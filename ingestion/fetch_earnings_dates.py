# ingestion/fetch_earnings_dates.py
import duckdb, time, pandas as pd,yfinance as yf
from datetime import datetime

from utilities.db_utilities import get_max_dates_by_stock
from utilities.api_functions import (get_earnings_data_from_api)
from utilities.data_utilities import to_float_or_none, get_alpha_vantage_api_key, read_stocks_to_fetch
from utilities.output_utilities import get_run_logs_dir
from config import STOCKS_START_DATE,ALPHAVANTAGE_CALLS_PER_MINUTE,EARNINGS_DATE_VALIDATION_WINDOW_DAYS
import os

def ingest_all_earnings_dates(con):
    already, inserted, failed = 0,0,0
    FAILED_EARNINGS_LOG_PATH = os.path.join(get_run_logs_dir(), "debug_failed_earnings_ingestion.txt")
    API_KEY = get_alpha_vantage_api_key()
    min_sleep = 60.0 / float(ALPHAVANTAGE_CALLS_PER_MINUTE)
    stocks = read_stocks_to_fetch()
    cutoff = pd.to_datetime(STOCKS_START_DATE).date()

    if not API_KEY:
        raise RuntimeError("Set ALPHAVANTAGE_API_KEY env var first.")
    # reset failure log each run (simple)
    with open(FAILED_EARNINGS_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("stock\terror\n")

    # cache current max earnings_date per stock
    max_earnings_date_by_stock = get_max_dates_by_stock(con, "earnings", "earnings_date")

    # heuristic freshness window (quarterly): if you already have something in last 90 days, skip
    today = datetime.now().date()
    fresh_window_days = 90

    for i, stock in enumerate(stocks, start=1):  
        stock_earn_max_date = max_earnings_date_by_stock.get(stock)

        if stock_earn_max_date is not None and (today - stock_earn_max_date).days <= fresh_window_days: # type: ignore
            already += 1
            print(f"{stock} is up to date")
            if i % 50 == 0:
                print(f"[{i}/{len(stocks)}] skipped(fresh): {already}, inserted: {inserted}, failed: {failed}")
            continue

        data = get_earnings_data_from_api(stock)    
        print(f"[{i}/{len(stocks)}] Fetching earnings data for {stock}...")
        try:
            if "quarterlyEarnings" not in data:
                raise RuntimeError(f"Bad payload keys: {list(data.keys())}. Snippet: {str(data)[:180]}")
            quarterly_earnings = data["quarterlyEarnings"]
            table_cols = ["stock", "reportedDate", "fiscalDateEnding", "reportedEPS", "estimatedEPS", "surprisePercentage"]
            rows = []   
            
            for quarter in quarterly_earnings:
                rows.append((stock, 
                             quarter["reportedDate"], 
                             quarter["fiscalDateEnding"], 
                             to_float_or_none(quarter["reportedEPS"]), 
                             to_float_or_none(quarter["estimatedEPS"]), 
                             to_float_or_none(quarter["surprisePercentage"])) )

            df = pd.DataFrame(rows, columns=table_cols)
            df = df.rename(columns={
                "reportedDate": "earnings_date",
                "fiscalDateEnding": "fiscal_end_date",
                "reportedEPS": "reported_eps",
                "estimatedEPS": "estimated_eps",
                "surprisePercentage": "surprise_percentage"
            })
            df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.date
            df["fiscal_end_date"] = pd.to_datetime(df["fiscal_end_date"]).dt.date
            
            df = df[df["earnings_date"] >= cutoff]
            df = df[df["fiscal_end_date"] >= cutoff] 

            if df.empty:
                already += 1
                time.sleep(min_sleep)
                continue

            df["surprise_percentage"] = df["surprise_percentage"] / 100
            df["ingested_at"] = datetime.now()

            count_before = con.execute("SELECT COUNT(*) FROM earnings WHERE stock = ?", [stock]).fetchone()[0] #type:ignore
            con.register("tmp_earnings_df", df)
            con.execute("INSERT OR IGNORE INTO earnings SELECT * FROM tmp_earnings_df")
            con.unregister("tmp_earnings_df")
            count_after = con.execute("SELECT COUNT(*) FROM earnings WHERE stock = ?", [stock]).fetchone()[0]#type:ignore
            added = count_after - count_before

            min_date, max_date = con.execute("""SELECT MIN(earnings_date), MAX(earnings_date) FROM earnings WHERE stock = ?;""", [stock]).fetchone() #type:ignore
            print(f"Added {added} rows ({min_date} -> {max_date})")
            if added != 0:
                inserted += 1
            print(f"Inserted {count_after} rows ({min_date} -> {max_date})")

        except Exception as e:
            failed += 1
            err = f"{type(e).__name__}: {e}"
            print(f"  FAILED {stock}: {err}")
            # ensure tmp view not left behind
            try:
                con.unregister("tmp_earnings_df")
            except Exception:
                pass
            with open(FAILED_EARNINGS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{stock}\t{err}\n")
        # always sleep a bit to respect rate limits
        time.sleep(min_sleep)
    
    print("\nIngesting Earnings Done.")
    print("already in DB:", already)
    print("inserted new:", inserted)
    print("failed:", failed)
    print("Failures saved to:", FAILED_EARNINGS_LOG_PATH)


def incremental_ingest_all_earnings_dates_yf(con):
    """
        Incremental earnings update using yfinance (no API key required).
        Fetches ~12 recent quarters + upcoming dates per stock.
        Skips stocks that already have a future earnings date in the DB.
    """
    stocks = read_stocks_to_fetch()
    today = datetime.now().date()
    already, inserted, failed = 0, 0, 0
    FAILED_LOG_PATH = os.path.join(get_run_logs_dir(), "debug_failed_earnings_ingestion.txt")

    with open(FAILED_LOG_PATH, "w") as f:
        f.write("stock\terror\n")

    # Skip stocks whose stored upcoming date is still comfortably far out.
    # Stocks due soon (or whose stored date just passed) are always rechecked,
    # since a stale placeholder estimate can otherwise never self-correct
    # before its own (wrong) date arrives.
    future_dates = set(
        row[0] for row in
        con.execute(
            "SELECT DISTINCT stock FROM earnings WHERE earnings_date > current_date + INTERVAL 7 DAY"
        ).fetchall()
    )

    for i, stock in enumerate(stocks, start=1):
        if stock in future_dates:
            already += 1
            if i % 100 == 0:
                print(f"[{i}/{len(stocks)}] skipped: {already}, inserted: {inserted}, failed: {failed}")
            continue

        try:
            ticker = yf.Ticker(stock)
            ed = ticker.earnings_dates
            if ed is None or ed.empty:
                failed += 1
                continue

            ed = ed.reset_index()
            ed = ed.rename(columns={
                "Earnings Date":  "earnings_date",
                "EPS Estimate":   "estimated_eps",
                "Reported EPS":   "reported_eps",
                "Surprise(%)":    "surprise_percentage",
            })

            ed["earnings_date"] = (
                pd.to_datetime(ed["earnings_date"])
                .dt.tz_localize(None)
                .dt.date
            )
            ed["stock"]           = stock
            ed["fiscal_end_date"] = None
            ed["surprise_percentage"] = ed["surprise_percentage"] / 100
            ed["ingested_at"]     = datetime.now()

            ed = ed[["stock", "earnings_date", "fiscal_end_date",
                     "reported_eps", "estimated_eps", "surprise_percentage", "ingested_at"]]

            # fiscal_end_date is None so the DB unique index can't deduplicate — filter manually
            existing = {
                row[0] for row in
                con.execute("SELECT earnings_date FROM earnings WHERE stock = ?", [stock]).fetchall()
            }
            ed = ed[~ed["earnings_date"].isin(existing)]
            if ed.empty:
                already += 1
                continue

            # For each freshly-fetched date (past or future), remove any unconfirmed
            # placeholder row within ±60 days — this is what clears a stale estimate
            # once the real (confirmed or corrected) date is known. Not restricted to
            # >= today: a just-reported date is exactly the anchor needed to clear an
            # old placeholder that was estimated too late.
            for new_date in ed["earnings_date"]:
                con.execute("""
                    DELETE FROM earnings
                    WHERE stock = ?
                      AND reported_eps IS NULL
                      AND ABS(DATEDIFF('day', earnings_date, ?)) <= 60
                """, [stock, new_date])

            count_before = con.execute(
                "SELECT COUNT(*) FROM earnings WHERE stock = ?", [stock]
            ).fetchone()[0]
            con.register("tmp_earnings_df", ed)
            con.execute("INSERT INTO earnings SELECT * FROM tmp_earnings_df")
            con.unregister("tmp_earnings_df")
            count_after = con.execute(
                "SELECT COUNT(*) FROM earnings WHERE stock = ?", [stock]
            ).fetchone()[0]

            added = count_after - count_before
            if added > 0:
                inserted += 1
                print(f"[{i}/{len(stocks)}] {stock}: +{added} rows")
            else:
                already += 1

        except Exception as e:
            failed += 1
            err = f"{type(e).__name__}: {e}"
            print(f"  FAILED {stock}: {err}")
            try:
                con.unregister("tmp_earnings_df")
            except Exception:
                pass
            with open(FAILED_LOG_PATH, "a") as f:
                f.write(f"{stock}\t{err}\n")

        time.sleep(0.3)

    print(f"\nIngesting Earnings Done (yfinance).")
    print(f"skipped/up-to-date: {already}, inserted: {inserted}, failed: {failed}")


def validate_upcoming_earnings_dates(con, days_ahead=EARNINGS_DATE_VALIDATION_WINDOW_DAYS, max_delta_days=30):
    """
        Cross-check upcoming unconfirmed earnings dates against ticker.calendar (company IR data).
        Corrects any date that differs by more than 1 day from the confirmed calendar date.
        days_ahead: only check dates within this many days out — dates further out get
        re-checked on a later run as they approach, so nothing is permanently skipped.
        max_delta_days: skip corrections where the calendar date is more than this many days out
        from the DB date — prevents next-quarter rollover false corrections.
        Called from pipeline/stage1.py after incremental_ingest_all_earnings_dates_yf.
    """
    today = datetime.now().date()

    rows = con.execute("""
        SELECT DISTINCT stock, earnings_date
        FROM earnings
        WHERE earnings_date >= CURRENT_DATE
          AND earnings_date <= CURRENT_DATE + CAST(? AS INTEGER)
          AND reported_eps IS NULL
        ORDER BY stock, earnings_date
    """, [days_ahead]).fetchall()

    if not rows:
        print("validate_upcoming_earnings_dates: no upcoming unconfirmed dates to check.")
        return {"corrected": [], "warnings": []}

    print(f"\nValidating {len(rows)} upcoming earnings dates (within {days_ahead} days) via ticker.calendar...")
    corrected = []
    warnings_list = []

    for stock, db_date in rows:
        if isinstance(db_date, pd.Timestamp):
            db_date = db_date.date()
        try:
            cal = yf.Ticker(stock).calendar
            if cal is None:
                warnings_list.append(stock)
                time.sleep(0.3)
                continue

            # calendar returns a dict {field: value_or_list} in yfinance 1.x
            if isinstance(cal, dict):
                ed_raw = cal.get("Earnings Date", [])
                if not isinstance(ed_raw, (list, tuple)):
                    ed_raw = [ed_raw] if ed_raw is not None else []
            elif isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.columns:
                    ed_raw = cal["Earnings Date"].dropna().tolist()
                else:
                    ed_raw = []
            else:
                ed_raw = []

            cal_dates = []
            for d in ed_raw:
                try:
                    cal_dates.append(pd.Timestamp(d).date())
                except Exception:
                    pass

            future_dates = [d for d in cal_dates if d >= today]
            if not future_dates:
                warnings_list.append(stock)
                time.sleep(0.3)
                continue

            cal_date = min(future_dates)
            diff = abs((cal_date - db_date).days)

            if diff > 0:
                if diff > max_delta_days:
                    warnings_list.append(stock)
                    print(f"  SKIPPED {stock}: {db_date} → {cal_date} ({diff:+d} days, exceeds max_delta={max_delta_days} — possible quarter rollover)")
                else:
                    con.execute("""
                        UPDATE earnings
                        SET earnings_date = ?
                        WHERE stock = ? AND earnings_date = ? AND reported_eps IS NULL
                    """, [cal_date, stock, db_date])
                    print(f"  CORRECTED {stock}: {db_date} → {cal_date} ({diff:+d} days)")
                    corrected.append((stock, db_date, cal_date))

        except Exception as e:
            warnings_list.append(stock)
            print(f"  WARNING {stock}: {type(e).__name__}: {e}")

        time.sleep(0.3)

    print(f"Validation done: {len(corrected)} corrected, {len(warnings_list)} no-calendar warnings.")
    return {"corrected": corrected, "warnings": warnings_list}


def get_next_earnings_dates():
    # TODO: Change to actually today (datetime.now())
    stocks = read_stocks_to_fetch()
    today = pd.Timestamp(datetime.now() ,tz="America/New_York")
    stock_dict = {}
    for i,stock in enumerate(stocks,start=1):
        if i%100==0:
            time.sleep(30)
        print(f"[{i}/{len(stocks)}] Fetching {stock} next Earnings Date...")
        try:
            ticker = yf.Ticker(stock)
            if ticker is not None:
                df = ticker.get_earnings_dates(limit=1, offset=0)
            else:
                raise ValueError(f"{ticker} returned None")
            if df is not None:
                df = df.reset_index()
                edates = pd.to_datetime(df["Earnings Date"])
            else:
                raise ValueError("Nothing from yfinance")
            for date in edates:
                if date >= today:
                    stock_dict[stock] = date.date()
        except Exception as e:
            print("ERROR - ", e)
            break
    next_earnings_df = pd.DataFrame(stock_dict.items(), columns=["stock","earnings_date"])
    next_earnings_df.to_csv("next_earnings_df.csv",index=False)
    print("DF created - next_earnings_df.csv")
    return next_earnings_df