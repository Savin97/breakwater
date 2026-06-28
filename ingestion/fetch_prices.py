# ingestion/fetch_prices.py
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from utilities.db_utilities import (stock_already_in_prices_db,get_max_dates_by_stock)
from utilities.api_functions import (fetch_daily_adjusted_prices)
from utilities.data_utilities import get_alpha_vantage_api_key, read_stocks_to_fetch
from config import (
    STOCKS_START_DATE,
    ALPHAVANTAGE_CALLS_PER_MINUTE)
def ingest_all_stocks(con):
    """
        This function updates the 'prices' table in the duckDB.
    """
    already, inserted, failed = 0,0,0
    FAILED_LOG_PATH = "output/debug_failed_price_ingestion.txt"
    API_KEY = get_alpha_vantage_api_key()
    min_sleep = 60.0 / float(ALPHAVANTAGE_CALLS_PER_MINUTE)
    stocks = read_stocks_to_fetch()
    if not stocks:
        raise ValueError("No stocks found.")
    cutoff = pd.to_datetime(STOCKS_START_DATE).date()

    if not API_KEY:
        raise RuntimeError("Set ALPHAVANTAGE_API_KEY env var first.")
    # reset failure log each run (simple)
    with open(FAILED_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("stock\terror\n")

    global_max_date = con.execute("SELECT MAX(date) FROM prices").fetchone()[0] # type: ignore
    max_date_by_stock = get_max_dates_by_stock(con, "prices", "date")
    print(f"Stocks: {len(stocks)}")
    print("Prices DB global max date:", global_max_date)

    for i, stock in enumerate(stocks, start=1):
        stock_max_date = max_date_by_stock.get(stock)
        print(f"\n{stock} Max date is {stock_max_date}")
        try:
            outputsize="full"
            if stock_already_in_prices_db(con, stock):
                if i % 50 == 0:
                    print(f"[{i}/{len(stocks)}] already in DB: {already}, inserted: {inserted}, failed: {failed}")
                # TODO: 18/2/26 implement "stale-ticker" detection rule
                # def trading_days_since(last_date, calendar="NYSE"):
                #     cal = mcal.get_calendar(calendar)
                #     schedule = cal.schedule(start_date=last_date, end_date=date.today())
                #     return len(schedule)
                # if trading_days_since(global_max_date) > 5: # >10 trading days
                #     with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f:
                #         f.write(f"{stock}\t{"Appears to be delisted, (last price: {stock_max_date})"}\n")
                #     # You could log something like:
                #     # DAY appears delisted (last price: 2026-02-03)
                #     # stock = "inactive" # mark ticker as inactive
                #     continue # skip ingestion attempts
                if (stock_max_date is not None) and (global_max_date is not None) and (stock_max_date >= global_max_date):
                    already += 1
                    print(f"{stock} is up to date")
                    continue
                else:
                    outputsize = "compact"

            print(f"[{i}/{len(stocks)}] Fetching {stock} ...")
            data = fetch_daily_adjusted_prices(stock, outputsize)
            # Basic failure handling (AlphaVantage often returns errors inside JSON)
            if "Error Message" in data:
                print(f"[{i}] {stock}: no data / error")
                time.sleep(min_sleep)
                continue

            ts = data["Time Series (Daily)"]

            rows = []
            for date_str, ohlc in ts.items():
                price = float(ohlc["5. adjusted close"])
                rows.append((stock, date_str, price))

            df = pd.DataFrame(rows, columns=["stock", "date", "price"])
            df["date"] = pd.to_datetime(df["date"]).dt.date
            if stock_max_date is not None:
                df = df[df["date"] > stock_max_date]   # only new days for that stock
            else:
                df = df[df["date"] >= cutoff]     # full ingest case
            if df.empty:
                # nothing new; update counters and move on
                already += 1
                time.sleep(min_sleep)
                continue

            df["ingested_at"] = datetime.now()

            con.register("tmp_prices", df)
            before = con.execute("SELECT COUNT(*) FROM prices WHERE stock = ?", [stock]).fetchone()[0] # type: ignore
            con.execute("INSERT OR IGNORE INTO prices SELECT * FROM tmp_prices;")
            con.unregister("tmp_prices")
            after  = con.execute("SELECT COUNT(*) FROM prices WHERE stock = ?", [stock]).fetchone()[0] # type: ignore
            added = after - before
            min_date, max_date = con.execute(
                "SELECT MIN(date), MAX(date) FROM prices WHERE stock = ?;",
                [stock]
            ).fetchone() # type: ignore
            print(f"Added {added} rows ({min_date} -> {max_date})")
            if added > 0:
                inserted += 1
            print(f"Inserted {after} rows ({min_date} -> {max_date})")
            # update cached max so later logic in same run is consistent
            max_date_by_stock[stock] = max_date
        except Exception as e:
            failed += 1
            err = f"{type(e).__name__}: {e}"
            print(f" FAILED {stock}: {err}")
            # ensure tmp view not left behind
            try:
                con.unregister("tmp_prices")
            except Exception:
                pass
            with open(FAILED_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{stock}\t{err}\n")
        # always sleep a bit to respect rate limits
        time.sleep(min_sleep)
    
    print("\nIngesting Prices Done.")
    print("skipped/up-to-date:", already)
    print("inserted new:", inserted)
    print("failed:", failed)
    print("failures saved to:", FAILED_LOG_PATH)
    return


_BATCH_SIZE = 100


def ingest_all_stocks_yf(con):
    """Incremental price update using yfinance (no API key required)."""
    stocks = read_stocks_to_fetch()
    if not stocks:
        raise ValueError("No stocks found.")

    global_max = con.execute("SELECT MAX(date) FROM prices").fetchone()[0]
    start = (global_max + timedelta(days=1)) if global_max else pd.to_datetime(STOCKS_START_DATE).date()
    end = date.today()

    if start >= end:
        print("Prices already up to date.")
        return

    print(f"Fetching prices {start} → {end} for {len(stocks)} stocks...")
    total_inserted = 0
    now = datetime.now()

    for batch_start in range(0, len(stocks), _BATCH_SIZE):
        batch = stocks[batch_start: batch_start + _BATCH_SIZE]
        try:
            raw = yf.download(
                batch, start=str(start), end=str(end),
                auto_adjust=True, progress=False,
            )
        except Exception as e:
            print(f"  Batch [{batch_start+1}–{batch_start+len(batch)}] failed: {e}")
            continue

        if raw.empty:
            continue

        closes = raw["Close"] if "Close" in raw.columns else raw
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(name=batch[0])

        rows = []
        for ticker in batch:
            if ticker not in closes.columns:
                continue
            series = closes[ticker].dropna()
            for dt, price in series.items():
                rows.append((ticker, dt.date() if hasattr(dt, "date") else dt, float(price), now))

        if not rows:
            continue

        df = pd.DataFrame(rows, columns=["stock", "date", "price", "ingested_at"])
        con.register("tmp_prices", df)
        before = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        con.execute("INSERT OR IGNORE INTO prices SELECT * FROM tmp_prices")
        con.unregister("tmp_prices")
        added = con.execute("SELECT COUNT(*) FROM prices").fetchone()[0] - before
        total_inserted += added
        print(f"  Batch [{batch_start+1}-{batch_start+len(batch)}]: +{added} rows")

    print(f"\nIngesting Prices Done (yfinance). Total inserted: {total_inserted}")