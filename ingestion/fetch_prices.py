# ingestion/fetch_prices.py
import os
import time
import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from utilities.db_utilities import (stock_already_in_prices_db, get_max_dates_by_stock,
                                    create_prices_table_if_not_exists)
from utilities.api_functions import (fetch_daily_adjusted_prices)
from utilities.data_utilities import get_alpha_vantage_api_key, read_stocks_to_fetch
from utilities.output_utilities import get_run_logs_dir
from config import (
    STOCKS_START_DATE,
    ALPHAVANTAGE_CALLS_PER_MINUTE)

logger = logging.getLogger(__name__)

def ingest_all_stocks(con):
    """
        This function updates the 'prices' table in the duckDB.
    """
    already, inserted, failed = 0,0,0
    FAILED_LOG_PATH = os.path.join(get_run_logs_dir(), "debug_failed_price_ingestion.txt")
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

BATCH_SIZE = 100

# ── yfinance ingestion: reconcile a recent window, never just append ────────────────────
# Adjusted closes are a moving target: a split, spin-off or dividend makes the provider
# re-adjust the stock's whole history. The previous design fetched only dates after the
# DB's GLOBAL max date and appended them, which (audit H4) left every older row on the
# old basis — MNST's 2:1 split of 2026-08-11 read as a -51.7% earnings reaction — and,
# because the start was global, one stock ingested ahead of the rest left a hole in all
# the others (2026-05-19..21, 497 of 500 stocks).
#
# Now every run re-fetches, per stock, a window that reaches back past its last stored
# row, and reconciles it (`reconcile_stock_prices`): rows in the window are replaced by
# the provider's (consistent) values, missing sessions in the window are filled, and if
# the provider's basis moved, every older row is rescaled by one factor onto the new
# basis — returns inside the old history are unchanged, the seam disappears. Every
# rescale is logged in `price_basis_log`. Run with `since=` to repair a longer stretch
# once (the 2026 history: since="2026-02-11", the last Alpha Vantage full-load session).
RECONCILE_LOOKBACK_DAYS = 45      # calendar days re-fetched every run
RECONCILE_OVERLAP_DAYS = 14       # and at least this far before the stock's last stored row
BASIS_REL_TOL = 1e-6              # float noise between two downloads of the same basis


class PriceBasisError(RuntimeError):
    """The fetched window shares no date with the stored history: the basis is unknowable."""


def reconcile_stock_prices(stored: pd.DataFrame, fresh: pd.DataFrame,
                           rel_tol: float = BASIS_REL_TOL) -> tuple[pd.DataFrame, float]:
    """Pure reconciliation of ONE stock. `stored`: its rows already in the DB [date, price];
    `fresh`: the provider's adjusted closes for a window [date, price] (consistent among
    themselves by construction — one download).

    Returns (window_rows, factor): `window_rows` replace every stored row on those dates
    and fill sessions the DB lacks; every OTHER stored row is multiplied by `factor`
    (1.0 when the basis did not move). The factor is fresh/stored on the earliest date
    both hold — the row that shares the basis of everything stored before the window.
    """
    fresh = fresh.dropna(subset=["price"])
    if stored.empty:
        return fresh[["date", "price"]].copy(), 1.0
    common = stored.merge(fresh, on="date", suffixes=("_stored", "_fresh"))
    if common.empty:
        raise PriceBasisError("fetched window does not overlap the stored history")
    first = common.sort_values("date").iloc[0]
    factor = float(first["price_fresh"] / first["price_stored"])
    if abs(factor - 1.0) <= rel_tol:
        factor = 1.0
    return fresh[["date", "price"]].copy(), factor


def _ensure_basis_log(con):
    con.execute("""CREATE TABLE IF NOT EXISTS price_basis_log (
        stock TEXT, applied_at TIMESTAMP, factor DOUBLE, rows_rescaled BIGINT,
        window_start DATE, window_rows BIGINT)""")


def apply_reconciliation(con, stock: str, window_rows: pd.DataFrame, factor: float,
                         now: datetime) -> dict:
    """Write one stock's reconciliation in a single transaction."""
    w = window_rows.copy()
    w["date"] = pd.to_datetime(w["date"]).dt.date
    w.insert(0, "stock", stock)
    w["ingested_at"] = now
    con.execute("BEGIN")
    try:
        con.register("tmp_window", w[["stock", "date", "price", "ingested_at"]])
        rescaled = 0
        if factor != 1.0:
            rescaled = con.execute(
                "SELECT COUNT(*) FROM prices WHERE stock = ? AND date NOT IN (SELECT date FROM tmp_window)",
                [stock]).fetchone()[0]
            con.execute("UPDATE prices SET price = price * ? WHERE stock = ? "
                        "AND date NOT IN (SELECT date FROM tmp_window)", [factor, stock])
        con.execute("DELETE FROM prices WHERE stock = ? AND date IN (SELECT date FROM tmp_window)",
                    [stock])
        con.execute("INSERT INTO prices SELECT * FROM tmp_window")
        if factor != 1.0:
            con.execute("INSERT INTO price_basis_log VALUES (?, ?, ?, ?, ?, ?)",
                        [stock, now, factor, rescaled, w["date"].min(), len(w)])
        con.unregister("tmp_window")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return {"stock": stock, "factor": factor, "rescaled": rescaled, "window_rows": len(w)}


def _download_closes(tickers, start, end) -> pd.DataFrame:
    """Long frame [stock, date, price] of adjusted closes. Isolated so tests stub it."""
    raw = yf.download(tickers, start=str(start), end=str(end), auto_adjust=True,
                      progress=False)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["stock", "date", "price"])
    closes = raw["Close"] if "Close" in raw.columns else raw
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(name=tickers[0])
    out = closes.stack().rename("price").reset_index()
    out.columns = ["date", "stock", "price"]
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
    return out[["stock", "date", "price"]]


def incremental_ingest_all_prices_yf(con, since=None, today=None):
    """Reconciling price update via yfinance (no API key). See the block comment above."""
    stocks = read_stocks_to_fetch(con, active_only=True)
    if not stocks:
        raise ValueError("No stocks found.")
    create_prices_table_if_not_exists(con)
    _ensure_basis_log(con)
    today = pd.Timestamp(today or date.today()).normalize()
    end = (today + pd.Timedelta(days=1)).date()           # yfinance's end is exclusive
    last = get_max_dates_by_stock(con, "prices", "date")
    now = datetime.now()

    def window_start(stock):
        if since is not None:
            return pd.Timestamp(since)
        s_last = last.get(stock)
        if s_last is None:
            return pd.Timestamp(STOCKS_START_DATE)
        return min(today - pd.Timedelta(days=RECONCILE_LOOKBACK_DAYS),
                   pd.Timestamp(s_last) - pd.Timedelta(days=RECONCILE_OVERLAP_DAYS))

    results, failed = [], []
    for batch_start in range(0, len(stocks), BATCH_SIZE):
        batch = stocks[batch_start: batch_start + BATCH_SIZE]
        starts = {s: window_start(s) for s in batch}
        try:
            fresh_all = _download_closes(batch, min(starts.values()).date(), end)
        except Exception as e:
            logger.warning(f"Batch [{batch_start+1}–{batch_start+len(batch)}] failed: {e}")
            failed += batch
            continue
        for stock in batch:
            fresh = fresh_all[(fresh_all["stock"] == stock) & (fresh_all["date"] >= starts[stock])]
            if fresh.empty:
                failed.append(stock)
                continue
            stored = con.execute("SELECT date, price FROM prices WHERE stock = ?", [stock]).df()
            stored["date"] = pd.to_datetime(stored["date"])
            try:
                rows, factor = reconcile_stock_prices(stored, fresh[["date", "price"]])
            except PriceBasisError as e:
                logger.warning(f"{stock}: {e} — left unchanged")
                failed.append(stock)
                continue
            results.append(apply_reconciliation(con, stock, rows, factor, now))
    moved = [r for r in results if r["factor"] != 1.0]
    logger.info(f"Prices reconciled (yfinance): {len(results)} stocks, "
                f"{sum(r['window_rows'] for r in results)} window rows, "
                f"{len(moved)} basis changes, {len(failed)} failed")
    for r in moved:
        logger.info(f"  basis change {r['stock']}: x{r['factor']:.6f} on {r['rescaled']} older rows")
    if failed:
        logger.warning(f"  not reconciled: {', '.join(sorted(set(failed)))}")
    return results
