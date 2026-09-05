"""Phase 0 audit — event-level announcement timestamps straight from the provider.

Ground truth for the timing analysis. NOTHING here looks at price behavior.
yfinance returns a tz-aware America/New_York index carrying the announcement time;
we keep it verbatim alongside the calendar date the DB stores.
"""
import time, random, logging
import pandas as pd, yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

from utilities.data_utilities import read_stocks_to_fetch

logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def one(stock):
    try:
        time.sleep(random.uniform(0.1, 0.5))
        d = yf.Ticker(stock).earnings_dates
        if d is None or d.empty:
            return stock, None, "empty"
        idx = d.index
        return stock, pd.DataFrame({
            "stock": stock,
            "announce_ts_ny": idx,                      # tz-aware, NY
            "earnings_date": idx.tz_localize(None).date,  # exactly what ingestion stores
            "reported_eps": d["Reported EPS"].values,
        }), None
    except Exception as e:
        return stock, None, f"{type(e).__name__}: {e}"


stocks = read_stocks_to_fetch()
print(f"fetching provider timestamps for {len(stocks)} tickers...")
frames, errs = [], {}
with ThreadPoolExecutor(max_workers=8) as pool:
    futs = {pool.submit(one, s): s for s in stocks}
    for i, f in enumerate(as_completed(futs), 1):
        s, df, err = f.result()
        if df is not None:
            frames.append(df)
        else:
            errs[s] = err
        if i % 100 == 0:
            print(f"  [{i}/{len(stocks)}] ok={len(frames)} err={len(errs)}")

out = pd.concat(frames, ignore_index=True)
out["earnings_date"] = pd.to_datetime(out["earnings_date"])
out = out.drop_duplicates(subset=["stock", "earnings_date"], keep="first")
out.to_parquet("audit/provider_timestamps.parquet")
print(f"\ntickers with data: {out['stock'].nunique()}   timestamps: {len(out)}   errors: {len(errs)}")
print(f"date range: {out['earnings_date'].min().date()} -> {out['earnings_date'].max().date()}")
print(out["announce_ts_ny"].dt.strftime("%H:%M").value_counts().head(15))
