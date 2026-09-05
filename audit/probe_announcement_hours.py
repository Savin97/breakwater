"""Phase 0 audit — what announcement times does the current provider actually return,
and how many fall inside market hours (09:30-16:00 ET) rather than cleanly BMO/AMC?"""
import random, time, collections
import pandas as pd, yfinance as yf
from utilities.data_utilities import read_stocks_to_fetch

random.seed(0)
stocks = read_stocks_to_fetch()
sample = random.sample(stocks, 60)
rows = []
for s in sample:
    try:
        d = yf.Ticker(s).earnings_dates
        if d is None or d.empty:
            continue
        for ts in d.index:
            rows.append({"stock": s, "ts": ts})
    except Exception as e:
        print("skip", s, type(e).__name__)
    time.sleep(0.4)

df = pd.DataFrame(rows)
df["hour"] = df["ts"].dt.hour + df["ts"].dt.minute / 60
def cls(h):
    if h < 9.5:  return "BMO (before 09:30)"
    if h >= 16:  return "AMC (16:00 or later)"
    return "INTRADAY (09:30-16:00)"
df["window"] = df["hour"].map(cls)
df.to_parquet("audit/announcement_hours.parquet")
print(f"tickers sampled: {df['stock'].nunique()}   timestamps: {len(df)}")
print(df["window"].value_counts(normalize=True).mul(100).round(1))
print("\nhour-of-day histogram:")
print(df["ts"].dt.strftime("%H:%M").value_counts().head(12))
per = df.groupby("stock")["window"].nunique()
print(f"\ntickers whose own history mixes >1 window: {(per > 1).sum()} / {len(per)}")
