"""Phase 0 audit probe — does the stored earnings_date follow the announcement's
calendar date (so BMO/AMC land on different trading days relative to the move)?

For each earnings event we measure the close-to-close return on the stored
earnings_date itself (day 0) and on the next trading day (day +1).
  - AMC announcement  -> the jump lands on day +1
  - BMO announcement  -> the jump lands on day 0
"""
import duckdb, pandas as pd, numpy as np
from config import DB_PATH

con = duckdb.connect(DB_PATH, read_only=True)
prices = con.execute("SELECT stock, date, price FROM prices ORDER BY stock, date").fetchdf()
earn = con.execute(
    "SELECT stock, earnings_date FROM earnings WHERE reported_eps IS NOT NULL ORDER BY stock, earnings_date"
).fetchdf()
con.close()

prices["date"] = pd.to_datetime(prices["date"])
earn["earnings_date"] = pd.to_datetime(earn["earnings_date"])

prices = prices.sort_values(["stock", "date"]).reset_index(drop=True)
prices["ret"] = prices.groupby("stock")["price"].pct_change()
prices["idx"] = prices.groupby("stock").cumcount()

# map (stock, date) -> row index within stock
lookup = prices.set_index(["stock", "date"])[["idx", "ret"]]

rows = []
by_stock = {s: g.reset_index(drop=True) for s, g in prices.groupby("stock")}
for stock, g in earn.groupby("stock"):
    p = by_stock.get(stock)
    if p is None:
        continue
    pos = pd.Series(p.index.values, index=p["date"].values)
    for d in g["earnings_date"]:
        i = pos.get(d)
        if i is None or np.isnan(i):
            rows.append({"stock": stock, "earnings_date": d, "on_trading_day": False,
                         "ret_d0": np.nan, "ret_d1": np.nan})
            continue
        i = int(i)
        r0 = p["ret"].iloc[i] if i < len(p) else np.nan
        r1 = p["ret"].iloc[i + 1] if i + 1 < len(p) else np.nan
        rows.append({"stock": stock, "earnings_date": d, "on_trading_day": True,
                     "ret_d0": r0, "ret_d1": r1})

ev = pd.DataFrame(rows)
ev.to_parquet("audit/events_timing_probe.parquet")

print(f"events with a confirmed result: {len(ev)}")
print(f"  falling on a non-trading day (no is_earnings_day row at all): "
      f"{(~ev['on_trading_day']).sum()} ({(~ev['on_trading_day']).mean():.2%})")

t = ev[ev["on_trading_day"]].dropna(subset=["ret_d0", "ret_d1"])
print(f"\nusable events: {len(t)}")
print(f"mean |ret| on day 0 (stored earnings_date): {t['ret_d0'].abs().mean():.4f}")
print(f"mean |ret| on day +1                      : {t['ret_d1'].abs().mean():.4f}")

# Per-stock verdict: which day carries the bigger typical move?
per = t.groupby("stock").agg(n=("ret_d0", "size"),
                             m0=("ret_d0", lambda x: x.abs().median()),
                             m1=("ret_d1", lambda x: x.abs().median()))
per = per[per["n"] >= 8]
per["verdict"] = np.where(per["m0"] > per["m1"], "day0 (BMO-like)", "day+1 (AMC-like)")
print("\nper-stock verdict (>=8 events):")
print(per["verdict"].value_counts())

known_bmo = ["JPM", "PG", "KO", "JNJ", "MCD", "WMT", "GS", "BAC", "CAT", "MMM", "VZ", "T"]
known_amc = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "NFLX", "AMD", "CRM", "ORCL"]
print("\nknown BMO reporters:")
print(per.reindex([s for s in known_bmo if s in per.index]))
print("\nknown AMC reporters:")
print(per.reindex([s for s in known_amc if s in per.index]))
