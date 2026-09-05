"""Phase 0 audit — how much does the discarded announcement time distort abs_reaction_3d?

Classifies each stock as BMO-like or AMC-like from the empirical location of its
earnings-day move, then rebuilds the reaction with a timing-aware baseline:
  AMC: close(t+3) / close(t)   - 1   (== current behavior)
  BMO: close(t+2) / close(t-1) - 1   (baseline = last PRE-announcement close)
"""
import duckdb, pandas as pd, numpy as np
from config import DB_PATH, EXTREME_EARNINGS_REACTION_THRESHOLD, LARGE_EARNINGS_REACTION_THRESHOLD

con = duckdb.connect(DB_PATH, read_only=True)
prices = con.execute("SELECT stock, date, price FROM prices ORDER BY stock, date").fetchdf()
earn = con.execute("SELECT stock, earnings_date FROM earnings WHERE reported_eps IS NOT NULL").fetchdf()
con.close()
prices["date"] = pd.to_datetime(prices["date"]); earn["earnings_date"] = pd.to_datetime(earn["earnings_date"])
prices = prices.sort_values(["stock", "date"]).reset_index(drop=True)
prices["ret"] = prices.groupby("stock")["price"].pct_change()

rows = []
for stock, p in prices.groupby("stock"):
    p = p.reset_index(drop=True)
    pos = pd.Series(p.index.values, index=p["date"].values)
    px, rt = p["price"].values, p["ret"].values
    n = len(p)
    for d in earn.loc[earn["stock"] == stock, "earnings_date"]:
        i = pos.get(d)
        if i is None or (isinstance(i, float) and np.isnan(i)):
            continue
        i = int(i)
        g = lambda k: px[i + k] if 0 <= i + k < n else np.nan
        rows.append({
            "stock": stock, "earnings_date": d,
            "ret_d0": rt[i] if i < n else np.nan,
            "ret_d1": rt[i + 1] if i + 1 < n else np.nan,
            "cur_r3": g(3) / g(0) - 1,                       # what the pipeline computes
            "bmo_r3": g(2) / g(-1) - 1,                      # timing-aware for a BMO event
        })
ev = pd.DataFrame(rows)

# per-stock BMO/AMC classification
per = ev.dropna(subset=["ret_d0", "ret_d1"]).groupby("stock").agg(
    n=("ret_d0", "size"), m0=("ret_d0", lambda x: x.abs().median()), m1=("ret_d1", lambda x: x.abs().median()))
per["timing"] = np.where(per["m0"] > per["m1"], "BMO", "AMC")
per.loc[per["n"] < 8, "timing"] = "unknown"
ev = ev.merge(per[["timing"]], left_on="stock", right_index=True, how="left")

ev["cur_abs"] = ev["cur_r3"].abs()
ev["fix_abs"] = np.where(ev["timing"] == "BMO", ev["bmo_r3"], ev["cur_r3"])
ev["fix_abs"] = ev["fix_abs"].abs()
ev.to_parquet("audit/events_bmo_bias.parquet")

X, L = EXTREME_EARNINGS_REACTION_THRESHOLD, LARGE_EARNINGS_REACTION_THRESHOLD
print("event counts by inferred announcement timing:")
print(ev["timing"].value_counts(dropna=False), "\n")

for tag in ["BMO", "AMC"]:
    s = ev[ev["timing"] == tag].dropna(subset=["cur_abs", "fix_abs"])
    print(f"--- {tag}  (n={len(s)}) ---")
    print(f"  mean |reaction_3d|   current={s['cur_abs'].mean():.4f}   timing-aware={s['fix_abs'].mean():.4f}"
          f"   ({s['fix_abs'].mean()/s['cur_abs'].mean()-1:+.1%})")
    print(f"  median               current={s['cur_abs'].median():.4f}   timing-aware={s['fix_abs'].median():.4f}")
    print(f"  p75                  current={s['cur_abs'].quantile(.75):.4f}   timing-aware={s['fix_abs'].quantile(.75):.4f}")
    print(f"  P(>= {L:.0%} 'large')     current={(s['cur_abs']>=L).mean():.3f}   timing-aware={(s['fix_abs']>=L).mean():.3f}")
    print(f"  P(>= {X:.0%} 'extreme')   current={(s['cur_abs']>=X).mean():.3f}   timing-aware={(s['fix_abs']>=X).mean():.3f}")
    print(f"  rank corr current vs timing-aware: {s['cur_abs'].corr(s['fix_abs'], method='spearman'):.3f}")

s = ev.dropna(subset=["cur_abs"])
print("\nbase rate of 'extreme' by inferred timing, as the pipeline measures it today:")
print(s.groupby("timing")["cur_abs"].agg(n="size", extreme_rate=lambda x: (x >= X).mean(), mean="mean"))
