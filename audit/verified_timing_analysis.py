"""Phase 0 audit, revision 2 — timing analysis restricted to events with an
INDEPENDENTLY OBSERVED event-level announcement timestamp.

No price behavior is used to classify anything. Classification comes from
audit/provider_timestamps.parquet (yfinance's tz-aware NY announcement time).
Price behavior appears once, clearly labelled, as a CHECK ON the provider labels.
"""
import math
import numpy as np, pandas as pd, duckdb
from config import DB_PATH, EXTREME_EARNINGS_REACTION_THRESHOLD as X

def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (c-h, c+h)

def ci(k, n):
    lo, hi = wilson(k, n)
    return f"{k/n:.3f} [{lo:.3f},{hi:.3f}]" if n else "n/a"

# ---------------------------------------------------------------- classification
ts = pd.read_parquet("audit/provider_timestamps.parquet")
ts["hour"] = ts["announce_ts_ny"].dt.hour + ts["announce_ts_ny"].dt.minute/60
ts["window"] = np.select(
    [ts["hour"] < 9.5, ts["hour"] >= 16.0],
    ["BMO", "AMC"], default="INTRADAY")

# ---------------------------------------------------------------- prices -> anchored reactions
con = duckdb.connect(DB_PATH, read_only=True)
prices = con.execute("SELECT stock,date,price FROM prices ORDER BY stock,date").fetchdf()
con.close()
prices["date"] = pd.to_datetime(prices["date"])

rows = []
for stock, p in prices.groupby("stock"):
    sub = ts[ts["stock"] == stock]
    if sub.empty: continue
    p = p.reset_index(drop=True)
    pos = pd.Series(p.index.values, index=p["date"].values)
    px = p["price"].values; n = len(px)
    ret = np.r_[np.nan, px[1:]/px[:-1] - 1]
    for d, w in zip(sub["earnings_date"], sub["window"]):
        i = pos.get(d)
        if i is None or (isinstance(i, float) and np.isnan(i)): continue
        i = int(i)
        g = lambda k: px[i+k] if 0 <= i+k < n else np.nan
        rows.append({
            "stock": stock, "earnings_date": d, "window": w,
            "ret_d0": ret[i] if i < n else np.nan,
            "ret_d1": ret[i+1] if i+1 < n else np.nan,
            "r3_asis":     g(3)/g(0) - 1,                        # pipeline's definition
            "r3_anchored": (g(3)/g(0) - 1) if w == "AMC" else (g(2)/g(-1) - 1),
        })
ev = pd.DataFrame(rows)

# ---------------------------------------------------------------- CHECK on the provider labels
print("="*78)
print("VALIDATION OF THE PROVIDER LABELS (price behavior used only to TEST them)")
print("="*78)
v = ev.dropna(subset=["ret_d0","ret_d1"])
chk = v.groupby("window").apply(lambda d: pd.Series({
    "n": len(d),
    "share_day0_dominant": (d["ret_d0"].abs() > d["ret_d1"].abs()).mean(),
    "med|ret_d0|": d["ret_d0"].abs().median(),
    "med|ret_d1|": d["ret_d1"].abs().median()}), include_groups=False)
print(chk.round(4).to_string())
print("\nby exact announce hour (is 16:00 / 06:00 a real time or a default fill?):")
vh = v.merge(ts[["stock","earnings_date","announce_ts_ny"]], on=["stock","earnings_date"])
vh["hh"] = vh["announce_ts_ny"].dt.strftime("%H:%M")
h = vh.groupby("hh").apply(lambda d: pd.Series({
    "n": len(d), "share_day0_dominant": (d["ret_d0"].abs() > d["ret_d1"].abs()).mean()}),
    include_groups=False)
print(h[h["n"] >= 50].round(3).to_string())
ev.to_parquet("audit/verified_events.parquet")

# ---------------------------------------------------------------- Q2: verified-only results
full = pd.read_parquet("output/full_df.parquet", columns=[
    "stock","date","earnings_date","is_earnings_day","abs_reaction_3d",
    "earnings_explosiveness_bucket","earnings_explosiveness_score","pre_earnings_drift_flag"])
full = full[(full["is_earnings_day"]==1) & full["abs_reaction_3d"].notna()
            & full["earnings_explosiveness_bucket"].notna()]
m = full.merge(ev, on=["stock","earnings_date"], how="inner").dropna(subset=["r3_asis","r3_anchored"])
m["a_asis"] = m["r3_asis"].abs(); m["a_anch"] = m["r3_anchored"].abs()
m["tier"] = m["earnings_explosiveness_bucket"].astype(str)
m["hc"] = (m["tier"]=="High Alert") & (m["pre_earnings_drift_flag"].fillna("").str.strip()!="")

print("\n" + "="*78)
print("Q2 — RESULTS ON VERIFIED-TIMESTAMP EVENTS ONLY (no inference anywhere)")
print("="*78)
print(f"events with a verified event-level timestamp AND a scored row: {len(m)}")
print(f"  date range: {m['date'].min().date()} -> {m['date'].max().date()}"
      f"   tickers: {m['stock'].nunique()}")
print(f"  window mix: " + "  ".join(f"{k}={v}" for k,v in m['window'].value_counts().items()))
print(f"\n  (for reference, total scored events in full_df: {len(full)} — "
      f"verified coverage {len(m)/len(full):.1%})")

TIERS = ["High Alert","Elevated","Normal"]
print("\n--- market baseline P(|reaction| >= 8%) ---")
for lbl, c in [("as measured today","a_asis"), ("timestamp-anchored","a_anch")]:
    k = int((m[c]>=X).sum()); print(f"  {lbl:20s} {ci(k, len(m))}   n={len(m)}")

print("\n--- tier hit rates P(>=8%), all verified events ---")
print(f"{'tier':<12}{'n':>7}   {'as measured':<24}{'timestamp-anchored':<24}")
for t in TIERS:
    s = m[m["tier"]==t]
    print(f"{t:<12}{len(s):>7}   {ci(int((s.a_asis>=X).sum()),len(s)):<24}"
          f"{ci(int((s.a_anch>=X).sum()),len(s)):<24}")
s = m[m["hc"]]
print(f"{'HighConv':<12}{len(s):>7}   {ci(int((s.a_asis>=X).sum()),len(s)):<24}"
      f"{ci(int((s.a_anch>=X).sum()),len(s)):<24}")

print("\n--- lift vs the corresponding baseline ---")
b_asis, b_anch = (m.a_asis>=X).mean(), (m.a_anch>=X).mean()
for t in TIERS + ["HighConv"]:
    s = m[m["hc"]] if t=="HighConv" else m[m["tier"]==t]
    print(f"  {t:<12} {(s.a_asis>=X).mean()/b_asis:>5.2f}x  ->  {(s.a_anch>=X).mean()/b_anch:>5.2f}x")

print("\n--- tier composition by verified window (row %) ---")
print((pd.crosstab(m["window"], m["tier"], normalize="index")*100).round(1).to_string())
print("\n--- share of each tier drawn from each window (row %) ---")
print((pd.crosstab(m["tier"], m["window"], normalize="index")*100).round(1).to_string())

print("\n--- tier hit rates WITHIN each verified window ---")
for w in ["BMO","AMC","INTRADAY"]:
    sw = m[m["window"]==w]
    if len(sw) < 30: continue
    bw_a, bw_n = (sw.a_asis>=X).mean(), (sw.a_anch>=X).mean()
    print(f"\n  === {w}  (n={len(sw)})   baseline {ci(int((sw.a_asis>=X).sum()),len(sw))}"
          f"  ->  {ci(int((sw.a_anch>=X).sum()),len(sw))}")
    print(f"  {'tier':<12}{'n':>6}   {'as measured':<24}{'timestamp-anchored':<24}{'lift(anch)':>10}")
    for t in TIERS:
        s = sw[sw["tier"]==t]
        if len(s)==0: continue
        lift = (s.a_anch>=X).mean()/bw_n if bw_n else np.nan
        print(f"  {t:<12}{len(s):>6}   {ci(int((s.a_asis>=X).sum()),len(s)):<24}"
              f"{ci(int((s.a_anch>=X).sum()),len(s)):<24}{lift:>9.2f}x")

print("\n--- capture: share of anchored >=8% moves landing in High Alert/Elevated ---")
big = m[m["a_anch"]>=X]
k = int(big["tier"].isin(["High Alert","Elevated"]).sum())
print(f"  overall            {ci(k,len(big))}   n={len(big)}")
for w in ["BMO","AMC"]:
    b = big[big["window"]==w]
    print(f"  {w:<18} {ci(int(b['tier'].isin(['High Alert','Elevated']).sum()),len(b))}   n={len(b)}")
m.to_parquet("audit/verified_scored_events.parquet")
