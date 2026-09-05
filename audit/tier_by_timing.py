"""Phase 0 audit — what does the discarded announcement time do to the product's tiers?"""
import numpy as np, pandas as pd
from config import EXTREME_EARNINGS_REACTION_THRESHOLD as X

tim = pd.read_parquet("audit/events_bmo_bias.parquet")[
    ["stock", "earnings_date", "timing", "cur_abs", "fix_abs"]]
ev = pd.read_parquet("output/full_df.parquet",
                     columns=["stock", "date", "earnings_date", "is_earnings_day",
                              "abs_reaction_3d", "earnings_explosiveness_bucket",
                              "earnings_explosiveness_score", "is_extreme_reaction"])
ev = ev[(ev["is_earnings_day"] == 1) & ev["abs_reaction_3d"].notna()]
ev = ev.merge(tim, on=["stock", "earnings_date"], how="left")
ev = ev[(ev["date"] >= "2015-01-01") & (ev["date"] <= "2025-12-31")]

print("TIER COMPOSITION by inferred announcement timing (2015-2025 events)")
ct = pd.crosstab(ev["timing"], ev["earnings_explosiveness_bucket"], normalize="index")
print((ct * 100).round(1))
print("\nshare of each tier's events that are AMC reporters:")
print(pd.crosstab(ev["earnings_explosiveness_bucket"], ev["timing"], normalize="index").mul(100).round(1))

print(f"\nP(|reaction| >= {X:.0%}) per tier, AS MEASURED TODAY vs TIMING-AWARE")
for tag in ["AMC", "BMO"]:
    s = ev[ev["timing"] == tag]
    g = s.groupby("earnings_explosiveness_bucket", observed=True).apply(
        lambda d: pd.Series({"n": len(d),
                             "hit_as_measured": (d["cur_abs"] >= X).mean(),
                             "hit_timing_aware": (d["fix_abs"] >= X).mean()}),
        include_groups=False)
    print(f"\n--- {tag} ---")
    print(g.round(3))

print("\ncapture: share of ALL timing-aware >=8% moves that the model tiered High Alert/Elevated")
big = ev[ev["fix_abs"] >= X]
flagged = big["earnings_explosiveness_bucket"].isin(["High Alert", "Elevated"])
print(f"  overall {flagged.mean():.3f}  (n={len(big)})")
print(big.groupby("timing").apply(
    lambda d: pd.Series({"n": len(d),
        "captured": d["earnings_explosiveness_bucket"].isin(["High Alert","Elevated"]).mean()}),
    include_groups=False).round(3))
