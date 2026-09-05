"""Phase 0 audit — is the score attached to an UPCOMING event one earnings event stale?

`df.sort_values("date").groupby("stock").last()` skips NaN column-by-column, so the
"latest row" it builds carries date/earnings_date from the true last price row but
score/bucket/lift from the last row where those were non-NaN — the last COMPLETED
earnings event. This recomputes the structural score with the most recent completed
event included and compares.
"""
import numpy as np, pandas as pd
from config import (BUCKET_ELEVATED_FLOOR, BUCKET_HIGH_ALERT_FLOOR,
                    EXTREME_EARNINGS_REACTION_THRESHOLD)

COLS = ["stock", "date", "earnings_date", "is_earnings_day", "abs_reaction_3d",
        "abs_reaction_p75", "abs_reaction_p75_rolling", "reaction_entropy",
        "earnings_explosiveness_score", "earnings_explosiveness_bucket",
        "earnings_explosiveness_bucket_structural", "stock_bucket_lift"]
df = pd.read_parquet("output/full_df.parquet", columns=COLS)

# --- 1. reproduce what the exporters see -------------------------------------------
latest = df.sort_values("date").groupby("stock").last()
today = pd.Timestamp("today").normalize()
upcoming = latest[latest["earnings_date"] >= today].copy()

ev = df[df["is_earnings_day"] == 1].sort_values(["stock", "date"])
last_event = ev.groupby("stock").last()

check = upcoming.join(last_event[["date", "earnings_explosiveness_score"]],
                      rsuffix="_lastevent")
same = np.isclose(check["earnings_explosiveness_score"],
                  check["earnings_explosiveness_score_lastevent"], equal_nan=True)
print(f"upcoming events exported: {len(upcoming)}")
print(f"  whose score is byte-identical to the score of the stock's LAST COMPLETED event: "
      f"{same.sum()} / {len(check)} ({same.mean():.1%})")
print(f"  median gap between the price row's date and the event the score came from: "
      f"{(check['date'] - check['date_lastevent']).dt.days.median():.0f} days\n")

# --- 2. recompute the score with the most recent completed event included ----------
def entropy(series, bins=8):
    s = series.dropna()
    if len(s) < bins:
        return np.nan
    hist, _ = np.histogram(s, bins)
    p = hist / hist.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p))

rows = []
for stock, g in ev.groupby("stock"):
    a = g["abs_reaction_3d"].dropna()
    if len(a) == 0:
        continue
    p75_roll = a.iloc[-28:].quantile(0.75) if len(a) >= 28 else np.nan
    p75_exp = a.quantile(0.75)
    p75 = p75_roll if not np.isnan(p75_roll) else p75_exp
    e3 = np.clip(p75 / 0.12, 0, 1)
    e4 = np.clip(entropy(a), 0, 1)
    if np.isnan(e4):
        e4 = 0.0
    rows.append({"stock": stock,
                 "score_now": 100 * np.clip(0.85 * e3 + 0.15 * e4, 0, 1),
                 "last_reaction": a.iloc[-1],
                 "n_events": len(a)})
now = pd.DataFrame(rows).set_index("stock")

cmp = upcoming[["earnings_date", "earnings_explosiveness_score",
                "earnings_explosiveness_bucket",
                "earnings_explosiveness_bucket_structural"]].join(now, how="inner")
cmp = cmp.rename(columns={"earnings_explosiveness_score": "score_stale",
                          "earnings_explosiveness_bucket": "tier_shipped",
                          "earnings_explosiveness_bucket_structural": "tier_stale_structural"})
cmp["delta"] = cmp["score_now"] - cmp["score_stale"]

def bucket(s):
    return pd.cut(s, [-np.inf, BUCKET_ELEVATED_FLOOR, BUCKET_HIGH_ALERT_FLOOR, np.inf],
                  labels=["Normal", "Elevated", "High Alert"])
cmp["tier_now_structural"] = bucket(cmp["score_now"])

print("score shipped for the upcoming event vs score recomputed including the last completed event:")
print(f"  mean |delta| = {cmp['delta'].abs().mean():.2f} points   "
      f"max |delta| = {cmp['delta'].abs().max():.2f}   "
      f"delta != 0 for {(cmp['delta'].abs() > 1e-9).mean():.1%} of events")
moved = cmp[cmp["tier_stale_structural"].astype(str) != cmp["tier_now_structural"].astype(str)]
print(f"  STRUCTURAL TIER CHANGES: {len(moved)} / {len(cmp)} ({len(moved)/len(cmp):.1%})")
print(pd.crosstab(cmp["tier_stale_structural"].astype(str),
                  cmp["tier_now_structural"].astype(str)))

X = EXTREME_EARNINGS_REACTION_THRESHOLD
blind = cmp[cmp["last_reaction"] >= X]
print(f"\nupcoming events whose most recent completed reaction was >= {X:.0%} and is NOT "
      f"in the shipped score's window: {len(blind)}")
print(cmp.sort_values("delta", key=abs, ascending=False)
        .head(15)[["earnings_date", "score_stale", "score_now", "delta",
                   "tier_stale_structural", "tier_now_structural", "last_reaction"]]
        .to_string())
cmp.to_parquet("audit/score_staleness.parquet")
