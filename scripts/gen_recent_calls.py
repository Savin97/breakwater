"""
Generate output/recent_calls.json from full_df.parquet.
Includes High Conviction, High Alert, and Elevated stocks with completed
reaction_3d, going back 6 weeks. JS on the landing page filters to last 14 days.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HA_BUCKETS = {"High Alert"}

def tier_for(row):
    if row.get("is_high_conviction"):
        return "hc", "HIGH CONVICTION ★"
    if row["earnings_explosiveness_bucket"] in HA_BUCKETS:
        return "high", "HIGH ALERT"
    if row["earnings_explosiveness_bucket"] == "Elevated":
        return "mid", "ELEVATED"
    return None, None

def main():
    parquet = Path("output/full_df.parquet")
    if not parquet.exists():
        print("output/full_df.parquet not found", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(parquet)

    cols = ["stock", "earnings_date", "earnings_explosiveness_bucket",
            "reaction_1d", "reaction_3d"]
    if "is_high_conviction" in df.columns:
        cols.append("is_high_conviction")

    earnings = df[df["is_earnings_day"] == 1][cols].copy()
    earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"])

    cutoff = pd.Timestamp.today() - pd.Timedelta(weeks=6)
    earnings = earnings[earnings["earnings_date"] >= cutoff]

    earnings["best_reaction"] = earnings["reaction_3d"].fillna(earnings["reaction_1d"])
    earnings = earnings.dropna(subset=["best_reaction"])

    include = HA_BUCKETS | {"Elevated"}
    earnings = earnings[earnings["earnings_explosiveness_bucket"].isin(include)]

    earnings["week_start"] = (
        earnings["earnings_date"]
        - pd.to_timedelta(earnings["earnings_date"].dt.dayofweek, unit="D")
    )

    records = []
    for _, row in earnings.iterrows():
        tier_class, tier_label = tier_for(row)
        if tier_class is None:
            continue
        move_pct = float(row["best_reaction"]) * 100
        records.append({
            "earnings_date": row["earnings_date"].strftime("%Y-%m-%d"),
            "week_start":    row["week_start"].strftime("%Y-%m-%d"),
            "ticker":        row["stock"],
            "tier_class":    tier_class,
            "tier_label":    tier_label,
            "move_pct":      move_pct,
        })

    records.sort(key=lambda x: (x["earnings_date"], -abs(x["move_pct"])), reverse=True)

    out = {
        "generated": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "calls": records,
    }
    Path("output/recent_calls.json").write_text(json.dumps(out, indent=2))
    print(f"Written {len(records)} calls → output/recent_calls.json")

if __name__ == "__main__":
    main()
