"""
Window sensitivity test for abs_reaction_p75_rolling.

Tests rolling window sizes [8, 10, 15, 20, 28] against the existing
forward_eval_onefactor framework (train 2005-2010, test 2011-2025).

Run with:
    .venv/bin/python -m testing.window_sensitivity 2>&1 | tee output/window_sensitivity.txt
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from testing.testing_functions import forward_eval_onefactor

print("Loading parquet...")
df = pd.read_parquet("output/full_df.parquet")

# Only earnings events; sort chronologically per stock (required for rolling to be correct)
earn = df[df["is_earnings_day"] == 1].copy()
earn["date"] = pd.to_datetime(earn["date"])
earn = earn.sort_values(["stock", "date"]).reset_index(drop=True)

print(f"Earnings events: {len(earn):,}  |  Stocks: {earn['stock'].nunique()}  |  Years: {earn['date'].dt.year.min()}–{earn['date'].dt.year.max()}\n")

windows = [8, 10, 15, 20, 28]
summary_rows = []

for w in windows:
    col = f"p75_roll_{w}"

    # Rolling p75 of past w reactions — shift(1) prevents look-ahead leakage
    earn[col] = (
        earn.groupby("stock")["abs_reaction_3d"]
        .transform(lambda x: x.shift(1).rolling(w, min_periods=w).quantile(0.75))
    )
    # Fallback: expanding p75 for stocks with fewer than w historical events
    expanding = (
        earn.groupby("stock")["abs_reaction_3d"]
        .transform(lambda x: x.shift(1).expanding(min_periods=2).quantile(0.75))
    )
    earn[col] = earn[col].fillna(expanding)

    stats, thr = forward_eval_onefactor(earn, col)
    test_stats = stats[stats["split"] == "TEST"].copy()

    avg_lift        = test_stats["lift"].mean()
    median_lift     = test_stats["lift"].median()
    pct_above_3x    = (test_stats["lift"] >= 3.0).mean()
    avg_n_regime    = test_stats["n_regime"].mean()
    avg_capture     = test_stats["regime_capture_of_extremes"].mean()

    summary_rows.append({
        "window":          w,
        "threshold":       round(thr, 4),
        "avg_lift":        round(avg_lift, 2),
        "median_lift":     round(median_lift, 2),
        "pct_years_>=3x":  round(pct_above_3x, 2),
        "avg_n_regime/yr": round(avg_n_regime, 0),
        "avg_capture":     round(avg_capture, 3),
    })

    print(f"=== Window {w} events  (train threshold = {thr:.4f}) ===")
    print(test_stats[["year", "n_regime", "baseline_extreme_rate", "regime_extreme_rate", "lift", "regime_capture_of_extremes"]]
          .to_string(index=False))
    print()

print("\n" + "="*70)
print("SUMMARY — OOS 2011–2025")
print("="*70)
summary = pd.DataFrame(summary_rows)
print(summary.to_string(index=False))

best = summary.loc[summary["avg_lift"].idxmax()]
print(f"\nBest avg_lift: window={int(best['window'])}  avg_lift={best['avg_lift']}x  pct_years_>=3x={best['pct_years_>=3x']}")
print("\nCurrent production window: 28")
