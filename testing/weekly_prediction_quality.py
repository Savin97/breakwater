"""
Evaluate weekly earnings prediction quality from April 10, 2026 onwards.
Prints per-event results week by week, then an aggregate summary by tier.
Saves full per-event data to testing/testing_results/weekly_prediction_quality.csv.

Usage:
    python testing/weekly_prediction_quality.py
    python testing/weekly_prediction_quality.py --start 2026-01-01
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import timedelta
from pathlib import Path

PARQUET     = "output/full_df.parquet"
DEFAULT_START = "2026-04-13"
THRESHOLDS  = [0.05, 0.08, 0.10]
BUCKET_ORDER = {"High Alert": 0, "Elevated": 1, "Normal": 2}
TIER_LABEL   = {"High Alert": "HIGH ALERT", "Elevated": "ELEVATED", "Normal": "NORMAL"}


def _save_confusion_matrix_png(with_outcome, out_path):
    PREDICTORS = [
        ("High Alert",        with_outcome["tier"] == "High Alert"),
        ("HA + Elevated",     with_outcome["tier"].isin(["High Alert", "Elevated"])),
        ("High Conviction ★", with_outcome["is_high_conviction"]),
    ]
    THRESHOLDS = [(5, "moved_5pct"), (8, "moved_8pct"), (10, "moved_10pct")]

    SURFACE   = "#fcfcfb"
    INK       = "#0b0b0b"
    INK_MUT   = "#898781"
    GRID      = "#e1e0d9"
    CORRECT   = "#d6f0d6"   # tinted good (#0ca30c)
    INCORRECT = "#faddda"   # tinted critical (#d03b3b)
    CORRECT_DARK   = "#0ca30c"
    INCORRECT_DARK = "#d03b3b"

    fig, axes = plt.subplots(len(THRESHOLDS), len(PREDICTORS), figsize=(11, 9))
    fig.patch.set_facecolor(SURFACE)

    for col_i, (pred_label, flagged) in enumerate(PREDICTORS):
        for row_i, (thresh, col) in enumerate(THRESHOLDS):
            ax = axes[row_i][col_i]
            ax.set_facecolor(SURFACE)

            actual = with_outcome[col]
            tp = ( flagged &  actual).sum()
            fp = ( flagged & ~actual).sum()
            fn = (~flagged &  actual).sum()
            tn = (~flagged & ~actual).sum()
            total = tp + fp + fn + tn

            cells = [
                (0, 1, tp, "TP", CORRECT,   CORRECT_DARK),
                (1, 1, fp, "FP", INCORRECT, INCORRECT_DARK),
                (0, 0, fn, "FN", INCORRECT, INCORRECT_DARK),
                (1, 0, tn, "TN", CORRECT,   CORRECT_DARK),
            ]

            for cx, cy, count, label, bg, ink in cells:
                ax.add_patch(patches.FancyBboxPatch(
                    (cx + 0.04, cy + 0.04), 0.92, 0.92,
                    boxstyle="round,pad=0.02",
                    facecolor=bg, edgecolor=GRID, linewidth=1,
                ))
                pct = count / total * 100 if total > 0 else 0
                ax.text(cx + 0.5, cy + 0.62, label,
                        ha="center", va="center", fontsize=9,
                        color=ink, fontweight="bold",
                        fontfamily="monospace")
                ax.text(cx + 0.5, cy + 0.38, str(count),
                        ha="center", va="center", fontsize=14,
                        color=INK, fontweight="bold")
                ax.text(cx + 0.5, cy + 0.18, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color=INK_MUT)

            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)

            # X-axis labels (actual)
            ax.set_xticks([0.5, 1.5])
            ax.set_xticklabels([f"Moved ≥{thresh}%", f"<{thresh}%"],
                               fontsize=8, color=INK_MUT)
            ax.xaxis.set_tick_params(length=0)

            # Y-axis labels (predicted) — only leftmost column
            if col_i == 0:
                ax.set_yticks([0.5, 1.5])
                ax.set_yticklabels(["Not flagged", "Flagged"],
                                   fontsize=8, color=INK_MUT, rotation=90, va="center")
                ax.yaxis.set_tick_params(length=0)
            else:
                ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_visible(False)

            if row_i == 0:
                ax.set_title(pred_label, fontsize=10, color=INK,
                             fontweight="bold", pad=10)

    # Row labels (thresholds) on the right
    for row_i, (thresh, _) in enumerate(THRESHOLDS):
        axes[row_i][-1].annotate(
            f"≥{thresh}% threshold",
            xy=(1.04, 0.5), xycoords="axes fraction",
            fontsize=8, color=INK_MUT, rotation=270, va="center", ha="left",
        )

    # Column headers
    fig.text(0.5, 0.04, "Actual outcome →", ha="center",
             fontsize=9, color=INK_MUT)
    fig.text(0.02, 0.5, "← Predicted", va="center", rotation=90,
             fontsize=9, color=INK_MUT)

    fig.suptitle("Confusion Matrices — Earnings Risk Predictions",
                 fontsize=13, color=INK, fontweight="bold", y=1.01)

    plt.tight_layout(rect=(0.04, 0.04, 0.96, 1.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  Saved → {out_path}")


def _fmt(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "   n/a"
    return f"{val * 100:+5.1f}%"


def run(start_date=DEFAULT_START):
    df = pd.read_parquet(PARQUET)
    df["date"]          = pd.to_datetime(df["date"])
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])

    earnings = df[
        (df["is_earnings_day"] == 1) &
        (df["earnings_date"] >= start_date) &
        df["earnings_explosiveness_bucket"].notna()
    ].copy()

    if earnings.empty:
        print("No earnings events found since that date.")
        return

    earnings["_week"]  = earnings["earnings_date"].apply(
        lambda d: d - timedelta(days=d.weekday())
    )
    earnings["_rank"]  = earnings["earnings_explosiveness_bucket"].map(BUCKET_ORDER).fillna(9)
    earnings           = earnings.sort_values(["_week", "_rank", "earnings_date", "stock"])

    rows        = []
    current_week = None

    for _, row in earnings.iterrows():
        week = row["_week"]

        if week != current_week:
            current_week = week
            week_end = week + timedelta(days=4)
            print(f"\n{'═' * 70}")
            print(f"  {week.strftime('%b %-d')} – {week_end.strftime('%b %-d, %Y')}")
            print(f"  {'─' * 66}")
            print(f"  {'STOCK':<8} {'DATE':<8} {'TIER':<12} {'SCORE':>5}  {'ACTUAL':>7}  {'HC':>3}")
            print(f"  {'─' * 66}")

        bucket  = row["earnings_explosiveness_bucket"]
        tier    = TIER_LABEL.get(bucket, bucket)
        hc      = bool(row.get("is_high_conviction", False))
        actual  = row.get("abs_reaction_3d")
        signed  = row.get("reaction_3d")

        print(f"  {row['stock']:<8} {row['earnings_date'].strftime('%b %-d'):<8} "
              f"{tier:<12} {row['risk_score']:>5.0f}  {_fmt(signed):>7}  {'★' if hc else ''}")

        rows.append({
            "week":             week.date(),
            "stock":            row["stock"],
            "earnings_date":    row["earnings_date"].date(),
            "tier":             bucket,
            "risk_score":       row["risk_score"],
            "is_high_conviction": hc,
            "reaction_3d":      signed,
            "abs_reaction_3d":  actual,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    results = pd.DataFrame(rows)
    with_outcome = results[results["abs_reaction_3d"].notna()].copy()
    for t in THRESHOLDS:
        with_outcome[f"moved_{int(t*100)}pct"] = with_outcome["abs_reaction_3d"] >= t

    base_n  = len(with_outcome)
    base_5  = with_outcome["moved_5pct"].mean()
    base_8  = with_outcome["moved_8pct"].mean()
    base_10 = with_outcome["moved_10pct"].mean()
    base_avg = with_outcome["abs_reaction_3d"].mean()

    print(f"\n\n{'═' * 70}")
    print(f"  SUMMARY  ({base_n} events with completed outcomes, "
          f"{len(results) - base_n} pending)")
    print(f"  {'─' * 66}")
    print(f"  {'TIER':<16} {'N':>4}  {'Avg|Move|':>9}  {'>5%':>5}  {'>8%':>5}  {'>10%':>5}  {'Lift@8%':>7}")
    print(f"  {'─' * 66}")

    for tier in ["High Alert", "Elevated", "Normal"]:
        sub = with_outcome[with_outcome["tier"] == tier]
        if sub.empty:
            continue
        r8   = sub["moved_8pct"].mean()
        lift = r8 / base_8 if base_8 > 0 else float("nan")
        print(f"  {tier:<16} {len(sub):>4}  {sub['abs_reaction_3d'].mean()*100:>8.1f}%  "
              f"{sub['moved_5pct'].mean()*100:>4.0f}%  {r8*100:>4.0f}%  "
              f"{sub['moved_10pct'].mean()*100:>4.0f}%  {lift:>6.2f}x")

    print(f"  {'─' * 66}")
    print(f"  {'ALL':<16} {base_n:>4}  {base_avg*100:>8.1f}%  "
          f"{base_5*100:>4.0f}%  {base_8*100:>4.0f}%  {base_10*100:>4.0f}%  {'1.00x':>7}")

    hc_sub = with_outcome[with_outcome["is_high_conviction"]]
    if not hc_sub.empty:
        hc_r8   = hc_sub["moved_8pct"].mean()
        hc_lift = hc_r8 / base_8 if base_8 > 0 else float("nan")
        print(f"\n  {'★ High Conviction':<16} {len(hc_sub):>4}  {hc_sub['abs_reaction_3d'].mean()*100:>8.1f}%  "
              f"{hc_sub['moved_5pct'].mean()*100:>4.0f}%  {hc_r8*100:>4.0f}%  "
              f"{hc_sub['moved_10pct'].mean()*100:>4.0f}%  {hc_lift:>6.2f}x")

    big_moves   = with_outcome[with_outcome["moved_8pct"]]
    flagged_big = big_moves[big_moves["tier"].isin(["High Alert", "Elevated"])]
    if len(big_moves) > 0:
        print(f"\n  Capture rate (≥8% moves caught by HA+Elevated): "
              f"{len(flagged_big)}/{len(big_moves)}  ({len(flagged_big)/len(big_moves)*100:.0f}%)")

    # ── Confusion matrices ────────────────────────────────────────────────────
    for label, flagged_mask in [
        ("High Alert",         with_outcome["tier"] == "High Alert"),
        ("High Alert+Elevated", with_outcome["tier"].isin(["High Alert", "Elevated"])),
        ("High Conviction",    with_outcome["is_high_conviction"]),
    ]:
        for threshold, col in [(5, "moved_5pct"), (8, "moved_8pct"), (10, "moved_10pct")]:
            actual_pos  = with_outcome[col]
            tp = ( flagged_mask &  actual_pos).sum()
            fp = ( flagged_mask & ~actual_pos).sum()
            fn = (~flagged_mask &  actual_pos).sum()
            tn = (~flagged_mask & ~actual_pos).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            print(f"\n  Confusion matrix — predicted: {label}  |  actual: |move| ≥ {threshold}%")
            print(f"  {'':20}  {'Actual YES':>12}  {'Actual NO':>10}")
            print(f"  {'Predicted YES':<20}  {'TP':>5} = {tp:<5}  {'FP':>5} = {fp}")
            print(f"  {'Predicted NO':<20}  {'FN':>5} = {fn:<5}  {'TN':>5} = {tn}")
            print(f"  Precision {precision*100:.0f}%  |  Recall {recall*100:.0f}%")

    out_path = Path("testing/testing_results/weekly_prediction_quality.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    _save_confusion_matrix_png(with_outcome,
                               Path("testing/testing_results/confusion_matrix.png"))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=DEFAULT_START,
                        help="Start date (YYYY-MM-DD), default: 2026-04-13")
    args = parser.parse_args()
    run(start_date=args.start)
