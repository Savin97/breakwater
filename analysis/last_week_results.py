"""
Last week's earnings results: prints table + saves results_chart.png.

Usage:
    python analysis/last_week_results.py
    python analysis/last_week_results.py --lookback 2   # go back 2 weeks
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import pandas as pd
from datetime import date, timedelta

from analysis.chart_results import generate_results_chart

PARQUET          = os.path.join(os.path.dirname(__file__), "..", "output", "full_df.parquet")
MOVED_THRESHOLD  = 0.05


def _week_bounds(lookback_weeks: int = 1) -> tuple[pd.Timestamp, pd.Timestamp]:
    today   = pd.Timestamp(date.today())
    monday  = today - timedelta(days=today.weekday() + 7 * lookback_weeks)
    return monday, monday + timedelta(days=4)


def _fmt(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "   n/a"
    sign = "+" if val >= 0 else "–"
    return f"{sign}{abs(val) * 100:4.1f}%"


def _surrounding_daily_rets(stock_df, earn_pos, offsets):
    out = {}
    for o in offsets:
        idx = earn_pos + o
        if 0 <= idx < len(stock_df):
            val = stock_df.at[idx, "daily_ret"]
            out[o] = float(val) if pd.notna(val) else None
        else:
            out[o] = None
    return out


def _cumulative(daily_rets_post):
    cum     = {}
    product = 1.0
    for k in sorted(daily_rets_post):
        r = daily_rets_post[k]
        if r is None:
            break
        product *= (1 + r)
        cum[k]   = product - 1
    return cum


def print_last_week_results(lookback_weeks: int = 1):
    df = pd.read_parquet(PARQUET)
    df["date"]          = pd.to_datetime(df["date"])
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])

    week_start, week_end = _week_bounds(lookback_weeks)

    earnings = df[
        (df["is_earnings_day"] == 1)
        & (df["earnings_date"] >= week_start)
        & (df["earnings_date"] <= week_end)
        & df["earnings_explosiveness_bucket"].notna()
    ].copy()

    if earnings.empty:
        print("No earnings events found for that week.")
        return

    bucket_order = {"High Alert": 0, "Elevated": 1, "Normal": 2}
    earnings["_rank"] = earnings["earnings_explosiveness_bucket"].map(bucket_order).fillna(9)
    earnings = earnings.sort_values(["_rank", "earnings_date", "stock"])

    stock_histories = {
        stock: grp.sort_values("date").reset_index(drop=True)
        for stock, grp in df.groupby("stock")
    }

    week_range = f"{week_start.strftime('%b %-d')} – {week_end.strftime('%b %-d')}"
    print(f"\nLAST WEEK'S RESULTS  ({week_range})")
    print("═" * 70)
    header = (f"  {'':6}  {'DATE':7}  {'TIER':12}  "
              f"{'–2d':>6} {'–1d':>6} {'ERN':>6}  "
              f"{'  +1d':>6} {'  +2d':>6} {'  +3d':>6} {'  +4d':>6} {'  +5d':>6}")
    print(header)
    print("  " + "─" * 68)

    chart_rows = []
    flagged_moved = flagged_total = ha_moved = ha_total = 0

    for _, row in earnings.iterrows():
        stock  = row["stock"]
        edate  = row["earnings_date"]
        bucket = row["earnings_explosiveness_bucket"]

        s = stock_histories.get(stock)
        if s is None:
            continue

        earn_positions = s.index[s["date"] == edate].tolist()
        if not earn_positions:
            continue
        pos = earn_positions[0]

        pre       = _surrounding_daily_rets(s, pos, [-2, -1, 0])
        post_daily = _surrounding_daily_rets(s, pos, [1, 2, 3, 4, 5])
        post_cum  = _cumulative(post_daily)

        abs_3d = post_cum.get(3)
        moved  = abs_3d is not None and abs(abs_3d) >= MOVED_THRESHOLD

        if bucket in ("High Alert", "Elevated"):
            flagged_total += 1
            if moved:
                flagged_moved += 1
        if bucket == "High Alert":
            ha_total += 1
            if moved:
                ha_moved += 1

        tier_short = {"High Alert": "HIGH ALERT", "Elevated": "ELEVATED",
                      "Normal": "NORMAL"}.get(bucket, bucket)
        date_str  = edate.strftime("%b %-d")
        pre_fmt   = " ".join(_fmt(pre.get(o)) for o in [-2, -1, 0])
        post_fmt  = " ".join(_fmt(post_cum.get(k)) for k in [1, 2, 3, 4, 5])
        print(f"  {stock:<6}  {date_str:<7}  {tier_short:<12}  {pre_fmt}  {post_fmt}")

        move = row["reaction_3d"] if pd.notna(row.get("reaction_3d")) else row.get("reaction_1d")
        if pd.notna(move):
            chart_rows.append({
                "stock":                      stock,
                "earnings_date":              edate,
                "earnings_explosiveness_bucket": bucket,
                "earnings_explosiveness_score": row["earnings_explosiveness_score"],
                "is_high_conviction":         row.get("is_high_conviction", False),
                "move":                       float(move),
            })

    print("  " + "─" * 68)
    print(f"  Columns: pre-earnings (–2d –1d 0d) | cumulative post-earnings (+1d thru +5d)")
    print()

    if flagged_total > 0:
        print(f"  Hit rate >5% move (flagged):  {flagged_moved}/{flagged_total}  "
              f"({flagged_moved / flagged_total * 100:.0f}%)")
    if ha_total > 0:
        print(f"  High Alert hit rate:          {ha_moved}/{ha_total}  "
              f"({ha_moved / ha_total * 100:.0f}%)")
    print()

    if chart_rows:
        chart_df = pd.DataFrame(chart_rows)
        generate_results_chart(chart_df, week_start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=1,
                        help="How many weeks back (default: 1 = last week)")
    args = parser.parse_args()
    print_last_week_results(lookback_weeks=args.lookback)
