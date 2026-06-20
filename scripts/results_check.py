"""
Print last week's earnings outcomes vs. model predictions.
Shows price moves for days -2, -1, 0 (earnings day), +1 through +5.
Reads output/full_df.parquet — run after monday_workflow.sh (or sync_pipeline.sh).

Usage:
    python scripts/results_check.py
    python scripts/results_check.py --lookback 12    # widen window
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import pandas as pd
import numpy as np
from datetime import date, timedelta

PARQUET       = os.path.join(os.path.dirname(__file__), "..", "output", "full_df.parquet")
MOVED_THRESHOLD = 0.05   # 5% = "notable move"
DEFAULT_LOOKBACK = 9     # calendar days back from today


def _fmt(val):
    """Format a decimal return as a coloured %-string, or '  n/a' if missing."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "   n/a"
    sign = "+" if val >= 0 else "–"
    return f"{sign}{abs(val)*100:4.1f}%"


def _surrounding_daily_rets(stock_df, earn_pos, offsets):
    """
    Return {offset: daily_ret} for each requested offset around earn_pos.
    stock_df must be sorted by date and positionally indexed (reset_index).
    """
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
    """
    Build cumulative returns from individual daily returns.
    daily_rets_post: dict {1: r1, 2: r2, ...} (already floats or None).
    Returns {1: cum1, 2: cum1*cum2-1, ...}  stops accumulating at first None.
    """
    cum = {}
    product = 1.0
    for k in sorted(daily_rets_post):
        r = daily_rets_post[k]
        if r is None:
            break
        product *= (1 + r)
        cum[k] = product - 1
    return cum


def run(lookback_days=DEFAULT_LOOKBACK):
    df = pd.read_parquet(PARQUET)
    df["date"]          = pd.to_datetime(df["date"])
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])

    today  = pd.Timestamp(date.today())
    cutoff = today - timedelta(days=lookback_days)

    earnings = df[
        (df["is_earnings_day"] == 1)
        & (df["date"] >= cutoff)
        & (df["date"] < today)
        & df["earnings_explosiveness_bucket"].notna()
    ].copy()

    if earnings.empty:
        print("No earnings events found in the past window.")
        return

    week_start = earnings["earnings_date"].min().strftime("%b %-d")
    week_end   = earnings["earnings_date"].max().strftime("%b %-d")

    bucket_order = {"High Alert": 0, "Elevated": 1, "Normal": 2}
    earnings["_rank"] = earnings["earnings_explosiveness_bucket"].astype(str).map(bucket_order).fillna(9)
    earnings = earnings.sort_values(["_rank", "earnings_date", "stock"])

    # Pre-compute per-stock sorted price history for fast positional lookup
    stock_histories = {
        stock: grp.sort_values("date").reset_index(drop=True)
        for stock, grp in df.groupby("stock")
    }

    print(f"\nLAST WEEK'S RESULTS  ({week_start} – {week_end})")
    print("═" * 70)
    header = f"  {'':6}  {'DATE':7}  {'TIER':12}  {'–2d':>6} {'–1d':>6} {'ERN':>6}  {'  +1d':>6} {'  +2d':>6} {'  +3d':>6} {'  +4d':>6} {'  +5d':>6}"
    print(header)
    print("  " + "─" * 68)

    flagged_moved = 0
    flagged_total = 0
    ha_moved = 0
    ha_total = 0

    for _, row in earnings.iterrows():
        stock  = row["stock"]
        edate  = row["earnings_date"]
        bucket = row["earnings_explosiveness_bucket"]

        s = stock_histories.get(stock)
        if s is None:
            continue

        # Find earnings day position
        earn_positions = s.index[s["date"] == edate].tolist()
        if not earn_positions:
            continue
        pos = earn_positions[0]

        # Pre-earnings individual daily returns
        pre = _surrounding_daily_rets(s, pos, [-2, -1, 0])

        # Post-earnings individual daily returns → cumulative
        post_daily = _surrounding_daily_rets(s, pos, [1, 2, 3, 4, 5])
        post_cum   = _cumulative(post_daily)

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

        tier_short = {"High Alert": "HIGH ALERT", "Elevated": "ELEVATED", "Normal": "NORMAL"}.get(bucket, bucket)
        date_str   = edate.strftime("%b %-d")

        pre_fmt  = " ".join(_fmt(pre.get(o)) for o in [-2, -1, 0])
        post_fmt = " ".join(_fmt(post_cum.get(k)) for k in [1, 2, 3, 4, 5])

        print(f"  {stock:<6}  {date_str:<7}  {tier_short:<12}  {pre_fmt}  {post_fmt}")

    print("  " + "─" * 68)
    print(f"  Columns: pre-earnings (-2d -1d 0d) | cumulative post-earnings (+1d thru +5d)")
    print()

    if flagged_total > 0:
        hit_pct = flagged_moved / flagged_total * 100
        print(f"  Hit rate >5% move (flagged):  {flagged_moved}/{flagged_total}  ({hit_pct:.0f}%)")
    if ha_total > 0:
        ha_pct = ha_moved / ha_total * 100
        print(f"  High Alert hit rate:          {ha_moved}/{ha_total}  ({ha_pct:.0f}%)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK,
                        help="Calendar days back from today to search for earnings events")
    args = parser.parse_args()
    run(lookback_days=args.lookback)
