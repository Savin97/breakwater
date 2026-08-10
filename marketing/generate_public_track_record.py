"""
Generate the delayed public track record used by the Harbor landing page.

The JSON remains compatible with the original recent_calls.json shape:
{
  "generated": "YYYY-MM-DD",
  "calls": [...]
}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from utilities.output_utilities import get_run_output_dir


DEFAULT_PARQUET = Path("output/full_df.parquet")
PUBLIC_BUCKETS = {"High Alert", "Elevated", "Normal"}
HIGH_ALERT_BUCKETS = {"High Alert"}
EXTREME_MOVE_PCT = 8.0


def _tier_for(row: pd.Series) -> tuple[str | None, str | None]:
    if bool(row.get("is_high_conviction", False)):
        return "hc", "HIGH CONVICTION ★"
    if row["earnings_explosiveness_bucket"] in HIGH_ALERT_BUCKETS:
        return "high", "HIGH ALERT"
    if row["earnings_explosiveness_bucket"] == "Elevated":
        return "mid", "ELEVATED"
    if row["earnings_explosiveness_bucket"] == "Normal":
        return "low", "NORMAL"
    return None, None


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def build_public_track_record(
    df: pd.DataFrame,
    *,
    weeks: int = 6,
    generated_at: pd.Timestamp | None = None,
) -> dict[str, Any]:
    generated_at = generated_at or pd.Timestamp.today()
    required = {
        "stock",
        "earnings_date",
        "earnings_explosiveness_bucket",
        "reaction_1d",
        "reaction_3d",
        "is_earnings_day",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cols = [
        "stock",
        "earnings_date",
        "earnings_explosiveness_bucket",
        "reaction_1d",
        "reaction_3d",
        "is_earnings_day",
    ]
    for optional in [
        "is_high_conviction",
        "earnings_explosiveness_score",
        "peer_percentile",
    ]:
        if optional in df.columns:
            cols.append(optional)

    earnings = df[df["is_earnings_day"] == 1][cols].copy()
    earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"], errors="coerce")
    earnings = earnings.dropna(subset=["earnings_date"])

    cutoff = generated_at.normalize() - pd.Timedelta(weeks=weeks)
    earnings = earnings[earnings["earnings_date"] >= cutoff]
    earnings = earnings[earnings["earnings_explosiveness_bucket"].isin(PUBLIC_BUCKETS)]

    earnings["best_reaction"] = earnings["reaction_3d"].fillna(earnings["reaction_1d"])
    earnings = earnings.dropna(subset=["best_reaction"])
    earnings["week_start"] = (
        earnings["earnings_date"]
        - pd.to_timedelta(earnings["earnings_date"].dt.dayofweek, unit="D")
    )

    records: list[dict[str, Any]] = []
    for _, row in earnings.iterrows():
        tier_class, tier_label = _tier_for(row)
        if tier_class is None:
            continue

        move_pct = float(row["best_reaction"]) * 100
        record: dict[str, Any] = {
            "earnings_date": row["earnings_date"].strftime("%Y-%m-%d"),
            "week_start": row["week_start"].strftime("%Y-%m-%d"),
            "ticker": row["stock"],
            "tier_class": tier_class,
            "tier_label": tier_label,
            "move_pct": move_pct,
            "abs_move_pct": abs(move_pct),
            "is_extreme_move": abs(move_pct) >= EXTREME_MOVE_PCT,
        }
        if "earnings_explosiveness_score" in row:
            record["risk_score"] = _safe_float(row["earnings_explosiveness_score"])
        if "peer_percentile" in row:
            record["peer_percentile"] = _safe_float(row["peer_percentile"])
        records.append(record)

    records.sort(key=lambda x: (x["earnings_date"], -abs(x["move_pct"])), reverse=True)
    high_alert = [r for r in records if r["tier_class"] in {"hc", "high"}]

    return {
        "generated": generated_at.strftime("%Y-%m-%d"),
        "description": "Delayed public record of completed Breakwater earnings risk calls.",
        "delay_policy": "Completed earnings events only; timely upcoming risk flags are reserved for the digest and dashboard.",
        "extreme_move_threshold_pct": EXTREME_MOVE_PCT,
        "summary": {
            "weeks": weeks,
            "total_calls": len(records),
            "high_alert_calls": len(high_alert),
            "high_alert_extreme_moves": sum(1 for r in high_alert if r["is_extreme_move"]),
        },
        "calls": records,
    }


def generate_public_track_record(
    parquet_path: Path = DEFAULT_PARQUET,
    output_path: Path | None = None,
    *,
    weeks: int = 6,
) -> Path:
    """output_path defaults to this run's timestamped output subfolder."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} not found")
    if output_path is None:
        output_path = Path(get_run_output_dir()) / "recent_calls.json"

    df = pd.read_parquet(parquet_path)
    data = build_public_track_record(df, weeks=weeks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Written {len(data['calls'])} calls -> {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--weeks", type=int, default=6)
    args = parser.parse_args(argv)

    try:
        generate_public_track_record(args.parquet, args.output, weeks=args.weeks)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0
