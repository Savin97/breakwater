"""
Generate the delayed public track record used by the Harbor landing page.

================================================================================
PAUSED — audit remediation item P4.3. THIS GENERATOR REFUSES TO RUN.
================================================================================
Every score, tier and lift it would publish was produced by the pre-audit feature
chain, which measured before-open (BMO) earnings reactions on the wrong sessions and
carried each call's tier from the stock's PREVIOUS completed event. A public track
record built from those calls would read as validation while measuring nothing.
See `audit/PHASE0_AUDIT_REV2.md` — Phase 4 (P4.1 retire published lift figures,
P4.2 void the archived pre-audit calls, P4.3 this pause).

HOW TO UN-PAUSE — all of these must be true FIRST:
  1. P3.1 — the whole feature chain rebuilt on verified announcement timestamps.
  2. P3.2 — thresholds re-fit (73/79, LIFT_TO_*, LIFT_PRIOR_STRENGTH, the 0.12
     ceiling, the 0.85/0.15 weights). Nothing about the corrected model is
     claimable before this.
  3. P3.3 — calibration stratified by announcement window, stratified lift as the
     headline number. This is the item the pause is formally blocked on.
  4. The source this publishes from contains post-rebuild calls only. Calls voided
     under P4.2 (`model_version = "0.3.1-preaudit"`, `void_for_track_record = TRUE`
     in db/predictions.duckdb) must stay excluded — see `drop_voided()` below and
     the `predictions_track_record` view in utilities/db_utilities.py.
Then, and only then, set TRACK_RECORD_PAUSED = False below and update this header.

The JSON remains compatible with the original recent_calls.json shape:
{
  "generated": "YYYY-MM-DD",
  "calls": [...]
}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from utilities.output_utilities import get_run_output_dir


DEFAULT_PARQUET = Path("output/full_df.parquet")
PUBLIC_BUCKETS = {"High Alert", "Elevated", "Normal"}
HIGH_ALERT_BUCKETS = {"High Alert"}
EXTREME_MOVE_PCT = 8.0

# ---------------------------------------------------------------------------
# P4.3 pause. Flip to False ONLY when the header checklist above is satisfied.
# ---------------------------------------------------------------------------
TRACK_RECORD_PAUSED = True

# The deliberate, explicit override. An env var alone is not enough: it must carry
# this exact token, so nothing can re-enable publication by setting a plausible-looking
# flag, and nobody can claim it happened by accident. Using it publishes numbers the
# audit has declared void.
PAUSE_OVERRIDE_ENV = "BREAKWATER_PUBLISH_VOID_TRACK_RECORD"
PAUSE_OVERRIDE_TOKEN = "yes-publish-preaudit-numbers-i-know-they-are-void"

PAUSE_MESSAGE = (
    "\n"
    "==============================================================================\n"
    " PUBLIC TRACK RECORD GENERATOR IS PAUSED (audit item P4.3) — nothing written.\n"
    "==============================================================================\n"
    " Reason : its inputs come from the pre-audit feature chain. Before-open (BMO)\n"
    "          earnings reactions were measured on the wrong sessions, so every\n"
    "          score, tier and lift downstream of them is invalid for publication.\n"
    " Blocked: on P3.3 (calibration stratified by announcement window), which in\n"
    "          turn needs P3.1 (chain rebuilt) and P3.2 (thresholds re-fit).\n"
    " Read   : audit/PHASE0_AUDIT_REV2.md, Phase 3 and Phase 4.\n"
    " Unpause: set TRACK_RECORD_PAUSED = False in\n"
    "          marketing/generate_public_track_record.py after the header\n"
    "          checklist in that file is genuinely satisfied.\n"
    f" Override (publishes void numbers, deliberate only): {PAUSE_OVERRIDE_ENV}"
    f"={PAUSE_OVERRIDE_TOKEN}\n"
    "=============================================================================="
)


class TrackRecordPaused(RuntimeError):
    """Raised when something asks for a public track record while P4.3 is in force."""


def pause_reason() -> str | None:
    """The reason publication is refused, or None if this generator may run.

    Callers that must not die (the pipeline's stage 5) check this and skip; callers
    that exist only to publish raise/exit on it.
    """
    if not TRACK_RECORD_PAUSED:
        return None
    if os.environ.get(PAUSE_OVERRIDE_ENV, "") == PAUSE_OVERRIDE_TOKEN:
        print(
            f"WARNING: {PAUSE_OVERRIDE_ENV} is set — generating a track record from "
            "pre-audit numbers that audit/PHASE0_AUDIT_REV2.md declares void (P4.1).",
            file=sys.stderr,
        )
        return None
    return PAUSE_MESSAGE


# ---------------------------------------------------------------------------
# P4.2 — calls voided for track-record purposes are filtered out here, by flag,
# so publication never depends on a human remembering which rows are poisoned.
# ---------------------------------------------------------------------------
VOID_FLAG_COLUMN = "void_for_track_record"
VOIDED_MODEL_VERSIONS = frozenset({"0.3.1-preaudit"})


def drop_voided(df: pd.DataFrame) -> pd.DataFrame:
    """Remove every call voided for track-record purposes (audit P4.2).

    Two independent markings, either of which is sufficient, because different inputs
    carry different columns: the boolean `void_for_track_record` flag and the
    `model_version` stamp. Voided calls are preserved in db/predictions.duckdb as
    history — void means "never counts towards a published record", not "deleted".
    Frames carrying neither column pass through untouched.
    """
    keep = pd.Series(True, index=df.index)
    if VOID_FLAG_COLUMN in df.columns:
        keep &= ~df[VOID_FLAG_COLUMN].fillna(False).astype(bool)
    if "model_version" in df.columns:
        keep &= ~df["model_version"].astype("string").isin(VOIDED_MODEL_VERSIONS).fillna(False)
    dropped = int((~keep).sum())
    if dropped:
        print(f"Excluded {dropped} call(s) voided for track record (audit P4.2).")
    return df[keep]


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
    reason = pause_reason()
    if reason is not None:
        raise TrackRecordPaused(reason)

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

    df = drop_voided(df)
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
    df: pd.DataFrame | None = None,
) -> Path | None:
    """output_path defaults to this run's timestamped output subfolder.

    Returns None without writing anything while the P4.3 pause is in force. It prints
    and returns rather than raising because pipeline/stage5.py calls this in the middle
    of a run that also produces the PDFs, the calendar and the dashboard parquets — the
    pause must stop publication, not the rest of the product.
    """
    reason = pause_reason()
    if reason is not None:
        print(reason, file=sys.stderr)
        return None

    if output_path is None:
        output_path = Path(get_run_output_dir()) / "recent_calls.json"

    if df is None:
        if not parquet_path.exists():
            raise FileNotFoundError(f"{parquet_path} not found")
        df = pd.read_parquet(parquet_path)
    data = build_public_track_record(df, weeks=weeks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
    print(f"Written {len(data['calls'])} calls -> {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    reason = pause_reason()
    if reason is not None:
        print(reason, file=sys.stderr)
        return 2

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


if __name__ == "__main__":
    # Without this, `python -m marketing.generate_public_track_record` imported the
    # module, ran nothing and exited 0 — a paused generator that looks like it
    # succeeded is worse than no pause at all. Exit code 2 = refused, see main().
    sys.exit(main())
