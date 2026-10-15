"""
Generate the delayed public track record used by the Harbor landing page.

================================================================================
PAUSED — THIS GENERATOR REFUSES TO RUN.
================================================================================
Model B (MODEL_VERSION 0.4.0) replaced the legacy 0.3.1 scorer on 2026-10-04. A public
track record must be PROSPECTIVE evidence: calls archived before the event, by the model
being advertised, scored afterwards against the corrected target. None exists yet.
Historical Model B backtest figures are research (analysis/model_c_refit/RESULTS.md) and
must never be presented as live performance.

What this publishes when un-paused — and nothing else:
  * calls from `predictions_track_record` in db/predictions.duckdb: the FIRST archived
    call per event by a version in config.TRACK_RECORD_MODEL_VERSIONS (0.4.0), scored
    from data before the event. Pre-0.4 calls (0.3.1, 0.3.1-preaudit — audit P4.2) are
    excluded by that view and again by `drop_voided()`.
  * each call's outcome on the validation target: `abs_reaction_3d_anchored >= 0.08`
    from the event frame. An event without a verified announcement time has no
    corrected outcome and is left out, never counted as a miss.

HOW TO UN-PAUSE — all must be true first:
  1. Enough prospective 0.4.0 calls have resolved to say something (decide the number
     before looking at the results).
  2. The figures to publish have been reviewed against audit/PHASE0_AUDIT_REV2.md's
     rules (stratified by announcement window; no crude lift).
  3. Michael has decided to publish.
Then set TRACK_RECORD_PAUSED = False below and update this header.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

import config
from feature_engineering.announcement_timing import TARGET_AVAILABLE
from utilities.db_utilities import read_track_record
from utilities.output_utilities import get_run_output_dir

EVENTS_PARQUET = Path("output/events_df.parquet")
PUBLIC_TIERS = {"High Alert", "Elevated", "Normal"}

# ---------------------------------------------------------------------------
# Pause. Flip to False ONLY when the header checklist above is satisfied.
# ---------------------------------------------------------------------------
TRACK_RECORD_PAUSED = True

# The deliberate, explicit override. An env var alone is not enough: it must carry this
# exact token, so nothing can re-enable publication by setting a plausible-looking flag.
PAUSE_OVERRIDE_ENV = "BREAKWATER_PUBLISH_TRACK_RECORD"
PAUSE_OVERRIDE_TOKEN = "yes-publish-unreviewed-0.4.0-calls"

PAUSE_MESSAGE = (
    "\n"
    "==============================================================================\n"
    " PUBLIC TRACK RECORD GENERATOR IS PAUSED — nothing written.\n"
    "==============================================================================\n"
    " Reason : Model B (0.4.0) has no prospective record yet. Pre-0.4 calls are void\n"
    "          for the 0.4.0 record, and historical backtests are not live calls.\n"
    " Unpause: see the checklist at the top of\n"
    "          marketing/generate_public_track_record.py.\n"
    f" Override (deliberate only): {PAUSE_OVERRIDE_ENV}={PAUSE_OVERRIDE_TOKEN}\n"
    "=============================================================================="
)


class TrackRecordPaused(RuntimeError):
    """Raised when something asks for a public track record while it is paused."""


def pause_reason() -> str | None:
    """The reason publication is refused, or None if this generator may run.

    Callers that must not die (the pipeline's stage 5) check this and skip; callers
    that exist only to publish raise/exit on it.
    """
    if not TRACK_RECORD_PAUSED:
        return None
    if os.environ.get(PAUSE_OVERRIDE_ENV, "") == PAUSE_OVERRIDE_TOKEN:
        print(f"WARNING: {PAUSE_OVERRIDE_ENV} is set — generating an unreviewed track "
              "record.", file=sys.stderr)
        return None
    return PAUSE_MESSAGE


# ---------------------------------------------------------------------------
# Voided and pre-0.4 calls are filtered out here, by flag and by version, so
# publication never depends on a human remembering which rows are excluded.
# ---------------------------------------------------------------------------
VOID_FLAG_COLUMN = "void_for_track_record"


def drop_voided(df: pd.DataFrame) -> pd.DataFrame:
    """Remove every call voided for track-record purposes (audit P4.2) and every call by
    a model version outside config.TRACK_RECORD_MODEL_VERSIONS. Voided calls stay in
    db/predictions.duckdb as history — excluded means "never counts toward a published
    record", not "deleted". A frame without `model_version` is refused: an unlabelled
    call cannot be shown to come from the advertised model."""
    if "model_version" not in df.columns:
        raise ValueError("calls carry no model_version — cannot tell which model made them")
    keep = df["model_version"].astype("string").isin(config.TRACK_RECORD_MODEL_VERSIONS).fillna(False)
    if VOID_FLAG_COLUMN in df.columns:
        keep &= ~df[VOID_FLAG_COLUMN].fillna(False).astype(bool)
    dropped = int((~keep).sum())
    if dropped:
        print(f"Excluded {dropped} call(s): voided or not a track-record model version.")
    return df[keep]


def load_archived_calls(db_path: str | None = None,
                        announcements: pd.DataFrame | None = None) -> pd.DataFrame:
    """Earliest eligible publication per event (utilities/db_utilities.read_track_record):
    published before the observed announcement, or before the report day where the time
    is unknown."""
    con = duckdb.connect(db_path or config.PREDICTIONS_DB_PATH, read_only=True)
    try:
        # From the table, not the stored view: an unmigrated DB keeps a stale view.
        return read_track_record(con, announcements)
    finally:
        con.close()


def build_public_track_record(
    calls: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    weeks: int = 6,
    generated_at: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Archived 0.4.0 calls for events in the last `weeks`, each with its corrected
    outcome. Calls without a corrected outcome are left out."""
    reason = pause_reason()
    if reason is not None:
        raise TrackRecordPaused(reason)

    generated_at = generated_at or pd.Timestamp.today()
    calls = drop_voided(calls).copy()
    calls["earnings_date"] = pd.to_datetime(calls["earnings_date"])
    cutoff = generated_at.normalize() - pd.Timedelta(weeks=weeks)
    calls = calls[(calls["earnings_date"] >= cutoff)
                  & (calls["earnings_date"] < generated_at.normalize())
                  & calls["tier"].isin(PUBLIC_TIERS)]

    ev = events_df[~events_df["is_pending"]][[
        "stock", "earnings_date", "reaction_3d_anchored", "reaction_3d_anchored_status"]].copy()
    ev["earnings_date"] = pd.to_datetime(ev["earnings_date"])
    rows = calls.merge(ev, on=["stock", "earnings_date"], how="inner")
    rows = rows[rows["reaction_3d_anchored_status"].eq(TARGET_AVAILABLE)]

    threshold_pct = config.EXTREME_EARNINGS_REACTION_THRESHOLD * 100
    records: list[dict[str, Any]] = []
    for _, r in rows.iterrows():
        move_pct = float(r["reaction_3d_anchored"]) * 100
        week_start = r["earnings_date"] - pd.Timedelta(days=r["earnings_date"].dayofweek)
        records.append({
            "earnings_date": r["earnings_date"].strftime("%Y-%m-%d"),
            "week_start": week_start.strftime("%Y-%m-%d"),
            "ticker": r["stock"],
            "tier_label": str(r["tier"]).upper(),
            "risk_percentile": None if pd.isna(r["risk_score"]) else float(r["risk_score"]),
            "call_asof": str(r["prediction_asof_date"])[:10],
            "model_version": str(r["model_version"]),
            "move_pct": move_pct,
            "abs_move_pct": abs(move_pct),
            "is_extreme_move": abs(move_pct) >= threshold_pct,
        })
    records.sort(key=lambda x: (x["earnings_date"], -x["abs_move_pct"]), reverse=True)
    return {
        "generated": generated_at.strftime("%Y-%m-%d"),
        "description": "Delayed public record of completed Breakwater earnings risk calls, "
                       "as archived before each event.",
        "delay_policy": "Completed earnings events only; upcoming calls are reserved for the "
                        "digest and dashboard.",
        "extreme_move_threshold_pct": threshold_pct,
        "calls": records,
    }


def generate_public_track_record(
    output_path: Path | None = None,
    *,
    weeks: int = 6,
    events_df: pd.DataFrame | None = None,
    calls: pd.DataFrame | None = None,
) -> Path | None:
    """Returns None without writing anything while the pause is in force. It prints and
    returns rather than raising because pipeline/stage5.py calls it in the middle of a
    run that also produces the PDFs, the calendar and the dashboard parquets — the pause
    must stop publication, not the rest of the product."""
    reason = pause_reason()
    if reason is not None:
        print(reason, file=sys.stderr)
        return None

    output_path = output_path or Path(get_run_output_dir()) / "recent_calls.json"
    if events_df is None:
        events_df = pd.read_parquet(EVENTS_PARQUET)
    if calls is None:
        timing = ["stock", "earnings_date", "announce_ts_ny", "announce_ts_observed_at"]
        calls = load_archived_calls(announcements=events_df[timing]
                                    if set(timing) <= set(events_df.columns) else None)
    data = build_public_track_record(calls, events_df, weeks=weeks)
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
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--weeks", type=int, default=6)
    args = parser.parse_args(argv)
    try:
        generate_public_track_record(args.output, weeks=args.weeks)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    # Without this, `python -m marketing.generate_public_track_record` imported the
    # module, ran nothing and exited 0 — a paused generator that looks like it
    # succeeded is worse than no pause at all. Exit code 2 = refused, see main().
    sys.exit(main())
