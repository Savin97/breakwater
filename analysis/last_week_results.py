"""
Last week's results: the calls we ARCHIVED for last week's events
(db/predictions.duckdb, view predictions_track_record — first 0.4.0 call per event) next
to each event's corrected outcome (`abs_reaction_3d_anchored` from the event frame).

    python -m analysis.last_week_results
    python -m analysis.last_week_results --lookback 2   # go back 2 weeks

Writes into this run's output folder:
  * last_week_results.txt — INTERNAL (tier, percentile, cutoff, ≥8% tally);
  * results_chart.png — for X (analysis/chart_last_week.py): the same archived calls,
    each at the size of its corrected move, largest first. Outcomes only, no tally.
    Nothing is drawn for a week without an archived 0.4.0 call.

What it deliberately does not do:
  * It never re-scores the past. A call is shown exactly as it was archived; a week with
    no archived 0.4.0 call shows nothing. Pre-0.4 calls (the legacy 0.3.1 model) are
    excluded — they are void for the 0.4.0 record.
  * It never uses the legacy report-date reaction, which mismeasures before-open
    reporters. An event without a verified announcement time has no corrected outcome and
    is listed as unresolved, never counted as a miss.
  * It posts nothing itself: the chart is a file to post by hand. The landing-page track
    record (marketing/generate_public_track_record.py) stays paused.
"""
import argparse
import os
from datetime import date, timedelta

import duckdb
import pandas as pd

import config
from analysis.chart_last_week import generate_last_week_chart
from feature_engineering.announcement_timing import TARGET_AVAILABLE
from scoring.model_b import HIGH_ALERT, ELEVATED, display_score, sort_by_risk
from utilities.db_utilities import read_track_record
from utilities.output_utilities import get_run_output_dir

EVENTS_PARQUET = "output/events_df.parquet"


def _week_bounds(lookback_weeks: int = 1, today=None) -> tuple[pd.Timestamp, pd.Timestamp]:
    today = pd.Timestamp(today if today is not None else date.today()).normalize()
    # On Sat/Sun snap forward to Monday so the formula sees the just-completed trading week
    if today.weekday() >= 5:
        today = today + timedelta(days=7 - today.weekday())
    monday = today - timedelta(days=today.weekday() + 7 * lookback_weeks)
    return monday, monday + timedelta(days=4)


def archived_calls(start, end, db_path=None, announcements=None) -> pd.DataFrame:
    """Earliest track-record-eligible publication per event reporting in [start, end].
    `announcements` (the event frame's announce_ts_ny / announce_ts_observed_at) lets a
    report-day publication count when it provably preceded the announcement."""
    db_path = db_path or config.PREDICTIONS_DB_PATH
    cols = ["stock", "earnings_date", "tier", "risk_score", "model_b_hist_mean_abs",
            "model_b_n_prior", "score_asof_date", "prediction_asof_date", "published_at",
            "generation_id", "revision", "model_version"]
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=cols)
    con = duckdb.connect(db_path, read_only=True)
    try:
        # From the table, not the stored view: an unmigrated DB keeps a stale view.
        calls = read_track_record(con, announcements)
    except duckdb.Error:
        calls = pd.DataFrame(columns=cols)
    finally:
        con.close()
    if calls.empty:
        return pd.DataFrame(columns=cols).assign(earnings_date=pd.to_datetime([]))
    calls["earnings_date"] = pd.to_datetime(calls["earnings_date"])
    out = calls[calls["earnings_date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    return out.reindex(columns=cols).reset_index(drop=True)


def results_table(calls: pd.DataFrame, events_df: pd.DataFrame) -> pd.DataFrame:
    """Archived calls joined to their corrected outcome (NaN where unresolved)."""
    ev = events_df[~events_df["is_pending"]][[
        "stock", "earnings_date", "announce_window", "reaction_3d_anchored",
        "abs_reaction_3d_anchored", "reaction_3d_anchored_status"]].copy()
    ev["earnings_date"] = pd.to_datetime(ev["earnings_date"])
    out = calls.merge(ev, on=["stock", "earnings_date"], how="left")
    resolved = out["reaction_3d_anchored_status"].eq(TARGET_AVAILABLE)
    out["moved_8pct"] = (out["abs_reaction_3d_anchored"]
                         >= config.EXTREME_EARNINGS_REACTION_THRESHOLD).where(resolved)
    return sort_by_risk(out.rename(columns={"tier": "risk_tier"})).rename(
        columns={"risk_tier": "tier"})


def format_results(tbl: pd.DataFrame, week_start, week_end) -> str:
    week_range = f"{week_start.strftime('%b %-d')} – {week_end.strftime('%b %-d')}"
    lines = [f"LAST WEEK'S RESULTS  ({week_range})  — INTERNAL, not for publication",
             "═" * 74]
    if tbl.empty:
        lines += ["  No archived 0.4.0 call for an event reporting this week.",
                  "  (Calls made by earlier model versions are not shown: void for the "
                  "0.4.0 record.)"]
        return "\n".join(lines)
    lines.append(f"  {'':6}  {'DATE':7}  {'TIER':11} {'PCTL':>4}  {'CALL ASOF':10}  "
                 f"{'WINDOW':7} {'3-SESSION':>9}  ≥8%")
    lines.append("  " + "─" * 72)
    for _, r in tbl.iterrows():
        pct = display_score(r["risk_score"])
        move = r["reaction_3d_anchored"]
        move_s = f"{move * 100:+8.1f}%" if pd.notna(move) else "  unresol."
        hit = "?" if pd.isna(r["moved_8pct"]) else ("✓" if r["moved_8pct"] else "·")
        lines.append(
            f"  {r['stock']:<6}  {r['earnings_date'].strftime('%b %-d'):<7}  {r['tier']:<11} "
            f"{('—' if pct is None else str(pct)):>4}  {str(r['score_asof_date'])[:10]:10}  "
            f"{str(r['announce_window'] if pd.notna(r['announce_window']) else '—'):7} "
            f"{move_s:>9}  {hit:>3}")
    lines.append("  " + "─" * 72)
    lines.append("  3-session = corrected move from the last close before the announcement.")
    lines.append("  unresol. / ? = no verified announcement time (or window not closed): "
                 "not counted either way.")
    for label, tiers in [("High Alert", [HIGH_ALERT]), ("High Alert + Elevated", [HIGH_ALERT, ELEVATED])]:
        sel = tbl[tbl["tier"].isin(tiers) & tbl["moved_8pct"].notna()]
        if len(sel):
            k = int(sel["moved_8pct"].sum())
            lines.append(f"  {label}: {k} of {len(sel)} resolved calls moved ≥ 8% "
                         f"(one week — too few events to judge the model by).")
    return "\n".join(lines)


def generate_last_week_results(lookback_weeks: int = 1, events_df: pd.DataFrame | None = None,
                               db_path=None, today=None) -> str:
    if events_df is None:
        events_df = pd.read_parquet(EVENTS_PARQUET)
    week_start, week_end = _week_bounds(lookback_weeks, today)
    announcements = events_df[[c for c in ("stock", "earnings_date", "announce_ts_ny",
                                           "announce_ts_observed_at") if c in events_df.columns]]
    if "announce_ts_observed_at" not in announcements.columns:
        announcements = None
    tbl = results_table(archived_calls(week_start, week_end, db_path, announcements), events_df)
    text = format_results(tbl, week_start, week_end)
    path = os.path.join(get_run_output_dir(), "last_week_results.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"Saved last week's results → {path} ({len(tbl)} archived 0.4.0 calls)")
    generate_last_week_chart(tbl, week_start)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=1,
                        help="How many weeks back (default: 1 = last week)")
    args = parser.parse_args()
    print(generate_last_week_results(lookback_weeks=args.lookback))
