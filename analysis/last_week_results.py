"""
Last week's results, for internal review: the calls we ARCHIVED for last week's events
(db/predictions.duckdb, view predictions_track_record — first 0.4.0 call per event) next
to each event's corrected outcome (`abs_reaction_3d_anchored` from the event frame).

    python -m analysis.last_week_results
    python -m analysis.last_week_results --lookback 2   # go back 2 weeks

Writes last_week_results.txt into this run's output folder. Nothing else.

What it deliberately does not do:
  * It never re-scores the past. A call is shown exactly as it was archived; a week with
    no archived 0.4.0 call shows nothing. Pre-0.4 calls (the legacy 0.3.1 model) are
    excluded — they are void for the 0.4.0 record.
  * It never uses the legacy report-date reaction, which mismeasures before-open
    reporters. An event without a verified announcement time has no corrected outcome and
    is listed as unresolved, never counted as a miss.
  * It publishes nothing. The public track record is paused
    (marketing/generate_public_track_record.py); the social results chart this module
    used to draw was retired with the 0.3.1 model.
"""
import argparse
import os
from datetime import date, timedelta

import duckdb
import pandas as pd

import config
from feature_engineering.announcement_timing import TARGET_AVAILABLE
from scoring.model_b import HIGH_ALERT, ELEVATED, display_score, sort_by_risk
from utilities.output_utilities import get_run_output_dir

EVENTS_PARQUET = "output/events_df.parquet"


def _week_bounds(lookback_weeks: int = 1, today=None) -> tuple[pd.Timestamp, pd.Timestamp]:
    today = pd.Timestamp(today if today is not None else date.today()).normalize()
    # On Sat/Sun snap forward to Monday so the formula sees the just-completed trading week
    if today.weekday() >= 5:
        today = today + timedelta(days=7 - today.weekday())
    monday = today - timedelta(days=today.weekday() + 7 * lookback_weeks)
    return monday, monday + timedelta(days=4)


def archived_calls(start, end, db_path=None) -> pd.DataFrame:
    """First archived track-record-eligible call per event reporting in [start, end]."""
    db_path = db_path or config.PREDICTIONS_DB_PATH
    cols = ["stock", "earnings_date", "tier", "risk_score", "model_b_hist_mean_abs",
            "model_b_n_prior", "score_asof_date", "prediction_asof_date", "model_version"]
    if not os.path.exists(db_path):
        return pd.DataFrame(columns=cols)
    con = duckdb.connect(db_path, read_only=True)
    try:
        out = con.execute(f"""
            SELECT {', '.join(cols)} FROM predictions_track_record
            WHERE earnings_date BETWEEN ? AND ?
        """, [pd.Timestamp(start).date(), pd.Timestamp(end).date()]).df()
    except duckdb.Error:          # a DB that predates the 0.4.0 view / columns
        out = pd.DataFrame(columns=cols)
    finally:
        con.close()
    out["earnings_date"] = pd.to_datetime(out["earnings_date"])
    return out


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
    tbl = results_table(archived_calls(week_start, week_end, db_path), events_df)
    text = format_results(tbl, week_start, week_end)
    path = os.path.join(get_run_output_dir(), "last_week_results.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"Saved last week's results → {path} ({len(tbl)} archived 0.4.0 calls)")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=1,
                        help="How many weeks back (default: 1 = last week)")
    args = parser.parse_args()
    print(generate_last_week_results(lookback_weeks=args.lookback))
