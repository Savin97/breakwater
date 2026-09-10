"""
Predictions for a range of whole Mon-Fri work weeks — backward, forward, or from an
explicit Monday.

    python -m analysis.predictions_range --weeks-back 12
    python -m analysis.predictions_range --weeks-forward 4
    python -m analysis.predictions_range --monday 2026-06-01 --weeks 3

The model is 0.3.1 exactly as it ships: the legacy `abs_reaction_3d` target, the
`config.py` bucket floors and lift gates, nothing from the methodology rebuild's
corrected target. The Phase 2 `*_anchored` columns are deliberately NOT reported here —
they are a parallel measurement that no threshold has been re-fit against, and mixing
them into a predictions sheet would invite reading them as the model's own output.

Two directions, two different things, and they are not equally trustworthy:

HISTORY (`is_pending == 0`).  The tier/score a completed event carries. Event-level
statistics are causal — every one is an expanding/shift(1) aggregate over that stock's
PRIOR events only — so this is close to what the model would have said going in. It is
still a RETRO-SCORE, not an archive: a few columns carried from the daily frame
(per-date cross-sectional ranks, the global quantile inside score_momentum_fragility)
are computed over the whole frame, so they know about days after the event. Where the
real published call exists it is shown alongside as `published_tier` — that archive
starts 2026-08-31, so it is empty for anything older.

Phase 1 changed nothing here: `assert_completed_parity` proves every completed event's
columns are identical to what the pre-audit daily pipeline produced.

UPCOMING (`is_pending == 1`).  Reported twice, because the audit changed what an
upcoming call says:
  `tier` / `risk_score`                    — the event frame's pending row: this stock's
                                             history through its most recent COMPLETED
                                             event.
  `tier_pre_audit` / `risk_score_pre_audit` — `df.sort_values("date").groupby("stock")
                                             .last()` over the daily frame, reproducing
                                             master exactly. `.last()` skips NaN column
                                             by column, so it reaches back past the
                                             blank in-between days to the stock's LAST
                                             COMPLETED EVENT — one earnings event stale
                                             (audit/PHASE0_AUDIT_REV2.md §Q4). This is
                                             what was actually published before the fix.
`pre_audit_differs` marks the events where the two disagree.

There is no pre-audit column for history rows. Reproducing what `.last()` would have
returned in a past week needs the daily frame as it stood that week, which we do not
keep; today's frame would answer with the event that has since completed.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from datetime import date

import duckdb
import numpy as np
import pandas as pd

from config import (PREDICTIONS_DB_PATH, MODEL_VERSION,
                    LARGE_EARNINGS_REACTION_THRESHOLD,
                    EXTREME_EARNINGS_REACTION_THRESHOLD)
from utilities.data_utilities import week_block_window

EVENTS_PARQUET = "output/events_df.parquet"
DAILY_PARQUET  = "output/full_df.parquet"
# Its own directory, NOT get_run_output_dir(): that one wipes and recreates today's run
# folder on first call in a process, so writing there from a separate process would
# delete the pipeline run's reports. These sheets are also worth keeping across days.
OUTPUT_DIR     = "output/predictions"

BUCKET_ORDER = {"High Alert": 0, "Elevated": 1, "Normal": 2}

# Carried through from the event frame unchanged.
SCORE_COLS = [
    "earnings_explosiveness_bucket", "earnings_explosiveness_bucket_structural",
    "earnings_explosiveness_score", "risk_score", "stock_bucket_lift",
    "is_high_conviction", "pre_earnings_drift_flag", "surprise_momentum_flag",
]
OUTCOME_COLS = ["reaction_1d", "reaction_3d", "reaction_5d", "abs_reaction_3d"]

# The subset of the daily frame the pre-audit reproduction needs. `.last()` skips NaN
# independently per column, so reading a subset gives the same answer as reading it all —
# and full_df.parquet is ~320 MB.
PRE_AUDIT_COLS = ["stock", "date", "earnings_date", *SCORE_COLS]


def _pre_audit_upcoming(daily_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """master's upcoming-call selection, verbatim: sort by date, groupby stock, .last().

    Kept as its own function so the thing being reproduced is visible in one place. Do
    not "fix" the NaN skipping — the skipping IS the pre-audit behaviour being measured.
    """
    if daily_df is None:
        daily_df = pd.read_parquet(DAILY_PARQUET, columns=PRE_AUDIT_COLS)
    missing = [c for c in PRE_AUDIT_COLS if c not in daily_df.columns]
    if missing:
        # Blank rather than absent, so the caller still gets the columns and a visible
        # gap instead of a KeyError halfway through assembling the sheet.
        print(f"  NOTE: daily frame has no {', '.join(missing)} — "
              f"pre-audit columns left blank.")
        daily_df = daily_df.reindex(columns=[*daily_df.columns, *missing])
    latest = daily_df.sort_values("date").groupby("stock").last().reset_index()
    return latest.rename(columns={
        "earnings_date": "earnings_date_pre_audit",
        **{c: f"{c}_pre_audit" for c in SCORE_COLS},
    })


def _published_calls() -> pd.DataFrame:
    """The archived first call per event, where one exists. Missing DB is not an error —
    the archive only starts 2026-08-31 and most windows predate it."""
    empty = pd.DataFrame(columns=["stock", "earnings_date", "published_tier",
                                  "published_risk_score", "published_asof"])
    if not os.path.exists(PREDICTIONS_DB_PATH):
        return empty
    con = duckdb.connect(PREDICTIONS_DB_PATH, read_only=True)
    try:
        out = con.execute("""
            SELECT stock, earnings_date,
                   tier                 AS published_tier,
                   risk_score           AS published_risk_score,
                   prediction_asof_date AS published_asof
            FROM predictions_first_call
        """).df()
    except duckdb.Error:
        return empty
    finally:
        con.close()
    out["earnings_date"] = pd.to_datetime(out["earnings_date"])
    return out


def predictions_range(weeks_back: int | None = None, weeks_forward: int | None = None,
                      monday=None, weeks: int = 1, events_df: pd.DataFrame | None = None,
                      daily_df: pd.DataFrame | None = None, pre_audit: bool = True,
                      today=None) -> pd.DataFrame:
    """One row per earnings event in the window. See the module docstring for what the
    history and upcoming halves each mean — they are not the same kind of number."""
    start, end = week_block_window(weeks_back=weeks_back, weeks_forward=weeks_forward,
                                   monday=monday, weeks=weeks, today=today)
    if events_df is None:
        events_df = pd.read_parquet(EVENTS_PARQUET)
    ev = events_df.copy()
    ev["earnings_date"] = pd.to_datetime(ev["earnings_date"])
    now = pd.Timestamp(today if today is not None else date.today()).normalize()

    in_window = ev["earnings_date"].between(start, end)

    history = ev[in_window & ~ev["is_pending"]].copy()
    history["row_kind"] = "history"

    upcoming = ev[in_window & ev["is_pending"]].copy()
    # A pending row whose earnings_date has already passed is a stale feed, not a call:
    # the ticker stopped updating and still carries an old forward-merged date. Counted
    # out loud rather than dropped silently (pipeline/events.py makes the same noise).
    stale = upcoming[upcoming["earnings_date"] < now]
    if not stale.empty:
        print(f"  NOTE: dropped {len(stale)} pending rows whose earnings_date is already "
              f"past (stale feed): {', '.join(sorted(stale['stock']))}")
    upcoming = upcoming[upcoming["earnings_date"] >= now].copy()
    upcoming["row_kind"] = "upcoming"

    out = pd.concat([history, upcoming], ignore_index=True)
    if out.empty:
        print(f"No earnings events between {start.date()} and {end.date()}.")
        return out

    keep = ["stock", "sector", "earnings_date", "row_kind", "score_asof_date",
            *SCORE_COLS, *[c for c in OUTCOME_COLS if c in out.columns]]
    out = out[[c for c in keep if c in out.columns]].copy()
    out.insert(0, "week_start",
               out["earnings_date"] - pd.to_timedelta(out["earnings_date"].dt.weekday, unit="D"))
    out["days_to_earnings"] = np.where(
        out["row_kind"] == "upcoming", (out["earnings_date"] - now).dt.days, np.nan)

    # Realized outcome, legacy target only. abs_reaction_3d is the scored target; fall
    # back to |reaction_3d| for any event where only the signed column survived.
    moved = out["abs_reaction_3d"] if "abs_reaction_3d" in out.columns else pd.Series(np.nan, index=out.index)
    if "reaction_3d" in out.columns:
        moved = moved.fillna(out["reaction_3d"].abs())
    out["moved_5pct"] = np.where(moved.notna(), moved >= LARGE_EARNINGS_REACTION_THRESHOLD, None)
    out["moved_8pct"] = np.where(moved.notna(), moved >= EXTREME_EARNINGS_REACTION_THRESHOLD, None)

    if pre_audit and (out["row_kind"] == "upcoming").any():
        pa = _pre_audit_upcoming(daily_df)
        out = out.merge(pa.drop(columns=["date"]), on="stock", how="left")
        # A pending row and master's .last() row both come off the stock's final daily
        # row, so the event they name must agree. Blank the reproduction where it does
        # not rather than align a call to the wrong event.
        wrong_event = (out["row_kind"] == "upcoming") & out["earnings_date_pre_audit"].notna() & \
                      (out["earnings_date_pre_audit"] != out["earnings_date"])
        if wrong_event.any():
            print(f"  NOTE: {int(wrong_event.sum())} upcoming events where the pre-audit "
                  f"selection names a different earnings date; left blank.")
        pa_cols = [f"{c}_pre_audit" for c in SCORE_COLS]
        # object dtype first: is_high_conviction_pre_audit arrives as bool, which cannot
        # hold the blank that says "no pre-audit call exists for this row".
        out[pa_cols] = out[pa_cols].astype(object)
        out.loc[wrong_event | (out["row_kind"] == "history"), pa_cols] = None
        out["pre_audit_differs"] = np.where(
            (out["row_kind"] == "upcoming")
            & out["earnings_explosiveness_bucket_pre_audit"].notna(),
            out["earnings_explosiveness_bucket_pre_audit"].astype(str)
            != out["earnings_explosiveness_bucket"].astype(str),
            None)
        out = out.drop(columns=["earnings_date_pre_audit"])

    out = out.merge(_published_calls(), on=["stock", "earnings_date"], how="left")
    out["model_version"] = MODEL_VERSION

    out["_rank"] = out["earnings_explosiveness_bucket"].map(BUCKET_ORDER).fillna(9)
    out = (out.sort_values(["week_start", "_rank", "earnings_date", "stock"])
              .drop(columns=["_rank"]).reset_index(drop=True))
    return out


def _hit(val) -> str:
    """? where the 3-day window has not closed yet — distinct from a measured miss."""
    if val is None or pd.isna(val):
        return "?"
    return "✓" if val else "·"


def _fmt_pct(val) -> str:
    if val is None or pd.isna(val):
        return "    n/a"
    return f"{val * 100:+6.1f}%"


def format_report(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> str:
    n_weeks = (end - start).days // 7 + 1
    lines = [
        f"PREDICTIONS  {start.date()} → {end.date()}  ({n_weeks} work week"
        f"{'s' if n_weeks != 1 else ''}, model {MODEL_VERSION})",
        "═" * 88,
    ]
    if df.empty:
        lines.append("  No events.")
        return "\n".join(lines)

    for week_start, block in df.groupby("week_start", sort=True):
        kind = "history" if (block["row_kind"] == "history").all() else \
               "upcoming" if (block["row_kind"] == "upcoming").all() else "mixed"
        lines.append("")
        lines.append(f"  Week of {week_start.strftime('%b %-d, %Y')}   "
                     f"({len(block)} events, {kind})")
        if kind == "history":
            lines.append(f"    {'':6} {'DATE':7} {'TIER':11} {'SCORE':>6} "
                         f"{'1d':>7} {'3d':>7} {'5d':>7}  {'≥5%':>4} {'≥8%':>4}")
        else:
            lines.append(f"    {'':6} {'DATE':7} {'TIER':11} {'SCORE':>6} "
                         f"{'PRE-AUDIT TIER':>15} {'DTE':>4}  HC")
        lines.append("    " + "─" * 82)
        for _, r in block.iterrows():
            tier = str(r["earnings_explosiveness_bucket"])
            score = r["earnings_explosiveness_score"]
            score_s = f"{score:6.1f}" if pd.notna(score) else "   n/a"
            base = (f"    {r['stock']:<6} {r['earnings_date'].strftime('%b %-d'):<7} "
                    f"{tier:<11} {score_s}")
            if r["row_kind"] == "history":
                lines.append(base + f" {_fmt_pct(r.get('reaction_1d'))}"
                                    f" {_fmt_pct(r.get('reaction_3d'))}"
                                    f" {_fmt_pct(r.get('reaction_5d'))}"
                                    f"  {_hit(r.get('moved_5pct')):>4}"
                                    f" {_hit(r.get('moved_8pct')):>4}")
            else:
                pa = r.get("earnings_explosiveness_bucket_pre_audit")
                pa_s = "n/a" if pa is None or pd.isna(pa) else str(pa)
                if r.get("pre_audit_differs"):
                    pa_s += " *"
                lines.append(base + f" {pa_s:>15} {int(r['days_to_earnings']):>4}"
                                    f"  {'★' if r.get('is_high_conviction') else ' '}")

    hist = df[df["row_kind"] == "history"]
    if not hist.empty:
        lines += ["", "  " + "─" * 84, "  HISTORY — realized hit rates (legacy abs_reaction_3d target)"]
        for label, mask in [("High Alert", hist["earnings_explosiveness_bucket"] == "High Alert"),
                            ("High Alert + Elevated",
                             hist["earnings_explosiveness_bucket"].isin(["High Alert", "Elevated"])),
                            ("All events", pd.Series(True, index=hist.index))]:
            sel = hist[mask & hist["moved_5pct"].notna()]
            if sel.empty:
                continue
            n = len(sel)
            m5 = int(sel["moved_5pct"].sum())
            m8 = int(sel["moved_8pct"].sum())
            lines.append(f"    {label:<24} n={n:<4}  ≥5%: {m5:>3}/{n:<4} ({m5/n*100:4.0f}%)"
                         f"   ≥8%: {m8:>3}/{n:<4} ({m8/n*100:4.0f}%)")
        pub = hist[hist["published_tier"].notna()]
        lines.append(f"    Archived published calls in window: {len(pub)} of {len(hist)} "
                     f"(the predictions DB starts 2026-08-31)")
        lines.append("    Retro-scored, not an archive — see the module docstring.")

    up = df[df["row_kind"] == "upcoming"]
    if not up.empty and "pre_audit_differs" in up.columns:
        diff = up[up["pre_audit_differs"] == True]
        compared = up[up["pre_audit_differs"].notna()]
        lines += ["", "  " + "─" * 84,
                  f"  UPCOMING — pre-audit selection disagrees on {len(diff)} of "
                  f"{len(compared)} events (marked *)"]
        for _, r in diff.iterrows():
            lines.append(f"    {r['stock']:<6} {r['earnings_date'].strftime('%b %-d')}   "
                         f"pre-audit {str(r['earnings_explosiveness_bucket_pre_audit']):<11} "
                         f"→ current {str(r['earnings_explosiveness_bucket'])}")
    return "\n".join(lines)


def run(weeks_back=None, weeks_forward=None, monday=None, weeks=1, pre_audit=True,
        out_dir=None, today=None) -> pd.DataFrame:
    start, end = week_block_window(weeks_back=weeks_back, weeks_forward=weeks_forward,
                                   monday=monday, weeks=weeks, today=today)
    df = predictions_range(weeks_back=weeks_back, weeks_forward=weeks_forward,
                           monday=monday, weeks=weeks, pre_audit=pre_audit, today=today)
    report = format_report(df, start, end)
    print(report)

    out_dir = out_dir or OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    stem = f"predictions_{start.date()}_{end.date()}"
    csv_path = os.path.join(out_dir, f"{stem}.csv")
    txt_path = os.path.join(out_dir, f"{stem}.txt")
    df.to_csv(csv_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nWrote {csv_path} ({len(df)} events)")
    print(f"Wrote {txt_path}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--weeks-back", type=int, help="N complete work weeks ending last Friday")
    g.add_argument("--weeks-forward", type=int, help="N whole work weeks from the next complete one")
    g.add_argument("--monday", help="explicit Monday (YYYY-MM-DD) to start from")
    p.add_argument("--weeks", type=int, default=1, help="blocks to take with --monday (default 1)")
    p.add_argument("--no-pre-audit", action="store_true",
                   help="skip the pre-audit reproduction (avoids reading full_df.parquet)")
    p.add_argument("--out-dir", default=None, help="where to write the CSV/TXT (default: today's run dir)")
    a = p.parse_args()
    run(weeks_back=a.weeks_back, weeks_forward=a.weeks_forward, monday=a.monday,
        weeks=a.weeks, pre_audit=not a.no_pre_audit, out_dir=a.out_dir)
