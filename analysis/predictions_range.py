"""
Predictions for a range of whole Mon-Fri work weeks — backward, forward, or from an
explicit Monday. A maintainer tool.

    python -m analysis.predictions_range --weeks-back 12
    python -m analysis.predictions_range --weeks-forward 4
    python -m analysis.predictions_range --monday 2026-06-01 --weeks 3

The model is Model B as it ships (scoring/model_b.py, MODEL_VERSION 0.4.0): Breakwater
Risk Percentile (`risk_score`), `risk_tier`, the raw signal `model_b_hist_mean_abs`, the
usable-prior count `model_b_n_prior` and the call cutoff `score_asof_date`.

HISTORY (`is_pending == 0`). Each completed event's Model B call as of its OWN cutoff —
the last session before the Monday of its report week — from prior corrected outcomes
complete by then. Fully causal, but still a RETRO-SCORE: nothing was published at the
time, and the percentile scale (scoring/model_b_reference.json) was built from 2014–2025
events. Outcomes are the validation target, `abs_reaction_3d_anchored` (≥ 5% / ≥ 8%);
an event without a verified announcement time shows `?`, never a miss. Where a call
was actually archived it is shown as `published_tier` with its model version — calls
before 0.4.0 came from the retired legacy model.

UPCOMING (`is_pending == 1`). The pending call, exactly as the pipeline publishes it.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
from datetime import date

import duckdb
import numpy as np
import pandas as pd

import config
from config import MODEL_VERSION, EXTREME_EARNINGS_REACTION_THRESHOLD
from feature_engineering.announcement_timing import TARGET_AVAILABLE
from scoring.model_b import display_score, sort_by_risk
from utilities.data_utilities import week_block_window

EVENTS_PARQUET = "output/events_df.parquet"
# Its own directory, NOT get_run_output_dir(): that one wipes and recreates today's run
# folder on first call in a process, so writing there from a separate process would
# delete the pipeline run's reports. These sheets are also worth keeping across days.
OUTPUT_DIR     = "output/predictions"

LARGE_MOVE = 0.05      # shown alongside the 8% validation target, descriptive only

SCORE_COLS = ["risk_tier", "risk_score", "model_b_hist_mean_abs", "model_b_n_prior",
              "score_asof_date"]
OUTCOME_COLS = ["announce_window", "reaction_3d_anchored", "abs_reaction_3d_anchored",
                "reaction_3d_anchored_status"]


def _published_calls(db_path=None) -> pd.DataFrame:
    """The archived first call per event, any model version. Missing DB is not an error —
    the archive only starts 2026-08-31. `published_void` and `published_model_version`
    travel with it: pre-0.4 calls came from the retired legacy model and are void for
    the 0.4.0 record."""
    empty = pd.DataFrame(columns=["stock", "earnings_date", "published_tier",
                                  "published_risk_score", "published_asof",
                                  "published_model_version", "published_void"])
    db_path = db_path or config.PREDICTIONS_DB_PATH
    if not os.path.exists(db_path):
        return empty
    con = duckdb.connect(db_path, read_only=True)
    try:
        cols = {r[0] for r in con.execute("DESCRIBE predictions_first_call").fetchall()}
        void = ("void_for_track_record" if "void_for_track_record" in cols else "FALSE")
        out = con.execute(f"""
            SELECT stock, earnings_date,
                   tier                 AS published_tier,
                   risk_score           AS published_risk_score,
                   prediction_asof_date AS published_asof,
                   model_version        AS published_model_version,
                   COALESCE({void}, FALSE) AS published_void
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
                      today=None, db_path=None) -> pd.DataFrame:
    """One row per earnings event in the window. See the module docstring for what the
    history and upcoming halves each mean."""
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
    # A pending row whose earnings_date has already passed is a stale feed, not a call.
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

    keep = ["stock", "sector", "earnings_date", "row_kind", *SCORE_COLS, *OUTCOME_COLS]
    out = out[[c for c in keep if c in out.columns]].copy()
    out.insert(0, "week_start",
               out["earnings_date"] - pd.to_timedelta(out["earnings_date"].dt.weekday, unit="D"))
    out["days_to_earnings"] = np.where(
        out["row_kind"] == "upcoming", (out["earnings_date"] - now).dt.days, np.nan)

    # Realized outcome on the corrected target only. Unresolved stays None ("?").
    resolved = out["reaction_3d_anchored_status"].eq(TARGET_AVAILABLE) \
        & out["abs_reaction_3d_anchored"].notna()
    moved = out["abs_reaction_3d_anchored"]
    out["moved_5pct"] = np.where(resolved, moved >= LARGE_MOVE, None)
    out["moved_8pct"] = np.where(resolved, moved >= EXTREME_EARNINGS_REACTION_THRESHOLD, None)

    out = out.merge(_published_calls(db_path), on=["stock", "earnings_date"], how="left")
    out["model_version"] = MODEL_VERSION

    out = sort_by_risk(out, by=["week_start"]).reset_index(drop=True)
    return out


def _hit(val) -> str:
    """? where there is no corrected outcome — distinct from a measured miss."""
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
        f"{'s' if n_weeks != 1 else ''}, Model B {MODEL_VERSION})",
        "═" * 92,
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
        lines.append(f"    {'':6} {'DATE':7} {'TIER':11} {'PCTL':>4} {'AVG|R|':>7} {'N':>3} "
                     f"{'CUTOFF':10}  {'3-SESSION':>9} {'≥5%':>4} {'≥8%':>4}  PUBLISHED")
        lines.append("    " + "─" * 86)
        for _, r in block.iterrows():
            pct = display_score(r["risk_score"])
            avg = r["model_b_hist_mean_abs"]
            pub = r.get("published_tier")
            pub_s = "" if pub is None or pd.isna(pub) else \
                f"{pub} ({r['published_model_version']}{', void' if r.get('published_void') else ''})"
            if r["row_kind"] == "history":
                outcome = (f"{_fmt_pct(r.get('reaction_3d_anchored')):>9} "
                           f"{_hit(r.get('moved_5pct')):>4} {_hit(r.get('moved_8pct')):>4}")
            else:
                outcome = f"{'in ' + str(int(r['days_to_earnings'])) + 'd':>9} {'':>4} {'':>4}"
            lines.append(
                f"    {r['stock']:<6} {r['earnings_date'].strftime('%b %-d'):<7} "
                f"{str(r['risk_tier']):<11} {('—' if pct is None else str(pct)):>4} "
                f"{('—' if pd.isna(avg) else f'{avg * 100:.1f}%'):>7} "
                f"{int(r['model_b_n_prior']):>3} "
                f"{str(r['score_asof_date'])[:10]:10}  {outcome}  {pub_s}")

    hist = df[df["row_kind"] == "history"]
    if not hist.empty:
        lines += ["", "  " + "─" * 88,
                  "  HISTORY — retro-scored Model B calls vs the corrected 3-session target"]
        for label, mask in [("High Alert", hist["risk_tier"] == "High Alert"),
                            ("High Alert + Elevated",
                             hist["risk_tier"].isin(["High Alert", "Elevated"])),
                            ("All scored events", hist["risk_tier"] != "Unscored")]:
            sel = hist[mask & hist["moved_8pct"].notna()]
            if sel.empty:
                continue
            n = len(sel)
            m5, m8 = int(sel["moved_5pct"].sum()), int(sel["moved_8pct"].sum())
            lines.append(f"    {label:<24} n={n:<4}  ≥5%: {m5:>3}/{n:<4} ({m5/n*100:4.0f}%)"
                         f"   ≥8%: {m8:>3}/{n:<4} ({m8/n*100:4.0f}%)")
        unresolved = int(hist["moved_8pct"].isna().sum())
        lines.append(f"    {unresolved} of {len(hist)} events have no corrected outcome "
                     f"(no verified announcement time) and are not counted.")
        pub = hist[hist["published_tier"].notna()]
        lines.append(f"    Archived published calls in window: {len(pub)} of {len(hist)} "
                     f"(the predictions DB starts 2026-08-31; calls before 0.4.0 came from "
                     f"the retired legacy model).")
        lines.append("    Retro-scored, not an archive — see the module docstring.")
    return "\n".join(lines)


def run(weeks_back=None, weeks_forward=None, monday=None, weeks=1, out_dir=None,
        today=None) -> pd.DataFrame:
    start, end = week_block_window(weeks_back=weeks_back, weeks_forward=weeks_forward,
                                   monday=monday, weeks=weeks, today=today)
    df = predictions_range(weeks_back=weeks_back, weeks_forward=weeks_forward,
                           monday=monday, weeks=weeks, today=today)
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
    p.add_argument("--out-dir", default=None, help=f"where to write the CSV/TXT (default {OUTPUT_DIR})")
    a = p.parse_args()
    run(weeks_back=a.weeks_back, weeks_forward=a.weeks_forward, monday=a.monday,
        weeks=a.weeks, out_dir=a.out_dir)
