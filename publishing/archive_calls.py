# publishing/archive_calls.py
"""Record PUBLISHED calls in the prediction archive.

    python -m publishing.archive_calls --published     # production: scripts/full_workflow.sh only
    BREAKWATER_PREDICTIONS_DB=output/scratch.duckdb python -m publishing.archive_calls
                                                       # scratch: any branch, for validation

The canonical archive (db/predictions.duckdb) is the prospective track record: a row there
says "this is the call we made public before the event". So nothing writes it as a side
effect of scoring — `run_pipeline` never archives. It is written only here, and only when
  * the target is a scratch DB (BREAKWATER_PREDICTIONS_DB points elsewhere), or
  * `--published` is passed (the publication workflow, after the dashboard push), the
    checkout is `master` (publishing/guard.py) and the calls file is from today's run.

What gets archived is `output/upcoming_df.parquet` — the exact file the workflow pushed to
the dashboard — restricted to the same whole Mon-Fri work week the digest mails
(work_week_window). An archived row is therefore byte-for-byte the call that went public.
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import date

import duckdb
import pandas as pd

import config
from config import MODEL_VERSION
from publishing.guard import PublicationRefused, is_canonical_archive, require_production_branch
from utilities.data_utilities import work_week_window
from utilities.db_utilities import create_predictions_table_if_not_exists

PUBLISHED_CALLS = "output/upcoming_df.parquet"
# A published call is the one from the run that just produced the dashboard; a calls file
# older than this means the run did not happen and archiving it would record stale calls.
MAX_CALLS_AGE_HOURS = 24


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def archive_calls(calls: pd.DataFrame, db_path: str, weeks: int = 1,
                  current_week: bool = False, today=None) -> int:
    """Write `calls` (upcoming Model B calls — the upcoming parquet, or pending event-frame
    rows) for one whole Mon-Fri work week into `db_path`. Returns the rows written.

    A pure writer: it decides nothing about whether the calls were published — the caller
    (main below, or a test with a temporary DB) does. Same window as the weekly digest;
    the lower bound is clamped to today so an event that already reported is never
    recorded. One row per (stock, earnings_date, prediction_asof_date): a same-day rerun
    UPSERTS, a later day inserts a new row, and predictions_first_call keeps the earliest.

    0.4.0 rows: `tier` = Model B tier (Unscored included), `risk_score` = Breakwater Risk
    Percentile (NULL when Unscored), `score_asof_date` = the call cutoff, plus the raw
    signal and usable-prior count. The legacy columns are NULL — they exist for old rows.
    """
    today = pd.Timestamp(today if today is not None else date.today()).normalize()
    window_start, window_end = work_week_window(weeks=weeks, current_week=current_week,
                                                today=today)
    start = max(window_start, today)          # never record an event that already reported
    run_week_start = today - pd.Timedelta(days=today.weekday())      # Monday of this week

    calls = calls.copy()
    if "is_pending" in calls.columns:
        calls = calls[calls["is_pending"].astype(bool)]
    calls["earnings_date"] = pd.to_datetime(calls["earnings_date"])
    upcoming = calls[(calls["earnings_date"] >= start) & (calls["earnings_date"] <= window_end)]

    if upcoming.empty:
        print(f"No earnings events between {start.date()} and {window_end.date()} — "
              "nothing to archive.")
        return 0

    n = len(upcoming)
    snap = pd.DataFrame({
        "prediction_asof_date":    today.date(),
        # Monday of the week we made the call, vs week_start = Monday of the week the
        # company reports. run_week records WHEN the call was made; it is not a grouping
        # key — predictions_first_call dedups on the event itself.
        "run_week":                run_week_start.date(),
        "week_start":              (upcoming["earnings_date"]
                                     - pd.to_timedelta(upcoming["earnings_date"].dt.weekday, unit="D")).dt.date,
        "stock":                   upcoming["stock"],
        "earnings_date":           upcoming["earnings_date"].dt.date,
        "tier":                    upcoming["risk_tier"].astype(str),
        "risk_score":              upcoming["risk_score"].astype(float),
        "is_high_conviction":      pd.Series([pd.NA] * n, index=upcoming.index, dtype="boolean"),
        "pre_earnings_drift_flag": pd.Series([pd.NA] * n, index=upcoming.index, dtype="string"),
        "surprise_momentum_flag":  pd.Series([pd.NA] * n, index=upcoming.index, dtype="string"),
        "model_version":           MODEL_VERSION,
        "git_commit":              _get_git_commit(),
        "score_asof_date":         pd.to_datetime(upcoming["score_asof_date"]).dt.date,
        "model_b_hist_mean_abs":   upcoming["model_b_hist_mean_abs"].astype(float),
        "model_b_n_prior":         upcoming["model_b_n_prior"].astype("int64"),
        "ingested_at":             pd.Timestamp.now(),
    })

    con = duckdb.connect(str(db_path))
    try:
        create_predictions_table_if_not_exists(con)
        col_list = ", ".join(snap.columns)
        key = ("stock", "earnings_date", "prediction_asof_date")
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in snap.columns if c not in key)
        con.register("tmp_predictions", snap)
        # Upsert, not DO NOTHING: re-running the same day after a fix must correct the
        # row; with DO NOTHING the first (possibly broken) run of the day would survive.
        con.execute(
            f"INSERT INTO predictions ({col_list}) SELECT {col_list} FROM tmp_predictions "
            f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {updates}"
        )
        con.unregister("tmp_predictions")
    finally:
        con.close()
    print(f"Archived {len(snap)} calls → {db_path} "
          f"(asof {today.date()}, week ending {window_end.date()})")
    return len(snap)


def check_may_archive(db_path, published: bool) -> str:
    """Raise PublicationRefused unless writing `db_path` is allowed; return the mode."""
    if not is_canonical_archive(db_path):
        return "scratch"
    if not published:
        raise PublicationRefused(
            "REFUSING TO WRITE THE CANONICAL PREDICTION ARCHIVE — it records published "
            "calls only, via scripts/full_workflow.sh (`--published`). For testing, set "
            "BREAKWATER_PREDICTIONS_DB to a scratch path.")
    require_production_branch("archive published calls")
    return "published"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Record published calls in the prediction archive.")
    ap.add_argument("--published", action="store_true",
                    help="the publication workflow has made these calls public (canonical archive)")
    ap.add_argument("--calls", default=PUBLISHED_CALLS, help="the published calls parquet")
    ap.add_argument("--weeks", type=int, default=1, choices=(1, 2))
    ap.add_argument("--current-week", action="store_true")
    a = ap.parse_args(argv)

    db_path = config.PREDICTIONS_DB_PATH
    try:
        mode = check_may_archive(db_path, a.published)
    except PublicationRefused as e:
        print(f"\n{e}\n", file=sys.stderr)
        return 3
    if not os.path.exists(a.calls):
        print(f"{a.calls} not found — run the pipeline first.", file=sys.stderr)
        return 1
    age_hours = (time.time() - os.path.getmtime(a.calls)) / 3600
    if mode == "published" and age_hours > MAX_CALLS_AGE_HOURS:
        print(f"REFUSING TO ARCHIVE: {a.calls} is {age_hours:.1f}h old (limit "
              f"{MAX_CALLS_AGE_HOURS}h) — these are not this run's published calls.",
              file=sys.stderr)
        return 3
    print(f"Archive mode: {mode} ({db_path})")
    archive_calls(pd.read_parquet(a.calls), db_path, weeks=a.weeks, current_week=a.current_week)
    return 0


if __name__ == "__main__":
    sys.exit(main())
