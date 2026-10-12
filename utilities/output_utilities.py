# utilities/output_utilities.py
import os
import shutil
from datetime import datetime

_run_output_dir = None


def get_run_output_dir(base="output"):
    """
    Today's output subfolder, e.g. output/output_2026_08_03 — one per calendar day, not
    per run. Wiped clean and recreated on first call each process, so a same-day rerun
    (e.g. a manual re-run of main.py) can't leave stale files (like a report PDF for a
    stock that no longer qualifies) sitting alongside the new outputs. Cached for the
    life of the process after that first call, so every output written during a single
    run lands in the same folder without re-wiping it.
    """
    global _run_output_dir
    if _run_output_dir is None:
        stamp = datetime.now().strftime("%Y_%m_%d")
        run_dir = os.path.join(base, f"output_{stamp}")
        if os.path.isdir(run_dir):
            shutil.rmtree(run_dir)
        os.makedirs(run_dir, exist_ok=True)
        _run_output_dir = run_dir
    return _run_output_dir


def get_run_logs_dir(base="output"):
    """logs/ subfolder inside today's run output dir, e.g. output/output_2026_08_03/logs."""
    logs_dir = os.path.join(get_run_output_dir(base), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def latest_run_output_dir(base="output"):
    """
    Most recently created output_<date> subfolder under base, for readers in a
    separate process that need to find the last run's non-parquet outputs
    (e.g. cron_weekly_digest.py locating per-stock PDF reports). Falls back to base
    itself if no dated run folder exists yet.
    """
    if not os.path.isdir(base):
        return base
    run_dirs = sorted(
        d for d in os.listdir(base)
        if d.startswith("output_") and os.path.isdir(os.path.join(base, d))
    )
    return os.path.join(base, run_dirs[-1]) if run_dirs else base


def write_daily_parquet(daily_df, path="output/full_df.parquet"):
    """The daily frame (no score) — research and price context only."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)   # ingest=False skips stage 1's directory_checks
    daily_df.to_parquet(path, index=False)
    print(f"Wrote {path}")


def write_events_parquet(events_df, path="output/events_df.parquet"):
    """The Model B-scored event frame every output reads."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    events_df.to_parquet(path, index=False)
    print(f"Wrote {path} ({len(events_df)} events)")
