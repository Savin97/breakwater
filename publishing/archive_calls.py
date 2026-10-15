# publishing/archive_calls.py
"""Record PUBLISHED calls in the prediction archive — append-only, one row per published
call per generation.

    python -m publishing.archive_calls --published --generation GEN   # scripts/full_workflow.sh only
    BREAKWATER_PREDICTIONS_DB=output/scratch.duckdb python -m publishing.archive_calls
                                                                     # scratch: any branch

The canonical archive (db/predictions.duckdb) is the prospective track record: a row there
says "this call was public at `published_at`". So nothing writes it as a side effect of
scoring — `run_pipeline` never archives — and the writer itself (not only this CLI)
refuses the canonical file unless it is handed a VERIFIED publication
(publishing/generation.load_publication: a generation whose receipt matched what the
server was actually serving) and the checkout is `master` (publishing/guard.py).

What a row records: the generation it was published in, the SHA-256 of that generation's
calls file, the timezone-aware publication time from the receipt, and the model version
the ARTIFACT says produced it (not whatever config the archiver happens to run).

History is never rewritten:
  * the same generation archived again (a retry) adds nothing;
  * a later generation whose call for an event is identical to the latest one already
    archived adds nothing — the earlier publication of that call stands;
  * a later generation that CHANGES an event's call appends a new revision. The original
    stays exactly as published, even when both happen on the same day, so the track
    record (utilities/db_utilities.read_track_record) can take the earliest eligible one.

The window is the whole Mon-Fri work week the digest mails (work_week_window), so the
record is "first eligible weekly-window call per event", not every call ever displayed.
"""
import argparse
import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone

import duckdb
import pandas as pd

import config
from publishing.generation import (PUBLICATION_ROOT, GenerationError, Publication,
                                   artifact_model_version, load_publication, mark_archived,
                                   sha256_file)
from publishing.guard import PublicationRefused, is_canonical_archive, require_production_branch
from utilities.data_utilities import work_week_window
from utilities.db_utilities import create_predictions_table_if_not_exists
from utilities.time_utilities import NY_TZ

PUBLISHED_CALLS = "output/upcoming_df.parquet"
# A publication older than this is not the run that just went live.
MAX_PUBLICATION_AGE_HOURS = 24

# The content that identifies a call. Two publications with the same call hash published
# the same call; anything that changes one of these is a revision.
CALL_FIELDS = ("stock", "earnings_date", "tier", "risk_score", "model_b_hist_mean_abs",
               "model_b_n_prior", "score_asof_date", "model_version")


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def _fmt(v) -> str:
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return ""
    if isinstance(v, float):
        return repr(round(v, 12))
    if isinstance(v, (pd.Timestamp, datetime)):
        return str(pd.Timestamp(v).date())
    return str(v)


def call_sha256(row) -> str:
    return hashlib.sha256("|".join(_fmt(row[f]) for f in CALL_FIELDS).encode()).hexdigest()


def scratch_publication(calls_path, now=None) -> Publication:
    """An UNVERIFIED publication for scratch archives (validation, tests)."""
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    calls = pd.read_parquet(calls_path)
    return Publication(generation_id=f"scratch-{now.strftime('%Y%m%dT%H%M%S%fZ')}",
                       published_at=now, artifact_sha256=sha256_file(calls_path),
                       model_version=artifact_model_version(calls),
                       calls_path=calls_path, verified=False)


def check_may_archive(db_path, publication: Publication | None) -> str:
    """Raise PublicationRefused unless writing `db_path` with `publication` is allowed."""
    if not is_canonical_archive(db_path):
        return "scratch"
    if publication is None or not publication.verified:
        raise PublicationRefused(
            "REFUSING TO WRITE THE CANONICAL PREDICTION ARCHIVE — it records published "
            "calls only, from a generation whose receipt matched the served dashboard "
            "(scripts/full_workflow.sh: `--published --generation`). For testing, set "
            "BREAKWATER_PREDICTIONS_DB to a scratch path.")
    require_production_branch("archive published calls")
    return "published"


def archive_calls(calls: pd.DataFrame, db_path: str, publication: Publication,
                  weeks: int = 1, current_week: bool = False) -> dict:
    """Append the publication's calls for one whole Mon-Fri work week to `db_path`.
    Returns {"inserted", "unchanged", "retried"}. See the module docstring for the rules."""
    check_may_archive(db_path, publication)
    published_at = pd.Timestamp(publication.published_at)
    if published_at.tzinfo is None:
        raise ValueError("published_at must be timezone-aware")
    today = published_at.tz_convert(NY_TZ).tz_localize(None).normalize()   # NY calendar day
    window_start, window_end = work_week_window(weeks=weeks, current_week=current_week,
                                                today=today)
    start = max(window_start, today)

    calls = calls.copy()
    if "is_pending" in calls.columns:
        calls = calls[calls["is_pending"].astype(bool)]
    if (calls["model_version"].astype(str) != publication.model_version).any():
        raise ValueError(f"calls carry a model_version other than the publication's "
                         f"({publication.model_version})")
    calls["earnings_date"] = pd.to_datetime(calls["earnings_date"])
    upcoming = calls[(calls["earnings_date"] >= start) & (calls["earnings_date"] <= window_end)]
    counts = {"inserted": 0, "unchanged": 0, "retried": 0}
    if upcoming.empty:
        print(f"No earnings events between {start.date()} and {window_end.date()} — nothing to archive.")
        return counts

    snap = pd.DataFrame({
        "prediction_asof_date":    today.date(),
        # run_week = Monday of the week we PUBLISHED; week_start = Monday of the week the
        # company reports. Neither is a key.
        "run_week":                (today - pd.Timedelta(days=today.weekday())).date(),
        "week_start":              (upcoming["earnings_date"]
                                     - pd.to_timedelta(upcoming["earnings_date"].dt.weekday, unit="D")).dt.date,
        "stock":                   upcoming["stock"],
        "earnings_date":           upcoming["earnings_date"].dt.date,
        "tier":                    upcoming["risk_tier"].astype(str),
        "risk_score":              upcoming["risk_score"].astype(float),
        "model_version":           publication.model_version,
        "git_commit":              _get_git_commit(),
        "score_asof_date":         pd.to_datetime(upcoming["score_asof_date"]).dt.date,
        "model_b_hist_mean_abs":   upcoming["model_b_hist_mean_abs"].astype(float),
        "model_b_n_prior":         upcoming["model_b_n_prior"].astype("int64"),
        "ingested_at":             pd.Timestamp.now(),
        "published_at":            published_at,
        "generation_id":           publication.generation_id,
        "artifact_sha256":         publication.artifact_sha256,
    })
    snap["call_sha256"] = snap.apply(call_sha256, axis=1)

    con = duckdb.connect(str(db_path))
    try:
        create_predictions_table_if_not_exists(con)
        con.execute("BEGIN")
        for row in snap.to_dict("records"):
            key = [row["stock"], row["earnings_date"]]
            if con.execute("SELECT 1 FROM predictions WHERE stock = ? AND earnings_date = ? "
                           "AND generation_id = ?", key + [row["generation_id"]]).fetchone():
                counts["retried"] += 1                       # same generation again
                continue
            latest = con.execute("""
                SELECT call_sha256, COALESCE(revision, 0) FROM predictions
                WHERE stock = ? AND earnings_date = ? AND published_at IS NOT NULL
                ORDER BY published_at DESC, revision DESC LIMIT 1""", key).fetchone()
            if latest and latest[0] == row["call_sha256"]:
                counts["unchanged"] += 1                     # the earlier publication stands
                continue
            row["revision"] = (latest[1] if latest else 0) + 1
            cols = list(row)
            con.execute(f"INSERT INTO predictions ({', '.join(cols)}) VALUES "
                        f"({', '.join('?' for _ in cols)})", [row[c] for c in cols])
            counts["inserted"] += 1
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    print(f"Archived generation {publication.generation_id} → {db_path}: "
          f"{counts['inserted']} new call(s)/revision(s), {counts['unchanged']} unchanged, "
          f"{counts['retried']} already archived (week ending {window_end.date()})")
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Record published calls in the prediction archive.")
    ap.add_argument("--published", action="store_true",
                    help="archive the verified generation --generation (canonical archive)")
    ap.add_argument("--generation", help="generation id under output/publication/")
    ap.add_argument("--calls", default=PUBLISHED_CALLS, help="scratch only: the calls parquet")
    ap.add_argument("--weeks", type=int, default=1, choices=(1, 2))
    ap.add_argument("--current-week", action="store_true")
    a = ap.parse_args(argv)

    db_path = config.PREDICTIONS_DB_PATH
    try:
        if a.published or a.generation:
            if not a.generation:
                raise PublicationRefused("--published needs --generation (the verified generation id)")
            publication = load_publication(PUBLICATION_ROOT / a.generation)
            age = (pd.Timestamp.now(tz="UTC") - publication.published_at).total_seconds() / 3600
            if age > MAX_PUBLICATION_AGE_HOURS:
                raise PublicationRefused(f"generation {a.generation} went live {age:.1f}h ago "
                                         f"(limit {MAX_PUBLICATION_AGE_HOURS}h) — not this run's")
        else:
            if not os.path.exists(a.calls):
                print(f"{a.calls} not found — run the pipeline first.", file=sys.stderr)
                return 1
            publication = scratch_publication(a.calls)
        mode = check_may_archive(db_path, publication)
    except (PublicationRefused, GenerationError) as e:
        print(f"\n{e}\n", file=sys.stderr)
        return 3
    print(f"Archive mode: {mode} ({db_path})")
    counts = archive_calls(pd.read_parquet(publication.calls_path), db_path, publication,
                           weeks=a.weeks, current_week=a.current_week)
    if mode == "published":
        mark_archived(PUBLICATION_ROOT / publication.generation_id, counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
