"""One-time seed of observed announcement timestamps into `earnings.announce_ts_ny`.

Why this exists
---------------
`audit/provider_timestamps.parquet` holds 12,269 event-level announcement times pulled
straight from yfinance during the Phase 0 audit. It is real evidence and there is no
reason to re-fetch it. But an audit artifact must not become a runtime input: nothing in
`pipeline/` may read that file, or the production path silently depends on a file whose
provenance, refresh cadence and schema nobody owns.

So it is loaded ONCE, here, into the database column that production does read
(`utilities.db_utilities.load_announcement_timing`). After this runs, the parquet is
back to being evidence. Ordinary ingestion keeps the column current from then on —
`fetch_one_earnings_dates` no longer throws the timestamp away.

Rules
-----
* Only fills NULLs. An observed timestamp already on a row is never overwritten.
* Only touches (stock, earnings_date) pairs that already exist in `earnings`. It creates
  no events; a timestamp for an event we do not store is simply skipped and counted.
* Records provenance in `announce_ts_source` and the pull date in
  `announce_ts_observed_at`, so a later reader can tell a backfilled timestamp from a
  freshly ingested one, and a schedule from a post-event observation.
* Idempotent: run it twice and the second run updates nothing.
* No timestamp is invented for the AlphaVantage date-only history. Those rows stay NULL
  and their events stay explicitly unresolved.

Usage
-----
    PYTHONPATH=. .venv/bin/python scripts/backfill_announcement_timestamps.py [--dry-run]
"""
import argparse
import sys

import duckdb
import pandas as pd

from config import DB_PATH
from utilities.db_utilities import create_earnings_table_if_not_exists

SOURCE_PARQUET = "audit/provider_timestamps.parquet"
SOURCE_LABEL = "audit_provider_timestamps_2026_09_05"

# When the audit actually pulled these timestamps from yfinance. Recorded as
# `announce_ts_observed_at` so the seeded rows carry the schedule-vs-observation
# distinction like any other row: a seeded event that had already reported by this date
# was OBSERVED after the fact and is frozen; a seeded event still upcoming on this date
# was a SCHEDULE and ingestion may refresh it later (external review of Phase 2, item 2).
# It is a fact about the pull, not a guess — do not move it to "now".
#
# Naive NY wall clock, like every other announcement-timing value
# (utilities.time_utilities). Midnight NY on the pull date is a deliberate LOWER BOUND on
# the moment of the pull: it can only make a seeded row look more like a schedule and so
# more refreshable, never less, which is the conservative direction under any host
# timezone.
SOURCE_OBSERVED_AT = pd.Timestamp("2026-09-05")


def load_seed(path=SOURCE_PARQUET) -> pd.DataFrame:
    """The audit parquet, reduced to exactly (stock, earnings_date, announce_ts_ny).

    The parquet's timestamps are tz-aware America/New_York; the DB column is naive NY
    local time, so the conversion drops the offset and keeps the wall clock. That is the
    same shape ingestion now writes, and the same shape the classifier reads.
    """
    ts = pd.read_parquet(path)
    ts = ts[["stock", "earnings_date", "announce_ts_ny"]].copy()
    ts["earnings_date"] = pd.to_datetime(ts["earnings_date"]).dt.date
    ny = pd.to_datetime(ts["announce_ts_ny"])
    if getattr(ny.dt, "tz", None) is not None:
        ny = ny.dt.tz_convert("America/New_York").dt.tz_localize(None)
    ts["announce_ts_ny"] = ny
    ts = ts.dropna(subset=["announce_ts_ny"])
    return ts.drop_duplicates(subset=["stock", "earnings_date"], keep="first")


def backfill(con, seed: pd.DataFrame, dry_run: bool = False) -> dict:
    create_earnings_table_if_not_exists(con)
    con.register("seed_ts", seed)

    stats = {"seed_rows": len(seed)}
    stats["matched_events"] = con.execute("""
        SELECT COUNT(*) FROM earnings e JOIN seed_ts s
          ON e.stock = s.stock AND e.earnings_date = s.earnings_date
    """).fetchone()[0]
    stats["would_fill"] = con.execute("""
        SELECT COUNT(*) FROM earnings e JOIN seed_ts s
          ON e.stock = s.stock AND e.earnings_date = s.earnings_date
        WHERE e.announce_ts_ny IS NULL
    """).fetchone()[0]
    stats["already_had_timestamp"] = stats["matched_events"] - stats["would_fill"]
    stats["seed_events_not_in_db"] = con.execute("""
        SELECT COUNT(*) FROM seed_ts s
        WHERE NOT EXISTS (SELECT 1 FROM earnings e
                          WHERE e.stock = s.stock AND e.earnings_date = s.earnings_date)
    """).fetchone()[0]

    # Rows this script seeded before `announce_ts_observed_at` existed carry the seed
    # label but no observation date, which leaves them un-classifiable as
    # schedule-vs-observation. Stamp the pull date on them once. Self-migrating and
    # idempotent, and it touches only rows this script itself wrote.
    stats["seeded_rows_missing_observed_at"] = con.execute("""
        SELECT COUNT(*) FROM earnings
        WHERE announce_ts_source = ? AND announce_ts_observed_at IS NULL
    """, [SOURCE_LABEL]).fetchone()[0]

    if not dry_run:
        con.execute("""
            UPDATE earnings SET announce_ts_observed_at = ?
             WHERE announce_ts_source = ? AND announce_ts_observed_at IS NULL
        """, [SOURCE_OBSERVED_AT.to_pydatetime(), SOURCE_LABEL])
        con.execute("""
            UPDATE earnings AS e
               SET announce_ts_ny = s.announce_ts_ny,
                   announce_ts_source = ?,
                   announce_ts_observed_at = ?
              FROM seed_ts AS s
             WHERE e.stock = s.stock
               AND e.earnings_date = s.earnings_date
               AND e.announce_ts_ny IS NULL
        """, [SOURCE_LABEL, SOURCE_OBSERVED_AT.to_pydatetime()])
        stats["filled"] = stats["would_fill"]
    con.unregister("seed_ts")

    stats["earnings_rows"] = con.execute("SELECT COUNT(*) FROM earnings").fetchone()[0]
    stats["with_timestamp"] = con.execute(
        "SELECT COUNT(*) FROM earnings WHERE announce_ts_ny IS NOT NULL").fetchone()[0]
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be filled and write nothing")
    ap.add_argument("--source", default=SOURCE_PARQUET)
    args = ap.parse_args()

    seed = load_seed(args.source)
    print(f"seed: {len(seed)} observed timestamps from {args.source}")
    con = duckdb.connect(DB_PATH)
    try:
        stats = backfill(con, seed, dry_run=args.dry_run)
    finally:
        con.close()

    for k, v in stats.items():
        print(f"  {k}: {v}")
    coverage = stats["with_timestamp"] / max(stats["earnings_rows"], 1)
    print(f"  timestamp coverage of the earnings table: {coverage:.1%}")
    if args.dry_run:
        print("\n(dry run — nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
