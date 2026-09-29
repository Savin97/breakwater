"""Delete iv_snapshots rows taken on days the NYSE was closed (one-time cleanup).

Before the market-hours guard in `ingestion/fetch_iv.py`, the IV cron ran on exchange
holidays and stored the previous session's leftover quotes as if they were that day's —
2026-06-19 (Juneteenth) and 2026-09-07 (Labor Day). The deleted rows are written to a
parquet file under backfills/ first, so this can be undone.

    PYTHONPATH=. .venv/bin/python scripts/delete_iv_rows_on_closed_days.py [--dry-run]

Idempotent: a second run finds nothing to delete. Run outside the cron windows.
"""
import argparse
import os
from datetime import datetime

import duckdb
import pandas as pd
import pandas_market_calendars as mcal

from config import DB_PATH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    con = duckdb.connect(DB_PATH, read_only=args.dry_run)
    dates = pd.to_datetime(
        con.execute("SELECT DISTINCT snapshot_date FROM iv_snapshots").fetch_df()["snapshot_date"]
    ).dt.date
    sessions = mcal.get_calendar("NYSE").valid_days(start_date=dates.min(), end_date=dates.max())
    session_dates = set(sessions.tz_localize(None).date)
    closed = sorted(d for d in dates if d not in session_dates)

    if not closed:
        print("No IV rows on closed days. Nothing to do.")
        return

    rows = con.execute(
        "SELECT * FROM iv_snapshots WHERE snapshot_date IN (SELECT UNNEST(?))", [closed]
    ).fetch_df()
    print(rows.groupby("snapshot_date").size().rename("rows").to_string())

    if args.dry_run:
        print(f"\n--dry-run: would delete {len(rows)} rows.")
        return

    os.makedirs("backfills", exist_ok=True)
    backup = f"backfills/iv_rows_on_closed_days_{datetime.now():%Y%m%d_%H%M%S}.parquet"
    rows.to_parquet(backup, index=False)
    deleted = con.execute(
        "DELETE FROM iv_snapshots WHERE snapshot_date IN (SELECT UNNEST(?))", [closed]
    ).fetchone()[0]
    con.close()
    print(f"\nDeleted {deleted} rows. Backup: {backup}")


if __name__ == "__main__":
    main()
