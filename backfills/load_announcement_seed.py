"""One-time load of historical announcement timestamps into `earnings.announce_ts_ny`.

Ingestion records the announcement time of every event it sees from now on. This fills
the history before that, from a seed file built elsewhere (the Benzinga and yfinance
sources are matched to our events on the research branch; the seed is the result).

The seed is a parquet with exactly these columns, one row per (stock, earnings_date):

    stock, earnings_date, announce_ts_ny, announce_ts_source, announce_ts_observed_at

`announce_ts_ny` and `announce_ts_observed_at` are naive New York wall clock, the same
convention ingestion writes. The seed is derived from licensed vendor data, so it is not
committed: copy it to the droplet by hand.

Rules
-----
* Only fills NULLs. A timestamp already on a row is never overwritten.
* Only touches (stock, earnings_date) pairs that already exist in `earnings`. It creates
  no events; a seed row for an event we do not store is skipped and counted.
* Every row keeps its own source and observation time, so a later reader can trace it.
* Idempotent: run it twice and the second run updates nothing.

Usage
-----
    PYTHONPATH=. .venv/bin/python backfills/load_announcement_seed.py SEED.parquet [--dry-run]
"""
import argparse
import sys

import duckdb
import pandas as pd

from config import DB_PATH
from utilities.db_utilities import create_earnings_table_if_not_exists

SEED_COLS = ["stock", "earnings_date", "announce_ts_ny", "announce_ts_source",
             "announce_ts_observed_at"]


def read_seed(path) -> pd.DataFrame:
    seed = pd.read_parquet(path)
    missing = set(SEED_COLS) - set(seed.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    seed = seed[SEED_COLS].copy()
    for col in ("announce_ts_ny", "announce_ts_observed_at"):
        ts = pd.to_datetime(seed[col])
        if getattr(ts.dt, "tz", None) is not None:
            raise ValueError(f"{col} must be naive New York wall clock, got {ts.dt.tz}")
        seed[col] = ts
    seed["earnings_date"] = pd.to_datetime(seed["earnings_date"]).dt.date
    seed = seed.dropna(subset=["stock", "earnings_date", "announce_ts_ny",
                               "announce_ts_source"])
    if seed.duplicated(["stock", "earnings_date"]).any():
        raise ValueError(f"{path} has more than one row for some (stock, earnings_date)")
    return seed


def load(con, seed: pd.DataFrame, dry_run: bool = False) -> dict:
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
    stats["seed_events_not_in_db"] = stats["seed_rows"] - stats["matched_events"]
    if not dry_run:
        con.execute("""
            UPDATE earnings AS e
               SET announce_ts_ny = s.announce_ts_ny,
                   announce_ts_source = s.announce_ts_source,
                   announce_ts_observed_at = s.announce_ts_observed_at
              FROM seed_ts AS s
             WHERE e.stock = s.stock
               AND e.earnings_date = s.earnings_date
               AND e.announce_ts_ny IS NULL
        """)
        stats["filled"] = stats["would_fill"]
    con.unregister("seed_ts")

    stats["earnings_rows"] = con.execute("SELECT COUNT(*) FROM earnings").fetchone()[0]
    stats["with_timestamp"] = con.execute(
        "SELECT COUNT(*) FROM earnings WHERE announce_ts_ny IS NOT NULL").fetchone()[0]
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seed", help="seed parquet")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be filled and write nothing")
    args = ap.parse_args()

    seed = read_seed(args.seed)
    print(f"seed: {len(seed)} timestamps from {args.seed}")
    con = duckdb.connect(DB_PATH)
    try:
        stats = load(con, seed, dry_run=args.dry_run)
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
