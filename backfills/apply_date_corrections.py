"""One-time correction of earnings dates that were stored days to weeks late.

Most of these rows came from the AlphaVantage history load of February 2026, whose report
dates for recent quarters often trail the real announcement. The corrections are built
elsewhere, by checking every stored date against Benzinga's confirmed report dates and
yfinance's (both must agree); this applies them.

The corrections file is a parquet, one row per wrong stored row:

    stock, old_date, new_date, action, announce_ts_ny, announce_ts_source,
    announce_ts_observed_at

action = "move"        no row exists at new_date: the stored row's date is changed to
                       new_date. EPS and every other column stay. If the row has no
                       announcement time, the one supplied is attached.
action = "delete_old"  a row for the same report already exists at new_date (ingestion
                       added it from yfinance): the stored row at old_date is deleted.

Rules
-----
* Every action re-checks the database first and is skipped, not forced, if the state is
  not what the file expected: old row gone, new row appeared where a move was planned,
  or new row missing where a delete was planned.
* Never creates an event.
* Idempotent: run it twice and the second run changes nothing.

Usage
-----
    PYTHONPATH=. .venv/bin/python backfills/apply_date_corrections.py CORRECTIONS.parquet [--dry-run]
"""
import argparse
import sys

import duckdb
import pandas as pd

from config import DB_PATH
from utilities.db_utilities import create_earnings_table_if_not_exists

CORRECTION_COLS = ["stock", "old_date", "new_date", "action", "announce_ts_ny",
                   "announce_ts_source", "announce_ts_observed_at"]
ACTIONS = {"move", "delete_old"}


def read_corrections(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = set(CORRECTION_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    df = df[CORRECTION_COLS].copy()
    bad = set(df["action"]) - ACTIONS
    if bad:
        raise ValueError(f"unknown actions {sorted(bad)}")
    for col in ("old_date", "new_date"):
        df[col] = pd.to_datetime(df[col]).dt.date
    for col in ("announce_ts_ny", "announce_ts_observed_at"):
        ts = pd.to_datetime(df[col])
        if getattr(ts.dt, "tz", None) is not None:
            raise ValueError(f"{col} must be naive New York wall clock")
        df[col] = ts
    if df.duplicated(["stock", "old_date"]).any():
        raise ValueError(f"{path} lists some (stock, old_date) more than once")
    if (df["old_date"] == df["new_date"]).any():
        raise ValueError(f"{path} has rows whose new_date equals old_date")
    return df


def _exists(con, stock, d) -> bool:
    return con.execute("SELECT COUNT(*) FROM earnings WHERE stock = ? AND earnings_date = ?",
                       [stock, d]).fetchone()[0] > 0


def _py(v):
    if v is None or pd.isna(v):
        return None
    return v.to_pydatetime() if hasattr(v, "to_pydatetime") else v


def apply(con, corrections: pd.DataFrame, dry_run: bool = False) -> dict:
    create_earnings_table_if_not_exists(con)
    stats = {"moved": 0, "deleted_old": 0, "skipped_old_row_gone": 0,
             "skipped_new_row_appeared": 0, "skipped_new_row_missing": 0}
    for r in corrections.itertuples(index=False):
        if not _exists(con, r.stock, r.old_date):
            stats["skipped_old_row_gone"] += 1
            continue
        new_exists = _exists(con, r.stock, r.new_date)
        if r.action == "move":
            if new_exists:
                stats["skipped_new_row_appeared"] += 1
                continue
            if not dry_run:
                con.execute("""
                    UPDATE earnings
                       SET earnings_date = ?,
                           announce_ts_ny = COALESCE(announce_ts_ny, ?),
                           announce_ts_source = CASE WHEN announce_ts_ny IS NULL
                                                     THEN ? ELSE announce_ts_source END,
                           announce_ts_observed_at = CASE WHEN announce_ts_ny IS NULL
                                                          THEN ? ELSE announce_ts_observed_at END
                     WHERE stock = ? AND earnings_date = ?
                """, [r.new_date, _py(r.announce_ts_ny),
                      r.announce_ts_source if _py(r.announce_ts_ny) is not None else None,
                      _py(r.announce_ts_observed_at), r.stock, r.old_date])
            stats["moved"] += 1
        else:
            if not new_exists:
                stats["skipped_new_row_missing"] += 1
                continue
            if not dry_run:
                con.execute("DELETE FROM earnings WHERE stock = ? AND earnings_date = ?",
                            [r.stock, r.old_date])
            stats["deleted_old"] += 1
    stats["earnings_rows"] = con.execute("SELECT COUNT(*) FROM earnings").fetchone()[0]
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corrections", help="corrections parquet")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing")
    args = ap.parse_args()

    corrections = read_corrections(args.corrections)
    print(f"corrections: {len(corrections)} rows "
          f"({(corrections['action'] == 'move').sum()} move, "
          f"{(corrections['action'] == 'delete_old').sum()} delete_old)")
    con = duckdb.connect(DB_PATH)
    try:
        con.execute("BEGIN TRANSACTION")
        stats = apply(con, corrections, dry_run=args.dry_run)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()

    for k, v in stats.items():
        print(f"  {k}: {v}")
    if args.dry_run:
        print("\n(dry run — nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
