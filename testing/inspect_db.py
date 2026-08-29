"""
Quick look at the DuckDB database — schema, row counts, date ranges, sample rows.

Usage:
    python scripts/inspect_db.py                        # summary of all tables
    python scripts/inspect_db.py prices                  # just one table, more detail
    python scripts/inspect_db.py --sql "SELECT * FROM earnings WHERE stock='AAPL' ORDER BY earnings_date DESC LIMIT 10"
    python scripts/inspect_db.py --csv                   # export all tables to output/db_csv/ for viewing in Excel/Numbers/etc.
    python scripts/inspect_db.py prices earnings --csv    # export just these tables
"""
import os
import sys
import argparse
import duckdb
import pandas as pd

sys.path.insert(0, ".")
from config import DB_PATH

CSV_OUT_DIR = "output/db_csv"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)


def summarize_table(con, table: str, sample_rows: int = 5):
    print(f"\n{'=' * 70}\n{table}\n{'=' * 70}")

    schema = con.execute(f"DESCRIBE {table}").fetchdf()
    print(schema.to_string(index=False))

    n_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"\nrows: {n_rows:,}")

    cols = set(schema["column_name"])
    if "stock" in cols:
        n_stocks = con.execute(f"SELECT COUNT(DISTINCT stock) FROM {table}").fetchone()[0]
        print(f"distinct stocks: {n_stocks}")

    for date_col in ("date", "earnings_date"):
        if date_col in cols:
            lo, hi = con.execute(f"SELECT MIN({date_col}), MAX({date_col}) FROM {table}").fetchone()
            print(f"{date_col} range: {lo} -> {hi}")

    if sample_rows:
        print(f"\nsample rows:")
        print(con.execute(f"SELECT * FROM {table} USING SAMPLE {sample_rows}").fetchdf().to_string(index=False))


def export_csv(con, table: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{table}.csv")
    con.execute(f"COPY {table} TO '{path}' (HEADER, DELIMITER ',')")
    n_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"Wrote {path} ({n_rows:,} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table", nargs="*", help="Table name(s) to inspect (default: all tables)")
    parser.add_argument("--sql", help="Run an arbitrary SQL query and print the result instead")
    parser.add_argument("--sample", type=int, default=5, help="Sample rows to show per table (default: 5)")
    parser.add_argument("--csv", action="store_true", help="Export table(s) to CSV instead of printing a summary")
    parser.add_argument("--out-dir", default=CSV_OUT_DIR, help=f"CSV output directory (default: {CSV_OUT_DIR})")
    args = parser.parse_args()

    con = duckdb.connect(DB_PATH, read_only=True)

    if args.sql:
        print(con.execute(args.sql).fetchdf().to_string(index=False))
        return

    tables = args.table or [row[0] for row in con.execute("SHOW TABLES").fetchall()]

    for table in tables:
        if args.csv:
            export_csv(con, table, args.out_dir)
        else:
            summarize_table(con, table, sample_rows=args.sample)


if __name__ == "__main__":
    main()
