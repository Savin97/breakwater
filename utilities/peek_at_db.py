# utilities/peek_at_db.py
"""
Peek at the DuckDB tables without writing SQL.

    python -m utilities.peek_at_db                      # freshness overview of every table
    python -m utilities.peek_at_db all 7d               # last 7 days of rows from every table
    python -m utilities.peek_at_db prices 7d            # last 7 days of prices
    python -m utilities.peek_at_db iv_snapshots 2w -s AVGO ADBE
    python -m utilities.peek_at_db predictions 1m --summary
    python -m utilities.peek_at_db earnings 30d --by ingested_at
    python -m utilities.peek_at_db prices 30d --csv        # also write output/peek/*.csv
    python -m utilities.peek_at_db all 7d --csv -n 0      # one CSV per table, -n = 0 is uncapped row amount
    python -m utilities.peek_at_db --sql "SELECT ..."

Windows are counted back from today, and rows always come back newest-first
(ORDER BY the table's time column DESC). Use --anchor latest to count back from
the newest row in the table instead (useful when a feed has stalled).

--csv writes exactly the rows you asked for: the same window and the same -n cap
as the printed table, so an export can never be bigger than the peek. It is
wider though - every column, not the terminal preview subset, since that subset
exists to fit a terminal. Use -n 0 for no row cap and --cols to narrow columns.

This is a read-only inspection tool: it opens the DB read_only and never writes.
Output goes to stdout via print, not logging - it is the thing you ran the
command to read, not a diagnostic.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys

import duckdb
import pandas as pd

import config

# Per-table default time column and a curated preview column list, so wide
# tables stay readable in a terminal. --all-cols overrides the preview list.
TABLE_SPECS = {
    "prices": {
        "time_col": "date",
        "cols": ["stock", "date", "price", "ingested_at"],
    },
    "merged_stock_data": {
        "time_col": "date",
        "cols": ["stock", "date", "price", "earnings_date", "reported_eps",
                 "surprise_percentage", "sector"],
    },
    "earnings": {
        "time_col": "earnings_date",
        "cols": ["stock", "earnings_date", "fiscal_end_date", "reported_eps",
                 "estimated_eps", "surprise_percentage", "ingested_at"],
    },
    "iv_snapshots": {
        "time_col": "snapshot_date",
        "cols": ["stock", "snapshot_date", "earnings_date", "days_to_earnings",
                 "current_price", "expiry_used", "atm_iv", "expected_move_pct",
                 "snapshot_hour"],
    },
    "eps_estimates": {
        "time_col": "snapshot_date",
        "cols": ["stock", "snapshot_date", "earnings_date", "eps_avg",
                 "eps_num_analysts", "eps_dispersion", "eps_revision_momentum",
                 "eps_trend_30d"],
    },
    "predictions": {
        "time_col": "prediction_asof_date",
        "cols": ["prediction_asof_date", "week_start", "stock", "earnings_date",
                 "tier", "risk_score", "is_high_conviction",
                 "pre_earnings_drift_flag", "model_version"],
    },
    "stock_data": {
        "time_col": "ingested_at",
        "cols": ["stock", "company_name", "sector", "sub_sector", "status",
                 "reason", "ingested_at"],
    },
}

# Fallback order when a table is not in TABLE_SPECS (e.g. one added later).
TIME_COL_FALLBACKS = ["date", "snapshot_date", "prediction_asof_date",
                      "earnings_date", "ingested_at"]

# output/peek/ rather than output/db_csv/ (existing full-table dumps, would collide)
# or output/output_<date>/ (wiped by get_run_output_dir on each pipeline run).
DEFAULT_CSV_DIR = os.path.join("output", "peek")

# With -n 0 --csv the user wants a complete file, not a complete screenful.
UNCAPPED_PRINT_ROWS = 25

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

DURATION_RE = re.compile(r"^(\d+)\s*([dwmy])?$", re.IGNORECASE)
DURATION_DAYS = {"d": 1, "w": 7, "m": 30, "y": 365}


def parse_window(text):
    """'7d' -> 7, '2w' -> 14, '3m' -> 90, '1y' -> 365, '30' -> 30, 'all' -> None."""
    text = text.strip().lower()
    if text in ("all", "*"):
        return None
    match = DURATION_RE.match(text)
    if not match:
        raise argparse.ArgumentTypeError(
            f"bad window {text!r} - use 7d, 2w, 3m, 1y, a plain number of days, or 'all'"
        )
    count, unit = match.groups()
    return int(count) * DURATION_DAYS[(unit or "d").lower()]


def safe_identifier(name, kind):
    if not IDENTIFIER_RE.match(name):
        sys.exit(f"refusing to interpolate unsafe {kind} name: {name!r}")
    return name


def attach_predictions(con):
    """predictions live in their own DB (config.PREDICTIONS_DB_PATH) so the weekly
    droplet sync, which overwrites breakwater.duckdb wholesale, cannot destroy them.
    Attach it and put both catalogs on the search path, so every caller here can still
    say `predictions` unqualified exactly as when it lived in the main DB."""
    if not os.path.exists(config.PREDICTIONS_DB_PATH):
        return
    con.execute(f"ATTACH '{config.PREDICTIONS_DB_PATH}' AS pred (READ_ONLY)")
    main_catalog = con.execute("SELECT current_database()").fetchone()[0]
    con.execute(f"SET search_path = '{main_catalog},pred'")


def list_tables(con):
    # SHOW ALL TABLES, not SHOW TABLES: the latter only sees the current catalog, which
    # would hide the attached predictions DB and report it as an unknown table.
    return [row.name for row in con.sql("SHOW ALL TABLES").df().itertuples()]


def table_columns(con, table):
    return [row[0] for row in con.sql(f"DESCRIBE {table}").fetchall()]


def resolve_time_col(con, table, override=None, strict=True):
    """Time column to filter on, or None if the table has none.

    strict=False is for the all-tables sweep: a --by column missing from one
    table should skip that table, not kill the whole run.
    """
    columns = table_columns(con, table)
    if override:
        if override not in columns:
            if not strict:
                return None
            sys.exit(f"{table} has no column {override!r}. Columns: {', '.join(columns)}")
        return override
    spec = TABLE_SPECS.get(table)
    if spec and spec["time_col"] in columns:
        return spec["time_col"]
    for candidate in TIME_COL_FALLBACKS:
        if candidate in columns:
            return candidate
    return None


def resolve_select_cols(con, table, requested, all_cols, strict=True):
    columns = table_columns(con, table)
    if requested:
        missing = [c for c in requested if c not in columns]
        if missing:
            if not strict:
                return None
            sys.exit(f"{table} has no column(s): {', '.join(missing)}. "
                     f"Columns: {', '.join(columns)}")
        return requested
    if all_cols:
        return columns
    spec = TABLE_SPECS.get(table)
    if spec:
        preview = [c for c in spec["cols"] if c in columns]
        if preview:
            return preview
    return columns


def build_filters(con, table, time_col, args):
    """Return (where_sql, params, window_description)."""
    clauses, params, description = [], [], "all rows"

    if time_col and args.window is not None:
        if args.anchor == "latest":
            anchor = con.execute(f"SELECT MAX({time_col}) FROM {table}").fetchone()[0]
            if anchor is None:
                return "", [], "table is empty"
            anchor = anchor.date() if isinstance(anchor, dt.datetime) else anchor
            anchor_label = f"latest {time_col} {anchor}"
        else:
            anchor = dt.date.today()
            anchor_label = f"today {anchor}"
        start = anchor - dt.timedelta(days=args.window)
        # Half-open upper bound: TIMESTAMP columns (ingested_at) would otherwise
        # lose everything after midnight on the anchor day, since the bound binds
        # as a date. For DATE columns "< anchor + 1 day" is the same as "<= anchor".
        clauses.append(f"{time_col} >= ? AND {time_col} < ?")
        params.extend([start, anchor + dt.timedelta(days=1)])
        description = f"{time_col} in [{start} .. {anchor}]  ({args.window}d back from {anchor_label})"
    elif time_col:
        description = f"all rows (ordered by {time_col})"

    if args.stocks and "stock" in table_columns(con, table):
        placeholders = ", ".join("?" for _ in args.stocks)
        clauses.append(f"stock IN ({placeholders})")
        params.extend([s.upper() for s in args.stocks])

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where, params, description


def limit_sql(limit):
    """(clause, params) for a row cap. limit=0 means no cap."""
    return (" LIMIT ?", [limit]) if limit else ("", [])


def order_clause(time_col, columns):
    """Newest first. Ties broken by ticker so repeated runs are stable."""
    if not time_col:
        return ""
    tiebreak = ", stock" if "stock" in columns else ""
    return f" ORDER BY {time_col} DESC{tiebreak}"


def csv_filename(table, time_col, args):
    """Self-describing name so different windows don't overwrite each other."""
    if args.window is None:
        span = "all"
    else:
        anchor = dt.date.today() if args.anchor == "today" else "latest"
        span = f"{args.window}d_to_{anchor}"
    parts = [table, time_col or "unsorted", span]
    if args.stocks:
        parts.append("-".join(s.upper() for s in args.stocks[:4]))
    if args.cols:
        # Otherwise a narrowed export silently overwrites the full-width one.
        parts.append("-".join(args.cols[:3]))
    if args.summary:
        parts.append("summary")
    return "_".join(parts) + ".csv"


def write_csv(frame, table, time_col, args):
    # NOTE: deliberately not output/output_<date>/ - get_run_output_dir() rmtree's
    # that folder on first call each process, which would delete these mid-run.
    os.makedirs(args.csv, exist_ok=True)
    path = os.path.join(args.csv, csv_filename(table, time_col, args))
    frame.to_csv(path, index=False)
    print(f"  -> wrote {len(frame):,} rows x {len(frame.columns)} cols to {path}")


def print_frame(frame):
    if frame.empty:
        print("  (no rows)")
        return
    with pd.option_context("display.width", 250,
                           "display.max_columns", 100,
                           "display.max_colwidth", 40):
        print(frame.to_string(index=False))


def overview(con, args):
    """Freshness and size of every table - the default when no table is named."""
    rows = []
    for table in list_tables(con):
        time_col = resolve_time_col(con, table)
        n_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        first = last = stale = None
        if time_col:
            first, last = con.execute(
                f"SELECT MIN({time_col}), MAX({time_col}) FROM {table}"
            ).fetchone()
            if last is not None:
                last_date = last.date() if isinstance(last, dt.datetime) else last
                stale = (dt.date.today() - last_date).days
        recent = None
        if time_col and args.window is not None:
            cutoff = dt.date.today() - dt.timedelta(days=args.window)
            recent = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE {time_col} >= ?", [cutoff]
            ).fetchone()[0]
        rows.append({
            "table": table,
            "rows": n_rows,
            "time_col": time_col or "-",
            "first": first,
            "last": last,
            "days_stale": stale,
            f"rows_last_{args.window}d": recent,
        })
    frame = pd.DataFrame(rows)
    if args.window is None:
        frame = frame.drop(columns=[c for c in frame.columns if c.startswith("rows_last_")])
    print(f"{config.DB_PATH}   ({len(frame)} tables, today is {dt.date.today()})\n")
    print_frame(frame)
    print("\nPeek at rows:  python -m utilities.peek_at_db <table> 7d [-s TICKER ...] [--summary]"
          "\nEvery table:   python -m utilities.peek_at_db all 7d"
          "\nExport to CSV: python -m utilities.peek_at_db all 7d --csv")


def peek_table(con, table, args, strict=True):
    time_col = resolve_time_col(con, table, args.by, strict=strict)
    columns = resolve_select_cols(con, table, args.cols, args.all_cols, strict=strict)
    if time_col is None and args.by:
        print(f"-- {table}: no {args.by!r} column, skipped")
        return
    if columns is None:
        print(f"-- {table}: missing requested column(s), skipped")
        return
    where, params, description = build_filters(con, table, time_col, args)

    total = con.execute(f"SELECT COUNT(*) FROM {table}{where}", params).fetchone()[0]
    print("=" * 100)
    print(f"{table}  -  {description}")
    # -n caps both the print and the CSV, except that -n 0 (no cap) with --csv
    # still trims the *print* - otherwise asking for a full export dumps the
    # whole window to the terminal as well.
    print_limit = args.limit or (UNCAPPED_PRINT_ROWS if args.csv else 0)
    shown = min(total, print_limit) if print_limit else total
    note = f" (all {total:,} in the CSV)" if args.csv and shown < total else ""
    print(f"{total:,} rows match" + (f", showing {shown:,}{note}" if not args.summary else ""))
    print("=" * 100)

    if total == 0:
        if time_col:
            newest = con.execute(f"SELECT MAX({time_col}) FROM {table}").fetchone()[0]
            print(f"  (no rows) - newest {time_col} in {table} is {newest}. "
                  f"Try a longer window or --anchor latest.")
        else:
            print("  (no rows)")
        return

    if args.summary:
        if not time_col:
            sys.exit(f"--summary needs a time column; {table} has none")
        has_stock = "stock" in table_columns(con, table)
        stock_expr = ", COUNT(DISTINCT stock) AS stocks" if has_stock else ""
        summary_sql = (f"SELECT {time_col}, COUNT(*) AS rows{stock_expr} FROM {table}{where} "
                       f"GROUP BY {time_col} ORDER BY {time_col} DESC")
        cap, cap_params = limit_sql(args.limit)
        export = con.execute(summary_sql + cap, params + cap_params).df()
        shown_frame = export.head(print_limit) if print_limit else export
        print_frame(shown_frame)
        if len(shown_frame) < len(export):
            print(f"  ... {len(shown_frame)} of {len(export)} dates shown")
        if args.csv:
            write_csv(export, table, time_col, args)
        return

    order = order_clause(time_col, columns)
    cap, cap_params = limit_sql(args.limit)
    frame = con.execute(
        f"SELECT {', '.join(columns)} FROM {table}{where}{order}"
        f"{limit_sql(print_limit)[0]}", params + limit_sql(print_limit)[1]
    ).df()
    print_frame(frame)

    if args.csv:
        # Same window and same -n cap as the display, so the file can never be
        # bigger than what you asked to look at. Wider though: every column,
        # since the terminal preview subset is a readability trim, not a filter.
        export_cols = args.cols or table_columns(con, table)
        export = con.execute(
            f"SELECT {', '.join(export_cols)} FROM {table}{where}"
            f"{order_clause(time_col, export_cols)}{cap}", params + cap_params
        ).df()
        write_csv(export, table, time_col, args)


def peek_all_tables(con, args):
    """One peek block per table - the rows themselves, not just the counts."""
    tables = list_tables(con)
    print(f"{config.DB_PATH}   ({len(tables)} tables, today is {dt.date.today()})\n")
    for table in tables:
        peek_table(con, table, args, strict=False)
        print()


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m utilities.peek_at_db",
        description="Peek at recent rows in the Breakwater DuckDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("table", nargs="?",
                        help="table to peek at, or 'all' for every table; "
                             "omit for a freshness overview")
    parser.add_argument("window", nargs="?", default="7d", type=parse_window,
                        help="time window: 7d, 2w, 3m, 1y, a plain number of days, or 'all' (default: 7d)")
    parser.add_argument("-s", "--stocks", nargs="+", metavar="TICKER",
                        help="restrict to these tickers")
    parser.add_argument("-n", "--limit", type=int, default=25,
                        help="max rows, for both the printed table and --csv "
                             "(default: 25; use 0 for no cap)")
    parser.add_argument("--by", metavar="COLUMN",
                        help="time column to filter and sort on (default: per-table, "
                             "e.g. earnings uses earnings_date - pass ingested_at to see what "
                             "was recently written instead)")
    parser.add_argument("--anchor", choices=["today", "latest"], default="today",
                        help="count the window back from today (default) or from the table's newest row")
    parser.add_argument("--summary", action="store_true",
                        help="row and ticker counts per date instead of the rows themselves")
    parser.add_argument("--cols", nargs="+", metavar="COLUMN", help="columns to show")
    parser.add_argument("--all-cols", action="store_true", help="show every column")
    parser.add_argument("--csv", nargs="?", const=DEFAULT_CSV_DIR, default=None, metavar="DIR",
                        help=f"also write the rows to CSV, one file per table "
                             f"(default dir: {DEFAULT_CSV_DIR}). Honours the window and -n; "
                             f"exports every column unless --cols is given.")
    parser.add_argument("--sql", metavar="QUERY", help="run an arbitrary read-only query instead")
    args = parser.parse_args(argv)

    con = duckdb.connect(config.DB_PATH, read_only=True)
    attach_predictions(con)
    try:
        if args.sql:
            frame = con.execute(args.sql).df()
            print_frame(frame)
            if args.csv:
                write_csv(frame, "sql_query", None, args)
            return
        if not args.table:
            overview(con, args)
            return
        if args.table.lower() in ("all", "*"):
            peek_all_tables(con, args)
            return

        tables = list_tables(con)
        if args.table not in tables:
            matches = [t for t in tables if args.table.lower() in t.lower()]
            if len(matches) != 1:
                sys.exit(f"unknown table {args.table!r}. Available: {', '.join(tables)}")
            args.table = matches[0]

        safe_identifier(args.table, "table")
        for column in (args.cols or []) + ([args.by] if args.by else []):
            safe_identifier(column, "column")

        peek_table(con, args.table, args)
    finally:
        con.close()


if __name__ == "__main__":
    main()
