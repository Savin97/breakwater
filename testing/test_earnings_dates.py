"""Earnings-date quality: which of two rows for one report survives the duplicate
cleanup, and the one-time correction of dates stored days to weeks late.

Run with:  pytest testing/test_earnings_dates.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import duckdb
import pandas as pd
import pytest

from ingestion.fetch_earnings_dates import EARNINGS_INSERT_COLS, _INSERT_COL_SQL
from utilities.db_utilities import (create_earnings_table_if_not_exists,
                                    clean_duplicate_earnings_from_db)
from backfills.apply_date_corrections import CORRECTION_COLS, apply, read_corrections

BZ = "massive_benzinga:snap:id"
YF = "yfinance_earnings_dates"


def _db(rows):
    """rows: [(stock, date, eps, announce_ts_ny, source)]"""
    con = duckdb.connect(":memory:")
    create_earnings_table_if_not_exists(con)
    df = pd.DataFrame([{
        "stock": s, "earnings_date": pd.Timestamp(d).date(), "fiscal_end_date": None,
        "reported_eps": eps, "estimated_eps": 1.0, "surprise_percentage": None,
        "ingested_at": pd.Timestamp("2026-02-18"),
        "announce_ts_ny": pd.Timestamp(ts) if ts else pd.NaT, "announce_ts_source": src,
        "announce_ts_observed_at": pd.Timestamp("2026-09-10") if ts else pd.NaT,
    } for s, d, eps, ts, src in rows])[EARNINGS_INSERT_COLS]
    con.register("tmp_earnings_df", df)
    con.execute(f"INSERT INTO earnings ({_INSERT_COL_SQL}) "
                f"SELECT {_INSERT_COL_SQL} FROM tmp_earnings_df")
    con.unregister("tmp_earnings_df")
    return con


def _dates(con, stock="KO"):
    return [str(r[0]) for r in con.execute(
        "SELECT earnings_date FROM earnings WHERE stock = ? ORDER BY earnings_date",
        [stock]).fetchall()]


# ── duplicate cleanup ────────────────────────────────────────────────────────

def test_cleanup_keeps_the_observed_date_over_a_late_unbacked_one():
    """KO: yfinance observed the report on 02-10; the AlphaVantage row says 02-17 and
    carries no timestamp. The old rule kept the later (wrong) date."""
    con = _db([("KO", "2026-02-10", 0.58, "2026-02-10 06:45", YF),
               ("KO", "2026-02-17", 0.58, None, None)])
    clean_duplicate_earnings_from_db(con)
    assert _dates(con) == ["2026-02-10"]


def test_cleanup_prefers_benzinga_over_yfinance():
    """BX: yfinance's old date is a day early; Benzinga confirms the stored one."""
    con = _db([("KO", "2023-04-19", 0.97, "2023-04-19 20:00", YF),
               ("KO", "2023-04-20", 0.97, "2023-04-20 06:55", BZ)])
    clean_duplicate_earnings_from_db(con)
    assert _dates(con) == ["2023-04-20"]


def test_cleanup_still_keeps_the_later_date_on_equal_evidence():
    con = _db([("KO", "2026-02-10", 0.58, None, None),
               ("KO", "2026-02-17", 0.58, None, None)])
    clean_duplicate_earnings_from_db(con)
    assert _dates(con) == ["2026-02-17"]


def test_cleanup_still_prefers_a_confirmed_row_over_a_placeholder():
    con = _db([("KO", "2026-02-10", None, "2026-02-10 06:45", BZ),
               ("KO", "2026-02-17", 0.58, None, None)])
    clean_duplicate_earnings_from_db(con)
    assert _dates(con) == ["2026-02-17"]


# ── one-time date corrections ────────────────────────────────────────────────

def _corr(rows):
    return pd.DataFrame([dict(zip(CORRECTION_COLS, (
        s, pd.Timestamp(o).date(), pd.Timestamp(n).date(), a,
        pd.Timestamp(ts) if ts else pd.NaT, src if ts else None,
        pd.Timestamp("2026-09-10 14:04:12") if ts else pd.NaT)))
        for s, o, n, a, ts, src in rows])


def test_move_changes_the_date_keeps_eps_and_attaches_the_time():
    con = _db([("KO", "2026-02-17", 0.58, None, None)])
    stats = apply(con, _corr([("KO", "2026-02-17", "2026-02-10", "move",
                               "2026-02-10 06:45", BZ)]))
    row = con.execute("SELECT earnings_date, reported_eps, announce_ts_ny, "
                      "announce_ts_source FROM earnings").fetchall()
    assert stats["moved"] == 1
    assert row == [(pd.Timestamp("2026-02-10").date(), 0.58,
                    pd.Timestamp("2026-02-10 06:45"), BZ)]


def test_move_never_overwrites_an_existing_time():
    con = _db([("KO", "2026-02-17", 0.58, "2026-02-17 16:00", YF)])
    apply(con, _corr([("KO", "2026-02-17", "2026-02-10", "move", "2026-02-10 06:45", BZ)]))
    assert con.execute("SELECT announce_ts_ny, announce_ts_source FROM earnings").fetchall() \
        == [(pd.Timestamp("2026-02-17 16:00"), YF)]


def test_move_without_a_usable_time_moves_only_the_date():
    con = _db([("KO", "2026-02-17", 0.58, None, None)])
    apply(con, _corr([("KO", "2026-02-17", "2026-02-10", "move", None, BZ)]))
    assert con.execute("SELECT earnings_date, announce_ts_ny, announce_ts_source "
                       "FROM earnings").fetchall() == [(pd.Timestamp("2026-02-10").date(), None, None)]


def test_delete_old_removes_the_late_duplicate_only():
    con = _db([("KO", "2026-02-10", 0.58, "2026-02-10 06:45", YF),
               ("KO", "2026-02-17", 0.58, None, None)])
    stats = apply(con, _corr([("KO", "2026-02-17", "2026-02-10", "delete_old", None, BZ)]))
    assert stats["deleted_old"] == 1
    assert _dates(con) == ["2026-02-10"]


def test_actions_are_skipped_when_the_database_is_not_as_expected():
    con = _db([("KO", "2026-02-10", 0.58, None, None),     # new row already there
               ("KO", "2026-05-01", 0.60, None, None)])
    stats = apply(con, _corr([
        ("KO", "2026-02-17", "2026-02-10", "move", None, BZ),        # old row gone
        ("KO", "2026-05-01", "2026-04-28", "delete_old", None, BZ),  # new row missing
    ]))
    assert stats["skipped_old_row_gone"] == 1
    assert stats["skipped_new_row_missing"] == 1
    assert _dates(con) == ["2026-02-10", "2026-05-01"]

    con = _db([("KO", "2026-02-10", 0.58, None, None), ("KO", "2026-02-17", 0.58, None, None)])
    stats = apply(con, _corr([("KO", "2026-02-17", "2026-02-10", "move", None, BZ)]))
    assert stats["skipped_new_row_appeared"] == 1
    assert _dates(con) == ["2026-02-10", "2026-02-17"]


def test_apply_is_idempotent_and_dry_run_writes_nothing():
    corr = _corr([("KO", "2026-02-17", "2026-02-10", "move", "2026-02-10 06:45", BZ)])
    con = _db([("KO", "2026-02-17", 0.58, None, None)])
    dry = apply(con, corr, dry_run=True)
    assert dry["moved"] == 1 and _dates(con) == ["2026-02-17"]
    apply(con, corr)
    second = apply(con, corr)
    assert second["moved"] == 0 and second["skipped_old_row_gone"] == 1
    assert _dates(con) == ["2026-02-10"]


def test_read_corrections_rejects_bad_files(tmp_path):
    bad = _corr([("KO", "2026-02-17", "2026-02-10", "move", None, BZ)])
    bad.loc[0, "action"] = "rewrite"
    bad.to_parquet(tmp_path / "bad.parquet")
    with pytest.raises(ValueError, match="unknown actions"):
        read_corrections(tmp_path / "bad.parquet")
    dup = pd.concat([_corr([("KO", "2026-02-17", "2026-02-10", "move", None, BZ)])] * 2)
    dup.to_parquet(tmp_path / "dup.parquet")
    with pytest.raises(ValueError, match="more than once"):
        read_corrections(tmp_path / "dup.parquet")
