"""Tests for `backfills/build_date_corrections.py`.

Synthetic vendor records go through the real `research.massive.normalize`, so the
identity-hazard rule under test is the production one.
"""
import pandas as pd

from backfills.build_date_corrections import CORRECTION_COLS, build
from research.massive import normalize

SNAP = "earnings_20260910T180412Z"


def _record(i, ticker="KO", date="2026-02-10", time="06:55:00"):
    return {"benzinga_id": f"id{i}", "ticker": ticker, "company_name": "Fake Co",
            "date": date, "time": time, "date_status": "confirmed", "currency": "USD",
            "importance": 1, "fiscal_year": 2025, "fiscal_period": "Q4",
            "last_updated": "2026-02-10T12:00:00Z"}


def _earnings(*rows):
    """rows: (stock, date, eps, announce_ts_source)"""
    return pd.DataFrame([{"stock": s, "earnings_date": pd.Timestamp(d), "reported_eps": eps,
                          "announce_ts_source": src} for s, d, eps, src in rows])


def _yf(*rows):
    return pd.DataFrame([{"stock": s, "earnings_date": pd.Timestamp(d).date()} for s, d in rows],
                        columns=["stock", "earnings_date"])


def _run(records, earnings, yf):
    vendor = normalize.normalize(records, snapshot_id=SNAP)
    return build(earnings, vendor, SNAP, yf)


def test_a_late_date_confirmed_by_both_sources_is_moved_with_the_time():
    out, stats = _run([_record(1)], _earnings(("KO", "2026-02-17", 0.58, None)),
                      _yf(("KO", "2026-02-10")))
    assert list(out.columns[:len(CORRECTION_COLS)]) == CORRECTION_COLS
    r = out.iloc[0]
    assert (r.stock, str(r.old_date), str(r.new_date), r.action) == \
        ("KO", "2026-02-17", "2026-02-10", "move")
    assert r.announce_ts_ny == pd.Timestamp("2026-02-10 06:55")
    assert r.announce_ts_source == f"massive_benzinga:{SNAP}:id1"


def test_an_existing_row_on_the_right_date_makes_it_a_delete():
    out, _ = _run([_record(1)],
                  _earnings(("KO", "2026-02-10", 0.58, "yfinance_earnings_dates"),
                            ("KO", "2026-02-17", 0.58, None)),
                  _yf())
    assert list(out.action) == ["delete_old"]
    assert str(out.iloc[0].old_date) == "2026-02-17"


def test_benzinga_alone_is_not_enough():
    """LOW 2014-05-21 is right as stored; Benzinga says 04-28 and yfinance has nothing."""
    out, stats = _run([_record(1, ticker="LOW", date="2014-04-28")],
                      _earnings(("LOW", "2014-05-21", 0.60, None)), _yf())
    assert out.empty and stats["benzinga_only_left_alone"] == 1


def test_yfinance_backing_the_stored_date_blocks_the_correction():
    out, stats = _run([_record(1)], _earnings(("KO", "2026-02-17", 0.58, None)),
                      _yf(("KO", "2026-02-17")))
    assert out.empty and stats["yfinance_disagrees"] == 1


def test_one_day_differences_and_exact_matches_are_left_alone():
    out, stats = _run([_record(1, date="2026-02-10"), _record(2, date="2025-11-04")],
                      _earnings(("KO", "2026-02-11", 0.58, None),
                                ("KO", "2025-11-04", 0.50, None)),
                      _yf(("KO", "2026-02-10")))
    assert out.empty
    assert stats["off_by_one_left_alone"] == 1 and stats["exact"] == 1


def test_a_record_nearer_another_stored_event_is_not_used():
    out, stats = _run([_record(1, date="2026-02-10")],
                      _earnings(("KO", "2026-02-17", 0.58, None),
                                ("KO", "2026-02-08", 0.50, None)),
                      _yf(("KO", "2026-02-10")))
    # the record belongs to the 02-08 row (2 days away), not the 02-17 one (7 days)
    assert list(out.old_date.astype(str)) == ["2026-02-08"]
    assert stats["benzinga_nearer_another_event"] == 1


def test_placeholders_and_future_events_are_never_corrected():
    out, _ = _run([_record(1, date="2026-02-10")],
                  _earnings(("KO", "2026-02-17", None, None)), _yf(("KO", "2026-02-10")))
    assert out.empty
