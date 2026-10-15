"""Price ingestion keeps one adjustment basis per stock (audit H4).

The auditor's reproduction, through the real ingestion function with a stubbed provider:
stored pre-split prices of 100, then a 2:1 split after which the provider reports 50 for
every date. The old append-only design stored 100 ... 50 and the corrected three-session
reaction read -50%, marked available. Now the fetched window is reconciled: stored rows in
it are replaced, sessions missing in it are filled, and older rows are rescaled onto the
provider's new basis — the reaction is 0%.

Run with:  pytest testing/test_price_reconciliation.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import duckdb
import numpy as np
import pandas as pd
import pytest

import ingestion.fetch_prices as FP
from feature_engineering.announcement_timing import TARGET_AVAILABLE, resolve_event_anchors
from utilities.db_utilities import create_prices_table_if_not_exists

S = pd.bdate_range("2026-07-01", periods=40)
TODAY = S[-1]


def _db(rows):
    con = duckdb.connect(":memory:")
    create_prices_table_if_not_exists(con)
    df = pd.DataFrame(rows, columns=["stock", "date", "price"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ingested_at"] = pd.Timestamp("2026-07-20")
    con.register("t", df); con.execute("INSERT INTO prices SELECT * FROM t"); con.unregister("t")
    return con


def _stub(monkeypatch, provider: pd.DataFrame, stocks=("AAA",)):
    monkeypatch.setattr(FP, "read_stocks_to_fetch", lambda con, active_only=False: list(stocks))
    def download(tickers, start, end):
        p = provider[(provider["date"] >= pd.Timestamp(start)) & (provider["date"] < pd.Timestamp(end))]
        return p[p["stock"].isin(tickers)].copy()
    monkeypatch.setattr(FP, "_download_closes", download)


def _prices(con, stock="AAA"):
    return con.execute("SELECT date, price FROM prices WHERE stock = ? ORDER BY date",
                       [stock]).df().assign(date=lambda d: pd.to_datetime(d["date"]))


def test_a_split_after_ingestion_no_longer_manufactures_a_reaction(monkeypatch):
    split = S[30]
    stored = [("AAA", d, 100.0) for d in S[:30]]                 # ingested before the split
    con = _db(stored)
    provider = pd.DataFrame({"stock": "AAA", "date": S, "price": 50.0})   # re-adjusted history
    _stub(monkeypatch, provider)
    FP.incremental_ingest_all_prices_yf(con, today=TODAY)
    px = _prices(con)
    assert len(px) == len(S) and np.allclose(px["price"], 50.0)
    log = con.execute("SELECT stock, factor FROM price_basis_log").fetchall()
    assert log == [("AAA", 0.5)]

    # the corrected target across the split: AMC report the session before the split
    events = pd.DataFrame({"stock": ["AAA"], "earnings_date": [S[29]], "is_pending": [False],
                           "announce_window": ["AMC"]})
    daily = px.assign(stock="AAA")
    out = resolve_event_anchors(events, daily).iloc[0]
    assert out["reaction_3d_anchored_status"] == TARGET_AVAILABLE
    assert out["reaction_3d_anchored"] == pytest.approx(0.0, abs=1e-12)


def test_the_old_append_only_result_is_what_the_auditor_reported():
    """Control: the stored series the old design produced really does read -50%."""
    px = pd.DataFrame({"stock": "AAA", "date": S, "price": [100.0] * 30 + [50.0] * 10})
    events = pd.DataFrame({"stock": ["AAA"], "earnings_date": [S[29]], "is_pending": [False],
                           "announce_window": ["AMC"]})
    out = resolve_event_anchors(events, px).iloc[0]
    assert out["reaction_3d_anchored"] == pytest.approx(-0.5)


def test_a_session_missing_inside_the_window_is_filled(monkeypatch):
    """The 2026-05-19..21 hole: one stock ingested ahead of the rest no longer moves
    everyone's start date past sessions they never got."""
    stored = [("AAA", d, 10.0 + i) for i, d in enumerate(S[:35]) if d not in (S[30], S[31])]
    con = _db(stored)
    provider = pd.DataFrame({"stock": "AAA", "date": S, "price": [10.0 + i for i in range(len(S))]})
    _stub(monkeypatch, provider)
    FP.incremental_ingest_all_prices_yf(con, today=TODAY)
    px = _prices(con)
    assert list(px["date"]) == list(S) and np.allclose(px["price"], provider["price"])
    assert con.execute("SELECT COUNT(*) FROM price_basis_log").fetchone()[0] == 0


def test_an_unchanged_basis_rewrites_nothing_and_a_rerun_is_idempotent(monkeypatch):
    base = [100.0 * (1.001 ** i) for i in range(len(S))]
    con = _db([("AAA", d, p) for d, p in zip(S[:35], base[:35])])
    provider = pd.DataFrame({"stock": "AAA", "date": S, "price": base})
    _stub(monkeypatch, provider)
    old = _prices(con).set_index("date")["price"]
    r1 = FP.incremental_ingest_all_prices_yf(con, today=TODAY)
    after = _prices(con).set_index("date")["price"]
    assert r1[0]["factor"] == 1.0 and (after.loc[old.index] == old).all()
    FP.incremental_ingest_all_prices_yf(con, today=TODAY)
    pd.testing.assert_series_equal(after, _prices(con).set_index("date")["price"])


def test_older_history_is_rescaled_by_one_factor_so_its_returns_do_not_change(monkeypatch):
    rng = np.random.default_rng(3)
    early = pd.bdate_range("2026-01-02", periods=60)
    hist = 80 * np.cumprod(1 + rng.normal(0, 0.01, len(early)))
    stored = [("AAA", d, p) for d, p in zip(early, hist)] + [("AAA", d, 100.0) for d in S[:30]]
    con = _db(stored)
    provider = pd.DataFrame({"stock": "AAA", "date": S, "price": [99.0] * len(S)})  # a dividend
    _stub(monkeypatch, provider)
    FP.incremental_ingest_all_prices_yf(con, today=TODAY)
    px = _prices(con).set_index("date")["price"]
    before = pd.Series(hist, index=early).pct_change().dropna()
    assert np.allclose(px.loc[early].pct_change().dropna(), before, atol=1e-12)
    assert px.loc[early[0]] == pytest.approx(hist[0] * 0.99)


def test_a_window_that_cannot_be_matched_to_the_history_is_left_alone(monkeypatch):
    con = _db([("AAA", d, 100.0) for d in pd.bdate_range("2025-01-02", periods=10)])
    provider = pd.DataFrame({"stock": "AAA", "date": S, "price": 50.0})   # no common date
    _stub(monkeypatch, provider)
    before = _prices(con)
    assert FP.incremental_ingest_all_prices_yf(con, since="2026-07-01", today=TODAY) == []
    pd.testing.assert_frame_equal(before, _prices(con))


def test_reconcile_is_pure_and_uses_the_earliest_common_date():
    stored = pd.DataFrame({"date": S[:5], "price": [10.0, 10.0, 20.0, 20.0, 20.0]})
    fresh = pd.DataFrame({"date": S[1:7], "price": [5.0, 10.0, 10.0, 10.0, 10.0, 10.0]})
    rows, factor = FP.reconcile_stock_prices(stored, fresh)
    assert factor == pytest.approx(0.5) and list(rows["date"]) == list(S[1:7])
