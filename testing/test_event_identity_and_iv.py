"""Earnings events survive report-day price gaps; IV context sits only beside its event.

Event identities (audit M1): through the REAL path — the stage-2 merge
(merge_prices_earnings_dates) → stage 3 → build_event_frame — a catalog event whose
ticker has no price row on the report date used to vanish. It now stays, with an explicit
status; strict endpoints still apply (no price is invented, no date rolled).

IV context (audit M2): the IV join carries the latest snapshot forward, so September IV
expiring September 18 could sit beside November 10 earnings. It is now blanked unless the
snapshot was taken for this event.

Run with:  pytest testing/test_event_identity_and_iv.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from feature_engineering.announcement_timing import TARGET_AVAILABLE
from pipeline.events import REPORT_DAY_COLS, attach_iv_context_status, build_event_frame
from pipeline.stage3 import stage3
from testing.test_pipeline import _build_stage2_df
from utilities.data_utilities import merge_prices_earnings_dates

SESSIONS = pd.bdate_range("2024-01-02", periods=160)
D_BMO, D_AMC = SESSIONS[100], SESSIONS[120]          # report days with NO price row for AAA
D_SAT = pd.Timestamp("2024-06-15")                    # a Saturday report date
D_OK = SESSIONS[60]                                   # an ordinary report day


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(7)
    prices = []
    for stock in ("AAA", "BBB"):
        px = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(SESSIONS)))
        for d, p in zip(SESSIONS, px):
            if stock == "AAA" and d in (D_BMO, D_AMC):
                continue                               # the ticker's ingestion gap
            prices.append({"stock": stock, "date": d, "price": p})
    prices = pd.DataFrame(prices)
    earnings = pd.DataFrame({
        "stock": ["AAA"] * 4 + ["BBB"],
        "earnings_date": [D_OK, D_BMO, D_AMC, D_SAT, D_OK],
        "reported_eps": 1.0, "estimated_eps": 0.9, "surprise_percentage": 5.0})
    daily = merge_prices_earnings_dates(prices.sort_values("date"),
                                        earnings.sort_values("earnings_date"))
    template = _build_stage2_df(n_days=10)
    for col in template.columns:
        if col not in daily.columns:
            daily[col] = "Tech" if col in ("sector", "sub_sector") else np.nan
    daily = stage3(daily[template.columns].copy())
    timing = pd.DataFrame({
        "stock": ["AAA"] * 4 + ["BBB"],
        "earnings_date": [D_OK, D_BMO, D_AMC, D_SAT, D_OK],
        "announce_ts_ny": [D_OK + pd.Timedelta(hours=7), D_BMO + pd.Timedelta(hours=7),
                           D_AMC + pd.Timedelta(hours=16, minutes=5), D_SAT + pd.Timedelta(hours=8),
                           D_OK + pd.Timedelta(hours=16, minutes=5)],
        "announce_ts_source": "test"})
    ev = build_event_frame(daily, timing).set_index(["stock", "earnings_date"])
    return ev, prices.set_index(["stock", "date"])["price"]


def test_every_catalog_event_inside_the_price_history_survives(frame):
    ev, _ = frame
    completed = ev[~ev["is_pending"].astype(bool)]
    assert set(completed.index) == {("AAA", D_OK), ("AAA", D_BMO), ("AAA", D_AMC),
                                    ("AAA", D_SAT), ("BBB", D_OK)}


def test_bmo_event_with_a_missing_report_day_keeps_its_strict_outcome(frame):
    """Anchor D-1 and endpoint D+2 exist, so the corrected outcome is computable — the
    report-day row is not part of a BMO window."""
    ev, px = frame
    r = ev.loc[("AAA", D_BMO)]
    assert r["anchor_session_status"] == "unresolved_price_gap"
    assert r["anchor_status"] == "resolved" and r["anchor_date"] == SESSIONS[99]
    assert r["reaction_3d_anchored_status"] == TARGET_AVAILABLE
    want = px[("AAA", SESSIONS[102])] / px[("AAA", SESSIONS[99])] - 1
    assert r["reaction_3d_anchored"] == pytest.approx(want, abs=1e-12)


def test_amc_event_with_a_missing_report_day_stays_unresolved(frame):
    ev, _ = frame
    r = ev.loc[("AAA", D_AMC)]
    assert r["anchor_status"] == "unresolved_price_gap"
    assert r["reaction_3d_anchored_status"] == "unresolved_price_gap"
    assert pd.isna(r["reaction_3d_anchored"])


def test_weekend_dated_event_is_kept_and_never_rolled(frame):
    ev, _ = frame
    r = ev.loc[("AAA", D_SAT)]
    assert r["anchor_session_status"] == "unresolved_no_session"
    assert r["anchor_status"] == "unresolved_no_session" and pd.isna(r["reaction_3d_anchored"])


def test_built_events_carry_no_report_day_price(frame):
    ev, _ = frame
    for key in [("AAA", D_BMO), ("AAA", D_AMC), ("AAA", D_SAT)]:
        r = ev.loc[key]
        assert r["date"] == key[1]
        for col in REPORT_DAY_COLS + ["reaction_3d", "abs_reaction_3d"]:
            assert pd.isna(r[col]), (key, col)


def test_events_with_a_report_day_row_are_unchanged(frame):
    ev, px = frame
    r = ev.loc[("AAA", D_OK)]
    assert r["price"] == px[("AAA", D_OK)] and r["anchor_session_status"] == "ok"


# ── IV context ────────────────────────────────────────────────────────────────

def _iv_events():
    return pd.DataFrame({
        "stock": ["AAA", "AAA", "BBB", "CCC"],
        "earnings_date": pd.to_datetime(["2026-09-15", "2026-11-10", "2026-11-10", "2026-11-10"]),
        "is_pending": [False, True, True, True],
        "expected_move_pct": [0.05, 0.06, 0.07, 0.04], "atm_iv": [0.4, 0.5, 0.6, 0.3],
        "iv_snapshot_date": pd.to_datetime(["2026-09-01", "2026-09-01", "2026-11-03", "2026-11-03"]),
        "iv_expiry_used": pd.to_datetime(["2026-09-18", "2026-09-18", "2026-11-13", "2026-11-06"]),
    })


def test_iv_from_the_previous_report_is_not_shown_beside_the_next():
    """The auditor's case: September IV (expiry September 18) beside November 10."""
    out = attach_iv_context_status(_iv_events()).set_index(["stock", "earnings_date"])
    nov = pd.Timestamp("2026-11-10")
    assert out.loc[("AAA", pd.Timestamp("2026-09-15")), "iv_context_status"] == "matched"
    stale = out.loc[("AAA", nov)]
    assert stale["iv_context_status"] == "suppressed_snapshot_before_previous_report"
    assert pd.isna(stale["expected_move_pct"]) and pd.isna(stale["atm_iv"]) \
        and pd.isna(stale["iv_snapshot_date"])
    assert out.loc[("BBB", nov), "iv_context_status"] == "matched"
    assert out.loc[("BBB", nov), "expected_move_pct"] == 0.07
    short = out.loc[("CCC", nov)]                       # expiry before the report
    assert short["iv_context_status"] == "suppressed_expiry_not_after_report"
    assert pd.isna(short["expected_move_pct"])


def test_customer_upcoming_file_omits_suppressed_iv(tmp_path):
    from streamlit_dash.streamlit_export import export_upcoming_df
    ev = attach_iv_context_status(_iv_events())
    ev["earnings_date"] = pd.Timestamp.today().normalize() + pd.Timedelta(days=10)
    ev["iv_snapshot_date"] = ev["iv_snapshot_date"]
    for c in ("risk_score", "model_b_hist_mean_abs"):
        ev[c] = [50.0, 50.0, 60.0, 40.0]
    ev["risk_tier"], ev["model_b_n_prior"], ev["model_version"] = "Normal", 10, "0.4.0"
    ev["score_asof_date"] = pd.Timestamp.today().normalize()
    ev["sector"] = ev["sub_sector"] = "Tech"
    path = tmp_path / "u.parquet"
    export_upcoming_df(ev, str(path))
    up = pd.read_parquet(path).set_index("stock")
    assert pd.isna(up.at["AAA", "expected_move_pct"]) and up.at["BBB", "expected_move_pct"] == 0.07
