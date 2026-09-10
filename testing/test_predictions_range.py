"""
Tests for the multi-week predictions range (analysis/predictions_range.py) and the
week-block arithmetic it rests on (utilities/data_utilities.week_block_window).

The window math is pure, so it is tested against fixed `today` values across every day
of the week — the bugs this code can have are all off-by-one-week bugs that only show
up on a Monday, a Friday, or a weekend.

The frame assembly is tested on a synthetic event frame: real history would make the
assertions depend on whatever the last pipeline run happened to produce.

Run with:  pytest testing/test_predictions_range.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from utilities.data_utilities import week_block_window, work_week_window
from analysis.predictions_range import predictions_range, format_report, _pre_audit_upcoming

# 2026-09-07 is a Monday.
MON, TUE, WED, THU, FRI, SAT, SUN = [f"2026-09-{d:02d}" for d in range(7, 14)]


# ── week_block_window ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("today", [MON, TUE, WED, THU, FRI, SAT, SUN])
def test_window_is_always_a_whole_monday_to_friday_block(today):
    for kw in (dict(weeks_back=3), dict(weeks_forward=3), dict(monday="2026-06-01", weeks=3)):
        start, end = week_block_window(today=today, **kw)
        assert start.weekday() == 0, kw
        assert end.weekday() == 4, kw
        assert (end - start).days == 4 + 7 * 2, kw


@pytest.mark.parametrize("today", [MON, TUE, WED, THU, FRI])
def test_weeks_back_excludes_the_part_spent_current_week(today):
    """The current week is never in a backward window, whatever weekday it is run —
    the same rule work_week_window applies going forward."""
    start, end = week_block_window(weeks_back=1, today=today)
    assert (start.date().isoformat(), end.date().isoformat()) == ("2026-08-31", "2026-09-04")


@pytest.mark.parametrize("today", [SAT, SUN])
def test_weeks_back_on_a_weekend_counts_the_finished_week(today):
    """Saturday's Mon-Fri is over, so it is the last complete block — matching
    last_week_results._week_bounds, which snaps a weekend forward to Monday."""
    start, end = week_block_window(weeks_back=1, today=today)
    assert (start.date().isoformat(), end.date().isoformat()) == ("2026-09-07", "2026-09-11")


def test_weeks_back_n_blocks_are_contiguous_and_end_last_friday():
    start, end = week_block_window(weeks_back=12, today=WED)
    assert end.date().isoformat() == "2026-09-04"          # Friday of last week
    assert (end - start).days == 4 + 7 * 11                # 12 contiguous blocks


@pytest.mark.parametrize("today", [MON, TUE, WED, THU, FRI, SAT, SUN])
@pytest.mark.parametrize("n", [1, 2, 4])
def test_weeks_forward_is_identical_to_the_published_digest_window(today, n):
    """A forward window must be the SAME window the weekly digest and the predictions
    snapshot select on, or the sheet and the email disagree about a week."""
    assert week_block_window(weeks_forward=n, today=today) == work_week_window(today=today, weeks=n)


def test_backward_and_forward_windows_never_overlap():
    for today in (MON, TUE, WED, THU, FRI, SAT, SUN):
        _, back_end = week_block_window(weeks_back=1, today=today)
        fwd_start, _ = week_block_window(weeks_forward=1, today=today)
        assert back_end < fwd_start


def test_explicit_monday_must_be_a_monday():
    with pytest.raises(ValueError, match="not a Monday"):
        week_block_window(monday="2026-09-09")


def test_exactly_one_direction_is_required():
    for kw in (dict(), dict(weeks_back=2, weeks_forward=2), dict(weeks_back=1, monday=MON)):
        with pytest.raises(ValueError, match="exactly one"):
            week_block_window(**kw)


@pytest.mark.parametrize("kw", [dict(weeks_back=0), dict(weeks_forward=0), dict(weeks_back=-3)])
def test_zero_or_negative_weeks_is_refused(kw):
    with pytest.raises(ValueError, match=">= 1"):
        week_block_window(**kw)


# ── predictions_range ─────────────────────────────────────────────────────────

TODAY = "2026-09-06"   # a Sunday: back window = Aug 31-Sep 4, forward = Sep 7-11


@pytest.fixture
def events():
    """Two stocks. AAA reported in the last complete week and has an upcoming call next
    week; BBB reported last week only. CCC is a dead feed: a pending row whose
    earnings_date has already gone by."""
    rows = [
        # stock, earnings_date, date(score asof), pending, bucket, score, r3d
        ("AAA", "2026-06-03", "2026-06-03", False, "Normal",     50.0,  0.02),
        ("AAA", "2026-09-02", "2026-09-02", False, "High Alert", 90.0,  0.11),
        ("AAA", "2026-09-09", "2026-09-04", True,  "Elevated",   75.0,  np.nan),
        ("BBB", "2026-09-03", "2026-09-03", False, "Elevated",   74.0, -0.06),
        ("CCC", "2026-02-11", "2026-02-05", True,  "Normal",     40.0,  np.nan),
    ]
    df = pd.DataFrame(rows, columns=["stock", "earnings_date", "date", "is_pending",
                                     "earnings_explosiveness_bucket",
                                     "earnings_explosiveness_score", "reaction_3d"])
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df["score_asof_date"] = pd.to_datetime(df["date"])
    df["sector"] = "Tech"
    df["risk_score"] = df["earnings_explosiveness_score"]
    df["earnings_explosiveness_bucket_structural"] = df["earnings_explosiveness_bucket"]
    df["stock_bucket_lift"] = 1.0
    df["is_high_conviction"] = False
    df["pre_earnings_drift_flag"] = ""
    df["surprise_momentum_flag"] = ""
    df["reaction_1d"] = df["reaction_3d"]
    df["reaction_5d"] = df["reaction_3d"]
    df["abs_reaction_3d"] = df["reaction_3d"].abs()
    return df


def test_weeks_back_returns_completed_events_only(events):
    out = predictions_range(weeks_back=1, events_df=events, today=TODAY, pre_audit=False)
    assert set(out["stock"]) == {"AAA", "BBB"}
    assert (out["row_kind"] == "history").all()
    assert out["earnings_date"].between("2026-08-31", "2026-09-04").all()
    # The June AAA event is outside the window and must not leak in.
    assert not (out["earnings_date"] < "2026-08-31").any()


def test_weeks_forward_returns_pending_events_only(events):
    out = predictions_range(weeks_forward=1, events_df=events, today=TODAY, pre_audit=False)
    assert list(out["stock"]) == ["AAA"]
    assert (out["row_kind"] == "upcoming").all()
    assert out["days_to_earnings"].tolist() == [3.0]


def test_a_pending_row_whose_date_has_passed_is_dropped(events):
    """CCC's feed stopped; its pending row still carries a February date. It is a stale
    feed, not a call, and must never be reported as either history or upcoming."""
    for kw in (dict(weeks_back=40), dict(monday="2026-02-09")):
        out = predictions_range(events_df=events, today=TODAY, pre_audit=False, **kw)
        assert "CCC" not in set(out["stock"])


def test_realized_outcome_flags_use_the_legacy_target(events):
    out = predictions_range(weeks_back=1, events_df=events, today=TODAY, pre_audit=False)
    aaa = out[out["stock"] == "AAA"].iloc[0]
    bbb = out[out["stock"] == "BBB"].iloc[0]
    assert bool(aaa["moved_5pct"]) and bool(aaa["moved_8pct"])       # +11%
    assert bool(bbb["moved_5pct"]) and not bool(bbb["moved_8pct"])   # -6%, absolute


def test_no_anchored_columns_are_reported(events):
    """Phase 2's corrected target is a parallel measurement no threshold has been re-fit
    against. It must not appear on a 0.3.1 predictions sheet."""
    events = events.copy()
    events["abs_reaction_3d_anchored"] = 0.99
    events["reaction_3d_anchored"] = 0.99
    out = predictions_range(weeks_back=1, events_df=events, today=TODAY, pre_audit=False)
    assert not [c for c in out.columns if "anchored" in c]


def test_upcoming_carries_both_the_current_and_the_pre_audit_call(events):
    """The pre-audit selection reaches back to the stock's last COMPLETED event, so AAA's
    upcoming call reads High Alert (its Sep 2 event) instead of the pending Elevated."""
    daily = pd.DataFrame([
        # the stock's final daily row: upcoming date present, tier columns blank
        {"stock": "AAA", "date": pd.Timestamp("2026-09-04"),
         "earnings_date": pd.Timestamp("2026-09-09"),
         "earnings_explosiveness_bucket": np.nan, "earnings_explosiveness_score": np.nan,
         "earnings_explosiveness_bucket_structural": np.nan, "risk_score": np.nan,
         "stock_bucket_lift": np.nan, "is_high_conviction": False,
         "pre_earnings_drift_flag": np.nan, "surprise_momentum_flag": np.nan},
        # the last completed earnings day, where the tier columns actually live
        {"stock": "AAA", "date": pd.Timestamp("2026-09-02"),
         "earnings_date": pd.Timestamp("2026-09-02"),
         "earnings_explosiveness_bucket": "High Alert", "earnings_explosiveness_score": 90.0,
         "earnings_explosiveness_bucket_structural": "High Alert", "risk_score": 90.0,
         "stock_bucket_lift": 1.0, "is_high_conviction": False,
         "pre_earnings_drift_flag": "", "surprise_momentum_flag": ""},
    ])
    out = predictions_range(weeks_forward=1, events_df=events, daily_df=daily, today=TODAY)
    row = out.iloc[0]
    assert row["earnings_explosiveness_bucket"] == "Elevated"
    assert row["earnings_explosiveness_bucket_pre_audit"] == "High Alert"
    assert bool(row["pre_audit_differs"]) is True


def test_pre_audit_reproduction_skips_nan_per_column(events):
    """The NaN skipping IS the bug being reproduced — a .last() that respected the final
    row would return NaN, not the stale tier, and the comparison would be empty."""
    daily = pd.DataFrame({
        "stock": ["AAA", "AAA"],
        "date": pd.to_datetime(["2026-09-02", "2026-09-04"]),
        "earnings_date": pd.to_datetime(["2026-09-02", "2026-09-09"]),
        "earnings_explosiveness_bucket": ["High Alert", np.nan],
        "earnings_explosiveness_score": [90.0, np.nan],
        "earnings_explosiveness_bucket_structural": ["High Alert", np.nan],
        "risk_score": [90.0, np.nan], "stock_bucket_lift": [1.0, np.nan],
        "is_high_conviction": [False, False],
        "pre_earnings_drift_flag": ["", np.nan], "surprise_momentum_flag": ["", np.nan],
    })
    pa = _pre_audit_upcoming(daily).iloc[0]
    assert pa["earnings_date_pre_audit"] == pd.Timestamp("2026-09-09")   # upcoming event
    assert pa["earnings_explosiveness_bucket_pre_audit"] == "High Alert"  # stale tier


def test_history_rows_get_no_pre_audit_call(events):
    """Reproducing what .last() would have said in a past week needs the daily frame as
    it stood then. Today's frame would answer with an event that has since completed, so
    the column is left blank rather than filled with a lookahead."""
    out = predictions_range(monday="2026-08-31", weeks=2, events_df=events,
                            daily_df=pd.DataFrame(columns=["stock", "date", "earnings_date"]),
                            today=TODAY)
    hist = out[out["row_kind"] == "history"]
    assert len(hist) == 2
    assert hist["earnings_explosiveness_bucket_pre_audit"].isna().all()
    assert hist["pre_audit_differs"].isna().all()


def test_a_window_spanning_today_carries_both_halves(events):
    out = predictions_range(monday="2026-08-31", weeks=2, events_df=events,
                            today=TODAY, pre_audit=False)
    assert set(out["row_kind"]) == {"history", "upcoming"}
    assert out.sort_values("week_start")["week_start"].is_monotonic_increasing


def test_empty_window_returns_empty_and_formats(events):
    out = predictions_range(monday="2026-07-06", events_df=events, today=TODAY, pre_audit=False)
    assert out.empty
    text = format_report(out, pd.Timestamp("2026-07-06"), pd.Timestamp("2026-07-10"))
    assert "No events." in text


def test_report_renders_both_halves(events):
    out = predictions_range(monday="2026-08-31", weeks=2, events_df=events,
                            today=TODAY, pre_audit=False)
    text = format_report(out, pd.Timestamp("2026-08-31"), pd.Timestamp("2026-09-11"))
    assert "Week of Aug 31, 2026" in text and "Week of Sep 7, 2026" in text
    assert "HISTORY — realized hit rates" in text
