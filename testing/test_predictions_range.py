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
from analysis.predictions_range import predictions_range, format_report

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
NO_DB = "/nonexistent/predictions.duckdb"


@pytest.fixture
def events():
    """AAA reported in the last complete week and has an upcoming call next week; BBB
    reported last week only, with no verified timing; CCC is a dead feed: a pending row
    whose earnings_date has already gone by."""
    rows = [
        # stock, earnings_date, cutoff, pending, tier, pct, anchored r3, status, legacy r3
        ("AAA", "2026-06-03", "2026-05-29", False, "Normal",     50.0,  0.02, "available", 0.02),
        ("AAA", "2026-09-02", "2026-08-28", False, "High Alert", 93.0,  0.11, "available", 0.01),
        ("AAA", "2026-09-09", "2026-09-04", True,  "Elevated",   85.0,  np.nan, "pending", np.nan),
        ("BBB", "2026-09-03", "2026-08-28", False, "Unscored",  np.nan, np.nan,
         "unresolved_no_timestamp", -0.20),
        ("CCC", "2026-02-11", "2026-02-06", True,  "Normal",     40.0,  np.nan, "pending", np.nan),
    ]
    df = pd.DataFrame(rows, columns=["stock", "earnings_date", "score_asof_date", "is_pending",
                                     "risk_tier", "risk_score", "reaction_3d_anchored",
                                     "reaction_3d_anchored_status", "reaction_3d"])
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    df["score_asof_date"] = pd.to_datetime(df["score_asof_date"])
    df["sector"] = "Tech"
    df["model_b_hist_mean_abs"] = [0.03, 0.07, 0.06, np.nan, 0.02]
    df["model_b_n_prior"] = [12, 13, 14, 3, 20]
    df["abs_reaction_3d_anchored"] = df["reaction_3d_anchored"].abs()
    df["abs_reaction_3d"] = df["reaction_3d"].abs()          # legacy, must be ignored
    df["announce_window"] = ["AMC", "BMO", "UNKNOWN", "UNKNOWN", "UNKNOWN"]
    return df


def _range(events, **kw):
    return predictions_range(events_df=events, today=TODAY, db_path=NO_DB, **kw)


def test_weeks_back_returns_completed_events_only(events):
    out = _range(events, weeks_back=1)
    assert set(out["stock"]) == {"AAA", "BBB"}
    assert (out["row_kind"] == "history").all()
    assert out["earnings_date"].between("2026-08-31", "2026-09-04").all()


def test_weeks_forward_returns_pending_events_only(events):
    out = _range(events, weeks_forward=1)
    assert list(out["stock"]) == ["AAA"]
    assert (out["row_kind"] == "upcoming").all()
    assert out["days_to_earnings"].tolist() == [3.0]


def test_a_pending_row_whose_date_has_passed_is_dropped(events):
    for kw in (dict(weeks_back=40), dict(monday="2026-02-09")):
        assert "CCC" not in set(_range(events, **kw)["stock"])


def test_outcomes_use_the_corrected_target_and_leave_unresolved_blank(events):
    out = _range(events, weeks_back=1).set_index("stock")
    assert bool(out.at["AAA", "moved_5pct"]) and bool(out.at["AAA", "moved_8pct"])   # +11%
    # BBB's legacy reaction is -20% but it has no verified timing: unknown, not a hit
    assert out.at["BBB", "moved_8pct"] is None and out.at["BBB", "moved_5pct"] is None


def test_reports_model_b_fields_and_nothing_legacy(events):
    out = _range(events, monday="2026-08-31", weeks=2)
    for col in ("risk_tier", "risk_score", "model_b_hist_mean_abs", "model_b_n_prior",
                "score_asof_date"):
        assert col in out.columns
    banned = ("explosiveness", "lift", "high_conviction", "drift_flag", "surprise", "pre_audit")
    assert not [c for c in out.columns if any(b in c for b in banned)]
    assert "abs_reaction_3d" not in out.columns and "reaction_3d" not in out.columns
    assert (out["model_version"] == "0.4.0").all()


def test_published_calls_carry_their_model_version(events, tmp_path):
    import duckdb
    from utilities.db_utilities import create_predictions_table_if_not_exists
    db = str(tmp_path / "p.duckdb")
    con = duckdb.connect(db)
    create_predictions_table_if_not_exists(con)
    con.execute("""INSERT INTO predictions (prediction_asof_date, stock, earnings_date, tier,
                   risk_score, model_version, score_asof_date)
                   VALUES ('2026-08-29', 'AAA', '2026-09-02', 'High Alert', 81, '0.3.1',
                           '2026-08-28')""")
    con.close()
    out = predictions_range(weeks_back=1, events_df=events, today=TODAY, db_path=db)
    aaa = out.set_index("stock").loc["AAA"]
    assert aaa["published_tier"] == "High Alert" and aaa["published_model_version"] == "0.3.1"
    text = format_report(out, pd.Timestamp("2026-08-31"), pd.Timestamp("2026-09-04"))
    assert "(0.3.1)" in text


def test_a_window_spanning_today_carries_both_halves(events):
    out = _range(events, monday="2026-08-31", weeks=2)
    assert set(out["row_kind"]) == {"history", "upcoming"}
    assert out.sort_values("week_start")["week_start"].is_monotonic_increasing


def test_empty_window_returns_empty_and_formats(events):
    out = _range(events, monday="2026-07-06")
    assert out.empty
    text = format_report(out, pd.Timestamp("2026-07-06"), pd.Timestamp("2026-07-10"))
    assert "No events." in text


def test_report_renders_both_halves(events):
    out = _range(events, monday="2026-08-31", weeks=2)
    text = format_report(out, pd.Timestamp("2026-08-31"), pd.Timestamp("2026-09-11"))
    assert "Week of Aug 31, 2026" in text and "Week of Sep 7, 2026" in text
    assert "retro-scored Model B calls vs the corrected 3-session target" in text
    assert "1 of 2 events have no corrected outcome" in text
