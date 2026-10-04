"""
Event-frame tests — Phase 1 of the methodology rebuild.

Two things must hold at once:

  * the refactor changes NOTHING about completed historical events, and
  * an upcoming (pending) event is scored from state that includes the most recently
    completed earnings reaction, which it never was before.

Tests run on a synthetic fixture so they are hermetic. The ones that need real history
(28+ events per stock for the rolling p75) are additionally run against
output/full_df.parquet when it exists, and skipped when it does not.

Run with:  pytest testing/test_event_frame.py -v
"""
import ast
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from config import LIFT_PRIOR_STRENGTH
from pipeline.stage3 import stage3
from pipeline.stage4 import stage4
from pipeline.events import (
    build_event_frame, score_event_frame, build_and_score_event_frame,
    completed_parity_report, pending_events, OUTCOME_COLS, PARITY_COLS,
)
from testing.test_pipeline import _build_stage2_df

FULL_DF_PATH = "output/full_df.parquet"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def daily_df():
    """Synthetic daily frame with a FUTURE earnings date on the tail rows, so every
    stock is eligible for a pending event. The trailing e_index is past the last price
    row, which is what gives those rows a positive days_to_earnings — exactly the shape
    real data has between two reports."""
    n_days = 250
    return stage4(stage3(_build_stage2_df(
        n_days=n_days, e_indices=(60, 120, 180, 220) + (n_days + 20,))))


@pytest.fixture(scope="module")
def events_df(daily_df):
    return score_event_frame(build_event_frame(daily_df))


@pytest.fixture(scope="module")
def real_daily_df():
    if not os.path.exists(FULL_DF_PATH):
        pytest.skip(f"{FULL_DF_PATH} not present — run the pipeline first")
    return pd.read_parquet(FULL_DF_PATH)


@pytest.fixture(scope="module")
def real_events_df(real_daily_df):
    return score_event_frame(build_event_frame(real_daily_df))


# ── Invariants 1-4: completed history must not move ───────────────────────────

def test_1_completed_scores_unchanged(daily_df, events_df):
    """Invariant 1 — completed historical scores are unchanged by the refactor."""
    diffs = completed_parity_report(events_df, daily_df)
    assert "earnings_explosiveness_score" not in diffs, diffs
    assert "risk_score" not in diffs, diffs


def test_2_completed_structural_tiers_unchanged(daily_df, events_df):
    """Invariant 2 — completed structural tiers are unchanged."""
    diffs = completed_parity_report(events_df, daily_df)
    assert "earnings_explosiveness_bucket_structural" not in diffs, diffs


def test_3_completed_final_tiers_unchanged(daily_df, events_df):
    """Invariant 3 — completed final lift-adjusted tiers are unchanged."""
    diffs = completed_parity_report(events_df, daily_df)
    assert "earnings_explosiveness_bucket" not in diffs, diffs


def test_4_completed_lift_unchanged(daily_df, events_df):
    """Invariant 4 — completed stock_bucket_lift is unchanged."""
    diffs = completed_parity_report(events_df, daily_df)
    assert "stock_bucket_lift" not in diffs, diffs


def test_1to4_full_parity_every_column(daily_df, events_df):
    """The whole PARITY_COLS surface, not just the four headline columns."""
    assert completed_parity_report(events_df, daily_df) == {}


def test_1to4_full_parity_on_real_history(real_daily_df, real_events_df):
    """Same guarantee on the real 45k-event history, where the 28-event rolling
    windows, the lift shrinkage and the entropy ffill chain are all actually exercised."""
    assert completed_parity_report(real_events_df, real_daily_df) == {}


def test_parity_report_is_not_vacuous(daily_df, events_df):
    """Guard the four tests above: if PARITY_COLS were empty or the frames misaligned,
    parity would pass while proving nothing."""
    assert len(PARITY_COLS) >= 20
    completed = events_df[~events_df["is_pending"]]
    assert len(completed) == int((daily_df["is_earnings_day"] == 1).sum()) > 0
    assert completed["earnings_explosiveness_score"].notna().any()
    assert completed["stock_bucket_lift"].notna().any()


# ── Invariants 5-6: pending event shape ───────────────────────────────────────

def test_5_exactly_one_pending_per_eligible_stock(daily_df, events_df):
    """Invariant 5 — exactly one pending event per eligible upcoming stock."""
    last_rows = (daily_df.sort_values(["stock", "date"], kind="mergesort")
                 .groupby("stock", as_index=False, sort=False).tail(1))
    eligible = set(last_rows.loc[
        last_rows["earnings_date"].notna() & (last_rows["days_to_earnings"] > 0), "stock"])
    pend = events_df[events_df["is_pending"]]
    assert set(pend["stock"]) == eligible
    assert not pend["stock"].duplicated().any()
    assert len(eligible) > 0, "fixture produces no eligible stock — test would be vacuous"


def test_5_pending_on_real_data(real_daily_df, real_events_df):
    pend = real_events_df[real_events_df["is_pending"]]
    assert not pend["stock"].duplicated().any()
    assert len(pend) > 400, "expected a pending row for most of the S&P 500"


def test_6_pending_has_no_realized_outcome(events_df):
    """Invariant 6 — a pending event carries no outcome, so nothing downstream can
    read a neighbouring event's result as this one's and the lift's expanding
    aggregations cannot count it as a non-extreme observation."""
    pend = events_df[events_df["is_pending"]]
    for col in OUTCOME_COLS:
        if col in pend.columns:
            assert pend[col].isna().all(), f"{col} is populated on a pending event"


def test_6_pending_earnings_date_is_in_the_future(events_df):
    pend = events_df[events_df["is_pending"]]
    assert (pend["earnings_date"] > pend["date"]).all()
    assert (pend["is_earnings_day"] == 0).all()


def test_6_pending_never_enters_the_daily_frame(daily_df, events_df):
    """The daily price frame must be untouched: no future-dated row can reach the
    rolling price windows or the per-date cross-sectional ranks."""
    assert "is_pending" not in daily_df.columns
    assert daily_df["date"].max() == events_df.loc[events_df["is_pending"], "date"].max()


# ── Invariants 7-8: the staleness fix itself ──────────────────────────────────

def test_7_pending_p75_includes_latest_completed_reaction(real_events_df):
    """Invariant 7 — the pending p75 is the quantile of the stock's last 28 COMPLETED
    reactions, i.e. the window ends at the most recent report. Before Phase 1 the
    shipped value was the p75 computed AT that report, whose own shift(1) excluded it:
    the window was a full quarter behind."""
    completed = real_events_df[~real_events_df["is_pending"]]
    checked = 0
    for stock, pend in real_events_df[real_events_df["is_pending"]].groupby("stock"):
        g = completed[completed["stock"] == stock].sort_values("earnings_date")["abs_reaction_3d"]
        if len(g) < 28:
            continue
        tail = g.iloc[-28:]
        if tail.isna().any():        # min_periods=28 makes the production value NaN too
            continue
        got = pend["abs_reaction_p75_rolling"].iloc[0]
        assert np.isclose(got, tail.quantile(0.75)), (
            f"{stock}: pending p75 {got} != {tail.quantile(0.75)}")
        checked += 1
    assert checked > 300, f"only {checked} stocks had a full window — test too weak"


def test_7_pending_p75_differs_from_the_stale_value_somewhere(real_events_df, real_daily_df):
    """The fix must be observable: for many stocks the pending p75 differs from the
    value groupby().last() used to ship."""
    stale = real_daily_df.sort_values("date").groupby("stock").last()
    pend = real_events_df[real_events_df["is_pending"]].set_index("stock")
    common = pend.index.intersection(stale.index)
    d = (pend.loc[common, "abs_reaction_p75_rolling"]
         - stale.loc[common, "abs_reaction_p75_rolling"]).abs()
    assert (d > 1e-12).sum() > 50, "pending p75 never differs — the fix is not taking effect"


def test_7_dropping_the_latest_event_moves_the_pending_p75(real_daily_df):
    """Differential proof of inclusion, independent of any formula: delete each stock's
    most recent completed event from the daily frame, rebuild, and the pending p75 must
    move for stocks whose 28-event window actually changed."""
    ev_full = score_event_frame(build_event_frame(real_daily_df))
    last_event_dates = (real_daily_df[real_daily_df["is_earnings_day"] == 1]
                        .groupby("stock")["date"].max())
    drop = pd.MultiIndex.from_arrays(
        [last_event_dates.index, last_event_dates.values], names=["stock", "date"])
    idx = pd.MultiIndex.from_frame(real_daily_df[["stock", "date"]])
    trimmed = real_daily_df[~idx.isin(drop)].copy()
    ev_trim = score_event_frame(build_event_frame(trimmed))

    a = ev_full[ev_full["is_pending"]].set_index("stock")["abs_reaction_p75_rolling"]
    b = ev_trim[ev_trim["is_pending"]].set_index("stock")["abs_reaction_p75_rolling"]
    common = a.index.intersection(b.index)
    moved = (a.loc[common] - b.loc[common]).abs() > 1e-12
    assert moved.sum() > 50, (
        "removing the latest completed event changed no pending p75 — the pending row "
        "is not reading it")


def test_8_pending_lift_includes_latest_completed_event(real_events_df):
    """Invariant 8 — the pending lift counts the stock's prior events in the pending
    tier through the most recent completed one, shrunk toward the market baseline over
    all completed events.

    Restricted to pending rows whose as-of date is the latest price date. A handful of
    tickers have a price feed that stops early (see the build_event_frame warning);
    their baseline is correctly taken as-of their own, earlier, date and the closed form
    below does not apply to them.
    """
    ev = real_events_df
    completed = ev[~ev["is_pending"]]
    global_prior = completed["is_extreme_reaction"].mean()
    latest = ev["date"].max()

    checked = 0
    for _, row in ev[ev["is_pending"] & (ev["date"] == latest)].iterrows():
        tier = str(row["earnings_explosiveness_bucket_structural"])
        hist = completed[(completed["stock"] == row["stock"]) &
                         (completed["earnings_explosiveness_bucket_structural"].astype(str) == tier)]
        expected = ((hist["is_extreme_reaction"].sum() + LIFT_PRIOR_STRENGTH * global_prior)
                    / (len(hist) + LIFT_PRIOR_STRENGTH)) / global_prior
        assert np.isclose(row["stock_bucket_lift"], expected, rtol=1e-9), (
            f"{row['stock']}: lift {row['stock_bucket_lift']} != {expected} "
            f"(tier={tier}, n_prior={len(hist)})")
        checked += 1
    assert checked > 400, f"only {checked} pending rows checked — test too weak"


def test_8_pending_lift_differs_from_the_stale_value(real_events_df, real_daily_df):
    """The stale lift was the one computed at the previous event, against a different
    market baseline and one fewer prior event. It must differ essentially everywhere."""
    stale = real_daily_df.sort_values("date").groupby("stock").last()
    pend = real_events_df[real_events_df["is_pending"]].set_index("stock")
    common = pend.index.intersection(stale.index)
    d = (pend.loc[common, "stock_bucket_lift"] - stale.loc[common, "stock_bucket_lift"]).abs()
    assert (d > 1e-12).mean() > 0.9, "pending lift matches the stale lift — fix not applied"


def test_8_pending_rows_do_not_move_the_market_baseline(real_events_df, real_daily_df):
    """A pending row must contribute nothing to the lift's global expanding baseline —
    otherwise 500 phantom 'not extreme' observations would deflate every stock's lift.
    Proven by completed lifts being untouched."""
    assert completed_parity_report(real_events_df, real_daily_df).get("stock_bucket_lift") is None


# ── Invariant 9: no consumer reads the sparse columns via groupby().last() ─────

CONSUMER_MODULES = [
    "streamlit_dash/streamlit_export.py",
    "analysis/save_predictions.py",
    "report/report_builder.py",
    "report/calendar_builder.py",
    "pipeline/stage5.py",
]


@pytest.mark.parametrize("path", CONSUMER_MODULES)
def test_9_no_groupby_last_in_upcoming_consumers(path):
    """Invariant 9 — no upcoming-score consumer may recover state with
    `groupby(...).last()`. Its per-column NaN skipping silently reaches back to the
    stock's last COMPLETED event, which is the whole bug. AST-level so a comment
    mentioning the old idiom (several do) cannot trip it."""
    root = os.path.join(os.path.dirname(__file__), "..")
    tree = ast.parse(open(os.path.join(root, path)).read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "last"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Attribute)
                and node.func.value.func.attr == "groupby"):
            raise AssertionError(f"{path}:{node.lineno} uses groupby(...).last()")


# ── Invariant 10: one implementation for historical and pending ───────────────

def test_10_same_cores_serve_daily_and_event_paths():
    """Invariant 10 — the daily pipeline and the event frame must call the SAME
    event-level functions, so the two can never drift apart."""
    import feature_engineering.event_features as core
    import scoring.scoring_features as scoring
    import feature_engineering.pre_earnings_stock_features as pre
    import feature_engineering.post_earnings_stock_features as post
    import pipeline.events as events

    shared = [
        (core.event_reaction_std, post), (core.event_reaction_entropy, post),
        (core.event_directional_bias, post),
        (core.event_abs_reaction_median, pre), (core.event_abs_reaction_p75, pre),
        (core.event_abs_reaction_p75_rolling, pre), (core.event_abs_reaction_p90_rolling, pre),
        (core.event_surprise_features, pre), (core.event_pre_earnings_drift_z, pre),
        (core.event_explosiveness_score, scoring), (core.event_stock_bucket_lift_values, scoring),
        (core.event_lift_adjusted_bucket, scoring), (core.event_high_conviction, scoring),
    ]
    for fn, module in shared:
        assert getattr(module, fn.__name__, None) is fn, (
            f"{module.__name__} does not import the shared core {fn.__name__}")
        assert getattr(events, fn.__name__, None) is fn, (
            f"pipeline.events does not import the shared core {fn.__name__}")


def test_10_pending_score_matches_the_core_formula(events_df):
    """The pending score is produced by the same formula as every historical event —
    no separate upcoming code path exists to drift."""
    pend = events_df[events_df["is_pending"]].copy()
    p75 = pend["abs_reaction_p75_rolling"].fillna(pend["abs_reaction_p75"])
    e3 = (p75 / 0.12).clip(0, 1)
    e4 = np.clip(pend["reaction_entropy"].fillna(0), 0, 1)
    expected = 100 * np.clip(0.85 * e3 + 0.15 * e4, 0, 1)
    got = pend["earnings_explosiveness_score"]
    both = expected.notna() & got.notna()
    assert both.any()
    assert np.allclose(got[both], expected[both])


# ── Invariant 11: score_asof_date ─────────────────────────────────────────────

def test_11_score_asof_date_is_internally_consistent(events_df):
    """Invariant 11 — every row states the observation date its state was computed
    from. A completed event's is its own event date; a pending event's is the latest
    price date, strictly before the event it is predicting."""
    assert events_df["score_asof_date"].notna().all()

    completed = events_df[~events_df["is_pending"]]
    assert (completed["score_asof_date"] == completed["earnings_date"]).all()

    pend = events_df[events_df["is_pending"]]
    assert (pend["score_asof_date"] < pend["earnings_date"]).all()
    assert (pend["score_asof_date"] == pend["date"]).all()


def test_11_pending_asof_is_the_latest_price_date(daily_df, events_df):
    pend = events_df[events_df["is_pending"]]
    for stock, row in pend.set_index("stock").iterrows():
        assert row["score_asof_date"] == daily_df.loc[daily_df["stock"] == stock, "date"].max()


def test_11_score_asof_date_survives_the_upcoming_export(real_events_df, tmp_path):
    from streamlit_dash.streamlit_export import export_upcoming_df
    out = tmp_path / "upcoming.parquet"
    export_upcoming_df(real_events_df, str(out))
    got = pd.read_parquet(out)
    if len(got):
        assert "score_asof_date" in got.columns
        assert got["score_asof_date"].notna().all()


# ── pending_events() helper ───────────────────────────────────────────────────

def test_pending_events_returns_only_pending_rows(events_df):
    out = pending_events(events_df)
    assert out["is_pending"].all()
    assert len(out) == int(events_df["is_pending"].sum())
    assert out["earnings_date"].is_monotonic_increasing


def test_build_and_score_asserts_parity(daily_df):
    """The pipeline-level entry point must fail loudly if a completed event ever moves."""
    ev = build_and_score_event_frame(daily_df)
    assert not ev.empty


# ── Pending drift flag: the legacy 1-60 day eligibility rule ──────────────────
#
# On the daily frame `engineer_pre_earnings_drift_flag` only wrote a flag onto a
# pre-earnings row when `days_to_earnings.between(1, 60)`; further out the flag stayed
# blank. The event frame must reproduce that exactly on pending rows — the Phase 1
# staleness fix is about WHICH event a score describes, not about widening the window
# a flag is emitted in.

def test_pending_drift_flag_blank_beyond_60_days(real_events_df):
    """No pending event more than 60 days out may carry a drift flag."""
    pend = real_events_df[real_events_df["is_pending"]]
    far = pend[~pend["days_to_earnings"].between(1, 60)]
    assert len(far) > 0, "no pending event beyond 60 days — test is vacuous"
    assert (far["pre_earnings_drift_flag"].fillna("") == "").all(), (
        "pending events beyond the legacy 1-60 day window carry a drift flag: "
        f"{far.loc[far['pre_earnings_drift_flag'].fillna('') != '', 'stock'].tolist()}"
    )


def test_pending_drift_flag_still_fires_inside_60_days(real_events_df):
    """Non-vacuity guard for the test above: the window still produces flags."""
    pend = real_events_df[real_events_df["is_pending"]]
    near = pend[pend["days_to_earnings"].between(1, 60)]
    assert (near["pre_earnings_drift_flag"].fillna("") != "").any(), (
        "the 1-60 day gate blanked every pending drift flag"
    )


# ── Weekly calendar: pipeline path vs dashboard path ──────────────────────────
#
# stage5 hands build_calendar_data the event frame, so the weekly calendar must read
# PENDING rows (it used to select from completed earnings-day rows and rendered nothing).
# The dashboard's "Export calendar HTML" button calls it WITHOUT an event frame, passing
# its own frame for a window the user picked; that path must keep selecting from the
# frame it was given, as before.

def _calendar_frames():
    from report.calendar_builder import build_calendar_data  # noqa: F401  (import check)
    completed = pd.DataFrame({
        "stock":                        ["AAA", "BBB", "CCC", "DDD"],
        "date":                         pd.to_datetime(["2026-07-01"] * 4),
        "earnings_date":                pd.to_datetime(["2026-07-01", "2026-07-02",
                                                        "2026-07-03", "2026-07-06"]),
        "is_earnings_day":              [1, 1, 1, 1],
        "is_extreme_reaction":          [1, 0, 0, 1],
        "earnings_explosiveness_bucket": ["High Alert", "Normal", "Elevated", "Normal"],
        "earnings_explosiveness_score": [85.0, 40.0, 75.0, 30.0],
        "momentum_fragility_score":     [0.1, 0.5, 0.9, 0.3],
        "sector":                       ["Tech", "Tech", "Energy", "Energy"],
        "sub_sector":                   ["a", "b", "c", "d"],
        "pre_earnings_drift_flag":      ["", "", "", ""],
        "surprise_momentum_flag":       ["", "", "", ""],
    })
    pending = completed.iloc[[0, 2]].copy()
    pending["earnings_date"] = pd.to_datetime(["2026-10-07", "2026-10-08"])
    pending["is_pending"] = True
    events = pd.concat([completed.assign(is_pending=False), pending], ignore_index=True)
    return completed, events


def test_calendar_reads_pending_rows_when_given_the_event_frame():
    from report.calendar_builder import build_calendar_data
    completed, events = _calendar_frames()
    got, summary, _ = build_calendar_data(completed, reference_date="2026-10-05",
                                          window_days=7, events_df=events)
    assert sorted(e["stock"] for e in got) == ["AAA", "CCC"]
    assert summary["n_total"] == 2


def test_calendar_without_event_frame_selects_from_the_frame_passed():
    """The dashboard export path (streamlit_dash/app.py) — master's behaviour."""
    from report.calendar_builder import build_calendar_data
    completed, _ = _calendar_frames()
    got, summary, _ = build_calendar_data(completed, reference_date="2026-07-01",
                                          window_days=3)
    assert sorted(e["stock"] for e in got) == ["AAA", "BBB", "CCC"]
    assert summary["n_total"] == 3


# ── Active universe: only currently active stocks get an upcoming call ────────
#
# `stock_data.status` (maintained by ingestion/fetch_sp500_sectors.py) is the authority.
# A stock that left the index keeps every completed event, but its final daily row —
# which still carries a future earnings_date from the DB — must not become a pending
# event: that is how a delisted name received a PDF and an archived call scored from
# prices two months old.

def _future_pending(events, when):
    """Copy of `events` with every pending row moved to `when`, so consumers that look
    forward from today see them. Moves rows; never adds any."""
    out = events.copy()
    out.loc[out["is_pending"], "earnings_date"] = pd.Timestamp(when)
    return out


@pytest.fixture(scope="module")
def events_active_aaa(daily_df):
    """BBB is inactive: only AAA may receive a pending row."""
    return build_and_score_event_frame(daily_df, active_stocks={"AAA"})


def test_active_stock_with_future_date_gets_one_pending_row(daily_df):
    ev = score_event_frame(build_event_frame(daily_df, active_stocks={"AAA", "BBB"}))
    pend = ev[ev["is_pending"]]
    assert sorted(pend["stock"]) == ["AAA", "BBB"]
    assert pend.groupby("stock").size().eq(1).all()


def test_inactive_stock_with_future_date_gets_no_pending_row(daily_df, events_active_aaa):
    last = daily_df.sort_values("date").groupby("stock").tail(1).set_index("stock")
    assert last.at["BBB", "days_to_earnings"] > 0, "BBB has no future date — test is vacuous"
    pend = events_active_aaa[events_active_aaa["is_pending"]]
    assert pend["stock"].tolist() == ["AAA"]


def test_inactive_stock_keeps_its_completed_history(events_df, events_active_aaa):
    def completed(ev, s):
        return ev[(~ev["is_pending"]) & (ev["stock"] == s)]
    assert len(completed(events_active_aaa, "BBB")) > 0
    assert len(completed(events_active_aaa, "BBB")) == len(completed(events_df, "BBB"))


def test_universe_filter_leaves_completed_events_identical(daily_df, events_df, events_active_aaa):
    assert completed_parity_report(events_active_aaa, daily_df) == {}
    a = events_df[~events_df["is_pending"]].reset_index(drop=True)
    b = events_active_aaa[~events_active_aaa["is_pending"]].reset_index(drop=True)
    cols = [c for c in PARITY_COLS if c in a.columns]
    pd.testing.assert_frame_equal(a[["stock", "earnings_date", *cols]],
                                  b[["stock", "earnings_date", *cols]])


def test_downstream_consumers_cannot_resurrect_an_inactive_stock(events_active_aaa, tmp_path):
    from streamlit_dash.streamlit_export import export_upcoming_df
    from report.calendar_builder import build_calendar_data
    soon = pd.Timestamp.today().normalize() + pd.Timedelta(days=3)
    ev = _future_pending(events_active_aaa, soon)

    assert pending_events(ev)["stock"].tolist() == ["AAA"]

    out = tmp_path / "upcoming.parquet"
    export_upcoming_df(ev, str(out))
    assert pd.read_parquet(out)["stock"].tolist() == ["AAA"]

    completed = ev[~ev["is_pending"]].copy()
    completed["momentum_fragility_score"] = completed.get("momentum_fragility_score", 0.0)
    got, _, _ = build_calendar_data(completed, reference_date=soon - pd.Timedelta(days=1),
                                    window_days=7, events_df=ev)
    assert [e["stock"] for e in got] == ["AAA"]


def test_prediction_snapshot_cannot_archive_an_inactive_call(events_active_aaa, tmp_path,
                                                             monkeypatch):
    import duckdb
    import analysis.save_predictions as sp
    from utilities.data_utilities import work_week_window
    db = tmp_path / "predictions.duckdb"
    monkeypatch.setattr(sp, "PREDICTIONS_DB_PATH", str(db))
    window_start, _ = work_week_window(weeks=1)
    ev = _future_pending(events_active_aaa, window_start + pd.Timedelta(days=1))

    sp.save_predictions_snapshot(ev)

    con = duckdb.connect(str(db), read_only=True)
    stocks = [r[0] for r in con.execute("SELECT stock FROM predictions").fetchall()]
    con.close()
    assert stocks == ["AAA"], stocks       # non-vacuous: the active call IS archived


def test_load_active_stocks_reads_stock_data_status():
    import duckdb
    from utilities.db_utilities import create_sectors_data_table_if_not_exists, load_active_stocks
    con = duckdb.connect(":memory:")
    create_sectors_data_table_if_not_exists(con)
    con.execute("""INSERT INTO stock_data (stock, status, reason) VALUES
        ('AAA', 'active', NULL),
        ('BK',  'inactive', 'renamed → BNY'),
        ('BNY', 'active', NULL),
        ('LW',  'inactive', 'removed from index — delisted/merged/unconfirmed')""")
    assert load_active_stocks(con) == {"AAA", "BNY"}


@pytest.mark.parametrize("path", ["pipeline/stage5.py"])
def test_production_supplies_the_active_universe(path):
    """`active_stocks=None` means "no filter" for synthetic tests; every production call
    site must pass the DB's active set explicitly."""
    tree = ast.parse(open(path).read())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "build_and_score_event_frame"]
    assert calls, f"{path} no longer builds the event frame"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "active_stocks" in kw, f"{path} builds the event frame without active_stocks"
        assert getattr(kw["active_stocks"].func, "id", None) == "load_pipeline_active_stocks"


def test_real_data_pending_events_are_exactly_the_active_stocks(real_daily_df):
    from config import DB_PATH
    import duckdb
    from utilities.db_utilities import load_active_stocks
    if not os.path.exists(DB_PATH):
        pytest.skip(f"{DB_PATH} not present")
    con = duckdb.connect(DB_PATH, read_only=True)
    active = load_active_stocks(con)
    con.close()
    unfiltered = build_event_frame(real_daily_df)
    filtered = build_event_frame(real_daily_df, active_stocks=active)
    before = set(unfiltered.loc[unfiltered["is_pending"], "stock"])
    after = set(filtered.loc[filtered["is_pending"], "stock"])
    assert after == before & active                 # no active stock lost
    assert not (after - active)                     # no inactive stock kept
    assert (~filtered["is_pending"]).sum() == (~unfiltered["is_pending"]).sum()
