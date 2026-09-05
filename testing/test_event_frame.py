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

GOLDEN_UPCOMING_PATH = "audit/phase1_golden/upcoming_df.parquet"


@pytest.fixture(scope="module")
def golden_upcoming():
    if not os.path.exists(GOLDEN_UPCOMING_PATH):
        pytest.skip(f"{GOLDEN_UPCOMING_PATH} not present")
    return pd.read_parquet(GOLDEN_UPCOMING_PATH).set_index("stock")


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


def test_pending_drift_flag_matches_legacy_golden(real_events_df, golden_upcoming):
    """Regression against the pre-refactor baseline: every pending drift flag must be
    the value the legacy daily path shipped for that stock. `pre_earnings_drift_flag`
    was never NaN on the daily frame, so the old `groupby('stock').last()` export read
    it from the true final row — it is the one upcoming field that was NOT stale, and
    Phase 1 must therefore leave it exactly as it was."""
    pend = real_events_df[real_events_df["is_pending"]].set_index("stock")
    common = pend.index.intersection(golden_upcoming.index)
    assert len(common) > 400, f"only {len(common)} stocks shared with the golden baseline"
    got = pend.loc[common, "pre_earnings_drift_flag"].fillna("")
    want = golden_upcoming.loc[common, "pre_earnings_drift_flag"].fillna("")
    mismatched = common[got.values != want.values]
    assert len(mismatched) == 0, (
        "pending drift flag differs from legacy for "
        f"{len(mismatched)} stocks: "
        + ", ".join(
            f"{s}({want.loc[s]!r}->{got.loc[s]!r}, dte={pend.at[s, 'days_to_earnings']:.0f})"
            for s in mismatched[:10]
        )
    )


def test_pending_high_conviction_matches_legacy_golden(real_events_df, golden_upcoming):
    """`is_high_conviction` = High Alert AND a drift flag. The tier half is corrected by
    Phase 1, the flag half must not be — so any HC change has to trace to a tier change
    on a stock that already carried a flag, never to a flag appearing out of window."""
    pend = real_events_df[real_events_df["is_pending"]].set_index("stock")
    common = pend.index.intersection(golden_upcoming.index)
    got = pend.loc[common, "is_high_conviction"].fillna(False).astype(bool)
    want = golden_upcoming.loc[common, "is_high_conviction"].fillna(False).astype(bool)
    changed = common[got.values != want.values]
    # Every HC change must be explained by the corrected tier, with the drift flag
    # identical to legacy on both sides.
    for stock in changed:
        assert (
            pend.at[stock, "pre_earnings_drift_flag"]
            == golden_upcoming.at[stock, "pre_earnings_drift_flag"]
        ), f"{stock}: high conviction moved because its drift flag moved"
        assert (
            pend.at[stock, "earnings_explosiveness_bucket"]
            != golden_upcoming.at[stock, "earnings_explosiveness_bucket"]
        ), f"{stock}: high conviction moved with neither tier nor flag changing"
