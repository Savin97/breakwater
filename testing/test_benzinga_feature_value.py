"""Tests for the Benzinga expectation-feature experiment.

The experiment's conclusion is only worth as much as its bookkeeping, so these tests are
about the five ways it could quietly cheat: a duplicated join inflating a stock, a missing
outcome becoming a "did not move" training example, a model trained on an outcome that had
not finished happening, a threshold or transform fitted on the test year, and two feature
groups scored on different populations.
"""
import numpy as np
import pandas as pd
import pytest

from feature_engineering.announcement_timing import TARGET_AVAILABLE
from testing.benzinga_feature_value import (
    ALERT_QUANTILE,
    AVAILABILITY_SUFFIX,
    BZ_BINARY,
    BZ_DEVIATION,
    BZ_GAP,
    BZ_LEVEL,
    GROUP_A,
    GROUP_NAMES,
    PRIMARY_STATUS,
    PRIMARY_TARGET,
    PRIMARY_THRESHOLD,
    SECONDARY_THRESHOLD,
    build_dataset,
    fit_predict,
    group_features,
    load_benzinga,
    nyse_sessions,
    outcome_ready_date,
    quarter_block_bootstrap,
    training_mask,
    walk_forward,
    within_stock_macro_auc,
)


# ──────────────────────────────────── fixtures ───────────────────────────────────────────
def _analysis(rows: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    d["earnings_date"] = pd.to_datetime(d["earnings_date"])
    d["event_clock"] = d["earnings_date"]
    d["anchor_date"] = d.get("anchor_date", d["earnings_date"])
    d["anchor_date"] = pd.to_datetime(d["anchor_date"])
    defaults = {"is_pending": False, PRIMARY_STATUS: TARGET_AVAILABLE, "stock": "AAA"}
    for col, value in defaults.items():
        if col not in d.columns:
            d[col] = value
        else:
            d[col] = d[col].fillna(value)
    for col in GROUP_A:
        if col not in d.columns:
            d[col] = 0.0
    return d


def _benzinga(rows: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    d["stock"] = d["ticker"].astype(str).str.upper()
    d["earnings_date"] = pd.to_datetime(d["date"])
    for col in [*BZ_LEVEL, *BZ_GAP, *BZ_DEVIATION]:
        if col not in d.columns:
            d[col] = np.nan
    for col in ("provider", "event_key", "fiscal_year", "fiscal_period", "last_updated",
                "retrieved_at_utc", "historical_estimate_vintage_verified",
                "eps_prior_check", "revenue_prior_check", "eps_regime_n",
                "revenue_regime_n"):
        if col not in d.columns:
            d[col] = "x"
    return d


# ──────────────────────────── duplicate-join rejection ───────────────────────────────────
def test_a_duplicated_vendor_row_is_rejected_not_silently_fanned_out():
    """A 1:many join would clone an event, double-count its outcome and let one company
    dominate a fold. It must stop the run, not be deduplicated by guesswork."""
    a = _analysis([{"earnings_date": "2018-05-01", PRIMARY_TARGET: 0.10}])
    bz = _benzinga([{"ticker": "AAA", "date": "2018-05-01"},
                    {"ticker": "AAA", "date": "2018-05-01"}])
    with pytest.raises(ValueError, match="not unique"):
        build_dataset(a, bz)


def test_a_duplicated_event_row_is_rejected():
    a = _analysis([{"earnings_date": "2018-05-01", PRIMARY_TARGET: 0.10},
                   {"earnings_date": "2018-05-01", PRIMARY_TARGET: 0.03}])
    bz = _benzinga([{"ticker": "AAA", "date": "2018-05-01"}])
    with pytest.raises(ValueError, match="not unique"):
        build_dataset(a, bz)


def test_the_join_is_one_to_one_and_losses_are_reported_by_year():
    a = _analysis([
        {"earnings_date": "2018-02-01", PRIMARY_TARGET: 0.10},
        {"earnings_date": "2018-05-01", PRIMARY_TARGET: 0.02},
        {"earnings_date": "2019-02-01", PRIMARY_TARGET: 0.09},
    ])
    bz = _benzinga([{"ticker": "AAA", "date": "2018-02-01"},
                    {"ticker": "AAA", "date": "2019-02-01"}])
    d, rep = build_dataset(a, bz)
    assert len(d) == 2                                   # the unmatched 2018-05 row is gone
    by_year = {r["year"]: r for r in rep.join_losses_by_year}
    assert by_year[2018]["events"] == 2 and by_year[2018]["unmatched"] == 1
    assert by_year[2019]["unmatched"] == 0
    assert d["event_id"].is_unique


# ──────────────────────────── missing-outcome handling ───────────────────────────────────
def test_an_unavailable_outcome_never_becomes_a_non_extreme_label():
    """The failure this guards against is silent and fatal: an unmeasurable reaction read
    as `y=0` teaches the model that those events are safe."""
    a = _analysis([
        {"earnings_date": "2018-02-01", PRIMARY_TARGET: 0.10},
        {"earnings_date": "2018-05-01", PRIMARY_TARGET: np.nan,
         PRIMARY_STATUS: "unresolved_no_timestamp"},
        {"earnings_date": "2018-08-01", PRIMARY_TARGET: 0.02,
         PRIMARY_STATUS: "unavailable_endpoint_price_gap"},
    ])
    bz = _benzinga([{"ticker": "AAA", "date": f"2018-{m:02d}-01"} for m in (2, 5, 8)])
    d, rep = build_dataset(a, bz)
    assert len(d) == 1 and d["y_primary"].iloc[0] == 1
    assert rep.exclusions["pending_or_outcome_unavailable"] == 2
    assert d["y_primary"].isin([0, 1]).all()


def test_labels_use_the_production_thresholds_on_the_corrected_horizon():
    a = _analysis([{"earnings_date": "2018-02-01", PRIMARY_TARGET: PRIMARY_THRESHOLD},
                   {"earnings_date": "2018-05-01", PRIMARY_TARGET: SECONDARY_THRESHOLD},
                   {"earnings_date": "2018-08-01", PRIMARY_TARGET: 0.0001}])
    bz = _benzinga([{"ticker": "AAA", "date": f"2018-{m:02d}-01"} for m in (2, 5, 8)])
    d, _ = build_dataset(a, bz)
    assert d["y_primary"].tolist() == [1, 0, 0]          # >= is inclusive at the threshold
    assert d["y_secondary"].tolist() == [1, 1, 0]


def test_a_pending_event_is_excluded():
    a = _analysis([{"earnings_date": "2018-02-01", PRIMARY_TARGET: 0.10, "is_pending": True}])
    bz = _benzinga([{"ticker": "AAA", "date": "2018-02-01"}])
    d, _ = build_dataset(a, bz)
    assert len(d) == 0


# ───────────────────────── training-cutoff / look-ahead enforcement ──────────────────────
def test_the_outcome_ready_date_is_three_sessions_after_the_anchor():
    sessions = nyse_sessions("2018-01-01", "2018-12-31")
    # 2018-12-26 Wed, 27 Thu, 28 Fri, 31 Mon are sessions; +3 from the 26th is the 31st.
    got = outcome_ready_date(pd.Series([pd.Timestamp("2018-12-26")]), sessions)
    assert got.iloc[0] == pd.Timestamp("2018-12-31")


def test_an_event_whose_window_closes_after_the_cutoff_cannot_be_trained_on():
    """Announced in December, resolved in January: available for TESTING later, never a
    training row for a model fitted on 1 January."""
    a = _analysis([
        {"earnings_date": "2018-06-01", "anchor_date": "2018-06-01", PRIMARY_TARGET: 0.10},
        {"earnings_date": "2018-12-28", "anchor_date": "2018-12-28", PRIMARY_TARGET: 0.01},
    ])
    bz = _benzinga([{"ticker": "AAA", "date": "2018-06-01"},
                    {"ticker": "AAA", "date": "2018-12-28"}])
    d, _ = build_dataset(a, bz)
    cutoff = pd.Timestamp("2019-01-01")
    mask = training_mask(d, cutoff)
    assert mask.tolist() == [True, False]
    assert (d.loc[mask, "outcome_ready_date"] < cutoff).all()


def test_an_event_with_no_placeable_anchor_is_never_a_training_row():
    a = _analysis([{"earnings_date": "2018-06-01", "anchor_date": pd.NaT,
                    PRIMARY_TARGET: 0.10}])
    bz = _benzinga([{"ticker": "AAA", "date": "2018-06-01"}])
    d, _ = build_dataset(a, bz)
    assert not d["outcome_ready_known"].iloc[0]
    assert not training_mask(d, pd.Timestamp("2019-01-01")).any()


def _walk_frame(n_years: int = 10, stocks: int = 40, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(stocks):
        for y in range(2014, 2014 + n_years):
            for q, month in enumerate((2, 5, 8, 11)):
                rows.append({
                    "event_id": f"S{s:02d}|{y}-{month:02d}",
                    "stock": f"S{s:02d}",
                    "event_clock": pd.Timestamp(year=y, month=month, day=15),
                    "quarter": f"{y}Q{q + 1}",
                    "outcome_ready_known": True,
                    "outcome_ready_date": pd.Timestamp(year=y, month=month, day=20),
                    "y_primary": int(rng.random() < 0.2),
                    "f1": rng.normal(), "f2": rng.normal(),
                })
    return pd.DataFrame(rows)


def test_walk_forward_never_trains_on_an_outcome_from_the_test_year_or_later(monkeypatch):
    seen = {}
    import testing.benzinga_feature_value as mod
    real = mod.fit_predict

    def spy(train, test, features, label):
        seen[int(test["event_clock"].dt.year.iloc[0])] = train["outcome_ready_date"].max()
        return real(train, test, features, label)

    monkeypatch.setattr(mod, "fit_predict", spy)
    pred = mod.walk_forward(_walk_frame(), ["f1", "f2"], "A", "y_primary")
    for year, max_ready in seen.items():
        assert max_ready < pd.Timestamp(year=year, month=1, day=1)
    for year, g in pred.groupby("test_year"):
        assert (g["event_id"].str.startswith(tuple(f"S{i:02d}" for i in range(40)))).all()
        assert g["event_id"].is_unique
    assert pred.groupby("test_year")["train_n"].first().is_monotonic_increasing


def test_no_event_is_both_trained_on_and_tested_in_the_same_fold():
    d = _walk_frame()
    for year in range(2018, 2024):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train_ids = set(d.loc[training_mask(d, cutoff), "event_id"])
        test_ids = set(d.loc[d["event_clock"].dt.year.eq(year), "event_id"])
        assert not (train_ids & test_ids)


# ────────────────────────── train-only preprocessing / thresholds ────────────────────────
def test_the_alert_threshold_comes_from_training_predictions_only():
    d = _walk_frame()
    train = d[d["event_clock"] < pd.Timestamp("2018-01-01")]
    test = d[d["event_clock"].dt.year.eq(2018)]
    p_test, p_train, alert = fit_predict(train, test, ["f1", "f2"], "y_primary")
    assert np.isclose(alert, np.quantile(p_train, ALERT_QUANTILE))
    # Perturbing the TEST rows must not move the threshold at all.
    shifted = test.copy()
    shifted["f1"] = shifted["f1"] + 100.0
    _p2, _pt2, alert2 = fit_predict(train, shifted, ["f1", "f2"], "y_primary")
    assert alert2 == alert


def test_test_rows_cannot_influence_the_fitted_model():
    d = _walk_frame()
    train = d[d["event_clock"] < pd.Timestamp("2018-01-01")]
    test = d[d["event_clock"].dt.year.eq(2018)]
    p_a, _t, _thr = fit_predict(train, test, ["f1", "f2"], "y_primary")
    bigger = pd.concat([test, test.assign(f1=test["f1"] * 50)], ignore_index=True)
    p_b, _t2, _thr2 = fit_predict(train, bigger, ["f1", "f2"], "y_primary")
    assert np.allclose(p_a, p_b[:len(p_a)])      # same model, row-by-row scoring


def test_missing_binary_flags_keep_an_availability_indicator_and_are_never_filled():
    a = _analysis([{"earnings_date": "2018-02-01", PRIMARY_TARGET: 0.10},
                   {"earnings_date": "2018-05-01", PRIMARY_TARGET: 0.02}])
    bz = _benzinga([{"ticker": "AAA", "date": "2018-02-01", "expected_loss_to_profit": 1.0},
                    {"ticker": "AAA", "date": "2018-05-01"}])
    d, _ = build_dataset(a, bz)
    assert d["expected_loss_to_profit"].isna().iloc[1]        # still missing, not False
    assert d["expected_loss_to_profit" + AVAILABILITY_SUFFIX].tolist() == [1.0, 0.0]
    for col in BZ_BINARY:
        assert col + AVAILABILITY_SUFFIX in group_features("D")


def test_group_a_never_contains_a_benzinga_column():
    bz_cols = set(BZ_LEVEL) | set(BZ_GAP) | set(BZ_DEVIATION)
    assert not (set(group_features("A")) & bz_cols)
    # Each step adds its own block plus the availability indicators for any binary flag in
    # that block — the indicators are part of the feature, not an extra one.
    def added(later, earlier):
        return {c for c in set(group_features(later)) - set(group_features(earlier))
                if not c.endswith(AVAILABILITY_SUFFIX)}
    assert added("B", "A") == set(BZ_LEVEL)
    assert added("C", "B") == set(BZ_GAP)
    assert added("D", "C") == set(BZ_DEVIATION)
    for flag in BZ_BINARY:
        assert flag + AVAILABILITY_SUFFIX in group_features("D")
    for g in GROUP_NAMES:
        assert len(group_features(g)) == len(set(group_features(g))), f"{g} has duplicates"


def test_the_leaky_and_non_point_in_time_columns_are_absent_from_every_group():
    """Phase 5 established these four are unusable; the experiment must not readmit them."""
    banned = {"surprise_percentage", "reported_eps", "daily_ret",
              "momentum_fragility_score", "earnings_explosiveness_z", "earnings_tail_z",
              PRIMARY_TARGET, "abs_reaction_3d", "reaction_3d_anchored"}
    for g in GROUP_NAMES:
        assert not (set(group_features(g)) & banned), g


# ───────────────────────────── identical comparison ids ──────────────────────────────────
def test_every_group_is_scored_on_exactly_the_same_held_out_events():
    d = _walk_frame()
    d["f3"] = np.random.default_rng(1).normal(size=len(d))
    a = walk_forward(d, ["f1", "f2"], "A", "y_primary")
    b = walk_forward(d, ["f1", "f2", "f3"], "B", "y_primary")
    assert a["event_id"].tolist() == b["event_id"].tolist()
    assert a["y"].tolist() == b["y"].tolist()
    delta = quarter_block_bootstrap(a, b, reps=20)
    assert delta["n"] == len(a)


def test_the_paired_bootstrap_refuses_a_mismatched_pairing():
    d = _walk_frame()
    a = walk_forward(d, ["f1", "f2"], "A", "y_primary")
    b = a.copy()
    b = pd.concat([b, b.head(1)], ignore_index=True)      # a duplicated event id
    with pytest.raises(Exception):
        quarter_block_bootstrap(a, b, reps=5)


def test_the_bootstrap_resamples_quarters_and_is_deterministic():
    d = _walk_frame()
    d["f3"] = np.random.default_rng(2).normal(size=len(d))
    a = walk_forward(d, ["f1", "f2"], "A", "y_primary")
    b = walk_forward(d, ["f1", "f2", "f3"], "B", "y_primary")
    one = quarter_block_bootstrap(a, b, reps=30, seed=7)
    two = quarter_block_bootstrap(a, b, reps=30, seed=7)
    assert one == two
    assert one["blocks"] == a["quarter"].nunique()
    assert one["delta_roc_auc_ci_lo"] <= one["delta_roc_auc"] <= one["delta_roc_auc_ci_hi"] \
        or np.isnan(one["delta_roc_auc"])


# ───────────────────────────── within-stock discrimination ───────────────────────────────
def test_within_stock_auc_skips_stocks_without_both_classes_and_counts_them():
    pred = pd.DataFrame({
        "stock": ["A", "A", "B", "B", "C", "C"],
        "y": [0, 1, 0, 0, 1, 0],
        "p": [0.1, 0.9, 0.2, 0.3, 0.8, 0.1],
    })
    out = within_stock_macro_auc(pred)
    assert out["within_stock_eligible_stocks"] == 2       # B has one class only
    assert out["within_stock_total_stocks"] == 3
    assert out["within_stock_eligible_events"] == 4
    assert np.isclose(out["within_stock_macro_auc"], 1.0)


def test_within_stock_auc_is_a_macro_average_not_pooled():
    """One stock ranked perfectly, one ranked backwards, different sizes: the macro average
    must be 0.5 regardless of how many events each contributes."""
    pred = pd.DataFrame({
        "stock": ["BIG"] * 6 + ["SMALL"] * 2,
        "y": [0, 0, 0, 1, 1, 1] + [1, 0],
        "p": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9] + [0.1, 0.9],
    })
    assert np.isclose(within_stock_macro_auc(pred)["within_stock_macro_auc"], 0.5)
