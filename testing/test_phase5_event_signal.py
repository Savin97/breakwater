"""Tests for Phase 5 — the event-specific signal audit (`research/phase5_event_signal.py`).

Phase 5 exists to tell a real event-specific signal apart from three things that imitate
one: a proxy for the stock phenotype, an aggregation artefact, and look-ahead. So the
tests are mostly about the ways the analysis could fool itself — a scaler fitted on the
test year, an unresolved outcome silently counted as "did not move 8%", a feature whose
whole apparent signal is between-stock — rather than about arithmetic.

Two synthetic fixtures carry most of the weight: a feature with ONLY between-stock signal,
which must look strong raw and null once the phenotype is controlled for, and a feature
with a genuine within-stock effect, which must survive both.
"""
import numpy as np
import pandas as pd
import pytest

from feature_engineering.announcement_timing import TARGET_AVAILABLE
from research.phase3_target_rebuild import CORRECTED_TARGET, PHASE3_PREFIX
from research.phase4_baselines import build_causal_baselines, prepare_analysis_frame
from research.phase5_event_signal import (
    CANDIDATES,
    CANDIDATES_BY_NAME,
    HEADLINE_FEATURES,
    STRUCTURAL,
    apply_block,
    derive_phase5_features,
    feature_audit,
    fit_block,
    paired_bootstrap_delta,
    pooled_pair_auc,
    residual_frame,
    residual_signal,
    select_survivors,
    structural_decile_auc,
    univariate_signal,
    walk_forward_predictions,
    within_stock_validation,
)

EXTREME = 0.08


# ────────────────────────────────────── fixtures ─────────────────────────────────────────
def _frame(rows: list[dict]) -> pd.DataFrame:
    """A minimal Phase-3-shaped event frame."""
    d = pd.DataFrame(rows)
    defaults = {
        "is_pending": False,
        "reaction_3d_anchored_status": TARGET_AVAILABLE,
        "sector": "Tech",
        "drift_30d": 0.0, "drift_60d": 0.0, "mom_5d": 0.0, "mom_20d": 0.0,
        "vol_10d": 0.02, "vol_30d": 0.02, "vol_ratio_10_to_30": 1.0,
        PHASE3_PREFIX + "risk_score": 50.0,
        PHASE3_PREFIX + "abs_reaction_p75": 0.06,
        PHASE3_PREFIX + "abs_reaction_p75_rolling": 0.06,
        PHASE3_PREFIX + "reaction_entropy": 0.5,
        PHASE3_PREFIX + "stock_bucket_lift": 1.0,
        PHASE3_PREFIX + "bucket_structural": "Normal",
        PHASE3_PREFIX + "bucket": "Normal",
        PHASE3_PREFIX + "n_prior_resolved_reactions": 0,
    }
    for col, value in defaults.items():
        if col not in d.columns:
            d[col] = value
        else:
            d[col] = d[col].fillna(value)
    d["earnings_date"] = pd.to_datetime(d["earnings_date"])
    d["phase3_announce_date"] = d.get("phase3_announce_date", d["earnings_date"])
    d["phase3_announce_date"] = pd.to_datetime(d["phase3_announce_date"])
    return d


def _panel(n_stocks: int, n_events: int, *, seed: int, kind: str) -> pd.DataFrame:
    """A panel where the event-level driver of |move| is controlled exactly.

    `kind="between"`: the candidate is a per-STOCK constant — a NOISY proxy for the stock's
    true scale. Constant within a stock, so it can carry no within-stock information, and
    noisier than the realized history, so it should add nothing to the structural baseline.
    `kind="between_exact"`: the candidate is the stock's EXACT scale parameter. Also
    constant within a stock, but a cleaner measure of the phenotype than the noisy expanding
    mean — which is precisely the case where controlling for the structural estimate
    under-controls. Used to pin that caveat, not as a success case.
    `kind="within"`: every stock has the same average risk, and the feature varies event to
    event and is the only thing that moves the outcome.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stocks):
        phenotype = rng.uniform(0.02, 0.10)
        proxy = float(phenotype * np.exp(rng.normal(0.0, 0.55)))
        for e in range(n_events):
            date = pd.Timestamp("2012-01-15") + pd.Timedelta(days=91 * e + s)
            if kind == "between":
                feature = proxy
                scale = phenotype
            elif kind == "between_exact":
                feature = phenotype
                scale = phenotype
            else:
                feature = float(rng.uniform(0.0, 1.0))
                scale = 0.055 * (0.4 + 1.2 * feature)
            move = float(abs(rng.normal(0.0, scale)))
            rows.append({"stock": f"S{s:03d}", "earnings_date": date,
                         CORRECTED_TARGET: move, "candidate": feature,
                         "sector": f"SEC{s % 4}"})
    return _frame(rows)


def _analysis(panel: pd.DataFrame) -> pd.DataFrame:
    d = derive_phase5_features(prepare_analysis_frame(panel))
    return d.dropna(subset=["y_extreme", STRUCTURAL])


# ───────────────────────────── point-in-time / shift behaviour ───────────────────────────
def test_the_z_own_transform_uses_only_the_stocks_prior_events():
    d = _frame([
        {"stock": "AAA", "earnings_date": "2020-01-01", CORRECTED_TARGET: 0.05, "vol_30d": 0.01},
        {"stock": "AAA", "earnings_date": "2020-04-01", CORRECTED_TARGET: 0.05, "vol_30d": 0.02},
        {"stock": "AAA", "earnings_date": "2020-07-01", CORRECTED_TARGET: 0.05, "vol_30d": 0.03},
        {"stock": "AAA", "earnings_date": "2020-10-01", CORRECTED_TARGET: 0.05, "vol_30d": 0.04},
        {"stock": "AAA", "earnings_date": "2021-01-01", CORRECTED_TARGET: 0.05, "vol_30d": 0.05},
        {"stock": "AAA", "earnings_date": "2021-04-01", CORRECTED_TARGET: 0.05, "vol_30d": 0.06},
    ])
    out = derive_phase5_features(prepare_analysis_frame(d))
    # min_periods=5 on the prior std: the first five events cannot have a z-score at all.
    assert out["vol_30d_z_own"].iloc[:5].isna().all()
    prior = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    expected = (0.06 - prior.mean()) / prior.std(ddof=1)
    assert np.isclose(out["vol_30d_z_own"].iloc[5], expected)


def test_the_z_own_transform_never_sees_the_current_event():
    """Doubling only the LAST event's vol must leave every earlier z-score untouched."""
    rows = [{"stock": "AAA", "earnings_date": f"20{20 + i // 4}-{1 + 3 * (i % 4):02d}-01",
             CORRECTED_TARGET: 0.05, "vol_30d": 0.02 + 0.001 * i} for i in range(10)]
    base = derive_phase5_features(prepare_analysis_frame(_frame(rows)))
    bumped_rows = [dict(r) for r in rows]
    bumped_rows[-1]["vol_30d"] = 99.0
    bumped = derive_phase5_features(prepare_analysis_frame(_frame(bumped_rows)))
    pd.testing.assert_series_equal(
        base["vol_30d_z_own"].iloc[:-1], bumped["vol_30d_z_own"].iloc[:-1])


def test_unresolved_outcomes_are_excluded_not_treated_as_non_extreme():
    d = _frame([
        {"stock": "AAA", "earnings_date": "2020-01-01", CORRECTED_TARGET: 0.20},
        {"stock": "AAA", "earnings_date": "2020-04-01", CORRECTED_TARGET: np.nan,
         "reaction_3d_anchored_status": "unresolved_no_timestamp"},
        {"stock": "AAA", "earnings_date": "2020-07-01", CORRECTED_TARGET: 0.01},
    ])
    out = prepare_analysis_frame(d)
    assert out["y_extreme"].isna().iloc[1]          # missing, not 0
    b = build_causal_baselines(d)
    assert b.loc[2, "n_prior_resolved_baseline"] == 1
    assert b.loc[2, STRUCTURAL] == 0.20             # the unresolved event contributes nothing


def test_same_day_reporters_do_not_see_each_others_outcomes():
    d = _frame([
        {"stock": "OLD", "earnings_date": "2019-12-01", CORRECTED_TARGET: 0.20},
        {"stock": "AAA", "earnings_date": "2020-01-01", CORRECTED_TARGET: 0.01},
        {"stock": "BBB", "earnings_date": "2020-01-01", CORRECTED_TARGET: 0.30},
    ])
    b = build_causal_baselines(d)
    assert b.loc[1, "market_prior_extreme_rate"] == 1.0
    assert b.loc[2, "market_prior_extreme_rate"] == 1.0   # not 0.5 — BBB cannot see AAA


# ──────────────────────────── walk-forward temporal separation ───────────────────────────
def _wf_sample() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for s in range(40):
        for e in range(24):
            date = pd.Timestamp("2017-02-01") + pd.Timedelta(days=91 * e)
            rows.append({"stock": f"S{s:02d}", "event_clock": date,
                         "y_extreme": float(rng.random() < 0.2),
                         STRUCTURAL: rng.uniform(0.02, 0.09),
                         "feat": rng.normal()})
    return pd.DataFrame(rows)


def test_walk_forward_never_trains_on_the_test_year_or_later():
    sample = _wf_sample()
    seen = {}
    real_fit = fit_block

    def spy(train, features, **kw):
        seen[len(seen)] = train["event_clock"].max()
        return real_fit(train, features, **kw)

    import research.phase5_event_signal as mod
    mod.fit_block = spy
    try:
        pred = walk_forward_predictions(sample, [STRUCTURAL, "feat"])
    finally:
        mod.fit_block = real_fit

    for year, block in pred.groupby("test_year"):
        assert block["event_clock"].min() >= pd.Timestamp(year=int(year), month=1, day=1)
        assert block["event_clock"].max() < pd.Timestamp(year=int(year) + 1, month=1, day=1)
    # every training block ended strictly before its own test year
    for i, max_train in enumerate(sorted(seen.values())):
        assert max_train < pd.Timestamp(year=2021 + i, month=1, day=1)
    assert pred.index.is_unique                      # no event scored twice


def test_scaling_and_imputation_constants_come_from_training_rows_only():
    train = pd.DataFrame({
        STRUCTURAL: [0.01, 0.02, 0.03, 0.04],
        "feat": [1.0, 2.0, 3.0, np.nan],
        "y_extreme": [0, 1, 0, 1],
    })
    fit = fit_block(train, [STRUCTURAL, "feat"])
    assert fit.medians["feat"] == 2.0                      # median of the TRAIN column
    assert np.isclose(fit.means["feat"], np.mean([1.0, 2.0, 3.0, 2.0]))

    # A test block with a wildly different distribution must not move the constants, and a
    # missing test value must be filled with the TRAIN median.
    test = pd.DataFrame({STRUCTURAL: [0.5, 0.5], "feat": [1000.0, np.nan]})
    p = apply_block(fit, test, [STRUCTURAL, "feat"])
    manual_z = (fit.medians["feat"] - fit.means["feat"]) / fit.stds["feat"]
    z_struct = (0.5 - fit.means[STRUCTURAL]) / fit.stds[STRUCTURAL]
    expected = fit.model.predict_proba([[z_struct, manual_z]])[0, 1]
    assert np.isclose(p[1], expected)


# ───────────────────────────── common-sample comparison guard ────────────────────────────
def test_walk_forward_models_are_compared_on_identical_rows():
    sample = _wf_sample()
    a = walk_forward_predictions(sample, [STRUCTURAL])
    b = walk_forward_predictions(sample, [STRUCTURAL, "feat"])
    assert a.index.equals(b.index)
    delta = paired_bootstrap_delta(b, a, reps=25)
    assert delta["n"] == len(a)


def test_bootstrap_resamples_stocks_and_is_deterministic():
    sample = _wf_sample()
    a = walk_forward_predictions(sample, [STRUCTURAL])
    b = walk_forward_predictions(sample, [STRUCTURAL, "feat"])
    one = paired_bootstrap_delta(b, a, reps=40, seed=11)
    two = paired_bootstrap_delta(b, a, reps=40, seed=11)
    assert one == two
    assert one["cluster"] == "stock"
    # Clustering must not be silently narrower than the row bootstrap on clustered data.
    rowwise = paired_bootstrap_delta(b, a, reps=40, seed=11, cluster=None)
    assert one["ci_hi"] - one["ci_lo"] >= 0.5 * (rowwise["ci_hi"] - rowwise["ci_lo"])


# ─────────────────────────────── the two synthetic panels ────────────────────────────────
@pytest.mark.parametrize("kind, seed", [("between", 11), ("between_exact", 15)])
def test_a_stock_constant_feature_is_null_within_stock_however_strong_it_looks(kind, seed):
    """The control that cannot be fooled.

    Both panels give the candidate a value that is CONSTANT within a stock, so neither can
    contain information about which quarter is dangerous. Raw AUC is high, because the
    candidate is a phenotype proxy. The structural-decile control reduces it but does NOT
    return it to 0.5 — `long_mean_abs_reaction` is only a noisy estimate of the phenotype,
    so a candidate measuring the same thing independently still adds inside a decile. That
    is errors-in-variables, not event-specific signal.

    The within-stock reading is exactly 0.5 in both cases, because a stock-constant feature
    ranks no quarter above another. That is why the survivor rule requires it.
    """
    d = _analysis(_panel(60, 20, seed=seed, kind=kind))
    uni = univariate_signal(d, ["candidate"])
    row = uni.iloc[0]
    assert row["roc_auc_oriented"] > 0.60, "a phenotype proxy should look strong raw"
    assert np.isclose(row["within_stock_pooled_auc"], 0.5), "stock-constant: no within signal"
    assert 0.5 < row["auc_within_structural_decile"] < row["roc_auc_oriented"], \
        "the decile control attenuates but does not eliminate — this is the caveat"
    assert select_survivors(uni) == [], "the within-stock requirement must reject it"


@pytest.mark.parametrize("kind, seed", [("between", 14), ("between_exact", 16)])
def test_a_walk_forward_gain_alone_does_not_prove_event_specific_signal(kind, seed):
    """Same caveat, stated against the Part 4 machinery.

    A stock-constant phenotype proxy can genuinely improve a walk-forward model whose only
    other input is the noisy expanding mean. It can even RAISE the fitted model's
    within-stock AUC, by diluting the weight on that mechanically anti-predictive expanding
    mean. So neither a positive `delta_auc` nor a rise in the model's within-stock AUC
    proves event-specific signal. The clean test is the within-stock AUC of the FEATURE
    itself, which is exactly 0.5 here and stays there.
    """
    d = _analysis(_panel(60, 24, seed=seed, kind=kind)).dropna(subset=["candidate"])
    base = walk_forward_predictions(d, [STRUCTURAL], first_year=2016, last_year=2017)
    plus = walk_forward_predictions(d, [STRUCTURAL, "candidate"],
                                    first_year=2016, last_year=2017)
    delta = paired_bootstrap_delta(plus, base, reps=60)
    assert delta["delta_auc"] > 0            # the model improves...
    ws = pooled_pair_auc(plus.assign(y_extreme=plus["y_extreme"]), "p", ["stock"])[0]
    assert ws < 0.5, "the model still cannot rank quarters inside a stock"
    assert np.isclose(pooled_pair_auc(d, "candidate", ["stock"])[0], 0.5), \
        "and the feature itself carries exactly no within-stock information"


def test_a_true_within_stock_feature_survives_both_controls():
    d = _analysis(_panel(60, 20, seed=12, kind="within"))
    uni = univariate_signal(d, ["candidate"])
    row = uni.iloc[0]
    assert row["within_stock_pooled_auc"] > 0.55
    assert row["auc_within_structural_decile"] > 0.55
    assert abs(row["spearman_vs_structural"]) < 0.2, "by construction it is not a phenotype"
    assert select_survivors(uni) == ["candidate"]


def test_a_true_within_stock_feature_improves_the_walk_forward_model():
    d = _analysis(_panel(60, 24, seed=13, kind="within"))
    d = d.dropna(subset=["candidate"])
    base = walk_forward_predictions(d, [STRUCTURAL], first_year=2016, last_year=2017)
    plus = walk_forward_predictions(d, [STRUCTURAL, "candidate"],
                                    first_year=2016, last_year=2017)
    delta = paired_bootstrap_delta(plus, base, reps=60)
    assert delta["delta_auc"] > 0.02
    assert delta["ci_lo"] > 0


def test_a_between_stock_feature_does_not_improve_the_walk_forward_model():
    d = _analysis(_panel(60, 24, seed=14, kind="between")).dropna(subset=["candidate"])
    base = walk_forward_predictions(d, [STRUCTURAL], first_year=2016, last_year=2017)
    plus = walk_forward_predictions(d, [STRUCTURAL, "candidate"],
                                    first_year=2016, last_year=2017)
    delta = paired_bootstrap_delta(plus, base, reps=60)
    assert abs(delta["delta_auc"]) < 0.03
    assert delta["ci_lo"] < 0 < delta["ci_hi"]


# ───────────────────────── within-stock metric construction itself ───────────────────────
def test_pooled_pair_auc_weights_by_pairs_not_by_stock():
    """One big stock that the feature ranks perfectly, one tiny stock it ranks backwards.

    The macro mean calls this 0.5. Pooling by pairs calls it what it is: mostly right."""
    rows = []
    for i, (y, s) in enumerate([(0, 0.0), (0, 1.0), (0, 2.0), (1, 3.0), (1, 4.0), (1, 5.0)]):
        rows.append({"stock": "BIG", "y_extreme": float(y), "f": s})
    rows += [{"stock": "TINY", "y_extreme": 1.0, "f": 0.0},
             {"stock": "TINY", "y_extreme": 0.0, "f": 1.0}]
    d = pd.DataFrame(rows)
    pooled, groups, events, pairs = pooled_pair_auc(d, "f", ["stock"])
    assert groups == 2 and events == 8
    assert pairs == 9 + 1                       # 3x3 from BIG, 1x1 from TINY
    assert np.isclose(pooled, 9 / 10)


def test_within_stock_views_flag_disagreement_rather_than_averaging_it():
    d = pd.DataFrame({
        "stock": ["A"] * 6 + ["B"] * 6,
        "event_clock": list(pd.date_range("2021-01-01", periods=6, freq="91D")) * 2,
        "y_extreme": [0, 0, 0, 1, 1, 1] * 2,
        "f": [0, 1, 2, 3, 4, 5] + [5, 4, 3, 2, 1, 0],
    })
    out = within_stock_validation(d, ["f"])
    row = out.iloc[0]
    assert np.isclose(row["pooled_pair_auc"], 0.5)
    assert not row["direction_agreed"] or row["views_above_half"] in (0, 5)


def test_the_sequence_reference_measures_pure_elapsed_time():
    """The seq reference must be computable and land near 0.5 when timing is irrelevant."""
    rng = np.random.default_rng(5)
    rows = []
    for s in range(30):
        for e in range(12):
            rows.append({"stock": f"S{s}", "event_clock": pd.Timestamp("2021-01-01")
                         + pd.Timedelta(days=91 * e), "y_extreme": float(rng.random() < 0.3),
                         "f": rng.normal()})
    out = within_stock_validation(pd.DataFrame(rows), ["f"])
    assert abs(out.iloc[0]["reference_seq_pooled_pair_auc"] - 0.5) < 0.08


# ──────────────────────────── residual formulation behaviour ─────────────────────────────
def test_residuals_are_measured_against_the_stocks_own_causal_history():
    d = _analysis(_panel(20, 16, seed=21, kind="within"))
    r = residual_frame(d)
    assert np.allclose(
        (r[CORRECTED_TARGET] - r[STRUCTURAL]).dropna(),
        r["residual_abs_move"].dropna())
    assert r["expected_abs_move"].equals(r[STRUCTURAL])


def test_a_within_stock_feature_predicts_positive_residuals():
    d = _analysis(_panel(60, 20, seed=22, kind="within")).dropna(subset=["candidate"])
    out = residual_signal(residual_frame(d), ["candidate"]).iloc[0]
    assert out["spearman_within_stock_log_ratio"] > 0.05
    assert out["residual_sign_auc"] > 0.53


def test_a_stock_constant_feature_reports_no_within_stock_residual_correlation():
    """Constant within a stock means UNDEFINED within-stock signal, which must be reported
    as missing rather than as a zero that could be mistaken for a measurement."""
    d = _analysis(_panel(60, 20, seed=23, kind="between")).dropna(subset=["candidate"])
    out = residual_signal(residual_frame(d), ["candidate"]).iloc[0]
    assert pd.isna(out["spearman_within_stock_log_ratio"])
    assert abs(out["residual_sign_auc"] - 0.5) < 0.05


# ────────────────────────────────── the audit registry ───────────────────────────────────
@pytest.mark.parametrize("name, reason_contains", [
    ("surprise_percentage", "pre-announcement"),
    ("daily_ret", "reaction"),
    ("momentum_fragility_score", "look-ahead"),
    ("days_to_earnings", "identically 0"),
    ("atm_iv", "coverage"),
    ("expected_move_pct", "coverage"),
    ("earnings_explosiveness_z", "LEGACY"),
])
def test_the_known_unsafe_columns_are_excluded_with_a_stated_reason(name, reason_contains):
    c = CANDIDATES_BY_NAME[name]
    assert not c.headline
    assert reason_contains in (c.exclude_reason + " " + c.leakage)
    assert name not in HEADLINE_FEATURES


def test_the_audit_table_reports_coverage_and_never_promotes_an_excluded_feature():
    d = _analysis(_panel(20, 12, seed=31, kind="within"))
    audit = feature_audit(d, d)
    assert set(audit["feature"]) == {c.name for c in CANDIDATES}
    for _i, row in audit.iterrows():
        c = CANDIDATES_BY_NAME[row["feature"]]
        if not c.headline:
            assert not row["headline_eligible"]
            assert row["exclusion_reason"]
    # A column absent from the frame can never be eligible, whatever the registry says.
    assert not audit.loc[audit["feature"].eq("atm_iv"), "headline_eligible"].iloc[0]


def test_structural_decile_auc_is_computed_inside_deciles_only():
    """A feature that equals the structural baseline carries no information inside a
    decile, however strong it looks overall."""
    rng = np.random.default_rng(9)
    n = 2000
    struct = rng.uniform(0.02, 0.10, n)
    y = (rng.random(n) < (struct * 6)).astype(float)
    d = pd.DataFrame({"y_extreme": y, STRUCTURAL: struct, "f": struct,
                      "stock": [f"S{i % 50}" for i in range(n)]})
    overall = pooled_pair_auc(d.assign(_all=0), "f", ["_all"])[0]
    inside, _pairs = structural_decile_auc(d, "f")
    assert overall > 0.6
    assert abs(inside - 0.5) < 0.05


def test_an_expanding_mean_of_the_outcome_is_negatively_within_stock_biased():
    """Why every structural predictor scored below 0.5 within stock in Phase 4.

    This panel has NO within-stock signal at all: each stock's absolute moves are i.i.d.
    draws from that stock's own fixed distribution. Nothing about quarter t predicts
    quarter t+1. Yet the expanding mean of the stock's own prior moves — which is exactly
    `long_mean_abs_reaction` — lands well below 0.5 within stock.

    The mechanism is regression to the mean in the estimator, not in the world. A large
    move lifts the running mean and it stays lifted for every later event of that stock,
    and those later events are ordinary. So high-running-mean events are systematically the
    ordinary ones, and the extreme event itself was preceded by a lower running mean than
    the quarters that follow it. A within-stock AUC below 0.5 for an expanding statistic is
    therefore the NULL expectation, not evidence of mean reversion in returns.
    """
    d = _analysis(_panel(80, 24, seed=77, kind="between_exact"))
    structural_ws = pooled_pair_auc(d, STRUCTURAL, ["stock"])[0]
    assert structural_ws < 0.45, "the expanding mean is mechanically anti-predictive"

    # The stock-constant truth, by contrast, sits exactly at the genuine null.
    assert np.isclose(pooled_pair_auc(d, "candidate", ["stock"])[0], 0.5)
