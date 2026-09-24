import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from feature_engineering.announcement_timing import TARGET_AVAILABLE
from research.phase3_target_rebuild import CORRECTED_TARGET, PHASE3_PREFIX
from research.phase4_baselines import (
    _within_stock_auc,
    build_causal_baselines,
    cohort_sample,
    paired_bootstrap_auc_delta,
    prepare_analysis_frame,
)


def _events(rows):
    d = pd.DataFrame(rows)
    defaults = {
        "is_pending": False,
        "reaction_3d_anchored_status": TARGET_AVAILABLE,
        "sector": "Tech",
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
    if "phase3_announce_date" not in d.columns:
        d["phase3_announce_date"] = d["earnings_date"]
    else:
        d["phase3_announce_date"] = pd.to_datetime(d["phase3_announce_date"])
    return d


def test_stock_baseline_never_reads_current_outcome():
    d = _events([
        {"stock": "AAA", "earnings_date": "2024-01-01", CORRECTED_TARGET: 0.10},
        {"stock": "AAA", "earnings_date": "2024-04-01", CORRECTED_TARGET: 0.02},
        {"stock": "AAA", "earnings_date": "2024-07-01", CORRECTED_TARGET: 0.12},
    ])
    b = build_causal_baselines(d)
    assert np.isnan(b.loc[0, "stock_prior_extreme_rate"])
    assert b.loc[1, "stock_prior_extreme_rate"] == 1.0
    assert b.loc[1, "last_abs_reaction"] == 0.10
    assert b.loc[2, "stock_prior_extreme_rate"] == 0.5
    assert np.isclose(b.loc[2, "long_mean_abs_reaction"], 0.06)


def test_same_day_market_and_sector_outcomes_do_not_leak_across_rows():
    d = _events([
        {"stock": "OLD", "sector": "Tech", "earnings_date": "2023-12-01", CORRECTED_TARGET: 0.10},
        {"stock": "AAA", "sector": "Tech", "earnings_date": "2024-01-01", CORRECTED_TARGET: 0.01},
        {"stock": "BBB", "sector": "Tech", "earnings_date": "2024-01-01", CORRECTED_TARGET: 0.20},
        {"stock": "CCC", "sector": "Tech", "earnings_date": "2024-02-01", CORRECTED_TARGET: 0.01},
    ])
    b = build_causal_baselines(d)
    # Both Jan-1 reporters see only OLD's prior result (= extreme), not each other.
    assert b.loc[1, "market_prior_extreme_rate"] == 1.0
    assert b.loc[2, "market_prior_extreme_rate"] == 1.0
    assert b.loc[1, "sector_prior_extreme_rate"] == 1.0
    assert b.loc[2, "sector_prior_extreme_rate"] == 1.0
    # Feb sees all three prior observations: extreme, non-extreme, extreme.
    assert np.isclose(b.loc[3, "market_prior_extreme_rate"], 2 / 3)


def test_unresolved_outcome_is_missing_information_not_a_non_extreme_trial():
    d = _events([
        {"stock": "AAA", "earnings_date": "2024-01-01", CORRECTED_TARGET: 0.10},
        {
            "stock": "AAA",
            "earnings_date": "2024-04-01",
            CORRECTED_TARGET: np.nan,
            "reaction_3d_anchored_status": "unresolved_no_timestamp",
        },
        {"stock": "AAA", "earnings_date": "2024-07-01", CORRECTED_TARGET: 0.02},
    ])
    b = build_causal_baselines(d)
    assert b.loc[2, "n_prior_resolved_baseline"] == 1
    assert b.loc[2, "stock_prior_extreme_rate"] == 1.0
    assert b.loc[2, "last_abs_reaction"] == 0.10


def test_recent_and_rolling_features_use_only_prior_resolved_events():
    rows = []
    vals = np.linspace(0.01, 0.30, 30)
    for i, value in enumerate(vals):
        rows.append({
            "stock": "AAA",
            "earnings_date": pd.Timestamp("2010-01-01") + pd.Timedelta(days=90 * i),
            CORRECTED_TARGET: value,
        })
    d = _events(rows)
    b = build_causal_baselines(d)
    idx = 29
    prior = vals[:29]
    assert np.isclose(b.loc[idx, "recent4_mean_abs_reaction"], np.mean(prior[-4:]))
    assert np.isclose(b.loc[idx, "recent8_mean_abs_reaction"], np.mean(prior[-8:]))
    assert np.isclose(b.loc[idx, "rolling28_p75_abs_reaction"], np.quantile(prior[-28:], 0.75))


def test_shrunk_stock_rate_uses_prior_market_rate_and_twenty_event_prior():
    d = _events([
        {"stock": "OLD", "earnings_date": "2023-01-01", CORRECTED_TARGET: 0.10},
        {"stock": "AAA", "earnings_date": "2023-02-01", CORRECTED_TARGET: 0.10},
        {"stock": "AAA", "earnings_date": "2023-05-01", CORRECTED_TARGET: 0.01},
    ])
    b = build_causal_baselines(d)
    # Before AAA's second event: market prior = 2/2 = 1, stock prior = 1/1 = 1.
    assert np.isclose(b.loc[2, "stock_prior_extreme_rate_shrunk20"], 1.0)


def test_mature28_cohort_is_a_true_common_row_sample():
    rows = []
    for i in range(35):
        rows.append({
            "stock": "AAA",
            "earnings_date": pd.Timestamp("2014-01-01") + pd.Timedelta(days=90 * i),
            CORRECTED_TARGET: 0.10 if i % 3 == 0 else 0.02,
            PHASE3_PREFIX + "n_prior_resolved_reactions": i,
            PHASE3_PREFIX + "risk_score": 80 if i % 3 == 0 else 60,
        })
    d = prepare_analysis_frame(_events(rows))
    sample, predictors = cohort_sample(d, "mature_28")
    assert len(sample) == 7
    assert sample[predictors].notna().all().all()
    assert (sample[PHASE3_PREFIX + "n_prior_resolved_reactions"] >= 28).all()


def test_within_stock_auc_requires_both_classes_and_is_weighted():
    d = pd.DataFrame({
        "stock": ["A"] * 4 + ["B"] * 4,
        "y_extreme": [0, 0, 1, 1, 0, 0, 0, 0],
        "score": [0.1, 0.2, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4],
    })
    n, macro, weighted = _within_stock_auc(d, "score")
    assert n == 1
    assert macro == weighted == 1.0


def test_paired_bootstrap_is_deterministic_and_positive_for_better_model():
    rng = np.random.default_rng(7)
    y = np.array([0] * 80 + [1] * 20)
    model = y + rng.normal(0, 0.15, len(y))
    baseline = rng.normal(0, 1.0, len(y))
    d = pd.DataFrame({
        "y_extreme": y,
        "model": model,
        "baseline": baseline,
    })
    a = paired_bootstrap_auc_delta(d, "model", "baseline", reps=100, seed=123)
    b = paired_bootstrap_auc_delta(d, "model", "baseline", reps=100, seed=123)
    assert a == b
    assert a["delta_auc"] > 0
    assert a["ci_lo"] > 0
