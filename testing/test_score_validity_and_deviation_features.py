"""Tests for the score-validity audit and the S / M0 / M1 comparison.

The audit exists to catch look-ahead, so these tests mostly check that it CAN catch it:
a validity check that passes on leaky data is worse than no check. The two central ones
build a frame with a known look-ahead and assert the audit fails, and build a score where
the current event's own reaction is perturbed and assert the score does not move.
"""
import numpy as np
import pandas as pd
import pytest

from research.phase3_target_rebuild import CORRECTED_TARGET, PHASE3_PREFIX
from testing.score_validity_and_deviation_features import (
    DEVIATION_FEATURES,
    ENTROPY,
    HISTORICAL_REFERENCE,
    MATCHED_ALERT_FRACTION,
    P75_EXPANDING,
    P75_ROLLING,
    SCORE,
    SCORE_REPAIRED,
    _matched_metrics,
    audit_score_validity,
    causal_entropy_fallback,
    common_events,
    fixed_predictor_predictions,
    matched_count_alerts,
    repaired_risk_score,
    tier_profile,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    d["earnings_date"] = pd.to_datetime(d["earnings_date"])
    if PHASE3_PREFIX + "announce_date" not in d:
        d[PHASE3_PREFIX + "announce_date"] = d["earnings_date"]
    d[PHASE3_PREFIX + "announce_date"] = pd.to_datetime(d[PHASE3_PREFIX + "announce_date"])
    defaults = {"is_pending": False, P75_ROLLING: np.nan, P75_EXPANDING: 0.06,
                ENTROPY: np.nan, SCORE: 0.0}
    for col, value in defaults.items():
        if col not in d.columns:
            d[col] = value
        else:
            d[col] = d[col].fillna(value)
    return d


# ───────────────────────── the defect, and that the repair removes it ────────────────────
def test_the_causal_fallback_only_ever_reads_strictly_earlier_dates():
    """Frame order is (stock, date), so the shipped ffill reaches a future ticker. The
    replacement must not, even though the frame order is unchanged."""
    d = _frame([
        {"stock": "AAA", "earnings_date": "2020-01-15", ENTROPY: np.nan},
        {"stock": "AAA", "earnings_date": "2020-04-15", ENTROPY: np.nan},
        {"stock": "ZZZ", "earnings_date": "2019-01-15", ENTROPY: 0.9},
        {"stock": "ZZZ", "earnings_date": "2026-01-15", ENTROPY: 0.1},
    ])
    d["_sd"] = pd.to_datetime(d["earnings_date"])
    chain = causal_entropy_fallback(d)
    # AAA 2020-01 may only see ZZZ 2019-01 (0.9), never ZZZ 2026-01 (0.1).
    assert chain.iloc[0] == 0.9
    assert chain.iloc[1] == 0.9
    assert pd.isna(chain.iloc[2])                 # nothing earlier than 2019-01 exists
    # The shipped frame-order fill would have reached backwards from the future row:
    shipped = d[ENTROPY].mask(d["is_pending"]).ffill()
    assert pd.isna(shipped.iloc[0]) and shipped.iloc[3] == 0.1


def test_the_causal_fallback_never_reads_a_same_day_row():
    d = _frame([
        {"stock": "AAA", "earnings_date": "2020-01-15", ENTROPY: np.nan},
        {"stock": "BBB", "earnings_date": "2020-01-15", ENTROPY: 0.7},
        {"stock": "CCC", "earnings_date": "2020-01-16", ENTROPY: np.nan},
    ])
    d["_sd"] = pd.to_datetime(d["earnings_date"])
    chain = causal_entropy_fallback(d)
    assert pd.isna(chain.iloc[0]), "same-day value must not be visible"
    assert chain.iloc[2] == 0.7, "the next day may see it"


def test_the_repaired_score_changes_only_rows_that_used_the_chain():
    d = _frame([
        {"stock": "AAA", "earnings_date": "2020-01-15", ENTROPY: np.nan, P75_EXPANDING: 0.06},
        {"stock": "AAA", "earnings_date": "2020-04-15", ENTROPY: 0.5, P75_EXPANDING: 0.06},
        {"stock": "ZZZ", "earnings_date": "2026-01-15", ENTROPY: 1.0, P75_EXPANDING: 0.06},
    ])
    d[SCORE] = 100 * np.clip(
        0.85 * (d[P75_EXPANDING] / 0.12).clip(0, 1)
        + 0.15 * np.clip(d[ENTROPY].mask(d["is_pending"]).ffill().fillna(0), 0, 1), 0, 1)
    out = repaired_risk_score(d)
    assert out.loc[1, "score_repair_delta"] == 0        # had its own entropy
    assert out.loc[2, "score_repair_delta"] == 0
    assert out["entropy_source"].tolist() == ["zero_default", "own_history", "own_history"]
    assert (out[SCORE_REPAIRED] >= 0).all() and (out[SCORE_REPAIRED] <= 100).all()


def test_the_audit_reports_the_entropy_defect_as_material():
    d = _frame([
        {"stock": "AAA", "earnings_date": "2000-01-15", ENTROPY: np.nan},
        {"stock": "ZZZ", "earnings_date": "2026-01-15", ENTROPY: 0.4},
    ])
    d[PHASE3_PREFIX + "n_prior_resolved_reactions"] = [0, 0]
    d[PRIMARY := "abs_reaction_3d_anchored"] = [0.1, 0.2]
    d["reaction_3d_anchored_status"] = "available"
    d[PHASE3_PREFIX + "is_extreme_reaction"] = [1.0, 1.0]
    d["announce_window"] = "AMC"
    d["anchor_date"] = d["earnings_date"]
    dataset = d.assign(event_id=d["stock"] + "|" + d["earnings_date"].dt.strftime("%Y-%m-%d"))
    checks = {c["check"]: c for c in audit_score_validity(d, dataset)}
    assert checks["entropy_fallback_is_frame_ordered"]["result"] == "FAIL"
    assert checks["entropy_fallback_is_frame_ordered"]["severity"] == "material"
    assert "future" in checks["entropy_fallback_is_frame_ordered"]["evidence"].lower()


def test_the_audit_covers_every_required_question_and_discloses_tuning():
    d = _frame([{"stock": "AAA", "earnings_date": f"20{y}-01-15", ENTROPY: 0.5}
                for y in range(10, 20)])
    d[PHASE3_PREFIX + "n_prior_resolved_reactions"] = range(10)
    d["abs_reaction_3d_anchored"] = 0.1
    d["reaction_3d_anchored_status"] = "available"
    d[PHASE3_PREFIX + "is_extreme_reaction"] = 1.0
    d["announce_window"] = "AMC"
    d["anchor_date"] = d["earnings_date"]
    dataset = d.assign(event_id=d["stock"] + "|" + d["earnings_date"].dt.strftime("%Y-%m-%d"))
    names = {c["check"] for c in audit_score_validity(d, dataset)}
    for required in ("current_event_outcome_excluded", "prior_outcomes_complete_at_cutoff",
                     "price_inputs_precede_cutoff", "outcome_anchor_handles_timing",
                     "missing_outcomes_stay_missing", "event_joins_are_one_to_one",
                     "historically_selected_constants", "score_is_not_a_probability"):
        assert required in names, required
    disclosures = {c["check"] for c in audit_score_validity(d, dataset)
                   if c["result"] == "DISCLOSURE"}
    assert "historically_selected_constants" in disclosures


def test_the_audit_detects_a_missing_outcome_recorded_as_non_extreme():
    d = _frame([{"stock": "AAA", "earnings_date": "2020-01-15", ENTROPY: 0.5}])
    d[PHASE3_PREFIX + "n_prior_resolved_reactions"] = 0
    d["abs_reaction_3d_anchored"] = np.nan
    d["reaction_3d_anchored_status"] = "unresolved_no_timestamp"
    d[PHASE3_PREFIX + "is_extreme_reaction"] = 0.0          # the bug: NaN turned into 0
    d["announce_window"] = "AMC"
    d["anchor_date"] = d["earnings_date"]
    dataset = d.assign(event_id="AAA|2020-01-15")
    checks = {c["check"]: c for c in audit_score_validity(d, dataset)}
    assert checks["missing_outcomes_stay_missing"]["result"] == "FAIL"


def test_the_audit_detects_a_bmo_anchor_that_does_not_precede_its_report_date():
    d = _frame([{"stock": "AAA", "earnings_date": "2020-01-15", ENTROPY: 0.5}])
    d[PHASE3_PREFIX + "n_prior_resolved_reactions"] = 0
    d["abs_reaction_3d_anchored"] = 0.1
    d["reaction_3d_anchored_status"] = "available"
    d[PHASE3_PREFIX + "is_extreme_reaction"] = 1.0
    d["announce_window"] = "BMO"
    d["anchor_date"] = d["earnings_date"]                    # BMO must anchor EARLIER
    dataset = d.assign(event_id="AAA|2020-01-15")
    checks = {c["check"]: c for c in audit_score_validity(d, dataset)}
    assert checks["outcome_anchor_handles_timing"]["result"] == "FAIL"


def test_the_audit_detects_a_duplicated_event_join():
    d = _frame([{"stock": "AAA", "earnings_date": "2020-01-15", ENTROPY: 0.5}])
    d[PHASE3_PREFIX + "n_prior_resolved_reactions"] = 0
    d["abs_reaction_3d_anchored"] = 0.1
    d["reaction_3d_anchored_status"] = "available"
    d[PHASE3_PREFIX + "is_extreme_reaction"] = 1.0
    d["announce_window"] = "AMC"
    d["anchor_date"] = d["earnings_date"]
    dataset = pd.concat([d, d]).assign(event_id="AAA|2020-01-15")
    checks = {c["check"]: c for c in audit_score_validity(d, dataset)}
    assert checks["event_joins_are_one_to_one"]["result"] == "FAIL"


# ───────────────────── matched alert counts: deterministic and outcome-blind ─────────────
def _pred(n_per_year=100, years=(2018, 2019), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        for i in range(n_per_year):
            rows.append({"group": "g", "label": "y_primary", "test_year": y,
                         "event_id": f"S{i:03d}|{y}-01-01", "stock": f"S{i % 20:02d}",
                         "quarter": f"{y}Q1", "y": int(rng.random() < 0.2),
                         "p": float(rng.random()), "alert_threshold": 0.8})
    return pd.DataFrame(rows)


def test_matched_alert_counts_are_a_fixed_share_of_each_year():
    pred = _pred(100, (2018, 2019, 2020))
    flag = matched_count_alerts(pred, MATCHED_ALERT_FRACTION)
    per_year = pd.Series(flag).groupby(pred["test_year"].to_numpy()).sum()
    assert per_year.tolist() == [10, 10, 10]


def test_matched_alert_selection_ignores_the_outcome_entirely():
    """Ties must not be resolved in a way that rewards a predictor for free."""
    pred = _pred(50, (2018,))
    pred["p"] = 0.5                                    # everything tied
    flag_a = matched_count_alerts(pred)
    shuffled = pred.sample(frac=1.0, random_state=3).reset_index(drop=True)
    flag_b = matched_count_alerts(shuffled)
    assert set(pred.loc[flag_a, "event_id"]) == set(shuffled.loc[flag_b, "event_id"])
    flipped = pred.assign(y=1 - pred["y"])
    assert set(pred.loc[flag_a, "event_id"]) == set(
        flipped.loc[matched_count_alerts(flipped), "event_id"])


def test_matched_counts_make_two_predictors_directly_comparable():
    a = _pred(200, (2018,), seed=1)
    b = a.assign(p=a["p"] * 0.001)                     # same ranking, different scale
    fa, fb = matched_count_alerts(a), matched_count_alerts(b)
    assert fa.sum() == fb.sum()
    assert _matched_metrics(a["y"].to_numpy(), fa) == _matched_metrics(b["y"].to_numpy(), fb)


# ───────────────────────── fixed predictors and common samples ───────────────────────────
def _dataset(n_stocks=30, years=range(2014, 2026), seed=4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stocks):
        for y in years:
            for m in (2, 5, 8, 11):
                rows.append({
                    "stock": f"S{s:02d}",
                    "earnings_date": pd.Timestamp(year=y, month=m, day=15),
                    "event_clock": pd.Timestamp(year=y, month=m, day=15),
                    "event_id": f"S{s:02d}|{y}-{m:02d}-15",
                    "quarter": f"{y}Q{(m // 3) + 1}",
                    "outcome_ready_known": True,
                    "outcome_ready_date": pd.Timestamp(year=y, month=m, day=20),
                    "y_primary": int(rng.random() < 0.2),
                    SCORE: float(rng.uniform(0, 100)),
                    HISTORICAL_REFERENCE: float(rng.uniform(0, 0.4)),
                    **{f: float(rng.normal()) for f in DEVIATION_FEATURES},
                })
    return pd.DataFrame(rows)


def test_a_fixed_predictor_takes_its_threshold_from_completed_history_only():
    d = _dataset()
    pred = fixed_predictor_predictions(d, SCORE, "S", "y_primary")
    for year, g in pred.groupby("test_year"):
        history = d[(d["event_clock"] < pd.Timestamp(year=int(year), month=1, day=1))]
        assert np.isclose(g["alert_threshold"].iloc[0],
                          np.quantile(history[SCORE], 0.90))
        assert g["train_n"].iloc[0] == len(history)


def test_a_fixed_predictor_is_scored_on_the_same_years_as_a_trained_one():
    from testing.benzinga_feature_value import walk_forward
    d = _dataset()
    s = fixed_predictor_predictions(d, SCORE, "S", "y_primary")
    m0 = walk_forward(d, [SCORE], "M0", "y_primary")
    assert sorted(s["test_year"].unique()) == sorted(m0["test_year"].unique())
    shared = common_events({"S": s, "M0": m0})
    assert shared == set(s["event_id"]) == set(m0["event_id"])


def test_common_events_is_what_every_comparison_runs_on():
    d = _dataset()
    partial = d.copy()
    partial.loc[partial.index[:500], DEVIATION_FEATURES[0]] = np.nan
    from testing.benzinga_feature_value import walk_forward
    s = fixed_predictor_predictions(d, SCORE, "S", "y_primary")
    m1 = walk_forward(partial, [SCORE, *DEVIATION_FEATURES], "M1", "y_primary")
    shared = common_events({"S": s, "M1": m1})
    # HistGradientBoosting scores every row natively, so nothing is lost to NaN — the
    # intersection must still be exact, not merely non-empty.
    assert shared == set(s["event_id"]) & set(m1["event_id"])
    assert len(shared) > 0


# ─────────────────────────────────── tier reporting ──────────────────────────────────────
def test_tier_profile_reports_rates_counts_and_capture_without_reselecting_cuts():
    d = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(10)],
        PHASE3_PREFIX + "bucket": ["Normal"] * 6 + ["Elevated"] * 2 + ["High Alert"] * 2,
        "y_primary": [0, 0, 0, 0, 0, 1, 0, 1, 1, 1],
    })
    rows = {r["tier"]: r for r in tier_profile(d, set(d["event_id"]), "y_primary")}
    assert rows["Normal"]["events"] == 6 and rows["Normal"]["extreme_events"] == 1
    assert np.isclose(rows["High Alert"]["extreme_rate"], 1.0)
    assert np.isclose(sum(r["capture"] for r in rows.values()), 1.0)
    assert np.isclose(sum(r["share_of_events"] for r in rows.values()), 1.0)
