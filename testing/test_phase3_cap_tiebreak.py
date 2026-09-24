"""Tests for the Phase 3 cap tie-break variant.

The variant's whole claim is "this changes ordering inside the tie and nothing else", so
the tests are mostly attempts to falsify that: a case where the capped group is NOT the
tied maximum must refuse to run, the tier must come out untouched, and the top-K accounting
must tell the truth about how much of a selection a tie decided.
"""
import numpy as np
import pandas as pd
import pytest

from research.phase3_cap_tiebreak import (
    P75,
    P75_CEILING,
    SCORE,
    TIER,
    V_RAW_P75,
    V_SHIPPED,
    V_TIEBREAK,
    add_variants,
    assert_tiebreak_is_order_safe,
    top_k_metrics,
    variant_row,
)


def _sample(p75_values, y_values, scores=None, tiers=None) -> pd.DataFrame:
    """A frame shaped like the prepared Phase 3 sample."""
    p75 = np.asarray(p75_values, float)
    score = (100 * np.clip(0.85 * np.clip(p75 / P75_CEILING, 0, 1) + 0.15, 0, 1)
             if scores is None else np.asarray(scores, float))
    n = len(p75)
    return pd.DataFrame({
        "event_id": [f"E{i:03d}" for i in range(n)],
        "stock": [f"S{i:03d}" for i in range(n)],
        "y": np.asarray(y_values, int),
        SCORE: score,
        P75: p75,
        TIER: tiers if tiers is not None else ["High Alert"] * n,
        "is_capped": p75 >= P75_CEILING,
        "quarter": ["2021Q1"] * n,
        "year": [2021] * n,
    })


# ───────────────────────────────── order safety ──────────────────────────────────────────
def test_the_tiebreak_only_moves_events_inside_the_tie():
    d = _sample([0.05, 0.09, 0.12, 0.20, 0.30], [0, 0, 1, 0, 1])
    out = add_variants(d)
    uncapped = ~out["is_capped"]
    # below the ceiling the variant is the score, exactly
    assert (out.loc[uncapped, V_TIEBREAK] == out.loc[uncapped, V_SHIPPED]).all()
    # inside the tie, ordering now follows p75
    cap = out[out["is_capped"]].sort_values(V_TIEBREAK)
    assert cap[P75].is_monotonic_increasing
    assert cap[V_SHIPPED].nunique() == 1          # the shipped score could not rank them


def test_the_tiebreak_never_reorders_a_capped_event_against_an_uncapped_one():
    d = add_variants(_sample([0.02, 0.06, 0.119, 0.12, 0.39], [0, 1, 0, 1, 1]))
    top = d.loc[d["is_capped"], V_TIEBREAK].min()
    assert (d.loc[~d["is_capped"], V_TIEBREAK] < top).all()


def test_it_refuses_to_run_when_the_capped_group_is_not_the_tied_maximum():
    """If some uncapped event outscored the cap, adding a positive term would reorder
    events the score deliberately separated — that is not a tie-break."""
    d = _sample([0.13, 0.05], [1, 0], scores=[90.0, 95.0])
    with pytest.raises(ValueError, match="at/above the cap"):
        assert_tiebreak_is_order_safe(d)
    with pytest.raises(ValueError):
        add_variants(d)


def test_it_refuses_when_capped_events_do_not_share_one_score():
    d = _sample([0.13, 0.20], [1, 0], scores=[100.0, 92.5])
    with pytest.raises(ValueError, match="one score"):
        assert_tiebreak_is_order_safe(d)


def test_order_safety_passes_on_a_clean_sample_and_reports_the_facts():
    facts = assert_tiebreak_is_order_safe(_sample([0.05, 0.13, 0.25], [0, 1, 1]))
    assert facts["order_safe"] and facts["capped_events"] == 2
    assert facts["distinct_scores_among_capped"] == 1
    assert facts["non_capped_at_or_above_capped_score"] == 0


def test_a_sample_with_no_capped_events_is_a_no_op():
    d = add_variants(_sample([0.02, 0.05, 0.09], [0, 1, 0]))
    assert (d[V_TIEBREAK] == d[V_SHIPPED]).all()


# ───────────────────────────── the tier must not move ────────────────────────────────────
def test_the_variant_leaves_the_tier_column_untouched():
    d = _sample([0.05, 0.13, 0.25], [0, 1, 1],
                tiers=["Normal", "High Alert", "High Alert"])
    out = add_variants(d)
    pd.testing.assert_series_equal(out[TIER], d[TIER])
    for variant in (V_SHIPPED, V_TIEBREAK, V_RAW_P75):
        assert variant in out.columns
    assert set(out[TIER]) == {"Normal", "High Alert"}


def test_tier_level_metrics_are_identical_across_variants():
    from research.phase3_cap_tiebreak import CORRECTED_TARGET
    from research.phase3_target_rebuild import variant_metrics
    d = add_variants(_sample([0.02, 0.05, 0.09, 0.13, 0.30] * 8,
                             [0, 0, 1, 1, 1] * 8,
                             tiers=(["Normal"] * 3 + ["High Alert"] * 2) * 8))
    d[CORRECTED_TARGET] = np.where(d["y"] == 1, 0.15, 0.01)
    rows = {v: variant_metrics(d, name=v, cohort="c", target_col=CORRECTED_TARGET,
                               score_col=v, bucket_col=TIER)
            for v in (V_SHIPPED, V_TIEBREAK, V_RAW_P75)}
    def same(a, b):
        # an empty tier yields NaN on both sides, and NaN != NaN
        return a == b or (isinstance(a, float) and isinstance(b, float)
                          and np.isnan(a) and np.isnan(b))

    tier_fields = [k for k in rows[V_SHIPPED] if k not in ("variant", "score_auc_ge_8")]
    for v in (V_TIEBREAK, V_RAW_P75):
        for f in tier_fields:
            assert same(rows[v][f], rows[V_SHIPPED][f]), f
    # and the ranking metric is the one field that IS allowed to differ
    assert rows[V_TIEBREAK]["score_auc_ge_8"] >= rows[V_SHIPPED]["score_auc_ge_8"] - 1e-12


# ──────────────────────────── top-K tells the truth about ties ───────────────────────────
def test_top_k_reports_how_much_of_the_selection_a_tie_decided():
    """Ten events, six tied at the top, asking for the top three: every slot is decided by
    the tie and the metric must say so."""
    d = _sample([0.13] * 6 + [0.05] * 4, [1, 1, 1, 0, 0, 0] + [0] * 4)
    m = top_k_metrics(add_variants(d), V_SHIPPED, 0.30)
    assert m["k"] == 3
    assert m["events_tied_at_the_cut"] == 6
    assert m["slots_decided_by_the_tie"] == 3
    assert m["share_of_selection_decided_by_tie"] == 1.0
    # expected precision = the tied block's own base rate (3 of 6 are extreme)
    assert np.isclose(m["expected_precision"], 0.5)


def test_expected_and_deterministic_agree_when_no_tie_straddles_the_cut():
    d = add_variants(_sample([0.30, 0.20, 0.13, 0.09, 0.05], [1, 1, 0, 0, 0]))
    m = top_k_metrics(d, V_TIEBREAK, 0.40)
    assert m["events_tied_at_the_cut"] == 1
    assert m["slots_decided_by_the_tie"] == 0
    assert np.isclose(m["deterministic_precision"], m["expected_precision"])
    assert np.isclose(m["expected_precision"], 1.0)


def test_the_tiebreak_removes_the_arbitrary_selection_the_shipped_score_leaves():
    """The point of the whole exercise: with the tie resolved, the top-K is determined."""
    d = add_variants(_sample([0.13, 0.15, 0.20, 0.30] + [0.05] * 6,
                             [0, 0, 1, 1] + [0] * 6))
    shipped = top_k_metrics(d, V_SHIPPED, 0.20)
    tie = top_k_metrics(d, V_TIEBREAK, 0.20)
    assert shipped["slots_decided_by_the_tie"] == 2      # arbitrary under the shipped score
    assert tie["slots_decided_by_the_tie"] == 0          # determined under the variant
    assert tie["expected_precision"] > shipped["expected_precision"]


def test_within_cap_auc_is_undefined_for_the_shipped_score_and_defined_for_the_variant():
    d = add_variants(_sample([0.13, 0.15, 0.20, 0.30, 0.05], [0, 0, 1, 1, 0]))
    shipped = variant_row(d, V_SHIPPED, "c", "pooled")
    tie = variant_row(d, V_TIEBREAK, "c", "pooled")
    assert shipped["within_cap_n"] == tie["within_cap_n"] == 4
    assert shipped["within_cap_auc"] == 0.5             # a constant ranks nothing
    assert tie["within_cap_auc"] == 1.0                 # p75 orders them correctly here


def test_pooled_auc_can_only_improve_or_hold_when_a_tie_is_broken_correctly():
    rng = np.random.default_rng(0)
    p75 = np.concatenate([rng.uniform(0.01, 0.119, 200), rng.uniform(0.12, 0.35, 60)])
    y = (rng.random(len(p75)) < np.clip(p75 * 3, 0, 0.9)).astype(int)
    d = add_variants(_sample(p75, y))
    a = variant_row(d, V_SHIPPED, "c", "pooled")["roc_auc"]
    b = variant_row(d, V_TIEBREAK, "c", "pooled")["roc_auc"]
    assert b >= a - 1e-12
