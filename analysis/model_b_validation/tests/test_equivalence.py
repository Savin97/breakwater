"""Production Model B must reproduce the frozen refit's Model B feature exactly.

Run:  PYTHONPATH=. .venv/bin/python -m pytest analysis/model_b_validation/tests -q
Kept out of `testing/` because it reads the refit oracle's saved frame, which will be
archived; the production invariants themselves live in testing/test_model_b.py.
"""
import pandas as pd
import pytest

from analysis.model_b_validation import validate as V


@pytest.fixture(scope="module")
def result():
    oracle = pd.read_parquet(V.ORACLE_FRAME)
    events, daily, prod = V.production_signal()
    return V.equivalence(oracle, prod), V.reference_identity(events, daily, oracle)


def test_same_events_cutoffs_counts_and_signal(result):
    eq, _ = result
    assert eq["only_in_oracle"] == eq["only_in_production"] == 0
    assert eq["cutoff_mismatch"] == eq["n_prior_mismatch"] == eq["hist_null_mismatch"] == 0
    assert eq["hist_abs_diff_max"] <= 1e-12
    assert eq["research_sample_n"] == 21927
    assert eq["research_sample_rank_mismatch"] == 0


def test_reference_sample_is_the_candidate_fit_sample(result):
    _, ref = result
    assert ref["reference_n"] == ref["oracle_candidate_fit_n"] == 20491
    assert ref["only_reference"] == ref["only_oracle"] == 0
