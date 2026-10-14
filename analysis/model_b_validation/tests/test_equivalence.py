"""Production Model B must reproduce the frozen refit's Model B feature exactly.

Run:  PYTHONPATH=. .venv/bin/python -m pytest analysis/model_b_validation/tests -q
Kept out of `testing/` because it imports the refit oracle's code, which will be archived;
the production invariants themselves live in testing/test_model_b.py and
testing/test_model_b_snapshot.py.

Runs on the pinned snapshot (scoring/model_b_snapshot.py), never on output/: a pipeline
run refreshes output/ from the live DB, and the oracle is a fixed 2026-09-28 picture.
"""
import pytest

from analysis.model_b_validation import validate as V
from scoring import model_b_snapshot as SNAP


@pytest.fixture(scope="module")
def result():
    try:
        oracle, events, daily = V.load_inputs()
    except SNAP.SnapshotUnavailable as e:
        pytest.skip(str(e))
    events, daily, prod = V.production_signal(events, daily)
    return V.equivalence(oracle, prod), V.reference_identity(events, daily, oracle)


def test_same_events_cutoffs_counts_and_signal(result):
    eq, _ = result
    assert eq["events_oracle"] == eq["events_production"] == 46200
    assert eq["only_in_oracle"] == eq["only_in_production"] == 0
    assert eq["cutoff_mismatch"] == eq["n_prior_mismatch"] == eq["hist_null_mismatch"] == 0
    assert eq["hist_abs_diff_max"] == 0.0
    assert eq["research_sample_n"] == 21927
    assert eq["research_sample_rank_mismatch"] == 0


def test_reference_sample_is_the_candidate_fit_sample(result):
    _, ref = result
    assert ref["reference_n"] == ref["oracle_candidate_fit_n"] == 20491
    assert ref["only_reference"] == ref["only_oracle"] == 0
