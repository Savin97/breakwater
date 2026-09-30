"""Tests for the historical options pilot (`research/options_pilot/`).

They pin the ways the pilot could flatter itself: an options quote from after the call
cutoff reaching a feature, an expiry that finishes before the news being treated as
covering it, a straddle assembled from mismatched legs, a model learning from its own
test year, variants compared on different rows. Synthetic data only — no network, no
parquet, no database.
"""
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from feature_engineering.announcement_timing import AMC, BMO
from research.options_pilot import build as B
from research.options_pilot import evaluate as E
from research.options_pilot import features as F

GRID = pd.bdate_range("2024-01-01", "2024-12-31").to_numpy(dtype="datetime64[ns]")


def _leg(date, exp, strike, cp, bid, ask, vol, delta):
    return {"date": date, "act_symbol": "AAA", "expiration": exp, "strike": str(strike),
            "call_put": cp, "bid": None if bid is None else str(bid),
            "ask": None if ask is None else str(ask), "vol": str(vol), "delta": str(delta)}


def _chain(date="2024-03-08", exps=("2024-03-15", "2024-03-22"), extra=()):
    rows = []
    for e in exps:
        for k, dc in ((95, 0.70), (100, 0.52), (105, 0.33)):
            rows.append(_leg(date, e, k, "Call", 2.0 + (100 - k) / 5, 2.2 + (100 - k) / 5, 0.40, dc))
            rows.append(_leg(date, e, k, "Put", 2.0 - (100 - k) / 5, 2.2 - (100 - k) / 5, 0.45,
                             dc - 1))
        rows.append(_leg(date, e, 90, "Put", 0.5, 0.6, 0.55, -0.24))
    rows += list(extra)
    return F.normalise_chain(rows)


# ───────────────────────────── the call-cutoff invariant ─────────────────────────────────
def test_snapshot_never_after_call_cutoff():
    chain_dates = GRID[::2]                          # a sparse, irregular source
    cutoff = pd.Timestamp("2024-03-08")              # a Friday
    cands = B.candidate_snapshots(cutoff, chain_dates, GRID)
    assert cands and all(d <= cutoff for d, _ in cands)
    # a snapshot dated after the cutoff exists but must never be offered
    assert (chain_dates > np.datetime64(cutoff)).any()


def test_select_snapshot_walks_back_and_records_age():
    chain_dates = GRID
    cutoff = pd.Timestamp("2024-03-08")
    have = {"2024-03-05"}                            # symbol only present 3 sessions back
    fetch = lambda d, s: [{"x": 1}] if d in have else []
    got = B.select_snapshot(cutoff, "AAA", chain_dates, GRID, fetch)
    assert got["match_status"] == "matched"
    assert got["snapshot_date"] == pd.Timestamp("2024-03-05") and got["snapshot_age"] == 3


def test_select_snapshot_does_not_forward_fill_old_snapshots():
    fetch = lambda d, s: [{"x": 1}] if d == "2024-01-02" else []
    got = B.select_snapshot(pd.Timestamp("2024-03-08"), "AAA", GRID, GRID, fetch)
    assert got["match_status"].startswith("none_within")


def test_weekend_dated_snapshot_counts_from_its_previous_session():
    grid = GRID
    sat = np.array([np.datetime64("2024-03-09")])
    assert grid[B.effective_session(sat, grid)[0]] == np.datetime64("2024-03-08")
    # and a Saturday snapshot is NOT offered for a Friday cutoff (strict date rule)
    chain_dates = np.array(["2024-03-02", "2024-03-09"], dtype="datetime64[ns]")
    cands = B.candidate_snapshots(pd.Timestamp("2024-03-08"), chain_dates, grid)
    assert [d for d, _ in cands] == [pd.Timestamp("2024-03-02")]
    assert cands[0][1] == 5


def test_symbol_history_is_point_in_time():
    assert B.source_symbol("META", "2021-06-04") == "FB"
    assert B.source_symbol("META", "2022-06-09") == "META"
    assert B.source_symbol("BRK-B", "2021-06-04") == "BRK.B"


# ───────────────────────────── expiry coverage ───────────────────────────────────────────
def test_amc_same_day_expiry_does_not_cover():
    assert not F.covers_announcement("2024-03-15", "2024-03-15", AMC)


def test_bmo_same_day_expiry_covers_explicitly():
    assert F.covers_announcement("2024-03-15", "2024-03-15", BMO)


def test_expiry_before_report_never_covers():
    for w in (BMO, AMC):
        assert not F.covers_announcement("2024-03-14", "2024-03-15", w)
        assert F.covers_announcement("2024-03-18", "2024-03-15", w)


def test_near_expiry_skips_amc_same_day():
    ch = _chain(exps=("2024-03-15", "2024-03-22"))
    amc = F.snapshot_features(ch, "2024-03-15", AMC)
    bmo = F.snapshot_features(ch, "2024-03-15", BMO)
    assert amc["near_expiration"] == pd.Timestamp("2024-03-22")
    assert bmo["near_expiration"] == pd.Timestamp("2024-03-15")


# ───────────────────────────── straddle construction ─────────────────────────────────────
def test_straddle_legs_share_strike_and_expiry():
    # a call-only strike nearer ATM must not be paired with a put from another strike
    extra = [_leg("2024-03-08", "2024-03-15", 101, "Call", 1.9, 2.0, 0.4, 0.50)]
    f = F.snapshot_features(_chain(extra=extra), "2024-03-12", AMC)
    assert f["atm_strike"] == 100.0
    assert f["near_expiration"] == pd.Timestamp("2024-03-15")


def test_mid_uses_only_same_row_bid_ask():
    f = F.snapshot_features(_chain(), "2024-03-12", AMC)
    assert f["atm_call_mid"] == pytest.approx((f["atm_call_bid"] + f["atm_call_ask"]) / 2)
    assert f["atm_put_mid"] == pytest.approx((f["atm_put_bid"] + f["atm_put_ask"]) / 2)
    assert f["expected_move"] == pytest.approx((f["atm_call_mid"] + f["atm_put_mid"]) / 100)


def test_bad_quotes_are_filtered_and_counted():
    extra = [_leg("2024-03-08", "2024-03-15", 110, "Call", 1.0, 0.5, 0.4, 0.2),    # ask < bid
             _leg("2024-03-08", "2024-03-15", 111, "Call", None, 0.5, 0.4, 0.2),   # no bid
             _leg("2024-03-08", "2024-03-15", 112, "Put", 0.1, 0.2, 0.4, 0.3)]     # put delta > 0
    f = F.snapshot_features(_chain(extra=extra), "2024-03-12", AMC)
    assert f["removed_ask_ge_bid"] == 1 and f["removed_bid_ask_present"] == 1
    assert f["removed_delta_valid"] == 1


def test_no_atm_pair_when_deltas_are_far_from_half():
    rows = [_leg("2024-03-08", "2024-03-15", 80, "Call", 20, 21, 0.4, 0.95),
            _leg("2024-03-08", "2024-03-15", 80, "Put", 0.1, 0.2, 0.4, -0.05)]
    f = F.snapshot_features(F.normalise_chain(rows), "2024-03-12", AMC)
    assert f["status"] == "no_atm_pair"


def test_atm_pair_far_from_its_parity_spot_is_rejected():
    rows = [_leg("2024-03-08", "2024-03-15", 20, "Call", 1.05, 1.10, 3.07, 0.50),
            _leg("2024-03-08", "2024-03-15", 20, "Put", 4.75, 4.85, 3.07, -0.50)]
    f = F.snapshot_features(F.normalise_chain(rows), "2024-03-12", AMC)
    assert f["status"] == "atm_far_from_parity_spot" and "expected_move" not in f


def test_term_structure_uses_a_later_expiry():
    f = F.snapshot_features(_chain(), "2024-03-12", AMC)
    assert f["far_expiration"] > f["near_expiration"]


# ───────────────────────────── IV relative to own history ────────────────────────────────
def test_iv_history_uses_no_future_snapshot():
    d = pd.date_range("2023-01-01", "2024-06-30", freq="W-FRI")
    vh = pd.DataFrame({"date": d, "act_symbol": "AAA", "iv_current": 0.3})
    snap = pd.Timestamp("2024-03-08")
    a = B.iv_relative_history("AAA", snap, vh)
    vh2 = vh.copy()
    vh2.loc[vh2["date"] > snap, "iv_current"] = 9.0   # poison the future
    b = B.iv_relative_history("AAA", snap, vh2)
    assert a == b
    assert a["vh_prior_max_date"] < snap


# ───────────────────────────── evaluation rules ──────────────────────────────────────────
def _toy_panel(n_per_year=200, years=range(2020, 2025), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        for i in range(n_per_year):
            d = pd.Timestamp(year=y, month=1 + i % 12, day=10)
            rows.append({"event_id": f"S{i % 40}|{d.date()}|{i}", "stock": f"S{i % 40}",
                         "year": y, "endpoint3_date": d + pd.Timedelta(days=5),
                         "announce_window": BMO if i % 2 else AMC,
                         "log_hist_mean_abs": rng.normal(), "log_vol_30d": rng.normal(),
                         "log_expected_move": rng.normal(), "log_atm_iv": rng.normal(),
                         "y_extreme": float(rng.random() < 0.2)})
    return pd.DataFrame(rows)


def test_training_year_strictly_before_test_year():
    panel = _toy_panel()
    preds, folds = E.walk_forward(panel, ["log_hist_mean_abs", "log_vol_30d"], range(2022, 2025))
    for y, fold in folds.items():
        assert fold["train_max_endpoint"] < pd.Timestamp(year=y, month=1, day=1)
    assert set(preds["test_year"]) == {2022, 2023, 2024}


def test_corrected_target_only():
    src = Path("research/options_pilot/evaluate.py").read_text()
    build = Path("research/options_pilot/build.py").read_text()
    for s in (src, build):
        # the legacy target never appears except as the anchored column's prefix
        assert "abs_reaction_3d\"" not in s and "abs_reaction_3d'" not in s
    assert E.TARGET == "y_extreme"


def test_population_requires_resolved_endpoint():
    src = Path("research/options_pilot/build.py").read_text()
    assert 'f["endpoint3_idx"].ge(0)' in src and 'f["y_extreme"].notna()' in src


def test_common_sample_comparisons_use_identical_event_ids():
    panel = _toy_panel()
    res = E.compare_on_common_sample(panel, {"C": ["log_hist_mean_abs", "log_vol_30d"],
                                             "C+em": ["log_hist_mean_abs", "log_vol_30d",
                                                      "log_expected_move"]},
                                     range(2022, 2025))
    ids = [set(p["event_id"]) for p in res.values()]
    assert all(i == ids[0] for i in ids)


def test_no_raw_vendor_data_tracked_by_git():
    tracked = subprocess.run(["git", "ls-files", "data/vendor", "output/options_pilot"],
                             capture_output=True, text=True, check=True).stdout.strip()
    assert tracked == ""
    ignored = subprocess.run(["git", "check-ignore", "-q",
                              "data/vendor/dolthub_options/x/q/abc.json.gz"])
    assert ignored.returncode == 0


def test_no_production_module_imports_the_pilot():
    for d in ("pipeline", "scoring", "ingestion", "feature_engineering", "report",
              "streamlit_dash", "analysis", "cron", "utilities"):
        for p in Path(d).rglob("*.py"):
            t = p.read_text(errors="ignore")
            assert "options_pilot" not in t and "dolthub" not in t.lower(), p


# ───────────────────────────── strict-after robustness rule ──────────────────────────────
def test_strict_after_rule_rejects_bmo_same_day_expiry():
    snap = pd.Timestamp("2024-03-08")
    assert F.is_covering("2024-03-15", snap, "2024-03-15", BMO, F.RULE_ANNOUNCEMENT)
    assert not F.is_covering("2024-03-15", snap, "2024-03-15", BMO, F.RULE_STRICT_AFTER)
    assert F.is_covering("2024-03-18", snap, "2024-03-15", AMC, F.RULE_STRICT_AFTER)


def test_strict_after_variant_is_computed_separately():
    ch = _chain(exps=("2024-03-15", "2024-03-22"))
    f = F.snapshot_features(ch, "2024-03-15", BMO)
    assert f["near_expiration"] == pd.Timestamp("2024-03-15")          # announcement-aware
    assert f["sa_near_expiration"] == pd.Timestamp("2024-03-22")       # strict-after


def test_expiry_counts_and_second_expiry():
    f = F.snapshot_features(_chain(exps=("2024-03-15", "2024-03-22")), "2024-03-12", AMC)
    assert f["n_covering_expiries"] == 2 and f["n_usable_covering_expiries"] == 2
    assert f["has_second_usable_expiry"]
    g = F.snapshot_features(_chain(exps=("2024-03-15",)), "2024-03-12", AMC)
    assert not g["has_second_usable_expiry"] and "term_ratio" not in g


def test_expiry_before_snapshot_never_covers():
    assert not F.is_covering("2024-03-07", "2024-03-08", "2024-03-06", AMC, F.RULE_ANNOUNCEMENT)


# ───────────────────────────── selection vs 2026 holdout ─────────────────────────────────
def test_2026_excluded_from_selection():
    panel = _toy_panel(years=range(2020, 2027))
    sel = E.selection_panel(panel)
    assert sel["year"].max() < E.HOLDOUT_YEAR
    assert E.LAST_TEST_YEAR < E.HOLDOUT_YEAR


def test_holdout_refuses_without_frozen_winner(tmp_path, monkeypatch):
    monkeypatch.setattr(E, "OUT", tmp_path)
    with pytest.raises(FileNotFoundError):
        E.holdout(_toy_panel(years=range(2020, 2027)), "strict")


def test_blocks_record_identical_rows_for_every_model():
    panel = _toy_panel()
    r, preds = E._block(panel, {"C": ["log_hist_mean_abs", "log_vol_30d"],
                                "C+iv": ["log_hist_mean_abs", "log_vol_30d", "log_atm_iv"]},
                        range(2022, 2025), reps=5, boot=True)
    assert np.array_equal(preds["C"]["event_id"], preds["C+iv"]["event_id"])
    assert "C+iv_vs_C" in r["deltas"] and r["n_oof"] == len(preds["C"])


def test_iv_gate_fails_without_volatility_history():
    panel = _toy_panel().assign(primary_ok=True, snapshot_age=0)
    g = E.iv_history_gate(panel, 1)
    assert not g["pass"]
    out = E.apply_iv_gate(panel.assign(log_iv_rel_hist=1.0), g)
    assert out["log_iv_rel_hist"].isna().all()
