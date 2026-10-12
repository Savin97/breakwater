"""The maintainer evaluation tools run on Model B data and measure the right thing.

testing/testing_functions.py (evaluation helpers) and testing/metric_testing.py (the
metrics script) are not production code, but a script that silently breaks is worse than
none. Every helper and every metrics section is run here on a synthetic event frame where
the Model B signal genuinely predicts the corrected move, and the invariants that make
the numbers meaningful are pinned: unresolved events are dropped (never misses), the
legacy target is never used, thresholds come from training years only.

Run with:  pytest testing/test_evaluation_tools.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

import testing.testing_functions as T
import testing.metric_testing as MT
from scoring import model_b as B


@pytest.fixture(scope="module")
def events():
    """60 stocks x quarterly events 2013-2025: each stock has a 'volatility level' that
    drives both its Model B signal and its moves, plus a few pending and unresolved rows."""
    rng = np.random.default_rng(11)
    rows = []
    for s in range(60):
        level = rng.uniform(0.02, 0.09)
        for k, d in enumerate(pd.date_range("2013-01-08", "2025-12-16", freq="13W-TUE")):
            move = abs(rng.normal(0, level))
            unresolved = rng.random() < 0.08
            rows.append({
                "stock": f"S{s:02d}", "earnings_date": d, "is_pending": False,
                "is_earnings_day": 1,
                "announce_window": "BMO" if rng.random() < 0.55 else "AMC",
                "reaction_3d_anchored": np.nan if unresolved else move * rng.choice([-1, 1]),
                "abs_reaction_3d_anchored": np.nan if unresolved else move,
                "reaction_3d_anchored_status": "unresolved_no_timestamp" if unresolved else "available",
                "abs_reaction_3d": 0.99,                              # legacy: must be ignored
                "model_b_hist_mean_abs": level * rng.uniform(0.8, 1.2),
                "model_b_n_prior": 8 + k,
                "vol_30d": level / 3 * rng.uniform(0.7, 1.3),
                "vol_10d": level / 3 * rng.uniform(0.6, 1.4),
                "vol_ratio_10_to_30": rng.uniform(0.6, 1.6),
            })
        rows.append({**rows[-1], "earnings_date": pd.Timestamp("2026-10-07"), "is_pending": True,
                     "is_earnings_day": 0, "abs_reaction_3d_anchored": np.nan,
                     "reaction_3d_anchored": np.nan, "reaction_3d_anchored_status": "pending"})
    ev = pd.DataFrame(rows)
    ev = B.engineer_risk_tier(B.engineer_risk_score(ev))
    return ev


# ── shared preparation ────────────────────────────────────────────────────────

def test_prepare_drops_pending_and_unresolved_and_ignores_the_legacy_target(events):
    ev = T.prepare_events(events)
    assert not ev["is_pending"].any()
    assert ev["reaction_3d_anchored_status"].eq("available").all()
    assert len(ev) == int((~events["is_pending"]
                          & events["reaction_3d_anchored_status"].eq("available")).sum())
    assert (ev["y"] == (ev["abs_reaction_3d_anchored"] >= 0.08)).all()   # not the 0.99 legacy
    assert ev["y"].mean() < 0.5


# ── testing_functions: every helper runs and measures something sensible ──────

def test_feature_diagnostics(events):
    rates = T.check_explosiveness_feature(events)
    assert rates.iloc[-1] > rates.iloc[0]
    table = T.check_feature_connection_to_large_reacion_metric(events)
    assert len(table) == 10 and table["p_extreme"].iloc[-1] > table["p_extreme"].iloc[0]
    assert (table["p_large_plus_ci_lo"] <= table["p_large_plus"]).all()
    corr = T.check_corr_of_features(events, cols=("model_b_hist_mean_abs", "vol_30d", "missing"))
    assert "missing" not in corr.columns and corr.loc["model_b_hist_mean_abs", "move"] > 0
    stats = T.check_score_metric(events)
    assert stats["min"] == 0 and stats["max"] == 100


def test_regime_tests(events):
    out = T.three_way_regime_test(events)
    assert list(out["group"])[0] == "ALL" and out["n"].iloc[2] <= out["n"].iloc[1]
    out = T.conditional_hit_rate_analysis(events)
    assert len(out) == 5 and out["p_extreme"].iloc[1] > out["p_extreme"].iloc[0]
    res = T.breakwater_regime_test(events)
    assert res["n_events"] > 0 and res["extreme_rate"] > res["baseline_extreme"]
    out = T.volatility_only_regime_test(events)
    assert out["capture_extreme"].between(0, 1).all() and out["capture_extreme"].iloc[0] == 1
    grid = T.evaluate_high_risk_earnings_regime(events)
    assert {"score_q", "other_q", "n", "extreme_rate"} <= set(grid.columns)
    summary = T.high_conviction_regime_test(events, overlays={"high vol": [("vol_30d", 0.8)]})
    assert {"High Alert", "High Alert + high vol"} <= set(summary.index)
    assert summary.loc["High Alert + high vol", "total_n"] <= summary.loc["High Alert", "total_n"]


def test_classification_metrics(events):
    ev = T.prepare_events(events)
    res = T.testing_precision_recall(events)
    assert res["TP"] + res["FN"] + res["FP"] + res["TN"] == len(ev)
    assert res["flagged"] == int((ev["risk_tier"] == "High Alert").sum())
    res = T.comparing_regime_results_to_volatility_only(events)
    assert abs(res["shuffled_precision"] - res["baseline"]) < 0.1     # the no-skill null
    assert res["precision"] > res["shuffled_precision"]


def test_out_of_sample_tools(events):
    out = T.check_feature_train_test(events)
    assert out["test_spearman"] > 0
    yearly = T.yearly_oos_report(events)
    assert len(yearly) > 5 and (yearly["spearman"] > 0).all()


def test_forward_eval_thresholds_come_from_train_years_only(events):
    ev = T.prepare_events(events)
    res, thr = T.forward_eval_onefactor(events)
    train = ev[ev["year"].isin(T.TRAIN_YEARS)]
    assert thr == pytest.approx(train["model_b_hist_mean_abs"].quantile(0.9))
    assert set(res["split"]) == {"TRAIN", "TEST"}
    assert res.loc[res["split"] == "TEST", "year"].min() == 2017
    res2, (t1, t2) = T.forward_eval_twofactor(events)
    assert t1 == thr and (res2["n_regime"] <= res.set_index(["split", "year"])
                          .loc[list(zip(res2["split"], res2["year"])), "n_regime"].values).all()


# ── metric_testing: every section runs ────────────────────────────────────────

def test_every_metrics_section_runs(events, tmp_path, monkeypatch, capsys):
    path = tmp_path / "events.parquet"
    events.to_parquet(path)
    monkeypatch.setattr(MT, "EVENTS_PARQUET", str(path))
    orig = MT.load_events
    monkeypatch.setattr(MT, "load_events", lambda path=None, **k: orig(str(tmp_path / "events.parquet"), **k))
    assert MT.main(["--feature", "vol_30d"]) == 0
    out = capsys.readouterr().out
    for title, *_ in MT.SECTIONS.values():
        assert f"=== {title} ===" in out
    assert "INTERNAL" in out and "ROC-AUC" in out and "ECE:" in out
    assert "FEATURE vs TARGET: vol_30d" in out


def test_metrics_rank_on_the_raw_signal(events):
    s = T.prepare_events(events)
    s = s.assign(move=s["abs_reaction_3d_anchored"], call_week=s["earnings_date"])
    hit, cap = MT._top(s["y"].to_numpy(), s["model_b_hist_mean_abs"].to_numpy(), 0.10)
    order = B.sort_by_risk(s).head(int(np.ceil(0.10 * len(s))))
    assert hit == pytest.approx(order["y"].mean())


# ── calibration.py and backtesting.py (migrated to Model B) ───────────────────

def test_calibration_runs_on_model_b_and_its_tiers_never_overlap(events, tmp_path, capsys):
    import testing.calibration as C
    ev = events.copy()
    ev.loc[ev.index[::7], "risk_tier"] = "Unscored"          # coverage row must appear
    ev.loc[ev["risk_tier"] == "Unscored", "risk_score"] = np.nan
    path = tmp_path / "events.parquet"
    ev.to_parquet(path)
    tables = C.run(str(path), str(tmp_path / "results"))
    out = capsys.readouterr().out
    assert "INTERNAL" in out and "Lift" not in out
    by = tables["by_bucket"].set_index("Bucket")
    assert {"ALL scored (baseline)", "Normal", "Elevated", "High Alert", "Unscored (coverage)"} <= set(by.index)
    n_scored = int(((ev["risk_tier"] != "Unscored") & ~ev["is_pending"]
                    & ev["reaction_3d_anchored_status"].eq("available")
                    & pd.to_datetime(ev["earnings_date"]).dt.year.between(2017, 2025)).sum())
    assert by.at["ALL scored (baseline)", "N Events"] == n_scored
    overlaps = tables["consistency"][tables["consistency"]["Tier"].str.startswith("OVERLAP")]
    assert len(overlaps) and (overlaps["N"] == 0).all()
    assert len(list((tmp_path / "results").glob("calibration_*.csv"))) == 9


def test_backtesting_runs_on_model_b(events, capsys):
    import testing.backtesting as BT
    BT.backtesting_suite(events)
    out = capsys.readouterr().out
    assert "threshold from 2013-2016" in out and "High Alert + vol_30d top 20%" in out
    assert "Avg spearman" in out
