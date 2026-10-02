"""Tests for the SEC feature test (`research/sec_features/`). Synthetic fixtures; no network."""
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.sec_features import build, evaluate, features as F
from research.sec_filings_pilot import paths as sp, timing

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "research" / "sec_features"
T = pd.Timestamp


def _strings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
        {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}


# ───────────────────────────────────────── inputs ───────────────────────────────────────
def test_current_corrected_event_frame_is_used():
    assert build.FRAME == Path("output/sec_features/feature_frame_current.parquet")
    assert sp.FEATURE_FRAME.name == "feature_frame_current.parquet"
    assert evaluate.OUT == Path("output/sec_features")
    src = (PKG / "frame.py").read_text()
    assert "build_phase3_events" in src and "build_feature_frame" in src


def test_no_target_enters_feature_construction():
    forbidden = {"y_extreme", "abs_r3", "abs_r1", "abs_reaction_3d_anchored", "abs_reaction_3d",
                 "reaction_3d_anchored", "reaction_1d_anchored", "is_extreme_reaction"}
    for name in ("build.py", "features.py", "audit.py"):
        f = PKG / name
        assert not forbidden & set(_strings(f)), name
        assert not forbidden & _names(f), name
    assert not forbidden & set(build.EVENT_COLS)


def test_sec_json_acceptance_time_is_never_used():
    for f in PKG.glob("*.py"):
        text = f.read_text()
        assert "acceptanceDateTime" not in text and "acceptance_json" not in text, f


# ──────────────────────────────────────── releases ──────────────────────────────────────
def _frame(dates_ann):
    rows = [{"event_id": f"S|{d}", "stock": "S", "ann_date": T(d), "report_date": T(d),
             "earnings_date": T(d)} for d in dates_ann]
    return pd.DataFrame(rows)


def _sf(rows):
    df = pd.DataFrame(rows, columns=["accession", "form", "filing_date", "items"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["stock"], df["cik"] = "S", 1
    df["base_form"] = df["form"].str.removesuffix("/A")
    return df


def _target(event_id, cutoff):
    cd = T(cutoff)
    return pd.DataFrame([{"event_id": event_id, "stock": "S", "call_cutoff_date": cd,
                          "call_cutoff_ts": timing.cutoff_instant(pd.Series([cd])).iloc[0]}])


def _prev(fe, sf, target, verified=None):
    rel = build.release_8k_per_event(fe, sf)
    return build.previous_releases(fe, rel, target, verified or {})


def test_own_release_is_never_the_previous_release():
    fe = _frame(["2024-01-25", "2024-04-25", "2024-07-25"])
    sf = _sf([("r1", "8-K", "2024-01-25", "2.02,9.01"), ("r2", "8-K", "2024-04-25", "2.02,9.01"),
              ("r3", "8-K", "2024-07-25", "2.02,9.01")])
    out = _prev(fe, sf, _target("S|2024-07-25", "2024-07-19")).iloc[0]
    assert out["prev_accession"] == "r2" and out["prev2_accession"] == "r1"
    assert "r3" not in set(out[["prev_accession", "prev2_accession"]])
    assert out["prev_filing_date"] < T("2024-07-19")


def test_previous_release_filed_after_cutoff_is_not_eligible():
    fe = _frame(["2024-04-25", "2024-07-25"])
    sf = _sf([("late", "8-K", "2024-07-22", "2.02")])        # filed after the 07-19 cutoff
    fe.loc[0, "ann_date"] = T("2024-07-20")                   # (a mis-dated previous event)
    out = _prev(fe, sf, _target("S|2024-07-25", "2024-07-19")).iloc[0]
    assert out["prev_status"] == "not_eligible"


def test_cutoff_day_release_needs_verified_time():
    fe = _frame(["2024-07-19", "2024-07-29"])
    sf = _sf([("x", "8-K", "2024-07-19", "2.02")])
    t = _target("S|2024-07-29", "2024-07-19")
    assert _prev(fe, sf, t).iloc[0]["prev_status"] == "not_eligible"
    ok = _prev(fe, sf, t, {"x": T("2024-07-19 08:00")}).iloc[0]
    late = _prev(fe, sf, t, {"x": T("2024-07-19 16:05")}).iloc[0]
    assert ok["prev_status"] == "ok" and late["prev_status"] == "not_eligible"


def test_second_prior_must_precede_first_and_stale_release_is_missing():
    fe = _frame(["2023-01-25", "2024-04-25", "2024-07-25"])
    sf = _sf([("a", "8-K", "2023-01-25", "2.02"), ("b", "8-K", "2024-04-25", "2.02")])
    out = _prev(fe, sf, _target("S|2024-07-25", "2024-07-19")).iloc[0]
    assert out["prev_status"] == "ok" and out["prev2_status"] == "too_old"
    fe2 = _frame(["2024-01-25", "2024-07-25"])
    sf2 = _sf([("a", "8-K", "2024-01-25", "2.02")])
    out2 = _prev(fe2, sf2, _target("S|2024-07-25", "2024-07-19")).iloc[0]
    assert out2["prev_status"] == "too_old"                 # 176 days > 140


def test_release_8k_needs_item_2_02_original_and_window():
    fe = _frame(["2024-04-25"])
    sf = _sf([("amend", "8-K/A", "2024-04-25", "2.02"), ("other", "8-K", "2024-04-25", "7.01"),
              ("far", "8-K", "2024-04-30", "2.02"), ("good", "8-K", "2024-04-26", "2.02,9.01")])
    rel = build.release_8k_per_event(fe, sf)
    assert list(rel["accession"]) == ["good"]


# ────────────────────────────────────────── 8-K ─────────────────────────────────────────
def test_8k_features_interval_and_recent_window():
    inset = pd.DataFrame({"form": ["8-K", "8-K/A", "8-K"], "items": ["2.02,9.01", "9.01", "5.02"],
                          "filing_date": pd.to_datetime(["2024-05-01", "2024-06-01", "2024-07-10"])})
    elig = pd.DataFrame({"form": ["8-K", "8-K", "8-K", "8-K"],
                         "items": ["2.02,9.01", "5.02", "7.01", "8.01"],
                         "filing_date": pd.to_datetime(["2024-05-01", "2024-07-10",
                                                        "2024-07-06", "2024-06-21"])})
    f = F.eightk_features(inset, elig, T("2024-07-19"))
    assert f["num_8k_since_periodic"] == 2                  # originals only
    assert f["has_2_02"] == 1 and f["has_5_02"] == 1 and f["has_7_01"] == 0
    assert f["num_material_item_types"] == 2                # 9.01 excluded
    assert f["num_8k_last_30d"] == 3                        # 06-21 is day 28
    assert f["has_recent_7_01_14d"] == 1 and f["has_recent_8_01_14d"] == 0  # 06-21 = 28 d
    assert f["has_recent_2_02_14d"] == 0


# ───────────────────────────────────────── text ─────────────────────────────────────────
REL = """<html><body>
<p>Acme Corp reports fourth quarter results. Revenue was $1.2 billion, up 5%, in a challenging and volatile environment for the industry this year.</p>
<p>Outlook: For fiscal 2025, the company expects revenue of $4.0 billion to $4.4 billion and diluted EPS of $2.10 to $2.30.</p>
<p>This release contains forward-looking statements that involve risks and uncertainties and are subject to the safe harbor provisions.</p>
</body></html>"""


def test_boilerplate_dropped_and_guidance_detected_with_widths():
    full, prose = F.release_text(REL)
    assert "safe harbor" not in full and "uncertainties" not in prose
    f = F.release_features(REL)
    assert f["guidance_present"] == 1
    assert f["eps_guidance_width"] == pytest.approx(0.2 / 2.2)
    assert f["revenue_guidance_width"] == pytest.approx(0.4 / 4.2)
    assert f["uncertainty_hits"] == 2                         # challenging, volatile


def test_no_guidance_without_forward_period_and_number():
    gs = F.guidance_sentences("Results were better than expected for the company overall.")
    assert gs == []


def test_cosine_distance_is_pairwise_and_bounded():
    a, b = Counter({"revenue": 2, "growth": 1}), Counter({"revenue": 2, "growth": 1})
    assert F.cosine_distance(a, b) == pytest.approx(0)
    assert F.cosine_distance(a, Counter({"loss": 3})) == pytest.approx(1)
    assert np.isnan(F.cosine_distance(a, Counter()))


# ───────────────────────────────────────── evaluation ───────────────────────────────────
def _panel(n_per_year=120, years=range(2014, 2027), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        for i in range(n_per_year):
            x = rng.normal(size=3)
            rows.append({"event_id": f"S{i % 40}|{y}-{i}", "stock": f"S{i % 40}", "year": y,
                         "announce_window": "BMO" if i % 2 else "AMC",
                         "endpoint3_date": T(f"{y}-06-01"),
                         "y_extreme": int(rng.random() < 1 / (1 + np.exp(-x[0]))),
                         "log_hist_mean_abs": x[0], "log_vol_30d": x[1], "z": x[2]})
    return pd.DataFrame(rows)


def test_walk_forward_trains_only_on_earlier_years_and_pairs_identical_rows():
    p = _panel()
    p.loc[p.index[::7], "z"] = np.nan
    from research.options_pilot.evaluate import compare_on_common_sample, walk_forward
    _, folds = walk_forward(p.dropna(), ["log_hist_mean_abs", "z"], [2017, 2018])
    for y, f in folds.items():
        assert f["train_max_endpoint"] < T(year=y, month=1, day=1)
    preds = compare_on_common_sample(p, {"C": ["log_hist_mean_abs", "log_vol_30d"],
                                         "C+z": ["log_hist_mean_abs", "log_vol_30d", "z"]},
                                     [2017, 2018])
    assert preds["C"]["event_id"].tolist() == preds["C+z"]["event_id"].tolist()
    assert not set(preds["C"]["event_id"]) & set(p.loc[p["z"].isna(), "event_id"])


def test_2026_is_excluded_from_selection(monkeypatch, tmp_path):
    p = _panel(n_per_year=80)
    for c in ["guidance_present", "uncertainty_rate", "uncertainty_change", "release_text_change",
              "l_uncertainty_rate", "l_num_8k_last_30d", "has_recent_2_02_14d"] + evaluate.K8:
        p[c] = np.random.default_rng(1).random(len(p))
    monkeypatch.setattr(evaluate, "panel", lambda: p.copy())
    monkeypatch.setattr(evaluate, "OUT", tmp_path)
    (tmp_path / "gates.json").write_text(json.dumps(
        {"release_ok": True, "guidance_present_ok": True, "uncertainty_ok": True, "width_ok": False}))
    out = evaluate.run_selection(reps=2)
    oof = pd.read_parquet(tmp_path / "oof_main.parquet")
    assert oof["test_year"].max() == 2025 and 2026 not in set(oof["test_year"])
    assert out["holdout_models"][0] == "C+compact"
