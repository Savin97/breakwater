"""Leakage and protocol checks for the Model C production-frame refit (SPEC.md).

Run:  PYTHONPATH=. .venv/bin/python -m pytest analysis/model_c_refit/tests -q
Kept out of `testing/` on purpose: this is a one-time validation, not production code.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from analysis.model_c_refit import frame as F
from analysis.model_c_refit import model as M


# ─────────────────────────────── synthetic production-shaped data ──────────────────────
def _synthetic(n_stocks=24, start="2010-01-04", end="2026-09-30", seed=7):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    daily, events = [], []
    for s in range(n_stocks):
        stock = f"S{s:02d}"
        vol = rng.uniform(0.008, 0.03)
        daily.append(pd.DataFrame({"stock": stock, "date": days,
                                   "vol_30d": vol * rng.uniform(0.8, 1.2, len(days))}))
        level = rng.uniform(0.02, 0.09)
        # one report roughly every quarter, on a random weekday
        for q in range(5, len(days) - 10, 63):   # room for a BMO anchor and the 3-session endpoint
            d = days[q + int(rng.integers(0, 5))]
            window = "AMC" if rng.random() < 0.5 else "BMO"
            gi = days.get_loc(d)
            anchor = d if window == "AMC" else days[gi - 1]
            r = abs(rng.normal(0, level + 2 * vol))
            events.append({"stock": stock, "earnings_date": d, "is_pending": False,
                           "announce_window": window, "anchor_date": anchor,
                           "anchor_status": "resolved",
                           "reaction_3d_anchored_status": "available",
                           "abs_reaction_3d_anchored": r,
                           "abs_reaction_3d": r * 0.3 if window == "BMO" else r,
                           "risk_score": 100 * min(level / 0.12, 1)})
    ev = pd.DataFrame(events).sort_values(["stock", "earnings_date"]).reset_index(drop=True)
    return ev, pd.concat(daily, ignore_index=True)


@pytest.fixture(scope="module")
def synth():
    ev, daily = _synthetic()
    return ev, daily, F.build_frame(ev, daily)


def _brute_hist(frame, i):
    """Recount event i's history directly from the definition."""
    r = frame.loc[i]
    prior = frame[(frame["stock"] == r["stock"]) & frame["y"].notna()
                  & (frame["endpoint_idx"] <= r["cutoff_idx"]) & (frame.index != i)]
    return len(prior), prior["abs_r3"].mean() if len(prior) else np.nan


# ──────────────────────────────────────────── history ──────────────────────────────────
def test_event_cannot_see_its_own_reaction(synth):
    ev, daily, frame = synth
    lab = frame[frame["y"].notna()]
    assert (lab["endpoint_idx"] > lab["cutoff_idx"]).all()
    # perturbing an event's own outcome must not move its own feature
    i = lab[lab["n_prior"] >= 8].index[5]
    ev2 = ev.copy()
    ev2.loc[i, "abs_reaction_3d_anchored"] = 9.99
    f2 = F.build_frame(ev2, daily)
    assert f2.loc[i, "hist_mean_abs"] == pytest.approx(frame.loc[i, "hist_mean_abs"])


def test_history_matches_the_definition_exactly(synth):
    _, _, frame = synth
    for i in frame[frame["y"].notna()].sample(60, random_state=1).index:
        n, m = _brute_hist(frame, i)
        assert frame.loc[i, "n_prior"] == n
        if n:
            assert frame.loc[i, "hist_mean_abs"] == pytest.approx(m)


def test_no_outcome_enters_history_before_its_endpoint():
    """Two reports of one stock in consecutive weeks: the first one's 3-session endpoint
    falls after the second one's call cutoff, so it must not count."""
    days = pd.bdate_range("2020-01-01", "2020-03-31")
    daily = pd.DataFrame({"stock": "X", "date": days, "vol_30d": 0.02})
    base = dict(stock="X", is_pending=False, announce_window="AMC", anchor_status="resolved",
                reaction_3d_anchored_status="available", risk_score=50.0)
    ev = pd.DataFrame([
        {**base, "earnings_date": pd.Timestamp("2020-02-06"), "anchor_date": pd.Timestamp("2020-02-06"),
         "abs_reaction_3d_anchored": 0.5},                       # Thu; endpoint Tue 2020-02-11
        {**base, "earnings_date": pd.Timestamp("2020-02-12"), "anchor_date": pd.Timestamp("2020-02-12"),
         "abs_reaction_3d_anchored": 0.01},                      # cutoff = Fri 2020-02-07
        {**base, "earnings_date": pd.Timestamp("2020-02-19"), "anchor_date": pd.Timestamp("2020-02-19"),
         "abs_reaction_3d_anchored": 0.02},                      # cutoff = Fri 2020-02-14
    ])
    f = F.build_frame(ev, daily)
    assert f.loc[1, "n_prior"] == 0                              # endpoint 02-11 > cutoff 02-07
    assert f.loc[2, "n_prior"] == 1                              # only the first is known by 02-14
    assert f.loc[2, "hist_mean_abs"] == pytest.approx(0.5)


# ──────────────────────────────────────────── clocks ───────────────────────────────────
def test_call_cutoff_is_the_last_session_before_the_report_week():
    days = pd.bdate_range("2024-01-01", "2024-02-29")
    grid = days.drop(pd.Timestamp("2024-01-12")).to_numpy(dtype="datetime64[ns]")  # a "holiday" Friday
    idx = F.call_cutoff_index(grid, pd.to_datetime(["2024-01-17", "2024-01-22", "2024-01-15"]))
    got = pd.to_datetime(grid[idx])
    assert list(got) == [pd.Timestamp("2024-01-11"),             # Wed of week of 01-15 → Thu 01-11
                         pd.Timestamp("2024-01-19"),             # Monday report → previous Friday
                         pd.Timestamp("2024-01-11")]


def test_cutoff_precedes_report_week_on_every_row(synth):
    _, _, frame = synth
    assert (frame["cutoff_date"] < frame["call_monday"]).all()
    assert (frame["call_monday"] <= frame["report_date"]).all()


# ──────────────────────────────────────────── vol_30d ──────────────────────────────────
def test_vol_is_read_on_or_before_the_cutoff(synth):
    ev, daily, frame = synth
    has = frame["vol_date"].notna()
    assert (frame.loc[has, "vol_date"] <= frame.loc[has, "cutoff_date"]).all()
    # for any event, a wildly different value on its stock's rows AFTER its cutoff must
    # not change the value it reads
    clocks = F.event_clocks(ev, F.market_session_grid(daily))
    for i in frame.sample(40, random_state=5).index:
        d2 = daily.copy()
        after = (d2["stock"] == frame.loc[i, "stock"]) & (d2["date"] > frame.loc[i, "cutoff_date"])
        d2.loc[after, "vol_30d"] = 99.0
        got = F.vol_at_cutoff(d2, ev, clocks).loc[i, "vol_30d"]
        assert got == pytest.approx(frame.loc[i, "vol_30d"], nan_ok=True)


def test_vol_fallback_never_reaches_past_the_tolerance():
    days = pd.bdate_range("2021-01-01", "2021-03-31")
    daily = pd.DataFrame({"stock": "X", "date": days, "vol_30d": np.arange(len(days), dtype=float)})
    daily = daily[(daily["date"] < "2021-02-01") | (daily["date"] > "2021-02-26")]  # 4-week hole
    ev = pd.DataFrame([{"stock": "X", "earnings_date": pd.Timestamp("2021-02-24"), "is_pending": False,
                        "announce_window": "AMC", "anchor_date": pd.Timestamp("2021-02-24"),
                        "anchor_status": "resolved", "reaction_3d_anchored_status": "available",
                        "abs_reaction_3d_anchored": 0.1, "risk_score": 1.0}])
    grid_daily = pd.DataFrame({"stock": "Y", "date": days, "vol_30d": 0.01})   # full market grid
    f = F.build_frame(ev, pd.concat([daily, grid_daily]))
    assert np.isnan(f.loc[0, "vol_30d"])                          # last row is > 7 days back


# ──────────────────────────────────────────── target ───────────────────────────────────
def test_target_is_always_the_anchored_target(synth):
    _, _, frame = synth
    lab = frame["y"].notna()
    exp = (frame.loc[lab, "abs_r3"] >= 0.08).astype(float)
    assert frame.loc[lab, "y"].equals(exp)
    ev, daily, _ = synth
    # the legacy column disagrees on BMO rows by construction; it must not matter
    assert not np.allclose(ev["abs_reaction_3d"], ev["abs_reaction_3d_anchored"])


def test_unresolved_target_stays_missing():
    ev, daily = _synthetic(n_stocks=3)
    ev.loc[[3, 4], "reaction_3d_anchored_status"] = "unresolved_no_timestamp"
    ev.loc[[3, 4], "abs_reaction_3d_anchored"] = np.nan
    ev.loc[5, "reaction_3d_anchored_status"] = "unavailable_endpoint_price_gap"
    f = F.build_frame(ev, daily)
    assert f.loc[[3, 4, 5], "y"].isna().all()
    assert f.loc[[3, 4, 5], "abs_r3"].isna().all()
    later = f[(f["stock"] == ev.loc[3, "stock"]) & (f.index > 5)].index[0]
    n, _ = _brute_hist(f, later)
    assert f.loc[later, "n_prior"] == n                           # the missing ones are not counted
    assert not F.population_mask(f).loc[[3, 4, 5]].any()


# ─────────────────────────────────────── protocol ──────────────────────────────────────
def test_b_and_c_are_scored_on_identical_rows(synth):
    from analysis.model_c_refit.evaluate import run_models
    _, _, frame = synth
    sample = frame[F.common_mask(frame) & (frame["year"] < 2026)]
    preds = run_models(sample)
    assert preds["p_B"].notna().equals(preds["p_C"].notna())
    assert preds["p_B"].notna().all()


def test_2026_cannot_influence_the_pre2026_results(synth):
    from analysis.model_c_refit.evaluate import _pre2026, run_models
    _, _, frame = synth
    base = run_models(_pre2026(frame)[F.common_mask(_pre2026(frame))])
    f2 = frame.copy()
    m26 = f2["year"] == 2026
    assert m26.any()
    f2.loc[m26, "y"] = 1 - f2.loc[m26, "y"]                       # scramble every 2026 label
    f2.loc[m26, "log_vol_30d"] = 5.0
    alt = run_models(_pre2026(f2)[F.common_mask(_pre2026(f2))])
    pd.testing.assert_frame_equal(base, alt)


def test_training_rows_end_before_each_test_year(synth):
    _, _, frame = synth
    sample = frame[F.common_mask(frame)]
    for year in M.TEST_YEARS:
        tr = sample[M.training_mask(sample, year)]
        assert (tr["endpoint_date"] < pd.Timestamp(year=year, month=1, day=1)).all()
        assert (tr["year"] < year).all()


def test_final_fit_excludes_2026(synth):
    from analysis.model_c_refit.evaluate import final_fits
    _, _, frame = synth
    train, fits = final_fits(frame)
    assert (train["year"] < 2026).all() and (train["endpoint_date"] < "2026-01-01").all()
    assert fits["C"].train_n == len(train)


def test_event_order_does_not_change_anything(synth):
    from analysis.model_c_refit.evaluate import run_models
    ev, daily, frame = synth
    shuffled = ev.sample(frac=1.0, random_state=3)
    f2 = F.build_frame(shuffled, daily).loc[frame.index]
    for c in ["n_prior", "hist_mean_abs", "vol_30d", "y", "cutoff_date", "endpoint_date"]:
        pd.testing.assert_series_equal(f2[c], frame[c], check_names=False)
    s1 = frame[F.common_mask(frame) & (frame["year"] < 2026)]
    s2 = f2[F.common_mask(f2) & (f2["year"] < 2026)].sample(frac=1.0, random_state=4)
    a = run_models(s1)
    b = run_models(s2).loc[a.index]
    np.testing.assert_allclose(a["p_C"], b["p_C"], rtol=1e-6)
    np.testing.assert_allclose(a["p_B"], b["p_B"], rtol=1e-6)


def test_holdout_refuses_without_a_frozen_spec(tmp_path, monkeypatch):
    from analysis.model_c_refit import evaluate as E
    monkeypatch.setattr(E, "FROZEN", tmp_path / "FROZEN.json")
    with pytest.raises(SystemExit):
        E.holdout()
    (tmp_path / "FROZEN.json").write_text(json.dumps({"spec_sha256": "not-the-spec"}))
    with pytest.raises(SystemExit):
        E.holdout()


# ─────────────────────────────── real production frame (if built) ─────────────────────
REAL = "output/model_c_refit/frame.parquet"


@pytest.fixture(scope="module")
def real():
    if not os.path.exists(REAL):
        pytest.skip("run `evaluate build` first")
    return pd.read_parquet(REAL)


def test_real_frame_obeys_the_rules(real):
    lab = real[real["y"].notna()]
    assert (lab["endpoint_idx"] > lab["cutoff_idx"]).all()
    assert (lab["reaction_3d_anchored_status"] == "available").all()
    assert (real.loc[real["y"].isna() & real["is_pending"].eq(False), "reaction_3d_anchored_status"]
            != "available").all()
    has = real["vol_date"].notna()
    assert (real.loc[has, "vol_date"] <= real.loc[has, "cutoff_date"]).all()
    assert (real["cutoff_date"] < real["call_monday"]).all()


def test_real_history_matches_the_definition(real):
    for i in real[real["y"].notna() & real["n_prior"].ge(8)].sample(80, random_state=2).index:
        n, m = _brute_hist(real, i)
        assert real.loc[i, "n_prior"] == n
        assert real.loc[i, "hist_mean_abs"] == pytest.approx(m)
