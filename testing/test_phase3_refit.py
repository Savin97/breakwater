"""Tests for the P3.2/P3.3 refit (`research/phase3_refit/`).

These pin the ways the refit could flatter itself: an outcome reaching a score before it
was observable, a model or cut point learning from its own test year, candidates being
compared on different rows. Synthetic frames only — no parquet, no database.
"""
import numpy as np
import pandas as pd
import pytest

from feature_engineering.announcement_timing import AMC, BMO, TARGET_AVAILABLE
from research.phase3_refit import calibration as cal
from research.phase3_refit.candidates import (
    promote,
    v031_score,
    v031_structural_tier,
)
from research.phase3_refit.features import (
    daily_asof,
    event_clocks,
    history_features,
    market_prior,
    outcome_frame,
    stock_bucket_lift,
)

GRID = pd.bdate_range("2024-01-01", "2024-12-31").to_numpy(dtype="datetime64[ns]")


def _events(rows):
    """Minimal Phase-3-shaped events. Each row: stock, report (date), anchor, r3, r1."""
    d = pd.DataFrame(rows)
    d["earnings_date"] = pd.to_datetime(d["report"])
    d["phase3_proxy_session_date"] = pd.NaT
    d["anchor_date"] = pd.to_datetime(d.get("anchor", d["report"]))
    d["is_pending"] = d.get("is_pending", False)
    d["abs_reaction_3d_anchored"] = d["r3"]
    d["reaction_1d_anchored"] = d.get("r1", d["r3"])
    d["reaction_3d_anchored_status"] = np.where(d["r3"].notna(), TARGET_AVAILABLE,
                                                "unresolved_no_timestamp")
    d["reaction_1d_anchored_status"] = np.where(d["reaction_1d_anchored"].notna(),
                                                TARGET_AVAILABLE, "unresolved_no_timestamp")
    return d


def _history(ev, cutoff="call"):
    clocks = event_clocks(ev, GRID)
    out = outcome_frame(ev)
    return history_features(ev, clocks, out, cutoff), clocks, out


def _series_of_quarters(stock="AAA", n=6, r3=0.05, start="2024-01-10"):
    dates = pd.date_range(start, periods=n, freq="35D")
    dates = [pd.Timestamp(GRID[np.searchsorted(GRID, np.datetime64(d))]) for d in dates]
    return [{"stock": stock, "report": d, "r3": r3} for d in dates]


# ─────────────────────────────── outcomes never leak into scores ─────────────────────────
def test_own_outcome_cannot_enter_own_score():
    rows = _series_of_quarters(n=5)
    a, _, _ = _history(_events(rows))
    rows[3]["r3"] = 0.50                      # change event 3's own outcome
    b, _, _ = _history(_events(rows))
    assert a.loc[3].equals(b.loc[3])
    assert a.loc[4, "hist_mean_abs_call"] != b.loc[4, "hist_mean_abs_call"]


def test_future_events_cannot_affect_earlier_scores():
    rows = _series_of_quarters(n=6)
    a, _, _ = _history(_events(rows))
    rows[5]["r3"] = 0.90
    rows.append({"stock": "AAA", "report": pd.Timestamp("2024-12-20"), "r3": 0.70})
    b, _, _ = _history(_events(rows))
    pd.testing.assert_frame_equal(a.iloc[:6], b.iloc[:6])


def test_outcome_counts_only_once_its_endpoint_is_observable():
    # Event 0 reports Wed 2024-03-06 (AMC, anchor = that day; 3d endpoint = Mon 03-11).
    # Event 1 of the same stock is called from Fri 03-08: endpoint not yet seen.
    # Event 2 is called from Fri 03-15: both are (event 1 ends exactly at that close).
    rows = [{"stock": "AAA", "report": "2024-03-06", "r3": 0.10},
            {"stock": "AAA", "report": "2024-03-12", "r3": 0.02},
            {"stock": "AAA", "report": "2024-03-19", "r3": 0.02}]
    h, clocks, _ = _history(_events(rows))
    assert pd.Timestamp(clocks.loc[0, "endpoint3_date"]) == pd.Timestamp("2024-03-11")
    assert pd.Timestamp(clocks.loc[1, "call_cutoff_date"]) == pd.Timestamp("2024-03-08")
    assert h.loc[1, "n_prior_call"] == 0
    assert h.loc[2, "n_prior_call"] == 2


def test_same_day_outcomes_do_not_leak_into_market_prior():
    rows = [{"stock": s, "report": "2024-05-08", "r3": r} for s, r in
            (("AAA", 0.20), ("BBB", 0.01), ("CCC", 0.30))]
    rows += [{"stock": "DDD", "report": "2024-04-03", "r3": 0.01}]
    ev = _events(rows)
    clocks, out = event_clocks(ev, GRID), outcome_frame(ev)
    prior = market_prior(clocks, out, "eve")
    # same-day events see only DDD, never one another
    assert prior.loc[[0, 1, 2]].tolist() == [0.0, 0.0, 0.0]
    # shuffling frame order changes nothing
    perm = ev.sample(frac=1.0, random_state=3)
    c2, o2 = event_clocks(perm, GRID), outcome_frame(perm)
    pd.testing.assert_series_equal(market_prior(c2, o2, "eve").sort_index(), prior)


def test_unresolved_target_stays_missing_never_zero():
    rows = _series_of_quarters(n=4)
    rows[1]["r3"] = np.nan
    rows[1]["r1"] = np.nan
    ev = _events(rows)
    out = outcome_frame(ev)
    assert np.isnan(out.loc[1, "y_extreme"]) and np.isnan(out.loc[1, "abs_r3"])
    h, _, _ = _history(ev)
    assert h.loc[3, "n_prior_call"] == 2          # events 0 and 2, not the unresolved one
    assert h.loc[3, "hist_mean_abs_call"] == pytest.approx(0.05)


def test_lift_uses_only_endpoint_safe_same_bucket_history():
    rows = _series_of_quarters(n=4, r3=0.20)
    ev = _events(rows)
    clocks, out = event_clocks(ev, GRID), outcome_frame(ev)
    bucket = pd.Series(["Normal"] * 4, index=ev.index, dtype=object)
    prior = pd.Series(0.2, index=ev.index)
    lift = stock_bucket_lift(ev, clocks, out, bucket, prior, strength=20)
    assert lift.loc[0] == 1.0                      # no history -> no opinion
    assert lift.loc[3] == pytest.approx(((3 * 1.0) + 20 * 0.2) / 23 / 0.2)
    bucket.loc[[0, 1, 2]] = "Elevated"             # different bucket -> not counted
    assert stock_bucket_lift(ev, clocks, out, bucket, prior, strength=20).loc[3] == 1.0


# ─────────────────────────────────── anchoring conventions ───────────────────────────────
def _toy_daily():
    days = pd.bdate_range("2024-06-03", "2024-06-21")
    return pd.DataFrame({"stock": "AAA", "date": days, "price": np.linspace(100, 110, len(days))})


@pytest.mark.parametrize("window,clock,anchor", [
    (BMO, "2024-06-12 07:00", "2024-06-11"),     # BMO: last close before = D-1
    (AMC, "2024-06-12 16:05", "2024-06-12"),     # AMC: last close before = D
])
def test_anchor_is_last_close_before_announcement(window, clock, anchor):
    from research.phase3_target_rebuild import reanchor_to_observed_timestamp
    daily = _toy_daily()
    ev = pd.DataFrame({"stock": ["AAA"], "earnings_date": [pd.Timestamp("2024-06-12")],
                       "announce_ts_ny": [pd.Timestamp(clock)], "announce_window": [window],
                       "is_pending": [False]})
    out = reanchor_to_observed_timestamp(ev, daily)
    assert pd.Timestamp(out.loc[0, "anchor_date"]) == pd.Timestamp(anchor)
    grid = np.sort(daily["date"].to_numpy(dtype="datetime64[ns]"))
    clocks = event_clocks(out.assign(phase3_proxy_session_date=pd.NaT), grid)
    a = int(np.searchsorted(grid, np.datetime64(pd.Timestamp(anchor))))
    assert clocks.loc[0, "endpoint3_idx"] == a + 3


# ──────────────────────────────── prediction-time cutoff ─────────────────────────────────
def test_call_cutoff_is_the_friday_before_the_report_week():
    ev = _events([{"stock": "AAA", "report": d, "r3": 0.05}
                  for d in ("2024-06-10", "2024-06-12", "2024-06-14")])
    clocks = event_clocks(ev, GRID)
    assert (clocks["call_cutoff_date"] == pd.Timestamp("2024-06-07")).all()
    assert clocks["eve_cutoff_date"].tolist() == [pd.Timestamp(d) for d in
                                                  ("2024-06-07", "2024-06-11", "2024-06-13")]


def test_daily_features_are_read_at_the_call_cutoff_not_later():
    days = pd.bdate_range("2024-06-03", "2024-06-14")
    daily = pd.DataFrame({"stock": "AAA", "date": days, "vol_30d": np.arange(len(days), dtype=float),
                          "drift_30d": 0.0})
    ev = _events([{"stock": "AAA", "report": "2024-06-14", "r3": 0.05}])
    clocks = event_clocks(ev, GRID)
    call = daily_asof(daily, clocks, ev, "call")
    eve = daily_asof(daily, clocks, ev, "eve")
    assert call.loc[0, "vol_30d_call"] == 4.0      # Fri 06-07 row
    assert eve.loc[0, "vol_30d_eve"] == 8.0        # Thu 06-13 row


def test_peer_reaction_unavailable_until_its_endpoint():
    from research.phase_5b_new_feature_testing.panel import build_panel
    from research.phase3_refit.features import phase5b_features_at
    days = pd.bdate_range("2024-01-02", "2024-06-28")
    rng = np.random.default_rng(0)
    daily = pd.concat([pd.DataFrame({"stock": s, "date": days, "sector": "Tech",
                                     "sub_sector": "X",
                                     "price": 100 * np.cumprod(1 + rng.normal(0, .01, len(days)))})
                       for s in ("AAA", "BBB", "CCC")])
    # BBB reports AMC Tue 06-11 (endpoint Wed 06-12); AAA reports Fri 06-14.
    ev = pd.DataFrame({
        "stock": ["BBB", "AAA", "CCC"], "sector": "Tech", "sub_sector": "X",
        "earnings_date": pd.to_datetime(["2024-06-11", "2024-06-14", "2024-05-20"]),
        "phase3_proxy_session_date": pd.NaT,
        "anchor_date": pd.to_datetime(["2024-06-11", "2024-06-13", "2024-05-20"]),
        "is_pending": False, "reaction_1d_anchored": [0.12, 0.01, 0.03],
        "reaction_1d_anchored_status": TARGET_AVAILABLE,
    })
    grid = build_panel(daily).grid
    clocks = event_clocks(ev, grid)
    call = phase5b_features_at(ev, clocks, daily, "call")
    eve = phase5b_features_at(ev, clocks, daily, "eve")
    # at the Friday-before call, only CCC's May reaction is observable to AAA
    assert call.loc[1, "peer_sec_rw_mean_abs_1d_20_call"] == pytest.approx(0.03)
    # on the eve (Thu 06-13) BBB's 06-12 endpoint has passed
    assert eve.loc[1, "peer_sec_rw_mean_abs_1d_20_eve"] > 0.03


# ──────────────────────────────────── walk-forward ───────────────────────────────────────
def _panel(n_years=5, per_year=300, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for yr in range(2018, 2018 + n_years):
        for k in range(per_year):
            x = rng.normal()
            v = rng.normal()
            p = 1 / (1 + np.exp(-(-1.5 + 0.8 * x + 0.3 * v)))
            rows.append({
                "stock": f"S{k % 60}", "year": yr,
                "report_date": pd.Timestamp(yr, 1 + k % 12, 1 + k % 25),
                "announce_window": BMO if k % 2 else AMC,
                "log_hist_mean_abs": x, "log_vol_30d": v, "a1_score": 50 + 10 * x,
                "a0_shipped": 50 + 10 * x, "log_peer_sec_rw": rng.normal(),
                "log_idio_vol_30d": rng.normal(), "y_extreme": float(rng.random() < p)})
    d = pd.DataFrame(rows)
    d["endpoint3_date"] = d["report_date"] + pd.Timedelta(days=5)
    d["is_bmo"] = d["announce_window"].eq(BMO).astype(float)
    return d


def test_training_rows_end_strictly_before_the_test_year():
    d = _panel()
    d.loc[d.index[:5], "report_date"] = pd.Timestamp("2019-12-30")
    d.loc[d.index[:5], "endpoint3_date"] = pd.Timestamp("2020-01-03")
    m = cal.training_mask(d, 2020)
    assert d.loc[m, "endpoint3_date"].max() < pd.Timestamp("2020-01-01")
    assert not m.iloc[:5].any()


@pytest.mark.parametrize("window_mode", ["W0", "W1", "W2"])
def test_fit_calibration_and_cuts_never_see_the_test_year(window_mode):
    d = _panel()
    pred_a, folds_a = cal.walk_forward(d, "C_structural_vol", window_mode, years=range(2020, 2023))
    poisoned = d.copy()
    test = poisoned["year"].eq(2021)
    poisoned.loc[test, "y_extreme"] = 1.0 - poisoned.loc[test, "y_extreme"]
    poisoned.loc[test, "log_vol_30d"] *= 50
    _, folds_b = cal.walk_forward(poisoned, "C_structural_vol", window_mode, years=range(2020, 2023))
    fa, fb = folds_a[2021], folds_b[2021]
    np.testing.assert_array_equal(fa.model.coef_, fb.model.coef_)
    pd.testing.assert_series_equal(fa.means, fb.means)
    pd.testing.assert_series_equal(fa.medians, fb.medians)
    assert fa.cuts == fb.cuts
    for w in fa.platt:
        np.testing.assert_array_equal(fa.platt[w].coef_, fb.platt[w].coef_)


def test_candidates_are_scored_on_identical_rows():
    d = _panel()
    idx = None
    for cand in ("A0_v031_shipped", "A1_v031_corrected", "B_structural", "C_structural_vol",
                 "D_phase5b_pair"):
        pred, _ = cal.walk_forward(d, cand, "W0", years=range(2020, 2023))
        idx = pred.index if idx is None else idx
        assert pred.index.equals(idx)


def test_tiers_are_deterministic():
    d = _panel()
    a, _ = cal.walk_forward(d, "C_structural_vol", "W0", years=range(2020, 2023))
    again, _ = cal.walk_forward(d, "C_structural_vol", "W0", years=range(2020, 2023))
    pd.testing.assert_frame_equal(a, again)
    # frame order does not matter beyond solver round-off
    b, _ = cal.walk_forward(d.sample(frac=1.0, random_state=7), "C_structural_vol", "W0",
                            years=range(2020, 2023))
    b = b.loc[a.index]
    np.testing.assert_allclose(a["p"], b["p"], rtol=1e-6, atol=1e-8)
    for rule in cal.TIER_RULES:
        assert (a[f"tier_{rule}"] == b[f"tier_{rule}"]).mean() > 0.999


def test_quantile_cuts_flag_ten_and_ten_percent_on_training():
    d = _panel()
    _, folds = cal.walk_forward(d, "C_structural_vol", "W0", years=[2021])
    f = folds[2021]
    train = d[cal.training_mask(d, 2021)]
    p = cal.probability(f, train, "W0")
    t = cal.assign_tiers(p, train["announce_window"].to_numpy(), f.cuts["Q_common"])
    assert (t == "High Alert").mean() == pytest.approx(0.10, abs=0.01)
    assert (t == "Elevated").mean() == pytest.approx(0.10, abs=0.01)
    tw = cal.assign_tiers(p, train["announce_window"].to_numpy(), f.cuts["Q_by_window"])
    for w in (BMO, AMC):
        m = train["announce_window"].eq(w).to_numpy()
        assert (tw[m] == "High Alert").mean() == pytest.approx(0.10, abs=0.015)


# ──────────────────────────────── 0.3.1 reconstruction ───────────────────────────────────
def test_v031_score_reproduces_the_shipped_formula():
    p75 = pd.Series([0.03, 0.12, 0.20, np.nan])
    ent = pd.Series([0.5, 2.0, 1.0, 0.3])
    got = v031_score(p75, ent)
    want = 100 * np.clip(0.85 * np.clip(p75 / 0.12, 0, 1) + 0.15 * np.clip(ent, 0, 1), 0, 1)
    np.testing.assert_allclose(got[:3], want[:3])
    # the ceiling flattens 0.12 and 0.20 onto the same score; removing it does not
    assert got[1] == got[2] == 100
    assert v031_score(p75, ent, cap=False)[2] > v031_score(p75, ent, cap=False)[1]


def test_v031_tiers_and_promotion_match_config():
    s = pd.Series([72.9, 73.0, 78.9, 79.0, 79.1])
    assert v031_structural_tier(s).tolist() == ["Normal", "Normal", "Elevated", "Elevated",
                                                "High Alert"]
    b = pd.Series(["Normal", "Normal", "Elevated", "High Alert"])
    lift = pd.Series([1.6, 3.1, 3.0, 0.5])
    assert promote(b, lift).tolist() == ["Elevated", "High Alert", "High Alert", "High Alert"]
