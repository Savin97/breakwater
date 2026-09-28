"""Phase 5B invariants: causality, leave-one-out, peer availability, evaluation hygiene,
and synthetic recovery / null behaviour.

Run from the repo root:  .venv/bin/python -m pytest research/phase_5b_new_feature_testing/tests -q
"""
import numpy as np
import pandas as pd
import pytest

import research.phase5_event_signal as p5
from feature_engineering.announcement_timing import TARGET_AVAILABLE
from research.phase3_target_rebuild import CORRECTED_TARGET, PHASE3_PREFIX
from research.phase4_baselines import prepare_analysis_frame
from research.phase5_event_signal import STRUCTURAL, walk_forward_predictions
from research.phase_5b_new_feature_testing import evaluate as ev
from research.phase_5b_new_feature_testing.panel import (
    build_panel,
    event_cutoff_index,
    issuer_of,
)
from research.phase_5b_new_feature_testing.peer_features import compute_peer_features
from research.phase_5b_new_feature_testing.price_features import (
    compute_price_features,
    rolling_vol_frames,
)
from research.phase_5b_new_feature_testing import registry as R

GRID = pd.bdate_range("2015-01-01", "2024-12-31")


# ─────────────────────────────────────── fixtures ────────────────────────────────────────
def _daily(n_stocks=12, seed=0, sessions=GRID, *, tickers=None, sectors=None, subs=None,
           ret_fn=None):
    rng = np.random.default_rng(seed)
    tickers = tickers or [f"S{i:02d}" for i in range(n_stocks)]
    sectors = sectors or [f"SEC{i % 2}" for i in range(len(tickers))]
    subs = subs or [f"SUB{i % 4}" for i in range(len(tickers))]
    m = rng.normal(0, 0.01, len(sessions))
    rows = []
    for k, t in enumerate(tickers):
        r = ret_fn(k, m, rng) if ret_fn else 0.9 * m + rng.normal(0, 0.015, len(sessions))
        price = 100 * np.cumprod(1 + r)
        rows.append(pd.DataFrame({"stock": t, "date": sessions, "price": price,
                                  "sector": sectors[k], "sub_sector": subs[k]}))
    return pd.concat(rows, ignore_index=True)


def _events(rows):
    d = pd.DataFrame(rows)
    defaults = {"is_pending": False, "sector": "SEC0", "sub_sector": "SUB0",
                "phase3_proxy_session_date": pd.NaT,
                "reaction_1d_anchored_status": TARGET_AVAILABLE,
                "reaction_3d_anchored_status": TARGET_AVAILABLE}
    for k, v in defaults.items():
        if k not in d.columns:
            d[k] = v
        elif not pd.isna(v):
            d[k] = d[k].fillna(v)
    d["earnings_date"] = pd.to_datetime(d["earnings_date"])
    d["phase3_proxy_session_date"] = pd.to_datetime(d["phase3_proxy_session_date"])
    if "anchor_date" in d:
        d["anchor_date"] = pd.to_datetime(d["anchor_date"])
    return d


def _phase4_frame(d):
    """Add the columns Phase 4's `prepare_analysis_frame` requires."""
    d = d.copy()
    for col, v in {PHASE3_PREFIX + "risk_score": 50.0, PHASE3_PREFIX + "abs_reaction_p75": 0.06,
                   PHASE3_PREFIX + "abs_reaction_p75_rolling": 0.06,
                   PHASE3_PREFIX + "reaction_entropy": 0.5,
                   PHASE3_PREFIX + "stock_bucket_lift": 1.0,
                   PHASE3_PREFIX + "bucket_structural": "Normal",
                   PHASE3_PREFIX + "bucket": "Normal",
                   PHASE3_PREFIX + "n_prior_resolved_reactions": 0}.items():
        if col not in d:
            d[col] = v
    d["phase3_announce_date"] = d["earnings_date"]
    return d


def _price_event(stock, date, **kw):
    return {"stock": stock, "earnings_date": date, "sector": "SEC0", "sub_sector": "SUB0", **kw}


# ──────────────────────────── cutoff / BMO / prior-only price data ───────────────────────
def test_cutoff_is_the_session_strictly_before_the_earliest_report_date():
    grid = GRID.to_numpy(dtype="datetime64[ns]")
    ev_ = _events([
        _price_event("S00", "2020-06-10"),                                   # Wednesday
        _price_event("S00", "2020-06-10", phase3_proxy_session_date="2020-06-09"),
        _price_event("S00", "2020-06-10", phase3_proxy_session_date="2020-06-11"),
        _price_event("S00", "2020-06-13"),                                   # Saturday
    ])
    c = event_cutoff_index(ev_, grid)
    got = [pd.Timestamp(grid[i]).date().isoformat() for i in c]
    assert got == ["2020-06-09", "2020-06-08", "2020-06-09", "2020-06-12"]


def test_bmo_event_day_return_cannot_enter_any_price_feature():
    daily = _daily()
    ev_ = _events([_price_event("S03", "2020-06-10", announce_window="BMO")])
    base = compute_price_features(build_panel(daily), ev_)
    shocked = daily.copy()
    m = (shocked["stock"] == "S03") & (shocked["date"] >= "2020-06-10")
    shocked.loc[m, "price"] *= 3.0          # a +200% "reaction" on D and everything after
    after = compute_price_features(build_panel(shocked), ev_)
    pd.testing.assert_frame_equal(base, after)


def test_price_features_use_prior_observations_only_and_do_react_to_them():
    daily = _daily()
    ev_ = _events([_price_event("S03", "2020-06-10")])
    base = compute_price_features(build_panel(daily), ev_)
    # every stock's prices after the cutoff are scrambled: nothing may move
    later = daily.copy()
    rng = np.random.default_rng(9)
    m = later["date"] > "2020-06-09"
    later.loc[m, "price"] *= np.exp(rng.normal(0, 0.2, m.sum()))
    pd.testing.assert_frame_equal(base, compute_price_features(build_panel(later), ev_))
    # a jump ON the cutoff session must move the jump features (the window is live)
    earlier = daily.copy()
    m = (earlier["stock"] == "S03") & (earlier["date"] >= "2020-06-09")
    earlier.loc[m, "price"] *= 1.25
    moved = compute_price_features(build_panel(earlier), ev_)
    assert moved.at[0, "max_abs_ret_20d"] > 0.2 > base.at[0, "max_abs_ret_20d"]


def test_own_slice_vol_matches_the_trailing_rolling_frame():
    panel = build_panel(_daily())
    vols = rolling_vol_frames(panel)
    ev_ = _events([_price_event("S05", "2021-03-03")])
    f = compute_price_features(panel, ev_, vols)
    c = int(f.at[0, "p5b_cutoff_session"])
    assert np.isclose(f.at[0, "vol_30d_cut"], vols["s30"][c, panel.col["S05"]])


def test_missing_session_yields_nan_return_not_a_stretched_one():
    daily = _daily()
    daily = daily[~((daily["stock"] == "S01") & (daily["date"] == "2020-03-04"))]
    panel = build_panel(daily)
    t = np.searchsorted(panel.grid, np.datetime64("2020-03-04"))
    i = panel.col["S01"]
    assert np.isnan(panel.ret[t, i]) and np.isnan(panel.ret[t + 1, i])


# ─────────────────────────────────────── leave-one-out ───────────────────────────────────
def test_market_and_sector_benchmarks_leave_the_stock_out():
    daily = _daily()
    base = build_panel(daily)
    bumped = daily.copy()
    rng = np.random.default_rng(1)
    m = bumped["stock"] == "S02"
    bumped.loc[m, "price"] *= np.exp(np.cumsum(rng.normal(0, 0.05, m.sum())))
    new = build_panel(bumped)
    i, j = base.col["S02"], base.col["S04"]                 # S04 shares S02's sector
    np.testing.assert_allclose(base.mkt_loo[:, i], new.mkt_loo[:, i], equal_nan=True)
    np.testing.assert_allclose(base.sec_loo[:, i], new.sec_loo[:, i], equal_nan=True)
    assert not np.allclose(base.sec_loo[1:, j], new.sec_loo[1:, j])  # others DO see S02


def test_share_class_twins_are_left_out_together():
    daily = _daily(tickers=["GOOG", "GOOGL", "AAA", "BBB", "CCC"],
                   sectors=["X"] * 5, subs=["Y"] * 5)
    base = build_panel(daily)
    bumped = daily.copy()
    m = bumped["stock"] == "GOOGL"
    bumped.loc[m, "price"] *= np.linspace(1, 3, m.sum())
    new = build_panel(bumped)
    g = base.col["GOOG"]
    np.testing.assert_allclose(base.sec_loo[:, g], new.sec_loo[:, g], equal_nan=True)
    assert issuer_of("GOOG") == issuer_of("GOOGL")


def test_sub_sector_peer_vol_excludes_the_stock_itself():
    daily = _daily()
    ev_ = _events([_price_event("S00", "2020-06-10")])
    base = compute_price_features(build_panel(daily), ev_)
    wild = daily.copy()
    m = (wild["stock"] == "S00") & (wild["date"] < "2020-06-10")
    wild.loc[m, "price"] *= np.exp(np.random.default_rng(2).normal(0, 0.1, m.sum()))
    out = compute_price_features(build_panel(wild), ev_)
    assert out.at[0, "sub_peer_med_vol_30d"] == base.at[0, "sub_peer_med_vol_30d"]
    assert out.at[0, "idio_vol_30d"] != base.at[0, "idio_vol_30d"]


# ───────────────────────────────────── peer availability ─────────────────────────────────
def _peer(stock, report, anchor, r1d, **kw):
    return {"stock": stock, "earnings_date": report, "anchor_date": anchor,
            "reaction_1d_anchored": r1d, "sector": "SEC0", "sub_sector": "SUB0", **kw}


def _target(stock="TGT", report="2020-06-10"):
    return {"stock": stock, "earnings_date": report, "anchor_date": "2020-06-09",
            "reaction_1d_anchored": 0.01, "sector": "SEC0", "sub_sector": "SUB0"}


def test_peer_reaction_is_unavailable_until_its_endpoint_session_has_closed():
    grid = GRID.to_numpy(dtype="datetime64[ns]")
    # target reports 2020-06-10 → cutoff 2020-06-09.
    ev_ = _events([
        _target(),
        _peer("AMC_ON_CUTOFF", "2020-06-09", "2020-06-09", 0.30),   # endpoint 06-10: NOT usable
        _peer("AMC_BEFORE", "2020-06-08", "2020-06-08", 0.10),      # endpoint 06-09: usable
        _peer("BMO_ON_CUTOFF", "2020-06-09", "2020-06-08", 0.20),   # endpoint 06-09: usable
    ])
    f = compute_peer_features(ev_, grid)
    assert f.at[0, "peer_sec_n_20"] == 2
    assert np.isclose(f.at[0, "peer_sec_mean_abs_1d_20"], 0.15)
    assert f.at[0, "peer_sec_max_abs_1d_20"] == 0.20


def test_a_later_or_same_day_peer_never_reaches_an_earlier_event():
    grid = GRID.to_numpy(dtype="datetime64[ns]")
    rows = [_target(), _peer("EARLY", "2020-06-01", "2020-06-01", 0.05)]
    base = compute_peer_features(_events(rows), grid)
    rows2 = rows + [_peer("SAME", "2020-06-10", "2020-06-09", 0.9),
                    _peer("LATER", "2020-06-12", "2020-06-12", 0.9)]
    more = compute_peer_features(_events(rows2), grid)
    cols = [c for c in base.columns if not c.startswith("peer_mkt")]
    pd.testing.assert_series_equal(base.loc[0, cols], more.loc[0, cols], check_names=False)


def test_unresolved_peer_outcomes_are_absent_not_zero():
    grid = GRID.to_numpy(dtype="datetime64[ns]")
    rows = [_target(), _peer("P1", "2020-06-01", "2020-06-01", 0.10)]
    base = compute_peer_features(_events(rows), grid)
    rows2 = rows + [_peer("P2", "2020-06-03", "2020-06-03", np.nan,
                          reaction_1d_anchored_status="unresolved_no_timestamp")]
    f = compute_peer_features(_events(rows2), grid)
    assert f.at[0, "peer_sec_n_20"] == base.at[0, "peer_sec_n_20"] == 1
    assert f.at[0, "peer_sec_mean_abs_1d_20"] == 0.10
    assert f.at[0, "peer_sec_n_unresolved_20"] == 1


def test_no_usable_peer_gives_nan_aggregates_and_zero_count():
    grid = GRID.to_numpy(dtype="datetime64[ns]")
    f = compute_peer_features(_events([_target()]), grid)
    assert f.at[0, "peer_sec_n_20"] == 0
    assert np.isnan(f.at[0, "peer_sec_mean_abs_1d_20"])
    assert np.isnan(f.at[0, "peer_sec_shock_20"])


def test_self_and_share_class_twin_are_never_peers():
    grid = GRID.to_numpy(dtype="datetime64[ns]")
    rows = [_target("GOOG"), _peer("GOOGL", "2020-06-01", "2020-06-01", 0.3),
            _peer("GOOG", "2020-06-02", "2020-06-02", 0.3)]
    f = compute_peer_features(_events(rows), grid)
    assert f.at[0, "peer_sec_n_20"] == 0


def test_same_day_row_order_cannot_change_any_feature():
    grid = GRID.to_numpy(dtype="datetime64[ns]")
    rows = [_target("A"), _target("B"), _target("C"),
            _peer("P", "2020-06-02", "2020-06-02", 0.1)]
    f1 = compute_peer_features(_events(rows), grid)
    f2 = compute_peer_features(_events(rows[::-1]), grid).iloc[::-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(f1, f2)


def test_peer_shock_uses_only_the_peers_prior_events():
    grid = GRID.to_numpy(dtype="datetime64[ns]")
    hist = [_peer("P", d, d, 0.02) for d in ("2019-06-03", "2019-09-03", "2019-12-03",
                                             "2020-03-03")]
    rows = [_target(), *hist, _peer("P", "2020-06-02", "2020-06-02", 0.10)]
    f = compute_peer_features(_events(rows), grid)
    expect = np.log((0.10 + 0.005) / (0.02 + 0.005))
    assert np.isclose(f.at[0, "peer_sec_shock_20"], expect)


# ──────────────────────────────────── evaluation hygiene ─────────────────────────────────
def _wf_sample(seed=3, signal=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(60):
        scale = rng.uniform(0.02, 0.09)
        for e in range(28):
            x = rng.normal()
            p = 1 / (1 + np.exp(-(-2.0 + 12 * (scale - 0.05) + signal * x)))
            rows.append({"stock": f"S{s:02d}",
                         "event_clock": pd.Timestamp("2018-02-01") + pd.Timedelta(days=91 * e + s),
                         "y_extreme": float(rng.random() < p),
                         STRUCTURAL: scale + rng.normal(0, 0.01), "vol_30d": rng.normal(),
                         "feat": x, "noise1": rng.normal(), "noise2": rng.normal(),
                         "own": rng.normal(), "peer": rng.normal()})
    return pd.DataFrame(rows)


def test_walk_forward_train_years_strictly_precede_the_test_year():
    seen = []
    real = p5.fit_block

    def spy(train, features, **kw):
        seen.append(train["event_clock"].max())
        return real(train, features, **kw)

    p5.fit_block = spy
    ev.fit_block = spy
    try:
        pred = walk_forward_predictions(_wf_sample(), [STRUCTURAL, "feat"])
        pix = ev.walk_forward_interaction(_wf_sample(), [STRUCTURAL], "own", "peer")
    finally:
        p5.fit_block = real
        ev.fit_block = real
    years = sorted(pred["test_year"].unique()) + sorted(pix["test_year"].unique())
    assert len(seen) == len(years)
    for mx, y in zip(seen, years):
        assert mx < pd.Timestamp(year=int(y), month=1, day=1)


def test_interaction_preprocessing_is_fitted_on_training_rows_only():
    s = _wf_sample()
    base = ev.walk_forward_interaction(s, [STRUCTURAL], "own", "peer")
    s2 = s.copy()
    last = s2["event_clock"].dt.year == 2024
    s2.loc[last, "peer"] = s2.loc[last, "peer"] * 1000 + 50   # only the final test year
    new = ev.walk_forward_interaction(s2, [STRUCTURAL], "own", "peer")
    early = base["test_year"] < 2024
    np.testing.assert_allclose(base.loc[early, "p"], new.loc[early, "p"])


def test_bootstrap_refuses_different_rows_and_all_models_share_rows():
    s = _wf_sample()
    a = walk_forward_predictions(s, [STRUCTURAL])
    b = walk_forward_predictions(s, [STRUCTURAL, "feat"])
    assert a.index.equals(b.index)
    with pytest.raises(ValueError):
        ev.bootstrap_delta(a.iloc[1:], b)
    specs = [("A_structural", [STRUCTURAL], "reference"),
             ("B_incumbent_structural+vol_30d", [STRUCTURAL, "vol_30d"], "incumbent"),
             ("C", [STRUCTURAL, "vol_30d", "feat"], "primary_single")]
    _m, _y, _b, preds = ev.evaluate_models(s, specs, reps=20)
    idx = [p.index for p in preds.values()]
    assert all(i.equals(idx[0]) for i in idx)


def test_decision_table_marks_the_incumbent_undefined_not_false():
    s = _wf_sample(signal=1.0)
    specs = [("A_structural", [STRUCTURAL], "reference"),
             ("B_incumbent_structural+vol_30d", [STRUCTURAL, "vol_30d"], "incumbent"),
             ("C_inc+feat", [STRUCTURAL, "vol_30d", "feat"], "primary_single")]
    m, y, b, _p = ev.evaluate_models(s, specs, reps=30)
    d = ev.decisions(m, y, b).set_index("model")
    assert d.at["B_incumbent_structural+vol_30d", "robust_beat"] is None
    assert d.at["C_inc+feat", "stat_incremental"] is True or d.at["C_inc+feat", "stat_incremental"] == True


def test_bootstrap_is_deterministic_and_clustered_by_stock():
    s = _wf_sample()
    a = walk_forward_predictions(s, [STRUCTURAL, "feat"])
    b = walk_forward_predictions(s, [STRUCTURAL])
    assert ev.bootstrap_delta(a, b, reps=30) == ev.bootstrap_delta(a, b, reps=30)


def test_null_features_produce_no_incremental_oof_signal():
    s = _wf_sample(seed=11, signal=0.0)
    inc = walk_forward_predictions(s, [STRUCTURAL, "vol_30d"])
    cand = walk_forward_predictions(s, [STRUCTURAL, "vol_30d", "feat", "noise1", "noise2"])
    d = ev.bootstrap_delta(cand, inc, reps=200)
    assert d["delta_auc_lo"] < 0 < d["delta_auc_hi"] or abs(d["delta_auc"]) < 0.005


def test_z_own_matches_phase5_on_vol_30d():
    rows = [{"stock": "AAA", "earnings_date": f"20{15 + i // 4}-{1 + 3 * (i % 4):02d}-05",
             CORRECTED_TARGET: 0.05, "vol_30d": 0.02 + 0.003 * ((i * 7) % 5)} for i in range(12)]
    d = _phase4_frame(_events(rows).assign(
        drift_30d=0.0, drift_60d=0.0, mom_5d=0.0, mom_20d=0.0, vol_10d=0.02,
        vol_ratio_10_to_30=1.0))
    a = p5.derive_phase5_features(prepare_analysis_frame(d))
    np.testing.assert_allclose(ev.z_own(a, "vol_30d"), a["vol_30d_z_own"], equal_nan=True)


def test_registry_has_the_preregistered_29_primary_features():
    assert len(R.PRIMARY) == 29
    assert set(ev.OWN_PRICE_REQUIRED) <= set(R.PRIMARY)
    assert not set(R.PRIMARY) & set(R.SECONDARY)


# ───────────────────────────────────── synthetic recovery ────────────────────────────────
def _peer_regime_panel(seed: int, regime: bool):
    """Sector earnings seasons: a sector-quarter latent scale drives every reaction in it.

    Stocks within a sector report on staggered sessions, so late reporters can observe
    early reporters' realized moves. With regime=False the scale is constant (null).
    """
    rng = np.random.default_rng(seed)
    grid = pd.bdate_range("2014-01-01", "2025-12-31")
    g = grid.to_numpy(dtype="datetime64[ns]")
    rows = []
    n_sec, per_sec = 6, 10
    # per-stock phenotype, independent of reporting ORDER (order drives the peer count)
    mult = {(s, k): rng.uniform(0.6, 1.4) for s in range(n_sec) for k in range(per_sec)}
    for q_start in pd.date_range("2014-01-15", "2025-10-15", freq="QS-JAN") + pd.Timedelta(days=14):
        base_idx = np.searchsorted(g, np.datetime64(q_start))
        for sec in range(n_sec):
            h = rng.choice([0.5, 2.0]) if regime else 1.0
            for k in range(per_sec):
                idx = base_idx + 3 * k + sec          # staggered: k-th reporter, 3 sessions apart
                if idx + 4 >= len(g):
                    continue
                stock = f"S{sec}_{k}"
                scale = 0.03 * h * mult[(sec, k)]
                r1 = rng.normal(0, scale)
                r3 = r1 + rng.normal(0, 0.3 * scale)
                rows.append({"stock": stock, "earnings_date": g[idx + 1], "anchor_date": g[idx],
                             "reaction_1d_anchored": r1, CORRECTED_TARGET: abs(r3),
                             "sector": f"SEC{sec}", "sub_sector": f"SUB{sec}_{k % 2}"})
    return _phase4_frame(_events(rows)), g


def _peer_signal_delta(seed, regime):
    events, grid = _peer_regime_panel(seed, regime)
    feats = compute_peer_features(events, grid)
    a = prepare_analysis_frame(events).join(feats[["peer_sec_mean_abs_1d_20", "peer_sec_n_20"]])
    a = a.dropna(subset=["y_extreme", STRUCTURAL])
    base = walk_forward_predictions(a, [STRUCTURAL])
    cand = walk_forward_predictions(a, [STRUCTURAL, "peer_sec_mean_abs_1d_20", "peer_sec_n_20"])
    return ev.bootstrap_delta(cand, base, reps=100)


def test_a_true_peer_shock_signal_is_recovered():
    d = _peer_signal_delta(seed=5, regime=True)
    assert d["delta_auc"] > 0.03 and d["delta_auc_lo"] > 0


def test_peer_features_without_a_peer_regime_add_nothing():
    d = _peer_signal_delta(seed=5, regime=False)
    assert d["delta_auc_lo"] < 0.005 and d["delta_auc"] < 0.01


def test_a_true_idiosyncratic_volatility_signal_is_recovered_and_beats_raw_vol():
    """Earnings moves scale with each stock's CURRENT idiosyncratic vol; the market's vol
    swings strongly and independently, so raw vol is a contaminated proxy for it."""
    rng = np.random.default_rng(21)
    sessions = pd.bdate_range("2012-01-02", "2025-12-31")
    T = len(sessions)
    block = np.arange(T) // 63
    n_blocks = block.max() + 1
    mkt_sigma = rng.choice([0.004, 0.03], size=n_blocks)[block]
    m = rng.normal(0, 1, T) * mkt_sigma
    n = 40
    idio_sigma = rng.uniform(0.008, 0.03, size=(n_blocks, n))
    tickers = [f"S{i:02d}" for i in range(n)]

    def ret_fn(k, _m, r):
        return 1.0 * m + r.normal(0, 1, T) * idio_sigma[block, k]

    daily = _daily(tickers=tickers, sectors=[f"SEC{i % 4}" for i in range(n)],
                   subs=[f"SUB{i % 8}" for i in range(n)], sessions=sessions, ret_fn=ret_fn)
    rows = []
    g = sessions.to_numpy(dtype="datetime64[ns]")
    for b in range(4, n_blocks):
        for k in range(n):
            idx = b * 63 + 45 + (k % 10)
            if idx >= T:
                continue
            move = abs(rng.normal(0, 4.0 * idio_sigma[b, k]))
            rows.append({"stock": tickers[k], "earnings_date": g[idx], CORRECTED_TARGET: move,
                         "sector": f"SEC{k % 4}", "sub_sector": f"SUB{k % 8}"})
    events = _phase4_frame(_events(rows))
    feats = compute_price_features(build_panel(daily), events)
    a = prepare_analysis_frame(events).join(feats[["idio_vol_30d", "vol_30d_cut"]])
    a = a.dropna(subset=["y_extreme", STRUCTURAL, "idio_vol_30d", "vol_30d_cut"])
    base = walk_forward_predictions(a, [STRUCTURAL])
    raw = walk_forward_predictions(a, [STRUCTURAL, "vol_30d_cut"])
    idio = walk_forward_predictions(a, [STRUCTURAL, "idio_vol_30d"])
    assert ev.bootstrap_delta(idio, base, reps=100)["delta_auc_lo"] > 0
    assert ev.bootstrap_delta(idio, raw, reps=100)["delta_auc_lo"] > 0
