"""Families 1–4, 6 and 7: own-stock and peer-volatility price features at the event cutoff.

Every value is a function of `panel.ret[: c + 1]` (and the LOO benchmarks up to c), where c
is the event's cutoff session from `panel.event_cutoff_index`. Nothing reads a session
after c, and the day-D session is always after c. The per-event loop is deliberately
plain: it makes "only rows up to c" visible in the code instead of hidden in a rolling
alignment, and the whole universe runs in seconds.

Definitions are fixed in PREREGISTRATION.md; this module implements them and nothing else.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from phase_5b_new_feature_testing.panel import PricePanel, event_cutoff_index

REG_WINDOW = 252
REG_MIN_OBS = 126
PEER_VOL_HIGH_Q = 0.80
PEER_VOL_HISTORY = 252
PEER_VOL_HISTORY_MIN = 126
MIN_PEERS = 2

PRICE_FEATURE_COLUMNS = [
    # family 1
    "idio_vol_30d", "idio_vol_10d", "idio_share_30d",
    # family 2
    "max_abs_ret_20d", "jump_share_20d", "semivar_balance_20d",
    # family 3
    "vol_log_ratio_5_30", "vol_log_ratio_10_60",
    # family 4
    "abs_idio_ret_5d", "abs_idio_ret_20d",
    # family 6
    "sub_peer_med_vol_10d", "sub_peer_med_vol_30d", "stock_vs_sub_vol_30d",
    "sub_peer_frac_high_vol",
    # family 7
    "sector_corr_60d", "sector_corr_change_20_120",
    # secondary
    "signed_idio_ret_20d", "std_abs_idio_ret_20d", "max_abs_idio_ret_20d",
    "vol_change_20_20", "vol_30d_cut", "sec_peer_med_vol_30d",
    # bookkeeping
    "p5b_cutoff_session", "p5b_reg_obs",
]


def _tail(a: np.ndarray, c: int, k: int) -> np.ndarray:
    """The k values ending at index c inclusive (fewer at the start of history)."""
    return a[max(0, c - k + 1): c + 1]


def _std(x: np.ndarray, min_obs: int) -> float:
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1)) if len(x) >= max(min_obs, 2) else np.nan


def _corr(a: np.ndarray, b: np.ndarray, min_obs: int) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < max(min_obs, 3):
        return np.nan
    x, y = a[ok], b[ok]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _log_ratio(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0:
        return np.nan
    return float(np.log(a / b))


def idiosyncratic_residuals(r: np.ndarray, mkt: np.ndarray, sec: np.ndarray,
                            min_obs: int = REG_MIN_OBS) -> tuple[np.ndarray, int]:
    """OLS residuals of r on [1, MKT, SEC-MKT] fitted on these rows only.

    Returns (residuals aligned to r with NaN where any input is missing, obs used). The
    caller passes the trailing window ending at the cutoff, so the fit sees only the past.
    """
    ok = np.isfinite(r) & np.isfinite(mkt) & np.isfinite(sec)
    e = np.full(len(r), np.nan)
    n = int(ok.sum())
    if n < min_obs:
        return e, n
    X = np.column_stack([np.ones(n), mkt[ok], sec[ok] - mkt[ok]])
    beta, *_ = np.linalg.lstsq(X, r[ok], rcond=None)
    e[ok] = r[ok] - X @ beta
    return e, n


def rolling_vol_frames(panel: PricePanel) -> dict[str, np.ndarray]:
    """Wide trailing stds used for PEER aggregates (family 6), each ending at its row.

    Row t of every array uses returns through t only (pandas trailing windows), so reading
    row c at an event is causal. The own-stock values in the per-event loop are computed
    from slices directly and are cross-checked against these in the tests.
    """
    ret = pd.DataFrame(panel.ret)
    s10 = ret.rolling(10, min_periods=8).std()
    s30 = ret.rolling(30, min_periods=20).std()
    q80 = s30.rolling(PEER_VOL_HISTORY, min_periods=PEER_VOL_HISTORY_MIN).quantile(
        PEER_VOL_HIGH_Q)
    return {"s10": s10.to_numpy(float), "s30": s30.to_numpy(float),
            "s30_q80": q80.to_numpy(float)}


def _peer_cols(panel: PricePanel, i: int, level: str) -> np.ndarray:
    groups = panel.sub_sector if level == "sub" else panel.sector
    g = groups[i]
    if g is None or (isinstance(g, float) and np.isnan(g)):
        return np.array([], dtype=int)
    return np.where((groups == g) & (panel.issuer != panel.issuer[i]))[0]


def compute_price_features(panel: PricePanel, events: pd.DataFrame,
                           vols: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """One row per event (same index as `events`) with every family 1–4/6/7 column."""
    vols = vols if vols is not None else rolling_vol_frames(panel)
    col_of = panel.col
    cut = event_cutoff_index(events, panel.grid)
    stocks = events["stock"].astype(str).to_numpy()
    peer_cache: dict[tuple[int, str], np.ndarray] = {}

    out = np.full((len(events), len(PRICE_FEATURE_COLUMNS)), np.nan)
    pos = {c: k for k, c in enumerate(PRICE_FEATURE_COLUMNS)}

    for row, (stock, c) in enumerate(zip(stocks, cut)):
        i = col_of.get(stock)
        out[row, pos["p5b_cutoff_session"]] = c
        if i is None or c < 1:
            continue
        r_all = panel.ret[: c + 1, i]           # nothing after the cutoff exists here
        v = {}

        # ── family 1 / 4: residuals from a trailing 252-session fit ──
        rw = _tail(r_all, c, REG_WINDOW)
        mw = _tail(panel.mkt_loo[:, i], c, REG_WINDOW)
        sw = _tail(panel.sec_loo[:, i], c, REG_WINDOW)
        e, n_obs = idiosyncratic_residuals(rw, mw, sw)
        v["p5b_reg_obs"] = n_obs
        e30, e10, e20, e5 = e[-30:], e[-10:], e[-20:], e[-5:]
        v["idio_vol_30d"] = _std(e30, 20)
        v["idio_vol_10d"] = _std(e10, 8)
        ok30 = np.isfinite(e30)
        if ok30.sum() >= 20:
            tot = np.var(rw[-30:][ok30], ddof=1)
            v["idio_share_30d"] = float(np.var(e30[ok30], ddof=1) / tot) if tot > 0 else np.nan
        f5, f20 = e5[np.isfinite(e5)], e20[np.isfinite(e20)]
        if len(f5) >= 4:
            v["abs_idio_ret_5d"] = abs(float(f5.mean() * 5))
        if len(f20) >= 15:
            s = float(f20.mean() * 20)
            v["abs_idio_ret_20d"] = abs(s)
            v["signed_idio_ret_20d"] = s
            v["max_abs_idio_ret_20d"] = float(np.abs(f20).max())
            if np.isfinite(v["idio_vol_30d"]) and v["idio_vol_30d"] > 0:
                v["std_abs_idio_ret_20d"] = abs(s) / (v["idio_vol_30d"] * np.sqrt(20))

        # ── family 2: jumps over the last 20 raw returns ──
        r20 = _tail(r_all, c, 20)
        r20 = r20[np.isfinite(r20)]
        if len(r20) >= 15:
            sq = r20 ** 2
            tot = sq.sum()
            v["max_abs_ret_20d"] = float(np.abs(r20).max())
            if tot > 0:
                v["jump_share_20d"] = float(sq.max() / tot)
                v["semivar_balance_20d"] = float((sq[r20 < 0].sum() - sq[r20 > 0].sum()) / tot)

        # ── family 3: acceleration ──
        s5, s10 = _std(_tail(r_all, c, 5), 4), _std(_tail(r_all, c, 10), 8)
        s30, s60 = _std(_tail(r_all, c, 30), 20), _std(_tail(r_all, c, 60), 45)
        v["vol_log_ratio_5_30"] = _log_ratio(s5, s30)
        v["vol_log_ratio_10_60"] = _log_ratio(s10, s60)
        v["vol_30d_cut"] = s30
        if c >= 39:
            v["vol_change_20_20"] = _log_ratio(_std(r_all[c - 19: c + 1], 15),
                                               _std(r_all[c - 39: c - 19], 15))

        # ── family 7: decoupling from the LOO sector ──
        sec_all = panel.sec_loo[: c + 1, i]
        v["sector_corr_60d"] = _corr(_tail(r_all, c, 60), _tail(sec_all, c, 60), 45)
        c20 = _corr(_tail(r_all, c, 20), _tail(sec_all, c, 20), 15)
        c120 = _corr(_tail(r_all, c, 120), _tail(sec_all, c, 120), 90)
        v["sector_corr_change_20_120"] = c20 - c120 if np.isfinite(c20) and np.isfinite(c120) else np.nan

        # ── family 6: LOO peer volatility regime at row c ──
        for level in ("sub", "sec"):
            key = (i, level)
            if key not in peer_cache:
                peer_cache[key] = _peer_cols(panel, i, level)
            peers = peer_cache[key]
            p30 = vols["s30"][c, peers]
            p30f = p30[np.isfinite(p30)]
            med30 = float(np.median(p30f)) if len(p30f) >= MIN_PEERS else np.nan
            if level == "sec":
                v["sec_peer_med_vol_30d"] = med30
                continue
            p10 = vols["s10"][c, peers]
            p10f = p10[np.isfinite(p10)]
            v["sub_peer_med_vol_10d"] = float(np.median(p10f)) if len(p10f) >= MIN_PEERS else np.nan
            v["sub_peer_med_vol_30d"] = med30
            v["stock_vs_sub_vol_30d"] = _log_ratio(s30, med30)
            q = vols["s30_q80"][c, peers]
            both = np.isfinite(p30) & np.isfinite(q)
            if both.sum() >= MIN_PEERS:
                v["sub_peer_frac_high_vol"] = float((p30[both] > q[both]).mean())

        for k, val in v.items():
            out[row, pos[k]] = val

    return pd.DataFrame(out, index=events.index, columns=PRICE_FEATURE_COLUMNS)
