"""Causal, prediction-time features for the P3.2/P3.3 refit.

Research-only. Reads the Phase 3 event frame and the daily frame; writes nothing and
touches no database. See PREREGISTRATION.md for the definitions this implements.

One availability rule governs every history lookup here:

    an outcome of event j may inform event i only if j's endpoint session
    (anchor + k market sessions) is at or before i's cutoff session.

It covers the stock's own history, the market-wide prior and the lift prior alike, so a
same-day or same-week outcome can never leak into a score, whatever the frame order.

Two cutoffs are built for every event:

* **call** — the last session strictly before the Monday of the report week. The weekly
  run scores from that close (`score_asof_date` on every 0.3.1 prediction row is the
  Friday before the report week). This is what Breakwater can actually know.
* **eve** — the last session strictly before the report date (Phase 5B's cutoff). Used
  only to measure how much a research result depends on information the product would
  not have had.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import EXTREME_EARNINGS_REACTION_THRESHOLD
from feature_engineering.announcement_timing import AMC, BMO, TARGET_AVAILABLE, market_session_grid
from feature_engineering.event_features import reaction_entropy

CORRECTED_TARGET = "abs_reaction_3d_anchored"
ROLLING_P75_WINDOW = 28
ASOF_TOLERANCE = pd.Timedelta(days=7)
CUTOFFS = ("call", "eve")


# ───────────────────────────────────────── clocks ───────────────────────────────────────
def report_date(events: pd.DataFrame) -> pd.Series:
    """min(earnings_date, phase3_proxy_session_date): the earliest the news can be."""
    ed = pd.to_datetime(events["earnings_date"]).dt.normalize().astype("datetime64[ns]")
    if "phase3_proxy_session_date" in events.columns:
        proxy = pd.to_datetime(events["phase3_proxy_session_date"]).dt.normalize()
        proxy = proxy.astype("datetime64[ns]")
        return ed.where(proxy.isna() | (ed <= proxy), proxy)
    return ed


def _last_session_before(grid: np.ndarray, dates: pd.Series) -> np.ndarray:
    """Grid index of the last session strictly before each date; -1 where none."""
    d = dates.to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(grid, d, side="left") - 1
    return np.where(pd.isna(d), -1, idx).astype(np.int64)


def _exact_index(grid: np.ndarray, dates: pd.Series) -> np.ndarray:
    """Grid index of each date when it is a session; -1 otherwise."""
    d = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(grid, d, side="left")
    ok = (~pd.isna(d)) & (pos < len(grid))
    ok &= grid[np.minimum(pos, len(grid) - 1)] == d
    return np.where(ok, pos, -1).astype(np.int64)


def event_clocks(events: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Cutoffs, anchors and outcome endpoints on the market-session grid."""
    rd = report_date(events)
    monday = rd - pd.to_timedelta(rd.dt.weekday, unit="D")
    out = pd.DataFrame(index=events.index)
    out["report_date"] = rd
    out["call_monday"] = monday
    out["call_cutoff_idx"] = _last_session_before(grid, monday)
    out["eve_cutoff_idx"] = _last_session_before(grid, rd)
    for name in CUTOFFS:
        idx = out[f"{name}_cutoff_idx"].to_numpy()
        out[f"{name}_cutoff_date"] = pd.Series(
            np.where(idx >= 0, grid[np.maximum(idx, 0)], np.datetime64("NaT", "ns")),
            index=events.index).astype("datetime64[ns]")

    anchor = _exact_index(grid, events["anchor_date"]) if "anchor_date" in events else \
        np.full(len(events), -1)
    out["anchor_idx"] = anchor
    for k in (1, 3):
        end = np.where(anchor >= 0, anchor + k, -1)
        out[f"endpoint{k}_idx"] = np.where(end < len(grid), end, -1)
    e3 = out["endpoint3_idx"].to_numpy()
    out["endpoint3_date"] = pd.Series(
        np.where(e3 >= 0, grid[np.maximum(e3, 0)], np.datetime64("NaT", "ns")),
        index=events.index).astype("datetime64[ns]")
    return out


def outcome_frame(events: pd.DataFrame) -> pd.DataFrame:
    """y and the usable reaction magnitudes. Unresolved stays NaN — never 0."""
    completed = ~events["is_pending"].astype(bool)
    ok3 = completed & events["reaction_3d_anchored_status"].eq(TARGET_AVAILABLE) \
        & events[CORRECTED_TARGET].notna()
    ok1 = completed & events["reaction_1d_anchored_status"].eq(TARGET_AVAILABLE) \
        & events["reaction_1d_anchored"].notna()
    out = pd.DataFrame(index=events.index)
    out["abs_r3"] = pd.to_numeric(events[CORRECTED_TARGET], errors="coerce").where(ok3)
    out["abs_r1"] = pd.to_numeric(events["reaction_1d_anchored"], errors="coerce").abs().where(ok1)
    out["y_extreme"] = np.where(
        out["abs_r3"].notna(),
        (out["abs_r3"] >= EXTREME_EARNINGS_REACTION_THRESHOLD).astype(float), np.nan)
    return out


# ─────────────────────────────────── own-stock history ──────────────────────────────────
def _p75(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    tail = values[-ROLLING_P75_WINDOW:] if len(values) >= ROLLING_P75_WINDOW else values
    return float(np.quantile(tail, 0.75))


def history_features(events: pd.DataFrame, clocks: pd.DataFrame, outcomes: pd.DataFrame,
                     cutoff: str) -> pd.DataFrame:
    """Stock-history statistics from outcomes observable at `cutoff`.

    Prior outcomes are ordered by endpoint session, the order in which they became known.
    An event can never see itself: its own endpoint is always after its own cutoff, and
    it is additionally excluded by position.
    """
    cut = clocks[f"{cutoff}_cutoff_idx"].to_numpy()
    end3 = clocks["endpoint3_idx"].to_numpy()
    end1 = clocks["endpoint1_idx"].to_numpy()
    r3 = outcomes["abs_r3"].to_numpy(float)
    r1 = outcomes["abs_r1"].to_numpy(float)
    # 0.3.1 entropy: 3-session magnitude, falling back to the 1-session one per event.
    ent_val = np.where(np.isfinite(r3), r3, r1)
    ent_end = np.where(np.isfinite(r3), end3, end1)

    n = len(events)
    cols = {c: np.full(n, np.nan) for c in
            ("n_prior", "hist_mean_abs", "hist_p75", "hist_entropy", "n_prior_extreme")}
    pos_of = {ix: p for p, ix in enumerate(events.index)}
    for _stock, sub in events.groupby("stock", sort=False):
        rows = np.array([pos_of[ix] for ix in sub.index])
        have3 = rows[np.isfinite(r3[rows]) & (end3[rows] >= 0)]
        o3 = have3[np.argsort(end3[have3], kind="mergesort")]
        e3, v3 = end3[o3], r3[o3]
        haveE = rows[np.isfinite(ent_val[rows]) & (ent_end[rows] >= 0)]
        oE = haveE[np.argsort(ent_end[haveE], kind="mergesort")]
        eE, vE = ent_end[oE], ent_val[oE]
        for p in rows:
            c = cut[p]
            if c < 0:
                continue
            k = np.searchsorted(e3, c, side="right")
            prior = v3[:k][o3[:k] != p]
            cols["n_prior"][p] = len(prior)
            if len(prior):
                cols["hist_mean_abs"][p] = float(prior.mean())
                cols["hist_p75"][p] = _p75(prior)
                cols["n_prior_extreme"][p] = float(
                    (prior >= EXTREME_EARNINGS_REACTION_THRESHOLD).sum())
            kE = np.searchsorted(eE, c, side="right")
            priorE = vE[:kE][oE[:kE] != p]
            if len(priorE) >= 8:
                cols["hist_entropy"][p] = float(reaction_entropy(pd.Series(priorE)))
    return pd.DataFrame({f"{k}_{cutoff}": v for k, v in cols.items()}, index=events.index)


def market_prior(clocks: pd.DataFrame, outcomes: pd.DataFrame, cutoff: str,
                 mask: pd.Series | None = None) -> pd.Series:
    """Market-wide P(extreme) over every outcome whose endpoint is <= the cutoff."""
    y = outcomes["y_extreme"]
    ok = y.notna() & clocks["endpoint3_idx"].ge(0)
    if mask is not None:
        ok &= mask
    end = clocks.loc[ok, "endpoint3_idx"].to_numpy()
    order = np.argsort(end, kind="mergesort")
    end = end[order]
    cum = np.concatenate([[0.0], np.cumsum(y[ok].to_numpy(float)[order])])
    k = np.searchsorted(end, clocks[f"{cutoff}_cutoff_idx"].to_numpy(), side="right")
    with np.errstate(invalid="ignore", divide="ignore"):
        prior = np.where(k > 0, cum[k] / np.maximum(k, 1), np.nan)
    return pd.Series(prior, index=clocks.index)


def stock_bucket_lift(events: pd.DataFrame, clocks: pd.DataFrame, outcomes: pd.DataFrame,
                      bucket: pd.Series, prior: pd.Series, *, strength: float,
                      cutoff: str = "call") -> pd.Series:
    """0.3.1's P(extreme | stock, bucket) / P(extreme | market), endpoint-safe.

    Only the stock's prior events in the SAME bucket whose outcome endpoint is <= this
    event's cutoff count. `prior` is the endpoint-safe market rate. No prior data -> 1.0,
    matching 0.3.1's "no opinion".
    """
    cut = clocks[f"{cutoff}_cutoff_idx"].to_numpy()
    end3 = clocks["endpoint3_idx"].to_numpy()
    y = outcomes["y_extreme"].to_numpy(float)
    b = bucket.astype(object).to_numpy()
    g = prior.to_numpy(float)
    lift = np.ones(len(events))
    pos_of = {ix: p for p, ix in enumerate(events.index)}
    for _stock, sub in events.groupby("stock", sort=False):
        rows = np.array([pos_of[ix] for ix in sub.index])
        have = rows[np.isfinite(y[rows]) & (end3[rows] >= 0) & pd.notna(b[rows])]
        for p in rows:
            if not np.isfinite(g[p]) or g[p] <= 0 or pd.isna(b[p]):
                continue
            j = have[(end3[have] <= cut[p]) & (b[have] == b[p]) & (have != p)]
            shrunk = (y[j].sum() + strength * g[p]) / (len(j) + strength)
            lift[p] = shrunk / g[p]
    return pd.Series(lift, index=events.index)


# ─────────────────────────────────── daily-frame lookups ────────────────────────────────
def daily_asof(daily: pd.DataFrame, clocks: pd.DataFrame, events: pd.DataFrame,
               cutoff: str, cols: tuple[str, ...] = ("vol_30d", "drift_30d")) -> pd.DataFrame:
    """The stock's daily-row values at the cutoff date (backward as-of, 7 days max).

    Production's pending row is the stock's last daily row and carries exactly these
    columns (`vol_30d` is `rolling(30).std().shift(1)`), so this reproduces what the live
    call is scored from — not a re-implementation of it.
    """
    left = pd.DataFrame({
        "_row": np.arange(len(events)),
        "stock": events["stock"].astype(str).to_numpy(),
        "date": clocks[f"{cutoff}_cutoff_date"].to_numpy(dtype="datetime64[ns]"),
    })
    have = left["date"].notna()
    right = daily[["stock", "date", *cols]].copy()
    right["stock"] = right["stock"].astype(str)
    right["date"] = pd.to_datetime(right["date"]).astype("datetime64[ns]")
    right = right.sort_values("date", kind="mergesort")
    merged = pd.merge_asof(
        left[have].sort_values("date", kind="mergesort"), right, on="date", by="stock",
        direction="backward", tolerance=ASOF_TOLERANCE)
    out = pd.DataFrame(np.nan, index=np.arange(len(events)), columns=list(cols))
    out.loc[merged["_row"].to_numpy(), list(cols)] = merged[list(cols)].to_numpy()
    out.index = events.index
    return out.add_suffix(f"_{cutoff}")


def phase5b_features_at(events: pd.DataFrame, clocks: pd.DataFrame, daily: pd.DataFrame,
                        cutoff: str) -> pd.DataFrame:
    """Phase 5B's idio vol and sector peer reaction level, with the cutoff moved.

    Phase 5B takes the cutoff as the last session strictly before
    min(earnings_date, proxy). For the call cutoff the query frame's earnings_date is set
    to the Monday of the report week and the proxy removed; the peer source rows (anchor
    dates, 1-session reactions) are left untouched, so peer availability is still decided
    by each peer's own endpoint.
    """
    from research.phase_5b_new_feature_testing.panel import build_panel
    from research.phase_5b_new_feature_testing.peer_features import compute_peer_features
    from research.phase_5b_new_feature_testing.price_features import compute_price_features

    query = events.copy()
    if cutoff == "call":
        query["earnings_date"] = clocks["call_monday"]
        query["phase3_proxy_session_date"] = pd.NaT
    panel = build_panel(daily[["stock", "date", "price", "sector", "sub_sector"]])
    price = compute_price_features(panel, query)
    peer = compute_peer_features(query, panel.grid)
    got = pd.concat([price[["idio_vol_30d"]], peer[["peer_sec_rw_mean_abs_1d_20"]]], axis=1)
    # the cutoff Phase 5B actually used must be the one this module computed
    used = price["p5b_cutoff_session"].to_numpy()
    want = clocks[f"{cutoff}_cutoff_idx"].to_numpy()
    both = np.isfinite(used) & (want >= 0)
    if not np.array_equal(used[both].astype(np.int64), want[both]):
        raise AssertionError(f"Phase 5B cutoff disagrees with the {cutoff} cutoff")
    return got.add_suffix(f"_{cutoff}")


# ───────────────────────────────────────── driver ───────────────────────────────────────
def build_feature_frame(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Every event of the Phase 3 frame, with outcomes, clocks and both cutoffs' features."""
    grid = market_session_grid(daily)
    clocks = event_clocks(events, grid)
    outcomes = outcome_frame(events)
    parts = [clocks, outcomes]
    for cutoff in CUTOFFS:
        parts.append(history_features(events, clocks, outcomes, cutoff))
        parts.append(daily_asof(daily, clocks, events, cutoff))
        parts.append(phase5b_features_at(events, clocks, daily, cutoff))
        parts.append(market_prior(clocks, outcomes, cutoff).rename(f"market_prior_{cutoff}"))
    keep = ["stock", "sector", "sub_sector", "earnings_date", "is_pending", "announce_window",
            "phase3_announce_date", "risk_score", "earnings_explosiveness_bucket",
            "earnings_explosiveness_bucket_structural", "announce_ts_source"]
    base = events[[c for c in keep if c in events.columns]].copy()
    out = pd.concat([base, *parts], axis=1)
    out["year"] = out["report_date"].dt.year
    out["is_bmo"] = out["announce_window"].eq(BMO).astype(float)
    out["window_ok"] = out["announce_window"].isin([BMO, AMC])
    return out
