"""Price panel on the market-session grid, event cutoffs, and leave-one-out benchmarks.

Research-only (Phase 5B). Reads a daily (stock, date, price, sector, sub_sector) frame and
the Phase 3 event frame; writes nothing and touches no database.

Three conventions carried by everything downstream:

* **The market-session grid** (`feature_engineering.announcement_timing.market_session_grid`)
  is the only unit of time. A return is `P_t / P_{t-1} - 1` where both are consecutive
  GRID sessions; a ticker missing a session gets NaN there, never a silently stretched
  multi-session return. This is the same refusal the Phase 2 anchored target makes.
* **The cutoff** of an event is the last session strictly before `min(earnings_date,
  phase3_proxy_session_date)`. For BMO it equals the anchor; for AMC it is one session more
  conservative. The day-D return is excluded whatever the window, which is what keeps a
  BMO reaction out of every feature.
* **Leave-one-out is by ISSUER**, not ticker: GOOG/GOOGL, FOX/FOXA and NWS/NWSA are one
  company, and leaving out only one share class would leave the stock's own return in its
  benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from feature_engineering.announcement_timing import market_session_grid

SHARE_CLASS_ISSUER = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA"}
BENCHMARK_CLIP = 0.50


def issuer_of(stock: str) -> str:
    s = str(stock)
    return SHARE_CLASS_ISSUER.get(s, s)


@dataclass
class PricePanel:
    """Wide grid-aligned returns plus the classification needed for peer groups."""
    grid: np.ndarray                 # datetime64[ns], sorted sessions
    stocks: np.ndarray               # column order of `ret`
    ret: np.ndarray                  # (T, N) float, NaN where not a consecutive-session return
    sector: np.ndarray               # (N,) object
    sub_sector: np.ndarray           # (N,) object
    issuer: np.ndarray               # (N,) object
    mkt_loo: np.ndarray              # (T, N) LOO-by-issuer equal-weight market return
    sec_loo: np.ndarray              # (T, N) LOO-by-issuer equal-weight sector return

    @property
    def col(self) -> dict[str, int]:
        return {s: i for i, s in enumerate(self.stocks)}


def _group_loo(values: np.ndarray, groups: np.ndarray, issuers: np.ndarray) -> np.ndarray:
    """Equal-weight mean of every OTHER issuer in the same group, per (session, stock).

    `groups` may be a single constant (the market). Sums and counts are taken over the
    group and over the stock's issuer; the issuer's contribution is subtracted from both.
    """
    finite = np.isfinite(values)
    v0 = np.where(finite, values, 0.0)
    n0 = finite.astype(float)
    out = np.full(values.shape, np.nan)
    for g in pd.unique(groups):
        in_g = groups == g
        g_sum = v0[:, in_g].sum(axis=1)
        g_n = n0[:, in_g].sum(axis=1)
        for iss in pd.unique(issuers[in_g]):
            cols = np.where(in_g & (issuers == iss))[0]
            i_sum = v0[:, cols].sum(axis=1)
            i_n = n0[:, cols].sum(axis=1)
            other_n = g_n - i_n
            with np.errstate(invalid="ignore", divide="ignore"):
                loo = np.where(other_n > 0, (g_sum - i_sum) / other_n, np.nan)
            out[:, cols] = loo[:, None]
    return out


def build_panel(daily: pd.DataFrame) -> PricePanel:
    """Grid-aligned return panel. `daily` needs stock, date, price, sector, sub_sector."""
    d = daily[["stock", "date", "price", "sector", "sub_sector"]].copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    grid = market_session_grid(d)
    wide = (d.drop_duplicates(["stock", "date"], keep="last")
              .pivot(index="date", columns="stock", values="price")
              .reindex(pd.DatetimeIndex(grid)))
    prices = wide.to_numpy(float)
    ret = np.full(prices.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ret[1:] = prices[1:] / prices[:-1] - 1.0
    ret[~np.isfinite(ret)] = np.nan

    stocks = wide.columns.to_numpy(dtype=object)
    meta = d.drop_duplicates("stock", keep="last").set_index("stock")
    sector = meta["sector"].reindex(stocks).astype(object).to_numpy()
    sub_sector = meta["sub_sector"].reindex(stocks).astype(object).to_numpy()
    issuer = np.array([issuer_of(s) for s in stocks], dtype=object)

    clipped = np.clip(ret, -BENCHMARK_CLIP, BENCHMARK_CLIP)
    mkt = _group_loo(clipped, np.zeros(len(stocks), dtype=int), issuer)
    sec = _group_loo(clipped, sector, issuer)
    return PricePanel(grid=grid, stocks=stocks, ret=ret, sector=sector,
                      sub_sector=sub_sector, issuer=issuer, mkt_loo=mkt, sec_loo=sec)


def event_cutoff_date(events: pd.DataFrame) -> pd.Series:
    """`min(earnings_date, phase3_proxy_session_date)` — the earliest the news can be."""
    ed = pd.to_datetime(events["earnings_date"]).dt.normalize()
    if "phase3_proxy_session_date" in events.columns:
        proxy = pd.to_datetime(events["phase3_proxy_session_date"]).dt.normalize()
        return ed.where(proxy.isna() | (ed <= proxy), proxy)
    return ed


def event_cutoff_index(events: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Grid index of the last session STRICTLY before the event's earliest possible date.

    -1 where no such session exists. Never the day-D session itself: for a BMO reporter
    that session's return is the reaction.
    """
    d = event_cutoff_date(events).to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(grid, d, side="left") - 1
    idx = np.where(pd.isna(d), -1, idx)
    return idx.astype(int)
