"""Model C refit — the labelled frame, built from the PRODUCTION event frame.

Rules: SPEC.md. Every function below is pure (frames in, frames out) so the leakage tests
can run them on synthetic data; `load_production_inputs()` is the only place that touches
the database or the parquet files, and it does so through the pipeline's own loaders.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import EXTREME_EARNINGS_REACTION_THRESHOLD
from feature_engineering.announcement_timing import TARGET_AVAILABLE, market_session_grid

TARGET = "abs_reaction_3d_anchored"
TARGET_STATUS = "reaction_3d_anchored_status"
MIN_PRIOR = 8
POP_YEARS = (2014, 2026)
VOL_ASOF_TOLERANCE = pd.Timedelta(days=7)
LOG_FLOOR = 1e-4
FULL_DF = "output/full_df.parquet"


# ───────────────────────────────────────── inputs ───────────────────────────────────────
def load_production_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(events, daily): the production event frame and the daily frame it was built from."""
    from pipeline.events import (build_event_frame, load_pipeline_active_stocks,
                                 load_pipeline_announcement_timing)
    daily = pd.read_parquet(FULL_DF)
    events = build_event_frame(daily, load_pipeline_announcement_timing(),
                               active_stocks=load_pipeline_active_stocks())
    return events, daily


# ───────────────────────────────────────── clocks ───────────────────────────────────────
def _exact_index(grid: np.ndarray, dates) -> np.ndarray:
    """Grid position of each date when it is a session, else -1."""
    d = pd.to_datetime(pd.Series(dates)).to_numpy(dtype="datetime64[ns]")
    pos = np.searchsorted(grid, d, side="left")
    ok = ~pd.isna(d) & (pos < len(grid))
    ok &= grid[np.minimum(pos, len(grid) - 1)] == d
    return np.where(ok, pos, -1).astype(np.int64)


def call_cutoff_index(grid: np.ndarray, report_dates) -> np.ndarray:
    """Grid position of the last session strictly before the Monday of the report week."""
    rd = pd.to_datetime(pd.Series(report_dates)).dt.normalize()
    monday = (rd - pd.to_timedelta(rd.dt.weekday, unit="D")).to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(grid, monday, side="left") - 1
    return np.where(pd.isna(monday), -1, idx).astype(np.int64)


def event_clocks(events: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(index=events.index)
    rd = pd.to_datetime(events["earnings_date"]).dt.normalize()
    out["report_date"] = rd
    out["call_monday"] = rd - pd.to_timedelta(rd.dt.weekday, unit="D")
    cut = call_cutoff_index(grid, rd.values)
    out["cutoff_idx"] = cut
    out["cutoff_date"] = pd.Series(
        np.where(cut >= 0, grid[np.maximum(cut, 0)], np.datetime64("NaT", "ns")),
        index=events.index).astype("datetime64[ns]")
    anchor = _exact_index(grid, events["anchor_date"].values)
    end = np.where(anchor >= 0, anchor + 3, -1)
    end = np.where(end < len(grid), end, -1)
    out["endpoint_idx"] = end
    out["endpoint_date"] = pd.Series(
        np.where(end >= 0, grid[np.maximum(end, 0)], np.datetime64("NaT", "ns")),
        index=events.index).astype("datetime64[ns]")
    return out


# ──────────────────────────────────────── outcomes ──────────────────────────────────────
def labelled_mask(events: pd.DataFrame) -> pd.Series:
    """Completed events whose corrected 3-session target is genuinely available."""
    return (~events["is_pending"].astype(bool)
            & events[TARGET_STATUS].eq(TARGET_AVAILABLE)
            & events[TARGET].notna())


def outcomes(events: pd.DataFrame) -> pd.DataFrame:
    ok = labelled_mask(events)
    out = pd.DataFrame(index=events.index)
    out["abs_r3"] = pd.to_numeric(events[TARGET], errors="coerce").where(ok)
    out["y"] = np.where(out["abs_r3"].notna(),
                        (out["abs_r3"] >= EXTREME_EARNINGS_REACTION_THRESHOLD).astype(float),
                        np.nan)
    return out


# ───────────────────────────────────── stock history ────────────────────────────────────
def history_features(events: pd.DataFrame, clocks: pd.DataFrame,
                     outs: pd.DataFrame) -> pd.DataFrame:
    """n_prior / hist_mean_abs from outcomes whose endpoint is on or before the cutoff.

    Prior outcomes are ordered by endpoint session — the order in which they became known —
    so the result does not depend on the order of the input rows.
    """
    cut = clocks["cutoff_idx"].to_numpy()
    end = clocks["endpoint_idx"].to_numpy()
    r3 = outs["abs_r3"].to_numpy(float)
    n = len(events)
    n_prior = np.zeros(n)
    mean_abs = np.full(n, np.nan)
    pos_of = pd.Series(np.arange(n), index=events.index)
    for _stock, ix in events.groupby("stock", sort=False).groups.items():
        rows = pos_of.loc[ix].to_numpy()
        have = rows[np.isfinite(r3[rows]) & (end[rows] >= 0)]
        order = have[np.argsort(end[have], kind="mergesort")]
        e_sorted, v_sorted = end[order], r3[order]
        csum = np.concatenate([[0.0], np.cumsum(v_sorted)])
        for p in rows:
            c = cut[p]
            if c < 0:
                continue
            k = int(np.searchsorted(e_sorted, c, side="right"))
            s, m = csum[k], k
            # own outcome can never be on or before own cutoff (asserted in build_frame);
            # excluded by identity regardless
            own = np.flatnonzero(order[:k] == p)
            if len(own):
                s -= v_sorted[own[0]]
                m -= 1
            n_prior[p] = m
            if m:
                mean_abs[p] = s / m
    return pd.DataFrame({"n_prior": n_prior, "hist_mean_abs": mean_abs}, index=events.index)


# ─────────────────────────────────────── vol at cutoff ──────────────────────────────────
def vol_at_cutoff(daily: pd.DataFrame, events: pd.DataFrame, clocks: pd.DataFrame,
                  tolerance: pd.Timedelta | None = VOL_ASOF_TOLERANCE) -> pd.DataFrame:
    """Production `vol_30d` on the stock's daily row at the cutoff (backward as-of)."""
    left = pd.DataFrame({"_row": np.arange(len(events)),
                         "stock": events["stock"].astype(str).to_numpy(),
                         "cutoff_date": clocks["cutoff_date"].to_numpy(dtype="datetime64[ns]")})
    left = left[left["cutoff_date"].notna()].sort_values("cutoff_date", kind="mergesort")
    right = daily[["stock", "date", "vol_30d"]].copy()
    right["stock"] = right["stock"].astype(str)
    right["date"] = pd.to_datetime(right["date"]).astype("datetime64[ns]")
    right = right.rename(columns={"date": "vol_date"}).sort_values("vol_date", kind="mergesort")
    m = pd.merge_asof(left, right, left_on="cutoff_date", right_on="vol_date", by="stock",
                      direction="backward",
                      tolerance=tolerance if tolerance is not None else pd.Timedelta(0))
    out = pd.DataFrame({"vol_30d": np.nan, "vol_date": pd.NaT}, index=np.arange(len(events)))
    out.loc[m["_row"].to_numpy(), "vol_30d"] = m["vol_30d"].to_numpy()
    out.loc[m["_row"].to_numpy(), "vol_date"] = m["vol_date"].to_numpy()
    out["vol_date"] = pd.to_datetime(out["vol_date"])
    out.index = events.index
    return out


# ────────────────────────────────────────── frame ───────────────────────────────────────
def build_frame(events: pd.DataFrame, daily: pd.DataFrame, *,
                vol_tolerance: pd.Timedelta | None = VOL_ASOF_TOLERANCE) -> pd.DataFrame:
    """One row per event of the production frame with clocks, outcome and features."""
    grid = market_session_grid(daily)
    clocks = event_clocks(events, grid)
    outs = outcomes(events)
    hist = history_features(events, clocks, outs)
    vol = vol_at_cutoff(daily, events, clocks, vol_tolerance)
    keep = ["stock", "earnings_date", "is_pending", "announce_window", TARGET_STATUS,
            "anchor_status", "anchor_date", "risk_score"]
    frame = pd.concat([events[[c for c in keep if c in events.columns]], clocks, outs,
                       hist, vol], axis=1)
    frame["year"] = frame["report_date"].dt.year
    frame["log_hist_mean_abs"] = np.log(frame["hist_mean_abs"].clip(lower=LOG_FLOOR))
    frame["log_vol_30d"] = np.log(pd.to_numeric(frame["vol_30d"]).clip(lower=LOG_FLOOR))

    lab = frame["y"].notna()
    # The rules every labelled row must satisfy (also covered by the tests).
    assert (frame.loc[lab, "endpoint_idx"] > frame.loc[lab, "cutoff_idx"]).all(), \
        "an event's own outcome endpoint is on or before its call cutoff"
    assert (frame.loc[lab, "cutoff_date"] < frame.loc[lab, "call_monday"]).all()
    has_vol = frame["vol_date"].notna()
    assert (frame.loc[has_vol, "vol_date"] <= frame.loc[has_vol, "cutoff_date"]).all(), \
        "vol_30d read from a row after the call cutoff"
    return frame


def population_mask(frame: pd.DataFrame, min_prior: int = MIN_PRIOR) -> pd.Series:
    return (frame["y"].notna() & frame["year"].between(*POP_YEARS)
            & frame["n_prior"].ge(min_prior) & frame["hist_mean_abs"].notna())


def common_mask(frame: pd.DataFrame, min_prior: int = MIN_PRIOR) -> pd.Series:
    """Rows where both B and C are defined — the identical-rows comparison sample."""
    return population_mask(frame, min_prior) & frame["vol_30d"].notna() & (frame["vol_30d"] > 0)


def exclusion_counts(frame: pd.DataFrame, min_prior: int = MIN_PRIOR) -> pd.DataFrame:
    """Sequential funnel from the whole event frame to the common sample."""
    steps = []
    m = pd.Series(True, index=frame.index)
    steps.append(("event frame (all rows)", m.sum()))
    m &= ~frame["is_pending"].astype(bool)
    steps.append(("completed (not pending)", m.sum()))
    status = frame[TARGET_STATUS]
    for s in sorted(status[m & status.ne(TARGET_AVAILABLE)].unique()):
        steps.append((f"  dropped: target {s}", int((m & status.eq(s)).sum())))
    m &= frame["y"].notna()
    steps.append(("corrected 3-session target available", m.sum()))
    m2 = m & frame["year"].between(*POP_YEARS)
    steps.append((f"  dropped: report year outside {POP_YEARS[0]}-{POP_YEARS[1]}",
                  int((m & ~frame["year"].between(*POP_YEARS)).sum())))
    m = m2
    steps.append((f"report year {POP_YEARS[0]}-{POP_YEARS[1]}", m.sum()))
    short = m & frame["n_prior"].lt(min_prior)
    steps.append((f"  dropped: fewer than {min_prior} usable prior outcomes", int(short.sum())))
    m &= frame["n_prior"].ge(min_prior)
    steps.append(("population (B defined)", m.sum()))
    novol = m & ~(frame["vol_30d"].notna() & (frame["vol_30d"] > 0))
    steps.append(("  dropped: vol_30d unavailable at cutoff", int(novol.sum())))
    m &= ~novol
    steps.append(("common sample (B and C defined)", m.sum()))
    return pd.DataFrame(steps, columns=["step", "events"])
