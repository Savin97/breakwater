"""Family 5: causal peer earnings shocks.

"Did companies like this one just react violently to THEIR earnings?" — measured only
from peer outcomes that were already observable when this event's features were frozen.

Availability is an ENDPOINT rule, not a report-date rule. A peer's 1-session anchored
reaction is `close(anchor + 1) / close(anchor) - 1`, so it exists only after the close of
session `anchor + 1`. It may feed event e only if that endpoint session is at or before
e's cutoff session c (the last session strictly before e's report date). Consequences the
tests pin:

* a peer reporting after the close on e's cutoff day is NOT usable (its endpoint is D);
* a same-day peer is never usable (its endpoint is after c by construction);
* a later-reporting peer can never reach an earlier event;
* an unresolved peer reaction is absent — excluded from numerator AND count, never 0.

Peer groups are today's sector / sub-sector (`stock_data`), constant per ticker in the
data: **not point-in-time**. Share-class twins are one issuer, counted once and never a
peer of themselves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import EXTREME_EARNINGS_REACTION_THRESHOLD, LARGE_EARNINGS_REACTION_THRESHOLD
from feature_engineering.announcement_timing import TARGET_AVAILABLE
from phase_5b_new_feature_testing.panel import (
    event_cutoff_date,
    event_cutoff_index,
    issuer_of,
)

WINDOWS = (10, 20, 40)
PRIMARY_WINDOW = 20
RECENCY_HALFLIFE = 10.0
SHOCK_OFFSET = 0.005
SHOCK_MIN_PRIOR = 4
LOCAL_MIN_SUB_PEERS = 2
LEVELS = ("sub", "sec")
REACTION_1D = "reaction_1d_anchored"
STATUS_1D = "reaction_1d_anchored_status"


def peer_event_table(events: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Every completed event with an AVAILABLE anchored 1-session reaction.

    Adds `endpoint_idx` (grid index of anchor + 1 — when the value becomes known) and
    `prior_mean_abs_1d` (the peer's own mean |r1d| over its strictly PRIOR resolved events,
    required ≥ SHOCK_MIN_PRIOR), the denominator of the peer shock.
    """
    ok = (~events["is_pending"].astype(bool)
          & events[STATUS_1D].eq(TARGET_AVAILABLE)
          & pd.to_numeric(events[REACTION_1D], errors="coerce").notna())
    p = events.loc[ok, ["stock", "sector", "sub_sector", "anchor_date", REACTION_1D]].copy()
    p["anchor_date"] = pd.to_datetime(p["anchor_date"]).dt.normalize()
    a = np.searchsorted(grid, p["anchor_date"].to_numpy(dtype="datetime64[ns]"), side="left")
    on_grid = (a < len(grid)) & (grid[np.minimum(a, len(grid) - 1)]
                                 == p["anchor_date"].to_numpy(dtype="datetime64[ns]"))
    p = p[on_grid & (a + 1 < len(grid))].copy()
    p["anchor_idx"] = a[on_grid & (a + 1 < len(grid))]
    p["endpoint_idx"] = p["anchor_idx"] + 1
    p["abs_r1d"] = pd.to_numeric(p[REACTION_1D]).abs()
    p["issuer"] = p["stock"].astype(str).map(issuer_of)

    p = p.sort_values(["stock", "anchor_idx"], kind="mergesort")
    g = p.groupby("stock", sort=False)["abs_r1d"]
    p["prior_n"] = g.cumcount()
    p["prior_mean_abs_1d"] = g.transform(lambda s: s.shift(1).expanding().mean())
    p.loc[p["prior_n"] < SHOCK_MIN_PRIOR, "prior_mean_abs_1d"] = np.nan
    p["shock_log"] = np.log((p["abs_r1d"] + SHOCK_OFFSET)
                            / (p["prior_mean_abs_1d"] + SHOCK_OFFSET))
    # One observation per issuer per announcement: GOOG and GOOGL are one company.
    p = p.drop_duplicates(["issuer", "anchor_idx"], keep="first")
    return p.sort_values("endpoint_idx", kind="mergesort").reset_index(drop=True)


def unresolved_event_table(events: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """Completed events WITHOUT an available 1-session reaction, placed at report session."""
    bad = (~events["is_pending"].astype(bool) & ~events[STATUS_1D].eq(TARGET_AVAILABLE))
    u = events.loc[bad, ["stock", "sector", "sub_sector"]].copy()
    d = event_cutoff_date(events.loc[bad]).to_numpy(dtype="datetime64[ns]")
    u["report_idx"] = np.searchsorted(grid, d, side="left")
    u["issuer"] = u["stock"].astype(str).map(issuer_of)
    return u.drop_duplicates(["issuer", "report_idx"]).sort_values("report_idx")


def _stats(ar: np.ndarray, end: np.ndarray, shock: np.ndarray, c: int,
           prefix: str, w: int) -> dict:
    n = len(ar)
    out = {f"{prefix}_n_{w}": float(n)}
    if n == 0:
        return out                                   # aggregates stay NaN, never 0
    wts = 0.5 ** ((c - end) / RECENCY_HALFLIFE)
    shock = shock[np.isfinite(shock)]
    out.update({
        f"{prefix}_mean_abs_1d_{w}": float(ar.mean()),
        f"{prefix}_max_abs_1d_{w}": float(ar.max()),
        f"{prefix}_frac_large_1d_{w}": float((ar >= LARGE_EARNINGS_REACTION_THRESHOLD).mean()),
        f"{prefix}_frac_extreme_1d_{w}": float((ar >= EXTREME_EARNINGS_REACTION_THRESHOLD).mean()),
        f"{prefix}_rw_mean_abs_1d_{w}": float(np.sum(wts * ar) / np.sum(wts)),
        f"{prefix}_shock_{w}": float(shock.mean()) if len(shock) else np.nan,
    })
    return out


def _arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"end": df["endpoint_idx"].to_numpy(np.int64),
            "ar": df["abs_r1d"].to_numpy(float),
            "shock": df["shock_log"].to_numpy(float),
            "issuer": df["issuer"].to_numpy(object),
            "sector": df["sector"].to_numpy(object)}


def compute_peer_features(events: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    """One row per event (index preserved) with every family-5 column."""
    peers = peer_event_table(events, grid)
    unres = unresolved_event_table(events, grid)
    cut = event_cutoff_index(events, grid)
    issuers = events["stock"].astype(str).map(issuer_of).to_numpy()
    groups = {"sub": events["sub_sector"].to_numpy(object),
              "sec": events["sector"].to_numpy(object)}
    gcol = {"sub": "sub_sector", "sec": "sector"}
    by_group = {lvl: {g: _arrays(df) for g, df in peers.groupby(gcol[lvl])} for lvl in LEVELS}
    unres_by_group = {lvl: {g: (df["report_idx"].to_numpy(np.int64), df["issuer"].to_numpy(object))
                            for g, df in unres.groupby(gcol[lvl])} for lvl in LEVELS}
    everything = _arrays(peers)
    empty = _arrays(peers.iloc[0:0])
    wmax = max(WINDOWS)
    pw = PRIMARY_WINDOW

    rows = []
    for k in range(len(events)):
        c = int(cut[k])
        rec: dict = {}
        if c < 0:
            rows.append(rec)
            continue
        for lvl in LEVELS:
            g = groups[lvl][k]
            gp = by_group[lvl].get(g, empty)
            lo = np.searchsorted(gp["end"], c - wmax + 1, side="left")
            hi = np.searchsorted(gp["end"], c, side="right")   # endpoint <= c: observable
            keep = gp["issuer"][lo:hi] != issuers[k]
            end = gp["end"][lo:hi][keep]
            ar = gp["ar"][lo:hi][keep]
            sh = gp["shock"][lo:hi][keep]
            for w in WINDOWS:
                m = end >= c - w + 1
                rec.update(_stats(ar[m], end[m], sh[m], c, f"peer_{lvl}", w))
            gu = unres_by_group[lvl].get(g)
            n_unres = 0.0
            if gu is not None:
                r, iss = gu
                n_unres = float(((r >= c - pw + 1) & (r <= c) & (iss != issuers[k])).sum())
            rec[f"peer_{lvl}_n_unresolved_{pw}"] = n_unres

        # market-wide earnings environment outside the stock's own sector (control)
        lo = np.searchsorted(everything["end"], c - pw + 1, side="left")
        hi = np.searchsorted(everything["end"], c, side="right")
        m = everything["sector"][lo:hi] != groups["sec"][k]
        mk = everything["ar"][lo:hi][m]
        rec[f"peer_mkt_exsec_mean_abs_1d_{pw}"] = float(mk.mean()) if len(mk) else np.nan

        # local = sub-sector where it has enough usable peers, else sector (secondary)
        src = "sub" if rec.get(f"peer_sub_n_{pw}", 0.0) >= LOCAL_MIN_SUB_PEERS else "sec"
        for stat in ("n", "mean_abs_1d", "shock", "frac_large_1d"):
            rec[f"peer_local_{stat}_{pw}"] = rec.get(f"peer_{src}_{stat}_{pw}", np.nan)
        rows.append(rec)

    return pd.DataFrame(rows, index=events.index).reindex(columns=peer_feature_columns())


def peer_feature_columns() -> list[str]:
    """Fixed output schema, whether or not any event had a usable peer."""
    stats = ("n", "mean_abs_1d", "max_abs_1d", "frac_large_1d", "frac_extreme_1d",
             "rw_mean_abs_1d", "shock")
    cols = [f"peer_{lvl}_{st}_{w}" for lvl in LEVELS for w in WINDOWS for st in stats]
    cols += [f"peer_{lvl}_n_unresolved_{PRIMARY_WINDOW}" for lvl in LEVELS]
    cols += [f"peer_mkt_exsec_mean_abs_1d_{PRIMARY_WINDOW}"]
    cols += [f"peer_local_{st}_{PRIMARY_WINDOW}" for st in ("n", "mean_abs_1d", "shock",
                                                            "frac_large_1d")]
    return cols
