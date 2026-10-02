"""Vendor-neutral rules for the confirmatory options test. Definitions: PREREGISTRATION.md.

No vendor code and no data here: an ingestion adapter (written once a source is chosen)
turns a vendor's files into the two frames these functions take:

* snapshots: one row per available snapshot for a ticker — `snapshot_ts` (tz-aware,
  America/New_York), `spot` (the source's underlying price at that snapshot, nominal),
  `spot_field` (which vendor field it came from), `source` (vendor/product/file id);
* chain: the contracts of ONE snapshot — `expiration`, `strike`, `call_put`
  ('Call'/'Put'), `bid`, `ask`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.options_pilot.features import (
    RULE_ANNOUNCEMENT,
    RULE_STRICT_AFTER,
    is_covering,
)

NY = "America/New_York"
CLOSE = pd.Timedelta(hours=16)
MAX_AGE_SESSIONS = 1
MAX_STRIKE_VS_SPOT = 0.05
REQUIRED_FIELDS = (
    "event_id", "call_cutoff_date", "prediction_cutoff_ts", "snapshot_ts", "snapshot_age",
    "expiration", "strike", "call_bid", "call_ask", "call_mid", "put_bid", "put_ask",
    "put_mid", "spot", "spot_field", "expected_move_pct", "denominator", "source", "rule",
)


def prediction_cutoff_ts(call_cutoff_date) -> pd.Timestamp:
    """The call-cutoff session's date at 16:00 New York time."""
    return (pd.Timestamp(call_cutoff_date).normalize() + CLOSE).tz_localize(NY)


def select_snapshot(snapshots: pd.DataFrame, call_cutoff_date, grid: np.ndarray) -> dict | None:
    """Latest snapshot with snapshot_ts <= the prediction cutoff, from the cutoff session
    or at most MAX_AGE_SESSIONS sessions before it. None otherwise — never later data."""
    cut_ts = prediction_cutoff_ts(call_cutoff_date)
    s = snapshots[snapshots["snapshot_ts"] <= cut_ts]
    if s.empty:
        return None
    best = s.sort_values("snapshot_ts", kind="mergesort").iloc[-1]
    gridx = pd.DatetimeIndex(grid)
    cut_i = gridx.searchsorted(pd.Timestamp(call_cutoff_date).normalize(), side="right") - 1
    snap_day = best["snapshot_ts"].tz_convert(NY).tz_localize(None).normalize()
    snap_i = gridx.searchsorted(snap_day, side="right") - 1
    age = int(cut_i - snap_i)
    if age > MAX_AGE_SESSIONS:
        return None
    assert best["snapshot_ts"] <= cut_ts
    return {**best.to_dict(), "snapshot_age": age, "prediction_cutoff_ts": cut_ts}


def clean(chain: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pre-registered quote filters, cumulative, with removal counts."""
    keep = pd.Series(True, index=chain.index)
    removed = {}
    for name, m in (("bid_ask_present", chain["bid"].notna() & chain["ask"].notna()),
                    ("bid_nonnegative", chain["bid"] >= 0),
                    ("ask_positive", chain["ask"] > 0),
                    ("ask_ge_bid", chain["ask"] >= chain["bid"])):
        new = keep & m.fillna(False)
        removed[name] = int((keep & ~new).sum())
        keep = new
    q = chain[keep].copy()
    q["mid"] = (q["bid"] + q["ask"]) / 2.0
    return q, removed


def expected_move(chain: pd.DataFrame, snap: dict, report_date, window: str,
                  rule: str = RULE_ANNOUNCEMENT) -> dict:
    """Spot-denominated expected move for one event, or a status explaining why not."""
    out = {"rule": rule, "denominator": "spot"}
    dup = int(chain.duplicated(["expiration", "strike", "call_put"]).sum())
    if dup:
        # Non-standard (adjusted) contracts or a bad merge; the adapter must resolve them.
        raise ValueError(f"{dup} duplicate (expiration, strike, call_put) rows in one chain")
    q, removed = clean(chain)
    out.update({f"removed_{k}": v for k, v in removed.items()})
    spot = snap.get("spot")
    if spot is None or not np.isfinite(spot) or spot <= 0:
        return {**out, "status": "no_spot"}
    snap_day = pd.Timestamp(snap["snapshot_ts"]).tz_convert(NY).tz_localize(None).normalize()
    # Covering expiries come from the RAW chain (as in the pilot): if the nearest one's
    # quotes all fail the filters the event gets no feature — never a later expiry.
    exps = sorted(chain["expiration"].unique())
    cov = [e for e in exps if is_covering(e, snap_day, report_date, window, rule)]
    if not cov:
        return {**out, "status": "no_covering_expiry"}
    near = cov[0]
    ex = q[q["expiration"] == near]
    c = ex[ex["call_put"] == "Call"].set_index("strike")
    p = ex[ex["call_put"] == "Put"].set_index("strike")
    ks = c.index.intersection(p.index)
    if len(ks) == 0:
        return {**out, "status": "no_pair", "expiration": pd.Timestamp(near)}
    cand = pd.DataFrame({"k": ks.to_numpy(dtype=float),
                         "dist": np.abs(ks.to_numpy(dtype=float) - spot),
                         "gap": np.abs(c.loc[ks, "mid"].to_numpy() - p.loc[ks, "mid"].to_numpy())})
    k = cand.sort_values(["dist", "gap", "k"], kind="mergesort")["k"].iloc[0]
    if abs(k / spot - 1) > MAX_STRIKE_VS_SPOT:
        return {**out, "status": "no_strike_near_spot", "expiration": pd.Timestamp(near)}
    cr, pr = c.loc[k], p.loc[k]
    out.update({
        "status": "ok", "expiration": pd.Timestamp(near), "strike": float(k),
        "call_bid": float(cr["bid"]), "call_ask": float(cr["ask"]), "call_mid": float(cr["mid"]),
        "put_bid": float(pr["bid"]), "put_ask": float(pr["ask"]), "put_mid": float(pr["mid"]),
        "spot": float(spot),
    })
    out["expected_move_pct"] = (out["call_mid"] + out["put_mid"]) / spot
    return out


def event_row(event_id: str, call_cutoff_date, report_date, window: str,
              snapshots: pd.DataFrame, chain_of, grid: np.ndarray,
              rule: str = RULE_ANNOUNCEMENT) -> dict:
    """One research row with every provenance field. `chain_of(snapshot)` returns the
    chain for the selected snapshot (supplied by the vendor adapter)."""
    row = {"event_id": event_id, "call_cutoff_date": pd.Timestamp(call_cutoff_date),
           "prediction_cutoff_ts": prediction_cutoff_ts(call_cutoff_date)}
    snap = select_snapshot(snapshots, call_cutoff_date, grid)
    if snap is None:
        return {**row, "status": "no_snapshot_within_limit"}
    row.update({k: snap[k] for k in ("snapshot_ts", "snapshot_age", "spot", "spot_field", "source")})
    row.update(expected_move(chain_of(snap), snap, report_date, window, rule))
    return row
