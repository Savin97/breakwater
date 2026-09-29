"""Build the earnings-date corrections that `backfills/apply_date_corrections.py` applies.

Why this exists
---------------
Checked against Benzinga's confirmed report dates, ~95.6% of stored events since 2013
sit on exactly the right date, but ~230 are 2-30 days off — almost all AlphaVantage rows
loaded in February 2026, whose report dates for recent quarters trail the announcement.
A wrong date moves the measured reaction to the wrong week. The duplicate cleanup used
to make it permanent: whenever yfinance added the correct date, the cleanup kept the
later, wrong one.

The droplet runs `master`, which has none of the vendor-matching code, so the matching
is done here and written to a corrections file that `master`'s loader applies.

A stored row is corrected only when ALL of these hold:

* It is a completed event with a reported EPS, on a ticker that is not an identity
  hazard (`identity_hazards`, audit condition 1).
* Benzinga has no confirmed record on its date, and exactly one confirmed record 2-30
  days away. One-day differences are left alone: they are as often a date convention
  (after the close vs. the next morning) or a yfinance shift as a real error.
* No other stored event of that stock is as close to the Benzinga date — the Benzinga
  record describes THIS report, not a neighbour.
* yfinance independently has the report on the Benzinga date, and does not also vouch
  for the stored date. Benzinga alone is not enough: before ~2020 its dates are
  sometimes filing dates or another company's (LOW 2014-05-21 is right as stored and
  Benzinga says 04-28; CB before 2016 is a different company), so two independent
  sources must agree before a stored date is overwritten. yfinance history reaches back
  only to ~2020, which is where the corrections stop; older wrong dates remain.

Output rows are `move` (no row at the new date: change the stored row's date) or
`delete_old` (ingestion already added the report on the right date: delete the late
duplicate). The file is derived from licensed vendor data, so it is written under the
gitignored `data/vendor/` and copied to the droplet by hand.

Usage
-----
    PYTHONPATH=. .venv/bin/python backfills/build_date_corrections.py [--db PATH]
"""
import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from config import DB_PATH
from backfills.build_announcement_seed import latest_normalized, load_vendor
from research.massive.paths import VENDOR_ROOT
from research.phase3_target_rebuild import (
    _map_vendor_tickers,
    _snapshot_observed_at,
    identity_hazards,
)
from scripts.backfill_announcement_timestamps import (
    SOURCE_PARQUET as YF_SOURCE_PARQUET,
    load_seed as load_yfinance_seed,
)

MIN_GAP_DAYS = 2
MAX_GAP_DAYS = 30
YF_SOURCES = ("yfinance_earnings_dates", "audit_provider_timestamps_2026_09_05")
CORRECTION_COLS = ["stock", "old_date", "new_date", "action", "announce_ts_ny",
                   "announce_ts_source", "announce_ts_observed_at"]


def _vendor_dates(vendor: pd.DataFrame, stocks: pd.Index) -> pd.DataFrame:
    """Confirmed Benzinga records mapped to our tickers, one row per (stock, report_date).

    `ts` is the vendor clock where it is usable and unambiguous for that date, else NaT.
    """
    v = vendor[vendor["is_confirmed"].fillna(False).astype(bool)].copy()
    v["stock"] = _map_vendor_tickers(v, stocks)
    v = v.dropna(subset=["stock", "report_date"])
    v["report_date"] = pd.to_datetime(v["report_date"]).dt.normalize()
    v["ts"] = v["announce_ts_vendor"].where(v["time_usable"].fillna(False).astype(bool))
    agg = v.groupby(["stock", "report_date"]).agg(
        ts=("ts", lambda s: s.dropna().iloc[0] if s.dropna().nunique() == 1 else pd.NaT),
        benzinga_id=("benzinga_id", "first"))
    return agg.reset_index()


def build(earnings: pd.DataFrame, vendor: pd.DataFrame, snapshot_id: str,
          yf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """earnings: the stored table (stock, earnings_date, reported_eps, announce_ts_source).
    yf: independent yfinance event dates (stock, earnings_date)."""
    e = earnings.copy()
    e["earnings_date"] = pd.to_datetime(e["earnings_date"]).dt.normalize()
    stocks = pd.Index(e["stock"].unique())
    hazards = set(identity_hazards(vendor, stocks)["stock"])
    bz = _vendor_dates(vendor, stocks)
    observed_at = _snapshot_observed_at(snapshot_id)

    yf_dates = pd.concat([
        yf[["stock", "earnings_date"]],
        e.loc[e["announce_ts_source"].isin(YF_SOURCES), ["stock", "earnings_date"]],
    ])
    yf_dates["earnings_date"] = pd.to_datetime(yf_dates["earnings_date"]).dt.normalize()
    yf_by_stock = {s: set(g["earnings_date"]) for s, g in yf_dates.groupby("stock")}
    bz_by_stock = {s: g.set_index("report_date") for s, g in bz.groupby("stock")}
    stored_by_stock = {s: pd.DatetimeIndex(np.sort(g["earnings_date"].unique()))
                       for s, g in e.groupby("stock")}

    today = pd.Timestamp.today().normalize()
    candidates = e[e["reported_eps"].notna() & (e["earnings_date"] <= today)
                   & ~e["stock"].isin(hazards)]
    stats = {"stored_events_checked": len(candidates), "identity_hazard_tickers": len(hazards),
             "exact": 0, "off_by_one_left_alone": 0, "no_benzinga_nearby": 0,
             "ambiguous_benzinga": 0, "benzinga_nearer_another_event": 0,
             "yfinance_disagrees": 0, "benzinga_only_left_alone": 0}
    rows = []
    for r in candidates.itertuples(index=False):
        recs = bz_by_stock.get(r.stock)
        if recs is None:
            stats["no_benzinga_nearby"] += 1
            continue
        if r.earnings_date in recs.index:
            stats["exact"] += 1
            continue
        gaps = (recs.index - r.earnings_date).days
        near = recs[np.abs(gaps) <= MAX_GAP_DAYS]
        if near.empty:
            stats["no_benzinga_nearby"] += 1
            continue
        near_gaps = np.abs((near.index - r.earnings_date).days)
        best = near_gaps.min()
        if (near_gaps == best).sum() > 1:
            stats["ambiguous_benzinga"] += 1
            continue
        new_date = near.index[near_gaps.argmin()]
        if best < MIN_GAP_DAYS:
            stats["off_by_one_left_alone"] += 1
            continue
        # Is another stored event at least as close to the Benzinga date? (A stored row
        # exactly ON new_date is the duplicate of this report, not a neighbour.)
        others = stored_by_stock[r.stock]
        others = others[(others != r.earnings_date) & (others != new_date)]
        if len(others) and np.abs((others - new_date).days).min() <= best:
            stats["benzinga_nearer_another_event"] += 1
            continue
        yfd = yf_by_stock.get(r.stock, set())
        yf_near = {d for d in yfd if abs((d - new_date).days) <= MAX_GAP_DAYS}
        if r.earnings_date in yfd or (yf_near and new_date not in yf_near):
            stats["yfinance_disagrees"] += 1
            continue
        if new_date not in yfd:
            stats["benzinga_only_left_alone"] += 1
            continue

        rec = near.loc[new_date]
        has_ts = pd.notna(rec["ts"])
        rows.append({
            "stock": r.stock,
            "old_date": r.earnings_date.date(),
            "new_date": new_date.date(),
            "action": "delete_old" if new_date in set(stored_by_stock[r.stock]) else "move",
            "announce_ts_ny": pd.Timestamp(rec["ts"]) if has_ts else pd.NaT,
            "announce_ts_source": (f"massive_benzinga:{snapshot_id}:{rec['benzinga_id']}"
                                   if has_ts else None),
            "announce_ts_observed_at": observed_at if has_ts else pd.NaT,
            "days_late": int((r.earnings_date - new_date).days),
            "yfinance_confirms": new_date in yfd,
        })
    out = pd.DataFrame(rows, columns=CORRECTION_COLS + ["days_late", "yfinance_confirms"])
    stats["corrections"] = len(out)
    stats["move"] = int((out["action"] == "move").sum())
    stats["delete_old"] = int((out["action"] == "delete_old").sum())
    stats["yfinance_confirms"] = int(out["yfinance_confirms"].sum())
    return out, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DB_PATH, help="DuckDB to read (read-only)")
    ap.add_argument("--source", type=Path, default=None,
                    help="normalized Benzinga parquet (default: newest under data/vendor)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    vendor, snapshot_id = load_vendor(args.source or latest_normalized())
    con = duckdb.connect(str(args.db), read_only=True)
    try:
        earnings = con.execute("SELECT stock, earnings_date, reported_eps, announce_ts_source "
                               "FROM earnings").df()
    finally:
        con.close()
    out, stats = build(earnings, vendor, snapshot_id, load_yfinance_seed(YF_SOURCE_PARQUET))

    path = args.out or VENDOR_ROOT / f"date_corrections_{snapshot_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {len(out)} corrections to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
