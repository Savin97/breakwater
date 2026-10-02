"""Build the announcement-timestamp seed that `backfills/load_announcement_seed.py` loads.

Why this exists
---------------
The droplet's database is the source of truth; every local copy is synced from it. So
the historical announcement times have to be loaded THERE, once, and the droplet runs
`master`, which carries none of the research code that matches vendor records to our
events. This script does the matching here and writes the result as a plain seed file:

    stock, earnings_date, announce_ts_ny, announce_ts_source, announce_ts_observed_at

which is copied to the droplet and loaded by `master`'s `load_announcement_seed.py`.

Two sources, in priority order:

1. Benzinga, snapshot of 2026-09-10 (`research/massive/`, verdict ACCEPT WITH CONDITIONS
   in `audit/BENZINGA_EARNINGS_AUDIT.md`). ~25,500 events back to ~2012, real minutes.
2. yfinance, `audit/provider_timestamps.parquet` (pulled 2026-09-05). Only for events
   Benzinga does not cover. Hour-rounded; agrees with Benzinga's window 99.6% of the time.

The seed is derived from licensed vendor data, so it is written under the gitignored
`data/vendor/` and never committed. Copy it to the droplet by hand.

Benzinga rules
--------------
* Audit condition 1 — identity: tickers flagged by `identity_hazards` (class-share
  spellings, reused symbols) are skipped entirely, not joined on today's ticker string.
* Audit condition 2 — midnight is "time unknown": only `time_usable` records are used,
  and `normalize()` marks `00:00:00` unusable.
* Audit condition 3 — `time` is NY wall clock: `announce_ts_vendor` is already naive NY.
* EXACT DATE MATCHES ONLY. Production anchors on `earnings.earnings_date` and takes only
  the clock from `announce_ts_ny` (`resolve_event_anchors`). A vendor record one day off
  (e.g. our date 02-19, vendor 02-18 16:05 AMC) would be anchored as AMC on 02-19 — the
  wrong session. The Phase 3 research re-anchored those on the vendor date; production
  cannot, so they are skipped and counted.
* Provenance per row: `announce_ts_source = massive_benzinga:<snapshot>:<benzinga_id>`,
  and `announce_ts_observed_at` = the snapshot's acquisition time in NY wall clock — a
  lower bound on when each page was fetched. Events still upcoming at that time are
  schedules and ingestion may refresh them; earlier ones are frozen.

Usage
-----
    PYTHONPATH=. .venv/bin/python backfills/build_announcement_seed.py
"""
import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

from config import DB_PATH
from research.massive.paths import NORMALIZED_ROOT, VENDOR_ROOT
from research.phase3_target_rebuild import (
    _snapshot_observed_at,
    identity_hazards,
    match_benzinga_timing,
)
SEED_COLS = ["stock", "earnings_date", "announce_ts_ny", "announce_ts_source",
             "announce_ts_observed_at"]

# The yfinance fallback: event-level announcement times pulled during the Phase 0 audit.
# These rows were first loaded into the DB by a one-time backfill script (since deleted);
# the label and pull date below are the provenance those rows already carry, so they
# must not change.
YF_SOURCE_PARQUET = "audit/provider_timestamps.parquet"
YF_SOURCE_LABEL = "audit_provider_timestamps_2026_09_05"

# When the audit actually pulled these timestamps from yfinance, recorded as
# `announce_ts_observed_at`: a seeded event that had already reported by this date was
# OBSERVED after the fact and is frozen; one still upcoming was a SCHEDULE and ingestion
# may refresh it later. It is a fact about the pull, not a guess — do not move it to "now".
#
# Naive NY wall clock, like every other announcement-timing value
# (utilities.time_utilities). Midnight NY on the pull date is a deliberate LOWER BOUND on
# the moment of the pull: it can only make a seeded row look more like a schedule and so
# more refreshable, never less, which is the conservative direction under any host
# timezone.
YF_OBSERVED_AT = pd.Timestamp("2026-09-05")


def load_yfinance_seed(path=YF_SOURCE_PARQUET) -> pd.DataFrame:
    """The audit parquet, reduced to exactly (stock, earnings_date, announce_ts_ny).

    The parquet's timestamps are tz-aware America/New_York; the DB column is naive NY
    local time, so the conversion drops the offset and keeps the wall clock. That is the
    same shape ingestion writes, and the same shape the classifier reads.
    """
    ts = pd.read_parquet(path)
    ts = ts[["stock", "earnings_date", "announce_ts_ny"]].copy()
    ts["earnings_date"] = pd.to_datetime(ts["earnings_date"]).dt.date
    ny = pd.to_datetime(ts["announce_ts_ny"])
    if getattr(ny.dt, "tz", None) is not None:
        ny = ny.dt.tz_convert("America/New_York").dt.tz_localize(None)
    ts["announce_ts_ny"] = ny
    ts = ts.dropna(subset=["announce_ts_ny"])
    return ts.drop_duplicates(subset=["stock", "earnings_date"], keep="first")


def latest_normalized(root: Path = NORMALIZED_ROOT) -> Path:
    candidates = sorted(Path(root).glob("benzinga_earnings_earnings_*.parquet"))
    if not candidates:
        raise FileNotFoundError(
            f"no normalized Benzinga parquet under {root} — copy it from the machine "
            "that ran research.massive.acquire (it is gitignored)")
    return candidates[-1]


def load_vendor(path: Path) -> tuple[pd.DataFrame, str]:
    vendor = pd.read_parquet(path)
    snapshots = vendor["snapshot_id"].dropna().unique()
    if len(snapshots) != 1:
        raise ValueError(f"{path} mixes snapshots {list(snapshots)}; expected exactly one")
    return vendor, str(snapshots[0])


def benzinga_seed(vendor: pd.DataFrame, keys: pd.DataFrame, snapshot_id: str
                  ) -> tuple[pd.DataFrame, dict]:
    """Exact-date, identity-safe Benzinga matches, in SEED_COLS."""
    keys = keys.copy()
    keys["earnings_date"] = pd.to_datetime(keys["earnings_date"]).dt.normalize()
    hazards = identity_hazards(vendor, pd.Index(keys["stock"].unique()))
    timing, audit = match_benzinga_timing(
        vendor, keys, tolerance_days=1, snapshot_id=snapshot_id,
        excluded_stocks=set(hazards["stock"]))

    stats = {
        "db_events": len(keys),
        "identity_hazard_tickers": hazards["stock"].nunique(),
        "identity_hazard_events": int(audit["match_status"].eq("identity_hazard").sum()),
        "no_vendor_record": int(audit["match_status"].eq("no_record").sum()),
        "ambiguous": int(audit["match_status"].eq("ambiguous_timestamp").sum()),
    }
    matched = audit[audit["match_status"].eq("matched")]
    exact = matched.loc[matched["date_delta_days"].eq(0), ["stock", "earnings_date"]]
    stats["skipped_date_off_by_one"] = len(matched) - len(exact)

    if timing.empty:
        seed = pd.DataFrame(columns=SEED_COLS)
    else:
        seed = timing.merge(exact, on=["stock", "earnings_date"], how="inner")[SEED_COLS]
    seed = seed.copy()
    seed["earnings_date"] = pd.to_datetime(seed["earnings_date"]).dt.date
    stats["benzinga_rows"] = len(seed)
    return seed, stats


def yfinance_fallback(yf: pd.DataFrame, keys: pd.DataFrame, taken: pd.DataFrame
                      ) -> pd.DataFrame:
    """yfinance rows for events we hold that Benzinga did not time, in SEED_COLS."""
    keys = keys.assign(earnings_date=pd.to_datetime(keys["earnings_date"]).dt.date)
    yf = yf.merge(keys[["stock", "earnings_date"]].drop_duplicates(),
                  on=["stock", "earnings_date"], how="inner")
    taken_keys = set(zip(taken["stock"], taken["earnings_date"]))
    untaken = pd.Series([k not in taken_keys for k in zip(yf["stock"], yf["earnings_date"])],
                        index=yf.index, dtype=bool)
    yf = yf[untaken].copy()
    yf["announce_ts_source"] = YF_SOURCE_LABEL
    yf["announce_ts_observed_at"] = YF_OBSERVED_AT
    return yf[SEED_COLS]


def build(vendor: pd.DataFrame, snapshot_id: str, keys: pd.DataFrame,
          yf: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    bz, stats = benzinga_seed(vendor, keys, snapshot_id)
    fallback = yfinance_fallback(yf, keys, bz)
    stats["yfinance_rows"] = len(fallback)
    seed = pd.concat([bz, fallback], ignore_index=True)
    seed["announce_ts_ny"] = pd.to_datetime(seed["announce_ts_ny"])
    seed["announce_ts_observed_at"] = pd.to_datetime(seed["announce_ts_observed_at"])
    seed = seed.sort_values(["stock", "earnings_date"], kind="mergesort").reset_index(drop=True)
    assert not seed.duplicated(["stock", "earnings_date"]).any()
    stats["seed_rows"] = len(seed)
    return seed, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=None,
                    help="normalized Benzinga parquet (default: newest under data/vendor)")
    ap.add_argument("--out", type=Path, default=None,
                    help="seed parquet to write (default: data/vendor/announcement_seed_<snapshot>.parquet)")
    args = ap.parse_args()

    path = args.source or latest_normalized()
    vendor, snapshot_id = load_vendor(path)
    print(f"vendor: {len(vendor)} records, snapshot {snapshot_id}, "
          f"observed_at {_snapshot_observed_at(snapshot_id)} NY")

    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        keys = con.execute("SELECT DISTINCT stock, earnings_date FROM earnings").df()
    finally:
        con.close()

    seed, stats = build(vendor, snapshot_id, keys, load_yfinance_seed(YF_SOURCE_PARQUET))
    out = args.out or VENDOR_ROOT / f"announcement_seed_{snapshot_id}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    seed.to_parquet(out, index=False)

    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {len(seed)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
