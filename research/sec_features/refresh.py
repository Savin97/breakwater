"""Compare the SEC pilot's event association before and after the 2026-09-29 date corrections.

Old: `output/sec_filings_pilot/pre_refresh_20261002/` (pilot run on Phase 3's frame).
New: `output/sec_filings_pilot/` (same code, same SEC snapshot, same identity chains and
timing rule, on `output/sec_features/feature_frame_current.parquet`).
Reads identity, clock and filing columns only.

    PYTHONPATH=. .venv/bin/python -m research.sec_features.refresh
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OLD = Path("output/sec_filings_pilot/pre_refresh_20261002")
NEW = Path("output/sec_filings_pilot")
OUT = Path("output/sec_features")
COLS = ["event_id", "stock", "earnings_date", "call_cutoff_date", "identity_status",
        "periodic_accession", "announce_window", "year"]
MATCH_DAYS = 45


def _infoset(p: Path) -> pd.Series:
    k = pd.read_parquet(p / "event_8k.parquet")
    k = k[k["relation"].isin(["after_periodic", "no_periodic"])]
    return k.groupby("event_id")["accession"].agg(lambda s: tuple(sorted(s)))


def compare() -> tuple[pd.DataFrame, dict]:
    o = pd.read_parquet(OLD / "events.parquet", columns=COLS)
    n = pd.read_parquet(NEW / "events.parquet", columns=COLS)
    o["k8"] = o["event_id"].map(_infoset(OLD)).apply(lambda x: x if isinstance(x, tuple) else ())
    n["k8"] = n["event_id"].map(_infoset(NEW)).apply(lambda x: x if isinstance(x, tuple) else ())
    both = o.merge(n, on="event_id", suffixes=("_old", "_new"))
    only_o = o[~o["event_id"].isin(n["event_id"])]
    only_n = n[~n["event_id"].isin(o["event_id"])]
    # An old-only and a new-only event of the same stock within 45 days = a moved date.
    pairs = only_o.merge(only_n, on="stock", suffixes=("_old", "_new"))
    pairs = pairs[(pairs["earnings_date_new"] - pairs["earnings_date_old"]).dt.days.abs()
                  <= MATCH_DAYS]
    pairs = pairs.sort_values("event_id_old").drop_duplicates("event_id_old").drop_duplicates(
        "event_id_new")
    moved = pairs.rename(columns={"stock": "stock_old"}).assign(stock_new=lambda d: d["stock_old"])
    both = both.assign(change="same_event_id")
    moved = moved.assign(change="earnings_date_changed")
    allp = pd.concat([both, moved], ignore_index=True)
    allp["cutoff_changed"] = allp["call_cutoff_date_old"] != allp["call_cutoff_date_new"]
    allp["periodic_changed"] = (allp["periodic_accession_old"].fillna("") !=
                                allp["periodic_accession_new"].fillna(""))
    allp["k8_changed"] = allp["k8_old"] != allp["k8_new"]
    allp["identity_changed"] = allp["identity_status_old"] != allp["identity_status_new"]
    gone = only_o[~only_o["event_id"].isin(pairs["event_id_old"])]
    added = only_n[~only_n["event_id"].isin(pairs["event_id_new"])]
    summary = {
        "events_old": len(o), "events_new": len(n), "same_event_id": len(both),
        "earnings_date_changed": len(moved),
        "dropped_from_population": len(gone), "added_to_population": len(added),
        "dropped_examples": gone["event_id"].head(20).tolist(),
        "added_examples": added["event_id"].head(20).tolist(),
        "cutoff_changed": int(allp["cutoff_changed"].sum()),
        "periodic_changed": int(allp["periodic_changed"].sum()),
        "k8_set_changed": int(allp["k8_changed"].sum()),
        "identity_status_changed": int(allp["identity_changed"].sum()),
        "unmapped_or_missing_old": int((~o["identity_status"].eq("mapped")).sum()),
        "unmapped_or_missing_new": int((~n["identity_status"].eq("mapped")).sum()),
        "no_periodic_old": int(o["periodic_accession"].isna().sum()),
        "no_periodic_new": int(n["periodic_accession"].isna().sum()),
        "new_identity_statuses": n["identity_status"].value_counts().to_dict(),
        "changed_by_kind": allp.groupby("change")[["cutoff_changed", "periodic_changed",
                                                  "k8_changed"]].sum().to_dict("index"),
    }
    changed = allp[allp[["cutoff_changed", "periodic_changed", "k8_changed",
                         "identity_changed"]].any(axis=1)]
    return changed, summary, gone, added


def main() -> int:
    changed, summary, gone, added = compare()
    OUT.mkdir(parents=True, exist_ok=True)
    changed.drop(columns=["k8_old", "k8_new"]).to_csv(OUT / "refresh_changed_events.csv", index=False)
    gone.drop(columns="k8").to_csv(OUT / "refresh_dropped_events.csv", index=False)
    added.drop(columns="k8").to_csv(OUT / "refresh_added_events.csv", index=False)
    (OUT / "refresh_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
