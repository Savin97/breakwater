"""The CURRENT corrected event frame, built exactly as Phase 3 built its own.

Phase 3's frame (`output/phase3_target_rebuild/phase3_events.parquet`, 2026-09-12) and the
SEC pilot both predate the 103 earnings-date corrections of 2026-09-29. This rebuilds the
same thing from today's `output/full_df.parquet` (rebuilt 2026-09-29 after the corrections)
with the same code and the same Benzinga snapshot: same timing match, same identity-hazard
exclusion, same re-anchoring, same causal call-cutoff features (`research.phase3_refit`).

    PYTHONPATH=. .venv/bin/python -m research.sec_features.frame

Writes `output/sec_features/phase3_events_current.parquet` and
`output/sec_features/feature_frame_current.parquet`. Nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from research import phase3_target_rebuild as p3
from research.phase3_refit.features import build_feature_frame

OUT = Path("output/sec_features")
EVENTS = OUT / "phase3_events_current.parquet"
FRAME = OUT / "feature_frame_current.parquet"


def build() -> dict:
    daily = pd.read_parquet(p3.FULL_DF_PATH)
    snapshot = p3.latest_snapshot()
    vendor = p3.normalize_snapshot(snapshot)
    keys = p3.event_keys_from_daily(daily)
    hazards = p3.identity_hazards(vendor, pd.Index(keys["stock"].unique()))
    timing, audit = p3.match_benzinga_timing(vendor, keys, tolerance_days=1, confirmed_only=True,
                                             snapshot_id=snapshot.name,
                                             excluded_stocks=set(hazards["stock"]))
    events = p3.build_phase3_events(daily, timing)
    OUT.mkdir(parents=True, exist_ok=True)
    events.to_parquet(EVENTS, index=False)
    frame = build_feature_frame(events, daily[["stock", "date", "price", "sector", "sub_sector",
                                               "vol_30d", "drift_30d"]])
    frame.to_parquet(FRAME)
    info = {"snapshot": snapshot.name, "full_df_mtime": pd.Timestamp(
        p3.FULL_DF_PATH.stat().st_mtime, unit="s").isoformat(), "events": len(events),
        "timing_matches": len(timing), "hazards": sorted(hazards["stock"])}
    (OUT / "frame_build.json").write_text(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    print(json.dumps(build(), indent=1))
