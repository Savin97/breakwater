"""Build every Phase 5B feature for every event of the Phase 3 event frame.

Reads `output/full_df.parquet` (prices + classification only) and
`output/phase3_target_rebuild/phase3_events.parquet`. Writes a feature cache under
`output/phase5b_free_features/`. No database, no production module is written to.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from phase_5b_new_feature_testing.panel import build_panel
from phase_5b_new_feature_testing.peer_features import compute_peer_features
from phase_5b_new_feature_testing.price_features import compute_price_features

FULL_DF_PATH = Path("output/full_df.parquet")
PHASE3_EVENTS_PATH = Path("output/phase3_target_rebuild/phase3_events.parquet")
RESULTS_DIR = Path("output/phase5b_free_features")
FEATURE_CACHE = RESULTS_DIR / "phase5b_event_features.parquet"


def load_daily(path: Path = FULL_DF_PATH) -> pd.DataFrame:
    return pd.read_parquet(path, columns=["stock", "date", "price", "sector", "sub_sector"])


def build_features(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    panel = build_panel(daily)
    price = compute_price_features(panel, events)
    peer = compute_peer_features(events, panel.grid)
    return pd.concat([price, peer], axis=1)


def main() -> int:
    t0 = time.time()
    events = pd.read_parquet(PHASE3_EVENTS_PATH)
    feats = build_features(events, load_daily())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    feats.assign(stock=events["stock"].values, earnings_date=events["earnings_date"].values
                 ).to_parquet(FEATURE_CACHE)
    print(f"wrote {FEATURE_CACHE} {feats.shape} in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
