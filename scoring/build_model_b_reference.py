# scoring/build_model_b_reference.py
"""
Builds `scoring/model_b_reference.json` — the frozen scale that turns the raw Model B
signal into the Breakwater Risk Percentile. A maintainer tool, run once; never on the
pipeline path.

    python -m scoring.build_model_b_reference                 # writes the artifact
    python -m scoring.build_model_b_reference --check         # rebuilds from the pinned
                                                              # snapshot, diffs, writes nothing
    python -m scoring.build_model_b_reference --check --live  # same on today's output/

`--check` rebuilds from the pinned 2026-09-28 snapshot (scoring/model_b_snapshot.py) the
artifact was frozen from, so it must match bit-for-bit. `--check --live` rebuilds from
today's output/: ingestion revises history, so a difference there is data drift, reported
for information — it never means the artifact should be regenerated.

Reference sample (the 0.4.0 production-fit sample, as in analysis/model_c_refit's
candidate production fit): completed events whose corrected 3-session outcome is
available, report year >= 2014, outcome endpoint before 2026-01-01, at least 8 usable
prior outcomes. Only the RAW SIGNAL of those events is used. Outcomes select which
events are in the sample (an event belongs to the fit sample once its own outcome is
known) but never shape the mapping: no hit rate, no threshold search, no fitting.

Changing the artifact changes what every percentile means. Regenerating it from newer
data is a model change and needs a new MODEL_VERSION.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.announcement_timing import market_session_grid
from scoring import model_b as B

EVENTS_PARQUET = "output/events_df.parquet"
DAILY_PARQUET = "output/full_df.parquet"

REFERENCE_FIRST_YEAR = 2014
REFERENCE_ENDPOINT_BEFORE = pd.Timestamp("2026-01-01")
PERCENTILE_STEP = 0.1
# SHA-256 of analysis/model_c_refit/SPEC.md as frozen in output/model_c_refit/FROZEN.json
# when the refit that selected Model B was run (2026-10-03). Provenance only.
REFIT_SPEC_SHA256 = "8ae23d283b7e81fe4e964dbf5c093f887c2762897f83c756710b1bf72f5cc69b"


def reference_sample(events: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """The events whose raw signal defines the scale, with that signal."""
    grid = market_session_grid(daily_df)
    sig = B.raw_signal(events, daily_df)
    end = B.outcome_endpoint_index(grid, events["anchor_date"].values)
    end_date = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns]")
    end_date[end >= 0] = grid[end[end >= 0]]
    year = pd.to_datetime(events["earnings_date"]).dt.year
    mask = (B.usable_outcome(events).notna() & sig["model_b_scored"]
            & year.ge(REFERENCE_FIRST_YEAR) & end_date.lt(REFERENCE_ENDPOINT_BEFORE))
    out = events.loc[mask, ["stock", "earnings_date"]].copy()
    out["model_b_hist_mean_abs"] = sig.loc[mask, "model_b_hist_mean_abs"]
    out["endpoint_date"] = end_date[mask]
    return out


def build_payload(sample: pd.DataFrame, daily_df: pd.DataFrame) -> dict:
    raw = sample["model_b_hist_mean_abs"].to_numpy(float)
    if np.isnan(raw).any():
        raise ValueError("reference sample contains an unscored event")
    pct = np.round(np.arange(0, 100 + PERCENTILE_STEP / 2, PERCENTILE_STEP), 6)
    values = np.quantile(raw, pct / 100.0)
    years = pd.to_datetime(sample["earnings_date"]).dt.year
    payload = {
        "model": B.MODEL_NAME,
        "model_version": "0.4.0",
        "raw_signal": "model_b_hist_mean_abs",
        "raw_signal_definition": B.SIGNAL_DEFINITION,
        "validation_target": "abs_reaction_3d_anchored >= 0.08",
        "min_prior": B.MIN_PRIOR,
        "reference_period": (f"report years {int(years.min())}-{int(years.max())}, outcome "
                             f"endpoint before {REFERENCE_ENDPOINT_BEFORE.date()}"),
        "reference_n": int(len(sample)),
        "reference_stocks": int(sample["stock"].nunique()),
        "percentile_semantics": "share of the reference sample at or below the raw signal "
                                "(linear between grid points); a rank, not a probability",
        "tier_bands": {"High Alert": ">= 90", "Elevated": "80 to < 90", "Normal": "< 80",
                       "Unscored": f"fewer than {B.MIN_PRIOR} usable prior corrected reactions"},
        "raw_signal_at_p80": float(np.quantile(raw, 0.80)),
        "raw_signal_at_p90": float(np.quantile(raw, 0.90)),
        "source": {
            "generator": "scoring/build_model_b_reference.py",
            "signal_implementation": "scoring/model_b.py (raw_signal)",
            "selected_by": "analysis/model_c_refit (SPEC.md, RESULTS.md)",
            "refit_spec_sha256": REFIT_SPEC_SHA256,
            "price_data_through": str(pd.to_datetime(daily_df["date"]).max().date()),
        },
        "generated_on": str(date.today()),
        "quantiles": {"percentiles": pct.tolist(), "values": [float(v) for v in values]},
    }
    payload["sha256"] = B.reference_checksum(payload)
    B.reference_from_payload(payload)          # validates monotonicity and checksum
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--check", action="store_true",
                    help="rebuild and compare with the committed artifact; write nothing")
    ap.add_argument("--live", action="store_true",
                    help="with --check: rebuild from today's output/ instead of the snapshot")
    a = ap.parse_args(argv)
    if a.check and not a.live:
        from scoring import model_b_snapshot
        events, daily = model_b_snapshot.load()
        print(f"inputs: pinned snapshot {model_b_snapshot.SNAPSHOT_DIR.name}")
    else:
        events = pd.read_parquet(EVENTS_PARQUET)
        daily = pd.read_parquet(DAILY_PARQUET, columns=["stock", "date"])
        print(f"inputs: {EVENTS_PARQUET} (live)")
    sample = reference_sample(events, daily)
    payload = build_payload(sample, daily)
    print(f"reference sample: n={payload['reference_n']}, {payload['reference_stocks']} "
          f"stocks, {payload['reference_period']}")
    print(f"raw signal at p80 {payload['raw_signal_at_p80']:.5f}, "
          f"p90 {payload['raw_signal_at_p90']:.5f}")
    if a.check:
        old = json.loads(B.REFERENCE_PATH.read_text())
        same = (old["reference_n"] == payload["reference_n"]
                and old["quantiles"]["values"] == payload["quantiles"]["values"])
        print("MATCHES the committed artifact" if same else
              "DIFFERS from the committed artifact" + (
                  " (live data drifted since 2026-09-28 — expected; do not regenerate)"
                  if a.live else " — the snapshot or the implementation changed"))
        return 0 if same else 1
    Path(B.REFERENCE_PATH).write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {B.REFERENCE_PATH} (sha256 {payload['sha256'][:12]}…)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
