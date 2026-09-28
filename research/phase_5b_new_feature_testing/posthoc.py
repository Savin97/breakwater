"""POST HOC — written AFTER the pre-registered Phase 5B results were read.

Not part of the pre-registration and not eligible to be called a Phase 5B "result". It asks
one question the pre-registered run raised but could not answer:

    The raw sector peer reaction LEVEL (`peer_sec_mean_abs_1d_20`) is statistically
    incremental over the incumbent, while the peer SHOCK (the same reactions relative to
    each peer's own history) is null. Is the level signal news about the current earnings
    season, or just a cleaner estimate of the sector's structural reaction phenotype?

Decomposition, on the SAME usable peer events (same endpoint rule, 20 sessions, sector
level, peers with >= 4 prior resolved events so both terms are defined):

    expected = mean over usable peers of their own prior mean |r1d|   (phenotype only)
    realized = mean over the same peers of their actual |r1d|         (phenotype + news)

If realized adds nothing once expected is in the model, the level signal is phenotype.
Also tested against Phase 4's `sector_prior_extreme_rate` (all-history sector base rate).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.phase5_event_signal import score_predictions, walk_forward_predictions
from research.phase_5b_new_feature_testing.build import FEATURE_CACHE, PHASE3_EVENTS_PATH, RESULTS_DIR, load_daily
from research.phase_5b_new_feature_testing.evaluate import INCUMBENT, assemble, bootstrap_delta, samples
from research.phase_5b_new_feature_testing.panel import event_cutoff_index, issuer_of
from research.phase_5b_new_feature_testing.peer_features import peer_event_table

W = 20


def expected_vs_realized(events: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:
    peers = peer_event_table(events, grid)
    peers = peers[peers["prior_mean_abs_1d"].notna()]
    cut = event_cutoff_index(events, grid)
    iss = events["stock"].astype(str).map(issuer_of).to_numpy()
    sec = events["sector"].to_numpy(object)
    groups = {g: (df["endpoint_idx"].to_numpy(), df["abs_r1d"].to_numpy(float),
                  df["prior_mean_abs_1d"].to_numpy(float), df["issuer"].to_numpy(object))
              for g, df in peers.groupby("sector")}
    out = np.full((len(events), 2), np.nan)
    for k, c in enumerate(cut):
        g = groups.get(sec[k])
        if g is None or c < 0:
            continue
        end, ar, m, pi = g
        lo, hi = np.searchsorted(end, c - W + 1), np.searchsorted(end, c, side="right")
        keep = pi[lo:hi] != iss[k]
        if keep.sum():
            out[k] = [m[lo:hi][keep].mean(), ar[lo:hi][keep].mean()]
    return pd.DataFrame(out, index=events.index,
                        columns=["posthoc_peer_sec_expected_abs_1d_20",
                                 "posthoc_peer_sec_realized_abs_1d_20"])


def main() -> int:
    from feature_engineering.announcement_timing import market_session_grid
    events = pd.read_parquet(PHASE3_EVENTS_PATH)
    grid = market_session_grid(load_daily())
    feats = pd.read_parquet(FEATURE_CACHE).drop(columns=["stock", "earnings_date"])
    analysis = assemble(events, feats).join(expected_vs_realized(events, grid))
    common = samples(analysis, "2019-01-01", "2025-12-31")["common"]

    E, Rz = "posthoc_peer_sec_expected_abs_1d_20", "posthoc_peer_sec_realized_abs_1d_20"
    SPR = "sector_prior_extreme_rate"
    specs = {
        "incumbent": INCUMBENT,
        "inc+peer_sec_mean_abs_1d_20 (pre-registered)": [*INCUMBENT, "peer_sec_mean_abs_1d_20"],
        "inc+expected": [*INCUMBENT, E],
        "inc+realized": [*INCUMBENT, Rz],
        "inc+expected+realized": [*INCUMBENT, E, Rz],
        "inc+sector_prior_extreme_rate": [*INCUMBENT, SPR],
        "inc+sector_prior_extreme_rate+peer_sec_mean_abs_1d_20":
            [*INCUMBENT, SPR, "peer_sec_mean_abs_1d_20"],
    }
    preds = {k: walk_forward_predictions(common, v) for k, v in specs.items()}
    rows = []
    comparisons = [
        ("inc+peer_sec_mean_abs_1d_20 (pre-registered)", "incumbent"),
        ("inc+expected", "incumbent"),
        ("inc+realized", "incumbent"),
        ("inc+expected+realized", "inc+expected"),        # does NEWS add beyond phenotype?
        ("inc+sector_prior_extreme_rate", "incumbent"),
        ("inc+sector_prior_extreme_rate+peer_sec_mean_abs_1d_20",
         "inc+sector_prior_extreme_rate"),
    ]
    for a, b in comparisons:
        s = score_predictions(preds[a], a, specs[a], "pooled_oof")
        d = bootstrap_delta(preds[a], preds[b])
        rows.append({"model": a, "vs": b, "roc_auc": s["roc_auc"],
                     "top10_capture": s["top10_capture"], "top20_capture": s["top20_capture"],
                     "within_stock_pooled_auc": s["within_stock_pooled_auc"],
                     **{k: v for k, v in d.items() if k.startswith("delta_auc")
                        or k.startswith("delta_top")}})
    corr = common[[E, Rz, "peer_sec_mean_abs_1d_20"]].corr(method="spearman")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "posthoc_peer_decomposition.csv", index=False)
    (RESULTS_DIR / "posthoc_peer_decomposition.json").write_text(json.dumps({
        "label": "POST HOC - not pre-registered",
        "spearman": corr.round(4).to_dict(),
        "coverage_expected": float(common[E].notna().mean()),
        "rows": out.to_dict(orient="records")}, indent=2, default=str))
    print(out.round(4).to_string())
    print(corr.round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
