"""POST HOC — one pre-declared combination, specified AFTER the main Phase 5B results.

The single model tested here, declared before it was run:

    structural + vol_30d + peer_sec_rw_mean_abs_1d_20 + idio_vol_30d

i.e. the incumbent plus the best single feature from each of the two families that showed
any incremental signal (family 5 peer earnings level, family 1 idiosyncratic vol). No other
combination is tried. Same common sample, walk-forward (2021-2025), block fitting and
stock-clustered bootstrap as `evaluate.py`; judged by the same three pre-registered rules.
Because the two features were chosen from the main results, a pass here is
confirmatory-looking but NOT an independent out-of-sample confirmation.
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from research.phase5_event_signal import score_predictions, walk_forward_predictions
from phase_5b_new_feature_testing.build import FEATURE_CACHE, PHASE3_EVENTS_PATH, RESULTS_DIR
from phase_5b_new_feature_testing.evaluate import (
    INCUMBENT,
    MIN_POSITIVE_YEARS,
    assemble,
    bootstrap_delta,
    samples,
)

PEER = "peer_sec_rw_mean_abs_1d_20"
IDIO = "idio_vol_30d"
PAIR = [*INCUMBENT, PEER, IDIO]


def main() -> int:
    events = pd.read_parquet(PHASE3_EVENTS_PATH)
    feats = pd.read_parquet(FEATURE_CACHE).drop(columns=["stock", "earnings_date"])
    common = samples(assemble(events, feats), "2019-01-01", "2025-12-31")["common"]

    specs = {"incumbent": INCUMBENT, "inc+peer": [*INCUMBENT, PEER],
             "inc+idio": [*INCUMBENT, IDIO], "inc+peer+idio (PAIR)": PAIR}
    preds = {k: walk_forward_predictions(common, v) for k, v in specs.items()}

    rows = []
    for a, b in [("inc+peer+idio (PAIR)", "incumbent"), ("inc+peer+idio (PAIR)", "inc+peer"),
                 ("inc+peer+idio (PAIR)", "inc+idio"), ("inc+peer", "incumbent"),
                 ("inc+idio", "incumbent")]:
        s = score_predictions(preds[a], a, specs[a], "pooled_oof")
        d = bootstrap_delta(preds[a], preds[b])
        yearly = {}
        for y, g in preds[a].groupby("test_year"):
            gb = preds[b].loc[g.index]
            yearly[int(y)] = (roc_auc_score(g["y_extreme"].astype(int), g["p"])
                              - roc_auc_score(gb["y_extreme"].astype(int), gb["p"]))
        pos = sum(v > 0 for v in yearly.values())
        rows.append({
            "model": a, "vs": b, "n": s["n"], "roc_auc": s["roc_auc"], "pr_auc": s["pr_auc"],
            "brier": s["brier"], "log_loss": s["log_loss"],
            "top10_hit_rate": s["top10_hit_rate"], "top10_capture": s["top10_capture"],
            "top20_capture": s["top20_capture"],
            "within_stock_pooled_auc": s["within_stock_pooled_auc"],
            **{k: v for k, v in d.items() if k.startswith("delta_")},
            "years_delta_auc_positive": pos,
            "yearly_delta_auc": ";".join(f"{y}:{v:+.4f}" for y, v in yearly.items()),
            "stat_incremental": d["delta_auc_lo"] > 0,
            "stable": pos >= MIN_POSITIVE_YEARS,
            "product_relevant": (d["delta_top10_capture_lo"] > 0) or (d["delta_top20_capture_lo"] > 0),
        })
    inc = score_predictions(preds["incumbent"], "incumbent", INCUMBENT, "pooled_oof")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "posthoc_pair.csv", index=False)
    (RESULTS_DIR / "posthoc_pair.json").write_text(json.dumps({
        "label": "POST HOC - single combination declared after main results",
        "pair_model": PAIR, "incumbent": inc, "comparisons": out.to_dict(orient="records"),
        "feature_spearman": float(common[[PEER, IDIO]].corr(method="spearman").iloc[0, 1]),
    }, indent=2, default=str))
    pd.set_option("display.width", 250)
    print(f"incumbent AUC {inc['roc_auc']:.4f} top10cap {inc['top10_capture']:.4f} "
          f"top20cap {inc['top20_capture']:.4f}")
    print(out.drop(columns=["n"]).round(4).T.to_string())
    print("spearman(peer, idio) =", round(float(common[[PEER, IDIO]].corr(method='spearman').iloc[0, 1]), 3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
