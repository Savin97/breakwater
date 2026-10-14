"""Production Model B vs the frozen refit oracle — the gate before any customer output.

    PYTHONPATH=. .venv/bin/python -m analysis.model_b_validation.validate          # pinned snapshot
    PYTHONPATH=. .venv/bin/python -m analysis.model_b_validation.validate --live   # today's output/

Research evidence, not production: it imports the frozen refit
(`analysis/model_c_refit/`) as the oracle, and nothing in production imports this file.
Writes only output/model_b_validation/ (gitignored).

1. Equivalence. Production `scoring.model_b.raw_signal` on the approved event frame vs
   the oracle frame (analysis/model_c_refit, built 2026-10-03 from the same DB): event
   identities, call cutoff, n_prior, hist_mean_abs, ranking. Both come from the pinned
   snapshot `scoring/model_b_snapshot.py` (prices through 2026-09-28), so the result is
   exact and does not move when the pipeline refreshes output/. `--live` runs the same
   comparison on today's output/ instead: a drift report — pending cutoffs follow the
   price horizon and ingestion revises history, so live differences are expected.
2. Reference sample = the oracle's candidate production-fit training rows, by identity.
3. Regression: B's walk-forward (oracle protocol) recomputed from the production signal;
   expectations from RESULTS.md, not targets.
4. Fixed 80/90 percentile bands on 2017–2025 OOF rows and the 2026 holdout. Descriptive
   only — the bands are never tuned on what this prints.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.model_c_refit import frame as F
from analysis.model_c_refit import model as M
from scoring import model_b as B
from scoring import model_b_snapshot as SNAP
from scoring.build_model_b_reference import reference_sample

# --live inputs only; the default inputs are the pinned snapshot (SNAP).
LIVE_ORACLE_FRAME = Path("output/model_c_refit/frame.parquet")
LIVE_EVENTS = "output/events_df.parquet"
LIVE_DAILY = "output/full_df.parquet"
OUT = Path("output/model_b_validation")
KEY = ["stock", "earnings_date"]

EXPECTED = {"oof": {"roc_auc": 0.715, "pr_auc": 0.357, "top10_capture": 0.251,
                    "top20_capture": 0.414},
            "2026": {"roc_auc": 0.717, "top10_capture": 0.203, "top20_capture": 0.386}}


def load_inputs(live: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(oracle, events, daily): the pinned snapshot, or today's output/ with `live`."""
    if live:
        return (pd.read_parquet(LIVE_ORACLE_FRAME), pd.read_parquet(LIVE_EVENTS),
                pd.read_parquet(LIVE_DAILY, columns=["stock", "date"]))
    events, daily = SNAP.load()
    return SNAP.load_oracle(), events, daily


def production_signal(events: pd.DataFrame, daily: pd.DataFrame):
    sig = B.raw_signal(events, daily)
    prod = pd.concat([events[KEY + ["is_pending"]], sig], axis=1)
    prod["earnings_date"] = pd.to_datetime(prod["earnings_date"])
    return events, daily, prod


def equivalence(oracle: pd.DataFrame, prod: pd.DataFrame) -> dict:
    o = oracle.copy()
    o["earnings_date"] = pd.to_datetime(o["earnings_date"])
    m = o.merge(prod, on=KEY, how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    diff = (both["hist_mean_abs"] - both["model_b_hist_mean_abs"]).abs()
    common = both[F.common_mask(both.set_index(both.index))]
    # ranking: order by oracle signal vs production signal on the research sample
    r_o = common["hist_mean_abs"].rank(method="min")
    r_p = common["model_b_hist_mean_abs"].rank(method="min")
    return {
        "events_oracle": int(len(o)), "events_production": int(len(prod)),
        "only_in_oracle": int((m["_merge"] == "left_only").sum()),
        "only_in_production": int((m["_merge"] == "right_only").sum()),
        "cutoff_mismatch": int((both["cutoff_date"] != both["score_asof_date"]).sum()),
        "n_prior_mismatch": int((both["n_prior"] != both["model_b_n_prior"]).sum()),
        "hist_null_mismatch": int((both["hist_mean_abs"].isna()
                                   != both["model_b_hist_mean_abs"].isna()).sum()),
        "hist_abs_diff_max": float(diff.max()),
        "hist_abs_diff_median": float(diff.median()),
        "hist_compared": int(diff.notna().sum()),
        "research_sample_n": int(len(common)),
        "research_sample_rank_mismatch": int((r_o != r_p).sum()),
        "scored_flag_mismatch_on_research_sample": int((~common["model_b_scored"]).sum()),
    }


def reference_identity(events, daily, oracle) -> dict:
    ref = reference_sample(events, daily)
    ref["earnings_date"] = pd.to_datetime(ref["earnings_date"])
    sample = oracle[F.common_mask(oracle)]
    train = sample[M.training_mask(sample, M.HOLDOUT_YEAR)].copy()
    train["earnings_date"] = pd.to_datetime(train["earnings_date"])
    a = set(map(tuple, ref[KEY].astype(str).values))
    b = set(map(tuple, train[KEY].astype(str).values))
    return {"reference_n": len(a), "oracle_candidate_fit_n": len(b),
            "only_reference": len(a - b), "only_oracle": len(b - a)}


def research_frame(oracle: pd.DataFrame, prod: pd.DataFrame, reference) -> pd.DataFrame:
    """The oracle's common sample, with the PRODUCTION signal swapped in for its own."""
    s = oracle[F.common_mask(oracle)].copy()
    s["earnings_date"] = pd.to_datetime(s["earnings_date"])
    s = s.drop(columns=["hist_mean_abs", "log_hist_mean_abs"]).merge(
        prod[KEY + ["model_b_hist_mean_abs"]], on=KEY, how="left", validate="1:1")
    s.index = oracle[F.common_mask(oracle)].index
    s["hist_mean_abs"] = s["model_b_hist_mean_abs"]
    s["log_hist_mean_abs"] = np.log(s["hist_mean_abs"].clip(lower=F.LOG_FLOOR))
    s["risk_score"] = B.to_percentile(s["hist_mean_abs"].to_numpy(float), reference)
    s["risk_tier"] = B.assign_tier(s["risk_score"].to_numpy()).to_numpy()
    return s


def regression(s: pd.DataFrame) -> dict:
    pre = s[s["year"] < M.HOLDOUT_YEAR]
    oof, fits = M.walk_forward(pre, "B")
    y = pre.loc[oof.index, "y"]
    # Production orders lists on the RAW signal (scoring.model_b.sort_by_risk). The refit's
    # pooled OOF "B" ranks by nine separately fitted yearly models, each standardised on its
    # own training window: identical order within every year, a different order across
    # years. The single fixed production map is the raw-signal order.
    rows = pre.loc[oof.index].assign(risk_tier=lambda d: B.assign_tier(d["risk_score"]).to_numpy(),
                                     model_b_hist_mean_abs=lambda d: d["hist_mean_abs"])
    order = B.sort_by_risk(rows).index
    prod_rank = pd.Series(np.arange(len(order), 0, -1, dtype=float), index=order)
    res = {"oof_n": int(len(oof)),
           "oof_B": M.metrics(y, oof["p"]),
           "oof_production_order": M.metrics(y, prod_rank.loc[oof.index]),
           "oof_raw_signal": M.metrics(y, pre.loc[oof.index, "hist_mean_abs"]),
           "oof_percentile": M.metrics(y, pre.loc[oof.index, "risk_score"]),
           "oof_within_year_order_identical": all(
               (g["p"].rank().to_numpy() == pre.loc[g.index, "hist_mean_abs"].rank().to_numpy()).all()
               for _, g in oof.groupby("test_year")),
           "oof_raw_ties": int(pre.loc[oof.index, "hist_mean_abs"].duplicated(keep=False).sum()),
           "oof_coef_positive_every_year": all(f.model.coef_[0][0] > 0 for f in fits.values())}
    train = s[M.training_mask(s, M.HOLDOUT_YEAR)]
    fit = M.fit(train, M.MODELS["B"], pd.Timestamp("2026-01-01"))
    test = s[s["year"].eq(M.HOLDOUT_YEAR)]
    res["holdout_n"] = int(len(test))
    res["holdout_B"] = M.metrics(test["y"], fit.predict(test))
    res["holdout_raw_signal"] = M.metrics(test["y"], test["hist_mean_abs"])
    res["holdout_raw_ties"] = int(test["hist_mean_abs"].duplicated(keep=False).sum())
    res["holdout_percentile"] = M.metrics(test["y"], test["risk_score"])
    res["fit_B"] = fit.coefficients()
    res["holdout_percentile_ties_at_100"] = int((test["risk_score"] >= 100).sum())
    return res


def bands(g: pd.DataFrame) -> dict:
    ha, el = g["risk_tier"].eq(B.HIGH_ALERT), g["risk_tier"].eq(B.ELEVATED)
    pos = g["y"].sum()
    return {"n": int(len(g)), "base_rate": float(g["y"].mean()),
            "share_high_alert": float(ha.mean()), "share_elevated": float(el.mean()),
            "high_alert_hit_rate": float(g.loc[ha, "y"].mean()) if ha.any() else None,
            "elevated_hit_rate": float(g.loc[el, "y"].mean()) if el.any() else None,
            "high_alert_capture": float(g.loc[ha, "y"].sum() / pos) if pos else None,
            "high_alert_plus_elevated_capture": float(g.loc[ha | el, "y"].sum() / pos) if pos else None}


def band_behaviour(s: pd.DataFrame) -> dict:
    oof = s[s["year"].between(2017, 2025)]
    hold = s[s["year"].eq(2026)]
    out = {"oof_2017_2025": bands(oof), "holdout_2026": bands(hold)}
    out["yearly"] = {int(y): bands(g) for y, g in s[s["year"].between(2017, 2026)].groupby("year")}
    out["by_window_oof"] = {w: bands(g) for w, g in oof.groupby("announce_window")}
    out["by_window_2026"] = {w: bands(g) for w, g in hold.groupby("announce_window")}
    # How the fixed bands relate to the pooled top-k slices the refit validated: the
    # top 10% by raw signal within 2017–2025 vs the High Alert set.
    k = int(np.ceil(0.10 * len(oof)))
    top = set(oof.sort_values("hist_mean_abs", ascending=False, kind="mergesort").index[:k])
    ha = set(oof.index[oof["risk_tier"].eq(B.HIGH_ALERT)])
    out["oof_top10_vs_high_alert"] = {"top10_n": k, "high_alert_n": len(ha),
                                     "overlap": len(top & ha)}
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Production Model B vs the frozen refit oracle.")
    ap.add_argument("--live", action="store_true",
                    help="compare on today's output/ (drift report) instead of the snapshot")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    oracle, events, daily = load_inputs(live=a.live)
    events, daily, prod = production_signal(events, daily)
    reference = B.load_reference()
    res = {"equivalence": equivalence(oracle, prod),
           "reference_identity": reference_identity(events, daily, oracle)}
    s = research_frame(oracle, prod, reference)
    res["regression"] = regression(s)
    res["bands"] = band_behaviour(s)
    res["expected"] = EXPECTED
    res["inputs"] = "live output/" if a.live else f"snapshot {SNAP.SNAPSHOT_DIR.name}"
    name = "validation_live.json" if a.live else "validation.json"
    (OUT / name).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("equivalence", "reference_identity")}, indent=1))
    r = res["regression"]
    for label, m in [("OOF B (pooled yearly fits)", r["oof_B"]),
                     ("OOF production order", r["oof_production_order"]),
                     ("OOF raw signal", r["oof_raw_signal"]),
                     ("OOF percentile", r["oof_percentile"]),
                     ("2026 B", r["holdout_B"]), ("2026 raw signal", r["holdout_raw_signal"]),
                     ("2026 percentile", r["holdout_percentile"])]:
        print(f"{label:28} n={m['n']:6} AUC {m['roc_auc']:.4f} PR {m['pr_auc']:.4f} "
              f"top10 cap {m['top10_capture']:.4f} top20 cap {m['top20_capture']:.4f}")
    print("within-year order B == raw:", r["oof_within_year_order_identical"],
          "| raw ties OOF:", r["oof_raw_ties"], "2026:", r["holdout_raw_ties"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
