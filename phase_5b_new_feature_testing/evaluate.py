"""Phase 5B evaluation: do free price/peer features beat `structural + vol_30d`?

Research-only. Reuses, unchanged: the Phase 3 corrected target, Phase 4's causal baselines
and mature-28 cohort, and Phase 5's common sample, walk-forward, block fitting, scoring,
pooled-pair and univariate machinery. What is new here is only (a) the pre-registered
feature set (PREREGISTRATION.md, registry.py), (b) a bootstrap that reports deltas on the
product metrics as well as AUC, and (c) the gated interaction.

The key comparison everywhere is  candidate − (structural + vol_30d),  on identical rows.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from research.phase4_baselines import _resolved_window, cohort_sample, prepare_analysis_frame
from research.phase5_event_signal import (
    FIRST_TEST_YEAR,
    HEADLINE_COHORT,
    LAST_TEST_YEAR,
    STRUCTURAL,
    apply_block,
    derive_phase5_features,
    feature_audit,
    fit_block,
    pooled_pair_auc,
    score_predictions,
    univariate_signal,
    walk_forward_predictions,
)
from phase_5b_new_feature_testing import registry as R
from phase_5b_new_feature_testing.build import FEATURE_CACHE, PHASE3_EVENTS_PATH, RESULTS_DIR

DEFAULT_START = "2019-01-01"
DEFAULT_END = "2025-12-31"
INCUMBENT = [STRUCTURAL, "vol_30d"]
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 51502
MIN_POSITIVE_YEARS = 4

# Own-stock price features must be present for a row to enter the common sample; peer
# aggregates may be missing and are imputed inside each training fold (PREREGISTRATION).
OWN_PRICE_REQUIRED = [
    "idio_vol_30d", "idio_vol_10d", "idio_share_30d", "max_abs_ret_20d", "jump_share_20d",
    "semivar_balance_20d", "vol_log_ratio_5_30", "vol_log_ratio_10_60", "abs_idio_ret_5d",
    "abs_idio_ret_20d", "sector_corr_60d", "sector_corr_change_20_120",
]
INTERACTION_OWN = "idio_vol_30d_z_own"
INTERACTION_PEER = "peer_sec_shock_20"


# ───────────────────────────────────── frame assembly ────────────────────────────────────
def z_own(d: pd.DataFrame, col: str) -> pd.Series:
    """Phase 5's `_z_own` transform, verbatim: expanding stats over the stock's PRIOR rows."""
    s = d.sort_values(["stock", "event_clock", "earnings_date"], kind="mergesort")
    grp = s.groupby("stock", sort=False)[col]
    mean_prior = grp.transform(lambda x: x.shift(1).expanding().mean())
    std_prior = grp.transform(lambda x: x.shift(1).expanding(min_periods=5).std())
    return ((s[col] - mean_prior) / std_prior.replace(0.0, np.nan)).reindex(d.index)


def assemble(events: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Phase 4/5 analysis frame with the Phase 5B columns joined on the event index."""
    analysis = derive_phase5_features(prepare_analysis_frame(events))
    new_cols = [c for c in feats.columns if c not in analysis.columns]
    analysis = analysis.join(feats[new_cols])
    analysis["idio_vol_30d_z_own"] = z_own(analysis, "idio_vol_30d")
    return analysis


def samples(analysis: pd.DataFrame, start: str, end: str) -> dict[str, pd.DataFrame]:
    window = _resolved_window(analysis, start, end)
    headline, _core = cohort_sample(window, HEADLINE_COHORT)
    audit = feature_audit(window, headline)
    eligible = audit.loc[audit["headline_eligible"], "feature"].tolist()
    phase5_common = headline.dropna(subset=["y_extreme", STRUCTURAL, *eligible]).copy()
    common = phase5_common.dropna(subset=OWN_PRICE_REQUIRED).copy()
    complete = common.dropna(subset=R.PRIMARY).copy()
    return {"window": window, "headline": headline, "phase5_common": phase5_common,
            "common": common, "complete_case": complete}


# ─────────────────────────────────────── diagnostics ─────────────────────────────────────
def feature_diagnostics(common: pd.DataFrame, headline: pd.DataFrame,
                        features: list[str]) -> pd.DataFrame:
    """Phase 5 univariate screen plus the Phase 5B extras (vs vol_30d, same-date)."""
    uni = univariate_signal(common, features).set_index("feature")
    rows = []
    for col in features:
        if col not in uni.index:
            continue
        d = common.dropna(subset=["y_extreme", col]).copy()
        sign = uni.at[col, "orientation"]
        d["_o"] = sign * pd.to_numeric(d[col], errors="coerce")
        same_date, _g, _e, sd_pairs = pooled_pair_auc(d, "_o", ["event_clock"])
        sp_vol = stats.spearmanr(d[col], d["vol_30d"]).statistic if len(d) > 2 else np.nan
        d["_vdec"] = pd.qcut(d["vol_30d"].rank(method="first"), 10, labels=False)
        vol_dec, _g, _e, _p = pooled_pair_auc(d, "_o", ["_vdec"])
        f = R.BY_NAME.get(col)
        rows.append({
            "feature": col,
            "family": f.family if f else "",
            "role": ("primary" if f.primary else "secondary") if f else "reference",
            "coverage_headline": float(headline[col].notna().mean()) if col in headline else np.nan,
            "coverage_common": float(common[col].notna().mean()),
            "spearman_vs_vol_30d": float(sp_vol),
            "same_date_pooled_auc": same_date,
            "same_date_pairs": sd_pairs,
            "auc_within_vol30_decile": vol_dec,
        })
    extra = pd.DataFrame(rows).set_index("feature")
    out = extra.join(uni.drop(columns=["derived_in_phase5"], errors="ignore"), how="left")
    return out.reset_index().sort_values(["role", "family", "feature"])


# ─────────────────────────────────── bootstrap (product) ─────────────────────────────────
def _top(y: np.ndarray, p: np.ndarray, frac: float) -> tuple[float, float]:
    k = max(1, int(math.ceil(frac * len(y))))
    top = y[np.argsort(-p, kind="mergesort")[:k]]
    tot = y.sum()
    return float(top.mean()), (float(top.sum() / tot) if tot > 0 else np.nan)


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    h10, c10 = _top(y, p, 0.10)
    _h20, c20 = _top(y, p, 0.20)
    return {"auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(np.mean((p - y) ** 2)), "top10_hit": h10, "top10_capture": c10,
            "top20_capture": c20}


def bootstrap_delta(pred_a: pd.DataFrame, pred_b: pd.DataFrame, *,
                    reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Stock-clustered CIs for metric(a) − metric(b) on identical rows.

    Same resampling unit as Phase 5's `paired_bootstrap_delta` (stocks), extended to the
    product metrics. Rows must match exactly; a mismatch is an error, not an inner join.
    """
    if not pred_a.index.equals(pred_b.index):
        raise ValueError("bootstrap_delta requires identical rows")
    y = pred_a["y_extreme"].to_numpy(int)
    a = pred_a["p"].to_numpy(float)
    b = pred_b["p"].to_numpy(float)
    point = {k: _metrics(y, a)[k] - _metrics(y, b)[k] for k in _metrics(y, a)}
    groups = pred_a.groupby("stock", sort=False).indices
    keys = list(groups)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in point}
    for _ in range(reps):
        pick = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([groups[keys[i]] for i in pick])
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        ma, mb = _metrics(yb, a[idx]), _metrics(yb, b[idx])
        for k in point:
            draws[k].append(ma[k] - mb[k])
    out = {"n": len(y), "reps_valid": len(draws["auc"])}
    for k, v in point.items():
        arr = np.asarray(draws[k], dtype=float)
        out[f"delta_{k}"] = v
        out[f"delta_{k}_lo"] = float(np.nanquantile(arr, 0.025))
        out[f"delta_{k}_hi"] = float(np.nanquantile(arr, 0.975))
    out["share_auc_positive"] = float((np.asarray(draws["auc"]) > 0).mean())
    return out


# ───────────────────────────────────── walk-forward ──────────────────────────────────────
def walk_forward_interaction(sample: pd.DataFrame, features: list[str], own: str,
                             peer: str) -> pd.DataFrame:
    """Walk-forward with ONE product term built from train-fold statistics only.

    Each component is median-imputed and standardised with the TRAINING block's constants,
    the product is formed, and the block model is fitted on features + product.
    """
    out = []
    for year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train = sample[sample["event_clock"] < cutoff].copy()
        test = sample[(sample["event_clock"] >= cutoff)
                      & (sample["event_clock"] < cutoff + pd.offsets.DateOffset(years=1))].copy()
        if len(test) == 0 or train["y_extreme"].nunique() < 2 or len(train) < 200:
            continue
        for frame in (train, test):
            frame["_ix"] = 1.0
        for col in (own, peer):
            med = train[col].median()
            mu = train[col].fillna(med).mean()
            sd = train[col].fillna(med).std(ddof=0) or 1.0
            for frame in (train, test):
                frame["_ix"] *= (frame[col].fillna(med) - mu) / sd
        feats = [*features, "_ix"]
        fit = fit_block(train, feats)
        out.append(pd.DataFrame({
            "stock": test["stock"].to_numpy(), "event_clock": test["event_clock"].to_numpy(),
            "test_year": year, "y_extreme": test["y_extreme"].to_numpy(),
            "p": apply_block(fit, test, feats), "train_n": len(train)}, index=test.index))
    return pd.concat(out)


def model_specs() -> list[tuple[str, list[str], str]]:
    """(label, features, kind). Fixed list — nothing is added after seeing results."""
    inc = INCUMBENT
    specs = [("A_structural", [STRUCTURAL], "reference"),
             ("B_incumbent_structural+vol_30d", inc, "incumbent"),
             ("B'_structural+idio_vol_30d", [STRUCTURAL, "idio_vol_30d"], "replacement"),
             ("B''_structural+vol_30d_cut", [STRUCTURAL, "vol_30d_cut"], "control")]
    specs += [(f"C_inc+{f}", [*inc, f], "primary_single") for f in R.PRIMARY]
    for fam in R.FAMILIES:
        specs.append((f"D_inc+family_{fam}", [*inc, *R.family_members(fam)], "family"))
    sub = [f for f in R.family_members(R.F5) if f.startswith("peer_sub_")]
    sec = [f for f in R.family_members(R.F5) if f.startswith("peer_sec_")]
    specs += [("D_inc+family_5_sub_only", [*inc, *sub], "family"),
              ("D_inc+family_5_sec_only", [*inc, *sec], "family"),
              ("E_inc+all_primary", [*inc, *R.PRIMARY], "all_primary"),
              ("R_inc+sector_vol_30d", [*inc, "sector_vol_30d"], "reference"),
              ("R_inc+sector_vol_30d+family_5_sub", [*inc, "sector_vol_30d", *sub], "reference"),
              ("R_inc+sector_vol_30d+family_6", [*inc, "sector_vol_30d",
                                                 *R.family_members(R.F6)], "reference"),
              ("R_inc+peer_mkt_exsec", [*inc, "peer_mkt_exsec_mean_abs_1d_20"], "reference"),
              ("R_inc+peer_mkt_exsec+family_5_sec",
               [*inc, "peer_mkt_exsec_mean_abs_1d_20", *sec], "reference")]
    specs += [(f"S_inc+{f}", [*inc, f], "secondary_single") for f in R.SECONDARY]
    return specs


def evaluate_models(common: pd.DataFrame, specs, *, reps: int, scope_tag: str = ""):
    preds = {}
    for label, feats, _kind in specs:
        preds[label] = walk_forward_predictions(common, feats)
    inc_label = "B_incumbent_structural+vol_30d"
    base_label = "A_structural"
    rows, yearly, boots = [], [], []
    for label, feats, kind in specs:
        pred = preds[label]
        r = score_predictions(pred, label, feats, "pooled_oof" + scope_tag)
        r["kind"] = kind
        rows.append(r)
        for y, g in pred.groupby("test_year"):
            ry = score_predictions(g, label, feats, f"year_{int(y)}")
            ry["test_year"] = int(y)
            ry["kind"] = kind
            gi = preds[inc_label].loc[g.index]
            ry["delta_auc_vs_incumbent"] = ry["roc_auc"] - roc_auc_score(
                gi["y_extreme"].astype(int), gi["p"])
            yearly.append(ry)
        if label != inc_label:
            b = bootstrap_delta(pred, preds[inc_label], reps=reps)
            b.update({"model": label, "kind": kind, "vs": "incumbent"})
            boots.append(b)
        if label in (inc_label, "B'_structural+idio_vol_30d"):
            b = bootstrap_delta(pred, preds[base_label], reps=reps)
            b.update({"model": label, "kind": kind, "vs": "structural"})
            boots.append(b)
    return pd.DataFrame(rows), pd.DataFrame(yearly), pd.DataFrame(boots), preds


def decisions(models: pd.DataFrame, yearly: pd.DataFrame, boots: pd.DataFrame) -> pd.DataFrame:
    b = boots[boots["vs"].eq("incumbent")].set_index("model")
    pos_years = (yearly.assign(pos=yearly["delta_auc_vs_incumbent"] > 0)
                 .groupby("model")["pos"].sum())
    n_years = yearly.groupby("model")["test_year"].nunique()
    out = models.set_index("model").join(
        b[[c for c in b.columns if c.startswith("delta_") or c == "share_auc_positive"]])
    out["years_delta_auc_positive"] = pos_years
    out["years_evaluated"] = n_years
    out["stat_incremental"] = out["delta_auc_lo"] > 0
    out["product_relevant"] = (out["delta_top10_capture_lo"] > 0) | (out["delta_top20_capture_lo"] > 0)
    out["stable"] = out["years_delta_auc_positive"] >= MIN_POSITIVE_YEARS
    out["robust_beat"] = out["stat_incremental"] & out["stable"]
    out["useful_to_breakwater"] = out["robust_beat"] & out["product_relevant"]
    for c in ("stat_incremental", "product_relevant", "stable", "robust_beat",
              "useful_to_breakwater"):
        # the incumbent has no delta against itself: undefined, not False
        out[c] = out[c].astype(object).where(out["delta_auc"].notna(), None)
    return out.reset_index()


# ───────────────────────────────────── peer diagnostics ──────────────────────────────────
def peer_conditional_rates(common: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Extreme-move rate by peer-shock tercile WITHIN structural terciles (descriptive)."""
    d = common.dropna(subset=[feature]).copy()
    d["structural_tercile"] = pd.qcut(d[STRUCTURAL].rank(method="first"), 3, labels=["low", "mid", "high"])
    d["peer_tercile"] = pd.qcut(d[feature].rank(method="first"), 3, labels=["low", "mid", "high"])
    g = d.groupby(["structural_tercile", "peer_tercile"], observed=True)["y_extreme"]
    t = g.agg(["mean", "size"]).reset_index().rename(columns={"mean": "extreme_rate", "size": "n"})
    t.insert(0, "feature", feature)
    return t


# ────────────────────────────────────────── driver ───────────────────────────────────────
def run(*, results_dir: Path = RESULTS_DIR, start: str = DEFAULT_START, end: str = DEFAULT_END,
        reps: int = BOOTSTRAP_REPS) -> dict:
    events = pd.read_parquet(PHASE3_EVENTS_PATH)
    feats = pd.read_parquet(FEATURE_CACHE).drop(columns=["stock", "earnings_date"])
    if not feats.index.equals(events.index):
        raise ValueError("feature cache is not aligned to the Phase 3 event frame; rebuild it")
    analysis = assemble(events, feats)
    S = samples(analysis, start, end)
    common, headline = S["common"], S["headline"]

    # 1. inventory + raw diagnostics
    inventory = R.feature_inventory()
    diag_feats = [*R.PRIMARY, *R.SECONDARY, "vol_30d", "vol_30d_z_own", "sector_vol_30d"]
    diag = feature_diagnostics(common, headline, diag_feats)

    # 2. walk-forward on the common sample
    specs = model_specs()
    models, yearly, boots, preds = evaluate_models(common, specs, reps=reps)

    # 3. gated interaction
    dec0 = decisions(models, yearly, boots).set_index("model")
    ck = results_dir / "_checkpoint"
    ck.mkdir(parents=True, exist_ok=True)
    models.to_pickle(ck / "models.pkl"); yearly.to_pickle(ck / "yearly.pkl")
    boots.to_pickle(ck / "boots.pkl")
    pd.to_pickle(preds, ck / "preds.pkl")
    gate_peer = bool(dec0.at["D_inc+family_5_peer_earnings", "stat_incremental"])
    gate_own = bool(dec0.at[f"C_inc+{INTERACTION_OWN}", "stat_incremental"])
    interaction = {"gate_family5_stat_incremental": gate_peer,
                   "gate_idio_vol_30d_z_own_stat_incremental": gate_own,
                   "run": gate_peer and gate_own,
                   "term": f"z({INTERACTION_OWN}) x z({INTERACTION_PEER})"}
    if interaction["run"]:
        mains = [*INCUMBENT, INTERACTION_OWN, INTERACTION_PEER]
        p_main = walk_forward_predictions(common, mains)
        p_ix = walk_forward_interaction(common, mains, INTERACTION_OWN, INTERACTION_PEER)
        interaction["main_effects_model"] = score_predictions(p_main, "mains", mains, "pooled_oof")
        interaction["interaction_model"] = score_predictions(p_ix, "mains+ix", mains, "pooled_oof")
        interaction["delta_vs_main_effects"] = bootstrap_delta(p_ix, p_main, reps=reps)
        interaction["delta_vs_incumbent"] = bootstrap_delta(
            p_ix, preds["B_incumbent_structural+vol_30d"], reps=reps)

    # 4. complete-case sensitivity (families + E only)
    cc = S["complete_case"]
    cc_specs = [s for s in specs if s[2] in ("reference", "incumbent", "family", "all_primary",
                                           "replacement")]
    cc_models, cc_yearly, cc_boots, _ = evaluate_models(cc, cc_specs, reps=reps,
                                                        scope_tag="_complete_case")
    cc_dec = decisions(cc_models, cc_yearly, cc_boots)

    dec = decisions(models, yearly, boots)

    # 5. peer diagnostics
    peer_cols = [f.name for f in R.FEATURES if f.family == R.F5]
    peer_diag = diag[diag["feature"].isin(peer_cols)].copy()
    cond = pd.concat([peer_conditional_rates(common, f) for f in
                      ("peer_sub_mean_abs_1d_20", "peer_sec_mean_abs_1d_20",
                       "peer_sec_shock_20", "peer_sub_shock_20", "peer_sec_frac_large_1d_20",
                       "peer_mkt_exsec_mean_abs_1d_20")], ignore_index=True)

    # ── write ──
    results_dir.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(results_dir / "feature_inventory.csv", index=False)
    diag.to_csv(results_dir / "feature_diagnostics.csv", index=False)
    dec.to_csv(results_dir / "walkforward_models.csv", index=False)
    yearly.to_csv(results_dir / "yearly_results.csv", index=False)
    boots.to_csv(results_dir / "bootstrap_deltas.csv", index=False)
    peer_diag.to_csv(results_dir / "peer_feature_diagnostics.csv", index=False)
    cond.to_csv(results_dir / "peer_conditional_rates.csv", index=False)
    cc_dec.to_csv(results_dir / "walkforward_models_complete_case.csv", index=False)

    def pooled(label, frame=dec):
        r = frame.set_index("model").loc[label]
        return {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
                for k, v in r.items() if k != "features"}

    keep = ["model", "kind", "roc_auc", "pr_auc", "top10_hit_rate", "top10_capture",
            "top20_capture", "within_stock_pooled_auc", "delta_auc", "delta_auc_lo",
            "delta_auc_hi", "delta_top10_capture", "delta_top10_capture_lo",
            "delta_top20_capture", "delta_top20_capture_lo", "years_delta_auc_positive",
            "stat_incremental", "product_relevant", "stable", "robust_beat",
            "useful_to_breakwater"]
    summary = {
        "phase": "phase5b_free_features",
        "preregistration": "phase_5b_new_feature_testing/PREREGISTRATION.md",
        "evaluation_window": {"start": start, "end": end},
        "incumbent": INCUMBENT,
        "samples": {k: int(len(v)) for k, v in S.items()},
        "common_sample": {"stocks": int(common["stock"].nunique()),
                          "base_rate": float(common["y_extreme"].mean()),
                          "oof_events": int(len(preds["A_structural"]))},
        "structural": pooled("A_structural"),
        "incumbent_model": pooled("B_incumbent_structural+vol_30d"),
        "incumbent_vs_structural": boots[(boots["vs"] == "structural")].to_dict(orient="records"),
        "families_and_all": dec[dec["kind"].isin(["family", "all_primary", "replacement",
                                                  "reference", "control"])][keep]
        .to_dict(orient="records"),
        "primary_singles_sorted": dec[dec["kind"].eq("primary_single")]
        .sort_values("delta_auc", ascending=False)[keep].to_dict(orient="records"),
        "robust_beats": dec.loc[dec["robust_beat"].eq(True), "model"].tolist(),
        "useful_to_breakwater": dec.loc[dec["useful_to_breakwater"].eq(True), "model"].tolist(),
        "interaction": interaction,
        "complete_case": cc_dec[keep].to_dict(orient="records"),
        "limitations": [
            "sector/sub_sector are today's classification, constant per ticker: not point-in-time",
            "walk-forward test years 2021-2025 only; univariate orientation is chosen on the "
            "whole cohort (diagnostic only, never used by a model)",
            "vol_30d (incumbent) is read on the stored earnings_date row; 14 BMO events in the "
            "window have a vendor announcement one session earlier, so their vol_30d includes "
            "the reaction day. vol_30d_cut is the leak-free control",
            "expanding own-history transforms (_z_own) have a mechanical within-stock bias; "
            "within-stock readings for them are compared with the seq reference, not with 0.5",
        ],
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    args = ap.parse_args(argv)
    s = run(results_dir=args.results_dir, reps=args.bootstrap_reps)
    print(json.dumps({k: s[k] for k in ("samples", "common_sample", "robust_beats",
                                         "useful_to_breakwater", "interaction")},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
