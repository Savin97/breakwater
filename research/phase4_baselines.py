"""Phase 4: simple causal baselines and signal decomposition for Breakwater.

Research-only. Reads the Phase-3 event artifact and asks whether corrected-history
model 0.3.1 earns its complexity against simple predictors built from information that
was available strictly before each earnings event.

Headline comparison: the mature-28 cohort, where every core baseline and 0.3.1 have
the same minimum historical information budget.

The suite separates three questions:
1. Phenotype: is "this stock often moves a lot on earnings" already enough?
2. Structural score: does p75 + entropy beat p75 alone?
3. Lift promotions: do final 0.3.1 tiers beat the structural tiers they promote?

No production code, thresholds, database, or vendor input is touched.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from config import EXTREME_EARNINGS_REACTION_THRESHOLD
from feature_engineering.announcement_timing import TARGET_AVAILABLE
from research.phase3_target_rebuild import CORRECTED_TARGET, PHASE3_PREFIX

PHASE3_EVENTS_PATH = Path("output/phase3_target_rebuild/phase3_events.parquet")
RESULTS_DIR = Path("output/phase4_baselines")
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2025-12-31"
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 40317

MODEL_SCORE = PHASE3_PREFIX + "risk_score"
N_PRIOR = PHASE3_PREFIX + "n_prior_resolved_reactions"

PROBABILITY_BASELINES = {
    "market_prior_extreme_rate",
    "sector_prior_extreme_rate",
    "stock_prior_extreme_rate",
    "stock_prior_extreme_rate_shrunk20",
    "recent8_extreme_rate",
}

# The clean apples-to-apples set for the headline mature-28 comparison.
CORE_MATURE28 = [
    "model_0_3_1_score",
    "stock_prior_extreme_rate",
    "stock_prior_extreme_rate_shrunk20",
    "last_abs_reaction",
    "recent4_mean_abs_reaction",
    "recent8_mean_abs_reaction",
    "recent8_extreme_rate",
    "long_mean_abs_reaction",
    "long_median_abs_reaction",
    "expanding_p75_abs_reaction",
    "rolling28_p75_abs_reaction",
]

CORE_MATURE8 = [
    c for c in CORE_MATURE28 if c != "rolling28_p75_abs_reaction"
]

TIER_MAP = {"Normal": 0.0, "Elevated": 1.0, "High Alert": 2.0}


def _event_clock(events: pd.DataFrame) -> pd.Series:
    """Observed announcement date where known, otherwise the stored event date."""
    ts = pd.to_datetime(events.get("phase3_announce_date"), errors="coerce")
    fallback = pd.to_datetime(events["earnings_date"], errors="coerce")
    return ts.fillna(fallback).dt.normalize()


def build_causal_baselines(
    events: pd.DataFrame,
    *,
    extreme_threshold: float = EXTREME_EARNINGS_REACTION_THRESHOLD,
    shrink_strength: float = 20.0,
) -> pd.DataFrame:
    """One row per event with baselines using only outcomes from PRIOR event dates.

    Histories are updated after an entire event-date group is scored. That matters for
    market/sector baselines: two companies reporting on the same date cannot observe one
    another's realized outcome merely because one row happened to sort first.
    """
    required = {"stock", "earnings_date", CORRECTED_TARGET, "is_pending"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events missing required columns: {sorted(missing)}")

    d = events.copy()
    d["_event_clock"] = _event_clock(d)
    if "sector" not in d.columns:
        d["sector"] = pd.NA

    # Current model/components are copied, not recomputed here.
    out = pd.DataFrame(index=d.index)
    out["model_0_3_1_score"] = pd.to_numeric(d.get(MODEL_SCORE), errors="coerce")
    model_p75 = pd.to_numeric(
        d.get(PHASE3_PREFIX + "abs_reaction_p75_rolling"), errors="coerce"
    )
    model_p75_fallback = pd.to_numeric(
        d.get(PHASE3_PREFIX + "abs_reaction_p75"), errors="coerce"
    )
    model_p75 = model_p75.fillna(model_p75_fallback)
    out["model_p75_component"] = (model_p75 / 0.12).clip(0, 1)
    out["model_entropy"] = pd.to_numeric(
        d.get(PHASE3_PREFIX + "reaction_entropy"), errors="coerce"
    )
    out["model_stock_bucket_lift"] = pd.to_numeric(
        d.get(PHASE3_PREFIX + "stock_bucket_lift"), errors="coerce"
    )
    structural = d.get(PHASE3_PREFIX + "bucket_structural")
    final = d.get(PHASE3_PREFIX + "bucket")
    out["model_structural_tier"] = (
        structural.astype(object).map(TIER_MAP) if structural is not None else np.nan
    )
    out["model_final_tier"] = (
        final.astype(object).map(TIER_MAP) if final is not None else np.nan
    )

    baseline_cols = [
        "market_prior_extreme_rate",
        "sector_prior_extreme_rate",
        "stock_prior_extreme_rate",
        "stock_prior_extreme_rate_shrunk20",
        "last_abs_reaction",
        "recent4_mean_abs_reaction",
        "recent8_mean_abs_reaction",
        "recent8_extreme_rate",
        "long_mean_abs_reaction",
        "long_median_abs_reaction",
        "expanding_p75_abs_reaction",
        "rolling28_p75_abs_reaction",
        "n_prior_resolved_baseline",
    ]
    for col in baseline_cols:
        out[col] = np.nan

    stock_abs: dict[str, list[float]] = defaultdict(list)
    stock_extreme: dict[str, list[float]] = defaultdict(list)
    sector_sum: dict[str, float] = defaultdict(float)
    sector_n: dict[str, int] = defaultdict(int)
    global_sum = 0.0
    global_n = 0

    ordered = d.sort_values(
        ["_event_clock", "stock", "earnings_date", "is_pending"],
        kind="mergesort",
        na_position="last",
    )

    for event_date, day in ordered.groupby("_event_clock", sort=True, dropna=False):
        global_prior = global_sum / global_n if global_n else np.nan

        # Score every row from state frozen before this event date.
        for idx, row in day.iterrows():
            stock = str(row["stock"])
            sector_raw = row.get("sector")
            sector = None if pd.isna(sector_raw) else str(sector_raw)

            ah = stock_abs[stock]
            eh = stock_extreme[stock]
            n = len(ah)
            out.at[idx, "n_prior_resolved_baseline"] = n

            out.at[idx, "market_prior_extreme_rate"] = global_prior
            if sector is not None and sector_n[sector]:
                out.at[idx, "sector_prior_extreme_rate"] = (
                    sector_sum[sector] / sector_n[sector]
                )

            if n:
                stock_rate = float(np.mean(eh))
                out.at[idx, "stock_prior_extreme_rate"] = stock_rate
                if pd.notna(global_prior):
                    out.at[idx, "stock_prior_extreme_rate_shrunk20"] = (
                        float(np.sum(eh)) + shrink_strength * global_prior
                    ) / (n + shrink_strength)
                out.at[idx, "last_abs_reaction"] = ah[-1]
                out.at[idx, "long_mean_abs_reaction"] = float(np.mean(ah))
                out.at[idx, "long_median_abs_reaction"] = float(np.median(ah))
                out.at[idx, "expanding_p75_abs_reaction"] = float(np.quantile(ah, 0.75))

            if n >= 4:
                out.at[idx, "recent4_mean_abs_reaction"] = float(np.mean(ah[-4:]))
            if n >= 8:
                out.at[idx, "recent8_mean_abs_reaction"] = float(np.mean(ah[-8:]))
                out.at[idx, "recent8_extreme_rate"] = float(np.mean(eh[-8:]))
            if n >= 28:
                out.at[idx, "rolling28_p75_abs_reaction"] = float(
                    np.quantile(np.asarray(ah[-28:], dtype=float), 0.75)
                )

        # Only after every row on this date has been scored may today's outcomes enter state.
        for idx, row in day.iterrows():
            if bool(row.get("is_pending", False)):
                continue
            target = pd.to_numeric(pd.Series([row.get(CORRECTED_TARGET)]), errors="coerce").iloc[0]
            status = row.get("reaction_3d_anchored_status")
            if pd.isna(target) or status != TARGET_AVAILABLE:
                continue

            stock = str(row["stock"])
            sector_raw = row.get("sector")
            sector = None if pd.isna(sector_raw) else str(sector_raw)
            target = float(target)
            extreme = float(target >= extreme_threshold)

            stock_abs[stock].append(target)
            stock_extreme[stock].append(extreme)
            global_sum += extreme
            global_n += 1
            if sector is not None:
                sector_sum[sector] += extreme
                sector_n[sector] += 1

    return out


def prepare_analysis_frame(events: pd.DataFrame) -> pd.DataFrame:
    """Attach causal baselines and evaluation bookkeeping to the Phase-3 event frame."""
    baselines = build_causal_baselines(events)
    d = pd.concat([events.copy(), baselines], axis=1)
    d["event_clock"] = _event_clock(d)
    d["year"] = d["event_clock"].dt.year
    d["y_extreme"] = np.where(
        d[CORRECTED_TARGET].notna(),
        d[CORRECTED_TARGET].ge(EXTREME_EARNINGS_REACTION_THRESHOLD).astype(float),
        np.nan,
    )
    return d


def _safe_auc(y: pd.Series, score: pd.Series) -> float:
    if len(y) == 0 or y.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y.astype(int), score))


def _safe_ap(y: pd.Series, score: pd.Series) -> float:
    if len(y) == 0 or y.nunique() < 2:
        return np.nan
    return float(average_precision_score(y.astype(int), score))


def _top_fraction_metrics(
    y: pd.Series,
    score: pd.Series,
    fraction: float,
) -> tuple[int, float, float, float]:
    n = len(y)
    if n == 0:
        return 0, np.nan, np.nan, np.nan
    k = max(1, int(math.ceil(fraction * n)))
    order = pd.DataFrame({"y": y.to_numpy(), "score": score.to_numpy()})
    order = order.sort_values("score", ascending=False, kind="mergesort")
    top = order.head(k)
    baseline = float(order["y"].mean())
    hit = float(top["y"].mean())
    lift = hit / baseline if baseline > 0 else np.nan
    total_hits = float(order["y"].sum())
    capture = float(top["y"].sum() / total_hits) if total_hits > 0 else np.nan
    return k, hit, lift, capture


def _within_stock_auc(d: pd.DataFrame, score_col: str) -> tuple[int, float, float]:
    """Macro and event-weighted AUC across stocks with both classes represented."""
    aucs: list[tuple[float, int]] = []
    for _stock, sub in d.groupby("stock", sort=False):
        s = sub.dropna(subset=["y_extreme", score_col])
        if len(s) < 4 or s["y_extreme"].nunique() < 2:
            continue
        aucs.append((float(roc_auc_score(s["y_extreme"].astype(int), s[score_col])), len(s)))
    if not aucs:
        return 0, np.nan, np.nan
    values = np.asarray([x[0] for x in aucs], dtype=float)
    weights = np.asarray([x[1] for x in aucs], dtype=float)
    return len(aucs), float(values.mean()), float(np.average(values, weights=weights))


def predictor_metrics(
    sample: pd.DataFrame,
    predictor: str,
    *,
    cohort: str,
    probability: bool = False,
) -> dict:
    d = sample.dropna(subset=["y_extreme", predictor]).copy()
    y = d["y_extreme"].astype(int)
    score = pd.to_numeric(d[predictor], errors="coerce")
    valid = score.notna()
    d, y, score = d.loc[valid], y.loc[valid], score.loc[valid]
    baseline = float(y.mean()) if len(y) else np.nan

    top10_n, top10_hit, top10_lift, top10_capture = _top_fraction_metrics(y, score, 0.10)
    top20_n, top20_hit, top20_lift, top20_capture = _top_fraction_metrics(y, score, 0.20)
    n_stocks, within_macro, within_weighted = _within_stock_auc(d, predictor)

    row = {
        "cohort": cohort,
        "predictor": predictor,
        "n": len(d),
        "base_rate": baseline,
        "roc_auc": _safe_auc(y, score),
        "average_precision": _safe_ap(y, score),
        "top10_n": top10_n,
        "top10_hit_rate": top10_hit,
        "top10_lift": top10_lift,
        "top10_capture": top10_capture,
        "top20_n": top20_n,
        "top20_hit_rate": top20_hit,
        "top20_lift": top20_lift,
        "top20_capture": top20_capture,
        "within_stock_n": n_stocks,
        "within_stock_auc_macro": within_macro,
        "within_stock_auc_weighted": within_weighted,
        "brier": np.nan,
        "log_loss": np.nan,
    }
    if probability and len(d):
        p = score.clip(1e-6, 1 - 1e-6)
        row["brier"] = float(brier_score_loss(y, p))
        row["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    return row


def _resolved_window(
    d: pd.DataFrame,
    start: str,
    end: str,
) -> pd.DataFrame:
    completed = ~d["is_pending"].astype(bool)
    target_ok = (
        d[CORRECTED_TARGET].notna()
        & d["reaction_3d_anchored_status"].eq(TARGET_AVAILABLE)
    )
    in_window = d["event_clock"].between(pd.Timestamp(start), pd.Timestamp(end))
    return d[completed & target_ok & in_window].copy()


def cohort_sample(d: pd.DataFrame, cohort: str) -> tuple[pd.DataFrame, list[str]]:
    """Return a same-row core comparison sample and its core predictor list."""
    if cohort == "mature_28":
        predictors = CORE_MATURE28
        out = d[d[N_PRIOR].ge(28)].copy()
    elif cohort == "mature_8":
        predictors = CORE_MATURE8
        out = d[d[N_PRIOR].ge(8)].copy()
    elif cohort == "all_with_history":
        predictors = [
            "model_0_3_1_score",
            "stock_prior_extreme_rate",
            "stock_prior_extreme_rate_shrunk20",
            "last_abs_reaction",
            "long_mean_abs_reaction",
            "long_median_abs_reaction",
            "expanding_p75_abs_reaction",
        ]
        out = d[d["n_prior_resolved_baseline"].ge(1)].copy()
    else:
        raise ValueError(f"unknown cohort: {cohort}")

    # Every core predictor is evaluated on exactly the same rows inside a cohort.
    out = out.dropna(subset=["y_extreme", *predictors]).copy()
    return out, predictors


def yearly_metrics(sample: pd.DataFrame, predictors: list[str], cohort: str) -> pd.DataFrame:
    rows = []
    for year, yr in sample.groupby("year", sort=True):
        for predictor in predictors:
            row = predictor_metrics(
                yr,
                predictor,
                cohort=cohort,
                probability=predictor in PROBABILITY_BASELINES,
            )
            row["year"] = int(year)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap_auc_delta(
    sample: pd.DataFrame,
    model_col: str,
    baseline_col: str,
    *,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Paired event bootstrap CI for AUC(model) - AUC(baseline)."""
    d = sample.dropna(subset=["y_extreme", model_col, baseline_col]).copy()
    y = d["y_extreme"].astype(int).to_numpy()
    model = pd.to_numeric(d[model_col], errors="coerce").to_numpy(dtype=float)
    baseline = pd.to_numeric(d[baseline_col], errors="coerce").to_numpy(dtype=float)
    if len(d) == 0 or len(np.unique(y)) < 2:
        return {
            "n": len(d),
            "reps_requested": reps,
            "reps_valid": 0,
            "model_auc": np.nan,
            "baseline_auc": np.nan,
            "delta_auc": np.nan,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
        }

    model_auc = float(roc_auc_score(y, model))
    baseline_auc = float(roc_auc_score(y, baseline))
    rng = np.random.default_rng(seed)
    deltas = []
    n = len(d)
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(
            roc_auc_score(yb, model[idx]) - roc_auc_score(yb, baseline[idx])
        )
    arr = np.asarray(deltas, dtype=float)
    return {
        "n": n,
        "reps_requested": reps,
        "reps_valid": len(arr),
        "model_auc": model_auc,
        "baseline_auc": baseline_auc,
        "delta_auc": model_auc - baseline_auc,
        "ci_lo": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
        "ci_hi": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
    }


def component_decomposition(sample: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Direct tests of the two pieces of extra 0.3.1 complexity."""
    pairs = [
        (
            "entropy_increment",
            "model_p75_component",
            "model_0_3_1_score",
            "Does adding entropy improve the exact capped-p75 component used by 0.3.1?",
        ),
        (
            "lift_promotion_increment",
            "model_structural_tier",
            "model_final_tier",
            "Do lift-based promotions improve the three-level structural tier?",
        ),
    ]
    rows = []
    for name, before, after, question in pairs:
        d = sample.dropna(subset=["y_extreme", before, after])
        y = d["y_extreme"].astype(int)
        before_auc = _safe_auc(y, d[before])
        after_auc = _safe_auc(y, d[after])
        rows.append({
            "cohort": cohort,
            "component": name,
            "question": question,
            "n": len(d),
            "before_predictor": before,
            "before_auc": before_auc,
            "after_predictor": after,
            "after_auc": after_auc,
            "delta_auc": after_auc - before_auc
            if pd.notna(before_auc) and pd.notna(after_auc)
            else np.nan,
        })
    return pd.DataFrame(rows)


def run(
    *,
    events_path: Path = PHASE3_EVENTS_PATH,
    results_dir: Path = RESULTS_DIR,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    bootstrap_reps: int = BOOTSTRAP_REPS,
) -> dict:
    events = pd.read_parquet(events_path)
    analysis = prepare_analysis_frame(events)
    window = _resolved_window(analysis, start, end)

    all_metrics = []
    yearly_frames = []
    component_frames = []
    cohort_sizes = {}

    cohort_data: dict[str, pd.DataFrame] = {}
    for cohort in ("all_with_history", "mature_8", "mature_28"):
        sample, core = cohort_sample(window, cohort)
        cohort_data[cohort] = sample
        cohort_sizes[cohort] = len(sample)

        # Core predictors: strict common sample.
        for predictor in core:
            all_metrics.append(
                predictor_metrics(
                    sample,
                    predictor,
                    cohort=cohort,
                    probability=predictor in PROBABILITY_BASELINES,
                )
            )

        # Diagnostics that are not allowed to shrink the common headline sample.
        optional = [
            "market_prior_extreme_rate",
            "sector_prior_extreme_rate",
            "model_p75_component",
            "model_entropy",
            "model_stock_bucket_lift",
            "model_structural_tier",
            "model_final_tier",
        ]
        for predictor in optional:
            if predictor in sample.columns:
                all_metrics.append(
                    predictor_metrics(
                        sample,
                        predictor,
                        cohort=cohort,
                        probability=predictor in PROBABILITY_BASELINES,
                    )
                )

        yearly_frames.append(yearly_metrics(sample, core, cohort))
        if cohort == "mature_28":
            component_frames.append(component_decomposition(sample, cohort))

    metrics = pd.DataFrame(all_metrics)
    yearly = pd.concat(yearly_frames, ignore_index=True)
    components = pd.concat(component_frames, ignore_index=True)

    headline = cohort_data["mature_28"]
    simple_names = [
        c for c in CORE_MATURE28 if c != "model_0_3_1_score"
    ]
    mature_rows = metrics[
        metrics["cohort"].eq("mature_28") & metrics["predictor"].isin(simple_names)
    ].dropna(subset=["roc_auc"])
    best_simple = (
        mature_rows.sort_values("roc_auc", ascending=False).iloc[0]["predictor"]
        if len(mature_rows)
        else None
    )
    bootstrap = (
        paired_bootstrap_auc_delta(
            headline,
            "model_0_3_1_score",
            str(best_simple),
            reps=bootstrap_reps,
        )
        if best_simple
        else {}
    )

    # Year-level stability summary for the mature-28 common sample.
    stability_rows = []
    mature_yearly = yearly[yearly["cohort"].eq("mature_28")]
    for predictor, sub in mature_yearly.groupby("predictor"):
        valid_auc = sub["roc_auc"].dropna()
        stability_rows.append({
            "predictor": predictor,
            "years": int(valid_auc.size),
            "mean_year_auc": float(valid_auc.mean()) if len(valid_auc) else np.nan,
            "min_year_auc": float(valid_auc.min()) if len(valid_auc) else np.nan,
            "max_year_auc": float(valid_auc.max()) if len(valid_auc) else np.nan,
            "mean_year_top10_lift": float(sub["top10_lift"].mean()),
        })
    stability = pd.DataFrame(stability_rows).sort_values(
        "mean_year_auc", ascending=False, na_position="last"
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    analysis.to_parquet(results_dir / "phase4_analysis_frame.parquet", index=False)
    metrics.to_csv(results_dir / "baseline_metrics.csv", index=False)
    yearly.to_csv(results_dir / "yearly_metrics.csv", index=False)
    stability.to_csv(results_dir / "yearly_stability.csv", index=False)
    components.to_csv(results_dir / "component_decomposition.csv", index=False)

    # Compact machine-readable headline.
    headline_metrics = (
        metrics[metrics["cohort"].eq("mature_28")]
        .sort_values("roc_auc", ascending=False, na_position="last")
    )
    summary = {
        "phase": "phase4_baseline_decomposition",
        "source": str(events_path),
        "evaluation_start": start,
        "evaluation_end": end,
        "headline_cohort": "mature_28",
        "cohort_sizes": cohort_sizes,
        "best_simple_baseline": best_simple,
        "paired_bootstrap_model_minus_best_simple": bootstrap,
        "headline_metrics": headline_metrics.to_dict(orient="records"),
        "component_decomposition": components.to_dict(orient="records"),
        "guardrails": [
            "all baselines use only resolved corrected outcomes from prior event dates",
            "same-day outcomes update history only after every event on that date is scored",
            "headline mature-28 core predictors are evaluated on exactly the same rows",
            "probability losses are reported only for predictors that are probabilities",
            "this phase benchmarks the existing structural model; it does not tune weights or thresholds",
        ],
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=Path, default=PHASE3_EVENTS_PATH)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    args = ap.parse_args()
    run(
        events_path=args.events,
        results_dir=args.results_dir,
        start=args.start,
        end=args.end,
        bootstrap_reps=args.bootstrap_reps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
