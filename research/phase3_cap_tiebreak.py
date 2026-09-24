"""Phase 3 variant test: break the score's tie at the 12% p75 ceiling.

Research only. No production scoring, threshold, weight or ingestion is changed, and no
tier cut point is re-fitted — that is the point of this particular variant.

The defect
----------
`event_explosiveness_score` is, for 97% of events, just

    score  =  85 x min(prior_p75 / 0.12, 1)  +  15

because the entropy term clips to 1.0 almost everywhere and contributes a flat offset. The
`.clip(0, 1)` is a hard ceiling: every stock whose prior 75th-percentile absolute reaction
reaches 12% collapses onto **one** value. Measured on the Phase 3 window that is 649 events
at exactly score 100.0, with zero non-capped events anywhere near them — the capped group
IS the tied maximum, exactly.

Those events are not interchangeable. Split them at their own median prior p75 and the
lower half realises a 42.8% extreme rate against 53.0% for the upper half: a ten-point
spread in outcome frequency inside a group the score declares identical.

Worse than losing ordering, it makes selection ARBITRARY. The top 5% of this sample is 648
events and the tied block is 649, so "the riskiest 5%" is currently decided by frame order,
and anything tighter than 5% is decided entirely by frame order.

The variant
-----------
The minimal intervention that could fix it, and deliberately the least invasive of the four
options considered: leave the score, the tiers and every published number alone, and break
the tie with the uncapped p75 the score already computed.

    V0  shipped        phase3_risk_score, unchanged
    V1  tie-break      score + (p75 - 0.12) where p75 > 0.12, else score
    V2  raw p75        p75 alone — a REFERENCE, not a proposal: it drops the ceiling, the
                       weights and the entropy term together, so it shows what the whole
                       transformation costs rather than what this change buys.

V1 is safe as arithmetic, not by assumption: the capped events are verified to be exactly
the tied maximum, so adding a non-negative term to them alone cannot reorder anything
against anything else. `assert_tiebreak_is_order_safe` enforces that and raises if a future
data change breaks it.

What can and cannot move
------------------------
The tier is untouched, so every tier-level number — hit rates, Wilson intervals, capture,
`selected_share` — must come out bit-identical. That is reported as a check, not a result.
Only the continuous ranking can move, and only inside the tied block, so the honest place
to look is top-K selection at K tight enough to cut into the tie, plus AUC restricted to
the capped group.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import EXTREME_EARNINGS_REACTION_THRESHOLD
from feature_engineering.announcement_timing import TARGET_AVAILABLE
from research.phase3_target_rebuild import (
    CORRECTED_TARGET,
    DEFAULT_END,
    DEFAULT_START,
    PHASE3_PREFIX,
    variant_metrics,
)

EVENTS_PATH = Path("output/phase3_target_rebuild/phase3_events.parquet")
RESULTS_DIR = Path("output/phase3_target_rebuild/cap_tiebreak")

SCORE = PHASE3_PREFIX + "risk_score"
TIER = PHASE3_PREFIX + "bucket"
P75_ROLLING = PHASE3_PREFIX + "abs_reaction_p75_rolling"
P75_EXPANDING = PHASE3_PREFIX + "abs_reaction_p75"
P75 = "_prior_p75"

# The ceiling in `feature_engineering.event_features.event_explosiveness_score`. Imported by
# value rather than re-derived, and NOT changed by this variant.
P75_CEILING = 0.12

TOP_K_FRACTIONS = (0.10, 0.05, 0.02, 0.01)
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 20260923

V_SHIPPED, V_TIEBREAK, V_RAW_P75 = "V0_shipped", "V1_tiebreak_in_cap", "V2_raw_p75_reference"


def prepare(events: pd.DataFrame, start: str = DEFAULT_START,
            end: str = DEFAULT_END) -> pd.DataFrame:
    """Phase 3's evaluation sample, with the uncapped p75 recovered alongside the score."""
    completed = ~events["is_pending"].astype(bool)
    target_ok = (events[CORRECTED_TARGET].notna()
                 & events["reaction_3d_anchored_status"].eq(TARGET_AVAILABLE))
    dates = pd.to_datetime(events[PHASE3_PREFIX + "announce_date"]
                           .fillna(events["earnings_date"]))
    in_window = dates.between(pd.Timestamp(start), pd.Timestamp(end))
    d = events[completed & target_ok & in_window].copy()
    d["_date"] = dates[completed & target_ok & in_window]
    d[P75] = d[P75_ROLLING].fillna(d[P75_EXPANDING])
    d = d.dropna(subset=[SCORE, TIER, P75, CORRECTED_TARGET]).copy()
    d["y"] = d[CORRECTED_TARGET].ge(EXTREME_EARNINGS_REACTION_THRESHOLD).astype(int)
    d["year"] = d["_date"].dt.year
    d["quarter"] = d["_date"].dt.to_period("Q").astype(str)
    d["event_id"] = d["stock"].astype(str) + "|" + d["_date"].dt.strftime("%Y-%m-%d")
    d["is_capped"] = d[P75] >= P75_CEILING
    return d.reset_index(drop=True)


def assert_tiebreak_is_order_safe(d: pd.DataFrame) -> dict:
    """The capped group must BE the tied maximum, or the tie-break is not a tie-break.

    If any non-capped event scored at or above the capped events, adding a positive term to
    the capped ones would reorder events that the shipped score deliberately separated, and
    the variant would no longer be the minimal intervention it claims to be.
    """
    capped, other = d.loc[d["is_capped"], SCORE], d.loc[~d["is_capped"], SCORE]
    facts = {
        "capped_events": int(len(capped)),
        "distinct_scores_among_capped": int(capped.nunique()),
        "capped_score": float(capped.iloc[0]) if len(capped) else float("nan"),
        "non_capped_at_or_above_capped_score": int((other >= capped.min()).sum())
        if len(capped) else 0,
    }
    if len(capped):
        if facts["distinct_scores_among_capped"] != 1:
            raise ValueError("capped events do not share one score; tie-break unsafe")
        if facts["non_capped_at_or_above_capped_score"] != 0:
            raise ValueError("a non-capped event scores at/above the cap; tie-break unsafe")
    facts["order_safe"] = True
    return facts


def add_variants(d: pd.DataFrame) -> pd.DataFrame:
    """V0 / V1 / V2 as three ranking columns over identical rows."""
    assert_tiebreak_is_order_safe(d)
    out = d.copy()
    out[V_SHIPPED] = out[SCORE]
    out[V_TIEBREAK] = out[SCORE] + (out[P75] - P75_CEILING).clip(lower=0)
    out[V_RAW_P75] = out[P75]
    return out


# ──────────────────────────────── top-K, with ties told honestly ─────────────────────────
def top_k_metrics(d: pd.DataFrame, score_col: str, fraction: float) -> dict:
    """Precision and capture in the top `fraction`, reporting what the tie does.

    Two numbers, because a tie at the cut is not a detail here — it is the finding.

    `deterministic_*` resolves ties by `event_id`, which is reproducible and
    outcome-independent but arbitrary in the sense that matters: nothing about the model
    chose it. `expected_*` is the exact mean over all random orderings of the tied block —
    what "the top 5%" is really worth when the cut falls inside a tie, since every tied
    event is equally likely to be picked. Where no tie straddles the cut the two agree.
    """
    n = len(d)
    k = max(1, int(math.ceil(fraction * n)))
    s = d[score_col].to_numpy(float)
    y = d["y"].to_numpy(int)
    total_hits = int(y.sum())

    order = np.lexsort((d["event_id"].to_numpy(), -s))
    det_hits = int(y[order[:k]].sum())

    # Exact expectation over random tie-breaking at the boundary.
    cut = np.sort(s)[::-1][k - 1]
    above = s > cut
    tied = s == cut
    n_above, hits_above = int(above.sum()), int(y[above].sum())
    n_tied, hits_tied = int(tied.sum()), int(y[tied].sum())
    need = k - n_above
    exp_hits = hits_above + (need * hits_tied / n_tied if n_tied else 0.0)

    base = float(y.mean())
    return {
        "k": k,
        "k_fraction": fraction,
        "deterministic_precision": det_hits / k,
        "deterministic_capture": det_hits / total_hits if total_hits else float("nan"),
        "deterministic_lift": (det_hits / k) / base if base else float("nan"),
        "expected_precision": exp_hits / k,
        "expected_capture": exp_hits / total_hits if total_hits else float("nan"),
        "expected_lift": (exp_hits / k) / base if base else float("nan"),
        "events_tied_at_the_cut": n_tied,
        "slots_decided_by_the_tie": max(0, need) if n_tied > 1 else 0,
        "share_of_selection_decided_by_tie": (max(0, need) / k) if n_tied > 1 else 0.0,
    }


def _auc(y, s) -> float:
    y = np.asarray(y, int)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def _ap(y, s) -> float:
    y = np.asarray(y, int)
    return float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def variant_row(d: pd.DataFrame, variant: str, cohort: str, scope: str) -> dict:
    y = d["y"].to_numpy(int)
    s = d[variant].to_numpy(float)
    row = {"variant": variant, "cohort": cohort, "scope": scope, "n": len(d),
           "base_rate": float(y.mean()), "roc_auc": _auc(y, s), "average_precision": _ap(y, s)}
    for f in TOP_K_FRACTIONS:
        m = top_k_metrics(d, variant, f)
        tag = f"top{int(f * 100)}"
        for key in ("deterministic_precision", "expected_precision", "expected_capture",
                    "expected_lift", "events_tied_at_the_cut",
                    "share_of_selection_decided_by_tie"):
            row[f"{tag}_{key}"] = m[key]
    # The change's direct target: ordering inside the tied block, and inside the tier that
    # block dominates. Both are NaN for V0 by construction — a constant cannot rank.
    cap = d[d["is_capped"]]
    row["within_cap_n"] = len(cap)
    row["within_cap_auc"] = _auc(cap["y"], cap[variant]) if len(cap) else float("nan")
    ha = d[d[TIER].astype(object).eq("High Alert")]
    row["within_high_alert_n"] = len(ha)
    row["within_high_alert_auc"] = _auc(ha["y"], ha[variant]) if len(ha) else float("nan")
    return row


def paired_block_bootstrap(d: pd.DataFrame, a: str, b: str, *, reps: int = BOOTSTRAP_REPS,
                           seed: int = BOOTSTRAP_SEED) -> dict:
    """b - a, resampling calendar-quarter blocks, on identical rows."""
    y = d["y"].to_numpy(int)
    sa, sb = d[a].to_numpy(float), d[b].to_numpy(float)
    blocks = d.groupby("quarter", sort=True).indices
    keys = list(blocks)

    def stats(idx):
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            return None
        sub = d.iloc[idx]
        # Bootstrapped at several K because the tie only binds where K cuts INTO the tied
        # block: at top-10% every capped event is selected either way and the difference is
        # identically zero, which is a fact about the selection rule, not an estimate.
        tops = []
        for frac in (0.05, 0.02, 0.01):
            tops.append(top_k_metrics(sub, b, frac)["expected_precision"]
                        - top_k_metrics(sub, a, frac)["expected_precision"])
        cap = sub[sub["is_capped"]]
        ca = _auc(cap["y"], cap[a]) if len(cap) and cap["y"].nunique() > 1 else np.nan
        cb = _auc(cap["y"], cap[b]) if len(cap) and cap["y"].nunique() > 1 else np.nan
        return (_auc(yy, sb[idx]) - _auc(yy, sa[idx]), *tops, cb - ca)

    point = stats(np.arange(len(d)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        picked = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([blocks[keys[i]] for i in picked])
        got = stats(idx)
        if got is not None:
            draws.append(got)
    arr = np.asarray(draws, float)
    names = ["delta_roc_auc", "delta_top5_expected_precision",
             "delta_top2_expected_precision", "delta_top1_expected_precision",
             "delta_within_cap_auc"]
    out = {"comparison": f"{b}_vs_{a}", "n": len(d), "blocks": len(keys),
           "reps_valid": int(len(arr))}
    for i, name in enumerate(names):
        out[name] = float(point[i]) if point else float("nan")
        col = arr[:, i] if len(arr) else np.array([])
        col = col[~np.isnan(col)]
        out[name + "_ci_lo"] = float(np.quantile(col, 0.025)) if len(col) else float("nan")
        out[name + "_ci_hi"] = float(np.quantile(col, 0.975)) if len(col) else float("nan")
        out[name + "_share_positive"] = float((col > 0).mean()) if len(col) else float("nan")
    return out


# ──────────────────────────────────────── the run ────────────────────────────────────────
def tier_identity_check(d: pd.DataFrame) -> pd.DataFrame:
    """Phase 3's own variant_metrics for each ranking, to prove the tier did not move.

    `variant_metrics` reads the tier column, which is the same object for all three
    rankings, so every tier field must be identical. Reported rather than asserted away:
    if a future edit to the variant touches the bucket, this table shows it immediately.
    """
    rows = []
    for variant in (V_SHIPPED, V_TIEBREAK, V_RAW_P75):
        for cohort, sample in _cohorts(d).items():
            rows.append(variant_metrics(
                sample, name=variant, cohort=cohort, target_col=CORRECTED_TARGET,
                score_col=variant, bucket_col=TIER))
    return pd.DataFrame(rows)


def _cohorts(d: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {"all_resolved": d,
            "mature_28": d[d[PHASE3_PREFIX + "n_prior_resolved_reactions"].ge(28)]}


def run(*, events_path: Path = EVENTS_PATH, results_dir: Path = RESULTS_DIR,
        start: str = DEFAULT_START, end: str = DEFAULT_END,
        bootstrap_reps: int = BOOTSTRAP_REPS) -> dict:
    events = pd.read_parquet(events_path)
    d = add_variants(prepare(events, start, end))
    safety = assert_tiebreak_is_order_safe(d)

    rows, yearly = [], []
    for cohort, sample in _cohorts(d).items():
        for variant in (V_SHIPPED, V_TIEBREAK, V_RAW_P75):
            rows.append(variant_row(sample, variant, cohort, "pooled"))
            if cohort == "all_resolved":
                for year, g in sample.groupby("year"):
                    yearly.append(variant_row(g, variant, cohort, f"year_{int(year)}"))

    deltas = []
    for cohort, sample in _cohorts(d).items():
        for a, b in ((V_SHIPPED, V_TIEBREAK), (V_SHIPPED, V_RAW_P75)):
            row = paired_block_bootstrap(sample, a, b, reps=bootstrap_reps)
            row["cohort"] = cohort
            deltas.append(row)

    tiers = tier_identity_check(d)
    tier_cols = [c for c in tiers.columns
                 if c not in ("variant", "cohort", "score_auc_ge_8")]
    unchanged = {}
    for cohort in _cohorts(d):
        sub = tiers[tiers["cohort"].eq(cohort)].set_index("variant")
        base = sub.loc[V_SHIPPED, tier_cols]
        unchanged[cohort] = bool(
            all(sub.loc[v, tier_cols].equals(base) for v in (V_TIEBREAK, V_RAW_P75)))

    metrics = pd.DataFrame(rows)
    yearly_df = pd.DataFrame(yearly)
    delta_df = pd.DataFrame(deltas)

    results_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / "variant_metrics.csv", index=False)
    yearly_df.to_csv(results_dir / "yearly_metrics.csv", index=False)
    delta_df.to_csv(results_dir / "paired_deltas.csv", index=False)
    tiers.to_csv(results_dir / "tier_identity_check.csv", index=False)
    d[["event_id", "stock", "_date", "year", "quarter", "y", TIER, P75, "is_capped",
       V_SHIPPED, V_TIEBREAK, V_RAW_P75]].to_csv(
        results_dir / "event_rankings.csv", index=False)

    cap = d[d["is_capped"]]
    summary = {
        "experiment": "phase3_cap_tiebreak",
        "status": "RESEARCH VARIANT — no production change, no threshold re-fit",
        "window": {"start": start, "end": end},
        "variants": {
            V_SHIPPED: "phase3_risk_score, unchanged",
            V_TIEBREAK: f"score + (prior_p75 - {P75_CEILING}) where p75 > {P75_CEILING}",
            V_RAW_P75: "prior p75 alone — reference, not a proposal",
        },
        "the_defect": {
            "p75_ceiling": P75_CEILING,
            "capped_events": int(len(cap)),
            "capped_share": float(d["is_capped"].mean()),
            "capped_share_of_high_alert": float(
                (d[TIER].astype(object).eq("High Alert") & d["is_capped"]).sum()
                / max((d[TIER].astype(object).eq("High Alert")).sum(), 1)),
            "capped_extreme_rate": float(cap["y"].mean()) if len(cap) else float("nan"),
            "overall_extreme_rate": float(d["y"].mean()),
            "capped_lower_half_extreme_rate": float(
                cap.loc[cap[P75] < cap[P75].median(), "y"].mean()) if len(cap) else float("nan"),
            "capped_upper_half_extreme_rate": float(
                cap.loc[cap[P75] >= cap[P75].median(), "y"].mean()) if len(cap) else float("nan"),
            "order_safety": safety,
        },
        "tier_metrics_unchanged": unchanged,
        "pooled_metrics": metrics.to_dict(orient="records"),
        "paired_deltas": delta_df.to_dict(orient="records"),
        "guardrails": [
            "tier cut points are NOT re-fitted; the bucket column is untouched",
            "the tie-break adds a non-negative term only to events verified to be the tied "
            "maximum, so it cannot reorder anything the score separated",
            "top-K reports both a deterministic tie rule and the exact expectation over "
            "random tie-breaking, because the cut falls inside the tie",
            "V2 (raw p75) is a reference for what the whole transformation costs, not a "
            "proposed change",
            "no production scoring, weight, threshold or ingestion is modified",
        ],
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", type=Path, default=EVENTS_PATH)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    a = ap.parse_args(argv)
    s = run(events_path=a.events, results_dir=a.results_dir, start=a.start, end=a.end,
            bootstrap_reps=a.bootstrap_reps)
    print(json.dumps({"the_defect": s["the_defect"],
                      "tier_metrics_unchanged": s["tier_metrics_unchanged"],
                      "paired_deltas": s["paired_deltas"]}, indent=2, default=str))
    print(f"\nwrote {a.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
