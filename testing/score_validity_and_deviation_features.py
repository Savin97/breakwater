"""Is the shipped score valid, and do growth-deviation features add to it?

Research evaluation only. No production scoring, threshold, ingestion or database is
changed, no API is called, and `main.py` is never run.

Two questions, deliberately kept apart
---------------------------------------
**Validity** is a question about code and time: can anything the score sees at prediction
time have been unknowable then? That is answered by tracing the score from its inputs to
the cutoff, not by looking at its AUC. A leaky score with a good AUC is worse than useless.

**Performance** is a question about history: given a score that is valid, how well did it
rank risk, and does anything improve on it? That is answered on held-out events.

Neither answers the third question — how a system would do live — and this module never
claims to. The Benzinga estimates are latest-vintage (`last_updated` on a 2014 event is
routinely a 2021-2023 timestamp, and the prior-period actuals forming the growth rates are
restated). That is a **limitation of unknown sign**: revisions could flatter a feature by
encoding what actually happened, or handicap it by replacing the consensus that actually
stood before the announcement. It is not a proven upper bound and is not treated as one.

What the validity audit found
------------------------------
One material defect, in the research score only, and it is disclosed rather than quietly
patched. `score_corrected_shadow` reproduces a known 0.3.1 quirk — the entropy fallback
`entropy.mask(is_pending).ffill()` — which forward-fills in FRAME order. The Phase 3 event
frame is sorted by (stock, earnings_date), so the row preceding a stock's first events is
the *last* event of the previous ticker alphabetically. Measured on the real frame: 23,725
events take their entropy from the chain, **99.8% of them from a future-dated row and 100%
from a different stock** — AAPL's 2000-04-19 event is filled from ticker A's 2026-08-26
event. On the held-out era this touches 2.29% of events (338 of 14,729) and moves the score
by up to 15 points.

The repair is confined to research: `repaired_risk_score()` replaces the frame-order fill
with a date-ordered one that can only see strictly earlier announcement dates. Both scores
are carried through every comparison, so the original reference is retained and the size of
the defect is visible rather than argued about. Production is untouched.

Three further disclosures, none of them repairable here:
* The 0.12 p75 ceiling, the 0.85/0.15 weights, the 73/79 tier cuts, `LIFT_PRIOR_STRENGTH`
  and the 1.5x/3.0x promotion gates were all selected on overlapping historical data. The
  score is therefore not a pristine out-of-sample instrument, whatever its AUC says.
* `_missing_aware_lift` advances its global prior with `.shift(1)` by ROW, so an event can
  see outcomes of other companies announcing on the SAME DATE (97.8% of events share their
  date with at least one other; up to 71 land on one day). This affects `phase3_bucket`,
  the tier — never `phase3_risk_score`, which does not read the lift.
* A 0-100 score is not a probability. Nothing in 0.3.1 calibrates it, so "79" does not mean
  79% and tier rates below are empirical frequencies, not the score's own claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BUCKET_ELEVATED_FLOOR,
    BUCKET_HIGH_ALERT_FLOOR,
    EXTREME_EARNINGS_REACTION_THRESHOLD,
    LARGE_EARNINGS_REACTION_THRESHOLD,
    LIFT_PRIOR_STRENGTH,
    LIFT_TO_ELEVATED,
    LIFT_TO_HIGH_ALERT,
    MODEL_VERSION,
)
from research.phase3_target_rebuild import CORRECTED_TARGET, PHASE3_PREFIX
from testing.benzinga_feature_value import (
    ALERT_QUANTILE,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    CLASSIFIER_PARAMS,
    FIRST_TEST_YEAR,
    LAST_TEST_YEAR,
    PRIMARY_STATUS,
    PRIMARY_TARGET,
    PRIMARY_THRESHOLD,
    REACTION_SESSIONS,
    SECONDARY_THRESHOLD,
    _ap,
    _auc,
    _git_state,
    _sha256,
    alert_metrics,
    build_dataset,
    load_benzinga,
    load_event_frame,
    quarter_block_bootstrap,
    walk_forward,
    within_stock_macro_auc,
)

RESULTS_DIR = Path("testing/testing_results/score_validity_and_deviation_features")

SCORE = PHASE3_PREFIX + "risk_score"
SCORE_REPAIRED = PHASE3_PREFIX + "risk_score_repaired"
TIER = PHASE3_PREFIX + "bucket"
ENTROPY = PHASE3_PREFIX + "reaction_entropy"
P75_ROLLING = PHASE3_PREFIX + "abs_reaction_p75_rolling"
P75_EXPANDING = PHASE3_PREFIX + "abs_reaction_p75"

# The simple causal historical-risk reference, frozen at Phase 4's settings: the stock's
# own prior extreme-event frequency shrunk toward the prior market rate with strength 20,
# built from the corrected target, with same-date outcomes entering history only after
# every event on that date has been scored.
HISTORICAL_REFERENCE = "stock_prior_extreme_rate_shrunk20"
HISTORICAL_REFERENCE_SHRINKAGE = 20.0

# M1's four features, fixed by the brief. The `abs_` pair are deterministic functions of
# the signed pair, so both share the signed pair's coverage.
DEVIATION_FEATURES = [
    "eps_growth_vs_normal",
    "revenue_growth_vs_normal",
    "abs_eps_growth_vs_normal",
    "abs_revenue_growth_vs_normal",
]
DEVIATION_COVERAGE_KEYS = ["eps_growth_vs_normal", "revenue_growth_vs_normal"]

MATCHED_ALERT_FRACTION = 0.10       # top decile within each held-out calendar year
TIER_ORDER = ["Normal", "Elevated", "High Alert"]


# ─────────────────────────── the research repair, isolated ───────────────────────────────
def causal_entropy_fallback(events: pd.DataFrame, date_col: str = "_sd") -> pd.Series:
    """Last entropy observed on a STRICTLY EARLIER announcement date.

    The shipped chain is `entropy.mask(is_pending).ffill()` over the frame as ordered, and
    the Phase 3 frame is ordered by (stock, earnings_date). "The previous row" is therefore
    the previous TICKER's last event, which is almost always dated in the future.

    This replacement keeps the intent — fall back on the most recent known entropy when a
    stock has fewer than eight of its own observations — while making "most recent" mean
    what it says. Collapsing to one value per date first is what guarantees strictness: a
    row can never be filled from another row announcing on its own date.
    """
    d = events.assign(_ent=events[ENTROPY].mask(events["is_pending"].astype(bool)))
    per_date = d.groupby(date_col)["_ent"].last().sort_index()
    prior = per_date.shift(1).ffill()
    return events[date_col].map(prior)


def repaired_risk_score(events: pd.DataFrame) -> pd.DataFrame:
    """0.3.1's structural score with the look-ahead removed and nothing else changed.

    Same 0.85/0.15 weights, same 0.12 ceiling, same clipping, same p75 fallback chain. The
    only difference is where a missing entropy value comes from.
    """
    out = events.copy()
    out["_sd"] = pd.to_datetime(
        out[PHASE3_PREFIX + "announce_date"].fillna(out["earnings_date"])).dt.normalize()
    p75 = out[P75_ROLLING].fillna(out[P75_EXPANDING])
    e3 = (p75 / 0.12).clip(0, 1)
    entropy = out[ENTROPY]
    causal_chain = causal_entropy_fallback(out)
    e4 = np.clip(entropy.fillna(causal_chain).fillna(0), 0, 1)
    out[SCORE_REPAIRED] = 100 * np.clip(0.85 * e3 + 0.15 * e4, 0, 1)

    shipped_chain = entropy.mask(out["is_pending"].astype(bool)).ffill()
    out["entropy_source"] = np.where(
        entropy.notna(), "own_history",
        np.where(shipped_chain.notna(), "ffill_chain", "zero_default"))
    out["score_repair_delta"] = out[SCORE_REPAIRED] - out[SCORE]
    return out


# ────────────────────────────────── 1. validity audit ────────────────────────────────────
def _check(name: str, question: str, passed: bool | None, evidence: str,
           severity: str = "info") -> dict:
    return {"check": name, "question": question,
            "result": "PASS" if passed is True else "FAIL" if passed is False else "DISCLOSURE",
            "severity": severity, "evidence": evidence}


def audit_score_validity(events: pd.DataFrame, dataset: pd.DataFrame) -> list[dict]:
    """Trace the shipped score from source inputs to the prediction cutoff.

    Each check is a question that could have a wrong answer, with the evidence attached.
    Passing checks are reported too: an audit that only lists failures cannot be read as
    coverage of anything.
    """
    checks = []
    sd = pd.to_datetime(events[PHASE3_PREFIX + "announce_date"]
                        .fillna(events["earnings_date"])).dt.normalize()
    completed = events[~events["is_pending"].astype(bool)].copy()

    # C1 — can the current event's own outcome reach its own score?
    # `_target_history_features` writes each row's statistics BEFORE appending that row's
    # reaction, so the score is a function of strictly prior reactions. Verified by a
    # direct perturbation test in the test suite; here we verify the structural claim that
    # every score input is an "n prior" statistic and n is never inflated by self.
    n_prior = events[PHASE3_PREFIX + "n_prior_resolved_reactions"]
    by_stock_pos = events.sort_values(["stock", "earnings_date"]).groupby("stock").cumcount()
    monotone = (events.sort_values(["stock", "earnings_date"])
                .groupby("stock")[PHASE3_PREFIX + "n_prior_resolved_reactions"]
                .apply(lambda s: s.is_monotonic_increasing).all())
    checks.append(_check(
        "current_event_outcome_excluded",
        "Can this event's own reaction enter its own pre-event score?",
        bool(monotone and (n_prior <= by_stock_pos.reindex(n_prior.index)).all()),
        f"n_prior_resolved_reactions is monotone non-decreasing within every stock and "
        f"never exceeds the event's own position in that stock's sequence "
        f"(max n_prior={int(n_prior.max())}); statistics are written before the current "
        f"reaction is appended. Perturbation test: test_current_event_reaction_cannot_"
        f"change_its_own_score."))

    # C2 — were prior outcomes actually finished by this event's cutoff?
    gaps = (completed.sort_values(["stock", "_ec" if "_ec" in completed else "earnings_date"])
            .assign(_d=lambda d: pd.to_datetime(d["earnings_date"]))
            .groupby("stock")["_d"].diff().dt.days)
    min_gap = float(gaps.min())
    checks.append(_check(
        "prior_outcomes_complete_at_cutoff",
        f"Could a prior event's {REACTION_SESSIONS}-session window still be open when the "
        "next event is scored?",
        bool(min_gap > 10),
        f"Minimum within-stock gap between consecutive events is {min_gap:.0f} calendar "
        f"days ({int((gaps <= 10).sum())} gaps <= 10 days), so a prior event's "
        f"{REACTION_SESSIONS}-session window is always closed. The history is ordered by "
        "announcement date, which is therefore also completion order."))

    # C3 — does the score read any price input at all?
    score_inputs = [P75_ROLLING, P75_EXPANDING, ENTROPY]
    checks.append(_check(
        "price_inputs_precede_cutoff",
        "Do price inputs used by the score precede the prediction cutoff?",
        True,
        "The score reads no price column. Its only inputs are "
        f"{score_inputs}, all quantiles/entropies of strictly prior corrected reactions. "
        "Price enters only through those completed reactions, never as a live series."))

    # C4 — announcement timing in the outcome anchor.
    # The reference must be the announcement date the anchor was actually computed from,
    # not `earnings_date`. Phase 3 anchors on the verified Benzinga announcement date, which
    # for a small number of events is `earnings_date + 1`; comparing against `earnings_date`
    # flags those as violations when the anchoring is correct and the stored calendar date
    # is the thing that is a day out.
    windows = completed["announce_window"].value_counts(dropna=False).to_dict()
    announce = pd.to_datetime(completed[PHASE3_PREFIX + "announce_date"]
                              .fillna(completed["earnings_date"])).dt.normalize()
    anchor = pd.to_datetime(completed["anchor_date"])
    has_anchor = anchor.notna()
    bmo = has_anchor & completed["announce_window"].eq("BMO")
    amc = has_anchor & completed["announce_window"].eq("AMC")
    bmo_bad = int((anchor[bmo] >= announce[bmo]).sum())
    amc_bad = int((anchor[amc] > announce[amc]).sum())
    date_shifted = int((announce != pd.to_datetime(completed["earnings_date"])).sum())
    checks.append(_check(
        "outcome_anchor_handles_timing",
        "Does the outcome anchor respect BMO/AMC announcement timing?",
        bool(bmo_bad == 0 and amc_bad == 0),
        f"Window mix on completed events: {windows}. Measured against the announcement date "
        f"the anchor was computed from: {int(bmo.sum())} BMO events, all anchored strictly "
        f"earlier ({bmo_bad} violations); {int(amc.sum())} AMC events, all anchored on or "
        f"before ({amc_bad} violations). Note that {date_shifted} completed events carry a "
        "verified announcement date one day after the stored `earnings_date`, so their "
        "anchor correctly sits ON `earnings_date` for BMO and one day after it for AMC. "
        "Anchoring itself is Phase 2's externally reviewed session-grid implementation; "
        "this re-asserts the directional invariant only."))

    # C5 — missing outcomes
    tgt = completed[PRIMARY_TARGET]
    status_ok = completed[PRIMARY_STATUS].eq("available")
    contradictions = int((tgt.notna() & ~status_ok).sum() + (tgt.isna() & status_ok).sum())
    extreme = completed[PHASE3_PREFIX + "is_extreme_reaction"]
    checks.append(_check(
        "missing_outcomes_stay_missing",
        "Is an unavailable outcome ever recorded as a non-extreme event?",
        bool(contradictions == 0 and (extreme.isna() == tgt.isna()).all()),
        f"{int(tgt.isna().sum())} completed events have no corrected target; "
        f"phase3_is_extreme_reaction is NaN on exactly those rows "
        f"({int(extreme.isna().sum())}), never 0. Status/target contradictions: "
        f"{contradictions}. The evaluation dataset drops them before labels are cut."))

    # C6 — joins
    checks.append(_check(
        "event_joins_are_one_to_one",
        "Could a vendor or event duplicate fan out an event?",
        bool(not dataset.duplicated(["stock", "earnings_date"]).any()
             and dataset["event_id"].is_unique),
        f"{len(dataset)} rows, {dataset['event_id'].nunique()} unique event ids, "
        "0 duplicates on (stock, earnings_date). The join asserts "
        "validate='one_to_one' and raises on violation."))

    # C7 — the defect
    ent = events[ENTROPY]
    chain = ent.mask(events["is_pending"].astype(bool)).ffill()
    uses_chain = ent.isna() & chain.notna()
    src = pd.Series(np.where(ent.notna() & ~events["is_pending"].astype(bool),
                             np.arange(len(events)), np.nan)).ffill()
    ok = uses_chain.to_numpy() & src.notna().to_numpy()
    future = int((sd.to_numpy()[ok] < sd.to_numpy()[src[ok].astype(int)]).sum())
    cross = int((events["stock"].to_numpy()[ok]
                 != events["stock"].to_numpy()[src[ok].astype(int)]).sum())
    era = (uses_chain.groupby(sd.dt.year // 5 * 5).sum().astype(int).to_dict())
    eval_keys = set(zip(dataset["stock"], pd.to_datetime(dataset["earnings_date"])))
    in_eval = int(sum(
        1 for flag, stock, day in zip(uses_chain, events["stock"],
                                      pd.to_datetime(events["earnings_date"]))
        if flag and (stock, day) in eval_keys))
    checks.append(_check(
        "entropy_fallback_is_frame_ordered",
        "Does the entropy fallback ever read a future or foreign row?",
        False,
        f"MATERIAL DEFECT. {int(uses_chain.sum())} events take entropy from the "
        f"frame-order ffill chain; {future} of them ({future / max(ok.sum(), 1):.1%}) are "
        f"filled from a FUTURE-dated row and {cross} ({cross / max(ok.sum(), 1):.1%}) from "
        "a DIFFERENT stock, because the frame is sorted by (stock, earnings_date). "
        "Repaired for research only by causal_entropy_fallback(); the shipped score is "
        "retained unchanged as the reference and production is not touched.",
        severity="material"))

    # C8-C10 — disclosures that cannot be repaired here
    checks.append(_check(
        "historically_selected_constants",
        "Were any constants in the score selected on data overlapping the evaluation?",
        None,
        f"Yes, and all of them: the 0.12 p75 ceiling, the 0.85/0.15 weights, the "
        f"{BUCKET_ELEVATED_FLOOR}/{BUCKET_HIGH_ALERT_FLOOR} tier cuts, "
        f"LIFT_PRIOR_STRENGTH={LIFT_PRIOR_STRENGTH} and the "
        f"{LIFT_TO_ELEVATED}x/{LIFT_TO_HIGH_ALERT}x promotion gates. None is refitted per "
        "fold, so there is no fold-level leakage, but absolute performance is optimistically "
        "biased and this is a controlled variant comparison, not an OOS certification.",
        severity="disclosure"))

    dup_dates = sd[~events["is_pending"].astype(bool)].duplicated(keep=False)
    checks.append(_check(
        "lift_global_prior_sees_same_day_rows",
        "Can the tier's lift prior see outcomes announced on the same day?",
        False,
        f"Yes, for the TIER only. _missing_aware_lift advances its global prior with "
        f".shift(1) by ROW, so an event can see same-date outcomes that sort before it. "
        f"{int(dup_dates.sum())} completed events ({dup_dates.mean():.1%}) share their "
        f"announcement date with another event; up to "
        f"{int(sd[~events['is_pending'].astype(bool)].value_counts().max())} land on one "
        "day. phase3_risk_score does NOT read the lift and is unaffected; phase3_bucket is. "
        "Tier results below are reported as descriptive with this caveat attached.",
        severity="material_tier_only"))

    checks.append(_check(
        "score_is_not_a_probability",
        "Does a 0-100 score mean a probability?",
        None,
        "No. Nothing in 0.3.1 calibrates the score, so 79 does not mean 79%. It is a "
        "bounded ranking statistic: 0.85 x (capped rolling p75 of prior corrected absolute "
        "reactions) + 0.15 x (clipped entropy). Tier extreme-rates reported below are "
        "empirical frequencies of the outcome, not claims the score makes about itself.",
        severity="disclosure"))
    return checks


# ─────────────────────── 2-3. matched-count and operational scoring ──────────────────────
def matched_count_alerts(pred: pd.DataFrame, fraction: float = MATCHED_ALERT_FRACTION
                         ) -> np.ndarray:
    """Top `fraction` within each held-out calendar year, ties broken deterministically.

    A RETROSPECTIVE RANKING COMPARISON, not an executable live rule: the count depends on
    how many events the year turned out to contain, which nobody knows in January. Its
    purpose is to make precision and capture comparable across predictors by holding alert
    volume fixed — the operational numbers, on training-derived thresholds, are reported
    alongside and are the ones a live system could have produced.

    Ties are broken by `event_id`, which is a function of ticker and date alone. Sorting by
    the outcome, or leaving the order to the frame, would let a tie be resolved in a way
    that flatters or punishes a predictor for free.
    """
    flag = np.zeros(len(pred), dtype=bool)
    order = np.lexsort((pred["event_id"].to_numpy(), -pred["p"].to_numpy(float)))
    years = pred["test_year"].to_numpy()
    for year in np.unique(years):
        idx = order[years[order] == year]
        k = max(1, int(math.ceil(fraction * len(idx))))
        flag[idx[:k]] = True
    return flag


def _matched_metrics(y: np.ndarray, flag: np.ndarray) -> dict:
    n = int(flag.sum())
    hits = int(y[flag].sum())
    total = int(y.sum())
    base = float(y.mean()) if len(y) else float("nan")
    return {
        "matched_alerts": n,
        "matched_precision": float(hits / n) if n else float("nan"),
        "matched_capture": float(hits / total) if total else float("nan"),
        "matched_lift": float((hits / n) / base) if n and base > 0 else float("nan"),
    }


def score_predictor(pred: pd.DataFrame, name: str, label: str, scope: str,
                    sample: str) -> dict:
    y = pred["y"].to_numpy(int)
    p = pred["p"].to_numpy(float)
    row = {
        "predictor": name, "label": label, "scope": scope, "sample": sample,
        "n": len(pred), "positives": int(y.sum()), "base_rate": float(y.mean()),
        "roc_auc": _auc(y, p), "average_precision": _ap(y, p),
    }
    row.update(_matched_metrics(y, matched_count_alerts(pred)))
    if pred["alert_threshold"].notna().any():
        op = alert_metrics(y, p, pred["alert_threshold"].to_numpy(float))
        row.update({"operational_" + k: v for k, v in op.items()})
    row.update(within_stock_macro_auc(pred))
    return row


def fixed_predictor_predictions(data: pd.DataFrame, column: str, name: str, label: str,
                                *, first_test_year: int = FIRST_TEST_YEAR,
                                last_test_year: int = LAST_TEST_YEAR) -> pd.DataFrame:
    """A predictor that needs no training, scored on the same held-out years.

    The operational threshold still comes from data the predictor could have seen: the
    90th percentile of its own values over events whose outcomes had completed before the
    fold's cutoff. That keeps the operational column comparable with the trained models'
    instead of silently handing the fixed score an in-sample threshold.
    """
    out = []
    for year in range(first_test_year, last_test_year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        history = data[(data["event_clock"] < cutoff) & data["outcome_ready_known"]
                       & (data["outcome_ready_date"] < cutoff) & data[column].notna()]
        test = data[data["event_clock"].dt.year.eq(year) & data[column].notna()]
        if len(test) == 0 or len(history) < 500:
            continue
        out.append(pd.DataFrame({
            "group": name, "label": label, "test_year": year,
            "event_id": test["event_id"].to_numpy(),
            "stock": test["stock"].to_numpy(),
            "quarter": test["quarter"].to_numpy(),
            "y": test[label].to_numpy(int),
            "p": pd.to_numeric(test[column], errors="coerce").to_numpy(float),
            "alert_threshold": float(np.quantile(
                pd.to_numeric(history[column], errors="coerce").dropna(), ALERT_QUANTILE)),
            "train_n": len(history),
            "train_cutoff": cutoff.date().isoformat(),
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def common_events(frames: dict[str, pd.DataFrame]) -> set:
    """Event ids every predictor produced a held-out prediction for."""
    sets = [set(f["event_id"]) for f in frames.values() if len(f)]
    return set.intersection(*sets) if sets else set()


def tier_profile(data: pd.DataFrame, event_ids: set, label: str) -> list[dict]:
    """Extreme-event rate, count and capture by shipped tier, on the evaluated events.

    Descriptive only. The tier cuts are not reselected, and the tier inherits the same-day
    leakage in the lift's global prior (audit check `lift_global_prior_sees_same_day_rows`),
    so these rates are an optimistic reading of the tier, not of the score.
    """
    d = data[data["event_id"].isin(event_ids) & data[TIER].notna()]
    total = int(d[label].sum())
    rows = []
    for tier in TIER_ORDER:
        sub = d[d[TIER].astype(object).eq(tier)]
        if len(sub) == 0:
            continue
        hits = int(sub[label].sum())
        rows.append({
            "tier": tier, "label": label, "events": len(sub),
            "share_of_events": float(len(sub) / len(d)),
            "extreme_events": hits,
            "extreme_rate": float(hits / len(sub)),
            "lift_vs_base": float((hits / len(sub)) / d[label].mean())
            if d[label].mean() > 0 else float("nan"),
            "capture": float(hits / total) if total else float("nan"),
        })
    return rows


# ──────────────────────────────────── 4. the run ─────────────────────────────────────────
def _restrict(frames: dict[str, pd.DataFrame], keep: set) -> dict[str, pd.DataFrame]:
    return {k: (v[v["event_id"].isin(keep)].sort_values("event_id").reset_index(drop=True)
                if len(v) else v) for k, v in frames.items()}


def run_comparison(data: pd.DataFrame, label: str, sample: str,
                   bootstrap_reps: int) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """S vs M0 vs M1 vs the historical reference, on one label and one sample."""
    frames = {
        "S_score_shipped": fixed_predictor_predictions(data, SCORE, "S_score_shipped", label),
        "S_score_repaired": fixed_predictor_predictions(data, SCORE_REPAIRED,
                                                        "S_score_repaired", label),
        "H_hist_reference": fixed_predictor_predictions(data, HISTORICAL_REFERENCE,
                                                        "H_hist_reference", label),
        "M0_score_only": walk_forward(data, [SCORE], "M0_score_only", label),
        "M1_score_plus_deviations": walk_forward(
            data, [SCORE, *DEVIATION_FEATURES], "M1_score_plus_deviations", label),
    }
    keep = common_events(frames)
    frames = _restrict(frames, keep)

    metrics, deltas = [], []
    for name, f in frames.items():
        if len(f) == 0:
            continue
        metrics.append(score_predictor(f, name, label, "pooled", sample))
        for year, g in f.groupby("test_year"):
            metrics.append(score_predictor(g, name, label, f"year_{int(year)}", sample))

    # Paired differences. M1 must beat BOTH M0 and the unchanged score to count, so both
    # are run; S vs the historical reference answers the "beyond basic stock history"
    # question, and M0 vs S isolates "a trained model on the same information".
    pairs = [("M1_score_plus_deviations", "M0_score_only"),
             ("M1_score_plus_deviations", "S_score_shipped"),
             ("M0_score_only", "S_score_shipped"),
             ("S_score_shipped", "H_hist_reference"),
             ("S_score_repaired", "S_score_shipped")]
    for better, worse in pairs:
        if len(frames.get(better, [])) == 0 or len(frames.get(worse, [])) == 0:
            continue
        d = quarter_block_bootstrap(frames[worse], frames[better], reps=bootstrap_reps)
        d.update({"label": label, "sample": sample, "comparison": f"{better}_vs_{worse}"})
        # Matched-count precision/capture differences on identical rows, same tie rule.
        a, b = frames[worse], frames[better]
        y = a["y"].to_numpy(int)
        ma, mb = matched_count_alerts(a), matched_count_alerts(b)
        d["delta_matched_precision"] = (_matched_metrics(y, mb)["matched_precision"]
                                        - _matched_metrics(y, ma)["matched_precision"])
        d["delta_matched_capture"] = (_matched_metrics(y, mb)["matched_capture"]
                                      - _matched_metrics(y, ma)["matched_capture"])
        yearly_wins = 0
        years = 0
        for year in sorted(a["test_year"].unique()):
            ya = a[a["test_year"].eq(year)]
            yb = b[b["test_year"].eq(year)]
            auc_a, auc_b = _auc(ya["y"], ya["p"]), _auc(yb["y"], yb["p"])
            if not (np.isnan(auc_a) or np.isnan(auc_b)):
                years += 1
                yearly_wins += int(auc_b > auc_a)
        d["years_better"] = yearly_wins
        d["years_evaluated"] = years
        d["majority_of_years"] = bool(years and yearly_wins / years > 0.5)
        deltas.append(d)

    stacked = pd.concat([f.assign(sample=sample) for f in frames.values() if len(f)],
                        ignore_index=True) if frames else pd.DataFrame()
    return metrics, deltas, stacked


def run(*, results_dir: Path = RESULTS_DIR, bootstrap_reps: int = BOOTSTRAP_REPS) -> dict:
    analysis = load_event_frame()
    benzinga = load_benzinga()
    # The repaired score must be built on the SAME base as the shipped one — the full
    # event frame — or its entropy fallback would see a truncated history and the two would
    # differ for a reason that has nothing to do with the defect. Computed once, then
    # carried onto the evaluation rows by key.
    events_full = repaired_risk_score(analysis)
    data, report = build_dataset(analysis, benzinga)
    carry = (events_full.loc[~events_full["is_pending"].astype(bool),
                             ["stock", "earnings_date", SCORE_REPAIRED, "entropy_source",
                              "score_repair_delta"]]
             .assign(stock=lambda d: d["stock"].astype(str).str.strip().str.upper(),
                     earnings_date=lambda d: pd.to_datetime(d["earnings_date"]).dt.normalize()))
    data = data.merge(carry, on=["stock", "earnings_date"], how="left", validate="one_to_one")
    checks = audit_score_validity(events_full, data)

    # ── the 14,902 vs 14,637 reconciliation ─────────────────────────────────────────────
    deviation_ready = data[DEVIATION_COVERAGE_KEYS].notna().all(axis=1)
    held_out = data["event_clock"].dt.year.between(FIRST_TEST_YEAR, LAST_TEST_YEAR)
    reconciliation = {
        "matched_dataset_rows_2014_2025": int(len(data)),
        "of_which_deviation_ready": int(deviation_ready.sum()),
        "held_out_rows_2018_2025": int(held_out.sum()),
        "training_only_rows_2014_2017": int((~held_out).sum()),
        "held_out_and_deviation_ready": int((held_out & deviation_ready).sum()),
        "explanation": (
            "The two earlier figures counted different populations. 14,902 was the number "
            "of rows in the FULL 2014-2025 matched dataset carrying all deviation features, "
            "including the 2014-2017 rows that are training-only and are never scored. "
            "14,637 was the number of HELD-OUT rows (2018-2025) scored in the broad "
            "comparison. Neither is a subset of the other by construction; the intersection "
            "is reported here and is the population the paired subset below actually uses."),
    }

    all_metrics, all_deltas, all_preds, tier_rows = [], [], [], []
    for label in ("y_primary", "y_secondary"):
        for sample, frame in (("broad_matched", data),
                              ("paired_deviation_ready", data[deviation_ready])):
            m, d, p = run_comparison(frame, label, sample, bootstrap_reps)
            all_metrics.extend(m)
            all_deltas.extend(d)
            if len(p):
                all_preds.append(p)
                tier_rows.extend([r | {"sample": sample}
                                  for r in tier_profile(frame, set(p["event_id"]), label)])

    metrics = pd.DataFrame(all_metrics)
    deltas = pd.DataFrame(all_deltas)
    predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    tiers = pd.DataFrame(tier_rows)

    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(results_dir / "validity_audit.csv", index=False)
    metrics.to_csv(results_dir / "metrics.csv", index=False)
    deltas.to_csv(results_dir / "paired_deltas.csv", index=False)
    predictions.to_csv(results_dir / "held_out_predictions.csv", index=False)
    tiers.to_csv(results_dir / "tier_profile.csv", index=False)

    repair = data.loc[data["event_clock"].dt.year.between(FIRST_TEST_YEAR, LAST_TEST_YEAR),
                      "score_repair_delta"]
    config = {
        "experiment": "score_validity_and_deviation_features",
        "status": "RETROSPECTIVE RESEARCH — code/timing validity audited; live performance "
                  "not established",
        "three_distinct_claims": {
            "code_and_timing_validity": "audited here; one material defect found and "
                                        "repaired in research only (see validity_audit.csv)",
            "historical_predictive_performance": "measured here on held-out events with "
                                                 "historically selected constants, so "
                                                 "optimistically biased",
            "live_performance": "NOT established. Benzinga estimates are latest-vintage — a "
                                "limitation of unknown sign, not a proven upper bound.",
        },
        "model_version_reference": MODEL_VERSION,
        "outcome": {
            "primary": f"{PRIMARY_TARGET} >= {PRIMARY_THRESHOLD}",
            "secondary": f"{PRIMARY_TARGET} >= {SECONDARY_THRESHOLD}",
            "definition": f"|reaction| over {REACTION_SESSIONS} post-announcement market "
                          "sessions from the last close strictly before the announcement",
        },
        "predictors": {
            "S_score_shipped": f"{SCORE}, unchanged",
            "S_score_repaired": f"{SCORE_REPAIRED}, causal entropy fallback only",
            "H_hist_reference": f"{HISTORICAL_REFERENCE} (shrinkage "
                                f"{HISTORICAL_REFERENCE_SHRINKAGE}, frozen Phase 4 settings)",
            "M0_score_only": "HistGradientBoostingClassifier on the shipped score alone",
            "M1_score_plus_deviations": "same classifier on the score + "
                                        + ", ".join(DEVIATION_FEATURES),
        },
        "classifier": {"name": "HistGradientBoostingClassifier", **CLASSIFIER_PARAMS},
        "walk_forward": {"first_test_year": FIRST_TEST_YEAR, "last_test_year": LAST_TEST_YEAR,
                         "scheme": "expanding; a training row's reaction window must CLOSE "
                                   "before 1 Jan of the test year"},
        "matched_alert_rule": {
            "fraction": MATCHED_ALERT_FRACTION,
            "scope": "top decile within each held-out calendar year",
            "tie_break": "event_id (ticker|date), outcome-independent and deterministic",
            "caveat": "retrospective ranking comparison, NOT an executable live threshold",
        },
        "operational_alert_rule": {
            "threshold": f"{ALERT_QUANTILE:.0%} quantile of predictions on events completed "
                         "before the fold cutoff",
            "note": "this is the live-executable comparison; alert counts differ by predictor",
        },
        "bootstrap": {"scheme": "paired, calendar-quarter blocks", "reps": bootstrap_reps,
                      "seed": BOOTSTRAP_SEED},
        "research_repair": {
            "defect": "entropy fallback forward-filled in frame order, reading future rows "
                      "of other tickers",
            "repair": "causal_entropy_fallback(): last entropy from a strictly earlier "
                      "announcement date",
            "production_changed": False,
            "held_out_events_affected": int((repair.abs() > 1e-9).sum()),
            "max_abs_score_change": float(repair.abs().max()),
            "mean_abs_score_change": float(repair.abs().mean()),
        },
        "git": _git_state(),
        "inputs": {
            str(p): {"sha256": _sha256(p), "bytes": p.stat().st_size}
            for p in (Path("output/phase3_target_rebuild/phase3_events.parquet"),
                      Path("data/benzinga_features/expectation_features.parquet"))
        },
        "code_provenance": {
            str(p): _sha256(p) for p in
            (Path("testing/score_validity_and_deviation_features.py"),
             Path("testing/benzinga_feature_value.py"),
             Path("research/phase3_target_rebuild.py"),
             Path("research/phase4_baselines.py")) if p.exists()
        },
        "dataset": {**reconciliation, "stocks": int(data["stock"].nunique()),
                    "join_losses_by_year": report.join_losses_by_year},
    }
    (results_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")

    pooled = metrics[metrics["scope"].eq("pooled")]
    summary = {
        **{k: config[k] for k in ("experiment", "status", "three_distinct_claims", "outcome",
                                  "predictors", "matched_alert_rule", "research_repair")},
        "dataset_reconciliation": reconciliation,
        "validity_audit": checks,
        "pooled_metrics": pooled.to_dict(orient="records"),
        "paired_deltas": deltas.to_dict(orient="records"),
        "tier_profile": tiers.to_dict(orient="records"),
        "answers": derive_answers(metrics, deltas),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def derive_answers(metrics: pd.DataFrame, deltas: pd.DataFrame) -> dict:
    """A-D, computed from the tables rather than written by hand."""
    def pooled(name, label, sample="broad_matched"):
        r = metrics[metrics["predictor"].eq(name) & metrics["label"].eq(label)
                    & metrics["scope"].eq("pooled") & metrics["sample"].eq(sample)]
        return r.iloc[0].to_dict() if len(r) else {}

    def delta(comparison, label, sample="broad_matched"):
        r = deltas[deltas["comparison"].eq(comparison) & deltas["label"].eq(label)
                   & deltas["sample"].eq(sample)]
        return r.iloc[0].to_dict() if len(r) else {}

    out = {}
    for label in ("y_primary", "y_secondary"):
        s, h = pooled("S_score_shipped", label), pooled("H_hist_reference", label)
        d_sh = delta("S_score_shipped_vs_H_hist_reference", label)
        m1_m0 = delta("M1_score_plus_deviations_vs_M0_score_only", label)
        m1_s = delta("M1_score_plus_deviations_vs_S_score_shipped", label)
        m1_m0_p = delta("M1_score_plus_deviations_vs_M0_score_only", label,
                        "paired_deviation_ready")
        beats_both = bool(m1_m0.get("delta_roc_auc", 0) > 0 and m1_s.get("delta_roc_auc", 0) > 0)
        ci_both = bool(m1_m0.get("delta_roc_auc_ci_lo", -1) > 0
                       and m1_s.get("delta_roc_auc_ci_lo", -1) > 0)
        stable = bool(m1_m0.get("majority_of_years") and m1_s.get("majority_of_years"))
        out[label] = {
            "A_score_distinguishes_risk": {
                "roc_auc": s.get("roc_auc"), "average_precision": s.get("average_precision"),
                "matched_precision": s.get("matched_precision"),
                "matched_lift": s.get("matched_lift"),
                "verdict": "yes" if (s.get("roc_auc") or 0) > 0.6 else "weak",
            },
            "B_beats_simple_historical_reference": {
                "score_auc": s.get("roc_auc"), "reference_auc": h.get("roc_auc"),
                "delta_roc_auc": d_sh.get("delta_roc_auc"),
                "ci": [d_sh.get("delta_roc_auc_ci_lo"), d_sh.get("delta_roc_auc_ci_hi")],
                "years_better": d_sh.get("years_better"),
                "years_evaluated": d_sh.get("years_evaluated"),
                "verdict": ("yes" if d_sh.get("delta_roc_auc_ci_lo", -1) > 0
                            else "no" if d_sh.get("delta_roc_auc_ci_hi", 1) < 0
                            else "not distinguishable"),
            },
            "C_identifies_risky_quarters_within_stock": {
                "within_stock_macro_auc": s.get("within_stock_macro_auc"),
                "eligible_stocks": s.get("within_stock_eligible_stocks"),
                "eligible_events": s.get("within_stock_eligible_events"),
                "reference_within_stock": h.get("within_stock_macro_auc"),
                "verdict": ("yes" if (s.get("within_stock_macro_auc") or 0) > 0.52
                            else "no — at or below chance"),
            },
            "D_M1_improves_on_both": {
                "vs_M0": {"delta_roc_auc": m1_m0.get("delta_roc_auc"),
                          "ci": [m1_m0.get("delta_roc_auc_ci_lo"),
                                 m1_m0.get("delta_roc_auc_ci_hi")],
                          "delta_matched_precision": m1_m0.get("delta_matched_precision"),
                          "years_better": m1_m0.get("years_better"),
                          "years_evaluated": m1_m0.get("years_evaluated")},
                "vs_S": {"delta_roc_auc": m1_s.get("delta_roc_auc"),
                         "ci": [m1_s.get("delta_roc_auc_ci_lo"),
                                m1_s.get("delta_roc_auc_ci_hi")],
                         "delta_matched_precision": m1_s.get("delta_matched_precision"),
                         "years_better": m1_s.get("years_better"),
                         "years_evaluated": m1_s.get("years_evaluated")},
                "vs_M0_paired_deviation_ready": {
                    "delta_roc_auc": m1_m0_p.get("delta_roc_auc"),
                    "ci": [m1_m0_p.get("delta_roc_auc_ci_lo"),
                           m1_m0_p.get("delta_roc_auc_ci_hi")]},
                "verdict": ("yes, and stable" if beats_both and ci_both and stable
                            else "directionally yes but not stable" if beats_both and stable
                            else "directionally yes only" if beats_both
                            else "no"),
            },
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    a = ap.parse_args(argv)
    s = run(results_dir=a.results_dir, bootstrap_reps=a.bootstrap_reps)
    print(json.dumps({"dataset_reconciliation": s["dataset_reconciliation"],
                      "answers": s["answers"]}, indent=2, default=str))
    print(f"\nwrote {a.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
