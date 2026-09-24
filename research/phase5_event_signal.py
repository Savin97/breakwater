"""Phase 5: does anything known before an announcement predict THIS event?

Research-only. Nothing here changes production scoring, thresholds, ingestion or the
production database, and no threshold is tuned.

The question
------------
Phase 4 established that essentially all of model 0.3.1's corrected-target discrimination
is a *stock phenotype*: "this company usually moves a lot on earnings". The expanding mean
of the stock's own prior corrected absolute reactions (`long_mean_abs_reaction`) reaches
ROC-AUC 0.698 on the mature-28 cohort, and the full 0.3.1 score reaches 0.692 — the model
does not beat its own crudest input.

Phase 5 asks the next question. Once that phenotype is accounted for, does any feature
already in this repository carry information about *which quarter* is dangerous for a
given company? That is a different and much harder question than "which companies are
dangerous", and it is the one a product has to answer, because the phenotype is already
known months in advance and prices itself in.

Structural reference: `long_mean_abs_reaction` (secondary: `expanding_p75_abs_reaction`).

The three ways a feature can look good and not be
--------------------------------------------------
1. **Structural correlation.** A feature that is merely a proxy for the phenotype inherits
   its AUC. Every candidate is therefore also scored *conditional* on the structural
   baseline: pooled-pair AUC restricted to pairs inside the same structural decile, an
   incremental walk-forward model, and the residual formulation of Part 5.
2. **Aggregation artefacts.** Phase 4's within-stock AUC is a plain mean over per-stock
   AUCs, which weights a stock with four events like a stock with twenty-eight and is
   undefined for stocks without both outcome classes. Part 1 re-derives it five other ways
   before any conclusion is drawn from it.
3. **Look-ahead hiding inside an "existing" column.** Several repository features are not
   point-in-time. `momentum_fragility_score` divides by `df["directional_bias"].abs()
   .quantile(0.90)` over the WHOLE frame; `surprise_percentage` is realized at the
   announcement; `daily_ret` on an event row is day D's return, which for a BMO reporter
   *is* the reaction. Part 2 audits every candidate and excludes these from headline
   testing rather than trusting the column name.

Two results about the METHOD, established on synthetic panels in
`testing/test_phase5_event_signal.py` before any real number was read
--------------------------------------------------------------------
**An expanding mean of the outcome is anti-predictive within stock by construction.** On a
panel where each stock's moves are i.i.d. draws from that stock's own fixed distribution —
zero within-stock signal anywhere — the expanding mean of prior moves still scores well
below 0.5 within stock. A large move lifts the running mean and it stays lifted for every
later event of that stock, and those later events are ordinary; so high-running-mean events
are systematically the ordinary ones. Phase 4's sub-0.5 within-stock AUCs for
`long_mean_abs_reaction` and friends are therefore the NULL expectation for that family of
estimator, not evidence of mean reversion in returns. `last_abs_reaction`, which does not
accumulate, sits at 0.499 on the real data — the control that confirms the mechanism.

**Controlling for a noisy structural estimate under-controls.** `long_mean_abs_reaction` is
an estimate of the phenotype, not the phenotype. A candidate that measures the same
phenotype more cleanly will therefore show AUC above 0.5 inside a structural decile and a
positive walk-forward delta, purely through errors-in-variables. Both of the headline
controls can be fooled this way; a stock-CONSTANT feature scores exactly 0.5 within stock
however precise it is. That is why the survivor rule requires the within-stock reading and
not only the incremental one.

What this phase deliberately does not do
-----------------------------------------
No production change, no threshold tuning, no new product score, no CAR, no new dataset,
no gradient boosting. Simple regularized logistic regression, walk-forward, scaled on
training data only.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from config import EXTREME_EARNINGS_REACTION_THRESHOLD
from research.phase3_target_rebuild import CORRECTED_TARGET, PHASE3_PREFIX
from research.phase4_baselines import (
    _resolved_window,
    _within_stock_auc,
    cohort_sample,
    prepare_analysis_frame,
)

PHASE3_EVENTS_PATH = Path("output/phase3_target_rebuild/phase3_events.parquet")
RESULTS_DIR = Path("output/phase5_event_signal")

DEFAULT_START = "2019-01-01"
DEFAULT_END = "2025-12-31"
HEADLINE_COHORT = "mature_28"

STRUCTURAL = "long_mean_abs_reaction"
STRUCTURAL_SECONDARY = "expanding_p75_abs_reaction"

# Walk-forward: train on everything strictly before the test year, test on that year.
# 2019-2020 are consumed as the first training block, so scoring starts in 2021.
FIRST_TEST_YEAR = 2021
LAST_TEST_YEAR = 2025

BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 50517

# Ridge-penalised logistic regression. C is fixed, never tuned on a test year; the point is
# a stable low-variance linear probe, not a fitted product model.
LOGIT_C = 1.0
LOGIT_MAX_ITER = 2000


# ─────────────────────────────────────── feature registry ────────────────────────────────
@dataclass(frozen=True)
class Candidate:
    """One column considered as an event-specific predictor.

    `pre_announcement` / `point_in_time` / `leakage` are judgements made by reading the
    code that produces the column, not inferred from the data. They are recorded here so
    the exclusion of a feature is auditable and arguable rather than silent.
    """
    name: str
    source: str
    description: str
    pre_announcement: bool
    point_in_time: bool
    leakage: str = ""
    exclude_reason: str = ""
    derived: bool = False

    @property
    def headline(self) -> bool:
        return self.pre_announcement and self.point_in_time and not self.exclude_reason


_PRE_STOCK = "feature_engineering/pre_earnings_stock_features.py"
_PRE_SECTOR = "feature_engineering/pre_earnings_sector_features.py"
_EVENTF = "feature_engineering/event_features.py"
_SCORING = "scoring/scoring_features.py"
_PHASE5 = "research/phase5_event_signal.py"
_INGEST = "ingestion (DB column)"

CANDIDATES: tuple[Candidate, ...] = (
    # ── realized volatility, all rolling().shift(1) on the daily grid → as of D-1 close ──
    Candidate("vol_10d", _PRE_STOCK, "10d rolling std of daily_ret, shift(1)", True, True),
    Candidate("vol_30d", _PRE_STOCK, "30d rolling std of daily_ret, shift(1)", True, True),
    Candidate("vol_ratio_10_to_30", _PRE_STOCK, "vol_10d / vol_30d — vol expansion", True, True),
    # ── drift and momentum, same shift(1) construction ──
    Candidate("drift_30d", _PRE_STOCK, "30d mean daily_ret, shift(1)", True, True),
    Candidate("drift_60d", _PRE_STOCK, "60d mean daily_ret, shift(1)", True, True),
    Candidate("abs_drift_30d", _PHASE5, "|drift_30d| — direction-free run-up size", True, True,
              derived=True),
    Candidate("mom_5d", _PRE_STOCK, "5d summed daily_ret, shift(1)", True, True),
    Candidate("mom_20d", _PRE_STOCK, "20d summed daily_ret, shift(1)", True, True),
    Candidate("abs_mom_5d", _PHASE5, "|mom_5d|", True, True, derived=True),
    Candidate("abs_mom_20d", _PHASE5, "|mom_20d|", True, True, derived=True),
    Candidate("pre_earnings_drift_z", _EVENTF,
              "drift_30d vs the stock's own prior pre-earnings drift, shift(1) expanding",
              True, True),
    # ── sector / cross-sectional context ──
    Candidate("sector_vol_10d", _PRE_SECTOR, "sector mean vol_10d, shift(1)", True, True),
    Candidate("sector_vol_30d", _PRE_SECTOR, "sector mean vol_30d, shift(1)", True, True),
    Candidate("sector_drift_60d", _PRE_SECTOR, "sector mean drift_60d, shift(1)", True, True),
    Candidate("stock_vs_sector_vol", _PRE_SECTOR, "stock vol vs its sector's", True, True),
    Candidate("sector_earnings_density", _PRE_SECTOR,
              "share of the sector reporting around this date — known from the calendar",
              True, True),
    Candidate("vol_ratio_cross_sectional_pct", _SCORING,
              "per-DATE cross-sectional rank of vol_ratio_10_to_30", True, True),
    Candidate("sector_vol_ratio_pct", _SCORING,
              "per-(sector,DATE) cross-sectional rank of vol_ratio_10_to_30", True, True),
    Candidate("momentum_pressure_regime", _SCORING,
              "regime score from per-DATE 80th-pct momentum thresholds", True, True),
    Candidate("vol_expansion_score", _SCORING,
              "0.3.1 vol component; built only from per-date ranks and sector ratios",
              True, True),
    # ── prior-event surprise history, shift(1) over the stock's own events ──
    Candidate("surprise_mean_5", _EVENTF, "mean EPS surprise over 5 PRIOR events", True, True),
    Candidate("surprise_std_5", _EVENTF, "std EPS surprise over 5 PRIOR events", True, True),
    Candidate("surprise_streak", _EVENTF, "signed beat/miss streak, PRIOR events", True, True),
    # ── Phase-5 causal within-stock normalisations (no new data, same construction as
    #    pre_earnings_drift_z: expanding mean/std over the stock's own PRIOR event rows) ──
    Candidate("vol_10d_z_own", _PHASE5, "vol_10d vs this stock's own prior event-day vol_10d",
              True, True, derived=True),
    Candidate("vol_30d_z_own", _PHASE5, "vol_30d vs this stock's own prior event-day vol_30d",
              True, True, derived=True),
    Candidate("vol_ratio_10_to_30_z_own", _PHASE5,
              "vol_ratio_10_to_30 vs this stock's own prior event-day values", True, True,
              derived=True),

    # ── excluded: zero historical coverage ──
    Candidate("expected_move_pct", _INGEST, "options-implied expected move",
              True, True, exclude_reason="no coverage before 2026 (forward-only snapshots)"),
    Candidate("atm_iv", _INGEST, "ATM implied volatility",
              True, True, exclude_reason="no coverage before 2026 (forward-only snapshots)"),
    Candidate("eps_dispersion", _INGEST, "analyst EPS dispersion",
              True, True, exclude_reason="no coverage before 2026 (forward-only snapshots)"),
    Candidate("eps_num_analysts", _INGEST, "analyst count",
              True, True, exclude_reason="no coverage before 2026 (forward-only snapshots)"),
    Candidate("eps_revision_momentum", _INGEST, "analyst revision momentum",
              True, True, exclude_reason="no coverage before 2026 (forward-only snapshots)"),
    Candidate("eps_trend_30d", _INGEST, "30d consensus EPS trend",
              True, True, exclude_reason="no coverage before 2026 (forward-only snapshots)"),
    Candidate("eps_revisions_up_30d", _INGEST, "upward revisions, 30d",
              True, True, exclude_reason="no coverage before 2026 (forward-only snapshots)"),
    Candidate("eps_revisions_down_30d", _INGEST, "downward revisions, 30d",
              True, True, exclude_reason="no coverage before 2026 (forward-only snapshots)"),

    # ── excluded: known only AT or AFTER the announcement ──
    Candidate("surprise_percentage", _INGEST, "THIS event's realized EPS surprise",
              False, True, leakage="realized at the announcement",
              exclude_reason="not known pre-announcement"),
    Candidate("reported_eps", _INGEST, "THIS event's reported EPS",
              False, True, leakage="realized at the announcement",
              exclude_reason="not known pre-announcement"),
    Candidate("daily_ret", _PRE_STOCK, "day-D return, unshifted",
              False, True,
              leakage="for a BMO reporter day D IS the first post-announcement session",
              exclude_reason="contains the reaction for 57% of the cohort"),

    # ── excluded: not point-in-time ──
    Candidate("momentum_fragility_score", _SCORING,
              "0.3.1 fragility component", True, False,
              leakage="divides by directional_bias.abs().quantile(0.90) over the WHOLE frame",
              exclude_reason="global quantile is look-ahead"),

    # ── excluded: degenerate on event rows ──
    Candidate("days_to_earnings", _PRE_STOCK, "days until the next report", True, True,
              exclude_reason="identically 0 on every event row"),
    Candidate("proximity_score", _SCORING, "closeness-to-earnings score", True, True,
              exclude_reason="identically 0 on every event row (derives from days_to_earnings)"),

    # ── excluded: structural by construction, or built on the legacy (wrong) target ──
    Candidate("earnings_explosiveness_z", _EVENTF,
              "legacy abs_reaction_median / vol_30d", True, True,
              exclude_reason="built on the LEGACY reaction history Phase 3 proved wrong for BMO"),
    Candidate("earnings_tail_z", _EVENTF, "legacy abs_reaction_p75 / vol_30d", True, True,
              exclude_reason="built on the LEGACY reaction history Phase 3 proved wrong for BMO"),
)

CANDIDATES_BY_NAME = {c.name: c for c in CANDIDATES}
HEADLINE_FEATURES = [c.name for c in CANDIDATES if c.headline]


def derive_phase5_features(d: pd.DataFrame) -> pd.DataFrame:
    """Add the Phase-5 derived columns. Causal by construction; no new data source.

    The `_z_own` transform is the same one `event_pre_earnings_drift_z` already applies to
    `drift_30d`: for each stock, expanding mean and std over that stock's own PRIOR event
    rows, via `shift(1)`. It answers "is this quarter unusual *for this company*", which is
    the only form in which a highly structural quantity like realized volatility can carry
    within-stock information.
    """
    out = d.copy()
    out["abs_drift_30d"] = out["drift_30d"].abs()
    out["abs_mom_5d"] = out["mom_5d"].abs()
    out["abs_mom_20d"] = out["mom_20d"].abs()

    out = out.sort_values(["stock", "event_clock", "earnings_date"], kind="mergesort")
    for col in ("vol_10d", "vol_30d", "vol_ratio_10_to_30"):
        grp = out.groupby("stock", sort=False)[col]
        mean_prior = grp.transform(lambda s: s.shift(1).expanding().mean())
        std_prior = grp.transform(lambda s: s.shift(1).expanding(min_periods=5).std())
        out[f"{col}_z_own"] = (out[col] - mean_prior) / std_prior.replace(0.0, np.nan)
    return out.sort_index()


# ───────────────────────────── Part 1: within-stock discrimination ───────────────────────
def _pair_concordance(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """(concordant + half the ties, total pairs) for one group, via the rank identity.

    AUC over a group equals (#concordant + 0.5 #tied) / (n_pos * n_neg). Returning the
    numerator and denominator instead of the ratio is what lets groups be POOLED by pair
    count rather than averaged as if a 4-event stock and a 28-event stock said equally much.
    """
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0, 0.0
    total = float(len(pos) * len(neg))
    auc = roc_auc_score(y, score)
    return float(auc) * total, total


def pooled_pair_auc(d: pd.DataFrame, score_col: str, by: list[str],
                    min_group: int = 2) -> tuple[float, int, int, float]:
    """AUC computed only from pairs that share every key in `by`, pooled over groups.

    Returns (auc, groups_used, events_used, comparable_pairs).

    This is the aggregation Phase 4's macro mean should be checked against. Pooling by
    pair count makes the estimate a single concordance probability over a well-defined pair
    population — "given one extreme and one non-extreme event of the SAME stock, how often
    does the feature rank the extreme one higher" — instead of an unweighted average of
    noisy per-stock estimates.
    """
    sub = d.dropna(subset=["y_extreme", score_col])
    num = den = 0.0
    groups = events = 0
    for _key, g in sub.groupby(by, sort=False, dropna=False):
        if len(g) < min_group:
            continue
        y = g["y_extreme"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            continue
        n, t = _pair_concordance(y, pd.to_numeric(g[score_col], errors="coerce").to_numpy(float))
        num += n
        den += t
        groups += 1
        events += len(g)
    if den == 0:
        return float("nan"), groups, events, 0.0
    return num / den, groups, events, den


def _grouped_transform_auc(d: pd.DataFrame, score_col: str, by: list[str],
                           how: str) -> float:
    """Pooled AUC after removing each group's location from the predictor.

    `how="demean"` subtracts the group mean; `how="rank"` replaces the value by its
    within-group percentile. Both answer the same question as `pooled_pair_auc` from the
    other direction — they keep every event, including those in groups with only one
    outcome class — so agreement between the three is what makes the within-stock reading
    trustworthy. `rank` additionally removes any monotone within-stock scaling, so a stock
    whose feature simply drifts in level cannot dominate the pooled statistic.
    """
    sub = d.dropna(subset=["y_extreme", score_col]).copy()
    values = pd.to_numeric(sub[score_col], errors="coerce")
    grp = values.groupby([sub[c] for c in by], sort=False)
    if how == "demean":
        adjusted = values - grp.transform("mean")
    elif how == "rank":
        adjusted = grp.rank(pct=True, method="average")
        adjusted = adjusted.where(grp.transform("size") > 1)
    else:
        raise ValueError(how)
    ok = adjusted.notna()
    y = sub.loc[ok, "y_extreme"].astype(int)
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, adjusted[ok]))


def within_stock_validation(sample: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Six views of the same question, plus the diagnostics needed to read them.

    Phase 4 reported one number: an unweighted mean of per-stock AUCs over stocks with at
    least four events and both outcome classes. That number was below 0.5 for every
    structural predictor, which is either a real mean-reversion effect or an artefact of
    aggregation, of the qualification rule, or of a time trend in the outcome. All three
    alternatives are measured here:

    * `phase4_macro_auc` / `phase4_weighted_auc` reproduce Phase 4 exactly.
    * `pooled_pair_auc` re-aggregates by pair count instead of by stock.
    * `pooled_pair_auc_within_stock_year` restricts pairs to the same stock AND the same
      calendar year, which removes any common time trend — the base rate moves from 0.116
      in 2021 to 0.214 in 2024, so a predictor that merely drifts upward over the sample
      would score above 0.5 on stock pairs alone.
    * `demeaned_auc` and `rank_within_stock_auc` keep every event, including those in
      stocks Phase 4's qualification rule discards.
    * `seq_auc` is the calibration reference: the within-stock AUC of the event's own
      sequence number, i.e. of pure elapsed time carrying no information at all.

    The same pooled-pair machinery then decomposes each predictor by which pairs it is
    allowed to compare — all pairs, pairs in the same year, pairs on the same DATE, pairs
    in the same stock — which separates three things a single AUC confounds: a market
    regime moving every stock at once (dies when pairs share a date), a cross-sectional
    phenotype (survives same-date, dies within stock), and genuine event-specific signal
    (survives within stock).
    """
    d = sample.copy()
    d = d.sort_values(["stock", "event_clock"], kind="mergesort")
    d["_seq"] = d.groupby("stock", sort=False).cumcount().astype(float)
    d["_year"] = d["event_clock"].dt.year

    seq_pooled, _, _, _ = pooled_pair_auc(d, "_seq", ["stock"])
    seq_macro = _within_stock_auc(d, "_seq")[1]

    rows = []
    for col in predictors:
        if col not in d.columns:
            continue
        n_stocks, macro, weighted = _within_stock_auc(d, col)
        pooled, groups, events, pairs = pooled_pair_auc(d, col, ["stock"])
        overall, _g, _e, _p = pooled_pair_auc(d.assign(_all=0), col, ["_all"])
        same_year, _g, _e, _p = pooled_pair_auc(d, col, ["_year"])
        same_date, _g, _e, same_date_pairs = pooled_pair_auc(d, col, ["event_clock"])
        pooled_yr, groups_yr, events_yr, pairs_yr = pooled_pair_auc(
            d, col, ["stock", "_year"])
        valid = d.dropna(subset=["y_extreme", col])
        rows.append({
            "predictor": col,
            "events_total": len(valid),
            "stocks_total": int(valid["stock"].nunique()),
            "phase4_qualifying_stocks": n_stocks,
            "phase4_macro_auc": macro,
            "phase4_weighted_auc": weighted,
            "overall_auc": overall,
            "pooled_pair_auc_same_year": same_year,
            "pooled_pair_auc_same_date": same_date,
            "same_date_pairs": same_date_pairs,
            "pooled_pair_auc": pooled,
            "pooled_pair_stocks": groups,
            "pooled_pair_events": events,
            "comparable_pairs": pairs,
            "pooled_pair_auc_within_stock_year": pooled_yr,
            "within_stock_year_groups": groups_yr,
            "within_stock_year_pairs": pairs_yr,
            "demeaned_auc": _grouped_transform_auc(d, col, ["stock"], "demean"),
            "rank_within_stock_auc": _grouped_transform_auc(d, col, ["stock"], "rank"),
            "reference_seq_pooled_pair_auc": seq_pooled,
            "reference_seq_macro_auc": seq_macro,
        })
    out = pd.DataFrame(rows)
    # A predictor is only called directional within-stock when every view agrees on the
    # side of 0.5 it sits on. Disagreement is reported, not resolved by picking a favourite.
    views = ["phase4_macro_auc", "pooled_pair_auc",
             "pooled_pair_auc_within_stock_year", "demeaned_auc", "rank_within_stock_auc"]
    above = (out[views] > 0.5).sum(axis=1)
    below = (out[views] < 0.5).sum(axis=1)
    out["views_above_half"] = above
    out["views_below_half"] = below
    out["direction_agreed"] = (above == len(views)) | (below == len(views))
    return out.sort_values("pooled_pair_auc", ascending=False)


# ────────────────────────────── Part 2: feature audit table ──────────────────────────────
def feature_audit(window: pd.DataFrame, headline_sample: pd.DataFrame) -> pd.DataFrame:
    """The registry, joined to what the data actually shows.

    Coverage is measured on the resolved evaluation window and on the headline cohort
    separately, because a feature can be complete on mature stocks and absent everywhere
    else — that is a coverage effect and must not be read as a quality effect.
    """
    rows = []
    for c in CANDIDATES:
        present = c.name in window.columns
        col_w = window[c.name] if present else pd.Series(dtype=float)
        col_m = headline_sample[c.name] if (present and c.name in headline_sample) \
            else pd.Series(dtype=float)
        by_year = (
            window.assign(_y=window["event_clock"].dt.year)
            .groupby("_y")[c.name].apply(lambda s: s.notna().mean())
            if present else pd.Series(dtype=float)
        )
        first_year = next((int(y) for y, v in by_year.items() if v >= 0.5), None)
        nunique = int(col_m.nunique(dropna=True)) if len(col_m) else 0
        rows.append({
            "feature": c.name,
            "source": c.source,
            "description": c.description,
            "derived_in_phase5": c.derived,
            "column_present": present,
            "coverage_resolved_window": float(col_w.notna().mean()) if len(col_w) else 0.0,
            "coverage_headline_cohort": float(col_m.notna().mean()) if len(col_m) else 0.0,
            "missing_share_headline": 1.0 - (float(col_m.notna().mean()) if len(col_m) else 0.0),
            "distinct_values_headline": nunique,
            "first_usable_year": first_year,
            "known_pre_announcement": c.pre_announcement,
            "point_in_time_safe": c.point_in_time,
            "suspected_leakage": c.leakage or "",
            "headline_eligible": bool(c.headline and present and nunique > 1),
            "exclusion_reason": c.exclude_reason or (
                "" if (c.headline and present and nunique > 1)
                else ("column absent from the event frame" if not present
                      else "constant on the headline cohort" if nunique <= 1 else "")),
        })
    return pd.DataFrame(rows).sort_values(
        ["headline_eligible", "coverage_headline_cohort", "feature"],
        ascending=[False, False, True])


# ─────────────────────── Part 3: univariate and incremental signal ───────────────────────
def _auc(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _ap(y, s):
    y = np.asarray(y, dtype=int)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, np.asarray(s, dtype=float)))


def _top_fraction(y: np.ndarray, s: np.ndarray, fraction: float) -> tuple[float, float, float]:
    """(hit rate, lift, capture) in the top `fraction` by score."""
    n = len(y)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    k = max(1, int(math.ceil(fraction * n)))
    order = np.argsort(-s, kind="mergesort")
    top = y[order[:k]]
    base = float(y.mean())
    hit = float(top.mean())
    total = float(y.sum())
    return hit, (hit / base if base > 0 else float("nan")), \
        (float(top.sum() / total) if total > 0 else float("nan"))


def structural_decile_auc(d: pd.DataFrame, col: str, structural: str = STRUCTURAL,
                          bins: int = 10) -> tuple[float, float]:
    """Pooled-pair AUC restricted to pairs inside the same structural decile.

    The model-free way to ask "does this feature add anything the phenotype does not
    already say". Deciles are cut on the evaluation sample itself, which is a mild
    in-sample convenience for a diagnostic; the walk-forward models in Part 4 are the
    out-of-sample answer.
    """
    sub = d.dropna(subset=["y_extreme", col, structural]).copy()
    if len(sub) < bins * 2:
        return float("nan"), 0.0
    sub["_dec"] = pd.qcut(sub[structural].rank(method="first"), bins, labels=False)
    auc, _groups, _events, pairs = pooled_pair_auc(sub, col, ["_dec"])
    return auc, pairs


def univariate_signal(sample: pd.DataFrame, predictors: list[str],
                      structural: str = STRUCTURAL) -> pd.DataFrame:
    """Raw signal, signal net of the phenotype, within-stock signal, yearly stability."""
    rows = []
    for col in predictors:
        if col not in sample.columns:
            continue
        d = sample.dropna(subset=["y_extreme", col])
        if len(d) == 0:
            continue
        y = d["y_extreme"].to_numpy(dtype=int)
        s = pd.to_numeric(d[col], errors="coerce").to_numpy(float)

        # Direction is fixed once, on the whole cohort, and reused everywhere: a feature
        # whose sign flips year to year must not be rescued by re-orienting it each year.
        raw_auc = _auc(y, s)
        oriented = s if (np.isnan(raw_auc) or raw_auc >= 0.5) else -s
        sign = 1 if (np.isnan(raw_auc) or raw_auc >= 0.5) else -1
        d = d.assign(_oriented=oriented)

        both = d.dropna(subset=[structural])
        spearman = (float(stats.spearmanr(both[col], both[structural]).statistic)
                    if len(both) > 2 else float("nan"))
        dec_auc, dec_pairs = structural_decile_auc(d.assign(**{col: oriented}), col, structural)
        ws_auc, ws_groups, _ws_events, ws_pairs = pooled_pair_auc(d, "_oriented", ["stock"])
        ws_year, _g, _e, _p = pooled_pair_auc(
            d.assign(_year=d["event_clock"].dt.year), "_oriented", ["stock", "_year"])

        yearly = []
        for year, g in d.groupby(d["event_clock"].dt.year):
            yearly.append((int(year), _auc(g["y_extreme"].to_numpy(int),
                                           g["_oriented"].to_numpy(float)), len(g)))
        yr_aucs = np.array([a for _y, a, _n in yearly if not np.isnan(a)], dtype=float)
        hit10, lift10, _cap10 = _top_fraction(y, oriented, 0.10)

        rows.append({
            "feature": col,
            "derived_in_phase5": CANDIDATES_BY_NAME[col].derived if col in CANDIDATES_BY_NAME else False,
            "n": len(d),
            "coverage": float(len(d) / len(sample)),
            "base_rate": float(y.mean()),
            "orientation": sign,
            "roc_auc_raw": raw_auc,
            "roc_auc_oriented": _auc(y, oriented),
            "pr_auc_oriented": _ap(y, oriented),
            "top10_hit_rate": hit10,
            "top10_lift": lift10,
            "spearman_vs_structural": spearman,
            "abs_spearman_vs_structural": abs(spearman) if spearman == spearman else np.nan,
            "auc_within_structural_decile": dec_auc,
            "structural_decile_pairs": dec_pairs,
            "within_stock_pooled_auc": ws_auc,
            "within_stock_stocks": ws_groups,
            "within_stock_pairs": ws_pairs,
            "within_stock_year_pooled_auc": ws_year,
            "years_evaluated": int(len(yr_aucs)),
            "years_auc_above_half": int((yr_aucs > 0.5).sum()),
            "mean_year_auc": float(yr_aucs.mean()) if len(yr_aucs) else np.nan,
            "min_year_auc": float(yr_aucs.min()) if len(yr_aucs) else np.nan,
            "max_year_auc": float(yr_aucs.max()) if len(yr_aucs) else np.nan,
            "yearly_auc": ";".join(f"{y}:{a:.3f}" for y, a, _n in yearly),
        })
    return pd.DataFrame(rows).sort_values("auc_within_structural_decile",
                                          ascending=False, na_position="last")


# ────────────────────────── Part 4: walk-forward incremental models ──────────────────────
@dataclass
class FitArtifacts:
    """Everything learned from a training block. Nothing here may touch a test row."""
    medians: pd.Series
    means: pd.Series
    stds: pd.Series
    model: LogisticRegression


def fit_block(train: pd.DataFrame, features: list[str], *, C: float = LOGIT_C,
              seed: int = 0) -> FitArtifacts:
    """Impute, scale and fit — every statistic estimated on the training rows ONLY.

    Median imputation and standardisation are part of the model, not of the data, so
    fitting them on the pooled sample would leak the test year's distribution into the
    training transform. Kept explicit rather than hidden in a Pipeline so the test suite
    can assert on the stored constants.
    """
    X = train[features].apply(pd.to_numeric, errors="coerce")
    medians = X.median()
    X = X.fillna(medians)
    means = X.mean()
    stds = X.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    Z = (X - means) / stds
    model = LogisticRegression(C=C, max_iter=LOGIT_MAX_ITER, random_state=seed)
    model.fit(Z.to_numpy(float), train["y_extreme"].to_numpy(int))
    return FitArtifacts(medians=medians, means=means, stds=stds, model=model)


def apply_block(fit: FitArtifacts, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = test[features].apply(pd.to_numeric, errors="coerce").fillna(fit.medians)
    Z = (X - fit.means) / fit.stds
    return fit.model.predict_proba(Z.to_numpy(float))[:, 1]


def walk_forward_predictions(sample: pd.DataFrame, features: list[str], *,
                             first_year: int = FIRST_TEST_YEAR,
                             last_year: int = LAST_TEST_YEAR) -> pd.DataFrame:
    """Out-of-fold probabilities from an expanding-window walk-forward.

    For test year Y the model sees only events whose announcement clock is strictly before
    1 January Y. No row is ever in both blocks, nothing is tuned on Y, and the imputation
    and scaling constants come from the training block alone.
    """
    out = []
    for year in range(first_year, last_year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train = sample[sample["event_clock"] < cutoff]
        test = sample[(sample["event_clock"] >= cutoff)
                      & (sample["event_clock"] < cutoff + pd.offsets.DateOffset(years=1))]
        if len(test) == 0 or train["y_extreme"].nunique() < 2 or len(train) < 200:
            continue
        fit = fit_block(train, features)
        out.append(pd.DataFrame({
            "stock": test["stock"].to_numpy(),
            "event_clock": test["event_clock"].to_numpy(),
            "test_year": year,
            "y_extreme": test["y_extreme"].to_numpy(),
            "p": apply_block(fit, test, features),
            "train_n": len(train),
        }, index=test.index))
    return pd.concat(out) if out else pd.DataFrame(
        columns=["stock", "event_clock", "test_year", "y_extreme", "p", "train_n"])


def score_predictions(pred: pd.DataFrame, label: str, features: list[str],
                      scope: str) -> dict:
    if len(pred) == 0:
        return {"model": label, "scope": scope, "n": 0}
    y = pred["y_extreme"].to_numpy(int)
    p = np.clip(pred["p"].to_numpy(float), 1e-6, 1 - 1e-6)
    hit10, lift10, cap10 = _top_fraction(y, p, 0.10)
    _h20, _l20, cap20 = _top_fraction(y, p, 0.20)
    ws, ws_groups, _e, ws_pairs = pooled_pair_auc(
        pred.assign(y_extreme=pred["y_extreme"]), "p", ["stock"])
    return {
        "model": label,
        "scope": scope,
        "n_features": len(features),
        "features": "+".join(features),
        "n": len(pred),
        "base_rate": float(y.mean()),
        "roc_auc": _auc(y, p),
        "pr_auc": _ap(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "top10_hit_rate": hit10,
        "top10_lift": lift10,
        "top10_capture": cap10,
        "top20_capture": cap20,
        "within_stock_pooled_auc": ws,
        "within_stock_stocks": ws_groups,
        "within_stock_pairs": ws_pairs,
    }


def paired_bootstrap_delta(pred_a: pd.DataFrame, pred_b: pd.DataFrame, *,
                           reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED,
                           cluster: str | None = "stock") -> dict:
    """CI for AUC(a) - AUC(b) on identical rows, resampled by `cluster`.

    Events of one company are not independent draws — the phenotype is the whole point of
    this phase — so the bootstrap resamples STOCKS, not rows. Resampling rows would give a
    confidently narrow interval around a difference that is really one company repeated.
    """
    joined = pred_a[["y_extreme", "p"]].join(
        pred_b[["p"]], how="inner", lsuffix="_a", rsuffix="_b")
    if cluster:
        joined = joined.join(pred_a[[cluster]], how="left")
    y = joined["y_extreme"].to_numpy(int)
    a = joined["p_a"].to_numpy(float)
    b = joined["p_b"].to_numpy(float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"n": len(y), "delta_auc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "reps_valid": 0, "share_positive": np.nan}
    delta = _auc(y, a) - _auc(y, b)
    rng = np.random.default_rng(seed)
    deltas = []
    if cluster:
        groups = joined.groupby(cluster, sort=False).indices
        keys = np.array(list(groups.keys()), dtype=object)
        for _ in range(reps):
            picked = rng.integers(0, len(keys), size=len(keys))
            idx = np.concatenate([groups[keys[i]] for i in picked])
            yb = y[idx]
            if len(np.unique(yb)) < 2:
                continue
            deltas.append(_auc(yb, a[idx]) - _auc(yb, b[idx]))
    else:
        n = len(y)
        for _ in range(reps):
            idx = rng.integers(0, n, size=n)
            yb = y[idx]
            if len(np.unique(yb)) < 2:
                continue
            deltas.append(_auc(yb, a[idx]) - _auc(yb, b[idx]))
    arr = np.asarray(deltas, dtype=float)
    return {
        "n": len(y),
        "auc_a": _auc(y, a),
        "auc_b": _auc(y, b),
        "delta_auc": delta,
        "ci_lo": float(np.quantile(arr, 0.025)) if len(arr) else np.nan,
        "ci_hi": float(np.quantile(arr, 0.975)) if len(arr) else np.nan,
        "share_positive": float((arr > 0).mean()) if len(arr) else np.nan,
        "reps_valid": int(len(arr)),
        "cluster": cluster or "row",
    }


# ──────────────────────────── Part 5: residual magnitude signal ──────────────────────────
def residual_frame(sample: pd.DataFrame, structural: str = STRUCTURAL) -> pd.DataFrame:
    """Corrected absolute move relative to what this stock's own history implies.

    A binary >=8% target answers "is this a tail event". It is a coarse instrument: a
    feature can say "bigger than normal FOR THIS COMPANY" — a sleepy stock going from 2% to
    5% — and move the binary AUC almost not at all, because neither event is extreme.

        residual_abs_move = corrected |reaction|  -  expanding mean of the stock's own
                            PRIOR corrected |reactions|

    `log_ratio` is the same comparison on a multiplicative scale, which is the right one
    for a quantity bounded below by zero and spanning an order of magnitude across stocks;
    it is also what makes residuals comparable between a 2%-normal stock and a 9%-normal
    one. Both come from `long_mean_abs_reaction`, which is already causal.
    """
    d = sample.copy()
    expected = pd.to_numeric(d[structural], errors="coerce")
    actual = pd.to_numeric(d[CORRECTED_TARGET], errors="coerce")
    d["expected_abs_move"] = expected
    d["residual_abs_move"] = actual - expected
    d["residual_ratio"] = actual / expected.replace(0.0, np.nan)
    d["residual_log_ratio"] = np.log(
        actual.replace(0.0, np.nan) / expected.replace(0.0, np.nan))
    d["residual_positive"] = (d["residual_abs_move"] > 0).astype(float)
    return d


def residual_signal(resid: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Does a pre-event feature predict a larger-than-normal move for that stock?

    Three readings per feature: the plain Spearman correlation with the residual, the
    within-stock Spearman (both sides demeaned by stock, so a company that is merely always
    volatile and always high on the feature contributes nothing), and the AUC for the sign
    of the residual. The within-stock column is the one that answers Phase 5's question.
    """
    rows = []
    for col in predictors:
        if col not in resid.columns:
            continue
        d = resid.dropna(subset=[col, "residual_abs_move", "residual_log_ratio"])
        if len(d) < 50:
            continue
        x = pd.to_numeric(d[col], errors="coerce")
        r_abs = stats.spearmanr(x, d["residual_abs_move"]).statistic
        r_log = stats.spearmanr(x, d["residual_log_ratio"]).statistic

        # Within-stock: demean both sides by stock before correlating.
        gx = x.groupby(d["stock"], sort=False)
        gy = d["residual_log_ratio"].groupby(d["stock"], sort=False)
        xr = gx.rank(pct=True) - 0.5
        yr = gy.rank(pct=True) - 0.5
        # A feature that is constant inside every stock has UNDEFINED within-stock
        # correlation. Reported as missing, never as 0.0, which would read as "measured and
        # found to be nothing" rather than "not measurable".
        multi = gx.transform("size") > 1
        varies = xr[multi].nunique() > 1 and yr[multi].nunique() > 1
        r_within = (stats.spearmanr(xr[multi], yr[multi]).statistic
                    if (multi.sum() > 2 and varies) else np.nan)

        sign_auc = _auc(d["residual_positive"].to_numpy(int), x.to_numpy(float))
        yearly = {int(y): stats.spearmanr(g[col], g["residual_log_ratio"]).statistic
                  for y, g in d.groupby(d["event_clock"].dt.year) if len(g) > 20}
        vals = np.array([v for v in yearly.values() if v == v], dtype=float)
        rows.append({
            "feature": col,
            "n": len(d),
            "spearman_residual_abs": float(r_abs),
            "spearman_residual_log_ratio": float(r_log),
            "spearman_within_stock_log_ratio": float(r_within) if r_within == r_within else np.nan,
            "residual_sign_auc": sign_auc,
            "years_evaluated": len(vals),
            "years_same_sign_as_pooled": int((np.sign(vals) == np.sign(r_log)).sum()) if len(vals) else 0,
            "mean_year_spearman": float(vals.mean()) if len(vals) else np.nan,
            "yearly_spearman": ";".join(f"{y}:{v:.3f}" for y, v in sorted(yearly.items())),
        })
    return pd.DataFrame(rows).sort_values(
        "spearman_within_stock_log_ratio", ascending=False, na_position="last")


# ───────────────────────────────────────── driver ────────────────────────────────────────
# A candidate survives the univariate screen if it discriminates inside a structural decile
# AND inside a stock, in the same direction, in a majority of years. The bar is deliberately
# about CONSISTENCY rather than size: at these sample sizes a single good year can move a
# pooled AUC by more than any real effect.
SURVIVE_DECILE_AUC = 0.52
SURVIVE_WITHIN_STOCK_AUC = 0.52
SURVIVE_MIN_YEAR_SHARE = 0.60


def select_survivors(uni: pd.DataFrame) -> list[str]:
    ok = (
        uni["auc_within_structural_decile"].ge(SURVIVE_DECILE_AUC)
        & uni["within_stock_pooled_auc"].ge(SURVIVE_WITHIN_STOCK_AUC)
        & (uni["years_auc_above_half"] / uni["years_evaluated"].replace(0, np.nan))
        .ge(SURVIVE_MIN_YEAR_SHARE)
    )
    return uni.loc[ok.fillna(False), "feature"].tolist()


def run(*, events_path: Path = PHASE3_EVENTS_PATH, results_dir: Path = RESULTS_DIR,
        start: str = DEFAULT_START, end: str = DEFAULT_END,
        bootstrap_reps: int = BOOTSTRAP_REPS) -> dict:
    events = pd.read_parquet(events_path)
    analysis = derive_phase5_features(prepare_analysis_frame(events))
    window = _resolved_window(analysis, start, end)
    headline, _core = cohort_sample(window, HEADLINE_COHORT)

    audit = feature_audit(window, headline)
    eligible = audit.loc[audit["headline_eligible"], "feature"].tolist()

    # Common sample: every model and every comparison uses exactly these rows, so a
    # difference between two models can never be a difference between two populations.
    common = headline.dropna(subset=["y_extreme", STRUCTURAL, *eligible]).copy()

    ws_predictors = [STRUCTURAL, STRUCTURAL_SECONDARY, "model_0_3_1_score",
                     "stock_prior_extreme_rate", "recent4_mean_abs_reaction",
                     "recent8_mean_abs_reaction", "last_abs_reaction",
                     "vol_30d", "vol_ratio_10_to_30", "vol_30d_z_own"]
    ws_predictors = [c for c in ws_predictors if c in headline.columns]
    within = within_stock_validation(headline, ws_predictors)

    uni = univariate_signal(common, eligible)
    survivors = select_survivors(uni)

    # ── walk-forward ──
    base_pred = walk_forward_predictions(common, [STRUCTURAL])
    rows = [score_predictions(base_pred, "structural", [STRUCTURAL], "pooled_oof")]
    yearly_rows = [
        score_predictions(g, "structural", [STRUCTURAL], f"year_{int(y)}")
        for y, g in base_pred.groupby("test_year")
    ]
    for r in yearly_rows:
        r["test_year"] = int(r["scope"].split("_")[1])

    deltas = []
    for feat in eligible:
        pred = walk_forward_predictions(common, [STRUCTURAL, feat])
        rows.append(score_predictions(pred, f"structural+{feat}", [STRUCTURAL, feat],
                                      "pooled_oof"))
        for y, g in pred.groupby("test_year"):
            r = score_predictions(g, f"structural+{feat}", [STRUCTURAL, feat], f"year_{int(y)}")
            r["test_year"] = int(y)
            yearly_rows.append(r)
        d = paired_bootstrap_delta(pred, base_pred, reps=bootstrap_reps)
        d["model"] = f"structural+{feat}"
        deltas.append(d)

    combo_rows = []
    if survivors:
        combo_features = [STRUCTURAL, *survivors]
        combo_pred = walk_forward_predictions(common, combo_features)
        rows.append(score_predictions(combo_pred, "structural+survivors", combo_features,
                                      "pooled_oof"))
        for y, g in combo_pred.groupby("test_year"):
            r = score_predictions(g, "structural+survivors", combo_features, f"year_{int(y)}")
            r["test_year"] = int(y)
            yearly_rows.append(r)
        d = paired_bootstrap_delta(combo_pred, base_pred, reps=bootstrap_reps)
        d["model"] = "structural+survivors"
        deltas.append(d)
        combo_rows.append(d)

    # A model's own score is also worth a walk-forward line, as a reference point.
    if "model_0_3_1_score" in common.columns:
        m_pred = walk_forward_predictions(common, ["model_0_3_1_score"])
        rows.append(score_predictions(m_pred, "model_0_3_1_score_only",
                                      ["model_0_3_1_score"], "pooled_oof"))
        d = paired_bootstrap_delta(m_pred, base_pred, reps=bootstrap_reps)
        d["model"] = "model_0_3_1_score_only"
        deltas.append(d)

    models = pd.DataFrame(rows)
    yearly = pd.DataFrame(yearly_rows)
    delta_df = pd.DataFrame(deltas).sort_values("delta_auc", ascending=False)

    resid = residual_frame(common)
    residuals = residual_signal(resid, eligible)

    # ── write ──
    results_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(results_dir / "feature_audit.csv", index=False)
    within.to_csv(results_dir / "within_stock_validation.csv", index=False)
    uni.to_csv(results_dir / "univariate_signal.csv", index=False)
    models.merge(delta_df[["model", "delta_auc", "ci_lo", "ci_hi", "share_positive", "cluster"]],
                 on="model", how="left").to_csv(
        results_dir / "walkforward_models.csv", index=False)
    yearly.to_csv(results_dir / "yearly_results.csv", index=False)
    residuals.to_csv(results_dir / "residual_signal.csv", index=False)

    base_row = models[models["model"].eq("structural")
                      & models["scope"].eq("pooled_oof")].iloc[0].to_dict()
    pooled = models[models["scope"].eq("pooled_oof")].sort_values("roc_auc", ascending=False)
    summary = {
        "phase": "phase5_event_specific_signal",
        "source": str(events_path),
        "evaluation_window": {"start": start, "end": end},
        "headline_cohort": HEADLINE_COHORT,
        "structural_reference": STRUCTURAL,
        "structural_secondary": STRUCTURAL_SECONDARY,
        "cohort": {
            "resolved_window_events": int(len(window)),
            "headline_events": int(len(headline)),
            "common_sample_events": int(len(common)),
            "stocks": int(common["stock"].nunique()),
            "base_rate": float(common["y_extreme"].mean()),
            "years": sorted(int(y) for y in common["event_clock"].dt.year.unique()),
        },
        "walk_forward": {
            "first_test_year": FIRST_TEST_YEAR,
            "last_test_year": LAST_TEST_YEAR,
            "oof_events": int(len(base_pred)),
            "structural_baseline": base_row,
        },
        "feature_audit": {
            "candidates_registered": len(CANDIDATES),
            "headline_eligible": len(eligible),
            "excluded": {c.name: (c.exclude_reason or c.leakage)
                         for c in CANDIDATES if not c.headline},
        },
        "within_stock_validation": within.to_dict(orient="records"),
        "univariate_survivors": survivors,
        "survivor_rule": {
            "auc_within_structural_decile_min": SURVIVE_DECILE_AUC,
            "within_stock_pooled_auc_min": SURVIVE_WITHIN_STOCK_AUC,
            "min_share_of_years_above_half": SURVIVE_MIN_YEAR_SHARE,
        },
        "univariate_top": uni.head(12).to_dict(orient="records"),
        "walk_forward_pooled": pooled.to_dict(orient="records"),
        "walk_forward_deltas_vs_structural": delta_df.to_dict(orient="records"),
        "residual_top": residuals.head(12).to_dict(orient="records"),
        "guardrails": [
            "corrected Phase 3 target only; unresolved outcomes stay missing and are never "
            "counted as non-extreme",
            "causal baselines come from Phase 4, where same-date outcomes enter history only "
            "after every event on that date has been scored",
            "every model comparison runs on one common sample of rows",
            "walk-forward trains strictly before the test year; imputation and scaling "
            "constants are fitted on the training block only",
            "bootstrap resamples stocks, not rows, because events of one company are not "
            "independent draws",
            "no production code, threshold, weight or product score is changed or tuned",
        ],
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", type=Path, default=PHASE3_EVENTS_PATH)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    args = ap.parse_args(argv)
    s = run(events_path=args.events, results_dir=args.results_dir, start=args.start,
            end=args.end, bootstrap_reps=args.bootstrap_reps)
    print(json.dumps({k: v for k, v in s.items()
                      if k in ("cohort", "univariate_survivors", "walk_forward")},
                     indent=2, default=str))
    print(f"\nwrote {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
