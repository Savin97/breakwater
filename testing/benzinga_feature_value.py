"""Do Benzinga expectation features improve earnings tail-risk prediction?

Research evaluation only. Nothing here changes production scoring, thresholds or
ingestion, nothing calls an API, and `main.py` is never run. The DuckDB file is opened
read-only if it is opened at all.

The question
------------
Phase 4 showed that essentially all of model 0.3.1's corrected-target discrimination is a
stock phenotype — "this company usually moves a lot on earnings" — and Phase 5 found only
one genuinely event-specific signal already in the repository, pre-event realized
volatility, worth about +0.011 AUC. Phase 5 also found that the features most likely to
carry event-level information (implied vol, expected move, analyst dispersion) have ZERO
historical coverage.

`data/benzinga_features/expectation_features.parquet` is the first candidate that does have
history: what the market expected this quarter's EPS and revenue to do, and how unusual
that expectation is for this company. This module asks whether it earns its place.

**This is retrospective research, not a point-in-time test.**
The vendor's estimates are LATEST VINTAGE. `last_updated` on a 2014 event is routinely a
2021-2023 timestamp, so a consensus figure may have been revised after the announcement it
is supposed to precede, and the "previous" actuals used to form growth rates are restated
values. The feature builder's own report says so in as many words: "no historical estimate
vintage has been verified" and "latest-vintage restated actuals may still introduce
historical look-ahead; chronological ordering alone does not establish PIT validity."
Every number this module produces is therefore an UPPER BOUND on what a live system could
have achieved. A negative result is still informative — a feature that cannot help even
with hindsight will not help without it — but a positive result is a reason to buy
point-in-time vintages and re-run, never a reason to ship.

Design commitments
------------------
* One row per earnings event, joined 1:1 and asserted, with join losses reported by year.
* Outcomes are excluded BEFORE labels are cut, so an unavailable reaction can never become
  a "did not move 8%" training example.
* Walk-forward only: train 2014-2017, test 2018; then expand the training window one year
  at a time through 2025. A training event must have had its whole reaction window CLOSE
  before the cutoff, not merely have been announced before it.
* One classifier, one set of hyper-parameters, one seed, for all four feature groups. No
  parameter search, no random split. Groups differ only by which columns they see.
* Identical training and test event ids inside every comparison, so a metric difference is
  never a difference between two populations.
* Missingness is preserved. The classifier splits on NaN natively; nothing is imputed, and
  no missing binary flag is silently turned into `False`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from config import (
    EXTREME_EARNINGS_REACTION_THRESHOLD,
    LARGE_EARNINGS_REACTION_THRESHOLD,
    MODEL_VERSION,
)
from feature_engineering.announcement_timing import TARGET_AVAILABLE
from research.phase3_target_rebuild import CORRECTED_TARGET, PHASE3_PREFIX
from research.phase4_baselines import prepare_analysis_frame
from research.phase5_event_signal import HEADLINE_FEATURES, derive_phase5_features

PHASE3_EVENTS_PATH = Path("output/phase3_target_rebuild/phase3_events.parquet")
BENZINGA_FEATURES_PATH = Path("data/benzinga_features/expectation_features.parquet")
RESULTS_DIR = Path("testing/testing_results/benzinga_feature_value")

# ── the experiment's fixed frame of reference ────────────────────────────────────────────
# Primary outcome: the production extreme-move threshold on the Phase 3 CORRECTED reaction,
# which spans 3 post-announcement market sessions measured from the last close strictly
# before the announcement. Secondary: the production large-move threshold on the same
# horizon. Both are read from config; neither is tuned here.
PRIMARY_TARGET = CORRECTED_TARGET                    # abs_reaction_3d_anchored
PRIMARY_STATUS = "reaction_3d_anchored_status"
PRIMARY_THRESHOLD = EXTREME_EARNINGS_REACTION_THRESHOLD   # 0.08
SECONDARY_THRESHOLD = LARGE_EARNINGS_REACTION_THRESHOLD   # 0.05
REACTION_SESSIONS = 3

FIRST_TRAIN_YEAR = 2014
FIRST_TEST_YEAR = 2018
LAST_TEST_YEAR = 2025

ALERT_QUANTILE = 0.90          # alert threshold, taken from TRAINING predictions only
BOOTSTRAP_REPS = 500
BOOTSTRAP_SEED = 20260913
SEED = 20260913

# One modest nonlinear classifier, fixed. `HistGradientBoostingClassifier` is chosen
# because it splits on NaN natively, which is what lets missingness be preserved rather
# than imputed — a hard requirement here, since Benzinga coverage is itself informative
# (a company with no usable prior EPS is a different kind of company).
CLASSIFIER_PARAMS = dict(
    max_iter=300,
    learning_rate=0.05,
    max_depth=4,
    max_leaf_nodes=15,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=SEED,
)

# ── feature groups ───────────────────────────────────────────────────────────────────────
# A: the existing pre-event baseline. Phase 4's causal structural baselines (built only
# from the stock's own PRIOR resolved corrected reactions, with same-date outcomes entering
# history only after every event on that date is scored) plus Phase 5's audited pre-event
# feature list, which already excludes the leaky and non-point-in-time columns:
# `surprise_percentage` (realized at the announcement), `daily_ret` (day D's return, which
# IS the reaction for a BMO reporter), `momentum_fragility_score` (a global quantile over
# the whole frame), and the two `earnings_*_z` columns built on the legacy target.
STRUCTURAL_BASELINE = [
    "long_mean_abs_reaction",
    "long_median_abs_reaction",
    "expanding_p75_abs_reaction",
    "stock_prior_extreme_rate",
    "stock_prior_extreme_rate_shrunk20",
    "recent4_mean_abs_reaction",
    "recent8_mean_abs_reaction",
    "recent8_extreme_rate",
    "last_abs_reaction",
    "sector_prior_extreme_rate",
    "market_prior_extreme_rate",
    "n_prior_resolved_baseline",
]
# `rolling28_p75_abs_reaction` is deliberately absent: it needs 28 prior resolved events and
# has 0.0% coverage in the 2014-2017 training block, so including it would put a column in
# group A that exists only in the later folds. Decided once, here, from coverage in the
# FIRST training block — never re-decided per fold.
GROUP_A = STRUCTURAL_BASELINE + list(HEADLINE_FEATURES)

BZ_LEVEL = [
    "expected_eps_growth",
    "expected_revenue_growth",
    "expected_eps_change_symmetric",
    "expected_loss_to_profit",
    "expected_profit_to_loss",
    "expected_loss_narrows",
    "expected_loss_widens",
    "eps_near_zero_prior",
    "abs_expected_eps_growth",
    "abs_expected_revenue_growth",
]
BZ_GAP = [
    "eps_revenue_growth_gap",
    "abs_eps_revenue_growth_gap",
    "eps_revenue_opposite_direction",
]
BZ_DEVIATION = [
    "eps_growth_vs_normal",
    "revenue_growth_vs_normal",
    "abs_eps_growth_vs_normal",
    "abs_revenue_growth_vs_normal",
    "eps_growth_regime_z",
    "revenue_growth_regime_z",
]
# Binary/flag columns whose NaN must never be read as "no". Each gets an explicit
# availability indicator so the classifier can tell "expected to stay profitable" from
# "we could not tell".
BZ_BINARY = [
    "expected_loss_to_profit", "expected_profit_to_loss", "expected_loss_narrows",
    "expected_loss_widens", "eps_near_zero_prior", "eps_revenue_opposite_direction",
]
AVAILABILITY_SUFFIX = "_is_present"

GROUP_B = GROUP_A + BZ_LEVEL
GROUP_C = GROUP_B + BZ_GAP
GROUP_D = GROUP_C + BZ_DEVIATION

REFERENCE_SCORE = PHASE3_PREFIX + "risk_score"
REFERENCE_TIER = PHASE3_PREFIX + "bucket"
TIER_ORDER = {"Normal": 0.0, "Elevated": 1.0, "High Alert": 2.0}


def group_features(name: str) -> list[str]:
    base = {"A": GROUP_A, "B": GROUP_B, "C": GROUP_C, "D": GROUP_D}[name]
    indicators = [c + AVAILABILITY_SUFFIX for c in BZ_BINARY if c in base]
    return base + indicators


GROUP_NAMES = ("A", "B", "C", "D")


# ─────────────────────────────── dataset construction ────────────────────────────────────
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nyse_sessions(start="2010-01-01", end="2030-12-31") -> np.ndarray:
    days = mcal.get_calendar("NYSE").valid_days(start, end)
    return np.sort(pd.DatetimeIndex(days).tz_localize(None).normalize().to_numpy())


def outcome_ready_date(anchor: pd.Series, sessions: np.ndarray | None = None,
                       horizon: int = REACTION_SESSIONS) -> pd.Series:
    """The date the outcome is fully observable: `horizon` sessions after the anchor close.

    "Announced before the cutoff" is not the same as "known before the cutoff". A company
    reporting on 28 December has a 3-session reaction window that closes in January, so a
    model trained on 1 January with that row would be using an outcome the world had not
    finished producing. The walk-forward filters on THIS date, not on the announcement.
    """
    sessions = nyse_sessions() if sessions is None else sessions
    a = pd.to_datetime(anchor).to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(sessions, a, side="left")
    target = idx + horizon
    out = np.full(len(a), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ok = (~pd.isna(anchor).to_numpy()) & (target < len(sessions))
    out[ok] = sessions[target[ok]]
    return pd.Series(out, index=anchor.index)


@dataclass
class DatasetReport:
    """Everything that was dropped, and why. Written out next to the metrics."""
    rows_events_total: int = 0
    rows_outcome_available: int = 0
    rows_benzinga_total: int = 0
    rows_joined: int = 0
    rows_final: int = 0
    join_losses_by_year: list = field(default_factory=list)
    exclusions: dict = field(default_factory=dict)


def load_event_frame(events_path: Path = PHASE3_EVENTS_PATH) -> pd.DataFrame:
    """Phase 3 corrected outcomes + Phase 4 causal baselines + Phase 5 derived features."""
    events = pd.read_parquet(events_path)
    return derive_phase5_features(prepare_analysis_frame(events))


def load_benzinga(path: Path = BENZINGA_FEATURES_PATH) -> pd.DataFrame:
    bz = pd.read_parquet(path)
    bz = bz.copy()
    bz["stock"] = bz["ticker"].astype(str).str.strip().str.upper()
    bz["earnings_date"] = pd.to_datetime(bz["date"]).dt.normalize()
    return bz


def build_dataset(analysis: pd.DataFrame, benzinga: pd.DataFrame, *,
                  first_year: int = FIRST_TRAIN_YEAR,
                  last_year: int = LAST_TEST_YEAR) -> tuple[pd.DataFrame, DatasetReport]:
    """One row per earnings event, 1:1 joined, labels cut only after exclusions.

    Order matters and is deliberate: restrict to completed events, drop events whose
    corrected outcome is unavailable, THEN join, THEN cut binary labels. Cutting labels
    first would turn "we could not measure this reaction" into "this reaction was not
    extreme", which is exactly the fabrication the Phase 2/3 rules forbid.
    """
    rep = DatasetReport()
    d = analysis.copy()
    rep.rows_events_total = int(len(d))

    completed = ~d["is_pending"].astype(bool)
    available = d[PRIMARY_STATUS].eq(TARGET_AVAILABLE) & d[PRIMARY_TARGET].notna()
    d = d[completed & available].copy()
    rep.rows_outcome_available = int(len(d))
    rep.exclusions["pending_or_outcome_unavailable"] = rep.rows_events_total - len(d)

    d["stock"] = d["stock"].astype(str).str.strip().str.upper()
    d["earnings_date"] = pd.to_datetime(d["earnings_date"]).dt.normalize()
    d["year"] = d["event_clock"].dt.year
    before = len(d)
    d = d[d["year"].between(first_year, last_year)].copy()
    rep.exclusions["outside_window"] = before - len(d)

    if d.duplicated(["stock", "earnings_date"]).any():
        raise ValueError("event frame is not unique on (stock, earnings_date)")
    bz = benzinga
    if bz.duplicated(["stock", "earnings_date"]).any():
        raise ValueError("benzinga features are not unique on (stock, earnings_date)")
    rep.rows_benzinga_total = int(len(bz))

    bz_cols = ["stock", "earnings_date", "provider", "event_key", "fiscal_year",
               "fiscal_period", "last_updated", "retrieved_at_utc",
               "historical_estimate_vintage_verified", "eps_prior_check",
               "revenue_prior_check", "eps_regime_n", "revenue_regime_n",
               *BZ_LEVEL, *BZ_GAP, *BZ_DEVIATION]
    bz_cols = [c for c in dict.fromkeys(bz_cols) if c in bz.columns]
    merged = d.merge(bz[bz_cols], on=["stock", "earnings_date"], how="left",
                     validate="one_to_one", indicator=True)
    merged["benzinga_matched"] = merged["_merge"].eq("both")
    merged = merged.drop(columns="_merge")
    rep.rows_joined = int(merged["benzinga_matched"].sum())

    by_year = (merged.groupby("year")
               .agg(events=("stock", "size"), matched=("benzinga_matched", "sum")))
    by_year["unmatched"] = by_year["events"] - by_year["matched"]
    by_year["match_rate"] = by_year["matched"] / by_year["events"]
    rep.join_losses_by_year = by_year.reset_index().to_dict(orient="records")

    # An event with no vendor row cannot be part of an A-vs-B comparison at all: group A
    # would see it and groups B-D would not, which is a population difference wearing the
    # costume of a feature difference.
    merged = merged[merged["benzinga_matched"]].copy()
    rep.exclusions["no_benzinga_record"] = rep.rows_outcome_available - len(merged) \
        - rep.exclusions["outside_window"]

    # Labels are cut here, on rows whose outcome is known to exist.
    merged["y_primary"] = (merged[PRIMARY_TARGET] >= PRIMARY_THRESHOLD).astype(int)
    merged["y_secondary"] = (merged[PRIMARY_TARGET] >= SECONDARY_THRESHOLD).astype(int)

    # Explicit availability indicators: a missing flag is "unknown", never "no".
    for col in BZ_BINARY:
        if col in merged.columns:
            merged[col + AVAILABILITY_SUFFIX] = merged[col].notna().astype(float)

    merged["outcome_ready_date"] = outcome_ready_date(merged["anchor_date"])
    # An event whose window cannot be closed on the session grid can still be TESTED (its
    # outcome is available) but can never be TRAINED on, because we cannot prove when it
    # became known. Marked, not dropped.
    merged["outcome_ready_known"] = merged["outcome_ready_date"].notna()
    merged["event_id"] = merged["stock"] + "|" + merged["earnings_date"].dt.strftime("%Y-%m-%d")
    merged["quarter"] = merged["event_clock"].dt.to_period("Q").astype(str)
    rep.rows_final = int(len(merged))
    return merged.reset_index(drop=True), rep


# ──────────────────────────────── walk-forward evaluation ────────────────────────────────
def training_mask(d: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Rows a model fitted at `cutoff` is allowed to have learned from.

    Two conditions, both necessary: the event was announced before the cutoff AND its
    reaction window had closed before the cutoff. Rows whose close date cannot be placed on
    the session grid are excluded from training rather than assumed safe.
    """
    return (d["event_clock"] < cutoff) & d["outcome_ready_known"] \
        & (d["outcome_ready_date"] < cutoff)


def test_mask(d: pd.DataFrame, year: int) -> pd.Series:
    return d["event_clock"].dt.year.eq(year)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                label: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit on `train`, score `test`, and take the alert threshold from TRAINING scores.

    Nothing is imputed or scaled: the classifier splits on NaN natively, so missingness
    reaches the model intact and no transformation has to be fitted at all. The single
    quantity estimated from data outside the model is the alert threshold, and it comes
    from the training predictions, never from the test year.
    """
    X_train = train[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X_test = test[features].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    model = HistGradientBoostingClassifier(**CLASSIFIER_PARAMS)
    model.fit(X_train, train[label].to_numpy(int))
    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]
    return p_test, p_train, float(np.quantile(p_train, ALERT_QUANTILE))


def walk_forward(d: pd.DataFrame, features: list[str], group: str, label: str,
                 *, first_test_year: int = FIRST_TEST_YEAR,
                 last_test_year: int = LAST_TEST_YEAR) -> pd.DataFrame:
    """Expanding-window walk-forward. Returns event-level held-out predictions."""
    out = []
    for year in range(first_test_year, last_test_year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train = d[training_mask(d, cutoff)]
        test = d[test_mask(d, year)]
        if len(test) == 0 or train[label].nunique() < 2 or len(train) < 500:
            continue
        p_test, _p_train, alert = fit_predict(train, test, features, label)
        out.append(pd.DataFrame({
            "group": group,
            "label": label,
            "test_year": year,
            "event_id": test["event_id"].to_numpy(),
            "stock": test["stock"].to_numpy(),
            "quarter": test["quarter"].to_numpy(),
            "y": test[label].to_numpy(int),
            "p": p_test,
            "alert_threshold": alert,
            "train_n": len(train),
            "train_cutoff": cutoff.date().isoformat(),
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ─────────────────────────────────────── metrics ─────────────────────────────────────────
def _auc(y, p) -> float:
    y = np.asarray(y, dtype=int)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def _ap(y, p) -> float:
    y = np.asarray(y, dtype=int)
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def alert_metrics(y: np.ndarray, p: np.ndarray, threshold: np.ndarray) -> dict:
    """Precision, capture and alert count at the TRAINING-derived threshold.

    `threshold` is per-row because each test year carries its own training block's 90th
    percentile; pooling the years means pooling their thresholds too, which is what a live
    system would actually have done.
    """
    flagged = p >= threshold
    n_alerts = int(flagged.sum())
    hits = int(y[flagged].sum()) if n_alerts else 0
    total = int(y.sum())
    return {
        "alerts": n_alerts,
        "alert_rate": float(flagged.mean()) if len(y) else float("nan"),
        "alert_precision": float(hits / n_alerts) if n_alerts else float("nan"),
        "alert_capture": float(hits / total) if total else float("nan"),
        "alert_lift": float((hits / n_alerts) / y.mean()) if n_alerts and y.mean() > 0
        else float("nan"),
    }


def within_stock_macro_auc(pred: pd.DataFrame, min_events: int = 2) -> dict:
    """Macro-average of per-stock AUCs over stocks that carry both outcome classes.

    Asks the question the pooled AUC cannot: given two quarters of the SAME company, one
    extreme and one not, does the model put them in the right order? Stocks without both
    classes are not scoreable and are counted, not imputed to 0.5.
    """
    aucs, events, stocks_total = [], 0, pred["stock"].nunique()
    for _stock, g in pred.groupby("stock", sort=False):
        if len(g) < min_events or g["y"].nunique() < 2:
            continue
        aucs.append(roc_auc_score(g["y"].to_numpy(int), g["p"].to_numpy(float)))
        events += len(g)
    return {
        "within_stock_macro_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "within_stock_eligible_stocks": len(aucs),
        "within_stock_total_stocks": int(stocks_total),
        "within_stock_eligible_events": int(events),
    }


def score_block(pred: pd.DataFrame, scope: str) -> dict:
    y = pred["y"].to_numpy(int)
    p = pred["p"].to_numpy(float)
    row = {
        "group": pred["group"].iloc[0],
        "label": pred["label"].iloc[0],
        "scope": scope,
        "n": len(pred),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "roc_auc": _auc(y, p),
        "average_precision": _ap(y, p),
    }
    row.update(alert_metrics(y, p, pred["alert_threshold"].to_numpy(float)))
    row.update(within_stock_macro_auc(pred))
    return row


def quarter_block_bootstrap(pred_a: pd.DataFrame, pred_b: pd.DataFrame, *,
                            reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict:
    """Paired differences B-minus-A, resampling whole calendar quarters.

    Events are not independent: a quarter shares a macro regime, and the same companies
    recur every quarter. Resampling rows would give a confidently narrow interval around a
    difference that is really one market episode repeated. Quarters are the natural block —
    they are also the unit an earnings dataset is generated in.
    """
    j = pred_a[["event_id", "quarter", "y", "p", "alert_threshold"]].merge(
        pred_b[["event_id", "p", "alert_threshold"]], on="event_id",
        suffixes=("_a", "_b"), validate="one_to_one")
    if len(j) == 0:
        return {"n": 0}
    y = j["y"].to_numpy(int)
    pa, pb = j["p_a"].to_numpy(float), j["p_b"].to_numpy(float)
    ta, tb = j["alert_threshold_a"].to_numpy(float), j["alert_threshold_b"].to_numpy(float)

    def stats(idx):
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            return None
        return (
            _auc(yy, pb[idx]) - _auc(yy, pa[idx]),
            _ap(yy, pb[idx]) - _ap(yy, pa[idx]),
            (alert_metrics(yy, pb[idx], tb[idx])["alert_precision"]
             - alert_metrics(yy, pa[idx], ta[idx])["alert_precision"]),
            (alert_metrics(yy, pb[idx], tb[idx])["alert_capture"]
             - alert_metrics(yy, pa[idx], ta[idx])["alert_capture"]),
        )

    point = stats(np.arange(len(j)))
    blocks = j.groupby("quarter", sort=True).indices
    keys = list(blocks)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        picked = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([blocks[keys[i]] for i in picked])
        s = stats(idx)
        if s is not None:
            draws.append(s)
    arr = np.asarray(draws, dtype=float)
    names = ["delta_roc_auc", "delta_average_precision", "delta_alert_precision",
             "delta_alert_capture"]
    out = {"n": len(j), "blocks": len(keys), "reps_valid": int(len(arr))}
    for i, name in enumerate(names):
        out[name] = float(point[i]) if point else float("nan")
        col = arr[:, i] if len(arr) else np.array([])
        col = col[~np.isnan(col)]
        out[name + "_ci_lo"] = float(np.quantile(col, 0.025)) if len(col) else float("nan")
        out[name + "_ci_hi"] = float(np.quantile(col, 0.975)) if len(col) else float("nan")
        out[name + "_share_positive"] = float((col > 0).mean()) if len(col) else float("nan")
    return out


# ───────────────────────────── descriptive reference benchmarks ──────────────────────────
def reference_benchmarks(d: pd.DataFrame, pred_index: pd.DataFrame, label: str) -> list[dict]:
    """The shipped score and tier, scored on exactly the held-out rows, untrained.

    These are DESCRIPTIVE, not competitors. Comparing a freshly trained gradient-boosted
    model against a hand-built 0-100 score would confound "the new features help" with "a
    trained model beats a hand-built score" — which is why groups A-D exist. These rows
    answer a different and still useful question: where does the thing currently in
    production sit on this same test set?
    """
    ref = d.set_index("event_id")
    rows = []
    for col, name in ((REFERENCE_SCORE, f"reference_score_{MODEL_VERSION}"),
                      (REFERENCE_TIER, f"reference_tier_{MODEL_VERSION}")):
        if col not in ref.columns:
            continue
        values = ref[col]
        if col == REFERENCE_TIER:
            values = values.astype(object).map(TIER_ORDER)
        for scope, sub in [("pooled", pred_index)] + [
                (f"year_{y}", g) for y, g in pred_index.groupby("test_year")]:
            s = values.reindex(sub["event_id"]).to_numpy(float)
            ok = ~np.isnan(s)
            y = sub["y"].to_numpy(int)[ok]
            if len(np.unique(y)) < 2:
                continue
            # The reference has no training block, so its alert threshold is its own 90th
            # percentile on the evaluated rows. Flagged as such: this is the one place a
            # threshold is not training-derived, and it FAVOURS the reference.
            thr = np.full(ok.sum(), float(np.quantile(s[ok], ALERT_QUANTILE)))
            row = {"group": name, "label": label, "scope": scope, "n": int(ok.sum()),
                   "positives": int(y.sum()), "base_rate": float(y.mean()),
                   "roc_auc": _auc(y, s[ok]), "average_precision": _ap(y, s[ok]),
                   "threshold_source": "in_sample_quantile_descriptive_only"}
            row.update(alert_metrics(y, s[ok], thr))
            row.update(within_stock_macro_auc(
                pd.DataFrame({"stock": sub["stock"].to_numpy()[ok], "y": y, "p": s[ok]})))
            rows.append(row)
    return rows


# ─────────────────────────────────────── the run ─────────────────────────────────────────
def _git_state() -> dict:
    def sh(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return "unavailable"
    return {"branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "commit": sh("git", "rev-parse", "HEAD"),
            "dirty": sh("git", "status", "--porcelain") != ""}


def run(*, events_path: Path = PHASE3_EVENTS_PATH,
        benzinga_path: Path = BENZINGA_FEATURES_PATH,
        results_dir: Path = RESULTS_DIR,
        bootstrap_reps: int = BOOTSTRAP_REPS) -> dict:
    analysis = load_event_frame(events_path)
    benzinga = load_benzinga(benzinga_path)
    data, report = build_dataset(analysis, benzinga)

    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows, delta_rows, prediction_frames = [], [], []

    labels = {"y_primary": PRIMARY_THRESHOLD, "y_secondary": SECONDARY_THRESHOLD}
    preds: dict[tuple[str, str], pd.DataFrame] = {}
    for label in labels:
        for group in GROUP_NAMES:
            p = walk_forward(data, group_features(group), group, label)
            if len(p) == 0:
                continue
            preds[(label, group)] = p
            prediction_frames.append(p)
            metrics_rows.append(score_block(p, "pooled"))
            for year, g in p.groupby("test_year"):
                metrics_rows.append(score_block(g, f"year_{int(year)}"))

        base = preds.get((label, "A"))
        if base is not None:
            for group in ("B", "C", "D"):
                other = preds.get((label, group))
                if other is None:
                    continue
                d_ = quarter_block_bootstrap(base, other, reps=bootstrap_reps)
                d_.update({"label": label, "comparison": f"{group}_vs_A", "scope": "pooled"})
                delta_rows.append(d_)
        metrics_rows.extend(reference_benchmarks(data, preds[(label, "A")], label))

    # ── C vs D on identical events that actually HAVE deviations ─────────────────────────
    # Group D adds features that are missing on 18-28% of events, and missing far more often
    # in 2020-2021 when the pandemic broke every company's "normal growth" history. On the
    # full sample, D-minus-C mixes a feature effect with a composition effect. Restricting
    # both to rows where every deviation column is present separates them.
    deviation_ready = data[BZ_DEVIATION].notna().all(axis=1)
    restricted = data[deviation_ready].copy()
    restricted_rows, restricted_delta = [], []
    if len(restricted) > 1000:
        for label in labels:
            r_preds = {}
            for group in ("C", "D"):
                p = walk_forward(restricted, group_features(group), group, label)
                if len(p) == 0:
                    continue
                p = p.assign(group=group + "_deviation_ready")
                r_preds[group] = p
                prediction_frames.append(p)
                restricted_rows.append(score_block(p, "pooled_deviation_ready"))
                for year, g in p.groupby("test_year"):
                    restricted_rows.append(score_block(g, f"year_{int(year)}_deviation_ready"))
            if "C" in r_preds and "D" in r_preds:
                d_ = quarter_block_bootstrap(r_preds["C"], r_preds["D"], reps=bootstrap_reps)
                d_.update({"label": label, "comparison": "D_vs_C_deviation_ready",
                           "scope": "pooled_deviation_ready"})
                restricted_delta.append(d_)

    metrics = pd.DataFrame(metrics_rows + restricted_rows)
    deltas = pd.DataFrame(delta_rows + restricted_delta)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    config = {
        "experiment": "benzinga_expectation_feature_value",
        "status": "RETROSPECTIVE RESEARCH — NOT point-in-time validated",
        "vintage_warning": (
            "Benzinga estimates are latest-vintage. `last_updated` on a 2014 event is "
            "routinely a 2021-2023 timestamp, and prior-period actuals are restated. "
            "Results are an UPPER BOUND on what a live system could have achieved."),
        "model_version_reference": MODEL_VERSION,
        "outcome": {
            "column": PRIMARY_TARGET,
            "definition": (f"|reaction| over {REACTION_SESSIONS} post-announcement market "
                           "sessions measured from the last close strictly before the "
                           "announcement (Phase 3 corrected anchoring)"),
            "primary_threshold": PRIMARY_THRESHOLD,
            "secondary_threshold": SECONDARY_THRESHOLD,
            "availability_rule": f"{PRIMARY_STATUS} == '{TARGET_AVAILABLE}' and target notna",
        },
        "prediction_cutoff": (
            "the anchor close — the last market close strictly before the announcement. "
            "Baseline price features are rolling().shift(1) on the daily grid (as of D-1 "
            "close); structural baselines use only PRIOR resolved corrected reactions, with "
            "same-date outcomes entering history only after every event on that date is "
            "scored; Benzinga expectations are pre-announcement consensus by construction "
            "but latest-vintage by provenance."),
        "walk_forward": {
            "first_train_year": FIRST_TRAIN_YEAR,
            "first_test_year": FIRST_TEST_YEAR,
            "last_test_year": LAST_TEST_YEAR,
            "scheme": "expanding window; train on all events whose reaction window CLOSED "
                      "before 1 Jan of the test year",
        },
        "classifier": {"name": "HistGradientBoostingClassifier", **CLASSIFIER_PARAMS},
        "alert_quantile": ALERT_QUANTILE,
        "alert_threshold_source": "90th percentile of TRAINING predictions",
        "bootstrap": {"scheme": "paired, resampling calendar-quarter blocks",
                      "reps": bootstrap_reps, "seed": BOOTSTRAP_SEED},
        "missingness": "preserved; HistGradientBoosting splits on NaN natively; no "
                       "imputation is fitted; binary flags carry explicit "
                       f"'{AVAILABILITY_SUFFIX}' indicators",
        "feature_groups": {g: group_features(g) for g in GROUP_NAMES},
        "git": _git_state(),
        "inputs": {
            str(events_path): {"sha256": _sha256(events_path),
                               "bytes": events_path.stat().st_size},
            str(benzinga_path): {"sha256": _sha256(benzinga_path),
                                 "bytes": benzinga_path.stat().st_size},
        },
        "dataset": {
            "rows_events_total": report.rows_events_total,
            "rows_outcome_available": report.rows_outcome_available,
            "rows_benzinga_total": report.rows_benzinga_total,
            "rows_joined": report.rows_joined,
            "rows_final": report.rows_final,
            "exclusions": report.exclusions,
            "deviation_ready_rows": int(deviation_ready.sum()),
            "stocks": int(data["stock"].nunique()),
            "primary_base_rate": float(data["y_primary"].mean()),
            "secondary_base_rate": float(data["y_secondary"].mean()),
        },
    }

    (results_dir / "config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    pd.DataFrame(report.join_losses_by_year).to_csv(
        results_dir / "join_losses_by_year.csv", index=False)
    metrics.to_csv(results_dir / "metrics.csv", index=False)
    deltas.to_csv(results_dir / "paired_deltas.csv", index=False)
    predictions.to_csv(results_dir / "held_out_predictions.csv", index=False)

    findings = derive_findings(metrics, deltas)
    pooled = metrics[metrics["scope"].str.startswith("pooled")]
    summary = {
        **{k: config[k] for k in ("experiment", "status", "vintage_warning", "outcome",
                                  "prediction_cutoff", "walk_forward")},
        "dataset": config["dataset"],
        "join_losses_by_year": report.join_losses_by_year,
        "findings": findings,
        "pooled_metrics": pooled.to_dict(orient="records"),
        "paired_deltas_vs_A": deltas.to_dict(orient="records"),
        "guardrails": [
            "outcomes excluded before labels were cut; an unavailable reaction is never a "
            "non-extreme training example",
            "1:1 join asserted with validate='one_to_one'; duplicates raise",
            "training rows require the reaction window to have CLOSED before the cutoff",
            "identical train/test event ids across groups within a comparison",
            "no imputation fitted; missingness preserved; binary flags carry availability "
            "indicators",
            "one classifier, one seed, no parameter search, no random split",
            "reference score/tier are descriptive only and use an in-sample threshold that "
            "favours them",
            "production scoring, thresholds and ingestion are untouched",
        ],
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def derive_findings(metrics: pd.DataFrame, deltas: pd.DataFrame) -> dict:
    """The verdict, computed from the tables rather than written by hand.

    A group "helps" only if its pooled AUC gain over A has a quarter-block CI that excludes
    zero AND it beats A in a majority of test years. The second condition is the one that
    usually fails: a pooled gain built out of two good years and six coin flips is a
    description of 2021 and 2023, not of a feature.
    """
    out = {}
    yearly = metrics[metrics["scope"].str.match(r"year_\d+$", na=False)].copy()
    yearly["year"] = yearly["scope"].str.extract(r"(\d+)").astype(int)
    for label in metrics["label"].dropna().unique():
        pooled = metrics[metrics["label"].eq(label) & metrics["scope"].eq("pooled")]
        by_group = pooled.set_index("group")
        if "A" not in by_group.index:
            continue
        piv = (yearly[yearly["label"].eq(label)]
               .pivot_table(index="year", columns="group", values="roc_auc"))
        entry = {"pooled_auc": {g: float(by_group.loc[g, "roc_auc"])
                                for g in by_group.index},
                 "groups": {}}
        for group in ("B", "C", "D"):
            if group not in by_group.index:
                continue
            row = deltas[deltas["label"].eq(label)
                         & deltas["comparison"].eq(f"{group}_vs_A")]
            delta = row.iloc[0].to_dict() if len(row) else {}
            wins = int((piv[group] > piv["A"]).sum()) if group in piv else 0
            years = int(piv[group].notna().sum()) if group in piv else 0
            ci_excludes_zero = bool(delta.get("delta_roc_auc_ci_lo", -1) > 0)
            entry["groups"][group] = {
                "delta_roc_auc": delta.get("delta_roc_auc"),
                "ci": [delta.get("delta_roc_auc_ci_lo"), delta.get("delta_roc_auc_ci_hi")],
                "ci_excludes_zero": ci_excludes_zero,
                "years_better_than_A": wins,
                "years_evaluated": years,
                "majority_of_years": bool(years and wins / years > 0.5),
                "delta_alert_precision": delta.get("delta_alert_precision"),
                "delta_alert_capture": delta.get("delta_alert_capture"),
                "within_stock_macro_auc": float(by_group.loc[group, "within_stock_macro_auc"]),
                "within_stock_delta_vs_A": float(by_group.loc[group, "within_stock_macro_auc"]
                                                 - by_group.loc["A", "within_stock_macro_auc"]),
                "verdict": ("helps" if (ci_excludes_zero and years and wins / years > 0.5)
                            else "inconclusive" if ci_excludes_zero or (years and wins / years > 0.5)
                            else "no measurable help"),
            }
        out[label] = entry
    restricted = deltas[deltas["comparison"].eq("D_vs_C_deviation_ready")]
    out["deviation_ready_D_vs_C"] = restricted.to_dict(orient="records")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--events", type=Path, default=PHASE3_EVENTS_PATH)
    ap.add_argument("--benzinga", type=Path, default=BENZINGA_FEATURES_PATH)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--bootstrap-reps", type=int, default=BOOTSTRAP_REPS)
    a = ap.parse_args(argv)
    s = run(events_path=a.events, benzinga_path=a.benzinga, results_dir=a.results_dir,
            bootstrap_reps=a.bootstrap_reps)
    print(json.dumps({"dataset": s["dataset"], "deltas": s["paired_deltas_vs_A"]},
                     indent=2, default=str))
    print(f"\nwrote {a.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
