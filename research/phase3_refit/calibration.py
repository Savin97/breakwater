"""Walk-forward fitting, window calibration and tier cut points.

For test year Y, everything — imputation medians, scaling, coefficients, Platt maps, tier
cut points — is estimated from training rows whose 3-session outcome endpoint falls
before 1 January Y, then frozen and applied to year Y. Nothing reads a test outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from research.phase3_refit.candidates import IMPUTABLE, MODEL_INPUTS, TIERS

LOGIT_C = 1.0
LOGIT_MAX_ITER = 2000
WINDOWS = ("W0", "W1", "W2")
TIER_RULES = ("Q_common", "Q_by_window", "P_ratio")
HIGH_ALERT_Q = 0.90
ELEVATED_Q = 0.80
P_HIGH_ALERT_RATIO = 2.0
P_ELEVATED_RATIO = 1.5


@dataclass
class Fold:
    """Everything learned from one training block."""
    year: int
    inputs: list[str]
    medians: pd.Series
    means: pd.Series
    stds: pd.Series
    model: LogisticRegression
    platt: dict[str, LogisticRegression] = field(default_factory=dict)
    cuts: dict[str, dict] = field(default_factory=dict)
    base_rate: float = np.nan
    train_n: int = 0


def training_mask(frame: pd.DataFrame, year: int) -> pd.Series:
    """Rows whose outcome was observable before 1 January `year`."""
    return frame["endpoint3_date"] < pd.Timestamp(year=year, month=1, day=1)


def _design(frame: pd.DataFrame, inputs: list[str], medians: pd.Series,
            means: pd.Series, stds: pd.Series) -> np.ndarray:
    X = frame[inputs].apply(pd.to_numeric, errors="coerce").fillna(medians)
    return ((X - means) / stds).to_numpy(float)


def fit_model(train: pd.DataFrame, inputs: list[str], year: int) -> Fold:
    X = train[inputs].apply(pd.to_numeric, errors="coerce")
    missing = [c for c in inputs if c not in IMPUTABLE and X[c].isna().any()]
    if missing:
        raise ValueError(f"non-imputable inputs missing in training: {missing}")
    medians = X.median()
    X = X.fillna(medians)
    means, stds = X.mean(), X.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    model = LogisticRegression(C=LOGIT_C, max_iter=LOGIT_MAX_ITER, random_state=0)
    model.fit(((X - means) / stds).to_numpy(float), train["y_extreme"].to_numpy(int))
    return Fold(year=year, inputs=inputs, medians=medians, means=means, stds=stds,
                model=model, base_rate=float(train["y_extreme"].mean()), train_n=len(train))


def raw_probability(fold: Fold, frame: pd.DataFrame) -> np.ndarray:
    return fold.model.predict_proba(
        _design(frame, fold.inputs, fold.medians, fold.means, fold.stds))[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_window_platt(fold: Fold, train: pd.DataFrame) -> None:
    """W1: a separate (intercept, slope) on logit(p) per window, fitted on training rows."""
    z = _logit(raw_probability(fold, train))
    for w, idx in train.groupby("announce_window").indices.items():
        y = train["y_extreme"].to_numpy(int)[idx]
        m = LogisticRegression(C=1e6, max_iter=LOGIT_MAX_ITER)
        m.fit(z[idx].reshape(-1, 1), y)
        fold.platt[w] = m


def probability(fold: Fold, frame: pd.DataFrame, window_mode: str) -> np.ndarray:
    p = raw_probability(fold, frame)
    if window_mode != "W1":
        return p
    z = _logit(p).reshape(-1, 1)
    out = np.full(len(frame), np.nan)
    for w, m in fold.platt.items():
        mask = frame["announce_window"].to_numpy() == w
        if mask.any():
            out[mask] = m.predict_proba(z[mask])[:, 1]
    return out


def fit_cuts(fold: Fold, p_train: np.ndarray, train: pd.DataFrame) -> None:
    """Tier cut points for every rule, from training-fold probabilities only."""
    fold.cuts["Q_common"] = {"_all": (np.quantile(p_train, ELEVATED_Q),
                                      np.quantile(p_train, HIGH_ALERT_Q))}
    by = {}
    for w, idx in train.groupby("announce_window").indices.items():
        by[w] = (np.quantile(p_train[idx], ELEVATED_Q), np.quantile(p_train[idx], HIGH_ALERT_Q))
    fold.cuts["Q_by_window"] = by
    b = fold.base_rate
    fold.cuts["P_ratio"] = {"_all": (P_ELEVATED_RATIO * b, P_HIGH_ALERT_RATIO * b)}


def assign_tiers(p: np.ndarray, windows: np.ndarray, cuts: dict) -> np.ndarray:
    out = np.full(len(p), "Normal", dtype=object)
    for key, (elev, high) in cuts.items():
        mask = np.ones(len(p), bool) if key == "_all" else (windows == key)
        out[mask & (p >= elev)] = "Elevated"
        out[mask & (p >= high)] = "High Alert"
    return out


def walk_forward(frame: pd.DataFrame, candidate: str, window_mode: str, *,
                 years: range, score_frame: pd.DataFrame | None = None
                 ) -> tuple[pd.DataFrame, dict[int, Fold]]:
    """Out-of-fold probabilities and tiers for one (candidate, window mode).

    `frame` holds the population (training and test rows). `score_frame`, if given, is
    scored by each fold as well — used to put prior events into tiers for the lift test.
    Returns (oof rows, {year: Fold}).
    """
    inputs = list(MODEL_INPUTS[candidate]) + (["is_bmo"] if window_mode == "W2" else [])
    rows, folds = [], {}
    for year in years:
        train = frame[training_mask(frame, year)]
        test = frame[frame["year"].eq(year)]
        if len(test) == 0 or train["y_extreme"].nunique() < 2:
            continue
        fold = fit_model(train, inputs, year)
        if window_mode == "W1":
            fit_window_platt(fold, train)
        p_train = probability(fold, train, window_mode)
        fit_cuts(fold, p_train, train)
        p = probability(fold, test, window_mode)
        out = pd.DataFrame({"p": p, "test_year": year, "train_n": len(train)}, index=test.index)
        w = test["announce_window"].to_numpy()
        for rule in TIER_RULES:
            out[f"tier_{rule}"] = assign_tiers(p, w, fold.cuts[rule])
        rows.append(out)
        folds[year] = fold
    return pd.concat(rows), folds


def tiers_for(fold: Fold, frame: pd.DataFrame, window_mode: str, rule: str) -> pd.Series:
    """Score any frame with a frozen fold and tier it. Rows lacking inputs stay NaN."""
    need = [c for c in fold.inputs if c not in IMPUTABLE]
    ok = frame[need].notna().all(axis=1)
    out = pd.Series(np.nan, index=frame.index, dtype=object)
    if ok.any():
        sub = frame[ok]
        p = probability(fold, sub, window_mode)
        out[ok] = assign_tiers(p, sub["announce_window"].to_numpy(), fold.cuts[rule])
    return out


TIER_ORDER = {t: i for i, t in enumerate(TIERS)}
