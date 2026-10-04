"""Model C refit — fitting, walk-forward, metrics, bootstrap, calibration. Rules: SPEC.md."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

MODELS = {"B": ["log_hist_mean_abs"], "C": ["log_hist_mean_abs", "log_vol_30d"]}
LOGIT_C = 1.0
MAX_ITER = 2000
TEST_YEARS = range(2017, 2026)
HOLDOUT_YEAR = 2026
BOOT_REPS = 500
BOOT_SEED = 32032


@dataclass
class Fit:
    inputs: list[str]
    means: pd.Series
    stds: pd.Series
    model: LogisticRegression
    train_n: int
    train_base_rate: float
    train_end: pd.Timestamp

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        X = ((frame[self.inputs] - self.means) / self.stds).to_numpy(float)
        return self.model.predict_proba(X)[:, 1]

    def coefficients(self) -> dict:
        return {"intercept": float(self.model.intercept_[0]),
                **{f"coef_z_{c}": float(b) for c, b in zip(self.inputs, self.model.coef_[0])},
                **{f"mean_{c}": float(self.means[c]) for c in self.inputs},
                **{f"std_{c}": float(self.stds[c]) for c in self.inputs},
                "train_n": self.train_n, "train_base_rate": self.train_base_rate,
                "train_endpoint_before": str(self.train_end.date())}


def training_mask(frame: pd.DataFrame, year: int) -> pd.Series:
    """Rows whose outcome was observable (endpoint) before 1 January `year`."""
    return frame["endpoint_date"] < pd.Timestamp(year=year, month=1, day=1)


def fit(train: pd.DataFrame, inputs: list[str], train_end: pd.Timestamp) -> Fit:
    X = train[inputs].astype(float)
    if X.isna().any().any():
        raise ValueError("missing model inputs in training rows")
    means, stds = X.mean(), X.std(ddof=0).replace(0.0, 1.0)
    m = LogisticRegression(C=LOGIT_C, max_iter=MAX_ITER, random_state=0)
    m.fit(((X - means) / stds).to_numpy(float), train["y"].to_numpy(int))
    return Fit(inputs, means, stds, m, len(train), float(train["y"].mean()), train_end)


def walk_forward(sample: pd.DataFrame, model: str, years=TEST_YEARS) -> tuple[pd.DataFrame, dict]:
    """OOF probabilities for `model` over `years`, plus the per-year fits."""
    rows, fits = [], {}
    for year in years:
        train = sample[training_mask(sample, year)]
        test = sample[sample["year"].eq(year)]
        if test.empty or train["y"].nunique() < 2:
            continue
        f = fit(train, MODELS[model], pd.Timestamp(year=year, month=1, day=1))
        rows.append(pd.DataFrame({"p": f.predict(test), "test_year": year}, index=test.index))
        fits[year] = f
    return pd.concat(rows), fits


def platt_walk_forward(sample: pd.DataFrame, score_col: str, years=TEST_YEARS) -> pd.Series:
    """A one-input logistic on a raw score (for the shipped 0.3.1 reference)."""
    out = []
    for year in years:
        train = sample[training_mask(sample, year) & sample[score_col].notna()]
        test = sample[sample["year"].eq(year)]
        m = LogisticRegression(C=1e6, max_iter=MAX_ITER)
        m.fit(train[[score_col]].to_numpy(float), train["y"].to_numpy(int))
        x = test[[score_col]].fillna(train[score_col].median()).to_numpy(float)
        out.append(pd.Series(m.predict_proba(x)[:, 1], index=test.index))
    return pd.concat(out)


# ────────────────────────────────────────── metrics ─────────────────────────────────────
def _top(y: np.ndarray, p: np.ndarray, frac: float) -> tuple[float, float, float]:
    k = max(1, int(math.ceil(frac * len(y))))
    top = y[np.argsort(-p, kind="mergesort")[:k]]
    base, tot = y.mean(), y.sum()
    return (float(top.mean()), float(top.sum() / tot) if tot else np.nan,
            float(top.mean() / base) if base else np.nan)


def metrics(y, p) -> dict:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    out = {"n": len(y), "base_rate": float(y.mean()) if len(y) else np.nan}
    if len(np.unique(y)) < 2:
        return out
    q = np.clip(p, 1e-6, 1 - 1e-6)
    out.update({"roc_auc": float(roc_auc_score(y, p)),
                "pr_auc": float(average_precision_score(y, p)),
                "brier": float(np.mean((q - y) ** 2)),
                "log_loss": float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))),
                "mean_p": float(q.mean())})
    for f in (0.10, 0.20):
        h, c, lift = _top(y, p, f)
        k = int(f * 100)
        out[f"top{k}_hit"], out[f"top{k}_capture"], out[f"top{k}_lift"] = h, c, lift
    return out


def within_week_top(y, p, week, frac: float) -> dict:
    """Top `frac` of each call week by p (at least one event per week), pooled."""
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float), "w": week})
    picked = []
    for _, g in df.groupby("w", sort=False):
        k = max(1, int(math.ceil(frac * len(g))))
        picked.append(g.sort_values("p", ascending=False, kind="mergesort").head(k))
    top = pd.concat(picked)
    return {"hit": float(top["y"].mean()), "capture": float(top["y"].sum() / df["y"].sum()),
            "share": len(top) / len(df)}


# ─────────────────────────────────── clustered bootstrap ────────────────────────────────
DELTA_KEYS = ["roc_auc", "pr_auc", "brier", "log_loss", "top10_hit", "top10_capture",
              "top20_hit", "top20_capture"]


def cluster_indices(stocks: np.ndarray, reps: int = BOOT_REPS, seed: int = BOOT_SEED):
    groups = pd.Series(np.arange(len(stocks))).groupby(stocks).indices
    keys = list(groups)
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        pick = rng.integers(0, len(keys), size=len(keys))
        yield np.concatenate([groups[keys[i]] for i in pick])


def bootstrap_delta(y, pa, pb, stocks, reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> dict:
    """a − b for every DELTA_KEYS metric: point, 95% CI, share of draws > 0."""
    y, pa, pb = np.asarray(y, int), np.asarray(pa, float), np.asarray(pb, float)

    def stat(idx):
        a, b = metrics(y[idx], pa[idx]), metrics(y[idx], pb[idx])
        return {k: a[k] - b[k] for k in DELTA_KEYS if k in a and k in b}

    point = stat(np.arange(len(y)))
    draws = {k: [] for k in point}
    for idx in cluster_indices(np.asarray(stocks), reps, seed):
        d = stat(idx)
        for k, v in d.items():
            draws[k].append(v)
    out = {}
    for k, v in point.items():
        a = np.asarray(draws[k], float)
        out[k] = {"delta": v, "lo": float(np.quantile(a, 0.025)),
                  "hi": float(np.quantile(a, 0.975)), "share_pos": float((a > 0).mean())}
    return out


# ──────────────────────────────────────── calibration ───────────────────────────────────
def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("bin")
    return pd.DataFrame({"n": g.size(), "p_lo": g["p"].min(), "p_hi": g["p"].max(),
                         "mean_p": g["p"].mean(), "observed": g["y"].mean()}).reset_index()


def calibration_line(y, p) -> dict:
    """Logistic recalibration y ~ a + b·logit(p): a = 0, b = 1 means calibrated."""
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    z = np.log(q / (1 - q)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=MAX_ITER).fit(z, np.asarray(y, int))
    return {"intercept": float(m.intercept_[0]), "slope": float(m.coef_[0][0]),
            "mean_p": float(q.mean()), "observed": float(np.mean(y))}
