"""P3.2 + P3.3 refit — walk-forward evaluation and the pre-registered selection of E.

Research-only: reads `output/phase3_target_rebuild/phase3_events.parquet` and
`output/full_df.parquet`, writes `output/phase3_refit/`. No production module, threshold,
database or vendor file is written. Definitions and decision rules: PREREGISTRATION.md.

    PYTHONPATH=. .venv/bin/python -m research.phase3_refit.evaluate
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from feature_engineering.announcement_timing import AMC, BMO
from research.phase3_refit import calibration as cal
from research.phase3_refit.candidates import (
    MECHANICS,
    MODEL_INPUTS,
    TIERS,
    add_model_inputs,
    promote,
    v031_score,
    v031_structural_tier,
)
from research.phase3_refit.features import (
    build_feature_frame,
    history_features,
    stock_bucket_lift,
)

PHASE3_EVENTS = Path("output/phase3_target_rebuild/phase3_events.parquet")
FULL_DF = Path("output/full_df.parquet")
PROVIDER_TS = Path("audit/provider_timestamps.parquet")
RESULTS = Path("output/phase3_refit")
FEATURE_CACHE = RESULTS / "feature_frame.parquet"

POP_YEARS = (2014, 2026)
TEST_YEARS = range(2017, 2026)
HOLDOUT_YEAR = 2026
MIN_PRIOR = 8
BOOT_REPS = 500
BOOT_SEED = 32032
LIFT_STRENGTHS = (20, 10, 40)
HC_Z = 1.5
CANDIDATES = list(MODEL_INPUTS)
REQUIRED = ["log_hist_mean_abs", "log_vol_30d", "log_idio_vol_30d", "a1_score", "a0_shipped"]


# ─────────────────────────────────────────── data ───────────────────────────────────────
def load_frame(rebuild: bool = False) -> pd.DataFrame:
    if FEATURE_CACHE.exists() and not rebuild:
        return pd.read_parquet(FEATURE_CACHE)
    events = pd.read_parquet(PHASE3_EVENTS)
    daily = pd.read_parquet(FULL_DF, columns=["stock", "date", "price", "sector", "sub_sector",
                                              "vol_30d", "drift_30d"])
    frame = build_feature_frame(events, daily)
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FEATURE_CACHE)
    return frame


def population(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[~frame["is_pending"].astype(bool) & frame["window_ok"]
                 & frame["y_extreme"].notna() & frame["year"].between(*POP_YEARS)
                 & frame["n_prior_call"].ge(MIN_PRIOR)].copy()


def common_rows(call: pd.DataFrame, eve: pd.DataFrame) -> pd.Index:
    ok = call[REQUIRED].notna().all(axis=1) & eve[REQUIRED].notna().all(axis=1)
    return call.index[ok]


# ────────────────────────────────────────── metrics ─────────────────────────────────────
def _top(y: np.ndarray, p: np.ndarray, frac: float) -> tuple[float, float, float]:
    k = max(1, int(math.ceil(frac * len(y))))
    top = y[np.argsort(-p, kind="mergesort")[:k]]
    base, tot = y.mean(), y.sum()
    return float(top.mean()), float(top.sum() / tot) if tot else np.nan, \
        float(top.mean() / base) if base else np.nan


def disc_metrics(y: np.ndarray, p: np.ndarray, probability: bool = True) -> dict:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    out = {"n": len(y), "base_rate": float(y.mean())}
    if len(np.unique(y)) < 2:
        return out
    out["roc_auc"] = float(roc_auc_score(y, p))
    out["pr_auc"] = float(average_precision_score(y, p))
    if probability:
        q = np.clip(p, 1e-6, 1 - 1e-6)
        out["brier"] = float(np.mean((q - y) ** 2))
        out["log_loss"] = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
        out["mean_p"] = float(q.mean())
    for f in (0.10, 0.20):
        h, c, l = _top(y, p, f)
        k = int(f * 100)
        out[f"top{k}_hit"], out[f"top{k}_capture"], out[f"top{k}_lift"] = h, c, l
    return out


def wilson(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def tier_table(y: np.ndarray, tier: np.ndarray, window: np.ndarray) -> list[dict]:
    """Per tier: size, hit rate [Wilson], capture, crude and window-stratified lift."""
    y = np.asarray(y, float)
    tier = np.asarray(tier, dtype=object)
    window = np.asarray(window, dtype=object)
    base = y.mean()
    wbase = {w: y[window == w].mean() for w in np.unique(window)}
    expected = np.array([wbase[w] for w in window])
    groups = {t: tier == t for t in TIERS}
    groups["Flagged"] = np.isin(tier, ["Elevated", "High Alert"])
    rows = []
    for name, m in groups.items():
        n, k = int(m.sum()), float(y[m].sum())
        lo, hi = wilson(k, n)
        rows.append({
            "tier": name, "n": n, "share": n / len(y) if len(y) else np.nan,
            "hit_rate": k / n if n else np.nan, "ci_lo": lo, "ci_hi": hi,
            "capture": k / y.sum() if y.sum() else np.nan,
            "lift": (k / n) / base if n and base else np.nan,
            "stratified_lift": (k / n) / expected[m].mean() if n else np.nan,
        })
    return rows


def calibration_rows(y: np.ndarray, p: np.ndarray, window: np.ndarray) -> list[dict]:
    rows = []
    for w in (BMO, AMC, "ALL"):
        m = np.ones(len(y), bool) if w == "ALL" else (window == w)
        if m.sum() == 0:
            continue
        yy, pp = y[m].astype(int), np.clip(p[m], 1e-6, 1 - 1e-6)
        z = np.log(pp / (1 - pp))
        from sklearn.linear_model import LogisticRegression
        fit = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), yy)
        rows.append({
            "window": w, "n": int(m.sum()), "observed": float(yy.mean()),
            "mean_p": float(pp.mean()), "gap": float(pp.mean() - yy.mean()),
            "brier": float(np.mean((pp - yy) ** 2)),
            "log_loss": float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))),
            "cal_slope": float(fit.coef_[0][0]), "cal_intercept": float(fit.intercept_[0]),
        })
    return rows


# ───────────────────────────────── stock-clustered bootstrap ────────────────────────────
def cluster_indices(stocks: np.ndarray, reps: int, seed: int):
    groups = pd.Series(np.arange(len(stocks))).groupby(stocks).indices
    keys = list(groups)
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        pick = rng.integers(0, len(keys), size=len(keys))
        yield np.concatenate([groups[keys[i]] for i in pick])


def bootstrap(stocks: np.ndarray, stat, *, reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> dict:
    """`stat(idx) -> {metric: delta}`. Returns point, 95% CI and share > 0 per metric."""
    point = stat(np.arange(len(stocks)))
    draws = {k: [] for k in point}
    for idx in cluster_indices(stocks, reps, seed):
        try:
            d = stat(idx)
        except ValueError:
            continue
        for k, v in d.items():
            draws[k].append(v)
    out = {}
    for k, v in point.items():
        a = np.asarray(draws[k], float)
        a = a[np.isfinite(a)]
        out[k] = v
        out[f"{k}_lo"] = float(np.quantile(a, 0.025)) if len(a) else np.nan
        out[f"{k}_hi"] = float(np.quantile(a, 0.975)) if len(a) else np.nan
        out[f"{k}_share_pos"] = float((a > 0).mean()) if len(a) else np.nan
    return out


def prob_delta_stat(y, pa, pb):
    def stat(idx):
        a, b = disc_metrics(y[idx], pa[idx]), disc_metrics(y[idx], pb[idx])
        keys = ["roc_auc", "pr_auc", "brier", "log_loss", "top10_hit", "top10_capture",
                "top20_hit", "top20_capture"]
        return {f"d_{k}": a[k] - b[k] for k in keys if k in a and k in b}
    return stat


def tier_delta_stat(y, ta, tb):
    def summary(t, yy):
        flag = np.isin(t, ["Elevated", "High Alert"])
        ha = t == "High Alert"
        return {"flag_capture": yy[flag].sum() / yy.sum(), "flag_hit": yy[flag].mean(),
                "ha_hit": yy[ha].mean(), "ha_capture": yy[ha].sum() / yy.sum(),
                "flag_share": flag.mean()}

    def stat(idx):
        a, b = summary(ta[idx], y[idx]), summary(tb[idx], y[idx])
        return {f"d_{k}": a[k] - b[k] for k in a}
    return stat


# ─────────────────────────────────────── driver pieces ──────────────────────────────────
def run_walk_forward(call: pd.DataFrame, eve: pd.DataFrame) -> tuple[dict, dict]:
    """OOF predictions for every (candidate, window mode) at the call cutoff, plus the
    event-eve versions of C and D (W0) for the prediction-time comparison."""
    preds, folds = {}, {}
    for cand in CANDIDATES:
        for wm in cal.WINDOWS:
            preds[(cand, wm, "call")], folds[(cand, wm, "call")] = cal.walk_forward(
                call, cand, wm, years=TEST_YEARS)
    for cand in ("B_structural", "C_structural_vol", "D_phase5b_pair"):
        preds[(cand, "W0", "eve")], folds[(cand, "W0", "eve")] = cal.walk_forward(
            eve, cand, "W0", years=TEST_YEARS)
    return preds, folds


def metric_tables(preds: dict, call: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    pooled, yearly, windowed, tiers, calib = [], [], [], [], []
    for (cand, wm, cut), pr in preds.items():
        d = call.loc[pr.index]
        y, p, w = d["y_extreme"].to_numpy(int), pr["p"].to_numpy(float), \
            d["announce_window"].to_numpy(object)
        key = {"candidate": cand, "window_mode": wm, "cutoff": cut}
        pooled.append({**key, **disc_metrics(y, p)})
        for yr in sorted(pr["test_year"].unique()):
            m = pr["test_year"].to_numpy() == yr
            yearly.append({**key, "year": int(yr), **disc_metrics(y[m], p[m])})
        for win in (BMO, AMC):
            m = w == win
            windowed.append({**key, "window": win, **disc_metrics(y[m], p[m])})
        for c in calibration_rows(y, p, w):
            calib.append({**key, **c})
        for rule in cal.TIER_RULES:
            t = pr[f"tier_{rule}"].to_numpy(object)
            for scope, m in (("ALL", np.ones(len(y), bool)), (BMO, w == BMO), (AMC, w == AMC)):
                for row in tier_table(y[m], t[m], w[m]):
                    tiers.append({**key, "rule": rule, "scope": scope, **row})
    return tuple(pd.DataFrame(x) for x in (pooled, yearly, windowed, tiers, calib))


def native_v031_tiers(frame: pd.DataFrame) -> pd.DataFrame:
    """A1's own tiers (73/79 + promotion on corrected, endpoint-safe lift) on every event,
    for each prior strength, plus A0's shipped tiers."""
    f = add_model_inputs(frame, "call")
    structural = v031_structural_tier(f["a1_score"])
    out = pd.DataFrame(index=frame.index)
    out["A1_structural"] = structural
    for k in LIFT_STRENGTHS:
        lift = stock_bucket_lift(frame, frame, frame, structural, frame["market_prior_call"],
                                 strength=k)
        out[f"A1_lift_{k}"] = lift
        out[f"A1_native_{k}"] = promote(structural, lift)
    out["A0_native"] = frame["earnings_explosiveness_bucket"].astype(object)
    out["A0_structural"] = frame["earnings_explosiveness_bucket_structural"].astype(object)
    return out


def select_e(boots: dict) -> dict:
    """The pre-registered selection rules, applied mechanically."""
    trail = []
    score = "B_structural"
    cb = boots["C-B call W0"]
    c_ok = cb["d_roc_auc_lo"] > 0 and cb["d_top10_capture_hi"] >= 0 and cb["d_top20_capture_hi"] >= 0
    trail.append(f"C over B: AUC CI lo {cb['d_roc_auc_lo']:+.4f}, capture CI hi "
                 f"{cb['d_top10_capture_hi']:+.4f}/{cb['d_top20_capture_hi']:+.4f} -> {c_ok}")
    if c_ok:
        score = "C_structural_vol"
        dc = boots["D-C call W0"]
        d_ok = dc["d_top10_capture_lo"] > 0 or dc["d_top20_capture_lo"] > 0
        trail.append(f"D over C: capture CI lo {dc['d_top10_capture_lo']:+.4f}/"
                     f"{dc['d_top20_capture_lo']:+.4f} -> {d_ok}")
        if d_ok:
            score = "D_phase5b_pair"
    w1, w2 = boots[f"W1-W0 {score}"], boots[f"W2-W0 {score}"]
    w1_ok, w2_ok = w1["d_log_loss_hi"] < 0, w2["d_log_loss_hi"] < 0
    trail.append(f"W1 log loss CI hi {w1['d_log_loss_hi']:+.5f} -> {w1_ok}; "
                 f"W2 {w2['d_log_loss_hi']:+.5f} -> {w2_ok}")
    window = "W0"
    if w1_ok and w2_ok:
        overlap = not (w1["d_log_loss_hi"] < w2["d_log_loss_lo"]
                       or w2["d_log_loss_hi"] < w1["d_log_loss_lo"])
        window = "W2" if overlap or w2["d_log_loss"] <= w1["d_log_loss"] else "W1"
    elif w1_ok or w2_ok:
        window = "W1" if w1_ok else "W2"
    return {"score": score, "window": window, "trail": trail}


def choose_tier_rule(tiers: pd.DataFrame, score: str, window: str) -> tuple[str, str]:
    t = tiers[(tiers.candidate == score) & (tiers.window_mode == window)
              & (tiers.cutoff == "call") & (tiers.tier == "Flagged")].set_index(["rule", "scope"])
    bmo_gain = t.loc[("Q_by_window", BMO), "capture"] - t.loc[("Q_common", BMO), "capture"]
    pooled_loss = t.loc[("Q_common", "ALL"), "capture"] - t.loc[("Q_by_window", "ALL"), "capture"]
    ok = bmo_gain > 0 and pooled_loss <= 0.01
    note = (f"Q_by_window BMO flagged capture {bmo_gain:+.4f}, pooled flagged capture "
            f"{-pooled_loss:+.4f} -> {'Q_by_window' if ok else 'Q_common'}")
    return ("Q_by_window" if ok else "Q_common"), note


# ───────────────────────────────── lift promotion and HC on E ───────────────────────────
def lift_promotion_test(frame: pd.DataFrame, e_pred: pd.DataFrame, e_folds: dict,
                        window: str, rule: str, oof: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Promote E's tiers with 0.3.1's gates; the stock's prior events are tiered by the
    same frozen fold model, and the lift prior is endpoint-safe."""
    full = add_model_inputs(frame, "call")
    rows, promoted_tiers = [], {}
    for k in LIFT_STRENGTHS:
        promoted = pd.Series(np.nan, index=e_pred.index, dtype=object)
        for year, fold in e_folds.items():
            bucket = cal.tiers_for(fold, full, window, rule)
            lift = stock_bucket_lift(frame, frame, frame, bucket, frame["market_prior_call"],
                                     strength=k)
            test_idx = e_pred.index[e_pred["test_year"].eq(year)]
            promoted.loc[test_idx] = promote(e_pred.loc[test_idx, f"tier_{rule}"],
                                             lift.loc[test_idx]).to_numpy()
        promoted_tiers[k] = promoted
        base = e_pred[f"tier_{rule}"]
        y = oof.loc[e_pred.index, "y_extreme"].to_numpy(float)
        for origin, dest in (("Normal", "Elevated"), ("Normal", "High Alert"),
                             ("Elevated", "High Alert")):
            m = (base.eq(origin) & promoted.eq(dest)).to_numpy()
            n = int(m.sum())
            rows.append({
                "prior_strength": k, "origin": origin, "destination": dest, "n_promoted": n,
                "promoted_hit": y[m].mean() if n else np.nan,
                "origin_hit_unpromoted": y[(base.eq(origin) & promoted.eq(origin)).to_numpy()].mean(),
                "destination_hit_native": y[(base.eq(dest)).to_numpy()].mean(),
            })
    return pd.DataFrame(rows), promoted_tiers


def drift_z_call(frame: pd.DataFrame) -> pd.Series:
    """drift_30d at the call cutoff vs the stock's own prior events' call-cutoff drift."""
    d = frame[~frame["is_pending"].astype(bool)].sort_values(
        ["stock", "call_cutoff_idx"], kind="mergesort")
    g = d.groupby("stock", sort=False)["drift_30d_call"]
    mean = g.transform(lambda s: s.shift(1).expanding().mean())
    std = g.transform(lambda s: s.shift(1).expanding(min_periods=5).std())
    return ((d["drift_30d_call"] - mean) / std.replace(0, np.nan)).reindex(frame.index)


# ─────────────────────────────── production-availability check ──────────────────────────
def yfinance_history(frame: pd.DataFrame) -> pd.DataFrame:
    """History features if only yfinance-timed events (2020+) had corrected outcomes."""
    ts = pd.read_parquet(PROVIDER_TS, columns=["stock", "earnings_date"])
    ts["earnings_date"] = pd.to_datetime(ts["earnings_date"]).dt.normalize()
    keys = set(zip(ts["stock"].astype(str), ts["earnings_date"]))
    has = pd.Series([(s, pd.Timestamp(d).normalize()) in keys for s, d in
                     zip(frame["stock"].astype(str), frame["earnings_date"])], index=frame.index)
    outcomes = frame[["abs_r3", "abs_r1", "y_extreme"]].copy()
    outcomes.loc[~has, ["abs_r3", "abs_r1", "y_extreme"]] = np.nan
    hist = history_features(frame, frame, outcomes, "call")
    hist["has_yf_timestamp"] = has
    return hist


# ───────────────────────────────────────────── run ──────────────────────────────────────
def run(rebuild: bool = False, reps: int = BOOT_REPS) -> dict:
    frame = load_frame(rebuild)
    pop = population(frame)
    call_all, eve_all = add_model_inputs(pop, "call"), add_model_inputs(pop, "eve")
    idx = common_rows(call_all, eve_all)
    call, eve = call_all.loc[idx], eve_all.loc[idx]
    RESULTS.mkdir(parents=True, exist_ok=True)

    coverage = {
        "population_events": int(len(pop)),
        "common_events": int(len(idx)),
        "common_stocks": int(call["stock"].nunique()),
        "natural_coverage": {c: float(call_all[[x for x in MODEL_INPUTS[c] if x != "log_peer_sec_rw"]]
                                      .notna().all(axis=1).mean()) for c in CANDIDATES},
        "peer_missing_share_call": float(call["log_peer_sec_rw"].isna().mean()),
        "events_by_year": call.groupby("year").size().to_dict(),
    }

    preds, folds = run_walk_forward(call, eve)
    pooled, yearly, windowed, tiers, calib = metric_tables(preds, call)
    oof_index = preds[("B_structural", "W0", "call")].index
    for k, pr in preds.items():
        assert pr.index.equals(oof_index), f"{k} scored a different row set"
    oof = call.loc[oof_index]
    y = oof["y_extreme"].to_numpy(int)
    stocks = oof["stock"].to_numpy(object)
    P = {k: v["p"].to_numpy(float) for k, v in preds.items()}

    boots = {}

    def boot(name, a, b):
        boots[name] = {"comparison": name, **bootstrap(stocks, prob_delta_stat(y, a, b), reps=reps)}

    boot("C-B call W0", P[("C_structural_vol", "W0", "call")], P[("B_structural", "W0", "call")])
    boot("D-C call W0", P[("D_phase5b_pair", "W0", "call")], P[("C_structural_vol", "W0", "call")])
    boot("C call-eve", P[("C_structural_vol", "W0", "call")], P[("C_structural_vol", "W0", "eve")])
    boot("D call-eve", P[("D_phase5b_pair", "W0", "call")], P[("D_phase5b_pair", "W0", "eve")])
    boot("D-C eve W0", P[("D_phase5b_pair", "W0", "eve")], P[("C_structural_vol", "W0", "eve")])
    for s in ("B_structural", "C_structural_vol", "D_phase5b_pair"):
        for wm in ("W1", "W2"):
            boot(f"{wm}-W0 {s}", P[(s, wm, "call")], P[(s, "W0", "call")])

    sel = select_e(boots)
    e_score, e_window = sel["score"], sel["window"]
    e_rule, rule_note = choose_tier_rule(tiers, e_score, e_window)
    sel["trail"].append(rule_note)
    e_key = (e_score, e_window, "call")
    e_pred = preds[e_key]

    boot("E-A0", P[e_key], P[("A0_v031_shipped", "W0", "call")])
    boot("E-A1", P[e_key], P[("A1_v031_corrected", "W0", "call")])
    boot("E-B", P[e_key], P[("B_structural", "W0", "call")])

    # ── mechanics of the old score, on the OOF rows ──
    mech_rows = []
    for name in [*MECHANICS, "log_hist_mean_abs"]:
        s = oof[name].to_numpy(float)
        mech_rows.append({"score": name, **disc_metrics(y, s, probability=False)})
    a1 = oof["a1_score"].to_numpy(float)
    at_ceiling = a1 >= 100 - 1e-9
    top5 = int(math.ceil(0.05 * len(a1)))
    mechanics = {
        "table": mech_rows,
        "share_at_score_100": float(at_ceiling.mean()),
        "events_at_score_100": int(at_ceiling.sum()),
        "top5pct_size": top5,
        "entropy_saturated_share": float((oof["hist_entropy_call"] >= 1).mean()),
        "entropy_missing_share": float(oof["hist_entropy_call"].isna().mean()),
        "spearman_a1_vs_p75": float(pd.Series(a1).corr(oof["p75_uncapped"].reset_index(drop=True),
                                                        method="spearman")),
        "hit_rate_at_ceiling_split_by_median_p75": None,
    }
    if at_ceiling.sum() > 20:
        ce = oof[at_ceiling]
        hi = ce["p75_uncapped"] >= ce["p75_uncapped"].median()
        mechanics["hit_rate_at_ceiling_split_by_median_p75"] = {
            "upper_half": float(ce.loc[hi, "y_extreme"].mean()),
            "lower_half": float(ce.loc[~hi, "y_extreme"].mean()),
            "n_each": int(hi.sum())}

    def score_delta(a, b):
        def stat(i):
            return {"d_roc_auc": roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]),
                    "d_top10_capture": _top(y[i], a[i], .1)[1] - _top(y[i], b[i], .1)[1],
                    "d_top20_capture": _top(y[i], a[i], .2)[1] - _top(y[i], b[i], .2)[1]}
        return stat
    for name, a, b in (("nocap-a1", "a1_nocap", "a1_score"),
                       ("a1-noentropy", "a1_score", "a1_noentropy"),
                       ("hist_mean-a1", "log_hist_mean_abs", "a1_score")):
        boots[f"mechanics {name}"] = {"comparison": f"mechanics {name}", **bootstrap(
            stocks, score_delta(oof[a].to_numpy(float), oof[b].to_numpy(float)), reps=reps)}

    # ── native 0.3.1 tiers and head-to-head tier comparison ──
    native = native_v031_tiers(frame)
    nat = native.loc[oof_index]
    e_tiers = e_pred[f"tier_{e_rule}"].to_numpy(object)
    w_oof = oof["announce_window"].to_numpy(object)
    native_rows = []
    for name in ("A0_native", "A0_structural", "A1_structural", "A1_native_20"):
        t = nat[name].to_numpy(object)
        for scope, m in (("ALL", np.ones(len(y), bool)), (BMO, w_oof == BMO), (AMC, w_oof == AMC)):
            for row in tier_table(y[m], t[m], w_oof[m]):
                native_rows.append({"tiering": name, "scope": scope, **row})
    for scope, m in (("ALL", np.ones(len(y), bool)), (BMO, w_oof == BMO), (AMC, w_oof == AMC)):
        for row in tier_table(y[m], e_tiers[m], w_oof[m]):
            native_rows.append({"tiering": f"E {e_score}/{e_window}/{e_rule}", "scope": scope, **row})
    native_df = pd.DataFrame(native_rows)
    yf = y.astype(float)
    for ref in ("A0_native", "A1_native_20"):
        boots[f"E tiers-{ref}"] = {"comparison": f"E tiers-{ref}", **bootstrap(
            stocks, tier_delta_stat(yf, e_tiers, nat[ref].to_numpy(object)), reps=reps)}
    # lift promotion inside 0.3.1 itself: does it help A1's structural tiers?
    boots["A1 promoted-structural"] = {"comparison": "A1 promoted-structural", **bootstrap(
        stocks, tier_delta_stat(yf, nat["A1_native_20"].to_numpy(object),
                                nat["A1_structural"].to_numpy(object)), reps=reps)}

    # ── tier yearly stability for E ──
    e_yearly = []
    for yr in TEST_YEARS:
        m = e_pred["test_year"].to_numpy() == yr
        for row in tier_table(y[m], e_tiers[m], w_oof[m]):
            e_yearly.append({"year": yr, **row})
        for row in tier_table(y[m], nat["A0_native"].to_numpy(object)[m], w_oof[m]):
            e_yearly.append({"year": yr, "reference": "A0_native", **row})
    e_yearly = pd.DataFrame(e_yearly)

    # ── volume curve: precision/capture at several flag rates (E vs A0 vs A1) ──
    curve = []
    for name, s in (("E", P[e_key]), ("A0", P[("A0_v031_shipped", "W0", "call")]),
                    ("A1", P[("A1_v031_corrected", "W0", "call")]),
                    ("B", P[("B_structural", "W0", "call")])):
        for f in (0.05, 0.10, 0.15, 0.20, 0.30):
            for scope, m in (("ALL", np.ones(len(y), bool)), (BMO, w_oof == BMO), (AMC, w_oof == AMC)):
                h, c, l = _top(y[m], s[m], f)
                curve.append({"model": name, "flag_rate": f, "scope": scope,
                              "hit": h, "capture": c, "lift": l})
    curve = pd.DataFrame(curve)

    # ── lift promotion on E ──
    lift_rows, promoted = lift_promotion_test(frame, e_pred, folds[e_key], e_window, e_rule, oof)
    for k, t in promoted.items():
        boots[f"E promoted{k}-E"] = {"comparison": f"E promoted{k}-E", **bootstrap(
            stocks, tier_delta_stat(yf, t.to_numpy(object), e_tiers), reps=reps)}

    # ── High Conviction on E ──
    z = drift_z_call(frame).loc[oof_index].to_numpy(float)
    ha = e_tiers == "High Alert"
    hc = ha & (np.abs(z) >= HC_Z)
    hc_info = {"high_alert_n": int(ha.sum()), "hc_n": int(hc.sum()),
               "hc_hit": float(yf[hc].mean()) if hc.any() else np.nan,
               "ha_not_hc_hit": float(yf[ha & ~hc].mean()),
               "hc_wilson": wilson(yf[hc].sum(), hc.sum()),
               "ha_not_hc_wilson": wilson(yf[ha & ~hc].sum(), (ha & ~hc).sum())}

    def hc_stat(i):
        h, a = hc[i], ha[i] & ~hc[i]
        return {"d_hit": yf[i][h].mean() - yf[i][a].mean()}
    hc_info["bootstrap"] = bootstrap(stocks, hc_stat, reps=reps)
    # 0.3.1's own HC as shipped, for reference, on the same rows
    ev_hc = pd.read_parquet(PHASE3_EVENTS, columns=["is_high_conviction"]).loc[oof_index]
    shipped_hc = ev_hc["is_high_conviction"].fillna(False).to_numpy(bool)
    shipped_ha = nat["A0_native"].to_numpy(object) == "High Alert"
    hc_info["shipped_hc_n"] = int(shipped_hc.sum())
    hc_info["shipped_hc_hit"] = float(yf[shipped_hc].mean()) if shipped_hc.any() else np.nan
    hc_info["shipped_ha_not_hc_hit"] = float(yf[shipped_ha & ~shipped_hc].mean())
    hc_info["hc_by_year"] = {int(yr): {"n": int((hc & (e_pred.test_year.to_numpy() == yr)).sum()),
                                       "hc_hit": float(yf[hc & (e_pred.test_year.to_numpy() == yr)].mean())
                                       if (hc & (e_pred.test_year.to_numpy() == yr)).any() else None,
                                       "ha_not_hc_hit": float(yf[ha & ~hc & (e_pred.test_year.to_numpy() == yr)].mean())}
                             for yr in TEST_YEARS}

    # ── 2026 holdout: scored once, after selection ──
    hold = {}
    hold_rows = []
    for cand in CANDIDATES:
        for wm in sorted({"W0", e_window}):
            pr, fd = cal.walk_forward(call, cand, wm, years=range(HOLDOUT_YEAR, HOLDOUT_YEAR + 1))
            hold[(cand, wm)] = (pr, fd)
            d = call.loc[pr.index]
            hold_rows.append({"candidate": cand, "window_mode": wm,
                              **disc_metrics(d["y_extreme"].to_numpy(int), pr["p"].to_numpy(float))})
            for win in (BMO, AMC):
                m = d["announce_window"].eq(win).to_numpy()
                hold_rows.append({"candidate": cand, "window_mode": wm, "window": win,
                                  **disc_metrics(d["y_extreme"].to_numpy(int)[m],
                                                 pr["p"].to_numpy(float)[m])})
    hold_df = pd.DataFrame(hold_rows)
    he, _ = hold[(e_score, e_window)]
    hd = call.loc[he.index]
    hold_tiers = []
    ht = he[f"tier_{e_rule}"].to_numpy(object)
    hw = hd["announce_window"].to_numpy(object)
    hy = hd["y_extreme"].to_numpy(float)
    for scope, m in (("ALL", np.ones(len(hy), bool)), (BMO, hw == BMO), (AMC, hw == AMC)):
        for row in tier_table(hy[m], ht[m], hw[m]):
            hold_tiers.append({"tiering": "E", "scope": scope, **row})
        a0 = native.loc[he.index, "A0_native"].to_numpy(object)
        for row in tier_table(hy[m], a0[m], hw[m]):
            hold_tiers.append({"tiering": "A0_native", "scope": scope, **row})
    hold_tiers = pd.DataFrame(hold_tiers)
    hold_calib = pd.DataFrame(calibration_rows(hy.astype(int), he["p"].to_numpy(float), hw))

    # ── production-availability sensitivity ──
    yfh = yfinance_history(frame)
    prod_rows = []
    for yr, (pr, fd) in [*((y_, (preds[e_key], folds[e_key][y_])) for y_ in (2023, 2024, 2025)),
                         (HOLDOUT_YEAR, (he, hold[(e_score, e_window)][1][HOLDOUT_YEAR]))]:
        sub_idx = pr.index[pr["test_year"].eq(yr)]
        base = call.loc[sub_idx]
        alt = base.copy()
        alt["hist_mean_abs_call"] = yfh.loc[sub_idx, "hist_mean_abs_call"]
        alt = add_model_inputs(alt, "call")
        n_yf = yfh.loc[sub_idx, "n_prior_call"]
        for min_n in (8, 4):
            ok = n_yf.ge(min_n).to_numpy()
            yy = base["y_extreme"].to_numpy(int)[ok]
            p_full = pr.loc[sub_idx, "p"].to_numpy(float)[ok]
            p_yf = cal.probability(fd, alt[ok], e_window)
            prod_rows.append({"year": yr, "min_prior": min_n,
                              "coverage": float(ok.mean()), "n": int(ok.sum()),
                              "auc_full_history": disc_metrics(yy, p_full).get("roc_auc"),
                              "auc_yfinance_history": disc_metrics(yy, p_yf).get("roc_auc"),
                              "top20_capture_full": disc_metrics(yy, p_full).get("top20_capture"),
                              "top20_capture_yf": disc_metrics(yy, p_yf).get("top20_capture")})
    prod_df = pd.DataFrame(prod_rows)

    # ── write ──
    boots_df = pd.DataFrame(boots.values())
    for name, df in (("pooled", pooled), ("yearly", yearly), ("by_window", windowed),
                     ("tiers", tiers), ("calibration", calib), ("native_tiers", native_df),
                     ("e_tier_yearly", e_yearly), ("volume_curve", curve),
                     ("lift_promotion", lift_rows), ("holdout_2026", hold_df),
                     ("holdout_2026_tiers", hold_tiers), ("holdout_2026_calibration", hold_calib),
                     ("production_availability", prod_df), ("bootstrap", boots_df),
                     ("mechanics", pd.DataFrame(mech_rows))):
        df.to_csv(RESULTS / f"{name}.csv", index=False)
    coefs = {}
    for yr, fd in {**folds[e_key], HOLDOUT_YEAR: hold[(e_score, e_window)][1][HOLDOUT_YEAR]}.items():
        coefs[int(yr)] = {"inputs": fd.inputs, "coef_std": fd.model.coef_[0].tolist(),
                          "intercept": float(fd.model.intercept_[0]),
                          "means": fd.means.to_dict(), "stds": fd.stds.to_dict(),
                          "base_rate": fd.base_rate, "train_n": fd.train_n,
                          "cuts": {r: {k: list(map(float, v)) for k, v in c.items()}
                                   for r, c in fd.cuts.items()}}
    oof_out = oof[["stock", "earnings_date", "report_date", "year", "announce_window",
                   "y_extreme"]].copy()
    for (cand, wm, cut), pr in preds.items():
        oof_out[f"p__{cand}__{wm}__{cut}"] = pr["p"]
    oof_out["E_tier"] = e_tiers
    oof_out["A0_native"] = nat["A0_native"].to_numpy(object)
    oof_out["A1_native_20"] = nat["A1_native_20"].to_numpy(object)
    oof_out.to_parquet(RESULTS / "oof_predictions.parquet")

    summary = {
        "phase": "P3.2+P3.3 refit",
        "coverage": coverage,
        "oof_events": int(len(oof)),
        "oof_years": list(TEST_YEARS),
        "selection": {**sel, "tier_rule": e_rule},
        "E": {"score": e_score, "window": e_window, "tier_rule": e_rule,
              "fold_coefficients": coefs},
        "mechanics": mechanics,
        "high_conviction": hc_info,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rebuild", action="store_true", help="recompute the feature frame")
    ap.add_argument("--reps", type=int, default=BOOT_REPS)
    args = ap.parse_args(argv)
    s = run(rebuild=args.rebuild, reps=args.reps)
    print(json.dumps({k: s[k] for k in ("coverage", "oof_events", "selection")}, indent=2,
                     default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
