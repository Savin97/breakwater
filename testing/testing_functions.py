# testing/testing_functions.py
"""
    Evaluation helpers for Breakwater — Model B (0.4.0) and any candidate feature.

    Every function takes the EVENT FRAME (output/events_df.parquet; the pipeline's
    events) and first reduces it with `prepare_events`: completed events whose corrected
    3-session outcome is available, with
        y        abs_reaction_3d_anchored >= 8%   (the Model B validation target)
        y_large  abs_reaction_3d_anchored >= 5%
        move     abs_reaction_3d_anchored
        year     report year
    Events without a verified announcement time have no corrected outcome and are dropped,
    never counted as misses. The legacy report-date reaction is never used.

    Defaults: score = `model_b_hist_mean_abs` (the raw Model B signal; `risk_score` ranks
    identically), dates = `earnings_date`. Anything else on the event frame can be passed
    in its place — e.g. `vol_30d`, `expected_move_pct` — which is how a candidate feature
    is tested against or on top of Model B.

    INTERNAL ONLY. Results on history are retro-scored (each event's Model B call computed
    from prior outcomes only) and are not a track record. When quoting a ratio, split it by
    `announce_window` (BMO vs AMC differ in base rate and tier mix).

    ---------------------------------------------------------------------------
    FEATURE DIAGNOSTICS
    ---------------------------------------------------------------------------
    check_explosiveness_feature(df, score)      quintile hit rates of a score
    check_feature_connection_to_large_reacion_metric(df, feature)
                                                decile table, ≥5% / ≥8% rates, Wilson CIs
    check_corr_of_features(df, cols)            Spearman matrix between signals
    check_score_metric(df, score, plots=False)  distribution, dispersion, quintile means

    ---------------------------------------------------------------------------
    REGIME TESTING (conditions are (column, quantile) pairs: column >= its quantile)
    ---------------------------------------------------------------------------
    three_way_regime_test(df, score, cond_a, cond_b)
    conditional_hit_rate_analysis(df, score, conditions)
    breakwater_regime_test(df, score, conditions)
    volatility_only_regime_test(df, vol_col, score)   competitive baseline vs Model B
    evaluate_high_risk_earnings_regime(df, score, other)  quantile robustness grid
    high_conviction_regime_test(df, overlays)         tier + overlay condition

    ---------------------------------------------------------------------------
    CLASSIFICATION METRICS
    ---------------------------------------------------------------------------
    testing_precision_recall(df, regime)        confusion matrix, precision, recall
    comparing_regime_results_to_volatility_only(df, feat1, feat2)
                                                signal persistence, shuffled-label null,
                                                first/second-half stability, top-vol baseline

    ---------------------------------------------------------------------------
    ROBUSTNESS & OUT-OF-SAMPLE
    ---------------------------------------------------------------------------
    check_feature_train_test(df, feature)       frozen train edges applied to test years
    yearly_oos_report(df, ...)                  walk-forward yearly corr and separation
    forward_eval_onefactor / forward_eval_twofactor
                                                train-quantile regimes evaluated per year
"""
import math

import numpy as np
import pandas as pd

from config import EXTREME_EARNINGS_REACTION_THRESHOLD
from feature_engineering.announcement_timing import TARGET_AVAILABLE

SCORE = "model_b_hist_mean_abs"
TARGET = "abs_reaction_3d_anchored"
LARGE_MOVE = 0.05
TRAIN_YEARS = range(2013, 2017)       # corrected timing is sparse before 2013
TEST_YEARS = range(2017, 2026)        # the refit's out-of-sample years


# ------------------------------------------------------
# Shared preparation
# ------------------------------------------------------

def prepare_events(df):
    """Completed events with an available corrected outcome, plus y / y_large / move /
    year. Accepts the event frame or an already-prepared frame."""
    if {"y", "y_large", "move", "year"} <= set(df.columns):
        return df
    ev = df
    if "is_pending" in ev.columns:
        ev = ev[~ev["is_pending"].astype(bool)]
    ev = ev[ev["reaction_3d_anchored_status"].eq(TARGET_AVAILABLE) & ev[TARGET].notna()].copy()
    ev["move"] = ev[TARGET]
    ev["y"] = (ev["move"] >= EXTREME_EARNINGS_REACTION_THRESHOLD).astype(int)
    ev["y_large"] = (ev["move"] >= LARGE_MOVE).astype(int)
    ev["year"] = pd.to_datetime(ev["earnings_date"]).dt.year
    return ev


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)) / denom
    return (center - half, center + half)


def _mask(ev, conditions):
    """AND of `column >= quantile(q)` over (column, q) pairs; q may be negative to mean
    `column <= quantile(-q)` (the low tail)."""
    m = pd.Series(True, index=ev.index)
    for col, q in conditions:
        m &= (ev[col] <= ev[col].quantile(-q)) if q < 0 else (ev[col] >= ev[col].quantile(q))
    return m


def _rates(sub, label):
    return {"group": label, "n": len(sub), "p_large_plus": sub["y_large"].mean(),
            "p_extreme": sub["y"].mean()}


# ------------------------------------------------------
# Feature diagnostics
# ------------------------------------------------------

def check_explosiveness_feature(df, score=SCORE):
    """Quintile buckets of `score`: extreme-move rate per bucket, top vs bottom quintile.
    For Model B the signal IS a stock's earnings explosiveness (its mean past corrected
    move), so this is the first thing to look at."""
    ev = prepare_events(df).dropna(subset=[score])
    print("Extreme move rate:", round(ev["y"].mean(), 4))
    ev["bucket"] = pd.qcut(ev[score], 5, duplicates="drop")
    print(ev.groupby("bucket", observed=True)["y"].agg(["mean", "size"]))
    top = ev[ev[score] >= ev[score].quantile(0.8)]
    bottom = ev[ev[score] <= ev[score].quantile(0.2)]
    print("Top quintile extreme rate:", round(top["y"].mean(), 4))
    print("Bottom quintile extreme rate:", round(bottom["y"].mean(), 4))
    return ev.groupby("bucket", observed=True)["y"].mean()


def check_feature_connection_to_large_reacion_metric(df, feature=SCORE):
    """Decile table of `feature`: ≥5% and ≥8% rates, ratio to the overall rate, Wilson
    95% CI on the ≥5% rate. Rates should rise from decile 1 to 10."""
    ev = prepare_events(df).dropna(subset=[feature])
    ev["decile"] = pd.qcut(ev[feature].rank(method="first"), q=10, labels=False) + 1
    table = (ev.groupby("decile")
               .agg(n=(feature, "size"), avg_value=(feature, "mean"),
                    p_large_plus=("y_large", "mean"), p_extreme=("y", "mean"))
               .reset_index())
    table["ratio_large_plus"] = table["p_large_plus"] / ev["y_large"].mean()
    table["ratio_extreme"] = table["p_extreme"] / ev["y"].mean()
    ci = [wilson_ci(p, int(n)) for p, n in zip(table["p_large_plus"], table["n"])]
    table["p_large_plus_ci_lo"] = [c[0] for c in ci]
    table["p_large_plus_ci_hi"] = [c[1] for c in ci]
    print(table.round(4).to_string(index=False))
    print("----------------------")
    print(ev[feature].describe())
    return table


def check_corr_of_features(df, cols=(SCORE, "vol_30d", "vol_10d", "expected_move_pct")):
    """Spearman correlation between signals (and with |move|). High correlation with the
    Model B signal means a candidate adds little on its own."""
    ev = prepare_events(df)
    usable = [c for c in cols if c in ev.columns and ev[c].notna().sum() >= 30]
    skipped = [c for c in cols if c not in usable]
    if skipped:
        # e.g. expected_move_pct: options snapshots only exist from 2026-09.
        print(f"skipped (absent or < 30 values): {skipped}")
    corr = ev[usable + ["move"]].corr(method="spearman")
    print(corr.round(3))
    return corr


def check_score_metric(df, score=SCORE, plots=False):
    """Distribution of a score rescaled 0-100, its per-report-date dispersion, its
    correlation with |move| and mean |move| by quintile. `plots=True` also draws the
    four diagnostic charts (blocks until closed)."""
    ev = prepare_events(df).dropna(subset=[score]).copy()
    lo, hi = ev[score].min(), ev[score].max()
    ev["scaled"] = 100 * (ev[score] - lo) / (hi - lo) if hi > lo else 50.0
    s = ev["scaled"]
    stats = {"count": int(s.count()), "min": s.min(), "max": s.max(), "mean": s.mean(),
             "median": s.median(), "std": s.std(),
             **{f"p{int(q * 100)}": s.quantile(q) for q in (0.10, 0.25, 0.75, 0.90, 0.95)}}
    print({k: round(float(v), 3) for k, v in stats.items()})
    counts, bins = np.histogram(s, bins=30)
    for i in range(len(counts)):
        print(f"{bins[i]:.1f} to {bins[i + 1]:.1f} : {counts[i]}")
    per_day = ev.groupby("earnings_date")["scaled"].agg(["mean", "std", "count"])
    print("per-report-date std (median):", round(float(per_day["std"].median()), 3))
    print("spearman vs |move|:", round(ev["scaled"].corr(ev["move"], method="spearman"), 3))
    ev["bucket"] = pd.qcut(ev["scaled"].rank(method="first"), q=5, labels=False)
    bucket_stats = ev.groupby("bucket")["move"].mean()
    print(bucket_stats.round(4))
    if plots:
        import matplotlib.pyplot as plt
        for title, draw in [
            (f"{score} distribution (0-100)", lambda: plt.hist(s, bins=30)),
            (f"{score} vs |move|", lambda: plt.scatter(ev["scaled"], ev["move"], alpha=0.25)),
            (f"mean |move| by {score} quintile", lambda: bucket_stats.plot(kind="bar")),
            (f"per-report-date dispersion of {score}",
             lambda: plt.hist(per_day["std"].dropna(), bins=30)),
        ]:
            plt.figure(figsize=(8, 5))
            draw()
            plt.title(title)
            plt.show()
    return stats


# ------------------------------------------------------
# Regime testing
# ------------------------------------------------------

def three_way_regime_test(df, score=SCORE, cond_a=("vol_30d", 0.5),
                          cond_b=("vol_ratio_10_to_30", 0.5), q=0.9):
    """Is tail risk materially higher when the score is in its top (1-q) AND two other
    conditions hold? Compares the joint regime with the baseline and with the score
    alone. Conditions are (column, quantile)."""
    ev = prepare_events(df).dropna(subset=[score, cond_a[0], cond_b[0]])
    if ev.empty:
        print(f"no events carry {score}, {cond_a[0]} and {cond_b[0]} together")
        return pd.DataFrame()
    alone = _mask(ev, [(score, q)])
    joint = _mask(ev, [(score, q), cond_a, cond_b])
    out = pd.DataFrame([_rates(ev, "ALL"), _rates(ev[alone], f"{score} top"),
                        _rates(ev[joint], f"{score} top + {cond_a[0]} + {cond_b[0]}")])
    print(out.round(4).to_string(index=False))
    return out


def conditional_hit_rate_analysis(df, score=SCORE,
                                  conditions=(("vol_30d", 0.9), ("vol_ratio_10_to_30", 0.9),
                                              ("vol_30d", -0.5)), q=0.9):
    """In which regimes does a high score turn into realized large moves? The score's
    top decile alone, then combined with each condition in turn."""
    ev = prepare_events(df).dropna(subset=[score])
    top = _mask(ev, [(score, q)])
    rows = [_rates(ev, "ALL"), _rates(ev[top], f"{score} top")]
    for col, cq in conditions:
        sub = ev.dropna(subset=[col])
        m = _mask(sub, [(score, q)]) & _mask(sub, [(col, cq)])
        side = f"low {col}" if cq < 0 else f"high {col}"
        rows.append(_rates(sub[m], f"{score} top + {side} (q {abs(cq)})"))
    out = pd.DataFrame(rows)
    print(out.round(4).to_string(index=False))
    return out


def breakwater_regime_test(df, score=SCORE, conditions=(("vol_30d", 0.5),), q=0.9):
    """One regime: score top (1-q) AND every condition. Returns its size and rates."""
    ev = prepare_events(df).dropna(subset=[score] + [c for c, _ in conditions])
    sub = ev[_mask(ev, [(score, q), *conditions])]
    res = {"n_events": len(sub), "extreme_rate": sub["y"].mean(),
           "large_plus_rate": sub["y_large"].mean(), "baseline_extreme": ev["y"].mean()}
    print(res)
    return res


def volatility_only_regime_test(df, vol_col="vol_30d", score=SCORE, q=0.9):
    """Competitive baseline: can recent volatility alone find the large moves? The vol
    top decile vs the Model B top decile on the SAME events, plus their overlap.
    `vol_30d` on an event row is as of the session before the report date — slightly
    later than Model B's call cutoff, which favours the baseline."""
    ev = prepare_events(df).dropna(subset=[vol_col, score])
    v, b = _mask(ev, [(vol_col, q)]), _mask(ev, [(score, q)])
    rows = [_rates(ev, "ALL"), _rates(ev[v], f"{vol_col} top"), _rates(ev[b], f"{score} top"),
            _rates(ev[v & b], "both"), _rates(ev[v & ~b], f"{vol_col} top only"),
            _rates(ev[b & ~v], f"{score} top only")]
    out = pd.DataFrame(rows)
    out["capture_extreme"] = [ev.loc[m, "y"].sum() / ev["y"].sum() for m in
                              (pd.Series(True, index=ev.index), v, b, v & b, v & ~b, b & ~v)]
    print(out.round(4).to_string(index=False))
    return out


def evaluate_high_risk_earnings_regime(df, score=SCORE, other="vol_30d"):
    """A strict regime (score top 10% AND `other` top 20%) and its robustness across a
    grid of quantiles; prints regime size, rates, and how concentrated it is in stocks."""
    ev = prepare_events(df).dropna(subset=[score, other])
    cond = _mask(ev, [(score, 0.9), (other, 0.8)])
    print("Sample size:", int(cond.sum()))
    print("Large move rate:", round(ev.loc[cond, "y_large"].mean(), 4))
    print("Extreme move rate:", round(ev.loc[cond, "y"].mean(), 4))
    print("Baseline extreme:", round(ev["y"].mean(), 4), " Baseline large:", round(ev["y_large"].mean(), 4))
    print(ev.loc[cond, "move"].describe().round(4))
    print("events per stock in regime:")
    print(ev.loc[cond].groupby("stock").size().describe().round(2))
    grid = []
    for sq in (0.8, 0.9):
        for oq in (0.7, 0.8):
            c = _mask(ev, [(score, sq), (other, oq)])
            if c.sum() > 30:
                grid.append({"score_q": sq, "other_q": oq, "n": int(c.sum()),
                             "extreme_rate": ev.loc[c, "y"].mean()})
    grid = pd.DataFrame(grid)
    print(grid.round(4).to_string(index=False))
    return grid


def high_conviction_regime_test(df, overlays=None, tier="High Alert", test_years=TEST_YEARS):
    """Does an overlay condition on top of a Model B tier improve the hit rate, or is it
    noise? `overlays` maps a label to (column, quantile) conditions; per-year and pooled
    rate, ratio to base and capture for the tier alone and the tier + each overlay."""
    ev = prepare_events(df)
    ev = ev[ev["year"].isin(test_years)]
    in_tier = ev["risk_tier"] == tier
    regimes = {tier: in_tier}
    for label, conds in (overlays or {}).items():
        sub = ev.dropna(subset=[c for c, _ in conds])
        m = pd.Series(False, index=ev.index)
        m.loc[sub.index] = _mask(sub, conds)
        regimes[f"{tier} + {label}"] = in_tier & m
    rows = []
    for y, g in ev.groupby("year"):
        base, K = g["y"].mean(), g["y"].sum()
        for label, mask in regimes.items():
            sub = g[mask.loc[g.index]]
            if len(sub):
                rows.append({"regime": label, "year": int(y), "n": len(sub),
                             "extreme_rate": sub["y"].mean(),
                             "ratio": sub["y"].mean() / base if base else np.nan,
                             "capture": sub["y"].sum() / K if K else np.nan})
    result = pd.DataFrame(rows)
    base_rate = ev["y"].mean()
    summary = (result.groupby("regime", sort=False)
                     .apply(lambda g: pd.Series({
                         "total_n": g["n"].sum(),
                         "extreme_rate": (g["extreme_rate"] * g["n"]).sum() / g["n"].sum(),
                         "avg_capture": g["capture"].mean()}), include_groups=False))
    summary["ratio"] = summary["extreme_rate"] / base_rate
    print(summary.round(3).to_string())
    print(f"\nBaseline extreme rate: {base_rate:.3f}")
    return summary


# ------------------------------------------------------
# Classification metrics
# ------------------------------------------------------

def testing_precision_recall(df, regime=None):
    """Confusion matrix of a regime against the ≥8% target. `regime` is a boolean column
    name or Series aligned to the event frame; default: the High Alert tier."""
    ev = prepare_events(df)
    if regime is None:
        flag = ev["risk_tier"] == "High Alert"
    elif isinstance(regime, str):
        flag = ev[regime].astype(bool)
    else:
        flag = regime.reindex(ev.index).fillna(False).astype(bool)
    extreme = ev["y"] == 1
    TP, FN = int((flag & extreme).sum()), int((~flag & extreme).sum())
    FP, TN = int((flag & ~extreme).sum()), int((~flag & ~extreme).sum())
    res = {"TP": TP, "FN": FN, "FP": FP, "TN": TN,
           "precision": TP / (TP + FP) if TP + FP else np.nan,
           "recall": TP / (TP + FN) if TP + FN else np.nan,
           "flagged": int(flag.sum()), "events": len(ev)}
    print(res)
    return res


def comparing_regime_results_to_volatility_only(df, feat1=SCORE, feat2="vol_30d",
                                                q=0.9, vol_q=0.98, seed=0):
    """Four checks on a two-feature regime (feat1 AND feat2 in their top 1-q):
      1. how much feat1 moves between a stock's consecutive events (persistence);
      2. precision against SHUFFLED labels — the no-skill null;
      3. precision in the first vs second half of the years (stability);
      4. precision of the top (1-vol_q) of feat2 alone (the volatility baseline)."""
    ev = prepare_events(df).dropna(subset=[feat1, feat2]).sort_values(["stock", "earnings_date"])
    ev["feat1_diff"] = ev.groupby("stock")[feat1].diff()
    print("feat1 change between consecutive events:")
    print(ev["feat1_diff"].abs().describe().round(5))

    regime = _mask(ev, [(feat1, q), (feat2, q)])
    shuffled = np.random.default_rng(seed).permutation(ev["y"].to_numpy())
    prec = lambda m, y: (y[m].sum() / m.sum()) if m.sum() else np.nan
    res = {"regime_n": int(regime.sum()),
           "precision": prec(regime.to_numpy(), ev["y"].to_numpy()),
           "shuffled_precision": prec(regime.to_numpy(), shuffled)}
    mid = ev["year"].median()
    for half, sub in (("first_half", ev[ev["year"] <= mid]), ("second_half", ev[ev["year"] > mid])):
        m = _mask(sub, [(feat1, q), (feat2, q)])
        res[f"{half}_precision"] = prec(m.to_numpy(), sub["y"].to_numpy())
    top_vol = _mask(ev, [(feat2, vol_q)])
    res["top_feat2_n"] = int(top_vol.sum())
    res["top_feat2_precision"] = prec(top_vol.to_numpy(), ev["y"].to_numpy())
    res["baseline"] = ev["y"].mean()
    print({k: round(float(v), 4) for k, v in res.items()})
    return res


# ------------------------------------------------------
# Robustness & out-of-sample
# ------------------------------------------------------

def check_feature_train_test(df, feature=SCORE, split_year=2017):
    """Freeze min/max scaling and quintile edges on events before `split_year`, apply them
    to later events: does the ordering and the bucket separation hold out of sample?"""
    ev = prepare_events(df).dropna(subset=[feature])
    pre, post = ev[ev["year"] < split_year].copy(), ev[ev["year"] >= split_year].copy()
    lo, hi = pre[feature].min(), pre[feature].max()
    scale = (lambda s: 100 * (s - lo) / (hi - lo)) if hi > lo else (lambda s: s * 0 + 50)
    pre["score_oos"], post["score_oos"] = scale(pre[feature]), scale(post[feature]).clip(0, 100)
    _, edges = pd.qcut(pre["score_oos"], q=5, retbins=True, duplicates="drop")
    edges[0], edges[-1] = -np.inf, np.inf
    pre["bucket"] = pd.cut(pre["score_oos"], bins=edges, labels=False)
    post["bucket"] = pd.cut(post["score_oos"], bins=edges, labels=False)
    out = {"train_spearman": pre["score_oos"].corr(pre["move"], method="spearman"),
           "test_spearman": post["score_oos"].corr(post["move"], method="spearman")}
    print(f"Train (<{split_year}) spearman: {out['train_spearman']:.3f}   "
          f"Test (>={split_year}) spearman: {out['test_spearman']:.3f}")
    print(pd.DataFrame({"train_mean_move": pre.groupby("bucket")["move"].mean(),
                        "test_mean_move": post.groupby("bucket")["move"].mean(),
                        "test_hit_rate": post.groupby("bucket")["y"].mean()}).round(4))
    return out


def yearly_oos_report(df, date_col="earnings_date", score_feature=SCORE, target_col=TARGET, q=5):
    """Walk-forward by year: scaling and bucket edges fitted on all prior years, applied
    to the test year; per-year correlation and top-vs-bottom bucket separation."""
    d = prepare_events(df)[[date_col, score_feature, target_col]].dropna()
    d[date_col] = pd.to_datetime(d[date_col])
    rows = []
    for y in sorted(d[date_col].dt.year.unique())[1:]:
        train = d[d[date_col] < pd.Timestamp(f"{y}-01-01")]
        test = d[d[date_col].dt.year == y]
        if len(train) < 500 or len(test) < 100:
            continue
        lo, hi = train[score_feature].min(), train[score_feature].max()
        if hi == lo:
            continue
        tr = (100 * (train[score_feature] - lo) / (hi - lo)).clip(0, 100)
        te = (100 * (test[score_feature] - lo) / (hi - lo)).clip(0, 100)
        _, edges = pd.qcut(tr, q=q, retbins=True, duplicates="drop")
        edges[0], edges[-1] = -np.inf, np.inf
        ev = pd.DataFrame({"score": te, "bucket": pd.cut(te, bins=edges, labels=False),
                           "target": test[target_col].values}).dropna(subset=["bucket"])
        means = ev.groupby("bucket")["target"].mean()
        rows.append({"year": y, "n_test": len(ev),
                     "spearman": ev["score"].corr(ev["target"], method="spearman"),
                     "bot_bucket_mean": means.iloc[0], "top_bucket_mean": means.iloc[-1],
                     "separation": means.iloc[-1] - means.iloc[0],
                     "top_over_bottom": means.iloc[-1] / means.iloc[0] if means.iloc[0] else np.nan})
    yearly = pd.DataFrame(rows).sort_values("year")
    print(yearly.round(4).to_string(index=False))
    print("Avg spearman:", round(yearly["spearman"].mean(), 3),
          "| Median:", round(yearly["spearman"].median(), 3),
          "| Years > 0.15:", round((yearly["spearman"] > 0.15).mean(), 3))
    return yearly


def _forward_stats(sub, split, mask_fn):
    rows = []
    for y, g in sub.groupby("year"):
        r = g[mask_fn(g)]
        N, K, n, k = len(g), int(g["y"].sum()), len(r), int(r["y"].sum())
        base = K / N if N else np.nan
        reg = k / n if n else np.nan
        rows.append({"split": split, "year": int(y), "N_earnings": N,
                     "baseline_extreme_rate": base, "n_regime": n,
                     "regime_extreme_rate": reg,
                     "ratio": reg / base if base and np.isfinite(reg) else np.nan,
                     "regime_share_of_events": n / N if N else np.nan,
                     "regime_capture_of_extremes": k / K if K else np.nan})
    return pd.DataFrame(rows).sort_values("year")


def forward_eval_onefactor(df, eval_feature=SCORE, train_years=TRAIN_YEARS,
                           test_years=TEST_YEARS, q=0.90):
    """Threshold = the feature's q-quantile over TRAIN years only; per-year regime rate,
    ratio to base and capture in train and test years."""
    ev = prepare_events(df).dropna(subset=[eval_feature])
    train, test = ev[ev["year"].isin(train_years)], ev[ev["year"].isin(test_years)]
    thr = float(train[eval_feature].quantile(q))
    fn = lambda g: g[eval_feature] >= thr
    return pd.concat([_forward_stats(train, "TRAIN", fn), _forward_stats(test, "TEST", fn)],
                     ignore_index=True), thr


def forward_eval_twofactor(df, feat1=SCORE, feat2="vol_30d", train_years=TRAIN_YEARS,
                           test_years=TEST_YEARS, q=0.90):
    """Same as forward_eval_onefactor for the regime feat1 AND feat2 above their
    train-year q-quantiles — e.g. does a second feature sharpen Model B's top decile?"""
    ev = prepare_events(df).dropna(subset=[feat1, feat2])
    train, test = ev[ev["year"].isin(train_years)], ev[ev["year"].isin(test_years)]
    thr1, thr2 = float(train[feat1].quantile(q)), float(train[feat2].quantile(q))
    fn = lambda g: (g[feat1] >= thr1) & (g[feat2] >= thr2)
    return pd.concat([_forward_stats(train, "TRAIN", fn), _forward_stats(test, "TEST", fn)],
                     ignore_index=True), (thr1, thr2)
