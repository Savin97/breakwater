# Not a testing suite — a maintainer metrics script for Model B (0.4.0).
#
#     .venv/bin/python -m testing.metric_testing                   # every section
#     .venv/bin/python -m testing.metric_testing --section profile # one section
#     .venv/bin/python -m testing.metric_testing --feature vol_30d # feature vs target
#
# INTERNAL ONLY — never publish these numbers. They are RETRO-SCORED history: each event's
# Model B call computed as of its own cutoff from prior outcomes only (the raw signal is
# causal), but nothing here was published at the time, and the percentile scale behind the
# tiers was built from the 2014-2025 events these tables cover. They describe how the
# fixed model behaves; they are not a track record (that is the predictions archive).
#
# Unit: completed events in output/events_df.parquet whose corrected 3-session outcome is
# available. Target: abs_reaction_3d_anchored >= 0.08. Events without a verified
# announcement time have no outcome and are excluded — never counted as misses. Unscored
# events (< 8 usable priors) are reported as a count and left out of ranking metrics.
# Before-open and after-close reporters differ (base rates, tier mix), so every pooled
# table is followed by its BMO / AMC split; never quote a pooled ratio without it.
import argparse
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import EXTREME_EARNINGS_REACTION_THRESHOLD
from feature_engineering.announcement_timing import TARGET_AVAILABLE
from scoring.model_b import HIGH_ALERT, ELEVATED, NORMAL, UNSCORED

warnings.filterwarnings("ignore")

EVENTS_PARQUET = "output/events_df.parquet"
SCORED_TIERS = [HIGH_ALERT, ELEVATED, NORMAL]
FROM_YEAR, TO_YEAR = 2017, 2025          # the refit's out-of-sample years
SHRINK_PRIOR = 20                         # events of prior weight in the tier calibration


# ── data ──────────────────────────────────────────────────────────────────────

def load_events(path=EVENTS_PARQUET, from_year=FROM_YEAR, to_year=TO_YEAR) -> pd.DataFrame:
    """Completed events with an available corrected outcome, in [from_year, to_year]."""
    ev = pd.read_parquet(path)
    ev = ev[~ev["is_pending"].astype(bool)
            & ev["reaction_3d_anchored_status"].eq(TARGET_AVAILABLE)
            & ev["abs_reaction_3d_anchored"].notna()].copy()
    ev["year"] = pd.to_datetime(ev["earnings_date"]).dt.year
    ev = ev[ev["year"].between(from_year, to_year)]
    ev["move"] = ev["abs_reaction_3d_anchored"]
    ev["y"] = (ev["move"] >= EXTREME_EARNINGS_REACTION_THRESHOLD).astype(int)
    ed = pd.to_datetime(ev["earnings_date"])
    ev["call_week"] = ed - pd.to_timedelta(ed.dt.weekday, unit="D")
    return ev


def scored(ev):
    return ev[ev["risk_tier"] != UNSCORED]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _by_window(ev, fn, title):
    print(f"\n  {title} — by announcement window")
    for w, g in ev.groupby("announce_window"):
        print(f"  [{w}]")
        fn(g, indent="    ")


# ── 1. tier profile ───────────────────────────────────────────────────────────

def tier_profile(ev, indent="  "):
    """Share of events, hit rate (≥8%, Wilson 95% CI), mean |move| and capture per tier."""
    total_hits = ev["y"].sum()
    print(f"{indent}{'tier':<11} {'n':>6} {'share':>6} {'hit rate [95% CI]':>24} "
          f"{'mean|move|':>10} {'capture':>8}")
    for t in SCORED_TIERS + [UNSCORED]:
        g = ev[ev["risk_tier"] == t]
        if g.empty:
            continue
        k, n = int(g["y"].sum()), len(g)
        lo, hi = wilson(k, n)
        print(f"{indent}{t:<11} {n:>6} {n / len(ev):>6.1%} {k / n:>8.3f} [{lo:.3f}, {hi:.3f}] "
              f"{g['move'].mean():>10.3f} {k / total_hits if total_hits else np.nan:>8.3f}")
    ha_el = ev[ev["risk_tier"].isin([HIGH_ALERT, ELEVATED])]
    print(f"{indent}{'all':<11} {len(ev):>6} {'':>6} {ev['y'].mean():>8.3f} {'':>15} "
          f"{ev['move'].mean():>10.3f}   (HA+Elevated capture "
          f"{ha_el['y'].sum() / total_hits if total_hits else np.nan:.3f})")


# ── 2. year-by-year tier stability ────────────────────────────────────────────

def tier_stability_by_year(ev, indent="  "):
    """Per year: base rate, share and hit rate of each tier, High Alert capture."""
    print(f"{indent}{'year':>5} {'n':>5} {'base':>6} | {'HA share':>8} {'HA hit':>7} "
          f"{'HA cap':>7} | {'El share':>8} {'El hit':>7} | {'No hit':>7}")
    for y, g in ev.groupby("year"):
        s = scored(g)
        ha, el, no = (s[s["risk_tier"] == t] for t in SCORED_TIERS)
        cap = ha["y"].sum() / s["y"].sum() if s["y"].sum() else np.nan
        print(f"{indent}{y:>5} {len(s):>5} {s['y'].mean():>6.3f} | {len(ha) / len(s):>8.1%} "
              f"{ha['y'].mean():>7.3f} {cap:>7.3f} | {len(el) / len(s):>8.1%} "
              f"{el['y'].mean():>7.3f} | {no['y'].mean():>7.3f}")


# ── 3. walk-forward tier-rate calibration ─────────────────────────────────────

def walkforward_tier_calibration(ev, all_events=None, indent="  "):
    """Is a tier's hit rate stable enough to quote? For each year, the rate expected from
    PRIOR years (shrunk toward their base rate by SHRINK_PRIOR events) vs the realized
    rate. ECE = event-weighted mean absolute gap, in percentage points. The tiers are
    fixed percentile bands — this measures drift, it never moves a cut."""
    hist = scored(all_events if all_events is not None else ev)
    rows = []
    for y in sorted(ev["year"].unique()):
        train, test = hist[hist["year"] < y], scored(ev[ev["year"] == y])
        if len(train) < 500 or test.empty:
            continue
        base = train["y"].mean()
        for t in SCORED_TIERS:
            tr, te = train[train["risk_tier"] == t], test[test["risk_tier"] == t]
            if te.empty:
                continue
            pred = (tr["y"].sum() + SHRINK_PRIOR * base) / (len(tr) + SHRINK_PRIOR)
            rows.append({"year": y, "tier": t, "predicted": pred, "actual": te["y"].mean(),
                         "n": len(te)})
    if not rows:
        print(f"{indent}(not enough prior history)")
        return
    cal = pd.DataFrame(rows)
    cal["err"] = (cal["predicted"] - cal["actual"]).abs()
    agg = cal.groupby("tier").apply(lambda g: pd.Series({
        "predicted": g["predicted"].mean(), "actual": g["actual"].mean(),
        "mean_abs_err": g["err"].mean(), "n": g["n"].sum()}), include_groups=False)
    print(agg.reindex(SCORED_TIERS).round(3).to_string().replace("\n", "\n" + indent))
    ece = (cal["err"] * cal["n"]).sum() / cal["n"].sum() * 100
    print(f"{indent}ECE: {ece:.2f} pp")


# ── 4. rank correlation by year ───────────────────────────────────────────────

def rank_correlation_by_year(ev, indent="  "):
    """Spearman and Pearson of the raw signal vs |move|, per year."""
    rows = []
    for y, g in scored(ev).groupby("year"):
        rows.append((y, g["model_b_hist_mean_abs"].corr(g["move"], method="spearman"),
                     g["model_b_hist_mean_abs"].corr(g["move"])))
    for y, sp, pe in rows:
        print(f"{indent}{y}: spearman {sp:.3f}  pearson {pe:.3f}")
    sp = [r[1] for r in rows]
    print(f"{indent}mean spearman {np.mean(sp):.3f} | min {np.min(sp):.3f} | "
          f"all positive: {all(x > 0 for x in sp)}")


# ── 5. between- vs within-stock decomposition ─────────────────────────────────

def correlation_decomposition(ev, indent="  ", seed=42):
    """Model B is a persistence signal by construction (a stock's own past moves). How
    much of its correlation with |move| is 'volatile stocks stay volatile' (between
    stocks) versus 'this quarter differs from this stock's usual' (within a stock)?"""
    s = scored(ev).copy()
    x, r = "model_b_hist_mean_abs", "move"
    pooled = s[x].corr(s[r], method="spearman")
    means = s.groupby("stock")[[x, r]].mean()
    between = means[x].corr(means[r], method="spearman")
    within = (s[x] - s.groupby("stock")[x].transform("mean")).corr(
        s[r] - s.groupby("stock")[r].transform("mean"), method="spearman")
    rng = np.random.default_rng(seed)
    shuffled = s.groupby("stock")[r].transform(lambda v: rng.permutation(v.to_numpy()))
    shuf = s[x].corr(shuffled, method="spearman")
    print(f"{indent}pooled spearman            {pooled:.3f}")
    print(f"{indent}between-stock (means)      {between:.3f}")
    print(f"{indent}within-stock (demeaned)    {within:.3f}")
    print(f"{indent}shuffle moves within stock {shuf:.3f}   (what survives with timing destroyed)")
    print(f"{indent}pooled - shuffle           {pooled - shuf:.3f}")


# ── 6. fixed decile calibration ───────────────────────────────────────────────

def decile_calibration(ev, indent="  "):
    """Realized rate per Breakwater Risk Percentile decile (fixed bands of the frozen
    reference — no edges are fitted here). Should rise monotonically."""
    s = scored(ev).copy()
    s["decile"] = np.minimum(s["risk_score"] // 10, 9).astype(int) + 1
    tbl = s.groupby("decile").agg(n=("y", "size"), hit_rate=("y", "mean"),
                                  mean_move=("move", "mean"))
    tbl["share"] = tbl["n"] / len(s)
    print(tbl.round(3).to_string().replace("\n", "\n" + indent))
    print(f"{indent}monotonic hit rate: {tbl['hit_rate'].is_monotonic_increasing}")


# ── 7. ranking metrics ────────────────────────────────────────────────────────

def _top(y, score, frac):
    k = max(1, math.ceil(frac * len(y)))
    top = y[np.argsort(-score, kind="mergesort")[:k]]
    return top.mean(), top.sum() / y.sum() if y.sum() else np.nan


def ranking_metrics(ev, indent="  "):
    """ROC-AUC, PR-AUC, pooled top-10/20% and within-call-week top-10/20% — the weekly list
    is what the product publishes. Ranked on the raw signal, as the product orders."""
    s = scored(ev)
    y, x = s["y"].to_numpy(), s["model_b_hist_mean_abs"].to_numpy()
    if len(np.unique(y)) < 2:
        print(f"{indent}(one class only)")
        return
    print(f"{indent}n {len(s)}  base rate {y.mean():.3f}  ROC-AUC {roc_auc_score(y, x):.3f}  "
          f"PR-AUC {average_precision_score(y, x):.3f}")
    for f in (0.10, 0.20):
        hit, cap = _top(y, x, f)
        picked = []
        for _, g in s.groupby("call_week"):
            k = max(1, math.ceil(f * len(g)))
            picked.append(g.nlargest(k, "model_b_hist_mean_abs"))
        wk = pd.concat(picked)
        print(f"{indent}top {int(f * 100)}%: pooled hit {hit:.3f} capture {cap:.3f} | "
              f"within-week hit {wk['y'].mean():.3f} capture {wk['y'].sum() / s['y'].sum():.3f}")


# ── 8. any feature vs the target ──────────────────────────────────────────────

def feature_vs_target(ev, feature, indent="  ", bins=10):
    """Decile table of any event-frame column against the corrected target, overall and
    inside each Model B tier (does it add anything the tier does not already say?)."""
    if feature not in ev.columns:
        print(f"{indent}{feature!r} is not an event-frame column")
        return
    s = scored(ev).dropna(subset=[feature]).copy()
    s["bin"] = pd.qcut(s[feature].rank(method="first"), bins, labels=False) + 1
    print(f"{indent}{feature}: n {len(s)}  spearman vs |move| "
          f"{s[feature].corr(s['move'], method='spearman'):.3f}  "
          f"vs Model B signal {s[feature].corr(s['model_b_hist_mean_abs'], method='spearman'):.3f}")
    tbl = s.groupby("bin").agg(n=("y", "size"), mean_value=(feature, "mean"),
                               hit_rate=("y", "mean"))
    print(tbl.round(4).to_string().replace("\n", "\n" + indent))
    print(f"{indent}within tier (spearman {feature} vs |move|):")
    for t in SCORED_TIERS:
        g = s[s["risk_tier"] == t]
        if len(g) > 30:
            print(f"{indent}  {t:<11} n {len(g):>5}  {g[feature].corr(g['move'], method='spearman'):.3f}")


# ── driver ────────────────────────────────────────────────────────────────────

SECTIONS = {
    "profile":     ("TIER PROFILE", tier_profile, True),
    "stability":   ("YEAR-BY-YEAR TIER STABILITY", tier_stability_by_year, False),
    "calibration": ("WALK-FORWARD TIER-RATE CALIBRATION", None, False),
    "rankcorr":    ("RANK CORRELATION BY YEAR", rank_correlation_by_year, False),
    "decomp":      ("BETWEEN- vs WITHIN-STOCK DECOMPOSITION", correlation_decomposition, True),
    "deciles":     ("PERCENTILE-DECILE CALIBRATION", decile_calibration, True),
    "ranking":     ("RANKING METRICS", ranking_metrics, True),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Model B maintainer metrics (internal only).")
    ap.add_argument("--section", choices=list(SECTIONS), default=None)
    ap.add_argument("--from-year", type=int, default=FROM_YEAR)
    ap.add_argument("--to-year", type=int, default=TO_YEAR)
    ap.add_argument("--feature", default=None, help="also run feature_vs_target on this column")
    a = ap.parse_args(argv)

    ev = load_events(from_year=a.from_year, to_year=a.to_year)
    all_ev = load_events(from_year=2000, to_year=a.to_year)
    n_uns = int((ev["risk_tier"] == UNSCORED).sum())
    print(f"Model B metrics — completed events {a.from_year}-{a.to_year} with a corrected "
          f"outcome: {len(ev)} ({n_uns} Unscored). INTERNAL, retro-scored, not a track record.")
    for key, (title, fn, split) in SECTIONS.items():
        if a.section and key != a.section:
            continue
        print(f"\n=== {title} ===")
        if key == "calibration":
            walkforward_tier_calibration(ev, all_ev)
            continue
        fn(ev)
        if split:
            _by_window(ev, fn, title.lower())
    if a.feature:
        print(f"\n=== FEATURE vs TARGET: {a.feature} ===")
        feature_vs_target(ev, a.feature)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
