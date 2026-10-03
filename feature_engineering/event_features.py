# feature_engineering/event_features.py
"""
Event-level feature and scoring cores — the single implementation of every statistic
that aggregates ACROSS a stock's earnings events.

Why this module exists
----------------------
Before Phase 1 these statistics were written only onto earnings-day rows of the daily
price frame, leaving them NaN everywhere else. Consumers then recovered "the current
state" with `df.sort_values("date").groupby("stock").last()`, which skips NaN column by
column and therefore reached back to the stock's LAST COMPLETED EVENT. Every upcoming
event shipped a score, tier and lift that were exactly one earnings event stale
(audit/PHASE0_AUDIT_REV2.md §Q4).

The fix does not change any statistic. Each function below takes an EVENT-INDEXED frame
— one row per (stock, earnings event), ordered oldest-first within a stock — and is
called from two places:

  1. the daily pipeline, on the earnings-day rows it has always used
  2. pipeline/events.py, on the same rows PLUS one pending row per stock

Every `.shift(1)` guard is preserved verbatim. A pending row carries a NaN outcome and
sorts last within its stock, so:

  * it cannot alter any completed row (shift/rolling/expanding only look backwards), and
  * its own statistics naturally cover every completed prior event, including the most
    recent one — which is precisely the staleness fix.

Nothing here is allowed to depend on the composition of the frame it is handed
(no cross-sectional ranks, no global quantiles); such scores stay on the daily frame
and are carried onto the event row. See pipeline/events.py CARRIED_COLS.
"""
import numpy as np
import pandas as pd

from config import (
    DEFAULT_REACTION_WINDOW,
    BUCKET_ELEVATED_FLOOR,
    BUCKET_HIGH_ALERT_FLOOR,
    LIFT_PRIOR_STRENGTH,
    LIFT_TO_ELEVATED,
    LIFT_TO_HIGH_ALERT,
)

# ---------------------------------------------------------------------------
# Historical reaction distribution
# ---------------------------------------------------------------------------

def event_reaction_std(events):
    """Std of the stock's last 8 absolute reactions, prior events only."""
    events["reaction_std"] = (
        events.groupby("stock")[DEFAULT_REACTION_WINDOW]
        .transform(lambda x: x.abs().shift(1).rolling(window=8, min_periods=3).std(ddof=1))
    )
    return events


def reaction_entropy(series: pd.Series, bins=8) -> float:
    """Shannon entropy of the stock's past absolute reactions. ~0 = every quarter the
    same size; higher = less predictable. Unchanged from the original implementation."""
    series = series.dropna()

    if len(series) < bins:
        return np.nan

    hist, _ = np.histogram(series, bins)
    probs = hist / hist.sum()
    probs = probs[probs > 0]  # avoids log(0)

    return -np.sum(probs * np.log(probs))


def event_reaction_entropy(events, source_col="_best_reaction"):
    """Expanding entropy over prior events. `source_col` is reaction_3d backfilled with
    reaction_1d, matching the daily implementation."""
    events["reaction_entropy"] = (
        events.groupby("stock")[source_col]
        .transform(lambda x: x.abs().shift(1).expanding().apply(reaction_entropy))
    )
    return events


def event_directional_bias(events):
    """Expanding mean of prior SIGNED reactions — does this stock tend to gap up or down?"""
    events["directional_bias"] = (
        events.groupby("stock")[DEFAULT_REACTION_WINDOW]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    return events


def event_abs_reaction_median(events):
    events["abs_reaction_median"] = (
        events.groupby("stock")[DEFAULT_REACTION_WINDOW]
        .transform(lambda x: x.abs().shift(1).expanding().median())
    )
    return events


def event_abs_reaction_p75(events):
    events["abs_reaction_p75"] = (
        events.groupby("stock")[DEFAULT_REACTION_WINDOW]
        .transform(lambda x: x.abs().shift(1).expanding().quantile(0.75))
    )
    return events


def event_abs_reaction_p75_rolling(events, window=28, percentile=0.75):
    """Rolling p75 over the last `window` prior events. Requires a full window; stocks
    with thinner history get NaN and fall back to the expanding p75 in scoring."""
    events["abs_reaction_p75_rolling"] = (
        events.groupby("stock")["abs_reaction_3d"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=window).quantile(percentile))
    )
    return events


def event_abs_reaction_p90_rolling(events, window=28, percentile=0.9):
    events["abs_reaction_p90_rolling"] = (
        events.groupby("stock")["abs_reaction_3d"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=window).quantile(percentile))
    )
    return events


# ---------------------------------------------------------------------------
# EPS surprise momentum
# ---------------------------------------------------------------------------

def _surprise_streak(x):
    """Signed run length of prior beats/misses: +4 = four straight beats."""
    shifted = x.shift(1)
    beat = (shifted >= 0).astype(float)   # 1=beat, 0=miss
    beat[shifted.isna()] = np.nan
    direction = beat.where(beat == 1, -1)
    direction[beat.isna()] = np.nan
    run_id = direction.ne(direction.shift()).cumsum()
    count = direction.groupby(run_id).cumcount() + 1
    streak = count * direction
    streak[beat.isna()] = np.nan
    return streak


def event_surprise_features(events):
    """surprise_mean_5 / surprise_std_5 / surprise_streak, all from prior events."""
    grp = events.groupby("stock")["surprise_percentage"]
    events["surprise_mean_5"] = grp.transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
    events["surprise_std_5"] = grp.transform(lambda x: x.shift(1).rolling(5, min_periods=3).std())
    events["surprise_streak"] = grp.transform(_surprise_streak)
    return events


def event_pre_earnings_drift_z(events):
    """How unusual is the drift into THIS event versus the stock's own prior
    pre-earnings drift distribution."""
    grp = events.groupby("stock")["drift_30d"]
    baseline = grp.transform(lambda x: x.shift(1).expanding().mean())
    std = grp.transform(lambda x: x.shift(1).expanding(min_periods=5).std())
    events["pre_earnings_drift_z"] = (events["drift_30d"] - baseline) / std
    return events


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def event_earnings_explosiveness(events, epsilon=1e-6):
    """Vol-normalised views of the historical reaction distribution. Retained for
    analysis; neither term feeds earnings_explosiveness_score."""
    events["earnings_explosiveness_z"] = (
        events["abs_reaction_median"] / np.maximum(events["vol_30d"], epsilon)
    )
    events["earnings_tail_z"] = (
        events["abs_reaction_p75"] / np.maximum(events["vol_30d"], epsilon)
    )
    return events


def event_explosiveness_score(events):
    """Structural score and its pre-promotion tier.

    Weights (0.85/0.15), the 0.12 p75 ceiling and the 73/79 cut points are model
    calibration and are NOT touched in Phase 1. The ungrouped `.ffill()` on
    reaction_entropy is a known defect (audit rev-2 §1f) deliberately preserved here so
    this refactor changes nothing; it is Phase 2+ work.

    That ffill is order-dependent, and pending rows sit between one stock's last event
    and the next stock's first. Letting them CONTRIBUTE to the chain would change 385
    completed events' scores. A pending row therefore reads the chain but does not
    update it, which reproduces the daily frame's chain exactly (the daily frame has no
    pending rows) while still giving a pending event its own entropy when it has one.
    """
    p75 = events["abs_reaction_p75_rolling"].fillna(events["abs_reaction_p75"])
    e3 = (p75 / 0.12).clip(0, 1)           # raw magnitude: 12% ceiling
    entropy_col = events["reaction_entropy"]
    if "is_pending" in events.columns:
        chain = entropy_col.mask(events["is_pending"]).ffill()
        filled = entropy_col.fillna(chain)
    else:
        filled = entropy_col.ffill()
    e4 = np.clip(filled.fillna(0), 0, 1)
    events["earnings_explosiveness_score"] = 100 * np.clip(0.85 * e3 + 0.15 * e4, 0, 1)

    events["earnings_explosiveness_bucket"] = pd.cut(
        events["earnings_explosiveness_score"],
        bins=[-np.inf, BUCKET_ELEVATED_FLOOR, BUCKET_HIGH_ALERT_FLOOR, np.inf],
        labels=["Normal", "Elevated", "High Alert"],
    )
    return events


def event_stock_bucket_lift_values(events) -> pd.Series:
    """P(extreme | stock, bucket) / P(extreme | market), both as-of each event.

    Returns a Series indexed like `events` so either caller can assign it.

    Ordering: sorted by date, with pending rows last within a date when the column is
    present. A pending row contributes NaN to `is_extreme_reaction`, which
    expanding().mean() and expanding().sum() both skip, so it can never move the market
    baseline or another stock's numerator.
    """
    sort_cols = ["date", "is_pending"] if "is_pending" in events.columns else ["date"]
    ev = events.sort_values(sort_cols, kind="mergesort")

    # Market baseline as of each event — expanding over ALL prior events, every stock.
    global_prior = ev["is_extreme_reaction"].expanding().mean().shift(1)

    # This stock's prior record within the same bucket.
    grp = ev.groupby(["stock", "earnings_explosiveness_bucket"], observed=True)["is_extreme_reaction"]
    n_prior = grp.cumcount()
    sum_prior = grp.transform(lambda s: s.shift(1).expanding().sum())

    shrunk = (sum_prior + LIFT_PRIOR_STRENGTH * global_prior) / (n_prior + LIFT_PRIOR_STRENGTH)
    return (shrunk / global_prior).replace([np.inf, -np.inf], np.nan).fillna(1.0)


def event_lift_adjusted_bucket(events):
    """Promote a tier where the stock's own record beats its structural score.
    A RECLASSIFICATION, not a rescaling — see scoring/scoring_features.py."""
    bucket = events["earnings_explosiveness_bucket"].astype(object)
    lift = events["stock_bucket_lift"]

    events["earnings_explosiveness_bucket_structural"] = bucket

    to_high_alert = bucket.isin(["Normal", "Elevated"]) & (lift >= LIFT_TO_HIGH_ALERT)
    to_elevated = (bucket == "Normal") & (lift >= LIFT_TO_ELEVATED) & ~to_high_alert

    adjusted = bucket.copy()
    adjusted[to_elevated] = "Elevated"
    adjusted[to_high_alert] = "High Alert"

    events["earnings_explosiveness_bucket"] = pd.Categorical(
        adjusted, categories=["Normal", "Elevated", "High Alert"], ordered=True
    )
    return events


def surprise_momentum_flag_values(streak, mean5, std5, base_mask) -> pd.Series:
    """Categorical surprise-momentum flag. Evaluated in priority order (later wins)."""
    flag = pd.Series("", index=streak.index)
    flag.loc[base_mask & (std5 > 0.20)] = "Erratic"
    flag.loc[base_mask & (streak <= -3)] = "Miss Streak"
    flag.loc[base_mask & (streak >= 4) & (mean5 > 0.05)] = "Beat Streak"
    flag.loc[base_mask & (streak >= 6)] = "Extended Beat Streak"
    return flag


def pre_earnings_drift_flag_values(z, base_mask) -> pd.Series:
    """Categorical flag from the pre-earnings drift z-score."""
    flag = pd.Series("", index=z.index)
    flag.loc[base_mask & (z >= 1.5)] = "Extended"
    flag.loc[base_mask & (z <= -1.5)] = "Compressed"
    return flag


def event_high_conviction(events, bucket):
    """High Alert AND a non-empty pre-earnings drift flag."""
    events["is_high_conviction"] = (
        (bucket == "High Alert")
        & events["pre_earnings_drift_flag"].notna()
        & (events["pre_earnings_drift_flag"] != "")
    )
    return events
