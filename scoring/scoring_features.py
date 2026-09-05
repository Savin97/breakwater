# scoring/scoring_features.py
import numpy as np, pandas as pd
from config import (
    LARGE_EARNINGS_REACTION_THRESHOLD,
    EXTREME_EARNINGS_REACTION_THRESHOLD,
    BUCKET_ELEVATED_FLOOR,
    BUCKET_HIGH_ALERT_FLOOR,
    LIFT_PRIOR_STRENGTH,
    LIFT_TO_ELEVATED,
    LIFT_TO_HIGH_ALERT)
from feature_engineering.event_features import (
    event_earnings_explosiveness,
    event_explosiveness_score,
    event_stock_bucket_lift_values,
    event_lift_adjusted_bucket,
    event_high_conviction,
    surprise_momentum_flag_values,
    pre_earnings_drift_flag_values)

def engineer_large_reaction(input_df):
    """
        Adds a binary column 'is_large_reaction' indicating if the earnings move is large (≥ threshold).
        Threshold can be set based on historical distribution of abs_reaction_3d or business needs.
    """
    df = input_df
    df["is_large_reaction"] = (df["abs_reaction_3d"] >= LARGE_EARNINGS_REACTION_THRESHOLD).astype("int8")
    return df

def engineer_extreme_reaction(input_df):
    """
        Adds a binary column 'is_extreme' indicating if the earnings move is extreme (≥ threshold).
        Threshold can be set based on historical distribution of abs_reaction_3d or business needs.
    """
    df = input_df
    df["is_extreme_reaction"] = (df["abs_reaction_3d"] >= EXTREME_EARNINGS_REACTION_THRESHOLD).astype("int8")
    return df

def engineer_vol_stress(input_df, ratio_col: str = "vol_ratio_10_to_30"):
    """
        If vol_ratio_10_to_30 is high, recent vol spiked relative to the recent baseline -> "stress".
        Define "stress" as "top X%".
        Typical starting points:
        Top 20% = "elevated"
        Top 10% = "high"
        Top 5% = "extreme"

        Adds percentile-based volatility stress flags using cross-sectional distribution per date.
        Leakage-safe if ratio_col is already computed using shift(1) rolling stats.

        Output columns:
        - vol_ratio_cross_pct: cross-sectional percentile rank on that date (0..1)
        - vol_stress_high: top (1-q_extreme)
        - vol_stress_elevated: top (1-q_high)
    """
    df = input_df
    q_high = 0.80  # elevated
    q_extreme = 0.90 # high / extreme

    # Prevent div by zero
    replaced = df[ratio_col].replace( [np.inf, -np.inf], np.nan)
    df[ratio_col] = replaced

    # Cross-sectional percentile rank per day
    df["vol_ratio_cross_sectional_pct"] = (
        df.groupby("date")[ratio_col]
            .rank(pct=True, method="average")
    )

    df["vol_stress_elevated"] = (
        (df["vol_ratio_cross_sectional_pct"] >= q_high) & (df[ratio_col] >= 1.10)
    ).astype(np.int8)

    df["vol_stress_extreme"] = ( df["vol_ratio_cross_sectional_pct"] >= q_extreme ).astype("int8")
    
    return df


def engineer_momentum_pressure(input_df, quantile = 0.8) -> pd.DataFrame:
    """ 
        top 20% of absolute momentum values per date
        cross-sectional and date-aligned
        Momentum pressure measures how unusually stretched a stock's price action 
        is relative to peers ahead of earnings,
        capturing both short-term crowding and longer-term trend extension.

    Returns
    -------
        - 'momentum_pressure_regime' : str
          {'normal', 'short_term_extreme', 'trend_extreme', 'crowded_trend'}
    """
    df = input_df

    abs5 = df["mom_5d"].abs()
    abs20 = df["mom_20d"].abs()

    threshold_5 = abs5.groupby(df["date"]).transform(lambda x: x.quantile(quantile))
    threshold_20 = abs20.groupby(df["date"]).transform(lambda x: x.quantile(quantile))

    mom_5 = abs5 > threshold_5
    mom_20 = abs20 > threshold_20

    df["momentum_pressure_regime"] = np.select(
        [~mom_5 & ~mom_20,  mom_5 & ~mom_20,  ~mom_5 & mom_20,  mom_5 & mom_20],
        ["normal", "short_term_extreme", "trend_extreme", "crowded_trend"],
        default="normal"
    )
    regime_score_map = {
        "normal": 0.0,
        "short_term_extreme": 35.0,
        "trend_extreme": 55.0,
        "crowded_trend": 100.0
    }
    df["momentum_pressure_regime"] = df["momentum_pressure_regime"].map(regime_score_map).fillna(0.0)
    return df


def engineer_earnings_explosiveness(input_df, epsilon = 1e-6):
    """
        Adds:
        - earnings_explosiveness (raw)          = abs_reaction_median_3d
        - earnings_explosiveness_z (normalized) = abs_reaction_median_3d / max(vol_30d, epsilon)
        Median-based explosiveness = "typical risk"
        - earnings_tail_z (optional)            = abs_reaction_p75_3d / max(vol_30d, epsilon)  (if column exists)
        P75-based explosiveness = "tail danger" ("When earnings go bad, this is how ugly it can get.")
        
        Assumptions:
        - med_col / p75_col are already computed using ONLY past earnings events (shifted).
        - vol_30d is rolling vol from daily returns (ideally shifted by 1 day).
    """
    return event_earnings_explosiveness(input_df, epsilon)

def engineer_sector_vol_stress(input_df: pd.DataFrame, q_high = 0.9) -> pd.DataFrame:
    """
    """
    df = input_df
    df["sector_vol_ratio_pct"] = (
        df.groupby(["sector", "date"])["vol_ratio_10_to_30"]
        .rank(pct=True, method="average")
    )
    df["sector_vol_stress_high"] = (df["sector_vol_ratio_pct"] >= q_high).astype("int8")
    return df

def engineer_proximity_score(input_df):
    df = input_df
    df["proximity_score"] = score_proximity(df)
    return df

def engineer_vol_expansion_score(input_df):
    df = input_df
    df["vol_expansion_score"] = score_vol_expansion(df)
    return df

def engineer_momentum_fragility_score(input_df):
    df = input_df
    df["momentum_fragility_score"] = score_momentum_fragility(df)
    return df
    
def engineer_earnings_explosiveness_score(input_df):
    """
        When this stock moves on earnings, how violent can it get?
        Uses rolling p75 (28 events); falls back to expanding p75 for thin history.
        Vol-normalized components removed — grid search showed they add noise, not signal.
        Signal is driven by raw p75 magnitude and reaction entropy.

        Deprecated:
        e1 = (df["earnings_explosiveness_z"].fillna(0) / 7).clip(0, 1)  # expanding median z, 7sigma ceiling                                                                    
        e2 = (p75 / vol / 7).clip(0, 1)                                  # rolling tail z, 7sigma ceiling                                                                       
    """
    # Cut points 73/79 and the 0.85/0.15 weights are model calibration; see
    # feature_engineering.event_features.event_explosiveness_score.
    return event_explosiveness_score(input_df)


def engineer_stock_bucket_lift(input_df):
    """
    How much more often does THIS stock blow up on earnings than the market baseline,
    given the bucket it currently sits in?

        lift = P(extreme | stock, bucket) / P(extreme | market)

    Both terms are as-of each event: the numerator uses only that stock's PRIOR
    earnings in the same bucket, the denominator only prior earnings market-wide.
    Same shift(1)-before-expanding guard used throughout feature_engineering/ —
    without it every historical row would be scored using its own future outcomes,
    which would silently inflate every backtest in the repo.

    The per-stock rate is shrunk toward the market baseline by LIFT_PRIOR_STRENGTH,
    so a stock with two prior events cannot swing its own tier on noise. A stock's
    first event in a bucket has no prior data and lands at lift = 1.0 (no opinion).

    Computed on earnings rows only, matching earnings_explosiveness_score; downstream
    consumers pick it up through groupby().last() the same way.
    """
    df = input_df
    earnings_mask = df["is_earnings_day"] == 1

    lift = event_stock_bucket_lift_values(
        df.loc[earnings_mask, ["stock", "date", "is_extreme_reaction",
                               "earnings_explosiveness_bucket"]]
    )
    df["stock_bucket_lift"] = np.nan
    df.loc[lift.index, "stock_bucket_lift"] = lift
    return df


def engineer_lift_adjusted_bucket(input_df):
    """
    Promote a stock's tier when its own history says it is riskier than its structural
    score implies. Keeps the pre-adjustment tier as earnings_explosiveness_bucket_structural.

    This is a RECLASSIFICATION, not a rescaling. Multiplying the score by the lift was
    measured and rejected — it drags top-decile lift from 3.70x to 2.98x because it
    corrupts the ordering of a score the lift is already 0.79 rank-correlated with.
    Used as a conditional gate instead, the same signal is strongly additive.
    """
    return event_lift_adjusted_bucket(input_df)


def engineer_surprise_momentum_flag(input_df):
    """
    Categorical flag derived from surprise_streak, surprise_mean_5, surprise_std_5.
    Computed on earnings days; forward-filled within each stock so pre-earnings rows
    carry the most recent earnings event's flag state (streak doesn't change between events).

    Flags (evaluated in priority order):
      "Extended Beat Streak" — streak >= 6: bar maximally elevated after long beat streak
      "Beat Streak"   — streak >= 4 and mean_5 > 0.05: consistently beating big
      "Miss Streak"   — streak <= -3: consecutive misses, expectations reset lower
      "Erratic"       — std_5 > 0.20: highly unpredictable surprise magnitude
      ""              — everything else (normal)
    """
    df = input_df
    earnings_mask = df["is_earnings_day"] == 1

    df["surprise_momentum_flag"] = surprise_momentum_flag_values(
        df["surprise_streak"], df["surprise_mean_5"], df["surprise_std_5"], earnings_mask
    )

    # Propagate earnings-day flag to subsequent non-earnings rows.
    # Earnings-day rows act as anchors — even "" resets the carry-forward so a streak
    # that ends on earnings day N doesn't bleed into the following pre-earnings rows.
    propagated = df["surprise_momentum_flag"].copy()
    propagated.loc[~earnings_mask] = np.nan          # clear non-earnings rows (fill targets)
    propagated = propagated.groupby(df["stock"]).ffill().fillna("")
    df.loc[~earnings_mask, "surprise_momentum_flag"] = propagated.loc[~earnings_mask]
    return df

def engineer_pre_earnings_drift_flag(input_df):
    """
    Categorical flag from pre_earnings_drift_z. Computed on earnings days using the
    historical z-score. Also computed for pre-earnings window rows (1-60 days before
    earnings) from current drift_30d vs the stock's full historical earnings-day drift
    distribution — so the upcoming-events view gets a meaningful current-state flag.

    Flags:
      "Extended"   — drift_z >= 1.5: running into earnings hotter than historical norm
      "Compressed" — drift_z <= -1.5: sold off more than usual heading into earnings
      ""           — normal range or insufficient history (< 5 prior earnings)
    """
    df = input_df
    earnings_mask = df["is_earnings_day"] == 1
    flag = pre_earnings_drift_flag_values(df["pre_earnings_drift_z"], earnings_mask)

    # For pre-earnings window rows: recompute from current drift_30d vs historical
    # earnings-day distribution. Uses full history as baseline (correct for upcoming
    # events; historical pre-earnings rows aren't consumed by any downstream path).
    pre_mask = (df["is_earnings_day"] == 0) & df["days_to_earnings"].between(1, 60)
    if pre_mask.any():
        past_stats = (
            df.loc[earnings_mask, ["stock", "drift_30d"]]
            .dropna(subset=["drift_30d"])
            .groupby("stock")["drift_30d"]
            .agg(_mean="mean", _std="std")
        )
        pre_df = df.loc[pre_mask, ["stock", "drift_30d"]].join(past_stats, on="stock")
        has_hist = (
            pre_df["_mean"].notna() & pre_df["_std"].notna() &
            (pre_df["_std"] > 0) & pre_df["drift_30d"].notna()
        )
        z_pre = (pre_df["drift_30d"] - pre_df["_mean"]) / pre_df["_std"]
        flag.loc[pre_df.index[has_hist & (z_pre >= 1.5)]]  = "Extended"
        flag.loc[pre_df.index[has_hist & (z_pre <= -1.5)]] = "Compressed"

    df["pre_earnings_drift_flag"] = flag
    return df

def engineer_total_risk_score(input_df):
    df = input_df
    df["risk_score"] = df["earnings_explosiveness_score"]
    return df


def engineer_high_conviction(input_df):
    df = input_df
    # earnings_explosiveness_bucket is only populated on earnings-day rows, but HC is
    # meaningful on pre-earnings rows too — pre_earnings_drift_flag is deliberately
    # computed there so the upcoming-events view gets a current-state flag. Carry the
    # last completed event's bucket forward locally (the stored column is left as-is);
    # without this the bucket is NaN on non-earnings rows and HC silently evaluates
    # False regardless of the stock's actual tier. Requires df sorted by [stock, date].
    bucket = df.groupby("stock")["earnings_explosiveness_bucket"].ffill()
    return event_high_conviction(df, bucket)

def classify_large_relative_earnings_move_bucket(input_df):
    """
        large_earnings_move = 1 if abs_reaction_3d ≥ abs_reaction_p75_rolling
        window: 20-40 past earnings for that stock; 28
    """
    df = input_df
    # Only meaningful on earnings rows and when p75, p90 aren't NaN
    eligible = (
        df["is_earnings_day"]
        & df[["abs_reaction_p75_rolling", "abs_reaction_p90_rolling"]].notna().all(axis=1)
    )
    conditions = [
        eligible & (df["abs_reaction_3d"] <  df["abs_reaction_p75_rolling"]),
        eligible & (df["abs_reaction_3d"] >= df["abs_reaction_p75_rolling"])
                 & (df["abs_reaction_3d"] <  df["abs_reaction_p90_rolling"]),
        eligible & (df["abs_reaction_3d"] >= df["abs_reaction_p90_rolling"]),
    ]

    # 0 = normal    # 1 = large (p75-p90)    # 2 = extreme (p90+)
    # Unknown where insufficient history
    df["earnings_move_bucket"] = np.select(conditions, [0, 1, 2], default=np.nan) 

    return df


# -------------------------
# Composite Scoring
# -------------------------

def score_proximity(df, horizon=30, power=1.5):
    """
        Pre-earnings proximity only.
        - days_to_earnings >= horizon -> 0
        - days_to_earnings <= 1 -> near 100 (but never applied on earnings day)
    """
    days_to_earnings = df["days_to_earnings"]
    
    # Only pre-earnings days (strictly > 0)
    pre_earnings_days = days_to_earnings > 0

    base = np.zeros(len(df), dtype=float)

    # normalize 1...horizon -> 1...0 (close -> high)
    x = 1 - np.clip(days_to_earnings[pre_earnings_days] / horizon, 0, 1)
    x = x ** power

    # Optional small discrete boosts (still only pre-earnings)
    boost = np.zeros(x.shape[0], dtype=float)
    near = x > 0.6
    boost += ((near & df.loc[pre_earnings_days, "is_earnings_window"]).astype(float)) * 0.05
    boost += ((near & df.loc[pre_earnings_days, "is_earnings_week"]).astype(float))   * 0.08

    base[pre_earnings_days] = np.clip(x + boost, 0, 1)
    proximity_score = 100 * base
    # # Normalize signals
    # # days_to_earnings >= 30  -> 0   # days_to_earnings <= 0   -> 100
    # base = 1 - np.clip(df["days_to_earnings"] / 30 , 0, 1)

    # # Non-linear pressure near earnings
    # base = base ** 1.5

    # boost = np.zeros(len(df))
    # near = base > 0.6
    # boost += ((near & df["is_earnings_window"]).astype(float)) * 0.05
    # boost += ((near & df["is_earnings_week"]).astype(float))   * 0.08
    # boost += ((near & df["is_earnings_day"]).astype(float))    * 0.15

    # proximity_score = 100 * np.clip(base + boost, 0, 1)
    # return proximity_score
    return proximity_score    

def score_vol_expansion(df):
    """
        Vol Expansion Risk = "Is volatility already stretched before earnings?"        
        Range: 0-100
        Meaning:
        0-30 -> calm
        30-60 -> warming up
        60-80 -> unstable
        80-100 -> volatility already breaking
    """
    # Normalize signals
    z1 = df["vol_ratio_cross_sectional_pct"].fillna(0).clip(0, 1)
    z2 = ((df["stock_vs_sector_vol"].fillna(1) - 1) / 1.5).clip(0, 1)
    z3 = df["sector_vol_ratio_pct"].fillna(0).clip(0, 1)

    base = 0.40 * z1 + 0.35 * z2 + 0.25 * z3  # 0..1-ish

    elevated = df["vol_stress_elevated"].fillna(0).astype(float)
    extreme  = df["vol_stress_extreme"].fillna(0).astype(float)
    sector_h = df["sector_vol_stress_high"].fillna(0).astype(float)

    # multiplier (soft) + small additive sector bump
    multiplier = 1 + 0.25 * elevated + 0.50 * extreme
    additive  = 0.08 * sector_h

    vol_expansion = 100 * np.clip(base * multiplier + additive, 0, 1)
    # # snap the score upward when conditions are structurally dangerous
    # boost = 0.0
    # boost = (
    #     df["vol_stress_elevated"].fillna(0).astype(float) * 0.20 +
    #     df["vol_stress_extreme"].fillna(0).astype(float)  * 0.40 +
    #     df["sector_vol_stress_high"].fillna(0).astype(float) * 0.20
    # )

    # base = 0.4 * z1 + 0.35 * z2 + 0.25 * z3
    # # score = 100*clip(base*(1+0.5*extreme+0.25*elevated)+0.1*sector_high, 0, 1)
    # vol_expansion = 100 * np.clip(base + boost, 0, 1)
    # return vol_expansion
    return vol_expansion


def score_momentum_fragility(df):
    """
        Is price positioning fragile right now?
        High score = price is balanced on a knife-edge.
    """
    # Mapping momentum fragility strings to floats
    PRESSURE_MAP = {
        "normal": 0.0,
        "short_term_extreme": 0.6,
        "trend_extreme": 0.75,
        "crowded_trend": 1.0,
    }
    # Interpretation: 
    # pressure -> crowding / exhaustion
    # bias -> one-sided positioning
    # sector drift -> late-cycle momentum

    bias_scale = df["directional_bias"].abs().quantile(0.90)

    m1 = df["momentum_pressure_regime"].map(PRESSURE_MAP).fillna(0)
    # m2: this achieves: 90% of observations live in [0,1),Top 10% saturate at 1, 
    # No arbitrary magic number, Stable across stocks and time
    m2 = np.clip(np.abs(df["directional_bias"].fillna(0)) / bias_scale, 0, 1) 
    m3 = np.clip(np.abs(df["sector_drift_60d"].fillna(0)) / 0.10, 0, 1)
    
    #m2 = (df["directional_bias"].abs() / bias_scale).clip(0, 1)
    base = (
        0.45 * m1 +   # positioning pressure
        0.35 * m3 +   # sector trend maturity
        0.20 * m2     # directional skew
    )
    base.describe()
    momentum_fragility_score = 100 * np.clip(base, 0, 1)
    return momentum_fragility_score
