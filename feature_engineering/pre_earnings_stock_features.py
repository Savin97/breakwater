# feature_engineering/pre_earnings_stock_features.py
"""
    Pre-earnings stock features:
    daily_ret
    drift_30d
    drift_60d
    vol_10d
    vol_30d
    mom_5d
    mom_20d
    market_cap_log
    cap_bucket
    beta_5y_monthly
    beta_bucket
    avg_dollar_volume
    past_large_move_freq *(computed only from past earnings, rolled forward safely)
    past_downside_tail_freq *
    past_small_move_freq *
    abs_reaction_std_3d * 
    abs_reaction_std_10d *
"""

import pandas as pd
import numpy as np

from config import (DEFAULT_REACTION_WINDOW,
                    SHORT_TERM_DRIFT,
                    LONG_TERM_DRIFT,
                    SHORT_TERM_VOLATILITY,
                    LONG_TERM_VOLATILITY,
                    SHORT_TERM_MOMENTUM,
                    LONG_TERM_MOMENTUM)
from feature_engineering.event_features import (
    event_abs_reaction_median,
    event_abs_reaction_p75,
    event_abs_reaction_p75_rolling,
    event_abs_reaction_p90_rolling,
    event_surprise_features,
    event_pre_earnings_drift_z)

def engineer_daily_ret(input_df):
    df = input_df
    # Daily return
    df['daily_ret'] = df.groupby('stock')['price'].pct_change()

    return df

def engineer_drift(input_df):
    df = input_df
    group = df.groupby('stock')['daily_ret']
    df[f'drift_30d'] = group.transform(lambda x: x.rolling(SHORT_TERM_DRIFT).mean().shift(1))
    df[f'drift_60d'] = group.transform(lambda x: x.rolling(LONG_TERM_DRIFT).mean().shift(1))

    return df

def engineer_volatility(input_df):
    df = input_df
    group = df.groupby('stock')['daily_ret']
    # Volatility (short + baseline)
    df[f'vol_10d'] = group.transform(lambda x: x.rolling(SHORT_TERM_VOLATILITY).std().shift(1))
    df[f'vol_30d'] = group.transform(lambda x: x.rolling(LONG_TERM_VOLATILITY).std().shift(1))
    # Vol expansion signal | by default vol_ratio_10_to_30
    df[f'vol_ratio_10_to_30'] = (
            df[f'vol_{SHORT_TERM_VOLATILITY}d'] / df[f'vol_{LONG_TERM_VOLATILITY}d']
    )
    return df

def engineer_momentum(input_df):
    df = input_df
    group = df.groupby('stock')['daily_ret']
    # Momentum (fast + standard)
    df['mom_5d']  = group.transform(lambda x: x.rolling(SHORT_TERM_MOMENTUM).sum().shift(1))
    df['mom_20d'] = group.transform(lambda x: x.rolling(LONG_TERM_MOMENTUM).sum().shift(1))

    return df

def engineer_earnings_windows(input_df):
    """
        Builds features:
        days_to_earnings: Earnings date - current date
        is_earnings_day: Earnings date - current date = 0
        is_earnings_week: Earnings date - current date <= 5
        is_earnings_window: Earnings date - current date <= 10
    """
    df = input_df
    df["days_to_earnings"] = (df["earnings_date"] - df["date"]).dt.days
    df["is_earnings_day"] = ( df["days_to_earnings"].notna()  # Avoids errors when days_to_earnings is N/A which leads to False->0
                               & (df["days_to_earnings"] == 0 ) ).astype("int8")
    df["is_earnings_week"] = ( df["days_to_earnings"].notna()  # Avoids errors when days_to_earnings is N/A which leads to False->0
                               & (df["days_to_earnings"].between(0, 5)) ).astype("int8")
    df["is_earnings_window"] = ( df["days_to_earnings"].notna()  # Avoids errors when days_to_earnings is N/A which leads to False->0
                               & (df["days_to_earnings"].between(0, 10)) ).astype("int8")

    return df

def engineer_abs_reaction_median(input_df):
    """
        Median of |DEFAULT_REACTION_WINDOW| over past earnings.

        Captures Typical size of earnings moves
        Robust to outliers

        High -> this stock usually moves on earnings
        Low -> earnings are often a non-event
        Intuition:
            - Median answers: "What usually happens on earnings?"

        Median so one crazy quarter doesn't dominate the signal
    """
    df = input_df
    
    # Separate earnings rows
    earnings_mask =  df[DEFAULT_REACTION_WINDOW].notna()
    earnings_df = df.loc[earnings_mask, ["stock","earnings_date", DEFAULT_REACTION_WINDOW]].copy()
    earnings_df = earnings_df.sort_values(["stock", "earnings_date"])

    earnings_df = event_abs_reaction_median(earnings_df)
    df.loc[earnings_mask, "abs_reaction_median"] = earnings_df["abs_reaction_median"]
    assert earnings_mask.sum() == len(earnings_df), "Mismatch: earnings rows vs earnings_df"
    return df

def engineer_abs_reaction_p75(input_df):
    """
        Compute the 75th percentile of historical absolute DEFAULT_REACTION_WINDOW earnings reactions
        for each stock, using only *past* earnings events.

        Intuition:
            - p75 answers: "When it moves meaningfully, how big does it often get?"
        This captures the stock's *upper-tail earnings volatility* without being
        dominated by single extreme outliers (unlike max or std).

        Construction details
        --------------------
        • Strictly non-leaky: statistics at time t are computed from events < t
        via shift(1)
        • Expanding window over the stock's earnings history
        • First earnings event per stock has no past history -> NaN (by design)
        Values are populated only on earnings rows; non-earnings rows remain NaN.
        
    """
    df = input_df
    # Separate earnings rows
    earnings_mask = df[DEFAULT_REACTION_WINDOW].notna()
    earnings_df = df.loc[earnings_mask, ["stock","earnings_date",DEFAULT_REACTION_WINDOW]].copy()
    earnings_df = earnings_df.sort_values(["stock", "earnings_date"])

    earnings_df = event_abs_reaction_p75(earnings_df)
    df.loc[earnings_mask, "abs_reaction_p75"] = earnings_df["abs_reaction_p75"]
    assert earnings_mask.sum() == len(earnings_df), "Mismatch: earnings rows vs earnings_df"

    return df


def engineer_abs_reaction_p75_rolling(df, window=28, percentile=0.75):
    """Rolling p75 over the stock's last `window` PRIOR earnings events.

    Computation lives in feature_engineering.event_features so the pending upcoming
    event uses the identical window (Phase 1). Stocks with < `window` events get NaN
    here and fall back to expanding abs_reaction_p75 in scoring.
    """
    earnings_mask = df["is_earnings_day"] == True
    earnings_df = event_abs_reaction_p75_rolling(df.loc[earnings_mask].copy(), window, percentile)
    df.loc[earnings_mask, "abs_reaction_p75_rolling"] = earnings_df["abs_reaction_p75_rolling"]
    return df

def engineer_abs_reaction_p90_rolling(df, window=28, percentile=0.9):
    """Rolling p90 over the stock's last `window` PRIOR earnings events."""
    earnings_mask = df["is_earnings_day"] == True
    earnings_df = event_abs_reaction_p90_rolling(df.loc[earnings_mask].copy(), window, percentile)
    df.loc[earnings_mask, "abs_reaction_p90_rolling"] = earnings_df["abs_reaction_p90_rolling"]
    return df

#---------------------------------------
# EPS surprise momentum features
#---------------------------------------
def engineer_surprise_features(input_df):
    """
    EPS surprise momentum features — computed on earnings rows only, all shift(1) to prevent leakage.

    Columns added (NaN on non-earnings days):
      surprise_beat      — 1 if prior report beat (surprise_percentage > 0), 0 miss, NaN if no estimate
      surprise_streak    — signed consecutive count: +4 = 4 straight beats, -2 = 2 straight misses
      surprise_mean_5    — rolling mean of surprise_percentage over last 5 earnings (min_periods=3)
      surprise_std_5     — rolling std over last 5 earnings (min_periods=3)
    """
    df = input_df
    earnings_mask = df["is_earnings_day"] == 1
    earnings_df = df.loc[earnings_mask, ["stock", "earnings_date", "surprise_percentage"]].copy()
    earnings_df = earnings_df.sort_values(["stock", "earnings_date"])

    earnings_df = event_surprise_features(earnings_df)

    for col in ["surprise_mean_5", "surprise_std_5", "surprise_streak"]:
        df.loc[earnings_mask, col] = earnings_df[col].values

    return df


def engineer_pre_earnings_drift_z(input_df):
    """
    Pre-earnings drift anomaly score — how unusual is the current 30d drift into earnings
    compared to this stock's own historical pre-earnings drift distribution?

    Uses drift_30d (already shift(1) from engineer_drift) so no additional shifting needed.
    Expanding window of past earnings-day drift_30d values, shifted by 1 event.

    Columns added (NaN on non-earnings days):
      pre_earnings_drift_z  — z-score of current drift_30d vs stock's own historical distribution.
                              Positive = running in hotter than usual. Negative = sold off more than usual.
    """
    df = input_df
    earnings_mask = df["is_earnings_day"] == 1
    earnings_df = df.loc[earnings_mask, ["stock", "earnings_date", "drift_30d"]].copy()
    earnings_df = earnings_df.sort_values(["stock", "earnings_date"])

    earnings_df = event_pre_earnings_drift_z(earnings_df)

    df.loc[earnings_mask, "pre_earnings_drift_z"] = earnings_df["pre_earnings_drift_z"]
    return df