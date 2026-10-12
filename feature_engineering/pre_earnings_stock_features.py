# feature_engineering/pre_earnings_stock_features.py
"""
    Daily per-stock price features and the earnings-window clock:
    daily_ret
    vol_10d, vol_30d, vol_ratio_10_to_30
    days_to_earnings, is_earnings_day, is_earnings_week, is_earnings_window

Nothing here feeds the Model B score (scoring/model_b.py), which reads only corrected
prior earnings outcomes. `days_to_earnings` / `is_earnings_day` are what the event frame
is built from. The volatility columns are generic price context kept on the daily frame
(output/full_df.parquet); every rolling window is shifted by one day, so a row never
sees its own return.

The legacy 0.3.1 scoring inputs that used to live here (drift, momentum, p75 history,
surprise streaks, drift z-score) were removed with that scorer in 0.4.0; they are in
the archive tag methodology-audit-archive-october-2026 and git history.
"""
from config import SHORT_TERM_VOLATILITY, LONG_TERM_VOLATILITY


def engineer_daily_ret(input_df):
    df = input_df
    df['daily_ret'] = df.groupby('stock')['price'].pct_change()
    return df


def engineer_volatility(input_df):
    df = input_df
    group = df.groupby('stock')['daily_ret']
    # Volatility (short + baseline)
    df[f'vol_10d'] = group.transform(lambda x: x.rolling(SHORT_TERM_VOLATILITY).std().shift(1))
    df[f'vol_30d'] = group.transform(lambda x: x.rolling(LONG_TERM_VOLATILITY).std().shift(1))
    df[f'vol_ratio_10_to_30'] = (
            df[f'vol_{SHORT_TERM_VOLATILITY}d'] / df[f'vol_{LONG_TERM_VOLATILITY}d']
    )
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
