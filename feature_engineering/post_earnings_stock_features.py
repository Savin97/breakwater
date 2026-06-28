# feature_engineering/post_earnings_stock_features.py
"""
    reaction_1d
    reaction_3d
    reaction_5d
    is_up
    is_down
    is_nochange
    reaction_std
    reportedEPS
    estimatedEPS
    surprise
    surprisePercentage
    surprise_bucket
"""
import pandas as pd
import numpy as np

from config import REACTION_THRESHOLD, DEFAULT_REACTION_WINDOW
from utilities.data_utilities import build_earnings_df

def engineer_earnings_reactions(df):
    """
        Compute forward post-earnings price reactions.

        For each stock and date, computes forward returns:
            reaction_k = price(t + k) / price(t) - 1
        for k in {1, 3, 5} trading days.

        Reactions are computed mechanically for all rows to preserve
        group alignment, then set to NaN on non-earnings days.

        Contract:
        - Requires columns: ["stock", "date", "price", "is_earnings_day"]
        - Output columns exist only on earnings days; non-earnings rows are NaN
        - Prevents leakage of post-event information into normal days
    """
    df = df.sort_values(["stock", "date"])
    group = df.groupby("stock")["price"]

    # forward returns from *today* to +k trading days
    df["reaction_1d"] = (group.shift(-1) / df["price"]) - (1)
    df["reaction_3d"] = (group.shift(-3) / df["price"]) - (1)
    df["reaction_5d"] = (group.shift(-5) / df["price"]) - (1)

    # keep only on earnings days (else NaN)
    mask = df["is_earnings_day"].astype(bool)
    for column in ["reaction_1d", "reaction_3d", "reaction_5d"]:
        df.loc[~mask, column] = np.nan # Apply NaN where the mask returns False

    # Assertion checks
    for i in [1,3,5]:
        assert df.loc[mask, f"reaction_{i}d"].notna().any() # At least 1 has a valid reaction
        assert df.loc[~mask, f"reaction_{i}d"].isna().all() # No reactions on non-earnings days
    return df

def engineer_abs_reaction_3d(df):
    earnings_mask = df["is_earnings_day"] == True
    # Absolute reaction
    df.loc[earnings_mask, "abs_reaction_3d"] = (
        df.loc[earnings_mask, "reaction_3d"].abs()
    )
    return df

def engineer_reaction_class(df):
    """
        Engineer is_up,is_down,is_nochange features:
        if DEFAULT_REACTION_WINDOW > REACTION_THRESHOLD
    """
    df["is_up"] = (df[DEFAULT_REACTION_WINDOW] > REACTION_THRESHOLD ).astype(int)
    df["is_down"] = (df[DEFAULT_REACTION_WINDOW] < - REACTION_THRESHOLD ).astype(int)
    df["is_nochange"] = ( df[DEFAULT_REACTION_WINDOW].abs() <= REACTION_THRESHOLD ).astype(int)
    return df

def engineer_reaction_std(df):
    """
        reaction_std of past 8 earnings dates (window=8)
        min periods required is min_periods=3
    """
    earnings_df = build_earnings_df(df)
    earnings_df["reaction_std"] = (
        earnings_df.groupby("stock")[DEFAULT_REACTION_WINDOW]
            .transform(lambda x: x.abs().shift(1).rolling(window=8, min_periods=3).std(ddof=1) )
    )
    df = df.merge(
        earnings_df[["stock","earnings_date","reaction_std"]],
        on=["stock","earnings_date"],
        how="left"
    )
    return df

def engineer_reaction_entropy(df) -> pd.DataFrame:
    """ """
    def reaction_entropy(series : pd.Series, bins = 8) -> float:   
        """ 
            series = pd.Series of absolute reactions (past only)
            Entropy (Shannon entropy) is defined as:
                H = - sum_over_i( p_i * log(p_i) )

            Properties:
                minimum ≈ 0 -> all reactions same bucket
                higher -> more chaotic

            Entropy needs a probability distribution, we'll use a stable default of 
            8 bins and compute a histogram.

            p_i = count in bin i / total past earnings
        """ 
        series = series.dropna()

        if len(series) < bins:
            return np.nan
        
        hist, _ = np.histogram( series, bins)
        probs =  hist/hist.sum()
        probs = probs[probs > 0] # avoids log(0)

        return -np.sum(probs * np.log(probs) )
    
    best_reaction = df["reaction_3d"].fillna(df["reaction_1d"])
    earnings_mask = best_reaction.notna()
    earnings_df = df.loc[earnings_mask].copy()
    earnings_df["_best_reaction"] = best_reaction[earnings_mask].values

    earnings_df["reaction_entropy"] = (
        earnings_df.groupby("stock")["_best_reaction"]
            .transform(lambda x: x.abs().shift(1).expanding().apply(reaction_entropy))
    )
    df.loc[earnings_mask, "reaction_entropy"] = earnings_df["reaction_entropy"].values
    return df

def engineer_directional_bias(df):
    """ 
        directional_bias = expanding mean of *past* signed reactions (no leakage).
        For each earnings event, bias is mean of prior earnings reactions for that stock.
    
        Answers:
        When this stock reacts to earnings, does it tend to move up, down, or is it symmetric?
    """

    earnings_df = build_earnings_df(df)
    earnings_df["directional_bias"] = (
        earnings_df
            .groupby("stock")[DEFAULT_REACTION_WINDOW]
                .transform(lambda x: x.shift(1).expanding().mean())
    )

    earnings_mask = df[DEFAULT_REACTION_WINDOW].notna()
    df.loc[earnings_mask, "directional_bias"] = earnings_df["directional_bias"]
    return df