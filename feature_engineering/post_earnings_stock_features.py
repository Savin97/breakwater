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
from feature_engineering.event_features import (
    event_reaction_std,
    event_reaction_entropy,
    event_directional_bias,
    reaction_entropy)

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
    df["is_up"] = (df[DEFAULT_REACTION_WINDOW] > REACTION_THRESHOLD ).astype("int8")
    df["is_down"] = (df[DEFAULT_REACTION_WINDOW] < - REACTION_THRESHOLD ).astype("int8")
    df["is_nochange"] = ( df[DEFAULT_REACTION_WINDOW].abs() <= REACTION_THRESHOLD ).astype("int8")
    return df

def engineer_reaction_std(df):
    """
        reaction_std of past 8 earnings dates (window=8)
        min periods required is min_periods=3

        Computation lives in feature_engineering.event_features.event_reaction_std so the
        pending upcoming event runs through the identical code path (Phase 1).
    """
    earnings_df = build_earnings_df(df).copy()
    earnings_df = event_reaction_std(earnings_df)
    df = df.merge(
        earnings_df[["stock","earnings_date","reaction_std"]],
        on=["stock","earnings_date"],
        how="left"
    )
    return df

def engineer_reaction_entropy(df) -> pd.DataFrame:
    """Expanding Shannon entropy of past absolute reactions.

    Computation lives in feature_engineering.event_features.event_reaction_entropy.
    """
    best_reaction = df["reaction_3d"].fillna(df["reaction_1d"])
    earnings_mask = best_reaction.notna()
    earnings_df = df.loc[earnings_mask].copy()
    earnings_df["_best_reaction"] = best_reaction[earnings_mask].values

    earnings_df = event_reaction_entropy(earnings_df)
    df.loc[earnings_mask, "reaction_entropy"] = earnings_df["reaction_entropy"].values
    return df

def engineer_directional_bias(df):
    """ 
        directional_bias = expanding mean of *past* signed reactions (no leakage).
        For each earnings event, bias is mean of prior earnings reactions for that stock.
    
        Answers:
        When this stock reacts to earnings, does it tend to move up, down, or is it symmetric?

        Computation lives in feature_engineering.event_features.event_directional_bias.
    """
    earnings_df = build_earnings_df(df).copy()
    earnings_df = event_directional_bias(earnings_df)

    earnings_mask = df[DEFAULT_REACTION_WINDOW].notna()
    df.loc[earnings_mask, "directional_bias"] = earnings_df["directional_bias"]
    return df
