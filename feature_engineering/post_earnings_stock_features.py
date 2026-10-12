# feature_engineering/post_earnings_stock_features.py
"""
    LEGACY post-earnings reactions, measured from the report-date close:
    reaction_1d, reaction_3d, reaction_5d, abs_reaction_3d

These assume every announcement lands after the close of the report date, which is
false for before-open (BMO) reporters (audit/PHASE0_AUDIT_REV2.md §Q1-Q2). They are kept
ONLY as the control the corrected, timestamp-anchored target is checked against: an AMC
anchored reaction must be bit-identical to the legacy one wherever the ticker has a row on
every session (feature_engineering/announcement_timing.py, test_announcement_timing.py).

Nothing scores, reports or displays them. Model B, every customer surface and every
outcome evaluation use `abs_reaction_3d_anchored` from the event frame;
testing/test_model_b.py asserts that statically.
"""
import numpy as np


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
    df.loc[earnings_mask, "abs_reaction_3d"] = (
        df.loc[earnings_mask, "reaction_3d"].abs()
    )
    return df
