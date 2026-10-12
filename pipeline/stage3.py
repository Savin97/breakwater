# pipeline/stage3.py
from feature_engineering.pre_earnings_stock_features import (
    engineer_daily_ret,
    engineer_volatility,
    engineer_earnings_windows)
from feature_engineering.post_earnings_stock_features import (
    engineer_earnings_reactions,
    engineer_abs_reaction_3d)


def stage3(stage2_df):
    """
    Stage 3 — daily preparation. No score is computed here.

    The earnings-window clock the event frame is built from, daily returns and volatility
    as price context, and the LEGACY report-date reactions, kept only as the control for
    the corrected anchored target (feature_engineering/post_earnings_stock_features.py).
    """
    print("--------------------\nStage 3 - Daily preparation...")
    df = stage2_df.copy().sort_values(["stock", "date"], kind="mergesort")
    df = engineer_daily_ret(df)
    df = engineer_volatility(df)
    df = engineer_earnings_windows(df)
    df = engineer_earnings_reactions(df)
    df = engineer_abs_reaction_3d(df)
    print("Stage 3 DONE")
    return df
