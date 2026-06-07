# pipeline/stage3.py
import pandas as pd
from config import INCREMENTAL_CACHED_COLS
from feature_engineering.pre_earnings_stock_features import (
    engineer_daily_ret,
    engineer_drift,
    engineer_volatility,
    engineer_momentum,
    engineer_abs_reaction_median,
    engineer_abs_reaction_p75,
    engineer_abs_reaction_p75_rolling,
    engineer_abs_reaction_p90_rolling,
    engineer_earnings_windows,
    engineer_surprise_features,
    engineer_pre_earnings_drift_z)
from feature_engineering.post_earnings_stock_features import (
    engineer_earnings_reactions,
    engineer_reaction_class,
    engineer_abs_reaction_3d,
    engineer_reaction_std,
    engineer_reaction_entropy,
    engineer_directional_bias)
from feature_engineering.pre_earnings_sector_features import (
    engineer_sector_drift_vol,
    engineer_stock_vs_sector_vol,
    engineer_sector_earnings_density)


def stage3(stage2_df, incremental=False):
    """
    Pipeline Stage 3 - Feature Engineering.

    incremental=False (default): full run — all features computed from scratch.
    incremental=True:  fast path — only price-dependent rolling features are
                       recomputed; expanding earnings stats are read from
                       output/full_df.parquet and broadcast across all rows.
                       Use only when no new earnings events have been reported
                       since the last full run (run_incremental() ensures this).
    """
    print("--------------------\nStage 3 - Feature Engineering...")
    stage3_df = stage2_df.copy()
    stage3_df = stage3_df.sort_values(["stock","date"], kind="mergesort")

    if incremental:
        # Price-dependent rolling features + cross-sectional sector features.
        feature_steps = [
            engineer_daily_ret,
            engineer_drift,
            engineer_volatility,
            engineer_momentum,
            engineer_earnings_windows,
            engineer_sector_drift_vol,
            engineer_stock_vs_sector_vol,
            engineer_sector_earnings_density,
        ]
        for feature in feature_steps:
            stage3_df = feature(stage3_df)

        # Read stable expanding stats (last non-null value per stock via skipna=True).
        cached_stats = (
            pd.read_parquet("output/full_df.parquet",
                            columns=["stock"] + INCREMENTAL_CACHED_COLS)
            .groupby("stock")[INCREMENTAL_CACHED_COLS]
            .last()
            .reset_index()
        )
        stage3_df = stage3_df.merge(cached_stats, on="stock", how="left")

    else:
        feature_steps = [
            engineer_daily_ret,
            engineer_drift,
            engineer_volatility,
            engineer_momentum,
            engineer_earnings_windows,
            engineer_earnings_reactions,
            engineer_reaction_class,
            engineer_reaction_std,
            engineer_reaction_entropy,
            engineer_directional_bias,
            engineer_abs_reaction_3d,
            engineer_abs_reaction_median,
            engineer_abs_reaction_p75,
            engineer_abs_reaction_p75_rolling,
            engineer_abs_reaction_p90_rolling,
            engineer_surprise_features,
            engineer_pre_earnings_drift_z,
            engineer_sector_drift_vol,
            engineer_stock_vs_sector_vol,
            engineer_sector_earnings_density,
        ]
        for feature in feature_steps:
            stage3_df = feature(stage3_df)

    if stage3_df is None:
        raise ValueError("\n---ERROR! Stage 3 Returned None.---\n")
    print("Stage 3 DONE")
    return stage3_df