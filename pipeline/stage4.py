# pipeline/stage4.py
from feature_engineering.scoring_features import (
    engineer_large_reaction,
    engineer_extreme_reaction,
    engineer_vol_stress,
    engineer_momentum_pressure,
    engineer_earnings_explosiveness,
    engineer_sector_vol_stress,
    engineer_proximity_score,
    engineer_vol_expansion_score,
    engineer_momentum_fragility_score,
    engineer_earnings_explosiveness_score,
    classify_large_relative_earnings_move_bucket,
    engineer_surprise_momentum_flag,
    engineer_pre_earnings_drift_flag,
    engineer_total_risk_score)


def stage4(stage3_df, incremental=False):
    """
    Stage 4 — Risk Scoring.

    incremental=False (default): full run — all scoring functions applied.
    incremental=True:  fast path — skips functions that require abs_reaction_3d
                       (is_large_reaction, is_extreme_reaction, earnings_move_bucket),
                       which is not available in the 90-day incremental window.
    """
    print("--------------------\nStage 4 - Risk Scoring...")
    stage4_df = stage3_df.copy()

    if incremental:
        features = [
            engineer_vol_stress,
            engineer_sector_vol_stress,
            engineer_momentum_pressure,
            engineer_earnings_explosiveness,
            engineer_proximity_score,
            engineer_vol_expansion_score,
            engineer_momentum_fragility_score,
            # engineer_earnings_explosiveness_score skipped: score and bucket are
            # read from cache in stage3 so the last-complete-event values are preserved.
            engineer_surprise_momentum_flag,
            engineer_pre_earnings_drift_flag,
            engineer_total_risk_score,
        ]
    else:
        features = [
            engineer_large_reaction,
            engineer_extreme_reaction,
            engineer_vol_stress,
            engineer_sector_vol_stress,
            engineer_momentum_pressure,
            engineer_earnings_explosiveness,
            engineer_proximity_score,
            engineer_vol_expansion_score,
            engineer_momentum_fragility_score,
            engineer_earnings_explosiveness_score,
            classify_large_relative_earnings_move_bucket,
            engineer_surprise_momentum_flag,
            engineer_pre_earnings_drift_flag,
            engineer_total_risk_score,
        ]

    for f in features:
        stage4_df = f(stage4_df)
    if stage4_df is None:
        raise ValueError("\n---ERROR! Stage 4 Returned None.---\n")
    print("Stage 4 DONE")
    return stage4_df