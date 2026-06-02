# pipeline/stage4.py
from risk_scoring.scoring_features import (
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
def stage4(stage3_df):
    """
        Risk Scoring and recommendation stage
        Returns a separate DF
    """
    print("--------------------\nStage 4 - Risk Scoring...")
    stage4_df = stage3_df.copy()
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
        engineer_total_risk_score
    ]
    for f in features:
        stage4_df = f(stage4_df)
    if stage4_df is None:
        raise ValueError("\n---ERROR! Stage 4 Returned None.---\n")
    print("Stage 4 DONE")
    return stage4_df