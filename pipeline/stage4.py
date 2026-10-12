# pipeline/stage4.py
from pipeline.events import build_pipeline_event_frame
from scoring.model_b import (
    engineer_score_asof_date,
    engineer_prior_history,
    engineer_risk_score,
    engineer_risk_tier,
    engineer_model_version,
    print_call_summary)


def stage4(daily_df):
    """
    Stage 4 — the event frame and Model B (scoring/model_b.py).

    One row per earnings event (completed + one pending per active stock), with corrected
    outcomes; then, for every event: its call cutoff, its usable corrected prior history,
    the Breakwater Risk Percentile and the tier. The one production magnitude scorer.
    `daily_df` supplies the prices, timing anchors and the market-session grid.
    """
    print("--------------------\nStage 4 - Event frame + Model B scoring...")
    events = build_pipeline_event_frame(daily_df)
    events = engineer_score_asof_date(events, daily_df)
    events = engineer_prior_history(events, daily_df)
    events = engineer_risk_score(events)
    events = engineer_risk_tier(events)
    events = engineer_model_version(events)
    print_call_summary(events)
    print("Stage 4 DONE")
    return events
