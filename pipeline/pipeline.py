# pipeline/pipeline.py
from pipeline.stage1 import stage1
from pipeline.stage2 import stage2
from pipeline.stage3 import stage3
from pipeline.stage4 import stage4
from pipeline.stage5 import stage5
from pipeline.events import build_and_score_event_frame, load_pipeline_announcement_timing

def run_pipeline(incremental):
    stage1(incremental=incremental)
    df = stage2()
    df = stage3(df)
    df = stage4(df)
    # Stage 4b — the event frame. Built from the daily frame, never merged back into it,
    # so no future-dated row can reach the rolling price windows or the per-date
    # cross-sectional ranks. Asserts that no completed event's score changed.
    events_df = build_and_score_event_frame(df, load_pipeline_announcement_timing())
    stage5(df, events_df)