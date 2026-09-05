# pipeline/stage5.py
from report.report_builder import generate_reports
from report.calendar_builder import generate_calendar
from streamlit_dash.streamlit_export import generate_streamlit_df
from analysis.chart_weekly import generate_weekly_earnings_chart
from marketing.generate_public_track_record import generate_public_track_record
from analysis.last_week_results import generate_last_week_results
from analysis.save_predictions import save_predictions_snapshot
from pipeline.events import build_and_score_event_frame


def stage5(df, events_df=None):
    """Outputs.

    `events_df` is the event frame (pipeline/events.py): one row per earnings event,
    completed or pending. Every forward-looking consumer reads its pending rows instead
    of `df.sort_values("date").groupby("stock").last()`, which returned the stock's last
    COMPLETED event's state. Built here when not supplied so the documented
    stage5(stage4(stage3(stage2()))) re-score one-liner keeps working.
    """
    print("--------------------\nStage 5 - Outputs...")
    if events_df is None:
        events_df = build_and_score_event_frame(df)

    def generate_full_parquet(df):
        df.to_parquet("output/full_df.parquet", index=False)
        print("Wrote output/full_df.parquet")
    generate_full_parquet(df)
    events_df.to_parquet("output/events_df.parquet", index=False)
    print(f"Wrote output/events_df.parquet ({len(events_df)} events)")
    generate_reports(df, events_df)
    generate_calendar(df, events_df=events_df)
    generate_weekly_earnings_chart()
    generate_last_week_results(df=df)
    generate_public_track_record(df=df)
    generate_streamlit_df(df, events_df)
    save_predictions_snapshot(events_df)
    print("Stage 5 DONE")
    return df
