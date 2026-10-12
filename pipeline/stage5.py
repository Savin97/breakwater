# pipeline/stage5.py
from utilities.output_utilities import write_daily_parquet, write_events_parquet
from report.report_builder import generate_reports
from report.calendar_builder import generate_calendar
from streamlit_dash.streamlit_export import generate_streamlit_df
from analysis.chart_weekly import generate_weekly_earnings_chart
from analysis.last_week_results import generate_last_week_results
from marketing.generate_public_track_record import generate_public_track_record
from pipeline.events import pending_events


def stage5(daily_df, events_df):
    """
    Stage 5 — outputs, all from the Model B-scored event frame.

    Everything here is LOCAL: files under output/. Nothing is deployed, mailed or
    archived — publication is scripts/full_workflow.sh, and the prediction archive is
    written only by publishing/archive_calls.py after the dashboard push.

    Forward-looking outputs read the PENDING rows, never
    `df.sort_values("date").groupby("stock").last()`. The dashboard parquets are written
    before the weekly chart, which reads upcoming_df.parquet.
    """
    print("--------------------\nStage 5 - Outputs...")
    write_daily_parquet(daily_df)
    write_events_parquet(events_df)
    generate_reports(events_df)
    generate_calendar(pending_events(events_df))
    generate_streamlit_df(events_df)
    generate_weekly_earnings_chart()
    generate_last_week_results(events_df=events_df)
    generate_public_track_record(events_df=events_df)
    print("Stage 5 DONE")
    return events_df
