# pipeline/stage5.py
from report.report_builder import generate_reports
from report.calendar_builder import generate_calendar
from streamlit_dash.streamlit_export import export_streamlit_df
from analysis.chart_weekly import generate_weekly_earnings_chart
from marketing.generate_public_track_record import generate_public_track_record
from analysis.last_week_results import generate_last_week_results
def stage5(df):
    print("--------------------\nStage 5 - Outputs...")
    generate_reports(df)
    generate_calendar(df)
    generate_weekly_earnings_chart()
    generate_last_week_results()
    generate_public_track_record()
    export_streamlit_df(df)
    print("Stage 5 DONE")
    return df
