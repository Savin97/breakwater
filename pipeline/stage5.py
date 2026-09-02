# pipeline/stage5.py
from report.report_builder import generate_reports
from report.calendar_builder import generate_calendar
from streamlit_dash.streamlit_export import generate_streamlit_df
from analysis.chart_weekly import generate_weekly_earnings_chart
from marketing.generate_public_track_record import generate_public_track_record
from analysis.last_week_results import generate_last_week_results
from analysis.save_predictions import save_predictions_snapshot
def stage5(df):
    print("--------------------\nStage 5 - Outputs...")

    def generate_full_parquet(df):
        df.to_parquet("output/full_df.parquet", index=False)
        print("Wrote output/full_df.parquet")
    generate_full_parquet(df)
    generate_reports(df)
    generate_calendar(df)
    generate_weekly_earnings_chart()
    generate_last_week_results(df=df)
    generate_public_track_record(df=df)
    generate_streamlit_df(df)
    save_predictions_snapshot(df)
    print("Stage 5 DONE")
    return df
