# streamlit_dash/app.py
import streamlit as st, sys, pandas as pd, warnings
from datetime import timedelta, date
# Streamlit page configuration
st.set_page_config(
    page_title="Breakwater",
    layout="wide"
)
warnings.filterwarnings('ignore')
from pathlib import Path
# Add project root (parent of "streamlit") to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# CSV with the dashboard output
# put streamlit_df.parquet in the output folder
DF_PATH = ROOT / "output/streamlit_df.parquet"
UPCOMING_PATH = ROOT / "output/upcoming_df.parquet"
from pipeline.pipeline import run_pipeline

@st.cache_data(show_spinner="Loading parquet…")
def get_full_df() -> pd.DataFrame:
    df = pd.read_parquet(DF_PATH)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    return df

@st.cache_data(show_spinner="Loading upcoming events…")
def get_upcoming_df() -> pd.DataFrame:
    if not UPCOMING_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(UPCOMING_PATH)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")
    return df

@st.cache_data(show_spinner="Loading dashboard data…")
def get_dashboard_df(use_cached_eps: bool = True) -> pd.DataFrame:
    """
    Calls your engine and returns the final dashboard dataframe.

    Assumes run_pipeline(...) returns a DataFrame with at least:
        Date, Stock, risk_level, risk_score, hist_xtreme_prob, base_xtreme_prob, risk_lift
    """
    df = get_full_df()

    # Sanity check for expected columns
    expected_cols = [
        "stock",
        "sector",
        "sub_sector",
        "earnings_date",
        "is_large_reaction",
        "is_extreme_reaction",
        "hist_extreme_prob",
        "global_hist_prob",
        "current_lift_vs_baseline",
        "current_lift_vs_same_bucket_global",
        "extreme_count",
        "risk_level",
        "risk_score",
        "base_extreme_prob",
        "pre_earnings_drift_flag",
        "surprise_momentum_flag",
        "is_high_conviction",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing expected columns in CSV: {missing}")

    if "earnings_date" in df.columns:
        df["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce")

    numeric_cols = [
        "risk_score",
        "hist_extreme_prob",
        "global_hist_prob",
        "current_lift_vs_baseline",
        "current_lift_vs_same_bucket_global",
        "base_extreme_prob",
        "extreme_count",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def sidebar_filters(df: pd.DataFrame, upcoming: pd.DataFrame) -> tuple:
    st.sidebar.header("Filters")

    stock_choice = "(All)"
    if "stock" in df.columns:
        stocks = sorted(df["stock"].dropna().unique())
        stock_choice = st.sidebar.selectbox("Stock", options=["(All)"] + stocks)
        if stock_choice != "(All)":
            df = df[df["stock"] == stock_choice]

    selected_sectors = None
    if "sector" in df.columns:
        sectors = sorted(df["sector"].dropna().unique())
        selected_sectors = st.sidebar.multiselect("Sector", options=sectors, default=sectors)
        if selected_sectors:
            df = df[df["sector"].isin(selected_sectors)]

    selected_risks = None
    if "risk_level" in df.columns:
        risk_levels = sorted(df["risk_level"].dropna().unique())
        selected_risks = st.sidebar.multiselect("Risk level", options=risk_levels, default=risk_levels)
        if selected_risks:
            df = df[df["risk_level"].isin(selected_risks)]

    lo, hi = None, None
    if "risk_score" in df.columns and not df["risk_score"].isna().all():
        min_rs = int(df["risk_score"].min())
        max_rs = int(df["risk_score"].max())
        lo, hi = st.sidebar.slider("Risk score range", min_value=min_rs, max_value=max_rs, value=(min_rs, max_rs))
        df = df[(df["risk_score"] >= lo) & (df["risk_score"] <= hi)]

    if "is_extreme_reaction" in df.columns:
        only_extreme = st.sidebar.checkbox("Only actual extreme reactions", value=False)
        if only_extreme:
            df = df[df["is_extreme_reaction"] == 1]

    if "is_large_reaction" in df.columns:
        only_large = st.sidebar.checkbox("Only actual large reactions", value=False)
        if only_large:
            df = df[df["is_large_reaction"] == 1]

    only_hc = False
    if "is_high_conviction" in df.columns:
        only_hc = st.sidebar.checkbox("High Conviction only", value=False)
        if only_hc:
            df = df[df["is_high_conviction"] == True]

    # Apply same filter state to upcoming df
    if not upcoming.empty:
        if stock_choice != "(All)" and "stock" in upcoming.columns:
            upcoming = upcoming[upcoming["stock"] == stock_choice]
        if selected_sectors and "sector" in upcoming.columns:
            upcoming = upcoming[upcoming["sector"].isin(selected_sectors)]
        if selected_risks and "earnings_explosiveness_bucket" in upcoming.columns:
            upcoming = upcoming[upcoming["earnings_explosiveness_bucket"].isin(selected_risks)]
        if lo is not None and "earnings_explosiveness_score" in upcoming.columns:
            upcoming = upcoming[
                (upcoming["earnings_explosiveness_score"] >= lo) &
                (upcoming["earnings_explosiveness_score"] <= hi)
            ]
        if only_hc and "is_high_conviction" in upcoming.columns:
            upcoming = upcoming[upcoming["is_high_conviction"] == True]

    return df, upcoming

def main():
    st.title("Breakwater - Earnings Risk Dashboard")

    with st.sidebar:
        st.markdown("### Data options")
        if st.button("Reload data from disk"):
            get_full_df.clear()
            get_dashboard_df.clear()
            get_upcoming_df.clear()

    raw_df = get_dashboard_df()
    raw_upcoming = get_upcoming_df()
    df, upcoming_filtered = sidebar_filters(raw_df, raw_upcoming)

    if df.empty:
        st.warning("No rows match the current filters.")
        return

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Earnings events", len(df))

    with col2:
        if "stock" in df.columns:
            st.metric("Unique stocks", df["stock"].nunique())

    with col3:
        if "risk_score" in df.columns:
            st.metric("Avg risk score", f"{df['risk_score'].mean():.1f}")

    with col4:
        if "is_extreme_reaction" in df.columns:
            st.metric("Extreme reactions", int(df["is_extreme_reaction"].sum()))

    with col5:
        if "is_high_conviction" in df.columns:
            st.metric("High Conviction", int(df["is_high_conviction"].sum()))

    tab_upcoming, tab_overview, tab_stock, tab_calendar = st.tabs(
        ["Upcoming Events", "Overview", "Stock drill-down", "Weekly Calendar"]
    )

    with tab_overview:
        st.subheader("Filtered earnings events")

        cols_to_show = [c for c in [
            "stock", "earnings_date", "sector",
            "risk_level", "risk_score", "is_high_conviction",
            "pre_earnings_drift_flag", "surprise_momentum_flag",
            "abs_reaction_3d",
            "is_large_reaction", "is_extreme_reaction",
            "hist_extreme_prob", "current_lift_vs_baseline",
        ] if c in df.columns]

        _period_options = {
            "1 month":  pd.DateOffset(months=1),
            "3 months": pd.DateOffset(months=3),
            "6 months": pd.DateOffset(months=6),
            "1 year":   pd.DateOffset(years=1),
            "2 years":  pd.DateOffset(years=2),
            "5 years":  pd.DateOffset(years=5),
        }
        period_label = st.radio(
            "Show events from last",
            options=list(_period_options.keys()),
            horizontal=True,
            index=3,
        )

        df_display = df.sort_values("earnings_date", ascending=False).copy()
        df_display["earnings_date"] = df_display["earnings_date"].dt.date
        cutoff = (pd.Timestamp(date.today()) - _period_options[period_label]).date()
        df_display = df_display[df_display["earnings_date"] >= cutoff]
        if "abs_reaction_3d" in df_display.columns:
            df_display["abs_reaction_3d"] = (df_display["abs_reaction_3d"] * 100).round(2)

        st.dataframe(
            df_display[cols_to_show],
            use_container_width=True,
            hide_index=True,
            column_config={
                "earnings_date":            st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                "stock":                    st.column_config.TextColumn("Ticker"),
                "sector":                   st.column_config.TextColumn("Sector"),
                "risk_level":               st.column_config.TextColumn("Risk Level"),
                "risk_score":               st.column_config.NumberColumn("Score", format="%.0f"),
                "is_high_conviction":       st.column_config.CheckboxColumn("HC ★"),
                "pre_earnings_drift_flag":  st.column_config.TextColumn("Drift"),
                "surprise_momentum_flag":   st.column_config.TextColumn("Surprise"),
                "abs_reaction_3d":          st.column_config.NumberColumn("Actual Move %", format="%.2f"),
                "is_large_reaction":        st.column_config.CheckboxColumn("Large"),
                "is_extreme_reaction":      st.column_config.CheckboxColumn("Extreme"),
                "hist_extreme_prob":        st.column_config.NumberColumn("P(Extreme)", format="%.3f"),
                "current_lift_vs_baseline": st.column_config.NumberColumn("Lift", format="%.2fx"),
            }
        )

        if "risk_level" in df.columns:
            st.markdown("#### Count of events by risk level")
            risk_counts = df["risk_level"].value_counts().sort_index()
            st.bar_chart(risk_counts)

    with tab_upcoming:
        st.subheader("Upcoming Earnings Events")

        upcoming = upcoming_filtered

        if upcoming.empty:
            st.info("No upcoming earnings found. Run the pipeline to generate upcoming_df.parquet.")
        else:
            window_days = st.radio(
                "Show events in the next",
                options=[14, 30, 60, 999],
                format_func=lambda x: "All" if x == 999 else f"{x} days",
                horizontal=True,
            )
            mask = upcoming["days_to_earnings"] <= window_days
            view = upcoming[mask].sort_values(
                ["earnings_date", "earnings_explosiveness_score"], ascending=[True, False]
            ).copy()

            if view.empty:
                st.info(f"No events in the next {window_days} days.")
            else:
                u1, u2, u3, u4 = st.columns(4)
                u1.metric("Events", len(view))
                u2.metric("High Alert", int((view["earnings_explosiveness_bucket"] == "High Alert").sum()))
                u3.metric("High Conviction ★", int(view["is_high_conviction"].sum()))
                u4.metric("Elevated", int((view["earnings_explosiveness_bucket"] == "Elevated").sum()))

                display_cols = [c for c in [
                    "earnings_date", "days_to_earnings", "stock", "sector",
                    "earnings_explosiveness_bucket", "earnings_explosiveness_score",
                    "peer_percentile", "pre_earnings_drift_flag",
                    "surprise_momentum_flag", "is_high_conviction",
                    "expected_move_pct", "iv_vs_hist_ratio",
                ] if c in view.columns]

                view = view.copy()
                view["earnings_date"] = view["earnings_date"].dt.date

                st.dataframe(
                    view[display_cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "earnings_date":                  st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                        "days_to_earnings":               st.column_config.NumberColumn("Days", format="%d"),
                        "stock":                          st.column_config.TextColumn("Ticker"),
                        "earnings_explosiveness_bucket":  st.column_config.TextColumn("Risk Level"),
                        "earnings_explosiveness_score":   st.column_config.NumberColumn("Score", format="%.0f"),
                        "peer_percentile":                st.column_config.NumberColumn("Percentile", format="%dth"),
                        "pre_earnings_drift_flag":        st.column_config.TextColumn("Drift"),
                        "surprise_momentum_flag":         st.column_config.TextColumn("Surprise Pattern"),
                        "is_high_conviction":             st.column_config.CheckboxColumn("HC ★"),
                        "expected_move_pct":              st.column_config.NumberColumn("IV Implied", format="%.1f%%"),
                        "iv_vs_hist_ratio":               st.column_config.NumberColumn("IV/Hist", format="%.2fx"),
                    },
                )

    with tab_stock:
        st.subheader("Single-stock history")

        if "stock" not in df.columns:
            st.info("stock column not found.")
            return

        stocks = sorted(df["stock"].dropna().unique())
        selected_stock = st.selectbox("Choose stock", options=stocks)

        stock_df = df[df["stock"] == selected_stock].copy()

        if "earnings_date" in stock_df.columns:
            stock_df = stock_df.sort_values("earnings_date")

        cols = [
            c for c in [
                "earnings_date",
                "stock",
                "risk_level",
                "risk_score",
                "pre_earnings_drift_flag",
                "surprise_momentum_flag",
                "is_high_conviction",
                "hist_extreme_prob",
                "base_extreme_prob",
                "current_lift_vs_baseline",
                "current_lift_vs_same_bucket_global",
                "is_large_reaction",
                "is_extreme_reaction",
            ]
            if c in stock_df.columns
        ]

        st.dataframe(stock_df[cols], use_container_width=True, hide_index=True)

        if {"earnings_date", "risk_score"}.issubset(stock_df.columns):
            chart_df = stock_df.set_index("earnings_date")["risk_score"]
            st.line_chart(chart_df)
            
    with tab_calendar:
        st.subheader("Earnings Risk Calendar")

        earn = get_full_df()

        # Merge upcoming events so the calendar spans past + future
        _upcoming_cal = get_upcoming_df()
        if not _upcoming_cal.empty:
            _cal_cols = [c for c in [
                "stock", "sector", "sub_sector", "earnings_date",
                "earnings_explosiveness_score", "earnings_explosiveness_bucket",
                "pre_earnings_drift_flag", "surprise_momentum_flag", "is_high_conviction",
            ] if c in _upcoming_cal.columns]
            earn = pd.concat([earn, _upcoming_cal[_cal_cols]], ignore_index=True)
            earn = earn.drop_duplicates(subset=["stock", "earnings_date"], keep="first")
            earn["earnings_date"] = pd.to_datetime(earn["earnings_date"], errors="coerce")

        today = pd.Timestamp(date.today())
        default_start = today - timedelta(days=7)
        default_end   = today + timedelta(days=14)

        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("From", value=default_start.date())
        with c2:
            end_date = st.date_input("To",   value=default_end.date())

        window = earn[
            (earn["earnings_date"] >= pd.Timestamp(start_date)) &
            (earn["earnings_date"] <= pd.Timestamp(end_date))
        ].copy()

        if window.empty:
            st.info("No earnings events in the selected window.")
        else:
            window["is_high_conviction"] = (
                (window["earnings_explosiveness_bucket"] == "High Alert") &
                (window["pre_earnings_drift_flag"].fillna("") != "")
            )

            today_dt = pd.Timestamp(date.today())
            window["status"] = window["earnings_date"].apply(
                lambda d: "Scheduled" if pd.Timestamp(d) > today_dt else "Reported"
            )

            window["earnings_date"] = window["earnings_date"].dt.date
            display = window[[
                "earnings_date", "stock", "sector", "sub_sector",
                "earnings_explosiveness_score", "earnings_explosiveness_bucket",
                "status", "pre_earnings_drift_flag", "surprise_momentum_flag",
                "is_high_conviction",
            ]].rename(columns={
                "earnings_date":                "Date",
                "stock":                        "Ticker",
                "sector":                       "Sector",
                "sub_sector":                   "Sub-Sector",
                "earnings_explosiveness_score": "Risk Score",
                "earnings_explosiveness_bucket": "Risk Level",
                "status":                       "Status",
                "pre_earnings_drift_flag":      "Drift",
                "surprise_momentum_flag":       "Surprise Pattern",
                "is_high_conviction":           "High Conviction",
            }).sort_values(["Date", "Risk Score"], ascending=[True, False])

            # Summary KPIs

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Events", len(display))
            k2.metric("High Conviction", int(display["High Conviction"].sum()))
            k3.metric("With Drift Flag",
                      int((display["Drift"].fillna("") != "").sum()))
            k4.metric("Avg Risk Score", f"{display['Risk Score'].mean():.1f}")
            sector_top = window["sector"].value_counts().index[0] if not window.empty else "—"
            k5.metric("Most Active Sector", sector_top)

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Date":             st.column_config.DateColumn("Date", format="DD MMM YYYY"),
                    "Risk Score":       st.column_config.NumberColumn("Risk Score", format="%.0f"),
                    "Status":           st.column_config.TextColumn("Status"),
                    "High Conviction":  st.column_config.CheckboxColumn("High Conviction"),
                },
            )

            with st.sidebar:
                st.markdown("---")
                if st.button("Export calendar HTML"):
                    sys.path.insert(0, str(ROOT))
                    from report.calendar_builder import generate_calendar
                    generate_calendar(earn, reference_date=start_date,
                                      window_days=(end_date - start_date).days)
                    st.success("Written to output/weekly_calendar.html")

if __name__ == "__main__":
    main()