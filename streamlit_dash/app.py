# streamlit_dash/app.py
import streamlit as st, sys, pandas as pd, warnings, duckdb
from datetime import timedelta, date

st.set_page_config(page_title="Breakwater", layout="wide")

warnings.filterwarnings('ignore')
from pathlib import Path
# Add project root (parent of "streamlit_dash") to Python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

css_path = Path(__file__).parent / "styles.css"
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from config import DB_PATH
from report.recommendations_builder import (PERCENTILE_EXPLAINER, SIGNAL_EXPLAINER, TIER_BANDS,
                                            UNSCORED_NOTE)
from scoring.model_b import TIERS, sort_by_risk
# Which publication generation is live, and its per-generation caches (audit H3).
from streamlit_dash.data import (ServedGenerationError, current_served, get_history_df,
                                 get_upcoming_df, today_ny, upcoming_view)

# Every view reads the same Model B fields: risk_tier, risk_score (Breakwater Risk
# Percentile — a rank, not a probability), model_b_hist_mean_abs, model_b_n_prior.
# Scores appear ONLY on upcoming calls. Past events show their corrected reaction and
# nothing else: pairing reconstructed scores with realized outcomes would read as a track
# record, which is paused. (Internal retro view: analysis/predictions_range.py.)
MODEL_B_COLUMN_CONFIG = {
    "earnings_date":         st.column_config.DateColumn("Date", format="DD MMM YYYY"),
    "stock":                 st.column_config.TextColumn("Ticker"),
    "sector":                st.column_config.TextColumn("Sector"),
    "sub_sector":            st.column_config.TextColumn("Sub-Sector"),
    "risk_tier":             st.column_config.TextColumn("Risk Tier"),
    "risk_score":            st.column_config.NumberColumn("Risk Percentile", format="%.0f"),
    "model_b_hist_mean_abs": st.column_config.NumberColumn("Avg Past Move %", format="%.1f"),
    "model_b_n_prior":       st.column_config.NumberColumn("Usable Past Reports", format="%d"),
    "days_to_earnings":      st.column_config.NumberColumn("Days", format="%d"),
    "expected_move_pct":     st.column_config.NumberColumn("Options Implied Move % (context)", format="%.1f"),
    "abs_reaction_3d_anchored": st.column_config.NumberColumn("Actual 3-Session Move %", format="%.2f"),
    "is_extreme_move_anchored": st.column_config.CheckboxColumn("Moved ≥ 8%"),
    "announce_window":       st.column_config.TextColumn("Timing"),
}


@st.cache_data(show_spinner=False)
def get_inactive_stocks() -> set:
    """Tickers marked status='inactive' in stock_data (delisted/merged/renamed)."""
    try:
        con = duckdb.connect(str(ROOT / DB_PATH), read_only=True)
        rows = con.execute("SELECT stock FROM stock_data WHERE status = 'inactive'").fetchall()
        con.close()
        return {r[0] for r in rows}
    except Exception:
        return set()


def label_stock(stock: str, inactive: set) -> str:
    return f"{stock} — inactive" if stock in inactive else stock


def ranked(df: pd.DataFrame) -> pd.DataFrame:
    """By date, then Model B risk: scored before Unscored, raw signal descending, ticker."""
    return sort_by_risk(df, by=["earnings_date"])


def as_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["earnings_date"] = out["earnings_date"].dt.date
    for col in ("model_b_hist_mean_abs", "expected_move_pct", "abs_reaction_3d_anchored"):
        if col in out.columns:
            out[col] = out[col] * 100
    if "risk_score" in out.columns:      # floored, so the number shown matches its tier band
        out["risk_score"] = out["risk_score"].apply(lambda v: v if pd.isna(v) else float(int(v)))
    return out


def sidebar_filters(upcoming: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    if upcoming.empty:
        return upcoming
    inactive = get_inactive_stocks()
    stocks = sorted(upcoming["stock"].dropna().unique())
    stock_choice = st.sidebar.selectbox(
        "Stock", options=["(All)"] + stocks,
        format_func=lambda s: s if s == "(All)" else label_stock(s, inactive))
    sectors = sorted(upcoming["sector"].dropna().unique())
    selected_sectors = st.sidebar.multiselect("Sector", options=sectors, default=sectors)
    selected_tiers = st.sidebar.multiselect("Risk tier", options=list(TIERS), default=list(TIERS))
    if stock_choice != "(All)":
        upcoming = upcoming[upcoming["stock"] == stock_choice]
    if selected_sectors:
        upcoming = upcoming[upcoming["sector"].isin(selected_sectors)]
    if selected_tiers:
        upcoming = upcoming[upcoming["risk_tier"].isin(selected_tiers)]
    return upcoming


def main():
    st.title("Breakwater - Earnings Risk Dashboard")
    st.caption(PERCENTILE_EXPLAINER + " " + SIGNAL_EXPLAINER)

    # Read on every page run: a new publication generation is a new cache entry, so the
    # page never shows the previous calls once the pointer has moved.
    try:
        served = current_served()
    except ServedGenerationError as e:
        st.error(f"Data unavailable: {e}")
        return
    today = today_ny()
    all_upcoming = upcoming_view(get_upcoming_df(served.key, str(served.calls_path)), today)
    upcoming = sidebar_filters(all_upcoming)
    history = get_history_df(served.key, str(served.history_path))
    st.sidebar.caption(f"Data generation: {served.generation_id or 'local output/'}")

    tab_upcoming, tab_stock, tab_calendar = st.tabs(
        ["Upcoming Events", "Past Earnings Reactions", "Weekly Calendar"])

    with tab_upcoming:
        st.subheader("Upcoming Earnings Events")
        if upcoming.empty:
            st.info("No upcoming earnings found. Run the pipeline to generate upcoming_df.parquet.")
        else:
            window_days = st.radio(
                "Show events in the next", options=[14, 30, 60, 999],
                format_func=lambda x: "All" if x == 999 else f"{x} days", horizontal=True)
            view = ranked(upcoming[upcoming["days_to_earnings"] <= window_days])
            if view.empty:
                st.info(f"No events in the next {window_days} days.")
            else:
                u1, u2, u3, u4 = st.columns(4)
                u1.metric("Events", len(view))
                u2.metric("High Alert", int((view["risk_tier"] == "High Alert").sum()))
                u3.metric("Elevated", int((view["risk_tier"] == "Elevated").sum()))
                u4.metric("Unscored", int((view["risk_tier"] == "Unscored").sum()))
                cols = [c for c in ["earnings_date", "days_to_earnings", "stock", "sector",
                                    "risk_tier", "risk_score", "model_b_hist_mean_abs",
                                    "model_b_n_prior", "expected_move_pct"] if c in view.columns]
                st.dataframe(as_display(view)[cols], width="stretch",
                             hide_index=True, column_config=MODEL_B_COLUMN_CONFIG)
                st.caption("Tiers — " + " · ".join(f"{t}: {TIER_BANDS[t]}" for t in TIERS)
                           + ". " + UNSCORED_NOTE + " Options implied move is market "
                             "context only; it does not enter the score or the tier.")

    with tab_stock:
        st.subheader("Past earnings reactions")
        st.caption("Each past report's 3-session move, measured from the last close before "
                   "the announcement. Reports without a verified announcement time have no "
                   "corrected move and are not shown.")
        stocks = sorted(history["stock"].dropna().unique()) if not history.empty else []
        if not stocks:
            st.info("No history parquet. Run the pipeline first.")
        else:
            inactive = get_inactive_stocks()
            selected = st.selectbox("Choose stock", options=stocks,
                                    format_func=lambda s: label_stock(s, inactive))
            sdf = history[(history["stock"] == selected)
                          & history["reaction_3d_anchored"].notna()].sort_values("earnings_date")
            if sdf.empty:
                st.info("No past report with a verified announcement time.")
            else:
                st.bar_chart(sdf.set_index("earnings_date")["reaction_3d_anchored"] * 100)
                st.dataframe(as_display(sdf)[["earnings_date", "announce_window",
                                              "abs_reaction_3d_anchored",
                                              "is_extreme_move_anchored"]],
                             width="stretch", hide_index=True,
                             column_config=MODEL_B_COLUMN_CONFIG)

    with tab_calendar:
        st.subheader("Earnings Risk Calendar")
        cal = all_upcoming
        if cal.empty:
            st.info("No upcoming events. Run the pipeline first.")
            return
        c1, c2 = st.columns(2)
        with c1:
            start_date = st.date_input("From", value=today.date(), min_value=today.date())
        with c2:
            end_date = st.date_input("To", value=(today + timedelta(days=14)).date(),
                                     min_value=today.date())
        window = cal[(cal["earnings_date"] >= pd.Timestamp(start_date))
                     & (cal["earnings_date"] <= pd.Timestamp(end_date))].copy()
        if window.empty:
            st.info("No earnings events in the selected window.")
        else:
            window = ranked(window)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Events", len(window))
            k2.metric("High Alert", int((window["risk_tier"] == "High Alert").sum()))
            k3.metric("Elevated", int((window["risk_tier"] == "Elevated").sum()))
            k4.metric("Unscored", int((window["risk_tier"] == "Unscored").sum()))
            cols = ["earnings_date", "stock", "sector", "sub_sector", "risk_tier",
                    "risk_score", "model_b_hist_mean_abs", "model_b_n_prior"]
            st.dataframe(as_display(window)[cols], width="stretch", hide_index=True,
                         column_config=MODEL_B_COLUMN_CONFIG)

            with st.sidebar:
                st.markdown("---")
                if st.button("Export calendar HTML"):
                    from report.calendar_builder import generate_calendar
                    written = generate_calendar(window, reference_date=start_date,
                                                window_days=(end_date - start_date).days)
                    if written:
                        st.success(f"Written to {written}")
                    else:
                        st.warning("No earnings events in window — nothing written.")


if __name__ == "__main__":
    main()
