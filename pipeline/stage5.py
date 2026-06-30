# pipeline/stage5.py
import os
from datetime import date
import pandas as pd
from report.report_builder import generate_report
from report.calendar_builder import generate_calendar
from report.recommendations_builder import build_recommendation
from report.chart_builder import generate_reactions_chart
from streamlit_dash.streamlit_export import export_streamlit_df

def stage5(df):
    print("--------------------\nStage 5 - Generating Report...")

    # Manual override: populate data/report_stocks.csv to generate reports for specific stocks
    # report_stocks_path = "data/report_stocks.csv"
    # if os.path.exists(report_stocks_path):
    #     stocks_to_report_for = pd.read_csv(report_stocks_path)["stock"].tolist()
    # else:
    #     stocks_to_report_for = []
    #     print("Warning: data/report_stocks.csv not found. No reports generated.")

    today = pd.Timestamp.today().normalize()
    cutoff = today + pd.Timedelta(days=7)
    latest_per_stock = (
        df.sort_values("date")
        .groupby("stock")
        .last()
        .reset_index()
    )
    mask = (
        (latest_per_stock["earnings_date"] >= today) &
        (latest_per_stock["earnings_date"] <= cutoff)
    )
    stocks_to_report_for = (
        latest_per_stock[mask]
        .sort_values("risk_score", ascending=False)["stock"]
        .tolist()
    )
    hc       = latest_per_stock[mask & latest_per_stock["is_high_conviction"]].sort_values("risk_score", ascending=False)["stock"].tolist()
    ha       = latest_per_stock[mask & (latest_per_stock["earnings_explosiveness_bucket"] == "High Alert") & ~latest_per_stock["is_high_conviction"]].sort_values("risk_score", ascending=False)["stock"].tolist()
    elevated = latest_per_stock[mask & (latest_per_stock["earnings_explosiveness_bucket"] == "Elevated")].sort_values("risk_score", ascending=False)["stock"].tolist()
    print(f"  {len(stocks_to_report_for)} stocks this week — *** HC: {hc or 'none'}  ** HA: {ha or 'none'}  * Elevated: {elevated or 'none'}")

    # Indexed for per-stock upcoming-date lookup; scores for universe-wide percentile (matches digest)
    latest_per_stock_idx = latest_per_stock.set_index("stock")
    all_latest_scores = latest_per_stock_idx["earnings_explosiveness_score"].dropna()

    company_names = pd.read_csv("data/sp500_full_info.csv", usecols=["ticker", "name"]).set_index("ticker")["name"]
    generated_date = date.today().strftime("%B %d, %Y")

    global_earnings_df = df[df["is_earnings_day"] == 1].copy()
    P_extreme_global = global_earnings_df["is_extreme_reaction"].mean()

    P_extreme_given_bucket = (
        global_earnings_df.groupby("earnings_explosiveness_bucket")["is_extreme_reaction"]
        .mean()
    )
    bucket_stats = pd.DataFrame({
        "global_hist_prob": P_extreme_given_bucket,
        "global_risk_lift_vs_baseline": P_extreme_given_bucket / P_extreme_global
    })

    # Pre-group by stock so the loop doesn't scan 2.8M rows per iteration
    df_by_stock = {s: grp for s, grp in df.groupby("stock")}

    for stock in stocks_to_report_for:
        stock_df = df_by_stock.get(stock)
        if stock_df is None or stock_df.empty:
            print(f"  No data found for {stock}, skipping.")
            continue
        earnings_df = stock_df[stock_df["is_earnings_day"] == 1]
        if earnings_df.empty:
            print(f"  No earnings events found for {stock}, skipping.")
            continue

        latest_row = earnings_df.iloc[-1]
        prior_strength = 20

        earnings_explosiveness_buckets = (
            earnings_df.groupby("earnings_explosiveness_bucket")["is_extreme_reaction"]
            .agg(extreme_count="sum", event_count="count")
        )
        earnings_explosiveness_buckets["shrunk_prob"] = (
            earnings_explosiveness_buckets["extreme_count"] +
            prior_strength * P_extreme_global
        ) / (
            earnings_explosiveness_buckets["event_count"] + prior_strength
        )
        earnings_explosiveness_buckets["global_hist_prob"] = bucket_stats.loc[earnings_explosiveness_buckets.index, "global_hist_prob"]
        earnings_explosiveness_buckets["lift_vs_baseline"] = (
            earnings_explosiveness_buckets["shrunk_prob"] / P_extreme_global
        )
        earnings_explosiveness_buckets["lift_vs_same_bucket_global"] = (
            earnings_explosiveness_buckets["shrunk_prob"] / earnings_explosiveness_buckets["global_hist_prob"]
        )

        current_bucket = latest_row["earnings_explosiveness_bucket"]
        if not isinstance(current_bucket, str):
            latest_row = earnings_df.iloc[-2]
            current_bucket = latest_row["earnings_explosiveness_bucket"]

        risk_score = f"{latest_row['risk_score']:.0f}"
        upcoming_date = pd.Timestamp(latest_per_stock_idx.loc[stock, "earnings_date"])
        current_earnings_date = upcoming_date.strftime("%B %d, %Y")
        sector = latest_row.get("sector", "")
        sub_sector = latest_row.get("sub_sector", "")
        company_name = company_names.get(stock, "")
        surprise_flag = str(latest_row.get("surprise_momentum_flag", "") or "")
        drift_flag    = str(latest_row.get("pre_earnings_drift_flag",  "") or "")
        high_conviction = (current_bucket == "High Alert") and bool(drift_flag)
        n_events = len(earnings_df)

        _rank_key = latest_per_stock_idx["abs_reaction_p75_rolling"].fillna(
            latest_per_stock_idx["abs_reaction_p75"]
        )
        peer_percentile = int(_rank_key.rank(pct=True).fillna(0).loc[stock] * 100)

        days_to_earnings = (upcoming_date.date() - date.today()).days

        # IV data (may be NaN if cron hasn't run yet for this stock)
        _exp_move = latest_row.get("expected_move_pct")
        _atm_iv   = latest_row.get("atm_iv")
        _p75      = latest_row.get("abs_reaction_p75_rolling")
        iv_implied_move_pct = float(_exp_move) if pd.notna(_exp_move) else None
        atm_iv_pct          = float(_atm_iv)   if pd.notna(_atm_iv)   else None
        iv_vs_hist_ratio    = (
            round(float(_exp_move) / float(_p75), 2)
            if pd.notna(_exp_move) and pd.notna(_p75) and float(_p75) > 0
            else None
        )

        reactions_chart_svg = generate_reactions_chart(earnings_df)
        P_extreme_global_rounded = round(P_extreme_global, 3)
        current_bucket_prob = f"{earnings_explosiveness_buckets.loc[current_bucket, 'shrunk_prob']:.3f}"
        current_lift_vs_baseline = f"{earnings_explosiveness_buckets.loc[current_bucket, 'lift_vs_baseline']:.3f}"
        current_lift_vs_same_bucket_global = f"{earnings_explosiveness_buckets.loc[current_bucket, 'lift_vs_same_bucket_global']:.3f}"
        earnings_explosiveness_buckets = earnings_explosiveness_buckets.reset_index()

        # Bayesian override: if the stock's actual extreme-move rate materially exceeds
        # what its model bucket implies, bump the reported risk level up.
        lift_for_report = float(current_bucket_prob) / float(P_extreme_global)
        if current_bucket == "Normal" and lift_for_report >= 1.5:
            effective_risk_level = "Elevated"
        elif current_bucket in ("Normal", "Elevated") and lift_for_report >= 3.0:
            effective_risk_level = "High Alert"
        else:
            effective_risk_level = current_bucket

        bucket_table_html = (
            earnings_explosiveness_buckets
            .drop(columns=["extreme_count"])
            .rename(columns={
                "earnings_explosiveness_bucket": "Risk Bucket",
                "event_count":                   "Events",
                "shrunk_prob":                   "Hist. Prob.",
                "global_hist_prob":              "Global Prob.",
                "lift_vs_baseline":              "Lift vs Baseline",
                "lift_vs_same_bucket_global":    "Lift vs Peers",
            })
            .to_html(index=False, classes="bucket-table", float_format=lambda x: f"{x:.3f}")
        )

        recommendation = build_recommendation(
            risk_level        = effective_risk_level,
            hist_extreme_prob = current_bucket_prob,
            base_extreme_prob = P_extreme_global_rounded,
            lift              = current_lift_vs_baseline,
            surprise_flag     = surprise_flag,
            drift_flag        = drift_flag,
            high_conviction   = high_conviction,
            stock             = stock,
            earnings_date     = current_earnings_date,
        )

        data_for_report = {
            "earnings_date":    current_earnings_date,
            "company_name":     company_name,
            "generated_date":   generated_date,
            "risk_level": effective_risk_level,
            "risk_score": risk_score,
            "sector": sector,
            "sub_sector": sub_sector,
            "n_events": n_events,
            "base_extreme_prob": P_extreme_global_rounded,
            "hist_extreme_prob": current_bucket_prob,
            "current_lift_vs_baseline": current_lift_vs_baseline,
            "current_lift_vs_same_bucket_global": current_lift_vs_same_bucket_global,
            "bucket_table": bucket_table_html,
            "surprise_flag":       surprise_flag,
            "drift_flag":          drift_flag,
            "high_conviction":     high_conviction,
            "recommendation":      recommendation,
            "peer_percentile":     peer_percentile,
            "days_to_earnings":    days_to_earnings,
            "reactions_chart_svg":   reactions_chart_svg,
            "iv_implied_move_pct":   iv_implied_move_pct,
            "atm_iv_pct":            atm_iv_pct,
            "iv_vs_hist_ratio":      iv_vs_hist_ratio,
        }
        generate_report(stock, data_for_report)

    print("--------------------")
    generate_calendar(df)
    export_streamlit_df(df)
    print("Stage 5 DONE")
    return df
