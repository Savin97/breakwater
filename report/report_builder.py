# report/report_builder.py
from datetime import date
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from weasyprint import HTML
from pathlib import Path
import pandas as pd

from report.chart_builder import generate_reactions_chart
from report.recommendations_builder import build_recommendation
from utilities.output_utilities import get_run_output_dir

def generate_report(stock, data):
    project_root = Path(__file__).resolve().parents[1]
    env = Environment(
        loader=FileSystemLoader("report/templates"),
        undefined=StrictUndefined,
        autoescape=True,)
    template = env.get_template("earnings_report.html")

    # Cover Page
    html_out = template.render(
        stock = stock,
        company_name = data.get("company_name", ""),
        earnings_date = data["earnings_date"],
        generated_date = data.get("generated_date", ""),
        risk_level = data["risk_level"],
        risk_score = data["risk_score"],
        hist_extreme_prob = data["hist_extreme_prob"],
        base_extreme_prob = data["base_extreme_prob"],
        current_lift_vs_baseline = data["current_lift_vs_baseline"],
        current_lift_vs_same_bucket_global = data["current_lift_vs_same_bucket_global"],
        bucket_table = data["bucket_table"],
        sector = data["sector"],
        sub_sector = data["sub_sector"],
        surprise_flag        = data.get("surprise_flag", ""),
        drift_flag           = data.get("drift_flag", ""),
        high_conviction      = data.get("high_conviction", False),
        recommendation       = data.get("recommendation", {}),
        peer_percentile      = data.get("peer_percentile"),
        days_to_earnings     = data.get("days_to_earnings"),
        reactions_chart_svg  = data.get("reactions_chart_svg", ""),
        iv_implied_move_pct  = data.get("iv_implied_move_pct"),
        atm_iv_pct           = data.get("atm_iv_pct"),
        iv_vs_hist_ratio     = data.get("iv_vs_hist_ratio"),
    )

    reports_dir = Path(get_run_output_dir()) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT_PATH = reports_dir / f"{stock}_report.pdf"
    HTML(string=html_out, base_url=project_root).write_pdf(REPORT_OUTPUT_PATH)


def generate_reports(df):
    today  = pd.Timestamp.today().normalize()
    cutoff = today + pd.Timedelta(days=7)
    latest_per_stock = df.sort_values("date").groupby("stock").last().reset_index()
    mask = (latest_per_stock["earnings_date"] >= today) & (latest_per_stock["earnings_date"] <= cutoff)

    stocks_to_report_for = latest_per_stock[mask].sort_values("risk_score", ascending=False)["stock"].tolist()
    if not stocks_to_report_for:
        raise ValueError(f"No stocks found to report for")
    
    hc       = latest_per_stock[mask & latest_per_stock["is_high_conviction"]].sort_values("risk_score", ascending=False)["stock"].tolist()
    ha       = latest_per_stock[mask & (latest_per_stock["earnings_explosiveness_bucket"] == "High Alert") & ~latest_per_stock["is_high_conviction"]].sort_values("risk_score", ascending=False)["stock"].tolist()
    elevated = latest_per_stock[mask & (latest_per_stock["earnings_explosiveness_bucket"] == "Elevated")].sort_values("risk_score", ascending=False)["stock"].tolist()
    print(f"{len(stocks_to_report_for)} stocks this week:\n*** HC: {hc or 'none'}\n** HA: {ha or 'none'}\n* Elevated: {elevated or 'none'}")

    latest_per_stock_idx = latest_per_stock.set_index("stock")
    
    company_names  = pd.read_csv("data/sp500_full_info.csv", usecols=["ticker", "name"]).set_index("ticker")["name"]
    generated_date = date.today().strftime("%B %d, %Y")
    df_by_stock    = {s: grp for s, grp in df.groupby("stock")}

    global_earnings_df     = df[df["is_earnings_day"] == 1].copy()
    P_extreme_global       = global_earnings_df["is_extreme_reaction"].mean()
    P_extreme_given_bucket = global_earnings_df.groupby("earnings_explosiveness_bucket")["is_extreme_reaction"].mean()
    bucket_stats           = pd.DataFrame({"global_hist_prob": P_extreme_given_bucket})

    rank_key         = latest_per_stock_idx["abs_reaction_p75_rolling"].fillna(latest_per_stock_idx["abs_reaction_p75"])
    peer_percentiles = rank_key.rank(pct=True).fillna(0)

    print(f"Generating reports: {', '.join(stocks_to_report_for)}")
    for stock in stocks_to_report_for:
        stock_df = df_by_stock.get(stock)
        if stock_df is None or stock_df.empty:
            print(f"  {stock}: no data, skipping.")
            continue
        earnings_df = stock_df[stock_df["is_earnings_day"] == 1]
        if earnings_df.empty:
            print(f"  {stock}: no earnings rows, skipping.")
            continue

        latest_row     = earnings_df.iloc[-1]
        current_bucket = latest_row["earnings_explosiveness_bucket"]
        if not isinstance(current_bucket, str):
            latest_row     = earnings_df.iloc[-2]
            current_bucket = latest_row["earnings_explosiveness_bucket"]

        prior_strength = 20
        eb = (
            earnings_df.groupby("earnings_explosiveness_bucket")["is_extreme_reaction"]
            .agg(extreme_count="sum", event_count="count")
        )
        eb["shrunk_prob"]               = (eb["extreme_count"] + prior_strength * P_extreme_global) / (eb["event_count"] + prior_strength)
        eb["global_hist_prob"]           = bucket_stats.loc[eb.index, "global_hist_prob"]
        eb["lift_vs_baseline"]           = eb["shrunk_prob"] / P_extreme_global
        eb["lift_vs_same_bucket_global"] = eb["shrunk_prob"] / eb["global_hist_prob"]

        upcoming_date   = pd.Timestamp(latest_per_stock_idx.loc[stock, "earnings_date"])
        surprise_flag   = str(latest_row.get("surprise_momentum_flag", "") or "")
        drift_flag      = str(latest_row.get("pre_earnings_drift_flag",  "") or "")
        high_conviction = bool(latest_per_stock_idx.loc[stock, "is_high_conviction"])

        # The lift-based tier promotion that used to live here now runs in stage4
        # (engineer_lift_adjusted_bucket), so current_bucket already reflects it and
        # every surface — dashboard, digest, parquet — agrees with this report.
        # The old local version applied only to PDFs, never moved risk_score, and
        # computed its lift from the stock's entire history including events after
        # the row being scored.
        current_bucket_prob                = f"{eb.loc[current_bucket, 'shrunk_prob']:.3f}"
        current_lift_vs_baseline           = f"{eb.loc[current_bucket, 'lift_vs_baseline']:.3f}"
        current_lift_vs_same_bucket_global = f"{eb.loc[current_bucket, 'lift_vs_same_bucket_global']:.3f}"

        bucket_table_html = (
            eb.reset_index()
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

        _exp_move = latest_row.get("expected_move_pct")
        _atm_iv   = latest_row.get("atm_iv")
        _p75      = latest_row.get("abs_reaction_p75_rolling")

        report_data = {
            "earnings_date":                      upcoming_date.strftime("%B %d, %Y"),
            "company_name":                       company_names.get(stock, ""),
            "generated_date":                     generated_date,
            "risk_level":                         current_bucket,
            "risk_score":                         f"{latest_row['risk_score']:.0f}",
            "sector":                             latest_row.get("sector", ""),
            "sub_sector":                         latest_row.get("sub_sector", ""),
            "n_events":                           len(earnings_df),
            "base_extreme_prob":                  round(P_extreme_global, 3),
            "hist_extreme_prob":                  current_bucket_prob,
            "current_lift_vs_baseline":           current_lift_vs_baseline,
            "current_lift_vs_same_bucket_global": current_lift_vs_same_bucket_global,
            "bucket_table":                       bucket_table_html,
            "surprise_flag":                      surprise_flag,
            "drift_flag":                         drift_flag,
            "high_conviction":                    high_conviction,
            "recommendation":                     build_recommendation(
                risk_level=current_bucket,
                hist_extreme_prob=current_bucket_prob,
                base_extreme_prob=round(P_extreme_global, 3),
                lift=current_lift_vs_baseline,
                surprise_flag=surprise_flag,
                drift_flag=drift_flag,
                high_conviction=high_conviction,
                stock=stock,
                earnings_date=upcoming_date.strftime("%B %d, %Y"),
            ),
            "peer_percentile":                    int(peer_percentiles.loc[stock] * 100),
            "days_to_earnings":                   (upcoming_date.date() - date.today()).days,
            "reactions_chart_svg":                generate_reactions_chart(earnings_df),
            "iv_implied_move_pct":                float(_exp_move) if pd.notna(_exp_move) else None,
            "atm_iv_pct":                         float(_atm_iv)   if pd.notna(_atm_iv)   else None,
            "iv_vs_hist_ratio":                   round(float(_exp_move) / float(_p75), 2)
                                                  if pd.notna(_exp_move) and pd.notna(_p75) and float(_p75) > 0
                                                  else None,
        }
        generate_report(stock, report_data)
