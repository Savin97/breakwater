# report/report_builder.py
from datetime import date
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from weasyprint import HTML
from pathlib import Path
import pandas as pd

from report.chart_builder import generate_reactions_chart, reactions_chart_summary
from report.recommendations_builder import (build_recommendation, PERCENTILE_EXPLAINER,
                                            SIGNAL_EXPLAINER, TIER_BANDS)
from scoring.model_b import display_score, sort_by_risk, MIN_PRIOR
from utilities.output_utilities import get_run_output_dir
from pipeline.events import pending_events


def render_report_html(stock, data):
    """The report's HTML, exactly as it is turned into the PDF."""
    env = Environment(
        loader=FileSystemLoader("report/templates"),
        undefined=StrictUndefined,
        autoescape=True,)
    template = env.get_template("earnings_report.html")
    return template.render(stock=stock, **data)


def generate_report(stock, data):
    project_root = Path(__file__).resolve().parents[1]
    html_out = render_report_html(stock, data)
    reports_dir = Path(get_run_output_dir()) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    HTML(string=html_out, base_url=project_root).write_pdf(reports_dir / f"{stock}_report.pdf")


def _fmt_date(d) -> str:
    return pd.Timestamp(d).strftime("%B %d, %Y")


def build_report_data(row: pd.Series, history: pd.DataFrame, company_name: str = "",
                      today=None) -> dict:
    """Everything the PDF template shows for one pending event — Model B fields from the
    event row, the corrected reaction chart from the stock's completed events, and the
    options snapshot as separate market context."""
    today = pd.Timestamp(today if today is not None else date.today()).normalize()
    upcoming = pd.Timestamp(row["earnings_date"])
    tier = str(row["risk_tier"])
    hist_abs = row.get("model_b_hist_mean_abs")
    n_prior = int(row["model_b_n_prior"])

    exp_move, atm_iv = row.get("expected_move_pct"), row.get("atm_iv")
    iv_date = row.get("iv_snapshot_date")
    has_iv = pd.notna(exp_move)
    return {
        "company_name":        company_name or "",
        "generated_date":      today.strftime("%B %d, %Y"),
        "earnings_date":       _fmt_date(upcoming),
        "days_to_earnings":    (upcoming - today).days,
        "sector":              row.get("sector", "") or "",
        "sub_sector":          row.get("sub_sector", "") or "",
        "risk_tier":           tier,
        "risk_tier_css":       tier.lower().replace(" ", "-"),
        "risk_score":          display_score(row.get("risk_score")),
        "tier_band":           TIER_BANDS[tier],
        "hist_mean_abs_pct":   (f"{hist_abs * 100:.1f}%" if pd.notna(hist_abs) else None),
        "n_prior":             n_prior,
        "min_prior":           MIN_PRIOR,
        "score_asof_date":     _fmt_date(row["score_asof_date"])
                               if pd.notna(row.get("score_asof_date")) else None,
        "percentile_explainer": PERCENTILE_EXPLAINER,
        "signal_explainer":    SIGNAL_EXPLAINER,
        "tier_bands":          TIER_BANDS,
        "recommendation":      build_recommendation(tier, row["stock"], _fmt_date(upcoming),
                                                    n_prior, hist_abs),
        "reactions_chart_svg": generate_reactions_chart(history),
        "chart_summary":       reactions_chart_summary(history),
        "iv_implied_move_pct": float(exp_move) if has_iv else None,
        "atm_iv_pct":          float(atm_iv) if has_iv and pd.notna(atm_iv) else None,
        "iv_snapshot_date":    _fmt_date(iv_date) if has_iv and pd.notna(iv_date) else None,
        "iv_snapshot_age_days": (today - pd.Timestamp(iv_date).normalize()).days
                                if has_iv and pd.notna(iv_date) else None,
        "model_version":       row.get("model_version", ""),
    }


def generate_reports(events_df, today=None):
    """Per-stock PDF reports for events reporting in the next 7 days, read from the
    PENDING rows of the Model B-scored event frame. Riskiest first; Unscored
    events get a report too, which says they are unscored and why."""
    today = pd.Timestamp(today if today is not None else date.today()).normalize()
    cutoff = today + pd.Timedelta(days=7)
    pend = pending_events(events_df)
    week = pend[(pend["earnings_date"] >= today) & (pend["earnings_date"] <= cutoff)].copy()
    if week.empty:
        raise ValueError("No stocks found to report for")
    week = sort_by_risk(week)

    by_tier = {t: g["stock"].tolist() for t, g in week.groupby("risk_tier", sort=False)}
    print(f"{len(week)} stocks this week: "
          + "; ".join(f"{t}: {', '.join(s)}" for t, s in by_tier.items()))

    company_names = pd.read_csv("data/sp500_full_info.csv",
                                usecols=["ticker", "name"]).set_index("ticker")["name"]
    completed = events_df[~events_df["is_pending"]]
    history_by_stock = {s: g for s, g in completed.groupby("stock")}
    empty = completed.iloc[0:0]

    print(f"Generating reports: {', '.join(week['stock'])}")
    for _, row in week.iterrows():
        stock = row["stock"]
        data = build_report_data(row, history_by_stock.get(stock, empty),
                                 company_names.get(stock, ""), today)
        generate_report(stock, data)
