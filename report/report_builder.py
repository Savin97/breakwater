# report/report_builder.py
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from weasyprint import HTML
from pathlib import Path

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
    )

    REPORT_OUTPUT_PATH = f"output/{stock}_report.pdf"
    HTML(string=html_out, base_url=project_root).write_pdf(REPORT_OUTPUT_PATH)
    print(f"Report created in {REPORT_OUTPUT_PATH}")
