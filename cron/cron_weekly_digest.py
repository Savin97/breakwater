# data_ingestion/cron_weekly_digest.py
"""
Weekly earnings risk digest — runs Monday 07:00 UTC.
Sends an HTML email to all subscribers listing High Alert and Elevated stocks
with earnings in the next 7 days.

Required .env variables:
    DIGEST_SMTP_HOST   e.g. smtp.gmail.com
    DIGEST_SMTP_PORT   e.g. 587
    DIGEST_SMTP_USER   sender email address
    DIGEST_SMTP_PASS   app password or SMTP password
"""
import os
import smtplib
import pandas as pd
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

PARQUET_PATH = "output/full_df.parquet"
SUBSCRIBERS  = "data/subscribers.txt"
SMTP_HOST    = os.getenv("DIGEST_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("DIGEST_SMTP_PORT", "587"))
SMTP_USER    = os.getenv("DIGEST_SMTP_USER", "")
SMTP_PASS    = os.getenv("DIGEST_SMTP_PASS", "")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_subscribers() -> list[str]:
    if not os.path.exists(SUBSCRIBERS):
        print(f"Subscribers file not found: {SUBSCRIBERS}")
        return []
    with open(SUBSCRIBERS) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _select_stocks(df: pd.DataFrame) -> pd.DataFrame:
    today  = pd.Timestamp.today().normalize()
    cutoff = today + pd.Timedelta(days=7)
    latest = (
        df[df["is_earnings_day"] == 1]
        .sort_values("earnings_date")
        .groupby("stock")
        .last()
        .reset_index()
    )
    mask = (
        latest["earnings_explosiveness_bucket"].isin(["High Alert", "Elevated"]) &
        (latest["earnings_date"] >= today) &
        (latest["earnings_date"] <= cutoff)
    )
    # High Alert first, then by risk_score desc
    return latest[mask].sort_values(
        ["earnings_explosiveness_bucket", "risk_score"],
        ascending=[True, False]
    ).reset_index(drop=True)


def _risk_badge(level: str) -> str:
    colors = {"High Alert": "#c0392b", "Elevated": "#e67e22"}
    bg = colors.get(level, "#999")
    return f'<span style="background:{bg};color:#fff;padding:2px 8px;border-radius:3px;font-size:11px;font-weight:700;">{level}</span>'


def _flag_text(row) -> str:
    parts = []
    drift    = str(row.get("pre_earnings_drift_flag",  "") or "")
    surprise = str(row.get("surprise_momentum_flag",   "") or "")
    if drift:    parts.append(f"Drift: {drift}")
    if surprise: parts.append(f"Pattern: {surprise}")
    return " &nbsp;|&nbsp; ".join(parts) if parts else "&mdash;"


def _iv_cell(row) -> str:
    exp = row.get("expected_move_pct")
    if pd.notna(exp):
        return f"{exp * 100:.1f}%"
    return "<span style='color:#bbb;'>—</span>"


def _build_html(stocks_df: pd.DataFrame, week_of: str) -> str:
    n_ha = (stocks_df["earnings_explosiveness_bucket"] == "High Alert").sum()
    n_el = (stocks_df["earnings_explosiveness_bucket"] == "Elevated").sum()

    rows_html = ""
    for _, r in stocks_df.iterrows():
        hc = r["earnings_explosiveness_bucket"] == "High Alert" and bool(r.get("pre_earnings_drift_flag"))
        ticker_cell = f"{r['stock']} &#9733;" if hc else r['stock']
        rows_html += f"""
        <tr style="border-bottom:1px solid #eee;">
          <td style="padding:8px 12px;font-weight:700;font-size:13px;">{ticker_cell}</td>
          <td style="padding:8px 12px;color:#555;font-size:12px;">{r.get('company_name', r['stock'])}</td>
          <td style="padding:8px 12px;font-size:12px;">{pd.Timestamp(r['earnings_date']).strftime('%b %d')}</td>
          <td style="padding:8px 12px;font-size:12px;color:#555;">{r.get('sector', '')}</td>
          <td style="padding:8px 12px;">{_risk_badge(r['earnings_explosiveness_bucket'])}</td>
          <td style="padding:8px 12px;font-size:12px;font-weight:700;">{r['risk_score']:.0f}</td>
          <td style="padding:8px 12px;font-size:12px;">{_iv_cell(r)}</td>
          <td style="padding:8px 12px;font-size:11px;color:#555;">{_flag_text(r)}</td>
        </tr>"""

    if not rows_html:
        rows_html = '<tr><td colspan="8" style="padding:16px;text-align:center;color:#999;">No High Alert or Elevated events this week.</td></tr>'

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family:Helvetica,Arial,sans-serif;font-size:14px;color:#1a1a1a;max-width:820px;margin:0 auto;padding:24px;">

  <div style="border-bottom:2px solid #1a1a1a;padding-bottom:12px;margin-bottom:20px;">
    <span style="font-size:20px;font-weight:700;">Breakwater</span>
    <span style="font-size:13px;color:#555;margin-left:12px;">Earnings Risk Digest</span>
    <span style="float:right;font-size:12px;color:#888;">Week of {week_of}</span>
  </div>

  <div style="background:#f5f5f5;border-radius:4px;padding:12px 16px;margin-bottom:20px;font-size:13px;">
    <strong>{n_ha} High Alert</strong> &nbsp;&nbsp; {n_el} Elevated &nbsp;&nbsp;
    <span style="color:#888;">earnings events in the next 7 days</span>
    &nbsp;&nbsp;&nbsp; &#9733; = High Conviction (High Alert + pre-earnings drift)
  </div>

  <table style="width:100%;border-collapse:collapse;">
    <thead>
      <tr style="background:#1a1a1a;color:#fff;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;">
        <th style="padding:8px 12px;text-align:left;">Stock</th>
        <th style="padding:8px 12px;text-align:left;">Company</th>
        <th style="padding:8px 12px;text-align:left;">Earnings</th>
        <th style="padding:8px 12px;text-align:left;">Sector</th>
        <th style="padding:8px 12px;text-align:left;">Risk Level</th>
        <th style="padding:8px 12px;text-align:left;">Score</th>
        <th style="padding:8px 12px;text-align:left;">Impl. Move</th>
        <th style="padding:8px 12px;text-align:left;">Flags</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <div style="margin-top:28px;padding-top:12px;border-top:1px solid #ddd;font-size:11px;color:#999;text-align:center;">
    Breakwater &mdash;
    <a href="https://harbor-markets.com/breakwater" style="color:#999;">harbor-markets.com/breakwater</a>
    &mdash; Not financial advice. For informational purposes only.
  </div>

</body>
</html>"""


def _send(recipients: list[str], subject: str, html: str):
    if not SMTP_USER or not SMTP_PASS:
        print("SMTP credentials not set — printing digest to stdout.\n")
        print(f"To: {recipients}\nSubject: {subject}\n")
        print(html)
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = ", ".join(recipients)
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, recipients, msg.as_string())
    print(f"Digest sent to {len(recipients)} subscriber(s).")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_weekly_digest():
    if not os.path.exists(PARQUET_PATH):
        print(f"Parquet not found: {PARQUET_PATH} — run the pipeline first.")
        return

    df         = pd.read_parquet(PARQUET_PATH)
    stocks_df  = _select_stocks(df)
    week_of    = date.today().strftime("%B %d, %Y")
    subject    = f"Breakwater — Earnings Risk Digest — Week of {week_of}"
    html       = _build_html(stocks_df, week_of)
    recipients = _load_subscribers()

    print(f"Digest: {len(stocks_df)} stocks selected, {len(recipients)} subscriber(s).")

    if not recipients:
        print("No subscribers — printing digest to stdout.\n")
        print(html)
        return

    _send(recipients, subject, html)


if __name__ == "__main__":
    run_weekly_digest()
