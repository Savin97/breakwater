# cron/cron_weekly_digest.py
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

PARQUET_PATH   = "output/full_df.parquet"
COMPANY_NAMES  = "data/sp500_full_info.csv"
SUBSCRIBERS    = "data/subscribers.txt"
SMTP_HOST      = os.getenv("DIGEST_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("DIGEST_SMTP_PORT", "587"))
SMTP_USER      = os.getenv("DIGEST_SMTP_USER", "")
SMTP_PASS      = os.getenv("DIGEST_SMTP_PASS", "")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _load_subscribers() -> list[str]:
    if not os.path.exists(SUBSCRIBERS):
        print(f"Subscribers file not found: {SUBSCRIBERS}")
        return []
    with open(SUBSCRIBERS) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _load_company_names() -> dict:
    if not os.path.exists(COMPANY_NAMES):
        return {}
    return pd.read_csv(COMPANY_NAMES, usecols=["ticker", "name"]).set_index("ticker")["name"].to_dict()


def _select_stocks(df: pd.DataFrame, company_names: dict) -> pd.DataFrame:
    today  = pd.Timestamp.today().normalize()
    cutoff = today + pd.Timedelta(days=7)
    all_latest = (
        df[df["is_earnings_day"] == 1]
        .sort_values("earnings_date")
        .groupby("stock")
        .last()
        .reset_index()
    )
    # Percentile from raw continuous score — sort ties by raw score
    all_latest["peer_percentile"] = (
        all_latest["earnings_explosiveness_score"].rank(pct=True) * 100
    ).fillna(0).astype(int)
    all_latest["company_name"] = all_latest["stock"].map(company_names)

    mask = (
        all_latest["earnings_explosiveness_bucket"].isin(["High Alert", "Elevated"]) &
        (all_latest["earnings_date"] >= today) &
        (all_latest["earnings_date"] <= cutoff)
    )
    return (
        all_latest[mask]
        .sort_values(["peer_percentile", "earnings_explosiveness_score"], ascending=[False, False])
        .reset_index(drop=True)
    )


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


def _is_hc(row) -> bool:
    return (
        row["earnings_explosiveness_bucket"] == "High Alert" and
        bool(row.get("pre_earnings_drift_flag")) and
        str(row.get("pre_earnings_drift_flag", "")).strip() != ""
    )


def _table_row(r, has_iv: bool, bg: str = "") -> str:
    hc = _is_hc(r)
    ticker_cell  = f"<strong>{r['stock']}</strong> &#9733;" if hc else f"<strong>{r['stock']}</strong>"
    company_cell = r.get("company_name") or "&mdash;"
    iv_td        = f'<td style="padding:8px 12px;font-size:12px;">{r["expected_move_pct"]*100:.1f}%</td>' if has_iv else ""
    row_style    = f'background:{bg};' if bg else ""
    return f"""
    <tr style="border-bottom:1px solid #e0e0e0;{row_style}">
      <td style="padding:8px 12px;font-size:13px;">{ticker_cell}</td>
      <td style="padding:8px 12px;color:#555;font-size:12px;">{company_cell}</td>
      <td style="padding:8px 12px;font-size:12px;">{pd.Timestamp(r['earnings_date']).strftime('%b %d')}</td>
      <td style="padding:8px 12px;font-size:12px;color:#555;">{r.get('sector', '')}</td>
      <td style="padding:8px 12px;">{_risk_badge(r['earnings_explosiveness_bucket'])}</td>
      <td style="padding:8px 12px;font-size:12px;font-weight:700;">{_ordinal(int(r['peer_percentile']))}</td>
      {iv_td}
      <td style="padding:8px 12px;font-size:11px;color:#555;">{_flag_text(r)}</td>
    </tr>"""


def _build_html(stocks_df: pd.DataFrame, week_of: str, date_range: str = "") -> str:
    n_total = len(stocks_df)
    n_ha    = int((stocks_df["earnings_explosiveness_bucket"] == "High Alert").sum())
    n_el    = int((stocks_df["earnings_explosiveness_bucket"] == "Elevated").sum())
    hc_mask = stocks_df.apply(_is_hc, axis=1)
    n_hc    = int(hc_mask.sum())
    has_iv  = stocks_df["expected_move_pct"].notna().any()

    iv_th   = '<th style="padding:8px 12px;text-align:left;">Impl. Move</th>' if has_iv else ""
    colspan = 8 if has_iv else 7

    # ── High Conviction section ───────────────────────────────────────────────
    hc_section = ""
    if n_hc:
        hc_rows = ""
        for _, r in stocks_df[hc_mask].iterrows():
            hc_rows += _table_row(r, has_iv, bg="#fdf6f0")
        hc_section = f"""
  <div style="margin-bottom:24px;">
    <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#c0392b;margin-bottom:8px;">
      High Conviction &#9733; — {n_hc} event{"s" if n_hc != 1 else ""}
    </div>
    <table style="width:100%;border-collapse:collapse;border:1px solid #f0d8cc;border-radius:3px;">
      <thead>
        <tr style="background:#c0392b;color:#fff;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;">
          <th style="padding:7px 12px;text-align:left;">Stock</th>
          <th style="padding:7px 12px;text-align:left;">Company</th>
          <th style="padding:7px 12px;text-align:left;">Earnings</th>
          <th style="padding:7px 12px;text-align:left;">Sector</th>
          <th style="padding:7px 12px;text-align:left;">Risk Level</th>
          <th style="padding:7px 12px;text-align:left;">Percentile</th>
          {iv_th}
          <th style="padding:7px 12px;text-align:left;">Flags</th>
        </tr>
      </thead>
      <tbody>{hc_rows}</tbody>
    </table>
  </div>"""

    # ── Full table ────────────────────────────────────────────────────────────
    rows_html = ""
    for _, r in stocks_df.iterrows():
        rows_html += _table_row(r, has_iv)

    if not rows_html:
        rows_html = f'<tr><td colspan="{colspan}" style="padding:16px;text-align:center;color:#999;">No High Alert or Elevated events this week.</td></tr>'

    date_line = date_range if date_range else f"Week of {week_of}"
    hc_pill   = f' &nbsp;·&nbsp; <strong style="color:#c0392b;">High Conviction &#9733; &mdash; {n_hc} event{"s" if n_hc != 1 else ""}</strong> &nbsp;·&nbsp; <span style="color:#888;">&#9733; = High Conviction (High Alert + pre-earnings drift)</span>' if n_hc else ""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family:Helvetica,Arial,sans-serif;font-size:14px;color:#1a1a1a;max-width:820px;margin:0 auto;padding:24px;">

  <div style="border-bottom:2px solid #1a1a1a;padding-bottom:12px;margin-bottom:20px;">
    <div style="font-size:20px;font-weight:700;">Breakwater <span style="font-size:13px;font-weight:400;color:#555;">Earnings Risk Digest</span></div>
    <div style="font-size:12px;color:#888;margin-top:3px;">{date_line}</div>
  </div>

  <div style="background:#f5f5f5;border-radius:4px;padding:12px 16px;margin-bottom:24px;font-size:13px;">
    <strong>{n_total} flagged earnings events</strong>
    &nbsp;·&nbsp; {n_ha} High Alert
    &nbsp;·&nbsp; {n_el} Elevated
    {hc_pill}
  </div>

  {hc_section}

  <div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:0.6px;color:#555;margin-bottom:8px;">
    All flagged events
  </div>
  <table style="width:100%;border-collapse:collapse;">
    <thead>
      <tr style="background:#1a1a1a;color:#fff;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;">
        <th style="padding:8px 12px;text-align:left;">Stock</th>
        <th style="padding:8px 12px;text-align:left;">Company</th>
        <th style="padding:8px 12px;text-align:left;">Earnings</th>
        <th style="padding:8px 12px;text-align:left;">Sector</th>
        <th style="padding:8px 12px;text-align:left;">Risk Level</th>
        <th style="padding:8px 12px;text-align:left;">Percentile</th>
        {iv_th}
        <th style="padding:8px 12px;text-align:left;">Flags</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <div style="margin-top:16px;font-size:11px;color:#999;line-height:1.7;">
    <strong>Percentile:</strong> Historical earnings-risk rank versus the S&amp;P 500 universe,
    based only on information available before the announcement.
    A 97th-percentile event ranks as riskier than approximately 97% of comparable historical events.<br>
    <strong>High Conviction &#9733;:</strong> High Alert with an active pre-earnings drift signal.
  </div>

  <div style="margin-top:20px;padding-top:12px;border-top:1px solid #ddd;font-size:11px;color:#999;text-align:center;">
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

    df            = pd.read_parquet(PARQUET_PATH)
    company_names = _load_company_names()
    stocks_df     = _select_stocks(df, company_names)
    today_dt      = date.today()
    cutoff_dt     = today_dt + pd.Timedelta(days=7).to_pytimedelta()
    week_of       = today_dt.strftime("%B %d, %Y")
    date_range    = f"{today_dt.strftime('%b %d')} – {cutoff_dt.strftime('%b %d, %Y')}"
    subject       = f"Breakwater — Earnings Risk Digest — {date_range}"
    html          = _build_html(stocks_df, week_of, date_range=date_range)
    recipients    = _load_subscribers()

    print(f"Digest: {len(stocks_df)} stocks selected, {len(recipients)} subscriber(s).")

    if not recipients:
        print("No subscribers — printing digest to stdout.\n")
        print(html)
        return

    _send(recipients, subject, html)


if __name__ == "__main__":
    run_weekly_digest()
