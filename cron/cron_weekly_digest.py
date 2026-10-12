# cron/cron_weekly_digest.py
"""
Weekly earnings risk digest. Sends an HTML email to all subscribers listing
High Alert and Elevated stocks (Model B tiers) with earnings in the next work week.

Runs LOCALLY, as the last step of scripts/full_workflow.sh — not on the droplet.
Two things are only available here: the PDF reports it attaches (written by stage5,
which the droplet does not run) and the DIGEST_SMTP_* credentials. Triggering it from
the workflow also removes a race: a Monday cron could fire before the weekly run and
mail out last week's numbers with nothing to indicate it had.

Required .env variables:
    DIGEST_SMTP_HOST   e.g. smtp.gmail.com
    DIGEST_SMTP_PORT   e.g. 587
    DIGEST_SMTP_USER   sender email address
    DIGEST_SMTP_PASS   app password or SMTP password
"""
import os
import smtplib
import time
import pandas as pd
from datetime import date
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from dotenv import load_dotenv

from utilities.output_utilities import latest_run_output_dir
from publishing.guard import PublicationRefused, require_production_branch
from utilities.data_utilities import work_week_window
from report.recommendations_builder import PERCENTILE_EXPLAINER, TIER_BANDS, UNSCORED_NOTE
from scoring.model_b import HIGH_ALERT, ELEVATED, UNSCORED, display_score, sort_by_risk

load_dotenv()

# The digest needs one row per upcoming event, which is exactly what upcoming_df.parquet
# already is: 496 rows / 46 KB, versus full_df.parquet's 2.9M rows / 323 MB. Reading the
# full frame here was unbounded (no column selection) and is what OOM-killed this job on
# the droplet's ~590 MB. upcoming_df is written by export_upcoming_df at the end of both
# the full and the incremental run, so it is always as fresh as the last scoring run.
PARQUET_PATH   = "output/upcoming_df.parquet"
COMPANY_NAMES  = "data/sp500_full_info.csv"
SUBSCRIBERS    = "data/subscribers.txt"
# The digest is sent immediately after a pipeline run, so anything older than a
# day means the run did not happen and the numbers are last week's.
MAX_PARQUET_AGE_HOURS = 24
REPORTS_DIR    = os.path.join(latest_run_output_dir(), "reports")
SMTP_HOST      = os.getenv("DIGEST_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("DIGEST_SMTP_PORT", "587"))
SMTP_USER      = os.getenv("DIGEST_SMTP_USER", "")
SMTP_PASS      = os.getenv("DIGEST_SMTP_PASS", "")


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _select_stocks(df: pd.DataFrame, company_names: dict,
                   weeks: int = 1, current_week: bool = False) -> pd.DataFrame:
    # Always a whole Mon-Fri work week — see work_week_window. The predictions snapshot
    # selects on the same window, so the table records exactly what was emailed.
    # df is upcoming_df: one row per pending event, Model B fields already attached.
    start, cutoff = work_week_window(weeks=weeks, current_week=current_week)
    all_latest = df.copy()
    all_latest["company_name"] = all_latest["stock"].map(company_names)
    mask = (
        all_latest["risk_tier"].isin([HIGH_ALERT, ELEVATED]) &
        (all_latest["earnings_date"] >= start) &
        (all_latest["earnings_date"] <= cutoff)
    )
    return sort_by_risk(all_latest[mask]).reset_index(drop=True)


def _risk_badge(level: str) -> str:
    colors = {HIGH_ALERT: "#c0392b", ELEVATED: "#e67e22"}
    bg = colors.get(level, "#999")
    return f'<span style="background:{bg};color:#fff;padding:2px 8px;border-radius:3px;font-size:11px;font-weight:700;">{level}</span>'


def _table_row(r, has_iv: bool) -> str:
    company_cell = r.get("company_name") or "&mdash;"
    iv = r.get("expected_move_pct")
    iv_td = (f'<td style="padding:8px 12px;font-size:12px;color:#777;">'
             f'{iv * 100:.1f}%</td>' if pd.notna(iv) else
             '<td style="padding:8px 12px;font-size:12px;color:#bbb;">&mdash;</td>') if has_iv else ""
    return f"""
    <tr style="border-bottom:1px solid #e0e0e0;">
      <td style="padding:8px 12px;font-size:13px;"><strong>{r['stock']}</strong></td>
      <td style="padding:8px 12px;color:#555;font-size:12px;">{company_cell}</td>
      <td style="padding:8px 12px;font-size:12px;">{pd.Timestamp(r['earnings_date']).strftime('%b %d')}</td>
      <td style="padding:8px 12px;font-size:12px;color:#555;">{r.get('sector', '')}</td>
      <td style="padding:8px 12px;">{_risk_badge(r['risk_tier'])}</td>
      <td style="padding:8px 12px;font-size:12px;font-weight:700;">{display_score(r['risk_score'])}</td>
      <td style="padding:8px 12px;font-size:12px;">{r['model_b_hist_mean_abs'] * 100:.1f}% <span style="color:#999;">({int(r['model_b_n_prior'])} reports)</span></td>
      {iv_td}
    </tr>"""


def _select_unscored(df: pd.DataFrame, weeks: int = 1, current_week: bool = False) -> list[str]:
    """Tickers reporting in the digest window that Model B cannot rank. Named in the email
    so leaving them out of the flagged table cannot read as 'low risk'."""
    start, cutoff = work_week_window(weeks=weeks, current_week=current_week)
    m = (df["risk_tier"] == UNSCORED) & (df["earnings_date"] >= start) & (df["earnings_date"] <= cutoff)
    return sorted(df.loc[m, "stock"])


def _build_html(stocks_df: pd.DataFrame, week_of: str, date_range: str = "",
                unscored: list[str] | None = None) -> str:
    n_total = len(stocks_df)
    n_ha    = int((stocks_df["risk_tier"] == HIGH_ALERT).sum())
    n_el    = int((stocks_df["risk_tier"] == ELEVATED).sum())
    has_iv  = "expected_move_pct" in stocks_df.columns and stocks_df["expected_move_pct"].notna().any()

    iv_th   = '<th style="padding:8px 12px;text-align:left;">Options Impl. Move*</th>' if has_iv else ""
    colspan = 8 if has_iv else 7

    rows_html = "".join(_table_row(r, has_iv) for _, r in stocks_df.iterrows())
    if not rows_html:
        rows_html = f'<tr><td colspan="{colspan}" style="padding:16px;text-align:center;color:#999;">No High Alert or Elevated events this week.</td></tr>'

    date_line = date_range if date_range else f"Week of {week_of}"
    unscored_line = (f'<div style="font-size:12px;color:#555;margin:-14px 0 22px 0;">'
                     f'<strong>Not ranked this week:</strong> {", ".join(unscored)}. '
                     f'{UNSCORED_NOTE}</div>') if unscored else ""
    iv_note = ("<br><strong>* Options implied move:</strong> market context from the last "
               "options snapshot. It does not enter the Breakwater score or tier.") if has_iv else ""

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family:Helvetica,Arial,sans-serif;font-size:14px;color:#1a1a1a;max-width:820px;margin:0 auto;padding:24px;">

  <div style="border-bottom:2px solid #1a1a1a;padding-bottom:12px;margin-bottom:20px;">
    <div style="font-size:20px;font-weight:700;">Breakwater <span style="font-size:13px;font-weight:400;color:#555;">Monday Earnings Risk Radar</span></div>
    <div style="font-size:12px;color:#888;margin-top:3px;">{date_line}</div>
  </div>

  <div style="font-size:13px;line-height:1.7;color:#444;margin-bottom:18px;">
    Breakwater ranks S&amp;P 500 earnings events by how large this stock's past earnings
    moves have been, using only information available before the announcement. It is about
    the size of a potential move, not its direction.
  </div>

  <div style="background:#f5f5f5;border-radius:4px;padding:12px 16px;margin-bottom:24px;font-size:13px;">
    <strong>{n_total} flagged earnings events</strong>
    &nbsp;·&nbsp; {n_ha} High Alert
    &nbsp;·&nbsp; {n_el} Elevated
  </div>
  {unscored_line}

  <table style="width:100%;border-collapse:collapse;">
    <thead>
      <tr style="background:#1a1a1a;color:#fff;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;">
        <th style="padding:8px 12px;text-align:left;">Stock</th>
        <th style="padding:8px 12px;text-align:left;">Company</th>
        <th style="padding:8px 12px;text-align:left;">Earnings</th>
        <th style="padding:8px 12px;text-align:left;">Sector</th>
        <th style="padding:8px 12px;text-align:left;">Risk Tier</th>
        <th style="padding:8px 12px;text-align:left;">Risk Percentile</th>
        <th style="padding:8px 12px;text-align:left;">Avg. Past Move</th>
        {iv_th}
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>

  <div style="margin-top:16px;font-size:11px;color:#999;line-height:1.7;">
    <strong>Risk Percentile:</strong> {PERCENTILE_EXPLAINER}<br>
    <strong>High Alert:</strong> {TIER_BANDS[HIGH_ALERT]}. <strong>Elevated:</strong> {TIER_BANDS[ELEVATED]}.<br>
    <strong>Avg. past move:</strong> mean 3-session move, either direction, over this stock's
    past reports with a verified announcement time.{iv_note}
  </div>

  <div style="margin-top:20px;background:#f8f8f8;border:1px solid #e5e5e5;border-radius:4px;padding:14px 16px;font-size:12px;line-height:1.7;color:#555;">
    <strong>Using Breakwater this week?</strong>
    Reply with one word: options, stock risk, portfolio, or research. It helps shape the dashboard.
    <div style="margin-top:10px;">
      <a href="https://harbor-markets.com/breakwater" style="display:inline-block;background:#1a1a1a;color:#fff;text-decoration:none;padding:8px 12px;border-radius:3px;font-size:11px;font-weight:700;">
        Open dashboard
      </a>
    </div>
  </div>

  <div style="margin-top:20px;padding-top:12px;border-top:1px solid #ddd;font-size:11px;color:#999;text-align:center;">
    Breakwater &mdash;
    <a href="https://harbor-markets.com/breakwater" style="color:#999;">harbor-markets.com/breakwater</a>
    &mdash; Not financial advice. For informational purposes only.<br>
    <a href="mailto:unsubscribe@harbor-markets.com?subject=Unsubscribe" style="color:#bbb;">Unsubscribe</a>
  </div>

</body>
</html>"""


def _collect_reports(tickers: list[str]) -> list[tuple[str, bytes]]:
    """Return (filename, bytes) for each ticker that has a report PDF."""
    found = []
    for ticker in tickers:
        path = os.path.join(REPORTS_DIR, f"{ticker}_report.pdf")
        if os.path.exists(path):
            with open(path, "rb") as f:
                found.append((os.path.basename(path), f.read()))
    return found


def _send(recipients: list[str], subject: str, html: str, attachments: list[tuple[str, bytes]] | None = None):
    if not SMTP_USER or not SMTP_PASS:
        print("SMTP credentials not set — printing digest to stdout.\n")
        print(f"To: {recipients}\nSubject: {subject}\n")
        print(html)
        return

    if attachments:
        msg = MIMEMultipart("mixed")
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(html, "html"))
        msg.attach(alt)
        for filename, data in attachments:
            part = MIMEBase("application", "pdf")
            part.set_payload(data)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename=filename)
            msg.attach(part)
    else:
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText(html, "html"))

    msg["Subject"] = subject
    msg["From"]    = SMTP_USER
    msg["To"]      = ", ".join(recipients)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, recipients, msg.as_string())
    n_att = len(attachments) if attachments else 0
    print(f"Digest sent to {len(recipients)} subscriber(s) with {n_att} report attachment(s).")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_weekly_digest(weeks: int = 1, current_week: bool = False):
    if not os.path.exists(PARQUET_PATH):
        print(f"Parquet not found: {PARQUET_PATH} — run the pipeline first.")
        return

    # Refuse to mail stale numbers. The digest is the product; sending last week's
    # tiers as though they were this week's is worse than sending nothing, and would
    # be invisible — every stock still looks plausible. Fail loudly instead.
    age_hours = (time.time() - os.path.getmtime(PARQUET_PATH)) / 3600
    if age_hours > MAX_PARQUET_AGE_HOURS:
        print(f"REFUSING TO SEND: {PARQUET_PATH} is {age_hours:.1f}h old "
              f"(limit {MAX_PARQUET_AGE_HOURS}h). Run the pipeline first.")
        return

    df            = pd.read_parquet(PARQUET_PATH)
    company_names = _load_company_names()
    stocks_df     = _select_stocks(df, company_names, weeks=weeks, current_week=current_week)
    start_dt, end_dt = work_week_window(weeks=weeks, current_week=current_week)
    week_of       = start_dt.strftime("%B %d, %Y")
    date_range    = f"{start_dt.strftime('%b %d')} – {end_dt.strftime('%b %d, %Y')}"
    subject       = f"Breakwater Monday Earnings Risk Radar — {date_range}"
    html          = _build_html(stocks_df, week_of, date_range=date_range,
                                unscored=_select_unscored(df, weeks=weeks, current_week=current_week))
    recipients    = _load_subscribers()

    attachments = _collect_reports(stocks_df["stock"].tolist())
    print(f"Digest: {len(stocks_df)} stocks selected, {len(attachments)} reports found, {len(recipients)} subscriber(s).")

    if not recipients:
        print("No subscribers — printing digest to stdout.\n")
        print(html)
        return

    if SMTP_USER and SMTP_PASS:
        # A real send is publication: production branch only (publishing/guard.py).
        # Without credentials _send only prints, which any branch may do.
        require_production_branch("send the weekly digest")
    _send(recipients, subject, html, attachments=attachments or None)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weeks", type=int, default=1, choices=(1, 2),
                    help="how many whole work weeks to cover (default 1)")
    ap.add_argument("--current-week", action="store_true",
                    help="cover THIS week's Mon-Fri whatever day it is run, instead of "
                         "the next complete work week")
    args = ap.parse_args()
    try:
        run_weekly_digest(weeks=args.weeks, current_week=args.current_week)
    except PublicationRefused as e:
        print(f"\n{e}\n")
        raise SystemExit(3)
