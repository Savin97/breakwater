# ingestion/fetch_eps_estimates.py
"""
Analyst EPS estimate snapshot tracker.

For each stock with upcoming earnings, fetches analyst consensus and dispersion
from yfinance and records:
  - eps_avg / eps_high / eps_low / eps_num_analysts  : current quarter consensus
  - eps_dispersion                                   : (high - low) / abs(avg)
  - eps_trend_7d/30d/60d/90d                        : how consensus has drifted
  - eps_revision_momentum                            : (avg - 90d) / abs(90d)
  - eps_revisions_up/down_7d/30d                    : analyst revision counts
  - revenue_avg / revenue_high / revenue_low         : revenue range from calendar

Data is stored in eps_estimates (one row per stock per day, idempotent).
Run separately via cron/cron_market_signals.py.
"""
import time
import warnings
import pandas as pd
import yfinance as yf
from datetime import datetime, date, timedelta
from config import DB_PATH


def _get_upcoming_earnings(con, days_ahead: int) -> pd.DataFrame:
    today_str  = date.today().isoformat()
    cutoff_str = (date.today() + timedelta(days=days_ahead)).isoformat()
    return con.execute("""
        SELECT stock, MIN(earnings_date) AS earnings_date
        FROM earnings
        WHERE earnings_date >= ?
          AND earnings_date <= ?
        GROUP BY stock
        ORDER BY earnings_date
    """, [today_str, cutoff_str]).fetchdf()


def _already_fetched_today(con) -> set:
    today_str = date.today().isoformat()
    rows = con.execute(
        "SELECT stock FROM eps_estimates WHERE snapshot_date = ?", [today_str]
    ).fetchdf()
    return set(rows["stock"].tolist())


def _to_float(val):
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _to_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def ingest_eps_estimates(con, days_ahead: int = 45, sleep_secs: float = 0.5):
    """
    Fetch analyst EPS estimate snapshots for all stocks with earnings within
    days_ahead days. Idempotent: skips stocks already fetched today.
    """
    today  = date.today()
    now    = datetime.now()

    done     = _already_fetched_today(con)
    upcoming = _get_upcoming_earnings(con, days_ahead)

    if upcoming.empty:
        print(f"No earnings found in the next {days_ahead} days.")
        return

    todo = upcoming[~upcoming["stock"].isin(done)].reset_index(drop=True)
    print(f"Upcoming earnings in next {days_ahead} days: {len(upcoming)} stocks")
    print(f"Already fetched today: {len(done)}  |  To fetch: {len(todo)}")

    inserted, skipped, failed = 0, 0, 0
    rows = []

    warnings.filterwarnings("ignore")

    for _, r in todo.iterrows():
        stock         = r["stock"]
        earnings_date = pd.Timestamp(r["earnings_date"]).date()

        try:
            ticker    = yf.Ticker(stock)
            est       = ticker.earnings_estimate
            trend     = ticker.eps_trend
            revisions = ticker.eps_revisions
            calendar  = ticker.calendar

            if est is None or est.empty or "0q" not in est.index:
                skipped += 1
                continue

            eps_avg          = _to_float(est.loc["0q", "avg"])
            eps_high         = _to_float(est.loc["0q", "high"])
            eps_low          = _to_float(est.loc["0q", "low"])
            eps_num_analysts = _to_int(est.loc["0q", "numberOfAnalysts"])

            if eps_avg is None:
                skipped += 1
                continue

            eps_dispersion = (
                (eps_high - eps_low) / abs(eps_avg)
                if eps_high is not None and eps_low is not None and eps_avg != 0
                else None
            )

            eps_trend_7d  = _to_float(trend.loc["0q", "7daysAgo"])  if trend is not None and "0q" in trend.index else None
            eps_trend_30d = _to_float(trend.loc["0q", "30daysAgo"]) if trend is not None and "0q" in trend.index else None
            eps_trend_60d = _to_float(trend.loc["0q", "60daysAgo"]) if trend is not None and "0q" in trend.index else None
            eps_trend_90d = _to_float(trend.loc["0q", "90daysAgo"]) if trend is not None and "0q" in trend.index else None

            eps_revision_momentum = (
                (eps_avg - eps_trend_90d) / abs(eps_trend_90d)
                if eps_trend_90d is not None and eps_trend_90d != 0
                else None
            )

            eps_revisions_up_7d    = _to_int(revisions.loc["0q", "upLast7days"])    if revisions is not None and "0q" in revisions.index else None
            eps_revisions_down_7d  = _to_int(revisions.loc["0q", "downLast7Days"])  if revisions is not None and "0q" in revisions.index else None
            eps_revisions_up_30d   = _to_int(revisions.loc["0q", "upLast30days"])   if revisions is not None and "0q" in revisions.index else None
            eps_revisions_down_30d = _to_int(revisions.loc["0q", "downLast30days"]) if revisions is not None and "0q" in revisions.index else None

            revenue_avg  = _to_float(calendar.get("Revenue Average")) if calendar else None
            revenue_high = _to_float(calendar.get("Revenue High"))     if calendar else None
            revenue_low  = _to_float(calendar.get("Revenue Low"))      if calendar else None

            rows.append({
                "stock":                   stock,
                "snapshot_date":           today,
                "earnings_date":           earnings_date,
                "eps_avg":                 eps_avg,
                "eps_high":                eps_high,
                "eps_low":                 eps_low,
                "eps_num_analysts":        eps_num_analysts,
                "eps_dispersion":          eps_dispersion,
                "eps_trend_7d":            eps_trend_7d,
                "eps_trend_30d":           eps_trend_30d,
                "eps_trend_60d":           eps_trend_60d,
                "eps_trend_90d":           eps_trend_90d,
                "eps_revision_momentum":   eps_revision_momentum,
                "eps_revisions_up_7d":     eps_revisions_up_7d,
                "eps_revisions_down_7d":   eps_revisions_down_7d,
                "eps_revisions_up_30d":    eps_revisions_up_30d,
                "eps_revisions_down_30d":  eps_revisions_down_30d,
                "revenue_avg":             revenue_avg,
                "revenue_high":            revenue_high,
                "revenue_low":             revenue_low,
                "ingested_at":             now,
            })
            inserted += 1
            print(f"  {stock}: eps_avg={eps_avg}  dispersion={eps_dispersion:.1%}  "
                  f"revision_mom={eps_revision_momentum:.1%}" if eps_dispersion and eps_revision_momentum
                  else f"  {stock}: eps_avg={eps_avg}")

        except Exception as e:
            failed += 1
            print(f"  FAILED {stock}: {type(e).__name__}: {e}")

        time.sleep(sleep_secs)

    if rows:
        df = pd.DataFrame(rows)
        con.register("tmp_eps", df)
        con.execute("INSERT INTO eps_estimates SELECT * FROM tmp_eps ON CONFLICT DO NOTHING")
        con.unregister("tmp_eps")

    print(f"\nEPS estimates done.  inserted={inserted}  skipped={skipped}  failed={failed}")
