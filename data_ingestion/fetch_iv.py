# data_ingestion/fetch_iv.py
"""
IV snapshot tracker — collects implied volatility for stocks with upcoming earnings.

For each stock with earnings within the next N days, fetches the options chain
for the nearest expiry AFTER the earnings date and records:
  - atm_iv          : average of ATM call + put implied vol
  - expected_move_pct: ATM straddle midpoint / stock price (market's priced-in earnings move)
  - expiry_used     : which expiry the chain was pulled from

Data is stored in iv_snapshots (one row per stock per day, idempotent).
Not wired into stage1 — run separately via data_ingestion/cron_iv.py.
"""
import time
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, timedelta
from config import DB_PATH


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_upcoming_earnings(con, days_ahead: int) -> pd.DataFrame:
    """Return one row per stock: nearest upcoming earnings date within days_ahead."""
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
        "SELECT stock FROM iv_snapshots WHERE snapshot_date = ?", [today_str]
    ).fetchdf()
    return set(rows["stock"].tolist())


def _get_current_price(con, stock: str) -> float | None:
    """Pull latest price from prices table (already updated by cron_ingest)."""
    row = con.execute(
        "SELECT price FROM prices WHERE stock = ? ORDER BY date DESC LIMIT 1",
        [stock]
    ).fetchone()
    return float(row[0]) if row else None


# ── Main ingestion ────────────────────────────────────────────────────────────

def ingest_iv_snapshots(con, days_ahead: int = 45, sleep_secs: float = 0.5):
    """
    Fetch IV snapshots for all stocks with earnings within days_ahead days.
    Idempotent: skips stocks already fetched today.
    """
    today   = date.today()
    now     = datetime.now()

    done    = _already_fetched_today(con)
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
        days_to_earn  = (earnings_date - today).days

        try:
            # ── Price (from DB, no extra API call) ───────────────────────────
            price = _get_current_price(con, stock)
            if not price or price <= 0:
                skipped += 1
                continue

            # ── Options chain ────────────────────────────────────────────────
            ticker  = yf.Ticker(stock)
            expiries = ticker.options          # tuple of 'YYYY-MM-DD' strings
            if not expiries:
                skipped += 1
                continue

            expiry_dates = pd.to_datetime(list(expiries))
            # Must be AFTER earnings to capture the earnings vol event
            valid = expiry_dates[expiry_dates > pd.Timestamp(earnings_date)]
            if valid.empty:
                skipped += 1
                continue

            expiry     = valid.min()
            expiry_str = expiry.strftime("%Y-%m-%d")

            chain = ticker.option_chain(expiry_str)
            calls = chain.calls
            puts  = chain.puts

            if calls.empty or puts.empty:
                skipped += 1
                continue

            # ── ATM strike ───────────────────────────────────────────────────
            atm_idx    = (calls["strike"] - price).abs().idxmin()
            atm_strike = float(calls.loc[atm_idx, "strike"])

            atm_call = calls[calls["strike"] == atm_strike]
            atm_put  = puts[puts["strike"] == atm_strike]

            if atm_call.empty or atm_put.empty:
                skipped += 1
                continue

            # ── IVs ──────────────────────────────────────────────────────────
            call_iv = atm_call["impliedVolatility"].values[0]
            put_iv  = atm_put["impliedVolatility"].values[0]

            if pd.isna(call_iv) or pd.isna(put_iv):
                skipped += 1
                continue

            atm_iv = (call_iv + put_iv) / 2.0

            # ── Expected move = straddle midpoint / price ────────────────────
            call_bid = atm_call["bid"].values[0]
            call_ask = atm_call["ask"].values[0]
            put_bid  = atm_put["bid"].values[0]
            put_ask  = atm_put["ask"].values[0]

            # Guard against stale/zero quotes
            if call_ask <= 0 or put_ask <= 0:
                skipped += 1
                continue

            call_mid = (call_bid + call_ask) / 2.0
            put_mid  = (put_bid  + put_ask)  / 2.0
            expected_move_pct = (call_mid + put_mid) / price

            rows.append({
                "stock":             stock,
                "snapshot_date":     today,
                "earnings_date":     earnings_date,
                "days_to_earnings":  days_to_earn,
                "current_price":     round(price, 4),
                "expiry_used":       expiry.date(),
                "atm_strike":        atm_strike,
                "atm_call_iv":       round(float(call_iv), 6),
                "atm_put_iv":        round(float(put_iv),  6),
                "atm_iv":            round(float(atm_iv),  6),
                "expected_move_pct": round(float(expected_move_pct), 6),
                "ingested_at":       now,
            })
            inserted += 1
            print(f"  {stock}: IV={atm_iv:.1%}  exp_move={expected_move_pct:.1%}  "
                  f"expiry={expiry_str}  days_to_earn={days_to_earn}")

        except Exception as e:
            failed += 1
            print(f"  FAILED {stock}: {type(e).__name__}: {e}")

        time.sleep(sleep_secs)

    # ── Bulk insert ───────────────────────────────────────────────────────────
    if rows:
        df = pd.DataFrame(rows)
        con.register("tmp_iv", df)
        con.execute("INSERT INTO iv_snapshots SELECT * FROM tmp_iv ON CONFLICT DO NOTHING")
        con.unregister("tmp_iv")

    print(f"\nIV snapshots done.  inserted={inserted}  skipped={skipped}  failed={failed}")
