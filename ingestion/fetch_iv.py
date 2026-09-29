# ingestion/fetch_iv.py
"""
IV snapshot tracker — collects implied volatility for stocks with upcoming earnings.

For each stock with earnings within the next N days, fetches the options chain
for the nearest expiry AFTER the earnings date and records:
  - atm_iv          : average of ATM call + put implied vol
  - expected_move_pct: ATM straddle midpoint / stock price (market's priced-in earnings move)
  - expiry_used     : which expiry the chain was pulled from

Data is stored in iv_snapshots (one row per stock per day, idempotent).
Not wired into stage1 — run separately via cron/cron_iv.py.
"""
import time
import warnings
from collections import Counter
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, timedelta
from config import (DB_PATH, IV_MAX_ATM_STRIKE_DISTANCE, IV_LIVE_PRICE_RETRIES,
                    IV_LIVE_PRICE_RETRY_WAIT_SECS)
from utilities.time_utilities import nyse_is_open


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


def _already_fetched_this_hour(con) -> set:
    today_str = date.today().isoformat()
    hour = datetime.now().hour
    rows = con.execute(
        "SELECT stock FROM iv_snapshots WHERE snapshot_date = ? AND snapshot_hour = ?",
        [today_str, hour]
    ).fetchdf()
    return set(rows["stock"].tolist())


def _usable_price(price) -> float | None:
    if price is None or pd.isna(price) or price <= 0:
        return None
    return float(price)


def _live_price(ticker, chain, retries: int = IV_LIVE_PRICE_RETRIES,
                wait_secs: float = IV_LIVE_PRICE_RETRY_WAIT_SECS) -> float | None:
    """The stock's live price, or None if Yahoo will not give one.

    First from the chain Yahoo already returned (no extra request); if that lacks it,
    from ticker.fast_info, up to `retries` times. This used to be the last close in the
    prices table, i.e. the PRIOR day's close for all four runs: on a day the stock
    gapped, the ATM strike was picked off a stale price and the straddle was divided
    by it. So there is deliberately no fallback to the stored close.
    """
    price = _usable_price((getattr(chain, "underlying", None) or {}).get("regularMarketPrice"))
    for _ in range(retries):
        if price is not None:
            break
        time.sleep(wait_secs)
        try:
            price = _usable_price(ticker.fast_info.last_price)
        except Exception:
            price = None
    return price


def _pick_atm_strike(calls: pd.DataFrame, puts: pd.DataFrame, price: float,
                     max_distance: float = IV_MAX_ATM_STRIKE_DISTANCE) -> float | None:
    """The strike nearest the price among strikes quoted on BOTH sides, or None.

    Picking from calls alone skipped the stock whenever Yahoo's put list lacked that
    exact strike, which on a thin chain was most of the time (TMO, BIIB, WAT, ... never
    got a row). None also when even the best common strike is more than max_distance
    from the price: that is not an at-the-money reading.
    """
    common = pd.Series(sorted(set(calls["strike"]) & set(puts["strike"])), dtype=float)
    if common.empty:
        return None
    strike = float(common.iloc[(common - price).abs().idxmin()])
    if abs(strike / price - 1) > max_distance:
        return None
    return strike


# ── Main ingestion ────────────────────────────────────────────────────────────

def ingest_iv_snapshots(con, days_ahead: int = 45, sleep_secs: float = 0.5):
    """
    Fetch IV snapshots for all stocks with earnings within days_ahead days.
    Idempotent: skips stocks already fetched today.
    """
    if not nyse_is_open():
        print("NYSE is closed right now (holiday, weekend, early close or outside hours) "
              "— quotes would be stale, so no IV snapshot is taken.")
        return

    today   = date.today()
    now     = datetime.now()
    hour    = now.hour

    done    = _already_fetched_this_hour(con)
    upcoming = _get_upcoming_earnings(con, days_ahead)

    if upcoming.empty:
        print(f"No earnings found in the next {days_ahead} days.")
        return

    todo = upcoming[~upcoming["stock"].isin(done)].reset_index(drop=True)
    print(f"Upcoming earnings in next {days_ahead} days: {len(upcoming)} stocks")
    print(f"Already fetched today: {len(done)}  |  To fetch: {len(todo)}")

    inserted, skipped, failed = 0, 0, 0
    skip_reasons = Counter()
    rows = []

    warnings.filterwarnings("ignore")

    for _, r in todo.iterrows():
        stock         = r["stock"]
        earnings_date = pd.Timestamp(r["earnings_date"]).date()
        days_to_earn  = (earnings_date - today).days

        if days_to_earn == 0:
            skipped += 1
            skip_reasons["reports_today"] += 1
            continue

        try:
            # ── Options chain ────────────────────────────────────────────────
            ticker  = yf.Ticker(stock)
            expiries = ticker.options          # tuple of 'YYYY-MM-DD' strings
            if not expiries:
                skipped += 1
                skip_reasons["no_expiries"] += 1
                continue

            expiry_dates = pd.to_datetime(list(expiries))
            # Must be AFTER earnings to capture the earnings vol event
            valid = expiry_dates[expiry_dates > pd.Timestamp(earnings_date)]
            if valid.empty:
                skipped += 1
                skip_reasons["no_expiry_after_earnings"] += 1
                continue

            expiry     = valid.min()
            expiry_str = expiry.strftime("%Y-%m-%d")

            chain = ticker.option_chain(expiry_str)
            calls = chain.calls
            puts  = chain.puts

            if calls.empty or puts.empty:
                skipped += 1
                skip_reasons["empty_chain"] += 1
                continue

            # ── Live price ───────────────────────────────────────────────────
            price = _live_price(ticker, chain)
            if price is None:
                skipped += 1
                skip_reasons["no_live_price"] += 1
                continue

            # ── ATM strike ───────────────────────────────────────────────────
            atm_strike = _pick_atm_strike(calls, puts, price)
            if atm_strike is None:
                skipped += 1
                skip_reasons["no_atm_strike"] += 1
                continue

            atm_call = calls[calls["strike"] == atm_strike]
            atm_put  = puts[puts["strike"] == atm_strike]

            # ── IVs ──────────────────────────────────────────────────────────
            call_iv = atm_call["impliedVolatility"].values[0]
            put_iv  = atm_put["impliedVolatility"].values[0]

            if pd.isna(call_iv) or pd.isna(put_iv):
                skipped += 1
                skip_reasons["no_iv"] += 1
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
                skip_reasons["zero_ask"] += 1
                continue

            call_mid = (call_bid + call_ask) / 2.0
            put_mid  = (put_bid  + put_ask)  / 2.0
            expected_move_pct = (call_mid + put_mid) / price

            rows.append({
                "stock":             stock,
                "snapshot_date":     today,
                "snapshot_hour":     hour,
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
    # Column list is explicit (not `SELECT *`) because the live table's physical
    # column order can drift from this list via ALTER TABLE (e.g. snapshot_hour
    # was appended at the end on existing tables) — a positional SELECT * would
    # silently misalign values into the wrong columns.
    if rows:
        df = pd.DataFrame(rows)
        cols = list(df.columns)
        col_list = ", ".join(cols)
        con.register("tmp_iv", df)
        con.execute(f"INSERT INTO iv_snapshots ({col_list}) SELECT {col_list} FROM tmp_iv ON CONFLICT DO NOTHING")
        con.unregister("tmp_iv")

    print(f"\nIV snapshots done.  inserted={inserted}  skipped={skipped}  failed={failed}")
    if skip_reasons:
        print("Skipped because: " + ", ".join(f"{k}={v}" for k, v in skip_reasons.most_common()))
