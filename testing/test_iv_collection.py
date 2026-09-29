"""IV snapshot collection: market-hours guard, at-the-money strike choice, and the read
filter that drops snapshots whose expiry misses the real earnings date.

Run with:  pytest testing/test_iv_collection.py -v
"""
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import duckdb
import pandas as pd
import pytest

from types import SimpleNamespace

from ingestion.fetch_iv import _live_price, _pick_atm_strike
from utilities.db_utilities import create_iv_table_if_not_exists, join_iv
from utilities.time_utilities import nyse_is_open


def _utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


# ── Market hours ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("instant, is_open", [
    (_utc(2026, 9, 28, 15, 0), True),    # Monday, 11:00 EDT
    (_utc(2026, 9, 28, 19, 30), True),   # Monday, 15:30 EDT — last scheduled slot
    (_utc(2026, 9, 7, 15, 0), False),    # Labor Day — the run that stored stale quotes
    (_utc(2026, 6, 19, 15, 0), False),   # Juneteenth
    (_utc(2026, 9, 26, 15, 0), False),   # Saturday
    (_utc(2026, 9, 28, 13, 0), False),   # 09:00 EDT, pre-open
    (_utc(2026, 11, 27, 16, 30), True),  # day after Thanksgiving, 11:30 EST
    (_utc(2026, 11, 27, 18, 0), False),  # same day, 13:00 EST early close
])
def test_nyse_is_open(instant, is_open):
    assert nyse_is_open(instant) is is_open


def test_nyse_is_open_refuses_a_naive_time():
    with pytest.raises(ValueError):
        nyse_is_open(datetime(2026, 9, 28, 15, 0))


# ── At-the-money strike ───────────────────────────────────────────────────────

def _chain(strikes):
    return pd.DataFrame({"strike": strikes})


def test_strike_must_exist_on_both_sides():
    # 100 is nearest in the calls but has no put; the old code skipped the stock here.
    calls = _chain([95.0, 100.0, 105.0])
    puts = _chain([95.0, 105.0])
    assert _pick_atm_strike(calls, puts, price=100.5) in (95.0, 105.0)
    assert _pick_atm_strike(calls, puts, price=103.0) == 105.0


def test_strike_too_far_from_price_is_not_at_the_money():
    # SPGI at 404.11 was recorded against a 445 strike — 10.1% away.
    calls = puts = _chain([445.0, 450.0])
    assert _pick_atm_strike(calls, puts, price=404.11, max_distance=0.10) is None
    assert _pick_atm_strike(calls, puts, price=430.0, max_distance=0.10) == 445.0


def test_no_common_strike():
    assert _pick_atm_strike(_chain([100.0]), _chain([110.0]), price=100.0) is None


# ── Live price ────────────────────────────────────────────────────────────────

class _Ticker:
    """Stands in for yf.Ticker: fast_info.last_price answers from a list, one per call."""
    def __init__(self, answers):
        self.answers, self.calls = list(answers), 0

    @property
    def fast_info(self):
        self.calls += 1
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return SimpleNamespace(last_price=answer)


def test_price_is_the_chains_live_price():
    # TMO 2026-09-29 11:00 EDT: live 672.12 while the stored prior close was 678.60.
    chain = SimpleNamespace(underlying={"regularMarketPrice": 672.12,
                                        "regularMarketPreviousClose": 678.6})
    ticker = _Ticker([])
    assert _live_price(ticker, chain, wait_secs=0) == 672.12
    assert ticker.calls == 0  # no extra request when the chain has it


@pytest.mark.parametrize("underlying", [None, {}, {"regularMarketPrice": None},
                                        {"regularMarketPrice": 0.0},
                                        {"regularMarketPrice": float("nan")}])
def test_chain_without_a_price_is_retried(underlying):
    ticker = _Ticker([ConnectionError("reset"), 671.5])
    assert _live_price(ticker, SimpleNamespace(underlying=underlying),
                       retries=2, wait_secs=0) == 671.5
    assert ticker.calls == 2


def test_gives_up_after_the_retries_and_never_uses_a_stale_price():
    ticker = _Ticker([None, 0.0, 999.0])
    assert _live_price(ticker, SimpleNamespace(underlying={}), retries=2, wait_secs=0) is None
    assert ticker.calls == 2


# ── Read filter: expiry must cover the real earnings date ─────────────────────

def _db(snapshots, earnings):
    con = duckdb.connect(":memory:")
    create_iv_table_if_not_exists(con)
    con.execute("CREATE TABLE earnings (stock TEXT, earnings_date DATE)")
    for stock, d in earnings:
        con.execute("INSERT INTO earnings VALUES (?, ?)", [stock, d])
    for stock, snap, earn, expiry, iv in snapshots:
        con.execute("""
            INSERT INTO iv_snapshots (stock, snapshot_date, snapshot_hour, earnings_date,
                                      expiry_used, atm_iv, expected_move_pct)
            VALUES (?, ?, 15, ?, ?, ?, 0.05)
        """, [stock, snap, earn, expiry, iv])
    return con


def test_snapshot_whose_expiry_lands_before_the_moved_date_is_dropped():
    # MU: collected against Sep 23 with a Sep 25 expiry; the date later moved to Sep 30.
    con = _db(
        snapshots=[
            ("MU", "2026-09-01", "2026-09-23", "2026-09-25", 0.40),  # misses the event
            ("MU", "2026-09-10", "2026-09-30", "2026-10-02", 0.60),  # covers it
        ],
        earnings=[("MU", "2026-06-24"), ("MU", "2026-09-30")],
    )
    df = pd.DataFrame({"stock": ["MU", "MU"], "date": pd.to_datetime(["2026-09-02", "2026-09-11"])})
    out = join_iv(df, con)
    assert out["atm_iv"].isna().iloc[0]
    assert out["atm_iv"].iloc[1] == 0.60


def test_snapshot_with_unchanged_date_is_kept():
    con = _db(
        snapshots=[("KO", "2026-09-01", "2026-10-20", "2026-10-23", 0.20)],
        earnings=[("KO", "2026-10-20")],
    )
    df = pd.DataFrame({"stock": ["KO"], "date": pd.to_datetime(["2026-09-01"])})
    assert join_iv(df, con)["atm_iv"].iloc[0] == 0.20
