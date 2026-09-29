"""Announcement-timestamp ingestion: the three timing columns on `earnings`, the
yfinance fetch that keeps the announcement time, the schedule-vs-observation refresh
rule, the one New York clock, and the one-time seed loader.

Ported from the methodology-rebuild branch's `test_announcement_timing.py` (the
ingestion half only; the event-frame and anchoring tests stay on that branch).

Run with:  pytest testing/test_announcement_ingest.py -v
"""
import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

# ── Schema and ingestion: the timestamp must survive the write path ───────────

def _fresh_db():
    import duckdb
    from utilities.db_utilities import create_earnings_table_if_not_exists
    con = duckdb.connect(":memory:")
    create_earnings_table_if_not_exists(con)
    return con


def test_earnings_schema_carries_the_timing_columns():
    con = _fresh_db()
    try:
        cols = {r[0]: r[1] for r in con.execute("DESCRIBE earnings").fetchall()}
    finally:
        con.close()
    assert cols.get("announce_ts_ny") == "TIMESTAMP"
    assert cols.get("announce_ts_source") == "VARCHAR"


def test_insert_column_list_matches_the_physical_table_order():
    """Both writers name their columns instead of `SELECT *`. If the list ever drifts
    from the table, a future column would shift values into the wrong slot silently."""
    from ingestion.fetch_earnings_dates import EARNINGS_INSERT_COLS
    con = _fresh_db()
    try:
        physical = [r[0] for r in con.execute("DESCRIBE earnings").fetchall()]
    finally:
        con.close()
    assert EARNINGS_INSERT_COLS == physical


def test_yfinance_ingestion_no_longer_discards_the_timestamp(monkeypatch):
    """The stated Phase 2 ingestion change: `fetch_one_earnings_dates` used to collapse
    yfinance's tz-aware NY index to a bare date with `.dt.date`, which is what forced
    every later timing analysis to infer BMO/AMC from price behavior. It must now keep
    the wall clock — while storing exactly the same `earnings_date` as before."""
    import ingestion.fetch_earnings_dates as fed

    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-05-01 06:30"), pd.Timestamp("2024-02-01 16:05")],
        name="Earnings Date").tz_localize("America/New_York")
    payload = pd.DataFrame(
        {"EPS Estimate": [0.9, 1.1], "Reported EPS": [1.0, 1.2], "Surprise(%)": [11.0, 9.0]},
        index=idx)

    class _Ticker:
        def __init__(self, symbol):
            self.earnings_dates = payload

    monkeypatch.setattr(fed.yf, "Ticker", _Ticker)
    monkeypatch.setattr(fed.time, "sleep", lambda *_: None)

    out = fed.fetch_one_earnings_dates("AAA")["earnings_dates_df"]
    assert list(out.columns) == fed.EARNINGS_INSERT_COLS
    assert list(out["announce_ts_ny"]) == [pd.Timestamp("2024-05-01 06:30"),
                                           pd.Timestamp("2024-02-01 16:05")]
    assert (out["announce_ts_source"] == fed.ANNOUNCE_TS_SOURCE_YFINANCE).all()
    # the calendar date is byte-for-byte what the old .dt.date produced
    assert list(out["earnings_date"]) == list(
        pd.to_datetime(idx).tz_localize(None).date)


# ── A pre-event schedule must not be frozen forever ──────────────────────────
# External review of Phase 2, item 2. `announce_ts_ny` collected while an event is still
# upcoming is a SCHEDULE: the issuer can move it and the provider can correct it. The
# NULL-only backfill this phase shipped with wrote such a schedule once and never looked
# at it again, so a historical anchored target could rest permanently on a time that
# never happened. `announce_ts_observed_at` is the minimum provenance needed to tell the
# two apart, and the refresh rule is written in terms of it.

_E = pd.Timestamp("2024-05-01").date()          # the report date used throughout


def _seed_event(con, announce_ts_ny, observed_at, source, ingested_at=None,
                earnings_date=None):
    from ingestion.fetch_earnings_dates import EARNINGS_INSERT_COLS, _INSERT_COL_SQL
    df = pd.DataFrame([{
        "stock": "AAA",
        "earnings_date": _E if earnings_date is None else earnings_date,
        "fiscal_end_date": None,
        "reported_eps": None, "estimated_eps": 0.9, "surprise_percentage": None,
        "ingested_at": ingested_at if ingested_at is not None else observed_at,
        "announce_ts_ny": announce_ts_ny, "announce_ts_source": source,
        "announce_ts_observed_at": observed_at,
    }])[EARNINGS_INSERT_COLS]
    con.register("tmp_earnings_df", df)
    con.execute(f"INSERT INTO earnings ({_INSERT_COL_SQL}) "
                f"SELECT {_INSERT_COL_SQL} FROM tmp_earnings_df")
    con.unregister("tmp_earnings_df")


def _stored(con):
    return con.execute("SELECT announce_ts_ny, announce_ts_source, "
                       "announce_ts_observed_at FROM earnings "
                       "WHERE stock = 'AAA'").fetchone()


def test_observed_at_column_exists_on_the_earnings_table():
    con = _fresh_db()
    try:
        cols = {r[0]: r[1] for r in con.execute("DESCRIBE earnings").fetchall()}
    finally:
        con.close()
    assert cols.get("announce_ts_observed_at") == "TIMESTAMP"


def test_refresh_fills_a_row_that_has_no_timestamp():
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    con = _fresh_db()
    try:
        _seed_event(con, pd.NaT, pd.NaT, None, ingested_at=pd.Timestamp("2024-04-01"))
        n = refresh_announcement_timestamp(
            con, "AAA", _E, pd.Timestamp("2024-05-01 06:30"), "yfinance_earnings_dates",
            pd.Timestamp("2024-05-03 02:00"))
        got = _stored(con)
    finally:
        con.close()
    assert n == 1
    assert got[0] == pd.Timestamp("2024-05-01 06:30")


def test_refresh_replaces_a_pre_event_schedule_with_a_later_observation():
    """The scenario the review asked for, end to end:

      * a timestamp is stored while the event is still upcoming (a SCHEDULE),
      * the provider later returns a DIFFERENT time for the same event,
      * after the refresh the stored historical timing is the NEWER observation.

    Here the correction also flips the classified window, which is the whole reason a
    stale schedule matters: it would have anchored the corrected target to the wrong
    session forever.
    """
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    con = _fresh_db()
    try:
        # observed 11 days BEFORE the announcement -> a schedule
        _seed_event(con, pd.Timestamp("2024-05-01 16:05"), pd.Timestamp("2024-04-20"),
                    "yfinance_earnings_dates")

        # observed two days AFTER it -> an observation of what actually happened
        n = refresh_announcement_timestamp(
            con, "AAA", _E, pd.Timestamp("2024-05-01 06:30"), "yfinance_earnings_dates",
            pd.Timestamp("2024-05-03 02:00"))
        got = _stored(con)
    finally:
        con.close()
    assert n == 1
    assert got[0] == pd.Timestamp("2024-05-01 06:30")
    assert got[2] == pd.Timestamp("2024-05-03 02:00")          # provenance updated
    assert got[1] == "yfinance_earnings_dates"


def test_refresh_never_overwrites_a_post_event_observation():
    """Once a timestamp has been observed AFTER the announcement it is a record of what
    happened, and no later fetch may move it."""
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    con = _fresh_db()
    try:
        _seed_event(con, pd.Timestamp("2024-05-01 06:30"), pd.Timestamp("2024-05-03"),
                    "yfinance_earnings_dates")
        n = refresh_announcement_timestamp(
            con, "AAA", _E, pd.Timestamp("2024-05-01 16:05"), "yfinance_earnings_dates",
            pd.Timestamp("2024-06-01"))
        got = _stored(con)
    finally:
        con.close()
    assert n == 0
    assert got[0] == pd.Timestamp("2024-05-01 06:30")
    assert got[2] == pd.Timestamp("2024-05-03")


def test_refresh_ignores_an_older_observation():
    """A stale fetch arriving late must not clobber a newer one. This is also what makes
    re-running ingestion idempotent."""
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    con = _fresh_db()
    try:
        _seed_event(con, pd.Timestamp("2024-05-01 16:05"), pd.Timestamp("2024-04-25"),
                    "yfinance_earnings_dates")
        n = refresh_announcement_timestamp(
            con, "AAA", _E, pd.Timestamp("2024-05-01 06:30"), "yfinance_earnings_dates",
            pd.Timestamp("2024-04-20"))                      # older than what is stored
        got = _stored(con)
    finally:
        con.close()
    assert n == 0
    assert got[0] == pd.Timestamp("2024-05-01 16:05")


def test_refresh_is_idempotent():
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    con = _fresh_db()
    try:
        _seed_event(con, pd.Timestamp("2024-05-01 16:05"), pd.Timestamp("2024-04-20"),
                    "yfinance_earnings_dates")
        args = ("AAA", _E, pd.Timestamp("2024-05-01 06:30"), "yfinance_earnings_dates",
                pd.Timestamp("2024-05-03 02:00"))
        assert refresh_announcement_timestamp(con, *args) == 1
        assert refresh_announcement_timestamp(con, *args) == 0     # now post-event
        got = _stored(con)
    finally:
        con.close()
    assert got[0] == pd.Timestamp("2024-05-01 06:30")


@pytest.mark.parametrize("ingested_at,expect_refresh", [
    (pd.Timestamp("2024-04-20"), True),    # row created BEFORE the event -> a schedule
    (pd.Timestamp("2024-05-03"), False),   # row created AFTER it -> treat as observed
])
def test_refresh_falls_back_to_ingested_at_on_a_pre_column_row(ingested_at, expect_refresh):
    """Rows written before `announce_ts_observed_at` existed have NULL in it. Their
    `ingested_at` is a lower bound on when the timestamp could have been observed, so it
    can only make a row look MORE like a schedule — never less. Using it is the
    conservative direction; with both NULL nothing is refreshed at all."""
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    con = _fresh_db()
    try:
        _seed_event(con, pd.Timestamp("2024-05-01 16:05"), pd.NaT,
                    "yfinance_earnings_dates", ingested_at=ingested_at)
        n = refresh_announcement_timestamp(
            con, "AAA", _E, pd.Timestamp("2024-05-01 06:30"), "yfinance_earnings_dates",
            pd.Timestamp("2024-05-10"))
        got = _stored(con)
    finally:
        con.close()
    assert n == (1 if expect_refresh else 0)
    assert got[0] == (pd.Timestamp("2024-05-01 06:30") if expect_refresh
                      else pd.Timestamp("2024-05-01 16:05"))


def test_refresh_does_nothing_when_the_observation_time_is_unknown():
    """Neither column set means we cannot say whether the stored value is a schedule.
    Refusing to touch it is the only safe answer."""
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    con = _fresh_db()
    try:
        _seed_event(con, pd.Timestamp("2024-05-01 16:05"), pd.NaT,
                    "yfinance_earnings_dates", ingested_at=pd.NaT)
        n = refresh_announcement_timestamp(
            con, "AAA", _E, pd.Timestamp("2024-05-01 06:30"), "yfinance_earnings_dates",
            pd.Timestamp("2024-05-10"))
        got = _stored(con)
    finally:
        con.close()
    assert n == 0
    assert got[0] == pd.Timestamp("2024-05-01 16:05")


def test_yfinance_fetch_records_when_it_observed_the_timestamp(monkeypatch):
    """The refresh rule is only as good as the observation time it is given, so the
    fetcher must stamp it at the moment of the fetch rather than leave it to be inferred
    later — in NY wall clock, the convention `announce_ts_ny` is in. The host-timezone
    half of that claim is tested in the timezone-convention section below."""
    import ingestion.fetch_earnings_dates as fed

    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-05-01 06:30"), pd.Timestamp("2024-02-01 16:05")],
        name="Earnings Date").tz_localize("America/New_York")
    payload = pd.DataFrame(
        {"EPS Estimate": [0.9, 1.1], "Reported EPS": [1.0, 1.2], "Surprise(%)": [11.0, 9.0]},
        index=idx)

    class _Ticker:
        def __init__(self, symbol):
            self.earnings_dates = payload

    monkeypatch.setattr(fed.yf, "Ticker", _Ticker)
    monkeypatch.setattr(fed.time, "sleep", lambda *_: None)
    from utilities.time_utilities import now_ny
    before = pd.Timestamp(now_ny())
    out = fed.fetch_one_earnings_dates("AAA")["earnings_dates_df"]
    after = pd.Timestamp(now_ny())

    assert "announce_ts_observed_at" in out.columns
    assert out["announce_ts_observed_at"].notna().all()
    assert (out["announce_ts_observed_at"] >= before).all()
    assert (out["announce_ts_observed_at"] <= after).all()
    # `ingested_at` covers the same moment but stays in the legacy machine-local
    # convention — it is an operational column, not part of the timing comparison, so it
    # equals the observation only on a host that is already on New York time.
    assert out["ingested_at"].notna().all()


# ── The observation timestamp has ONE timezone convention ────────────────────
# External review of Phase 2, item 3. `announce_ts_ny` is naive NEW YORK wall clock and
# `announce_ts_observed_at` is compared against it directly, so it must be NY wall clock
# too. It used to be `datetime.now()`, which is the wall clock of whatever host ran the
# pipeline. On a host east of New York — UTC, or Israel at UTC+2/+3 — an observation made
# HOURS BEFORE an announcement produces a NUMBER LARGER than the NY announcement time, so
# the refresh rule reads a schedule as a post-event observation and freezes it into the
# historical record permanently. The classification must depend on the New York event
# clock and on nothing else.

import contextlib
import time as _stdtime
from datetime import datetime as _dt, timedelta as _td, timezone as _tz
from zoneinfo import ZoneInfo as _ZI

from utilities.time_utilities import (
    NY_TZ, MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS, now_ny, to_ny_wall_clock,
    utc_to_ny_wall_clock,
)

# Hosts this pipeline plausibly runs on, plus the two extremes of the offset range.
# Israel is the one that actually motivated the fix: it leads New York by 6 or 7 hours
# depending on whose DST is in effect, and the two zones switch on different dates.
_HOST_TZS = ["America/New_York", "UTC", "Asia/Jerusalem", "Europe/London",
             "Pacific/Kiritimati"]

_HAS_TZSET = hasattr(_stdtime, "tzset")
_needs_tzset = pytest.mark.skipif(not _HAS_TZSET, reason="TZ simulation needs time.tzset")


@contextlib.contextmanager
def _host_tz(name):
    """Run the block as if the machine's local timezone were `name`.

    This is the real thing, not a mock: it moves what bare `datetime.now()` returns,
    which is exactly the dependency being tested.
    """
    previous = os.environ.get("TZ")
    os.environ["TZ"] = name
    _stdtime.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        _stdtime.tzset()


def _wall(instant_utc, tz_name):
    """The naive wall clock a bare `datetime.now()` on a host in `tz_name` would return
    at the absolute instant `instant_utc` (given naive, in UTC)."""
    return pd.Timestamp(instant_utc.replace(tzinfo=_tz.utc)
                        .astimezone(_ZI(tz_name)).replace(tzinfo=None))


@_needs_tzset
@pytest.mark.parametrize("host", _HOST_TZS)
def test_now_ny_reads_the_new_york_clock_on_every_host(host):
    """`now_ny()` is the same value on every host at the same moment; bare
    `datetime.now()` is not. The second assertion is the bug, stated positively."""
    with _host_tz(host):
        got = now_ny()
        host_local = _dt.now()
    truth = _dt.now(_tz.utc).astimezone(NY_TZ).replace(tzinfo=None)
    assert abs(got - truth) < _td(seconds=5)

    expected_host = _dt.now(_tz.utc).astimezone(_ZI(host)).replace(tzinfo=None)
    assert abs(host_local - expected_host) < _td(seconds=5)
    ny_offset = _dt.now(_tz.utc).astimezone(NY_TZ).utcoffset()
    host_offset = _dt.now(_tz.utc).astimezone(_ZI(host)).utcoffset()
    if ny_offset != host_offset:
        assert abs(host_local - got) > _td(minutes=30)


@_needs_tzset
@pytest.mark.parametrize("host", _HOST_TZS)
@pytest.mark.parametrize("instant_utc,expected_ny", [
    (_dt(2024, 1, 15, 21, 5), pd.Timestamp("2024-01-15 16:05")),   # EST, UTC-5
    (_dt(2024, 7, 25, 20, 5), pd.Timestamp("2024-07-25 16:05")),   # EDT, UTC-4
    (_dt(2024, 3, 10,  6, 0), pd.Timestamp("2024-03-10 01:00")),   # minutes before spring-forward
    (_dt(2024, 3, 10,  7, 30), pd.Timestamp("2024-03-10 03:30")),  # minutes after it
    (_dt(2024, 11, 3,  5, 30), pd.Timestamp("2024-11-03 01:30")),  # first pass of the repeated hour
    (_dt(2024, 11, 3,  6, 30), pd.Timestamp("2024-11-03 01:30")),  # second pass of it
])
def test_the_ny_convention_is_dst_correct_and_host_independent(host, instant_utc, expected_ny):
    """The zone, not an offset constant, decides the wall clock — including across both
    US transitions, and regardless of when the HOST's own DST switches (Israel's dates
    differ from the US's, which is what makes the offset 6 hours in March and 7 in July)."""
    with _host_tz(host):
        assert utc_to_ny_wall_clock(instant_utc) == expected_ny
        assert to_ny_wall_clock(instant_utc.replace(tzinfo=_tz.utc)) == expected_ny
        # already NY wall clock -> nothing to convert, returned verbatim
        assert to_ny_wall_clock(expected_ny.to_pydatetime()) == expected_ny


# (label, announce_ts_ny, observation instant in UTC, may the stored value be refreshed)
#
# Every observation here is timed relative to the NEW YORK announcement clock. Read in
# Israel's or UTC's wall clock instead, the second row flips from schedule to
# observation — that is the failure this section exists to rule out.
_OBSERVATION_CASES = [
    ("pre-event, days ahead, EST",
     pd.Timestamp("2024-01-24 16:05"), _dt(2024, 1, 20, 12, 0), True),
    ("pre-event, 90 minutes ahead, EDT",
     pd.Timestamp("2024-07-25 06:30"), _dt(2024, 7, 25, 9, 0), True),
    ("post-event, same session, EDT",
     pd.Timestamp("2024-07-25 06:30"), _dt(2024, 7, 25, 14, 0), False),
    ("pre-event across the spring-forward boundary",
     pd.Timestamp("2024-03-12 06:30"), _dt(2024, 3, 8, 20, 0), True),
    ("post-event across the fall-back boundary",
     pd.Timestamp("2024-11-01 16:05"), _dt(2024, 11, 5, 9, 0), False),
    ("pre-event minutes before an AMC call",
     pd.Timestamp("2024-07-25 16:05"), _dt(2024, 7, 25, 19, 50), True),
]


@_needs_tzset
@pytest.mark.parametrize("host", _HOST_TZS)
@pytest.mark.parametrize("label,announce,instant_utc,expect_refresh",
                         _OBSERVATION_CASES,
                         ids=[c[0] for c in _OBSERVATION_CASES])
def test_schedule_vs_observation_is_identical_on_every_host(
        host, label, announce, instant_utc, expect_refresh):
    """The whole rule, end to end, under five host timezones.

    The stored row is stamped the way production stamps it — the observation instant
    rendered in New York — and the refresh outcome must come out the same everywhere.
    """
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    event_date = announce.date()
    stamp = _wall(instant_utc, "America/New_York")
    corrected = announce + _td(hours=1)
    later = _wall(instant_utc + _td(days=30), "America/New_York")

    with _host_tz(host):
        con = _fresh_db()
        try:
            con.execute("DELETE FROM earnings")
            _seed_event(con, announce, stamp, "yfinance_earnings_dates",
                        earnings_date=event_date)
            n = refresh_announcement_timestamp(
                con, "AAA", event_date, corrected, "yfinance_earnings_dates", later)
            stored = _stored(con)
        finally:
            con.close()

    assert n == (1 if expect_refresh else 0), label
    assert stored[0] == (corrected if expect_refresh else announce), label


@_needs_tzset
@pytest.mark.parametrize("host", ["UTC", "Asia/Jerusalem"])
def test_a_host_local_stamp_would_have_frozen_a_pre_event_schedule(host):
    """The regression, pinned. Stamping the SAME instant in the host's own wall clock —
    what `datetime.now()` did — turns a schedule observed 90 minutes before a BMO call
    into a permanent post-event record on any host east of New York.

    This test asserts the broken behaviour deliberately, so that a future change which
    reintroduces `datetime.now()` here fails the test above rather than passing both.
    """
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    announce = pd.Timestamp("2024-07-25 06:30")            # BMO, EDT
    instant_utc = _dt(2024, 7, 25, 9, 0)                   # 05:00 NY — still upcoming
    host_stamp = _wall(instant_utc, host)
    ny_stamp = _wall(instant_utc, "America/New_York")
    assert host_stamp > announce                           # reads as post-event...
    assert ny_stamp < announce                             # ...but it is not

    def _try(stamp):
        con = _fresh_db()
        try:
            _seed_event(con, announce, stamp, "yfinance_earnings_dates",
                        earnings_date=announce.date())
            return refresh_announcement_timestamp(
                con, "AAA", announce.date(), pd.Timestamp("2024-07-25 07:15"),
                "yfinance_earnings_dates", pd.Timestamp("2024-07-26 09:00"))
        finally:
            con.close()

    with _host_tz(host):
        assert _try(host_stamp) == 0        # frozen — the bug
        assert _try(ny_stamp) == 1          # refreshed — the fix


@_needs_tzset
@pytest.mark.parametrize("host", _HOST_TZS)
def test_the_fetcher_stamps_the_observation_in_new_york_time(monkeypatch, host):
    """Where the value is actually produced. `announce_ts_observed_at` must come out of
    `now_ny()` on every host; `ingested_at` deliberately keeps its legacy machine-local
    convention and is NOT part of the announcement-timing comparison."""
    import ingestion.fetch_earnings_dates as fed

    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-05-01 06:30"), pd.Timestamp("2024-02-01 16:05")],
        name="Earnings Date").tz_localize("America/New_York")
    payload = pd.DataFrame(
        {"EPS Estimate": [0.9, 1.1], "Reported EPS": [1.0, 1.2], "Surprise(%)": [11.0, 9.0]},
        index=idx)

    class _Ticker:
        def __init__(self, symbol):
            self.earnings_dates = payload

    monkeypatch.setattr(fed.yf, "Ticker", _Ticker)
    monkeypatch.setattr(fed.time, "sleep", lambda *_: None)

    with _host_tz(host):
        before = now_ny()
        out = fed.fetch_one_earnings_dates("AAA")["earnings_dates_df"]
        after = now_ny()
        host_local = pd.Timestamp(_dt.now())

    observed = out["announce_ts_observed_at"]
    assert observed.notna().all()
    assert (observed >= before).all() and (observed <= after).all()

    ny_offset = _dt.now(_tz.utc).astimezone(NY_TZ).utcoffset()
    host_offset = _dt.now(_tz.utc).astimezone(_ZI(host)).utcoffset()
    if ny_offset != host_offset:
        # the host clock is somewhere else entirely, and the column did not follow it
        assert (observed - host_local).abs().min() > _td(minutes=30)


@_needs_tzset
@pytest.mark.parametrize("host", ["UTC", "Asia/Jerusalem"])
def test_a_legacy_ingested_at_is_never_compared_raw_against_the_ny_clock(host):
    """The fallback path. Rows written before `announce_ts_observed_at` existed carry a
    machine-local `ingested_at` whose timezone convention was never recorded, so it
    cannot be compared against the NY announcement clock directly.

    Here the row was ingested 90 minutes BEFORE a BMO announcement by a host east of New
    York, so its raw value sorts after the announcement. Comparing raw would freeze a
    schedule on the strength of a timezone artefact; widening it into a lower bound that
    holds for any host keeps it refreshable, which is the conservative direction.
    """
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    announce = pd.Timestamp("2024-07-25 06:30")
    ingested_at = _wall(_dt(2024, 7, 25, 9, 0), host)      # 05:00 NY, still upcoming
    assert ingested_at > announce                          # raw comparison says post-event

    with _host_tz(host):
        con = _fresh_db()
        try:
            _seed_event(con, announce, pd.NaT, "yfinance_earnings_dates",
                        ingested_at=ingested_at, earnings_date=announce.date())
            n = refresh_announcement_timestamp(
                con, "AAA", announce.date(), pd.Timestamp("2024-07-25 07:15"),
                "yfinance_earnings_dates", pd.Timestamp("2024-07-26 09:00"))
            stored = _stored(con)
        finally:
            con.close()
    assert n == 1
    assert stored[0] == pd.Timestamp("2024-07-25 07:15")
    # and the refresh replaces the ambiguity with a real NY-convention observation, so
    # this row never takes the widened path again
    assert stored[2] == pd.Timestamp("2024-07-26 09:00")


@pytest.mark.parametrize("ingested_at", [
    pd.Timestamp("2024-07-27 09:00"),      # two days later — post-event under any host
    pd.Timestamp("2024-08-30 00:00"),      # a month later
])
def test_an_unambiguous_legacy_ingested_at_still_freezes_the_row(ingested_at):
    """The widening must not swallow the rule. A legacy `ingested_at` more than
    MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS past the announcement was an observation on every
    possible host, and stays frozen."""
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    announce = pd.Timestamp("2024-07-25 06:30")
    assert ingested_at - _td(hours=MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS) > announce
    con = _fresh_db()
    try:
        _seed_event(con, announce, pd.NaT, "yfinance_earnings_dates",
                    ingested_at=ingested_at, earnings_date=announce.date())
        n = refresh_announcement_timestamp(
            con, "AAA", announce.date(), pd.Timestamp("2024-07-25 07:15"),
            "yfinance_earnings_dates", pd.Timestamp("2024-09-01 09:00"))
        stored = _stored(con)
    finally:
        con.close()
    assert n == 0
    assert stored[0] == announce


def test_the_widening_is_exactly_the_worst_case_host_offset():
    """19 hours is UTC+14 against New York on EST. Not a round number picked for comfort:
    it is the largest a host wall clock can lead New York's."""
    extreme = _dt(2024, 1, 15, 12, 0)                       # EST is in effect
    ny = utc_to_ny_wall_clock(extreme)
    ahead = max(
        (_wall(extreme, zone) - ny) for zone in
        ["Pacific/Kiritimati", "Pacific/Apia", "Asia/Jerusalem", "UTC", "Asia/Tokyo"]
    )
    assert ahead <= _td(hours=MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS)
    assert _wall(extreme, "Pacific/Kiritimati") - ny == _td(hours=19)


def test_no_module_stamps_the_observation_from_the_host_clock():
    """Static guard, the same shape as `test_6_the_classifier_never_touches_price`.

    Nothing anywhere may assign `announce_ts_observed_at` from a bare `datetime.now()` /
    `pd.Timestamp.now()`. There is one correct source and it is `now_ny()`.
    """
    import pathlib
    offenders = []
    for directory in ("ingestion", "pipeline", "utilities", "scripts",
                      "backfills", "feature_engineering", "analysis"):
        for path in sorted(pathlib.Path(directory).glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                targets = [t for t in node.targets if isinstance(t, ast.Subscript)]
                names = [t.slice.value for t in targets
                         if isinstance(t.slice, ast.Constant)
                         and isinstance(t.slice.value, str)]
                if "announce_ts_observed_at" not in names:
                    continue
                src = ast.unparse(node.value)
                if "datetime.now(" in src or "Timestamp.now(" in src:
                    offenders.append(f"{path}:{node.lineno}: {src}")
    assert not offenders, (
        "announce_ts_observed_at must be stamped with utilities.time_utilities.now_ny(); "
        "a host-local clock silently makes the schedule/observation classification depend "
        "on where the pipeline runs:\n" + "\n".join(offenders))


# ── The one-time seed loader (backfills/load_announcement_seed.py) ───────────

def _seed(rows):
    from backfills.load_announcement_seed import SEED_COLS
    return pd.DataFrame([dict(zip(SEED_COLS, r)) for r in rows])


def test_seed_loader_fills_nulls_only_and_is_idempotent():
    from backfills.load_announcement_seed import load
    con = _fresh_db()
    try:
        _seed_event(con, pd.NaT, pd.NaT, None)
        _seed_event(con, pd.Timestamp("2024-08-01 16:00"), pd.Timestamp("2024-08-02"),
                    "yfinance_earnings_dates", earnings_date=pd.Timestamp("2024-08-01").date())
        seed = _seed([
            ("AAA", _E, pd.Timestamp("2024-05-01 06:45"), "massive_benzinga:snap:id1",
             pd.Timestamp("2026-09-10 14:04:12")),
            ("AAA", pd.Timestamp("2024-08-01").date(), pd.Timestamp("2024-08-01 06:00"),
             "massive_benzinga:snap:id2", pd.Timestamp("2026-09-10 14:04:12")),   # must NOT overwrite
            ("ZZZ", _E, pd.Timestamp("2024-05-01 16:00"), "massive_benzinga:snap:id3",
             pd.Timestamp("2026-09-10 14:04:12")),                                # event we do not hold
        ])
        first = load(con, seed)
        second = load(con, seed)
        rows = con.execute("SELECT earnings_date, announce_ts_ny, announce_ts_source, "
                           "announce_ts_observed_at FROM earnings ORDER BY earnings_date"
                           ).fetchall()
        n_rows = con.execute("SELECT COUNT(*) FROM earnings").fetchone()[0]
    finally:
        con.close()
    assert first["filled"] == 1 and first["seed_events_not_in_db"] == 1
    assert second["filled"] == 0 and second["already_had_timestamp"] == 2
    assert n_rows == 2                                          # no event created
    assert rows[0][1:] == (pd.Timestamp("2024-05-01 06:45"), "massive_benzinga:snap:id1",
                           pd.Timestamp("2026-09-10 14:04:12"))
    assert rows[1][1:3] == (pd.Timestamp("2024-08-01 16:00"), "yfinance_earnings_dates")


def test_seed_loader_dry_run_writes_nothing():
    from backfills.load_announcement_seed import load
    con = _fresh_db()
    try:
        _seed_event(con, pd.NaT, pd.NaT, None)
        stats = load(con, _seed([("AAA", _E, pd.Timestamp("2024-05-01 06:45"), "src",
                                  pd.Timestamp("2026-09-10"))]), dry_run=True)
        got = _stored(con)
    finally:
        con.close()
    assert stats["would_fill"] == 1 and "filled" not in stats
    assert pd.isna(got[0])


def test_seed_file_must_be_naive_ny_and_one_row_per_event(tmp_path):
    from backfills.load_announcement_seed import read_seed
    aware = _seed([("AAA", _E, pd.Timestamp("2024-05-01 06:45", tz="America/New_York"),
                    "src", pd.Timestamp("2026-09-10"))])
    aware.to_parquet(tmp_path / "aware.parquet")
    with pytest.raises(ValueError, match="naive"):
        read_seed(tmp_path / "aware.parquet")

    dup = _seed([("AAA", _E, pd.Timestamp("2024-05-01 06:45"), "a", pd.Timestamp("2026-09-10")),
                 ("AAA", _E, pd.Timestamp("2024-05-01 16:00"), "b", pd.Timestamp("2026-09-10"))])
    dup.to_parquet(tmp_path / "dup.parquet")
    with pytest.raises(ValueError, match="more than one row"):
        read_seed(tmp_path / "dup.parquet")


def test_the_ingest_runs_against_a_database_that_predates_the_columns():
    """The droplet's DB has the original 7 columns. The first ingest after deploy must
    add the three timing columns and then insert with the named column list."""
    import duckdb
    from ingestion.fetch_earnings_dates import EARNINGS_INSERT_COLS, _INSERT_COL_SQL
    from utilities.db_utilities import create_earnings_table_if_not_exists
    con = duckdb.connect(":memory:")
    try:
        con.execute("""CREATE TABLE earnings (stock TEXT, earnings_date DATE,
                       fiscal_end_date DATE, reported_eps DOUBLE, estimated_eps DOUBLE,
                       surprise_percentage DOUBLE, ingested_at TIMESTAMP)""")
        con.execute("INSERT INTO earnings VALUES ('OLD', DATE '2020-01-01', NULL, 1, 1, 0, "
                    "TIMESTAMP '2020-01-02')")
        create_earnings_table_if_not_exists(con)
        df = pd.DataFrame([{
            "stock": "NEW", "earnings_date": _E, "fiscal_end_date": None,
            "reported_eps": None, "estimated_eps": 0.9, "surprise_percentage": None,
            "ingested_at": pd.Timestamp("2024-04-01"),
            "announce_ts_ny": pd.Timestamp("2024-05-01 16:05"),
            "announce_ts_source": "yfinance_earnings_dates",
            "announce_ts_observed_at": pd.Timestamp("2024-04-01"),
        }])[EARNINGS_INSERT_COLS]
        con.register("tmp_earnings_df", df)
        con.execute(f"INSERT INTO earnings ({_INSERT_COL_SQL}) "
                    f"SELECT {_INSERT_COL_SQL} FROM tmp_earnings_df")
        got = con.execute("SELECT stock, announce_ts_ny FROM earnings ORDER BY stock").fetchall()
    finally:
        con.close()
    assert got == [("NEW", pd.Timestamp("2024-05-01 16:05")), ("OLD", None)]
