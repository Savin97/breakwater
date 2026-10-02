"""When a filing became public, when a Breakwater call was made, and the rule between them.

    a filing enters an event's information set only if it was public by the call cutoff

The call cutoff (reused, not re-derived)
---------------------------------------
Phase 3's `call_cutoff_date` (`research/phase3_refit/features.py::event_clocks`): the last
market session strictly before the Monday of the report week, where the report date is
min(earnings_date, the session implied by the observed announcement timestamp). The weekly
run scores from that session's close, so the cutoff *instant* is that session's official
NYSE close — 16:00 New York, or 13:00 on an early-close day (the day after Thanksgiving is
often a Friday cutoff). Read from `pandas_market_calendars`, as
`utilities.time_utilities.nyse_is_open` does. The weekly run happens later, never earlier.

SEC's clocks — what was found (RESULTS.md §4)
---------------------------------------------
* `filingDate` (submissions JSON, index page): the official filing date. EDGAR accepts
  submissions 06:00-22:00 ET; one accepted after 17:30 ET gets the NEXT business day's
  date (Forms 3/4/5 until 22:00). So the filing was accepted no later than 22:00 ET on its
  filing date, and never on a later day.
* "Accepted" on the filing index page and `<ACCEPTANCE-DATETIME>` in the `.hdr.sgml`
  header: identical to the second in every filing checked, both New York wall clock.
* `acceptanceDateTime` in the submissions JSON, e.g. `2025-04-11T14:45:50.000Z`: for some
  filers it is true UTC (NVIDIA: 20:27Z = 16:27 EDT, as the index page says); for others
  it is shifted by one EXTRA UTC offset (JPMorgan 06:45 ET is stored as 14:45Z, Apple 16:30
  ET as 02:30Z next day). A morning filing shifted this way still looks plausible, so the
  error cannot be detected per filing. **It is recorded, audited, and never used to admit
  a filing.**

The rule (fixed here before any event was mapped)
------------------------------------------------
    filing_date <  cutoff_date   -> eligible  (accepted by 22:00 ET the day before at the
                                               latest, so before the cutoff close; no
                                               time of day is assumed)
    filing_date == cutoff_date   -> eligible only with a VERIFIED acceptance time (index
                                    page / header, NY wall clock) <= the cutoff close;
                                    otherwise excluded — treated as date-only
    filing_date >  cutoff_date   -> never

Ordering inside an information set (which 8-Ks come "after" the periodic filing) uses
the filing date too: an 8-K is after the periodic filing only on a strictly later filing
date. A same-day 8-K is ambiguous without a verified time and is counted separately, not
included.
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd

from utilities.time_utilities import NY_TZ

EXACT = "exact_verified"           # acceptance from the index page / header (NY wall clock)
DATE_ONLY = "date_only"            # filing date only
_JSON_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")
_INDEX_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
EDGAR_OPEN = pd.Timedelta(hours=6)
EDGAR_CLOSE = pd.Timedelta(hours=22)
LATE_FILING_CUTOFF = pd.Timedelta(hours=17, minutes=30)


def parse_json_acceptance_as_utc(s) -> pd.Timestamp:
    """The submissions-JSON value read literally (as UTC), in naive NY wall clock.

    For AUDIT ONLY: for a large share of filers this literal reading is wrong by one UTC
    offset (module docstring). Raises on an unrecognised format rather than guessing.
    """
    if s is None or s == "" or (isinstance(s, float) and np.isnan(s)):
        return pd.NaT
    if not _JSON_RE.match(str(s)):
        raise ValueError(f"unrecognised acceptanceDateTime {s!r}")
    return pd.Timestamp(str(s)[:-1]).tz_localize("UTC").tz_convert(NY_TZ).tz_localize(None)


def parse_verified_acceptance(s) -> pd.Timestamp:
    """Index-page "Accepted" (`YYYY-MM-DD HH:MM:SS`) or header `YYYYMMDDHHMMSS`, both NY wall
    clock; a tz-aware datetime is converted to NY. Naive datetimes are refused."""
    if s is None or s == "" or (isinstance(s, float) and np.isnan(s)):
        return pd.NaT
    if isinstance(s, str):
        if _INDEX_RE.match(s):
            return pd.Timestamp(s)
        if re.fullmatch(r"\d{14}", s):
            return pd.Timestamp(f"{s[:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}:{s[12:]}")
        raise ValueError(f"unrecognised acceptance {s!r}")
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        raise ValueError("naive datetime given as an acceptance time; pass a string or tz-aware")
    return ts.tz_convert(NY_TZ).tz_localize(None)


@lru_cache(maxsize=1)
def _nyse_closes(start: str = "1995-01-01", end: str = "2030-12-31") -> pd.Series:
    import pandas_market_calendars as mcal
    sch = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    close = sch["market_close"].dt.tz_convert(NY_TZ).dt.tz_localize(None)
    close.index = pd.to_datetime(close.index).normalize()
    return close


def cutoff_instant(cutoff_dates: pd.Series) -> pd.Series:
    """The NYSE close of each cutoff session, naive NY. Raises if a date is not a session —
    the cutoff is a session by construction, so that would be a bug upstream, not a gap."""
    closes = _nyse_closes()
    d = pd.to_datetime(cutoff_dates).dt.normalize()
    out = d.map(closes)
    bad = d.notna() & out.isna()
    if bad.any():
        raise ValueError(f"{int(bad.sum())} cutoff dates are not NYSE sessions, e.g. "
                         f"{d[bad].iloc[0].date()}")
    return out.astype("datetime64[ns]")


def _dt64(x, n=None) -> np.ndarray:
    a = pd.to_datetime(pd.Series(np.asarray(x).ravel())).to_numpy("datetime64[ns]") \
        if not (isinstance(x, np.ndarray) and x.dtype == "datetime64[ns]") else x
    return np.broadcast_to(a, (n,)) if n is not None and a.shape == (1,) else a


def eligible(filing_date, cutoff_date, cutoff_ts, verified_acceptance=None) -> np.ndarray:
    """THE invariant, vectorised (module docstring). Missing dates are never eligible.

    `cutoff_date` / `cutoff_ts` may be scalars or arrays aligned with `filing_date`.
    """
    fd = _dt64(filing_date)
    n = len(fd)
    fd = fd.astype("datetime64[D]").astype("datetime64[ns]")
    cd = _dt64([cutoff_date] if np.ndim(cutoff_date) == 0 else cutoff_date, n)
    cd = cd.astype("datetime64[D]").astype("datetime64[ns]")
    ct = _dt64([cutoff_ts] if np.ndim(cutoff_ts) == 0 else cutoff_ts, n)
    va = np.full(n, np.datetime64("NaT", "ns"), "datetime64[ns]") if verified_acceptance is None \
        else _dt64(verified_acceptance)
    with np.errstate(invalid="ignore"):
        before = fd < cd
        va_day = va.astype("datetime64[D]").astype("datetime64[ns]")
        same_day = (fd == cd) & ~np.isnat(va) & (va_day <= fd) & (va <= ct)
    ok = before | same_day
    return ok & ~np.isnat(fd) & ~np.isnat(cd) & ~np.isnat(ct)


LATE_SAME_DAY_FORMS = frozenset({"3", "4", "5", "3/A", "4/A", "5/A"})


def json_acceptance_consistent(json_utc_ny: pd.Series, filing_date: pd.Series,
                               business_days: pd.DatetimeIndex,
                               form: pd.Series | None = None) -> pd.Series:
    """Is the literal (UTC) reading of the JSON timestamp consistent with EDGAR's rules?

    True only if the NY time lies inside EDGAR hours AND the filing date is either the same
    day (accepted by 17:30; by 22:00 for Forms 3/4/5) or the next business day (accepted
    after 17:30). A False is proof the reading is wrong for that record; a True proves
    nothing (a morning filing shifted by +4/5 h still looks fine). Also applied to the
    VERIFIED index-page times, as a check on the rule itself.
    """
    t = pd.to_datetime(json_utc_ny)
    fd = pd.to_datetime(filing_date).dt.normalize()
    tod = t - t.dt.normalize()
    in_hours = (tod >= EDGAR_OPEN) & (tod <= EDGAR_CLOSE)
    d = t.dt.normalize()
    pos = np.searchsorted(business_days.values, d.values.astype("datetime64[ns]"), side="right")
    nxt = pd.Series(business_days.values[np.minimum(pos, len(business_days) - 1)], index=t.index)
    late_ok = pd.Series(False, index=t.index) if form is None else \
        pd.Series(form, index=t.index).isin(LATE_SAME_DAY_FORMS)
    same = fd.eq(d) & ((tod <= LATE_FILING_CUTOFF) | late_ok)
    next_ok = fd.eq(nxt) & (tod > LATE_FILING_CUTOFF)
    return (in_hours & (same | next_ok)).where(t.notna())
