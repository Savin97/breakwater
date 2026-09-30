"""Item 1A: is the assumed snapshot calendar (volatility_history dates) option_chain's?

Two checks, no outcome read:

1. **Targeted** — every weekday that is NOT on the assumed calendar and lies in a window
   where it could have changed a match: after an event's selected snapshot and at or
   before its call cutoff, or anywhere in the 10-session walk-back window of an event that
   found nothing. If option_chain has rows on such a date, that event's snapshot age was
   overstated (never understated: the date is still <= the cutoff).
2. **Broad sample** — random calendar dates and random off-calendar weekdays, stratified
   by year, probed at date level; plus symbol-level presence on calendar dates for liquid
   and thin names.

    PYTHONPATH=. .venv/bin/python -m research.options_pilot.calendar_check
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from research.options_pilot import source
from research.options_pilot.build import MAX_WALK_SESSIONS, OUT, load_calendar, session_grid

SEED = 11
PER_YEAR_ON = 8
PER_YEAR_OFF = 12
SYMBOLS = ["AAPL", "JPM", "KO", "ALLE", "AIZ", "CPB"]   # liquid ... thin


def next_chain_date(after: str) -> str | None:
    """First option_chain date strictly after `after` — a primary-key seek (<1 s), unlike
    an equality probe of an absent date, which the API answers slowly."""
    r = source.query(f"SELECT date FROM option_chain {{REF}} WHERE date > '{after}' "
                     "ORDER BY date LIMIT 1")
    return r[0]["date"][:10] if r else None


def date_has_chain(d: str) -> bool:
    prev = (pd.Timestamp(d) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return next_chain_date(prev) == d


def targeted_dates(ev: pd.DataFrame, calendar: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    cal = set(pd.to_datetime(calendar))
    gridx = pd.DatetimeIndex(grid)
    rows = []
    for r in ev.itertuples(index=False):
        cut = pd.Timestamp(r.call_cutoff_date)
        ci = gridx.searchsorted(cut, side="right") - 1
        if r.match_status == "matched":
            lo = pd.Timestamp(r.snapshot_date)
        else:
            lo = gridx[max(ci - MAX_WALK_SESSIONS, 0)] - pd.Timedelta(days=1)
        for d in pd.bdate_range(lo + pd.Timedelta(days=1), cut):
            if d not in cal:
                rows.append((r.event_id, d))
    return pd.DataFrame(rows, columns=["event_id", "date"])


def windows(ev: pd.DataFrame, grid: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Merged (exclusive start, inclusive end) matching windows over all events."""
    gridx = pd.DatetimeIndex(grid)
    iv = []
    for r in ev.itertuples(index=False):
        cut = pd.Timestamp(r.call_cutoff_date)
        ci = gridx.searchsorted(cut, side="right") - 1
        lo = pd.Timestamp(r.snapshot_date) if r.match_status == "matched" else \
            gridx[max(ci - MAX_WALK_SESSIONS, 0)] - pd.Timedelta(days=1)
        if lo < cut:
            iv.append((lo, cut))
    iv.sort()
    merged = []
    for lo, hi in iv:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def chain_dates_in(window) -> list[str]:
    """Every option_chain date in (lo, hi], by successive primary-key seeks."""
    lo, hi = window
    out, cur = [], lo.strftime("%Y-%m-%d")
    while True:
        nxt = next_chain_date(cur)
        if nxt is None or pd.Timestamp(nxt) > hi:
            return out
        out.append(nxt)
        cur = nxt


def main() -> int:
    ev = pd.read_parquet(OUT / "event_options.parquet")
    calendar, grid = load_calendar(), session_grid()
    rng = np.random.default_rng(SEED)

    tgt = targeted_dates(ev, calendar, grid)
    win = windows(ev, grid)
    with ThreadPoolExecutor(4) as ex:
        seen = sorted({d for ds in ex.map(chain_dates_in, win) for d in ds})
    calset = set(pd.to_datetime(calendar).strftime("%Y-%m-%d"))
    extra = [d for d in seen if d not in calset]
    tgt["chain_exists"] = tgt["date"].dt.strftime("%Y-%m-%d").isin(extra)
    tgt.to_csv(OUT / "calendar_targeted.csv", index=False)
    pd.DataFrame({"chain_date": seen, "on_calendar": [d in calset for d in seen]}).to_csv(
        OUT / "calendar_window_dates.csv", index=False)
    hit = tgt[tgt["chain_exists"]]
    print(f"targeted: {len(win)} merged windows; option_chain dates found in them {len(seen)}, "
          f"of which NOT on the calendar {len(extra)} {extra}; affected events "
          f"{hit['event_id'].nunique()}", flush=True)

    cal = pd.Series(pd.to_datetime(calendar))
    on, off = [], []
    wd = pd.Series(pd.bdate_range(cal.min(), cal.max()))
    offcal = wd[~wd.isin(set(cal))]
    for y in range(2019, 2027):
        c = cal[cal.dt.year == y]
        o = offcal[offcal.dt.year == y]
        on += list(rng.choice(c.to_numpy(), size=min(PER_YEAR_ON, len(c)), replace=False))
        off += list(rng.choice(o.to_numpy(), size=min(PER_YEAR_OFF, len(o)), replace=False))
    on = sorted(pd.Timestamp(x).strftime("%Y-%m-%d") for x in on)
    off = sorted(pd.Timestamp(x).strftime("%Y-%m-%d") for x in off)
    with ThreadPoolExecutor(4) as ex:
        on_found = list(ex.map(date_has_chain, on))
        off_found = list(ex.map(date_has_chain, off))
        sym = list(ex.map(lambda d: {s: len(v) for s, v in source.chains_batch(d, SYMBOLS).items()}, on))
    samp = pd.concat([
        pd.DataFrame({"date": on, "on_calendar": True, "chain_exists": on_found}),
        pd.DataFrame({"date": off, "on_calendar": False, "chain_exists": off_found})])
    samp.to_csv(OUT / "calendar_sample.csv", index=False)
    symdf = pd.DataFrame(sym, index=on)
    symdf.to_csv(OUT / "calendar_symbol_sample.csv")
    print(f"sample: calendar dates with chain {sum(on_found)}/{len(on)}; "
          f"off-calendar weekdays with chain {sum(off_found)}/{len(off)}")
    print("symbol presence on sampled calendar dates (share with rows):")
    print((symdf > 0).mean().round(3).to_string())
    print("off-calendar dates WITH chain:", [d for d, f in zip(off, off_found) if f])
    print("calendar dates WITHOUT chain:", [d for d, f in zip(on, on_found) if not f])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
