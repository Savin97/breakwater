"""The source's snapshot calendar.

A DISTINCT/GROUP BY over option_chain times out on the public API, and walking the key
one date at a time is too slow under the API's variable latency. `volatility_history`
is small enough to list by date, and option_chain is written in the same daily commit.
So the calendar is the volatility_history date list, VERIFIED here:

1. every volatility_history date is probed for option_chain rows;
2. a random sample of weekdays that are NOT volatility_history dates is probed for
   option_chain rows (should find none).

(2019-01-01 .. 2020-04-14 was probed exhaustively day by day: identical, 89 of 89.)
A chain date missing from the calendar could only make a snapshot look older than it
is, never newer, so an error here cannot leak post-cutoff information.

    PYTHONPATH=. .venv/bin/python -m research.options_pilot.dates
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from research.options_pilot import source
from research.options_pilot.build import CHAIN_DATES, OUT, VH_DATES

N_OFF_CALENDAR_SAMPLE = 150
SEED = 7


def vh_dates() -> list[str]:
    rows, off = [], 0
    while True:
        r = source.query("SELECT date, COUNT(*) n FROM volatility_history {REF} "
                         f"GROUP BY date ORDER BY date LIMIT 1000 OFFSET {off}")
        rows += r
        if len(r) < 1000:
            break
        off += 1000
    return [r["date"][:10] for r in rows]


def exists(d: str) -> bool:
    return bool(source.query(f"SELECT date FROM option_chain {{REF}} WHERE date = '{d}' LIMIT 1"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    vh = vh_dates()
    pd.DataFrame({"date": vh, "exists": True}).to_csv(VH_DATES, index=False)
    with ThreadPoolExecutor(4) as ex:
        on = list(ex.map(exists, vh))
    pd.DataFrame({"date": vh, "exists": on}).to_csv(CHAIN_DATES, index=False)
    wd = pd.bdate_range("2020-04-15", vh[-1]).strftime("%Y-%m-%d")
    off = sorted(set(wd) - set(vh))
    rng = np.random.default_rng(SEED)
    sample = sorted(rng.choice(off, size=min(N_OFF_CALENDAR_SAMPLE, len(off)), replace=False))
    with ThreadPoolExecutor(4) as ex:
        found = list(ex.map(exists, sample))
    pd.DataFrame({"date": sample, "chain_exists": found}).to_csv(OUT / "off_calendar_check.csv",
                                                                 index=False)
    print(f"vh dates {len(vh)}, with option_chain rows {sum(on)}; "
          f"off-calendar weekdays sampled {len(sample)}, with rows {sum(found)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
