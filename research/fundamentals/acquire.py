"""SEC XBRL Company Facts for every CIK of the SEC pilot's identity chains — the only
network code in this package.

    export SEC_USER_AGENT="<name> <contact email>"
    PYTHONPATH=. .venv/bin/python -m research.fundamentals.acquire [--finish]

Endpoint (official SEC, no third party):

    https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json

Company Facts is a MUTABLE endpoint (every new filing adds facts), so it is captured into
an immutable snapshot with the SEC pilot's `Snapshot` class — resumable `.partial` folder,
per-file SHA-256 manifest, atomic rename, read-only files — under
`data/vendor/sec/xbrl_snapshots/` (gitignored). The CIKs are exactly those of the pilot's
`cik_segments.csv`; there is no second ticker -> CIK system here.

Why Company Facts can be used point in time: every value is listed once PER FILING that
reported it, with that filing's accession number (`accn`) and filing date (`filed`). A
value restated in a later filing appears as a separate entry under the later accession; the
original entry stays. `build.py` keeps only entries whose `accn` is the exact original
10-Q/10-K that the stock's chain attributes to the quarter, so nothing filed later — a
restatement, an amendment, a later comparative — can reach an earlier event.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from research.sec_filings_pilot import acquire, paths

log = logging.getLogger(__name__)

XBRL_SNAPSHOT_ROOT = paths.SEC_ROOT / "xbrl_snapshots"
SEGMENTS = paths.OUT / "cik_segments.csv"
INTERVAL_S = 0.125            # 8 request starts per second (SEC's limit is 10)


def companyfacts_url(cik: int) -> str:
    return f"{acquire.DATA}/api/xbrl/companyfacts/CIK{int(cik):010d}.json"


def companyfacts_name(cik: int) -> str:
    return f"companyfacts/CIK{int(cik):010d}.json"


def chain_ciks() -> list[int]:
    return sorted(pd.read_csv(SEGMENTS)["cik"].astype(int).unique())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finish", action="store_true", help="seal the open snapshot")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    client = acquire.SECClient(min_interval=INTERVAL_S)
    snap = acquire.Snapshot(root=XBRL_SNAPSHOT_ROOT, client=client)
    ciks = chain_ciks()
    missing = []
    for i, cik in enumerate(ciks):
        if snap.fetch(companyfacts_name(cik), companyfacts_url(cik), missing_ok=True) is None:
            missing.append(cik)
        if (i + 1) % 50 == 0:
            log.info("companyfacts %d/%d", i + 1, len(ciks))
    log.info("done: %d CIKs, %d without Company Facts %s, %d requests", len(ciks),
             len(missing), missing, client.n_requests)
    if a.finish:
        print("sealed", snap.finish())
    else:
        print("open snapshot", snap.dir, "- rerun with --finish to seal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
