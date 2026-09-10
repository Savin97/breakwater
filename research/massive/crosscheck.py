"""Independent completeness checks on a cursor walk.

A cursor-paginated API gives no total. "Followed `next_url` until it stopped" is a
statement about the client, not about the data: a walk that silently loses a page boundary
ends in exactly the same way as a complete one. So the acquisition is re-derived two other
ways and the `benzinga_id` sets are compared.

    partitioned   one query per calendar year (`date.gte`/`date.lte`). Each year is small
                  enough to come back in a single response, so this traversal barely uses
                  the cursor at all. It is the primary evidence.
    reverse       the same walk with the sort order flipped.

Both are checks, not evidence about the world, so neither is persisted as a snapshot. The
results are written to `vendor/massive/crosscheck_<snapshot>.json` so the audit report can
quote them without re-running them.

    PYTHONPATH=. .venv/bin/python -m research.massive.crosscheck
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from research.massive import paths
from research.massive.acquire import iter_records, latest_snapshot
from research.massive.client import EARNINGS_PATH, MAX_PAGE_LIMIT, MassiveClient

log = logging.getLogger(__name__)

REVERSE_SORT = "date.desc,ticker.desc"
FORWARD_SORT = "date.asc,ticker.asc"
PARTITION_YEARS = range(2009, 2030)


def _walk_ids(client, params) -> tuple[set, int]:
    ids, pages = set(), 0
    for pages, _url, payload in client.iter_pages(path=EARNINGS_PATH, params=params):
        ids.update(r.get("benzinga_id") for r in payload.get("results") or [])
    return ids, pages


def partitioned_ids(client, years=PARTITION_YEARS, sort: str = FORWARD_SORT):
    """Every id, gathered one calendar year at a time."""
    ids, per_year = set(), {}
    for year in years:
        got, pages = _walk_ids(client, {"limit": MAX_PAGE_LIMIT, "sort": sort,
                                        "date.gte": f"{year}-01-01",
                                        "date.lte": f"{year}-12-31"})
        per_year[str(year)] = {"records": len(got), "responses": pages}
        ids |= got
        log.info("crosscheck: %d -> %d records in %d response(s)", year, len(got), pages)
    return ids, per_year


def crosscheck(snapshot: Path | None = None, client: MassiveClient | None = None,
               out_dir: Path | None = None, reverse: bool = True) -> dict:
    snapshot = Path(snapshot) if snapshot else latest_snapshot()
    if snapshot is None:
        raise SystemExit("no finished snapshot to check")
    have = {r.get("benzinga_id") for r in iter_records(snapshot)}
    client = client or MassiveClient()

    part_ids, per_year = partitioned_ids(client)
    result = {
        "snapshot": snapshot.name,
        "snapshot_sort": FORWARD_SORT,
        "snapshot_ids": len(have),
        "partitioned": {
            "ids": len(part_ids),
            "per_year": per_year,
            "missing_from_partitioned": len(have - part_ids),
            "extra_in_partitioned": len(part_ids - have),
            "identical": have == part_ids,
            "example_differences": sorted((have ^ part_ids))[:20],
        },
    }
    if reverse:
        rev_ids, rev_pages = _walk_ids(client, {"limit": MAX_PAGE_LIMIT, "sort": REVERSE_SORT})
        result["reverse"] = {
            "sort": REVERSE_SORT,
            "ids": len(rev_ids),
            "responses": rev_pages,
            "missing_from_reverse": len(have - rev_ids),
            "extra_in_reverse": len(rev_ids - have),
            "identical": have == rev_ids,
        }
    out_dir = Path(out_dir or paths.MASSIVE_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"crosscheck_{snapshot.name}.json").write_text(json.dumps(result, indent=2))
    return result


def load(snapshot_name: str, out_dir: Path | None = None) -> dict | None:
    f = Path(out_dir or paths.MASSIVE_ROOT) / f"crosscheck_{snapshot_name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--no-reverse", action="store_true")
    args = ap.parse_args(argv)
    r = crosscheck(args.snapshot, reverse=not args.no_reverse)
    print(json.dumps(r, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
