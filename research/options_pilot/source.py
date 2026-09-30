"""DoltHub `post-no-preference/options` — a pinned, cached, read-only query client.

Research-only. Nothing in `pipeline/`, `scoring/`, `ingestion/` or any production path
imports this module, and it never touches a database file.

Pinning
-------
Every query runs `AS OF` one fixed commit (`PINNED_COMMIT`). The DoltHub SQL API does not
accept a commit hash as the ref in the URL, so the query itself carries the pin: every
`{REF}` placeholder in a query becomes `AS OF '<commit>'`. A query without `{REF}` is
refused, so nothing can silently read today's master.

Cache
-----
Each result is stored gzipped under `data/vendor/dolthub_options/<commit>/q/<sha>.json.gz`,
keyed by the SHA-256 of the exact SQL. `data/vendor/` is gitignored: this is third-party
market data and must never be committed. A cached result is returned as-is, so a re-run
reads nothing new from the network and cannot drift.

Why targeted queries
--------------------
The database is several GB. Aggregates over `option_chain` time out on the public API
(55 s deadline), but point lookups on the leading primary-key columns `(date,
act_symbol)` return in ~4 s. Everything below is built from those.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

import requests

OWNER_REPO = "post-no-preference/options"
API = f"https://www.dolthub.com/api/v1alpha1/{OWNER_REPO}/master"
# Head of master when the pilot started (2026-09-29 06:34:57 UTC,
# "volatility_history 2026-09-28 update"). Every figure in RESULTS.md is AS OF this commit.
PINNED_COMMIT = "1ug5hqta1o786faoh8fv89q7grrd00jj"
CACHE_ROOT = Path("data/vendor/dolthub_options")
TIMEOUT_SECS = 300
RETRIES = 4


class QueryError(RuntimeError):
    pass


def cache_dir(commit: str = PINNED_COMMIT) -> Path:
    return CACHE_ROOT / commit / "q"


def pin(sql: str, commit: str = PINNED_COMMIT) -> str:
    if "{REF}" not in sql:
        raise ValueError("query must pin its tables with {REF}")
    return sql.replace("{REF}", f"AS OF '{commit}'")


def _key(sql: str) -> str:
    return hashlib.sha256(sql.encode()).hexdigest()


def query(sql: str, *, commit: str = PINNED_COMMIT, offline: bool = False) -> list[dict]:
    """Rows of `sql` (with `{REF}` pinned), from cache if present.

    Raises QueryError on an API error (a timed-out query is an error, never an empty
    result: an empty list only ever means the database returned no rows).
    """
    pinned = pin(sql, commit)
    path = cache_dir(commit) / f"{_key(pinned)}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as fh:
            return json.load(fh)["rows"]
    if offline:
        raise QueryError(f"not cached (offline): {pinned[:120]}")
    last = None
    for attempt in range(RETRIES):
        try:
            r = requests.get(API, params={"q": pinned}, timeout=TIMEOUT_SECS)
            d = r.json()
        except (requests.RequestException, ValueError) as e:
            last = str(e)
            time.sleep(2 * (attempt + 1))
            continue
        if d.get("query_execution_status") == "Success":
            rows = d.get("rows") or []
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wt") as fh:
                json.dump({"sql": pinned, "commit": commit, "rows": rows}, fh)
            tmp.rename(path)
            return rows
        last = d.get("query_execution_message")
        time.sleep(2 * (attempt + 1))
    raise QueryError(f"{last}: {pinned[:160]}")


# ─────────────────────────────────────── helpers ────────────────────────────────────────
def chain_exists_on(date: str) -> bool:
    return bool(query(f"SELECT date FROM option_chain {{REF}} WHERE date = '{date}' LIMIT 1"))


def chain(date: str, symbol: str) -> list[dict]:
    """Every option_chain row for one (snapshot date, symbol)."""
    sym = symbol.replace("'", "")
    return query("SELECT * FROM option_chain {REF} "
                 f"WHERE date = '{date}' AND act_symbol = '{sym}'")


ROW_LIMIT = 1000        # the public API truncates every result at 1,000 rows
SYMBOLS_PER_BATCH = 25


def chains_batch(date: str, symbols: list[str], *, offline: bool = False) -> dict[str, list[dict]]:
    """option_chain rows for many symbols on one snapshot date, paged under the row cap.

    Pages are ordered by the table's primary key, so LIMIT/OFFSET paging is deterministic;
    a page shorter than ROW_LIMIT ends the walk. Every requested symbol appears in the
    result (an empty list = the source has no rows for it on that date).
    """
    out = {s: [] for s in symbols}
    syms = sorted({s.replace("'", "") for s in symbols})
    for i in range(0, len(syms), SYMBOLS_PER_BATCH):
        chunk = ",".join(f"'{s}'" for s in syms[i:i + SYMBOLS_PER_BATCH])
        off = 0
        while True:
            rows = query("SELECT * FROM option_chain {REF} "
                         f"WHERE date = '{date}' AND act_symbol IN ({chunk}) "
                         "ORDER BY act_symbol, expiration, strike, call_put "
                         f"LIMIT {ROW_LIMIT} OFFSET {off}", offline=offline)
            for r in rows:
                out.setdefault(r["act_symbol"], []).append(r)
            if len(rows) < ROW_LIMIT:
                break
            off += ROW_LIMIT
    return out


def volatility_history(symbol: str) -> list[dict]:
    """Every volatility_history row for one symbol (small table)."""
    sym = symbol.replace("'", "")
    return query(f"SELECT * FROM volatility_history {{REF}} WHERE act_symbol = '{sym}'")
