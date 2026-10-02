"""Polite, cached access to SEC EDGAR — the only module here that touches the network.

SEC endpoints used (all official, no third-party mirrors):

    https://www.sec.gov/files/company_tickers.json            current ticker -> CIK map
    https://data.sec.gov/submissions/CIK##########.json       filing history, newest ~1,000
    https://data.sec.gov/submissions/CIK##########-submissions-NNN.json   older history
    https://www.sec.gov/Archives/edgar/data/<cik>/<acc>/<acc-with-dashes>-index.htm
                                                              filing index: every document,
                                                              its type (EX-99.1 ...) and the
                                                              "Accepted" timestamp
    https://www.sec.gov/Archives/edgar/data/<cik>/<acc>/<file>   a document of the filing

SEC's automated-access rules (https://www.sec.gov/os/accessing-edgar-data): a User-Agent
that identifies the requester with contact information, and at most 10 requests/second.
The User-Agent is read from `SEC_USER_AGENT` and never written to a committed file; this
module refuses to send a request without it. Requests are serialised and spaced at
`MIN_INTERVAL_S` (5/s — half the limit).

Two caches, because SEC serves two kinds of thing
-------------------------------------------------
* **Mutable** endpoints (`company_tickers.json`, `submissions/`) change every day. They are
  captured into a *snapshot*: `snapshots/<id>.partial/` while running (resumable — files
  already there are not refetched), then an atomic rename to `snapshots/<id>/`, a manifest
  with each file's SHA-256 and `retrieved_at`, and every file chmod'ed read-only. A
  finished snapshot is never written to again; a newer picture of SEC is a new snapshot.
* **Archive** documents (`/Archives/edgar/data/...`) are immutable once disseminated. They
  are fetched once into `archives/` and never refetched. `verify_archive()` re-downloads
  a file and compares digests; a difference is appended to `archive_changes.jsonl` and the
  stored file is left as it was — never silently overwritten.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import stat
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from research.sec_filings_pilot import paths

log = logging.getLogger(__name__)

WWW = "https://www.sec.gov"
DATA = "https://data.sec.gov"
TICKERS_URL = f"{WWW}/files/company_tickers.json"
MIN_INTERVAL_S = 0.2
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRIES = 6
USER_AGENT_ENV = "SEC_USER_AGENT"


class SECAccessError(RuntimeError):
    pass


def user_agent() -> str:
    ua = os.environ.get(USER_AGENT_ENV, "").strip()
    if not ua or "@" not in ua:
        raise SECAccessError(
            f"{USER_AGENT_ENV} must be set to '<name/organisation> <contact email>' — SEC "
            "blocks anonymous automated access. It is read from the environment only.")
    return ua


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class SECClient:
    """Serialised, rate-limited GET with retry. `session` is injectable for tests."""

    def __init__(self, session=None, min_interval: float = MIN_INTERVAL_S):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent(),
                                     "Accept-Encoding": "gzip, deflate"})
        self.min_interval = min_interval
        self._last = 0.0
        self.n_requests = 0
        self._lock = threading.Lock()

    def _wait_turn(self) -> None:
        """Request STARTS are spaced >= min_interval apart even across threads, so the rate
        limit holds however many workers share this client."""
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.n_requests += 1

    def get(self, url: str) -> requests.Response:
        for attempt in range(MAX_RETRIES):
            self._wait_turn()
            try:
                r = self.session.get(url, timeout=60)
            except requests.RequestException as e:
                log.warning("GET %s failed (%s), retry %d", url, e, attempt + 1)
                time.sleep(2 ** attempt)
                continue
            if r.status_code in RETRY_STATUSES:
                log.warning("GET %s -> %d, retry %d", url, r.status_code, attempt + 1)
                time.sleep(max(2 ** attempt, 1))
                continue
            return r
        raise SECAccessError(f"GET {url} failed after {MAX_RETRIES} attempts")


# ─────────────────────────────────── mutable: snapshots ──────────────────────────────────
def submissions_url(cik: int) -> str:
    return f"{DATA}/submissions/CIK{int(cik):010d}.json"


def submissions_page_url(name: str) -> str:
    return f"{DATA}/submissions/{name}"


def _make_read_only(p: Path) -> None:
    p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def latest_snapshot(root: Path = paths.SNAPSHOT_ROOT) -> Path:
    done = sorted(p for p in root.glob("sec_*") if p.is_dir() and not p.name.endswith(".partial"))
    if not done:
        raise FileNotFoundError(f"no finished SEC snapshot under {root}")
    return done[-1]


class Snapshot:
    """One capture of SEC's mutable endpoints. Files are stored gzip'ed, bytes as served."""

    def __init__(self, snapshot_id: str | None = None, root: Path = paths.SNAPSHOT_ROOT,
                 client: SECClient | None = None):
        root.mkdir(parents=True, exist_ok=True)
        if snapshot_id is None:
            partial = sorted(root.glob("sec_*.partial"))
            snapshot_id = partial[-1].name.removesuffix(".partial") if partial else \
                "sec_" + utc_now().strftime("%Y%m%dT%H%M%SZ")
        self.id = snapshot_id
        self.final = root / snapshot_id
        if self.final.exists():
            raise FileExistsError(f"{self.final} is a finished snapshot; it is never reopened")
        self.dir = root / f"{snapshot_id}.partial"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "fetch_log.jsonl"
        self.client = client

    def _path(self, name: str) -> Path:
        return self.dir / f"{name}.gz"

    def fetch(self, name: str, url: str, missing_ok: bool = False) -> bytes | None:
        p = self._path(name)
        if p.exists():
            return gzip.decompress(p.read_bytes())
        r = self.client.get(url)
        if r.status_code == 404 and missing_ok:
            self._log(name, url, None, 404)
            return None
        if r.status_code != 200:
            raise SECAccessError(f"{url} -> HTTP {r.status_code}")
        body = r.content
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(body))
        tmp.rename(p)
        self._log(name, url, body, 200)
        return body

    def _log(self, name, url, body, status):
        rec = {"name": name, "url": url, "status": status, "retrieved_at": utc_now().isoformat(),
               "bytes": None if body is None else len(body),
               "sha256": None if body is None else sha256(body)}
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def finish(self) -> Path:
        recs = {}
        for line in self.log_path.read_text().splitlines():
            r = json.loads(line)
            recs[r["name"]] = r        # last word per file
        files = sorted(recs.values(), key=lambda r: r["name"])
        manifest = {"snapshot_id": self.id, "finished_at": utc_now().isoformat(),
                    "n_files": len(files), "files": files,
                    "snapshot_sha256": sha256("".join(
                        f"{r['name']}:{r['sha256']}\n" for r in files).encode())}
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
        self.dir.rename(self.final)
        for p in self.final.rglob("*"):
            if p.is_file():
                _make_read_only(p)
        return self.final


def read_snapshot_file(snapshot_dir: Path, name: str) -> dict | None:
    p = snapshot_dir / f"{name}.gz"
    if not p.exists():
        return None
    return json.loads(gzip.decompress(p.read_bytes()))


def acquire_submissions(snap: Snapshot, ciks) -> None:
    """Main submissions JSON plus every older page it lists, for each CIK."""
    ciks = sorted({int(c) for c in ciks})
    for i, cik in enumerate(ciks):
        body = snap.fetch(f"submissions/CIK{cik:010d}.json", submissions_url(cik), missing_ok=True)
        if body is None:
            continue
        for extra in json.loads(body).get("filings", {}).get("files", []):
            snap.fetch(f"submissions/{extra['name']}", submissions_page_url(extra["name"]))
        if (i + 1) % 50 == 0:
            log.info("submissions %d/%d (requests so far %d)", i + 1, len(ciks),
                     snap.client.n_requests)


# ─────────────────────────────────── immutable: archives ────────────────────────────────
def accession_nodash(acc: str) -> str:
    return acc.replace("-", "")


def archive_url(cik: int, acc: str, filename: str) -> str:
    return f"{WWW}/Archives/edgar/data/{int(cik)}/{accession_nodash(acc)}/{filename}"


def index_url(cik: int, acc: str) -> str:
    return archive_url(cik, acc, f"{acc}-index.htm")


def archive_path(cik: int, acc: str, filename: str, root: Path = paths.ARCHIVE_ROOT) -> Path:
    return root / str(int(cik)) / accession_nodash(acc) / filename


class Archive:
    """Fetch-once store for immutable EDGAR archive files."""

    def __init__(self, client: SECClient | None, root: Path = paths.ARCHIVE_ROOT,
                 manifest: Path = paths.ARCHIVE_MANIFEST, changes: Path = paths.ARCHIVE_CHANGES):
        self.client, self.root, self.manifest, self.changes = client, root, manifest, changes

    def get(self, cik: int, acc: str, filename: str) -> tuple[bytes | None, dict]:
        """(bytes or None, record). Cached files are returned without a request."""
        p = archive_path(cik, acc, filename, self.root)
        url = archive_url(cik, acc, filename)
        if p.exists():
            b = p.read_bytes()
            return b, {"url": url, "status": 200, "cached": True, "bytes": len(b)}
        r = self.client.get(url)
        rec = {"cik": int(cik), "accession": acc, "filename": filename, "url": url,
               "status": r.status_code, "retrieved_at": utc_now().isoformat(),
               "content_type": r.headers.get("Content-Type"), "cached": False}
        if r.status_code != 200:
            self._append(self.manifest, rec)
            return None, rec
        b = r.content
        rec.update(bytes=len(b), sha256=sha256(b))
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_bytes(b)
        tmp.rename(p)
        _make_read_only(p)
        self._append(self.manifest, rec)
        return b, rec

    def verify(self, cik: int, acc: str, filename: str) -> bool:
        """Re-download a cached file; record (never apply) any change. True if identical."""
        p = archive_path(cik, acc, filename, self.root)
        old = p.read_bytes()
        r = self.client.get(archive_url(cik, acc, filename))
        same = r.status_code == 200 and sha256(r.content) == sha256(old)
        if not same:
            self._append(self.changes, {"cik": int(cik), "accession": acc, "filename": filename,
                                        "old_sha256": sha256(old), "status": r.status_code,
                                        "new_sha256": sha256(r.content) if r.status_code == 200
                                        else None, "checked_at": utc_now().isoformat()})
        return same

    _append_lock = threading.Lock()

    @classmethod
    def _append(cls, path: Path, rec: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with cls._append_lock, path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


# ───────────────────────────────────────── CLI ──────────────────────────────────────────
def main(argv=None) -> int:
    """Acquire a snapshot: the ticker map, then submissions for every candidate CIK."""
    from research.sec_filings_pilot import mapping
    ap = argparse.ArgumentParser()
    ap.add_argument("--finish", action="store_true", help="seal the open snapshot")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    snap = Snapshot(client=SECClient())
    tickers = json.loads(snap.fetch("company_tickers.json", TICKERS_URL))
    ciks = mapping.candidate_ciks(tickers)
    log.info("acquiring submissions for %d CIKs", len(ciks))
    acquire_submissions(snap, ciks)
    log.info("done: %d requests this run", snap.client.n_requests)
    if args.finish:
        print("sealed", snap.finish())
    else:
        print("open snapshot", snap.dir, "- rerun with --finish to seal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
