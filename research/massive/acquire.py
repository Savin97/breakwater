"""Acquire the complete Benzinga earnings history into an immutable raw snapshot.

What a snapshot is
------------------
    vendor/massive/earnings/<snapshot_id>/
        pages/page_00001.json.gz ...   the vendor payloads, byte-for-byte as received
        manifest.json                  acquisition metadata + per-page SHA-256
        SHA256SUMS                     the same digests in `sha256sum -c` format

`<snapshot_id>` is `earnings_<UTC acquisition start, compact ISO>`. Nothing ever writes
into a finished snapshot: acquisition happens in a sibling `<snapshot_id>.partial/`
directory and the finished directory only comes into existence by an atomic rename, at
which point every file in it is chmod'ed read-only. `acquire()` refuses outright if the
target snapshot directory already exists.

Why that matters here: this snapshot is the evidence for a source-acceptance decision.
An analysis that can be re-run against a directory somebody topped up in between is not
evidence of anything. The manifest's `snapshot_sha256` — the digest of the ordered list
of page digests — is the one number that pins a whole snapshot.

Interruption
------------
`progress.json` inside the `.partial/` directory records the cursor for the NEXT page
after every page is safely on disk, so an interrupted run resumes exactly where it
stopped rather than restarting or, worse, finishing short. A `.partial/` directory is
never a snapshot: `latest_snapshot()` and the normalizer only ever see finished ones.
"""
import argparse
import gzip
import hashlib
import json
import logging
import os
import shutil
import stat
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from research.massive import paths
from research.massive.client import EARNINGS_PATH, MAX_PAGE_LIMIT, MassiveClient, scrub

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
CHECKSUMS_NAME = "SHA256SUMS"
PROGRESS_NAME = "progress.json"
PAGES_DIR = "pages"

# Deterministic ordering is what makes a cursor walk complete. `last_updated.desc` — the
# vendor default — reorders under us whenever a record is edited mid-walk, which can drop
# or duplicate rows across page boundaries. Report date is immutable for history, so
# `date.asc,ticker.asc` is stable for everything except the projected tail.
ACQUISITION_PARAMS = {"limit": MAX_PAGE_LIMIT, "sort": "date.asc,ticker.asc"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def snapshot_id(started: datetime | None = None) -> str:
    started = started or _utc_now()
    return "earnings_" + started.strftime("%Y%m%dT%H%M%SZ")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _freeze(directory: Path) -> None:
    """Make every file in a finished snapshot read-only, then the directories too."""
    for p in sorted(directory.rglob("*"), reverse=True):
        mode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        if p.is_dir():
            mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        os.chmod(p, mode)
    os.chmod(directory, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
             | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _write_page(pages_dir: Path, page: int, payload: dict) -> dict:
    """Persist one raw payload verbatim (gzipped) and return its manifest entry."""
    name = f"page_{page:05d}.json.gz"
    target = pages_dir / name
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    tmp = target.with_suffix(target.suffix + ".tmp")
    # mtime=0 so the gzip container is a pure function of the payload and the digest is
    # reproducible across runs of the same data.
    with gzip.GzipFile(filename="", mode="wb", fileobj=open(tmp, "wb"), mtime=0) as gz:
        gz.write(body)
    os.replace(tmp, target)
    return {
        "page": page,
        "file": f"{PAGES_DIR}/{name}",
        "records": len(payload.get("results") or []),
        "request_id": payload.get("request_id"),
        "sha256": _sha256_file(target),
        "bytes": target.stat().st_size,
    }


def _load_progress(partial: Path) -> dict:
    f = partial / PROGRESS_NAME
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError:
        log.warning("massive: unreadable %s, restarting this snapshot", f)
        return {}


def _save_progress(partial: Path, progress: dict) -> None:
    tmp = partial / (PROGRESS_NAME + ".tmp")
    tmp.write_text(json.dumps(progress, indent=2, sort_keys=True))
    os.replace(tmp, partial / PROGRESS_NAME)


def latest_snapshot(root: Path | None = None) -> Path | None:
    """The newest FINISHED snapshot, or None. `.partial` directories are invisible."""
    root = Path(root or paths.EARNINGS_SNAPSHOT_ROOT)
    if not root.is_dir():
        return None
    done = [p for p in root.iterdir()
            if p.is_dir() and not p.name.endswith(".partial") and (p / MANIFEST_NAME).exists()]
    return max(done, key=lambda p: p.name) if done else None


def read_manifest(snapshot: Path) -> dict:
    return json.loads((Path(snapshot) / MANIFEST_NAME).read_text())


def iter_records(snapshot: Path):
    """Yield every vendor record in a finished snapshot, in acquisition order."""
    snapshot = Path(snapshot)
    for entry in read_manifest(snapshot)["pages"]:
        with gzip.open(snapshot / entry["file"], "rt", encoding="utf-8") as fh:
            for record in json.load(fh).get("results") or []:
                yield record


def verify_snapshot(snapshot: Path) -> dict:
    """Re-hash every page and confirm it matches the manifest.

    Returns a report rather than raising, so a caller can decide what a mismatch means.
    """
    snapshot = Path(snapshot)
    manifest = read_manifest(snapshot)
    bad, missing = [], []
    digests = []
    for entry in manifest["pages"]:
        f = snapshot / entry["file"]
        if not f.exists():
            missing.append(entry["file"])
            continue
        actual = _sha256_file(f)
        digests.append(actual)
        if actual != entry["sha256"]:
            bad.append(entry["file"])
    recomputed = hashlib.sha256("\n".join(digests).encode()).hexdigest()
    return {
        "snapshot": snapshot.name,
        "pages": len(manifest["pages"]),
        "missing": missing,
        "corrupt": bad,
        "snapshot_sha256": manifest.get("snapshot_sha256"),
        "recomputed_sha256": recomputed,
        "ok": not bad and not missing and recomputed == manifest.get("snapshot_sha256"),
    }


def acquire(root: Path | None = None, client: MassiveClient | None = None,
            params: dict | None = None, path: str = EARNINGS_PATH,
            snapshot_name: str | None = None, max_pages: int | None = None,
            resume: bool = True) -> Path:
    """Walk the vendor's cursor to exhaustion and freeze the result. Returns the path.

    Raises FileExistsError if the finished snapshot already exists — a snapshot is
    immutable evidence and is never re-opened, appended to or overwritten.
    """
    root = Path(root or paths.EARNINGS_SNAPSHOT_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    name = snapshot_name or snapshot_id(started)
    final = root / name
    if final.exists():
        raise FileExistsError(
            f"snapshot {final} already exists; snapshots are immutable — acquire a new one")

    partial = root / f"{name}.partial"
    pages_dir = partial / PAGES_DIR
    pages_dir.mkdir(parents=True, exist_ok=True)

    progress = _load_progress(partial) if resume else {}
    if not resume:
        shutil.rmtree(pages_dir, ignore_errors=True)
        pages_dir.mkdir(parents=True, exist_ok=True)
    entries = progress.get("pages", [])
    start_url = progress.get("next_url")
    first_page = len(entries)
    if entries:
        log.info("massive: resuming %s after page %d (%d records so far)",
                 name, first_page, sum(e["records"] for e in entries))

    client = client or MassiveClient()
    query = dict(params or ACQUISITION_PARAMS)
    t0 = time.monotonic()
    for offset, (_, url, payload) in enumerate(
            client.iter_pages(path=path, params=query, start_url=start_url,
                              max_pages=max_pages), start=1):
        entry = _write_page(pages_dir, first_page + offset, payload)
        entry["url"] = scrub(url)
        entries.append(entry)
        _save_progress(partial, {"pages": entries,
                                 "next_url": scrub(payload.get("next_url") or ""),
                                 "params": query})
        log.info("massive: page %d  records=%d  total=%d",
                 entry["page"], entry["records"], sum(e["records"] for e in entries))

    total = sum(e["records"] for e in entries)
    digests = [e["sha256"] for e in entries]
    manifest = {
        "source": "Massive / Benzinga Earnings",
        "endpoint": f"{client.base_url}{path}",
        "request_params": query,
        "acquired_start_utc": started.isoformat(),
        "acquired_end_utc": _utc_now().isoformat(),
        "elapsed_seconds": round(time.monotonic() - t0, 1),
        "http_requests": client.request_count,
        "pages": entries,
        "record_count": total,
        "snapshot_sha256": hashlib.sha256("\n".join(digests).encode()).hexdigest(),
        "acquired_by": "research/massive/acquire.py",
        "python": sys.version.split()[0],
        # Stated so the normalizer's assumption is auditable from the raw snapshot alone.
        "vendor_time_field_documented_as": "24-hour HH:MM:SS EST (vendor documentation)",
        "notes": "Raw vendor payloads, unmodified. Never a production input.",
    }
    (partial / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    (partial / CHECKSUMS_NAME).write_text(
        "".join(f"{e['sha256']}  {e['file']}\n" for e in entries))
    (partial / PROGRESS_NAME).unlink(missing_ok=True)

    os.replace(partial, final)      # atomic: the snapshot exists only once it is whole
    _freeze(final)
    log.info("massive: snapshot %s complete — %d pages, %d records", name, len(entries), total)
    return final


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-pages", type=int, default=None,
                    help="stop early (smoke test); the snapshot is then knowingly partial")
    ap.add_argument("--verify", metavar="SNAPSHOT", default=None,
                    help="re-hash a finished snapshot against its manifest and exit")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    if args.verify:
        print(json.dumps(verify_snapshot(Path(args.verify)), indent=2))
        return 0

    out = acquire(max_pages=args.max_pages, resume=not args.no_resume)
    print(json.dumps(read_manifest(out) | {"pages": f"<{len(read_manifest(out)['pages'])} entries>"},
                     indent=2, default=str))
    print(f"\nsnapshot: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
