# publishing/generation.py
"""Immutable publication generations — what the dashboard serves and what the archive records.

A generation is the set of files ONE publication makes public, frozen when it is built:

    output/publication/<generation_id>/
        upcoming_df.parquet    the calls
        streamlit_df.parquet   past corrected reactions (no score)
        reports/<T>_report.pdf the PDF reports the digest attaches
        manifest.json          generation id, created_at (UTC), git commit, the ARTIFACT's
                               model_version, the percentile reference checksum, and the
                               SHA-256 + row count of every file
        receipt.json           written only after the generation was activated on the
                               server AND the served pointer and file hash were read back
                               and matched: published_at (UTC, timezone-aware)
        archived.json          written by publishing/archive_calls after the canonical
                               archive recorded this generation

On the server (scripts/full_workflow.sh) a generation is uploaded into
`output/generations/<generation_id>/` — invisible — and then made live by replacing the
one-line pointer `output/CURRENT_GENERATION` with an atomic `mv`. The dashboard reads the
pointer on every page run and caches by generation (streamlit_dash/data.py), so once the
pointer moves no viewer is served the previous calls. The archive (publishing/archive_calls)
records a generation only from its receipt: nothing is archived before it is public, and
every archived call carries the generation, the artifact hash and the publication time.

    python -m publishing.generation build                       # prints the generation id last
    python -m publishing.generation receipt GEN --served-pointer P --served-sha S
    python -m publishing.generation check-live GEN              # before replacing GEN

`check-live` closes the one gap the receipt cannot: a generation that went live but whose
run died before its receipt or archive. Replacing it would leave calls customers saw out of
the record, and the next generation would be archived as the first public call. So the
workflow refuses to activate a new generation until the live one is receipted AND
archived, and prints how to finish it (its receipt time is the read-back time — never
earlier than the moment it was visible).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_ROOT = REPO_ROOT / "output" / "publication"
CALLS_FILE = "upcoming_df.parquet"
HISTORY_FILE = "streamlit_df.parquet"
FILES = (CALLS_FILE, HISTORY_FILE)
REPORTS = "reports"                      # PDFs inside a generation: reports/<T>_report.pdf
MANIFEST = "manifest.json"
RECEIPT = "receipt.json"
ARCHIVED = "archived.json"
POINTER = "CURRENT_GENERATION"           # on the server, under output/
# Bumped when the dashboard needs different files or columns; the dashboard refuses a
# generation whose schema it does not know instead of rendering it wrongly.
DASHBOARD_SCHEMA = 1


class GenerationError(RuntimeError):
    """A generation is incomplete, altered, or does not match what was served."""


@dataclass(frozen=True)
class Publication:
    """One publication, as the archive needs it. `verified` is True only when it came from
    a receipt that matched the served generation (load_publication)."""
    generation_id: str
    published_at: pd.Timestamp          # timezone-aware
    artifact_sha256: str                # SHA-256 of the published calls file
    model_version: str                  # read from the artifact, not from config
    calls_path: Path
    verified: bool


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def artifact_model_version(calls: pd.DataFrame) -> str:
    """The single model version the calls file says produced it."""
    versions = sorted(calls["model_version"].dropna().astype(str).unique()) \
        if "model_version" in calls.columns else []
    if len(versions) != 1:
        raise GenerationError(f"calls file must carry exactly one model_version, found {versions}")
    return versions[0]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def build_generation(src_dir=REPO_ROOT / "output", root=PUBLICATION_ROOT, now=None,
                     reports_dir=None) -> dict:
    """Freeze the current output files — and the run's PDF reports, which the digest
    attaches — into a new generation; return its manifest."""
    src_dir, root = Path(src_dir), Path(root)
    if reports_dir is None:
        from utilities.output_utilities import latest_run_output_dir
        reports_dir = Path(latest_run_output_dir()) / "reports"
    reports_dir = Path(reports_dir)
    now = now or datetime.now(timezone.utc)
    for name in FILES:
        if not (src_dir / name).exists():
            raise GenerationError(f"{src_dir / name} not found — run the pipeline first")
    calls = pd.read_parquet(src_dir / CALLS_FILE)
    version = artifact_model_version(calls)
    if version != config.MODEL_VERSION:
        raise GenerationError(f"calls file was produced by model {version}, this code is "
                              f"{config.MODEL_VERSION} — rerun the pipeline")
    shas = {name: sha256_file(src_dir / name) for name in FILES}
    gen_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}-{shas[CALLS_FILE][:10]}"
    gen_dir = root / gen_id
    gen_dir.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in FILES:
        shutil.copyfile(src_dir / name, gen_dir / name)
        if sha256_file(gen_dir / name) != shas[name]:
            raise GenerationError(f"{name} changed while it was being copied")
        files[name] = {"sha256": shas[name], "rows": int(len(pd.read_parquet(gen_dir / name)))}
    pdfs = sorted(reports_dir.glob("*_report.pdf")) if reports_dir.is_dir() else []
    if pdfs:
        (gen_dir / REPORTS).mkdir()
    for pdf in pdfs:
        rel = f"{REPORTS}/{pdf.name}"
        shutil.copyfile(pdf, gen_dir / rel)
        files[rel] = {"sha256": sha256_file(gen_dir / rel)}
    reference = json.loads((REPO_ROOT / "scoring" / "model_b_reference.json").read_text())
    manifest = {
        "generation_id": gen_id,
        "created_at": now.isoformat(),
        "git_commit": _git_commit(),
        "model_version": version,
        "reference_sha256": reference["sha256"],
        "dashboard_schema": DASHBOARD_SCHEMA,
        "files": files,
    }
    (gen_dir / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def verify_generation(gen_dir) -> dict:
    """The manifest, after checking every file against it."""
    gen_dir = Path(gen_dir)
    if not (gen_dir / MANIFEST).exists():
        raise GenerationError(f"{gen_dir} has no {MANIFEST}")
    manifest = json.loads((gen_dir / MANIFEST).read_text())
    if manifest.get("generation_id") != gen_dir.name:
        raise GenerationError(f"{gen_dir} holds the manifest of {manifest.get('generation_id')}")
    for name, meta in manifest["files"].items():
        if not (gen_dir / name).exists() or sha256_file(gen_dir / name) != meta["sha256"]:
            raise GenerationError(f"{gen_dir / name} does not match its manifest")
    return manifest


def write_receipt(gen_dir, served_pointer: str, served_sha256: str, now=None) -> dict:
    """Record that this generation is live — only if the server's pointer names it and the
    file it serves is byte-identical to the manifest. A retry for the same generation
    returns the existing receipt unchanged (the first publication time stands)."""
    gen_dir = Path(gen_dir)
    manifest = verify_generation(gen_dir)
    gen_id = manifest["generation_id"]
    if served_pointer.strip() != gen_id:
        raise GenerationError(f"server serves generation {served_pointer.strip()!r}, not {gen_id}")
    if served_sha256.strip() != manifest["files"][CALLS_FILE]["sha256"]:
        raise GenerationError("the served calls file differs from the generation's")
    if (gen_dir / RECEIPT).exists():
        return json.loads((gen_dir / RECEIPT).read_text())
    receipt = {"generation_id": gen_id,
               "published_at": (now or datetime.now(timezone.utc)).isoformat(),
               "served_pointer": served_pointer.strip(),
               "served_calls_sha256": served_sha256.strip()}
    (gen_dir / RECEIPT).write_text(json.dumps(receipt, indent=1) + "\n")
    return receipt


def load_publication(gen_dir) -> Publication:
    """A verified Publication from a generation's manifest and receipt."""
    gen_dir = Path(gen_dir)
    manifest = verify_generation(gen_dir)
    if not (gen_dir / RECEIPT).exists():
        raise GenerationError(f"{gen_dir.name} has no receipt — it was never verified as served")
    receipt = json.loads((gen_dir / RECEIPT).read_text())
    if (receipt.get("generation_id") != manifest["generation_id"]
            or receipt.get("served_calls_sha256") != manifest["files"][CALLS_FILE]["sha256"]):
        raise GenerationError(f"{gen_dir.name}: receipt does not match the manifest")
    published_at = pd.Timestamp(receipt["published_at"])
    if published_at.tzinfo is None:
        raise GenerationError(f"{gen_dir.name}: published_at carries no timezone")
    calls = pd.read_parquet(gen_dir / CALLS_FILE)
    if artifact_model_version(calls) != manifest["model_version"]:
        raise GenerationError(f"{gen_dir.name}: calls file and manifest disagree on the model")
    return Publication(generation_id=manifest["generation_id"], published_at=published_at,
                       artifact_sha256=manifest["files"][CALLS_FILE]["sha256"],
                       model_version=manifest["model_version"],
                       calls_path=gen_dir / CALLS_FILE, verified=True)


def mark_archived(gen_dir, counts: dict, now=None) -> None:
    """Record that the canonical archive holds this generation (publishing/archive_calls)."""
    (Path(gen_dir) / ARCHIVED).write_text(json.dumps(
        {**counts, "archived_at": (now or datetime.now(timezone.utc)).isoformat()}, indent=1) + "\n")


def unfinished(gen_dir) -> str | None:
    """Why a generation that is (or was) live is not fully recorded yet; None when done."""
    gen_dir = Path(gen_dir)
    if not (gen_dir / RECEIPT).exists():
        return "it went live but has no receipt (the run stopped before the read-back)"
    if not (gen_dir / ARCHIVED).exists():
        return "it has a receipt but was never archived"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build or confirm a publication generation.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="freeze output/ into a new generation; prints its id last")
    r = sub.add_parser("receipt", help="record that GEN is live, after reading the server back")
    r.add_argument("generation")
    r.add_argument("--served-pointer", required=True)
    r.add_argument("--served-sha", required=True)
    c = sub.add_parser("check-live", help="exit 5 if the live generation GEN is not fully recorded")
    c.add_argument("generation", nargs="?", default="")
    a = ap.parse_args(argv)
    if a.cmd == "check-live":
        gen = (a.generation or "").strip()
        if not gen:
            print("No generation is live yet.")
            return 0
        if not (PUBLICATION_ROOT / gen).is_dir():
            print(f"WARNING: live generation {gen} was not built on this machine; cannot "
                  "check its receipt/archive here.", file=sys.stderr)
            return 0
        why = unfinished(PUBLICATION_ROOT / gen)
        if why:
            print(f"\nREFUSING TO REPLACE live generation {gen}: {why}.\n"
                  f"Finish it first (same generation, nothing rebuilt):\n"
                  f"  ssh <remote> sha256sum <repo>/output/generations/{gen}/{CALLS_FILE}\n"
                  f"  python -m publishing.generation receipt {gen} --served-pointer {gen} --served-sha <sha>\n"
                  f"  python -m publishing.archive_calls --published --generation {gen}\n"
                  f"  python -m cron.cron_weekly_digest --generation {gen}\n", file=sys.stderr)
            return 5
        print(f"Live generation {gen} is receipted and archived.")
        return 0
    try:
        if a.cmd == "build":
            m = build_generation()
            print(f"Built generation {m['generation_id']} ({m['files'][CALLS_FILE]['rows']} calls, "
                  f"model {m['model_version']})")
            print(m["generation_id"])
        else:
            rec = write_receipt(PUBLICATION_ROOT / a.generation, a.served_pointer, a.served_sha)
            print(f"Generation {rec['generation_id']} confirmed live at {rec['published_at']}")
    except GenerationError as e:
        print(f"\nPUBLICATION FAILED: {e}\n", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
