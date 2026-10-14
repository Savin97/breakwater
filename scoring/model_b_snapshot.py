# scoring/model_b_snapshot.py
"""The pinned inputs Model B 0.4.0 was frozen from — an immutable research snapshot.

Exact-match checks need the data AS IT WAS when 0.4.0 was approved: the percentile
reference rebuilds bit-for-bit, production scores reproduce the approved baseline, and
production reproduces the frozen refit's cutoffs, prior counts, signal and ranking on all
46,200 events. output/ cannot hold that data: every pipeline run rewrites
output/events_df.parquet and output/full_df.parquet from the live DB, and ingestion
legitimately revises history (2026-10: one-day earnings-date corrections for EQT, JBHT,
PKG and STZ moved 14 prior counts and the reference sample from 20,491 to 20,493). So the
exact checks read this snapshot, and the routine data tests keep reading output/.

    data/snapshots/model_b_0.4.0_20260928/
      events.parquet              the approved event frame (prices through 2026-09-28)
      session_grid.parquet        every market session in that price data, through 2026-09-28
      refit_oracle_frame.parquet  analysis/model_c_refit's frame (2026-10-03), the oracle
      MANIFEST.json               provenance

Gitignored: the event frame carries vendor-derived announcement timing, which must not
enter the public repo. The files are written once by `create`, made read-only, and checked
against SHA256 below on every load — a missing snapshot is reported as unavailable, a
changed one as corrupt, never silently used. Nothing on the pipeline path imports this
module (testing/test_model_b_snapshot.py).

Model B reads only the market-session grid from the daily frame (`market_session_grid`),
so the snapshot keeps the session dates, not 300 MB of prices.

Provenance (created 2026-10-13 by `create`; MANIFEST.json repeats the sources):
  * events.parquet — byte copy of output/model_b_validation/events_approved_baseline.parquet,
    the event frame the 0.4.0 runs of 2026-10-04 and 2026-10-06 wrote from the DB with prices
    through 2026-09-28 (the same bytes were saved three times over those runs).
  * session_grid.parquet — the unique dates <= 2026-09-28 of master's 2026-10-12
    output/full_df.parquet. The 2026-10-06 daily frame itself had been overwritten by then;
    the grid taken from the later file is shown to be the one 0.4.0 was frozen on because
    with it the snapshot rebuilds scoring/model_b_reference.json bit-for-bit (n 20,491,
    identical quantile values, price data through 2026-09-28), re-scores all 46,200 events
    to the approved Model B columns unchanged, and matches the refit oracle exactly.
  * refit_oracle_frame.parquet — byte copy of output/model_c_refit/frame.parquet, written by
    `analysis.model_c_refit.evaluate build` on 2026-10-03.

Fresh clones: the snapshot is private and not in git (licensed vendor-derived timing in a
public repo). Without it the six exact-match tests — four in
testing/test_model_b_snapshot.py, two in analysis/model_b_validation/tests — skip with
SnapshotUnavailable, and `validate` / `build_model_b_reference --check` stop with it; the
rest of the suite runs. To run them, copy the folder privately from a machine that has it
(a copy is kept in the gitignored output/backups_2026-10-13/snapshots_copy/) into
data/snapshots/; the hashes below prove it is the right one. Never commit it, never
refresh it, never edit SHA256 to make a check pass.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = REPO_ROOT / "data" / "snapshots" / "model_b_0.4.0_20260928"
PRICE_DATA_THROUGH = pd.Timestamp("2026-09-28")

EVENTS = "events.parquet"
SESSION_GRID = "session_grid.parquet"
ORACLE = "refit_oracle_frame.parquet"
MANIFEST = "MANIFEST.json"

# The committed fingerprint of the snapshot. Never edit these to make a check pass: a
# mismatch means the files are not the ones 0.4.0 was approved on.
SHA256 = {
    EVENTS:       "556f47971fe41e8d939f6dea491160625a5c8fc673c0a546258bc79f304fdeca",
    SESSION_GRID: "e466494238a6e0914a1421b0df16eeefdac436c0a902729c48f3476258e9d874",
    ORACLE:       "898aea928c41511b377376d68cbe4775f09499a77768daa0e073b05f99ca1ebd",
}


class SnapshotUnavailable(RuntimeError):
    """The snapshot is not on this machine (it is gitignored)."""


class SnapshotCorrupt(RuntimeError):
    """A snapshot file exists but is not the approved one."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def available(directory: Path = SNAPSHOT_DIR) -> bool:
    return all((directory / name).exists() for name in SHA256)


def verify(directory: Path = SNAPSHOT_DIR, expected: dict | None = None) -> None:
    """Raise SnapshotUnavailable if any file is missing, SnapshotCorrupt if any differs."""
    expected = SHA256 if expected is None else expected
    missing = [n for n in expected if not (directory / n).exists()]
    if missing:
        raise SnapshotUnavailable(
            f"Model B snapshot not on this machine ({directory}; missing {missing}). It is "
            "gitignored; restore it from a machine that has it.")
    bad = {n: _sha256(directory / n) for n in expected}
    bad = {n: h for n, h in bad.items() if h != expected[n]}
    if bad:
        raise SnapshotCorrupt(f"Model B snapshot files differ from the approved ones: "
                              f"{sorted(bad)} in {directory}")


def load(directory: Path = SNAPSHOT_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(events, daily) as of 2026-09-28. `daily` holds only `date` — all Model B reads."""
    verify(directory)
    return (pd.read_parquet(directory / EVENTS), pd.read_parquet(directory / SESSION_GRID))


def load_oracle(directory: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    verify(directory)
    return pd.read_parquet(directory / ORACLE)


def create(events_src, daily_src, oracle_src, directory: Path = SNAPSHOT_DIR) -> dict:
    """Write the snapshot once. Refuses if the directory already exists — a snapshot is
    never refreshed; new data is a new snapshot (and, for the reference, a new model
    version). Returns the SHA-256 of each file, to be committed in SHA256 above."""
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"{directory} exists — snapshots are never overwritten")
    daily = pd.read_parquet(daily_src, columns=["date"])
    dates = pd.to_datetime(daily["date"])
    grid = pd.DataFrame({"date": np.sort(pd.unique(dates[dates <= PRICE_DATA_THROUGH]))})
    if grid["date"].max() != PRICE_DATA_THROUGH:
        raise ValueError(f"{daily_src} does not reach {PRICE_DATA_THROUGH.date()}")

    directory.mkdir(parents=True)
    (directory / EVENTS).write_bytes(Path(events_src).read_bytes())
    grid.to_parquet(directory / SESSION_GRID, index=False)
    (directory / ORACLE).write_bytes(Path(oracle_src).read_bytes())
    hashes = {n: _sha256(directory / n) for n in (EVENTS, SESSION_GRID, ORACLE)}
    (directory / MANIFEST).write_text(json.dumps({
        "purpose": "Pinned inputs of Model B 0.4.0 (scoring/model_b_snapshot.py)",
        "price_data_through": str(PRICE_DATA_THROUGH.date()),
        "created_on": str(date.today()),
        "sources": {EVENTS: str(events_src),
                    SESSION_GRID: f"unique dates <= {PRICE_DATA_THROUGH.date()} of {daily_src}",
                    ORACLE: str(oracle_src)},
        "sha256": hashes,
    }, indent=1) + "\n")
    read_only = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    for p in directory.iterdir():
        os.chmod(p, read_only)
    os.chmod(directory, read_only | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return hashes
