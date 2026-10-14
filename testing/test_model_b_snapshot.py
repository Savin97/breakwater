"""Model B 0.4.0 against its pinned 2026-09-28 inputs (scoring/model_b_snapshot.py).

Deterministic: these read the immutable snapshot, never output/, so a pipeline run on
refreshed data cannot change them. The snapshot is gitignored; on a machine without it the
exact-match tests skip and say why, and a snapshot that exists but differs fails.
Live-data invariants stay in testing/test_model_b.py.
"""
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scoring import model_b as B
from scoring import model_b_snapshot as SNAP

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def snap():
    try:
        return SNAP.load()
    except SNAP.SnapshotUnavailable as e:
        pytest.skip(str(e))


# ── exact match on the approved inputs ───────────────────────────────────────────────

def test_snapshot_rebuilds_the_frozen_reference_bit_for_bit(snap):
    from scoring.build_model_b_reference import build_payload, reference_sample
    events, daily = snap
    payload = build_payload(reference_sample(events, daily), daily)
    frozen = B.load_reference().meta
    assert payload["reference_n"] == frozen["reference_n"] == 20491
    assert payload["reference_stocks"] == frozen["reference_stocks"]
    assert payload["quantiles"]["values"] == frozen["quantiles"]["values"]
    assert payload["raw_signal_at_p80"] == frozen["raw_signal_at_p80"]
    assert payload["raw_signal_at_p90"] == frozen["raw_signal_at_p90"]
    assert payload["source"]["price_data_through"] == frozen["source"]["price_data_through"] \
        == str(SNAP.PRICE_DATA_THROUGH.date())


def test_snapshot_reproduces_the_approved_scores(snap):
    """The snapshot's event frame is the approved 0.4.0 output; re-scoring it with today's
    code must give back every Model B column unchanged — cutoff, prior count, raw signal,
    percentile, tier, version."""
    events, daily = snap
    rescored = B.score_events(events, daily)
    assert len(rescored) == len(events) == 46200
    for col in B.MODEL_B_COLS:
        pd.testing.assert_series_equal(rescored[col], events[col], check_names=True,
                                       obj=f"Model B column {col}")


def test_snapshot_pending_calls_as_approved(snap):
    events, _ = snap
    pend = events[events["is_pending"].astype(bool)]
    assert pend["risk_tier"].value_counts().to_dict() == \
        {"Normal": 352, "Elevated": 72, "High Alert": 57, "Unscored": 4}


def test_snapshot_files_are_read_only(snap):
    for name in SNAP.SHA256:
        mode = os.stat(SNAP.SNAPSHOT_DIR / name).st_mode
        assert not mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH), name


# ── the snapshot mechanism itself (no real data needed) ──────────────────────────────

def _inputs(tmp_path):
    ev = tmp_path / "ev.parquet"
    pd.DataFrame({"stock": ["A"], "earnings_date": [pd.Timestamp("2026-09-01")]}).to_parquet(ev)
    daily = tmp_path / "daily.parquet"
    dates = pd.to_datetime(["2026-09-25", "2026-09-28", "2026-09-28", "2026-09-29"])
    pd.DataFrame({"stock": ["A", "A", "B", "A"], "date": dates}).to_parquet(daily)
    oracle = tmp_path / "oracle.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(oracle)
    return ev, daily, oracle


def _make_writable(d: Path):
    os.chmod(d, 0o755)
    for p in d.iterdir():
        os.chmod(p, 0o644)


def test_create_writes_read_only_files_and_only_sessions_through_the_cutoff(tmp_path):
    d = tmp_path / "snap"
    hashes = SNAP.create(*_inputs(tmp_path), directory=d)
    try:
        grid = pd.read_parquet(d / SNAP.SESSION_GRID)
        assert grid["date"].tolist() == list(pd.to_datetime(["2026-09-25", "2026-09-28"]))
        assert json.loads((d / SNAP.MANIFEST).read_text())["sha256"] == hashes
        SNAP.verify(d, expected=hashes)
        assert not os.stat(d / SNAP.EVENTS).st_mode & stat.S_IWUSR
        assert not os.stat(d).st_mode & stat.S_IWUSR
    finally:
        _make_writable(d)


def test_create_never_overwrites_a_snapshot(tmp_path):
    d = tmp_path / "snap"
    d.mkdir()
    with pytest.raises(FileExistsError):
        SNAP.create(*_inputs(tmp_path), directory=d)


def test_a_changed_file_is_corrupt_not_used(tmp_path):
    d = tmp_path / "snap"
    hashes = SNAP.create(*_inputs(tmp_path), directory=d)
    _make_writable(d)
    pd.DataFrame({"x": [2]}).to_parquet(d / SNAP.ORACLE)
    with pytest.raises(SNAP.SnapshotCorrupt):
        SNAP.verify(d, expected=hashes)


def test_a_missing_snapshot_is_unavailable(tmp_path):
    with pytest.raises(SNAP.SnapshotUnavailable):
        SNAP.verify(tmp_path / "nowhere")


def test_the_snapshot_is_gitignored():
    path = SNAP.SNAPSHOT_DIR.relative_to(REPO) / SNAP.EVENTS
    assert subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=REPO).returncode == 0


def test_the_pinned_fingerprint_is_complete():
    assert set(SNAP.SHA256) == {SNAP.EVENTS, SNAP.SESSION_GRID, SNAP.ORACLE}
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in SNAP.SHA256.values())


# ── nothing routine can reach it ─────────────────────────────────────────────────────

def test_no_pipeline_or_output_module_loads_the_snapshot():
    entry = ["pipeline.pipeline", "cron.cron_weekly_digest", "analysis.predictions_range",
             "analysis.last_week_results", "marketing.generate_public_track_record",
             "streamlit_dash.streamlit_export", "publishing.archive_calls",
             "scoring.build_model_b_reference"]
    code = ("import sys, importlib\n"
            f"for m in {entry!r}: importlib.import_module(m)\n"
            "print('scoring.model_b_snapshot' in sys.modules)")
    out = subprocess.check_output([sys.executable, "-c", code], text=True, cwd=REPO)
    assert out.strip().splitlines()[-1] == "False", out


def test_only_the_snapshot_module_names_the_snapshot_directory():
    """No other module can write there by accident: the directory is spelled in one place."""
    hits = []
    for f in REPO.rglob("*.py"):
        rel = f.relative_to(REPO)
        if rel.parts[0] in {".venv", "data", "output"} or "__pycache__" in rel.parts:
            continue
        if rel == Path("scoring/model_b_snapshot.py") or rel == Path(__file__).relative_to(REPO):
            continue
        text = f.read_text(errors="ignore")
        if '"snapshots"' in text or "data/snapshots" in text:
            hits.append(str(rel))
    assert hits == []
