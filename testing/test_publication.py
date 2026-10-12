"""Scoring ≠ publishing ≠ archival.

  * Scoring and local outputs (run_pipeline / stage 5) never write the prediction archive,
    on any branch — they produce files under output/ and nothing else.
  * The canonical archive (db/predictions.duckdb) is written only by
    `publishing.archive_calls --published`, on master, from today's published calls file.
  * A scratch archive (BREAKWATER_PREDICTIONS_DB) can be written anywhere.
  * The digest refuses a real send off master.

Every test uses temporary files; the real canonical archive's hash is asserted unchanged.

Run with:  pytest testing/test_publication.py -v
"""
import ast
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import duckdb
import numpy as np
import pandas as pd
import pytest

import config
import publishing.guard as guard
import publishing.archive_calls as AC
from scoring import model_b as B
from utilities.data_utilities import work_week_window

REPO = Path(__file__).resolve().parents[1]
CANONICAL = REPO / "db" / "predictions.duckdb"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope="module", autouse=True)
def canonical_untouched():
    before = _sha(CANONICAL) if CANONICAL.exists() else None
    yield
    after = _sha(CANONICAL) if CANONICAL.exists() else None
    assert before == after, "a publication test modified db/predictions.duckdb"


def _events(today=None):
    """Scored event frame: pending calls this coming work week and within 7 days (so the
    reports run), plus a corrected history per stock."""
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    start, _ = work_week_window(today=today)
    soon = [max(today + pd.Timedelta(days=1), start), max(today + pd.Timedelta(days=2), start)]
    soon = [d if d.weekday() < 5 else d + pd.Timedelta(days=7 - d.weekday()) for d in soon]
    pend = pd.DataFrame({
        "stock": ["AAA", "BBB", "CCC"],
        "earnings_date": [soon[0], soon[1], soon[1]],
        "model_b_hist_mean_abs": [0.09, 0.03, 0.06],
        "model_b_n_prior": [30, 25, 4],
        "expected_move_pct": [0.08, np.nan, np.nan], "atm_iv": [0.5, np.nan, np.nan],
        "iv_snapshot_date": [today - pd.Timedelta(days=2), pd.NaT, pd.NaT],
    })
    pend["is_pending"] = True
    pend = B.engineer_risk_tier(B.engineer_risk_score(pend))
    hist = pd.DataFrame([{"stock": s, "earnings_date": d, "is_pending": False,
                          "reaction_3d_anchored": 0.03 * (-1) ** k,
                          "abs_reaction_3d_anchored": 0.03,
                          "reaction_3d_anchored_status": "available"}
                         for s in ["AAA", "BBB", "CCC"]
                         for k, d in enumerate(pd.date_range("2025-01-14", periods=6, freq="91D"))])
    ev = pd.concat([hist, pend], ignore_index=True)
    ev["sector"], ev["sub_sector"] = "Technology", "Software"
    ev["score_asof_date"] = today - pd.Timedelta(days=3)
    ev["model_version"] = "0.4.0"
    ev["announce_window"] = "AMC"
    ev["is_extreme_move_anchored"] = np.where(ev["is_pending"], np.nan, 0.0)
    ev["is_earnings_day"] = (~ev["is_pending"]).astype(int)
    return ev


# ── 1. Scoring / local outputs never archive ──────────────────────────────────

@pytest.mark.parametrize("path", ["pipeline/pipeline.py", "pipeline/stage1.py", "pipeline/stage2.py",
                                  "pipeline/stage3.py", "pipeline/stage4.py", "pipeline/stage5.py",
                                  "pipeline/events.py", "main.py"])
def test_the_pipeline_never_imports_the_archiver(path):
    tree = ast.parse((REPO / path).read_text())
    mods = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    mods += [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not [m for m in mods if m.startswith("publishing")], (path, mods)
    called = {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
              for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not called & {"archive_calls", "save_predictions_snapshot", "main"} - {None} or \
        path == "main.py", (path, called)


def test_stage5_writes_local_outputs_and_leaves_the_archive_alone(tmp_path, monkeypatch):
    """Run the real stage 5 in a sandbox whose db/predictions.duckdb is a copy of the
    canonical archive: reports, calendar, parquets and chart appear; the archive copy is
    byte-for-byte unchanged."""
    if not CANONICAL.exists():
        pytest.skip("db/predictions.duckdb not present")
    for d in ("report/templates", "report/img"):
        shutil.copytree(REPO / d, tmp_path / d)
    (tmp_path / "data").mkdir()
    shutil.copy(REPO / "data" / "sp500_full_info.csv", tmp_path / "data")
    (tmp_path / "db").mkdir()
    sandbox_db = tmp_path / "db" / "predictions.duckdb"
    shutil.copy(CANONICAL, sandbox_db)
    before = _sha(sandbox_db)

    monkeypatch.chdir(tmp_path)
    import utilities.output_utilities as ou
    monkeypatch.setattr(ou, "_run_output_dir", None)
    monkeypatch.setattr(config, "PREDICTIONS_DB_PATH", "db/predictions.duckdb")
    from pipeline.stage5 import stage5

    ev = _events()
    stage5(pd.DataFrame({"stock": ["AAA"], "date": [pd.Timestamp.today().normalize()]}), ev)

    run_dir = Path(ou._run_output_dir)
    assert sorted(p.name for p in (run_dir / "reports").glob("*.pdf")) == \
        ["AAA_report.pdf", "BBB_report.pdf", "CCC_report.pdf"]
    assert (run_dir / "weekly_calendar.html").exists()
    for f in ("full_df.parquet", "events_df.parquet", "streamlit_df.parquet", "upcoming_df.parquet"):
        assert (tmp_path / "output" / f).exists(), f
    assert _sha(sandbox_db) == before


# ── 2. The canonical archive: --published, master, today's calls only ────────

@pytest.fixture
def published_setup(tmp_path, monkeypatch):
    """A temporary file standing in for the canonical archive, and a calls parquet."""
    canon = tmp_path / "canonical.duckdb"
    monkeypatch.setattr(guard, "CANONICAL_PREDICTIONS_DB", canon)
    monkeypatch.setattr(config, "PREDICTIONS_DB_PATH", str(canon))
    calls = tmp_path / "upcoming_df.parquet"
    ev = _events()
    ev[ev["is_pending"]].drop(columns="is_pending").to_parquet(calls)
    return canon, calls, ev


def _rows(db):
    if not Path(db).exists():
        return pd.DataFrame()
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute("SELECT * FROM predictions ORDER BY stock").df()
    finally:
        con.close()


def test_master_publication_archives_exactly_the_published_calls(published_setup, monkeypatch):
    canon, calls, ev = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "master")
    assert AC.main(["--published", "--calls", str(calls)]) == 0
    got = _rows(canon).set_index("stock")
    published = pd.read_parquet(calls)
    start, end = work_week_window()
    want = published[pd.to_datetime(published["earnings_date"]).between(max(start, pd.Timestamp.today().normalize()), end)]
    assert sorted(got.index) == sorted(want["stock"])
    for s, r in want.set_index("stock").iterrows():
        assert got.at[s, "tier"] == r["risk_tier"]
        assert got.at[s, "model_b_hist_mean_abs"] == r["model_b_hist_mean_abs"]
        assert (pd.isna(got.at[s, "risk_score"]) and pd.isna(r["risk_score"])) or \
            got.at[s, "risk_score"] == r["risk_score"]
    assert (got["model_version"] == "0.4.0").all()


def test_same_day_rerun_upserts(published_setup, monkeypatch):
    canon, calls, _ = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "master")
    AC.main(["--published", "--calls", str(calls)])
    n = len(_rows(canon))
    AC.main(["--published", "--calls", str(calls)])
    assert len(_rows(canon)) == n > 0


@pytest.mark.parametrize("branch", ["methodology-rebuild", "HEAD", "(not a git checkout)"])
def test_non_master_cannot_create_canonical_calls(published_setup, monkeypatch, capsys, branch):
    canon, calls, _ = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: branch)
    assert AC.main(["--published", "--calls", str(calls)]) == 3
    assert f"current branch: {branch}" in capsys.readouterr().err
    assert not canon.exists()


def test_canonical_archive_needs_the_published_flag_even_on_master(published_setup, monkeypatch):
    canon, calls, _ = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "master")
    assert AC.main(["--calls", str(calls)]) == 3
    assert not canon.exists()


def test_stale_calls_file_is_not_archived_as_published(published_setup, monkeypatch):
    canon, calls, _ = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "master")
    old = time.time() - 3 * 86400
    os.utime(calls, (old, old))
    assert AC.main(["--published", "--calls", str(calls)]) == 3
    assert not canon.exists()


def test_the_real_canonical_path_is_recognised():
    assert guard.is_canonical_archive(REPO / "db" / "predictions.duckdb")
    assert guard.is_canonical_archive("db/predictions.duckdb") == (Path.cwd() == REPO)
    assert not guard.is_canonical_archive(REPO / "output" / "scratch.duckdb")


# ── 3. Scratch archives work anywhere ─────────────────────────────────────────

def test_scratch_archive_works_on_any_branch(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch.duckdb"
    monkeypatch.setattr(config, "PREDICTIONS_DB_PATH", str(scratch))
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "methodology-rebuild")
    calls = tmp_path / "upcoming.parquet"
    ev = _events()
    ev[ev["is_pending"]].to_parquet(calls)
    assert AC.main(["--calls", str(calls)]) == 0
    assert len(_rows(scratch)) > 0


def test_env_var_points_validation_runs_at_scratch(tmp_path):
    import subprocess
    out = subprocess.check_output(
        [sys.executable, "-c", "import config, publishing.guard as g; "
                               "print(g.is_canonical_archive(config.PREDICTIONS_DB_PATH))"],
        env={**os.environ, "BREAKWATER_PREDICTIONS_DB": str(tmp_path / "s.duckdb")},
        text=True, cwd=REPO)
    assert out.strip() == "False"


# ── 4. The digest refuses a real send off master ──────────────────────────────

@pytest.fixture
def digest(tmp_path, monkeypatch):
    import cron.cron_weekly_digest as dg
    up = tmp_path / "upcoming.parquet"
    ev = _events()
    start, _ = work_week_window()
    p = ev[ev["is_pending"]].copy()
    p["earnings_date"] = start + pd.Timedelta(days=1)
    p.to_parquet(up)
    subs = tmp_path / "subs.txt"
    subs.write_text("someone@example.com\n")
    monkeypatch.setattr(dg, "PARQUET_PATH", str(up))
    monkeypatch.setattr(dg, "SUBSCRIBERS", str(subs))
    monkeypatch.setattr(dg, "REPORTS_DIR", str(tmp_path))
    sent = []

    class FakeSMTP:
        def __init__(self, *a, **k): sent.append("connect")
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, *a): pass
        def sendmail(self, *a): sent.append("sendmail")
    monkeypatch.setattr(dg.smtplib, "SMTP", FakeSMTP)
    return dg, sent


def test_non_master_cannot_send_the_digest(digest, monkeypatch):
    dg, sent = digest
    monkeypatch.setattr(dg, "SMTP_USER", "u"); monkeypatch.setattr(dg, "SMTP_PASS", "p")
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "methodology-rebuild")
    with pytest.raises(guard.PublicationRefused, match="methodology-rebuild"):
        dg.run_weekly_digest()
    assert sent == []


def test_master_sends_the_digest(digest, monkeypatch):
    dg, sent = digest
    monkeypatch.setattr(dg, "SMTP_USER", "u"); monkeypatch.setattr(dg, "SMTP_PASS", "p")
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "master")
    dg.run_weekly_digest()
    assert "sendmail" in sent


def test_without_credentials_any_branch_may_preview(digest, monkeypatch, capsys):
    dg, sent = digest
    monkeypatch.setattr(dg, "SMTP_USER", ""); monkeypatch.setattr(dg, "SMTP_PASS", "")
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "methodology-rebuild")
    dg.run_weekly_digest()
    assert sent == [] and "SMTP credentials not set" in capsys.readouterr().out


def test_the_digest_names_unscored_events_instead_of_dropping_them(digest):
    dg, _ = digest
    df = pd.read_parquet(dg.PARQUET_PATH)
    html = dg._build_html(dg._select_stocks(df, {}), "x", unscored=dg._select_unscored(df))
    assert "Not ranked this week:</strong> CCC" in html and "not a low-risk rating" in html
