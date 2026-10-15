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


# ── 2. The canonical archive: only a verified generation, only on master ─────

import publishing.generation as G
from publishing.generation import GenerationError, Publication
from utilities.db_utilities import create_predictions_table_if_not_exists, read_track_record


@pytest.fixture
def published_setup(tmp_path, monkeypatch):
    """A temporary file standing in for the canonical archive, the pipeline's output files,
    and a generation built from them and confirmed as served (receipt written)."""
    canon = tmp_path / "canonical.duckdb"
    monkeypatch.setattr(guard, "CANONICAL_PREDICTIONS_DB", canon)
    monkeypatch.setattr(config, "PREDICTIONS_DB_PATH", str(canon))
    root = tmp_path / "publication"
    monkeypatch.setattr(G, "PUBLICATION_ROOT", root)
    monkeypatch.setattr(AC, "PUBLICATION_ROOT", root)
    out = tmp_path / "output"
    out.mkdir()
    ev = _events()
    ev[ev["is_pending"]].drop(columns="is_pending").to_parquet(out / G.CALLS_FILE)
    ev[~ev["is_pending"]].to_parquet(out / G.HISTORY_FILE)
    manifest = G.build_generation(src_dir=out, root=root)
    gen = manifest["generation_id"]
    G.write_receipt(root / gen, gen, manifest["files"][G.CALLS_FILE]["sha256"])
    return canon, root / gen, ev


def _rows(db):
    if not Path(db).exists():
        return pd.DataFrame()
    con = duckdb.connect(str(db), read_only=True)
    try:
        return con.execute("SELECT * FROM predictions ORDER BY stock, published_at").df()
    finally:
        con.close()


def test_master_archives_exactly_the_verified_generation(published_setup, monkeypatch):
    canon, gen_dir, ev = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "master")
    assert AC.main(["--published", "--generation", gen_dir.name]) == 0
    got = _rows(canon).set_index("stock")
    published = pd.read_parquet(gen_dir / G.CALLS_FILE)
    pub = G.load_publication(gen_dir)
    today = pub.published_at.tz_convert("America/New_York").tz_localize(None).normalize()
    start, end = work_week_window(today=today)
    want = published[pd.to_datetime(published["earnings_date"]).between(max(start, today), end)]
    assert sorted(got.index) == sorted(want["stock"]) and len(got) > 0
    for s, r in want.set_index("stock").iterrows():
        assert got.at[s, "tier"] == r["risk_tier"]
        assert got.at[s, "model_b_hist_mean_abs"] == r["model_b_hist_mean_abs"]
        assert (pd.isna(got.at[s, "risk_score"]) and pd.isna(r["risk_score"])) or \
            got.at[s, "risk_score"] == r["risk_score"]
    assert (got["model_version"] == "0.4.0").all()
    assert (got["generation_id"] == gen_dir.name).all() and (got["revision"] == 1).all()
    assert (got["artifact_sha256"] == G.sha256_file(gen_dir / G.CALLS_FILE)).all()
    assert (pd.to_datetime(got["published_at"], utc=True) == pub.published_at).all()


def test_archiving_the_same_generation_again_adds_nothing(published_setup, monkeypatch):
    canon, gen_dir, _ = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "master")
    AC.main(["--published", "--generation", gen_dir.name])
    first = _rows(canon)
    AC.main(["--published", "--generation", gen_dir.name])
    pd.testing.assert_frame_equal(first, _rows(canon))


@pytest.mark.parametrize("branch", ["methodology-rebuild", "HEAD", "(not a git checkout)"])
def test_non_master_cannot_create_canonical_calls(published_setup, monkeypatch, capsys, branch):
    canon, gen_dir, _ = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: branch)
    assert AC.main(["--published", "--generation", gen_dir.name]) == 3
    assert f"current branch: {branch}" in capsys.readouterr().err
    assert not canon.exists()


def test_canonical_archive_needs_a_verified_generation_even_on_master(published_setup, monkeypatch):
    canon, gen_dir, _ = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "master")
    assert AC.main(["--calls", str(gen_dir / G.CALLS_FILE)]) == 3       # no generation
    assert AC.main(["--published"]) == 3                                  # no id
    assert not canon.exists()


def test_the_writer_itself_refuses_the_canonical_archive_without_a_receipt(published_setup, monkeypatch):
    """The auditor's bypass: calling archive_calls() directly. The check is in the writer."""
    canon, gen_dir, _ = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "master")
    unverified = AC.scratch_publication(gen_dir / G.CALLS_FILE)
    with pytest.raises(guard.PublicationRefused):
        AC.archive_calls(pd.read_parquet(gen_dir / G.CALLS_FILE), str(canon), unverified)
    assert not canon.exists()


def test_a_generation_without_a_receipt_is_never_archived(published_setup, monkeypatch, tmp_path):
    canon, gen_dir, _ = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "master")
    (gen_dir / G.RECEIPT).unlink()
    assert AC.main(["--published", "--generation", gen_dir.name]) == 3
    assert not canon.exists()


def test_an_altered_generation_is_never_archived(published_setup, monkeypatch):
    canon, gen_dir, ev = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "master")
    calls = pd.read_parquet(gen_dir / G.CALLS_FILE)
    calls.loc[0, "risk_score"] = 1.0
    calls.to_parquet(gen_dir / G.CALLS_FILE)
    assert AC.main(["--published", "--generation", gen_dir.name]) == 3
    assert not canon.exists()


def test_a_stale_publication_is_not_archived(published_setup, monkeypatch):
    canon, gen_dir, _ = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "master")
    import json
    rec = json.loads((gen_dir / G.RECEIPT).read_text())
    rec["published_at"] = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)).isoformat()
    (gen_dir / G.RECEIPT).write_text(json.dumps(rec))
    assert AC.main(["--published", "--generation", gen_dir.name]) == 3
    assert not canon.exists()


def test_the_receipt_requires_the_served_generation_and_bytes(tmp_path, monkeypatch):
    out = tmp_path / "output"; out.mkdir()
    ev = _events()
    ev[ev["is_pending"]].to_parquet(out / G.CALLS_FILE); ev.to_parquet(out / G.HISTORY_FILE)
    m = G.build_generation(src_dir=out, root=tmp_path / "pub")
    gen_dir = tmp_path / "pub" / m["generation_id"]
    sha = m["files"][G.CALLS_FILE]["sha256"]
    with pytest.raises(GenerationError):
        G.write_receipt(gen_dir, "some-older-generation", sha)            # pointer not moved
    with pytest.raises(GenerationError):
        G.write_receipt(gen_dir, m["generation_id"], "0" * 64)            # other bytes served
    assert not (gen_dir / G.RECEIPT).exists()
    first = G.write_receipt(gen_dir, m["generation_id"], sha)
    assert pd.Timestamp(first["published_at"]).tzinfo is not None
    assert G.write_receipt(gen_dir, m["generation_id"], sha) == first     # retry: unchanged


def test_build_refuses_calls_from_another_model_version(tmp_path):
    out = tmp_path / "output"; out.mkdir()
    ev = _events()
    p = ev[ev["is_pending"]].assign(model_version="0.3.1")
    p.to_parquet(out / G.CALLS_FILE); ev.to_parquet(out / G.HISTORY_FILE)
    with pytest.raises(GenerationError):
        G.build_generation(src_dir=out, root=tmp_path / "pub")


def test_the_archived_version_is_the_artifact_s_not_config_s(published_setup, monkeypatch):
    canon, gen_dir, _ = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "master")
    monkeypatch.setattr(config, "MODEL_VERSION", "9.9.9")
    assert AC.main(["--published", "--generation", gen_dir.name]) == 0
    assert set(_rows(canon)["model_version"]) == {"0.4.0"}


def test_check_live_blocks_a_live_generation_until_receipted_and_archived(published_setup, monkeypatch, capsys):
    canon, gen_dir, _ = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "master")
    gen = gen_dir.name
    receipt = (gen_dir / G.RECEIPT).read_text()
    (gen_dir / G.RECEIPT).unlink()
    assert G.main(["check-live", gen]) == 5                         # live, no receipt
    assert f"receipt {gen} --served-pointer {gen}" in capsys.readouterr().err
    (gen_dir / G.RECEIPT).write_text(receipt)
    assert G.main(["check-live", gen]) == 5                         # receipt, not archived
    assert AC.main(["--published", "--generation", gen]) == 0
    assert (gen_dir / G.ARCHIVED).exists()
    assert G.main(["check-live", gen]) == 0                         # fully recorded
    assert G.main(["check-live", ""]) == 0                          # first publication
    assert G.main(["check-live", "built-elsewhere"]) == 0           # warned, not blocked


def test_finishing_an_interrupted_generation_keeps_its_own_calls(published_setup, monkeypatch):
    """Recovery after a stop between activation and receipt: the SAME generation is
    receipted and archived — its calls, its hash — and published_at is the read-back time,
    never earlier than it was visible."""
    canon, gen_dir, _ = published_setup
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "master")
    (gen_dir / G.RECEIPT).unlink()
    before = pd.Timestamp.now(tz="UTC")
    m = G.verify_generation(gen_dir)
    G.write_receipt(gen_dir, gen_dir.name, m["files"][G.CALLS_FILE]["sha256"])
    assert AC.main(["--published", "--generation", gen_dir.name]) == 0
    rows = _rows(canon)
    assert set(rows["generation_id"]) == {gen_dir.name}
    assert (rows["artifact_sha256"] == m["files"][G.CALLS_FILE]["sha256"]).all()
    assert (pd.to_datetime(rows["published_at"], utc=True) >= before).all()


def test_the_real_canonical_path_is_recognised():
    assert guard.is_canonical_archive(REPO / "db" / "predictions.duckdb")
    assert guard.is_canonical_archive("db/predictions.duckdb") == (Path.cwd() == REPO)
    assert not guard.is_canonical_archive(REPO / "output" / "scratch.duckdb")


# ── 3. Prospective and append-only: the auditor's H1 and H2 reproductions ─────

NY = "America/New_York"


def _call(stock, ed, tier, score, cutoff, version="0.4.0"):
    return {"stock": stock, "earnings_date": pd.Timestamp(ed), "is_pending": True,
            "risk_tier": tier, "risk_score": score, "model_b_hist_mean_abs": score / 1000,
            "model_b_n_prior": 20, "score_asof_date": pd.Timestamp(cutoff),
            "model_version": version}


def _publish(db, calls, when, gen):
    pub = Publication(generation_id=gen, published_at=pd.Timestamp(when, tz=NY).tz_convert("UTC"),
                      artifact_sha256="x" * 64, model_version="0.4.0",
                      calls_path=Path("unused"), verified=False)
    return AC.archive_calls(pd.DataFrame(calls), str(db), pub)


def test_a_same_day_change_appends_a_revision_and_the_first_call_stands(tmp_path):
    """H2: High Alert / 93 then Normal / 10 on the same day. Both are kept; the track
    record holds the first (it was public first)."""
    db = tmp_path / "p.duckdb"
    c1 = [_call("AAA", "2026-10-15", "High Alert", 93.0, "2026-10-09")]
    c2 = [_call("AAA", "2026-10-15", "Normal", 10.0, "2026-10-09")]
    assert _publish(db, c1, "2026-10-12 09:00", "g1")["inserted"] == 1
    assert _publish(db, c2, "2026-10-12 15:00", "g2")["inserted"] == 1
    rows = _rows(db)
    assert rows[["tier", "risk_score", "revision", "generation_id"]].values.tolist() == \
        [["High Alert", 93.0, 1, "g1"], ["Normal", 10.0, 2, "g2"]]
    con = duckdb.connect(str(db), read_only=True)
    tr = read_track_record(con)
    view = con.execute("SELECT * FROM predictions_track_record").df()
    con.close()
    for t in (tr, view):
        assert t["risk_score"].tolist() == [93.0] and t["tier"].tolist() == ["High Alert"]


def test_an_identical_republication_adds_no_row(tmp_path):
    db = tmp_path / "p.duckdb"
    c = [_call("AAA", "2026-10-15", "High Alert", 93.0, "2026-10-09")]
    _publish(db, c, "2026-10-12 09:00", "g1")
    assert _publish(db, c, "2026-10-12 15:00", "g2") == {"inserted": 0, "unchanged": 1, "retried": 0}
    assert _publish(db, c, "2026-10-12 09:00", "g1") == {"inserted": 0, "unchanged": 0, "retried": 1}
    assert len(_rows(db)) == 1


def _announcements(ts, observed):
    return pd.DataFrame({"stock": ["AAA"], "earnings_date": [pd.Timestamp("2026-10-12")],
                         "announce_ts_ny": [pd.Timestamp(ts) if ts else pd.NaT],
                         "announce_ts_observed_at": [pd.Timestamp(observed) if observed else pd.NaT]})


@pytest.mark.parametrize("published, ts, observed, eligible", [
    # H1 as the auditor reproduced it: report Oct 12, published Oct 12, cutoff Oct 9
    ("2026-10-12 10:00", None, None, False),                                  # timing unknown
    ("2026-10-12 06:00", "2026-10-12 07:00", "2026-10-13 06:00", True),       # before a BMO
    ("2026-10-12 10:00", "2026-10-12 07:00", "2026-10-13 06:00", False),      # after a BMO
    ("2026-10-12 10:00", "2026-10-12 16:05", "2026-10-13 06:00", True),       # before an AMC
    ("2026-10-12 10:00", "2026-10-12 16:05", "2026-10-11 06:00", False),      # only a schedule
    ("2026-10-11 18:00", None, None, True),                                   # the day before
])
def test_only_publications_before_the_announcement_count(tmp_path, published, ts, observed, eligible):
    db = tmp_path / "p.duckdb"
    _publish(db, [_call("AAA", "2026-10-12", "High Alert", 93.0, "2026-10-09")], published, "g1")
    con = duckdb.connect(str(db), read_only=True)
    with_timing = read_track_record(con, _announcements(ts, observed))
    conservative = read_track_record(con)
    view = con.execute("SELECT * FROM predictions_track_record").df()
    con.close()
    assert (len(with_timing) == 1) == eligible
    before_day = pd.Timestamp(published) < pd.Timestamp("2026-10-12")
    assert (len(conservative) == 1) == before_day == (len(view) == 1)      # date-only rule


def test_a_late_revision_never_hides_an_earlier_eligible_call(tmp_path):
    """Earliest ELIGIBLE publication, not the latest revision and not the earliest row."""
    db = tmp_path / "p.duckdb"
    _publish(db, [_call("AAA", "2026-10-15", "Elevated", 85.0, "2026-10-09")], "2026-10-12 09:00", "g1")
    _publish(db, [_call("AAA", "2026-10-15", "High Alert", 95.0, "2026-10-09")], "2026-10-15 17:00", "g2")
    con = duckdb.connect(str(db), read_only=True)
    tr = read_track_record(con)
    con.close()
    assert tr["risk_score"].tolist() == [85.0] and tr["generation_id"].tolist() == ["g1"]


# ── 4. Scratch archives work anywhere ─────────────────────────────────────────

def test_scratch_archive_works_on_any_branch(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch.duckdb"
    monkeypatch.setattr(config, "PREDICTIONS_DB_PATH", str(scratch))
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "methodology-rebuild")
    calls = tmp_path / "upcoming.parquet"
    ev = _events()
    ev[ev["is_pending"]].to_parquet(calls)
    assert AC.main(["--calls", str(calls)]) == 0
    rows = _rows(scratch)
    assert len(rows) > 0 and rows["generation_id"].str.startswith("scratch-").all()


def test_env_var_points_validation_runs_at_scratch(tmp_path):
    import subprocess
    out = subprocess.check_output(
        [sys.executable, "-c", "import config, publishing.guard as g; "
                               "print(g.is_canonical_archive(config.PREDICTIONS_DB_PATH))"],
        env={**os.environ, "BREAKWATER_PREDICTIONS_DB": str(tmp_path / "s.duckdb")},
        text=True, cwd=REPO)
    assert out.strip() == "False"


# ── 5. The digest mails the published generation, and only from master ───────

@pytest.fixture
def digest(tmp_path, monkeypatch):
    """Local outputs, a verified generation built from them (calls + PDF reports), and a
    fake SMTP that keeps every message."""
    import cron.cron_weekly_digest as dg
    out = tmp_path / "output"; out.mkdir()
    reports = tmp_path / "run" / "reports"; reports.mkdir(parents=True)
    ev = _events()
    start, _ = work_week_window()
    p = ev[ev["is_pending"]].copy()
    p["earnings_date"] = start + pd.Timedelta(days=1)
    p.to_parquet(out / G.CALLS_FILE)
    ev[~ev["is_pending"]].to_parquet(out / G.HISTORY_FILE)
    (reports / "AAA_report.pdf").write_bytes(b"%PDF published AAA")
    root = tmp_path / "publication"
    monkeypatch.setattr(G, "PUBLICATION_ROOT", root)
    m = G.build_generation(src_dir=out, root=root, reports_dir=reports)
    G.write_receipt(root / m["generation_id"], m["generation_id"], m["files"][G.CALLS_FILE]["sha256"])
    subs = tmp_path / "subs.txt"
    subs.write_text("someone@example.com\n")
    monkeypatch.setattr(dg, "PARQUET_PATH", str(out / G.CALLS_FILE))
    monkeypatch.setattr(dg, "SUBSCRIBERS", str(subs))
    monkeypatch.setattr(dg, "REPORTS_DIR", str(reports))
    sent = []

    class FakeSMTP:
        def __init__(self, *a, **k): sent.append("connect")
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def starttls(self): pass
        def login(self, *a): pass
        def sendmail(self, frm, to, msg): sent.append(msg)
    monkeypatch.setattr(dg.smtplib, "SMTP", FakeSMTP)
    return dg, sent, m["generation_id"], root


def _mail(raw):
    """(html, {attachment name: bytes}) of a sent message."""
    import email
    msg = email.message_from_string(raw)
    html, files = "", {}
    for part in msg.walk():
        if part.get_content_type() == "text/html":
            html = part.get_payload(decode=True).decode()
        elif part.get_filename():
            files[part.get_filename()] = part.get_payload(decode=True)
    return html, files


def _creds(dg, monkeypatch, branch):
    monkeypatch.setattr(dg, "SMTP_USER", "u"); monkeypatch.setattr(dg, "SMTP_PASS", "p")
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: branch)


def test_non_master_cannot_send_the_digest(digest, monkeypatch):
    dg, sent, gen, _ = digest
    _creds(dg, monkeypatch, "methodology-rebuild")
    with pytest.raises(guard.PublicationRefused, match="methodology-rebuild"):
        dg.run_weekly_digest(generation=gen)
    assert sent == []


def test_master_sends_the_published_generation(digest, monkeypatch):
    dg, sent, gen, _ = digest
    _creds(dg, monkeypatch, "master")
    dg.run_weekly_digest(generation=gen)
    html, files = _mail(sent[-1])
    assert "AAA" in html and files == {"AAA_report.pdf": b"%PDF published AAA"}


def test_a_real_send_without_a_generation_is_refused(digest, monkeypatch):
    dg, sent, _, _ = digest
    _creds(dg, monkeypatch, "master")
    with pytest.raises(guard.PublicationRefused, match="--generation"):
        dg.run_weekly_digest()
    assert sent == []


def test_files_changed_after_publication_never_reach_the_email(digest, monkeypatch):
    """The regression: after the dashboard went live, output/ is rewritten (a rerun, a
    hand edit) — new tiers and a new PDF. The email still describes the published calls."""
    dg, sent, gen, _ = digest
    _creds(dg, monkeypatch, "master")
    changed = pd.read_parquet(dg.PARQUET_PATH)
    changed["risk_tier"] = "Normal"                                     # AAA was High Alert
    changed.loc[changed["stock"] == "BBB", "risk_tier"] = "High Alert"
    changed.to_parquet(dg.PARQUET_PATH)
    Path(dg.REPORTS_DIR, "AAA_report.pdf").write_bytes(b"%PDF CHANGED")
    dg.run_weekly_digest(generation=gen)
    html, files = _mail(sent[-1])
    published = pd.read_parquet(G.PUBLICATION_ROOT / gen / G.CALLS_FILE).set_index("stock")
    assert published.at["AAA", "risk_tier"] == "High Alert" and "AAA" in html
    assert "BBB" not in html                                            # the edit is not mailed
    assert files == {"AAA_report.pdf": b"%PDF published AAA"}


@pytest.mark.parametrize("damage", ["calls", "report", "receipt"])
def test_an_altered_or_unconfirmed_generation_is_not_mailed(digest, monkeypatch, damage):
    dg, sent, gen, root = digest
    _creds(dg, monkeypatch, "master")
    d = root / gen
    if damage == "calls":
        c = pd.read_parquet(d / G.CALLS_FILE); c["risk_tier"] = "Normal"; c.to_parquet(d / G.CALLS_FILE)
    elif damage == "report":
        (d / G.REPORTS / "AAA_report.pdf").write_bytes(b"%PDF tampered")
    else:
        (d / G.RECEIPT).unlink()
    with pytest.raises(guard.PublicationRefused, match="REFUSING TO SEND"):
        dg.run_weekly_digest(generation=gen)
    assert sent == []


def test_without_credentials_any_branch_may_preview(digest, monkeypatch, capsys):
    dg, sent, _, _ = digest
    monkeypatch.setattr(dg, "SMTP_USER", ""); monkeypatch.setattr(dg, "SMTP_PASS", "")
    monkeypatch.setattr(guard, "current_branch", lambda repo=None: "methodology-rebuild")
    dg.run_weekly_digest()                                       # local output/, no generation
    assert sent == [] and "SMTP credentials not set" in capsys.readouterr().out


def test_the_digest_names_unscored_events_instead_of_dropping_them(digest):
    dg, *_ = digest
    df = pd.read_parquet(dg.PARQUET_PATH)
    html = dg._build_html(dg._select_stocks(df, {}), "x", unscored=dg._select_unscored(df))
    assert "Not ranked this week:</strong> CCC" in html and "not a low-risk rating" in html
