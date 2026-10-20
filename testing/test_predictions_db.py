"""The predictions archive (db/predictions.duckdb) under 0.4.0.

The archive is evidence, not a clean-room schema: old 0.3.1 rows are kept exactly as
written, the schema only grows, and only 0.4.0 calls may ever count toward a track record.
Every test here works on a COPY; the canonical file's hash is asserted unchanged.

Run with:  pytest testing/test_predictions_db.py -v
"""
import hashlib
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import duckdb
import numpy as np
import pandas as pd
import pytest

from utilities.data_utilities import work_week_window

CANONICAL = "db/predictions.duckdb"


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.fixture(scope="module", autouse=True)
def canonical_untouched():
    """No test in this module may change the canonical archive."""
    before = _sha(CANONICAL) if os.path.exists(CANONICAL) else None
    yield
    after = _sha(CANONICAL) if os.path.exists(CANONICAL) else None
    assert before == after, "a test modified db/predictions.duckdb"


@pytest.fixture
def db_copy(tmp_path):
    if not os.path.exists(CANONICAL):
        pytest.skip(f"{CANONICAL} not present")
    dst = tmp_path / "predictions.duckdb"
    shutil.copy(CANONICAL, dst)
    return str(dst)


def _pending_week():
    """Pending Model B calls inside the window the archiver records."""
    start, _ = work_week_window(weeks=1)
    d = start + pd.Timedelta(days=1)
    ev = pd.DataFrame({
        "stock": ["HHH", "NNN", "UUU"],
        "earnings_date": [d, d, d + pd.Timedelta(days=1)],
        "is_pending": True,
        "risk_score": [93.7, 41.0, np.nan],
        "risk_tier": ["High Alert", "Normal", "Unscored"],
        "model_b_hist_mean_abs": [0.081, 0.032, 0.05],
        "model_b_n_prior": [30, 41, 5],
        "score_asof_date": [start - pd.Timedelta(days=3)] * 3,
        "model_version": "0.4.0",
    })
    return ev


def _archive(ev, db, when=None, gen=None):
    """Archive `ev` as a scratch publication (never the canonical file) at `when`."""
    from pathlib import Path
    from publishing.archive_calls import archive_calls
    from publishing.generation import Publication
    when = pd.Timestamp(when) if when is not None else pd.Timestamp.now(tz="UTC")
    pub = Publication(generation_id=gen or f"test-{when.value}", published_at=when,
                      artifact_sha256="0" * 64, model_version="0.4.0",
                      calls_path=Path("unused"), verified=False)
    return archive_calls(ev, db, pub)


def _rows(db, where="TRUE"):
    con = duckdb.connect(db, read_only=True)
    try:
        return con.execute(f"SELECT * FROM predictions WHERE {where} "
                           "ORDER BY stock, earnings_date, prediction_asof_date").df()
    finally:
        con.close()


def test_old_rows_are_preserved_exactly(db_copy):
    old = _rows(db_copy)
    assert len(old) > 0 and set(old["model_version"]) <= {"0.3.1", "0.3.1-preaudit"}
    _archive(_pending_week(), db_copy)
    after = _rows(db_copy, "model_version <> '0.4.0'")
    # Everything as published — version, tier, score, dates, commit, timestamps. Only the
    # track-record flag and its reason belong to the migration (tested below).
    published = [c for c in old.columns if c not in VOID_COLS + PUBLICATION_COLS]
    pd.testing.assert_frame_equal(old[published].reset_index(drop=True),
                                  after[published].reset_index(drop=True))
    # the new columns exist and are NULL on every pre-0.4 row: their publication time was
    # never recorded, so it is not invented
    for col in ("model_b_hist_mean_abs", "model_b_n_prior") + PUBLICATION_COLS:
        assert after[col].isna().all(), col


VOID_COLS = ("void_for_track_record", "void_reason")
PUBLICATION_COLS = ("published_at", "generation_id", "artifact_sha256", "call_sha256", "revision")


def test_every_legacy_call_is_void_with_its_reason_and_nothing_else_moves(db_copy):
    from utilities.db_utilities import (LEGACY_0_3_1_VOID_REASON, LEGACY_STALE_SCORE_NOTE,
                                        create_predictions_table_if_not_exists, read_track_record)
    old = _rows(db_copy)
    con = duckdb.connect(db_copy)
    create_predictions_table_if_not_exists(con)
    create_predictions_table_if_not_exists(con)                    # idempotent
    after = con.execute("SELECT * FROM predictions ORDER BY stock, earnings_date, "
                        "prediction_asof_date").df()
    view = con.execute("SELECT * FROM predictions_track_record").df()
    computed = read_track_record(con)
    con.close()
    assert len(after) == len(old)
    legacy = after[after["model_version"] != "0.4.0"]
    assert legacy["void_for_track_record"].all()
    pre = legacy[legacy["model_version"] == "0.3.1-preaudit"]
    assert pre["void_reason"].str.contains("P4.2").all()
    v031 = legacy[legacy["model_version"] == "0.3.1"]
    expected = LEGACY_0_3_1_VOID_REASON + np.where(v031["score_asof_date"].isna(),
                                                   LEGACY_STALE_SCORE_NOTE, "")
    assert (v031["void_reason"] == expected).all()
    published = [c for c in old.columns if c not in VOID_COLS + PUBLICATION_COLS]
    pd.testing.assert_frame_equal(old[published], after[published])
    assert set(view["model_version"]) <= {"0.4.0"} and set(computed["model_version"]) <= {"0.4.0"}


def test_new_rows_store_the_model_b_fields(db_copy):
    _archive(_pending_week(), db_copy)
    new = _rows(db_copy, "model_version = '0.4.0'").set_index("stock")
    assert sorted(new.index) == ["HHH", "NNN", "UUU"]
    assert new.at["HHH", "tier"] == "High Alert" and new.at["HHH", "risk_score"] == 93.7
    assert new.at["HHH", "model_b_hist_mean_abs"] == 0.081 and new.at["HHH", "model_b_n_prior"] == 30
    assert new.at["UUU", "tier"] == "Unscored" and pd.isna(new.at["UUU", "risk_score"])
    assert new.at["UUU", "model_b_n_prior"] == 5
    assert new["score_asof_date"].notna().all()
    for legacy in ("is_high_conviction", "pre_earnings_drift_flag", "surprise_momentum_flag"):
        assert new[legacy].isna().all(), legacy


def test_a_same_day_change_appends_a_revision_and_keeps_the_original(db_copy):
    ev = _pending_week()
    t0 = pd.Timestamp.now(tz="UTC")
    _archive(ev, db_copy, when=t0, gen="g1")
    n = len(_rows(db_copy))
    ev.loc[0, "risk_score"], ev.loc[0, "model_b_hist_mean_abs"] = 95.5, 0.09
    _archive(ev, db_copy, when=t0 + pd.Timedelta(minutes=5), gen="g2")
    rows = _rows(db_copy)
    assert len(rows) == n + 1                      # only HHH changed; one revision appended
    hhh = rows[(rows["stock"] == "HHH") & (rows["model_version"] == "0.4.0")].sort_values("revision")
    assert hhh["risk_score"].tolist() == [93.7, 95.5] and hhh["revision"].tolist() == [1, 2]


def test_track_record_view_holds_only_0_4_0_calls(db_copy):
    _archive(_pending_week(), db_copy)
    con = duckdb.connect(db_copy, read_only=True)
    tr = con.execute("SELECT * FROM predictions_track_record").df()
    first = con.execute("SELECT * FROM predictions_first_call").df()
    con.close()
    assert set(tr["model_version"]) == {"0.4.0"}
    assert len(tr) == 3
    assert (first["model_version"] != "0.4.0").any()       # the old calls are still there


def test_the_view_takes_the_first_0_4_0_call_even_after_an_older_one(tmp_path):
    from utilities.db_utilities import create_predictions_table_if_not_exists
    db = str(tmp_path / "p.duckdb")
    con = duckdb.connect(db)
    create_predictions_table_if_not_exists(con)
    con.execute("""INSERT INTO predictions (prediction_asof_date, stock, earnings_date, tier,
        risk_score, model_version, score_asof_date, published_at, generation_id) VALUES
        ('2026-09-28', 'AAA', '2026-10-08', 'Elevated', 76, '0.3.1', '2026-09-26', '2026-09-28 12:00:00+00', 'a'),
        ('2026-10-04', 'AAA', '2026-10-08', 'High Alert', 92, '0.4.0', '2026-10-02', '2026-10-04 12:00:00+00', 'b'),
        ('2026-10-05', 'AAA', '2026-10-08', 'High Alert', 93, '0.4.0', '2026-10-02', '2026-10-05 12:00:00+00', 'c'),
        ('2026-10-09', 'BBB', '2026-10-08', 'Normal', 10, '0.4.0', '2026-10-09', '2026-10-09 12:00:00+00', 'd')""")
    create_predictions_table_if_not_exists(con)      # re-creates the views
    tr = con.execute("SELECT * FROM predictions_track_record").df()
    con.close()
    assert tr["stock"].tolist() == ["AAA"]               # BBB's call was made after the event
    assert tr["risk_score"].iloc[0] == 92 and tr["model_version"].iloc[0] == "0.4.0"


def test_eligibility_is_filtered_before_the_first_call_is_taken(tmp_path):
    """A voided or legacy FIRST call must not hide a later eligible 0.4.0 call."""
    from utilities.db_utilities import create_predictions_table_if_not_exists, read_track_record
    db = str(tmp_path / "p.duckdb")
    con = duckdb.connect(db)
    create_predictions_table_if_not_exists(con)
    con.execute("""INSERT INTO predictions (prediction_asof_date, stock, earnings_date, tier,
        risk_score, model_version, score_asof_date, void_for_track_record, published_at, generation_id) VALUES
        -- voided 0.3.1 first, then a legitimate 0.4.0 call
        ('2026-10-04', 'AAA', '2026-10-15', 'Elevated',   76, '0.3.1', '2026-10-02', TRUE, '2026-10-04 12:00:00+00', 'AAA2026-10-04'),
        ('2026-10-11', 'AAA', '2026-10-15', 'High Alert', 92, '0.4.0', '2026-10-09', FALSE, '2026-10-11 12:00:00+00', 'AAA2026-10-11'),
        -- voided 0.4.0 first, then a legitimate 0.4.0 call
        ('2026-10-04', 'BBB', '2026-10-15', 'Normal',     30, '0.4.0', '2026-10-02', TRUE, '2026-10-04 12:00:00+00', 'BBB2026-10-04'),
        ('2026-10-11', 'BBB', '2026-10-15', 'Elevated',   85, '0.4.0', '2026-10-09', FALSE, '2026-10-11 12:00:00+00', 'BBB2026-10-11'),
        -- 0.3.1 call the migration voids, then a legitimate 0.4.0 call
        ('2026-10-04', 'CCC', '2026-10-15', 'High Alert', 90, '0.3.1', '2026-10-02', FALSE, '2026-10-04 12:00:00+00', 'CCC2026-10-04'),
        ('2026-10-11', 'CCC', '2026-10-15', 'Normal',     12, '0.4.0', '2026-10-09', FALSE, '2026-10-11 12:00:00+00', 'CCC2026-10-11'),
        -- only legacy calls: no track-record entry at all
        ('2026-10-04', 'DDD', '2026-10-15', 'High Alert', 95, '0.3.1', '2026-10-02', FALSE, '2026-10-04 12:00:00+00', 'DDD2026-10-04')""")
    create_predictions_table_if_not_exists(con)
    view = con.execute("SELECT * FROM predictions_track_record ORDER BY stock").df()
    computed = read_track_record(con).sort_values("stock").reset_index(drop=True)
    con.close()
    for tr in (view, computed):
        assert tr["stock"].tolist() == ["AAA", "BBB", "CCC"]
        assert tr["risk_score"].tolist() == [92, 85, 12]
        assert set(tr["model_version"]) == {"0.4.0"}
        assert (tr["prediction_asof_date"].astype(str) == "2026-10-11").all()
    pd.testing.assert_frame_equal(view, computed)


# ── Older archive files: a stale stored view must never become the track record ──

def _db_from_git(rev, tmp_path):
    """The archive exactly as committed at `rev` (pinned, immutable git objects)."""
    try:
        blob = subprocess.check_output(["git", "show", f"{rev}:db/predictions.duckdb"],
                                       cwd=os.path.join(os.path.dirname(__file__), ".."),
                                       stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"git object {rev}:db/predictions.duckdb not available")
    dst = tmp_path / f"predictions_{rev}.duckdb"
    dst.write_bytes(blob)
    return str(dst)


# a7edae6 = 5cc7e64's archive (38 rows): its stored view predates the 0.4.0 version
# filter. a2159de = master's archive (35 rows): no void flag, no score_asof_date, no view.
REBUILD_STALE, MASTER_OLD = "a7edae6", "a2159de"


def test_the_stale_view_really_returned_legacy_calls(tmp_path):
    """The bug being guarded against, reproduced on the real file."""
    con = duckdb.connect(_db_from_git(REBUILD_STALE, tmp_path), read_only=True)
    stale = con.execute("SELECT model_version FROM predictions_track_record").df()
    con.close()
    assert len(stale) == 20 and set(stale["model_version"]) == {"0.3.1"}


@pytest.mark.parametrize("rev", [REBUILD_STALE, MASTER_OLD])
def test_readers_never_show_legacy_calls_from_an_old_archive(rev, tmp_path):
    from analysis.last_week_results import archived_calls
    from marketing.generate_public_track_record import load_archived_calls
    from utilities.db_utilities import read_track_record
    db = _db_from_git(rev, tmp_path)
    before = _sha(db)
    con = duckdb.connect(db, read_only=True)
    assert read_track_record(con).empty
    con.close()
    assert archived_calls(pd.Timestamp("2026-08-01"), pd.Timestamp("2026-12-31"), db).empty
    assert load_archived_calls(db).empty
    assert _sha(db) == before                         # readers never migrate or write


@pytest.mark.parametrize("rev,n", [(REBUILD_STALE, 38), (MASTER_OLD, 35)])
def test_migration_refreshes_the_view_and_voids_every_legacy_call(rev, n, tmp_path):
    from utilities.db_utilities import create_predictions_table_if_not_exists
    db = _db_from_git(rev, tmp_path)
    con = duckdb.connect(db)
    create_predictions_table_if_not_exists(con)
    rows = con.execute("SELECT * FROM predictions").df()
    tr = con.execute("SELECT * FROM predictions_track_record").df()
    con.close()
    assert len(rows) == n and rows["void_for_track_record"].all()
    assert rows["void_reason"].notna().all()
    assert tr.empty


def test_nothing_but_the_definition_reads_the_stored_view():
    """Readers use read_track_record(); only db_utilities names the view in SQL."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    hits = []
    for f in root.rglob("*.py"):
        rel = f.relative_to(root)
        if rel.parts[0] in {".venv", "testing", "output", "data"} or "__pycache__" in rel.parts:
            continue
        if rel == Path("utilities/db_utilities.py"):
            continue
        if "FROM predictions_track_record" in f.read_text(errors="ignore"):
            hits.append(str(rel))
    assert hits == []


def test_env_var_redirects_the_archive(tmp_path):
    out = subprocess.check_output(
        [sys.executable, "-c", "import config; print(config.PREDICTIONS_DB_PATH)"],
        env={**os.environ, "BREAKWATER_PREDICTIONS_DB": str(tmp_path / "x.duckdb")},
        text=True, cwd=os.path.join(os.path.dirname(__file__), ".."))
    assert out.strip() == str(tmp_path / "x.duckdb")


# ── Prospective evaluation: last week's results and the paused track record ───

def _completed_outcomes():
    start, _ = work_week_window(weeks=1)
    d = start + pd.Timedelta(days=1)
    return pd.DataFrame({
        "stock": ["HHH", "NNN", "UUU"], "earnings_date": [d, d, d + pd.Timedelta(days=1)],
        "is_pending": False, "announce_window": ["AMC", "BMO", "UNKNOWN"],
        "reaction_3d_anchored": [-0.12, 0.03, np.nan],
        "abs_reaction_3d_anchored": [0.12, 0.03, np.nan],
        "reaction_3d_anchored_status": ["available", "available", "unresolved_no_timestamp"],
        "abs_reaction_3d": [0.01, 0.30, 0.25],            # legacy, must be ignored
    })


def test_last_week_results_reads_archived_calls_and_corrected_outcomes(db_copy):
    from analysis.last_week_results import archived_calls, results_table, format_results
    _archive(_pending_week(), db_copy)
    start, end = work_week_window(weeks=1)
    tbl = results_table(archived_calls(start, end, db_copy), _completed_outcomes()).set_index("stock")
    assert bool(tbl.at["HHH", "moved_8pct"]) is True             # corrected 12%
    assert bool(tbl.at["NNN", "moved_8pct"]) is False            # legacy 30% ignored
    assert pd.isna(tbl.at["UUU", "moved_8pct"])                   # unresolved, not a miss
    text = format_results(tbl.reset_index(), start, end)
    assert "INTERNAL" in text and "1 of 1 resolved calls moved" in text


def _week_with_more_calls():
    """_pending_week plus two calls on the same day, so the size order is tested."""
    start, _ = work_week_window(weeks=1)
    d = start + pd.Timedelta(days=1)
    extra = pd.DataFrame({
        "stock": ["EEE", "BBB"], "earnings_date": [d, d], "is_pending": True,
        "risk_score": [84.0, 12.0], "risk_tier": ["Elevated", "Normal"],
        "model_b_hist_mean_abs": [0.06, 0.02], "model_b_n_prior": [20, 20],
        "score_asof_date": [start - pd.Timedelta(days=3)] * 2, "model_version": "0.4.0"})
    calls = pd.concat([_pending_week(), extra], ignore_index=True)
    outcomes = pd.concat([_completed_outcomes(), pd.DataFrame({
        "stock": ["EEE", "BBB"], "earnings_date": [d, d], "is_pending": False,
        "announce_window": ["BMO", "AMC"], "reaction_3d_anchored": [-0.05, 0.08],
        "abs_reaction_3d_anchored": [0.05, 0.08],
        "reaction_3d_anchored_status": ["available", "available"],
        "abs_reaction_3d": [0.40, 0.0]})], ignore_index=True)
    return calls, outcomes


def test_last_week_chart_orders_each_day_by_size_of_the_corrected_move(db_copy):
    from analysis.last_week_results import archived_calls, results_table
    from analysis.chart_last_week import chart_rows
    calls, outcomes = _week_with_more_calls()
    _archive(calls, db_copy)
    start, end = work_week_window(weeks=1)
    plotted, unmeasured = chart_rows(results_table(archived_calls(start, end, db_copy), outcomes))
    # Size, not sign and not tier: −12% above +8% above −5% above +3%; legacy moves ignored.
    assert plotted["stock"].tolist() == ["HHH", "BBB", "EEE", "NNN"]
    assert plotted["y"].round(6).tolist() == [12.0, 8.0, 5.0, 3.0]
    assert unmeasured == ["UUU"]                 # unresolved: named, never drawn as a small move


def test_last_week_chart_draws_archived_calls_and_nothing_for_an_empty_week(db_copy, tmp_path):
    from analysis.last_week_results import archived_calls, results_table
    from analysis.chart_last_week import generate_last_week_chart
    calls, outcomes = _week_with_more_calls()
    _archive(calls, db_copy)
    start, end = work_week_window(weeks=1)
    tbl = results_table(archived_calls(start, end, db_copy), outcomes)
    out = tmp_path / "results_chart.png"
    assert generate_last_week_chart(tbl, start, str(out)) == str(out) and out.stat().st_size > 0
    empty = tmp_path / "empty.png"
    assert generate_last_week_chart(tbl.iloc[0:0], start, str(empty)) == ""
    assert not empty.exists()


def test_last_week_chart_wording_is_honest():
    from analysis import chart_last_week as C
    from testing.test_customer_outputs import assert_honest
    for text in (C.MOVE_NOTE, C.NOT_PLOTTED_NOTE, C.UNSCORED_LEGEND):
        assert_honest(text, "last week's chart")
    assert "not low risk" in C.UNSCORED_LEGEND


def test_last_week_results_ignores_pre_0_4_calls(db_copy):
    from analysis.last_week_results import archived_calls
    calls = archived_calls(pd.Timestamp("2026-08-31"), pd.Timestamp("2026-10-02"), db_copy)
    assert calls.empty                                   # the archive holds only 0.3.1 calls


def test_track_record_is_paused_and_writes_nothing(tmp_path):
    from marketing.generate_public_track_record import (generate_public_track_record,
                                                        build_public_track_record,
                                                        TrackRecordPaused, main)
    assert generate_public_track_record(tmp_path / "x.json") is None
    assert not (tmp_path / "x.json").exists()
    with pytest.raises(TrackRecordPaused):
        build_public_track_record(pd.DataFrame(), pd.DataFrame())
    assert main([]) == 2


def test_track_record_when_unpaused_uses_only_0_4_0_calls_and_corrected_outcomes(monkeypatch):
    import marketing.generate_public_track_record as tr
    monkeypatch.setattr(tr, "TRACK_RECORD_PAUSED", False)
    out = _completed_outcomes()
    calls = pd.DataFrame({
        "stock": ["HHH", "NNN", "UUU", "OLD"],
        "earnings_date": list(out["earnings_date"]) + [out["earnings_date"].iloc[0]],
        "tier": ["High Alert", "Normal", "Unscored", "High Alert"],
        "risk_score": [93.7, 41.0, np.nan, 80.0],
        "prediction_asof_date": ["2026-10-04"] * 4,
        "model_version": ["0.4.0", "0.4.0", "0.4.0", "0.3.1"],
    })
    events = pd.concat([out, out.iloc[[0]].assign(stock="OLD")], ignore_index=True)
    rec = tr.build_public_track_record(calls, events, generated_at=pd.Timestamp.today() + pd.Timedelta(days=14))
    got = {c["ticker"]: c for c in rec["calls"]}
    assert set(got) == {"HHH", "NNN"}              # OLD is 0.3.1; UUU unscored+unresolved
    assert got["HHH"]["is_extreme_move"] and not got["NNN"]["is_extreme_move"]
    assert got["NNN"]["move_pct"] == pytest.approx(3.0)


def test_drop_voided_refuses_unlabelled_calls():
    from marketing.generate_public_track_record import drop_voided
    with pytest.raises(ValueError):
        drop_voided(pd.DataFrame({"stock": ["A"]}))
