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


def _rows(db, where="TRUE"):
    con = duckdb.connect(db, read_only=True)
    try:
        return con.execute(f"SELECT * FROM predictions WHERE {where} "
                           "ORDER BY stock, earnings_date, prediction_asof_date").df()
    finally:
        con.close()


def test_old_rows_are_preserved_exactly(db_copy):
    from publishing.archive_calls import archive_calls as save_predictions_snapshot
    old = _rows(db_copy)
    assert len(old) > 0 and set(old["model_version"]) <= {"0.3.1", "0.3.1-preaudit"}
    save_predictions_snapshot(_pending_week(), db_path=db_copy)
    after = _rows(db_copy, "model_version <> '0.4.0'")
    pd.testing.assert_frame_equal(old.reset_index(drop=True),
                                  after[old.columns].reset_index(drop=True))
    # the new columns exist and are NULL on every pre-0.4 row
    assert after["model_b_hist_mean_abs"].isna().all() and after["model_b_n_prior"].isna().all()


def test_new_rows_store_the_model_b_fields(db_copy):
    from publishing.archive_calls import archive_calls as save_predictions_snapshot
    save_predictions_snapshot(_pending_week(), db_path=db_copy)
    new = _rows(db_copy, "model_version = '0.4.0'").set_index("stock")
    assert sorted(new.index) == ["HHH", "NNN", "UUU"]
    assert new.at["HHH", "tier"] == "High Alert" and new.at["HHH", "risk_score"] == 93.7
    assert new.at["HHH", "model_b_hist_mean_abs"] == 0.081 and new.at["HHH", "model_b_n_prior"] == 30
    assert new.at["UUU", "tier"] == "Unscored" and pd.isna(new.at["UUU", "risk_score"])
    assert new.at["UUU", "model_b_n_prior"] == 5
    assert new["score_asof_date"].notna().all()
    for legacy in ("is_high_conviction", "pre_earnings_drift_flag", "surprise_momentum_flag"):
        assert new[legacy].isna().all(), legacy


def test_same_day_rerun_upserts(db_copy):
    from publishing.archive_calls import archive_calls as save_predictions_snapshot
    ev = _pending_week()
    save_predictions_snapshot(ev, db_path=db_copy)
    n = len(_rows(db_copy))
    ev.loc[0, "risk_score"], ev.loc[0, "model_b_hist_mean_abs"] = 95.5, 0.09
    save_predictions_snapshot(ev, db_path=db_copy)
    rows = _rows(db_copy)
    assert len(rows) == n
    hhh = rows[(rows["stock"] == "HHH") & (rows["model_version"] == "0.4.0")]
    assert len(hhh) == 1 and hhh["risk_score"].iloc[0] == 95.5
    assert hhh["model_b_hist_mean_abs"].iloc[0] == 0.09


def test_track_record_view_holds_only_0_4_0_calls(db_copy):
    from publishing.archive_calls import archive_calls as save_predictions_snapshot
    save_predictions_snapshot(_pending_week(), db_path=db_copy)
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
        risk_score, model_version, score_asof_date) VALUES
        ('2026-09-28', 'AAA', '2026-10-08', 'Elevated', 76, '0.3.1', '2026-09-26'),
        ('2026-10-04', 'AAA', '2026-10-08', 'High Alert', 92, '0.4.0', '2026-10-02'),
        ('2026-10-05', 'AAA', '2026-10-08', 'High Alert', 93, '0.4.0', '2026-10-02'),
        ('2026-10-09', 'BBB', '2026-10-08', 'Normal', 10, '0.4.0', '2026-10-09')""")
    create_predictions_table_if_not_exists(con)      # re-creates the views
    tr = con.execute("SELECT * FROM predictions_track_record").df()
    con.close()
    assert tr["stock"].tolist() == ["AAA"]               # BBB's call was made after the event
    assert tr["risk_score"].iloc[0] == 92 and tr["model_version"].iloc[0] == "0.4.0"


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
    from publishing.archive_calls import archive_calls as save_predictions_snapshot
    from analysis.last_week_results import archived_calls, results_table, format_results
    save_predictions_snapshot(_pending_week(), db_path=db_copy)
    start, end = work_week_window(weeks=1)
    tbl = results_table(archived_calls(start, end, db_copy), _completed_outcomes()).set_index("stock")
    assert bool(tbl.at["HHH", "moved_8pct"]) is True             # corrected 12%
    assert bool(tbl.at["NNN", "moved_8pct"]) is False            # legacy 30% ignored
    assert pd.isna(tbl.at["UUU", "moved_8pct"])                   # unresolved, not a miss
    text = format_results(tbl.reset_index(), start, end)
    assert "INTERNAL" in text and "1 of 1 resolved calls moved" in text


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
