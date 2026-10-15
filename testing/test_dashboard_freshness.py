"""The dashboard serves the current publication (audit H3).

Reproduces the auditor's scenario with the REAL cached getters: a calls file with score
91 is replaced by one with score 12. Before the fix the getter kept returning 91 until its
cache was cleared by hand; now a new generation (server) or a rewritten file (local) is a
new cache key. Also: day counts are recomputed at serve time and reported events leave
Upcoming Events.

Run with:  pytest testing/test_dashboard_freshness.py -v
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

import publishing.generation as G
from streamlit_dash import data as D


def _calls(score, earnings="2026-10-20"):
    return pd.DataFrame({"stock": ["AAA"], "earnings_date": [pd.Timestamp(earnings)],
                         "risk_score": [float(score)], "risk_tier": ["High Alert"],
                         "days_to_earnings": [99], "model_version": ["0.4.0"]})


def _served_scores(root):
    s = D.current_served(root)
    return s, D.get_upcoming_df(s.key, str(s.calls_path))["risk_score"].tolist()


def _activate(root, gen, score, schema=G.DASHBOARD_SCHEMA):
    d = root / "output" / "generations" / gen
    d.mkdir(parents=True)
    _calls(score).to_parquet(d / G.CALLS_FILE)
    _calls(score).to_parquet(d / G.HISTORY_FILE)
    (d / G.MANIFEST).write_text(json.dumps({"generation_id": gen, "dashboard_schema": schema}))
    (root / "output" / G.POINTER).write_text(gen + "\n")


def test_a_new_generation_replaces_the_cached_calls_without_a_manual_clear(tmp_path):
    _activate(tmp_path, "gen-1", 91)
    s1, scores = _served_scores(tmp_path)
    assert scores == [91.0] and s1.generation_id == "gen-1"
    _activate(tmp_path, "gen-2", 12)                                   # pointer moves
    s2, scores = _served_scores(tmp_path)
    assert scores == [12.0] and s2.generation_id == "gen-2"


def test_a_rewritten_local_file_replaces_the_cached_calls(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    _calls(91).to_parquet(out / G.CALLS_FILE)
    _, scores = _served_scores(tmp_path)
    assert scores == [91.0]
    _calls(12).to_parquet(out / G.CALLS_FILE)
    st = (out / G.CALLS_FILE).stat()
    os.utime(out / G.CALLS_FILE, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    _, scores = _served_scores(tmp_path)
    assert scores == [12.0]


def test_a_pointer_to_a_missing_or_newer_generation_is_refused(tmp_path):
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / G.POINTER).write_text("not-there\n")
    with pytest.raises(D.ServedGenerationError):
        D.current_served(tmp_path)
    _activate(tmp_path, "gen-future", 50, schema=G.DASHBOARD_SCHEMA + 1)
    with pytest.raises(D.ServedGenerationError):
        D.current_served(tmp_path)


def test_days_are_recomputed_and_reported_events_dropped_at_serve_time():
    df = pd.concat([_calls(91, "2026-10-12"), _calls(50, "2026-10-14"), _calls(12, "2026-10-20")])
    view = D.upcoming_view(df, pd.Timestamp("2026-10-14"))
    assert view["earnings_date"].dt.strftime("%m-%d").tolist() == ["10-14", "10-20"]
    assert view["days_to_earnings"].tolist() == [0, 6]                 # not the stored 99


def test_the_app_reads_only_through_the_generation_aware_getters():
    src = (Path(__file__).resolve().parents[1] / "streamlit_dash" / "app.py").read_text()
    assert "read_parquet" not in src                     # every parquet read goes through data.py
    assert "current_served()" in src and "upcoming_view(" in src
    assert "get_upcoming_df(served.key" in src and "get_history_df(served.key" in src
