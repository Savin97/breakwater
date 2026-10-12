"""
Event-frame tests — one row per earnings event, completed or pending.

  * every completed earnings day becomes exactly one completed event;
  * every ACTIVE stock with a future earnings date gets exactly one pending event, which
    carries no outcome;
  * no consumer recovers "current state" with groupby(...).last();
  * an inactive stock keeps its history but can never resurface as an upcoming call.

Model B scoring on this frame is tested in testing/test_model_b.py.

Run with:  pytest testing/test_event_frame.py -v
"""
import ast
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

from pipeline.stage3 import stage3
from scoring.model_b import score_events
from pipeline.events import build_event_frame, pending_events, OUTCOME_COLS
from testing.test_pipeline import _build_stage2_df

FULL_DF_PATH = "output/full_df.parquet"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def daily_df():
    """Synthetic daily frame with a FUTURE earnings date on the tail rows, so every
    stock is eligible for a pending event. The trailing e_index is past the last price
    row, which is what gives those rows a positive days_to_earnings — exactly the shape
    real data has between two reports."""
    n_days = 250
    return stage3(_build_stage2_df(
        n_days=n_days, e_indices=(60, 120, 180, 220) + (n_days + 20,)))


@pytest.fixture(scope="module")
def events_df(daily_df):
    return score_events(build_event_frame(daily_df), daily_df)


@pytest.fixture(scope="module")
def real_daily_df():
    if not os.path.exists(FULL_DF_PATH):
        pytest.skip(f"{FULL_DF_PATH} not present — run the pipeline first")
    return pd.read_parquet(FULL_DF_PATH, columns=["stock", "date", "price", "earnings_date",
                                                  "is_earnings_day", "days_to_earnings"])


# ── Shape ─────────────────────────────────────────────────────────────────────

def test_one_completed_event_per_earnings_day(daily_df, events_df):
    completed = events_df[~events_df["is_pending"]]
    assert len(completed) == int((daily_df["is_earnings_day"] == 1).sum()) > 0
    assert not completed.duplicated(["stock", "earnings_date"]).any()


def test_exactly_one_pending_per_eligible_stock(daily_df, events_df):
    last_rows = (daily_df.sort_values(["stock", "date"], kind="mergesort")
                 .groupby("stock", as_index=False, sort=False).tail(1))
    eligible = set(last_rows.loc[
        last_rows["earnings_date"].notna() & (last_rows["days_to_earnings"] > 0), "stock"])
    pend = events_df[events_df["is_pending"]]
    assert set(pend["stock"]) == eligible
    assert not pend["stock"].duplicated().any()
    assert len(eligible) > 0, "fixture produces no eligible stock — test would be vacuous"


def test_pending_has_no_realized_outcome(events_df):
    """A pending event carries no outcome, so nothing downstream can read a neighbouring
    event's result as this one's."""
    pend = events_df[events_df["is_pending"]]
    for col in OUTCOME_COLS:
        if col in pend.columns:
            assert pend[col].isna().all(), col


def test_pending_earnings_date_is_in_the_future(events_df):
    pend = events_df[events_df["is_pending"]]
    assert (pend["earnings_date"] > pend["date"]).all()


def test_pending_never_enters_the_daily_frame(daily_df, events_df):
    """The daily frame is never modified: building the event frame adds no rows to it."""
    assert "is_pending" not in daily_df.columns


def test_pending_events_returns_only_pending_rows(events_df):
    out = pending_events(events_df)
    assert out["is_pending"].all()
    assert len(out) == int(events_df["is_pending"].sum())
    assert out["earnings_date"].is_monotonic_increasing


# ── No consumer may recover current state with .last() ────────────────────────

CONSUMER_MODULES = [
    "streamlit_dash/streamlit_export.py",
    "publishing/archive_calls.py",
    "report/report_builder.py",
    "report/calendar_builder.py",
    "pipeline/stage5.py",
]


@pytest.mark.parametrize("path", CONSUMER_MODULES)
def test_no_groupby_last_in_upcoming_consumers(path):
    """`groupby(...).last()` skips NaN per column and silently reaches back to the stock's
    last COMPLETED event — the bug the event frame exists to fix. AST-level so a comment
    mentioning the old idiom cannot trip it."""
    root = os.path.join(os.path.dirname(__file__), "..")
    tree = ast.parse(open(os.path.join(root, path)).read())
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "last"
                and isinstance(node.func.value, ast.Call)
                and isinstance(node.func.value.func, ast.Attribute)
                and node.func.value.func.attr == "groupby"):
            raise AssertionError(f"{path}:{node.lineno} uses groupby(...).last()")


# ── Weekly calendar reads the rows it is given ────────────────────────────────

def test_calendar_shows_only_rows_inside_the_window(events_df):
    from report.calendar_builder import build_calendar_data
    pend = pending_events(events_df)
    when = pend["earnings_date"].min()
    got, summary, _ = build_calendar_data(pend, reference_date=when, window_days=0)
    assert sorted(e["stock"] for e in got) == sorted(pend.loc[pend["earnings_date"] == when, "stock"])
    assert summary["n_total"] == len(got) > 0


# ── Active universe: only currently active stocks get an upcoming call ────────
#
# `stock_data.status` (maintained by ingestion/fetch_sp500_sectors.py) is the authority.
# A stock that left the index keeps every completed event, but its final daily row —
# which still carries a future earnings_date from the DB — must not become a pending
# event: that is how a delisted name received a PDF and an archived call scored from
# prices two months old.

def _future_pending(events, when):
    """Copy of `events` with every pending row moved to `when`, so consumers that look
    forward from today see them. Moves rows; never adds any."""
    out = events.copy()
    out.loc[out["is_pending"], "earnings_date"] = pd.Timestamp(when)
    return out


@pytest.fixture(scope="module")
def events_active_aaa(daily_df):
    """BBB is inactive: only AAA may receive a pending row."""
    return score_events(build_event_frame(daily_df, active_stocks={"AAA"}), daily_df)


def test_active_stock_with_future_date_gets_one_pending_row(daily_df):
    ev = build_event_frame(daily_df, active_stocks={"AAA", "BBB"})
    pend = ev[ev["is_pending"]]
    assert sorted(pend["stock"]) == ["AAA", "BBB"]


def test_inactive_stock_with_future_date_gets_no_pending_row(daily_df, events_active_aaa):
    last = daily_df.sort_values("date").groupby("stock").tail(1).set_index("stock")
    assert last.at["BBB", "days_to_earnings"] > 0, "BBB has no future date — test is vacuous"
    pend = events_active_aaa[events_active_aaa["is_pending"]]
    assert pend["stock"].tolist() == ["AAA"]


def test_inactive_stock_keeps_its_completed_history(events_df, events_active_aaa):
    def completed(ev, s):
        return ev[(~ev["is_pending"]) & (ev["stock"] == s)]
    assert len(completed(events_active_aaa, "BBB")) > 0
    pd.testing.assert_frame_equal(
        completed(events_active_aaa, "BBB").reset_index(drop=True),
        completed(events_df, "BBB").reset_index(drop=True))


def test_downstream_consumers_cannot_resurrect_an_inactive_stock(events_active_aaa, tmp_path):
    from streamlit_dash.streamlit_export import export_upcoming_df
    from report.calendar_builder import build_calendar_data
    soon = pd.Timestamp.today().normalize() + pd.Timedelta(days=3)
    ev = _future_pending(events_active_aaa, soon)

    assert pending_events(ev)["stock"].tolist() == ["AAA"]

    out = tmp_path / "upcoming.parquet"
    export_upcoming_df(ev, str(out))
    assert pd.read_parquet(out)["stock"].tolist() == ["AAA"]

    got, _, _ = build_calendar_data(pending_events(ev),
                                    reference_date=soon - pd.Timedelta(days=1), window_days=7)
    assert [e["stock"] for e in got] == ["AAA"]


def test_prediction_snapshot_cannot_archive_an_inactive_call(events_active_aaa, tmp_path):
    import duckdb
    import publishing.archive_calls as sp
    from utilities.data_utilities import work_week_window
    db = tmp_path / "predictions.duckdb"
    window_start, _ = work_week_window(weeks=1)
    ev = _future_pending(events_active_aaa, window_start + pd.Timedelta(days=1))

    sp.archive_calls(ev, db_path=str(db))

    con = duckdb.connect(str(db), read_only=True)
    stocks = [r[0] for r in con.execute("SELECT stock FROM predictions").fetchall()]
    con.close()
    assert stocks == ["AAA"], stocks       # non-vacuous: the active call IS archived


def test_load_active_stocks_reads_stock_data_status():
    import duckdb
    from utilities.db_utilities import create_sectors_data_table_if_not_exists, load_active_stocks
    con = duckdb.connect(":memory:")
    create_sectors_data_table_if_not_exists(con)
    con.execute("""INSERT INTO stock_data (stock, status, reason) VALUES
        ('AAA', 'active', NULL),
        ('BK',  'inactive', 'renamed → BNY'),
        ('BNY', 'active', NULL),
        ('LW',  'inactive', 'removed from index — delisted/merged/unconfirmed')""")
    assert load_active_stocks(con) == {"AAA", "BNY"}


def test_production_supplies_the_active_universe():
    """`active_stocks=None` means "no filter" for synthetic tests; the production event
    frame must pass the DB's active set explicitly."""
    tree = ast.parse(open("pipeline/events.py").read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build_pipeline_event_frame")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "build_event_frame"]
    assert calls
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "active_stocks" in kw
        assert getattr(kw["active_stocks"].func, "id", None) == "load_pipeline_active_stocks"
    assert "build_pipeline_event_frame(" in open("pipeline/stage4.py").read()


def test_real_data_pending_events_are_exactly_the_active_stocks(real_daily_df):
    from config import DB_PATH
    import duckdb
    from utilities.db_utilities import load_active_stocks
    if not os.path.exists(DB_PATH):
        pytest.skip(f"{DB_PATH} not present")
    con = duckdb.connect(DB_PATH, read_only=True)
    active = load_active_stocks(con)
    con.close()
    unfiltered = build_event_frame(real_daily_df)
    filtered = build_event_frame(real_daily_df, active_stocks=active)
    before = set(unfiltered.loc[unfiltered["is_pending"], "stock"])
    after = set(filtered.loc[filtered["is_pending"], "stock"])
    assert after == before & active                 # no active stock lost
    assert not (after - active)                     # no inactive stock kept
    assert (~filtered["is_pending"]).sum() == (~unfiltered["is_pending"]).sum()
    assert len(after) > 400, "expected a pending row for most of the S&P 500"


def test_stage4_runs_exactly_the_model_b_steps(daily_df, monkeypatch):
    """Stage 4 lists the Model B steps itself; it must give the same frame as
    score_events, which the reference builder, the validation and the tests use."""
    import pipeline.stage4 as s4
    monkeypatch.setattr(s4, "build_pipeline_event_frame", lambda d: build_event_frame(d))
    pd.testing.assert_frame_equal(s4.stage4(daily_df),
                                  score_events(build_event_frame(daily_df), daily_df))
