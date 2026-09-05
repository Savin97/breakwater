"""
Phase 2 tests — verified announcement timing and the parallel anchored targets.

Two things must hold at once:

  * the LEGACY target is untouched — `reaction_{1,3,5}d` and `abs_reaction_3d` are the
    control this phase is measured against, and every Phase 1 parity guarantee survives;
  * the corrected target is anchored to the last close before the announcement, using an
    OBSERVED timestamp and never a price-behavior inference, with everything that cannot
    be anchored explicitly unresolved and counted.

The numbered tests map one-to-one onto the ten required invariants.

Run with:  pytest testing/test_announcement_timing.py -v
"""
import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from pipeline.stage3 import stage3
from pipeline.stage4 import stage4
from pipeline.events import (
    build_event_frame, build_and_score_event_frame, attach_announcement_timing,
    completed_parity_report, OUTCOME_COLS,
)
from feature_engineering.announcement_timing import (
    AMC, BMO, INTRADAY, UNKNOWN, ANCHOR_OFFSET,
    RESOLVED, PENDING, UNRESOLVED_NO_TIMESTAMP, UNRESOLVED_INTRADAY,
    UNRESOLVED_NO_SESSION, UNRESOLVED_PRICE_GAP, UNRESOLVED_NO_PRIOR_SESSION,
    ANCHORED_OUTCOME_COLS,
    classify_announce_window, resolve_event_anchors, resolved_events,
)
from testing.test_pipeline import _build_stage2_df

FULL_DF_PATH = "output/full_df.parquet"
LEGACY_REACTION_COLS = ["reaction_1d", "reaction_3d", "reaction_5d", "abs_reaction_3d"]

# The four cases the fixture must exercise. Times are NY wall clock.
_FIXTURE_TIMES = {0: "06:30", 1: "16:05", 2: "11:00", 3: None}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def daily_df():
    n_days = 250
    return stage4(stage3(_build_stage2_df(
        n_days=n_days, e_indices=(60, 120, 180, 220) + (n_days + 20,))))


@pytest.fixture(scope="module")
def timing_df(daily_df):
    """Synthetic observed timestamps: one BMO, one AMC, one INTRADAY and one event left
    without a timestamp, per stock. Every branch of the resolver gets exercised."""
    ev = daily_df[daily_df["is_earnings_day"] == 1][["stock", "earnings_date"]]
    rows = []
    for stock, sub in ev.groupby("stock"):
        for i, ed in enumerate(sorted(sub["earnings_date"].unique())):
            t = _FIXTURE_TIMES.get(i)
            if t is None:
                continue
            rows.append({
                "stock": stock,
                "earnings_date": pd.Timestamp(ed),
                "announce_ts_ny": pd.Timestamp(f"{pd.Timestamp(ed):%Y-%m-%d} {t}"),
                "announce_ts_source": "test_fixture",
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def events_df(daily_df, timing_df):
    return build_and_score_event_frame(daily_df, timing_df)


@pytest.fixture(scope="module")
def real_events_df():
    if not os.path.exists(FULL_DF_PATH):
        pytest.skip(f"{FULL_DF_PATH} not present — run the pipeline first")
    import duckdb
    from config import DB_PATH
    from utilities.db_utilities import load_announcement_timing
    daily = pd.read_parquet(FULL_DF_PATH)
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        timing = load_announcement_timing(con)
    finally:
        con.close()
    if timing.empty:
        pytest.skip("no announcement timestamps in the DB — run "
                    "scripts/backfill_announcement_timestamps.py")
    return build_and_score_event_frame(daily, timing), daily


# ── The window classifier ─────────────────────────────────────────────────────

@pytest.mark.parametrize("clock,expected", [
    ("2024-05-01 04:00", BMO),
    ("2024-05-01 06:00", BMO),
    ("2024-05-01 09:29", BMO),
    ("2024-05-01 09:30", INTRADAY),   # at the open — not before it
    ("2024-05-01 12:00", INTRADAY),
    ("2024-05-01 15:59", INTRADAY),
    ("2024-05-01 16:00", AMC),        # at the close — after it for our purposes
    ("2024-05-01 20:00", AMC),
    (None, UNKNOWN),
])
def test_window_is_a_pure_function_of_the_clock(clock, expected):
    got = classify_announce_window(pd.Series([pd.Timestamp(clock) if clock else pd.NaT]))
    assert got.iloc[0] == expected


def test_window_is_unknown_without_a_timestamp(daily_df):
    """No timestamp must mean UNKNOWN, never a default window."""
    events = attach_announcement_timing(build_event_frame(daily_df), None)
    assert (events["announce_window"] == UNKNOWN).all()
    assert events["announce_ts_ny"].isna().all()


def test_tz_aware_timestamps_are_read_in_new_york(daily_df):
    """A tz-aware input must be converted to NY wall clock, not read as UTC. 21:00 UTC
    is 17:00 NY in summer — AMC, not INTRADAY."""
    ts = pd.Series([pd.Timestamp("2024-07-01 21:00", tz="UTC")])
    assert classify_announce_window(ts).iloc[0] == AMC


# ── Invariant 1 — the legacy target is untouched ──────────────────────────────

def test_1_legacy_reaction_columns_unchanged_by_timing(daily_df, timing_df):
    """Invariant 1 — adding announcement timing must not move a single legacy value."""
    without = build_event_frame(daily_df, None)
    with_timing = build_event_frame(daily_df, timing_df)
    assert len(without) == len(with_timing)
    for col in LEGACY_REACTION_COLS:
        a = without[col].to_numpy(dtype="float64")
        b = with_timing[col].to_numpy(dtype="float64")
        assert np.array_equal(a, b, equal_nan=True), f"{col} moved when timing was added"


def test_1_legacy_columns_match_the_daily_frame(daily_df, events_df):
    """The event frame's legacy reactions are still the daily frame's own values."""
    left = events_df[~events_df["is_pending"]].set_index(["stock", "earnings_date"]).sort_index()
    right = (daily_df[daily_df["is_earnings_day"] == 1]
             .set_index(["stock", "earnings_date"]).sort_index())
    for col in LEGACY_REACTION_COLS:
        assert np.array_equal(left[col].to_numpy(dtype="float64"),
                              right[col].to_numpy(dtype="float64"), equal_nan=True), col


def test_1_anchored_columns_are_parallel_not_replacements(events_df):
    for col in LEGACY_REACTION_COLS + ANCHORED_OUTCOME_COLS:
        assert col in events_df.columns, f"{col} missing — the two targets must coexist"


# ── Invariant 2 — AMC anchored == legacy, bit for bit ─────────────────────────

def test_2_amc_anchored_is_bit_identical_to_legacy(events_df):
    """Invariant 2 — AMC was never mismeasured, so the correction must be a no-op there.
    This is the control proving the anchoring code moves nothing on its own."""
    amc = events_df[(events_df["announce_window"] == AMC)
                    & (events_df["anchor_status"] == RESOLVED)]
    assert len(amc) > 0, "no resolved AMC events — test is vacuous"
    for k in (1, 3, 5):
        assert np.array_equal(amc[f"reaction_{k}d_anchored"].to_numpy(dtype="float64"),
                              amc[f"reaction_{k}d"].to_numpy(dtype="float64"),
                              equal_nan=True), f"AMC reaction_{k}d_anchored != legacy"
    assert np.array_equal(amc["abs_reaction_3d_anchored"].to_numpy(dtype="float64"),
                          amc["abs_reaction_3d"].to_numpy(dtype="float64"), equal_nan=True)


def test_2_amc_anchor_is_the_report_date_close(events_df):
    amc = events_df[(events_df["announce_window"] == AMC)
                    & (events_df["anchor_status"] == RESOLVED)]
    assert (amc["anchor_date"] == amc["earnings_date"]).all()


def test_2_amc_bit_identical_on_real_history(real_events_df):
    events, _ = real_events_df
    amc = events[(events["announce_window"] == AMC) & (events["anchor_status"] == RESOLVED)]
    assert len(amc) > 1000, f"only {len(amc)} resolved AMC events on real data"
    for k in (1, 3, 5):
        assert np.array_equal(amc[f"reaction_{k}d_anchored"].to_numpy(dtype="float64"),
                              amc[f"reaction_{k}d"].to_numpy(dtype="float64"),
                              equal_nan=True), f"AMC reaction_{k}d_anchored != legacy"


# ── Invariant 3 — the BMO anchor precedes the announcement ────────────────────

def test_3_bmo_anchor_close_strictly_precedes_the_announcement(events_df):
    """Invariant 3 — a BMO announcement lands before the open of the report date, so the
    last clean pre-announcement close is the PREVIOUS session's."""
    bmo = events_df[(events_df["announce_window"] == BMO)
                    & (events_df["anchor_status"] == RESOLVED)]
    assert len(bmo) > 0, "no resolved BMO events — test is vacuous"
    assert (bmo["anchor_date"] < bmo["earnings_date"]).all()
    assert (bmo["anchor_date"] < bmo["announce_ts_ny"]).all()


def test_3_bmo_anchor_is_the_immediately_preceding_session(daily_df, events_df):
    """Not merely earlier — the session immediately before, with no session skipped."""
    bmo = events_df[(events_df["announce_window"] == BMO)
                    & (events_df["anchor_status"] == RESOLVED)]
    for stock, sub in bmo.groupby("stock"):
        sessions = np.sort(daily_df.loc[daily_df["stock"] == stock, "date"]
                           .to_numpy(dtype="datetime64[ns]"))
        for ed, ad in zip(sub["earnings_date"], sub["anchor_date"]):
            i = int(np.searchsorted(sessions, np.datetime64(ed, "ns")))
            assert sessions[i - 1] == np.datetime64(ad, "ns")


def test_3_bmo_anchor_precedes_on_real_history(real_events_df):
    events, _ = real_events_df
    bmo = events[(events["announce_window"] == BMO) & (events["anchor_status"] == RESOLVED)]
    assert len(bmo) > 1000, f"only {len(bmo)} resolved BMO events on real data"
    assert (bmo["anchor_date"] < bmo["earnings_date"]).all()


# ── Invariant 4 — both windows span the same number of sessions ───────────────

def test_4_anchored_3d_spans_three_post_announcement_sessions(daily_df, events_df):
    """Invariant 4 — the event clock counts POST-ANNOUNCEMENT trading sessions, so BMO
    and AMC 3d cover three each: AMC close(D)->close(D+3), BMO close(D-1)->close(D+2)."""
    res = events_df[events_df["anchor_status"] == RESOLVED]
    assert len(res) > 0
    checked = {AMC: 0, BMO: 0}
    for stock, sub in res.groupby("stock"):
        s = daily_df[daily_df["stock"] == stock].sort_values("date")
        sessions = s["date"].to_numpy(dtype="datetime64[ns]")
        px = s["price"].to_numpy(dtype="float64")
        for row in sub.itertuples():
            i = int(np.searchsorted(sessions, np.datetime64(row.earnings_date, "ns")))
            a = i + ANCHOR_OFFSET[row.announce_window]
            assert sessions[a] == np.datetime64(row.anchor_date, "ns")
            for k in (1, 3, 5):
                got = getattr(row, f"reaction_{k}d_anchored")
                if a + k >= len(px):
                    assert np.isnan(got)
                    continue
                # exactly k sessions of price action starting at the first session the
                # market can trade the news in
                assert got == px[a + k] / px[a] - 1
                first_tradeable = sessions[a + 1]
                last_session = sessions[a + k]
                n_sessions = int(np.searchsorted(sessions, last_session)
                                 - np.searchsorted(sessions, first_tradeable) + 1)
                assert n_sessions == k
            checked[row.announce_window] += 1
    assert checked[AMC] > 0 and checked[BMO] > 0, f"one window unexercised: {checked}"


def test_4_bmo_and_amc_use_the_same_session_count_on_real_history(real_events_df):
    events, daily = real_events_df
    res = resolved_events(events)
    sample = pd.concat([res[res["announce_window"] == w].head(200) for w in (BMO, AMC)])
    sessions_by_stock = {
        s: sub.sort_values("date")["date"].to_numpy(dtype="datetime64[ns]")
        for s, sub in daily[daily["stock"].isin(sample["stock"])].groupby("stock")
    }
    for row in sample.itertuples():
        sess = sessions_by_stock[row.stock]
        a = int(np.searchsorted(sess, np.datetime64(row.anchor_date, "ns")))
        i = int(np.searchsorted(sess, np.datetime64(row.earnings_date, "ns")))
        assert a == i + ANCHOR_OFFSET[row.announce_window]


# ── Invariant 5 — INTRADAY / UNKNOWN get no primary target ────────────────────

def test_5_intraday_and_unknown_have_no_anchored_target(events_df):
    """Invariant 5 — an announcement inside the session has no clean pre-announcement
    close, and no timestamp means no window at all. Neither is guessed at."""
    bad = events_df[events_df["announce_window"].isin([INTRADAY, UNKNOWN])
                    & ~events_df["is_pending"]]
    assert len(bad) > 0, "no INTRADAY/UNKNOWN completed events — test is vacuous"
    for col in ANCHORED_OUTCOME_COLS:
        assert bad[col].isna().all(), f"{col} populated on an unanchorable event"
    assert bad["anchor_date"].isna().all()
    assert set(bad["anchor_status"]) <= {UNRESOLVED_INTRADAY, UNRESOLVED_NO_TIMESTAMP,
                                         UNRESOLVED_NO_SESSION, UNRESOLVED_PRICE_GAP}


def test_5_unanchorable_on_real_history(real_events_df):
    events, _ = real_events_df
    bad = events[events["announce_window"].isin([INTRADAY, UNKNOWN]) & ~events["is_pending"]]
    assert len(bad) > 0
    for col in ANCHORED_OUTCOME_COLS:
        assert bad[col].isna().all(), f"{col} populated on an unanchorable event"


def test_5_pending_events_are_never_resolved(events_df):
    pend = events_df[events_df["is_pending"]]
    assert (pend["anchor_status"] == PENDING).all()
    for col in OUTCOME_COLS:
        if col in pend.columns:
            assert pend[col].isna().all(), f"{col} is populated on a pending event"


# ── Invariant 6 — the window is never inferred from price ─────────────────────

PRICE_TERMS = {
    "price", "prices", "ret", "returns", "daily_ret", "reaction", "reaction_1d",
    "reaction_3d", "reaction_5d", "abs_reaction_3d", "vol", "drift", "momentum",
    "is_extreme_reaction", "is_large_reaction", "abs",
}


def test_6_the_classifier_never_touches_price():
    """Invariant 6 — `classify_announce_window` is a pure function of the observed clock.

    Enforced statically, not by inspection. Revision 1 of the audit labelled a ticker BMO
    because its day-0 move exceeded its day-+1 move; every corrected number that followed
    was circular. The classifier must never be able to regress into that, so no
    price-derived name may appear anywhere in its body.
    """
    src = open("feature_engineering/announcement_timing.py").read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "classify_announce_window")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    names |= {c.value for c in ast.walk(fn)
              if isinstance(c, ast.Constant) and isinstance(c.value, str)}
    leaked = names & PRICE_TERMS
    assert not leaked, f"classify_announce_window references price-derived names: {leaked}"


def test_6_window_is_independent_of_the_price_series(daily_df, timing_df):
    """The same timestamps over a DIFFERENT price series must give the same windows.
    A behavior-derived label could not survive this."""
    shuffled = daily_df.copy()
    rng = np.random.default_rng(7)
    shuffled["price"] = shuffled.groupby("stock")["price"].transform(
        lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
    a = build_event_frame(daily_df, timing_df)
    b = build_event_frame(shuffled, timing_df)
    assert (a["announce_window"].to_numpy() == b["announce_window"].to_numpy()).all()
    assert a["anchor_status"].equals(b["anchor_status"])


def test_6_resolver_reads_no_reaction_column(daily_df, timing_df):
    """The resolver anchors from price ROW POSITIONS; it must not consult any realized
    outcome. Blanking every reaction column must leave the anchoring identical."""
    events = build_event_frame(daily_df, timing_df)
    blanked = events.copy()
    for col in LEGACY_REACTION_COLS + ANCHORED_OUTCOME_COLS:
        if col in blanked.columns:
            blanked[col] = np.nan
    left = resolve_event_anchors(events, daily_df)
    right = resolve_event_anchors(blanked, daily_df)
    assert left["anchor_status"].equals(right["anchor_status"])
    assert np.array_equal(left["reaction_3d_anchored"].to_numpy(dtype="float64"),
                          right["reaction_3d_anchored"].to_numpy(dtype="float64"),
                          equal_nan=True)


# ── Invariant 7 — unresolved events cannot enter the corrected dataset ────────

def test_7_resolved_events_is_the_only_gate(events_df):
    res = resolved_events(events_df)
    assert (res["anchor_status"] == RESOLVED).all()
    assert (~res["is_pending"]).all()
    assert res["announce_window"].isin([BMO, AMC]).all()
    assert res["anchor_date"].notna().all()


def test_7_no_unresolved_event_carries_an_anchored_target(events_df):
    """Invariant 7 — the corrected target exists ONLY where the anchor is real, so an
    unresolved event cannot slip into corrected calibration by carrying a value."""
    unres = events_df[events_df["anchor_status"] != RESOLVED]
    assert len(unres) > 0
    for col in ANCHORED_OUTCOME_COLS:
        assert unres[col].isna().all(), f"{col} populated on {unres['anchor_status'].iloc[0]}"


def test_7_on_real_history(real_events_df):
    events, _ = real_events_df
    unres = events[events["anchor_status"] != RESOLVED]
    for col in ANCHORED_OUTCOME_COLS:
        assert unres[col].isna().all(), f"{col} populated on an unresolved event"
    res = resolved_events(events)
    assert len(res) > 5000, f"only {len(res)} resolved events"
    assert res["announce_window"].isin([BMO, AMC]).all()


# ── Invariant 8 — provenance is recorded ──────────────────────────────────────

def test_8_every_resolved_event_records_its_provenance(events_df):
    """Invariant 8 — a resolved event must say WHERE its timestamp came from, so a
    backfilled one can always be told from a freshly ingested one."""
    res = resolved_events(events_df)
    assert res["announce_ts_ny"].notna().all()
    assert res["announce_ts_source"].notna().all()
    assert (res["announce_ts_source"].astype(str).str.strip() != "").all()


def test_8_provenance_on_real_history(real_events_df):
    events, _ = real_events_df
    res = resolved_events(events)
    assert res["announce_ts_ny"].notna().all()
    assert res["announce_ts_source"].notna().all()


def test_8_announce_ts_agrees_with_the_window(events_df):
    """The recorded timestamp and the recorded window must not be able to disagree."""
    res = resolved_events(events_df)
    hour = res["announce_ts_ny"].dt.hour + res["announce_ts_ny"].dt.minute / 60
    assert (hour[res["announce_window"] == BMO] < 9.5).all()
    assert (hour[res["announce_window"] == AMC] >= 16.0).all()


# ── Invariant 9 — non-session and price-gap cases are counted, not rolled ─────

def test_9_a_non_session_date_is_unresolved_not_rolled(daily_df, timing_df):
    """Invariant 9 — a report date the ticker has no price row for must come back
    unresolved with a reason, never quietly moved to the next session."""
    events = build_event_frame(daily_df, timing_df)
    victim = events.index[(events["announce_window"] == BMO) & ~events["is_pending"]][0]
    moved = events.copy()
    # a Saturday: inside the price history, not a session for anyone
    bad_date = pd.Timestamp(moved.at[victim, "earnings_date"]) + pd.Timedelta(days=0)
    while bad_date.weekday() != 5 or bad_date in set(daily_df["date"]):
        bad_date += pd.Timedelta(days=1)
    moved.at[victim, "earnings_date"] = bad_date
    out = resolve_event_anchors(moved, daily_df)
    assert out.at[victim, "anchor_status"] == UNRESOLVED_NO_SESSION
    assert pd.isna(out.at[victim, "anchor_date"])
    assert out.at[victim, "reaction_3d_anchored"] != out.at[victim, "reaction_3d_anchored"]


def test_9_a_missing_price_row_is_a_gap_not_a_calendar_problem(daily_df, timing_df):
    """A date the market traded but this ticker has no row for is an INGESTION gap, and
    must be labelled separately from a weekend/holiday date — the two have different
    causes and different fixes (audit Q6)."""
    events = build_event_frame(daily_df, timing_df)
    victim = events.index[(events["announce_window"] == AMC) & ~events["is_pending"]][0]
    stock = events.at[victim, "stock"]
    ed = pd.Timestamp(events.at[victim, "earnings_date"])
    holed = daily_df[~((daily_df["stock"] == stock) & (daily_df["date"] == ed))]
    assert (holed["date"] == ed).any(), "the other stock must still trade that day"
    out = resolve_event_anchors(events, holed)
    assert out.at[victim, "anchor_status"] == UNRESOLVED_PRICE_GAP
    assert out.at[victim, "anchor_session_status"] == UNRESOLVED_PRICE_GAP
    assert pd.isna(out.at[victim, "anchor_date"])


def test_9_bmo_on_the_first_price_row_has_no_anchor(daily_df, timing_df):
    """A BMO event on a ticker's very first session has no prior close. Unresolved,
    not silently anchored to itself."""
    events = build_event_frame(daily_df, timing_df)
    victim = events.index[(events["announce_window"] == BMO) & ~events["is_pending"]][0]
    stock = events.at[victim, "stock"]
    ed = pd.Timestamp(events.at[victim, "earnings_date"])
    truncated = daily_df[~((daily_df["stock"] == stock) & (daily_df["date"] < ed))]
    out = resolve_event_anchors(events, truncated)
    assert out.at[victim, "anchor_status"] == UNRESOLVED_NO_PRIOR_SESSION
    assert pd.isna(out.at[victim, "anchor_date"])


def test_9_every_unresolved_event_has_a_stated_reason(events_df):
    from feature_engineering.announcement_timing import ANCHOR_STATUSES
    assert set(events_df["anchor_status"]) <= set(ANCHOR_STATUSES)
    assert events_df["anchor_status"].notna().all()
    assert events_df["anchor_session_status"].notna().all()


# ── Invariant 10 — Phase 1 legacy parity still holds ──────────────────────────

def test_10_phase1_completed_parity_still_green(daily_df, timing_df):
    """Invariant 10 — every Phase 1 guarantee about the LEGACY columns survives Phase 2.
    build_and_score_event_frame asserts this on every run; this pins it in CI."""
    events = build_and_score_event_frame(daily_df, timing_df)
    assert completed_parity_report(events, daily_df) == {}


def test_10_parity_holds_with_and_without_timing(daily_df, timing_df):
    for timing in (None, timing_df):
        events = build_and_score_event_frame(daily_df, timing)
        assert completed_parity_report(events, daily_df) == {}


def test_10_parity_on_real_history(real_events_df):
    events, daily = real_events_df
    assert completed_parity_report(events, daily) == {}


def test_10_scores_are_identical_with_and_without_timing(daily_df, timing_df):
    """Phase 2 adds a parallel target and changes no score. Announcement timing must not
    reach the production score at all — that is Phase 3's job, after a re-fit."""
    from pipeline.events import PARITY_COLS
    a = build_and_score_event_frame(daily_df, None)
    b = build_and_score_event_frame(daily_df, timing_df)
    for col in PARITY_COLS:
        x, y = a[col], b[col]
        if x.dtype.kind in "fi":
            assert np.array_equal(x.to_numpy(dtype="float64"),
                                  y.to_numpy(dtype="float64"), equal_nan=True), col
        else:
            assert x.astype(object).where(x.notna(), "<NA>").equals(
                y.astype(object).where(y.notna(), "<NA>")), col


# ── The audit artifact must not be a runtime dependency ───────────────────────

@pytest.mark.parametrize("path", [
    "pipeline/events.py",
    "pipeline/stage5.py",
    "pipeline/pipeline.py",
    "feature_engineering/announcement_timing.py",
    "utilities/db_utilities.py",
])
def test_no_pipeline_module_reads_the_audit_parquet(path):
    """audit/provider_timestamps.parquet seeded the DB column ONCE, via
    scripts/backfill_announcement_timestamps.py. If a production module ever reads it
    directly, the pipeline has silently acquired a dependency on an audit artifact."""
    tree = ast.parse(open(path).read())
    literals = [c.value for c in ast.walk(tree)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)]
    docstrings = {ast.get_docstring(n, clean=False) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef))}
    offenders = [t for t in literals
                 if "provider_timestamps" in t and t not in docstrings]
    assert not offenders, (
        f"{path} references the audit timestamp parquet in code: {offenders}")


# ── Schema and ingestion: the timestamp must survive the write path ───────────

def _fresh_db():
    import duckdb
    from utilities.db_utilities import create_earnings_table_if_not_exists
    con = duckdb.connect(":memory:")
    create_earnings_table_if_not_exists(con)
    return con


def test_earnings_schema_carries_the_timing_columns():
    con = _fresh_db()
    try:
        cols = {r[0]: r[1] for r in con.execute("DESCRIBE earnings").fetchall()}
    finally:
        con.close()
    assert cols.get("announce_ts_ny") == "TIMESTAMP"
    assert cols.get("announce_ts_source") == "VARCHAR"


def test_insert_column_list_matches_the_physical_table_order():
    """Both writers name their columns instead of `SELECT *`. If the list ever drifts
    from the table, a future column would shift values into the wrong slot silently."""
    from ingestion.fetch_earnings_dates import EARNINGS_INSERT_COLS
    con = _fresh_db()
    try:
        physical = [r[0] for r in con.execute("DESCRIBE earnings").fetchall()]
    finally:
        con.close()
    assert EARNINGS_INSERT_COLS == physical


def test_a_timestamped_row_round_trips_through_the_write_path():
    """End to end on the actual SQL: insert a row carrying an observed timestamp, read it
    back through load_announcement_timing, and classify it."""
    from ingestion.fetch_earnings_dates import EARNINGS_INSERT_COLS, _INSERT_COL_SQL
    from utilities.db_utilities import load_announcement_timing
    con = _fresh_db()
    try:
        df = pd.DataFrame([{
            "stock": "AAA", "earnings_date": pd.Timestamp("2024-05-01").date(),
            "fiscal_end_date": None, "reported_eps": 1.0, "estimated_eps": 0.9,
            "surprise_percentage": 0.1, "ingested_at": pd.Timestamp("2024-05-02"),
            "announce_ts_ny": pd.Timestamp("2024-05-01 06:30"),
            "announce_ts_source": "yfinance_earnings_dates",
        }, {
            "stock": "BBB", "earnings_date": pd.Timestamp("2024-05-01").date(),
            "fiscal_end_date": None, "reported_eps": 1.0, "estimated_eps": 0.9,
            "surprise_percentage": 0.1, "ingested_at": pd.Timestamp("2024-05-02"),
            "announce_ts_ny": pd.NaT, "announce_ts_source": None,
        }])[EARNINGS_INSERT_COLS]
        con.register("tmp_earnings_df", df)
        con.execute(f"INSERT INTO earnings ({_INSERT_COL_SQL}) "
                    f"SELECT {_INSERT_COL_SQL} FROM tmp_earnings_df")
        con.unregister("tmp_earnings_df")
        timing = load_announcement_timing(con)
    finally:
        con.close()

    # the date-only row must not appear at all — a NULL timestamp is not a window
    assert list(timing["stock"]) == ["AAA"]
    assert timing.loc[0, "announce_ts_ny"] == pd.Timestamp("2024-05-01 06:30")
    assert classify_announce_window(timing["announce_ts_ny"]).iloc[0] == BMO


def test_load_announcement_timing_degrades_on_a_pre_phase2_database():
    """An old DB without the column must give "no observed timing", not a crash."""
    import duckdb
    from utilities.db_utilities import load_announcement_timing
    con = duckdb.connect(":memory:")
    try:
        con.execute("CREATE TABLE earnings (stock TEXT, earnings_date DATE)")
        out = load_announcement_timing(con)
    finally:
        con.close()
    assert out.empty
    assert list(out.columns) == ["stock", "earnings_date",
                                 "announce_ts_ny", "announce_ts_source"]


def test_backfill_is_idempotent_and_only_fills_nulls():
    """The audit parquet is a one-time seed. Re-running must change nothing, and a
    timestamp already observed must never be overwritten."""
    from ingestion.fetch_earnings_dates import EARNINGS_INSERT_COLS, _INSERT_COL_SQL
    from scripts.backfill_announcement_timestamps import backfill
    con = _fresh_db()
    try:
        rows = pd.DataFrame([{
            "stock": s, "earnings_date": pd.Timestamp("2024-05-01").date(),
            "fiscal_end_date": None, "reported_eps": 1.0, "estimated_eps": 0.9,
            "surprise_percentage": 0.1, "ingested_at": pd.Timestamp("2024-05-02"),
            "announce_ts_ny": ts, "announce_ts_source": src,
        } for s, ts, src in [("AAA", pd.NaT, None),
                             ("BBB", pd.Timestamp("2024-05-01 16:00"), "yfinance_earnings_dates")]
        ])[EARNINGS_INSERT_COLS]
        con.register("tmp_earnings_df", rows)
        con.execute(f"INSERT INTO earnings ({_INSERT_COL_SQL}) "
                    f"SELECT {_INSERT_COL_SQL} FROM tmp_earnings_df")
        con.unregister("tmp_earnings_df")

        seed = pd.DataFrame([
            {"stock": "AAA", "earnings_date": pd.Timestamp("2024-05-01").date(),
             "announce_ts_ny": pd.Timestamp("2024-05-01 07:00")},
            {"stock": "BBB", "earnings_date": pd.Timestamp("2024-05-01").date(),
             "announce_ts_ny": pd.Timestamp("2024-05-01 05:00")},   # must NOT overwrite
            {"stock": "CCC", "earnings_date": pd.Timestamp("2024-05-01").date(),
             "announce_ts_ny": pd.Timestamp("2024-05-01 06:00")},   # event we do not hold
        ])
        first = backfill(con, seed)
        assert first["filled"] == 1
        assert first["seed_events_not_in_db"] == 1
        second = backfill(con, seed)
        assert second["filled"] == 0 and second["already_had_timestamp"] == 2

        got = con.execute("SELECT stock, announce_ts_ny, announce_ts_source "
                          "FROM earnings ORDER BY stock").fetchall()
    finally:
        con.close()
    assert got[0][1] == pd.Timestamp("2024-05-01 07:00")
    assert got[0][2] == "audit_provider_timestamps_2026_09_05"
    # the pre-existing observed timestamp survives, with its own provenance
    assert got[1][1] == pd.Timestamp("2024-05-01 16:00")
    assert got[1][2] == "yfinance_earnings_dates"


def test_yfinance_ingestion_no_longer_discards_the_timestamp(monkeypatch):
    """The stated Phase 2 ingestion change: `fetch_one_earnings_dates` used to collapse
    yfinance's tz-aware NY index to a bare date with `.dt.date`, which is what forced
    every later timing analysis to infer BMO/AMC from price behavior. It must now keep
    the wall clock — while storing exactly the same `earnings_date` as before."""
    import ingestion.fetch_earnings_dates as fed

    idx = pd.DatetimeIndex(
        [pd.Timestamp("2024-05-01 06:30"), pd.Timestamp("2024-02-01 16:05")],
        name="Earnings Date").tz_localize("America/New_York")
    payload = pd.DataFrame(
        {"EPS Estimate": [0.9, 1.1], "Reported EPS": [1.0, 1.2], "Surprise(%)": [11.0, 9.0]},
        index=idx)

    class _Ticker:
        def __init__(self, symbol):
            self.earnings_dates = payload

    monkeypatch.setattr(fed.yf, "Ticker", _Ticker)
    monkeypatch.setattr(fed.time, "sleep", lambda *_: None)

    out = fed.fetch_one_earnings_dates("AAA")["earnings_dates_df"]
    assert list(out.columns) == fed.EARNINGS_INSERT_COLS
    assert list(out["announce_ts_ny"]) == [pd.Timestamp("2024-05-01 06:30"),
                                           pd.Timestamp("2024-02-01 16:05")]
    assert (out["announce_ts_source"] == fed.ANNOUNCE_TS_SOURCE_YFINANCE).all()
    # the calendar date is byte-for-byte what the old .dt.date produced
    assert list(out["earnings_date"]) == list(
        pd.to_datetime(idx).tz_localize(None).date)
    assert list(classify_announce_window(out["announce_ts_ny"])) == [BMO, AMC]
