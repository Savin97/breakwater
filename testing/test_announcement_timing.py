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
    UNRESOLVED_ANCHOR_BEFORE_HISTORY,
    ANCHORED_OUTCOME_COLS, ANCHORED_STATUS_COLS, ANCHORED_REACTION_WINDOWS,
    DEFAULT_ANCHORED_TARGET,
    TARGET_AVAILABLE, TARGET_ENDPOINT_BEYOND_GRID, TARGET_ENDPOINT_PRICE_GAP,
    TARGET_ENDPOINT_AFTER_LAST_PRICE,
    classify_announce_window, market_session_grid, resolve_event_anchors,
    anchor_resolved_events, resolved_events,
)


def _gap_free(events):
    """Events whose every anchored horizon actually resolved to a real endpoint.

    An AMC event with a missing session inside its window is NOT expected to match the
    legacy value: `.shift(-k)` counts the ticker's rows and steps over the hole, the
    corrected target counts MARKET SESSIONS and refuses. Bit-identity is a claim about
    the gap-free events, and `test_1_grid_*` below pins the gapped ones separately.
    """
    return events[events[ANCHORED_STATUS_COLS].eq(TARGET_AVAILABLE).all(axis=1)]
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
    amc = _gap_free(events[(events["announce_window"] == AMC)
                           & (events["anchor_status"] == RESOLVED)])
    assert len(amc) > 1000, f"only {len(amc)} gap-free resolved AMC events on real data"
    for k in (1, 3, 5):
        assert np.array_equal(amc[f"reaction_{k}d_anchored"].to_numpy(dtype="float64"),
                              amc[f"reaction_{k}d"].to_numpy(dtype="float64"),
                              equal_nan=True), f"AMC reaction_{k}d_anchored != legacy"


def test_2_amc_legacy_disagreement_is_only_ever_a_session_gap(real_events_df):
    """The converse of the control: every AMC event where the corrected target differs
    from the legacy one must be explained by a missing session, never by the anchoring
    code drifting on its own."""
    events, _ = real_events_df
    amc = events[(events["announce_window"] == AMC) & (events["anchor_status"] == RESOLVED)]
    for k in (1, 3, 5):
        x = amc[f"reaction_{k}d_anchored"].to_numpy(dtype="float64")
        y = amc[f"reaction_{k}d"].to_numpy(dtype="float64")
        differs = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        if not differs.any():
            continue
        statuses = amc.loc[differs, f"reaction_{k}d_anchored_status"]
        assert (statuses != TARGET_AVAILABLE).all(), (
            f"AMC reaction_{k}d_anchored differs from legacy on an event whose endpoint "
            f"was available: {amc.loc[differs & statuses.eq(TARGET_AVAILABLE).reindex(amc.index, fill_value=False), ['stock', 'earnings_date']]}")


# ── Invariant 3 — the BMO anchor precedes the announcement ────────────────────

def test_3_bmo_anchor_close_strictly_precedes_the_announcement(events_df):
    """Invariant 3 — a BMO announcement lands before the open of the report date, so the
    last clean pre-announcement close is the PREVIOUS session's."""
    bmo = events_df[(events_df["announce_window"] == BMO)
                    & (events_df["anchor_status"] == RESOLVED)]
    assert len(bmo) > 0, "no resolved BMO events — test is vacuous"
    assert (bmo["anchor_date"] < bmo["earnings_date"]).all()
    assert (bmo["anchor_date"] < bmo["announce_ts_ny"]).all()


def test_3_bmo_anchor_is_the_immediately_preceding_market_session(daily_df, events_df):
    """Not merely earlier — the MARKET session immediately before, with none skipped.
    Checked against the market grid rather than the ticker's own rows, because the
    ticker's rows are exactly what must not define the offset."""
    grid = market_session_grid(daily_df)
    bmo = events_df[(events_df["announce_window"] == BMO)
                    & (events_df["anchor_status"] == RESOLVED)]
    assert len(bmo) > 0
    for ed, ad in zip(bmo["earnings_date"], bmo["anchor_date"]):
        i = int(np.searchsorted(grid, np.datetime64(ed, "ns")))
        assert grid[i] == np.datetime64(ed, "ns")
        assert grid[i - 1] == np.datetime64(ad, "ns")


def test_3_bmo_anchor_precedes_on_real_history(real_events_df):
    events, _ = real_events_df
    bmo = events[(events["announce_window"] == BMO) & (events["anchor_status"] == RESOLVED)]
    assert len(bmo) > 1000, f"only {len(bmo)} resolved BMO events on real data"
    assert (bmo["anchor_date"] < bmo["earnings_date"]).all()


# ── Invariant 4 — both windows span the same number of sessions ───────────────

def test_4_anchored_3d_spans_three_post_announcement_sessions(daily_df, events_df):
    """Invariant 4 — the event clock counts POST-ANNOUNCEMENT trading sessions, so BMO
    and AMC 3d cover three each: AMC close(D)->close(D+3), BMO close(D-1)->close(D+2)."""
    grid = market_session_grid(daily_df)
    res = events_df[events_df["anchor_status"] == RESOLVED]
    assert len(res) > 0
    checked = {AMC: 0, BMO: 0}
    for stock, sub in res.groupby("stock"):
        s = daily_df[daily_df["stock"] == stock].sort_values("date")
        by_date = dict(zip(s["date"].to_numpy(dtype="datetime64[ns]"),
                           s["price"].to_numpy(dtype="float64")))
        for row in sub.itertuples():
            i = int(np.searchsorted(grid, np.datetime64(row.earnings_date, "ns")))
            a = i + ANCHOR_OFFSET[row.announce_window]
            assert grid[a] == np.datetime64(row.anchor_date, "ns")
            for k in (1, 3, 5):
                got = getattr(row, f"reaction_{k}d_anchored")
                status = getattr(row, f"reaction_{k}d_anchored_status")
                if a + k >= len(grid):
                    assert np.isnan(got) and status == TARGET_ENDPOINT_BEYOND_GRID
                    continue
                # exactly k MARKET sessions of price action, starting at the first
                # session the market can trade the news in
                assert int(np.searchsorted(grid, grid[a + k])
                           - np.searchsorted(grid, grid[a + 1]) + 1) == k
                if grid[a + k] not in by_date or grid[a] not in by_date:
                    assert np.isnan(got) and status != TARGET_AVAILABLE
                    continue
                assert status == TARGET_AVAILABLE
                assert got == by_date[grid[a + k]] / by_date[grid[a]] - 1
            checked[row.announce_window] += 1
    assert checked[AMC] > 0 and checked[BMO] > 0, f"one window unexercised: {checked}"


def test_4_bmo_and_amc_use_the_same_session_count_on_real_history(real_events_df):
    events, daily = real_events_df
    grid = market_session_grid(daily)
    res = resolved_events(events)
    sample = pd.concat([res[res["announce_window"] == w].head(200) for w in (BMO, AMC)])
    for row in sample.itertuples():
        a = int(np.searchsorted(grid, np.datetime64(row.anchor_date, "ns")))
        i = int(np.searchsorted(grid, np.datetime64(row.earnings_date, "ns")))
        assert grid[i] == np.datetime64(row.earnings_date, "ns")
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


def test_9_bmo_before_the_tickers_history_has_no_anchor(daily_df, timing_df):
    """A BMO event on a ticker's very first session has no prior close of its own, even
    though the market traded that session. Unresolved with THAT reason — a young ticker
    is not an ingestion bug — and never silently anchored to itself."""
    events = build_event_frame(daily_df, timing_df)
    victim = events.index[(events["announce_window"] == BMO) & ~events["is_pending"]][0]
    stock = events.at[victim, "stock"]
    ed = pd.Timestamp(events.at[victim, "earnings_date"])
    truncated = daily_df[~((daily_df["stock"] == stock) & (daily_df["date"] < ed))]
    assert (truncated["date"] < ed).any(), "the other stock must still trade before ed"
    out = resolve_event_anchors(events, truncated)
    assert out.at[victim, "anchor_status"] == UNRESOLVED_ANCHOR_BEFORE_HISTORY
    assert pd.isna(out.at[victim, "anchor_date"])


def test_9_bmo_on_the_first_market_session_has_no_prior_session(daily_df, timing_df):
    """When the MARKET grid itself has nothing before the report date there is no prior
    session at all, which is a different fact from the ticker being young."""
    events = build_event_frame(daily_df, timing_df)
    victim = events.index[(events["announce_window"] == BMO) & ~events["is_pending"]][0]
    ed = pd.Timestamp(events.at[victim, "earnings_date"])
    truncated = daily_df[daily_df["date"] >= ed]          # every ticker, not just one
    out = resolve_event_anchors(events, truncated)
    assert out.at[victim, "anchor_status"] == UNRESOLVED_NO_PRIOR_SESSION
    assert pd.isna(out.at[victim, "anchor_date"])


def test_9_every_unresolved_event_has_a_stated_reason(events_df):
    from feature_engineering.announcement_timing import ANCHOR_STATUSES
    assert set(events_df["anchor_status"]) <= set(ANCHOR_STATUSES)
    assert events_df["anchor_status"].notna().all()
    assert events_df["anchor_session_status"].notna().all()


# ── The market-session grid, not the ticker's observed rows ───────────────────
# External review of Phase 2, item 1. Anchors and endpoints are positions on the
# canonical market-session grid; the ticker must then have a price row on those EXACT
# dates. Each test below deletes ONE stock's row for a session the other stock still
# trades, so the market grid is unchanged and only the ticker's coverage moves.


def _grid_of(daily_df):
    return market_session_grid(daily_df)


def _victim(events, window):
    """The first completed, resolved event in `window` with room to move around it."""
    m = ((events["announce_window"] == window) & ~events["is_pending"]
         & (events["anchor_status"] == RESOLVED))
    return events.index[m][0]


def _drop_row(daily_df, stock, date):
    holed = daily_df[~((daily_df["stock"] == stock)
                       & (daily_df["date"] == pd.Timestamp(date)))]
    assert (holed["date"] == pd.Timestamp(date)).any(), \
        "another stock must still establish that the market traded that session"
    assert len(holed) == len(daily_df) - 1
    return holed


def test_grid_a_missing_anchor_row_never_falls_back_to_the_session_before(
        daily_df, timing_df):
    """BMO anchor = the market session before D. If the ticker has no row for THAT
    session, the answer is "unavailable", not D-2.

    Counting positions in the ticker's own rows silently returns D-2 here and produces a
    two-session window labelled as one. Nothing in the output would show it.
    """
    events = build_event_frame(daily_df, timing_df)
    victim = _victim(events, BMO)
    stock = events.at[victim, "stock"]
    anchor = pd.Timestamp(events.at[victim, "anchor_date"])
    grid = _grid_of(daily_df)
    d_minus_2 = pd.Timestamp(grid[int(np.searchsorted(grid, np.datetime64(anchor))) - 1])

    holed = _drop_row(daily_df, stock, anchor)
    out = resolve_event_anchors(events, holed)
    assert out.at[victim, "anchor_status"] == UNRESOLVED_PRICE_GAP
    assert pd.isna(out.at[victim, "anchor_date"])
    assert out.at[victim, "anchor_date"] != d_minus_2
    for col in ANCHORED_OUTCOME_COLS:
        assert pd.isna(out.at[victim, col]), f"{col} produced without a real anchor"


def test_grid_a_missing_endpoint_row_never_stretches_the_window(daily_df, timing_df):
    """AMC k-endpoint = market session D+k. If the ticker has no row for that exact
    session, the outcome is unavailable — never the NEXT row the ticker does have, which
    is what `.shift(-k)` on the ticker's own rows returns.

    The legacy columns do exactly that and are left alone; the corrected target must not.
    """
    events = build_event_frame(daily_df, timing_df)
    base = events
    victim = _victim(events, AMC)
    stock = events.at[victim, "stock"]
    anchor = pd.Timestamp(base.at[victim, "anchor_date"])
    grid = _grid_of(daily_df)
    a = int(np.searchsorted(grid, np.datetime64(anchor)))
    endpoint_3d = pd.Timestamp(grid[a + 3])
    next_row_price = float(daily_df.loc[(daily_df["stock"] == stock)
                                        & (daily_df["date"] == pd.Timestamp(grid[a + 4])),
                                        "price"].iloc[0])
    anchor_price = float(daily_df.loc[(daily_df["stock"] == stock)
                                      & (daily_df["date"] == anchor), "price"].iloc[0])
    stretched = next_row_price / anchor_price - 1

    holed = _drop_row(daily_df, stock, endpoint_3d)
    out = resolve_event_anchors(events, holed)
    # the anchor is untouched — only the far end of the window moved
    assert out.at[victim, "anchor_status"] == RESOLVED
    assert pd.Timestamp(out.at[victim, "anchor_date"]) == anchor
    assert out.at[victim, "reaction_3d_anchored_status"] == TARGET_ENDPOINT_PRICE_GAP
    assert pd.isna(out.at[victim, "reaction_3d_anchored"])
    assert pd.isna(out.at[victim, "abs_reaction_3d_anchored"])
    assert not np.isclose(np.nan_to_num(out.at[victim, "reaction_3d_anchored"], nan=-999),
                          stretched), "the window was stretched to the next observed row"
    # the horizons that do not cross the hole are unaffected
    assert out.at[victim, "reaction_1d_anchored_status"] == TARGET_AVAILABLE
    assert out.at[victim, "reaction_1d_anchored"] == base.at[victim, "reaction_1d_anchored"]


def test_grid_a_bmo_report_date_gap_costs_only_the_horizons_that_need_it(
        daily_df, timing_df):
    """A BMO event anchors on D-1, so a missing row on D itself does not break the
    anchor — it breaks exactly the 1-session horizon, whose endpoint IS D. The 3- and
    5-session horizons end later and survive. Per-horizon availability, not a blanket
    verdict."""
    events = build_event_frame(daily_df, timing_df)
    base = events
    victim = _victim(events, BMO)
    stock = events.at[victim, "stock"]
    ed = pd.Timestamp(events.at[victim, "earnings_date"])

    holed = _drop_row(daily_df, stock, ed)
    out = resolve_event_anchors(events, holed)
    assert out.at[victim, "anchor_status"] == RESOLVED
    assert pd.Timestamp(out.at[victim, "anchor_date"]) == pd.Timestamp(base.at[victim, "anchor_date"])
    assert out.at[victim, "anchor_session_status"] == UNRESOLVED_PRICE_GAP
    assert out.at[victim, "reaction_1d_anchored_status"] == TARGET_ENDPOINT_PRICE_GAP
    assert pd.isna(out.at[victim, "reaction_1d_anchored"])
    for k in (3, 5):
        assert out.at[victim, f"reaction_{k}d_anchored_status"] == TARGET_AVAILABLE
        assert out.at[victim, f"reaction_{k}d_anchored"] == base.at[victim, f"reaction_{k}d_anchored"]


def test_grid_endpoint_past_the_end_of_the_grid_is_named_separately(daily_df, timing_df):
    """An unfinished window and a missing row are different facts and get different
    names, so "we do not know yet" is never counted as an ingestion bug."""
    events = build_event_frame(daily_df, timing_df)
    victim = _victim(events, AMC)
    grid = _grid_of(daily_df)
    a = int(np.searchsorted(grid, np.datetime64(pd.Timestamp(events.at[victim, "anchor_date"]))))
    # cut the whole market's history off between the 3- and 5-session endpoints
    truncated = daily_df[daily_df["date"] <= pd.Timestamp(grid[a + 3])]
    out = resolve_event_anchors(events, truncated)
    assert out.at[victim, "anchor_status"] == RESOLVED
    assert out.at[victim, "reaction_3d_anchored_status"] == TARGET_AVAILABLE
    assert out.at[victim, "reaction_5d_anchored_status"] == TARGET_ENDPOINT_BEYOND_GRID
    assert pd.isna(out.at[victim, "reaction_5d_anchored"])
    assert out.at[victim, "reaction_5d_anchored_status"] != TARGET_ENDPOINT_PRICE_GAP


def test_grid_is_the_market_grid_not_the_tickers_rows(daily_df, timing_df):
    """The grid must come from the whole loaded frame. A hole in ONE ticker cannot
    remove a session from it, which is what makes the checks above meaningful."""
    victim_stock = sorted(daily_df["stock"].unique())[0]
    holed = daily_df[daily_df["stock"] != victim_stock]
    assert len(_grid_of(daily_df)) == len(_grid_of(daily_df[daily_df["stock"] == victim_stock]))
    assert set(_grid_of(holed)) <= set(_grid_of(daily_df))


# ── The corrected-target gate: anchor resolved vs target available ────────────
# External review of Phase 2, item 3.


def test_gate_requires_the_requested_target_to_be_present(events_df):
    """`resolved_events` is the calibration/training gate, so it must require the
    outcome the caller is going to consume — not merely a real anchor."""
    gated = resolved_events(events_df)
    assert gated[DEFAULT_ANCHORED_TARGET].notna().all()
    assert (gated["anchor_status"] == RESOLVED).all()
    assert (~gated["is_pending"]).all()


def test_gate_is_per_target(events_df):
    for target in ANCHORED_OUTCOME_COLS:
        gated = resolved_events(events_df, target=target)
        assert gated[target].notna().all(), target
        assert len(gated) <= len(anchor_resolved_events(events_df))


def test_gate_target_none_is_the_anchor_only_slice(events_df):
    assert resolved_events(events_df, target=None).index.equals(
        anchor_resolved_events(events_df).index)


def test_gate_rejects_a_target_that_is_not_an_anchored_outcome(events_df):
    with pytest.raises(ValueError):
        resolved_events(events_df, target="abs_reaction_3d")     # the LEGACY column


def test_gate_excludes_an_event_whose_endpoint_row_is_missing(daily_df, timing_df):
    """The concrete failure the gate exists for: a resolved anchor whose 3-session
    endpoint the ticker has no row for must not reach a corrected calibration."""
    events = build_event_frame(daily_df, timing_df)
    victim = _victim(events, AMC)
    stock, anchor = events.at[victim, "stock"], pd.Timestamp(events.at[victim, "anchor_date"])
    grid = _grid_of(daily_df)
    a = int(np.searchsorted(grid, np.datetime64(anchor)))
    holed = _drop_row(daily_df, stock, pd.Timestamp(grid[a + 3]))

    scored = build_and_score_event_frame(holed, timing_df, verify=False)
    row = scored.index[(scored["stock"] == stock)
                       & (scored["earnings_date"] == events.at[victim, "earnings_date"])
                       & ~scored["is_pending"]][0]
    assert scored.at[row, "anchor_status"] == RESOLVED
    assert row in anchor_resolved_events(scored).index
    assert row not in resolved_events(scored).index


def test_gate_accounts_for_the_paired_row_difference_on_real_history(real_events_df):
    """The three counts that get confused with each other, pinned as an identity:

        anchor resolved  >=  anchored target available  >=  paired with the legacy one

    and each step is fully explained by a NAMED status, never by a silent dropna.
    """
    events, _ = real_events_df
    anchored = anchor_resolved_events(events)
    gated = resolved_events(events)
    paired = gated.dropna(subset=["abs_reaction_3d"])
    assert len(anchored) >= len(gated) >= len(paired)

    # step 1: every anchor that loses its target says why, in reaction_3d_anchored_status
    lost = anchored[anchored[DEFAULT_ANCHORED_TARGET].isna()]
    assert len(anchored) - len(gated) == len(lost)
    assert (lost["reaction_3d_anchored_status"] != TARGET_AVAILABLE).all()
    assert lost["reaction_3d_anchored_status"].isin({
        TARGET_ENDPOINT_BEYOND_GRID, TARGET_ENDPOINT_AFTER_LAST_PRICE,
        TARGET_ENDPOINT_PRICE_GAP}).all()

    # step 2: the remaining difference is the LEGACY column being absent, which is a
    # property of the comparison and not of the gate. A BMO 3d window closes one session
    # earlier than the legacy one, so this is expected at the right edge of history.
    legacy_missing = gated[gated["abs_reaction_3d"].isna()]
    assert len(gated) - len(paired) == len(legacy_missing)
    assert legacy_missing[DEFAULT_ANCHORED_TARGET].notna().all()


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
            "announce_ts_observed_at": pd.Timestamp("2024-05-02 03:00"),
        }, {
            "stock": "BBB", "earnings_date": pd.Timestamp("2024-05-01").date(),
            "fiscal_end_date": None, "reported_eps": 1.0, "estimated_eps": 0.9,
            "surprise_percentage": 0.1, "ingested_at": pd.Timestamp("2024-05-02"),
            "announce_ts_ny": pd.NaT, "announce_ts_source": None,
            "announce_ts_observed_at": pd.NaT,
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
    assert list(out.columns) == ["stock", "earnings_date", "announce_ts_ny",
                                 "announce_ts_source", "announce_ts_observed_at"]


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
            "announce_ts_observed_at": obs,
        } for s, ts, src, obs in [
            ("AAA", pd.NaT, None, pd.NaT),
            ("BBB", pd.Timestamp("2024-05-01 16:00"), "yfinance_earnings_dates",
             pd.Timestamp("2024-05-02 03:00")),
        ]])[EARNINGS_INSERT_COLS]
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


# ── A pre-event schedule must not be frozen forever ──────────────────────────
# External review of Phase 2, item 2. `announce_ts_ny` collected while an event is still
# upcoming is a SCHEDULE: the issuer can move it and the provider can correct it. The
# NULL-only backfill this phase shipped with wrote such a schedule once and never looked
# at it again, so a historical anchored target could rest permanently on a time that
# never happened. `announce_ts_observed_at` is the minimum provenance needed to tell the
# two apart, and the refresh rule is written in terms of it.

_E = pd.Timestamp("2024-05-01").date()          # the report date used throughout


def _seed_event(con, announce_ts_ny, observed_at, source, ingested_at=None,
                earnings_date=None):
    from ingestion.fetch_earnings_dates import EARNINGS_INSERT_COLS, _INSERT_COL_SQL
    df = pd.DataFrame([{
        "stock": "AAA",
        "earnings_date": _E if earnings_date is None else earnings_date,
        "fiscal_end_date": None,
        "reported_eps": None, "estimated_eps": 0.9, "surprise_percentage": None,
        "ingested_at": ingested_at if ingested_at is not None else observed_at,
        "announce_ts_ny": announce_ts_ny, "announce_ts_source": source,
        "announce_ts_observed_at": observed_at,
    }])[EARNINGS_INSERT_COLS]
    con.register("tmp_earnings_df", df)
    con.execute(f"INSERT INTO earnings ({_INSERT_COL_SQL}) "
                f"SELECT {_INSERT_COL_SQL} FROM tmp_earnings_df")
    con.unregister("tmp_earnings_df")


def _stored(con):
    return con.execute("SELECT announce_ts_ny, announce_ts_source, "
                       "announce_ts_observed_at FROM earnings "
                       "WHERE stock = 'AAA'").fetchone()


def test_observed_at_column_exists_on_the_earnings_table():
    con = _fresh_db()
    try:
        cols = {r[0]: r[1] for r in con.execute("DESCRIBE earnings").fetchall()}
    finally:
        con.close()
    assert cols.get("announce_ts_observed_at") == "TIMESTAMP"


def test_refresh_fills_a_row_that_has_no_timestamp():
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    con = _fresh_db()
    try:
        _seed_event(con, pd.NaT, pd.NaT, None, ingested_at=pd.Timestamp("2024-04-01"))
        n = refresh_announcement_timestamp(
            con, "AAA", _E, pd.Timestamp("2024-05-01 06:30"), "yfinance_earnings_dates",
            pd.Timestamp("2024-05-03 02:00"))
        got = _stored(con)
    finally:
        con.close()
    assert n == 1
    assert got[0] == pd.Timestamp("2024-05-01 06:30")


def test_refresh_replaces_a_pre_event_schedule_with_a_later_observation():
    """The scenario the review asked for, end to end:

      * a timestamp is stored while the event is still upcoming (a SCHEDULE),
      * the provider later returns a DIFFERENT time for the same event,
      * after the refresh the stored historical timing is the NEWER observation.

    Here the correction also flips the classified window, which is the whole reason a
    stale schedule matters: it would have anchored the corrected target to the wrong
    session forever.
    """
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    from utilities.db_utilities import load_announcement_timing
    con = _fresh_db()
    try:
        # observed 11 days BEFORE the announcement -> a schedule
        _seed_event(con, pd.Timestamp("2024-05-01 16:05"), pd.Timestamp("2024-04-20"),
                    "yfinance_earnings_dates")
        assert classify_announce_window(pd.Series([_stored(con)[0]])).iloc[0] == AMC

        # observed two days AFTER it -> an observation of what actually happened
        n = refresh_announcement_timestamp(
            con, "AAA", _E, pd.Timestamp("2024-05-01 06:30"), "yfinance_earnings_dates",
            pd.Timestamp("2024-05-03 02:00"))
        got = _stored(con)
        timing = load_announcement_timing(con)
    finally:
        con.close()
    assert n == 1
    assert got[0] == pd.Timestamp("2024-05-01 06:30")
    assert got[2] == pd.Timestamp("2024-05-03 02:00")          # provenance updated
    assert got[1] == "yfinance_earnings_dates"
    assert timing.loc[0, "announce_ts_ny"] == pd.Timestamp("2024-05-01 06:30")
    assert timing.loc[0, "announce_ts_observed_at"] == pd.Timestamp("2024-05-03 02:00")
    assert classify_announce_window(timing["announce_ts_ny"]).iloc[0] == BMO


def test_refresh_never_overwrites_a_post_event_observation():
    """Once a timestamp has been observed AFTER the announcement it is a record of what
    happened, and no later fetch may move it."""
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    con = _fresh_db()
    try:
        _seed_event(con, pd.Timestamp("2024-05-01 06:30"), pd.Timestamp("2024-05-03"),
                    "yfinance_earnings_dates")
        n = refresh_announcement_timestamp(
            con, "AAA", _E, pd.Timestamp("2024-05-01 16:05"), "yfinance_earnings_dates",
            pd.Timestamp("2024-06-01"))
        got = _stored(con)
    finally:
        con.close()
    assert n == 0
    assert got[0] == pd.Timestamp("2024-05-01 06:30")
    assert got[2] == pd.Timestamp("2024-05-03")


def test_refresh_ignores_an_older_observation():
    """A stale fetch arriving late must not clobber a newer one. This is also what makes
    re-running ingestion idempotent."""
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    con = _fresh_db()
    try:
        _seed_event(con, pd.Timestamp("2024-05-01 16:05"), pd.Timestamp("2024-04-25"),
                    "yfinance_earnings_dates")
        n = refresh_announcement_timestamp(
            con, "AAA", _E, pd.Timestamp("2024-05-01 06:30"), "yfinance_earnings_dates",
            pd.Timestamp("2024-04-20"))                      # older than what is stored
        got = _stored(con)
    finally:
        con.close()
    assert n == 0
    assert got[0] == pd.Timestamp("2024-05-01 16:05")


def test_refresh_is_idempotent():
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    con = _fresh_db()
    try:
        _seed_event(con, pd.Timestamp("2024-05-01 16:05"), pd.Timestamp("2024-04-20"),
                    "yfinance_earnings_dates")
        args = ("AAA", _E, pd.Timestamp("2024-05-01 06:30"), "yfinance_earnings_dates",
                pd.Timestamp("2024-05-03 02:00"))
        assert refresh_announcement_timestamp(con, *args) == 1
        assert refresh_announcement_timestamp(con, *args) == 0     # now post-event
        got = _stored(con)
    finally:
        con.close()
    assert got[0] == pd.Timestamp("2024-05-01 06:30")


@pytest.mark.parametrize("ingested_at,expect_refresh", [
    (pd.Timestamp("2024-04-20"), True),    # row created BEFORE the event -> a schedule
    (pd.Timestamp("2024-05-03"), False),   # row created AFTER it -> treat as observed
])
def test_refresh_falls_back_to_ingested_at_on_a_pre_column_row(ingested_at, expect_refresh):
    """Rows written before `announce_ts_observed_at` existed have NULL in it. Their
    `ingested_at` is a lower bound on when the timestamp could have been observed, so it
    can only make a row look MORE like a schedule — never less. Using it is the
    conservative direction; with both NULL nothing is refreshed at all."""
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    con = _fresh_db()
    try:
        _seed_event(con, pd.Timestamp("2024-05-01 16:05"), pd.NaT,
                    "yfinance_earnings_dates", ingested_at=ingested_at)
        n = refresh_announcement_timestamp(
            con, "AAA", _E, pd.Timestamp("2024-05-01 06:30"), "yfinance_earnings_dates",
            pd.Timestamp("2024-05-10"))
        got = _stored(con)
    finally:
        con.close()
    assert n == (1 if expect_refresh else 0)
    assert got[0] == (pd.Timestamp("2024-05-01 06:30") if expect_refresh
                      else pd.Timestamp("2024-05-01 16:05"))


def test_refresh_does_nothing_when_the_observation_time_is_unknown():
    """Neither column set means we cannot say whether the stored value is a schedule.
    Refusing to touch it is the only safe answer."""
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    con = _fresh_db()
    try:
        _seed_event(con, pd.Timestamp("2024-05-01 16:05"), pd.NaT,
                    "yfinance_earnings_dates", ingested_at=pd.NaT)
        n = refresh_announcement_timestamp(
            con, "AAA", _E, pd.Timestamp("2024-05-01 06:30"), "yfinance_earnings_dates",
            pd.Timestamp("2024-05-10"))
        got = _stored(con)
    finally:
        con.close()
    assert n == 0
    assert got[0] == pd.Timestamp("2024-05-01 16:05")


def test_yfinance_fetch_records_when_it_observed_the_timestamp(monkeypatch):
    """The refresh rule is only as good as the observation time it is given, so the
    fetcher must stamp it at the moment of the fetch rather than leave it to be inferred
    later — in NY wall clock, the convention `announce_ts_ny` is in. The host-timezone
    half of that claim is tested in the timezone-convention section below."""
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
    from utilities.time_utilities import now_ny
    before = pd.Timestamp(now_ny())
    out = fed.fetch_one_earnings_dates("AAA")["earnings_dates_df"]
    after = pd.Timestamp(now_ny())

    assert "announce_ts_observed_at" in out.columns
    assert out["announce_ts_observed_at"].notna().all()
    assert (out["announce_ts_observed_at"] >= before).all()
    assert (out["announce_ts_observed_at"] <= after).all()
    # `ingested_at` covers the same moment but stays in the legacy machine-local
    # convention — it is an operational column, not part of the timing comparison, so it
    # equals the observation only on a host that is already on New York time.
    assert out["ingested_at"].notna().all()


def test_backfill_stamps_the_seed_pull_date_as_the_observation_time():
    """The audit seed is evidence pulled on a known day. Recording that day is what lets
    the rule above tell a seeded post-event observation (frozen) from a seeded schedule
    for an event that had not happened yet (refreshable)."""
    from scripts.backfill_announcement_timestamps import (
        backfill, SOURCE_LABEL, SOURCE_OBSERVED_AT)
    con = _fresh_db()
    try:
        _seed_event(con, pd.NaT, pd.NaT, None, ingested_at=pd.Timestamp("2024-04-01"))
        backfill(con, pd.DataFrame([{
            "stock": "AAA", "earnings_date": _E,
            "announce_ts_ny": pd.Timestamp("2024-05-01 06:30")}]))
        got = _stored(con)
    finally:
        con.close()
    assert got[1] == SOURCE_LABEL
    assert got[2] == SOURCE_OBSERVED_AT
    # the seeded event reported long before the pull, so it is a post-event observation
    assert got[2] > got[0]


def test_backfill_migrates_observed_at_onto_rows_it_seeded_earlier():
    """The column was added after the seed had already run once. Stamping the pull date
    on those rows is a one-time, idempotent migration — without it 12,068 rows would be
    permanently un-classifiable as schedule or observation."""
    from scripts.backfill_announcement_timestamps import (
        backfill, SOURCE_LABEL, SOURCE_OBSERVED_AT)
    con = _fresh_db()
    try:
        _seed_event(con, pd.Timestamp("2024-05-01 06:30"), pd.NaT, SOURCE_LABEL,
                    ingested_at=pd.Timestamp("2024-04-01"))
        first = backfill(con, pd.DataFrame(columns=["stock", "earnings_date",
                                                    "announce_ts_ny"]))
        got = _stored(con)
        second = backfill(con, pd.DataFrame(columns=["stock", "earnings_date",
                                                     "announce_ts_ny"]))
    finally:
        con.close()
    assert first["seeded_rows_missing_observed_at"] == 1
    assert got[2] == SOURCE_OBSERVED_AT
    assert second["seeded_rows_missing_observed_at"] == 0


# ── The observation timestamp has ONE timezone convention ────────────────────
# External review of Phase 2, item 3. `announce_ts_ny` is naive NEW YORK wall clock and
# `announce_ts_observed_at` is compared against it directly, so it must be NY wall clock
# too. It used to be `datetime.now()`, which is the wall clock of whatever host ran the
# pipeline. On a host east of New York — UTC, or Israel at UTC+2/+3 — an observation made
# HOURS BEFORE an announcement produces a NUMBER LARGER than the NY announcement time, so
# the refresh rule reads a schedule as a post-event observation and freezes it into the
# historical record permanently. The classification must depend on the New York event
# clock and on nothing else.

import contextlib
import time as _stdtime
from datetime import datetime as _dt, timedelta as _td, timezone as _tz
from zoneinfo import ZoneInfo as _ZI

from utilities.time_utilities import (
    NY_TZ, MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS, now_ny, to_ny_wall_clock,
    utc_to_ny_wall_clock,
)

# Hosts this pipeline plausibly runs on, plus the two extremes of the offset range.
# Israel is the one that actually motivated the fix: it leads New York by 6 or 7 hours
# depending on whose DST is in effect, and the two zones switch on different dates.
_HOST_TZS = ["America/New_York", "UTC", "Asia/Jerusalem", "Europe/London",
             "Pacific/Kiritimati"]

_HAS_TZSET = hasattr(_stdtime, "tzset")
_needs_tzset = pytest.mark.skipif(not _HAS_TZSET, reason="TZ simulation needs time.tzset")


@contextlib.contextmanager
def _host_tz(name):
    """Run the block as if the machine's local timezone were `name`.

    This is the real thing, not a mock: it moves what bare `datetime.now()` returns,
    which is exactly the dependency being tested.
    """
    previous = os.environ.get("TZ")
    os.environ["TZ"] = name
    _stdtime.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        _stdtime.tzset()


def _wall(instant_utc, tz_name):
    """The naive wall clock a bare `datetime.now()` on a host in `tz_name` would return
    at the absolute instant `instant_utc` (given naive, in UTC)."""
    return pd.Timestamp(instant_utc.replace(tzinfo=_tz.utc)
                        .astimezone(_ZI(tz_name)).replace(tzinfo=None))


@_needs_tzset
@pytest.mark.parametrize("host", _HOST_TZS)
def test_now_ny_reads_the_new_york_clock_on_every_host(host):
    """`now_ny()` is the same value on every host at the same moment; bare
    `datetime.now()` is not. The second assertion is the bug, stated positively."""
    with _host_tz(host):
        got = now_ny()
        host_local = _dt.now()
    truth = _dt.now(_tz.utc).astimezone(NY_TZ).replace(tzinfo=None)
    assert abs(got - truth) < _td(seconds=5)

    expected_host = _dt.now(_tz.utc).astimezone(_ZI(host)).replace(tzinfo=None)
    assert abs(host_local - expected_host) < _td(seconds=5)
    ny_offset = _dt.now(_tz.utc).astimezone(NY_TZ).utcoffset()
    host_offset = _dt.now(_tz.utc).astimezone(_ZI(host)).utcoffset()
    if ny_offset != host_offset:
        assert abs(host_local - got) > _td(minutes=30)


@_needs_tzset
@pytest.mark.parametrize("host", _HOST_TZS)
@pytest.mark.parametrize("instant_utc,expected_ny", [
    (_dt(2024, 1, 15, 21, 5), pd.Timestamp("2024-01-15 16:05")),   # EST, UTC-5
    (_dt(2024, 7, 25, 20, 5), pd.Timestamp("2024-07-25 16:05")),   # EDT, UTC-4
    (_dt(2024, 3, 10,  6, 0), pd.Timestamp("2024-03-10 01:00")),   # minutes before spring-forward
    (_dt(2024, 3, 10,  7, 30), pd.Timestamp("2024-03-10 03:30")),  # minutes after it
    (_dt(2024, 11, 3,  5, 30), pd.Timestamp("2024-11-03 01:30")),  # first pass of the repeated hour
    (_dt(2024, 11, 3,  6, 30), pd.Timestamp("2024-11-03 01:30")),  # second pass of it
])
def test_the_ny_convention_is_dst_correct_and_host_independent(host, instant_utc, expected_ny):
    """The zone, not an offset constant, decides the wall clock — including across both
    US transitions, and regardless of when the HOST's own DST switches (Israel's dates
    differ from the US's, which is what makes the offset 6 hours in March and 7 in July)."""
    with _host_tz(host):
        assert utc_to_ny_wall_clock(instant_utc) == expected_ny
        assert to_ny_wall_clock(instant_utc.replace(tzinfo=_tz.utc)) == expected_ny
        # already NY wall clock -> nothing to convert, returned verbatim
        assert to_ny_wall_clock(expected_ny.to_pydatetime()) == expected_ny


# (label, announce_ts_ny, observation instant in UTC, may the stored value be refreshed)
#
# Every observation here is timed relative to the NEW YORK announcement clock. Read in
# Israel's or UTC's wall clock instead, the second row flips from schedule to
# observation — that is the failure this section exists to rule out.
_OBSERVATION_CASES = [
    ("pre-event, days ahead, EST",
     pd.Timestamp("2024-01-24 16:05"), _dt(2024, 1, 20, 12, 0), True),
    ("pre-event, 90 minutes ahead, EDT",
     pd.Timestamp("2024-07-25 06:30"), _dt(2024, 7, 25, 9, 0), True),
    ("post-event, same session, EDT",
     pd.Timestamp("2024-07-25 06:30"), _dt(2024, 7, 25, 14, 0), False),
    ("pre-event across the spring-forward boundary",
     pd.Timestamp("2024-03-12 06:30"), _dt(2024, 3, 8, 20, 0), True),
    ("post-event across the fall-back boundary",
     pd.Timestamp("2024-11-01 16:05"), _dt(2024, 11, 5, 9, 0), False),
    ("pre-event minutes before an AMC call",
     pd.Timestamp("2024-07-25 16:05"), _dt(2024, 7, 25, 19, 50), True),
]


@_needs_tzset
@pytest.mark.parametrize("host", _HOST_TZS)
@pytest.mark.parametrize("label,announce,instant_utc,expect_refresh",
                         _OBSERVATION_CASES,
                         ids=[c[0] for c in _OBSERVATION_CASES])
def test_schedule_vs_observation_is_identical_on_every_host(
        host, label, announce, instant_utc, expect_refresh):
    """The whole rule, end to end, under five host timezones.

    The stored row is stamped the way production stamps it — the observation instant
    rendered in New York — and the refresh outcome must come out the same everywhere.
    """
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    event_date = announce.date()
    stamp = _wall(instant_utc, "America/New_York")
    corrected = announce + _td(hours=1)
    later = _wall(instant_utc + _td(days=30), "America/New_York")

    with _host_tz(host):
        con = _fresh_db()
        try:
            con.execute("DELETE FROM earnings")
            _seed_event(con, announce, stamp, "yfinance_earnings_dates",
                        earnings_date=event_date)
            n = refresh_announcement_timestamp(
                con, "AAA", event_date, corrected, "yfinance_earnings_dates", later)
            stored = _stored(con)
        finally:
            con.close()

    assert n == (1 if expect_refresh else 0), label
    assert stored[0] == (corrected if expect_refresh else announce), label


@_needs_tzset
@pytest.mark.parametrize("host", ["UTC", "Asia/Jerusalem"])
def test_a_host_local_stamp_would_have_frozen_a_pre_event_schedule(host):
    """The regression, pinned. Stamping the SAME instant in the host's own wall clock —
    what `datetime.now()` did — turns a schedule observed 90 minutes before a BMO call
    into a permanent post-event record on any host east of New York.

    This test asserts the broken behaviour deliberately, so that a future change which
    reintroduces `datetime.now()` here fails the test above rather than passing both.
    """
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    announce = pd.Timestamp("2024-07-25 06:30")            # BMO, EDT
    instant_utc = _dt(2024, 7, 25, 9, 0)                   # 05:00 NY — still upcoming
    host_stamp = _wall(instant_utc, host)
    ny_stamp = _wall(instant_utc, "America/New_York")
    assert host_stamp > announce                           # reads as post-event...
    assert ny_stamp < announce                             # ...but it is not

    def _try(stamp):
        con = _fresh_db()
        try:
            _seed_event(con, announce, stamp, "yfinance_earnings_dates",
                        earnings_date=announce.date())
            return refresh_announcement_timestamp(
                con, "AAA", announce.date(), pd.Timestamp("2024-07-25 07:15"),
                "yfinance_earnings_dates", pd.Timestamp("2024-07-26 09:00"))
        finally:
            con.close()

    with _host_tz(host):
        assert _try(host_stamp) == 0        # frozen — the bug
        assert _try(ny_stamp) == 1          # refreshed — the fix


@_needs_tzset
@pytest.mark.parametrize("host", _HOST_TZS)
def test_the_fetcher_stamps_the_observation_in_new_york_time(monkeypatch, host):
    """Where the value is actually produced. `announce_ts_observed_at` must come out of
    `now_ny()` on every host; `ingested_at` deliberately keeps its legacy machine-local
    convention and is NOT part of the announcement-timing comparison."""
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

    with _host_tz(host):
        before = now_ny()
        out = fed.fetch_one_earnings_dates("AAA")["earnings_dates_df"]
        after = now_ny()
        host_local = pd.Timestamp(_dt.now())

    observed = out["announce_ts_observed_at"]
    assert observed.notna().all()
    assert (observed >= before).all() and (observed <= after).all()

    ny_offset = _dt.now(_tz.utc).astimezone(NY_TZ).utcoffset()
    host_offset = _dt.now(_tz.utc).astimezone(_ZI(host)).utcoffset()
    if ny_offset != host_offset:
        # the host clock is somewhere else entirely, and the column did not follow it
        assert (observed - host_local).abs().min() > _td(minutes=30)


@_needs_tzset
@pytest.mark.parametrize("host", ["UTC", "Asia/Jerusalem"])
def test_a_legacy_ingested_at_is_never_compared_raw_against_the_ny_clock(host):
    """The fallback path. Rows written before `announce_ts_observed_at` existed carry a
    machine-local `ingested_at` whose timezone convention was never recorded, so it
    cannot be compared against the NY announcement clock directly.

    Here the row was ingested 90 minutes BEFORE a BMO announcement by a host east of New
    York, so its raw value sorts after the announcement. Comparing raw would freeze a
    schedule on the strength of a timezone artefact; widening it into a lower bound that
    holds for any host keeps it refreshable, which is the conservative direction.
    """
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    announce = pd.Timestamp("2024-07-25 06:30")
    ingested_at = _wall(_dt(2024, 7, 25, 9, 0), host)      # 05:00 NY, still upcoming
    assert ingested_at > announce                          # raw comparison says post-event

    with _host_tz(host):
        con = _fresh_db()
        try:
            _seed_event(con, announce, pd.NaT, "yfinance_earnings_dates",
                        ingested_at=ingested_at, earnings_date=announce.date())
            n = refresh_announcement_timestamp(
                con, "AAA", announce.date(), pd.Timestamp("2024-07-25 07:15"),
                "yfinance_earnings_dates", pd.Timestamp("2024-07-26 09:00"))
            stored = _stored(con)
        finally:
            con.close()
    assert n == 1
    assert stored[0] == pd.Timestamp("2024-07-25 07:15")
    # and the refresh replaces the ambiguity with a real NY-convention observation, so
    # this row never takes the widened path again
    assert stored[2] == pd.Timestamp("2024-07-26 09:00")


@pytest.mark.parametrize("ingested_at", [
    pd.Timestamp("2024-07-27 09:00"),      # two days later — post-event under any host
    pd.Timestamp("2024-08-30 00:00"),      # a month later
])
def test_an_unambiguous_legacy_ingested_at_still_freezes_the_row(ingested_at):
    """The widening must not swallow the rule. A legacy `ingested_at` more than
    MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS past the announcement was an observation on every
    possible host, and stays frozen."""
    from ingestion.fetch_earnings_dates import refresh_announcement_timestamp
    announce = pd.Timestamp("2024-07-25 06:30")
    assert ingested_at - _td(hours=MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS) > announce
    con = _fresh_db()
    try:
        _seed_event(con, announce, pd.NaT, "yfinance_earnings_dates",
                    ingested_at=ingested_at, earnings_date=announce.date())
        n = refresh_announcement_timestamp(
            con, "AAA", announce.date(), pd.Timestamp("2024-07-25 07:15"),
            "yfinance_earnings_dates", pd.Timestamp("2024-09-01 09:00"))
        stored = _stored(con)
    finally:
        con.close()
    assert n == 0
    assert stored[0] == announce


def test_the_widening_is_exactly_the_worst_case_host_offset():
    """19 hours is UTC+14 against New York on EST. Not a round number picked for comfort:
    it is the largest a host wall clock can lead New York's."""
    extreme = _dt(2024, 1, 15, 12, 0)                       # EST is in effect
    ny = utc_to_ny_wall_clock(extreme)
    ahead = max(
        (_wall(extreme, zone) - ny) for zone in
        ["Pacific/Kiritimati", "Pacific/Apia", "Asia/Jerusalem", "UTC", "Asia/Tokyo"]
    )
    assert ahead <= _td(hours=MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS)
    assert _wall(extreme, "Pacific/Kiritimati") - ny == _td(hours=19)


def test_no_module_stamps_the_observation_from_the_host_clock():
    """Static guard, the same shape as `test_6_the_classifier_never_touches_price`.

    Nothing anywhere may assign `announce_ts_observed_at` from a bare `datetime.now()` /
    `pd.Timestamp.now()`. There is one correct source and it is `now_ny()`.
    """
    import pathlib
    offenders = []
    for directory in ("ingestion", "pipeline", "utilities", "scripts",
                      "feature_engineering", "analysis"):
        for path in sorted(pathlib.Path(directory).glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                targets = [t for t in node.targets if isinstance(t, ast.Subscript)]
                names = [t.slice.value for t in targets
                         if isinstance(t.slice, ast.Constant)
                         and isinstance(t.slice.value, str)]
                if "announce_ts_observed_at" not in names:
                    continue
                src = ast.unparse(node.value)
                if "datetime.now(" in src or "Timestamp.now(" in src:
                    offenders.append(f"{path}:{node.lineno}: {src}")
    assert not offenders, (
        "announce_ts_observed_at must be stamped with utilities.time_utilities.now_ny(); "
        "a host-local clock silently makes the schedule/observation classification depend "
        "on where the pipeline runs:\n" + "\n".join(offenders))
