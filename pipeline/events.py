# pipeline/events.py
"""
The event frame — one row per earnings event, completed or pending. Model B is scored
on it (pipeline/stage4.py → scoring/model_b.py) and every output reads it.

Why an event frame
------------------
Before the methodology rebuild, statistics that aggregate across a stock's earnings
events lived only on the earnings-day rows of the daily price frame, and consumers
recovered "current state" with `df.sort_values("date").groupby("stock").last()`. That
skipped NaN column by column and reached back to the stock's LAST COMPLETED EVENT, so
every shipped upcoming call was one earnings event stale (audit/PHASE0_AUDIT_REV2.md §Q4).

What it holds
-------------
* every completed earnings event — one per earnings-day row of the daily frame;
* one PENDING row per active stock with a future earnings date, taken from that stock's
  final daily row (price context such as IV as of the latest data) and carrying no
  outcome;
* the verified announcement timing and the corrected, timestamp-anchored outcomes
  (feature_engineering/announcement_timing.py), which are Model B's only input.

The daily frame is never modified and nothing here is merged back into it.
"""
import numpy as np
import pandas as pd

from config import EXTREME_EARNINGS_REACTION_THRESHOLD
from feature_engineering.announcement_timing import (
    ANCHORED_OUTCOME_COLS,
    TARGET_AVAILABLE,
    classify_announce_window,
    resolve_event_anchors,
    resolution_summary,
)

# Realized outcomes. A pending event has none — blanked so nothing downstream can read a
# neighbouring event's result as this one's. The legacy report-date reactions are the
# anchoring control only (feature_engineering/post_earnings_stock_features.py).
OUTCOME_COLS = [
    "reaction_1d", "reaction_3d", "reaction_5d", "abs_reaction_3d",
    *ANCHORED_OUTCOME_COLS,
    "is_extreme_move_anchored",
]


def load_pipeline_announcement_timing() -> pd.DataFrame:
    """The pipeline's one call site for verified announcement timing.

    Opens the production DuckDB read-only and returns whatever `earnings.announce_ts_ny`
    holds. This is the ONLY place in `pipeline/` that knows where announcement timing
    lives, and it is a database column — never a seed or vendor file. Ordinary ingestion
    keeps the column current.
    """
    import duckdb
    from config import DB_PATH
    from utilities.db_utilities import load_announcement_timing

    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        timing = load_announcement_timing(con)
    finally:
        con.close()
    print(f"  announcement timing: {len(timing)} observed timestamps from the earnings table")
    return timing


def load_pipeline_active_stocks() -> set[str]:
    """The pipeline's one call site for the current universe: `stock_data.status ==
    'active'`, read from the production DuckDB (read-only). Only active stocks receive a
    pending event.

    Raises rather than returning an empty set: an empty `stock_data` table means a broken
    database, and an empty universe would silently ship no calls at all.
    """
    import duckdb
    from config import DB_PATH
    from utilities.db_utilities import load_active_stocks

    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        active = load_active_stocks(con)
    finally:
        con.close()
    if not active:
        raise RuntimeError("stock_data has no 'active' tickers — refusing to build an "
                           "event frame with no eligible upcoming events")
    print(f"  active universe: {len(active)} stocks (stock_data.status = 'active')")
    return active


def attach_announcement_timing(events: pd.DataFrame,
                               timing_df: pd.DataFrame | None) -> pd.DataFrame:
    """Join the observed announcement timestamps and classify the window.

    `timing_df` is (stock, earnings_date, announce_ts_ny, announce_ts_source,
    announce_ts_observed_at) as returned by
    `utilities.db_utilities.load_announcement_timing`. `announce_ts_observed_at` is when
    the provider was observed to say that time; it is provenance only — no anchoring
    decision reads it. Passing None means "no observed timing available": every event
    comes out UNKNOWN and unresolved. That is the honest degradation and deliberately the
    DEFAULT, so a production path can never acquire announcement timing by accident and a
    unit test can never acquire a database by accident.
    """
    events = events.copy()
    if timing_df is None or len(timing_df) == 0:
        events["announce_ts_ny"] = pd.NaT
        events["announce_ts_source"] = None
        events["announce_ts_observed_at"] = pd.NaT
    else:
        cols = ["stock", "earnings_date", "announce_ts_ny", "announce_ts_source"]
        t = timing_df[cols + [c for c in ("announce_ts_observed_at",)
                              if c in timing_df.columns]].copy()
        t["earnings_date"] = pd.to_datetime(t["earnings_date"])
        t = t.drop_duplicates(subset=["stock", "earnings_date"], keep="first")
        events = events.merge(t, on=["stock", "earnings_date"], how="left")
        if "announce_ts_observed_at" not in events.columns:
            events["announce_ts_observed_at"] = pd.NaT
    events["announce_window"] = classify_announce_window(events["announce_ts_ny"]).values
    return events


def build_event_frame(daily_df: pd.DataFrame,
                      timing_df: pd.DataFrame | None = None,
                      active_stocks=None) -> pd.DataFrame:
    """One row per (stock, earnings event): every completed event, plus one pending row
    per eligible stock.

    A stock is eligible for a pending row when its final daily row carries a future
    earnings_date (days_to_earnings > 0) AND it is in `active_stocks`. A stock whose
    final row IS its earnings day gets no pending row — the next date is not yet known
    from the forward merge_asof.

    `active_stocks` is the current universe (`stock_data.status == 'active'`; the
    pipeline passes `load_pipeline_active_stocks()`). A stock that left the index keeps
    every COMPLETED event — that is real history — but gets no upcoming call. `None`
    applies no universe filter (unit tests on synthetic frames); production must pass it.

    `timing_df` carries the observed announcement timestamps. It decides
    `announce_window`, `anchor_date`, `anchor_status` and the corrected
    `reaction_*_anchored` outcomes that Model B reads. A pending event does not need a
    timestamp of its own to be scored.
    """
    completed = daily_df[daily_df["is_earnings_day"] == 1].copy()
    completed["is_pending"] = False

    # True final row per stock — .tail(1), never .last(), which would skip NaN per column.
    last_rows = (
        daily_df.sort_values(["stock", "date"], kind="mergesort")
        .groupby("stock", as_index=False, sort=False)
        .tail(1)
    )
    pending = last_rows[
        last_rows["earnings_date"].notna() & (last_rows["days_to_earnings"] > 0)
    ].copy()
    if active_stocks is not None:
        active = set(active_stocks)
        dropped = sorted(pending.loc[~pending["stock"].isin(active), "stock"])
        pending = pending[pending["stock"].isin(active)]
        if dropped:
            print(f"  {len(dropped)} stocks not in the active universe get no pending "
                  f"event (history kept): {', '.join(dropped)}")
    pending["is_pending"] = True
    pending["is_earnings_day"] = 0
    for col in OUTCOME_COLS:
        if col in pending.columns:
            pending[col] = np.nan

    events = pd.concat([completed, pending], ignore_index=True)
    events = events.sort_values(
        ["stock", "earnings_date", "is_pending"], kind="mergesort"
    ).reset_index(drop=True)

    # A pending row's price context (IV snapshot, sector) is only as fresh as its stock's
    # last price row. Surfaced rather than dropped: dropping it would silently shrink the
    # universe. Model B itself reads only completed corrected outcomes.
    if not pending.empty:
        latest = daily_df["date"].max()
        stale = pending.loc[pending["date"] < latest, "stock"].tolist()
        if stale:
            print(f"  NOTE: {len(stale)} pending events whose last price row is older than "
                  f"{latest.date()} (stale feed): {', '.join(sorted(stale))}")
    events["event_id"] = (
        events["stock"].astype(str) + "|" + events["earnings_date"].dt.strftime("%Y-%m-%d")
    )

    # Verified announcement timing and the corrected anchored outcomes. Runs on the
    # assembled frame, after the pending rows exist, so a pending event is anchored as
    # `pending` rather than picking up an anchor for an outcome that has not happened.
    events = attach_announcement_timing(events, timing_df)
    events = pd.concat([events, resolve_event_anchors(events, daily_df)], axis=1)

    # The validation target as a flag: 1/0 where the corrected 3-session outcome is
    # available, NaN everywhere else (pending, unresolved) — never 0 for "unknown".
    avail = (~events["is_pending"]) & events["reaction_3d_anchored_status"].eq(TARGET_AVAILABLE)
    events["is_extreme_move_anchored"] = np.where(
        avail, (events["abs_reaction_3d_anchored"] >= EXTREME_EARNINGS_REACTION_THRESHOLD)
        .astype(float), np.nan)
    return events


def build_pipeline_event_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    """The production event frame: the DB's verified timing and active universe."""
    print("--------------------\nEvent frame...")
    events = build_event_frame(daily_df, load_pipeline_announcement_timing(),
                               active_stocks=load_pipeline_active_stocks())
    n_pending = int(events["is_pending"].sum())
    print(f"Event frame: {len(events)} events "
          f"({len(events) - n_pending} completed, {n_pending} pending)")
    print(resolution_summary(events))
    return events


def pending_events(events: pd.DataFrame, on_or_after=None) -> pd.DataFrame:
    """The upcoming-event rows. This replaces every
    `df.sort_values("date").groupby("stock").last()` in the reporting layer."""
    out = events[events["is_pending"]].copy()
    if on_or_after is not None:
        out = out[out["earnings_date"] >= pd.Timestamp(on_or_after)]
    return out.sort_values("earnings_date").reset_index(drop=True)
