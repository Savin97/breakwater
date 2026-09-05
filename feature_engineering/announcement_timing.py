# feature_engineering/announcement_timing.py
"""
Verified announcement timing and the timing-aware event clock — Phase 2 of the
methodology rebuild (audit/PHASE0_AUDIT_REV2.md Q1/Q2/Q6).

The problem
-----------
`reaction_kd = close(D+k)/close(D) - 1` assumes the announcement lands after the close
of the report date D. That is true for AMC reporters and false for BMO reporters, whose
first post-announcement session IS D. On 6,570 timestamped BMO events the measured
P(|reaction| >= 8%) is 0.041; anchored correctly it is 0.173. The target, not the model,
was wrong.

The rule
--------
An announcement window is a property of an OBSERVED TIMESTAMP and of nothing else.

    NY clock < 09:30  -> BMO        (before the open)
    NY clock >= 16:00 -> AMC        (after the close)
    otherwise         -> INTRADAY   (mid-session; the anchor is ambiguous)
    no timestamp      -> UNKNOWN

**Never infer the window from realized price behavior.** Revision 1 of the audit did
exactly that — it labelled a ticker BMO because its day-0 move exceeded its day-+1 move —
and every "corrected" number it produced was circular: the label was a function of the
same price series the corrected target re-anchors onto. Nothing in this module reads a
price, a return or a reaction; `test_announcement_timing.py` asserts that statically.

The anchor
----------
The anchor is the last close STRICTLY BEFORE the announcement:

    AMC  -> the report-date close        close(D)
    BMO  -> the previous session's close close(D-1)

and the corrected reaction spans k post-announcement trading SESSIONS from there:

    AMC k:  close(D+k)   / close(D)   - 1
    BMO k:  close(D+k-1) / close(D-1) - 1

Both therefore cover exactly k sessions beginning with the first session the market can
trade the news in. AMC is unchanged by construction, which is the bit-identity check.

INTRADAY and UNKNOWN get NO primary anchored target. They are carried with an explicit
`anchor_status` and excluded, never guessed at and never rolled.

Positions, not calendars
------------------------
Every offset above is a position in the stock's own ordered price rows, so a weekend,
a holiday or a halt shifts the window automatically. Calendar arithmetic would not.
"""
import numpy as np
import pandas as pd

BMO = "BMO"
AMC = "AMC"
INTRADAY = "INTRADAY"
UNKNOWN = "UNKNOWN"

ANNOUNCE_WINDOWS = (BMO, AMC, INTRADAY, UNKNOWN)

# Windows whose anchor is unambiguous. INTRADAY is deliberately absent: an announcement
# inside the session has no clean pre-announcement close, and picking one either leaks
# part of the reaction into the anchor or discards part of it.
ANCHORABLE_WINDOWS = (AMC, BMO)

# Offset, in the stock's own trading rows, from the report date to the anchor session.
ANCHOR_OFFSET = {AMC: 0, BMO: -1}

MARKET_OPEN_HOUR = 9.5     # 09:30 NY
MARKET_CLOSE_HOUR = 16.0   # 16:00 NY

# anchor_status vocabulary. Only "resolved" may enter a corrected target.
RESOLVED = "resolved"
PENDING = "pending"
UNRESOLVED_NO_TIMESTAMP = "unresolved_no_timestamp"
UNRESOLVED_INTRADAY = "unresolved_intraday"
UNRESOLVED_NO_SESSION = "unresolved_no_session"
UNRESOLVED_PRICE_GAP = "unresolved_price_gap"
UNRESOLVED_NO_PRIOR_SESSION = "unresolved_no_prior_session"

ANCHOR_STATUSES = (
    RESOLVED, PENDING,
    UNRESOLVED_NO_TIMESTAMP, UNRESOLVED_INTRADAY,
    UNRESOLVED_NO_SESSION, UNRESOLVED_PRICE_GAP, UNRESOLVED_NO_PRIOR_SESSION,
)

UNRESOLVED_REASONS = {
    UNRESOLVED_NO_TIMESTAMP:
        "no independently observed announcement timestamp for this event",
    UNRESOLVED_INTRADAY:
        "announced mid-session; no unambiguous pre-announcement close",
    UNRESOLVED_NO_SESSION:
        "the report date is not a market trading session (weekend/holiday/bad date)",
    UNRESOLVED_PRICE_GAP:
        "the market traded that day but this ticker has no price row (ingestion gap)",
    UNRESOLVED_NO_PRIOR_SESSION:
        "BMO event on the ticker's first price row; no prior close to anchor to",
    PENDING:
        "upcoming event; no realized outcome to anchor yet",
}

# The corrected outcome columns. Deliberately PARALLEL to reaction_{1,3,5}d /
# abs_reaction_3d, which stay exactly as they are: the legacy target is the control this
# phase is measured against, and Phase 3 is where anything switches over.
ANCHORED_REACTION_WINDOWS = (1, 3, 5)
ANCHORED_REACTION_COLS = [f"reaction_{k}d_anchored" for k in ANCHORED_REACTION_WINDOWS]
ANCHORED_OUTCOME_COLS = ANCHORED_REACTION_COLS + ["abs_reaction_3d_anchored"]

TIMING_COLS = [
    "announce_ts_ny", "announce_ts_source", "announce_window",
    "anchor_date", "anchor_status", "anchor_session_status",
]


def classify_announce_window(announce_ts_ny) -> pd.Series:
    """Announcement window from the observed NY wall-clock time, and from nothing else.

    `announce_ts_ny` is naive NY local time (see utilities/db_utilities.py). A missing
    timestamp is UNKNOWN — it is never filled in, defaulted, or guessed from behavior.
    """
    ts = pd.to_datetime(pd.Series(announce_ts_ny).values, errors="coerce")
    ts = pd.Series(ts, index=pd.RangeIndex(len(ts)))
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    hour = ts.dt.hour + ts.dt.minute / 60.0
    window = pd.Series(UNKNOWN, index=ts.index, dtype=object)
    window[ts.notna() & (hour < MARKET_OPEN_HOUR)] = BMO
    window[ts.notna() & (hour >= MARKET_CLOSE_HOUR)] = AMC
    window[ts.notna() & (hour >= MARKET_OPEN_HOUR) & (hour < MARKET_CLOSE_HOUR)] = INTRADAY
    return window


def _stock_price_series(daily_df):
    """{stock: (ordered date array, aligned price array)} — the ticker's own session grid.

    Taken from the daily frame the legacy reactions were computed on, so an AMC anchored
    reaction is the same arithmetic on the same floats as the legacy one.
    """
    d = daily_df.sort_values(["stock", "date"], kind="mergesort")
    out = {}
    for stock, sub in d.groupby("stock", sort=False):
        out[stock] = (sub["date"].to_numpy(dtype="datetime64[ns]"),
                      sub["price"].to_numpy(dtype="float64"))
    return out


def resolve_event_anchors(events: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Anchor every event and compute the timing-aware reactions.

    Returns a frame indexed like `events` with `anchor_date`, `anchor_status`,
    `anchor_session_status` and the anchored reaction columns. Legacy reaction columns
    are neither read nor written here.

    Nothing is rolled. A report date that is not a session, or that the ticker has no
    price row for, comes back unresolved with the reason recorded — Q6 of the audit
    withdrew auto-rolling precisely because those two cases have different causes and
    rolling hides both.
    """
    idx = events.index
    n_ev = len(events)
    anchor_date = np.full(n_ev, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    status = np.full(n_ev, UNRESOLVED_NO_TIMESTAMP, dtype=object)
    session_status = np.full(n_ev, "ok", dtype=object)
    reactions = {c: np.full(n_ev, np.nan) for c in ANCHORED_REACTION_COLS}

    market_sessions = pd.Index(pd.unique(daily_df["date"].to_numpy(dtype="datetime64[ns]")))
    series_by_stock = _stock_price_series(daily_df)

    ed_all = events["earnings_date"].to_numpy(dtype="datetime64[ns]")
    window_all = events["announce_window"].to_numpy(dtype=object)
    pending_all = events["is_pending"].to_numpy(dtype=bool)

    for stock, rows in events.groupby("stock", sort=False).indices.items():
        dates, px = series_by_stock.get(
            stock, (np.array([], dtype="datetime64[ns]"), np.array([], dtype="float64")))
        n = len(px)
        ed = ed_all[rows]
        pos = np.searchsorted(dates, ed)
        # Session bookkeeping runs for EVERY event, whether or not it has a timestamp, so
        # a missing price row is counted even when the window is also unknown.
        # Invariant 9: surface both, hide neither.
        on_grid = (pos < n) & (dates[np.minimum(pos, max(n - 1, 0))] == ed) if n else \
            np.zeros(len(rows), dtype=bool)
        # A pending event's date is in the future by construction, so it is never on the
        # price grid; that is not a data defect and must not be counted as one.
        off = ~on_grid & ~pending_all[rows]
        session_status[rows[pending_all[rows]]] = PENDING
        if off.any():
            # A date the market traded but this ticker has no row for is an ingestion
            # gap, not a calendar problem — Q6 counted 20 such events and they must not
            # be confused with genuine weekend/holiday dates.
            is_gap = np.isin(ed[off], market_sessions.to_numpy())
            session_status[rows[off]] = np.where(is_gap, UNRESOLVED_PRICE_GAP,
                                                 UNRESOLVED_NO_SESSION)

        w = window_all[rows]
        anchorable = np.isin(w, list(ANCHOR_OFFSET))
        offset = np.where(w == BMO, ANCHOR_OFFSET[BMO], ANCHOR_OFFSET[AMC])
        a = pos + offset

        st = np.where(
            pending_all[rows], PENDING,
            np.where(w == UNKNOWN, UNRESOLVED_NO_TIMESTAMP,
            np.where(~anchorable, UNRESOLVED_INTRADAY,
            np.where(~on_grid, session_status[rows],
            np.where(a < 0, UNRESOLVED_NO_PRIOR_SESSION, RESOLVED)))))
        status[rows] = st

        ok = st == RESOLVED
        if not ok.any():
            continue
        rows_ok, a_ok = rows[ok], a[ok]
        anchor_date[rows_ok] = dates[a_ok]
        for k in ANCHORED_REACTION_WINDOWS:
            # k post-announcement sessions measured from the anchor close. The forward
            # leg may run past the end of a still-unfolding window; that leaves a NaN
            # outcome on a resolved anchor, exactly as the legacy columns do, and the
            # diagnostics count it.
            f = a_ok + k
            inside = f < n
            vals = np.full(len(a_ok), np.nan)
            vals[inside] = px[f[inside]] / px[a_ok[inside]] - 1
            reactions[f"reaction_{k}d_anchored"][rows_ok] = vals

    out = pd.DataFrame(
        {"anchor_date": anchor_date, "anchor_status": status,
         "anchor_session_status": session_status, **reactions},
        index=idx,
    )
    out["abs_reaction_3d_anchored"] = out["reaction_3d_anchored"].abs()
    return out


def resolution_summary(events: pd.DataFrame) -> str:
    """One-line-per-bucket census of what was and was not anchored. Printed on every
    run: the 318 events that silently vanished in Q6 vanished because nothing counted."""
    completed = events[~events["is_pending"]]
    lines = [
        f"  announcement windows: "
        + "  ".join(f"{w}={int((completed['announce_window'] == w).sum())}"
                    for w in ANNOUNCE_WINDOWS)
    ]
    counts = completed["anchor_status"].value_counts()
    n_res = int(counts.get(RESOLVED, 0))
    lines.append(f"  anchored (resolved): {n_res} of {len(completed)} completed events "
                 f"({n_res / max(len(completed), 1):.1%})")
    for st, n in counts.items():
        if st == RESOLVED:
            continue
        lines.append(f"    unresolved {st}: {n} — {UNRESOLVED_REASONS.get(st, '')}")
    gaps = completed["anchor_session_status"].value_counts()
    for st in (UNRESOLVED_PRICE_GAP, UNRESOLVED_NO_SESSION):
        if st in gaps:
            lines.append(f"    report-date session check: {st} on {int(gaps[st])} events "
                         f"(counted, never rolled)")
    incomplete = int((completed["anchor_status"].eq(RESOLVED)
                      & completed["reaction_3d_anchored"].isna()).sum())
    if incomplete:
        lines.append(f"    resolved but forward window incomplete: {incomplete}")
    return "\n".join(lines)


def resolved_events(events: pd.DataFrame) -> pd.DataFrame:
    """The corrected-target dataset: completed events with a verified window and a real
    anchor. This is the ONLY slice that may feed corrected calibration or training —
    invariant 7. Everything else is excluded explicitly and counted, never dropped
    quietly and never patched up with an inferred window."""
    return events[(~events["is_pending"]) & (events["anchor_status"] == RESOLVED)]
