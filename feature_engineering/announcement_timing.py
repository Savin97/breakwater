# feature_engineering/announcement_timing.py
"""
Verified announcement timing and the timing-aware event clock — Phase 2 of the
methodology rebuild (audit/PHASE0_AUDIT_REV2.md Q1/Q2/Q6).

The problem
-----------
`reaction_kd = close(D+k)/close(D) - 1` assumes the announcement lands after the close
of the report date D. That is true for AMC reporters and false for BMO reporters, whose
first post-announcement session IS D. On 6,570 timestamped BMO events the measured
P(|reaction| >= 8%) is 0.041; anchored correctly it is 0.173.

What that does and does not establish: the LEGACY TARGET IS PROVEN WRONG FOR BMO
EVENTS — it measures a window that starts one session after the news. It establishes
nothing about the model. The validity and the incremental value of the scoring model
remain unestablished, pending the corrected-history rebuild (Phase 3) and a
competitive-baseline validation. No claim about model quality may be read off this
module or off the diagnostics built on it.

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
trade the news in. AMC is unchanged by construction wherever the ticker has a price row
on every session in the window, which is the bit-identity check.

INTRADAY and UNKNOWN get NO primary anchored target. They are carried with an explicit
`anchor_status` and excluded, never guessed at and never rolled.

The market-session grid, not the ticker's own rows
--------------------------------------------------
Every offset above is a position on the CANONICAL MARKET-SESSION GRID — the sorted set
of dates the loaded price data shows the market trading — and never a position in the
ticker's own price rows.

That distinction is the whole of external-review item 1. Counting positions in the
ticker's own rows silently absorbs a missing price row: a ticker with a three-session
ingestion hole after its report date gets `shift(-3)` landing on D+6, so the "3-session"
window is really six sessions long, and a BMO event whose D-1 row is missing anchors to
D-2, a two-session window mislabelled as one. Both are invisible: the arithmetic
succeeds and produces a number.

So the grid decides WHICH dates are required, and the ticker must then have a price row
on those EXACT dates:

    anchor session      = grid[i + offset]        i = the report date's grid index
    k-session endpoint  = grid[i + offset + k]

    AMC:  anchor = D,   k-endpoint = D+k
    BMO:  anchor = D-1, k-endpoint = D+(k-1)

A required date the ticker has no row for leaves that outcome UNAVAILABLE with a stated
reason. It is never satisfied by the nearest neighbouring row, and no calendar-day
arithmetic is used anywhere.
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

# Offset, in MARKET SESSIONS, from the report date to the anchor session.
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
UNRESOLVED_ANCHOR_BEFORE_HISTORY = "unresolved_anchor_before_history"

ANCHOR_STATUSES = (
    RESOLVED, PENDING,
    UNRESOLVED_NO_TIMESTAMP, UNRESOLVED_INTRADAY,
    UNRESOLVED_NO_SESSION, UNRESOLVED_PRICE_GAP, UNRESOLVED_NO_PRIOR_SESSION,
    UNRESOLVED_ANCHOR_BEFORE_HISTORY,
)

UNRESOLVED_REASONS = {
    UNRESOLVED_NO_TIMESTAMP:
        "no independently observed announcement timestamp for this event",
    UNRESOLVED_INTRADAY:
        "announced mid-session; no unambiguous pre-announcement close",
    UNRESOLVED_NO_SESSION:
        "the report date is not a market trading session (weekend/holiday/bad date)",
    UNRESOLVED_PRICE_GAP:
        "the market traded the required anchor session but this ticker has no price "
        "row for it (ingestion gap)",
    UNRESOLVED_NO_PRIOR_SESSION:
        "BMO event on the first session of the loaded market grid; no prior close",
    UNRESOLVED_ANCHOR_BEFORE_HISTORY:
        "the required anchor session precedes this ticker's first price row",
    PENDING:
        "upcoming event; no realized outcome to anchor yet",
}

# ── Per-target availability ───────────────────────────────────────────────────
# An anchor being resolved says the PRE-announcement close is real. It says nothing
# about the far end of the window. These statuses carry that second question, one per
# horizon, so "anchor resolved" and "target available" can never be confused again
# (external-review item 3).
TARGET_AVAILABLE = "available"
TARGET_ENDPOINT_BEYOND_GRID = "unavailable_endpoint_beyond_market_grid"
TARGET_ENDPOINT_AFTER_LAST_PRICE = "unavailable_endpoint_after_last_price"
TARGET_ENDPOINT_PRICE_GAP = "unavailable_endpoint_price_gap"

TARGET_UNAVAILABLE_REASONS = {
    TARGET_ENDPOINT_BEYOND_GRID:
        "the endpoint session is past the end of the loaded market grid (the window "
        "has not finished unfolding)",
    TARGET_ENDPOINT_AFTER_LAST_PRICE:
        "the endpoint session is past this ticker's last price row",
    TARGET_ENDPOINT_PRICE_GAP:
        "the market traded the endpoint session but this ticker has no price row for "
        "it (ingestion gap)",
}

# The corrected outcome columns. Deliberately PARALLEL to reaction_{1,3,5}d /
# abs_reaction_3d, which stay exactly as they are: the legacy target is the control this
# phase is measured against, and Phase 3 is where anything switches over.
ANCHORED_REACTION_WINDOWS = (1, 3, 5)
ANCHORED_REACTION_COLS = [f"reaction_{k}d_anchored" for k in ANCHORED_REACTION_WINDOWS]
ANCHORED_OUTCOME_COLS = ANCHORED_REACTION_COLS + ["abs_reaction_3d_anchored"]
ANCHORED_STATUS_COLS = [f"reaction_{k}d_anchored_status" for k in ANCHORED_REACTION_WINDOWS]

# The corrected target Phase 3 will be fit against, and the default gate below.
DEFAULT_ANCHORED_TARGET = "abs_reaction_3d_anchored"

# Which status column backs each anchored outcome column.
ANCHORED_STATUS_FOR = {f"reaction_{k}d_anchored": f"reaction_{k}d_anchored_status"
                       for k in ANCHORED_REACTION_WINDOWS}
ANCHORED_STATUS_FOR["abs_reaction_3d_anchored"] = "reaction_3d_anchored_status"

TIMING_COLS = [
    "announce_ts_ny", "announce_ts_source", "announce_ts_observed_at",
    "announce_window", "anchor_date", "anchor_status", "anchor_session_status",
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


def market_session_grid(daily_df) -> np.ndarray:
    """The canonical market-session grid: every date the loaded price data shows the
    market trading, sorted and deduplicated.

    Derived from the same frame the legacy reactions were computed on, so it needs no
    exchange calendar dependency and cannot drift from the data. A date absent here is a
    date we have no evidence the market traded.
    """
    return np.sort(pd.unique(daily_df["date"].to_numpy(dtype="datetime64[ns]")))


def _stock_price_series(daily_df):
    """{stock: (ordered date array, aligned price array)} — the ticker's observed rows.

    Used ONLY to look up a price on an exact date chosen by the grid. Positions in this
    array are never used as an offset unit.
    """
    d = daily_df.sort_values(["stock", "date"], kind="mergesort")
    out = {}
    for stock, sub in d.groupby("stock", sort=False):
        out[stock] = (sub["date"].to_numpy(dtype="datetime64[ns]"),
                      sub["price"].to_numpy(dtype="float64"))
    return out


def _exact_positions(dates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Index of each target date in `dates`, or -1 where that EXACT date is absent.

    Never the nearest neighbour, and never the next row along: accepting a neighbour is
    precisely what stretches a k-session window past a missing row. NaT targets come
    back -1.
    """
    out = np.full(len(targets), -1, dtype=np.int64)
    if len(dates) == 0 or len(targets) == 0:
        return out
    p = np.searchsorted(dates, targets)
    hit = (p < len(dates)) & (dates[np.minimum(p, len(dates) - 1)] == targets)
    out[hit] = p[hit]
    return out


_NAT = np.datetime64("NaT", "ns")


def resolve_event_anchors(events: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Anchor every event on the market-session grid and compute the timing-aware
    reactions.

    Returns a frame indexed like `events` with `anchor_date`, `anchor_status`,
    `anchor_session_status`, the anchored reaction columns and a per-horizon
    `reaction_{k}d_anchored_status`. Legacy reaction columns are neither read nor
    written here.

    Nothing is rolled and nothing is approximated. A report date that is not a market
    session, an anchor session the ticker has no row for, and an endpoint session the
    ticker has no row for are three different failures with three different names —
    Q6 of the audit withdrew auto-rolling precisely because collapsing them hides both
    calendar problems and ingestion bugs.
    """
    idx = events.index
    n_ev = len(events)
    anchor_date = np.full(n_ev, _NAT, dtype="datetime64[ns]")
    status = np.full(n_ev, UNRESOLVED_NO_TIMESTAMP, dtype=object)
    session_status = np.full(n_ev, "ok", dtype=object)
    reactions = {c: np.full(n_ev, np.nan) for c in ANCHORED_REACTION_COLS}
    target_status = {c: np.full(n_ev, "", dtype=object) for c in ANCHORED_STATUS_COLS}

    grid = market_session_grid(daily_df)
    n_grid = len(grid)
    series_by_stock = _stock_price_series(daily_df)
    empty = (np.array([], dtype="datetime64[ns]"), np.array([], dtype="float64"))

    ed_all = events["earnings_date"].to_numpy(dtype="datetime64[ns]")
    window_all = events["announce_window"].to_numpy(dtype=object)
    pending_all = events["is_pending"].to_numpy(dtype=bool)

    for stock, rows in events.groupby("stock", sort=False).indices.items():
        dates, px = series_by_stock.get(stock, empty)
        n_px = len(px)
        ed = ed_all[rows]
        pend = pending_all[rows]
        w = window_all[rows]

        # ── report-date bookkeeping, for EVERY event whatever its window ──────
        # Informational only: it says whether the report date itself is a session this
        # ticker has a row for. Invariant 9 — surface it, hide nothing. A pending
        # event's date is in the future by construction, so its absence is not a defect.
        i_grid = _exact_positions(grid, ed)
        pos_report = _exact_positions(dates, ed)
        session_status[rows] = np.where(
            pend, PENDING,
            np.where(pos_report >= 0, "ok",
                     np.where(i_grid >= 0, UNRESOLVED_PRICE_GAP, UNRESOLVED_NO_SESSION)))

        # ── the anchor session, chosen on the grid ────────────────────────────
        anchorable = np.isin(w, ANCHORABLE_WINDOWS)
        offset = np.where(w == BMO, ANCHOR_OFFSET[BMO], ANCHOR_OFFSET[AMC])
        a_grid = i_grid + offset                       # meaningless where i_grid < 0
        a_ok_grid = (i_grid >= 0) & (a_grid >= 0)
        a_date = np.where(a_ok_grid, grid[np.clip(a_grid, 0, max(n_grid - 1, 0))], _NAT) \
            if n_grid else np.full(len(rows), _NAT, dtype="datetime64[ns]")
        pos_anchor = _exact_positions(dates, a_date)

        # "before this ticker's history" and "a hole inside it" have different causes:
        # the first is a young/newly-listed ticker, the second an ingestion bug.
        before_hist = (pos_anchor < 0) & (
            np.full(len(rows), True) if n_px == 0 else (a_date < dates[0]))

        st = np.where(
            pend, PENDING,
            np.where(w == UNKNOWN, UNRESOLVED_NO_TIMESTAMP,
            np.where(~anchorable, UNRESOLVED_INTRADAY,
            np.where(i_grid < 0, UNRESOLVED_NO_SESSION,
            np.where(a_grid < 0, UNRESOLVED_NO_PRIOR_SESSION,
            np.where(pos_anchor >= 0, RESOLVED,
            np.where(before_hist, UNRESOLVED_ANCHOR_BEFORE_HISTORY,
                     UNRESOLVED_PRICE_GAP)))))))
        status[rows] = st

        ok = st == RESOLVED
        anchor_date[rows[ok]] = a_date[ok]

        for k in ANCHORED_REACTION_WINDOWS:
            rcol, scol = f"reaction_{k}d_anchored", f"reaction_{k}d_anchored_status"
            f_grid = a_grid + k
            beyond = ~a_ok_grid | (f_grid >= n_grid)
            f_date = np.where(beyond, _NAT, grid[np.clip(f_grid, 0, max(n_grid - 1, 0))]) \
                if n_grid else np.full(len(rows), _NAT, dtype="datetime64[ns]")
            pos_f = _exact_positions(dates, f_date)
            after_last = (pos_f < 0) & ~beyond & (
                np.full(len(rows), True) if n_px == 0 else (f_date > dates[-1]))

            ts = np.where(
                ~ok, st,
                np.where(beyond, TARGET_ENDPOINT_BEYOND_GRID,
                np.where(pos_f >= 0, TARGET_AVAILABLE,
                np.where(after_last, TARGET_ENDPOINT_AFTER_LAST_PRICE,
                         TARGET_ENDPOINT_PRICE_GAP))))
            target_status[scol][rows] = ts

            good = ok & (pos_f >= 0)
            if good.any():
                vals = np.full(len(rows), np.nan)
                vals[good] = px[pos_f[good]] / px[pos_anchor[good]] - 1
                reactions[rcol][rows] = vals

    out = pd.DataFrame(
        {"anchor_date": anchor_date, "anchor_status": status,
         "anchor_session_status": session_status, **reactions, **target_status},
        index=idx,
    )
    out["abs_reaction_3d_anchored"] = out["reaction_3d_anchored"].abs()
    return out


def resolution_summary(events: pd.DataFrame) -> str:
    """One-line-per-bucket census of what was and was not anchored. Printed on every
    run: the 318 events that silently vanished in Q6 vanished because nothing counted."""
    completed = events[~events["is_pending"]]
    n = max(len(completed), 1)
    lines = [
        "  announcement windows: "
        + "  ".join(f"{w}={int((completed['announce_window'] == w).sum())}"
                    for w in ANNOUNCE_WINDOWS)
    ]
    counts = completed["anchor_status"].value_counts()
    n_res = int(counts.get(RESOLVED, 0))
    lines.append(f"  anchored (resolved): {n_res} of {len(completed)} completed events "
                 f"({n_res / n:.1%})")
    for st, cnt in counts.items():
        if st == RESOLVED:
            continue
        lines.append(f"    unresolved {st}: {cnt} — {UNRESOLVED_REASONS.get(st, '')}")
    gaps = completed["anchor_session_status"].value_counts()
    for st in (UNRESOLVED_PRICE_GAP, UNRESOLVED_NO_SESSION):
        if st in gaps:
            lines.append(f"    report-date session check: {st} on {int(gaps[st])} events "
                         f"(counted, never rolled)")
    # "anchor resolved" is not "target available" — say so with numbers, every run.
    for k in ANCHORED_REACTION_WINDOWS:
        scol = f"reaction_{k}d_anchored_status"
        if scol not in completed.columns:
            continue
        sub = completed.loc[completed["anchor_status"] == RESOLVED, scol]
        avail = int((sub == TARGET_AVAILABLE).sum())
        lines.append(f"    reaction_{k}d_anchored available on {avail} of {n_res} "
                     f"resolved anchors")
        for st, cnt in sub[sub != TARGET_AVAILABLE].value_counts().items():
            lines.append(f"      {st}: {cnt} — {TARGET_UNAVAILABLE_REASONS.get(st, '')}")
    return "\n".join(lines)


def anchor_resolved_events(events: pd.DataFrame) -> pd.DataFrame:
    """Completed events whose ANCHOR is real — the pre-announcement close exists.

    This is the anchoring control slice (it is what the AMC bit-identity check runs on).
    It is NOT the training/calibration gate: an anchor can be resolved while the far end
    of the window has not closed yet, or falls on a session this ticker has no price row
    for. Use `resolved_events()` for anything that consumes an outcome.
    """
    return events[(~events["is_pending"]) & (events["anchor_status"] == RESOLVED)]


def resolved_events(events: pd.DataFrame,
                    target: str | None = DEFAULT_ANCHORED_TARGET) -> pd.DataFrame:
    """The corrected-target dataset: completed events with a verified window, a real
    anchor, AND a non-null value for the requested anchored target.

    This is the ONLY slice that may feed corrected calibration or training — invariant
    7. Everything else is excluded explicitly and counted, never dropped quietly and
    never patched up with an inferred window.

    External-review item 3: "anchor resolved" and "target available" are separate
    questions and this helper now answers the second one. `target` names the anchored
    outcome the caller is actually going to consume (default
    `abs_reaction_3d_anchored`); `target=None` drops back to the anchor-only slice and
    is equivalent to `anchor_resolved_events`, which is what a caller studying the
    anchoring itself should use.
    """
    out = anchor_resolved_events(events)
    if target is None:
        return out
    if target not in ANCHORED_OUTCOME_COLS:
        raise ValueError(
            f"{target!r} is not an anchored outcome column; expected one of "
            f"{ANCHORED_OUTCOME_COLS}")
    return out[out[target].notna()]
