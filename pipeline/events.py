# pipeline/events.py
"""
The event frame — one row per earnings event, completed or pending.

Phase 1 of the methodology rebuild (audit/PHASE0_AUDIT_REV2.md §Q5).

Problem it solves
-----------------
Every statistic that aggregates across a stock's earnings events was written only onto
earnings-day rows of the daily price frame. Consumers recovered "current state" with
`df.sort_values("date").groupby("stock").last()`, which skips NaN column by column and
so reached back to the stock's LAST COMPLETED EVENT. 100% of shipped upcoming scores,
tiers and lifts were one earnings event stale.

How it is fixed
---------------
`build_event_frame` appends ONE pending row per eligible stock, taken from that stock's
final daily row (so its price features are as-of today) and stamped with the upcoming
earnings_date. `score_event_frame` then runs the shared cores in
feature_engineering/event_features.py over completed and pending rows together.

Because a pending row
  * carries a NaN outcome, and
  * sorts last within its stock (its earnings_date is later than every completed one),
every `.shift(1)` / `.rolling()` / `.expanding()` guard is preserved: completed rows
cannot see it, and its own statistics naturally span every completed prior event
INCLUDING the most recent. No statistic, threshold or weight changes.

The daily frame is never modified. Nothing here is inserted into it, so merge_asof,
the per-stock rolling price windows and the `groupby("date")` cross-sectional ranks are
all untouched.

Guarantee
---------
`assert_completed_parity` proves that every completed event's recomputed columns are
identical to what the daily pipeline produced. It runs on every pipeline execution.
"""
import numpy as np
import pandas as pd

from config import DEFAULT_REACTION_WINDOW
from feature_engineering.event_features import (
    event_reaction_std,
    event_reaction_entropy,
    event_directional_bias,
    event_abs_reaction_median,
    event_abs_reaction_p75,
    event_abs_reaction_p75_rolling,
    event_abs_reaction_p90_rolling,
    event_surprise_features,
    event_pre_earnings_drift_z,
    event_earnings_explosiveness,
    event_explosiveness_score,
    event_stock_bucket_lift_values,
    event_lift_adjusted_bucket,
    event_high_conviction,
    surprise_momentum_flag_values,
    pre_earnings_drift_flag_values,
)
from scoring.scoring_features import classify_large_relative_earnings_move_bucket

# Realized outcomes. A pending event has none — blanked so nothing downstream can read a
# neighbouring event's result as this one's, and so the lift's expanding aggregations
# (which skip NaN) cannot count a pending row as a non-extreme observation.
OUTCOME_COLS = [
    "reaction_1d", "reaction_3d", "reaction_5d", "abs_reaction_3d",
    "is_up", "is_down", "is_nochange",
    "is_large_reaction", "is_extreme_reaction", "earnings_move_bucket",
]

# History-dependent state recomputed on the event frame. Blanked on pending rows at
# construction so a stale carried value can never survive a core failing to overwrite it.
RECOMPUTED_COLS = [
    "reaction_std", "reaction_entropy", "directional_bias",
    "abs_reaction_median", "abs_reaction_p75",
    "abs_reaction_p75_rolling", "abs_reaction_p90_rolling",
    "surprise_mean_5", "surprise_std_5", "surprise_streak",
    "pre_earnings_drift_z",
    "earnings_explosiveness_z", "earnings_tail_z",
    "earnings_explosiveness_score", "earnings_explosiveness_bucket",
    "earnings_explosiveness_bucket_structural", "stock_bucket_lift",
    "surprise_momentum_flag", "pre_earnings_drift_flag",
    "risk_score", "is_high_conviction",
]

# Carried verbatim from the source daily row, never recomputed here: these are
# properties of a DAY, not of an event history, and several depend on the composition of
# the frame they are computed over (cross-sectional ranks per date, a global quantile in
# score_momentum_fragility). Recomputing them on a ~46k-row event frame would silently
# change their meaning. Phase 1 changes no scoring, so they stay on the daily frame.
CARRIED_NOTE = (
    "vol_*, sector_*, momentum_pressure_regime, proximity_score, vol_expansion_score, "
    "momentum_fragility_score, expected_move_pct/atm_iv, eps_* — all as-of score_asof_date"
)


def build_event_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (stock, earnings event): every completed event, plus one pending row
    per eligible stock.

    A stock is eligible for a pending row when its final daily row carries a future
    earnings_date (days_to_earnings > 0). A stock whose final row IS its earnings day
    gets no pending row — the next date is not yet known from the forward merge_asof.
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
    pending["is_pending"] = True
    pending["is_earnings_day"] = 0

    for col in OUTCOME_COLS + RECOMPUTED_COLS:
        if col in pending.columns:
            pending[col] = np.nan

    events = pd.concat([completed, pending], ignore_index=True)
    # Pending last within a stock (its earnings_date is the furthest out) and, for the
    # global lift ordering, last within a date.
    events = events.sort_values(
        ["stock", "earnings_date", "is_pending"], kind="mergesort"
    ).reset_index(drop=True)

    events["score_asof_date"] = events["date"]

    # A pending row is only as fresh as its stock's last price row. A ticker whose feed
    # has stopped (delisted, halted, renamed) still carries a future earnings_date from
    # the DB and so still produces a pending row — scored, correctly, as of a stale
    # date. Surfaced rather than dropped: dropping it would silently shrink the universe.
    if not pending.empty:
        latest = daily_df["date"].max()
        stale = pending.loc[pending["date"] < latest, "stock"].tolist()
        if stale:
            print(f"  NOTE: {len(stale)} pending events scored as-of a price date older "
                  f"than {latest.date()} (stale feed): {', '.join(sorted(stale))}")
    events["event_id"] = (
        events["stock"].astype(str) + "|" + events["earnings_date"].dt.strftime("%Y-%m-%d")
    )
    return events


def _apply_core(events, mask, core, out_cols, **kwargs):
    """Run an event-level core over `mask` rows PLUS every pending row, and write the
    result back. `mask` mirrors the row set the daily pipeline uses for that statistic,
    so completed rows see an identical sequence; the pending rows ride along so their own
    statistics cover all completed prior events."""
    sel = mask | events["is_pending"]
    sub = core(events.loc[sel].copy(), **kwargs)
    for col in out_cols:
        events.loc[sel, col] = sub[col].values
    return events


def score_event_frame(events: pd.DataFrame) -> pd.DataFrame:
    """Recompute every history-dependent statistic and score on the event frame.

    Row-set masks below mirror the daily pipeline exactly (see the corresponding
    engineer_* functions); only the pending rows are additional.
    """
    events = events.copy()
    events["_best_reaction"] = events["reaction_3d"].fillna(events["reaction_1d"])

    completed = ~events["is_pending"]
    has_reaction = events[DEFAULT_REACTION_WINDOW].notna()   # build_earnings_df mask
    has_best = events["_best_reaction"].notna()              # entropy mask

    events = _apply_core(events, has_reaction, event_reaction_std, ["reaction_std"])
    events = _apply_core(events, has_best, event_reaction_entropy, ["reaction_entropy"])
    events = _apply_core(events, has_reaction, event_directional_bias, ["directional_bias"])
    events = _apply_core(events, has_reaction, event_abs_reaction_median, ["abs_reaction_median"])
    events = _apply_core(events, has_reaction, event_abs_reaction_p75, ["abs_reaction_p75"])
    events = _apply_core(events, completed, event_abs_reaction_p75_rolling, ["abs_reaction_p75_rolling"])
    events = _apply_core(events, completed, event_abs_reaction_p90_rolling, ["abs_reaction_p90_rolling"])
    events = _apply_core(events, completed, event_surprise_features,
                         ["surprise_mean_5", "surprise_std_5", "surprise_streak"])
    events = _apply_core(events, completed, event_pre_earnings_drift_z, ["pre_earnings_drift_z"])

    events = event_earnings_explosiveness(events)
    events = event_explosiveness_score(events)

    # Lift needs every event, pending included — a pending row contributes NaN and so
    # cannot move the market baseline it is measured against.
    events["stock_bucket_lift"] = np.nan
    lift = event_stock_bucket_lift_values(
        events[["stock", "date", "is_pending", "is_extreme_reaction",
                "earnings_explosiveness_bucket"]]
    )
    events.loc[lift.index, "stock_bucket_lift"] = lift
    events = event_lift_adjusted_bucket(events)
    events = classify_large_relative_earnings_move_bucket(events)

    all_rows = pd.Series(True, index=events.index)
    events["surprise_momentum_flag"] = surprise_momentum_flag_values(
        events["surprise_streak"], events["surprise_mean_5"], events["surprise_std_5"], all_rows
    )
    events["pre_earnings_drift_flag"] = pre_earnings_drift_flag_values(
        events["pre_earnings_drift_z"], all_rows
    )
    events["risk_score"] = events["earnings_explosiveness_score"]
    events = event_high_conviction(events, events["earnings_explosiveness_bucket"])

    return events.drop(columns=["_best_reaction"])


# Columns that must match the daily pipeline exactly on completed events. The two flag
# columns are excluded: on the daily frame they are computed on earnings-day rows and
# then propagated forward onto pre-earnings rows, which the event frame has none of —
# their earnings-day values are covered by PARITY_COLS via is_high_conviction and are
# asserted separately below.
PARITY_COLS = [
    "reaction_std", "reaction_entropy", "directional_bias",
    "abs_reaction_median", "abs_reaction_p75",
    "abs_reaction_p75_rolling", "abs_reaction_p90_rolling",
    "surprise_mean_5", "surprise_std_5", "surprise_streak",
    "pre_earnings_drift_z", "earnings_explosiveness_z", "earnings_tail_z",
    "earnings_explosiveness_score", "stock_bucket_lift", "earnings_move_bucket",
    "earnings_explosiveness_bucket", "earnings_explosiveness_bucket_structural",
    "surprise_momentum_flag", "pre_earnings_drift_flag",
    "risk_score", "is_high_conviction",
]


def completed_parity_report(events: pd.DataFrame, daily_df: pd.DataFrame) -> dict:
    """Compare every completed event's recomputed columns against the daily pipeline.
    Returns {column: n_mismatched}; an empty dict means exact agreement."""
    left = events[~events["is_pending"]].set_index(["stock", "earnings_date"]).sort_index()
    right = (
        daily_df[daily_df["is_earnings_day"] == 1]
        .set_index(["stock", "earnings_date"]).sort_index()
    )
    if not left.index.equals(right.index):
        raise AssertionError(
            f"completed event index mismatch: {len(left)} event rows vs {len(right)} "
            f"earnings-day rows in the daily frame"
        )

    diffs = {}
    for col in PARITY_COLS:
        if col not in right.columns:
            continue
        a, b = left[col], right[col]
        if a.dtype.kind in "fi" and b.dtype.kind in "fi":
            same = np.array_equal(a.to_numpy(dtype="float64"),
                                  b.to_numpy(dtype="float64"), equal_nan=True)
            n_bad = 0 if same else int((~np.isclose(
                a.to_numpy(dtype="float64"), b.to_numpy(dtype="float64"), equal_nan=True)).sum())
        else:
            eq = (a.astype(object).where(a.notna(), "<NA>") ==
                  b.astype(object).where(b.notna(), "<NA>"))
            n_bad = int((~eq).sum())
        if n_bad:
            diffs[col] = n_bad
    return diffs


def assert_completed_parity(events: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    diffs = completed_parity_report(events, daily_df)
    if diffs:
        raise AssertionError(
            "Event-frame refactor changed completed historical events — this must never "
            f"happen in Phase 1. Mismatched columns: {diffs}"
        )


def build_and_score_event_frame(daily_df: pd.DataFrame, verify: bool = True) -> pd.DataFrame:
    events = score_event_frame(build_event_frame(daily_df))
    if verify:
        assert_completed_parity(events, daily_df)
    n_pending = int(events["is_pending"].sum())
    print(f"Event frame: {len(events)} events "
          f"({len(events) - n_pending} completed, {n_pending} pending)")
    return events


def pending_events(events: pd.DataFrame, on_or_after=None) -> pd.DataFrame:
    """The upcoming-event rows. This replaces every
    `df.sort_values("date").groupby("stock").last()` in the reporting layer."""
    out = events[events["is_pending"]].copy()
    if on_or_after is not None:
        out = out[out["earnings_date"] >= pd.Timestamp(on_or_after)]
    return out.sort_values("earnings_date").reset_index(drop=True)
