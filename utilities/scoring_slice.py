# utilities/scoring_slice.py
"""
Loads the part of output/full_df.parquet that incremental scoring actually reads.

Why this exists
---------------
The droplet has ~590 MB of usable RAM. Reading full_df.parquet into pandas costs far
more than that, which is what produced the 75 OOM kills in the daily cron. But almost
none of the file is needed: of 2,911,875 rows only 45,693 (1.6%) are earnings days, and
the expensive per-stock statistics are aggregates over each stock's ~105 earnings events.
The other 98.4% are daily price rows those statistics never touch.

So we load two things: every earnings-day row (all history) plus the recent price window.

Why it streams instead of using filters=
----------------------------------------
pyarrow's `filters=` only prunes whole row groups using their statistics. This file has
3 row groups of ~970k rows each and earnings days are scattered through all of them, so
nothing is pruned: it decompresses all 2.9M rows and filters afterwards. Measured
2026-09-02: 584 MB that way, versus 232 MB streaming batches and filtering each one.
The resulting frame is only 12 MB either way — the peak is entirely read buffers.

CRITICAL — the returned slice is NOT contiguous in time
-------------------------------------------------------
Consecutive historical rows for a stock are ~90 days apart. `.diff()`, `.pct_change()`
and row-based `.rolling()` over them silently produce garbage that looks plausible:
recomputing daily_ret on AAPL's earnings rows gives 0.262 where the true value is 0.0104,
a 90-day move mislabelled as a one-day move. Never recompute price-derived features from
these rows — use the stored values, and aggregate only ACROSS events (groupby(stock)),
which is event-ordered and therefore valid. Recompute only within the contiguous window.
"""
from datetime import date, timedelta

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from config import INCREMENTAL_CACHED_COLS

# Columns the incremental path reads off historical earnings rows. Keep this list tight:
# peak memory scales with column count, not with the 12 MB of rows we keep.
#   - keys//masks:  stock, date, is_earnings_day, earnings_date, days_to_earnings
#   - drift_30d:    the baseline engineer_pre_earnings_drift_flag builds its z-score from
#   - the rest:     INCREMENTAL_CACHED_COLS, carried per-row rather than broadcast per-stock
SLICE_KEY_COLS = [
    "stock", "date", "is_earnings_day", "earnings_date", "days_to_earnings", "drift_30d",
]
SLICE_COLS = SLICE_KEY_COLS + [c for c in INCREMENTAL_CACHED_COLS if c not in SLICE_KEY_COLS]

# Lowest measured peak (232 MB); 250k batches cost 391 MB for the same result.
SLICE_BATCH_SIZE = 25_000


def read_scoring_slice(parquet_path="output/full_df.parquet",
                       lookback_days=None,
                       batch_size=SLICE_BATCH_SIZE,
                       columns=None):
    """
    Return every earnings-day row, plus every row within `lookback_days` of today if given.

    Streams the file in batches and filters each one, so peak memory stays near the size
    of a single batch rather than the size of the file.
    """
    columns = list(columns or SLICE_COLS)
    parquet_file = pq.ParquetFile(parquet_path)

    cutoff = None
    if lookback_days is not None:
        cutoff = pd.Timestamp(date.today() - timedelta(days=lookback_days))

    kept = []
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
        table = pa.Table.from_batches([batch])
        mask = pc.equal(table["is_earnings_day"], 1)
        if cutoff is not None:
            mask = pc.or_(mask, pc.greater_equal(table["date"], cutoff))
        kept.append(table.filter(mask))

    if not kept:
        return pd.DataFrame(columns=columns)
    return pa.concat_tables(kept).to_pandas()


def load_earnings_history(parquet_path="output/full_df.parquet", before=None,
                          batch_size=SLICE_BATCH_SIZE):
    """
    Earnings-day rows only, optionally restricted to those strictly before `before`.

    Used by stage3's incremental path: the recent window is rebuilt from the DB, so the
    parquet only needs to supply the older events the rolling baselines aggregate over.
    Passing `before` avoids returning a second, staler copy of rows the DB already gave us.
    """
    df = read_scoring_slice(parquet_path, lookback_days=None, batch_size=batch_size)
    if before is not None:
        df = df[df["date"] < pd.Timestamp(before)]
    return df.reset_index(drop=True)


def attach_earnings_history(stage3_df, parquet_path="output/full_df.parquet"):
    """
    Concatenate the stock's earnings-day history onto an incremental (windowed) frame.

    Call this AFTER the price-feature loop has run, never before: those features must be
    computed on the contiguous DB window alone. See the module docstring for why.

    Events inside the window already have a row rebuilt from the DB with fresh price
    features, so their stored statistics are merged onto that row rather than replacing
    it — the parquet copy carries only the slice columns and would blank vol/momentum/
    sector for a stock whose latest row is an earnings day. Older events exist only in the
    parquet and cannot collide with the window, so they are concatenated.

    Nothing is forward-filled onto non-earnings rows. The full path leaves these columns
    NaN off earnings days and lets the flag functions propagate, so filling them here
    diverges from the full run rather than matching it: an ffill was measured carrying a
    stale streak of 27 onto ADSK's 2026-08-27 earnings row, where the full path has NaN
    because the just-reported surprise is not in the DB yet, manufacturing an "Extended
    Beat Streak" the full path declines to assert. Consumers read the latest row via
    groupby().last(), which skips NaN and so picks up the last completed event.
    """
    history = load_earnings_history(parquet_path)
    if history.empty:
        return stage3_df

    window_start = stage3_df["date"].min()

    in_window = history.loc[history["date"] >= window_start,
                            ["stock", "date"] + INCREMENTAL_CACHED_COLS]
    stage3_df = stage3_df.merge(in_window, on=["stock", "date"], how="left")

    older = history[history["date"] < window_start]
    stage3_df = pd.concat([older, stage3_df], ignore_index=True)
    return stage3_df.sort_values(["stock", "date"], kind="mergesort")
