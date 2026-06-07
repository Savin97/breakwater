# pipeline/incremental.py
import os
from datetime import date
import duckdb
import pandas as pd
from config import DB_PATH, INCREMENTAL_LOOKBACK_DAYS
from pipeline.stage2 import stage2
from pipeline.stage3 import stage3
from pipeline.stage4 import stage4
from streamlit_dash.streamlit_export import export_upcoming_df


def run_incremental():
    """
    Fast incremental update (~5-10s vs 80s full run).

    Loads only the last INCREMENTAL_LOOKBACK_DAYS of prices, reads stable
    expanding stats from full_df.parquet, recomputes price-dependent features
    and scoring, and writes only upcoming_df.parquet.

    Falls back to run_pipeline() automatically if new earnings events are
    detected in the DB that aren't in the parquet yet.
    """
    if _has_new_earnings():
        from pipeline.pipeline import run_pipeline
        return run_pipeline()
    df = stage2(lookback_days=INCREMENTAL_LOOKBACK_DAYS)
    df = stage3(df, incremental=True)
    df = stage4(df, incremental=True)
    export_upcoming_df(df)


def _has_new_earnings():
    """
    Returns True if the DB contains earnings events that occurred after the
    last full pipeline run (approximated by the max price date per stock in
    the parquet).

    Compares max(earnings_date in DB, past events only) against
    max(price date in parquet) per stock. An event on a date where no price
    row exists (e.g. earnings released on a market holiday) is NOT counted as
    new — the pipeline can't process it regardless.
    """
    parquet_path = "output/full_df.parquet"
    if not os.path.exists(parquet_path):
        return True

    cached = pd.read_parquet(parquet_path, columns=["stock", "date", "earnings_date", "is_earnings_day"])
    parquet_max_price = cached.groupby("stock")["date"].max()

    con = duckdb.connect(DB_PATH)
    db_max_df = con.execute(
        "SELECT stock, MAX(earnings_date) AS max_date FROM earnings "
        "WHERE earnings_date <= ? GROUP BY stock",
        [date.today().isoformat()]
    ).fetch_df()
    con.close()

    db_max = pd.to_datetime(db_max_df.set_index("stock")["max_date"])
    common = db_max.index.intersection(parquet_max_price.index)
    new_stocks = db_max[common] > parquet_max_price[common]
    if new_stocks.any():
        print(f"New earnings events detected ({new_stocks.sum()} stocks) — running full pipeline.")
    return bool(new_stocks.any())
