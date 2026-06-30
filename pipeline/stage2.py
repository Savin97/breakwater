# pipeline/stage2.py
import duckdb
from datetime import date, timedelta
from config import (
    CORRECT_STOCK_COL_NAME,
    LIST_OF_POSSIBLE_STOCK_COL_NAMES,
    PRICES_PROVIDER,
    DB_PATH)
from utilities.data_utilities import (
    parse_date, parse_numeric,
    change_column_name,
    dedup_earnings,
    merge_prices_earnings_dates,
    map_sector_data_to_main_df)
from utilities.db_utilities import join_iv, join_eps_estimates


def stage2(lookback_days=None):
    """
    Stage 2 — Data Ingestion.

    lookback_days=None  (default): full load — all historical prices and earnings.
    lookback_days=N     (incremental): loads only the last N days of prices plus
                        earnings dates within the same window (past + all future),
                        giving a ~45K-row DataFrame instead of 2.8M rows.
                        90 days is sufficient warm-up for the longest rolling window (drift_60d).

    Returns a DataFrame:
        stock | price | date | earnings_date | sector | sub_sector |
        estimated_eps | reported_eps | surprise_percentage |
        expected_move_pct | atm_iv | iv_snapshot_date
    """
    print("--------------------\nStage 2 - Data Ingestion...")
    con = duckdb.connect(DB_PATH)

    if lookback_days is not None:
        cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()
        prices_df = con.execute(
            "SELECT stock, price, date FROM prices WHERE date >= ? ORDER BY stock, date",
            [cutoff]
        ).fetch_df()
        # Include recent past + all future earnings so merge_asof attaches
        # the correct upcoming earnings_date to each recent price row.
        earnings_df = con.execute(
            "SELECT stock, earnings_date, reported_eps, estimated_eps, surprise_percentage "
            "FROM earnings WHERE earnings_date >= ? ORDER BY stock, earnings_date",
            [cutoff]
        ).fetch_df()
    else:
        prices_df = con.execute(
            "SELECT stock, price, date FROM prices ORDER BY stock, date"
        ).fetch_df()
        earnings_df = con.execute(
            "SELECT stock, earnings_date, reported_eps, estimated_eps, surprise_percentage "
            "FROM earnings ORDER BY stock, earnings_date"
        ).fetch_df()

    stock_data_df = con.execute("SELECT * FROM stock_data ORDER BY stock").fetch_df()

    prices_df["date"] = parse_date(prices_df["date"])
    earnings_df["earnings_date"] = parse_date(earnings_df["earnings_date"])
    earnings_df = dedup_earnings(earnings_df)
    prices_df = prices_df.sort_values("date")
    earnings_df = earnings_df.sort_values("earnings_date")

    df = map_sector_data_to_main_df(prices_df, stock_data_df)
    df = merge_prices_earnings_dates(df, earnings_df)
    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    df["date"] = parse_date(df["date"])
    df["earnings_date"] = parse_date(df["earnings_date"])
    df["price"] = parse_numeric(df["price"])

    df = join_iv(df, con)
    df = join_eps_estimates(df, con)

    con.close()
    df = df.sort_values("date",ascending=False)
    df.head(5000).to_csv("merged_df.csv",index=False)
    exit()
    print("Stage 2 DONE")
    return df