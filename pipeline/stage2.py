# pipeline/stage2.py
import duckdb
from config import (
    CORRECT_STOCK_COL_NAME,
    LIST_OF_POSSIBLE_STOCK_COL_NAMES,
    PRICES_PROVIDER,
    DB_PATH)
from data_ingestion.data_utilities import (
    parse_date, parse_numeric, 
    change_column_name,
    merge_prices_earnings_dates,
    map_sector_data_to_main_df)
def stage2():
    """
        Second stage of the pipeline - Data Ingestion:
        1. Import tables as needed for the pipeline. For example, if start date is set to 2020-01-01 until today, then
        the pipeline should take what it needs from the database and put it in a pandas df.
        2. Merge DFs to one df.

        Returns a df:
        stock | price | date | earnings_date | sector | sub_sector | estimated_eps | reported_eps | surprise_percentage
    """
    print("--------------------\nStage 2 - Data Ingestion...")
    con = duckdb.connect(DB_PATH)
    # Loading tables
    prices_df = con.execute("SELECT stock, price, date FROM prices ORDER BY stock, date").fetch_df()
    earnings_df = con.execute("SELECT stock, earnings_date, reported_eps, estimated_eps, surprise_percentage FROM earnings ORDER BY stock,earnings_date").fetch_df()
    stock_data_df = con.execute("SELECT * FROM stock_data ORDER BY stock").fetch_df()

    # Latest IV snapshot per stock (left join — NaN if not yet collected)
    iv_df = con.execute("""
        SELECT DISTINCT ON (stock) stock, expected_move_pct, atm_iv, snapshot_date AS iv_snapshot_date
        FROM iv_snapshots
        ORDER BY stock, snapshot_date DESC
    """).fetch_df()

    prices_df["date"] = parse_date(prices_df["date"])
    earnings_df["earnings_date"] = parse_date(earnings_df["earnings_date"])
    prices_df = prices_df.sort_values("date")
    earnings_df = earnings_df.sort_values("earnings_date")
    # Merging
    df = map_sector_data_to_main_df(prices_df, stock_data_df)
    df = merge_prices_earnings_dates(df, earnings_df) # df that holds stock prices, earnings dates
    # Sort, make sure "price" is numeric, make sure dates are datetime
    df = df.sort_values(["stock", "date"]).reset_index(drop=True)
    df["date"] = parse_date(df["date"])
    df["earnings_date"] = parse_date(df["earnings_date"])
    df["price"] = parse_numeric(df["price"])

    # Join IV — one value per stock, broadcast across all rows for that stock
    if not iv_df.empty:
        df = df.merge(iv_df, on="stock", how="left")
    else:
        df["expected_move_pct"] = float("nan")
        df["atm_iv"]            = float("nan")
        df["iv_snapshot_date"]  = None

    con.close()
    print("Stage 2 DONE")
    return df