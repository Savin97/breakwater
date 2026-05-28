# pipeline/stage1.py
import duckdb, warnings
from data_ingestion.db_functions import (
    create_prices_table_if_not_exists, 
    create_earnings_table_if_not_exists,
    create_sectors_data_table_if_not_exists, 
    create_iv_table_if_not_exists,
    merge_tables)
from data_ingestion.fetch_prices import ingest_all_stocks, ingest_all_stocks_yf
from data_ingestion.fetch_earnings_dates import ingest_all_earnings_dates, ingest_all_earnings_dates_yf, get_next_earnings_dates
from data_ingestion.fetch_sp500_sectors import ingest_all_sector_data
from data_ingestion.data_utilities import directory_checks
from data_ingestion.fetch_iv import ingest_iv_snapshots
from config import DB_PATH
def stage1(update:bool):
    """
        Building / Updating DB
        1. Create DB/Make sure it exists.
        2. Create prices, earnings, sector tables / make sure they exist.
        3. Update tables or choose to leave them as-is (introduce a switch for this)
    """
    print("Stage 1 - Building / Updating DB...")
    warnings.filterwarnings('ignore')
    directory_checks()
    con = duckdb.connect(DB_PATH)
    create_prices_table_if_not_exists(con)
    create_earnings_table_if_not_exists(con)
    create_sectors_data_table_if_not_exists(con)
    create_iv_table_if_not_exists(con)
    if update == True:
        ingest_all_stocks_yf(con)
        ingest_all_earnings_dates_yf(con)
        ingest_all_sector_data(con)
        merge_tables(con)
        ingest_iv_snapshots(con)

    con.close()
    print("Stage 1 DONE")
    return