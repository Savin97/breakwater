# pipeline/stage1.py
import duckdb, warnings
from utilities.db_utilities import (
    verify_tables_existence,
    merge_tables)
from ingestion.fetch_prices import ingest_all_stocks, incremental_ingest_all_stocks_yf
from ingestion.fetch_earnings_dates import ingest_all_earnings_dates, incremental_ingest_all_earnings_dates_yf, get_next_earnings_dates, validate_upcoming_earnings_dates
from ingestion.fetch_sp500_sectors import ingest_all_sector_data
from utilities.data_utilities import directory_checks
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
    verify_tables_existence(con)
    
    if update == True:
        incremental_ingest_all_stocks_yf(con)
        incremental_ingest_all_earnings_dates_yf(con)
        validate_upcoming_earnings_dates(con)
        ingest_all_sector_data(con)
        merge_tables(con)

    con.close()
    print("Stage 1 DONE")
    return