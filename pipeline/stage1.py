# pipeline/stage1.py
import duckdb, warnings
from utilities.db_utilities import (verify_tables_existence, merge_tables, clean_duplicate_earnings_from_db)
from ingestion.fetch_prices import ingest_all_stocks, incremental_ingest_all_prices_yf
from ingestion.fetch_earnings_dates import ingest_all_earnings_dates, incremental_ingest_all_earnings_dates_yf, get_next_earnings_dates, validate_upcoming_earnings_dates
from ingestion.fetch_sp500_sectors import ingest_all_sp500_data
from utilities.data_utilities import directory_checks
from utilities.logging_utilities import setup_logging
from config import DB_PATH

def stage1(incremental:bool):
    print("Stage 1 - Building / Updating DB...")
    setup_logging()
    warnings.filterwarnings('ignore')
    directory_checks()
    con = duckdb.connect(DB_PATH)
    verify_tables_existence(con)
    ingest_all_sp500_data(con)
    if incremental:
        incremental_ingest_all_prices_yf(con)
        incremental_ingest_all_earnings_dates_yf(con)
    else:
        ingest_all_stocks(con)
        ingest_all_earnings_dates(con)
    validate_upcoming_earnings_dates(con)
    clean_duplicate_earnings_from_db(con)
    merge_tables(con)
    con.close()
    print("Stage 1 DONE")
    return