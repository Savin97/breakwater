import duckdb
from config import DB_PATH
from utilities.db_utilities import create_iv_table_if_not_exists
from utilities.logging_utilities import setup_logging
from ingestion.fetch_iv import ingest_iv_snapshots

setup_logging()
con = duckdb.connect(DB_PATH)
create_iv_table_if_not_exists(con)
ingest_iv_snapshots(con)
con.close()
