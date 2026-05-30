import duckdb
from config import DB_PATH
from data_ingestion.db_functions import create_iv_table_if_not_exists
from data_ingestion.fetch_iv import ingest_iv_snapshots

con = duckdb.connect(DB_PATH)
create_iv_table_if_not_exists(con)
ingest_iv_snapshots(con)
con.close()
