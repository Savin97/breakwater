import duckdb
from config import DB_PATH
from utilities.db_utilities import create_eps_estimates_table_if_not_exists
from ingestion.fetch_eps_estimates import ingest_eps_estimates

con = duckdb.connect(DB_PATH)
create_eps_estimates_table_if_not_exists(con)
ingest_eps_estimates(con)
con.close()
