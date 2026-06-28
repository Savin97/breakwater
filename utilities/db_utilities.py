# utilities/db_utilities.py
def create_prices_table_if_not_exists(con):
    # ensure table exists (match your schema)
    con.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            stock TEXT,
            date  DATE,
            price DOUBLE,
            ingested_at TIMESTAMP
        );
    """)
    # uniqueness constraint - only one row per (stock, date) pair, no duplicates allowed
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS prices_stock_date_uq ON prices(stock, date)")
    print("Prices table ready.")

def create_earnings_table_if_not_exists(con):
    con.execute("""
                    CREATE TABLE IF NOT EXISTS earnings (
                    stock TEXT,
                    earnings_date DATE,
                    fiscal_end_date DATE,
                    reported_eps DOUBLE,
                    estimated_eps DOUBLE,
                    surprise_percentage DOUBLE,
                    ingested_at TIMESTAMP
                ); """)
    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS earnings_unique
        ON earnings(stock, earnings_date, fiscal_end_date);
    """)
    print("Earnings table ready.")

def create_sectors_data_table_if_not_exists(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
        stock TEXT PRIMARY KEY,
        company_name TEXT,
        sector TEXT,
        sub_sector TEXT,
        ingested_at TIMESTAMP
    ); """)
    print("Stock Data table ready.")

def merge_tables(con):
    """
        Tables:
        prices (
            stock TEXT,
            date  DATE,
            price DOUBLE,
            ingested_at TIMESTAMP
        )

        earnings (
            stock TEXT,
            earnings_date DATE,
            fiscal_end_date DATE,
            reported_eps DOUBLE,
            estimated_eps DOUBLE,
            surprise_percentage DOUBLE,
            ingested_at TIMESTAMP
        )

        stock_data (
            stock TEXT PRIMARY KEY,
            company_name TEXT,
            sector TEXT,
            sub_sector TEXT,
            ingested_at TIMESTAMP
        )
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS merged_stock_data AS 
        SELECT p.stock, p.date, p.price, p.ingested_at,
            e.earnings_date, e.fiscal_end_date, e.reported_eps, e.estimated_eps, e.surprise_percentage,
            sd.sector, sd.sub_sector
        FROM prices p 
        LEFT JOIN earnings e ON p.stock = e.stock AND p.date = e.earnings_date
        LEFT JOIN stock_data sd ON p.stock = sd.stock;
        """)
    print("Merged Stock Data Table ready.") 

def stock_already_in_prices_db(con, stock: str) -> bool:
    n = con.execute("SELECT COUNT(*) FROM prices WHERE stock = ?;", [stock]).fetchone()[0]
    return n > 0

def get_max_dates_by_stock(con, table: str, date_col: str) -> dict[str, object]:
    rows = con.execute(f"""
        SELECT stock, MAX({date_col}) AS max_date
        FROM {table}
        GROUP BY stock
    """).fetchall()
    return {stock: max_date for stock, max_date in rows}

def test_db(con):
    print("\n\n---------------------\n")
    # Describe all tables
    print("Table description in the DB:")
    print(con.execute(
        """SELECT
        table_name,
        column_name,
        data_type
        FROM information_schema.columns
        ORDER BY table_name, ordinal_position; """).fetchall())
    
    earnings_db_df = con.execute("""
        SELECT *
        FROM earnings
        ORDER BY stock,earnings_date; """).df()
    
    #earnings_db_df.to_csv("earnings_db_df.csv",index=False)
    prices_count_df = con.execute("""
        SELECT stock, COUNT(*) n, MIN(date) mind, MAX(date) maxd
        FROM prices
        GROUP BY stock
        ORDER BY stock
    """).df()   
    # prices_count_df.to_csv("count_prices_db_test.csv",index=False)
    # print("\ncreated test db in count_db_test.csv\n")
    
    testing_if_all_fetched = con.execute("""
        WITH mx AS (SELECT MAX(date) AS global_max FROM prices)
        SELECT p.stock, MAX(p.date) AS max_date
        FROM prices p, mx
        GROUP BY p.stock, mx.global_max
        HAVING MAX(p.date) < mx.global_max
        ORDER BY max_date
        """).df()

    print(testing_if_all_fetched.head())
    print(con.execute("SELECT COUNT(DISTINCT stock) FROM prices").fetchone())
    print(con.execute("SELECT DISTINCT stock FROM prices ORDER BY stock").fetchdf().head())
    print(con.execute("SELECT COUNT(*) FROM prices").fetchone())

    # ---------------------
    # Earnings Table
    # ---------------------

    print("\n\n---------------------\nEarnings Table\n")
    earnings_count_df = con.execute("""
        SELECT stock, COUNT(*) n, MIN(earnings_date) mind, MAX(earnings_date) maxd
        FROM earnings
        GROUP BY stock
        ORDER BY stock
    """).df()   
    # earnings_count_df.to_csv("count_earnings_db_test.csv",index=False)
    # print("\ncreated earnings_count_df.csv\n")
    testing_if_all_fetched = con.execute("""
        WITH mx AS (SELECT MAX(earnings_date) AS global_max FROM earnings)
        SELECT e.stock, MAX(e.earnings_date) AS max_earnings_date
        FROM earnings e, mx
        GROUP BY e.stock, mx.global_max
        HAVING MAX(e.earnings_date) < mx.global_max
        ORDER BY max_earnings_date
        """).df()

    print(testing_if_all_fetched.head())
    print("Number of unique stocks in earnings: ", con.execute("SELECT COUNT(DISTINCT stock) FROM earnings").fetchone())
    print(con.execute("SELECT DISTINCT stock FROM earnings ORDER BY stock").df().head())
    print("Number of rows in earnings: ", con.execute("SELECT COUNT(*) FROM earnings").fetchone())

    # ---------------------
    # Stock Sector Table
    # ---------------------
    print("\n\n---------------------\nStock Sector Table\n")
    print(con.execute("SELECT * FROM stock_data;").df().head())
    print(con.execute("SELECT COUNT(*) FROM stock_data;").fetchone())

    stock_sector_count_df = con.execute("""
        SELECT stock, COUNT(*) n
        FROM stock_data
        GROUP BY stock
        ORDER BY stock
    """).df()   
    print(stock_sector_count_df.head())
    print("Number of unique stocks in stock_data: ", con.execute("SELECT COUNT(DISTINCT stock) FROM earnings").fetchone())
    print(con.execute("SELECT DISTINCT stock FROM stock_data ORDER BY stock").df().head())
    print("Number of rows in stock_data: ", con.execute("SELECT COUNT(*) FROM stock_data").fetchone())

    # ---------------------
    # Merged Table
    # ---------------------
    print("\n\n---------------------\nMerged Table\n---------------------")

    print(con.execute("SELECT * FROM merged_stock_data").df().head())
    merged_count_df = con.execute("""
        SELECT stock, COUNT(*) n, MIN(date) min_date, MAX(date) max_date, MIN(earnings_date) min_earnigns_date, MAX(earnings_date) max_earnings_date
        FROM merged_stock_data
        GROUP BY stock
        ORDER BY stock
    """).df()   

    print(merged_count_df.head())
    print("Number of unique stocks in merged_stock_data: ", con.execute("SELECT COUNT(DISTINCT stock) FROM merged_stock_data").fetchone())
    print(con.execute("SELECT DISTINCT stock FROM merged_stock_data ORDER BY stock").df().head())
    print("Number of rows in merged_stock_data: ", con.execute("SELECT COUNT(*) FROM merged_stock_data").fetchone())


def join_iv(df, con):
    """Join latest IV snapshot per stock onto df. Left join — NaN if not collected."""
    import pandas as pd
    iv_df = con.execute("""
        SELECT DISTINCT ON (stock) stock, expected_move_pct, atm_iv,
               snapshot_date AS iv_snapshot_date
        FROM iv_snapshots
        ORDER BY stock, snapshot_date DESC
    """).fetch_df()
    if not iv_df.empty:
        return df.merge(iv_df, on="stock", how="left")
    df["expected_move_pct"] = float("nan")
    df["atm_iv"]            = float("nan")
    df["iv_snapshot_date"]  = None
    return df


def join_eps_estimates(df, con):
    """Join latest EPS estimate snapshot per stock onto df. Left join — NaN if not collected."""
    eps_cols = [
        "eps_avg", "eps_high", "eps_low", "eps_num_analysts",
        "eps_dispersion", "eps_revision_momentum",
        "eps_trend_7d", "eps_trend_30d", "eps_trend_60d", "eps_trend_90d",
        "eps_revisions_up_7d", "eps_revisions_down_7d",
        "eps_revisions_up_30d", "eps_revisions_down_30d",
        "revenue_avg", "revenue_high", "revenue_low", "eps_snapshot_date",
    ]
    eps_df = con.execute("""
        SELECT DISTINCT ON (stock)
            stock, eps_avg, eps_high, eps_low, eps_num_analysts,
            eps_dispersion, eps_revision_momentum,
            eps_trend_7d, eps_trend_30d, eps_trend_60d, eps_trend_90d,
            eps_revisions_up_7d, eps_revisions_down_7d,
            eps_revisions_up_30d, eps_revisions_down_30d,
            revenue_avg, revenue_high, revenue_low,
            snapshot_date AS eps_snapshot_date
        FROM eps_estimates
        ORDER BY stock, snapshot_date DESC
    """).fetch_df()
    if not eps_df.empty:
        return df.merge(eps_df, on="stock", how="left")
    for col in eps_cols:
        df[col] = float("nan")
    return df


def create_eps_estimates_table_if_not_exists(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS eps_estimates (
            stock                   TEXT,
            snapshot_date           DATE,
            earnings_date           DATE,
            eps_avg                 DOUBLE,
            eps_high                DOUBLE,
            eps_low                 DOUBLE,
            eps_num_analysts        INTEGER,
            eps_dispersion          DOUBLE,
            eps_trend_7d            DOUBLE,
            eps_trend_30d           DOUBLE,
            eps_trend_60d           DOUBLE,
            eps_trend_90d           DOUBLE,
            eps_revision_momentum   DOUBLE,
            eps_revisions_up_7d     INTEGER,
            eps_revisions_down_7d   INTEGER,
            eps_revisions_up_30d    INTEGER,
            eps_revisions_down_30d  INTEGER,
            revenue_avg             DOUBLE,
            revenue_high            DOUBLE,
            revenue_low             DOUBLE,
            ingested_at             TIMESTAMP
        )
    """)
    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS eps_estimates_uq
        ON eps_estimates(stock, snapshot_date)
    """)
    print("EPS estimates table ready.")


def create_iv_table_if_not_exists(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS iv_snapshots (
            stock             TEXT,
            snapshot_date     DATE,
            snapshot_hour     INTEGER,
            earnings_date     DATE,
            days_to_earnings  INTEGER,
            current_price     DOUBLE,
            expiry_used       DATE,
            atm_strike        DOUBLE,
            atm_call_iv       DOUBLE,
            atm_put_iv        DOUBLE,
            atm_iv            DOUBLE,
            expected_move_pct DOUBLE,
            ingested_at       TIMESTAMP
        )
    """)
    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS iv_snapshots_uq
        ON iv_snapshots(stock, snapshot_date, snapshot_hour)
    """)
    print("IV snapshots table ready.")