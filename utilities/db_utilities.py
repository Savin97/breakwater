# utilities/db_utilities.py
"""
    Function definitions:
    create_prices_table_if_not_exists
    create_earnings_table_if_not_exists
    create_sectors_data_table_if_not_exists
    create_iv_table_if_not_exists
    create_eps_estimates_table_if_not_exists
    create_predictions_table_if_not_exists
"""
import pandas as pd
import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)

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
    # Phase 2 (audit/PHASE0_AUDIT_REV2.md Q1) — the observed announcement time, so the
    # BMO/AMC window is a recorded fact rather than something a later analysis infers
    # from price behavior. Naive NY LOCAL time; the column name carries the zone, which
    # keeps it reproducible regardless of the reader's session timezone.
    # announce_ts_source records where the timestamp came from, so provenance survives
    # into the event frame (invariant 8). NULL means "not observed" — never a default,
    # never back-filled from a guess. The AlphaVantage history is date-only and stays
    # NULL forever; those events are simply unresolved.
    con.execute("ALTER TABLE earnings ADD COLUMN IF NOT EXISTS announce_ts_ny TIMESTAMP")
    con.execute("ALTER TABLE earnings ADD COLUMN IF NOT EXISTS announce_ts_source TEXT")


def load_announcement_timing(con) -> pd.DataFrame:
    """Observed announcement timestamps, keyed by (stock, earnings_date).

    The event frame's only source of announcement timing. Reads the `earnings` table and
    nothing else — in particular it never reads audit/provider_timestamps.parquet, which
    is a one-time seed loaded into this column by
    scripts/backfill_announcement_timestamps.py, not a runtime input.

    Returns an empty frame (not an error) on a DB predating the column, so an old
    database degrades to "every event UNKNOWN and unresolved" rather than to a crash.
    """
    cols = {row[0] for row in con.execute("DESCRIBE earnings").fetchall()}
    if "announce_ts_ny" not in cols:
        return pd.DataFrame(columns=["stock", "earnings_date",
                                     "announce_ts_ny", "announce_ts_source"])
    out = con.execute("""
        SELECT stock, earnings_date, announce_ts_ny, announce_ts_source
        FROM earnings
        WHERE announce_ts_ny IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY stock, earnings_date ORDER BY announce_ts_ny
        ) = 1
        ORDER BY stock, earnings_date
    """).fetch_df()
    out["earnings_date"] = pd.to_datetime(out["earnings_date"])
    out["announce_ts_ny"] = pd.to_datetime(out["announce_ts_ny"])
    return out

def create_sectors_data_table_if_not_exists(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
        stock TEXT PRIMARY KEY,
        company_name TEXT,
        sector TEXT,
        sub_sector TEXT,
        status TEXT DEFAULT 'active',
        reason TEXT,
        ingested_at TIMESTAMP
    ); """)

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


def create_predictions_table_if_not_exists(con):
    """The calls we published, one row per stock per earnings event per run day.
    Scoped to one whole Mon-Fri work week (see save_predictions.py and
    work_week_window) — the same window the digest emails, so this is the product
    record rather than a dump of every scored event. Two week columns, easy to confuse:
      week_start — Monday of the week the COMPANY REPORTS (derived from earnings_date)
      run_week   — Monday of the week WE MADE THE CALL (derived from prediction_asof_date)
    Backtest against predictions_first_call (the view below); use this table directly
    only to study how a call drifted between the first call and the event.
    """
    con.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_asof_date    DATE,
            run_week                DATE,
            week_start              DATE,
            stock                   TEXT,
            earnings_date           DATE,
            tier                    TEXT,
            risk_score              DOUBLE,
            is_high_conviction      BOOLEAN,
            pre_earnings_drift_flag TEXT,
            surprise_momentum_flag  TEXT,
            model_version           TEXT,
            git_commit              TEXT,
            score_asof_date         DATE,
            ingested_at             TIMESTAMP
        );
    """)
    # Self-migration for DBs created before run_week existed. Needed because
    # scripts/sync_pipeline.sh overwrites the local DB with the droplet's copy, so a
    # schema change applied on only one side gets silently reverted on the next sync.
    con.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS run_week DATE")
    # score_asof_date: the observation date the tier was computed from. Added in the
    # Phase 1 event-frame rebuild so staleness is a recorded fact rather than something
    # a later audit has to infer. Rows written before it exists stay NULL — those are
    # the pre-rebuild calls, whose score came from the previous completed event.
    con.execute("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS score_asof_date DATE")
    con.execute("""
        UPDATE predictions
        SET run_week = date_trunc('week', prediction_asof_date)
        WHERE run_week IS NULL
    """)

    # One row per (stock, earnings_date) per run DAY. A same-day re-run upserts in place
    # (see save_predictions.py) so a re-score after a fix corrects the row rather than
    # being dropped; a run on a new date inserts a fresh row, preserving the trajectory.
    con.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS predictions_uq
        ON predictions(stock, earnings_date, prediction_asof_date);
    """)

    # The product-of-record: the FIRST call we ever made on each event. Query this for
    # hit rates — reading the raw table instead counts an event once per day it was
    # scored. Keyed on the event, not on run_week: a rolling work-week window can cover
    # the same event from two different run weeks (a Thursday run and the following
    # Monday both see an event 4 days out), which under the old (stock, earnings_date,
    # run_week) key produced two rows for one call.
    con.execute("DROP VIEW IF EXISTS predictions_week_open")
    con.execute("""
        CREATE OR REPLACE VIEW predictions_first_call AS
        SELECT DISTINCT ON (stock, earnings_date) *
        FROM predictions
        ORDER BY stock, earnings_date, prediction_asof_date;
    """)

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


def clean_duplicate_earnings_from_db(con, window_days=30):
    """Delete duplicate earnings rows from the DB where two dates for the same stock
    fall within window_days of each other and at least one has reported_eps (confirmed
    past event). Keeps the higher-quality row: prefers reported_eps not null, then
    later date. Pairs where both are NULL are left alone — those are upcoming/unconfirmed
    events handled by validate_upcoming_earnings_dates.
    """
    pairs = con.execute("""
        SELECT e1.stock,
               e1.earnings_date AS date1, e1.reported_eps AS eps1,
               e2.earnings_date AS date2, e2.reported_eps AS eps2
        FROM earnings e1
        JOIN earnings e2
          ON e1.stock = e2.stock
         AND e1.earnings_date < e2.earnings_date
         AND (e2.earnings_date - e1.earnings_date) <= ?
        WHERE e1.reported_eps IS NOT NULL OR e2.reported_eps IS NOT NULL
    """, [window_days]).fetchall()

    if not pairs:
        return

    to_delete = []
    for stock, date1, eps1, date2, eps2 in pairs:
        has1, has2 = eps1 is not None, eps2 is not None
        if has1 and not has2:
            to_delete.append((stock, date2))
        elif has2 and not has1:
            to_delete.append((stock, date1))
        else:
            to_delete.append((stock, date1))  # both confirmed — keep later date

    for stock, drop_date in to_delete:
        con.execute(
            "DELETE FROM earnings WHERE stock = ? AND earnings_date = ?",
            [stock, drop_date]
        )
    logger.info(f"clean_duplicate_earnings: removed {len(to_delete)} duplicate rows from DB.")


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


def join_dfs_by_stock_and_date(df, right_df, right_date_col, value_cols):
    """Join right_df onto df matching on stock AND date, not stock alone.

    Each row gets the most recent right_df entry dated on or before that row's own
    date; rows older than the earliest entry get NaN. Matching on stock alone (the
    previous behaviour) copied the newest values onto every historical row, so a 2008
    row carried 2026 data — lookahead that silently poisons any backtest on these
    columns.

    right_date_col is the date column in right_df; value_cols are the columns it
    contributes. Row order and index of df are preserved.
    """
    import pandas as pd

    if right_df.empty:
        for col in value_cols + [right_date_col]:
            df[col] = float("nan")
        return df

    right_df = right_df.copy()
    right_df[right_date_col] = pd.to_datetime(right_df[right_date_col])
    right_df["right_date"]   = right_df[right_date_col]
    right_df = right_df.sort_values("right_date", kind="mergesort")

    out = df.copy()
    out["left_date"] = pd.to_datetime(out["date"])
    out["row_order"] = range(len(out))
    out = out.sort_values("left_date", kind="mergesort")

    merged = pd.merge_asof(
        out, right_df,
        left_on="left_date", right_on="right_date",
        by="stock", direction="backward",
    )
    merged = merged.sort_values("row_order", kind="mergesort")
    merged = merged.drop(columns=["left_date", "row_order", "right_date"])
    merged.index = df.index
    return merged


def join_iv(df, con):
    """Join implied-volatility readings onto df by stock and date.

    NaN for rows dated before IV collection began.
    """
    iv_df = con.execute("""
        SELECT DISTINCT ON (stock, snapshot_date)
               stock, expected_move_pct, atm_iv,
               snapshot_date AS iv_snapshot_date
        FROM iv_snapshots
        ORDER BY stock, snapshot_date, snapshot_hour DESC
    """).fetch_df()
    return join_dfs_by_stock_and_date(
        df, iv_df, "iv_snapshot_date", ["expected_move_pct", "atm_iv"]
    )


def join_eps_estimates(df, con):
    """Join analyst EPS estimates onto df by stock and date.

    NaN for rows dated before estimate collection began.
    """
    eps_cols = [
        "eps_avg", "eps_high", "eps_low", "eps_num_analysts",
        "eps_dispersion", "eps_revision_momentum",
        "eps_trend_7d", "eps_trend_30d", "eps_trend_60d", "eps_trend_90d",
        "eps_revisions_up_7d", "eps_revisions_down_7d",
        "eps_revisions_up_30d", "eps_revisions_down_30d",
        "revenue_avg", "revenue_high", "revenue_low",
    ]
    eps_df = con.execute("""
        SELECT DISTINCT ON (stock, snapshot_date)
            stock, eps_avg, eps_high, eps_low, eps_num_analysts,
            eps_dispersion, eps_revision_momentum,
            eps_trend_7d, eps_trend_30d, eps_trend_60d, eps_trend_90d,
            eps_revisions_up_7d, eps_revisions_down_7d,
            eps_revisions_up_30d, eps_revisions_down_30d,
            revenue_avg, revenue_high, revenue_low,
            snapshot_date AS eps_snapshot_date
        FROM eps_estimates
        ORDER BY stock, snapshot_date
    """).fetch_df()
    return join_dfs_by_stock_and_date(df, eps_df, "eps_snapshot_date", eps_cols)

def verify_tables_existence(con):
    create_prices_table_if_not_exists(con)
    create_earnings_table_if_not_exists(con)
    create_sectors_data_table_if_not_exists(con)
    create_iv_table_if_not_exists(con)
    create_eps_estimates_table_if_not_exists(con)
    # create_predictions_table_if_not_exists is deliberately NOT called here: predictions
    # live in PREDICTIONS_DB_PATH, not this DB. save_predictions.py creates it there.
    print("DB Tables Set Up")


def incremental_of_full_ingestion_from_db(con, lookback_days):
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
    return prices_df, earnings_df
