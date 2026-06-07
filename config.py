# config.py
from datetime import date
STOCK_LIST_PATH = "data/stock_list.csv"
DB_PATH = "data/breakwater.duckdb"
OUTPUT_PATH = "output/"

# Global Parameters
STOCKS_START_DATE = "2000-01-01"
STOCKS_END_DATE = date.today().isoformat()
DEFAULT_REACTION_WINDOW = "reaction_3d" # Model will use 3 days after earnings
REACTION_THRESHOLD = 0.007
SHORT_TERM_DRIFT = 30 # 30 past days
LONG_TERM_DRIFT = 60 # 60 past days
SHORT_TERM_VOLATILITY = 10 # 10 past days
LONG_TERM_VOLATILITY = 30 # 30 past days
SHORT_TERM_MOMENTUM = 5 # 5 past days
LONG_TERM_MOMENTUM = 20 # 20 past days
LARGE_EARNINGS_REACTION_THRESHOLD = 0.05 # Based on 75th percentile of abs_reaction_3d
EXTREME_EARNINGS_REACTION_THRESHOLD = 0.08 # Based on 90th percentile of abs_reaction_3d

# API Parameters
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
PRICES_PROVIDER = "ALPHAVANTAGE"
ALPHAVANTAGE_CALLS_PER_MINUTE=75
BACKOFF_SECONDS = 20.0  
MAX_RETRIES = 5
DEFAULT_FETCH_CHUNK_SIZE = 50
CORRECT_STOCK_COL_NAME = "stock"
LIST_OF_POSSIBLE_STOCK_COL_NAMES = ["ticker", "Ticker", "Symbol", "symbol", "Stock", "stock"]

# Incremental pipeline — expanding stats that are stable between earnings events.
# Cached from full_df.parquet in incremental mode instead of recomputed.
INCREMENTAL_CACHED_COLS = [
    "abs_reaction_median", "abs_reaction_p75",
    "abs_reaction_p75_rolling", "abs_reaction_p90_rolling",
    "reaction_std", "reaction_entropy", "directional_bias",
    "surprise_streak", "surprise_mean_5", "surprise_std_5",
    "pre_earnings_drift_z",
    # Cached directly to avoid score drift when the most recent earnings event
    # has an incomplete reaction window (reaction_entropy=NaN).
    "earnings_explosiveness_score", "earnings_explosiveness_bucket",
]
INCREMENTAL_LOOKBACK_DAYS = 90  # warm-up buffer for longest rolling window (drift_60d = 60d)