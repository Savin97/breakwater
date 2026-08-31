# config.py

# Bump manually on scoring-logic changes; recorded on every predictions snapshot row
# alongside the git commit, so backtests can tell which model version produced a call.
#
# Pre-1.0 on purpose — the model is still being developed and nothing here is a frozen
# release. History, renumbered 2026-08-31 (earlier snapshot rows carry the OLD labels;
# use git_commit to place them, it is unambiguous):
#   0.1   — was labelled "1.0". Structural score + fixed 73/79 bucket cuts.
#   0.2   — was labelled "1.1". Adds lift-based tier promotion (commit f3dd1e2).
#   0.3.1 — is_high_conviction now carries the last completed event's bucket forward,
#           so HC is correct on pre-earnings rows instead of silently False.
MODEL_VERSION = "0.3.1"

STOCK_LIST_PATH = "data/stock_list.csv"
DB_PATH = "db/breakwater.duckdb"
# Predictions live in their OWN database on purpose. scripts/full_workflow.sh pulls the
# droplet's breakwater.duckdb and overwrites the local copy, so anything stored there is
# destroyed on the next weekly run; the droplet also has six cron writers a day, making
# pushing our copy back unsafe. This file is written only by us, and nothing syncs it.
PREDICTIONS_DB_PATH = "db/predictions.duckdb"
OUTPUT_PATH = "output/"

# Global Parameters
STOCKS_START_DATE = "2000-01-01"
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

# Bucket cut points on earnings_explosiveness_score. Chosen by OOS decile calibration
# (see testing/testing.py); (73, 79) minimises ECE across the 2011-2025 walk-forward.
# Also used as score floors so a lift-reclassified event cannot score below the tier
# it is labelled with.
BUCKET_ELEVATED_FLOOR   = 73
BUCKET_HIGH_ALERT_FLOOR = 79

# Lift-based tier reclassification. A stock whose own past earnings blow up far more
# often than the market baseline is treated as riskier than its structural score alone
# implies. Measured OOS 2015-2026: Normal events with lift >= 1.5 realise P(>=8%) = 0.238
# vs 0.058 for the Normal events left behind, and 0.250 for the genuine Elevated bucket —
# i.e. they behave like Elevated, so they are labelled Elevated.
# prior_strength shrinks thin per-stock samples toward the market baseline.
LIFT_PRIOR_STRENGTH     = 20
LIFT_TO_ELEVATED        = 1.5
LIFT_TO_HIGH_ALERT      = 3.0
EARNINGS_DATE_VALIDATION_WINDOW_DAYS = 20 # How far ahead to cross-check unconfirmed earnings dates against ticker.calendar
EARNINGS_RECHECK_WINDOW_DAYS = 14 # Stocks reporting within this many days are always re-fetched, so a wrong estimated date can still self-correct

# Logging
LOG_LEVEL = "INFO"  # "DEBUG" adds per-stock detail; "INFO" shows summaries only
# Third-party libraries that flood the output with their own logging.
# Each is capped at the level given, so our own LOG_LEVEL only affects our code.
NOISY_LIBRARIES = {
    "yfinance":            "WARNING",
    "urllib3":             "WARNING",
    "peewee":              "WARNING",
    "matplotlib":          "WARNING",
    "fontTools":           "WARNING",
    "weasyprint":          "ERROR",    # warns on every ignored CSS property
    "weasyprint.progress": "ERROR",    # logs each PDF build step
}

# API Parameters
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"
PRICES_PROVIDER = "ALPHAVANTAGE"
ALPHAVANTAGE_CALLS_PER_MINUTE=75
BACKOFF_SECONDS = 20.0
MAX_RETRIES = 5
DEFAULT_FETCH_CHUNK_SIZE = 50
YFINANCE_MAX_WORKERS = 8  # concurrent yfinance fetches in the earnings-date ingestion loops
# Random pause before each yfinance call so concurrent workers don't fire every request at once
YFINANCE_JITTER_MIN_SECONDS = 0.05
YFINANCE_JITTER_MAX_SECONDS = 0.15
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
    # Lift and the structural (pre-reclassification) bucket: both need
    # is_extreme_reaction, so they cannot be recomputed in the incremental window.
    "stock_bucket_lift", "earnings_explosiveness_bucket_structural",
]
INCREMENTAL_LOOKBACK_DAYS = 90  # warm-up buffer for longest rolling window (drift_60d = 60d)