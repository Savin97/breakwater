# config.py
import os

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
#           Legacy target (mismeasures before-open reporters); retired in 0.4.0.
#   0.4.0 — corrected-target Model B (scoring/model_b.py): causal mean prior anchored
#           earnings reaction; percentile score against a frozen reference
#           (scoring/model_b_reference.json); tiers are fixed percentile bands; no legacy
#           lift, promotions or High Conviction.
MODEL_VERSION = "0.4.0"
# Model versions whose archived calls may count toward a track record
# (db/predictions.duckdb, view predictions_track_record). Pre-0.4 calls are history only.
TRACK_RECORD_MODEL_VERSIONS = ("0.4.0",)

STOCK_LIST_PATH = "data/stock_list.csv"
DB_PATH = "db/breakwater.duckdb"
# Predictions live in their OWN database on purpose. scripts/full_workflow.sh pulls the
# droplet's breakwater.duckdb and overwrites the local copy, so anything stored there is
# destroyed on the next weekly run; the droplet also has six cron writers a day, making
# pushing our copy back unsafe. This file is written only by us, and nothing syncs it.
# BREAKWATER_PREDICTIONS_DB points a validation run at a scratch copy so it can never
# write the canonical archive.
PREDICTIONS_DB_PATH = os.environ.get("BREAKWATER_PREDICTIONS_DB", "db/predictions.duckdb")
OUTPUT_PATH = "output/"

# Global Parameters
STOCKS_START_DATE = "2000-01-01"
SHORT_TERM_VOLATILITY = 10 # 10 past days
LONG_TERM_VOLATILITY = 30 # 30 past days
# The Model B validation target: abs_reaction_3d_anchored >= this (scoring/model_b.py).
EXTREME_EARNINGS_REACTION_THRESHOLD = 0.08

# IV snapshots: the strike used as "at the money" must be within this fraction of the price.
# Yahoo sometimes returns a thin chain (a handful of strikes), and the nearest strike it
# offers can be far from the money — SPGI at $404 was recorded against a $445 strike.
IV_MAX_ATM_STRIKE_DISTANCE = 0.10
# IV snapshots: when the options chain comes back without the stock's live price, ask Yahoo
# for it separately this many times, waiting this long before each try, before skipping.
IV_LIVE_PRICE_RETRIES = 2
IV_LIVE_PRICE_RETRY_WAIT_SECS = 2.0
EARNINGS_DATE_VALIDATION_WINDOW_DAYS = 20 # How far ahead to cross-check unconfirmed earnings dates against ticker.calendar
EARNINGS_RECHECK_WINDOW_DAYS = 14 # Stocks reporting within this many days are always re-fetched, so a wrong estimated date can still self-correct
# A stock is normally skipped once its next earnings date is far out. That also stopped us
# ever collecting the RESULT of the event it just reported: yfinance publishes reported_eps
# within a day or two, but we would not ask again until ~14 days before the NEXT report,
# roughly 80 days later. Measured 2026-09-02: 100% of events under 30 days old had a NULL
# reported_eps, still ~80% at 31-90 days. So also re-fetch a stock whose most recent past
# event is still missing its result, bounded to this window — unbounded, a permanently
# missing old event would force a re-fetch of that stock on every run forever.
EARNINGS_RESULT_BACKFILL_DAYS = 30

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
