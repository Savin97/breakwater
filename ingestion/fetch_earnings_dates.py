# ingestion/fetch_earnings_dates.py
import time, random, logging, pandas as pd, yfinance as yf
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from utilities.db_utilities import get_max_dates_by_stock
from utilities.api_functions import (get_earnings_data_from_api)
from utilities.data_utilities import to_float_or_none, get_alpha_vantage_api_key, read_stocks_to_fetch
from utilities.output_utilities import get_run_logs_dir
from utilities.time_utilities import now_ny, MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS
from config import (
    STOCKS_START_DATE,
    ALPHAVANTAGE_CALLS_PER_MINUTE,
    EARNINGS_DATE_VALIDATION_WINDOW_DAYS,
    EARNINGS_RECHECK_WINDOW_DAYS,
    EARNINGS_RESULT_BACKFILL_DAYS,
    YFINANCE_MAX_WORKERS,
    YFINANCE_JITTER_MIN_SECONDS,
    YFINANCE_JITTER_MAX_SECONDS,
)
import os

logger = logging.getLogger(__name__)

# Physical column order of the `earnings` table. Both writers below name their columns
# explicitly rather than `SELECT *`, so adding a column (Phase 2 added two) can never
# silently shift values into the wrong slot.
EARNINGS_INSERT_COLS = ["stock", "earnings_date", "fiscal_end_date",
                        "reported_eps", "estimated_eps", "surprise_percentage",
                        "ingested_at", "announce_ts_ny", "announce_ts_source",
                        "announce_ts_observed_at"]
_INSERT_COL_SQL = ", ".join(EARNINGS_INSERT_COLS)

ANNOUNCE_TS_SOURCE_YFINANCE = "yfinance_earnings_dates"

# A stored announcement timestamp is REFRESHABLE while it is still a schedule and frozen
# once it has been observed after the fact.
#
#   observed_at <= announce_ts_ny   the provider told us this BEFORE the announcement.
#                                   That is a SCHEDULE. Issuers move them, and providers
#                                   correct them, so it must not become the permanent
#                                   historical record.
#   observed_at >  announce_ts_ny   the provider told us this AFTER the announcement. It
#                                   is an observation of what happened and is never
#                                   overwritten.
#
# Both sides of that comparison are naive NEW YORK wall clock (utilities.time_utilities).
# The classification is a statement about the New York event clock and must not change
# with the timezone of the host running the pipeline: `datetime.now()` on a UTC or
# Israeli box reads hours ahead of NY, which would make a pre-event observation look
# post-event and freeze a schedule into the historical record forever.
#
# `ingested_at` stands in where `announce_ts_observed_at` is NULL (rows written before
# the column existed). It is a legacy operational column written in MACHINE-LOCAL time by
# whichever host ran that ingestion, and its convention is not recorded, so it is never
# compared against announce_ts_ny directly. It is first widened into a guaranteed lower
# bound on the same instant in NY terms — a host clock can lead New York by at most
# MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS — so the row is frozen only when it was observed after
# the announcement under EVERY possible host timezone. The residual error therefore only
# ever goes one way: an ambiguous legacy row can be treated as a schedule when it was
# really an observation, and is then replaced by a strictly newer, correctly-stamped
# observation of the same event. That is the conservative direction. The opposite error —
# a false post-event classification — is the one that is unrecoverable, because it welds a
# time that never happened onto the historical record. If both columns are NULL the
# comparison is NULL, the row does not match, and nothing is overwritten.
#
# The refresh also requires the incoming observation to be strictly newer than the one it
# replaces, so re-running ingestion is idempotent and an older observation can never
# clobber a newer one. Against a legacy row that bound is the widened one too, for the
# same reason; a row only takes the widened path once, because the refresh writes a real
# NY-convention `announce_ts_observed_at` and every later comparison is exact.

# Naive NY wall clock if we have it; otherwise the legacy machine-local `ingested_at`
# widened into a lower bound that holds for any host timezone.
_OBSERVED_AT_NY = (f"COALESCE(announce_ts_observed_at, "
                   f"ingested_at - INTERVAL {MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS} HOUR)")

_REFRESH_ANNOUNCE_TS_SQL = f"""
    UPDATE earnings
       SET announce_ts_ny = ?,
           announce_ts_source = ?,
           announce_ts_observed_at = ?
     WHERE stock = ? AND earnings_date = ?
       AND (announce_ts_ny IS NULL
            OR ({_OBSERVED_AT_NY} <= announce_ts_ny
                AND ? > {_OBSERVED_AT_NY}))
"""


def refresh_announcement_timestamp(con, stock, earnings_date, announce_ts_ny,
                                   source, observed_at) -> int:
    """Store an observed announcement timestamp, refreshing a stale pre-event schedule.

    Returns the number of rows changed. See `_REFRESH_ANNOUNCE_TS_SQL` for the rule; the
    short version is fill-if-empty, replace-if-still-a-schedule, never touch a
    post-event observation.

    `announce_ts_ny` and `observed_at` are both naive NEW YORK wall clock; pass
    `utilities.time_utilities.now_ny()` for an observation made right now, never
    `datetime.now()`.

    Keyed on (stock, earnings_date), so it corrects the TIME of an event whose calendar
    date is unchanged. A provider correction that moves the date itself is a different
    row and is handled by the placeholder-clearing DELETE in the caller.
    """
    def _py(v):
        return v.to_pydatetime() if hasattr(v, "to_pydatetime") else v

    ts, obs = _py(announce_ts_ny), _py(observed_at)
    if ts is None or pd.isna(ts):
        return 0
    changed = con.execute(_REFRESH_ANNOUNCE_TS_SQL,
                          [ts, source, obs, stock, earnings_date, obs]).fetchone()
    return int(changed[0]) if changed else 0

def fetch_one_earnings_dates(stock: str):
    """Network fetch + pandas reshape only — no DB access. Mirrors the fetch/reshape
    portion of incremental_ingest_all_earnings_dates_yf's loop body exactly.

    Returns {"stock": ..., "earnings_dates_df": ..., "error": ...}:
      - got data:     earnings_dates_df = dataframe, error = None
      - no data:      earnings_dates_df = None,      error = None       — not logged, matches current behavior
      - fetch failed: earnings_dates_df = None,      error = exception   — logged, matches current behavior
    """
    try:
        time.sleep(random.uniform(YFINANCE_JITTER_MIN_SECONDS, YFINANCE_JITTER_MAX_SECONDS))
        earnings_dates_df = yf.Ticker(stock).earnings_dates
        if earnings_dates_df is None or earnings_dates_df.empty:
            return {"stock": stock, "earnings_dates_df": None, "error": None}

        earnings_dates_df = earnings_dates_df.reset_index()
        earnings_dates_df = earnings_dates_df.rename(columns={
            "Earnings Date":  "earnings_date",
            "EPS Estimate":   "estimated_eps",
            "Reported EPS":   "reported_eps",
            "Surprise(%)":    "surprise_percentage",
        })
        # yfinance's index is tz-aware America/New_York and carries the ANNOUNCEMENT TIME.
        # Phase 2: keep it. Discarding it with .dt.date is what forced every timing
        # analysis to infer BMO/AMC from price behavior, which is circular
        # (audit/PHASE0_AUDIT_REV2.md Q1). The calendar date is still stored exactly as
        # before — announce_ts_ny is additive, nothing downstream of earnings_date moves.
        announce_ts_ny = (
            pd.to_datetime(earnings_dates_df["earnings_date"])
            .dt.tz_localize(None)          # tz-aware -> NY LOCAL wall clock, kept verbatim
        )
        earnings_dates_df["announce_ts_ny"]      = announce_ts_ny
        earnings_dates_df["announce_ts_source"]  = ANNOUNCE_TS_SOURCE_YFINANCE
        earnings_dates_df["earnings_date"]       = announce_ts_ny.dt.date
        earnings_dates_df["stock"]               = stock
        earnings_dates_df["fiscal_end_date"]     = None
        earnings_dates_df["surprise_percentage"] = earnings_dates_df["surprise_percentage"] / 100
        earnings_dates_df["ingested_at"]         = datetime.now()
        # WHEN the provider was observed saying this. A row fetched while the event is
        # still upcoming is a schedule; one fetched afterwards is an observation. Only
        # this column can tell them apart later, so it is written at the moment of the
        # fetch and never inferred.
        #
        # In naive NEW YORK wall clock, the same convention as `announce_ts_ny`, because
        # the refresh rule compares the two directly. `datetime.now()` would put the
        # host's timezone into that comparison: on a UTC or Israeli box a schedule
        # fetched hours before the announcement would read as later than it and be frozen
        # into the historical record. `ingested_at` above keeps its own legacy
        # machine-local convention — it is an operational column, not part of the
        # announcement-timing comparison.
        earnings_dates_df["announce_ts_observed_at"] = now_ny()
        earnings_dates_df.loc[earnings_dates_df["announce_ts_ny"].isna(),
                              "announce_ts_observed_at"] = pd.NaT
        earnings_dates_df = earnings_dates_df[EARNINGS_INSERT_COLS]

        return {"stock": stock, "earnings_dates_df": earnings_dates_df, "error": None}
    except Exception as e:
        return {"stock": stock, "earnings_dates_df": None, "error": e}


def ingest_all_earnings_dates(con):
    already, inserted, failed = 0,0,0
    FAILED_EARNINGS_LOG_PATH = os.path.join(get_run_logs_dir(), "debug_failed_earnings_ingestion.txt")
    API_KEY = get_alpha_vantage_api_key()
    min_sleep = 60.0 / float(ALPHAVANTAGE_CALLS_PER_MINUTE)
    stocks = read_stocks_to_fetch()
    cutoff = pd.to_datetime(STOCKS_START_DATE).date()

    if not API_KEY:
        raise RuntimeError("Set ALPHAVANTAGE_API_KEY env var first.")
    # reset failure log each run (simple)
    with open(FAILED_EARNINGS_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("stock\terror\n")

    # cache current max earnings_date per stock
    max_earnings_date_by_stock = get_max_dates_by_stock(con, "earnings", "earnings_date")

    # heuristic freshness window (quarterly): if you already have something in last 90 days, skip
    today = datetime.now().date()
    fresh_window_days = 90

    for i, stock in enumerate(stocks, start=1):  
        stock_earn_max_date = max_earnings_date_by_stock.get(stock)

        if stock_earn_max_date is not None and (today - stock_earn_max_date).days <= fresh_window_days: # type: ignore
            already += 1
            print(f"{stock} is up to date")
            if i % 50 == 0:
                print(f"[{i}/{len(stocks)}] skipped(fresh): {already}, inserted: {inserted}, failed: {failed}")
            continue

        data = get_earnings_data_from_api(stock)    
        print(f"[{i}/{len(stocks)}] Fetching earnings data for {stock}...")
        try:
            if "quarterlyEarnings" not in data:
                raise RuntimeError(f"Bad payload keys: {list(data.keys())}. Snippet: {str(data)[:180]}")
            quarterly_earnings = data["quarterlyEarnings"]
            table_cols = ["stock", "reportedDate", "fiscalDateEnding", "reportedEPS", "estimatedEPS", "surprisePercentage"]
            rows = []   
            
            for quarter in quarterly_earnings:
                rows.append((stock, 
                             quarter["reportedDate"], 
                             quarter["fiscalDateEnding"], 
                             to_float_or_none(quarter["reportedEPS"]), 
                             to_float_or_none(quarter["estimatedEPS"]), 
                             to_float_or_none(quarter["surprisePercentage"])) )

            df = pd.DataFrame(rows, columns=table_cols)
            df = df.rename(columns={
                "reportedDate": "earnings_date",
                "fiscalDateEnding": "fiscal_end_date",
                "reportedEPS": "reported_eps",
                "estimatedEPS": "estimated_eps",
                "surprisePercentage": "surprise_percentage"
            })
            df["earnings_date"] = pd.to_datetime(df["earnings_date"]).dt.date
            df["fiscal_end_date"] = pd.to_datetime(df["fiscal_end_date"]).dt.date
            
            df = df[df["earnings_date"] >= cutoff]
            df = df[df["fiscal_end_date"] >= cutoff] 

            if df.empty:
                already += 1
                time.sleep(min_sleep)
                continue

            df["surprise_percentage"] = df["surprise_percentage"] / 100
            df["ingested_at"] = datetime.now()
            # AlphaVantage returns a reported DATE and no time. Left NULL — inventing a
            # plausible hour here would put a fabricated timestamp behind a column whose
            # whole purpose is that it was independently observed.
            df["announce_ts_ny"] = pd.NaT
            df["announce_ts_source"] = None
            df["announce_ts_observed_at"] = pd.NaT
            df = df[EARNINGS_INSERT_COLS]

            count_before = con.execute("SELECT COUNT(*) FROM earnings WHERE stock = ?", [stock]).fetchone()[0] #type:ignore
            con.register("tmp_earnings_df", df)
            con.execute(f"INSERT OR IGNORE INTO earnings ({_INSERT_COL_SQL}) SELECT {_INSERT_COL_SQL} FROM tmp_earnings_df")
            con.unregister("tmp_earnings_df")
            count_after = con.execute("SELECT COUNT(*) FROM earnings WHERE stock = ?", [stock]).fetchone()[0]#type:ignore
            added = count_after - count_before

            min_date, max_date = con.execute("""SELECT MIN(earnings_date), MAX(earnings_date) FROM earnings WHERE stock = ?;""", [stock]).fetchone() #type:ignore
            print(f"Added {added} rows ({min_date} -> {max_date})")
            if added != 0:
                inserted += 1
            print(f"Inserted {count_after} rows ({min_date} -> {max_date})")

        except Exception as e:
            failed += 1
            err = f"{type(e).__name__}: {e}"
            print(f"  FAILED {stock}: {err}")
            # ensure tmp view not left behind
            try:
                con.unregister("tmp_earnings_df")
            except Exception:
                pass
            with open(FAILED_EARNINGS_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"{stock}\t{err}\n")
        # always sleep a bit to respect rate limits
        time.sleep(min_sleep)
    
    print("\nIngesting Earnings Done.")
    print("already in DB:", already)
    print("inserted new:", inserted)
    print("failed:", failed)
    print("Failures saved to:", FAILED_EARNINGS_LOG_PATH)


def incremental_ingest_all_earnings_dates_yf(con):
    """
        Incremental earnings update using yfinance (no API key required).
        Fetches ~12 recent quarters + upcoming dates per stock.
        Skips stocks that already have a future earnings date in the DB.

        Fetches run concurrently (YFINANCE_MAX_WORKERS); DB writes stay sequential
        and in stock order, since the connection isn't safe to share across threads.
    """
    stocks = read_stocks_to_fetch(con, active_only=True)
    already, inserted, failed = 0, 0, 0
    FAILED_LOG_PATH = os.path.join(get_run_logs_dir(), "debug_failed_earnings_ingestion.txt")

    with open(FAILED_LOG_PATH, "w") as f:
        f.write("stock\terror\n")

    # Skip stocks whose stored upcoming date is still comfortably far out.
    # Stocks due soon (or whose stored date just passed) are always rechecked,
    # since a stale placeholder estimate can otherwise never self-correct
    # before its own (wrong) date arrives.
    # ...but never skip a stock whose most recent event is still missing its result.
    # The event has happened, yfinance has the reported EPS within a day or two, and
    # without this the stock is not asked again until ~14 days before its NEXT report.
    stocks_to_skip = set(
        row[0] for row in
        con.execute(
            "SELECT DISTINCT stock FROM earnings "
            "WHERE earnings_date > current_date + CAST(? AS INTEGER) "
            "  AND stock NOT IN ("
            "        SELECT stock FROM earnings "
            "        WHERE reported_eps IS NULL "
            "          AND earnings_date <= current_date "
            "          AND earnings_date >= current_date - CAST(? AS INTEGER))",
            [EARNINGS_RECHECK_WINDOW_DAYS, EARNINGS_RESULT_BACKFILL_DAYS]
        ).fetchall()
    )

    # Fetch every stock that needs one up front, several at a time. Only the network
    # call and pandas reshape happen here — no DB access, so nothing touches `con`
    # from a worker thread. All the writes below stay sequential, in stock order.
    stocks_to_fetch = [stock for stock in stocks if stock not in stocks_to_skip]
    fetch_results = {}
    with ThreadPoolExecutor(max_workers=YFINANCE_MAX_WORKERS) as pool:
        pending = {pool.submit(fetch_one_earnings_dates, stock): stock for stock in stocks_to_fetch}
        for future in as_completed(pending):
            fetch_results[pending[future]] = future.result()

    for i, stock in enumerate(stocks, start=1):
        if stock in stocks_to_skip:
            already += 1
            if i % 100 == 0:
                logger.debug(f"[{i}/{len(stocks)}] skipped: {already}, inserted: {inserted}, failed: {failed}")
            continue

        result = fetch_results[stock]
        earnings_dates_df = result["earnings_dates_df"]
        if earnings_dates_df is None:
            failed += 1
            if result["error"] is not None:
                err = f"{type(result['error']).__name__}: {result['error']}"
                logger.error(f"FAILED {stock}: {err}")
                with open(FAILED_LOG_PATH, "a") as f:
                    f.write(f"{stock}\t{err}\n")
            continue

        try:
            # Backfill results onto events we already store. The filter below drops any
            # date already in the DB and the write below is a plain INSERT, so a row held
            # as an unconfirmed placeholder would keep its NULL reported_eps until the
            # date itself changed — which is why 100% of events under 30 days old had no
            # result. Only fills NULLs; a confirmed row is never overwritten.
            confirmed = earnings_dates_df[earnings_dates_df["reported_eps"].notna()]
            for row in confirmed.itertuples(index=False):
                con.execute("""
                    UPDATE earnings
                       SET reported_eps = ?, estimated_eps = ?, surprise_percentage = ?
                     WHERE stock = ? AND earnings_date = ? AND reported_eps IS NULL
                """, [row.reported_eps, row.estimated_eps, row.surprise_percentage,
                      stock, row.earnings_date])

            # Store the announcement timestamp on rows we already hold. The dedup filter
            # below drops any date already in the DB, so without this an event ingested
            # before Phase 2 would stay timestamp-less forever and remain permanently
            # unresolved.
            #
            # This used to fill NULLs only, which froze a pre-event SCHEDULE into the
            # historical record permanently: a date fetched while the event was upcoming
            # kept whatever hour the provider had pencilled in, even after the provider
            # published the real one. `refresh_announcement_timestamp` replaces a stored
            # schedule with a later observation and still never overwrites a timestamp
            # that was already observed after the fact.
            for row in earnings_dates_df.itertuples(index=False):
                if pd.isna(row.announce_ts_ny):
                    continue
                refresh_announcement_timestamp(
                    con, stock, row.earnings_date, row.announce_ts_ny,
                    row.announce_ts_source, row.announce_ts_observed_at)

            # fiscal_end_date is None so the DB unique index can't deduplicate — filter manually
            existing = {
                row[0] for row in
                con.execute("SELECT earnings_date FROM earnings WHERE stock = ?", [stock]).fetchall()
            }
            earnings_dates_df = earnings_dates_df[~earnings_dates_df["earnings_date"].isin(existing)]
            if earnings_dates_df.empty:
                already += 1
                continue

            # For each freshly-fetched date (past or future), remove any unconfirmed
            # placeholder row within ±60 days — this is what clears a stale estimate
            # once the real (confirmed or corrected) date is known. Not restricted to
            # >= today: a just-reported date is exactly the anchor needed to clear an
            # old placeholder that was estimated too late.
            for new_date in earnings_dates_df["earnings_date"]:
                con.execute("""
                    DELETE FROM earnings
                    WHERE stock = ?
                      AND reported_eps IS NULL
                      AND ABS(DATEDIFF('day', earnings_date, ?)) <= 60
                """, [stock, new_date])

            count_before = con.execute(
                "SELECT COUNT(*) FROM earnings WHERE stock = ?", [stock]
            ).fetchone()[0]
            con.register("tmp_earnings_df", earnings_dates_df)
            con.execute(f"INSERT INTO earnings ({_INSERT_COL_SQL}) SELECT {_INSERT_COL_SQL} FROM tmp_earnings_df")
            con.unregister("tmp_earnings_df")
            count_after = con.execute(
                "SELECT COUNT(*) FROM earnings WHERE stock = ?", [stock]
            ).fetchone()[0]

            added = count_after - count_before
            if added > 0:
                inserted += 1
                logger.debug(f"[{i}/{len(stocks)}] {stock}: +{added} rows")
            else:
                already += 1

        except Exception as e:
            failed += 1
            err = f"{type(e).__name__}: {e}"
            logger.error(f"FAILED {stock}: {err}")
            try:
                con.unregister("tmp_earnings_df")
            except Exception:
                pass
            with open(FAILED_LOG_PATH, "a") as f:
                f.write(f"{stock}\t{err}\n")

    logger.info(f"Ingesting Earnings Done (yfinance). skipped/up-to-date: {already}, inserted: {inserted}, failed: {failed}")

def fetch_upcoming_earnings_date(stock: str, today):
    """Network fetch + parsing only — no DB access. Reads the company's IR calendar
    and returns just the nearest confirmed upcoming earnings date.

    Returns {"stock": ..., "upcoming_earnings_date": ..., "error": ...}:
      - got a date:     upcoming_earnings_date = date, error = None
      - no usable date: upcoming_earnings_date = None, error = None      — counted as a warning, not logged
      - fetch failed:   upcoming_earnings_date = None, error = exception  — logged, matches current behavior
    """
    try:
        time.sleep(random.uniform(YFINANCE_JITTER_MIN_SECONDS, YFINANCE_JITTER_MAX_SECONDS))
        calendar = yf.Ticker(stock).calendar
        if calendar is None:
            return {"stock": stock, "upcoming_earnings_date": None, "error": None}

        # calendar returns a dict {field: value_or_list} in yfinance 1.x
        if isinstance(calendar, dict):
            raw_dates = calendar.get("Earnings Date", [])
            if not isinstance(raw_dates, (list, tuple)):
                raw_dates = [raw_dates] if raw_dates is not None else []
        elif isinstance(calendar, pd.DataFrame):
            if "Earnings Date" in calendar.columns:
                raw_dates = calendar["Earnings Date"].dropna().tolist()
            else:
                raw_dates = []
        else:
            raw_dates = []

        calendar_dates = []
        for raw_date in raw_dates:
            try:
                calendar_dates.append(pd.Timestamp(raw_date).date())
            except Exception:
                pass

        future_dates = [d for d in calendar_dates if d >= today]
        if not future_dates:
            return {"stock": stock, "upcoming_earnings_date": None, "error": None}

        return {"stock": stock, "upcoming_earnings_date": min(future_dates), "error": None}
    except Exception as e:
        return {"stock": stock, "upcoming_earnings_date": None, "error": e}


def validate_upcoming_earnings_dates(con, days_ahead=EARNINGS_DATE_VALIDATION_WINDOW_DAYS, max_delta_days=30):
    """
        Cross-check upcoming unconfirmed earnings dates against ticker.calendar (company IR data).
        Corrects any date that differs by more than 1 day from the confirmed calendar date.
        days_ahead: only check dates within this many days out — dates further out get
        re-checked on a later run as they approach, so nothing is permanently skipped.
        max_delta_days: skip corrections where the calendar date is more than this many days out
        from the DB date — prevents next-quarter rollover false corrections.
        Called from pipeline/stage1.py after incremental_ingest_all_earnings_dates_yf.

        Calendar fetches run concurrently (YFINANCE_MAX_WORKERS), one per unique stock;
        the date comparison and any UPDATE stay sequential, in the original row order.
    """
    today = datetime.now().date()

    rows = con.execute("""
        SELECT DISTINCT stock, earnings_date
        FROM earnings
        WHERE earnings_date >= CURRENT_DATE
          AND earnings_date <= CURRENT_DATE + CAST(? AS INTEGER)
          AND reported_eps IS NULL
        ORDER BY stock, earnings_date
    """, [days_ahead]).fetchall()

    if not rows:
        logger.info("validate_upcoming_earnings_dates: no upcoming unconfirmed dates to check.")
        return {"corrected": [], "warnings": []}

    logger.info(f"Validating {len(rows)} upcoming earnings dates (within {days_ahead} days) via ticker.calendar...")
    corrected = []
    warnings_list = []

    # One calendar fetch per stock, several at a time. A stock can appear on more than
    # one row here (multiple upcoming unconfirmed dates), so fetching per unique stock
    # also avoids repeat calls the old row-by-row loop was making.
    unique_stocks = sorted({stock for stock, _ in rows})
    fetch_results = {}
    with ThreadPoolExecutor(max_workers=YFINANCE_MAX_WORKERS) as pool:
        pending = {pool.submit(fetch_upcoming_earnings_date, stock, today): stock for stock in unique_stocks}
        for future in as_completed(pending):
            fetch_results[pending[future]] = future.result()

    for stock, db_date in rows:
        if isinstance(db_date, pd.Timestamp):
            db_date = db_date.date()

        result = fetch_results[stock]
        if result["error"] is not None:
            warnings_list.append(stock)
            logger.warning(f"{stock}: {type(result['error']).__name__}: {result['error']}")
            continue

        calendar_date = result["upcoming_earnings_date"]
        if calendar_date is None:
            warnings_list.append(stock)
            continue

        diff = abs((calendar_date - db_date).days)

        if diff > 0:
            if diff > max_delta_days:
                warnings_list.append(stock)
                logger.warning(f"SKIPPED {stock}: {db_date} → {calendar_date} ({diff:+d} days, exceeds max_delta={max_delta_days} — possible quarter rollover)")
            else:
                con.execute("""
                    UPDATE earnings
                    SET earnings_date = ?
                    WHERE stock = ? AND earnings_date = ? AND reported_eps IS NULL
                """, [calendar_date, stock, db_date])
                logger.info(f"CORRECTED {stock}: {db_date} → {calendar_date} ({diff:+d} days)")
                corrected.append((stock, db_date, calendar_date))

    logger.info(f"Validation done: {len(corrected)} corrected, {len(warnings_list)} no-calendar warnings.")
    return {"corrected": corrected, "warnings": warnings_list}


def get_next_earnings_dates():
    stocks = read_stocks_to_fetch()
    today = pd.Timestamp(datetime.now() ,tz="America/New_York")
    stock_dict = {}
    for i,stock in enumerate(stocks,start=1):
        if i%100==0:
            time.sleep(30)
        print(f"[{i}/{len(stocks)}] Fetching {stock} next Earnings Date...")
        try:
            ticker = yf.Ticker(stock)
            if ticker is not None:
                df = ticker.get_earnings_dates(limit=1, offset=0)
            else:
                raise ValueError(f"{ticker} returned None")
            if df is not None:
                df = df.reset_index()
                edates = pd.to_datetime(df["Earnings Date"])
            else:
                raise ValueError("Nothing from yfinance")
            for date in edates:
                if date >= today:
                    stock_dict[stock] = date.date()
        except Exception as e:
            print("ERROR - ", e)
            break
    next_earnings_df = pd.DataFrame(stock_dict.items(), columns=["stock","earnings_date"])
    next_earnings_df.to_csv("next_earnings_df.csv",index=False)
    print("DF created - next_earnings_df.csv")
    return next_earnings_df