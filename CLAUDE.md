# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Continuity

At the start of every conversation, read `.claude/memory/MEMORY.md` to restore context from previous sessions. At the end of every conversation (or when asked), update it with what was done and what's next.

**Memory location:** All session memory lives in `.claude/memory/` inside this repo. It syncs via git across machines and is the single source of truth. Do NOT write to the Claude harness auto-memory location (`~/.claude/projects/*/memory/`) — that path is not used for this project and will diverge.

## External Brain

Durable cross-project knowledge lives at `/home/Michael/projects/brain`.

Use the external brain for source-backed research notes, durable decisions, project maps, and concepts that should persist across Breakwater, Polymarket, and musicology work. Keep `.claude/memory/` for Breakwater session continuity and immediate handoff notes.

Before doing architecture work, research synthesis, model-evaluation planning, or durable documentation, check `/home/Michael/projects/brain/projects/breakwater.md` and relevant notes under `/home/Michael/projects/brain/wiki/`.

## What This Is

Breakwater is an earnings tail-risk model for S&P 500 stocks. It ingests price/earnings/sector data, engineers features, scores each stock's upcoming earnings event on a risk scale, and produces reports and a Streamlit dashboard.

## Commands

```bash
# Run the full pipeline (see the incremental note under Stage 1 before running)
python main.py

# Re-score WITHOUT re-ingesting — no API calls, uses the existing DuckDB.
# This is what you want after changing anything in stage3/stage4.
python -c "from pipeline.stage2 import stage2; from pipeline.stage3 import stage3; \
from pipeline.stage4 import stage4; from pipeline.stage5 import stage5; stage5(stage4(stage3(stage2())))"

# Run backtesting suite (reads output/full_df.parquet)
python -m testing.backtesting

# Historical calibration tables (reads output/full_df.parquet)
python -m testing.calibration

# Launch the Streamlit dashboard
streamlit run streamlit_dash/app.py

# Ad-hoc feature/score testing
python -m testing.testing

# Run tests
pytest testing/test_pipeline.py
```

The project uses a `.venv` (Python 3.14). Activate with `source .venv/bin/activate` or prefix commands with `.venv/bin/python`.

## Pipeline Architecture

`main.py` → `pipeline/pipeline.py` → five sequential stages:

| Stage | File | What it does |
|---|---|---|
| 1 | `pipeline/stage1.py` | Creates/updates DuckDB at `db/breakwater.duckdb`. **Both settings ingest — the flag picks the provider.** `incremental=True` uses the free yfinance fetchers (`incremental_ingest_all_prices_yf`, `incremental_ingest_all_earnings_dates_yf`). `incremental=False` uses the legacy AlphaVantage full-history fetchers, which **require a paid AlphaVantage key**. `main.py` currently passes `incremental=False`. To re-score without ingesting at all, skip stage 1 entirely (see Commands). |
| 2 | `pipeline/stage2.py` | Reads `prices`, `earnings`, `stock_data` tables from DB; merges into a single DataFrame. Also dedups earnings and asserts data freshness. |
| 3 | `pipeline/stage3.py` | Calls ~20 feature-engineering functions in sequence; each appends columns and returns the df. `incremental=True` recomputes only price-dependent rolling features and reads expanding earnings stats from `config.INCREMENTAL_CACHED_COLS`. |
| 4 | `pipeline/stage4.py` | Calls the risk-scoring functions; produces `risk_score` (0–100), `earnings_explosiveness_bucket`, and component scores. `incremental=True` skips everything needing `abs_reaction_3d`. |
| 4b | `pipeline/events.py` | Builds the **event frame** — one row per earnings event, every completed event plus one pending row per stock — attaches verified announcement timing, and asserts completed-event parity against the daily frame. Written to `output/events_df.parquet`. |
| 5 | `pipeline/stage5.py` | Writes `output/full_df.parquet`, then generates PDF reports, the weekly calendar, charts, the public track record, the Streamlit export, and a predictions snapshot. |

The intermediate output between stages is a pandas DataFrame. Stage 3 and 4 functions all follow the same pattern: accept `input_df`, copy it, add columns, return it — never mutate in place.

## Data Storage

- **DuckDB** (`db/breakwater.duckdb`, gitignored): `prices (stock, date, price)`, `earnings (stock, earnings_date, fiscal_end_date, reported_eps, estimated_eps, surprise_percentage, announce_ts_ny, announce_ts_source)`, `stock_data (stock, company_name, sector, sub_sector, status, reason)`, plus `iv_snapshots` and `eps_estimates`.
- **Parquet** (`output/full_df.parquet`): the fully engineered + scored DataFrame, written at the top of stage 5. This is the source of truth for backtesting, calibration, and reporting.
- **Parquet** (`output/streamlit_df.parquet`, `output/upcoming_df.parquet`): produced by `streamlit_dash/streamlit_export.py`; consumed by the Streamlit app.
- **Parquet** (`output/events_df.parquet`, ~15 MB): the event frame. One row per earnings event; `is_pending == 0` is history, `is_pending == 1` is the upcoming call. Every forward-looking consumer reads this, never `groupby("stock").last()`.
- **Stock universe** (`data/stock_list.csv`): the list of stocks to process.

## Announcement Timing and the Corrected Target (Phase 2)

`reaction_1d/3d/5d` and `abs_reaction_3d` measure `close(D+k)/close(D)`, which assumes the
announcement lands after the close of the report date. That is true for AMC reporters and
**false for BMO reporters**, whose first post-announcement session is D itself. On 6,573
timestamped BMO events the measured P(|reaction| ≥ 8%) is 0.041; anchored correctly it is
0.173. See `audit/PHASE0_AUDIT_REV2.md` §Q1–Q2 and `audit/PHASE2_DIAGNOSTICS.md`.

**What this does and does not establish.** The legacy target is *proven wrong for BMO
events*. Nothing here establishes anything about the model: its validity and its
incremental value remain **unestablished**, pending the corrected-history rebuild (Phase 3)
and a competitive-baseline validation. Do not restate this finding as a claim about the
model being right.

**The legacy columns are unchanged and remain the production target.** Phase 2 adds a
*parallel* corrected target so the two can be compared on equal terms; nothing switches over
until the whole historical chain is rebuilt and the thresholds re-fit, which is Phase 3.

`feature_engineering/announcement_timing.py` owns this. Event-frame columns:

| column | meaning |
|---|---|
| `announce_ts_ny` | observed announcement time, naive **NY local** |
| `announce_ts_source` | provenance of that timestamp |
| `announce_ts_observed_at` | **when** the provider was observed saying it — the schedule-vs-observation flag |
| `announce_window` | `BMO` (<09:30) / `AMC` (≥16:00) / `INTRADAY` / `UNKNOWN` — a pure function of the clock |
| `anchor_date` | last close **strictly before** the announcement: AMC → close(D), BMO → close(D−1) |
| `anchor_status` | `resolved`, `pending`, or `unresolved_{no_timestamp,intraday,no_session,price_gap,no_prior_session,anchor_before_history}` |
| `anchor_session_status` | whether the report date is a session and this ticker has a row for it |
| `reaction_{1,3,5}d_anchored`, `abs_reaction_3d_anchored` | k post-announcement **market sessions** from the anchor |
| `reaction_{1,3,5}d_anchored_status` | per-horizon availability: `available`, or `unavailable_endpoint_{beyond_market_grid,after_last_price,price_gap}`, or the `anchor_status` when the anchor itself failed |

**Anchors and endpoints are positions on the market-session grid**
(`market_session_grid(daily_df)` — every date the loaded price data shows the market
trading), never positions in the ticker's own price rows, and never calendar arithmetic:

```
anchor session     = grid[i + offset]        i = the report date's grid index
k-session endpoint = grid[i + offset + k]    AMC offset 0, BMO offset −1
```

The ticker must then have a price row on those **exact** dates. Counting the ticker's own
rows silently absorbs a hole — `.shift(-3)` over a three-session gap yields a six-session
window, and a BMO event missing its D−1 row anchors to D−2 — and the arithmetic still
returns a number. The legacy columns do exactly this and are deliberately left alone; the
corrected target refuses and says which session it was missing. So an AMC anchored
reaction is bit-identical to the legacy one **wherever the ticker has a row on every
session in the window** (4,832 of 4,842 AMC events); the 10 that differ are all missing
sessions and are enumerated in `audit/PHASE2_DIAGNOSTICS.md` §5. That equality is the
control.

Rules that must not be relaxed:

- **Never infer BMO/AMC from realized price behavior.** Audit rev-1 did (ticker labelled BMO
  because its day-0 move exceeded its day-+1 move) and every "corrected" number it produced
  was circular. `test_6_the_classifier_never_touches_price` enforces this statically.
- **Never fabricate a timestamp.** The AlphaVantage date-only history stays NULL and its
  events stay unresolved. 25.0% of completed events are resolved; the rest are counted, not
  guessed at.
- **Never auto-roll a non-trading-day date.** A weekend/holiday date and a date the market
  traded but we failed to ingest have different causes; rolling hides both (§Q6).
- **Only `resolved_events()` may feed a corrected calibration**, and "anchor resolved" is
  not "target available". `anchor_resolved_events()` is the anchoring control slice;
  `resolved_events(events, target=...)` additionally requires that anchored outcome to be
  non-null and defaults to `abs_reaction_3d_anchored`. On the current data that is
  11,417 resolved anchors → 11,412 with a 3d target → 11,410 also carrying the legacy
  column for a paired comparison; every step is accounted for in the diagnostics §3.
- **A pre-event timestamp is a schedule, not a record.** A timestamp observed while the
  event was still upcoming may be corrected later, so ingestion refreshes it when a newer
  observation arrives (`refresh_announcement_timestamp`). A timestamp observed *after* the
  announcement is never overwritten. `announce_ts_observed_at` is what tells the two
  apart; where it is NULL, `ingested_at` stands in as a lower bound, and where both are
  NULL nothing is refreshed.
- `audit/provider_timestamps.parquet` is **evidence, not a runtime input**. It seeded
  `earnings.announce_ts_ny` once via `scripts/backfill_announcement_timestamps.py`;
  ingestion keeps the column current from there. A test asserts no `pipeline/` module reads it.

```bash
# one-time seed of the audit timestamps into the DB (idempotent)
PYTHONPATH=. .venv/bin/python scripts/backfill_announcement_timestamps.py [--dry-run]

# dataset diagnostics — window mix, coverage by year, legacy vs anchored, unresolved reasons
PYTHONPATH=. .venv/bin/python -m audit.phase2_diagnostics
```

## Feature Engineering Conventions

All rolling stats use `.shift(1)` to prevent leakage. Target variable is `abs_reaction_3d` — the absolute 3-day post-earnings price move.

Key engineered columns:
- `daily_ret`: daily pct change
- `drift_30d / drift_60d`: rolling mean of daily_ret
- `vol_10d / vol_30d`: rolling std; `vol_ratio_10_to_30` for stress detection
- `mom_5d / mom_20d`: rolling sum of daily_ret
- `days_to_earnings`, `is_earnings_day`, `is_earnings_week`, `is_earnings_window`
- `reaction_1d / 3d / 5d`: price move 1/3/5 days after earnings
- `reaction_std`, `reaction_entropy`, `directional_bias`: historical reaction distribution stats
- `sector_drift_60d`, `sector_vol_10d/30d`, `stock_vs_sector_vol`, `sector_earnings_density`

## Risk Scoring

Stage 4 produces these component scores (all in `scoring/scoring_features.py`):
- `proximity_score`: how close the stock is to earnings
- `vol_expansion_score`: vol expansion relative to baseline
- `momentum_fragility_score`: momentum divergence signal
- `earnings_explosiveness_score`: historical tail-risk profile score (0–100)
- `stock_bucket_lift`: P(extreme | stock, bucket) / P(extreme | market), computed causally from prior events only and shrunk toward the market baseline
- `earnings_explosiveness_bucket`: `Normal` / `Elevated` / `High Alert` — cut from the score, then promoted where `stock_bucket_lift` clears its threshold
- `earnings_explosiveness_bucket_structural`: the pre-promotion bucket, kept for analysis
- `risk_score`: currently a pass-through of `earnings_explosiveness_score`
- `is_high_conviction`: `High Alert` **and** a non-empty `pre_earnings_drift_flag`

**The tier is the product, not the score.** The bucket is decided by two inputs (structural
score + lift) while `risk_score` carries only the first, so a lift-promoted event can sit in a
higher tier than a higher-scoring event. That is deliberate: it means "mild structural profile,
violent personal history." Do not "fix" it by flooring the score to the tier boundary or by
multiplying the score by the lift — the latter was measured and drops top-decile lift from
3.70x to 2.98x, because lift is ~0.79 rank-correlated with the score and corrupts its ordering
when blended. As a conditional gate the same signal is strongly additive: capture of ≥8% moves
goes 43.9% → 57.0% with `High Alert` purity unchanged at 0.409.

Thresholds are in `config.py`: `LARGE_EARNINGS_REACTION_THRESHOLD = 0.05`,
`EXTREME_EARNINGS_REACTION_THRESHOLD = 0.08`, bucket cut points
`BUCKET_ELEVATED_FLOOR = 73` / `BUCKET_HIGH_ALERT_FLOOR = 79`, and promotion gates
`LIFT_TO_ELEVATED = 1.5` / `LIFT_TO_HIGH_ALERT = 3.0` with `LIFT_PRIOR_STRENGTH = 20`.

## Key Configuration (`config.py`)

- `DB_PATH`: path to DuckDB file
- `STOCKS_START_DATE / STOCKS_END_DATE`: date range for price data
- `DEFAULT_REACTION_WINDOW`: `"reaction_3d"` — the primary reaction metric
- `PRICES_PROVIDER`: `"ALPHAVANTAGE"` — only used by the legacy `incremental=False` ingestion path, which needs a paid key. The yfinance path (`incremental=True`) ignores it and is the one in routine use; its knobs are `YFINANCE_MAX_WORKERS` and the jitter settings.
- `INCREMENTAL_CACHED_COLS`: expanding earnings stats read from the previous `full_df.parquet` in incremental mode instead of recomputed. Anything needing `abs_reaction_3d` must be listed here, or it will be silently missing on incremental runs.

## Backtesting

`testing/backtesting.py` — standalone module, reads `output/full_df.parquet` directly. The `backtesting_suite()` function runs calibration, lift, hit rates, and year-by-year OOS checks. `testing/testing_functions.py` contains all the individual test helpers.

`testing/calibration.py` — the acceptance gate for any scoring change. Produces tier hit rates
with 95% Wilson CIs, capture rate, percentile bands, per-tier year-by-year stability, and a
score/bucket consistency check. Use it to **compare two variants on equal terms**, not to certify
an absolute hit rate: the 73/79 cut points were themselves selected on this same window, so
absolute numbers are optimistically biased.

The train/test split convention used in `testing/testing.py`: pre-2015 = train, post-2015 = OOS test.
