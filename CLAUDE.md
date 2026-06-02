# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Continuity

At the start of every conversation, read `.claude/memory/MEMORY.md` to restore context from previous sessions. At the end of every conversation (or when asked), update it with what was done and what's next.

**Memory location:** All session memory lives in `.claude/memory/` inside this repo. It syncs via git across machines and is the single source of truth. Do NOT write to the Claude harness auto-memory location (`~/.claude/projects/*/memory/`) — that path is not used for this project and will diverge.

## What This Is

Breakwater is an earnings tail-risk model for S&P 500 stocks. It ingests price/earnings/sector data, engineers features, scores each stock's upcoming earnings event on a risk scale, and produces reports and a Streamlit dashboard.

## Commands

```bash
# Run the full pipeline
python main.py

# Run backtesting suite (reads output/full_df.parquet)
python -m testing.backtesting

# Prepare streamlit_df.csv from output/full_df.parquet
python prep_for_streamlit.py

# Launch the Streamlit dashboard
streamlit run streamlit_dash/app.py

# Ad-hoc feature/score testing
python testing.py

# Run tests
pytest
```

The project uses a `.venv` (Python 3.14). Activate with `source .venv/bin/activate` or prefix commands with `.venv/bin/python`.

## Pipeline Architecture

`main.py` → `pipeline/pipeline.py` → five sequential stages:

| Stage | File | What it does |
|---|---|---|
| 1 | `pipeline/stage1.py` | Creates/updates DuckDB at `data/breakwater.duckdb`. Set `update=True` in `pipeline.py` to re-fetch from APIs; defaults to `False` (skip re-ingestion). |
| 2 | `pipeline/stage2.py` | Reads `prices`, `earnings`, `stock_data` tables from DB; merges into a single DataFrame. |
| 3 | `pipeline/stage3.py` | Calls ~18 feature-engineering functions in sequence; each appends columns in-place and returns the df. |
| 4 | `pipeline/stage4.py` | Calls ~12 risk-scoring functions; produces `risk_score` (0–100) and intermediate component scores. |
| 5 | `pipeline/stage5.py` | Reads `output/full_df.parquet` (note: ignores the passed df), computes Bayesian bucket stats, writes `report_txt.txt`. `stocks_to_report_for` is an empty list — populate it to generate per-stock reports. |

The intermediate output between stages is a pandas DataFrame. Stage 3 and 4 functions all follow the same pattern: accept `input_df`, copy it, add columns, return it — never mutate in place.

## Data Storage

- **DuckDB** (`data/breakwater.duckdb`): three tables — `prices (stock, date, price)`, `earnings (stock, earnings_date, reported_eps, estimated_eps, surprise_percentage)`, `stock_data (stock, sector, sub_sector)`.
- **Parquet** (`output/full_df.parquet`): the fully engineered + scored DataFrame written after stage 3/4. This is the source of truth for backtesting and reporting.
- **CSV** (`streamlit_df.csv`): produced by `prep_for_streamlit.py`; consumed by the Streamlit app.
- **Stock universe** (`data/stock_list.csv`): the list of stocks to process.

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

Stage 4 produces these component scores (all in `risk_scoring/scoring_features.py`):
- `proximity_score`: how close the stock is to earnings
- `vol_expansion_score`: vol expansion relative to baseline
- `momentum_fragility_score`: momentum divergence signal
- `earnings_explosiveness_score`: historical tail-risk profile score
- `earnings_explosiveness_bucket`: categorical bucket (`low / moderate / elevated / high / extreme`)
- `risk_score`: weighted composite of the above (0–100)

Thresholds are in `config.py`: `LARGE_EARNINGS_REACTION_THRESHOLD = 0.007`, `EXTREME_EARNINGS_REACTION_THRESHOLD = 0.08`.

## Key Configuration (`config.py`)

- `DB_PATH`: path to DuckDB file
- `STOCKS_START_DATE / STOCKS_END_DATE`: date range for price data
- `DEFAULT_REACTION_WINDOW`: `"reaction_3d"` — the primary reaction metric
- `PRICES_PROVIDER`: currently `"ALPHAVANTAGE"`; rate-limit params (`ALPHAVANTAGE_CALLS_PER_MINUTE`, `BACKOFF_SECONDS`) also live here

## Backtesting

`backtesting/backtesting.py` — standalone module, reads `output/full_df.parquet` directly. The `backtesting_suite()` function runs calibration, lift, hit rates, and year-by-year OOS checks. `backtesting/testing_functions.py` contains all the individual test helpers.

The train/test split convention used in `testing.py`: pre-2015 = train, post-2015 = OOS test.
