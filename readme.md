# Breakwater

Earnings tail-risk model for S&P 500 stocks. Scores each upcoming earnings event on a risk scale (0–100) and surfaces the ~15–20 events per week most likely to produce large moves.

> **No performance figures are published at present.** The scoring model is being replaced and will be re-fit on timestamp-anchored earnings reactions; new figures will be published only after that model has been audited. The previously published figures (40% vs 6.9%, 5.8x lift, a consistent 2015–2025 OOS record) were derived from a target that mismeasured before-open announcements, were overstated by roughly 3x, and are retracted. The history is recorded in `audit/PHASE0_AUDIT_REV2.md`.

Live at **harbor-markets.com/breakwater** (Streamlit dashboard). Weekly email digest sent Monday mornings.

---

## Quick Commands

```bash
source .venv/bin/activate

python main.py                         # Full pipeline (stages 1–5)
python -m testing.backtesting          # Backtesting suite (reads output/full_df.parquet)
streamlit run streamlit_dash/app.py    # Launch dashboard locally
python -m testing.calibration          # Calibration tables → testing/testing_results/
```

## Monday Workflow

```bash
bash scripts/full_workflow.sh
```

This script does everything in one shot:
1. Pull `breakwater.duckdb` from the droplet (rsync)
2. Run full pipeline → `output/full_df.parquet`
3. Generate `output/weekly_chart.png` + `output/recent_calls.json`
4. Push output parquets to the droplet (so the live dashboard updates)
5. Push `recent_calls.json` to the harbor webpage
6. Print last week's results

---

## Pipeline Architecture

`main.py` → `pipeline/pipeline.py` → five sequential stages:

| Stage | File | What it does |
|---|---|---|
| 1 | `pipeline/stage1.py` | Creates/updates DuckDB. Set `incremental=True` in `pipeline.py` to re-fetch from APIs; default `False` skips ingestion. |
| 2 | `pipeline/stage2.py` | Reads `prices`, `earnings`, `stock_data` from DB; merges into a single DataFrame. Calls `dedup_earnings()` to remove yfinance duplicates. |
| 3 | `pipeline/stage3.py` | ~18 feature-engineering functions in sequence, each appending columns and returning the df. |
| 4 | `pipeline/stage4.py` | ~12 risk-scoring functions; produces `risk_score` (0–100) and component scores. |
| 5 | `pipeline/stage5.py` | Reads `output/full_df.parquet`, computes Bayesian bucket stats, writes `report_txt.txt`, generates per-stock PDF reports for `stocks_to_report_for`. |

For faster runs (no new earnings data): `pipeline/incremental.py` → `run_incremental()` — 0.8s vs 80s, scores bit-for-bit identical.

---

## Codebase Map

```
breakwater/
├── main.py                         # Entry point
├── config.py                       # All constants and paths
│
├── pipeline/
│   ├── pipeline.py                 # Orchestrates stages 1–5
│   ├── stage1.py – stage5.py       # Pipeline stages (see above)
│   ├── events.py                   # Stage 4b: event frame (one row per earnings event + one pending row per stock)
│   └── incremental.py              # Fast incremental update path
│
├── data_ingestion/
│   ├── fetch_prices.py             # yfinance price ingestion
│   ├── fetch_earnings_dates.py     # yfinance earnings dates ingestion
│   ├── fetch_iv.py                 # Options IV fetching (post-close)
│   ├── fetch_sp500_sectors.py      # Sector/company name data
│   ├── db_functions.py             # DuckDB table creation + helpers
│   ├── data_utilities.py           # dedup_earnings() and other DB utilities
│   └── api_functions.py            # Shared HTTP/rate-limit helpers
│
├── feature_engineering/
│   ├── pre_earnings_stock_features.py   # Vol, momentum, drift, timing features
│   ├── post_earnings_stock_features.py  # Reaction stats, entropy, directional bias
│   ├── pre_earnings_sector_features.py  # Sector-level aggregates
│   ├── event_features.py               # Event-level statistics shared by history and the pending row
│   ├── announcement_timing.py          # BMO/AMC from observed timestamps; timestamp-anchored reactions
│   └── scoring_features.py             # Component scores → risk_score + bucket
│
├── cron/
│   ├── cron_ingest.py              # Daily 06:00 UTC: prices + earnings update
│   ├── cron_iv.py                  # Weekdays 20:30 UTC: IV snapshot
│   └── cron_weekly_digest.py       # Monday 07:00 UTC: email digest to subscribers
│
├── report/
│   ├── report_builder.py           # Per-stock HTML/PDF report generation
│   ├── recommendations_builder.py  # Recommendation block (Normal/Elevated/HA/HC)
│   ├── chart_builder.py            # Historical reactions bar chart (SVG)
│   ├── chart_weekly.py             # Weekly earnings calendar chart (PNG)
│   ├── chart_results.py            # Last-week outcomes chart (PNG)
│   ├── calendar_builder.py         # Weekly calendar HTML
│   └── templates/                  # earnings_report.html, styles.css, weekly_calendar.html
│
├── streamlit_dash/
│   ├── app.py                      # Streamlit dashboard (Overview, Bucket Stats, Upcoming, Calendar tabs)
│   └── streamlit_export.py         # Exports streamlit_df.parquet + upcoming_df.parquet
│
├── testing/
│   ├── backtesting.py              # Full backtesting suite (calibration, lift, OOS)
│   ├── calibration.py              # Calibration tables by bucket/percentile/year
│   ├── testing_functions.py        # Individual test helpers
│   ├── metric_testing.py           # Ad-hoc feature/score metric script (not a test suite)
│   ├── test_*.py                   # pytest suite (`pytest testing`)
│   └── testing_results/            # CSVs: calibration_by_bucket, _by_percentile, _year_by_year
│
├── scripts/
│   ├── full_workflow.sh          # Full Monday workflow (see above)
│   ├── gen_recent_calls.py         # Generates output/recent_calls.json for landing page
│   ├── last_week_results.py        # Prints last week's earnings outcomes
│   ├── results_check.py            # Full price history around past earnings events
│   ├── sync_pipeline.sh            # rsync helper: pull the DB from the droplet
│   └── sync_iv.sh                  # rsync helper: pull IV data from droplet
│
├── data/
│   ├── breakwater.duckdb           # Source of truth for all raw data
│   ├── stock_list.csv              # S&P 500 universe (~500 tickers)
│   ├── sp500_full_info.csv         # Company names + sector + sub-sector
│   └── subscribers.txt             # Email digest subscriber list
│
└── output/
    ├── full_df.parquet             # Fully engineered + scored DataFrame (all history)
    ├── streamlit_df.parquet        # Historical view for dashboard
    ├── upcoming_df.parquet         # Forward-looking view for dashboard Upcoming tab
    ├── events_df.parquet           # Event frame: one row per earnings event (history + pending)
    ├── recent_calls.json           # Last 2 weeks of calls for landing page
    ├── weekly_chart.png            # Weekly earnings calendar chart
    ├── results_chart.png           # Last-week outcomes chart
    ├── weekly_calendar.html        # HTML calendar
    └── reports/                    # Per-stock PDF reports (e.g. NVDA_report.pdf)
```

---

## Data Storage

**DuckDB** (`data/breakwater.duckdb`) — three tables:
- `prices (stock, date, price)` — unique index (stock, date)
- `earnings (stock, earnings_date, fiscal_end_date, reported_eps, estimated_eps, surprise_percentage)` — surprise stored as decimal (÷100)
- `stock_data (stock, company_name, sector, sub_sector)`

**Parquet** (`output/full_df.parquet`) — source of truth for backtesting, reporting, and the dashboard.

---

## Infrastructure

**DigitalOcean droplet** (`harbor-markets.com`):
- Repo: `/var/www/breakwater` — deploy by pushing locally + `git pull` on droplet
- Crons run ingestion, IV, and digest automatically (see `cron/`)
- Dashboard served at `harbor-markets.com/breakwater` (systemd + Nginx)

**Website** (`harbor-markets.com`): separate repo `Savin97/harbor_webpage`, server at `/var/www/harbor_webpage`.

---

## Model Performance

None published. The audit measurement of the legacy tiers on timestamp-anchored outcomes is
historical record in `audit/PHASE0_AUDIT_REV2.md` §Q2–Q3, not current product evidence.
Figures for the replacement model will appear here only after it has been fitted and audited.
