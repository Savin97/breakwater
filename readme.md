# Breakwater

Earnings tail-risk model for S&P 500 stocks. Scores each upcoming earnings event on a risk scale (0–100) and surfaces the ~15–20 events per week most likely to produce large moves. On 11,496 events with verified announcement timestamps, High Alert stocks move ≥8% on earnings **45.5%** of the time [42.9, 48.2] vs. a **20.4%** base rate [19.7, 21.1] — a **1.91x lift, stratified by announcement window**.

> **Performance figures: `audit/PHASE0_AUDIT_REV2.md` is the authority.** Every number in this file traces to it. The previously published figures (40% vs 6.9%, 5.8x lift, a consistent 2015–2025 OOS record) were derived from a target that mismeasured before-open announcements; they were overstated by roughly 3x and are retracted. **Do not reintroduce them, and do not add a performance figure that is not stated in that audit.** Quote the stratified lift, never the crude one: the crude-vs-stratified gap is composition, not model skill (audit §Q2).

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
│   ├── testing.py                  # Ad-hoc feature/score testing
│   ├── window_sensitivity.py       # Grid search over reaction window
│   └── testing_results/            # CSVs: calibration_by_bucket, _by_percentile, _year_by_year
│
├── scripts/
│   ├── full_workflow.sh          # Full Monday workflow (see above)
│   ├── gen_recent_calls.py         # Generates output/recent_calls.json for landing page
│   ├── last_week_results.py        # Prints last week's earnings outcomes
│   ├── results_check.py            # Full price history around past earnings events
│   ├── sync_pipeline.sh            # rsync helper: pull DB, push parquets
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

## Key Model Numbers

All figures from `audit/PHASE0_AUDIT_REV2.md` §Q2, measured on **n=11,496 events with verified
provider announcement timestamps**. Hit rates are timestamp-anchored; lift is **stratified by
announcement window** (observed rate ÷ the rate the tier's BMO/AMC mix alone would produce with
zero within-window skill). Market baseline over the same events: **20.4%** [19.7, 21.1].

| Tier | n | P(move ≥8%) [95% CI] | Stratified lift | Crude lift |
|---|---:|---|---:|---:|
| Normal | 9,120 | 15.7% [14.9, 16.4] | 0.80x | 0.77x |
| Elevated | 1,022 | 29.3% [26.5, 32.1] | 1.23x | 1.43x |
| High Alert | 1,354 | 45.5% [42.9, 48.2] | **1.91x** | 2.23x |
| High Conviction (HA + drift flag) | 162 | 58.6% [50.9, 65.9] | **2.46x** | 2.88x |

High Alert and Elevated together capture **39.0%** [37.1, 41.0] of all ≥8% moves.

**Scope, which must travel with these numbers:**

- The sample is dated 2008–2026, but verified timestamps reach back only to about 2020 for
  most tickers, and cover 25.4% of all scored events. **There is no 15-year or 2015–2025
  out-of-sample record.** Nothing above is out-of-sample: the 73/79 cut points were selected
  on this distribution and are being re-fit (audit §Q3).
- The edge is concentrated in after-close reporters. Capture is **67.8%** on AMC events and
  **9.3%** [7.8, 11.1] on BMO. Within AMC — the half that was never mismeasured — High Alert
  lift is **1.87x**. On BMO, 96.0% of events are `Normal` and `Normal` lift is **0.94x**, i.e.
  no discrimination; the model's value on BMO is **unestablished** pending Phase 3 (audit §Q3).
- Knowing only that a company reports after the close is worth **1.42x** with no model at all.
  That is why the headline must be stratified.
