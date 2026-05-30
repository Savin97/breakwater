# Session Memory Index

This file is read by Claude Code at the start of each session to restore context.
Entries are updated at the end of each session. Most recent first.

---

## 2026-05-30 — Digest layout frozen, ready for historical evaluation (session 2 of 2)

**Final digest changes (end of session):**
- HC section title: "★ High Conviction" → "High Conviction ★"
- Summary bar: "High Conviction ★ — N events · ★ = High Conviction (High Alert + pre-earnings drift)"
- Footer split into two lines: Percentile definition + HC definition
- "Overdue Miss" → "Extended Beat Streak" in scoring_features.py
- Layout is now frozen per GPT review — stop iterating on presentation

**Next session — historical evaluation:**
Build a script that runs the digest selection logic across past earnings weeks and produces:
- Total earnings events per week vs. number surfaced
- Capture rate for moves ≥8%, ≥10%, ≥15% by tier (Normal/Elevated/High Alert/HC)
- False-negative rate among omitted stocks
- Calibration by percentile band
- Comparison vs. simple baseline (recent realized vol)

This validates the core product claim: "Breakwater reduces the earnings calendar while retaining a disproportionate share of the largest moves."

**Also pending (lower priority):**
- Uncapped percentile: rank by `abs_reaction_p75_rolling` pre-clip to break 97th-percentile ties
- Flag glossary for digest (wait to see if users ask for it)
- Payment gate on harbor-markets.com (website repo, separate)

## 2026-05-30 — Major product build session

**Shipped:**
- IV (expected_move_pct, atm_iv, iv_vs_hist_ratio) joined in stage2, shown in per-stock reports as "Options Market Signal" section
- Coverage automation: stage5 auto-selects High Alert + Elevated stocks with earnings in 14-day window (manual CSV override kept commented)
- Reports now output to `output/reports/`
- Weekly email digest: `cron/cron_weekly_digest.py` — sends HTML email Mondays 07:00 UTC, reads full_df.parquet, `data/subscribers.txt` for list
- Cron scripts moved from `data_ingestion/` to `cron/` folder
- Digest: ordinal suffixes, company names, IV column hidden when empty, HC section at top, explicit date range, percentile display replacing raw score
- "Overdue Miss" renamed to "Extended Beat Streak" in scoring_features.py
- Reports: footer updated to harbor-markets.com, ordinal suffix fixed

**IV cron bug fixed:** `cron_iv.py` was importing `create_iv_table_if_not_exists` from wrong module — fixed to import from `data_ingestion.db_functions`

**Droplet crontab updated to:** `cron.cron_ingest`, `cron.cron_iv`, `cron.cron_weekly_digest`

**Next model work (not done):** Compute percentile from uncapped raw score (pre-clip p75) to differentiate 100-scored stocks; historical calibration tables

## 2026-05-19 — Report content additions + yfinance migration (in progress)

**Report additions (completed):**
- Created `report/chart_builder.py` — `generate_reactions_chart(earnings_df, n=16)` returns SVG string of bar chart (green/red bars, ±8% threshold lines, darker shading for extreme events)
- Added to stage5: peer_percentile (Xth percentile vs S&P 500), days_to_earnings, reactions_chart_svg
- HTML: historical reactions chart section, peer percentile stat block, days-to-earnings in meta row
- CSS: `.chart-container`, `.peer-note` added to styles.css
- All 4 reports (AAPL, NVDA, TSLA, MSFT) regenerate cleanly

**yfinance migration (partially done, NOT yet tested):**
- AlphaVantage subscription cancelled — need yfinance replacement
- Added `ingest_all_stocks_yf(con)` to `data_ingestion/fetch_prices.py` — batch download, incremental from global max date, chunks of 100
- Added `ingest_all_earnings_dates_yf(con)` to `data_ingestion/fetch_earnings_dates.py` — uses `yf.Ticker().earnings_dates`, skips stocks with future dates already in DB, manual dedup since fiscal_end_date=None bypasses unique index
- `pipeline/stage1.py`: old AlphaVantage calls commented out (19/5/26), new yf functions active
- `config.py`: `STOCKS_END_DATE` now uses `date.today().isoformat()` dynamically; added `from datetime import date` at top
- **NOT yet tested** — killed before smoke test could complete. Test first thing next session.

**Next:** Run smoke test on 3 stocks (AAPL, TSLA, NVDA), verify prices + earnings update correctly, then run full pipeline with update=True.

---

## 2026-05-18 — Memory setup + context recovery

Set up `.claude/memory/` inside the repo so session notes sync via git across machines.

---

## 2026-05-18 — Backtesting high_conviction + Report recommendation block

**Backtesting:**
- Validated `is_high_conviction` (High Alert + drift flag): 4.93x OOS lift, 12 events/yr
- Tested `surprise_momentum_flag` sub-categories: Beat/Miss Streak and Erratic add modest lift (~4.2x); Overdue Miss is below High Alert baseline (3.64x) — noise
- Tested "HA + Drift OR (Surprise ex-OM)": 4.24x lift, 75 events/yr — better coverage but lower precision
- Decision: `is_high_conviction` stays as drift-only (4.93x); broader definition has no clear home in the 3-tier system
- 3-tier system: Normal = calm, Elevated = risky, High Alert = very dangerous; High Conviction is a highlight within High Alert

**Report recommendation block:**
- Created `report/recommendations_builder.py` — `build_recommendation()` returns headline, body, action, flag_lines
- Wired into `stage5.py`, `report_builder.py`, HTML template, CSS
- 4 tiers: Normal (no action), Elevated (light caution), High Alert (reduce/hedge), High Alert + HC (strongest language)
- Flag explanations (drift + surprise) shown for Elevated and High Alert; suppressed for Normal
- `risk_score` = `earnings_explosiveness_score` (they are the same thing now)

**Report testing:**
- Fixed `stage5.py` to use `data/sp500_full_info.csv` (was incorrectly referencing `sp500_data.csv`)
- Successfully generated reports for AAPL, NVDA, TSLA, MSFT
- Reports confirmed working end-to-end with recommendation block rendering correctly

**Next:** Reports need further work — content and design TBD. Dashboard is lower priority.

---

## Product & Business
- [Product direction](project_direction.md) — target market (retail options traders), value prop, pricing, live URLs (as of 2026-05-30)
- [Next to build](next_to_build.md) — prioritized build list: IV into reports → coverage automation → weekly email digest (as of 2026-05-30)

## Infrastructure
- [DigitalOcean droplet](infra_digitalocean.md) — cron_ingest.py + cron_iv.py run here daily (IV post-close)

## DuckDB Schema — data/breakwater.duckdb

**prices:** stock, date (DATE), price (DOUBLE), ingested_at — unique index (stock, date)
**earnings:** stock, earnings_date (DATE), fiscal_end_date (DATE), reported_eps, estimated_eps, surprise_percentage (DOUBLE), ingested_at — unique index (stock, earnings_date, fiscal_end_date); fiscal_end_date=None for yfinance rows (manual dedup in code); surprise_percentage stored as decimal (÷100)
**stock_data:** stock (PK), company_name, sector, sub_sector, ingested_at
**merged_stock_data:** denormalised join of the above — NOT used by pipeline (stage2 reads raw tables directly)

---

## 2026-05-17 — Streamlit dashboard overhaul (other machine)

- Created `pipeline/streamlit_export.py`: generates `streamlit_df.csv` with Bayesian bucket stats; now called automatically at end of stage 5 (replaces manual `prep_for_streamlit.py` step)
- Dashboard: added `pre_earnings_drift_flag`, `surprise_momentum_flag`, `is_high_conviction` columns to Overview and Bucket Stats tabs
- Added "High Conviction only" sidebar filter and metric card
- Weekly Calendar tab: removed `momentum_fragility_score` / "Positioning"; replaced with timing flags and `is_high_conviction`
- `is_high_conviction` = "High Alert" bucket AND non-empty `pre_earnings_drift_flag`

**Next:** Unknown — pick up from here.

---
