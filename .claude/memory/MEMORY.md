# Session Memory Index

This file is read by Claude Code at the start of each session to restore context.
Entries are updated at the end of each session. Most recent first.

---

## 2026-06-07 — Codebase audit & refactor

- [Full change log](codebase_audit_2026_06_07.md) — every file touched, what changed and why, what was deliberately kept. Read this before debugging any score/pipeline regression introduced after this date.
- Backtesting verified bit-for-bit identical before and after.
- Key changes: fixed WMB ticker bug, fixed 3× `.to_numpy()` alignment risk, removed ~34 redundant `df.copy()` calls across feature/scoring functions, stage5 loop pre-grouped by stock, streamlit filter copy removed, `engineer_timing_danger` deleted, 3 dead file/folders deleted.

---

## 2026-06-01 — Calibration, percentile fix, landing page, dashboard diagnosis

**Model work:**
- Built `testing/calibration.py` — historical calibration tables (by bucket, capture rate, percentile band, year-by-year). Results: High Alert 40.2% P(≥8%) vs 6.9% base (5.8x lift), HC 52.4%, consistent 2015–2025.
- Fixed uncapped percentile ranking in `cron/cron_weekly_digest.py`, `pipeline/stage5.py`, `testing/calibration.py` — now ranks by `abs_reaction_p75_rolling.fillna(abs_reaction_p75)` instead of clipped `earnings_explosiveness_score`. 99–100th percentile band now populated (53.8% P(≥8%)).
- Product framing decision: sell "which 15–20 events matter this week," lead with ≥8% lift story. Don't try to fix false-negative rate at ≥5% — Normal bucket big moves are structurally unpredictable.

**Infrastructure / repo:**
- Consolidated memory to `.claude/memory/` only — deleted harness memory files, updated CLAUDE.md.
- Renamed `cv_website` → `harbor_webpage` locally, on GitHub (Savin97/harbor_webpage), and re-cloned on server at `/var/www/harbor_webpage`.
- Cleaned harbor_webpage repo: moved CV/portfolio to `cv/` subfolder, deleted stale `app.py`, `streamlit_df.csv`, `requirements.txt`.

**Landing page:**
- Built `harbor_webpage/index.html` — dark, minimal product landing page. Eyebrow + serif H1 + email capture (Formspree placeholder). Three numbered feature bullets. No stats/methodology revealed.
- **TODO:** Sign up at formspree.io, replace `REPLACE_WITH_YOUR_ID` in both form action attributes, push + git pull on server.

**Dashboard diagnosis:**
- Dashboard "staleness" is not a bug — Q1 2026 earnings season ended ~May 26th, Q2 starts mid-July. Data IS current.
- Root issue: `streamlit_export.py` only exports `is_earnings_day == 1` rows — dashboard is historical-only, no upcoming events view.
- Parked as next build item: add upcoming events tab to dashboard (see next_to_build.md).
- Workflow established: pull DuckDB from droplet → run pipeline locally → scp `streamlit_df.parquet` back.

## 2026-05-31 — Report/digest consistency fixes (pipeline/stage5.py)

**Two inconsistencies fixed between digest and PDF reports (both in `pipeline/stage5.py`):**

1. **Wrong earnings date in report**: Report was using `earnings_df.iloc[-1]["earnings_date"]` (last `is_earnings_day==1` row = past event). Fixed to use `latest_per_stock_idx.loc[stock, "earnings_date"]` — the forward-filled upcoming date from the latest price row. LULU: was showing Mar 17, now shows Jun 04 correctly.

2. **Percentile mismatch**: Report was ranking `earnings_explosiveness_score` against earnings-day rows only (→ 94th). Digest uses `rank(pct=True)` across all stocks' latest rows (→ 97th). Fixed report to use the same `rank(pct=True)` approach on `latest_per_stock_idx`. Also removed now-unused `latest_scores` and `n_universe` variables.

**Verified**: Regenerated reports, LULU report confirmed showing Jun 04 + 97th pct — matching digest.

**Next priorities:**
1. Uncapped percentile ranking (rank by `abs_reaction_p75_rolling` pre-clip to break ties at 97th)
2. Historical calibration tables (capture rate by tier across past 20–30 weeks)
3. Stripe payment link + fix landing page form (website repo)

---

## 2026-05-31 — Bug fixes + report delivery shipped

**Bugs fixed (same root cause in two places):**
- `cron/cron_weekly_digest.py` `_select_stocks()` was filtering `is_earnings_day == 1` before grouping — this made it impossible to find upcoming earnings (future rows never have `is_earnings_day == 1`). Fixed to `df.sort_values("date").groupby("stock").last()` — groupby skipna=True pulls the last non-null bucket from historical data, and the latest earnings_date from the most recent row.
- `pipeline/stage5.py` auto-selection had the identical bug — reports were being generated for 0 stocks. Same fix applied.

**New features:**
- Digest now attaches PDF reports for flagged stocks (`_collect_reports()` in cron_weekly_digest.py, MIMEBase attachment via `email.mime.base`)
- Unsubscribe mailto link added to digest footer

**Verified end-to-end:** ran digest, received email with HTML + 6 PDF attachments (LULU, PANW, CRWD, ULTA, AVGO, COO).

**Next priorities (from plan):**
1. Uncapped percentile ranking (rank by `abs_reaction_p75_rolling` pre-clip to break ties at 97th)
2. Historical calibration tables (capture rate by tier across past 20–30 weeks)
3. Stripe payment link + fix landing page form (website repo)

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
- [DigitalOcean droplet](infra_digitalocean.md) — cron schedule, droplet path, stale tickers list
- [Window sensitivity](window-sensitivity.md) — grid search confirmed window=28 optimal (4.49x avg lift, 100% years ≥3x)

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
