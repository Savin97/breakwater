# Session Memory Index

This file is read by Claude Code at the start of each session to restore context.
Entries are updated at the end of each session. Most recent first.

---

- [Social media strategy](social_media_strategy.md) — platforms, cadence, content rules, weekly workflow (added Jun 9, 2026)
- [Reddit/X marketing playbook](reddit_marketing_playbook.md) — comment tone, data angles, soft Breakwater plug, real examples from Jun 23 2026 (MU, FDX, NKE, NOW)

## 2026-08-29 — Logging framework + report logo fix (UNCOMMITTED, local only)

Continuation of the same session as the parallelization entry below — both are uncommitted together.

**Logging (new `utilities/logging_utilities.py::setup_logging()`):**
- `logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)-7s %(message)s",
  datefmt="%H:%M:%S", stream=sys.stdout, force=True)`. stdout (not stderr) so cron shell
  redirection keeps working. No FileHandler — `get_run_output_dir()` rmtree-on-first-call would
  delete it out from under us.
- Config: `LOG_LEVEL = "INFO"` and `NOISY_LIBRARIES` (a **dict** of library -> level cap).
  User preference: constants belong in config.py with clear names, not module-level in code.
- **Why NOISY_LIBRARIES exists (learned the hard way):** configuring the root logger *unleashed*
  third-party logging that had been silent because logging was never configured. First DEBUG run
  drowned in yfinance internals; first full pipeline run drowned in WeasyPrint's "Ignored `fill:
  #c0392b`" CSS warnings (dozens per report x 10 reports) — net output got WORSE than prints.
  yfinance/urllib3/peewee/matplotlib/fontTools capped at WARNING; **weasyprint + weasyprint.progress
  need ERROR**, since their CSS noise is logged at WARNING level.
- `setup_logging()` is called from `pipeline/stage1.py` (after its first print) and from
  `cron/cron_iv.py` / `cron/cron_eps_estimates.py` (they bypass the pipeline).
  **`main.py` is deliberately untouched — user wants it kept clean.**
- Converted: `fetch_earnings_dates.py` (2 live fns only), `fetch_prices.py`
  (`incremental_ingest_all_prices_yf` only), `fetch_sp500_sectors.py`,
  `db_utilities.py` (`clean_duplicate_earnings_from_db`'s one line).
  Levels: per-stock noise -> DEBUG, summaries/CORRECTED -> INFO, SKIPPED/no-calendar -> WARNING,
  FAILED -> ERROR (still also written to the `debug_failed_*.txt` files).
- **Deliberately NOT converted:** legacy AlphaVantage fns + unused `get_next_earnings_dates`;
  all of `testing/` (interactive scripts, ~200 prints); and **product output** — the weekly risk
  table, backtesting `to_string()` tables, "Saved chart → ..." lines. Rule agreed with user:
  *logging = "what the program is doing / went wrong" (diagnostics); print = "the thing you ran the
  command to read" (output).* Timestamping table rows would wreck them.

**Report logo bug — found BY the logging change, fixed:**
`report/templates/earnings_report.html` + `weekly_calendar.html` pointed at `webpage/img/breakwater_logo.png`,
which **has never existed in this repo**. Every customer-facing PDF had been rendering a broken image,
silently, because WeasyPrint's ERROR went nowhere without logging configured. Fixed by creating
`report/img/` and copying `harbor_webpage/assets/logo.png` (1200x434, user picked from 4 variants)
to `report/img/breakwater_logo.png`, then repointing both templates. `base_url=project_root` in
`report_builder.py:50`, so `report/img/...` resolves correctly. Verified by rendering AVGO_report.pdf
to PNG — logo displays in header; PDFs went ~30KB -> ~188KB.

## 2026-08-28/29 — Fixed IV/EPS cron ingestion end-to-end (4 stacked bugs)

`eps_estimates` was empty and `iv_snapshots` had stalled since Jun 26. Four independent bugs,
each masking the next. All fixed and verified live. Details in
[infra_digitalocean.md](infra_digitalocean.md).

1. **Cron invocation** — crontab called `python /var/www/breakwater/cron/cron_iv.py` by path, so
   `cron/` (not the repo root) landed on `sys.path` → `ModuleNotFoundError: No module named
   'config'` on every run since ~Jun 29. Fix: always `cd /var/www/breakwater && python -m
   cron.<module>`, as `cron_ingest` already did.
2. **Positional INSERT** — `INSERT INTO iv_snapshots SELECT * FROM tmp_iv` is positional, but the
   live table has `snapshot_hour` appended last (added later via ALTER TABLE) while the current
   `CREATE TABLE` declares it 3rd → integer hour shoved into the `earnings_date` DATE column,
   `ConversionException: BIGINT -> DATE`. Fix: explicit column lists in `ingestion/fetch_iv.py`
   and `fetch_eps_estimates.py`. **Never use positional `SELECT *` inserts against a table whose
   physical column order can drift.**
3. **DuckDB write lock** — `cron_iv` and `cron_eps_estimates` were scheduled on the same minute;
   DuckDB allows one writer, so the loser died outright (zero rows, not partial). Fix: stagger.
4. **Timezone** — system TZ is UTC and Ubuntu's Debian cron **ignores `CRON_TZ`**, so ET-intended
   times ran as UTC, putting 2 of 4 IV runs pre-market where options have no bid/ask. Fix: cron
   times written in UTC inside the 14:30–20:00 UTC DST-safe window.

**Verified 2026-08-28:** post-fix runs at 16:30/18:00/19:30 UTC each inserted 32 rows.
**Also:** dependencies upgraded + pinned (local and droplet now match; `requirements.txt` was
UTF-16, now UTF-8); `validate_upcoming_earnings_dates` narrowed to a 20-day window
(`EARNINGS_DATE_VALIDATION_WINDOW_DAYS`), cutting it from 501 to ~165 yfinance calls.

**Next / open:**
- Monday 2026-08-31 is the first full day on the new schedule — expect IV hours 15/16/18/19 and
  EPS at 14:45 UTC.
- Droplet is one commit behind master (at `942de67`; master `117ee16`) — undeployed, affects the
  reporting path not the crons.
- Unexplained `eps_estimates` gap on Aug 6–7 (both weekdays, no rows) — never root-caused.
- Stale untracked files on droplet: `data/breakwater.duckdb` (superseded by the `db/` move) and
  `next_earnings_df.csv`. Confirm before deleting the DB file.

## 2026-08-29 — Parallelized yfinance earnings fetches (UNCOMMITTED, local only)

**Done — both per-ticker yfinance loops in `ingestion/fetch_earnings_dates.py` now fetch concurrently.**
Pattern used in both: extract the network+parse work into a standalone function that touches
NO DB and returns a dict; run those across a `ThreadPoolExecutor`; then keep the original
sequential DB-write loop, iterating in the original order, reading pre-fetched results from a
dict keyed by ticker. DuckDB connections aren't thread-safe and `con.register("tmp_earnings_df",…)`
uses a fixed view name, so DB writes must stay serialized — that constraint drove the whole design.

- `fetch_one_earnings_dates(stock)` → `{"stock", "earnings_dates_df", "error"}`. Three outcomes kept
  distinct to preserve old behavior: data / no-data (silent) / exception (printed + written to
  `debug_failed_earnings_ingestion.txt`).
- `fetch_upcoming_earnings_date(stock, today)` → `{"stock", "upcoming_earnings_date", "error"}`. Reads
  `.calendar` (company IR confirmed date) — a *different* endpoint from `.earnings_dates` (history +
  Yahoo's often-wrong estimate); that difference is the whole point of the validate cross-check.
  Now deduped to one fetch per unique stock (the old row-by-row loop refetched stocks having
  multiple upcoming dates).
- Renamed `future_dates` → `stocks_to_skip` (held tickers, not dates). Recheck window 7d → 14d,
  moved to config.
- Removed the per-iteration `time.sleep(0.3)` in both loops; worker cap + jitter replace it.
- New in `config.py`: `YFINANCE_MAX_WORKERS = 8`, `YFINANCE_JITTER_MIN_SECONDS = 0.05`,
  `YFINANCE_JITTER_MAX_SECONDS = 0.15`, `EARNINGS_RECHECK_WINDOW_DAYS = 14`.

**Measured:** 40-ticker benchmark 59.6s → 5.1s (**11.7x**), identical data both ways, 0 errors.
Single yfinance fetch ≈ 0.8–2.4s, so this loop was ~100% network-bound — DB work in it is ~0ms.
Validate loop: 16 dates in 1.8s (was ~25-30s serial). Full `main.py` run: 1m30s, exit 0, all stages OK.

**Watch item:** one `SSLError: Connection reset by peer` on BF-B during a full run. Investigated:
BF-B alone sequentially 5/5 fine, and 180 concurrent fetches (60 tickers × 3 rounds @ 8 workers)
produced ZERO errors — so it reads as transient network flakiness, not throttling. If resets start
clustering on real runs, drop `YFINANCE_MAX_WORKERS` to 4–5 (single easy lever).

**State: NOT committed, NOT deployed.** All changes are working-tree only; droplet still runs the old
sequential code until a commit + `git pull` there. `cron/cron_ingest.py`'s `stage1(update=True)` bug
noted on 2026-08-15 was already fixed by other work — nothing to do.

**Update:** the logging pass (the other half of the original "too verbose / too slow" work) was
implemented later the same day — see the 2026-08-29 logging entry above. Both are uncommitted together.

## 2026-08-19 — Diagnosed risk_score/bucket inconsistency; fix planned, deferred to own branch

**Bug:** a stock can show a lower `risk_score` than another while being in a higher risk group (Elevated/High Alert/High Conviction). **Cause:** `risk_score` = raw `earnings_explosiveness_score` (`scoring/scoring_features.py:263`), internally consistent with its bucket cut — but `report/report_builder.py:99-120` and `streamlit_dash/streamlit_export.py:6-53` each separately reimplement a per-stock "historical lift" bump that changes the *displayed* label without ever touching `risk_score`. Three uncoordinated implementations, only one feeds the sorted number.

**Decision:** fold the lift into `risk_score` itself in stage4 so score and bucket stay consistent everywhere, not just fix the label. Real model change (not plumbing) — needs a causal (shift/expanding, no lookahead) version of the lift, re-picked bucket thresholds, and a calibration re-run (`testing/calibration.py` + `testing/testing.py` correlation/decile checks vs. checked-in baselines) as a go/no-go gate before merging. `report/calendar_builder.py:6-10` already documents a past blend attempt hurting signal — treat "helps" as unproven until the calibration diff says so.

Full plan: `~/.claude/plans/binary-jingling-cake.md` (harness-local, may not survive across machines). **Deferred — user wants this on a separate branch, later.** Next session: branch first, implement, run calibration comparison before touching report_builder/streamlit_export cleanup.

## 2026-08-15 — Ticker lifecycle tracking, droplet DB reorg, logging plan (branch stock-lifecycle-status)

**Active/inactive ticker lifecycle (shipped, merged to master, deployed):**
- `stock_data` gained `status` ('active'/'inactive') and `reason` columns. Reconciled every run in `ingestion/fetch_sp500_sectors.py` (function later renamed `ingest_all_sector_data`→`ingest_all_sp500_data`, `get_sp500_sectors`→`get_current_sp500` by other work) against the live Wikipedia S&P 500 table. Tickers not in the live list get `status='inactive'` — either `reason='renamed → X'` (via `data/ticker_renames.csv`, a manually-maintained old_ticker,new_ticker,note map) or a generic delisted/merged reason.
- `utilities/data_utilities.py::read_stocks_to_fetch(con=None, active_only=False)` — when `active_only=True`, filters out inactive tickers. Used by the price/earnings incremental fetch loops so dead tickers stop being retried every run (was 503→490 stocks fetched).
- `data/ticker_renames.csv` populated after web research distinguishing true renames from mergers/index-removals: **BK→BNY** (NYSE ticker change 2026-05-21), **SATS→ECHO** (Nasdaq ticker change 2026-06-24). Confirmed NOT renames (stay plain inactive): DAY, EA, HOLX (all taken private/delisted entirely), CTRA (merged into pre-existing DVN, not a same-company rename), CAG/CPB/EPAM/POOL/LW/PAYC/MTCH/MOH (still trading, just demoted out of the S&P 500 index — not delisted at all).
- Dashboard (`streamlit_dash/app.py`): stock pickers now label inactive tickers (`"HOLX — inactive"`) via `get_inactive_stocks()` + `format_func`, instead of hiding them — historical browsing of delisted stocks stays intact.

**Droplet DB schema-drift bug found + fixed:** `scripts/sync_pipeline.sh` pulls the droplet's `breakwater.duckdb` and *overwrites local* — so the schema migration above kept getting silently wiped locally every time it was re-synced, and `ingest_all_sp500_data`'s INSERT would throw a `BinderException` on the missing columns (silently swallowed by its own try/except, logged not crashed). Root-caused via git reflog + file mtimes before the user mentioned the sync script. Fixed by migrating the *droplet's* DB directly (`ALTER TABLE stock_data ADD COLUMN IF NOT EXISTS status/reason`), then re-syncing.

**Repo reorg (shipped, merged to master, deployed):**
- Moved `breakwater.duckdb` from `data/` into a new `db/` folder (gitignored). `config.DB_PATH = "db/breakwater.duckdb"`. Updated `directory_checks()`, and all three sync scripts (`sync_pipeline.sh`, `sync_iv.sh`, `full_workflow.sh`).
- `.gitignore`: removed the `data/` and blanket `*.csv` ignore rules so `stock_list.csv`/`ticker_renames.csv`/`sp500_full_info.csv` sync via git instead of manual scp. Kept `data/subscribers.txt` explicitly ignored (contains a real email address).
- Deployed to droplet: moved its DB into `db/` too, `git pull`ed, restarted the `breakwater.service` systemd unit (serves the live Streamlit dashboard) — verified clean restart via journalctl.

**Known bug, NOT yet fixed:** `cron/cron_ingest.py` still calls `stage1(update=True)`, but `stage1`'s parameter was renamed to `incremental` by other/parallel work on this repo (`pipeline/stage1.py` now `def stage1(incremental:bool)`, `pipeline/pipeline.py::run_pipeline(incremental=True)`). This will raise `TypeError` on the droplet's next 6am `cron_ingest` run. Found while re-verifying the logging plan below; queued to fix as part of that pass (one-line change) since it's already in scope there.

**Logging cleanup — planned and approved, NOT yet implemented (next session should start here):**
Plan file (harness-local, may not persist across machines): `~/.claude/plans/keen-pondering-mochi.md`. Summary in case that's gone:
- New `utilities/logging_utilities.py::setup_logging()` — `logging.basicConfig(level=os.getenv("BREAKWATER_LOG_LEVEL","INFO"), format="%(asctime)s %(levelname)-8s %(name)s: %(message)s", stream=sys.stdout, force=True)`. No FileHandler (crontab already redirects stdout/stderr to log files; `get_run_output_dir()` does an rmtree-on-first-call that would delete a FileHandler's file out from under it).
- Call it as the first line of `main.py`, `cron/cron_ingest.py`, `cron/cron_iv.py`, `cron/cron_eps_estimates.py`.
- Convert print()→logging in: `ingestion/fetch_prices.py` (`incremental_ingest_all_prices_yf` only), `ingestion/fetch_earnings_dates.py` (`incremental_ingest_all_earnings_dates_yf`, `validate_upcoming_earnings_dates` only), `ingestion/fetch_sp500_sectors.py` (both functions), `utilities/db_utilities.py` (`clean_duplicate_earnings_from_db`'s one line), `pipeline/stage1.py`, `pipeline/pipeline.py`. Leave legacy AlphaVantage functions (`ingest_all_stocks`, `ingest_all_earnings_dates`, unused `get_next_earnings_dates`) and stage2–5/`fetch_iv.py`/`fetch_eps_estimates.py` untouched — not noisy, not in scope.
- Level scheme: DEBUG = per-ticker routine noise (~500 items), INFO = banners/summaries/batch results/`CORRECTED` lines, WARNING = recoverable edge cases (`SKIPPED`/no-calendar), ERROR = per-ticker `FAILED` lines.
- Bundle the `cron_ingest.py` bug fix (`update=True`→`incremental=True`) into this pass since that file is already being touched.
- **Deferred to a separate follow-up** (user said "let's take it slow"): parallelizing `incremental_ingest_all_earnings_dates_yf` and `validate_upcoming_earnings_dates`'s per-ticker yfinance loops via `ThreadPoolExecutor(max_workers=8)` — two-phase design (Phase 1: concurrent network fetch only, zero DB access, workers return a result dataclass and never raise; Phase 2: sequential DB writes in original ticker order, identical to today's logic) to avoid DuckDB single-connection thread-safety issues and the fixed `con.register("tmp_earnings_df",...)` name collision. Verified via source inspection that the installed yfinance's `YfData` singleton + `curl_cffi` session are both explicitly thread-safe; residual risk is Yahoo-side throttling, not client crashes — mitigated by conservative worker count + jittered per-request delay, should be validated empirically on a ~30-50 ticker subset before trusting on the full run.

## 2026-08-15 — Pipeline fixes, weekly eval script, test suite

- **Stray `exit()` in stage2.py** was killing the pipeline silently — parquet never updated. Removed.
- **`assert_df_fresh`** added to `utilities/data_utilities.py`; called at end of stage2 — raises if max price date >10 days old.
- **`clean_duplicate_earnings_from_db`** added to `utilities/db_utilities.py`; called in stage1 on update — eliminates persistent FDS/GIS/JBL/NKE dedup noise. Three-layer dedup stack: validate_upcoming → clean_duplicate_from_db → dedup_earnings (in-memory).
- **`is_high_conviction`** promoted to first-class stage4 column via `engineer_high_conviction` in `scoring/scoring_features.py`. Was previously computed ad-hoc in streamlit_export.
- **stage5.py** stripped to 17 lines; report orchestration moved into `report/report_builder.py`.
- **`testing/weekly_prediction_quality.py`** (new): evaluates prediction quality for every week since a given start date. Prints per-event tables + tier summary + confusion matrices. Saves CSV + confusion matrix PNG (`testing/testing_results/`). OOS 2024+: High Alert 3.35x lift@8%, HC 3.97x, 50% capture rate.
- **`testing/test_pipeline.py`** (new): 68 pytest tests — column existence, no-leakage shift checks, reaction correctness, score ranges, bucket labels, high-conviction invariants. `pytest testing/test_pipeline.py -v` → 68/68 in 0.52s.

## 2026-07-27 — Monday workflow log audit: fixed price-fetch date bug, logged 6 open issues

User pasted the full `scripts/full_workflow.sh` output log (~1200 lines) and flagged it as "insanity" — mostly a single bug's blast radius. Diagnosed the whole log; fixed the worst offender, rest are open.

**Fixed:**
- `ingestion/fetch_prices.py` `incremental_ingest_all_prices_yf()`: `end = date.today()` was passed straight to `yf.download(..., end=...)`, whose `end` is *exclusive*. On any run where the DB's last price date was Friday (i.e. every Monday run), `start`=Saturday, `end`=Monday(excluded) → the fetch window contained zero trading days, so **all 503 tickers** logged "possibly delisted; no price data found" and `Total inserted: 0` — pure noise, nothing actually delisted. Changed to `end = date.today() + timedelta(days=1)` so today is included. `timedelta` was already imported.

**Open issues found in the same log, not yet fixed (in rough priority order):**
1. **`BK` and `CTRA` are genuinely dead on yfinance** — HTTP 404 "Quote not found for symbol." Need remapping (ticker changes?) or pruning from `data/stock_list.csv`.
2. **`data/stock_list.csv` is stale** — 12 tickers appear in earnings/calendar data but aren't in the stock list file: BK, CAG, CPB, CTRA, DAY, EPAM, HOLX, LW, MOH, MTCH, PAYC, POOL. Logs `PROBLEM! X not in my stock_list file` for each, every run, during the GICS sector merge in stage1.
3. **Earnings-date source is unreliable ~20% of the time.** The `ticker.calendar` validation pass in stage1 corrected 97 of 501 tickers' dates this run (several by exactly +7 days — looks like a systematic weekly-offset pattern worth investigating on its own), skipped 4 as bad quarter-rollovers (>78 day jumps), and *separately*, the end-of-stage5 sanity check still flagged 10 stocks with `earnings_date >90 days out` (GOOG, GOOGL, NOW, RJF, DHI, EW, KMI, OTIS, TEL, AMP) — meaning the validation pass isn't catching everything.
4. **`dedup_earnings` removes ~555 duplicate rows every single run** (stage2) instead of once. The print message (`re-run stage1 with incremental=True to clean DB`) says how to fix it at the source, but `full_workflow.sh` always runs with `incremental=False`, so the same duplicates get stripped in-memory every week instead of being deleted from the DB.
5. **`report/calendar_builder.py`'s HTML calendar output looks dead.** `generate_calendar(df)` (called from `stage5.py` with the historical `full_df`) filters on `is_earnings_day == 1`, which is essentially never true for *future* earnings dates — so it prints `Weekly calendar: no scored earnings events in window.` and skips writing `output/weekly_calendar.html`, even though the very next step (`analysis/chart_weekly.py`, reading `output/upcoming_df.parquet` instead) finds the same 114 upcoming events fine and prints/charts them. Worth checking whether anything (dashboard, website) still depends on `weekly_calendar.html` — if so it's been silently stale.
6. **Ordinal-suffix bug in the uncommitted `_print_weekly_table` addition to `analysis/chart_weekly.py`** (this was the file showing as modified in `git status` at session start): `pct = f"{row['peer_percentile']:.0f}th"` always appends "th" regardless of the number, so the console table prints "91th", "61th", "71th", "41th", "2th", "3th", "1th" instead of 91st/61st/71st/41st/2nd/3rd/1st. Trivial fix, same pattern as the ordinal fix already done elsewhere (2026-05-30 digest work) — reuse that logic instead of reimplementing.

**Also noted, not necessarily a bug:** `dedup_earnings` window/behavior and the earnings-date +7-day correction pattern in #3 might share a root cause (weekly-cadence data source lag) — worth checking together if revisiting earnings ingestion.

## 2026-06-22 — Landing page overhaul + pipeline data quality fixes

**Landing page (harbor_webpage):**
- Added stats band: 40% vs 7%, 500+ stocks, 25 years of data (data goes back to 2000)
- Added "How tiers are assigned" section (4 signals, plain language)
- Rewrote feature bullets as benefits; strengthened CTA line
- Recent calls section now dynamic: `script.js` fetches `/recent_calls.json`, renders last 2 weeks with data (all tiers)
- `recent_calls.json` generated by `scripts/gen_recent_calls.py` in breakwater repo, pushed to `/var/www/harbor_webpage/` via full_workflow.sh

**Breakwater pipeline fixes:**
- `data_ingestion/data_utilities.py`: `dedup_earnings(window_days=30)` — removes duplicate earnings rows from yfinance returning slightly different dates for same event. Called in stage2 after loading earnings. Removed 537 duplicates on first run.
- `feature_engineering/post_earnings_stock_features.py`: `engineer_reaction_entropy` now uses `reaction_3d.fillna(reaction_1d)` as best_reaction — includes most-recent incomplete earnings events in entropy computation
- `feature_engineering/scoring_features.py`: entropy uses `.ffill().fillna(0)` as fallback for stocks with no prior entropy at all
- `pipeline/stage2.py`: imports and calls `dedup_earnings`

**Droplet rename:** `/var/www/Breakwater` → `/var/www/breakwater` (lowercase). Updated crontab, systemd service, all local scripts, info file, memory.

**Next:** Dashboard upcoming events view (dashboard is still historical-only — no forward-looking tab)

## 2026-06-09 — Flag fix + social media launch

**Flag fix:**
- `pre_earnings_drift_flag` and `surprise_momentum_flag` were always empty for upcoming events (both only populated on `is_earnings_day==1` rows; `export_upcoming_df` uses latest price row = `is_earnings_day==0`)
- Fixed in `feature_engineering/scoring_features.py`:
  - `engineer_pre_earnings_drift_flag`: now also computes for pre-earnings window rows (1-60 days before earnings) using current `drift_30d` vs stock's historical earnings-day drift distribution
  - `engineer_surprise_momentum_flag`: now forward-fills within each stock after computing on earnings days; earnings-day rows act as reset anchors (even "" resets the carry-forward)
- Also fixed: `surprise_percentage >= 0` now counts as beat in streak (was `> 0`)
- Verified: backtesting metrics unchanged (High Alert 3.82x, HA+Drift 4.94x, avg corr 0.432)
- ORCL now shows: Extended drift + Erratic surprise → `is_high_conviction = True`

**Social media:**
- First weekly post: week of Jun 9, 2026. Chart generated via `report/chart_weekly.py`
- Posts drafted for X and Reddit (see social_media_strategy.md for templates)
- Strategy saved to `.claude/memory/social_media_strategy.md`

## 2026-06-07 — Incremental pipeline + codebase audit

**Incremental pipeline (new):**
- `stage2(lookback_days=N)` toggle: loads only last N days of prices (default=None = full load)
- `stage3(incremental=True)`: skips expanding earnings stats, reads cached values from `full_df.parquet` via `groupby().last(skipna=True)`; cached cols defined in `config.INCREMENTAL_CACHED_COLS`
- `stage4(incremental=True)`: skips functions requiring `abs_reaction_3d`; `earnings_explosiveness_score` + `earnings_explosiveness_bucket` read from cache to avoid score drift when most recent earnings has incomplete reaction window
- `pipeline/incremental.py`: `run_incremental()` auto-detects new earnings via per-stock max-date comparison (DB vs parquet), falls back to `run_pipeline()` if any found
- `cron/cron_ingest.py`: now calls `run_incremental()` after `stage1(incremental=True)`
- Result: **0.8s** incremental vs 80s full run; scores bit-for-bit identical to full pipeline
- Constants in `config.py`: `INCREMENTAL_CACHED_COLS`, `INCREMENTAL_LOOKBACK_DAYS = 90`

**Audit & refactor (same commit):**
- [Full change log](codebase_audit_2026_06_07.md) — every file touched, what changed and why, what was deliberately kept. Read this before debugging any score/pipeline regression.
- Key changes: fixed WMB ticker bug, fixed 3× `.to_numpy()` alignment risk, removed ~34 redundant `df.copy()` calls, stage5 loop pre-grouped by stock, `engineer_timing_danger` deleted, 3 dead file/folders deleted.
- Backtesting lift numbers verified unchanged after all changes (High Alert 3.82x, HA+Drift 4.94x).

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
- Added `incremental_ingest_all_prices_yf(con)` to `data_ingestion/fetch_prices.py` — batch download, incremental from global max date, chunks of 100
- Added `incremental_ingest_all_earnings_dates_yf(con)` to `data_ingestion/fetch_earnings_dates.py` — uses `yf.Ticker().earnings_dates`, skips stocks with future dates already in DB, manual dedup since fiscal_end_date=None bypasses unique index
- `pipeline/stage1.py`: old AlphaVantage calls commented out (19/5/26), new yf functions active
- `config.py`: `STOCKS_END_DATE` now uses `date.today().isoformat()` dynamically; added `from datetime import date` at top
- **NOT yet tested** — killed before smoke test could complete. Test first thing next session.

**Next:** Run smoke test on 3 stocks (AAPL, TSLA, NVDA), verify prices + earnings update correctly, then run full pipeline with incremental=True.

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
