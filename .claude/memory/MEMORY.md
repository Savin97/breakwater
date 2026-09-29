# Session Memory

Read at the start of every session. Newest entry first. At the end of a session, add a
dated entry at the top of the log and rewrite **Current state**. The full text of older
entries is in git history (condensed 2026-09-29 from 1,304 lines + 7 side files).

## How to write for this user — read before replying

**No consultant jargon. Say the plain thing.** Corrected 2026-09-28: "scope it" and
"park it" were both used and neither landed. Write "plan out what it involves before
starting" and "leave it for now". Same for deliverable, workstream, bandwidth, circle
back, align on, surface it: if a word describes a *process* rather than a *thing being
done*, replace it with the thing being done.

- **Do not invent categories.** "Phase 3 deliverable" was made up; the real list is
  P3.1-P3.5 in `audit/PHASE0_AUDIT_REV2.md`. Check the source before naming something.
- **Lead with the concrete artifact, not the statistics.** A table of real events
  (tickers, scores, outcomes) explained the score-ceiling problem instantly after
  several paragraphs of AUCs and confidence intervals had not.

## Current state — updated 2026-09-29 (end of session)

- Two branches, one folder: `master` (production, droplet) and `methodology-rebuild` (the
  rebuild, NOT ready to merge — user's call). Master was merged into the rebuild on
  2026-09-29 (`fc5c2d2`); keep doing that after fixes on master, never the reverse yet.
- Production is still **0.3.1 on the legacy target**; no research result has been ported.
- Everything is committed. Full suite: all pass, 2 Phase 1 golden tests skip (their
  snapshot predates the data).
- Timestamp history is in both DBs: 26,657 / 47,503 rows. Date corrections and the IV
  collection fix are live on the droplet and in the local DB (synced 2026-09-29 13:36 UTC;
  full_df + events_df rebuilt: 25,987 of 45,715 completed events anchored, 56.8%).
- The 18 "stale feed" pending events (DAY, HOLX, BK, EQR, BLDR, TTD, ...) are all
  `inactive` in stock_data (left the index / BK renamed BNY) — not an ingest bug, but
  they still get a pending row in the event frame.
- Next, in order: choose model B vs C → port the
  refit (file list in `research/phase3_refit/RESULTS.md` §10) → historical options data
  for an implied-move baseline (ORATS the lead candidate; Leung & Santoli note in the brain).
- IV live-price fix (`8fcf3b7`): check it's pulled on the droplet and that iv.log looks right.
- Housekeeping DONE 2026-09-29: droplet .bak files, stray CSV and local bak deleted; Labor
  Day IV backup moved to droplet /root/iv_rows_on_closed_days_20260929_130616.parquet.

## Retracted figures — do not reuse

Any lift, hit rate or capture figure in an entry dated **before 2026-09-05** (4.5x, 4.49x,
3.7x, 3.72x, 4.8x, 4.93x, 40%, 52%, 56.8%, 6.9% base, "15 years OOS", "stable 2015-2025")
was measured on the legacy target, which mismeasures BMO events. They are kept only so the
history reads straight. `audit/PHASE0_AUDIT_REV2.md` is the authority; CLAUDE.md lists the
verified replacements.

---

## 2026-09-29 — IV snapshots use the live price (master 8fcf3b7)

- Was using the prior close all day. Now the live price from the options chain, retried
  via `fast_info`, else skip. **Never fall back to the stored close.**
- TMO/WAT were skipped by the old strike rule (holes in Yahoo's put list); fixed.

## 2026-09-29 — Repo cleanup: memory, branches, IV fix

- Memory folded from 1,304 lines + 7 side files into this one file (dated entries kept).
- Deleted branches `announce-timing-on-master`, `fix-earnings-dates` and the
  `breakwater-ivfix` worktree. CLAUDE.md now has the two-branch rule and makes the brain
  suggest-only.
- IV fix on master (`26b9ef8`): no runs when NYSE is closed, ATM strike quoted on both
  sides within 10% of price, skip reasons logged, `join_iv` drops snapshots whose expiry
  precedes the current earnings date (745 of 21,331 rows). Deleted 408 Labor Day rows on
  the droplet (backup parquet in its `backfills/`); it had no Juneteenth rows.
- Merge conflict notes: kept the rebuild's `load_announcement_timing`; the vendor guard
  test now allows exactly the `'massive_benzinga:'` provenance label (master's date
  cleanup compares it).
- CLAUDE.md corrections: `main.py` runs `incremental=True` (since 2026-08-15); stage 3/4
  incremental modes are dead code; functions don't copy their input.

## 2026-09-29 — Late earnings dates corrected on the droplet

- master 83b9fbe: `apply_date_corrections.py` + cleanup keeps the row whose date an
  observed announcement backs (Benzinga > yfinance > later date). Pulled on droplet.
- All 103 corrections applied (KO 2026-02-17 -> 02-10, PEP 2025-02-24 -> 02-04,
  AZO 2024-06-07 -> 05-21); 0 duplicate pairs; timed rows 26,554 -> 26,657.
- Only fixed where Benzinga AND yfinance agree (2021+). Left alone: 242 older
  Benzinga-only disputes (Benzinga pre-2020 is sometimes filing dates / other companies)
  and DOC's history (ticker taken over by Healthpeak).
- Stray `next_earnings_df.csv` untracked in the droplet repo folder.

## 2026-09-29 — Timestamp history verified

- Droplet after the 06:00 ingest: 26,554 timestamped of 47,503 (+8 vs local, nothing lost).
- Local: 25,508 Benzinga rows, 482 stocks, 2011-2026. Matches
  `data/vendor/announcement_seed_earnings_20260910T180412Z.parquet` except 7 pending
  events refreshed by yfinance (same BMO/AMC window). `events_df`: 25,883 of 45,713
  completed events anchored.
- The old "production DB has ~250 timestamps" blocker is resolved.

## 2026-09-29 — IV/EPS collection health check (droplet, data Aug 28 - Sep 28)

- Usable IV data starts 2026-08-28 (4 runs/day since). EPS since 2026-08-10.
- Every weekday Aug 28-Sep 28 has all 4 IV hours + the EPS run. No crashes, duplicates
  or nulls. Only failure: EPS 2026-09-09 lost OMC + LRCX (Yahoo rate limit).
- **Problems found, not fixed (need user decision):**
  1. Runs went ahead on Labor Day 2026-09-07 → 408 IV rows of stale quotes. No holiday check.
  2. `current_price` = the PRIOR day's close from the DB for all 4 runs; ATM strike and
     `expected_move_pct` use it, not a live price.
  3. Earnings dates that move after a snapshot: 1,497 IV rows (9%) carry an old date;
     526 rows / 14 stocks (ACN BSX DAL FDS FDX JBL MU QCOM REGN SO SYF TMUS TRV VZ) used
     an expiry BEFORE the real date, so that IV does not cover the event. EPS: 433 rows.
  4. 28 stocks never get an IV row (TMO, EA, BIIB, NVR, ECL, WAT, AVB, IEX, ...): no
     expiries, or a thin chain with no matching ATM put. Skip reasons are not logged.
  5. WRONG, retracted same day: both tables DO have unique indexes (`iv_snapshots_uq`,
     `eps_estimates_uq`; they show in duckdb_indexes(), not duckdb_constraints()), so
     `ON CONFLICT DO NOTHING` works.

## 2026-09-28 — Announcement times loaded into the droplet DB

- Why it had to be done on the droplet: `full_workflow.sh` rsyncs the droplet DB down
  and overwrites local, so anything loaded locally is wiped. Pushing a DB up is unsafe
  (six cron writers). User's decision: put the timing change on master, load the times
  into the droplet DB once.
- Master `5d82127` pulled on the droplet; `backfills/load_announcement_seed.py` filled
  26,543 (25,515 Benzinga + 1,028 yfinance). The seed is built here by
  `backfills/build_announcement_seed.py` (skips 214 off-by-one dates, 21 identity-hazard
  tickers). Loader is idempotent: a second run fills 0.
- Deploy window: outside the cron runs (06:00 UTC; weekdays 14:45-19:30 UTC).

## 2026-09-28 — Earnings dates on the wrong day (built, not committed/deployed)

- An EPS-backlog run (EARNINGS_RESULT_BACKFILL_DAYS=120, fixed missing EPS 124 → 6)
  inserted 102 rows duplicating existing events at a different date, e.g. KO 2026-02-10
  (yfinance, real) vs 2026-02-17 (stored AlphaVantage-era, late). The cleanup
  `clean_duplicate_earnings_from_db` keeps the LATER row, i.e. the wrong one; it ran
  2026-09-29 and KO is still on 02-17.
- Vs Benzinga (2013+, hazards excluded): 24,410 exact of 25,545; 230 off by 2-30 days
  (177 too late, 216 of them AlphaVantage rows ingested Feb 2026), 177 off by 1 day.
- Fix on branch `fix-earnings-dates` (off master): cleanup keeps Benzinga-timed > any-timed
  > later date (`_date_evidence`); `backfills/apply_date_corrections.py` (move/delete_old,
  idempotent); 179 tests pass. Here: `backfills/build_date_corrections.py` →
  `data/vendor/date_corrections_*.parquet`, **103 corrections**, only where Benzinga AND
  yfinance agree (Benzinga alone is too often wrong pre-2020: LOW 2014, TPL filing
  dates, CB = another company). So only 2021+ is fixed; 242 Benzinga-only candidates left
  alone. Simulated on a droplet copy: clean, 0 duplicate pairs left.
- DOC is still an identity hazard (keeps yfinance/Healthpeak history).

## 2026-09-28 — P3.2 + P3.3 refit: results, not promoted (uncommitted)

`research/phase3_refit/` + `testing/test_phase3_refit.py` → `output/phase3_refit/`.
Write-up in `RESULTS.md`, rules fixed in `PREREGISTRATION.md` first.

- Every feature at the real call time (last close before the report week's Monday).
  Walk-forward OOF 2017-2025, 15,870 events; 2026 YTD holdout 1,328.
- AUC: shipped 0.668, same formula on corrected history 0.706, B (history only) 0.712,
  **C = logistic(log mean prior corrected |r3|, log vol_30d) 0.723**. BMO AUC 0.616 → 0.691.
  2026 holdout: C 0.716 vs shipped 0.637.
- Top-20% capture +2.3pt vs shipped; at top 10% no model beats any other.
- Remove: the 12% cap (687 tied events), entropy (99.1% saturated), lift promotion (worse
  than flagging the same count by score, post hoc), High Conviction (-3.6pt vs rest of
  High Alert).
- Call-time vs eve: no difference. Window-specific calibration and per-window cuts: no gain.
- vol_30d makes flag volume follow the market (14.5-40%/yr); B stays flat at 20-26%.
  **Open product choice: B vs C.** Also open: coefficient source (Benzinga history vs
  yfinance 2020+; yfinance-only gives 2026 AUC 0.7125 vs 0.7131).

## 2026-09-24 — Phase 5B: free price/peer features exhausted (committed 04a5b9e)

`research/phase_5b_new_feature_testing/` (RESULTS.md, PREREGISTRATION.md).
- 29 features / 7 families vs structural + vol_30d (0.7097). Only single features beat
  it robustly, best is the sector peer reaction level (+0.006 AUC). **None improves
  top-10%/20% capture.** Post-hoc pair (+ idio vol): 0.7167, still no capture gain.
- Verdict: ~0.006 AUC left in free data vs ~0.054 from the timing fix. Next rational
  step is point-in-time historical options/analyst data — verify vintage before buying.

## 2026-09-24 — Phases 3-5: the model does not beat a one-line baseline (committed 177be6e)

- **The data got better, and that is the real asset.** Corrected anchoring moves
  P(|r| ≥ 8%) 0.118 → 0.189; 16% of BMO events flip their extreme label, 0.25% of AMC.
  The same baseline scores 0.648 on free data vs 0.702 on corrected history — ~5x any
  feature effect found anywhere.
- **The score adds nothing over its own input:** rho 0.9997 with the raw prior p75.
  Entropy clips to 1.0 on 97.1% of events, so score ≈ 85·min(p75/0.12, 1) + 15. Tier cuts:
  score 73 ↔ p75 ~8.2%, 79 ↔ ~9.0%. Plain expanding mean of prior |reaction| is best (0.7078).
- **Within a stock the score is below chance (0.43)** — it ranks companies, not quarters.
  (Expanding stats are mechanically anti-predictive within stock, so ~0.43 is near the null.)
- The original backtest compared features against each other, never against the raw
  history they are computed from. Different question, not a flawed test.
- BMO on corrected target + history (2019-2025): High Alert 126 → 566 events, hit 36.2%,
  lift 2.25x; still flags 3x fewer than AMC — a threshold problem, fixed in the refit.
- Other findings: the 0.12 ceiling flattens 649 events onto score 100 (ordering at the
  top decided by frame order); the entropy `ffill` reads a FUTURE event of a DIFFERENT
  ticker (nil after 2018, not nil pre-2015); `_missing_aware_lift` advances its global
  prior by row, so events see same-day outcomes; wrapping the score in a GBM loses 0.011
  AUC; pre-event realized vol is the only event-specific signal (+0.011); Benzinga
  expectation features are negative (-0.018, latest-vintage estimates).
- IV, expected move and analyst data have **zero historical coverage** — forward-only
  from 2026. The features most likely to carry quarter-level signal cannot be tested.

## 2026-09-10 — Benzinga source evaluation: accept with conditions

Write-up: `audit/BENZINGA_EARNINGS_AUDIT.md`. Snapshot `earnings_20260910T180412Z`,
`snapshot_sha256 17a5e76693c0c232…`. 296,334 records, raw pages in gitignored
`data/vendor/`, 48 tests in `testing/test_massive_earnings.py`.
1. **`date.desc` pagination silently loses 39%.** Completeness proved by re-acquiring
   year by year: identical 296,334 ids.
2. **`time` is NY local wall clock**, not the documented "EST" (tested against yfinance).
3. yfinance timestamps are hour-rounded, so exact-minute agreement is meaningless.
   Window agreement 99.63%; only 17 genuine contradictions.
4. **Ticker joins are unsafe** (BF-B, BRK-B spellings; company_name not point-in-time;
   SNDK reused). `00:00:00` = unknown (7.5%), never BMO.
- Usable timing: 91% by 2015, 99.8% from 2019; ≥80% of the universe has 28 prior timed
  events only at year end 2019.

## 2026-09-06 — Multi-week predictions (`analysis/predictions_range.py`)

- Model 0.3.1 as shipped, legacy target only (user's ask). History is a retro-score,
  not an archive. Upcoming is shown twice: current and `*_pre_audit` (the one-event-stale
  published call). **The NaN skipping is what is being reproduced — never "fix" it.**
- Writes `output/predictions/`, never `get_run_output_dir()`. Details in CLAUDE.md.

## 2026-09-05 — Phase 2: announcement timing + parallel anchored target (+ 3 review fixes)

Commits `0ecec2c`, `a4475a9`, `a3bd276` on `methodology-rebuild`. Mechanism and rules are
in CLAUDE.md. Things a successor must not undo:
1. **AMC anchored == legacy, bit for bit** where the ticker has every session. That is the control.
2. **Never infer BMO/AMC from price** (audit rev-1 did; every number was circular).
3. **Never fabricate a timestamp; never auto-roll a non-session date.**
4. `resolved_events()` is the only gate into a corrected calibration.
5. Anchors are positions on the market-session grid, never the ticker's own rows.
6. One clock: naive NY wall time via `now_ny()`. The host runs UTC+3 (Israel); a
   host-clock stamp would freeze schedules into history permanently.
- Still open: 24 events on sessions where the ticker has no price row (mostly the
  2026-05-19..21 ingestion hole) — an ingestion bug, counted, never rolled.
- `get_next_earnings_dates()` (~line 560) labels `datetime.now()` as NY — same bug class,
  dead-ish legacy helper, left alone.

## 2026-09-05 — Phase 1: event frame, upcoming-score staleness fixed

- `groupby("stock").last()` skips NaN per column, so 100% of shipped upcoming calls were
  one earnings event stale (audit §Q4). Fix: `pipeline/events.py`, one row per event plus
  one pending row per stock; consumers read `is_pending == 1`.
- Rules: **never put a pending row in the daily frame**; a pending row reads the entropy
  `ffill` chain without updating it (letting it contribute moved 385 scores).
- Found along the way: `calendar_builder` was dead (rendered zero events every run);
  `save_predictions.py` had a live `NameError`.
- Open: **15 stale price feeds** (AVB, BK, CAG, CPB, CTRA, DAY, EA, EPAM, EQR, HOLX, LW,
  MOH, MTCH, PAYC, POOL) still carry a future earnings date with prices stopping as early
  as 2026-02-03. `scoring_slice.py` and `INCREMENTAL_CACHED_COLS` now redundant; clean up later.

## 2026-09-04 — First full end-to-end run; one work week per email

- `full_workflow.sh` ran end to end: pipeline → PDFs → parquets to droplet → digest →
  user confirmed the email arrived.
- **Product rule (user): every email covers exactly one whole Mon-Fri work week.** More
  must be explicitly asked for; weekend dates excluded. Monday run = this week, any other
  day = next week (running Tuesday means Wed-Fri are never emailed — keep Monday the habit).
  `--current-week`, `--weeks N`.
- `work_week_window()` is shared by the digest and the predictions snapshot — they drifted
  once (Friday run emailed ORCL/ADBE/COO/CPRT and recorded none). Do not re-inline it.
- **Backtest against the view `predictions_first_call`** (renamed from
  `predictions_week_open`), keyed on (stock, earnings_date).

## 2026-09-02 — Scope decision: weekly only

- **User: "weekly is enough."** No daily scoring on the droplet. The droplet only runs
  ingest + IV/EPS crons and serves Streamlit. The weekly local run produces everything
  and sends the digest. Do not resurrect droplet scoring without the user asking.
- Predictions stay local: stage5 writes git-tracked `db/predictions.duckdb`. One writer.
- `pipeline/incremental.py` has no caller and a latent `TypeError` (line 27). Deletable.
- Memory measurements: the full pipeline peaks at ~5.5 GB (frame 1.9 GB); float32 would
  save ~734 MB — `testing/calibration.py` is the gate. A droplet big enough to run the
  pipeline is an 8 GB box.
- EPS results were never backfilled: ingestion is INSERT-only and the skip rule hid last
  quarter's result. Fixed with an UPDATE pass + `EARNINGS_RESULT_BACKFILL_DAYS`.
- **Trap:** emulating `.last()` with an ffill manufactures signals (invented an "Extended
  Beat Streak" on ADSK). Leave flag columns NaN off earnings days. **Do not fill them.**
- **Repo conventions (user):** `pipeline/` holds only pipeline stages; stages read as named
  function calls; droplet and local must produce identical results.
- **The repo is PUBLIC** (`Savin97/breakwater`).

## 2026-09-01 / 08-31 — cron ingest-only, predictions table, High Conviction bug

- `cron/cron_ingest.py` is ingest-only (`stage1(incremental=True)`); 285 MB, fits the droplet.
- **stdout is block-buffered under cron and lost on SIGKILL** — only stderr lines in
  `/var/log/breakwater_ingest.log` show real progress.
- Merged (not rebased) on purpose: `config.py` cites `f3dd1e2` and predictions rows store
  `git_commit`; a rebase orphans both.
- Predictions: `analysis/save_predictions.py`, own DB `db/predictions.duckdb`
  (un-ignored — it is the only copy), upsert keyed (stock, earnings_date,
  prediction_asof_date). Archive starts 2026-08-31.
- `is_high_conviction` was silently False on ~98% of rows (NaN bucket off earnings days).
  **The suite passed before and after** — the fixture had zero High Alert rows. Lesson:
  check a new test fails against the bug before trusting a green suite. Other
  fixture-driven tests may be vacuous; not audited.
- MODEL_VERSION renumbered to 0.3.1 (old "1.0" = 0.1, "1.1" = 0.2; mapping in config.py).
- Droplet has **no backups**.

## 2026-08-29 — Lift promotion, logging, parallel yfinance fetches

- Lift promotion (`f3dd1e2`): Normal → Elevated at lift ≥ 1.5, → High Alert at ≥ 3.0.
  Measured [retracted] capture 43.6% → 56.8%; multiplying score by lift dropped top-decile
  lift 3.70x → 2.98x, so lift is a gate, not a multiplier. (The 2026-09-28 refit found
  promotion worse than flagging the same count by score.)
- Logging: `utilities/logging_utilities.setup_logging()`, stdout, `LOG_LEVEL` and
  `NOISY_LIBRARIES` in config.py (weasyprint needs ERROR). `main.py` deliberately untouched.
  Rule: logging = what the program is doing; print = the output you ran it to read.
- Report logo pointed at a file that never existed; every PDF had a broken image. Now
  `report/img/breakwater_logo.png`.
- Parallel fetches: network work in a thread pool, DB writes stay sequential (DuckDB
  connections aren't thread-safe). 11.7x faster. If SSL resets cluster, drop
  `YFINANCE_MAX_WORKERS` to 4-5.
- The note here that `main.py` runs `incremental=False` (the paid path) was already wrong:
  it has passed `incremental=True` (yfinance) since `117ee16`, 2026-08-15. Corrected 2026-09-29.
- User preference: constants go in config.py with clear names.

## 2026-08-02 → 08-28 — Droplet cron incidents (from infra notes)

Repo at `/var/www/breakwater`, deploy = push + `git pull` on the droplet. Website repo:
local `/home/Michael/projects/harbor_webpage`, server `/var/www/harbor_webpage`, GitHub
`Savin97/harbor_webpage`.
1. **Invoke cron scripts as modules**: `cd /var/www/breakwater && .venv/bin/python -m
   cron.<module>`. By file path, `cron/` lands on `sys.path` → `No module named 'config'`
   (IV stalled from 2026-06-26, EPS never ran).
2. **Never positional `SELECT *` inserts** against a table whose column order can drift
   (`snapshot_hour` ended up in a DATE column).
3. **DuckDB allows one writer** — never schedule two DB jobs on the same minute.
4. **Droplet TZ is UTC and Ubuntu's cron ignores `CRON_TZ`.** Times are written in UTC
   inside 14:30-20:00 UTC (market hours under both EDT and EST). Don't change the system
   timezone: `snapshot_hour` / `ingested_at` history is UTC.

```
45 14 * * 1-5   cd /var/www/breakwater && .venv/bin/python -m cron.cron_eps_estimates >> /var/log/breakwater/eps_estimates.log 2>&1
0  15 * * 1-5   ... -m cron.cron_iv >> /var/log/breakwater/iv.log 2>&1
30 16 * * 1-5   ... -m cron.cron_iv
0  18 * * 1-5   ... -m cron.cron_iv
30 19 * * 1-5   ... -m cron.cron_iv
0 6 * * *       ... -m cron.cron_ingest >> /var/log/breakwater_ingest.log 2>&1
```
Pre-fix crontab backup: `/root/crontab.backup.20260828`.
- Ticker lifecycle (2026-08-10/15): `stock_data.status` reconciled against Wikipedia each
  run; renames in `data/ticker_renames.csv` (BK → BNY, SATS → ECHO).
- **`full_workflow.sh` overwrites the local DB with the droplet's** — it silently wiped a
  local schema migration twice. Migrate the droplet DB, then sync down.
- Still open: `eps_estimates` gap Aug 6-7 never explained; stale untracked
  `data/breakwater.duckdb` + `next_earnings_df.csv` on the droplet.

## 2026-06-23 — Reddit/X comment playbook (from marketing notes)

- Comment with real data; never hype, never trash a name; no directional calls — the
  model flags tail risk, not direction. Soft plug: "I ran it through a risk model I use".
- Data to pull: tier, peer percentile, recent 3d reactions, beat rate, beat-but-fell
  rate, expected move (IV), `iv_vs_hist_ratio`, `pre_earnings_drift_flag`.
- Angles that landed: "beat but stock fell" (NKE 73% beat rate, fell after 45% of beats);
  options under/overpricing via `iv_vs_hist_ratio` (FDX 0.72, MU 1.63); "Compressed drift";
  "coiled spring" (calm quarters + Compressed + High Alert); macro-catalyst pushback with
  real price data; flag tail risk when someone is on margin.
- Examples written that day: MU, FDX, NKE. Not in universe: SOFI, ELF, CELH, RVLV.

## 2026-06-09 — Social media strategy (from marketing notes)

- X: primary, 3x/week (Mon/Tue weekly watch with chart, Wed/Thu outcomes, Fri optional).
  Reddit: comment in any thread on a stock reporting that week — r/stocks, wallstreetbets,
  investing, StockMarket, thetagang, options, SecurityAnalysis, Daytrading.
- **Content rules:** no model lift numbers, calibration stats or methodology; no mention
  of ML/models; not options or trading advice; yes to factual history (avg move,
  frequency); "I track earnings tail risk across the S&P 500" framing; always link
  harbor-markets.com, never the raw dashboard.
- Weekly: `python report/chart_weekly.py` → `output/weekly_chart.png`, pull stats, post,
  then comment on Reddit threads.
- Template: `★ $TICKER (High Alert) — avg ~X% move, Y of 8 quarters moved >5%. [context]`
  … `Full weekly tracker → harbor-markets.com`.

## 2026-06-07 — Codebase audit + incremental pipeline

- Removed per-function `df.copy()` from ~34 functions (stage entries are the copy
  boundary), fixed three positional `.to_numpy()` assignments, deleted dead code. ~120s → ~80s.
- User asked to keep: reaction_1d/5d, is_up/down, z-score columns, the p90 bucket chain,
  AlphaVantage ingestion functions.
- Incremental pipeline built (`INCREMENTAL_CACHED_COLS`); its "bit-for-bit identical" claim
  was never true for the flags. Nothing runs it now (see 2026-09-02).

## 2026-05-17 → 07-27 — Early sessions

- 2026-07-27: `yf.download(end=...)` is exclusive → `end = today + 1`. Still open then:
  `data/stock_list.csv` stale; yfinance dates wrong ~20% of the time, often +7 days.
- 2026-06-01: `testing/calibration.py` built. [Retracted] High Alert 40.2% vs 6.9% base,
  HC 52.4%. Product framing: sell "which 15-20 events matter this week"; don't chase false
  negatives in Normal. `cv_website` renamed `harbor_webpage`.
- 2026-05-31: digest and reports were reading past events for upcoming dates — fixed
  (the `.last()` approach later turned out one event stale; see Phase 1).
- 2026-05-30: IV (`expected_move_pct`, `atm_iv`) into reports; weekly digest built;
  stage5 auto-selects High Alert + Elevated. Layout frozen.
- 2026-05-27: window grid search chose rolling-28 for p75 ([retracted] 4.49x avg lift).
  The 2026-09-24 work found a plain expanding mean does better.
- 2026-05-19: AlphaVantage cancelled; yfinance is the active ingest path.
- 2026-05-18: HC (High Alert + drift flag) chosen over surprise-based variants
  ([retracted] 4.93x); memory moved into `.claude/memory/`.

## 2026-05-30 — Product direction and build list

- Target: retail/prosumer options traders (straddle/strangle buyers, hedgers) — not
  institutional quants. Sells move *magnitude*, not direction. Price $50-200/month.
- Live: harbor-markets.com (landing page), harbor-markets.com/breakwater (Streamlit).
- Revenue path: Stripe payment gate on the dashboard; the email digest as delivery.
  **Not built yet.**
- The original pitch ("4.5x lift, 15 years OOS", "lead the landing page with it") is
  **retracted** — P4.1 retires every published figure until Phase 3 is ported.
- Deferred: IV signal validation (`iv_vs_hist_ratio`, time-aware join) until IV history
  accumulates; weekly chart polish. Not for now: SHAP, sector models, API, portfolio views.
