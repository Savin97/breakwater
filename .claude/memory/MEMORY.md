# Session Memory Index

This file is read by Claude Code at the start of each session to restore context.
Entries are updated at the end of each session. Most recent first.

---

- [Social media strategy](social_media_strategy.md) — platforms, cadence, content rules, weekly workflow (added Jun 9, 2026)
- [Reddit/X marketing playbook](reddit_marketing_playbook.md) — comment tone, data angles, soft Breakwater plug, real examples from Jun 23 2026 (MU, FDX, NKE, NOW)

## 2026-09-05 — PHASE 2 REVIEW FIX #3: timezone convention (commit `a3bd276`)

**Third external review of `a4475a9` found one remaining Phase 2 correctness bug: the
observation timestamp had no fixed timezone convention. Fixed and committed, NOT yet
pushed. Still awaiting re-review; do not start Phase 3.**

The bug: `announce_ts_ny` is naive NY wall clock, `announce_ts_observed_at` was
`datetime.now()` — the HOST's clock. This machine runs UTC+3 (Israel), 7h ahead of NY, so
a schedule observed hours before an announcement read as LATER than it, was classified
post-event, and would have been frozen into the historical record forever.

- New `utilities/time_utilities.py`: `now_ny()` (zoneinfo America/New_York, naive),
  `to_ny_wall_clock`, `utc_to_ny_wall_clock`, `MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS = 19`.
- `fetch_one_earnings_dates` stamps observed_at with `now_ny()`. `ingested_at` keeps its
  legacy machine-local convention deliberately (operational column, not in the comparison).
- The `ingested_at` fallback is no longer compared raw: widened by 19h (UTC+14 vs EST)
  into a host-independent lower bound, so a row freezes only if it was post-event under
  EVERY host tz. Error runs only toward "still a schedule" — recoverable; a false
  post-event classification is not.
- +78 tests (263 total). Host tz simulated for real via `TZ` env + `time.tzset()`, five
  zones x pre/post-event x both US DST transitions; the old host-local stamp is pinned as
  the regression; static test forbids `datetime.now()` for observed_at.

**No data repair needed** — all 12,068 stored observed_at values came from the backfill's
fixed pull date (2026-09-05), never from a host clock. `audit.phase2_diagnostics` output
byte-identical; no anchoring/target/score/threshold change.

Still open from the previous round (unchanged): the 24 ingestion-gap events, 25.0%
timestamp coverage, and `get_next_earnings_dates()` line ~560 labels `datetime.now()` as
tz-aware NY — same bug class, but it is a dead-ish legacy helper feeding no timing column,
so it was left alone to keep the diff to the reviewed defect.

---

## 2026-09-05 — PHASE 2 REVIEW FIXES (commit `a4475a9`, branch `methodology-rebuild`)

**External review of `0ecec2c` found two target-integrity issues. All four items are
implemented, committed and PUSHED (`d300a89..9d52648` -> `origin/methodology-rebuild`).
Still NOT merged to master; awaiting re-review of these fixes. Do not start Phase 3 and do
not touch model/scoring parameters until it clears.**

Nothing in scoring moved: no threshold, weight, lift gate, legacy reaction column,
BMO/AMC definition, intraday rule or entropy change. `assert_completed_parity` clean.

1. **Market-session grid anchoring.** Anchors/endpoints were positional offsets in each
   ticker's own price rows, which silently absorbs a missing row (`.shift(-3)` over a
   3-session hole spans 6; a BMO event missing D-1 anchors to D-2). Now positions on
   `market_session_grid(daily_df)`, and the ticker must have a row on the EXACT required
   dates. New per-horizon `reaction_{k}d_anchored_status`. Impact measured: all 11,417
   resolved anchors unchanged, 3 outcome values withdrawn (AMAT 3d/5d, CSCO 5d — the
   2026-05-19..21 ingestion hole). AMC bit-identity now asserted on the 4,832 gap-free
   AMC events; the 10 that differ are enumerated in the diagnostics §5.
2. **`announce_ts_observed_at`.** A timestamp observed before the event is a SCHEDULE and
   was being frozen forever by the NULL-only backfill. `refresh_announcement_timestamp`
   replaces a schedule with a strictly newer observation, never overwrites a post-event
   observation, falls back to `ingested_at` when observed_at is NULL, and refuses when
   both are NULL. Backfill stamps the 2026-09-05 pull date and self-migrated the 12,068
   seeded rows (11,582 frozen as post-event, 486 still schedules).
3. **Gate split.** `anchor_resolved_events()` = anchoring control slice.
   `resolved_events(events, target="abs_reaction_3d_anchored")` = the calibration gate,
   now requiring the target non-null. The 11,417 vs 11,411 question is accounted for
   exactly: 11,417 anchors -> 11,412 with a 3d target -> 11,410 paired with the legacy
   column (2 BMO events at the right edge whose corrected window closes a session
   earlier). Printed every run in diagnostics §3.
4. **Doc correction.** "The target, not the model, was wrong" removed everywhere. The
   supportable claim is: the legacy target is proven wrong for BMO; model validity and
   incremental value remain UNESTABLISHED pending the Phase 3 rebuild and a
   competitive-baseline validation.

Tests: 185 pass across test_announcement_timing / test_event_frame / test_pipeline.
`audit/PHASE2_DIAGNOSTICS.md` regenerated. `output/events_df.parquet` rebuilt.

### Files touched (9)
`feature_engineering/announcement_timing.py` (rewritten resolver + gate),
`pipeline/events.py` (carries `announce_ts_observed_at`),
`utilities/db_utilities.py` (new column + loader prefers newest observation),
`ingestion/fetch_earnings_dates.py` (`refresh_announcement_timestamp`,
`EARNINGS_INSERT_COLS` +1), `scripts/backfill_announcement_timestamps.py` (stamps +
self-migrates observed_at), `audit/phase2_diagnostics.py`, `audit/PHASE2_DIAGNOSTICS.md`,
`CLAUDE.md`, `testing/test_announcement_timing.py` (+38 tests, 78 total).

### To resume
```bash
.venv/bin/python -m pytest testing/test_announcement_timing.py testing/test_event_frame.py testing/test_pipeline.py -q
PYTHONPATH=. .venv/bin/python -m audit.phase2_diagnostics
# rebuild events_df without re-ingesting or re-running stage5's reports:
PYTHONPATH=. .venv/bin/python -c "import pandas as pd; from pipeline.events import build_and_score_event_frame, load_pipeline_announcement_timing; ev=build_and_score_event_frame(pd.read_parquet('output/full_df.parquet'), load_pipeline_announcement_timing()); ev.to_parquet('output/events_df.parquet', index=False)"
```

### Open, NOT fixed here (deliberately out of scope)
- **24 events sit on a session the market traded but the ticker has no price row for**
  (diagnostics §7). Mostly one three-session ingestion hole, 2026-05-19..21, plus
  2026-06-16/24/25 and two 2006/2008 SPGI rows. This is an INGESTION bug, not a timing
  bug. It is counted, never rolled. Fixing the price feed would restore the 3 anchored
  outcomes withdrawn by the grid fix and add ~24 events to the frame.
- Timestamp coverage is 25.0% of completed events and effectively zero before 2020
  (95%+ from 2021). That is the binding constraint on any Phase 3 walk-forward re-fit —
  it cannot claim a window the timestamps do not cover.
- `announce_ts_observed_at` refresh is keyed on (stock, earnings_date), so it corrects
  the TIME of an event whose calendar date is unchanged. A provider correction that moves
  the DATE is a different row, handled by the existing placeholder-clearing DELETE.

---

## 2026-09-05 — PHASE 2: verified announcement timing + parallel anchored target

**Branch `methodology-rebuild` (renamed from `methodology-rebuild-phase-1`, old remote
deleted). Phase 1 approved through `8ce659c`; Phase 2 is commit `0ecec2c`. Both pushed to
`origin/methodology-rebuild`. NOT merged to master — Phase 2 is awaiting external review,
same posture Phase 1 had. Do not start Phase 3 until it clears.**

### What Phase 2 does — and deliberately does NOT do
Adds a **parallel** corrected outcome. `reaction_{1,3,5}d` / `abs_reaction_3d` are
untouched and remain the production target; they are the CONTROL. Nothing switches over
until the historical chain is rebuilt and 73/79 etc. re-fit — that is Phase 3.
Explicitly deferred and untouched: 73/79, 0.85/0.15, the 0.12 ceiling, lift gates and
prior strength, same-day global-lift ordering, the cross-stock entropy ffill.

### The mechanism
- `feature_engineering/announcement_timing.py` — window classification and anchoring.
  Window is a pure function of the OBSERVED NY clock: `<09:30` BMO, `>=16:00` AMC, else
  INTRADAY, no timestamp = UNKNOWN. Anchor = last close strictly BEFORE the announcement
  (AMC → close(D), BMO → close(D-1)); anchored reaction spans k post-announcement
  SESSIONS via price-row positions, never calendar arithmetic.
- `earnings.announce_ts_ny` + `announce_ts_source` (naive NY local; self-migrating
  ALTER TABLE). `ingestion/fetch_earnings_dates.py` no longer throws the timestamp away
  — that `.dt.date` is the original sin. Both writers now name columns explicitly instead
  of `SELECT *`, and there is a per-row UPDATE that backfills the timestamp onto rows the
  dedup filter would otherwise skip forever.
- `scripts/backfill_announcement_timestamps.py` — one-time, idempotent seed of
  `audit/provider_timestamps.parquet` into that column. **This is how the audit artifact
  enters production: once, as a seed. No `pipeline/` module reads that parquet, and a
  test enforces it.** Filled 12,068 of 12,269; 201 seed events are not in our DB.
- `pipeline/events.py` — `build_event_frame(daily_df, timing_df=None)`. Default None
  means "no observed timing → everything UNKNOWN/unresolved". The PIPELINE loads it
  explicitly (`load_pipeline_announcement_timing`), so production can't acquire timing by
  accident and unit tests can't acquire a database by accident.

### Numbers (audit/PHASE2_DIAGNOSTICS.md, reproducible via `python -m audit.phase2_diagnostics`)
- Windows on 45,701 completed: BMO 6,575 / AMC 4,842 / INTRADAY 98 / UNKNOWN 34,186.
- **Resolved 11,417 (25.0%).** Coverage ~96% for 2021-2026, ~26% in 2020, ~0 before.
  That is the binding constraint on any Phase 3 walk-forward.
- P(|reaction_3d| >= 8%): ALL 0.128 -> 0.204, **BMO 0.041 -> 0.173**, AMC 0.246 -> 0.246.
  Reproduces `audit/verified_timing_analysis.py` to 3 dp — the point of the exercise.
- BMO |reaction| is ~1.75x larger at every quantile (p10 through p99), mean 0.0276 ->
  0.0482. A uniform level shift, not a tail artifact.
- Unresolved: 34,186 no_timestamp + 98 intraday. 4 resolved events have an incomplete
  forward window.
- **24 price-gap events** (market traded, ticker has no row) — mostly 2026-05-19..21 plus
  SPGI 2006/2008. An INGESTION bug; still not fixed, still not rolled. 299 non-session
  dates, 972 outside the ticker's own price history.

### Things a successor must not undo
1. **AMC anchored == legacy, bit for bit.** That is the control proving the anchoring code
   moves nothing on its own. If it ever fails, the anchoring is wrong, not the legacy.
2. **Never infer BMO/AMC from price.** `test_6_the_classifier_never_touches_price` walks
   the AST of `classify_announce_window` and fails on any price-derived name. Rev-1 of the
   audit made exactly this mistake and every corrected number it published was circular.
3. **Never fabricate a timestamp.** AlphaVantage history is date-only and stays NULL.
4. **Never auto-roll a non-session date** (audit Q6) — weekend/holiday and ingestion-gap
   have different causes and rolling hides both.
5. `resolved_events()` is the ONLY gate into a corrected calibration. Unresolved events
   carry NaN anchored targets so they cannot enter one by accident.

### Green
- Phase 1 completed-event parity `{}` on all 45,701 events, with AND without timing.
- Every `PARITY_COLS` value identical with and without timing — timing does not reach the
  score at all.
- `python -m testing.calibration` still byte-identical to `audit/phase1_golden/calibration_pre.txt`.
- Pending drift flags still 0 diffs vs the legacy golden; High Conviction still 5.
- **160 tests pass** (107 + 53 new in `testing/test_announcement_timing.py`, one per
  invariant plus non-vacuity guards).

### Next (Phase 3 — nothing about the corrected model is claimable before it)
Rebuild the chain from anchored outcomes in the §Q3 order, then re-fit 73/79, `LIFT_TO_*`,
`LIFT_PRIOR_STRENGTH`, the 0.12 ceiling and the 0.85/0.15 weights, then calibrate
STRATIFIED by announcement window. Decide the pre-2020 policy: a shorter honest window vs
paid historical timestamps. **Do not re-fit on inferred labels.**

## 2026-09-05 — PHASE 1 HANDOFF: event frame landed, upcoming-score staleness fixed

**Branch `methodology-rebuild-phase-1`, commit `e197506` (base `09e7861`). NOT merged to
master. NEXT STEP IS EXTERNAL REVIEW OF THIS COMMIT — do not start Phase 2 until it clears.**

### What was wrong
Every forward-looking consumer recovered upcoming state with
`df.sort_values("date").groupby("stock").last()`. `GroupBy.last()` skips NaN *per column*,
and the scoring columns are NaN off earnings days, so the row it returned took
date/earnings_date from today but score/tier/lift from the stock's **last completed
event**. 100% of shipped upcoming calls were exactly one earnings event stale
(audit/PHASE0_AUDIT_REV2.md §Q4).

### What was implemented
- `pipeline/events.py` — the event frame: one row per earnings event, every completed
  event plus **one pending row per eligible stock** (that stock's final daily row, outcome
  blanked, `is_pending=True`), written to `output/events_df.parquet`. Built in a new
  stage 4b between stage4 and stage5.
- `feature_engineering/event_features.py` — the event-level cores. The daily pipeline AND
  the event frame both call them, so historical and pending events cannot drift apart.
- Consumers now read `is_pending == 1` explicitly: `streamlit_export.py`,
  `save_predictions.py`, `report_builder.py` (also dropped its second stale source,
  `earnings_df.iloc[-1]`), `calendar_builder.py`.
- `score_asof_date` added to the event frame, `upcoming_df.parquet` and the predictions
  archive.

Every `shift(1)` kept verbatim. A pending row carries a NaN outcome and sorts last within
its stock, so it cannot touch any completed row, while its own
`shift(1).expanding()/rolling()` spans all completed prior events including the most
recent — **that is the fix, not a new statistic.** No model calibration changed.

### Two rules for anyone continuing this
1. **Never put a pending row in the daily frame.** It would corrupt `merge_asof`, the
   per-stock rolling price windows and the `groupby("date")` cross-sectional ranks.
2. **The cross-stock `reaction_entropy.ffill()` in the score is order-dependent** and
   pending rows sit between one stock's last event and the next stock's first. Letting
   them contribute moved 385 completed scores. A pending row now reads that chain without
   updating it (`entropy.mask(is_pending).ffill()`). Cost real time to find; the parity
   assertion is what caught it.

### Parity / tests — all green
- Daily frame **byte-identical**: 87 cols × 2,914,315 rows, after parquet round-trip.
- All 22 history-dependent columns **identical on all 45,701 completed events**
  (`completed_parity_report` → `{}`; asserted on every pipeline run).
- `python -m testing.calibration` output **identical** to the frozen baseline.
- **103 tests pass** (73 existing + 30 new in `testing/test_event_frame.py`, covering all
  12 required invariants with non-vacuity guards).
- Baseline evidence + regeneration procedure: `audit/phase1_golden/README.md`.

### Shipped effect
8 of 495 upcoming final tiers change (ADBE, ISRG, NXPI, EA, CMG, WDC, PTC, WSM);
High Conviction 5 → 12. NXPI moves on a byte-identical score — its stale lift 1.400 sat
under the 1.5 gate, the correct 1.535 clears it.

### Found along the way (fixed here, worth knowing)
- `calendar_builder` was **dead**: it selected its forward window out of
  `is_earnings_day == 1` rows, i.e. completed events with past dates, so it rendered zero
  events every run. Now reads pending rows; window opens today.
- `save_predictions.py` had a live `NameError` (`week_end` vs `window_end`) on its success
  path — stage 5 could not complete.
- `report_builder` needed a guard: a fresh tier can be one the stock has never held, which
  `KeyError`'d the bucket-stats lookup. Reindexed over all three tiers.

### Known remaining issues (NOT addressed — deliberate)
- **`score_asof_date` exposed 15 stale price feeds**: AVB, BK, CAG, CPB, CTRA, DAY, EA,
  EPAM, EQR, HOLX, LW, MOH, MTCH, PAYC, POOL — still carrying a future earnings date with
  prices stopping as far back as 2026-02-03. Reported, not dropped. Investigate.
- **Two secondary fields also de-staled** (follows from the shared-core invariant):
  `surprise_momentum_flag` changed on 172/500; `pre_earnings_drift_flag` on 26 and
  `is_high_conviction` on 7. All the flag changes are **>60 days out** (min 61) — the old
  daily branch only fired within `days_to_earnings.between(1, 60)`. **Inside 60 days,
  which is every near-term deliverable, the flags are identical.**
- Cross-stock `reaction_entropy.ffill()` defect — still there on purpose.
- `scoring_slice.py` and `INCREMENTAL_CACHED_COLS` — still there on purpose. The event
  frame makes both redundant (5 MB vs 323 MB), but clean up only after this is proven.
- Announcement-time / BMO-AMC correction — **Phase 2**, not started. See
  `audit/PHASE0_AUDIT_REV2.md` for the plan and for what may NOT be claimed until the
  historical chain is rebuilt (every published lift figure is overstated).
- Predictions archive untouched: the 10 pre-audit rows keep `score_asof_date = NULL`,
  which is itself the marker that their score came from the previous completed event.

## 2026-09-04/05 — First full end-to-end run; digest + predictions scoped to a work week

**THE CHAIN WORKS END TO END.** `full_workflow.sh` ran: pipeline -> 5 PDFs -> parquets
rsynced to droplet -> digest sent -> **user confirmed the email arrived**. That was the
last unverified link (`_send` and the attachment path had never executed).

### Product rule the user stated — do not violate it
**Every email covers exactly one whole Mon-Fri work week.** More than that must be
explicitly asked for. Weekend earnings dates "make no sense" and are excluded.
- Default: the next COMPLETE work week. Monday run = this week; any other day = next
  week's Mon-Fri. **Consequence the user accepted:** running Tuesday means Wed-Fri of
  that week are never emailed or recorded. Argues for keeping Monday the habit.
- `--current-week`: this week's Mon-Fri whatever day it is run.
- `--weeks 2`: two whole blocks.

### What was built
- `utilities/data_utilities.work_week_window(today, weeks, current_week)` — **one helper,
  imported by BOTH the digest and the predictions snapshot.** They computed windows
  separately before and drifted, which is the bug below. Do not re-inline it.
- `analysis/save_predictions.py` and `cron/cron_weekly_digest.py::_select_stocks` both
  select on it. Digest gained `--weeks` / `--current-week` argparse.
- save_predictions clamps the lower bound to today, so `--current-week` on a Friday emails
  the whole week but records only what has not reported — no hindsight in our own backtest.
- The table stays the WIDER record: all tiers, while the email shows High Alert/Elevated.

### The bug this fixed (found by inspecting the first real run)
The Friday 2026-09-04 run **emailed ORCL, ADBE, COO, CPRT and recorded none of them.**
The digest used a rolling today..+7 window; save_predictions used today..Sunday. On a
Friday that is Sep 4-6, when nothing reports. Silent — the table simply had no rows.
Verified fixed by simulation: "emailed but not recorded: none".

### View renamed and re-keyed
`predictions_week_open` -> **`predictions_first_call`**, `DISTINCT ON (stock,
earnings_date)` instead of including `run_week`. A rolling window lets a Thursday run and
the following Monday see the same event from two different run weeks, which under the old
key produced two rows for one call. Old view is explicitly DROPped. `run_week` survives as
a column recording WHEN the call was made — no longer a grouping key.
**Backtest against `predictions_first_call`.**

### Backfill's first production run — it worked
0-14 day events: 100% missing EPS -> **24%**. 15-30 days: 100% -> **2%**.
**But 31-60 days is still ~85% missing (306 events)** — `EARNINGS_RESULT_BACKFILL_DAYS=30`
bounds it, so the pre-existing backlog outside 30 days was not swept. It will clear on the
old slow path over ~2 months, or immediately with one run at 120 then set back.

### Still open
- `pipeline/incremental.py` — callerless, latent `TypeError` on line 27.
- dtype experiment (float32, ~734 MB of 1910 MB frame; calibration is the gate).
- 7 stocks with `earnings_date` >90 days out — **probably NOT a bug**, they are
  off-calendar fiscal years whose next report is genuinely 97-112 days out. The
  `export_upcoming_df` warning threshold of 90 days is just tight. Raise it or drop it.
- Brain: still untouched, still undecided.

## 2026-09-02 — Slice landed, digest fixed, and the product scope narrowed to weekly

**SCOPE DECISION, made by the user this session — read this before planning anything:**
**"weekly is enough."** No daily scoring on the droplet. The weekly local `full_workflow.sh`
run produces everything; the digest is sent from that run. This retires the whole
"droplet generates predictions automatically" thread that the last three sessions were
building toward. Do not resurrect it without the user asking.

### Consequences of that decision
- **Predictions stay local.** `analysis/save_predictions.py` already runs in stage5 on the
  weekly run and writes git-tracked `db/predictions.duckdb`. One writer, git as archive.
  The "truth lives on the droplet, local copies it down" design is **moot — do not build it.**
- **Nothing runs the incremental path any more.** `cron_ingest` is ingest-only, the droplet
  does not score, `main.py` runs the full pipeline. `pipeline/incremental.py` has **no caller
  anywhere** and still has a latent `TypeError` on line 27 (`run_pipeline()` called with no
  args against signature `run_pipeline(incremental)`). Deleting it is defensible.
- The droplet needs **no** SMTP, no subscribers file, no digest cron. Its only job is the
  06:00 ingest plus serving Streamlit.

### Shipped
1. **Slice loading** (`9e914cf`, deployed). `utilities/scoring_slice.py` +
   `attach_earnings_history()`, called from stage3's incremental branch.
   Droplet measured: **369 MB, 14s, exit 0** (was 1030 MB, OOM). Local: 427 MB.
2. **Parity is exact.** Full vs incremental `upcoming_df`: **all 23 columns, all 496 rows
   identical**, `peer_percentile` included. The old divergence (17 stocks losing
   `pre_earnings_drift_flag`, 8 on `surprise_momentum_flag`, ORCL/DECK flipping
   `is_high_conviction`) is gone. xfail marker removed; suite **73 passed**.
3. **Digest** reads `output/upcoming_df.parquet` not `full_df.parquet`: **128 MB** (was
   multi-GB, unbounded). Added `MAX_PARQUET_AGE_HOURS = 24` — refuses to send stale numbers
   rather than silently mailing last week's tiers. Wired in as step 5 of `full_workflow.sh`.
   Sent LOCALLY on purpose: the PDF attachments come from stage5 (droplet has 0 PDFs in its
   run dirs) and the SMTP creds are in the local `.env`. Also kills a race a Monday cron
   would have had with the manual weekly run.
4. **06:00 droplet cron succeeded for the first time** — 74s, exit 0, DB current.

### Measurements that correct earlier notes
- **The 274 MB slice estimate was WRONG.** A single `pd.read_parquet(filters=...)` costs
  **584 MB**: `filters=` only prunes whole row groups by statistics, this file has 3 row
  groups of ~970k rows, and earnings days are scattered through all of them, so it
  decompresses everything and filters afterwards. Streaming 25,000-row batches gives
  **232 MB** for a bit-identical result. That is why `scoring_slice.py` streams.
- **The full pipeline peaks at 5562 MB**, not the "2000+ MB" recorded earlier. Measured
  breakdown: the frame itself is **1910 MB** — float64 1468 MB (63 cols), str 288 MB (6),
  datetime 93 MB, int8 32 MB. Peak is ~3x the frame because each stage copies.
  float64 -> float32 would save **~734 MB** of frame (-> ~1176 MB) and proportionally more
  of the peak. **The categorical idea is moot** — string columns are already `str` dtype,
  not object; there are zero object columns. Cost of float32: precision changes, so
  `testing/calibration.py` is the acceptance gate before trusting it.
- Server sizing, if the droplet ever needs to run the full pipeline: that is an **8 GB**
  box (~$48/mo vs the current $6), not 2 GB. **Try dtypes first** — float32 + categoricals
  could plausibly halve it and helps local runs too. Cheaper experiment, measurable.

### Earnings results were never backfilled (fixed 2026-09-02)
`reported_eps` was NULL for **100% of events under 30 days old** and ~80% at 31-90 days,
filling only at 91-180. Two causes, both needed fixing — the first alone does nothing:
1. The skip rule (`WHERE earnings_date > current_date + 14`) asks "do we know the next
   date?" and thereby also suppresses "do we have last quarter's result?". Once a stock
   reports and yfinance hands us the next date ~90 days out, it is not fetched again for
   ~80 days. Now also re-fetches stocks whose most recent past event lacks a result,
   bounded by `EARNINGS_RESULT_BACKFILL_DAYS = 30`.
2. **The ingestion is INSERT-ONLY.** It filters out every date already in the DB and then
   does a plain `INSERT` — no upsert. So a placeholder row's NULL could only be corrected
   by the ±60-day DELETE path, which fires only for a date it has not seen. Added an
   explicit UPDATE pass before the filter; fills NULLs only, never overwrites a confirmed
   row. Verified on a DB copy: 5 stocks backfilled, **0 rows added, 0 duplicates**.
Yahoo had the data all along (ADSK 3.30/+5.64%, CRM 5.90/+80.36%) — we were not asking.
**First run after this clears an ~80-day backlog: ~191 stocks fetched instead of 19, then
it settles to the few that reported recently.**
**Worth what exactly:** `surprise_momentum_flag` is DISPLAY ONLY — digest, PDF reports,
calendar, dashboard, predictions table. It does not feed `earnings_explosiveness_score`,
the tier, or `is_high_conviction` (drift flag only). Backtesting 2026-05-18 measured the
surprise sub-categories at ~4.2x against a High Alert baseline of 3.82x and rejected them
for high conviction. So this fixes "blank reads as normal when it means unknown", nothing
in the ranking.

### A trap worth remembering
Emulating the old `groupby().last()` broadcast with an ffill **manufactures signals**.
It carried a stale streak of 27 onto ADSK's 2026-08-27 earnings row, where the full path
has NaN because the just-reported surprise is not in the DB yet — inventing an "Extended
Beat Streak" the full path declines to assert. Six stocks affected. The full path leaves
these columns NaN off earnings days and lets the flag functions propagate; consumers read
`groupby().last()`, which skips NaN. **Do not fill them.** Reasoning is in the code.

### Repo conventions the user stated
- `pipeline/` holds **only** pipeline stages or versions of the pipeline. Nothing else.
  (`scoring_slice.py` -> `utilities/`, `save_predictions.py` -> `analysis/` for this reason.)
- Stages must read simply: prefer a named function call over an inline code block.
- Droplet and local **must** produce identical results; a difference is a defect.

### Open
- **`data/subscribers.txt` is tracked as an EMPTY file** (committed 0 bytes in `acb79f5`,
  so the address was never published — an earlier warning of mine about that was wrong).
  The user added it to `.gitignore` line 22, but **gitignore does not apply to tracked
  files**: the working copy (18 bytes, real address) still shows as modified and a
  `git add .` would commit it. Needs `git rm --cached data/subscribers.txt`.
- Uncommitted at time of writing: the digest + workflow change, plus the user's own
  `streamlit_dash/app.py` and new `styles.css`.
- 7 stocks with `earnings_date` >90 days out (CRM, DG, DLTR, HRL, NDSN, SNPS, ULTA).
- Brain (`/home/Michael/projects/brain`): still undecided, still untouched, leave alone.
- Droplet has **no backups**. It now matters less — nothing unique lives there.

## 2026-09-01 — cron_ingest made ingest-only, branches consolidated, parity test filed

**GOAL AS IT STOOD THAT DAY:** automatic weekly predictions + digest emails ON THE DROPLET.
**SUPERSEDED 2026-09-02** — the user chose weekly-only, so the droplet does not score.
See the 2026-09-02 entry above before acting on anything in this entry.

### State of the tree — END OF SESSION, all clean
- **One branch: `master`.** Local == `origin/master` == **`d9567d8`**, working tree clean.
  `risk_score_proposed_fix` and `stock-lifecycle-status` were verified fully contained
  (0 unique commits each) and **deleted** local + remote. Old tips: `2b79144`, `b387d82`.
- **DROPLET IS DEPLOYED** — pulled to `d9567d8`, ingest-only `cron_ingest` in place, tree
  clean apart from an untracked `next_earnings_df.csv`. Tomorrow's 06:00 run should succeed;
  **check `/var/log/breakwater_ingest.log` first thing** to confirm it did.
- **Suite is GREEN**: `72 passed, 1 xfailed`. The parity failure is now
  `xfail(strict=True)`, not a red test. Do not delete the marker by hand — `strict=True`
  makes the test FAIL the moment the bug is fixed, which is the signal to remove it.
- Commits from this session: `3b1283c` (ingest-only), `e0c8632` (merge), `b19219c`
  (refactor + parity test), `d9567d8` (memory prune).

### Done today
1. **`cron/cron_ingest.py` is now ingest-only** (`3b1283c`) — just `stage1(incremental=True)`.
   Dropped from the daily job: `_has_new_earnings()` (391 MB), the `run_pipeline()` fallback,
   and stage2-4 + `export_upcoming_df`. Header comment says why; do not add scoring back.
2. **Measured on the droplet: 285 MB peak, 91s, exit 0**, ~308 MB headroom against ~593
   available. First successful ingest in weeks; droplet DB now current to 2026-09-02.
3. **Merged `origin/master`** (`e0c8632`) — 4 web-UI commits deleting the tracked calibration
   CSVs. Took origin's side; `.gitignore` line 20 already covered them, they just predated it.
   **Merged, not rebased, on purpose:** `config.py` cites `f3dd1e2` by SHA and the predictions
   table stores `git_commit` per row — a rebase orphans both.
4. Pushed master, THEN deleted the branches. Order mattered: all three recent commits were on
   origin *only* via `origin/risk_score_proposed_fix`.

### Two facts worth keeping
- **stdout is block-buffered under cron and lost on SIGKILL.** `Stage 1 DONE` appears nowhere
  in `/var/log/breakwater_ingest.log` despite stage1 completing every time. Only the yfinance
  **stderr** lines mark real progress there. Do not infer where the job died from missing prints.
- `run_incremental_pipeline()` now has **no callers anywhere** — dead until the slice work
  revives it. Its line 27 calls `run_pipeline()` with no args while the signature is
  `run_pipeline(incremental)`; latent `TypeError`, currently unreachable. Fix when rebuilding.
  Also: `streamlit_dash/app.py` imports `run_pipeline` but never calls it — near miss, checked.

### Next, in order — ALL DONE 2026-09-02, see the entry above
The slice work, the digest parquet change and the deploy all landed. Kept only so the
2026-09-01 record reads straight; do not work from this list.

### Decisions taken 2026-09-01 (so they are not re-litigated)
- ~~**Where predictions live once the droplet generates them:** droplet writes, workflow
  pulls it down.~~ **REVERSED 2026-09-02 — DO NOT BUILD.** The droplet does not score, so
  predictions stay local exactly as they already are: stage5 writes git-tracked
  `db/predictions.duckdb` on the weekly run. One writer, git as archive. Nothing to do.
- **Brain (`/home/Michael/projects/brain`): undecided, leave alone.** User: "i havent used the
  brain yet, it was an idea." Every file there is dated 2026-06-29 and it is not a git repo.
  `projects/breakwater.md` has drifted (`data/breakwater.duckdb` -> now `db/`,
  `monday_workflow.sh` -> now `full_workflow.sh`). Breakwater's CLAUDE.md still routes every
  session there before architecture work; that detour is currently worthless. Do not spend
  time on it unless asked.
- **Subscribers:** `data/subscribers.txt` was un-ignored by the user. **The repo is PUBLIC**
  (`api.github.com/repos/Savin97/breakwater` -> `"private": false`) — user believed it was
  private and judged the exposure acceptable on that basis; flagged, user's call, one address.
  Real fix when wanted: a `subscribers` table (email, subscribed_at, status,
  unsubscribe_token) fed by the landing-page form, digest reads the table not the file.
  Needs a decision on whether the form posts to the droplet or a third party (Formspree).
  Own session, not a tweak.

### Also open
- Droplet has **no backups at all** — the only backup line in its crontab is Ubuntu's
  commented-out example. Disk is 37% used, 15 GB free, uptime 8 days (prior boot ran 102
  days), 5 kernel OOM events all self-inflicted by the pipeline. Stable enough to write to,
  not safe as the only copy.
- Droplet still needs `data/subscribers.txt` and `DIGEST_SMTP_*` in `.env` before any digest
  email can send.
- `run_incremental_pipeline()` has **no callers anywhere** — dead until the slice work revives
  it. Its line 27 calls `run_pipeline()` with no args while the signature is
  `run_pipeline(incremental)`; latent `TypeError`, currently unreachable. Fix when rebuilding.
- `utilities/db_utilities.py` has no trailing newline.

## 2026-08-31 (session 2) — Droplet memory diagnosis + plan for automated weekly predictions

**GOAL (stated by user):** weekly predictions generated automatically ON THE DROPLET, and
weekly_digest emails sent automatically — at first only to the user, to test. Everything below
is groundwork for that. **We stopped here; this is tomorrow's work.**

### State of the tree — SUPERSEDED, see the 2026-09-01 entry above
This section described the tree as of 2026-08-31 (3 unpushed commits, red suite, cron_ingest
not yet ingest-only). All of it was resolved on 2026-09-01. The **measurements, the slice plan
and the bug analysis below are still current and still the plan** — only the tree state changed.

### Why the droplet cron has never worked (measured, not guessed)
Droplet: **961 MiB RAM, ~581 MiB available** (Streamlit `breakwater.service` holds the rest).
`/var/log/breakwater_ingest.log`: **75 `Killed`** (OOM) and only **2 `TypeError`**. The TypeError
(`stage1(update=True)`) only appeared after the 2026-08-29 pull; before that it was pure OOM.
**Fixing the TypeError does not fix the cron — it just moves the failure back to the OOM.**

| path | peak RSS | on droplet |
|---|---|---|
| full `run_pipeline` | 2000+ MB | OOM (the 75 Killed lines) |
| incremental path **as built** | **1030 MB** | OOM — the "fast path" does not fit either |
| `_has_new_earnings()` alone | 391 MB | reads 4 cols x 2.9M rows |
| slice, all 87 cols | 500 MB | fits, only ~80 MB headroom — too thin |
| **slice, 21 needed cols** | **274 MB** | **fits, ~300 MB headroom** |
| the sliced data itself | 10 MB | rest is import overhead + duckdb .df() buffer |

Memory is dominated by **reading `full_df.parquet` (323 MB) into pandas**, not by the 90-day
window. So the incremental design saves CPU but not memory, and memory is the binding constraint.

### The plan: load only the rows the computation actually reads
Key counts: 2,911,875 total rows, but only **45,693 are earnings days (1.6%)**. Slow per-stock
stats (15 cols x 503 stocks = **7,545 numbers**) are aggregates over ~91 events per stock; the
other 98.4% of rows are daily prices the slow stats never touch.

**The slice = all earnings-day rows (every year) + the last 90 days of prices = ~76,494 rows (2.6%).**

**CRITICAL CONSTRAINT — the slice is NOT contiguous in time.** Consecutive historical rows are
~90 days apart. Anything using `.diff()`, `.pct_change()` or row-based `.rolling()` silently
produces garbage there. Measured on AAPL: recomputed `pct_change` = **0.262** where the true
stored `daily_ret` = **0.0104** — a 90-day return mislabelled as daily, 25x wrong, and it looks
plausible. So the work MUST be split:
- **historical earnings rows** -> use the **stored** per-event values (`abs_reaction_3d`,
  `drift_30d`, `reaction_*`). NEVER recompute anything price-derived from them. Only aggregate
  ACROSS events (`groupby(stock)` rolling/expanding), which is event-ordered and therefore valid.
- **recent 90-day window** -> contiguous, recompute price features normally.

**Bonus: this also fixes the flag-parity bug** (below) for free — that bug exists precisely because
the 90-day window holds ~1 earnings event per stock, so the drift baseline cannot be built. The
slice holds all ~91.

**Do not assume bit-identical — prove it with the parity test.** (I over-claimed twice today.)

### The bug the red test documents (full vs incremental divergence)
The incremental path strips flags: on real data **17 stocks lose `pre_earnings_drift_flag`**,
8 differ on `surprise_momentum_flag`, and **`is_high_conviction` goes True->False (ORCL, DECK)**.
Score/bucket/percentile match exactly (they come from cache). Cause: `engineer_pre_earnings_drift_flag`
builds its baseline from `drift_30d` over the stock's earnings-day rows *present in the loaded
frame*; only **4 of 500 stocks have >=2 earnings days** in a 90-day window, so std is NaN,
`has_hist` is False, and the flag falls back to `""`.
**This becomes BLOCKING for the goal:** if the droplet generates the weekly predictions, HC — the
headline signal — would be empty in every digest.
Origin: `INCREMENTAL_CACHED_COLS` was created 7 Jun (`c0f82c4`); the drift flag's full-history
dependency landed 9 Jun (`1682385`), two days later, and nothing reconciled them. CLAUDE.md states
the rule as "anything needing `abs_reaction_3d`", which is **narrower than the real hazard** —
the real rule is *any function that aggregates over the stock's earnings-day history*.

### Also needed for the goal
1. **`cron_weekly_digest.py` does `pd.read_parquet(PARQUET_PATH)` with NO column selection** —
   the whole 2.9M x ~100 frame, several GB. It should read `output/upcoming_df.parquet`
   (**46 KB**), which already has tier, score, percentile, both flags, HC and earnings_date.
   Cheapest win of the lot.
2. **Decide where predictions live if the droplet writes them.** `db/predictions.duckdb` is now
   git-tracked and written locally; both sides writing it = binary git conflicts.
3. Droplet needs `data/subscribers.txt` (gitignored) with just the user's address, and the
   `DIGEST_SMTP_*` vars in its `.env`.
4. Every week has earnings (2025+: median 12 stocks, max 179, **0 weeks with none**), so
   `_has_new_earnings()` is essentially always True and the full-run fallback always fires.
   Guard or remove it — on 961 MB it can never succeed.

### Suggested order
`cron_ingest` genuinely ingest-only (stops the bleeding) -> digest reads `upcoming_df` ->
the slice loading change -> parity test green -> then deploy.

### Other open items from earlier today
- `week_start` is now always == `run_week` in the predictions table (week-only scope makes it
  redundant). Harmless; drop only if wanted.
- 9 stocks have `earnings_date` >90 days out (CRM, CRWD, DG, DLTR, HRL, NDSN, SNPS, ULTA, WDAY) —
  ingestion data quality, untouched.
- **Other fixture-driven tests may be vacuous** like the HC ones were (the synthetic fixture
  yields no High Alert rows). Not audited.

## 2026-08-31 — Predictions table + is_high_conviction fix (branch `risk_score_proposed_fix`)

**Committed:** `b0db658` (HC fix + tests). Predictions-storage commit staged separately.

**Predictions snapshot shipped** — `analysis/save_predictions.py::save_predictions_snapshot(df)`,
called at the end of stage5. Records what we published: `prediction_asof_date`, `run_week`,
`week_start`, stock, `earnings_date`, `tier`, `risk_score`, `is_high_conviction`, both flags,
`model_version`, `git_commit`.
- **Scope is the run week only** (`today..Sunday`), not every future event — the table is the
  product record. Window starts at *today*, not Monday, so a mid-week re-run can't write a
  prediction for an event that already reported (would leak hindsight into backtests). Cut at
  Sunday not Friday because 115 earnings_dates in history land on a weekend (bad source data).
- **Lives in its own `db/predictions.duckdb`** (`config.PREDICTIONS_DB_PATH`), NOT breakwater.duckdb.
  Verified on the droplet: it has **no** predictions table and never writes one (its cron runs
  `run_incremental()`, which skips stage5). `scripts/full_workflow.sh` pulls the droplet's
  breakwater.duckdb and *overwrites local*, so a table living there is destroyed every weekly run.
  Pushing our copy back was rejected: the droplet has **six cron writers/day** (ingest 06:00,
  eps 14:45, IV 15:00/16:30/18:00/19:30 UTC) and rsync over a live DuckDB risks corruption plus
  loss of IV snapshots, which cannot be refetched.
- **Un-ignored in .gitignore** — it is the only copy and a lost week is unrecoverable. Needed
  `db/*` + `**/db/*` instead of `db/`, because git cannot re-include a file whose parent dir is
  excluded.
- **Upsert**, keyed `(stock, earnings_date, prediction_asof_date)`. `ON CONFLICT DO NOTHING` was
  wrong: the first run of a day won, so a broken snapshot survived a same-day re-score (this bit us
  — required a manual DELETE). New date = new row, preserving drift toward the event.
- **`predictions_week_open` view** = earliest surviving call per event per run week (the Monday
  call). **Backtest against the view, not the table** — the raw table double-counts any event
  scored on several days that week.
- `utilities/peek_at_db.py` attaches the second DB + sets search_path so `peek_at_db predictions`
  still works unqualified; `list_tables` needs `SHOW ALL TABLES` (plain `SHOW TABLES` sees only the
  current catalog).

**`is_high_conviction` was silently False on ~98% of rows** (fixed, `b0db658`).
`engineer_high_conviction` compared `earnings_explosiveness_bucket` per row, but that column exists
**only on earnings-day rows** — NaN in between, and `(NaN == "High Alert")` is False. Score/bucket
survive this because `.groupby().last()` skips NaN and reaches back; a bool has no NaN to skip, so
the wrong value wins. Fix carries the last completed event's bucket forward *locally inside the
function*; stored columns untouched, so calibration groupbys and backtesting see identical data.
Earnings-day rows bit-identical; ~31k rows changed, all False -> True.
- **Who was affected:** `report_builder`'s weekly High Conviction list and the predictions table
  (both read the raw column). Dashboard + weekly chart were already correct — they recompute it.
- **Rejected as too broad:** ffilling the stored score/bucket columns. Only the bool benefits;
  every other consumer already reaches back via `.last()`.

**The test suite proved nothing here — 68/68 passed before AND after.** The synthetic fixture in
`testing/test_pipeline.py` yields **zero High Alert rows**, so all three HC tests were vacuous
(`.all()` on an empty frame is True), and two asserted the *old buggy* invariant and would have
failed on real data. Now compare against the carried bucket + three direct tests of
`engineer_high_conviction`, each confirmed to fail against the old implementation. **Lesson: check a
new test actually fails against the bug before trusting a green suite.** Other fixture-driven tests
may be vacuous for the same reason — not audited.

**MODEL_VERSION renumbered to `0.3.1`** (pre-1.0 — not a finished product). Old `"1.0"` = **0.1**,
old `"1.1"` = **0.2**; mapping recorded in config.py. Stored rows also carry `git_commit`, which
places them unambiguously — needed because the 08-29 rows were labelled "1.0" while actually being
lift-era `f3dd1e2` output.

**Resolves open item 3 of the 2026-08-29 entry** (mislabeled prediction rows): table was cleared and
re-saved; now 10 rows, asof 2026-08-31, model_version 0.3.1, HC=2 (ORCL, DECK).

**Open:**
1. **Not pushed / not deployed.** Branch not merged to master, nothing pushed. Droplet deploys by
   `git pull`, so pushing master is effectively a deploy.
2. **`db/predictions.duckdb` has no backup until committed** — un-ignoring only makes git *see* it.
3. **9 stocks have earnings_date >90 days out** (CRM, CRWD, DG, DLTR, HRL, NDSN, SNPS, ULTA, WDAY) —
   same class as WDAY's 174-day gap. Ingestion data-quality issue, untouched.
4. `.claude/memory/MEMORY.md` "UNCOMMITTED" markers on the two 2026-08-29 entries were stale.
   **Re-verified and corrected 2026-09-01:** `utilities/logging_utilities.py`,
   `ingestion/fetch_earnings_dates.py` and `report/img/breakwater_logo.png` are all tracked.

## 2026-08-29 — Logging framework + report logo fix

*(Was recorded as uncommitted; committed since — `utilities/logging_utilities.py` and
`report/img/breakwater_logo.png` are tracked as of 2026-09-01.)*

Continuation of the same session as the parallelization entry below. Both were uncommitted at
the time; both are committed as of 2026-09-01.

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

## 2026-08-29 — Parallelized yfinance earnings fetches

*(Was recorded as uncommitted; committed since — in `ingestion/fetch_earnings_dates.py`
as of 2026-09-01.)*

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

## 2026-08-29 — Lift-based tier promotion built on branch `risk_score_proposed_fix` (COMMITTED f3dd1e2)

**Supersedes the 2026-08-19 plan below — that plan's approach was tested and rejected.**

**What shipped into the working tree:** `engineer_stock_bucket_lift` + `engineer_lift_adjusted_bucket`
in `scoring/scoring_features.py`, wired into `pipeline/stage4.py`. Lift = P(extreme|stock,bucket) /
P(extreme|market), causal (prior events only, shift(1)-before-expanding), shrunk by
`LIFT_PRIOR_STRENGTH=20`. Normal events with lift>=1.5 promote to Elevated, >=3.0 to High Alert.
Structural bucket preserved as `earnings_explosiveness_bucket_structural`. Duplicate logic deleted
from `report/report_builder.py`. `MODEL_VERSION` -> 1.1.

**Measured OOS 2015-2025:** capture of >=8% moves **43.6% -> 56.8%**, >=15% 66.2% -> 76.1%.
High Alert unchanged (40.3%, 3.72x), Normal cleaner (7.0% -> 5.8%), Elevated 25.2% -> 23.7%
(inside both CIs, not distinguishable). Selection 12.7% -> 18.9% (~4.8 -> ~7 names/week).
High Conviction **bit-for-bit identical** (234 events, 0.534, 4.81x). 68/68 tests pass.

**Rejected by measurement — do not retry:** multiplying risk_score by lift drops top-decile lift
3.70x -> 2.98x (lift is ~0.79 rank-correlated with the score and corrupts its ordering). Ranking
drift-first (treating HC as a tier above High Alert) drops P@10 2.25x -> 1.89x — HC is an overlay,
and that original design is correct. Deleting the report_builder bump outright was also wrong: its
promoted events realise 0.238 vs 0.058 for the Normal events left behind, i.e. real signal.

**Why score and lift disagree** (the crux): score is p75 of abs reactions (rolling 28, capped at
12%) = the *typical* move; lift is frequency of >=8% = the *tail*. p75 structurally ignores the top
25%. EW: typical move 2.4% but 25% of events >=8%. Separately, EBAY scores **72.976** against a 73
cutoff while exceeding 8% in 75% of its events — so lift is currently patching threshold brittleness
as well as finding tail-heavy names.

**Open, in priority order:**
1. **73/79 cutoffs and the lift gate must be tuned together** — lift is doing two jobs (real tail
   signal + patching the hard 73 boundary). Sequential tuning double-counts.
2. **Unresolved: what `risk_score` is for.** Tier now uses two signals, score carries one, so they
   disagree — Elevated currently spans 29.68-78.97 vs Normal 8.71-72.98 (16,698 overlapping pairs,
   up from 0). Either accept it (score = within-tier sort key) or floor the score to the tier
   boundary (rejected once as assigning a non-measurement). A measured but NOT-implemented option:
   gate promotion on structural score >=50 — capture 57.2% -> 57.0% (~free), Elevated quality
   0.241 -> 0.250, worst ordering violations gone.
3. ~~**Predictions table has mislabeled rows**~~ — RESOLVED 2026-08-31: table cleared and
   re-saved, and the insert is now an upsert so a same-day re-run corrects rather than skips.
   See the 2026-08-31 entry.
4. **Incremental path untested** — `stock_bucket_lift` and `earnings_explosiveness_bucket_structural`
   added to `INCREMENTAL_CACHED_COLS` but `run_incremental()` never exercised. Droplet cron uses it.

**Gotcha found:** `main.py` runs `run_pipeline(incremental=False)` = the **paid AlphaVantage** path;
`incremental=True` is the free yfinance one. CLAUDE.md documented this backwards and has been fixed,
along with several dead paths (`prep_for_streamlit.py`, root `testing.py`, `backtesting/`,
`data/breakwater.duckdb`, `streamlit_df.csv`). `scripts/full_workflow.sh` calls `main.py`, so it hits
the paid path as written. To re-score without ingesting, run stage2->3->4 directly.

## Earlier sessions — compressed (2026-05-17 → 2026-08-19)

Condensed 2026-09-01 from full session notes. Kept: facts still load-bearing today.
Dropped: narrative, superseded plans, and next-step lists that have since been done or
overtaken. Full text is in git history if ever needed.

**2026-08-19 — risk_score/bucket inconsistency diagnosed.** A stock could show a lower
`risk_score` than another while sitting in a higher tier, because `report_builder.py` and
`streamlit_export.py` each reimplemented a "historical lift" bump that changed the label but
never the number — three uncoordinated implementations. **Superseded:** fixed properly in
`f3dd1e2` by moving lift into stage4 as a tier reclassification.

**2026-08-15 — Ticker lifecycle + repo reorg (shipped).**
- `stock_data` gained `status` ('active'/'inactive') and `reason`, reconciled every run against
  the live Wikipedia S&P 500 table. `read_stocks_to_fetch(active_only=True)` stops dead tickers
  being retried (503 -> 490).
- `data/ticker_renames.csv`: **BK -> BNY** (2026-05-21), **SATS -> ECHO** (2026-06-24). Verified
  NOT renames: DAY, EA, HOLX (taken private/delisted), CTRA (merged into pre-existing DVN).
  CAG/CPB/EPAM/POOL/LW/PAYC/MTCH/MOH are still trading — merely dropped from the index.
  *(These are the names still producing daily "possibly delisted" noise in the ingest log.)*
- DB moved `data/` -> `db/breakwater.duckdb` (gitignored).
- **Hazard worth remembering:** `full_workflow.sh` pulls the droplet's DB and *overwrites local*,
  which silently wiped a local schema migration twice. Migrate the droplet's DB, then re-sync.

**2026-08-15 — Pipeline fixes + first test suite.** Removed a stray `exit()` in stage2 that was
silently killing the pipeline; added `assert_df_fresh` (raises if max price date >10 days old) and
`clean_duplicate_earnings_from_db`; promoted `is_high_conviction` to a real stage4 column via
`engineer_high_conviction`; created `testing/test_pipeline.py` and
`testing/weekly_prediction_quality.py`.

**2026-07-27 — Monday-log audit.** Fixed the big one: `yf.download(end=...)` is **exclusive**, so
every Monday run fetched a window containing zero trading days and all 503 tickers logged
"possibly delisted". Now `end = today + 1 day`.
Open issues found then that are **still open**: `data/stock_list.csv` is stale (12 tickers appear
in earnings data but not the list); earnings dates from yfinance are wrong ~20% of the time, many
by exactly +7 days; `report/calendar_builder.py` filters `is_earnings_day == 1` so it never emits
`weekly_calendar.html` for *future* events — check whether anything still reads that file.

**2026-06-22 — Landing page + dedup.** Landing page stats band and dynamic recent-calls section
(`recent_calls.json`, pushed to harbor_webpage by `full_workflow.sh`). Added `dedup_earnings`
(yfinance returns slightly different dates for the same event). Droplet path lowercased to
`/var/www/breakwater`.

**2026-06-09 — Flag fix.** `pre_earnings_drift_flag` and `surprise_momentum_flag` were always
empty for *upcoming* events, since both only populated `is_earnings_day == 1` rows while
`export_upcoming_df` reads the latest price row. Drift flag now also computes on pre-earnings
window rows (1-60 days out) — **this is the same function whose baseline breaks the incremental
path today.** Surprise flag now forward-fills within each stock, earnings rows acting as resets.

**2026-06-07 — Incremental pipeline built.** `stage2(lookback_days=N)`, `stage3(incremental=True)`
reading cached expanding stats from `full_df.parquet` per `INCREMENTAL_CACHED_COLS`,
`stage4(incremental=True)` skipping anything needing `abs_reaction_3d`, and `run_incremental()`
with a `_has_new_earnings()` fallback to the full run. Claimed 0.8s vs 80s and "bit-for-bit
identical" — **the identity claim was never true for the flags** (see the parity bug above); the
drift flag's full-history dependency landed two days later and nothing reconciled them.
Same commit: codebase audit — [full change log](codebase_audit_2026_06_07.md).

**2026-06-01 — Calibration + percentile fix.** Built `testing/calibration.py`. High Alert 40.2%
P(>=8%) vs 6.9% base; HC 52.4%; stable 2015-2025. Fixed uncapped percentile ranking to use
`abs_reaction_p75_rolling.fillna(abs_reaction_p75)` instead of the clipped score. Product framing
decision: sell "which 15-20 events matter this week", lead with the >=8% lift story; do not chase
false negatives in the Normal bucket — those moves are structurally unpredictable.
Memory consolidated to `.claude/memory/`; `cv_website` renamed `harbor_webpage`.

**2026-05-31 — Report/digest consistency.** Reports used the last `is_earnings_day` row (a *past*
event) for the earnings date, and ranked percentile against earnings-day rows only. Both switched
to the forward-filled latest row + `rank(pct=True)`, matching the digest.

**2026-05-31 — Digest selection bug.** `_select_stocks()` filtered `is_earnings_day == 1` *before*
grouping, so upcoming earnings could never be found — future rows never have that flag. Fixed to
`sort_values("date").groupby("stock").last()`. Identical bug in stage5's auto-selection. Digest
gained PDF attachments and an unsubscribe link.

**2026-05-30 — Product build + digest frozen.** IV (`expected_move_pct`, `atm_iv`) joined in
stage2 and shown in reports; stage5 auto-selects High Alert + Elevated within 14 days; weekly
digest `cron/cron_weekly_digest.py` (Mondays 07:00 UTC, reads `full_df.parquet`, list in
`data/subscribers.txt`); cron scripts moved to `cron/`. Layout frozen — stop iterating on it.

**2026-05-19 — Reports + yfinance migration.** `report/chart_builder.py` reactions chart, peer
percentile, days-to-earnings. AlphaVantage subscription cancelled; `incremental_ingest_*_yf`
functions added and made the active path.

**2026-05-18 — Memory setup.** `.claude/memory/` created inside the repo so notes sync via git.

**2026-05-18 — HC validated + recommendation block.** `is_high_conviction` (High Alert + drift
flag) measured at 4.93x OOS lift, ~12 events/yr. Broader definitions tested and rejected: adding
surprise sub-categories gave 4.24x at 75 events/yr — more coverage, worse precision. HC stays
drift-only. `report/recommendations_builder.py` added with 4 tiers of language.

**2026-05-17 — Dashboard overhaul.** Streamlit export automated into stage5; flags and
`is_high_conviction` surfaced in Overview / Bucket Stats / Weekly Calendar tabs.

## Product & Business
- [Product direction](project_direction.md) — target market (retail options traders), value prop, pricing, live URLs (as of 2026-05-30)
- [Next to build](next_to_build.md) — prioritized build list: IV into reports → coverage automation → weekly email digest (as of 2026-05-30)

## Infrastructure
- [DigitalOcean droplet](infra_digitalocean.md) — cron schedule, droplet path, stale tickers list
- [Window sensitivity](window-sensitivity.md) — grid search confirmed window=28 optimal (4.49x avg lift, 100% years ≥3x)

## DuckDB Schema — `db/breakwater.duckdb`

**prices:** stock, date (DATE), price (DOUBLE), ingested_at — unique index (stock, date)
**earnings:** stock, earnings_date (DATE), fiscal_end_date (DATE), reported_eps, estimated_eps, surprise_percentage (DOUBLE), ingested_at — unique index (stock, earnings_date, fiscal_end_date); fiscal_end_date=None for yfinance rows (manual dedup in code); surprise_percentage stored as decimal (÷100)
**stock_data:** stock (PK), company_name, sector, sub_sector, ingested_at
**merged_stock_data:** denormalised join of the above — NOT used by pipeline (stage2 reads raw tables directly)

---
