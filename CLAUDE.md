# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Continuity

At the start of every conversation, read `.claude/memory/MEMORY.md` to restore context from previous sessions. At the end of every conversation (or when asked), update it with what was done and what's next.

**Memory location:** All session memory lives in `.claude/memory/` inside this repo. It syncs via git across machines and is the single source of truth. Do NOT write to the Claude harness auto-memory location (`~/.claude/projects/*/memory/`) — that path is not used for this project and will diverge.

## Branches and Folders

Two branches: `master` (production — the droplet runs it) and `methodology-rebuild` (the
audit rebuild, merged into master only when the user says it is ready). One folder: this
one. **Do not create other branches, worktrees or copies of the repo without asking.**
Small fixes go on master; afterwards master is merged into methodology-rebuild, never the
other way round until the rebuild is ready. The user commits; give them the `-m` message.

**Scoring ≠ publishing ≠ archival.**
- *Scoring + local outputs*: `run_pipeline` (stages 1–5). Writes only under `output/`; never
  deploys, mails or writes the prediction archive — on any branch.
- *Publishing*: `scripts/full_workflow.sh`, master only. It sources `scripts/deploy_guard.sh`;
  from any other branch (or a detached HEAD) it pulls the DB and runs the pipeline locally,
  then refuses loudly (exit 3, branch printed) before anything public. No override. On
  master (since 2026-10-14, audit H1–H3): `publishing.generation build` freezes the calls +
  history parquets into an immutable **generation** (`output/publication/<id>/`, manifest
  with SHA-256s, the artifact's model version, the reference checksum); the run refuses
  (exit 4) unless the droplet's `git rev-parse HEAD` equals this commit; the generation is
  uploaded to `output/generations/<id>/`; `publishing.generation check-live` refuses (exit 5)
  to replace a live generation that has no receipt or was never archived (its calls were
  public; it prints how to finish that SAME generation); then one atomic `mv` of
  `output/CURRENT_GENERATION` makes it live and removes the old flat parquets (a dashboard
  still on old code shows nothing rather than stale calls); the served pointer and
  calls-file hash are read back and
  `publishing.generation receipt` writes `receipt.json` (`published_at`, UTC) only if they
  match. The dashboard (`streamlit_dash/data.py`) reads the pointer on every page run and
  caches per generation, recomputes days-to-earnings from today's NY date and drops
  reported events — no manual cache clear, no stale calls after a publication.
- *Archival*: `python -m publishing.archive_calls --published --generation <id>`, after the
  receipt. Append-only, one row per call per generation (`published_at` TIMESTAMPTZ,
  `generation_id`, `artifact_sha256`, `call_sha256`, `revision`; unique on stock +
  earnings_date + generation_id). A retry of the same generation adds nothing; an identical
  call in a later generation adds nothing (the first publication stands); a changed call
  appends a revision — the original is never overwritten, even the same day. The WRITER
  (`archive_calls()`, not only the CLI) refuses the canonical DB without a verified receipt
  and off master, and stamps the artifact's model version, not config's. Window: the whole
  Mon-Fri work week the digest mails, so the record is *first eligible weekly-window calls*.
  Any other target (`BREAKWATER_PREDICTIONS_DB=…`) is scratch and allowed anywhere.
- *Track-record eligibility* (`utilities/db_utilities.py`): 0.4.0, not void, `score_asof_date <
  earnings_date`, and **published before the announcement** — `read_track_record(con,
  announcements)` trusts an announcement time only if it was observed at or after itself
  (a schedule is not a record); otherwise, and in the stored view, the call must predate
  00:00 NY on the report date. Filtering precedes picking the earliest publication per event
  (by `published_at`, then revision). Legacy rows have no `published_at`: never eligible.
- *Digest*: `python -m cron.cron_weekly_digest --generation <id>`. A real send (SMTP
  credentials set) requires master AND `--generation`, and reads the calls and the attached
  PDF reports (frozen into the generation at build) only from the verified bundle — never
  `output/`, which may have changed since; a receipt older than 24 h is refused. Without
  credentials it previews from `output/upcoming_df.parquet` on any branch.
- Partial failure: any step failing stops the run (set -e). Code check or upload fails →
  nothing live, nothing archived. Activation done but the read-back does not match → no
  receipt, nothing archived or mailed; investigate what the droplet serves. Receipt written
  but archive fails → rerun `--published --generation <id>` within 24 h (idempotent).
  Archive succeeds (`archived.json` written) → digest fails → resend with `--generation <id>`.
`testing/test_deploy_guard.py`, `testing/test_publication.py`, `testing/test_predictions_db.py`
and `testing/test_dashboard_freshness.py` pin all of this.

The complete methodology research (Phase 3–5 evaluation, Model C research code, options,
SEC and fundamentals experiments, vendor tooling, audit scripts and artifacts) is preserved
at the tag `methodology-audit-archive-october-2026`. Since 2026-10-03 `methodology-rebuild`
is being rebuilt from `master` as a clean production integration: production fixes only,
no research code. Recover anything archived from the tag selectively; never restore the
research tree wholesale.

## Brain

`/home/Michael/projects/brain`: the user's cross-project research notes (papers, finance
concepts). When reading or adding to it would help (a research question, a paper or
finding worth keeping), suggest it to the user — don't use it unasked. Example: it holds
Leung & Santoli (2014) on the options-implied earnings jump, relevant to testing Breakwater
against the implied move.

## What This Is

Breakwater is an earnings-move risk model for S&P 500 stocks. It ingests price/earnings/sector
data, builds one row per earnings event with corrected (timestamp-anchored) reactions, scores
every event with **Model B** (0.4.0, `scoring/model_b.py`), and produces PDF reports, a weekly
calendar and digest, a Streamlit dashboard and an archive of every published call. It ranks the
SIZE of a potential move, never its direction.

## Performance Claims — `audit/PHASE0_AUDIT_REV2.md` Is The Authority

**Read it before writing, quoting or restoring any performance number, anywhere in this repo
or in any customer-facing or marketing surface.** Every figure published before it was derived
from a target that mismeasured before-open (BMO) announcements and was overstated by roughly
3x. Remediation item **P4.1 is "Retire every published lift figure until P3.3."**

Verified figures for the **retired legacy 0.3.1 tiers** (audit §Q2, n=11,496 events with provider
announcement timestamps) — history, not evidence about Model B:

| quantity | verified | retracted value it replaces |
|---|---|---|
| market baseline P(\|reaction\| ≥ 8%) | **0.204** [0.197, 0.211] | 6.9% |
| High Alert hit rate | **0.455** [0.429, 0.482] | 40% |
| Elevated hit rate | **0.293** [0.265, 0.321] | ~18% |
| Normal hit rate | **0.157** [0.149, 0.164] | ~4% |
| High Conviction hit rate | **0.586** [0.509, 0.659], n=162 | 52% |
| High Alert lift | **1.91x stratified** (2.23x crude) | 3.70x / 5.8x |
| High Conviction lift | **2.46x stratified** (2.88x crude) | 4.78x / 4.83x |
| capture of ≥8% moves, High Alert + Elevated | **0.390** [0.371, 0.410] | 42% |

Rules that follow from it:

- **Quote the stratified lift, never the crude one.** The crude-vs-stratified gap is
  composition, not model skill (audit §Q2). Reporting AMC alone is worth 1.42x with no model.
- **There is no 15-year and no 2015–2025 out-of-sample record.** The figures above were
  measured when verified timestamps covered 25.4% of scored events and reached back only to
  ~2020 for most tickers (audit §Q3 coverage constraint). Nothing above is out-of-sample: the
  73/79 cuts were selected on this distribution (and were retired in 0.4.0). Coverage has since
  grown (see the Benzinga section), but that does not change these figures or license new
  ones — a figure measured on the wider history is a Phase 3 result and needs the same
  scrutiny.
- **The demonstrated edge is 1.87x within AMC and unestablished within BMO** (audit §Q3). On
  BMO, 96.0% of events are `Normal` and `Normal` lift is 0.94x.
- **No lift figure is displayed to users** (P4.1), and since 0.4.0 no lift is computed at
  all: the legacy scorer (explosiveness score, `stock_bucket_lift`, promotions, High
  Conviction, drift/surprise flags, 73/79 cuts) was deleted. `testing/test_customer_outputs.py`
  enforces that no customer surface names any of it.
- **Model B has no published performance figure.** Its research validation
  (`analysis/model_c_refit/RESULTS.md`: OOF 2017–2025 ROC-AUC 0.715, 2026 holdout 0.717) is a
  historical backtest — never quote it as live performance or in marketing. The customer
  score is a percentile RANK; never present it, or anything else, as a probability, chance,
  multiple of a market rate, expected move or direction.
- **Track record: prospective 0.4.0 calls only.** Every pre-0.4 archived call is voided
  (`void_for_track_record`, with `void_reason`; versions, scores and dates kept as
  published): the 10 `0.3.1-preaudit` calls under P4.2 and, since 2026-10-13, all 53 later
  `0.3.1` calls (self-migration in `create_predictions_table_if_not_exists`). The track
  record is the first ELIGIBLE 0.4.0 call per event — eligibility is filtered before the
  first call is taken — defined once in `track_record_select()` (`utilities/db_utilities.py`).
  Readers call `read_track_record()`, which runs it on the table: a stored
  `predictions_track_record` view is only as current as that file's last migration (the
  archive at 5cc7e64 had one returning 20 legacy calls). `marketing/generate_public_track_record.py`
  stays paused until enough 0.4.0 calls have resolved and Michael decides to publish.

## Commands

```bash
# Run the full pipeline (ingests via yfinance, then scores and writes reports)
python main.py

# Re-score WITHOUT re-ingesting — no API calls, uses the existing DuckDB.
python -c "from pipeline.pipeline import run_pipeline; run_pipeline(ingest=False)"

# ANY validation run: point the prediction archive at a scratch copy first, so the
# canonical db/predictions.duckdb is never written.
BREAKWATER_PREDICTIONS_DB=output/scratch_predictions.duckdb python -c "..."

# Launch the Streamlit dashboard
streamlit run streamlit_dash/app.py

# Model B calls + corrected outcomes for whole Mon-Fri work weeks (reads output/events_df.parquet)
python -m analysis.predictions_range --weeks-back 12
python -m analysis.predictions_range --weeks-forward 4
python -m analysis.predictions_range --monday 2026-06-01 --weeks 3

# Archived 0.4.0 calls vs corrected outcomes for last week. last_week_results.txt is
# INTERNAL (tier, percentile, >=8% tally); results_chart.png (analysis/chart_last_week.py)
# is for X, posted by hand: the same archived calls, each day's column ordered by the SIZE
# of the corrected 3-session move, no tally or rate. Nothing drawn without an archived
# 0.4.0 call; moves not measurable yet (Thu AMC / Fri on a Monday run) are named, not drawn.
python -m analysis.last_week_results

# Archive calls into a SCRATCH DB (validation; the canonical DB needs --published on master)
BREAKWATER_PREDICTIONS_DB=output/scratch_predictions.duckdb python -m publishing.archive_calls

# Internal Model B evaluation (retro-scored, never published)
python -m testing.metric_testing
python -m testing.calibration
python -m testing.backtesting

# Rebuild the percentile reference from the pinned 2026-09-28 snapshot and diff it against
# the committed artifact — must match bit-for-bit (writes nothing). --live: same on today's
# output/, a drift report only; never a reason to regenerate.
python -m scoring.build_model_b_reference --check
python -m scoring.build_model_b_reference --check --live

# Production Model B vs the frozen refit oracle, on the pinned snapshot (research evidence)
PYTHONPATH=. python -m analysis.model_b_validation.validate          # --live: today's output/
PYTHONPATH=. python -m pytest analysis/model_b_validation/tests analysis/model_c_refit/tests -q
# (`analysis.model_c_refit.evaluate build` takes the legacy A0 score from the pinned snapshot)

# Run the whole test suite
pytest testing
```

The project uses a `.venv` (Python 3.14). Activate with `source .venv/bin/activate` or prefix commands with `.venv/bin/python`.

## Pipeline Architecture

`main.py` → `pipeline/pipeline.py` (`run_pipeline(incremental=True, ingest=True)`). Each stage is a plain list of steps — one `df = engineer_...(df)` / `generate_...()` call per line:

| Step | File | What it does |
|---|---|---|
| 1 | `pipeline/stage1.py` | Creates/updates DuckDB at `db/breakwater.duckdb`. **Both settings ingest — the flag picks the provider.** `incremental=True` uses the free yfinance fetchers; `incremental=False` uses the legacy AlphaVantage full-history fetchers, which **require a paid AlphaVantage key**. `ingest=False` skips stage 1 entirely. |
| 2 | `pipeline/stage2.py` | Reads `prices`, `earnings`, `stock_data` (+ IV and EPS snapshots, joined as-of each row's date) into the daily frame. Dedups earnings and asserts data freshness. |
| 3 | `pipeline/stage3.py` | Daily preparation only: `days_to_earnings`/`is_earnings_day` (what the event frame is built from), `daily_ret`, `vol_10d/30d`, and the LEGACY report-date reactions, kept solely as the anchoring control. No score. |
| 4 | `pipeline/stage4.py` | Builds the **event frame** (`pipeline/events.build_pipeline_event_frame`) — one row per earnings event, every completed event plus one pending row per **active** stock (`stock_data.status = 'active'`; a stock that left the index keeps its history but gets no upcoming call), with verified announcement timing, the corrected anchored outcomes and `is_extreme_move_anchored` — then runs the **Model B** steps from `scoring/model_b.py` in order: `engineer_score_asof_date`, `engineer_prior_history`, `engineer_risk_score`, `engineer_risk_tier`, `engineer_model_version`. The one production magnitude scorer. `score_events` runs the same steps on a copy for callers outside the pipeline (a test pins the two together). |
| 5 | `pipeline/stage5.py` | LOCAL outputs only: `output/full_df.parquet`, `output/events_df.parquet`, the PDF reports, the weekly calendar, the dashboard parquets, the weekly chart (after the parquets), last week's results and the public track record (paused — returns without writing). It does NOT archive calls — see "Scoring ≠ publishing ≠ archival". |

Every forward-looking consumer reads the event frame's pending rows (`pending_events`), never
`groupby("stock").last()`.

## Data Storage

- **DuckDB** (`db/breakwater.duckdb`, gitignored): `prices (stock, date, price, ingested_at)` (+ `price_basis_log`), `earnings (stock, earnings_date, fiscal_end_date, reported_eps, estimated_eps, surprise_percentage, ingested_at, announce_ts_ny, announce_ts_source, announce_ts_observed_at)`, `stock_data (stock, company_name, sector, sub_sector, status, reason)`, plus `iv_snapshots` and `eps_estimates`.
- **Parquet** (`output/full_df.parquet`): the daily frame (prices, snapshots, stage 3 columns). Carries NO score; nothing customer-facing reads a score from it.
- **Parquet** (`output/streamlit_df.parquet`, `output/upcoming_df.parquet`): produced by `streamlit_dash/streamlit_export.py`; consumed by the Streamlit app.
- **Parquet** (`output/events_df.parquet`, ~15 MB): the Model B-scored event frame. One row per earnings event; `is_pending == 0` is history (each scored as of its own cutoff), `is_pending == 1` is the upcoming call. Every forward-looking consumer reads this.
- **Prediction archive** (`db/predictions.duckdb`, tracked): one row per PUBLISHED call per generation (append-only; see *Archival* above), written only by `publishing/archive_calls.py --published --generation` from the master publication workflow. Only grows; old 0.3.1 rows are evidence: their published values are never rewritten (only the void flag and reason, set by the self-migration). 0.4.0 rows add `model_b_hist_mean_abs`, `model_b_n_prior` and the cutoff in `score_asof_date`, with the legacy flag columns NULL.
- **Reference artifact** (`scoring/model_b_reference.json`, tracked): the frozen percentile scale, checksummed.
- **Stock universe** (`data/stock_list.csv`): the list of stocks to process.

## Price and Event Integrity (independent audit, 2026-10-14)

`audit/INDEPENDENT_AUDIT_REMEDIATION_2026-10-14.md` holds the measurements. Rules:

- **Prices keep one adjustment basis per stock.** yfinance ingestion
  (`ingestion/fetch_prices.incremental_ingest_all_prices_yf`) re-fetches a window per stock
  reaching back past its last stored row and reconciles it (`reconcile_stock_prices`): window
  rows replaced, missing sessions filled, older rows rescaled by ONE factor onto the
  provider's current basis (returns inside them unchanged), every rescale logged in
  `price_basis_log`. Never go back to appending only dates after the global max: that left
  MNST's 2026-08-11 split as a −51.7% reaction and a 497-stock hole on 2026-05-19..21.
  Existing damage is repaired by the same function with `since="2026-02-11"` — run on the
  DROPLET DB only, after review. Validated 2026-10-15 on a disposable copy (audit file §6):
  MNST fixed, May hole filled, no pre-window change beyond float rounding, rerun-safe; it
  changes 30 2026 outcomes > 0.1%, adds 19, and moves one live call (MNST → Normal). It
  fetches ACTIVE stocks only: inactive POOL/TAP keep two small 2026 basis errors.
- **Event identities come from the catalog**, not from price rows on the report date
  (`pipeline/events._completed_event_rows`): a report day with no price row keeps its event,
  with `anchor_session_status` `unresolved_price_gap` / `unresolved_no_session`; report-day
  columns stay blank, nothing is rolled or invented, strict endpoints still decide
  availability.
- **IV context only beside its own event** (`attach_iv_context_status`): snapshot after the
  previous report, on or before this one, expiry after this one; else blanked with a status.
- The Alpha Vantage full loads (pre-2026-02) and yfinance disagree on old history (mostly
  pre-2016); measured effect on the reference is in the fifth decimal and no change is
  made — do not "fix" it without a third source.

## Announcement Timing and the Corrected Target (Phase 2)

`reaction_1d/3d/5d` and `abs_reaction_3d` measure `close(D+k)/close(D)`, which assumes the
announcement lands after the close of the report date. That is true for AMC reporters and
**false for BMO reporters**, whose first post-announcement session is D itself. On 6,573
timestamped BMO events the measured P(|reaction| ≥ 8%) is 0.041; anchored correctly it is
0.173. See `audit/PHASE0_AUDIT_REV2.md` §Q1–Q2.

**What this does and does not establish.** The legacy target is *proven wrong for BMO
events*. Nothing here establishes anything about the model: its validity and its
incremental value remain **unestablished**, pending the corrected-history rebuild (Phase 3)
and a competitive-baseline validation. Do not restate this finding as a claim about the
model being right.

**Since 0.4.0 the corrected target is the production target.** Model B reads only
`abs_reaction_3d_anchored` of completed prior events with `reaction_3d_anchored_status ==
"available"`. The legacy `reaction_*`/`abs_reaction_3d` columns are still computed but only as
the control the anchoring is checked against (the AMC bit-identity below); no score, output or
evaluation reads them, and `testing/test_model_b.py` asserts that statically.

`feature_engineering/announcement_timing.py` owns this. Event-frame columns:

| column | meaning |
|---|---|
| `announce_ts_ny` | observed announcement time, naive **NY local** |
| `announce_ts_source` | provenance of that timestamp |
| `announce_ts_observed_at` | **when** the provider was observed saying it, naive **NY local** — the schedule-vs-observation flag |
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
session in the window**; the AMC events that differ are all missing sessions (10 of 4,842
on the Phase 2 data). That equality is the control.

Rules that must not be relaxed:

- **Never infer BMO/AMC from realized price behavior.** Audit rev-1 did (ticker labelled BMO
  because its day-0 move exceeded its day-+1 move) and every "corrected" number it produced
  was circular. `test_6_the_classifier_never_touches_price` enforces this statically.
- **Never fabricate a timestamp.** The AlphaVantage date-only history stays NULL and its
  events stay unresolved. Since the 2026-09-28 Benzinga seed, 25,883 of 45,713 completed
  events (56.6%) are resolved — ~90%+ of each year from 2013, none before 2011; the rest are
  counted, not guessed at.
- **Never auto-roll a non-trading-day date.** A weekend/holiday date and a date the market
  traded but we failed to ingest have different causes; rolling hides both (§Q6).
- **Only `resolved_events()` may feed a corrected calibration**, and "anchor resolved" is
  not "target available". `anchor_resolved_events()` is the anchoring control slice;
  `resolved_events(events, target=...)` additionally requires that anchored outcome to be
  non-null and defaults to `abs_reaction_3d_anchored`. After the Benzinga seed that is
  25,883 resolved anchors → 25,879 with a 3d target. `resolution_summary()` in
  `announcement_timing.py` reports the counts on whatever DB it is run on.
- **A pre-event timestamp is a schedule, not a record.** A timestamp observed while the
  event was still upcoming may be corrected later, so ingestion refreshes it when a newer
  observation arrives (`refresh_announcement_timestamp`). A timestamp observed *after* the
  announcement is never overwritten. `announce_ts_observed_at` is what tells the two
  apart; where it is NULL, `ingested_at` stands in as a lower bound, and where both are
  NULL nothing is refreshed.
- **One clock: naive New York wall time.** `announce_ts_ny` and `announce_ts_observed_at`
  are both naive NY local, because the refresh rule compares them directly and the
  BMO/AMC cut points (09:30, 16:00) are NY wall-clock facts.
  `utilities/time_utilities.now_ny()` is the only sanctioned source — `datetime.now()`
  returns the *host's* clock, so on a UTC or Israel box an observation made hours before
  an announcement reads as later than it, and a schedule gets frozen into the historical
  record permanently. A static test forbids stamping `announce_ts_observed_at` from a
  host clock; `test_schedule_vs_observation_is_identical_on_every_host` runs the whole
  rule under five host timezones and across both US DST transitions.
  The legacy `ingested_at` is **machine-local and its convention was never recorded**, so
  it is never compared against `announce_ts_ny` raw: the fallback first widens it by
  `MAX_HOST_CLOCK_AHEAD_OF_NY_HOURS` (19 — UTC+14 against EST) into a lower bound that
  holds on any host, and a row is frozen only if it was post-event under *every* possible
  host timezone. The residual error therefore only ever runs toward "still a schedule",
  which a later correctly-stamped observation repairs; a false post-event classification
  would be permanent. Every `announce_ts_observed_at` currently in the DB came from a
  seed's fixed pull date (2026-09-05 audit, 2026-09-10 Benzinga) or from `now_ny()`, so no
  stored row carries a host-local stamp.
- **Seeds are evidence, not runtime inputs.** A yfinance timestamp pull (2026-09-05)
  seeded `earnings.announce_ts_ny` first, via a one-time backfill; the Benzinga seed
  (below) superseded most of it. Both are done, and the files and scripts that produced
  them live in the archive tag `methodology-audit-archive-october-2026`. Ingestion keeps
  the column current from there. Tests assert no production module reads either seed or
  any vendor file (`test_announcement_timing.py`, `test_production_guards.py`).

## Historical Timing Source: Benzinga (seeded once)

The Benzinga Earnings API (via Massive) is Breakwater's **historical announcement-timing
source**. The evaluation verdict is **ACCEPT WITH CONDITIONS** in
`audit/BENZINGA_EARNINGS_AUDIT.md`, which is the artifact to read before doing anything
with it.

**How it reached production — once, as a seed, never as a feed.** On 2026-09-28 snapshot
`earnings_20260910T180412Z` was turned into a seed (25,515 Benzinga exact-date rows + 1,028
yfinance fallback; 214 off-by-one dates and 21 identity-hazard tickers skipped) and
`backfills/load_announcement_seed.py` loaded it into the **droplet** DB, which then syncs
down. On 2026-09-29 `backfills/apply_date_corrections.py` applied the earnings-date
corrections built from the same snapshot. Both loaders are idempotent and already applied.
The acquisition tooling, the seed and correction builders, and the vendor-matching code
are in the archive tag `methodology-audit-archive-october-2026`, not in this tree. No
pipeline stage calls Benzinga or reads vendor files; ingestion keeps timestamps current
from yfinance. Load on the droplet, never locally — `full_workflow.sh` overwrites the
local DB with the droplet's.

Rules for any future use of this vendor, none of which may be relaxed:

- **Raw vendor data lives in `data/vendor/`, which is gitignored.** Licensed third-party
  data in a public repo, and a raw vendor file must never become a production input.
  `testing/test_production_guards.py` asserts statically that no production module
  mentions it and that the directory stays ignored. Never `git add -A` / `git add .`
  in this repo.
- **Never `sort=date.desc` against this vendor.** Its descending cursor returns 180,033 of
  296,334 records and then reports itself finished, with no error — a silent 39% loss.
  Forward or date-partitioned pagination only, and verify the count.
- **`time` is New York local wall clock, despite being documented as "EST"** (audit §2).
  A fixed UTC-5 reading would move every EDT event an hour across the 16:00 cut.
- **`00:00:00` is the vendor's filler for "time unknown", not a BMO announcement.** 7.5%
  of records carry it.
- **Use the production window classifier** (`classify_announce_window` in
  `feature_engineering/announcement_timing.py`), never a second set of cut points, and
  never anything price-derived.
- **Do not join on today's ticker string.** `BF-B` exists in the vendor only as `BF.B` and
  `BFB`; `BRK-B` holds 29 records while `BRK.B` holds 62. `company_name` is not
  point-in-time, and `SNDK` is a reused symbol across a 3,297-day hole under the *same*
  name.

The binding constraint on any future use: usable timing does not reach a steady state
until 2015, and ≥80% of today's universe does not carry 28 prior timed events until year
end **2019**.

## Multi-Week Predictions (`analysis/predictions_range.py`)

Model B calls for N whole Mon-Fri work weeks, backward or forward: tier, percentile, raw
signal, usable priors, cutoff; for history, the corrected 3-session outcome (≥ 5% / ≥ 8%,
`?` where unresolved — never a miss) and any archived call with its model version. Writes
`output/predictions/predictions_<start>_<end>.{csv,txt}` — its own directory, never
`get_run_output_dir()`, which wipes today's run folder on first call. Week arithmetic is
`utilities/data_utilities.week_block_window()`, asserted identical to the digest's window.

History rows are fully causal (each event scored from outcomes complete by its own cutoff)
but still a RETRO-SCORE: nothing was published at the time, and the percentile scale was
built from 2014–2025 events. Archived calls before 0.4.0 came from the retired legacy model.

## Model B (`scoring/model_b.py`) — the production model since 0.4.0

Selected in `analysis/model_c_refit/` (SPEC.md, RESULTS.md): B beats the shipped 0.3.1 score;
C (B + `vol_30d`) adds ROC-AUC but not the top-k ranking the product publishes, and the raw
probability level drifts by year — so ship B and expose a rank, not a probability.

- **Raw signal** `model_b_hist_mean_abs`: mean `abs_reaction_3d_anchored` over the stock's
  prior events with `reaction_3d_anchored_status == "available"` whose endpoint session
  (anchor + 3 market sessions on the grid) is **on or before** the event's call cutoff.
  `model_b_n_prior` counts them. The event's own outcome never enters.
- **Call cutoff** `score_asof_date`: last market session strictly before the Monday of the
  report week. A midweek rerun keeps it; an event whose cutoff has not happened yet uses the
  latest session in the data (never a future one). The pending event needs no timestamp of
  its own.
- **Eligibility**: `model_b_n_prior >= 8`, else `risk_tier = "Unscored"`, `risk_score = NaN`
  (never zero risk; the raw mean is still recorded when any prior exists).
- **Score** `risk_score`: the Breakwater Risk Percentile — the raw signal's position in the
  frozen reference distribution `scoring/model_b_reference.json` (2014–2025 production-fit
  sample, n 20,491, ≥ 8 priors, 1001-point quantile grid, checksummed; regenerate only with
  `scoring/build_model_b_reference.py`, and regenerating is a model change). A presentation
  scale only: no future data enters any raw signal. Display with `display_score()` (floored,
  so the number shown always sits in its tier's band).
- **Ordering**: every list (PDF order, calendar, dashboard, digest, chart table, maintainer
  sheets) goes through `sort_by_risk()`: scored before Unscored, then the RAW
  `model_b_hist_mean_abs` descending, then ticker. Never sort on `risk_score` — it clips to
  0/100 outside the reference range, where distinct raw signals would tie. (The refit's
  pooled OOF top-10 capture 0.2506 ranks by nine separately fitted yearly models; the single
  production map gives 0.2472 pooled and the identical 0.254 within-week, which is what the
  product publishes.)
- **Tiers**: High Alert ≥ 90, Elevated 80–90, Normal < 80, Unscored. Fixed percentile bands,
  not fitted on outcomes — do not move them because of what a backtest shows. Nothing else
  (drift, surprise, lift, volatility, IV) alters a score or tier.
- **Scope**: Model B is primarily a cross-sectional structural earnings-move-risk model. It
  distinguishes stocks that historically have larger corrected earnings reactions; it has
  little demonstrated ability to identify which quarter of the same stock will be unusually
  large (`testing/metric_testing.py`, between- vs within-stock decomposition). Not a defect —
  but never describe it as a quarter-timing model. Customer language: "structural
  earnings-move risk for upcoming earnings events". Keep the decomposition figures internal.
- **Unscored ≠ low risk.** `n_prior < 8` → Unscored stays the rule. It means insufficient
  corrected history, and short-history events are not calmer. Every surface labels it
  (`UNSCORED_NOTE` in `report/recommendations_builder.py`): sorted last for usability, but
  marked "insufficient history — not low risk"; the chart and digest name the Unscored
  tickers they cannot rank instead of dropping them silently.
- **Equivalence**: production reproduces the frozen refit's B feature exactly on all 46,200
  events (identities, cutoffs, n_prior, signal bit-identical; reference sample = the refit's
  candidate-fit sample) — `analysis/model_b_validation/`.
- **Pinned snapshot** (`scoring/model_b_snapshot.py`): the inputs 0.4.0 was approved on —
  the event frame, the market sessions through 2026-09-28 and the refit oracle frame — in
  gitignored, read-only `data/snapshots/model_b_0.4.0_20260928/`, SHA-256-checked on every
  load. Every exact-match check runs on it (`testing/test_model_b_snapshot.py`, the
  equivalence tests, `--check`), never on `output/`: pipeline runs refresh `output/` and
  ingestion revises history (2026-10 date corrections moved the live reference sample to
  20,493). A missing snapshot skips those tests with a message; a changed one fails. Never
  refresh it; never edit its fingerprint to make a check pass. Live-data tests assert only
  invariants that hold on any data. Provenance is in the module docstring. **Fresh clones
  do not have it** — it is private (vendor-derived timing; never commit it): the six
  exact-match tests skip there and `validate` / `--check` stop, until the folder is copied
  in privately; its hashes prove it is the right one.
- **Public dashboard**: scores appear only on upcoming calls. Past events show their corrected
  reaction alone — never a reconstructed score/tier beside an outcome (track record paused).
  Retro-scored history is internal: `analysis/predictions_range.py`.
- **Options/IV** (implied move, ATM IV, snapshot date) is market context on the PDF, digest and
  dashboard only. The old IV-vs-historical-p75 ratio was removed (it used the legacy target).
- Customer wording lives in one place, `report/recommendations_builder.py`. Never write
  "X% chance", "probability of an 8% move", "Nx market", "expected to move" or anything
  directional.

## Key Configuration (`config.py`)

- `MODEL_VERSION = "0.4.0"`, with the version history; `TRACK_RECORD_MODEL_VERSIONS = ("0.4.0",)`.
- `DB_PATH`; `PREDICTIONS_DB_PATH` (overridable with `BREAKWATER_PREDICTIONS_DB`).
- `EXTREME_EARNINGS_REACTION_THRESHOLD = 0.08`: the Model B validation target.
- `PRICES_PROVIDER`: `"ALPHAVANTAGE"` — only used by the legacy `incremental=False` ingestion path, which needs a paid key. The yfinance path (`incremental=True`) ignores it and is the one in routine use; its knobs are `YFINANCE_MAX_WORKERS` and the jitter settings.

## Evaluation

- `testing/calibration.py` (`python -m testing.calibration`): six calibration tables for the
  Model B tiers — hit rates at ≥1/3/5/8/10/15% with Wilson CIs and an Unscored coverage row,
  capture, fixed percentile bands, High Alert by year, every tier by year, score/tier
  consistency — pooled and BMO/AMC; CSVs to `testing/testing_results/` (gitignored).
- `testing/backtesting.py` (`python -m testing.backtesting`): train-year (2013–2016) top-10%
  threshold held through 2017–2025, walk-forward yearly report, High Alert + overlay test.
  Scored events only.

Maintainer evaluation tools, both on the event frame with the corrected target (unresolved
events dropped, never misses), internal only — retro-scored, not a track record:
- `testing/metric_testing.py` (`python -m testing.metric_testing [--section …] [--feature col]`):
  tier profile with Wilson CIs, year-by-year tier stability, walk-forward tier-rate
  calibration (ECE), rank correlation by year, between- vs within-stock decomposition,
  percentile-decile calibration, ROC/PR-AUC and pooled + within-week top-k, feature vs
  target; every pooled table split by BMO/AMC.
- `testing/testing_functions.py`: the evaluation helpers (quintile/decile diagnostics, regime
  tests with (column, quantile) conditions, a vol-only competitive baseline, precision/recall
  with a shuffled-label null, train/test and walk-forward checks). Defaults: score
  `model_b_hist_mean_abs`, train 2013–2016, test 2017+. Pass any event-frame column to test a
  candidate feature against or on top of Model B.
`testing/test_evaluation_tools.py` runs every helper and every metrics section.

Kept but not on the pipeline path (restored 2026-10-04 at Michael's request):
`scoring/scoring_features.py` (the retired 0.3.1 scorer, now self-contained: the event-level
cores from the deleted `feature_engineering/event_features.py` and the legacy 73/79 and lift
constants live inside it; most of its inputs are no longer produced by stage 3),
`feature_engineering/pre_earnings_sector_features.py` (sector drift/vol features, not called
by stage 3) and `analysis/chart_results.py` (the retired 0.3.1 results chart — score
axis, 73/79, legacy reaction; replaced by `analysis/chart_last_week.py` on 2026-10-20). None of them may feed a production score or tier;
`testing/test_model_b.py` checks no pipeline or output module loads them.

Model B's evidence is `analysis/model_c_refit/RESULTS.md` (selection) and
`analysis/model_b_validation/` (production equivalence, regression, percentile-band
behaviour). Prospective evaluation is the archive: first 0.4.0 calls in
`predictions_track_record` against `abs_reaction_3d_anchored >= 0.08`
(`analysis/last_week_results.py`, internal).
