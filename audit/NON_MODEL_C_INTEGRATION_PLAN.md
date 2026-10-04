# First integration plan: everything except Model C

Written 2026-10-03. Planning only: no branch was reset, no file restored, no database written.

- **Base:** `master` = `8fcf3b7` (local and `origin/master` agree; refs not re-fetched).
- **Source of restored files:** tag `methodology-audit-archive-october-2026` → `39fe089`
  (identical to the current `methodology-rebuild` tip).
- **Why wholesale restores are safe in principle:** the merge base of master and the tag
  *is* master's tip. Every archived file already contains all of master's changes, so
  restoring a file from the tag never drops a master fix. The only risks are what each
  file *drops in* (dependencies, references to removed paths), and those are traced below.
- **Fixed decisions applied:** production timing rules win (no ±1-day vendor matching, no
  proxy dates, no rolling); no Model C; research stays archived; `predictions_range.py`
  is kept; future research branches from future master.

Checks behind this plan, all read-only:

- **Static import closure.** An AST walk over every `.py` file in the restore set below
  resolves each `from X import name` against the restore set (tag version) or master.
  Result: **0 missing modules, 0 missing names, 0 imports of `research.*`, `audit.*` or
  `backfills.build_*`**.
- **Reverse callers.** Every caller of a function whose signature changes
  (`stage5`, `generate_reports`, `generate_calendar`, `build_calendar_data`,
  `generate_streamlit_df`, `export_upcoming_df`, `save_predictions_snapshot`,
  `load_announcement_timing`) was found across the whole tree. Two callers stay as they
  are on master and are affected; both are covered in "Risks".
- **Readers of upcoming state.** Every forward-looking reader goes through
  `pipeline.events.pending_events` or `output/upcoming_df.parquet`, which
  `streamlit_export` builds from pending rows. That covers the digest
  (`cron/cron_weekly_digest.py`), the weekly chart (`analysis/chart_weekly.py`) and the
  dashboard. No remaining production reader uses `groupby("stock").last()`.
- **Predictions DB.** `db/predictions.duckdb` on master and on the tag were compared row
  by row through read-only copies in the job scratch directory.

---

## 1. Summary

| | count |
|---|---|
| files restored **wholesale** from the archive tag | **28** |
| files needing a **SELECTIVE PATCH** | **6** |
| **new** file (port of 2 guard tests out of an archived test file) | **1** |
| optional one-line comment patch (`config.py`) | 1 |
| files to **delete** from master (deliberate deletions on the rebuild) | **15** |
| **one-time migrations** to perform | **1**: merge the 28 September calls into `db/predictions.duckdb` |
| files left **exactly as master has them** but named here because they matter | 9 |

Nothing in this integration changes a score, tier or threshold on completed events.
`pipeline/events.py` asserts on every run that completed events are identical to the
daily pipeline (`assert_completed_parity`). The visible behaviour changes are:

1. Upcoming calls come from the pending event row instead of the stale
   `.last()` row (audit §Q4).
2. The weekly calendar shows upcoming events again (it rendered zero).
3. The event frame gains timing columns and the parallel `*_anchored` target. Nothing
   consumes them for scoring.
4. Retracted figures disappear from the report, readme and marketing.
5. The public track record stays paused.
6. Predictions get `score_asof_date` and the pre-audit calls are voided.

---

## 2. Restore manifest

`source` = `methodology-audit-archive-october-2026` unless stated.
Type: **P** permanent production code · **T** permanent production test · **M** one-time
migration/backfill · **Doc** documentation · **—** deletion.

### 2a. Wholesale restores (28)

| file | restore? | source | type | why | dependencies (imports → / imported by ←) | tests | notes |
|---|---|---|---|---|---|---|---|
| `feature_engineering/announcement_timing.py` | yes, wholesale | tag | P | BMO/AMC from timestamp only; canonical market-session grid; anchors; `reaction_{1,3,5}d_anchored` + per-horizon status; explicit `unresolved_*` states; `resolved_events()` | → numpy, pandas only. ← `pipeline/events.py` | `test_announcement_timing.py` | Implements the strict production rules (exact `earnings_date`, no rolling, refuse on missing session). Contains none of the research proxy-date logic, which lives only in `research/phase3_target_rebuild.py` |
| `feature_engineering/event_features.py` | yes, wholesale | tag | P | One event-indexed implementation of every across-events statistic, so the pending row is computed by the same code as history | → `config`. ← pre/post feature files, `scoring_features.py`, `pipeline/events.py` | `test_event_frame.py` (parity, invariants) | Pure refactor of master's lambdas; parity proven on every run |
| `feature_engineering/pre_earnings_stock_features.py` | yes, wholesale | tag | P | Delegates median/p75/rolling p75/p90/surprise/drift-z to `event_features` | → `event_features`, `config`. ← stage3 | `test_event_frame.py`, `test_pipeline.py` | No number changes |
| `feature_engineering/post_earnings_stock_features.py` | yes, wholesale | tag | P | Delegates reaction_std/entropy/directional_bias | → `event_features`. ← stage3 | same | No number changes |
| `scoring/scoring_features.py` | yes, wholesale | tag | P | Delegates scoring cores so pending rows are scored identically | → `event_features`, `config`. ← stage4, `pipeline/events.py` | same | Still the legacy 0.3.1 score. Replaced later by Model C |
| `utilities/db_utilities.py` | yes, wholesale | tag | P | `load_announcement_timing()` (event frame's only timing source, reads the DB only); predictions schema: `score_asof_date`, `void_for_track_record`, `void_reason`, idempotent voiding of the 10 pre-audit calls, `predictions_track_record` view | ← `pipeline/events.py`, `save_predictions.py`, marketing generator | `test_announcement_timing.py` (loader, schema) | The schema changes are **self-applying** each time `create_predictions_table_if_not_exists` runs. That is runtime code, not a separate migration |
| `pipeline/events.py` | yes, wholesale | tag | P | Stage 4b event frame: every completed event + one pending row per stock; timing attached; completed-event parity asserted; `pending_events()` | → `event_features`, `announcement_timing`, `scoring_features`, `db_utilities`, `config`. ← `pipeline.py`, `stage5.py`, `report_builder`, `calendar_builder`, `streamlit_export`, `save_predictions` | `test_event_frame.py`, `test_announcement_timing.py` | Reads timing from the `earnings` table only. A static test forbids reading `audit/provider_timestamps.parquet` |
| `pipeline/pipeline.py` | yes, wholesale | tag | P | Inserts stage 4b and passes `events_df` to stage5 | → `pipeline.events`. ← `main.py` | `test_event_frame.py` | Must land together with `stage5.py` |
| `pipeline/stage5.py` | yes, wholesale | tag | P | Writes `output/events_df.parquet`; passes `events_df` to every forward-looking consumer; builds it itself if called without one (keeps the re-score one-liner working) | → `pipeline.events`, the 4 consumers below. ← `pipeline.py` | `test_event_frame.py` | Must land together with the 4 consumers (their signatures change) |
| `report/report_builder.py` | yes, wholesale | tag | P | PDF reports from the pending row; tier reindex fixes KeyError | → `pipeline.events.pending_events`. ← stage5 | indirect (`test_event_frame`) | |
| `streamlit_dash/streamlit_export.py` | yes, wholesale | tag | P | `upcoming_df.parquet` from pending rows; adds `score_asof_date` | → `pipeline.events`. ← stage5; `pipeline/incremental.py` (dead, see Risks) | `test_event_frame.py::…export_upcoming_df…` | `upcoming_df` feeds the digest, weekly chart and dashboard. That format has shipped from the rebuild since 2026-09-06 |
| `analysis/save_predictions.py` | yes, wholesale | tag | P | Archives the pending-row call; records `score_asof_date` | → `pipeline.events`, `db_utilities`. ← stage5 | none direct | |
| `marketing/generate_public_track_record.py` | yes, wholesale | tag | P | P4.3 pause (refuses to publish; stage5 skips cleanly), P4.2 `drop_voided()` | ← stage5 (same call signature as master) | none direct | Keeps the public record paused. Introduces no figures |
| `report/templates/earnings_report.html` | yes, wholesale | tag | P | Removes retracted "4.78× OOS lift"; per-stock lift rows commented out (P4.1) | ← `report_builder.py` | — | Shows the verified audit figures for the **current** legacy tiers. Must be rewritten when Model C ships |
| `utilities/data_utilities.py` | yes, wholesale | tag | P | Adds `week_block_window()` | ← `analysis/predictions_range.py`, its test | `test_predictions_range.py` | Kept because `predictions_range.py` is kept (decision D) |
| `analysis/predictions_range.py` | yes, wholesale | tag | P | Maintainer CLI: predictions for N whole work weeks | → `config`, `data_utilities`; reads `output/events_df.parquet`, `db/predictions.duckdb` (read-only) | `test_predictions_range.py` | Its `*_pre_audit` columns stay meaningful until Model C. Strip them then (see §6) |
| `scripts/full_workflow.sh` | yes, wholesale | tag | P | Skips the `recent_calls.json` push while P4.3 is in force instead of aborting under `set -e` before the digest | — | — | Without this, the Monday run dies at step 4 once master's generator is paused |
| `scripts/sync_pipeline.sh` | yes, wholesale | tag | P | Pull-only, deliberately (`85a929d`) | — | — | |
| `testing/metric_testing.py` (rename of `testing/testing.py`) | yes, as a rename | tag | P (maintainer script) | User's own rename (`39fe089`); file reads a parquet at import time, so "testing.py" read as a test suite | → `testing.testing_functions` | — | `git mv testing/testing.py testing/metric_testing.py` then restore content. References fixed in the CLAUDE.md/readme patches and the optional `config.py` patch |
| `testing/test_announcement_timing.py` | yes, wholesale | tag | T | 71 tests: classifier cut points, session-grid anchoring, refusal on gaps, never infer timing from price, `now_ny`/host-clock rules, schema write path, no pipeline module reads the audit parquet | → `announcement_timing`, `pipeline.events`, ingestion, `db_utilities`, `time_utilities`, `testing.test_pipeline` | itself | Real-data tests skip without `output/full_df.parquet` or DB timestamps. The audit-parquet guard stays useful once the parquet is gone |
| `testing/test_pipeline.py` | yes, wholesale | tag | T | Docstring update only | — | itself | |
| `testing/test_predictions_range.py` | yes, wholesale | tag | T | 20 tests for the CLI | → `analysis.predictions_range`, `data_utilities` | itself | |
| `marketing/positioning.md` | yes, wholesale | tag | Doc | Replaces retracted proof points with verified audit figures + scope limits | cites `audit/PHASE0_AUDIT_REV2.md` (restored) | — | Re-derive when Model C ships |
| `marketing/post_templates.md` | yes, wholesale | tag | Doc | Retractions; results template on hold | same | — | |
| `marketing/content_calendar.md` | yes, wholesale | tag | Doc | Track-record posts paused | same | — | |
| `marketing/reddit_playbook.md` | yes, wholesale | tag | Doc | Retracted figures removed from suggested comments | same | — | |
| `audit/PHASE0_AUDIT_REV2.md` | yes, wholesale | tag | Doc | The authority for every performance number; cited by the readme, template, marketing, generator and `db_utilities` void reason | — | — | Only file in `audit/` besides the Benzinga audit and these two plan documents |
| `audit/BENZINGA_EARNINGS_AUDIT.md` | yes, wholesale | tag | Doc | Provenance of ~25.5k timestamps in the production `earnings` table (`announce_ts_source = massive_benzinga:…`) and of the NY-clock/midnight rules | cited by CLAUDE.md | — | Frozen document; its generator stays archived |

### 2b. SELECTIVE PATCH (6)

| file | restore? | source | type | why | dependencies | tests | notes |
|---|---|---|---|---|---|---|---|
| `.gitignore` | **SELECTIVE PATCH** | tag, edited | P | **Must be the first file restored after the reset.** Master does not ignore `data/vendor/` (**16 GB of licensed Benzinga data on this machine**), `data/benzinga_earnings/` (201 MB), `data/benzinga_features/` (8.5 MB) or `audit/phase1_golden/daily_df_pre.parquet` (309 MB). On master they would show up as untracked, and one `git add -A` would put licensed data in a public repo | — | the ported vendor guard (2c) | Take the tag's vendor and Benzinga ignore lines. Reword the comment that points at `research/massive/paths.py` and `test_massive_earnings.py` (both gone) to point at the new guard test. The `daily_df_pre.parquet` line can go once that local file is deleted (§8); keep it until then |
| `testing/test_event_frame.py` | **SELECTIVE PATCH** | tag, minus golden tests | T | 30 tests guarding parity, pending-row invariants, no `.last()` in production, drift-flag window | → `pipeline.events`, `event_features`, feature files, `scoring_features`, `streamlit_export`, `testing.test_pipeline` | itself | From the tag, **remove** `GOLDEN_UPCOMING_PATH`, the `golden_upcoming` fixture, `test_pending_drift_flag_matches_legacy_golden` and `test_pending_high_conviction_matches_legacy_golden`. Their fixture (`audit/phase1_golden/upcoming_df.parquet`) is not restored and they have skipped on every run since the data moved past the snapshot. Everything else as in the tag |
| `report/calendar_builder.py` | **SELECTIVE PATCH** | tag + master fallback | P | The tag version fixes the weekly calendar (pending rows, window opens today). But it returns **no events when called without `events_df`**, which is how `streamlit_dash/app.py:480` (the dashboard's "Export calendar HTML" button, stays as master) calls it. On master that button exports the selected window from completed rows; after a wholesale restore it would always say "no scored earnings events" | → `pipeline.events.pending_events`. ← stage5, `streamlit_dash/app.py` | add one test for the `events_df=None` path | From the **tag**: `events_df` parameter, pending-row window, `reference_date` default = today. From **master**: when `events_df is None`, select the window from `earn_all` as before. Small new code: one branch. Alternative (bigger): pass `events_df` from the app, but the droplet has no `events_df.parquet` (`full_workflow.sh` does not push it) |
| `readme.md` | **SELECTIVE PATCH** | tag, edited | Doc | Tag version retracts the 40% / 5.8x / "2015–2025 OOS" headline and table (both still live on master) | cites `audit/PHASE0_AUDIT_REV2.md` | — | Take the tag text. Fix the layout tree: drop `window_sensitivity.py`, `testing.py` → `metric_testing.py`, add `pipeline/events.py`, `feature_engineering/announcement_timing.py`, `event_features.py`. Add no new figures |
| `CLAUDE.md` | **SELECTIVE PATCH** | tag, edited | Doc | Tag version holds the rules production now depends on: performance-claims table, timing rules, event frame, `now_ny`, schedule-vs-observation, consumers read pending rows | — | — | **Keep from tag:** everything above plus Branches, Brain (tag wording supersedes master's; master was merged into it), Commands, Pipeline (with 4b), Data Storage, Announcement Timing, Multi-Week Predictions, Risk Scoring, Backtesting. **Remove or rewrite:** the `audit.phase2_diagnostics` command; both citations of `audit/PHASE2_DIAGNOSTICS.md` (restate the fact inline: "10 of 4,842 AMC events differ, all missing sessions"); the "Historical Timing Source: Massive / Benzinga" section's `research.massive.*` commands, `research/` rules and `backfills/build_announcement_seed.py` story. Keep a short paragraph: timestamps were seeded once on 2026-09-28 from Benzinga snapshot `earnings_20260910T180412Z` + yfinance, loaded by `backfills/load_announcement_seed.py`; the seed builder and vendor tooling are in the archive tag; never `sort=date.desc`; vendor time is NY wall clock; midnight = unknown; never join on today's ticker. `python -m testing.testing` → `python -m testing.metric_testing`; `testing/testing.py` → `testing/metric_testing.py`. "The legacy columns … remain the production target" stays true until Model C |
| `.claude/memory/MEMORY.md` | **SELECTIVE PATCH** | **working tree** (not the tag) | Doc | Session memory required by CLAUDE.md | — | — | The working-tree copy has today's uncommitted entries, which the tag lacks. **Copy it out before the reset** (§9 step 2) and put it back after. Add an integration entry; drop "Current state" bullets that describe research directories as present |

### 2c. New file (1) and optional patch (1)

| file | restore? | source | type | why | dependencies | tests | notes |
|---|---|---|---|---|---|---|---|
| `testing/test_production_guards.py` *(new name, suggested)* | **NEW** (port) | `testing/test_massive_earnings.py` @ tag, 2 tests + helpers | T | Keeps two production invariants whose only home is an archived research test file: `test_no_production_module_depends_on_a_raw_vendor_file`, `test_the_vendor_directory_is_gitignored` | port the helpers `_python_files`, `_code_only` and the `PRODUCTION_DIRS` list; drop `research.massive` imports | itself | Keep the `'massive_benzinga:'` provenance-label exemption: `utilities/db_utilities.py` on master compares that stored label in the duplicate-date cleanup |
| `config.py` | optional | master + 1 comment | P | Line 38 comment points at `testing/testing.py` | — | — | Comment only. Skip if you want `config.py` untouched |

### 2d. Deletions from master (15)

Deliberate deletions made on the rebuild branch. Nothing on master imports any of them (checked with `git grep` on master).

| file | restore? | type | why |
|---|---|---|---|
| `.claude/memory/codebase_audit_2026_06_07.md`, `infra_digitalocean.md`, `next_to_build.md`, `project_direction.md`, `reddit_marketing_playbook.md`, `social_media_strategy.md`, `window-sensitivity.md` (7) | delete | — | Folded into `MEMORY.md` (`fa2ab55`). Note: `infra_digitalocean.md` holds the droplet crontab; check its content made it into MEMORY.md or the brain before deleting |
| `info/droplet_project_update_instructions.txt`, `info/metrics_to_build.txt`, `info/recall_precision_for_different_quantiles_results.ods` (3) | delete | — | Removed as unused (`2297aaf`) |
| `pipeline/__pycache__/pipeline.cpython-39.pyc` | delete | — | Committed bytecode |
| `scripts/delete_iv_rows_on_closed_days.py` | delete | M (done) | One-time IV cleanup, run 2026-09-29; deleted deliberately by the user |
| `testing/inspect_db.py`, `testing/weekly_prediction_quality.py`, `testing/window_sensitivity.py` (3) | delete | — | Removed as unused (`2297aaf`) |

### 2e. Stays exactly as master has it (named because it matters)

| file | type | why it stays |
|---|---|---|
| `ingestion/fetch_earnings_dates.py` | P | Already holds every runtime timing/date rule: `refresh_announcement_timestamp` (schedule vs observation), the 19-hour host-clock widening, duplicate cleanup preferring the observed-timestamp row. Not in the diff |
| `utilities/time_utilities.py` | P | `now_ny`, `nyse_is_open`, already on master |
| `ingestion/fetch_iv.py`, `cron/*` | P | IV fixes already on master; cron unchanged |
| `config.py` | P | Unchanged by the rebuild (bar the optional comment) |
| `backfills/load_announcement_seed.py` | M (done 2026-09-28) | One-time loader, already applied on the droplet. Stays because master's `testing/test_announcement_ingest.py` imports it. Remove later only together with those tests |
| `backfills/apply_date_corrections.py` | M (done 2026-09-29) | Same; imported by master's `testing/test_earnings_dates.py` |
| `streamlit_dash/app.py` | P | Unchanged; depends on the `calendar_builder` patch above |
| `pipeline/incremental.py` | P (dead) | Nothing imports it, and it is already broken on master (`run_pipeline()` without its argument). After the restore its `export_upcoming_df(df)` would also be wrong. Leave it; remove or fix in a later cleanup |
| `testing/test_announcement_ingest.py`, `test_earnings_dates.py`, `test_iv_collection.py` | T | Master's own production tests; unchanged |

### 2f. Dependency questions, per group

Restoring a single file of a group on its own breaks master. Restore groups whole.

| group | files | imports outside the group | anything archive-only? | breaks master if restored alone? | protected by |
|---|---|---|---|---|---|
| G1 timing | `announcement_timing.py` | none | no | no (nothing calls it until G3) | `test_announcement_timing.py` |
| G2 shared features | `event_features.py` + pre/post feature files + `scoring_features.py` | `config` | no | `event_features.py` alone is harmless; the other three need it | `test_event_frame.py`, `test_pipeline.py` |
| G3 event frame + consumers | `events.py`, `pipeline.py`, `stage5.py`, `report_builder.py`, `calendar_builder.py`, `streamlit_export.py`, `save_predictions.py`, `db_utilities.py` | G1, G2, `config` | no | **yes**: stage5 and the consumers change signatures together, and `save_predictions` needs `db_utilities`' new `score_asof_date` column | `test_event_frame.py`, `test_announcement_timing.py` |
| G4 retractions | template, `generate_public_track_record.py`, `full_workflow.sh`, readme, marketing docs, `PHASE0_AUDIT_REV2.md` | none (generator uses `predictions_track_record` from G3's `db_utilities`, read path only) | no | the generator alone is fine; **`full_workflow.sh` must come with the generator** or the Monday run aborts | — |
| G5 maintainer CLI | `predictions_range.py`, `data_utilities.py`, its test | reads `output/events_df.parquet` (made by G3) | no | needs G3 to have produced `events_df.parquet` at least once | `test_predictions_range.py` |

**Data assumptions:** no restored file reads anything production will not have.
`events.py` and `db_utilities.load_announcement_timing` read the `earnings` table (the
DB already holds the seeded timestamps: 26,657 of 47,503 rows). No restored file reads
`data/vendor/`, `audit/*.parquet` or `output/phase3_*`. On a DB without timestamps every
event is UNKNOWN/unresolved, which is the intended degradation, not a crash.

**Inventory correction:** every file `FINAL_BRANCH_INVENTORY.md` marked A is clean of
archive-only dependencies. The inventory did not catch three things, all handled above:

1. the dashboard calendar-export regression;
2. master's `.gitignore` exposing `data/vendor/`;
3. `MEMORY.md` must come from the working tree, not the tag.

---

## 3. Permanent vs one-time vs docs

| kind | items |
|---|---|
| **Permanent production code** | `announcement_timing.py`, `event_features.py`, `pre_earnings_stock_features.py`, `post_earnings_stock_features.py`, `scoring_features.py`, `db_utilities.py`, `data_utilities.py`, `pipeline/events.py`, `pipeline/pipeline.py`, `pipeline/stage5.py`, `report_builder.py`, `calendar_builder.py` (patched), `earnings_report.html`, `streamlit_export.py`, `save_predictions.py`, `generate_public_track_record.py`, `predictions_range.py`, `full_workflow.sh`, `sync_pipeline.sh`, `metric_testing.py` (maintainer script), `.gitignore` (patched) |
| **Permanent production tests** | `test_announcement_timing.py`, `test_event_frame.py` (patched), `test_pipeline.py`, `test_predictions_range.py`, new `test_production_guards.py`, plus master's unchanged `test_announcement_ingest.py`, `test_earnings_dates.py`, `test_iv_collection.py` |
| **One-time migration, still to do** | merge 28 September calls into `db/predictions.duckdb` (§5) |
| **One-time migrations, already done, NOT restored** | `backfills/build_announcement_seed.py` (seed built 2026-09-28), `backfills/build_date_corrections.py` (corrections built 2026-09-29), `scripts/delete_iv_rows_on_closed_days.py` (run 2026-09-29), `audit/fetch_provider_timestamps.py` (pulled 2026-09-05) |
| **One-time loaders already on master** | `backfills/load_announcement_seed.py`, `backfills/apply_date_corrections.py` (kept as master; tests import them) |
| **Runtime logic that keeps history correct going forward** (already on master) | `refresh_announcement_timestamp`, duplicate-date cleanup preferring observed timestamps, `now_ny`, schema self-migration in `create_earnings_table_if_not_exists` |
| **Identity protection** | Production has no vendor join, so it needs no runtime identity guard; it ingests yfinance by today's ticker plus `data/ticker_renames.csv`. `identity_hazards()` lived only in `research/phase3_target_rebuild.py` for the one-time seed. **Not restored.** Recover it from the tag if a vendor join is ever done again |
| **Documentation** | `PHASE0_AUDIT_REV2.md`, `BENZINGA_EARNINGS_AUDIT.md`, readme, CLAUDE.md, MEMORY.md, 4 marketing files, these two plan documents (keep or drop at your choice) |
| **Do not restore** | §7 |

---

## 4. Downstream consumers (complete list)

| consumer | on master reads | after integration reads | change |
|---|---|---|---|
| PDF reports (`report_builder`) | `.last()` + `earnings_df.iloc[-1]` (stale) | `pending_events(events_df)` | restored |
| Weekly calendar (`calendar_builder`, via stage5) | completed rows only → **zero events every run** | pending rows, window from today | restored + patch |
| Dashboard calendar export (`app.py` → `generate_calendar`) | completed rows in the picked window | same, via the master fallback in the patch | patch keeps it working |
| `upcoming_df.parquet` (`streamlit_export`) | `.last()` (stale) | pending rows | restored |
| Weekly digest (`cron/cron_weekly_digest.py`) | `upcoming_df.parquet` | same file, now correct | none needed |
| Weekly chart (`analysis/chart_weekly.py`) | `upcoming_df.parquet` | same file, now correct | none needed |
| Dashboard (`streamlit_dash/app.py`) | `full_df`, `streamlit_df`, `upcoming_df` | same files; `upcoming_df` has `score_asof_date` extra | none needed (already served this format since 2026-09-06) |
| Predictions archive (`save_predictions`) | `.last()` (stale) | pending rows + `score_asof_date` | restored |
| Public track record (`generate_public_track_record`) | publishes | paused (P4.3), voided calls dropped (P4.2) | restored |
| `analysis/last_week_results.py`, `results_check.py`, `chart_results.py` | completed events in `full_df` | unchanged | none needed (history only) |
| `pipeline/incremental.py` | — | dead; already broken | leave (Risks) |

---

## 5. `db/predictions.duckdb`

### 5.1 What differs

Compared row by row (read-only copies of both blobs):

| | master (`7684cf8…`) | archive tag = working tree (`07af710…`) |
|---|---|---|
| rows | 10 | 38 |
| `model_version = '0.3.1-preaudit'`, voided | 0 | 10 (the same 10 master rows, re-stamped) |
| `model_version = '0.3.1'` | 10 (2026-08-31, `f3dd1e2`) | 28 (2026-09-06 → 09-28) |
| columns | no `score_asof_date` / `void_*` | has them |
| views | `predictions_week_open` (old) | `predictions_first_call`, `predictions_track_record` |
| unique index | `predictions_uq (stock, earnings_date, prediction_asof_date)` | same |

- Master keys missing from the tag copy: **0**.
- Rows sharing a key that differ outside `model_version`/`void_*`: **0**.
- **The tag copy is a strict superset of master's.**
- The 28 extra rows: 2026-09-06 (ADBE, COO, CPRT, KR, ORCL), 09-12 (LEN), 09-20 (AZO,
  COST, CTAS, DRI, GIS, PAYX), 09-27 and 09-28 (ACN, CAG, CCL, FDS, JBL, MKC, MU, NKE).
  Each was written by a commit that exists only on `methodology-rebuild`.
- The working-tree file is byte-identical to the tag's blob (`git hash-object` = `07af710`).
  Its mtime is 2026-09-29 18:11, so no call has been written since.

### 5.2 Can master or the droplet hold newer calls?

- **Master in git:** no. Its blob dates from `2b79144` (2026-08-31). `origin/master` =
  local master.
- **Droplet:** the evidence says no. `save_predictions_snapshot` runs only in stage5. The
  droplet runs `cron_ingest` (stage1 only), `cron_iv` and `cron_eps_estimates`, and serves
  the dashboard. The digest runs locally from `full_workflow.sh`. Master's own
  `.gitignore` says of this file: "it is the only copy, nothing syncs it."
  `full_workflow.sh` never pushes it. **This is inference, not observation.** Confirm
  read-only before the migration (§9 step 1).
- **The real risk is local and comes from timing.** The pipeline writes to whatever
  `db/predictions.duckdb` the checked-out branch has. Two ways to lose calls:
  1. A Monday run happens on `methodology-rebuild` before the reset. Its new rows sit
     uncommitted in a tracked file, and `git reset --hard master` **silently destroys
     them**.
  2. The reset happens and a Monday run follows before the migration. It appends to
     master's 10-row copy, and a later whole-file copy of the old 38-row file **destroys
     the new rows**.

  Both are avoided by backing up first and merging by key, not by whole-file replacement.

### 5.3 Keep tracking it in git for this integration?

Yes. Changing that is an architecture decision, and it is out of scope. The integration
commit carries the merged file. **Later:** the repository should probably stop tracking a
mutable production database (a binary that changes weekly, conflicts on any branch
switch and is backed up only by commits). Decide that separately.

### 5.4 Merge procedure (the one-time migration)

Key: `(stock, earnings_date, prediction_asof_date)`, the existing unique index.

1. **Before any reset:**
   `cp db/predictions.duckdb ~/breakwater_backups/predictions_rebuild_$(date +%Y%m%d_%H%M%S).duckdb`.
   Note whether `git diff --quiet -- db/predictions.duckdb` is clean. If it is not, a run
   wrote new rows since the last commit, and the backup holds them.
2. **Droplet, read-only:**
   `ssh root@harbor-markets.com 'ls -la /var/www/breakwater/db/; git -C /var/www/breakwater status --short db/'`.
   If a `predictions.duckdb` there is modified or newer than master's blob, `scp` a copy
   into `~/breakwater_backups/` as a third source.
3. **After the reset** (the working file is now master's 10-row copy), merge into a
   **fresh file**, never in place:

   ```sql
   -- duckdb ~/breakwater_backups/merged.duckdb
   ATTACH 'db/predictions.duckdb'                          AS m (READ_ONLY);  -- master copy
   ATTACH '~/breakwater_backups/predictions_rebuild_….duckdb' AS r (READ_ONLY);  -- backup
   -- (ATTACH the droplet copy as d too, if step 2 produced one)

   -- Check A: nothing in master that the backup lacks (expected 0)
   SELECT count(*) FROM (SELECT stock, earnings_date, prediction_asof_date FROM m.predictions
                         EXCEPT SELECT stock, earnings_date, prediction_asof_date FROM r.predictions);
   -- Check B: shared keys agree outside the voiding columns (expected 0)
   SELECT count(*) FROM (
     SELECT prediction_asof_date, run_week, week_start, stock, earnings_date, tier, risk_score,
            is_high_conviction, pre_earnings_drift_flag, surprise_momentum_flag, git_commit, ingested_at
     FROM m.predictions
     EXCEPT
     SELECT prediction_asof_date, run_week, week_start, stock, earnings_date, tier, risk_score,
            is_high_conviction, pre_earnings_drift_flag, surprise_momentum_flag, git_commit, ingested_at
     FROM r.predictions);
   ```

   - **If A = 0 and B = 0** (true today), the key-merge result *is* the backup's row set.
     Copy the backup into the merged file:
     `CREATE TABLE predictions AS SELECT * FROM r.predictions` plus the unique index. Or
     simply copy the file.
   - **If either is non-zero** (a run wrote to master's copy, or a droplet copy exists):
     1. Start from the backup's table (it has the newer schema).
     2. Add the missing columns to any older source by running
        `create_predictions_table_if_not_exists` on a scratch copy of it, never on the
        original.
     3. Run `INSERT INTO predictions SELECT … FROM <source> ON CONFLICT DO NOTHING`, column
        by column, for each other source.
     4. Report every conflicting key where the two sources disagree, and resolve it by
        hand. Do not overwrite.
4. Open the merged file through `utilities.db_utilities.create_predictions_table_if_not_exists`
   (idempotent: views, voiding, `run_week`).
5. Verify:
   - `count(*) >= 38`;
   - 10 rows voided as `0.3.1-preaudit`;
   - 28 rows `0.3.1` dated 09-06…09-28;
   - every key from every source present;
   - `predictions_track_record` excludes the voided rows.
6. Move the merged file to `db/predictions.duckdb` and commit it with the integration.
   Keep the backups outside the repo until the droplet has pulled and one Monday run has
   appended correctly.
7. **Droplet pull:** check `git -C /var/www/breakwater status --short db/` first. If the
   droplet copy is locally modified, `git pull` refuses rather than overwriting. That is a
   safe failure; go back to step 3 with that copy.

---

## 6. DEFERRED TO MODEL C INTEGRATION

Left out on purpose. Nothing here is implemented.

- **Causal historical mean absolute reaction:** `hist_mean_abs` on
  `abs_reaction_3d_anchored`, ≥ 8 prior outcomes, an outcome counted only once its
  3-session endpoint is at or before the call cutoff. New core in `event_features.py`.
  Research version: `research/phase3_refit/features.history_features`.
- **Call-cutoff construction on the production event frame:** `call_cutoff_date` (last
  session strictly before the Monday of the report week) and `endpoint3_date` columns.
  `vol_30d` read as of the call cutoff for history rows; production carries the
  earnings-day value today. Research version: `features.event_clocks`, `daily_asof`.
- **Model C fitting and calibration:** yearly walk-forward, log inputs, training-fold
  standardisation, L2 logistic C=1, train = endpoint before 1 January. Offline fit script
  plus a versioned coefficients file. Research: `research/phase3_refit/calibration.py`,
  `candidates.py`.
- **Coefficients:** refit on the **production** `events_df` (strict timing rules). The
  research coefficients (−1.854 / 0.740 / 0.275) and cuts (0.238 / 0.328) were measured on
  the research frame and must not be copied.
- **Score and probability semantics:** what `risk_score` means (probability vs 0–100);
  `MODEL_VERSION` bump; report, dashboard and digest wording.
- **Thresholds and tiers:** common quantile cuts (top 10% High Alert, next 10% Elevated)
  from training-fold probabilities. Replaces `BUCKET_ELEVATED_FLOOR` / `BUCKET_HIGH_ALERT_FLOOR`.
- **Replacing the old scoring system:** remove the 12% ceiling, entropy term,
  `stock_bucket_lift` and lift promotion (`LIFT_*`, `LIFT_PRIOR_STRENGTH`), High Conviction.
  Retire the legacy cores in `event_features.py` and `scoring_features.py`.
- **Production-vs-research equivalence:** on the production frame, rebuild the research
  leakage tests as production tests: own outcome excluded, future events cannot move
  earlier scores, endpoint availability, call cutoff = Friday before report week,
  training ends before test year, cuts never see the test year, 10%/10% flag rates,
  deterministic tiers. Template: `testing/test_phase3_refit.py` @ tag. Measure how far
  production-frame Model C sits from the research numbers, and report it.
- **Acceptance gate:** rewrite `testing/calibration.py` for the anchored target and
  probabilities (tier hit rates, capture, calibration by window, stratified lift).
- **Public claims:** new figures only from a fresh audit of the production Model C.
  Rewrite the report template, readme, `positioning.md` and the CLAUDE.md claims table at
  that point. Re-check the P4.3 un-pause checklist.
- **`predictions_range.py`:** drop the `*_pre_audit` / `pre_audit_differs` columns. They
  reproduce the `.last()` bug, which means nothing once the old score is gone.
- **Model C write-up:** bring `research/phase3_refit/RESULTS.md` across as a doc (e.g.
  `docs/model_c.md`) with the production-frame numbers, not now.

---

## 7. Cleanup manifest: NOT restored

Everything below disappears from the tree when the branch is rebuilt from master, and
stays recoverable from the tag.

| group | paths |
|---|---|
| Research package root | `research/__init__.py` |
| Phase 3 target rebuild (±1-day vendor match, proxy dates) | `research/phase3_target_rebuild.py`, `testing/test_phase3_target_rebuild.py` |
| Phase 3 refit / Model C evaluation machinery | `research/phase3_refit/` (`__init__`, `PREREGISTRATION.md`, `RESULTS.md`*, `candidates.py`, `features.py`, `calibration.py`, `evaluate.py`), `testing/test_phase3_refit.py`, `research/phase3_cap_tiebreak.py`, `testing/test_phase3_cap_tiebreak.py` |
| Phase 4 / 5 baselines and feature search | `research/phase4_baselines.py`, `research/phase5_event_signal.py`, `research/score_validity_and_deviation_features.py`, `research/benzinga_feature_value.py`, `research/phase_5b_new_feature_testing/` (incl. its `tests/`), `testing/test_phase4_baselines.py`, `test_phase5_event_signal.py`, `test_score_validity_and_deviation_features.py`, `test_benzinga_feature_value.py` |
| Vendor tooling (Massive/Benzinga) | `research/massive/` (10 files), `testing/test_massive_earnings.py` (2 guards ported first) |
| Options (paused) | `research/options_pilot/`, `research/options_confirmatory/`, `testing/test_options_pilot.py`, `testing/test_options_confirmatory.py` |
| SEC research | `research/sec_filings_pilot/`, `research/sec_features/` (incl. `manual_audit.csv`), `testing/test_sec_filings.py`, `testing/test_sec_features.py` |
| Fundamentals research | `research/fundamentals/`, `testing/test_fundamentals.py` |
| One-time builders (done) | `backfills/build_announcement_seed.py`, `backfills/build_date_corrections.py`, `testing/test_build_announcement_seed.py`, `testing/test_build_date_corrections.py` |
| Audit scripts | `audit/fetch_provider_timestamps.py`, `phase2_diagnostics.py`, `probe_announcement_hours.py`, `probe_announcement_timing.py`, `quantify_bmo_bias.py`, `quantify_score_staleness.py`, `staleness_final_tier.py`, `stratified_lift.py`, `tier_by_timing.py`, `verified_timing_analysis.py` |
| Superseded / generated audit docs | `audit/PHASE0_AUDIT.md` (rev 1, circular), `audit/PHASE2_DIAGNOSTICS.md`, `audit/FINAL_BRANCH_INVENTORY.md` (untracked; keep or drop) |
| Binary/generated audit data | `audit/announcement_hours.parquet`, `events_bmo_bias.parquet`, `events_timing_probe.parquet`, `provider_timestamps.parquet`, `score_staleness.parquet`, `staleness_final_tier.parquet`, `verified_events.parquet`, `verified_scored_events.parquet` |
| Phase 1 parity fixtures | `audit/phase1_golden/` (all 20 tracked files: `BASE_SHA.txt`, `README.md`, `calibration_pre.txt`, 2 parquets, 13 CSVs, 2 PNGs) |

\* `RESULTS.md` comes back with the Model C integration as a doc (§6), not now.

Local files that are ignored or untracked today and are not part of git at all. Decide
what to do with them in the next task:

- `audit/phase1_golden/daily_df_pre.parquet` (309 MB): regenerable, archive-only. Delete.
- `data/vendor/` (16 GB), `data/benzinga_earnings/`, `data/benzinga_features/`: licensed
  data. Keep or move, but **never let it become visible to git** (the `.gitignore` patch
  goes first).
- `output/phase3_*`, `output/phase4_baselines`, `output/phase5*`, `output/options_pilot`,
  `output/fundamentals`, `output/sec_*`: research outputs, already ignored.

---

## 8. Risks for the first integration

1. **Licensed data exposed by the reset (highest).** Master's `.gitignore` does not cover
   `data/vendor/` (16 GB), `data/benzinga_*` or the 309 MB parity parquet. Restore the
   patched `.gitignore` immediately after the reset. Before any `git add`, confirm with
   `git status --short | grep -E 'vendor|benzinga|daily_df_pre'` returning nothing.
   Stage files by name, never `git add -A`.
2. **Losing prediction rows.** See §5.2. Back up before the reset, and merge by key.
3. **Uncommitted work in the tree.** `MEMORY.md` (modified) and the two untracked plan
   documents. `git reset --hard` keeps untracked files but destroys the `MEMORY.md` edits.
   Copy them out first.
4. **Dashboard calendar export.** Without the `calendar_builder` patch, the droplet
   dashboard's export button always returns nothing after master is pulled.
5. **Monday run on master without the workflow fix.** If the generator is restored but
   `full_workflow.sh` is not, `set -e` aborts the run before the digest. They are in the
   same manifest. Do not split them.
6. **Partial restores.** G3 must land whole (§2f).
7. **Dead code.** `pipeline/incremental.py` is already broken and becomes more wrong. It
   is not called. Leave it and note it.
8. **Droplet deploy.** After master moves, `git pull` on the droplet brings the new code.
   - Runtime there is stage1, IV, EPS (unchanged) and the dashboard (patched calendar
     import path; `pipeline.events` imports only pandas/numpy/duckdb/config).
   - Check the droplet's `db/` status before pulling (§5.4 step 7).
9. **Not risky:** completed-event scores. `assert_completed_parity` raises on any drift,
   on every run.

---

## 9. Proposed execution sequence (next task, not done here)

1. **Check state.** Confirm `git status` shows only `.claude/memory/MEMORY.md` (M) and
   the untracked `audit/FINAL_BRANCH_INVENTORY.md` and `audit/NON_MODEL_C_INTEGRATION_PLAN.md`.
   Confirm no pipeline run is due during the work. Run the read-only droplet check (§5.4
   step 2).
2. **Back up outside the repo**, e.g. into `~/breakwater_backups/<timestamp>/`:
   - `db/predictions.duckdb`;
   - `.claude/memory/MEMORY.md`;
   - both plan documents.

   Record `git hash-object db/predictions.duckdb`.
3. **Archive is on the remote**: checked 2026-10-03, `git ls-remote origin` shows the tag
   → `39fe089`. Re-check just before the reset.
4. **Rebuild the branch:** `git checkout methodology-rebuild && git reset --hard master`.
5. **Restore the patched `.gitignore` first.** Then check that `git status` shows no
   vendor, benzinga or parity files.
6. **Restore the wholesale files** from the tag (§2a): `git checkout methodology-audit-archive-october-2026 -- <the 27 paths>`,
   plus the rename `git mv testing/testing.py testing/metric_testing.py`, then restore
   its content from the tag.
7. **Apply the SELECTIVE PATCHES** (§2b) and the new guard test (§2c). Put back
   `MEMORY.md` from the backup and add the integration entry.
8. **Apply the 15 deletions** (§2d) with `git rm`.
9. **Run the predictions migration** (§5.4 steps 3–5) into a fresh file, verify it, then
   move it into `db/predictions.duckdb`.
10. **Targeted tests:**
    `pytest testing/test_announcement_timing.py testing/test_event_frame.py testing/test_pipeline.py testing/test_predictions_range.py testing/test_production_guards.py testing/test_announcement_ingest.py testing/test_earnings_dates.py testing/test_iv_collection.py`
    plus the new calendar fallback test.
11. **Full suite:** `pytest testing`. Expect no skips from golden tests (removed) and no
    research tests (gone).
12. **Re-score without ingesting** (the stage2→stage5 one-liner in CLAUDE.md). Check:
    - `events_df.parquet` is written and the parity assert passes;
    - the calendar has events;
    - the track record prints its pause message;
    - one new prediction row per upcoming call in the merged DB, if run in a window.
13. **Inspect the diff:** `git diff --stat master` and `git diff master --name-status`
    against §2. Also check:
    - no `research/` paths;
    - nothing under `audit/` except the 2 docs (+ plan docs if kept);
    - `git grep -nE "research\.|research/|phase1_golden|PHASE2_DIAGNOSTICS|phase2_diagnostics" -- ':!audit/*.md'` is empty
      apart from intended mentions.
14. **Commit** (you commit; message drafted at that point).
15. **Later, separately:**
    - merge to master when you say;
    - droplet `git pull` after the §5.4 step 7 check;
    - first Monday run on the new master;
    - only then start the Model C refit on the production event frame (§6).
