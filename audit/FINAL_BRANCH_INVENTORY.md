# Final branch inventory: `master...methodology-rebuild`

Written 2026-10-03. Read-only inventory: nothing was deleted, moved or changed to produce it.

- Diff: `git diff master...methodology-rebuild`. The merge base is master's tip (`8fcf3b7`), so
  the three-dot and two-dot diffs are the same: **186 changed paths**.
- Archive: tag `methodology-audit-archive-october-2026` is an annotated tag. It points to
  `39fe089`, the current `methodology-rebuild` tip. Everything marked C or D below can
  still be recovered from it.
- Method: classified by tracing imports and callers (`grep` over every `from`/`import` in
  `pipeline/ ingestion/ feature_engineering/ scoring/ utilities/ analysis/ report/
  streamlit_dash/ marketing/ backfills/ audit/ testing/`), reading every production diff,
  and checking which research code each test file imports. Filenames alone decided nothing.

## Summary

| category | files | what it is |
|---|---|---|
| **A** keep / production required | **30** | event frame, announcement timing, the shared event-level feature module, consumer migrations off `.last()`, predictions schema + voiding, P4 retractions in report/readme/marketing, production tests, process docs |
| **B** keep / documentation | **3** | `audit/PHASE0_AUDIT_REV2.md`, `audit/BENZINGA_EARNINGS_AUDIT.md`, `research/phase3_refit/RESULTS.md` |
| **C** archive only, remove from final tree | **135** | 79 `research/`, 39 `audit/`, 15 research/one-time-builder tests, 2 one-time `backfills/` builders |
| **D** obsolete, stays deleted | **15** | folded memory side-files, `info/`, a `.pyc`, unused testing scripts, the one-time IV cleanup script |
| **E** needs decision | **3** | `analysis/predictions_range.py` + its test + `week_block_window()` (its only caller) |

**The main finding:** the runtime pipeline imports **nothing** from `research/` or `audit/`.
The only production-tree code that imports research is the two one-time
`backfills/build_*.py` builders. Both have already been run, and master already has the
loaders that applied their output. The research and audit trees can be removed wholesale.
The three B files are the only exceptions.

## Things that were surprising

1. **Production's weekly calls already come from the rebuild branch.** The committed
   `db/predictions.duckdb` on `methodology-rebuild` holds 28 calls dated 2026-09-06 to
   2026-09-28. Every one carries a `git_commit` that exists only on `methodology-rebuild`
   (`f586f9f`, `6afd8d3`, `3142c8f`, `177be6e`, `2297aaf`), and every one has
   `score_asof_date`, a column that exists only on this branch. `full_workflow.sh` runs the
   pipeline locally from this one folder, so the Monday digest has been built by the
   event-frame code since early September. The droplet (ingestion, IV, dashboard) runs
   master. "Production is on master" is true for the droplet only.
2. **Master still ships retracted figures and the stale-call bug.** On master,
   `report/templates/earnings_report.html` prints "4.78× OOS lift", `readme.md` claims
   "40% … 5.8x lift, consistent 2015–2025 OOS", the public track-record generator is not
   paused, and every upcoming call comes from `groupby("stock").last()` (one event stale,
   audit §Q4). The fixes are all on this branch. None of them depends on Model C.
3. **Model C's research code does not use production's corrected reactions.**
   `research/phase3_refit` trains on `research/phase3_target_rebuild.build_phase3_events`.
   That function matches the **raw Benzinga vendor snapshot** with a ±1-day tolerance and
   re-anchors on the vendor timestamp's own date, rolling weekend/holiday timestamps to a
   session (`_phase3_proxy_session_dates`). Production's `events_df` anchors on
   `earnings_date` plus the clock of the DB's `announce_ts_ny`. It loaded exact-date
   matches only, skipping 214 off-by-one dates, and it refuses to roll non-session dates
   (CLAUDE.md rule). The two anchored histories are close but not identical. **Every
   published Model C number (AUC 0.723, coefficients −1.854 / 0.740 / 0.275, cuts
   0.238 / 0.328) was measured on the research frame, not on the frame production would
   train on.**
4. **Two golden tests in `testing/test_event_frame.py` can never run again.** They skip
   whenever the data is newer than `audit/phase1_golden/upcoming_df.parquet`, which is
   now always. They are the "2 skip" in every suite run.
5. Minor: `CLAUDE.md` (Commands, Backtesting) and `config.py:38` still say
   `testing/testing.py`, renamed to `testing/metric_testing.py`. `readme.md` still lists
   the deleted `window_sensitivity.py`.

## Detailed table

Grouped files are generated artifacts or self-contained research packages. Every
production-tree source file has its own row. Statuses: added unless marked *(deleted)* or
*(renamed)*. Files marked modified on master are noted in the reason column where it matters.

#### Category A

| file | category | reason | production dependency / caller | action later |
|---|---|---|---|---|
| `.claude/memory/MEMORY.md` | **A** | Session memory, mandated by CLAUDE.md and synced via git | process | Keep; trim research log at merge |
| `.gitignore` | **A** | Ignores data/vendor/ (licensed Benzinga data), Benzinga tooling dirs, the 309 MB parity frame | repo hygiene; test_massive_earnings asserts vendor/ is ignored | Keep (drop the phase1_golden line at cleanup) |
| `CLAUDE.md` | **A** | Project instructions; holds the performance-claims rules and timing rules production depends on | process | Keep, but rewrite at merge: drop research-package sections, Phase-2 'parallel target' framing once Model C ships |
| `analysis/save_predictions.py` | **A** | Archives the pending-row call (not the stale .last() call) and records score_asof_date | pipeline/stage5.py | Keep |
| `db/predictions.duckdb` | **A** | The archive of published calls. Rebuild copy = superset: 10 pre-audit calls (now stamped 0.3.1-preaudit, voided) + 28 calls 2026-09-06..09-28, every one written by methodology-rebuild commits | save_predictions.py writes; predictions_range.py, last_week_results read | Keep the rebuild copy (master's has only the 10 voided calls). See open decision on tracking it in git |
| `feature_engineering/announcement_timing.py` | **A** | BMO/AMC classifier, market-session grid, anchor resolution, reaction_{1,3,5}d_anchored + status columns, resolved_events() | pipeline/events.py (runtime); audit/, research/, many tests | Keep. This is the production corrected-reaction implementation Model C must train on |
| `feature_engineering/event_features.py` | **A** | Single event-indexed implementation of every across-events statistic, shared by the daily path and the pending row | pre_/post_earnings_stock_features.py, scoring/scoring_features.py, pipeline/events.py | Keep. Add hist_mean_abs on the anchored target here; the 0.3.1-only cores (p75/entropy/lift/HC) go when 0.3.1 is retired |
| `feature_engineering/post_earnings_stock_features.py` | **A** | Delegates reaction_std/entropy/directional_bias to event_features (no number changes; parity-tested) | pipeline/stage3.py | Keep |
| `feature_engineering/pre_earnings_stock_features.py` | **A** | Delegates median/p75/rolling p75/p90/surprise/drift-z to event_features | pipeline/stage3.py | Keep |
| `marketing/content_calendar.md` | **A** | Marks track-record posts paused | marketing | Keep |
| `marketing/generate_public_track_record.py` | **A** | P4.3 pause (refuses to publish) + P4.2 drop_voided() | pipeline/stage5.py | Keep; un-pause only per its header checklist |
| `marketing/positioning.md` | **A** | Replaces retracted proof points with verified audit figures + scope limits | marketing source of truth | Keep; re-derive when Model C ships |
| `marketing/post_templates.md` | **A** | Retracts figures; holds results template | marketing | Keep |
| `marketing/reddit_playbook.md` | **A** | Retracts figures in suggested comments | marketing | Keep |
| `pipeline/events.py` | **A** | Stage 4b event frame: one row per event incl. one pending row per stock; fixes the one-event-stale call (audit §Q4); attaches DB timing and the anchored target; asserts completed-event parity on every run | pipeline/pipeline.py, pipeline/stage5.py; pending_events() used by report_builder, calendar_builder, streamlit_export, save_predictions | Keep. Model C scores on this frame |
| `pipeline/pipeline.py` | **A** | Wires stage 4b between stage4 and stage5 | main.py | Keep |
| `pipeline/stage5.py` | **A** | Writes output/events_df.parquet; hands events_df to every forward-looking consumer | pipeline/pipeline.py | Keep |
| `readme.md` | **A** | Retracts '40% / 5.8x / 2015-2025 OOS' headline (still on master) | public repo front page | Keep; still lists deleted window_sensitivity.py |
| `report/calendar_builder.py` | **A** | Weekly calendar reads pending rows; fixes pre-existing bug where the calendar rendered zero events | pipeline/stage5.py, Streamlit export button | Keep |
| `report/report_builder.py` | **A** | PDF reports read the pending row; fixes KeyError when a stock enters a tier it never held | pipeline/stage5.py | Keep |
| `report/templates/earnings_report.html` | **A** | Replaces retracted 4.78x OOS lift with verified figures; suppresses per-stock lift rows (P4.1) | report/report_builder.py | Keep. Rewrite figures when Model C ships |
| `scoring/scoring_features.py` | **A** | Delegates scoring cores to event_features so the pending row is scored by the same code | pipeline/stage4.py, pipeline/events.py | Keep until Model C replaces the 0.3.1 score |
| `scripts/full_workflow.sh` | **A** | Skips the recent_calls.json push while P4.3 pause is on, instead of aborting the Monday run under set -e | run by hand (Monday workflow) | Keep |
| `scripts/sync_pipeline.sh` | **A** | Deliberately reduced to pull-only (85a929d, 2026-09-29): it no longer runs the pipeline or pushes parquets; full_workflow.sh does that | run by hand | Keep |
| `streamlit_dash/streamlit_export.py` | **A** | upcoming_df from pending rows; adds score_asof_date | pipeline/stage5.py | Keep |
| `testing/metric_testing.py` *(renamed)* | **A** | Rename of master's testing/testing.py (legacy 0.3.1 one-factor OOS script) with a 'Not a testing suite' comment; content 99% unchanged | CLI only | Keep the rename; retire with 0.3.1. Fix CLAUDE.md/config.py references to testing/testing.py |
| `testing/test_announcement_timing.py` | **A** | 71 tests: classifier cut points, session-grid anchoring, refusal on gaps, never-infer-from-price, now_ny/host-clock rules, schema write path, no pipeline module reads the audit parquet | pipeline.events, announcement_timing, ingestion, db_utilities | Keep |
| `testing/test_event_frame.py` | **A** | 30 tests: completed-event parity, pending-row invariants, no .last() in production, drift-flag window | pipeline.events, event_features, scoring, streamlit_export | Keep; delete the 2 golden tests (permanently skipped; fixture is archive-only) |
| `testing/test_pipeline.py` | **A** | Docstring-only change on master's existing pipeline test | pipeline stages | Keep |
| `utilities/db_utilities.py` | **A** | load_announcement_timing() (event frame's only timing source); predictions schema: score_asof_date, void_for_track_record/void_reason, predictions_track_record view (P4.2) | pipeline/events.py; analysis/save_predictions.py; marketing generator | Keep |

#### Category B

| file | category | reason | production dependency / caller | action later |
|---|---|---|---|---|
| `audit/BENZINGA_EARNINGS_AUDIT.md` | **B** | Provenance of the 25.5k Benzinga timestamps now in the production earnings table (announce_ts_source = massive_benzinga:...); explains the NY-clock and midnight-filler rules production depends on | cited by CLAUDE.md; generated by research/massive/report.py | Keep as a frozen document (its generator is archived) |
| `audit/PHASE0_AUDIT_REV2.md` | **B** | The authority for every performance number (CLAUDE.md); cited by readme, positioning.md, report template, track-record generator, db_utilities void_reason | cited by A files (text only) | Keep |
| `research/phase3_refit/RESULTS.md` | **B** | The specification and evidence for Model C (inputs, call cutoff, walk-forward, common cuts, coefficients, removal of ceiling/entropy/lift/HC) | none | Keep; consider moving to docs/ when Model C is ported |

#### Category C

| file | category | reason | production dependency / caller | action later |
|---|---|---|---|---|
| `audit/PHASE0_AUDIT.md` | **C** | Audit rev 1, superseded; its 'corrected' numbers were circular (inferred BMO from price) | none | Archive |
| `audit/PHASE2_DIAGNOSTICS.md` | **C** | Generated output of audit/phase2_diagnostics.py; CLAUDE.md cites §5 as the AMC control | none | Archive; drop the CLAUDE.md citation or restate the fact inline |
| `audit/*.parquet` (9 files):<br>`audit/announcement_hours.parquet`<br>`audit/events_bmo_bias.parquet`<br>`audit/events_timing_probe.parquet`<br>`audit/phase1_golden/streamlit_df.parquet`<br>`audit/phase1_golden/upcoming_df.parquet`<br>`audit/score_staleness.parquet`<br>`audit/staleness_final_tier.parquet`<br>`audit/verified_events.parquet`<br>`audit/verified_scored_events.parquet` | **C** | Generated audit artifacts (announcement_hours, events_bmo_bias, events_timing_probe, score_staleness, staleness_final_tier, verified_events, verified_scored_events) | none | Archive |
| `audit/fetch_provider_timestamps.py` | **C** | One-time yfinance pull that produced provider_timestamps.parquet | none | Archive |
| `audit/phase1_golden/BASE_SHA.txt` | **C** | Phase 1 parity evidence: pre-refactor commit | none | Archive |
| `audit/phase1_golden/README.md` | **C** | Phase 1 parity procedure; says itself the files are 'evidence, not fixtures' | none | Archive |
| `audit/phase1_golden/calibration_pre.txt` | **C** | Pre-refactor calibration output (parity evidence) | none | Archive |
| `audit/phase1_golden/testing_results_pre/*` (14 files):<br>`audit/phase1_golden/testing_results_pre/april13-aug15/calibration_by_bucket.csv`<br>`audit/phase1_golden/testing_results_pre/april13-aug15/calibration_by_percentile.csv`<br>`audit/phase1_golden/testing_results_pre/april13-aug15/calibration_capture_rate.csv`<br>`audit/phase1_golden/testing_results_pre/april13-aug15/calibration_year_by_year.csv`<br>`audit/phase1_golden/testing_results_pre/april13-aug15/confusion_matrix.png`<br>`audit/phase1_golden/testing_results_pre/april13-aug15/weekly_prediction_quality.csv`<br>`audit/phase1_golden/testing_results_pre/calibration_by_bucket.csv`<br>`audit/phase1_golden/testing_results_pre/calibration_by_percentile.csv`<br>`audit/phase1_golden/testing_results_pre/calibration_capture_rate.csv`<br>`audit/phase1_golden/testing_results_pre/calibration_consistency.csv`<br>`audit/phase1_golden/testing_results_pre/calibration_tier_stability.csv`<br>`audit/phase1_golden/testing_results_pre/calibration_year_by_year.csv`<br>`audit/phase1_golden/testing_results_pre/confusion_matrix.png`<br>`audit/phase1_golden/testing_results_pre/weekly_prediction_quality.csv` | **C** | Pre-refactor calibration CSVs + confusion-matrix PNGs (parity evidence) | none | Archive |
| `audit/phase2_diagnostics.py` | **C** | Dataset diagnostics on the Phase 2 target (window mix, coverage, legacy vs anchored) | none; imports announcement_timing | Archive. announcement_timing.resolution_summary() already covers ongoing coverage reporting |
| `audit/probe_announcement_hours.py` | **C** | Exploratory probe | none | Archive |
| `audit/probe_announcement_timing.py` | **C** | Exploratory probe | none | Archive |
| `audit/provider_timestamps.parquet` | **C** | yfinance timestamp evidence (2026-09-05); already loaded into the DB. Read only by build_announcement_seed.py and research | none at runtime (test asserts no pipeline module reads it) | Archive |
| `audit/quantify_bmo_bias.py` | **C** | Audit analysis script | none | Archive |
| `audit/quantify_score_staleness.py` | **C** | Audit analysis script (§Q4) | none | Archive |
| `audit/staleness_final_tier.py` | **C** | Audit analysis script | none | Archive |
| `audit/stratified_lift.py` | **C** | Audit analysis script (produced the 1.91x stratified lift) | none | Archive |
| `audit/tier_by_timing.py` | **C** | Audit analysis script | none | Archive |
| `audit/verified_timing_analysis.py` | **C** | Audit analysis script (§Q2 verified figures) | none | Archive |
| `backfills/build_announcement_seed.py` | **C** | One-time builder of the Benzinga/yfinance seed (done 2026-09-28; loaded on the droplet by master's load_announcement_seed.py). Imports research.massive.paths and research.phase3_target_rebuild | none at runtime | Archive. Master already has the loader |
| `backfills/build_date_corrections.py` | **C** | One-time builder of the earnings-date corrections (applied 2026-09-29 by master's apply_date_corrections.py). Imports research.* | none at runtime | Archive |
| `research/__init__.py` | **C** | Package marker for research/ | none | Archive |
| `research/benzinga_feature_value.py` | **C** | Research: Benzinga feature value | none | Archive |
| `research/fundamentals/*` (7 files):<br>`research/fundamentals/PREREGISTRATION.md`<br>`research/fundamentals/RESULTS.md`<br>`research/fundamentals/__init__.py`<br>`research/fundamentals/acquire.py`<br>`research/fundamentals/audit.py`<br>`research/fundamentals/build.py`<br>`research/fundamentals/evaluate.py` | **C** | XBRL fundamentals vs Model C: NO USEFUL VALUE | none | Archive |
| `research/massive/*` (10 files):<br>`research/massive/__init__.py`<br>`research/massive/acquire.py`<br>`research/massive/client.py`<br>`research/massive/completeness.py`<br>`research/massive/crosscheck.py`<br>`research/massive/identity.py`<br>`research/massive/normalize.py`<br>`research/massive/paths.py`<br>`research/massive/report.py`<br>`research/massive/validate.py` | **C** | Benzinga acquisition/normalisation/validation/report tooling (the seed is already in the DB; no pipeline stage may call it) | backfills/build_* (one-time) and research only | Archive |
| `research/options_confirmatory/*` (5 files):<br>`research/options_confirmatory/DATA_SOURCES.md`<br>`research/options_confirmatory/PAUSED.md`<br>`research/options_confirmatory/PREREGISTRATION.md`<br>`research/options_confirmatory/__init__.py`<br>`research/options_confirmatory/core.py` | **C** | Pre-registered options test, designed but not run; paused pending paid data | none | Archive; restore from the tag if data is bought |
| `research/options_pilot/*` (11 files):<br>`research/options_pilot/PAUSED.md`<br>`research/options_pilot/PREREGISTRATION.md`<br>`research/options_pilot/RESULTS.md`<br>`research/options_pilot/__init__.py`<br>`research/options_pilot/build.py`<br>`research/options_pilot/calendar_check.py`<br>`research/options_pilot/coverage.py`<br>`research/options_pilot/dates.py`<br>`research/options_pilot/evaluate.py`<br>`research/options_pilot/features.py`<br>`research/options_pilot/source.py` | **C** | Options pilot on DoltHub data: WEAK / UNCERTAIN; paused | none | Archive |
| `research/phase3_cap_tiebreak.py` | **C** | Research: 12% ceiling tie-break | none | Archive |
| `research/phase3_refit/PREREGISTRATION.md` | **C** | Model C pre-registration | none | Archive (RESULTS.md is kept) |
| `research/phase3_refit/__init__.py` | **C** | Package marker | none | Archive |
| `research/phase3_refit/calibration.py` | **C** | Walk-forward logistic fit, standardisation, quantile tier cuts (Model C research implementation) | none | Archive after reimplementing |
| `research/phase3_refit/candidates.py` | **C** | Candidate definitions A0-D incl. Model C's inputs | none | Archive after reimplementing (see Model C map) |
| `research/phase3_refit/evaluate.py` | **C** | Evaluation driver, bootstrap, tables | none | Archive |
| `research/phase3_refit/features.py` | **C** | Call-cutoff clocks, endpoint-availability rule, hist_mean_abs, vol_30d as-of (Model C research implementation) | none | Archive after reimplementing |
| `research/phase3_target_rebuild.py` | **C** | Phase 3 corrected-target rebuild from the raw vendor snapshot (±1-day match, re-anchoring on the vendor date) | backfills/build_* (one-time); phase3_refit, phase4/5, sec_features, options_pilot | Archive. Production equivalent is announcement_timing + DB timestamps |
| `research/phase4_baselines.py` | **C** | Research: simple baselines vs model | none | Archive |
| `research/phase5_event_signal.py` | **C** | Research: event-signal candidates | none | Archive |
| `research/phase_5b_new_feature_testing/*` (13 files):<br>`research/phase_5b_new_feature_testing/PREREGISTRATION.md`<br>`research/phase_5b_new_feature_testing/RESULTS.md`<br>`research/phase_5b_new_feature_testing/__init__.py`<br>`research/phase_5b_new_feature_testing/build.py`<br>`research/phase_5b_new_feature_testing/evaluate.py`<br>`research/phase_5b_new_feature_testing/pair.py`<br>`research/phase_5b_new_feature_testing/panel.py`<br>`research/phase_5b_new_feature_testing/peer_features.py`<br>`research/phase_5b_new_feature_testing/posthoc.py`<br>`research/phase_5b_new_feature_testing/price_features.py`<br>`research/phase_5b_new_feature_testing/registry.py`<br>`research/phase_5b_new_feature_testing/tests/__init__.py`<br>`research/phase_5b_new_feature_testing/tests/test_phase5b.py` | **C** | Phase 5B free price/peer features (incl. its own tests/ subpackage); D candidate rejected | research/phase3_refit/features.py (peer features for candidate D only) | Archive |
| `research/score_validity_and_deviation_features.py` | **C** | Research | none | Archive |
| `research/sec_features/*` (10 files):<br>`research/sec_features/PREREGISTRATION.md`<br>`research/sec_features/RESULTS.md`<br>`research/sec_features/__init__.py`<br>`research/sec_features/audit.py`<br>`research/sec_features/build.py`<br>`research/sec_features/evaluate.py`<br>`research/sec_features/features.py`<br>`research/sec_features/frame.py`<br>`research/sec_features/manual_audit.csv`<br>`research/sec_features/refresh.py` | **C** | SEC text/8-K features vs Model C: NO USEFUL VALUE. frame.py builds the current corrected research frame | none | Archive |
| `research/sec_filings_pilot/*` (10 files):<br>`research/sec_filings_pilot/RESULTS.md`<br>`research/sec_filings_pilot/__init__.py`<br>`research/sec_filings_pilot/acquire.py`<br>`research/sec_filings_pilot/audit.py`<br>`research/sec_filings_pilot/build.py`<br>`research/sec_filings_pilot/documents.py`<br>`research/sec_filings_pilot/mapping.py`<br>`research/sec_filings_pilot/paths.py`<br>`research/sec_filings_pilot/sample.py`<br>`research/sec_filings_pilot/timing.py` | **C** | SEC filings feasibility pilot | sec_features | Archive |
| `testing/test_benzinga_feature_value.py` | **C** | Research tests | research | Archive |
| `testing/test_build_announcement_seed.py` | **C** | Tests a one-time builder; imports research.massive | backfills/build_announcement_seed.py | Archive |
| `testing/test_build_date_corrections.py` | **C** | Tests a one-time builder; imports research.massive | backfills/build_date_corrections.py | Archive |
| `testing/test_fundamentals.py` | **C** | Research tests | research | Archive |
| `testing/test_massive_earnings.py` | **C** | 34 tests of research/massive; includes 2 guards worth porting (no production module reads data/vendor; vendor dir gitignored) | research.massive | Archive; port the 2 guards into test_announcement_timing first |
| `testing/test_options_confirmatory.py` | **C** | Research tests | research | Archive |
| `testing/test_options_pilot.py` | **C** | Research tests | research | Archive |
| `testing/test_phase3_cap_tiebreak.py` | **C** | Research tests | research | Archive |
| `testing/test_phase3_refit.py` | **C** | 17 tests of the Model C research code; the leakage/cutoff tests are the template for production Model C tests | research.phase3_refit | Archive after porting the leakage tests |
| `testing/test_phase3_target_rebuild.py` | **C** | Research tests | research.phase3_target_rebuild | Archive |
| `testing/test_phase4_baselines.py` | **C** | Research tests | research | Archive |
| `testing/test_phase5_event_signal.py` | **C** | Research tests | research | Archive |
| `testing/test_score_validity_and_deviation_features.py` | **C** | Research tests | research | Archive |
| `testing/test_sec_features.py` | **C** | Research tests | research | Archive |
| `testing/test_sec_filings.py` | **C** | Research tests | research | Archive |

#### Category D

| file | category | reason | production dependency / caller | action later |
|---|---|---|---|---|
| `.claude/memory/*.md` (7 files):<br>`.claude/memory/codebase_audit_2026_06_07.md`<br>`.claude/memory/infra_digitalocean.md`<br>`.claude/memory/next_to_build.md`<br>`.claude/memory/project_direction.md`<br>`.claude/memory/reddit_marketing_playbook.md`<br>`.claude/memory/social_media_strategy.md`<br>`.claude/memory/window-sensitivity.md` | **D** | Side memory files folded into MEMORY.md (fa2ab55) | none | Stay deleted |
| `info/*` (3 files):<br>`info/droplet_project_update_instructions.txt`<br>`info/metrics_to_build.txt`<br>`info/recall_precision_for_different_quantiles_results.ods` | **D** | Unused notes/spreadsheet removed in 2297aaf | none | Stay deleted |
| `pipeline/__pycache__/pipeline.cpython-39.pyc` *(deleted)* | **D** | Committed bytecode | none | Stay deleted |
| `scripts/delete_iv_rows_on_closed_days.py` *(deleted)* | **D** | One-time IV cleanup, done 2026-09-29; deleted deliberately by the user | none | Stay deleted (still on master: delete there too) |
| `testing/inspect_db.py` *(deleted)* | **D** | Unused, removed in 2297aaf | none | Stay deleted |
| `testing/weekly_prediction_quality.py` *(deleted)* | **D** | Unused, removed in 2297aaf | none | Stay deleted |
| `testing/window_sensitivity.py` *(deleted)* | **D** | Unused, removed in 2297aaf | none | Stay deleted |

#### Category E

| file | category | reason | production dependency / caller | action later |
|---|---|---|---|---|
| `analysis/predictions_range.py` | **E** | Multi-week predictions CLI on 0.3.1; half its output (*_pre_audit, pre_audit_differs) exists to measure the staleness bug | CLI only; nothing in the pipeline calls it | Decide whether a multi-week CLI is a product tool; if yes, strip *_pre_audit at Model C port |
| `testing/test_predictions_range.py` | **E** | Tests the CLI above (20 tests) | analysis/predictions_range.py | Goes with predictions_range |
| `utilities/data_utilities.py` | **E** | Only change is week_block_window(); its only callers are predictions_range and its test | analysis/predictions_range.py | Goes with predictions_range |

## Production dependency map

What the runtime pipeline (`main.py` → `pipeline/pipeline.py`) actually reaches. Every
node is on `methodology-rebuild`. Nodes marked *new* or *changed* differ from master.

```
main.py
└─ pipeline/pipeline.py (changed)
   ├─ stage1 → ingestion/fetch_earnings_dates.py ─ utilities/time_utilities.py  (both already on master)
   ├─ stage2 / stage3 / stage4
   │    ├─ feature_engineering/pre_earnings_stock_features.py  (changed) ─┐
   │    ├─ feature_engineering/post_earnings_stock_features.py (changed) ─┼─ feature_engineering/event_features.py (new)
   │    └─ scoring/scoring_features.py                          (changed) ─┘
   ├─ stage 4b  pipeline/events.py (new)
   │    ├─ feature_engineering/event_features.py
   │    ├─ feature_engineering/announcement_timing.py (new)
   │    ├─ scoring/scoring_features.classify_large_relative_earnings_move_bucket
   │    └─ utilities/db_utilities.load_announcement_timing (changed)  ← earnings.announce_ts_ny in DuckDB
   └─ stage5 (changed) — writes output/full_df.parquet and output/events_df.parquet
        ├─ report/report_builder.py   (changed) ─┐
        ├─ report/calendar_builder.py (changed) ─┤
        ├─ streamlit_dash/streamlit_export.py (changed) ─┼─ pipeline.events.pending_events
        ├─ analysis/save_predictions.py (changed) ─┘ → db/predictions.duckdb (schema in db_utilities)
        └─ marketing/generate_public_track_record.py (changed: P4.3 pause, P4.2 drop_voided)

Not reached at runtime:
  research/**            — imported only by research/, research tests, backfills/build_*.py
  audit/*.py             — standalone scripts; import announcement_timing / config only
  audit/*.parquet        — read by research + build_announcement_seed.py only
                           (test_no_pipeline_module_reads_the_audit_parquet guards this)
  backfills/build_*.py   — one-time builders, already run; import research.massive.paths and
                           research.phase3_target_rebuild
  backfills/load_announcement_seed.py, apply_date_corrections.py — already on master, unchanged
  analysis/predictions_range.py — CLI only
```

**No production code depends on anything classified C.** The only C-classified code that
anything in the production tree imports is `research.massive.paths` and
`research.phase3_target_rebuild`, and only the two `backfills/build_*.py` files import
them. Those files are C themselves.

## Model C promotion map

Model C (`research/phase3_refit/RESULTS.md` §7):
`p = sigmoid(b0 + b1·z(log hist_mean_abs) + b2·z(log vol_30d))`, refit each year on
everything observable before 1 January; High Alert = top 10% and Elevated = next 10% of
training-fold probabilities (common cuts, no BMO/AMC split); ≥ 8 prior corrected outcomes.
The research code that demonstrated it is **not** the code that should ship. It runs
on a vendor-derived frame, computes both call and eve cutoffs, and carries candidates
A0–D, bootstraps and Phase 5B peer features.

| concept | research source (demonstrated it) | production implementation today | recommended final home | after the port |
|---|---|---|---|---|
| Corrected historical reaction (`abs_reaction_3d_anchored`, status columns) | `research/phase3_target_rebuild.py` `match_benzinga_timing`, `reanchor_to_observed_timestamp`, `_phase3_proxy_session_dates` (reads raw vendor snapshot) | **Already exists:** `feature_engineering/announcement_timing.resolve_event_anchors` on DB timestamps, attached by `pipeline/events.py` | Keep `announcement_timing.py` as the only implementation | Discard the research version. Re-measure Model C on production's `events_df` before shipping (see open decision 1) |
| Event / call-time construction | `research/phase3_refit/features.event_clocks`: call cutoff = last session strictly before the Monday of the report week; endpoint = anchor + 3 sessions | `pipeline/events.py` has one pending row per stock scored at the latest data date; `score_asof_date` records it. No call-cutoff or endpoint column on history rows | Add `call_cutoff_date` and `endpoint3_date` to the event frame in `pipeline/events.py` (session-grid arithmetic from `announcement_timing.market_session_grid`) | Discard `research/phase3_refit/features.py` |
| Historical mean absolute reaction (`hist_mean_abs`, ≥ 8 prior, outcome counts only once its endpoint ≤ cutoff) | `research/phase3_refit/features.history_features` (per-stock Python loop over endpoint order) | None. Production's history stats (`abs_reaction_median`, `_p75`, entropy) use the **legacy** target and `shift(1)` | New core `event_hist_mean_abs` in `feature_engineering/event_features.py` on `abs_reaction_3d_anchored`, ordered by `endpoint3_date`, gated by cutoff | Discard research loop. Legacy p75/median/entropy cores retire with 0.3.1 |
| `vol_30d` at the call cutoff | `research/phase3_refit/features.daily_asof` (as-of merge onto the daily frame, 7-day tolerance) | **Already exists:** `pre_earnings_stock_features` `vol_30d` = `rolling(30).std().shift(1)`; already on every event row (`pipeline/events.py` carries daily columns as-of the row: earnings day for history, latest session for pending — see `CARRIED_NOTE`) | Keep. For history rows, carry the value as of `call_cutoff_date`, not the event-day value | Discard `daily_asof` |
| Walk-forward fit (log transform, train-fold standardisation, L2 logistic C=1, train = endpoint < 1 Jan Y) | `research/phase3_refit/calibration.fit_model`, `training_mask`, `raw_probability`; `candidates._log` (floor 1e-4) | None | Offline yearly fit script, e.g. `scoring/model_c_fit.py`, writing a small versioned coefficients file (means, stds, coefs, cuts, train cutoff). Runtime `scoring/model_c.py` only applies the frozen file. Bump `MODEL_VERSION` | Discard research calibration module. W1/W2 window variants and P_ratio cuts were rejected: do not port |
| Tier cuts (common quantiles of training probabilities, 0.80 / 0.90) | `calibration.fit_cuts` (`Q_common`), `assign_tiers` | `config.py` 73/79 floors + lift promotion | Same fit script, cuts stored with the coefficients | Remove `BUCKET_*_FLOOR`, `LIFT_*`, `LIFT_PRIOR_STRENGTH`, `stock_bucket_lift`, High Conviction (RESULTS §6 recommends removing all of them) |
| Score calibration / acceptance gate | `research/phase3_refit/evaluate.py` (`disc_metrics`, `tier_table`, `calibration_rows`, stock-clustered `bootstrap`) | `testing/calibration.py` (legacy target, Wilson CIs, 73/79-specific) | Rewrite `testing/calibration.py` for the anchored target and probabilities: hit rate by tier, capture, calibration by window, stratified lift. Port only `wilson` and the cluster bootstrap if wanted | Discard evaluate.py |

Dependencies of the production port: `announcement_timing.py`, `event_features.py`,
`pipeline/events.py`, the DB timestamps (Benzinga seed + yfinance ingestion), `vol_30d`
in stage 3, scikit-learn (already used by research and `metric_testing.py`).

Tests that should survive into the Model C port:

- Keep as-is: `testing/test_announcement_timing.py`, `testing/test_event_frame.py` (minus
  the 2 golden tests), `testing/test_pipeline.py`, plus master's `test_announcement_ingest.py`,
  `test_earnings_dates.py`, `test_iv_collection.py`.
- Re-write against the production implementation, using `testing/test_phase3_refit.py` as
  the template: `test_own_outcome_cannot_enter_own_score`,
  `test_future_events_cannot_affect_earlier_scores`,
  `test_outcome_counts_only_once_its_endpoint_is_observable`,
  `test_unresolved_target_stays_missing_never_zero`,
  `test_call_cutoff_is_the_friday_before_the_report_week`,
  `test_daily_features_are_read_at_the_call_cutoff_not_later`,
  `test_training_rows_end_strictly_before_the_test_year`,
  `test_fit_calibration_and_cuts_never_see_the_test_year`,
  `test_quantile_cuts_flag_ten_and_ten_percent_on_training`, `test_tiers_are_deterministic`.
- Port from `testing/test_massive_earnings.py` before archiving it:
  `test_no_production_module_depends_on_a_raw_vendor_file`,
  `test_the_vendor_directory_is_gitignored`.
- Drop when 0.3.1 retires: `test_v031_*`, everything that pins 73/79 or lift promotion.

## Deletion candidate summary

All recoverable from tag `methodology-audit-archive-october-2026`.

| remove wholesale | files | keep from it |
|---|---|---|
| `research/` | 80 | `research/phase3_refit/RESULTS.md` (or move it to `docs/model_c.md`) |
| `audit/` | 41 (316 MB on disk incl. the ignored 309 MB parity frame) | `audit/PHASE0_AUDIT_REV2.md`, `audit/BENZINGA_EARNINGS_AUDIT.md` |
| `audit/phase1_golden/` | 20 | nothing |
| `backfills/build_announcement_seed.py`, `backfills/build_date_corrections.py` | 2 | master's loaders stay as they are |
| research tests in `testing/` | 15: `test_massive_earnings`, `test_phase3_refit`, `test_phase3_target_rebuild`, `test_phase3_cap_tiebreak`, `test_phase4_baselines`, `test_phase5_event_signal`, `test_score_validity_and_deviation_features`, `test_benzinga_feature_value`, `test_options_pilot`, `test_options_confirmatory`, `test_sec_filings`, `test_sec_features`, `test_fundamentals`, `test_build_announcement_seed`, `test_build_date_corrections` | port the guards/leakage tests listed above first |
| 2 golden tests inside `testing/test_event_frame.py` | – | rest of the file |
| `.gitignore` line for `audit/phase1_golden/daily_df_pre.parquet` | – | vendor/Benzinga ignores stay |

Tracked generated/binary files:

| file(s) | verdict |
|---|---|
| `audit/*.parquet` (8, incl. `provider_timestamps.parquet`) | archive only. The timestamps are already in the DB, and no runtime reader exists |
| `audit/phase1_golden/*.parquet`, `*.csv`, `*.png`, `calibration_pre.txt` | archive only. Parity evidence for a refactor that is finished; the parity check itself runs on every pipeline run (`assert_completed_parity`) without them |
| `research/sec_features/manual_audit.csv` | archive only |
| `db/predictions.duckdb` | **genuinely needed**: it is the only record of published calls. Not regenerable. See open decision 3 |
| `data/*.csv`, `report/img/breakwater_logo.png` | unchanged by this branch; out of scope |
| Untracked local backups `db/6_09_26_db_backup_breakwater.duckdb`, `db/predictions.duckdb.bak_prevoid_20260920` | not in git; nothing to do for the merge |

No golden outputs are needed as permanent fixtures. Phase 1 parity is checked live, and
the Model C port will need its own check (open decision 1), not these files.

## Open decisions

1. **Which frame does Model C get fitted and judged on in production?** The research
   frame re-anchors on the vendor date (±1 day) and rolls weekend/holiday timestamps to a
   session. Production anchors on `earnings_date` with exact-date DB timestamps and
   refuses to roll (a CLAUDE.md rule). Options: (a) refit Model C on production's
   `events_df` and accept slightly different coefficients and figures; recommended,
   because it keeps the vendor out of runtime and the rules intact. (b) Bring the
   proxy-session rule into `announcement_timing.py` on DB timestamps only, which means
   relaxing "never auto-roll". Either way the published Model C numbers must be
   re-measured on the frame that ships.
2. **Ship the non-Model-C fixes to master now, or wait for Model C?** The event frame
   (stale-call fix), calendar fix, P4.1–P4.3 retractions and the predictions voiding are
   all independent of Model C, and the local Monday run already uses them. Master
   (droplet, public readme) still shows the retracted 4.78× and 5.8× figures. A first
   merge of just the A + B files would be clean; Model C would then be a second, smaller
   change.
3. **Keep `db/predictions.duckdb` in git?** It is production data that keeps changing,
   committed as a binary ("updated predictions db" commits). If it stays tracked, the
   rebuild copy must win at merge, because master's copy lacks the 28 September calls and
   the void flags. The alternative is backing it up off-repo (e.g. alongside the droplet DB).
4. **`analysis/predictions_range.py` (E): keep as a product tool?** Recommended: keep the
   multi-week CLI if you actually use it, but strip the `*_pre_audit` / `pre_audit_differs`
   half, which only measures the old staleness bug, when Model C ships. If not used,
   archive it with its test and `week_block_window()`.
5. **Where does future research live?** Options confirmation is paused, not cancelled. If
   paid data arrives, the test needs `research/options_confirmatory/` plus the Model C
   research frame (`phase3_target_rebuild`, `phase3_refit`, `sec_features/frame.py`), all
   of which this cleanup removes from master. Choices: branch from the archive tag at that
   point (recommended; nothing on master changes), or keep a `research/` area on master.
   This needs deciding before deleting `research/`, but it does not block the merge in
   decision 2.
