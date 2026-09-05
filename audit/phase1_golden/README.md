# Phase 1 golden baseline

Frozen **before** any code change, to prove the event-frame refactor moves nothing.

- `BASE_SHA.txt` — commit the baseline was taken at.
- `calibration_pre.txt` + `testing_results_pre/` — `python -m testing.calibration` output
  and CSVs at that commit.
- `upcoming_df.parquet`, `streamlit_df.parquet` — what the exporters shipped under the
  old `groupby("stock").last()` mechanism, kept for the before/after tier diff.
- `daily_df_pre.parquet` — full pre-change daily frame. **Gitignored (309 MB).**
  Regenerate at the base SHA with:

  ```bash
  PYTHONPATH=. .venv/bin/python -c "
  from pipeline.stage2 import stage2; from pipeline.stage3 import stage3
  from pipeline.stage4 import stage4
  stage4(stage3(stage2())).to_parquet('audit/phase1_golden/daily_df_pre.parquet', index=False)"
  ```

## Parity procedure

1. **Daily frame** — re-run stage 2→4 after the change and compare every column
   against `daily_df_pre.parquet`. Result: byte-identical (87/87 columns, 2,914,315 rows).
2. **Completed events** — `pipeline.events.completed_parity_report` compares all 22
   history-dependent columns on all 45,701 completed events against the daily frame.
   Result: `{}`. This runs on every pipeline execution via `assert_completed_parity`,
   and in CI via `testing/test_event_frame.py`.
3. **Calibration** — `python -m testing.calibration` diffed against `calibration_pre.txt`.
   Result: identical.

Neither 1 nor 2 depends on this directory at run time; the tests rebuild from
`output/full_df.parquet`. The files here are evidence, not fixtures.
