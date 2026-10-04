# Model C production-frame refit — specification

Written 2026-10-03, **before any outcome of this refit was computed**. Everything below is
fixed. Anything decided after results are read is marked **POST HOC** in RESULTS.md.

The archived Model C (tag `methodology-audit-archive-october-2026`,
`research/phase3_refit/`) is the conceptual source. Its event frame is NOT used: it matched
vendor records to events with a ±1-day tolerance and re-anchored on a proxy report date,
which production deliberately does not do. Here everything is computed from the
production event frame.

## Data

* **Event frame**: `pipeline.events.build_event_frame(daily, timing, active_stocks)` on
  `output/full_df.parquet` (the production daily frame), with `timing` and `active_stocks`
  loaded from `db/breakwater.duckdb` by the pipeline's own loaders
  (`load_pipeline_announcement_timing`, `load_pipeline_active_stocks`). No production
  timing rule is changed.
* **Market-session grid**: `announcement_timing.market_session_grid(daily)`.
* **Target**: `y = abs_reaction_3d_anchored >= 0.08` (`EXTREME_EARNINGS_REACTION_THRESHOLD`),
  only where `reaction_3d_anchored_status == "available"` and the event is completed.
  Everything else is unlabelled and never enters a fit or a metric. Unresolved is never 0.

## Clocks

* **Report date** = `earnings_date` (production; no proxy date).
* **Call cutoff** = the last market session strictly before the Monday of the report
  date's week. This is the close the weekend run scores from.
* **Outcome endpoint** of a labelled event = the session 3 market sessions after its
  `anchor_date` on the grid (the session whose close ends `reaction_3d_anchored`).
* A prior event's outcome may enter event i's features only if its endpoint session is
  **on or before** i's call cutoff session. The event's own outcome is excluded by
  identity as well as by this rule.

## Features (as of the call cutoff)

* `n_prior` = number of the stock's labelled outcomes usable at the cutoff.
* `hist_mean_abs` = mean of those outcomes' `abs_reaction_3d_anchored`.
* `vol_30d` = the production column `full_df.vol_30d`
  (`daily_ret.rolling(30).std().shift(1)` per stock) on the stock's daily row dated at the
  call cutoff; if the stock has no row that day, its latest row up to 7 calendar days
  earlier (the archived rule). Counts of exact vs fallback rows are reported.
* Model inputs: `log(max(x, 1e-4))`, standardised with the training fold's mean and
  population std (ddof 0).

## Population

Completed events with the target available (BMO/AMC by construction), report year
**2014–2026**, `n_prior >= 8` (**MIN_PRIOR = 8**, the archived pre-registration). No
imputation: an event without 8 usable prior outcomes, or without `vol_30d`, is excluded
and counted.

**Common sample** = population rows with both `hist_mean_abs` and `vol_30d`. B and C are
compared on these identical rows. B on its own (vol not required) is reported as a
sensitivity only.

## Models

* **B**: logistic on `z(log hist_mean_abs)`.
* **C**: logistic on `z(log hist_mean_abs)`, `z(log vol_30d)`.
* scikit-learn `LogisticRegression(C=1.0, max_iter=2000, random_state=0)`, L2, never tuned.
  One common model; no BMO/AMC adjustment (the archived W0, which its selection kept).

## Walk-forward

* Test years **2017–2025** (by report-date year).
* For test year Y: train on population rows whose outcome endpoint is before 1 January Y;
  fit standardisation and coefficients there; score the year-Y rows.
* Pooled out-of-fold (OOF) metrics are over all 2017–2025 test rows.

## Holdout

* **2026 YTD** is scored **once**, after the pre-2026 results are written and this
  specification is frozen (a file `output/model_c_refit/FROZEN.json` carrying this file's
  SHA-256 is written by the pre-2026 run; the holdout step refuses to run without it or if
  this file has changed).
* Holdout model = trained on every population row with endpoint before 2026-01-01. The
  same fit is the **candidate production fit** whose coefficients are reported.

## Metrics

ROC-AUC, PR-AUC (average precision), Brier, log loss, mean p; top-10% and top-20% hit rate,
capture and lift (top k = ceil(frac·n) by p, stable sort) — all as the archived
`disc_metrics`. Reported pooled OOF, per test year, per window (BMO/AMC), and for 2026.

Supplementary product view: top 10% / 20% **within each call week** (events grouped by the
Monday of the report week), pooled — the weekly list is what the product publishes.

C − B deltas: stock-clustered bootstrap, **500 reps, seed 32032** (archived), 95%
percentile CIs, for ROC-AUC, PR-AUC, Brier, log loss, top-10/top-20 hit and capture.

Calibration: OOF reliability by decile of p (mean p vs observed rate, n), by year (mean p vs
base rate), calibration intercept/slope (logistic of y on logit p), the same for 2026.

## Sensitivities (pre-2026 only; reported, never used to change the specification)

* MIN_PRIOR = 4 and 12.
* `vol_30d` exact-date only (no 7-day fallback).

## Verdict criteria (fixed now)

* **Survives, suitable for productionisation**: C − B ROC-AUC CI > 0, no top-10%/top-20%
  capture CI entirely below 0, C beats B in a majority of test years on ROC-AUC, and C's
  2026 ROC-AUC and top-20% capture are not below B's by more than the OOF CI width.
* **Survives weakly / needs a product decision**: C − B ROC-AUC CI > 0 but top-k behaviour
  is flat or year-unstable, or the gain does not hold in 2026; or B is as good for the
  product (top-k) and simpler.
* **Does not survive**: C − B ROC-AUC CI includes 0 or C is worse than B on top-k.

Independently of C vs B, the refit must also show that the B/C family beats the shipped
0.3.1 score (`risk_score` on the event row, legacy history) on the same rows; A0 is scored
as a reference using a Platt map fitted on each training fold.
