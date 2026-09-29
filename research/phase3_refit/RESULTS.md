# P3.2 + P3.3 refit — results

Research only. Nothing in production changed: no scoring code, threshold, config value or
`MODEL_VERSION` was touched. Definitions and decision rules were fixed in
`PREREGISTRATION.md` before any outcome was read. Anything decided afterwards is marked
**POST HOC**.

```bash
PYTHONPATH=. .venv/bin/python -m research.phase3_refit.evaluate [--rebuild]   # ~15 min, 500 reps
.venv/bin/python -m pytest testing/test_phase3_refit.py -q                    # 20 tests
```

Artifacts: `output/phase3_refit/` (`pooled.csv`, `yearly.csv`, `by_window.csv`, `tiers.csv`,
`native_tiers.csv`, `calibration.csv`, `volume_curve.csv`, `bootstrap.csv`,
`lift_promotion.csv`, `holdout_2026*.csv`, `production_availability.csv`, `summary.json`,
`oof_predictions.parquet`).

## Setup

* Target `abs_reaction_3d_anchored >= 0.08`, Benzinga-anchored (Phase 3 artifact).
* **Every feature is computed at the real call time**: the last close before the Monday of
  the report week. That is the close the weekend run scores from; every 0.3.1 prediction row
  has `score_asof_date` = the Friday before the report week. A prior outcome counts only once
  its 3-session endpoint is at or before that close. This applies to the stock's own
  history, the market prior and the lift prior.
* Population: BMO/AMC events 2014–2026 with at least 8 prior corrected outcomes. Common
  sample 21,420 events, 471 stocks. Every candidate is scored on the same rows.
* Walk-forward test years 2017–2025, **15,870 out-of-fold events** (BMO 9,248, base rate
  0.148; AMC 6,622, base rate 0.214; pooled 0.175). For year Y, training uses only events
  whose outcome endpoint falls before 1 January Y.
* **2026 YTD holdout**: 1,328 events, never used by any earlier phase. Scored once, after
  E was selected.
* Stock-clustered bootstrap, 500 reps, for every comparison below.

## 1. Current v0.3.1

`risk_score = 100 * clip(0.85 * min(p75/0.12, 1) + 0.15 * clip(entropy, 0, 1))`. Here p75 is
the rolling-28 (else expanding) p75 of the stock's prior legacy `abs_reaction_3d`. Tiers cut
at 73/79, then promoted where the stock/tier lift ≥ 1.5 (→ Elevated) or ≥ 3.0 (→ High Alert).
High Conviction = High Alert with a non-empty drift flag.

On the corrected target (OOF 2017–2025):

| | AUC | PR-AUC | BMO AUC | AMC AUC | top-10% hit / capture | top-20% capture |
|---|---|---|---|---|---|---|
| **A0** as shipped (legacy history) | 0.668 | 0.320 | **0.616** | 0.715 | 0.413 / 0.235 | 0.387 |
| **A1** same formula, corrected history | 0.706 | 0.332 | 0.673 | 0.723 | 0.430 / 0.245 | 0.404 |

A0's shipped tiers: 97.1% of BMO events are Normal, and BMO flagged capture is **7.3%**. That
is the audit's BMO defect, and it is still in production.

## 2. Candidates (all logistic, L2 C=1, standardised on the training fold)

| | inputs (log-transformed) |
|---|---|
| B `structural` | mean of the stock's prior corrected \|3-session reaction\| |
| C `structural_vol` | B + `vol_30d` at the call cutoff (the value production's pending row already carries) |
| D `phase5b_pair` | C + sector peer reaction level (`peer_sec_rw_mean_abs_1d_20`) + `idio_vol_30d` |

## 3. Walk-forward results (call cutoff, common calibration)

| | AUC | PR-AUC | Brier | log loss | Δ vs B, AUC [95% CI] |
|---|---|---|---|---|---|
| A0 | 0.668 | 0.320 | 0.1361 | 0.4392 | |
| A1 | 0.706 | 0.332 | 0.1331 | 0.4262 | |
| B | 0.712 | 0.348 | 0.1323 | 0.4232 | — |
| **C** | **0.723** | 0.350 | 0.1315 | 0.4188 | **+0.011 [+0.005, +0.016]** |
| D | 0.729 | 0.354 | 0.1308 | 0.4159 | D−C +0.006 [+0.003, +0.008] |

Yearly AUC: C beats A0 in all 9 years, by 0.04–0.07. C beats B in 6 of 9 years.
The biggest single-year gain is 2020 (+0.041). **POST HOC:** without 2020, C−B is +0.007
[+0.002, +0.011], so the gain is smaller but still there.

**Call-time vs eve:** C call−eve AUC +0.0003 [−0.0004, +0.0009]; D −0.0003 [−0.0021, +0.0015].
The 0–4 sessions between the weekend call and the event eve change nothing measurable.
Research done at event eve was not borrowing information the product lacks.

## 4. Product metrics

At equal volume (`volume_curve.csv`, OOF pooled):

| flagged | A0 hit / capture | A1 | B | C |
|---|---|---|---|---|
| top 5% | 0.465 / 0.133 | 0.463 / 0.132 | 0.472 / 0.135 | 0.472 / 0.135 |
| top 10% | 0.413 / 0.235 | 0.430 / 0.245 | 0.430 / 0.245 | 0.427 / 0.244 |
| top 20% | 0.339 / 0.387 | 0.354 / 0.404 | 0.359 / 0.409 | 0.359 / 0.410 |
| top 30% | 0.291 / 0.499 | 0.311 / 0.533 | 0.315 / 0.540 | 0.322 / 0.551 |

* C − A0: top-20% capture **+0.023 [+0.007, +0.044]**; top-10% capture +0.008
  [−0.007, +0.020] (not significant).
* C − B and D − C: no top-10% or top-20% capture difference. Every CI crosses zero.
  **The AUC gains from vol, peer and idio vol all come from the middle of the ranking.** This
  repeats Phase 5B.
* Top-10% lift ≈ 2.44x for A1, B and C alike.

## 5. BMO vs AMC

| C, OOF | n | base | AUC | PR-AUC | top-20% capture | mean p | calibration gap | slope |
|---|---|---|---|---|---|---|---|---|
| BMO | 9,248 | 0.148 | 0.691 | 0.274 | 0.383 | 0.138 | −1.0 pt | 0.90 |
| AMC | 6,622 | 0.214 | 0.740 | 0.408 | 0.411 | 0.198 | −1.6 pt | 0.95 |

For comparison, A0 on BMO: AUC 0.616, top-20% capture 0.303, and the Platt-calibrated gap
is −4.4 pt.

**Window-specific calibration does not help.** W1 − W0 log loss +0.0002 [−0.0001, +0.0005];
W2 − W0 +0.0001 [−0.0002, +0.0003]. The same holds for B and D. One common model is
already calibrated within each window to about 1–1.5 points, because the stock history
already carries the BMO/AMC difference.

C tiers with common cuts (top 10% / next 10% of training probabilities):

| | HA share | HA hit | flagged share | flagged hit | flagged capture | stratified lift (flagged) |
|---|---|---|---|---|---|---|
| ALL | 12.7% | 0.401 | 24.6% | 0.340 | 0.477 | 1.80x |
| BMO | 6.8% | 0.340 | 15.9% | 0.295 | 0.317 | 2.00x |
| AMC | 20.9% | 0.429 | 36.8% | 0.367 | 0.632 | 1.71x |

Compare A0 shipped: flagged 17.8%, hit 0.351, capture 0.356, stratified lift 1.69x; BMO
flagged share 2.9%, capture 7.3%.

**Per-window cut points** (pre-registered rule 3 selected them) raise BMO flagged capture
0.317 → 0.482 and lower AMC capture 0.632 → 0.454. **POST HOC, at equal total volume:** a
common cut gives pooled hit 0.338 vs 0.320 and capture 0.495 vs 0.468. Per-window cuts do
not add skill; they move flags from AMC to BMO. My rule 3 measured capture, which rises with
volume, so it rewarded that shift. I recommend **common cut points** and flag this as a
product choice (how many BMO names to show), not a modelling result.

INTRADAY (45 events) and UNKNOWN have no anchored target and are excluded. They should be
tiered from the common model with no window-specific claim.

## 6. Score mechanics

* **12% ceiling:** 687 OOF events (4.3%) sit tied at score 100.0. Split at their own median
  p75, they go extreme at **51.7% vs 43.1%**. The ceiling costs 0.0002 AUC [−0.0000, +0.0004]
  and nothing in top-10%/top-20% capture. It is a tie-breaking defect, not a
  discrimination one. **Remove it:** a model on the raw magnitude never ties.
* **Entropy:** at or above 1.0 (clipped) on **99.1%** of events, so it is a flat +15.
  Removing it changes AUC by 0.0001 [−0.0000, +0.0002]. **Remove it.**
* **0.85/0.15 weights and p75:** rank correlation between A1 and uncapped p75 is 0.9998.
  The expanding mean beats the 0.3.1 score by +0.005 [+0.0005, +0.009] AUC. Nothing in the
  old construction is worth keeping.
* **Lift promotion:** on A1, promotion raises flagged capture +3.7 pt only by flagging 2.9 pt
  more events, and flagged precision falls −1.8 pt. On C, pre-registered rule 4 technically
  passes: 418 promoted events hit 25.8%, near Elevated's 27.6%. **POST HOC, at equal
  volume:** promoting the same number of the next-best events *by score* hits 26.8%, and
  under common cuts 20.4% vs 27.8%. Lift promotion is worse than flagging a few more events
  by score. **Remove it**, along with `LIFT_PRIOR_STRENGTH` / `LIFT_TO_*`.
* **High Conviction** (C High Alert plus |drift z| ≥ 1.5 at the call cutoff): 314 events hit
  33.1% vs 36.7% for the rest of High Alert. The difference is −3.6 pt [−9.3, +2.3], and HC
  is lower in 6 of 9 years. **Drop it.** The shipped HC (legacy, event-eve drift) looks
  better, 51.5% vs 41.2% on 132 events, but it cannot be reproduced at call time.

## 7. Recommended score

Pre-registered selection: **E = C, common calibration.**

```
p = sigmoid(b0 + b1 * z(log hist_mean_abs) + b2 * z(log vol_30d))
hist_mean_abs = mean |3-session corrected reaction| over the stock's prior events whose
                endpoint is at or before the call close; at least 8 required
vol_30d       = production's vol_30d on the stock's row at the call close
```

Standardised coefficients are stable across folds: structural 0.73–0.79, vol_30d
0.28–0.33. The current fit (trained on everything observable before 2026-01-01) is
intercept −1.854, structural 0.740, vol_30d 0.275, with common cuts Elevated p ≥ 0.238 and
High Alert p ≥ 0.328. Exact scaling constants are in `summary.json → E.fold_coefficients`.
The model is refit yearly on everything observable before 1 January.

**POST HOC caveat that should shape the product decision:** vol_30d makes the flagged
volume follow the market. Under common cuts, C flags 14.5%–40% of events depending on the
year (2020: 40%, and High Alert precision falls to 0.34 vs B's 0.40). B, on history alone,
flags a steady 20–26% every year with the same top-20% capture. If a stable weekly list
matters more than +0.011 AUC and better log loss, B is the defensible alternative.

## 8. Tiers and calibration

* One common model and one common probability calibration. No BMO/AMC adjustment: tested
  three ways, none improves out-of-fold log loss.
* High Alert = top 10%, Elevated = next 10% of the training-fold probabilities, refit yearly.
  0.3.1 flags about 9% / 9%, so this matches its volume. Realised flag rates drift (see §7),
  so the cut points should be monitored.
* Headline diagnostic: flagged stratified lift **1.80x** (0.3.1 shipped: 1.69x); BMO 2.00x on
  15.9% of BMO events, vs 2.53x on 2.9%.

## 2026 holdout (scored once)

| | AUC | BMO AUC | AMC AUC | top-10% hit | top-20% capture |
|---|---|---|---|---|---|
| A0 shipped | 0.637 | 0.602 | 0.707 | 0.466 | 0.333 |
| A1 | 0.702 | 0.682 | 0.716 | 0.436 | 0.361 |
| B | 0.713 | 0.691 | 0.733 | 0.489 | 0.368 |
| **C** | **0.716** | 0.687 | 0.742 | **0.511** | 0.374 |
| D | 0.718 | 0.693 | 0.742 | 0.496 | 0.358 |

The ordering holds on data no phase has touched. Calibration under-predicts 2026: base rate
0.242 vs mean p 0.207. The base rate has been rising since 2021, which argues for yearly
refits. E's tiers (per-window cuts, as the rule selected) flag 33% of 2026 events (hit 0.389, capture 0.530). A0 flags 21% (hit
0.408, capture 0.358) and only 4.4% of BMO.

## Production availability

* **Production cannot compute E today.** The corrected history here comes from the Benzinga
  snapshot, which is research-only. The production DB's `earnings.announce_ts_ny` holds
  **about 250 timestamps** (2020–2026). The 12,068-row yfinance backfill is gone again, most
  likely overwritten by the droplet sync noted on 2026-09-06.
* If that backfill were restored, yfinance-only history would be enough. Using only events
  with a yfinance timestamp (2020+), coverage at ≥8 prior outcomes is 98–99% from 2023.
  E's AUC: 2023 0.708 vs 0.726 with full history, 2024 0.722 vs 0.739, 2025 0.689 vs 0.691,
  **2026 0.7125 vs 0.7131**. The gap has closed.

## 9. v0.3.2 decision

```
DO NOT PROMOTE YET
```

The model side of the checklist passes. The score uses the corrected target, is causal at
the real call time, and has no known leak. It is simpler than 0.3.1 (two inputs, no cap, no
entropy, no lift, no HC), beats the strongest simple baseline, and does not degrade product
metrics (+2.3 pt top-20% capture vs 0.3.1). BMO AUC goes 0.616 → 0.691 and BMO is no longer
97% Normal. Cut points are fitted causally, the result holds without 2020, and the 2026
holdout confirms it.

**Item 10 fails: production cannot reproduce the research result.** Production has no
corrected outcome history (about 250 timestamps) and no code path that builds one. Shipping
now would mean scoring every stock on missing history, or quietly falling back to the legacy
target this rebuild exists to replace. Two decisions are also still open: whether yearly
coefficients may be fitted from the licensed Benzinga research history or only from the
2020+ yfinance history, and B vs C given the volume instability in §7.

What would make it promotable, in order:

1. Restore the yfinance timestamps (`scripts/backfill_announcement_timestamps.py`) and stop
   the droplet sync from overwriting them. Without this, nothing else matters.
2. Decide the source for fitting coefficients (see above).
3. Port E (list below), then add a parity test: production scores on history equal
   `oof_predictions.parquet` for the same fold.

## 10. Files that would change in production (at promotion)

* `pipeline/events.py` — build corrected per-stock history from `reaction_3d_anchored` with
  the endpoint rule. Attach `vol_30d` at the call close for completed events (pending rows
  already carry it).
* `feature_engineering/event_features.py` — replace `event_explosiveness_score`,
  `event_stock_bucket_lift_values`, `event_lift_adjusted_bucket`, `event_high_conviction`
  with a two-input scorer and cut-point tiering. Delete the entropy ffill.
* `scoring/scoring_features.py`, `pipeline/stage4.py` — remove the lift/promotion/HC stages.
* `config.py` — delete `BUCKET_*_FLOOR`, `LIFT_*`. Add fitted coefficients and cut points
  (or a small JSON artifact). `MODEL_VERSION = "0.3.2"`.
* Consumers of `stock_bucket_lift`, `is_high_conviction` and
  `earnings_explosiveness_bucket_structural`: `report/`, `streamlit_dash/`,
  `analysis/save_predictions.py`, `cron/cron_weekly_digest.py`,
  `marketing/generate_public_track_record.py`.
* `testing/test_event_frame.py`, `testing/test_pipeline.py` — `assert_completed_parity`
  compares against the legacy daily-frame score and must be rewritten for the new score.

## Limitations

* The event window used at call time is the eventual Benzinga record, not the pre-event
  schedule. 97.7% of events share the stock's previous window, so the effect is small but
  non-zero.
* Sector groupings (D only) are today's, not point-in-time.
* The walk-forward starts in 2017 because corrected history is thin before 2014.
* 18 model variants × 3 tier rules were evaluated. Selection followed fixed rules, and the
  two rules that turned out to reward volume are overridden **POST HOC** above, as stated.
