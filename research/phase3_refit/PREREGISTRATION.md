# P3.2 + P3.3 refit — pre-registration

Written 2026-09-28, **before any outcome in this phase was computed**. Everything below
is fixed. Anything chosen after results are read is labelled **POST HOC** in RESULTS.md.

Sources: `output/phase3_target_rebuild/phase3_events.parquet` (Benzinga-anchored corrected
target and history) and `output/full_df.parquet` (daily prices, `vol_30d`, `drift_30d`).
Target: `y = abs_reaction_3d_anchored >= 0.08`. Unresolved outcomes stay missing.

## Clocks

* `report_date` = min(`earnings_date`, `phase3_proxy_session_date`).
* **Call cutoff** = last market session strictly before the Monday of `report_date`'s
  week. That is the close the weekend run scores from (`score_asof_date` in
  `db/predictions.duckdb` is the Friday before the report week on every 0.3.1 row).
* **Eve cutoff** = last session strictly before `report_date` (Phase 5B's cutoff).
* A prior event's 3-session outcome is usable at cutoff c only if its endpoint session
  (anchor + 3) is <= c. The same rule applies to the stock's own history, the market-wide
  prior and the lift prior. A peer's 1-session reaction needs anchor + 1 <= c.
* A model used for test year Y is trained on events whose 3-session endpoint is before
  1 January Y.

## Population

Completed events, 3-session anchored target available, window BMO or AMC, report year
2014–2026, at least 8 prior corrected outcomes usable at the call cutoff.
INTRADAY (45 events) and UNKNOWN are excluded: no anchored target exists for them.

Walk-forward test years **2017–2025** (training starts 2014). **2026 YTD is a held-out
confirmation year**: it is scored once, after E is selected, by a model trained on
everything with an endpoint before 2026-01-01. No earlier phase used 2026.

Common sample = rows where every candidate's inputs exist, except the peer feature, which
may be missing and is median-imputed inside the training fold (as in Phase 5B).

## Features (as of the call cutoff; `_eve` versions as of the eve cutoff)

| name | definition |
|---|---|
| `hist_mean_abs` | mean of the stock's prior corrected \|r3\| (Phase 4 `long_mean_abs_reaction`) |
| `hist_p75` | p75 of the last 28 prior \|r3\| if >= 28 exist, else expanding p75 (0.3.1's input) |
| `hist_entropy` | 0.3.1 entropy (8 bins, >= 8 obs) over prior \|r3\|, falling back to \|r1\| per event |
| `vol_30d` | `full_df.vol_30d` on the stock's row at the cutoff date (backward as-of, 7 days max) — exactly what the production pending row carries |
| `idio_vol_30d` | Phase 5B definition, cutoff moved |
| `peer_sec_rw_mean_abs_1d_20` | Phase 5B definition, cutoff moved |
| `drift_30d` | `full_df.drift_30d` at the cutoff date (for High Conviction only) |

## Score candidates

* **A0 `v031_shipped`** — the `risk_score` / `earnings_explosiveness_bucket` 0.3.1 actually
  computes (legacy history). Reference only.
* **A1 `v031_corrected`** — 0.3.1 formula on the corrected history above:
  `100*clip(0.85*min(hist_p75/0.12,1) + 0.15*clip(hist_entropy,0,1))`, missing entropy -> 0
  (the cross-ticker `ffill` is a known defect and is not reproduced), tiers 73/79, lift
  promotion with prior strength 20 and gates 1.5 / 3.0, endpoint-safe priors.
* Mechanics diagnostics on A1: `a1_nocap` (p75 uncapped + entropy term), `a1_noentropy`
  (capped p75 only), `p75_uncapped` (hist_p75 alone).
* **B `structural`** — logistic on `log(hist_mean_abs)`.
* **C `structural_vol`** — logistic on `log(hist_mean_abs)`, `log(vol_30d)`.
* **D `phase5b_pair`** — C + `log(peer_sec_rw_mean_abs_1d_20)` + `log(idio_vol_30d)`.
  Chosen post hoc in Phase 5B; its 0.7167 is not independent confirmation.

All logistic models: standardise on the training fold, L2, C = 1.0, never tuned. Log
transform chosen a priori because every input is a positive, right-skewed magnitude.
A scores are Platt-calibrated on the training fold when a probability is needed.

## Window handling (P3.3)

* **W0** common model, common calibration.
* **W1** common score, window-specific Platt recalibration fitted on the training fold.
* **W2** window indicator (`is_bmo`) added as a logistic feature.

The event's own window is used at call time (it is on the published schedule; 97.7% of
events share the stock's previous window).

## Tier rules

* **Q10/10 (primary)**: High Alert = top 10%, Elevated = next 10%, cut points = the 90th /
  80th percentiles of the frozen model's probabilities on the training fold. 0.3.1 flags
  ~9% / ~9% in 2014–2026, so this is volume-matched to it.
* **Q10/10-by-window**: the same percentiles computed separately within BMO and AMC.
* **P (sensitivity)**: High Alert p >= 2.0x training base rate, Elevated p >= 1.5x.
* A1 is also shown with its native 73/79 + promotion tiers.

## Metrics

ROC-AUC, PR-AUC, Brier, log loss (probabilities only), top-10% / top-20% hit rate,
capture and lift, tier sizes / hit rates / capture / lift, **stratified lift** (observed
over the rate expected from window mix alone, audit §Q2), per-year, per-window,
calibration by window (mean p vs realised, Brier, log loss). Stock-clustered bootstrap,
500 reps, fixed seed, for every comparison named below.

## Selection rules for E (fixed now)

1. **Score.** Start from B. Use C if C − B has AUC CI > 0 and no top-10%/top-20% capture
   CI entirely below 0. Then use D over that only if D − C has a top-10% or top-20%
   capture CI > 0 (Phase 5B's "product relevant" rule), measured at the call cutoff.
2. **Window.** W0 unless W1 or W2 lowers pooled OOF log loss with CI < 0. If both do, the
   one with lower log loss; if their CIs overlap, W2 (one parameter instead of two).
3. **Tiers.** Q10/10 on E's probability. Q10/10-by-window replaces it only if it raises
   BMO capture without lowering pooled top-20% capture by more than 1 point.
4. **Lift promotion** is kept only if promoted events' OOF hit rate is closer to the
   destination tier than to the origin tier AND tier capture rises with CI > 0 at no loss
   of High Alert precision beyond 1 point. Tested at 0.3.1 settings (20, 1.5, 3.0), with
   prior strength 10 and 40 as sensitivity. No gate search.
5. **High Conviction** (High Alert AND |drift z| >= 1.5 at the call cutoff) is kept only if
   HC − (High Alert without HC) hit rate has a stock-clustered CI > 0.

## Production-availability sensitivity

Recompute E's history using only events that also carry a yfinance timestamp
(`audit/provider_timestamps.parquet`, 2020+) — the most production could have if its
backfill were restored — and report coverage and 2023–2026 performance.

## Promotion to 0.3.2

Only if the checklist in the task brief is met **and** production can reproduce E from
its own data. Otherwise `MODEL_VERSION` stays 0.3.1.
