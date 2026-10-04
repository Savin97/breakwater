# Model C on the production event frame — results

Research only. No production scoring, tier, report, dashboard, `MODEL_VERSION` or database
was changed. The specification (`SPEC.md`) was written before any outcome was computed;
the 2026 holdout was scored once, after `FROZEN.json` recorded SPEC.md's SHA-256.

```bash
PYTHONPATH=. .venv/bin/python -m analysis.model_c_refit.evaluate build      # ~5 s
PYTHONPATH=. .venv/bin/python -m analysis.model_c_refit.evaluate pre2026    # ~75 s, writes FROZEN.json
PYTHONPATH=. .venv/bin/python -m analysis.model_c_refit.evaluate holdout    # refuses without FROZEN.json
PYTHONPATH=. .venv/bin/python -m analysis.model_c_refit.evaluate compare
PYTHONPATH=. .venv/bin/python -m pytest analysis/model_c_refit/tests -q     # 17 leakage/protocol checks
```

Outputs: `output/model_c_refit/` (gitignored). Data: `output/full_df.parquet` and
`db/breakwater.duckdb` as synced 2026-09-29; prices end 2026-09-28.

## Verdict

**Model C survives weakly; the choice between B and C is a product decision.**

* The structural signal survives the production refit essentially unchanged: on the
  15,795 out-of-sample events both runs share, production gives B 0.711 / C 0.722 ROC-AUC
  against the archived 0.712 / 0.722.
* Both B and C clearly beat the shipped 0.3.1 score (C − A0: ROC-AUC +0.053 [+0.042,
  +0.067], top-20% capture +2.2 pt [+0.7, +4.0]; BMO ROC-AUC 0.69 vs 0.62).
* C's gain over B is real in ROC-AUC (+0.010 [+0.005, +0.015]), log loss and Brier, but
  **not in the ranking the product publishes**: top-10% capture −0.8 pt [−1.6, +0.3],
  top-20% capture −0.1 pt [−1.2, +1.0], within-week top-10% and top-20% equal. In 2026, C −
  B ROC-AUC is +0.002 [−0.006, +0.011].

The pre-registered "survives" bullets are all met (ROC-AUC CI > 0, no top-k CI entirely
below 0, C ahead in 6 of 9 years, 2026 not worse). But the spec also says the verdict rests
on top-k behaviour, and there C and B are the same. The B/C family is ready to replace
0.3.1; whether the shipped model is B or C turns on what the product exposes (see
Calibration) and on how much weekly flag volume may move with the market, which
`vol_30d` drives (archived finding, not re-measured here because no tier thresholds were set).

## Data

| step | events |
|---|---:|
| production event frame | 46,200 (45,715 completed, 485 pending) |
| target unresolved: no timestamp | −19,666 |
| target unresolved: intraday | −62 |
| target endpoint unavailable (after last price 2, price gap 1, beyond grid 1) | −4 |
| **corrected 3-session target available** | **25,983** |
| report year outside 2014–2026 | −2,674 |
| fewer than 8 usable prior outcomes | −1,382 |
| `vol_30d` unavailable at the cutoff | 0 |
| **common sample (B and C on identical rows)** | **21,927** (499 stocks) |

Pre-2026 common sample 20,491; out-of-sample 2017–2025: **16,315** (BMO 9,462, AMC 6,853,
494 stocks). 2026 holdout: **1,436** (BMO 823, AMC 613), reports 2026-01-13 → 2026-09-24.

| year | target available | common | BMO | AMC | base rate |
|---|---:|---:|---:|---:|---:|
| 2014 | 1,596 | 984 | 604 | 380 | 0.090 |
| 2015 | 1,671 | 1,576 | 947 | 629 | 0.129 |
| 2016 | 1,701 | 1,616 | 974 | 642 | 0.154 |
| 2017 | 1,730 | 1,663 | 996 | 667 | 0.114 |
| 2018 | 1,744 | 1,688 | 1,003 | 685 | 0.174 |
| 2019 | 1,784 | 1,729 | 1,009 | 720 | 0.157 |
| 2020 | 1,837 | 1,762 | 1,013 | 749 | 0.188 |
| 2021 | 1,936 | 1,801 | 1,031 | 770 | 0.124 |
| 2022 | 1,944 | 1,844 | 1,073 | 771 | 0.211 |
| 2023 | 1,952 | 1,928 | 1,108 | 820 | 0.185 |
| 2024 | 1,984 | 1,953 | 1,108 | 845 | 0.224 |
| 2025 | 1,980 | 1,947 | 1,121 | 826 | 0.217 |
| 2026 | 1,450 | 1,436 | 823 | 613 | 0.247 |

Missing history: an event needs 8 usable prior corrected outcomes; without them it is
excluded, never imputed. 612 of the 1,382 short-history exclusions are in 2014 (timed
history starts 2012–2013); later years lose 14–135 per year. Every common-sample row read
`vol_30d` on the exact cutoff date; the 7-day fallback was never used.

## Method

* **Target**: `abs_reaction_3d_anchored >= 0.08`, completed events with
  `reaction_3d_anchored_status == "available"` only. Production timing rules unchanged.
* **Call cutoff**: last market session strictly before the Monday of the `earnings_date`
  week (market-session grid from the daily frame).
* **B**: logistic on `z(log hist_mean_abs)`; `hist_mean_abs` = mean `abs_reaction_3d_anchored`
  of the stock's prior events whose outcome endpoint (anchor + 3 sessions) is on or before
  the cutoff, at least 8.
* **C**: B + `z(log vol_30d)`, production `vol_30d` on the stock's row at the cutoff.
* L2 logistic, C = 1, standardised on the training fold. Walk-forward test years 2017–2025;
  year Y trains on rows whose endpoint is before 1 January Y. 2026 = holdout, fitted on
  everything with endpoint before 2026-01-01.
* Stock-clustered bootstrap, 500 reps, seed 32032.

## Results (out of sample, 2017–2025, identical rows)

| | ROC-AUC | PR-AUC | Brier | log loss | top-10% hit / capture | top-20% hit / capture |
|---|---|---|---|---|---|---|
| A0 shipped 0.3.1 | 0.672 | 0.327 | 0.1377 | 0.4431 | 0.428 / 0.239 | 0.349 / 0.390 |
| **B** | 0.715 | 0.357 | 0.1338 | 0.4267 | 0.448 / 0.251 | 0.370 / 0.414 |
| **C** | 0.725 | 0.359 | 0.1331 | 0.4225 | 0.433 / 0.242 | 0.369 / 0.413 |

Base rate 0.179, n 16,315.

**C − B, 95% CI** (share of draws > 0): ROC-AUC **+0.0102 [+0.0055, +0.0154]** (1.00);
PR-AUC +0.0019 [−0.0037, +0.0081] (0.76); Brier −0.0007 [−0.0013, −0.0001]; log loss
−0.0042 [−0.0062, −0.0023]; top-10% capture −0.0082 [−0.0156, +0.0031] (0.10); top-20%
capture −0.0014 [−0.0120, +0.0097] (0.43).

**Within each call week** (what the weekly list does): top 10% — B hit 0.401 / capture
0.254, C 0.402 / 0.254; top 20% — B 0.343 / 0.406, C 0.348 / 0.411.

**Yearly**

| year | n | base | AUC A0 | AUC B | AUC C | top-10 cap B | top-10 cap C | top-20 cap B | top-20 cap C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2017 | 1,663 | 0.114 | 0.713 | 0.763 | 0.772 | 0.344 | 0.323 | 0.513 | 0.540 |
| 2018 | 1,688 | 0.174 | 0.659 | 0.717 | 0.726 | 0.224 | 0.235 | 0.408 | 0.405 |
| 2019 | 1,729 | 0.157 | 0.659 | 0.717 | 0.716 | 0.243 | 0.246 | 0.404 | 0.401 |
| 2020 | 1,762 | 0.188 | 0.634 | 0.643 | 0.683 | 0.202 | 0.193 | 0.340 | 0.373 |
| 2021 | 1,801 | 0.124 | 0.709 | 0.756 | 0.755 | 0.286 | 0.277 | 0.504 | 0.460 |
| 2022 | 1,844 | 0.211 | 0.653 | 0.679 | 0.690 | 0.238 | 0.238 | 0.385 | 0.387 |
| 2023 | 1,928 | 0.185 | 0.677 | 0.736 | 0.738 | 0.300 | 0.280 | 0.448 | 0.434 |
| 2024 | 1,953 | 0.224 | 0.686 | 0.735 | 0.747 | 0.220 | 0.222 | 0.405 | 0.403 |
| 2025 | 1,947 | 0.217 | 0.637 | 0.696 | 0.691 | 0.213 | 0.201 | 0.363 | 0.339 |

C ≥ B ROC-AUC in 6 of 9 years (largest gain 2020, +0.040); top-10% capture C ≥ B in 4 of 9;
top-20% capture C ≥ B in 3 of 9. B and C beat A0 in every year.

**BMO vs AMC**

| | n | base | AUC B | AUC C | C − B AUC [CI] | top-10 cap B / C | top-20 cap B / C |
|---|---:|---:|---:|---:|---|---|---|
| BMO | 9,462 | 0.149 | 0.680 | 0.692 | +0.012 [+0.005, +0.021] | 0.224 / 0.220 | 0.370 / 0.384 |
| AMC | 6,853 | 0.220 | 0.732 | 0.741 | +0.009 [+0.003, +0.015] | 0.223 / 0.226 | 0.406 / 0.409 |

A0 on BMO: AUC 0.617, top-20% capture 0.297 — the audit's BMO defect; B/C repair it.

**Sensitivities (pre-2026; not used to change the spec)**: MIN_PRIOR 4 → C − B ROC-AUC
+0.010 [+0.005, +0.015] (n 16,618); MIN_PRIOR 12 → +0.011 [+0.007, +0.016] (n 15,984).
Exact-date-only `vol_30d` is identical to the primary run (no fallback was ever used).

## 2026 holdout (scored once)

| | ROC-AUC | PR-AUC | Brier | top-10% hit / capture | top-20% hit / capture |
|---|---|---|---|---|---|
| A0 | 0.651 | 0.381 | 0.1778 | 0.507 / 0.206 | 0.444 / 0.361 |
| B | 0.717 | 0.428 | 0.1708 | 0.500 / 0.203 | 0.476 / 0.386 |
| C | 0.720 | 0.437 | 0.1681 | 0.549 / 0.223 | 0.469 / 0.380 |

n 1,436, base rate 0.247. C − B: ROC-AUC +0.002 [−0.006, +0.011]; top-10% capture +0.020
[−0.008, +0.037]; top-20% capture −0.006 [−0.021, +0.021]; Brier −0.0027 [−0.0048,
−0.0008]. By window: AMC AUC B 0.737 / C 0.742; BMO 0.690 / 0.688. Within-week top 10%:
B 0.237 / C 0.234 capture; top 20%: both 0.366.

## Comparison with archived Model C

| | archived (2026-09-28) | production refit |
|---|---|---|
| frame | Benzinga matched ±1 day, proxy report date | production event frame, exact dates |
| B / C sample (this refit's definition) | 21,648 | 21,927 |
| OOF events 2017–2025 | 15,870 | 16,315 |
| B / C OOF ROC-AUC | 0.712 / 0.723 | 0.715 / 0.725 |
| C − B ROC-AUC | +0.011 [+0.005, +0.016] | +0.010 [+0.005, +0.015] |
| top-10% capture B / C | 0.245 / 0.244 | 0.251 / 0.242 |
| top-20% capture B / C | 0.409 / 0.410 | 0.414 / 0.413 |
| 2026 C ROC-AUC (n) | 0.716 (1,328) | 0.720 (1,436) |
| coefficients C (intercept / hist / vol) | −1.854 / 0.740 / 0.275 | −1.830 / 0.751 / 0.271 |

The archived run reported a common sample of 21,420 because it also required the inputs of
candidates A1 and D; 21,648 is its sample under this refit's B/C definition.

**Why the samples differ** (archived vs production, same B/C definition; 21,487 shared):

* 161 archived-only events: **97 have no timestamp in production — every one was matched in
  the archived run to a vendor record dated one day off** (79 a day earlier, 18 a day
  later; 66 stocks), exactly the ±1-day substitution production refuses. 8 are INTRADAY
  under the production timestamp, 55 drop below 8 usable priors once those events are
  gone, 1 moved date.
* 440 production-only events: 315 had no archived target (no confirmed exact vendor match)
  but carry an independently observed yfinance timestamp in production; 102 did not exist
  in the 2026-09-28 frame (all 102 are in the 2026-10-02 rebuild; 89 are 2025–2026 reports,
  consistent with the 2026-09-29 earnings-date corrections); 23 had < 8 archived priors.
* Against the archived rules rebuilt on today's data (2026-10-02 frame), the 102 refresh
  cases disappear: 160 vs 326, the rest identical. So the rule change itself moves ~1.4% of
  events in each direction, and the data refresh adds ~100.
* On the 21,487 shared events: 6 labels differ, 15 windows differ, 1 call cutoff differs
  (the archived proxy date), `hist_mean_abs` identical at the median (p99 |Δ| 0.005).
  `n_prior` differs on 3,766 rows because the extra and missing timed events shift counts.
* On the 15,795 OOF events both runs scored: archived B 0.712 / C 0.722, production B 0.711
  / C 0.722. **The signal is the same; the production rules cost nothing.**

## Candidate production fit (NOT deployed)

Trained on every common-sample event with outcome endpoint before 2026-01-01: report
years 2014–2025, **n 20,491**, base rate 0.1688.

```
p = sigmoid(-1.8303 + 0.7513 * z_hist + 0.2707 * z_vol)
z_hist = (ln(max(hist_mean_abs, 1e-4)) - (-3.238277)) / 0.443920
z_vol  = (ln(max(vol_30d,       1e-4)) - (-4.209207)) / 0.458417
```

B for reference: `sigmoid(-1.8113 + 0.8503 * z_hist)`, same standardisation.

## Calibration

* Pooled OOF, C: calibration intercept +0.009, slope 0.94; mean p 0.166 vs observed 0.179.
  By decile: within ~1 pt at the ends, under-predicts by 2.5–4.5 pt in deciles 3–8
  (e.g. mean p 0.213 vs observed 0.257).
* **By year the level drifts with the market**: calibration intercept ranges −0.46 (2020:
  p 0.206 vs observed 0.188) to +0.56 (2024: p 0.169 vs observed 0.224). Slopes 0.72–1.18.
* 2026: mean p 0.212 vs observed 0.247 (intercept +0.19, slope 0.97); the base rate has
  risen every year since 2021 and a model trained on 2014–2025 (base 0.169) lags it.
* **Finding**: the ranking is stable, the probability level is not. A raw probability is
  not credible enough to show customers as "X% chance of an 8% move" — in a given year it
  can be off by 3–6 points on average. Expose a percentile / risk rank (or tier) instead;
  a probability would need recency recalibration and its own audit first.

## Leakage and protocol checks (`tests/`, 17 pass)

Own reaction cannot enter own score; history equals a brute-force recount (synthetic and 80
real events); an outcome counts only once its endpoint is on or before the cutoff; cutoff
is the last session before the report week (holiday Friday included); `vol_30d` is read on
or before the cutoff and is unmoved by any later row; the target is the anchored one, never
the legacy column; unresolved targets stay missing and are not counted as history; B and C
use identical rows; scrambling every 2026 label and feature leaves the pre-2026 results
bit-identical; training rows always end before the test year; the final fit excludes 2026;
shuffling event order changes nothing; the holdout refuses to run without the frozen spec.
