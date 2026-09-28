# Phase 5B — free price/peer features vs `structural + vol_30d`

Research only. No production code, threshold, score, report, dashboard or database was
touched. Definitions were fixed in `PREREGISTRATION.md` before any outcome was read.
Artifacts: `output/phase5b_free_features/`. Reproduce:

```bash
.venv/bin/python -m phase_5b_new_feature_testing.build      # features, ~30 s
.venv/bin/python -m phase_5b_new_feature_testing.evaluate   # models + bootstrap, ~20 min
.venv/bin/python -m phase_5b_new_feature_testing.posthoc    # POST HOC decomposition
.venv/bin/python -m pytest phase_5b_new_feature_testing/tests -q   # 26 tests
```

## Setup

* **Sample:** exactly Phase 5's common sample. Mature-28 cohort, 2019–2025, **10,934 events,
  441 stocks**, base rate 0.179. The own-stock price-feature requirement removed no rows.
  **Walk-forward:** test years 2021–2025, **8,410 out-of-fold events**, Phase 5's
  `walk_forward_predictions` unchanged.
* **Incumbent reproduced:** structural 0.6986 → structural + vol_30d **0.7097**. The
  difference is +0.0111 [0.0058, 0.0167], identical to Phase 5.
* **Scale of the search:** 29 primary features in 7 families, 23 secondary features,
  71 walk-forward models. Each model has stock-clustered bootstrap deltas vs the incumbent
  (500 reps) on AUC, PR-AUC, Brier, top-10% hit rate, top-10% capture and top-20% capture.
* **How to read the multiplicity:** 29 primary features were tested at 95%, so about 1–2
  false "statistically incremental" results are expected by chance.

## Headline table (pooled OOF, deltas vs incumbent, 95% stock-clustered CI)

| model | AUC | ΔAUC [CI] | top-10% capture Δ | top-20% capture Δ | years ΔAUC>0 |
|---|---|---|---|---|---|
| incumbent structural + vol_30d | 0.7097 | — | (0.231) | (0.400) | — |
| + peer_sec_rw_mean_abs_1d_20 | 0.7157 | **+0.0060 [+0.0020, +0.0102]** | +0.005 [−0.007, …] | −0.003 [−0.012, …] | 4/5 |
| + peer_sec_mean_abs_1d_20 | 0.7152 | **+0.0055 [+0.0017, +0.0096]** | +0.005 [−0.007, …] | −0.003 [−0.013, …] | 4/5 |
| + peer_sub_mean_abs_1d_20 | 0.7129 | **+0.0032 [+0.0008, +0.0059]** | +0.003 [−0.010, …] | +0.001 [−0.006, …] | 4/5 |
| + idio_vol_30d | 0.7108 | **+0.0011 [+0.0005, +0.0017]** | +0.003 [−0.003, …] | −0.001 [−0.005, …] | 4/5 |
| + idio_share_30d | 0.7107 | **+0.0010 [+0.0001, +0.0020]** | +0.003 [−0.004, …] | +0.001 [−0.007, …] | 4/5 |
| structural + idio_vol_30d (replaces vol_30d) | 0.7103 | +0.0006 [−0.0022, +0.0039] | −0.001 | +0.003 | 2/5 |
| + family 5 (all 12 peer features) | 0.7086 | −0.0011 [−0.0074, +0.0054] | −0.001 | −0.002 | 2/5 |
| + family 6 peer vol | 0.7042 | **−0.0054 [−0.0088, −0.0016]** | −0.001 | −0.010 | 1/5 |
| + all 29 primary (E) | 0.7056 | −0.0041 [−0.0123, +0.0038] | −0.002 | −0.011 | 2/5 |

The pre-registered rules give these outcomes:

* **Robust beat** (AUC CI > 0 and ΔAUC > 0 in ≥ 4 of 5 years): 6 primary single features.
  They are the four peer-level features above plus `idio_vol_30d` and `idio_share_30d`.
* **Useful to Breakwater** (robust beat plus a product-metric CI > 0): **none**.
* **Complete-case sensitivity** (5,090 OOF events, every primary feature present): the same
  picture. No family and not the combined model beats the incumbent.

## Answers

**1. Did any new free feature robustly beat `structural + vol_30d`?**
Statistically yes, practically no. Six single features pass the pre-registered
"robust beat" rule. The largest is the recency-weighted sector peer reaction level, at
+0.006 AUC. That is about half the size of the vol_30d gain the user already called
"modest", and 1–2 of these six passes are expected by chance.

None of them improves any product metric with a CI above zero. Every family model and the
combined model is flat or worse, because correlated features dilute and add variance.

**2. Which feature or family had the largest genuinely incremental event-specific signal?**
The **sector peer earnings reaction level** (family 5). In a model with the incumbent it
raises within-stock pooled OOF AUC from 0.481 to 0.511. Its univariate within-stock-year
AUC is 0.533, the highest of any feature here; vol_30d is 0.518.

The pre-registered *peer shock* (reactions relative to each peer's own history) is null:
AUC 0.51, above 0.5 in only 3 of 7 years, ΔAUC −0.0015. A post-hoc decomposition
(`posthoc.py`, labelled as such) splits the level into two parts:

* **Peer phenotype** (what these peers normally move): +0.0014, not significant.
* **Realized reactions on top of that**: +0.0023 [−0.0003, +0.0048], borderline.

The level also still adds +0.0033 [+0.0003, +0.0062] beyond Phase 4's all-history
`sector_prior_extreme_rate`. Conclusion: roughly half phenotype and half a weak real
"violent earnings season in this sector" effect. The seasonal half is not statistically
secure on its own.

**3. Did any feature improve the top of the ranking?** No. Top-10% capture moves by at most
+0.5 points with a CI spanning zero for every model. Top-20% capture never improves
significantly. The best point estimate, family-5 sub-sector only at +1.0 point, has a CI
of [−0.003, …]. Every AUC gain here comes from the middle of the ranking.

**4. Do peer earnings reactions predict subsequent peer earnings?** Weakly, and only as a
level:

* Within the top structural tercile, events whose sector peers moved most in the prior 20
  sessions went extreme 33.7% of the time vs 21.6% for the quietest tercile
  (`peer_conditional_rates.csv`). Part of that gap is sector phenotype that coarse
  terciles don't remove.
* In the walk-forward, the incremental effect is +0.003 to +0.006 AUC.
* The "shock" framing, where a peer moving more than *its own* normal predicts the next
  reporter, is not supported.
* A market-wide control (all *other* sectors' reactions) adds nothing (−0.0007), so the
  signal is sector-local rather than an overall earnings-season regime.

**5. Does idiosyncratic volatility beat ordinary vol_30d?** Marginally at best:

* As a **replacement**, structural + idio_vol_30d scores 0.7103 vs 0.7097, a difference of
  +0.0006 [−0.0022, +0.0039]. Not different.
* As an **addition**, +0.0011 [+0.0005, +0.0017]: tiny but consistent.
* The own-history version `idio_vol_30d_z_own` *hurts* (−0.0022), even though its
  univariate within-stock AUC (0.548) matches vol_30d_z_own.

**6. Is sub-sector information more useful than sector?** No. Sector-level peer features
beat sub-sector ones on every reading: +0.0055 vs +0.0032 ΔAUC, 97% vs 65% coverage.
Sub-sectors are too thin (median 3 tickers, 57 of 127 have ≤ 2). Sub-sector peer
*volatility* (family 6) is actively harmful: −0.0054, CI entirely below zero. On top of
sector_vol_30d, sub-sector peer earnings add +0.0006 [−0.0036, +0.0051].

**7. Are gains stable across years?** Only for the single small features, at 4 of 5 years
each. Even the best, peer_sec_rw_mean_abs, lost 0.006 in 2022, and its 2025 gain (+0.010)
is its largest. The family and combined models are negative in 2021–2023 and positive
only in 2024–2025, so they are regime-dependent.

**8. Is there enough signal left in prices, calendar and classification to justify more
engineering?** No. Seven economically distinct families, built causally and pre-registered,
produced these results:

* Two small, statistically real single-feature effects (sector peer reaction level
  +0.006; idio vol +0.001).
* Nothing at the top of the ranking.
* Nothing that survives being combined.

The jump, acceleration, residual-move and decoupling families are null or negative. This
repeats Phase 5's finding that raw momentum and drift fail. The price panel has been mined
from enough independent angles that further variants would be multiple-comparison fishing.

**9. Is historical options/analyst data the rational next acquisition?** Yes. The
remaining event-specific signal in free data is about 0.006 AUC and invisible at the top
decile. The Phase 3 timing correction was worth about +0.054, roughly nine times more.
Implied move (the market's own forecast of this event's size) and analyst dispersion or
revisions are the obvious untested sources of genuinely event-specific information.

Before buying, one data check is needed: confirm that point-in-time history exists
(snapshot timestamps before each announcement), not latest-vintage.
`testing/score_validity_and_deviation_features.py` already documents the vintage problem
in the Benzinga estimates.

If anything from this phase is kept, it should be a single feature,
`peer_sec_rw_mean_abs_1d_20`, re-validated as the lone pre-specified addition in the next
phase. It should not become a new product score.

## Limitations

* Sector/sub-sector are today's classification, constant per ticker: **not point-in-time**.
* The univariate orientation in `feature_diagnostics.csv` is chosen on the whole cohort.
  It is diagnostic only and no model uses it.
* The incumbent's `vol_30d` is read on the stored `earnings_date` row. For 14 BMO events
  in the window the vendor announcement is one session earlier, so their vol_30d includes
  the reaction day. The leak-free control `vol_30d_cut` scores identically (ΔAUC −0.0000),
  so this has no material effect.
* The interaction gate failed, as pre-registered. Neither family 5 as a whole nor
  `idio_vol_30d_z_own` was incremental, so the interaction was **not run**.
* `posthoc.py` was written after the results were read and is labelled as post hoc
  throughout.

## POST HOC — the one two-feature combination (`pair.py`)

Declared before running, chosen from the results above (so not independent confirmation):
`structural + vol_30d + peer_sec_rw_mean_abs_1d_20 + idio_vol_30d`. Same sample, same
walk-forward (8,410 OOF events), same bootstrap. Output: `posthoc_pair.{csv,json}`.

| comparison | ΔAUC [95% CI] | Δ top-10% capture [CI] | Δ top-20% capture [CI] | years ΔAUC>0 |
|---|---|---|---|---|
| pair − incumbent | **+0.0070 [+0.0027, +0.0116]** | +0.009 [−0.006, +0.015] | −0.001 [−0.011, +0.016] | 4/5 (2022 −0.006) |
| pair − (inc + peer) | +0.0010 [+0.0004, +0.0017] | +0.004 [−0.003, +0.007] | +0.002 [−0.009, +0.006] | 4/5 |
| pair − (inc + idio) | +0.0059 [+0.0018, +0.0103] | +0.006 [−0.006, +0.012] | −0.001 [−0.011, +0.014] | 4/5 |

Pair AUC 0.7167 vs 0.7097; top-10% hit rate 0.433 vs 0.417; Brier −0.0009 (CI below 0).
The features are only mildly correlated (Spearman 0.23), so their gains add: ≈ 0.006 + 0.001.
Statistically incremental and stable; **not product-relevant** (no capture CI above zero).
Nearly half the gain is the single year 2025 (+0.014). The conclusion is unchanged.
