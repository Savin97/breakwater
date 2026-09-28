# Phase 5B — pre-registration

Written **before** any Phase 5B feature was evaluated against the outcome. Everything
below is fixed; nothing in this file is to be edited after results are read. Choices made
after reading results must go in the results write-up, labelled as post hoc.

Inputs inspected before writing this file (none of them performance): the Phase 3/4/5 code
and outputs, the Phase 3 event frame's columns, the price panel's extent (2000-01-03 →
2026-09-18, 503 tickers, 6,718 sessions), the distribution of daily returns (max |r| 4.7),
sub-sector sizes (127 sub-sectors, median 3 tickers, 57 with ≤ 2), and the count of events
whose vendor announcement date precedes the stored `earnings_date` (132 with a target,
14 of them BMO in 2019–2025).

## Question

Can price/peer features built only from data Breakwater already holds improve prediction
of THIS event's corrected tail outcome beyond `structural + vol_30d` (Phase 5 incumbent,
walk-forward AUC ≈ 0.7097)? `structural` (`long_mean_abs_reaction`) alone is kept for
attribution.

## Target, cohort, protocol (all reused from Phases 3–5, unchanged)

* Target `y_extreme = abs_reaction_3d_anchored ≥ 0.08` (Phase 3 corrected target).
* Resolved window 2019-01-01 → 2025-12-31; headline cohort `mature_28`.
* **Common sample** = Phase 5's common sample (mature-28 rows with every Phase 5
  headline-eligible feature non-null), intersected with rows where every *own-stock
  price-derived* primary feature (families 1–4, 7 and `stock_vs_sub_vol_30d`'s
  numerator) is non-null. Peer aggregates (families 5, 6) are allowed to be missing and
  are median-imputed inside each training fold; the peer count (`n`) is never missing and
  carries the absence information. A complete-case sensitivity is also reported.
* Walk-forward: Phase 5 `walk_forward_predictions` — expanding years, test 2021–2025,
  L2 logistic regression C = 1.0, median imputation and scaling fitted on the training
  block only. No feature selection inside or outside folds.
* Bootstrap: 500 reps, resampling **stocks**, fixed seed, deltas vs the incumbent for
  ROC-AUC, PR-AUC, Brier, top-decile hit rate, top-decile capture, top-20% capture.

## Feature cutoff

For an event with stored `earnings_date` E and Phase 3 proxy session date P (the
announcement's session under Phase 3's clock, when a timestamp exists):
`D = min(E, P)`; the cutoff session `c` is the **last market session strictly before D**
on `market_session_grid`. Every price-derived value uses returns through close(c) only.
For BMO, c is the anchor session (D−1); for AMC, c = D−1, one session more conservative
than the anchor. The day-D return never enters any feature, whatever the window.

Returns are close-to-close on the **market session grid**: `r_t = P_t / P_{t−1} − 1`
only when the ticker has a price on both sessions; a missing session yields NaN, never a
multi-session return.

## Benchmarks (families 1, 4, 7)

* Issuer key collapses share-class twins: GOOG≡GOOGL, FOX≡FOXA, NWS≡NWSA. "Leave one out"
  always leaves out the whole issuer.
* `MKT_{-i}(t)`: equal-weight mean of all other issuers' returns at t.
  `SEC_{-i}(t)`: same within the stock's sector. For benchmark aggregation only, each
  constituent return is clipped to ±50% (one glitch must not move everyone's benchmark).
* Exposure: OLS over the 252 sessions ending at c (≥ 126 complete obs),
  `r = a + b1·MKT_{-i} + b2·(SEC_{-i} − MKT_{-i}) + e`. Residuals `e` are evaluated in that
  window with those coefficients.

## Primary features (29) — fixed

| family | feature | definition |
|---|---|---|
| 1 idio vol | `idio_vol_30d` | std(e) over last 30 sessions (≥ 20 obs) |
| 1 | `idio_vol_10d` | std(e) over last 10 sessions (≥ 8) |
| 1 | `idio_share_30d` | var(e)/var(r) on the same 30-session rows |
| 1 | `idio_vol_30d_z_own` | `idio_vol_30d` z-scored against the stock's own PRIOR event rows (expanding mean, expanding std ≥ 5 prior), exactly Phase 5's `_z_own` construction |
| 2 jumps | `max_abs_ret_20d` | max \|r\| over last 20 sessions (≥ 15) |
| 2 | `jump_share_20d` | max r² / Σ r² over last 20 |
| 2 | `semivar_balance_20d` | (Σ r²·[r<0] − Σ r²·[r>0]) / Σ r² over last 20 |
| 3 accel | `vol_log_ratio_5_30` | log(std r last 5 (≥4) / std r last 30 (≥20)) |
| 3 | `vol_log_ratio_10_60` | log(std r last 10 (≥8) / std r last 60 (≥45)) |
| 4 residual move | `abs_idio_ret_5d` | \|mean(e) × 5\| over last 5 (≥ 4) |
| 4 | `abs_idio_ret_20d` | \|mean(e) × 20\| over last 20 (≥ 15) |
| 5 peer shocks (L = `sub`, `sec`) | `peer_L_n_20` | usable distinct-issuer peer events in window |
| 5 | `peer_L_mean_abs_1d_20` | mean \|reaction_1d_anchored\| of usable peers |
| 5 | `peer_L_max_abs_1d_20` | max of the same |
| 5 | `peer_L_frac_large_1d_20` | share with \|r1d\| ≥ 0.05 (`LARGE_EARNINGS_REACTION_THRESHOLD`) |
| 5 | `peer_L_rw_mean_abs_1d_20` | weighted mean, weight 0.5^(age/10), age = c − endpoint session |
| 5 | `peer_L_shock_20` | mean of log((\|r1d\|+0.005)/(m_j+0.005)); m_j = peer's own mean \|r1d\| over its ≥ 4 PRIOR resolved events |
| 6 peer vol | `sub_peer_med_vol_10d` | median over LOO sub-sector peers of their 10-session std at c (≥ 2 peers) |
| 6 | `sub_peer_med_vol_30d` | same, 30 sessions |
| 6 | `stock_vs_sub_vol_30d` | log(own 30-session std / `sub_peer_med_vol_30d`) |
| 6 | `sub_peer_frac_high_vol` | share of LOO sub-sector peers whose 30-session std at c exceeds their own 80th percentile over the trailing 252 sessions (≥ 126) |
| 7 decoupling | `sector_corr_60d` | corr(r, SEC_{-i}) over last 60 (≥ 45) |
| 7 | `sector_corr_change_20_120` | corr over last 20 (≥ 15) minus corr over last 120 (≥ 90) |

Peer usability (family 5): a peer event j (different issuer, same sub-sector/sector,
`reaction_1d_anchored_status == available`) is usable at event e only if its 1-session
endpoint session `anchor_j + 1` satisfies `c_e − 19 ≤ endpoint_j ≤ c_e`. Having an earlier
`earnings_date` is not sufficient. Unresolved peer outcomes are excluded, never zero;
with no usable peer, every aggregate except `n` is NaN. Twins are counted once per
(issuer, anchor date). Classification is today's `sector`/`sub_sector` from
`stock_data`, constant per ticker in the data — **not point-in-time**; this is a known
limitation, not corrected here.

## Secondary (sensitivity / diagnostics — never selected into a primary model)

`signed_idio_ret_20d`, `std_abs_idio_ret_20d` (= `abs_idio_ret_20d` / (`idio_vol_30d`·√20)),
`max_abs_idio_ret_20d`, `vol_change_20_20` (log std last 20 / std of the 20 before),
`vol_30d_cut` (plain 30-session std at the Phase 5B cutoff — leak control for `vol_30d`),
peer windows 10 and 40 for `mean_abs_1d` and `shock`, `peer_L_frac_extreme_1d_20`
(threshold 0.08), `peer_local_*` (sub-sector when ≥ 2 usable sub-sector peers else
sector), `peer_mkt_exsec_mean_abs_1d_20` (all other sectors — market-wide earnings
environment control), `peer_L_n_unresolved_20` (diagnostic), sector-level LOO peer vol
(`sec_peer_med_vol_30d`).

## Models

A `structural`; B `structural + vol_30d` (**incumbent**); B′ `structural + idio_vol_30d`
(replacement); C incumbent + each primary feature; D incumbent + each family (family 5
also split into `sub` and `sec`); E incumbent + all 29 primary. References:
incumbent + `sector_vol_30d` and incumbent + `sector_vol_30d` + family 5 `sub`.

## Decision rules (fixed)

* **Statistically incremental**: stock-clustered 95% CI of ΔAUC vs incumbent has lo > 0.
* **Product-relevant**: CI lo > 0 for Δ top-decile capture **or** Δ top-20% capture.
* **Stable**: ΔAUC > 0 in ≥ 4 of the 5 test years.
* **Robust beat** = statistically incremental AND stable; "useful to Breakwater" additionally
  requires product-relevant.

## The one interaction (gated)

Run only if BOTH (i) family 5 (all primary peer features) and (ii) `idio_vol_30d_z_own`
individually are statistically incremental over the incumbent. Term:
`z(idio_vol_30d_z_own) × z(peer_sec_shock_20)`, both train-fold imputed and standardised,
added to incumbent + both main effects. Sector level is used for the peer term because
sub-sector coverage is structurally thin (median 3 tickers per sub-sector) — a coverage
argument, not a performance one. If the gate fails, the interaction is not run.
