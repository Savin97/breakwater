# Historical options pilot — results

Research only. No production module, score, threshold, report, dashboard, prediction
archive or `MODEL_VERSION` was touched. Rules were fixed in `PREREGISTRATION.md` (plus
Amendments 1 and 2) before any outcome was read. Nothing below was chosen after seeing a
result unless it says **POST HOC**.

```bash
PYTHONPATH=. .venv/bin/python -m research.options_pilot.build            # match + features (cached)
PYTHONPATH=. .venv/bin/python -m research.options_pilot.build --vol-history
PYTHONPATH=. .venv/bin/python -m research.options_pilot.calendar_check   # snapshot calendar
PYTHONPATH=. .venv/bin/python -m research.options_pilot.coverage         # Phase A tables
PYTHONPATH=. .venv/bin/python -m research.options_pilot.evaluate         # pre-2026, ~10 min
PYTHONPATH=. .venv/bin/python -m research.options_pilot.evaluate --holdout   # 2026, once
.venv/bin/python -m pytest testing/test_options_pilot.py -q
```

Source: DoltHub `post-no-preference/options`, every query `AS OF` commit
`1ug5hqta1o786faoh8fv89q7grrd00jj` (master, 2026-09-29 06:34:57 UTC). License CC BY-SA 4.0.
Raw query results are cached in gitignored `data/vendor/dolthub_options/`. Tables in
`output/options_pilot/`: `event_options.parquet` (the per-event audit file),
`coverage_*.csv`, `calendar_*.csv`, `identity_audit.parquet`, `composition.csv`,
`sector_mix.csv`, `metrics_pooled.csv`, `metrics_yearly.csv`, `deltas_bootstrap.csv`,
`summary.json`, `winner.json`, `oof_{strict,medium,broad}.parquet`, `holdout_2026.json`.

## Conclusion

```
FREE OPTIONS PILOT SHOWS WEAK / UNCERTAIN VALUE
```

Expected move from the free chains adds to Model C, and it's the first new input in any
phase to move top-20% capture with a CI above zero. Measured against the pre-registered
rules:

* ✔ AUC +0.013 [+0.008, +0.020] and top-20% capture +1.8 points [+0.5, +3.6] (hit rate
  35.9% → 37.5%) on 8,241 identical events, 2021–2025.
* ✔ The gain holds at every staleness limit and under the live collector's expiry rule.
* ✔ 2026 has the same sign on every headline metric.
* ✘ The pre-registered year-stability test: AUC is up in only 3 of 5 years.
* ✘ Top-10% capture (+0.7 pt) and the 2026 deltas all have CIs that cross zero.
* ✘ The gain is mostly BMO; AMC top-20% capture is flat.

So it's real but small, about the size of what `vol_30d` added over B in Phase 3. It's
concentrated in the 20% band rather than the top 10%, and its year-to-year spread is
wide. Worth a properly sized test on professional data (§9); not enough to build on from
this source.

## 1. Is the source usable?

Yes, for a research question. Not as a production source.

| | |
|---|---|
| Tables | `option_chain` (bid, ask, IV, delta, gamma, theta, vega, rho per contract), `volatility_history` (IV/HV summary per symbol and date) |
| Dates | 2019-02-09 → 2026-09-28, 1,291 `volatility_history` dates |
| Cadence | 2019 weekly, **Saturday-dated** (Friday's quotes, verified byte-identical); 2020–mid-2024 Mon/Wed/Fri; mid-2024 onward daily. Each date committed ~06:34 UTC the next morning |
| Chains | near the money, median 84 contracts and 3 expirations per stock and date; nearest covering expiry a median 14 days after the snapshot |
| Symbols | **estimate**: ~530 (Feb 2019) → ~1,050 (mid-2019) → 1,520–1,640 from 2020, from `volatility_history` row counts per date. An exact `option_chain` count is not obtainable: every aggregate over that table times out on the public API. **Verified**: 476 of our 477 population stocks appear (NVR never does) |
| Provenance | unknown — the repo has a license and no README; the quote provider, capture time and IV/greek model are undocumented |

**Snapshot calendar (item 1A).** The matching assumed `option_chain` dates = `volatility_history`
dates. Checked two ways (`calendar_check.py`):

* Every chain date inside every event's matching window was enumerated by primary-key
  seeks: 677 dates in 56 windows. **6 were missing from the assumed calendar**, all in
  2023-11-16 → 11-24 (`volatility_history` has a gap there, `option_chain` does not).
* Broad sample, 8 calendar and 12 off-calendar weekdays per year 2019–2026: 64/64 calendar
  dates have chains; 2/77 off-calendar weekdays do — the same two November 2023 dates. On
  the sampled calendar dates AAPL, JPM, KO, ALLE, AIZ and CPB (liquid to thin) were present
  100% of the time.

Fix: the calendar is now the union of both lists. **22 events** were in affected windows; 20
changed snapshot (4–8 sessions stale → same day), all 2023-11. Phase A was regenerated.
`snapshot_date ≤ call_cutoff_date` holds on every row before and after.

**Quote quality — on the matched-chain slice only** (the 13,042 chains actually used,
1,192,806 contracts; not the upstream database as a whole): 0 duplicate keys, 0 missing
bid/ask, 0 negative quotes, 86 ask < bid, 745 zero asks, 6,550 IV ≤ 0 or > 5 (0.55%),
0 implausible deltas. Selected ATM pairs: strike a median 0.5% from its own parity spot
(p95 2.4%), call Δ + |put Δ| within 0.026 of 1 at p95, relative straddle spread median 12.5%
(p95 60%). Amendment 1 removed 5 pairs whose greeks were computed off a wrong spot.

## 2. How many Breakwater events were matched?

Population = Phase 3's (completed, BMO/AMC, corrected target with resolved endpoint, ≥ 8
prior outcomes, C inputs present), report year ≥ 2019: **14,035 events, 477 stocks**.

| year | events | same-day | ≤ 1 session | ≤ 3 | ≤ 5 | none ≤ 5 |
|---|---|---|---|---|---|---|
| 2019 | 1,738 | 0.5% | 0.5% | 0.5% | 72.1% | 27.9% |
| 2020 | 1,769 | 91.2% | 92.6% | 92.9% | 93.1% | 6.9% |
| 2021 | 1,803 | 91.1% | 91.6% | 93.8% | 93.8% | 6.2% |
| 2022 | 1,832 | 72.9% | 76.1% | 94.4% | 94.4% | 5.6% |
| 2023 | 1,853 | 94.3% | 94.8% | 95.0% | 95.0% | 5.0% |
| 2024 | 1,871 | 92.6% | 92.7% | 92.8% | 96.7% | 3.3% |
| 2025 | 1,830 | 98.6% | 98.7% | 98.7% | 98.7% | 1.3% |
| 2026 | 1,339 | 90.9% | 99.7% | 99.8% | 99.8% | 0.2% |
| **all** | **14,035** | **79.1%** | **80.7%** | **83.4%** | **92.9%** | **7.1%** |

2019 is ~5 sessions stale by construction: a Saturday snapshot is never used for a Friday
cutoff (the literal date rule), so 2019 gets the previous week's quotes. 2022's dip is four
weeks where the source skipped a Friday. BMO 94.2% / AMC 91.1% matched within 5 sessions.

**Expiry availability (item 1B), matched events** (`coverage_expiry.csv`):

| | matched | ≥ 1 covering expiry | usable pair on nearest | ≥ 2 usable covering | 0 usable covering | second usable later expiry (term structure) |
|---|---|---|---|---|---|---|
| all | 13,042 | 13,042 | 12,877 | 12,486 | 53 | 12,480 |

Per year the pattern is the same; 2019 is thinnest (967 of 1,253 with a second usable
expiry). Term structure is therefore evaluated on 99.3% of the main-tier covered rows,
and that block says so.

**Identity.** Class shares are dot-spelled in the source (`BRK.B`); symbols are
point-in-time. Seventeen renames are mapped with switch dates read off the source itself
(FB→META, ANTM→ELV, UTX→RTX, …); the source only picks up a new symbol days to months
after a rename, and those gaps stay missing. Inherited from Phase 3 and absent from this
population: APP, BF-B, BRK-B, CBOE, CPAY, CPB, DELL, DOC, DOW, ES, FISV, GE, GEN, HON, IR,
LIN, MAR, ORCL, SMCI, SNDK, WBD (21 identity hazards). Price continuity (parity spot ÷
adjusted close): 55 clean-split jumps; the other 12 are all spin-offs (T/WarnerMedia,
MMM/Solventum, LH/Fortrea, WDC/Sandisk, FTV ×2, DD/Qnity, J/Amentum, HWM/Arconic, DTE,
BDX/Waters) plus CMG 50:1. No identity errors found.

| names | events | usable ≤ 1 session | usable ≤ 5 |
|---|---|---|---|
| GOOG / GOOGL | 31 / 31 | 83.9% / 83.9% | 96.8% / 96.8% |
| FOX / FOXA | 22 / 22 | 54.5% / 90.9% | 59.1% / 95.5% |
| NWS / NWSA | 31 / 31 | 61.3% / 61.3% | 71.0% / 74.2% |
| 17 inactive names (AVB, BK, BLDR, CAG, CTRA, DAY, EA, EPAM, EQR, HOLX, LW, MOH, MTCH, PAYC, POOL, TAP, TTD) | 482 | 77.0% | 85.5% |

## 3. Is there meaningful coverage bias?

Main tier (≤ 1 session), test years 2021–2025 (`composition.csv`, `sector_mix.csv`):

| | events | stocks | base rate | BMO share | top-10 stocks' share | inactive-name share |
|---|---|---|---|---|---|---|
| all eligible | 9,189 | 473 | 0.191 | 57.2% | 2.2% | 3.5% |
| covered | 8,241 | 470 | 0.184 | 57.9% | 2.5% | 3.5% |
| not covered | 948 | 356 | 0.255 | 51.5% | 18.0% | 3.9% |

Uncovered events are 48% from 2022 (the skipped Fridays) and concentrated in names the
source added late (ARES, CVNA, XYZ, TTD, APO, WDAY, TPL, KKR, BX). Sector mix of covered vs
all differs by ≤ 0.7 points in every sector. No market-cap or liquidity field is available
without a new data task, so none is compared.

Model C as Phase 3 fitted it (full Benzinga history, its own out-of-fold predictions):

| C (Phase 3 fit) | n | base | AUC | top-10% capture | top-20% capture |
|---|---|---|---|---|---|
| all events 2021–2025 | 9,068 | 0.189 | 0.719 | 0.240 | 0.407 |
| covered | 8,128 | 0.181 | 0.706 | 0.225 | 0.386 |
| not covered | 940 | 0.255 | 0.788 | 0.254 | 0.479 |

**The covered sample is harder for C, not easier.** Every C-vs-options comparison below is
on identical rows, so selection cannot create a delta; it can only make the covered
numbers lower than the full-population ones.

## 4–6. Walk-forward results (main tier: snapshot ≤ 1 session; test years 2021–2025)

Main tier chosen by the pre-registered count rule (8,241 OOF events = 80% of the ≤ 5 tier).
Training starts 2020 (2019 has no ≤ 1 snapshot). Logistic, standardised on the training
fold, L2 C = 1, refit yearly on events whose outcome endpoint precedes 1 January.

**Primary block — identical 8,241 events, 470 stocks:**

| model | AUC | PR-AUC | Brier | log loss | top-10% hit / capture / lift | top-20% hit / capture / lift | BMO AUC | AMC AUC |
|---|---|---|---|---|---|---|---|---|
| C | 0.707 | 0.334 | 0.1390 | 0.4395 | 0.412 / 0.224 / 2.24x | 0.359 / 0.391 / 1.95x | 0.680 | 0.725 |
| **C + expected move** | **0.720** | 0.350 | 0.1373 | 0.4339 | 0.426 / 0.232 / 2.31x | **0.375 / 0.409 / 2.04x** | 0.699 | 0.732 |
| C + ATM IV | 0.718 | 0.342 | 0.1378 | 0.4348 | 0.416 / 0.226 / 2.26x | 0.367 / 0.399 / 2.00x | 0.705 | 0.723 |
| C + EM + ATM IV | 0.723 | 0.349 | 0.1372 | 0.4329 | 0.427 / 0.232 / 2.32x | 0.376 / 0.409 / 2.05x | 0.709 | 0.729 |

**Deltas vs C, 95% stock-clustered bootstrap (500 reps, paired on identical rows):**

| vs C | ΔAUC | Δ top-10% hit | Δ top-10% capture | Δ top-20% hit | Δ top-20% capture |
|---|---|---|---|---|---|
| + expected move | **+0.013 [+0.008, +0.020]** | +1.3 pt [−0.7, +3.6] | +0.7 pt [−0.4, +2.0] | **+1.6 pt [+0.5, +3.2]** | **+1.8 pt [+0.5, +3.6]** |
| + ATM IV | **+0.011 [+0.005, +0.020]** | +0.4 pt [−1.5, +3.2] | +0.2 pt [−0.8, +1.7] | +0.8 pt [−0.8, +2.6] | +0.9 pt [−0.8, +2.9] |
| + EM + IV | **+0.016 [+0.009, +0.024]** | +1.5 pt [−0.7, +4.1] | +0.8 pt [−0.4, +2.2] | **+1.7 pt [+0.1, +3.6]** | **+1.9 pt [+0.1, +3.9]** |
| EM + IV vs EM alone | +0.003 [−0.001, +0.008] | +0.1 pt | +0.1 pt [−0.8, +1.1] | +0.1 pt | +0.1 pt [−1.3, +1.3] |

**Secondary features, each on its own rows (C refit there):**

| feature | rows (share of covered) | vs C: ΔAUC | vs C: Δ top-20% capture | added to C+EM+IV: ΔAUC | added to C+EM+IV: Δ top-20% capture |
|---|---|---|---|---|---|
| put skew | 8,064 (97.7%) | −0.002 [−0.004, +0.001] | −0.4 pt [−1.3, +0.4] | −0.001 [−0.002, +0.001] | −0.3 pt [−0.9, +0.5] |
| term structure | 8,211 (99.3%) | +0.003 [+0.000, +0.006] | −0.3 pt [−1.1, +1.2] | −0.000 [−0.001, +0.000] | −0.1 pt [−0.5, +0.6] |
| IV vs own history | 8,143 (98.8%) | **+0.007 [+0.004, +0.011]** | **+1.5 pt [+0.2, +3.2]** | **−0.003 [−0.005, −0.002]** | −0.3 pt [−1.3, +0.5] |
| compact (all five) | 7,941 (96.4%) | +0.011 [+0.004, +0.018] | +1.5 pt [−0.7, +3.3] | vs C+EM+IV: **−0.005 [−0.007, −0.002]** | −0.5 pt [−1.3, +0.6] |

IV-relative-history passed its build gate (Amendment 2 §4: `volatility_history` row on the
snapshot date for 99.6% of covered events; Spearman 0.96 with the chain's later-expiry ATM
IV). Alone it helps; on top of expected move and ATM IV it hurts. Skew is null or harmful.
Term structure is null. The compact set is **worse** than EM + IV by itself.

**Answers:**

* **4. Expected move improves C.** AUC +0.013 and top-20% capture +1.8 points, both with CIs
  clear of zero. Top-10% capture +0.7 points, CI crosses zero.
* **5. ATM IV** adds about as much AUC (+0.011) but nothing significant at the top. Once
  expected move is in, IV adds nothing (+0.003 AUC, CI crosses zero).
* **6. Skew, term structure, relative IV:** nothing beyond expected move + IV. Relative IV
  duplicates what expected move already carries.

## 7. Stable across years and BMO/AMC?

C vs C + expected move, main tier:

| slice | n | base | C AUC | ΔAUC | C top-10% capture | Δ | C top-20% capture | Δ |
|---|---|---|---|---|---|---|---|---|
| 2021 | 1,631 | 0.122 | 0.746 | −0.000 | 0.271 | +1.0 pt | 0.437 | **+6.5 pt** |
| 2022 | 1,381 | 0.196 | 0.656 | −0.004 | 0.219 | −1.1 pt | 0.356 | +0.7 pt |
| 2023 | 1,720 | 0.172 | 0.709 | +0.020 | 0.253 | +2.4 pt | 0.409 | +0.7 pt |
| 2024 | 1,719 | 0.213 | 0.737 | +0.008 | 0.218 | −0.5 pt | 0.387 | +3.8 pt |
| 2025 | 1,790 | 0.214 | 0.687 | +0.013 | 0.185 | +1.0 pt | 0.342 | +2.1 pt |
| BMO | 4,772 | 0.162 | 0.680 | **+0.020** | 0.205 | +1.9 pt | 0.381 | +1.6 pt |
| AMC | 3,469 | 0.214 | 0.725 | +0.008 | 0.213 | +0.9 pt | 0.401 | −0.5 pt |

* Top-20% capture is positive in **5 of 5** years, but more than a third of the pooled gain
  is 2021 alone (+6.5 pt); 2022 and 2023 are +0.7.
* AUC is positive in **3 of 5** (2021 flat, 2022 slightly negative).
* Top-10% capture is positive in 3 of 5.
* The gain is mostly **BMO**: +0.020 AUC, versus +0.008 for AMC, where top-20% capture is
  slightly negative. This fits the one known weakness of C (BMO AUC 0.68 vs AMC 0.73).

**Staleness sensitivity** (the same comparison under each tier; the main result is not
chosen by it):

| tier | OOF events | ΔAUC C+EM | Δ top-20% capture C+EM | ΔAUC C+EM+IV | Δ top-20% capture C+EM+IV |
|---|---|---|---|---|---|
| ≤ 1 session (main) | 8,241 | +0.013 [+0.008, +0.020] | +1.8 pt [+0.5, +3.6] | +0.016 [+0.009, +0.024] | +1.9 pt [+0.1, +3.9] |
| ≤ 3 | 8,621 | +0.013 [+0.008, +0.019] | +2.0 pt [+0.6, +3.8] | +0.016 [+0.010, +0.024] | +2.0 pt [+0.4, +3.8] |
| ≤ 5 (adds 2020 as a test year) | 10,318 | +0.013 [+0.008, +0.019] | +1.8 pt [+0.4, +3.4] | +0.014 [+0.007, +0.021] | +1.4 pt [+0.0, +3.1] |

**Expiry-rule robustness** (the live collector's strictly-after-report-date rule; 141
covered rows change expiry): C+EM ΔAUC +0.013 [+0.008, +0.020], Δ top-20% capture
+1.7 pt [+0.3, +3.6]. The result does not depend on the BMO same-day-expiry choice.

## Frozen before 2026 was opened

Applied Amendment 2 §7 to the main tier's pre-2026 results (`winner.json`, written by
`evaluate.py` before `--holdout` ran):

| candidate | ΔAUC CI > 0 | a capture CI > 0 | mean Δ capture (top 10/20) |
|---|---|---|---|
| C + EM | yes | yes (top 20%) | +1.25 pt |
| C + IV | yes | no | +0.53 pt |
| C + EM + IV | yes | yes (top 20%) | +1.32 pt |
| C + compact | yes | no | +0.98 pt |

C + EM and C + EM + IV qualify; their mean capture gains are within 0.002, so the rule takes
the one with fewer inputs.

**Winner: C + expected move.** Inputs `log hist_mean_abs`, `log vol_30d`,
`log expected_move`; expected move = (C_mid + P_mid) / K of the pre-registered ATM pair on
the first announcement-aware covering expiry of the latest snapshot ≤ the call cutoff, at
most 1 session old; rows lacking an input are not scored (no imputation); logistic,
standardised on the training fold, L2 C = 1, trained on every covered event with an
outcome endpoint before 2026-01-01.

Against the pre-registered conclusion rules, before 2026: the capture CI (top 20%) and the
AUC CI are above zero ✔. Top-20% capture is positive in 5 of 5 years ✔, but AUC in only
3 of 5 ✘. 2026 not yet seen.

## 8. Does 2026 confirm it? (scored once, after the freeze above)

Models trained on every covered event with an outcome endpoint before 2026-01-01; 2026
YTD, snapshot ≤ 1 session: **1,326 events, base rate 0.247** (higher than any training
year).

| model | AUC | PR-AUC | log loss | top-10% hit / capture / lift | top-20% hit / capture / lift | BMO AUC | AMC AUC |
|---|---|---|---|---|---|---|---|
| C | 0.719 | 0.419 | 0.509 | 0.519 / 0.211 / 2.10x | 0.462 / 0.376 / 1.88x | 0.689 | 0.744 |
| **C + EM (frozen winner)** | **0.729** | 0.440 | 0.499 | **0.556 / 0.226 / 2.26x** | 0.470 / 0.382 / 1.91x | 0.706 | 0.747 |
| C + IV | 0.726 | 0.427 | 0.503 | 0.511 / 0.208 / 2.07x | 0.462 / 0.376 / 1.88x | 0.705 | 0.741 |
| C + EM + IV | 0.731 | 0.435 | 0.499 | 0.504 / 0.205 / 2.04x | 0.481 / 0.391 / 1.95x | 0.711 | 0.744 |

| 2026, vs C | ΔAUC | Δ top-10% hit | Δ top-10% capture | Δ top-20% hit | Δ top-20% capture |
|---|---|---|---|---|---|
| **+ EM (winner)** | +0.010 [−0.004, +0.022] | +3.8 pt [−3.0, +10.5] | +1.5 pt [−1.3, +4.3] | +0.8 pt [−2.3, +5.0] | +0.6 pt [−1.9, +4.1] |
| + EM + IV | +0.012 [−0.001, +0.024] | −1.5 pt [−8.2, +7.5] | −0.6 pt [−3.4, +3.0] | +1.9 pt [−1.7, +6.0] | +1.5 pt [−1.3, +4.9] |
| + compact (1,299 rows) | +0.015 [+0.004, +0.027] | +0.8 pt | +0.3 pt [−3.2, +3.1] | +3.5 pt | +2.8 pt [−1.0, +5.3] |

* **The direction is confirmed.** The frozen winner beats C on AUC, PR-AUC, log loss and
  every top-of-ranking metric, and BMO again gains most (+0.017 AUC).
* **The size isn't.** Nine months of events can't resolve a 1–2 point capture difference:
  every capture CI spans roughly ±3–5 points.
* **One contradiction.** In 2026 the top-10% gain goes to EM alone (+1.5 pt) and the top-20%
  gain to EM + IV (+1.5 pt), the reverse of 2021–2025. Pre-2026 the two were within 0.002;
  2026 says the same thing, that they're indistinguishable.
* **POST HOC, not used for anything:** the compact set, worse than EM + IV before 2026, is
  the best of all in 2026 (ΔAUC CI above zero). With one year and this sample size it's
  noise-level either way.

## 9. Is the signal strong enough to justify paying for professional 2013–present data?

**Yes, for one well-defined test. Not on the expectation of a big jump.**

* **This is the first acquisition-worthy signal in the rebuild.** Phase 5B's 29 free
  features and Phase 3's vol/peer additions never moved top-10% or top-20% capture;
  expected move moves top-20% capture by ~1.8 points with a CI above zero.
* **It lands exactly where C is weakest.** BMO AUC 0.680 → 0.699; BMO is the half of the
  calendar the audit found broken.
* **The free source understates what options can do, for fixable reasons:**
  * The nearest covering expiry is a median 14 days out, so the "expected move" is two
    weeks of ordinary vol plus the event.
  * History starts in 2020, so 5 test years and CIs of ±1.5 points on a 1.8-point effect.
  * Quotes are at an unknown time of day, and greeks come from an undocumented model.
* **What professional data fixes:**
  * Daily history from ~2013 (ORATS was the lead candidate) roughly doubles the test years.
  * Weekly expiries and a vendor-computed implied earnings move isolate the event itself
    (Leung & Santoli 2014, in the brain notes).
  * Documented capture times.
* **Realistic expectation:** a gain of the same order, measured precisely, perhaps a few
  points of top-20% capture. That's not a new product. **Buy only if the price is modest
  relative to that,** and check point-in-time snapshots (not end-of-day recomputed
  history) before paying.
* **The test to pre-register on the professional data:** C vs C + implied earnings move
  on 2014–2025, the same call cutoff, the same rules. The bar to clear: top-20% capture
  CI above zero, positive in ≥ 70% of years, and a top-10% capture effect whose CI
  excludes zero.

## Caveats

* **Unknown provenance.** Nothing documents the free source's quote provider, capture
  time or IV/greek model. A few rows carry greeks off a wrong spot (Amendment 1).
* **Saturday-dated 2019 snapshots were not used for Friday cutoffs** (literal date rule).
  2019 appears only in the ≤ 5 tier, as training.
* **A non-pre-registered interpretive choice:** the "positive in ≥ 4 of 5 years" rule was
  read as applying to AUC and to the capture metric that qualified. Top-20% capture passes
  (5/5); AUC fails (3/5), and that failure is why this lands on WEAK rather than MATERIAL.
* **Model C here is refit on covered 2020+ rows,** not Phase 3's 2014+ fit. Phase 3's own C
  on the same covered rows scores 0.706 AUC vs 0.707 here, so the benchmark is equivalent.
* **Base rates move a lot by year** (0.122 in 2021 → 0.247 in 2026). Calibration drift
  affects C and C + EM alike, since they're compared on identical rows.
* **Sector is today's classification** (inherited from Phase 3). No size or liquidity
  proxy was compared.
* **One source limitation:** `volatility_history` has a gap on 2023-11-16..24, so the
  IV-history feature is missing for those 20 events.
