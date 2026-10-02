# Confirmatory options test — pre-registration

Written 2026-10-01, **before any professional options data has been acquired or seen**.
Nothing in production changes because of this document, whatever the result.

Follows `research/options_pilot/` (free DoltHub data, verdict **WEAK / UNCERTAIN VALUE**:
C + expected move improved top-20% capture +1.8 pt, CI above zero; top-10% +0.7 pt, CI
crossing zero; gain mostly BMO; 2026 same sign, not significant). This test is
confirmatory: **one feature, one comparison, one primary metric.** The pilot's sample is
not reused as evidence.

## Question

Using trustworthy point-in-time historical options quotes, does the options-implied
expected move available at Breakwater's weekly call cutoff improve **top-10% capture** of
corrected ≥ 8% earnings moves beyond Model C?

## Comparison

| model | inputs |
|---|---|
| **C** (fixed benchmark, not redesigned) | `log hist_mean_abs`, `log vol_30d` |
| **C + EM** | the same + `log expected_move_pct` |

Not tested: ATM IV, skew, term structure, IV-vs-history, combinations, XGBoost. The pilot
tested them and found nothing beyond EM.

## Population

Built independently from the professional source. Phase 3's corrected event frame
(`output/phase3_target_rebuild/phase3_events.parquet` via the Phase 3 feature frame):

* completed event, announcement window **BMO or AMC**;
* `abs_reaction_3d_anchored` available with its 3-session endpoint resolved;
* ≥ 8 prior corrected outcomes usable at the call cutoff, both C inputs present;
* an options snapshot passing the rules below.

Target `y = abs_reaction_3d_anchored ≥ 0.08`. The legacy `abs_reaction_3d` is never used.

Eligible events before options coverage (counted 2026-10-01): 2014 1,034 · 2015 1,585 ·
2016 1,629 · 2017 1,670 · 2018 1,695 · 2019 1,738 · 2020 1,769 · 2021 1,803 · 2022 1,832 ·
2023 1,853 · 2024 1,871 · 2025 1,830 · **2026 1,339 (holdout)**. 2013 has 50 — the ≥ 8
prior-outcome rule makes 2014 the practical start whatever the vendor offers.

## Clock and snapshot

* **Call cutoff** = Phase 3's `call_cutoff_date` (last session before the Monday of the
  report week). **Prediction cutoff timestamp** = that date at 16:00 America/New_York.
* **Snapshot** = the source's snapshot on the call-cutoff date with capture timestamp
  ≤ the prediction cutoff (a near-close snapshot such as 15:45/15:46 qualifies). If that
  session has none, the latest snapshot from the **one** previous session; otherwise the
  event is excluded and counted. Nothing is forward-filled.
* **Invariant, no exceptions:** `snapshot_ts ≤ prediction_cutoff_ts`, asserted on every
  row and tested.
* Every event row keeps: event ID, call cutoff, snapshot timestamp, snapshot age
  (sessions), expiry, strike, call bid/ask/mid, put bid/ask/mid, underlying spot and its
  source field, `expected_move_pct`, denominator used, vendor + product + file/request ID.

## Expected move (the one feature)

1. **Quote filters** per contract: bid and ask present; bid ≥ 0; ask > 0; ask ≥ bid.
   Removals counted per filter.
2. **Expiry**: the first expiration on/after the snapshot date that covers the
   announcement — after the report date: yes; on the report date: **BMO yes, AMC no**;
   before: never. (Same rule as the pilot, `research/options_pilot/features.py`.)
3. **Spot** `S` = the source's underlying price captured **at the same snapshot**
   (nominal, same day — never Breakwater's split/dividend-adjusted price).
4. **ATM pair**: the strike `K` with a valid call **and** put in that expiry minimising
   `|K − S|`; ties → smaller `|C_mid − P_mid|`, then lower strike. Must satisfy
   `|K / S − 1| ≤ 5%`, else no feature.
5. `expected_move_pct = (C_mid + P_mid) / S`, mids = (bid + ask) / 2 of the same rows.
   Model input: `log(expected_move_pct)`.

**Fallback, fixed now:** if the source has no contemporaneous spot for ≥ 5% of eligible
events, the whole experiment uses the pilot's chain-internal definition instead (ATM by
`|Δ_call − 0.5| + |Δ_put + 0.5|`, deltas in [0.30, 0.70], strike within 10% of the parity
spot `K + C_mid − P_mid`, `EM = (C_mid + P_mid) / K`). If spot is missing on < 5%, those
events are excluded. The choice is made from coverage counts before any outcome is joined,
and written into an amendment.

Deltas and IVs, if supplied, are kept for diagnostics only. The feature never needs them
(except under the fallback).

## Model and walk-forward

Phase 3's logistic, unchanged: standardise on the training fold, L2, C = 1, no tuning. For
test year Y, training = events whose outcome endpoint precedes 1 January Y; scaling and
coefficients come from training rows only. **First test year** = the first year whose
training block has ≥ 1,000 rows and ≥ 150 positives (expected 2015 if 2014 is covered,
2017 if the source starts in 2016). Test years run through 2025. C and C + EM are always
compared on **identical event IDs** (a hash is saved). Events without the feature are
excluded from both models; coverage bias is reported as in the pilot (covered vs
uncovered: counts, base rate, BMO/AMC, year, sector, ticker concentration, and Phase 3's C
on each).

## Metrics

**Primary:** Δ top-10% capture (C + EM − C), pooled over the test years, with a 95%
stock-clustered bootstrap CI (**2,000** paired reps).

Secondary, all with the same bootstrap: top-10% hit rate and lift; top-20% capture, hit
rate, lift; ROC-AUC; PR-AUC; Brier; log loss; each by test year. **BMO and AMC**: the same
fitted models evaluated within each window (no window-specific model); Δ top-10% and
Δ top-20% capture and ΔAUC with CIs. The pilot's "mostly BMO" is a hypothesis here.

## Decision (applied to pre-2026 results; 2026 then checked)

* **EXPECTED MOVE CONFIRMED — CANDIDATE FOR PRODUCTION**: Δ top-10% capture > 0 with CI
  lower bound > 0; Δ top-20% capture ≥ 0; ΔAUC > 0; Δ top-10% capture positive in ≥ 60% of
  test years; the 2026 Δ top-10% capture ≥ 0.
* **EXPECTED MOVE REAL BUT TOO SMALL / UNSTABLE FOR PRODUCTION**: not confirmed, but
  ΔAUC CI > 0 or Δ top-20% capture CI > 0.
* **FREE-PILOT RESULT DID NOT REPLICATE**: neither ΔAUC nor Δ top-20% capture has a CI
  above zero.
* **PROFESSIONAL DATA SOURCE UNSUITABLE** (checked first, before outcomes): point-in-time
  status not established in writing; or < 80% of eligible 2016–2025 events get a valid
  feature at ≤ 1 session; or > 5% of selected pairs fail the quote filters.

AUC alone never decides.

## Robustness (reported, never used to choose)

1. The live collector's rule: expiry strictly after the report date (both windows).
2. Staleness ≤ 3 sessions.
3. If the primary uses spot: the chain-internal fallback definition on the same events.

## 2026

Held out. Scored once, with models trained on every event whose endpoint precedes
2026-01-01, only after a dated amendment records the frozen source rules, definition,
population and pre-2026 results. Never used to change anything.

## Power — read before paying

The pilot's top-10% capture CI half-width was ±1.2 pt on 8,241 events. Scaling by √n:

| source start | first test year | OOF events | expected CI half-width | power if true Δ = 0.7 pt | 1.0 pt | 1.5 pt |
|---|---|---|---|---|---|---|
| 2014 | 2015 | ~19,300 | ±0.8 pt | ~40% | ~68% | ~95% |
| 2016 | 2017 | ~16,100 | ±0.9 pt | ~33% | ~58% | ~90% |

Rough (normal approximation, clustered-bootstrap width assumed to scale with √events).
**If the true top-10% effect is the pilot's +0.7 pt, this test more likely fails its
primary criterion than passes, even with perfect data.** The purchase is worth it only if
you accept that answer, or expect cleaner quotes (weekly expiries, a true close snapshot)
to make the effect larger. Starting in 2016 rather than 2014 costs ~5–10 points of power.

## Tests (to be written with the ingestion code)

The vendor-neutral selection and feature rules are implemented and tested now in
`research/options_confirmatory/core.py` / `testing/test_options_confirmatory.py`:
snapshot never after the prediction cutoff; one-session staleness limit; AMC same-day
expiry excluded; BMO same-day included; strict-after rule separate; call and put share
strike and expiry; mids from the same row's bid/ask; spot denominator; strike within 5% of
spot; every required provenance field present. With ingestion: identical event IDs in
every paired comparison; training years < test year; 2026 absent from selection; target
is the corrected column; no raw vendor data tracked by git.
