# Historical options pilot — pre-registration

Written 2026-09-29, **before any target outcome was joined to any options feature**. At the
time of writing I had looked at: the source schema, its snapshot-date cadence, a handful of
raw chains (AAPL, META/FB, BRK.B, BF.B, DOC, SNDK) and Breakwater's price semantics. No
feature has been compared with `y`. Anything chosen after outcomes are read is labelled
**POST HOC** in RESULTS.md.

Research only. No production module, threshold, `MODEL_VERSION`, report, dashboard or
prediction archive changes, whatever the result.

## Question

Does options information available at Breakwater's real weekly call cutoff improve
prediction of corrected ≥8% earnings moves beyond Model C — especially at the top of the
ranking (top 10% / top 20%)?

## Source

DoltHub `post-no-preference/options`, every query `AS OF` commit
`1ug5hqta1o786faoh8fv89q7grrd00jj` (master head at 2026-09-29 06:34:57 UTC). Tables
`option_chain` (quotes + greeks per contract) and `volatility_history` (one IV/HV summary row
per symbol and date). Raw query results are cached, gzipped and keyed by SQL hash, in the
gitignored `data/vendor/dolthub_options/`. Nothing is committed.

## Population

Phase 3's refit population (`research/phase3_refit/`), unchanged: completed events,
announcement window BMO or AMC, `abs_reaction_3d_anchored` available with its 3-session
endpoint resolved, ≥ 8 prior corrected outcomes at the call cutoff, both Model C inputs
present. Restricted to report year ≥ 2019 (the source starts 2019-02-09).

**Target:** `y = abs_reaction_3d_anchored ≥ 0.08` (the Phase 3 frame's `y_extreme`). The
legacy `abs_reaction_3d` is never used.

## Clock and snapshot selection

* **Call cutoff** = Phase 3's `call_cutoff_date`, reused as stored: the last market session
  strictly before the Monday of the report week.
* **Snapshot** = the latest `option_chain` date **≤ call cutoff** on which the source has
  rows for the event's symbol. The date comparison is literal: a snapshot dated after the
  cutoff is never considered, even a Saturday-dated one that carries Friday's quotes (2019
  is Saturday-dated; this makes most of 2019 5 sessions stale, which is accepted).
* **Age** = market sessions between the snapshot's quote session (the snapshot date, or the
  last session before it if it is a weekend/holiday date) and the cutoff.
* Walk back at most 10 sessions. Beyond that the event is "missing". Nothing is
  forward-filled; a snapshot is only ever used for the symbol it was fetched for.

**Staleness tiers:** strict ≤ 1 session, medium ≤ 3, broad ≤ 5. **Main result** = the
tightest tier that keeps a useful sample, decided from row counts only: ≥ 4,000
out-of-fold events **and** ≥ 70% of the broad tier's out-of-fold events. Otherwise the next
tier. All three tiers are reported.

## Identity

The source keys contracts on the symbol as traded on the snapshot date and spells class
shares with a dot (`BRK.B`, `BF.B`). Mapping: Breakwater `X-Y` → `X.Y`, plus an explicit,
dated rename table (FB → META from 2022-06-09; others only if the audit finds them and
verifies them). A ticker string that belonged to a different company at snapshot time (e.g.
DOC = Physicians Realty before 2024) is excluded for those dates, never joined. The audit
checks, per stock, that the parity-implied spot (`K + C_mid − P_mid`) divided by
Breakwater's adjusted close changes only at clean split ratios. Exclusions are listed.

## Quote filters (per contract, in this order; each filter's removals reported)

1. bid and ask present; 2. bid ≥ 0 and ask ≥ 0; 3. ask ≥ bid; 4. ask > 0;
5. 0 < IV ≤ 5.0; 6. call delta in [0, 1], put delta in [−1, 0].

Nothing else is removed. Duplicate primary keys cannot exist (the table's PK is
date, symbol, expiration, strike, type) and are counted anyway.

## Expiry that covers the announcement

* Expiry after the report date: covers.
* Expiry on the report date: **BMO covers** (the news is out before that day's open and the
  option trades through that session); **AMC does not** (it expires at 16:00, before the
  news).
* Expiry before the report date: never.

Report date = Phase 3's `report_date`. Near expiry = the first covering expiration in the
snapshot. Note the difference from the live collector (`ingestion/fetch_iv.py`), which
requires expiry strictly after the earnings date for both windows. That rule is the
robustness check, not the primary definition.

## ATM pair

In the near expiry, among strikes with a filtered call **and** put at the same strike,
choose the pair minimising `|Δcall − 0.5| + |Δput + 0.5|`, subject to both |Δ| in
[0.30, 0.70]. Ties: smaller `|C_mid − P_mid|`, then the lower strike. No pair → no features.
**Amendment 1 (2026-09-29, after the coverage audit, before any outcome was joined):** the
chosen strike must also lie within 10% of its own put-call-parity spot `K + C_mid − P_mid`
(the live collector's 10% rule). Reason: a handful of rows carry greeks computed off a wrong
spot (KEY 2024-07-12: 20 strike, call 1.08, put 4.80, parity spot 16.3, both deltas ≈ 0.5,
both IVs 3.07). The pair is chosen from the chain alone. Breakwater's stored prices are split- and
dividend-adjusted and are never compared with a nominal strike.

## Features

| name | definition | model input |
|---|---|---|
| **expected move** (primary) | `(C_mid + P_mid) / K`, mids = (bid+ask)/2 of the same row | `log` |
| **ATM IV** (primary) | mean of the pair's call and put IV | `log` |
| put skew (secondary) | IV of the near-expiry put whose delta is closest to −0.25, accepted only within [−0.35, −0.15], **minus** ATM IV. No interpolation | raw |
| term structure (secondary) | near ATM IV / ATM IV of the first later expiry that has a valid ATM pair (same rule) | `log` |
| IV vs own history (secondary) | `volatility_history.iv_current` on the snapshot date / median `iv_current` over the same symbol's **weekly** rows (the last source date of each Mon–Sun week) dated strictly before the snapshot, within 365 days; ≥ 20 prior weeks required | `log` |

The IV-history feature uses the provider's own IV summary (definition undocumented) because
`volatility_history` has at least one row per week per symbol back to 2019, which makes a
causal lookback possible. Sampling one row per week gives 2019 (weekly source) and 2025 (daily source) the
same lookback density. Rebuilding it from past chains would need every past chain. It
never uses knowledge of which past dates were earnings dates.

Every feature row carries `snapshot_date`, and `snapshot_date ≤ call_cutoff_date` is
asserted on the whole table and tested.

## Models (logistic, identical to Phase 3: standardise on training fold, L2, C = 1)

Primary comparisons, all on the **same rows** (covered sample: both primary features
present, age within the tier):

* **C** = `log hist_mean_abs`, `log vol_30d` (Phase 3's Model C, unchanged)
* **C + expected move**, **C + ATM IV**, **C + expected move + ATM IV**
* **C + compact** = C + expected move + ATM IV + every secondary feature whose coverage in the
  main covered sample is ≥ 80% (coverage, not outcome, decides)

Secondary comparisons: each secondary feature on the rows where it exists. C, C + feature,
C + EM + IV and C + EM + IV + feature are refit on exactly those rows.

No XGBoost, no feature sweep, no tuning.

## Walk-forward

For test year Y, training rows are covered events whose 3-session endpoint is before
1 January Y. Scaling and coefficients come from training rows only. **First test year** =
the first year whose training block has ≥ 1,000 rows and ≥ 150 positives. Test years run
through 2025. **2026 YTD is held out** and scored once, with `--holdout`, after everything
above is final.

## Metrics and decision

Headline: top-10% and top-20% hit rate, capture and lift. Also ROC-AUC, PR-AUC, Brier,
log loss, BMO and AMC AUC, year by year. Deltas versus C on identical rows, with 95%
stock-clustered bootstrap CIs (500 reps).

**Coverage bias:** Phase 3's own out-of-fold C (trained on full Benzinga history) is
reported on all its events in the test years, on the covered events and on the uncovered
ones. Base rate, BMO/AMC mix, sector mix, stock count and typical history/vol are compared
between the covered and uncovered events.

Conclusion rules:

* **MATERIAL INCREMENTAL VALUE**: C + expected move, or C + EM + IV, improves top-10% **or**
  top-20% capture versus C with the 95% CI above zero, the AUC delta CI is above zero, the
  point estimate is positive in ≥ 4 of 5 test years (or the equivalent share), and the 2026
  holdout delta has the same sign.
* **WEAK / UNCERTAIN VALUE**: an AUC / log-loss gain with CI above zero but no top-of-ranking
  capture gain whose CI clears zero, or gains that are unstable across years or windows.
* **SOURCE TOO POOR TO ANSWER**: the main tier has < 2,000 out-of-fold events, or coverage is
  so selective that the covered sample is unrepresentative (for example < 25% of events in
  the test years).

Robustness, reported but not used to choose anything: the strictly-after-report-date expiry
rule (the live collector's rule).

## Amendment 2 — the freeze (2026-09-30, before any outcome was joined)

Written after the Phase A audit and before `evaluate.py` has read `y` for any row. It
fills the gaps Phase A exposed and freezes everything the evaluation uses. Nothing here
was chosen by looking at a result.

**Frozen as already written above, unchanged:** population, call cutoff, snapshot matching
(latest source date ≤ call cutoff, literal date comparison, no forward fill, 10-session
walk-back), staleness tiers and the rule that picks the main one, announcement-aware
expiry rule, ATM-pair rule plus Amendment 1, quote filters, target, logistic model form,
first-test-year rule, metrics, bootstrap (stock-clustered, 500 reps, paired on identical
rows).

**Clarifications and additions:**

1. **Snapshot calendar.** The source's date list is the `volatility_history` date list,
   verified by `calendar_check.py` (targeted + sampled probes of `option_chain`). If the
   check finds an `option_chain` date missing from that list inside any event's matching
   window, the calendar gains it and Phase A is regenerated before evaluation.
2. **Term structure.** The later expiry's ATM pair must pass the same rules as the near
   pair, Amendment 1 included. No valid later pair → the feature is missing. No farther
   substitute is ever chosen beyond "the first later expiry with a valid pair".
3. **Comparison blocks (every block compares models on identical rows; the row count and
   a hash of the event IDs are saved):**
   * *primary* — C, C+EM, C+IV, C+EM+IV on rows with both primary features;
   * *compact* — C, C+EM+IV, C+compact on rows with every compact input;
   * *secondary k* — C, C+k, C+EM+IV, C+EM+IV+k on rows where feature k exists.
   Each block states its share of the covered sample, so no feature is quietly evaluated
   on a different population.
4. **IV relative to own history — build gate.** `volatility_history.iv_current` (weekly,
   see above) is used only if, on the main-tier covered sample (2020–2025): (a) a
   `volatility_history` row exists on the event's own snapshot date for ≥ 80% of events,
   and (b) its Spearman correlation with the chain's **later-expiry** ATM IV (the one less
   inflated by the earnings event) is ≥ 0.70 — evidence it measures the same stock's IV.
   Neither check reads an outcome. Fail → the feature is omitted and the reason reported.
   Its z-score variant is **not** added: one form only.
5. **Robustness — live collector's expiry rule.** The same primary block recomputed with
   the near expiry chosen as the first expiry **strictly after** the report date for both
   windows (`ingestion/fetch_iv.py`). Reported next to, never instead of, the main rule.
6. **2026 is removed from every selection run** (`selection_panel`) — not merely excluded
   by the training mask. Test years stop at 2025.
7. **Winner rule (applied to the main tier's pre-2026 walk-forward only).** Candidates:
   C+EM, C+IV, C+EM+IV, C+compact.
   * A candidate *qualifies* if its ΔAUC 95% CI is above zero **and** its Δ top-10% or
     Δ top-20% capture CI is above zero.
   * Among qualifiers: highest mean of Δ top-10% and Δ top-20% capture; within 0.002 of
     the best → fewer inputs.
   * No qualifier → the candidate with the highest ΔAUC among those whose ΔAUC CI is
     above zero, labelled *weak*.
   * None of those either → C+EM+IV, labelled *for information*.
   The winner, its inputs, transforms (log for expected move, ATM IV, term ratio, IV
   ratio; skew raw), missing handling (a model scores only rows with all its inputs; no
   imputation), tier and model form are written to `output/options_pilot/winner.json` and
   into RESULTS.md **before** `--holdout` runs. `--holdout` refuses to run without that
   file and asserts the compact set has not changed.
8. **2026 holdout.** Models trained on every covered row with an outcome endpoint before
   2026-01-01, scored once on 2026 YTD. All candidates and C are reported, the frozen
   winner is the one that counts.
9. **Conclusion rules** as written above, with the winner in place of "C + expected move,
   or C + EM + IV".
