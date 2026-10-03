# Breakwater Phase 0 — Audit Revision 2

Answers the six methodological challenges raised against `audit/PHASE0_AUDIT.md`.
**No production code changed.** New evidence: `audit/provider_timestamps.parquet`
(12,269 event-level announcement timestamps, 500 tickers, fetched this session).

**Headline: the challenge was correct.** The rev-1 "timing-aware" analysis classified
100% of its events by price behavior, which is circular for the specific claim it was
used to support. Redone on observed timestamps, the direction of every finding holds but
**one rev-1 conclusion is withdrawn** (§3).

---

## Q1 — How every event in the rev-1 analysis was classified

`audit/quantify_bmo_bias.py` lines 27-33 are the whole classifier:

```python
per = ev.groupby("stock").agg(m0=("ret_d0", lambda x: x.abs().median()),
                              m1=("ret_d1", lambda x: x.abs().median()))
per["timing"] = np.where(per["m0"] > per["m1"], "BMO", "AMC")   # <-- price behavior
per.loc[per["n"] < 8, "timing"] = "unknown"
```

### Exact accounting — all 45,380 events

| Classification source | Events | Share |
|---|---:|---:|
| Event-level provider timestamps | **0** | 0.0% |
| Static ticker label **derived from price behavior** (median \|ret_d0\| vs \|ret_d1\| over the ticker's full history, ≥8 events) — 25,965 BMO + 19,402 AMC | **45,367** | 99.97% |
| Price-behavior inference of any kind | **45,367** | 99.97% |
| Unclassified (`unknown`, <8 events) | 13 | 0.03% |
| Any other inference | 0 | 0.0% |

There is no second category: the static ticker label **is** the price-behavior inference.
Every event inherited one label per ticker for all time.

### Why this invalidates the rev-1 corrected numbers

Two distinct circularities:

1. **Selection → magnitude.** A ticker was called BMO *because* its day-0 move exceeds its
   day-+1 move. The "corrected" reaction then re-anchors to include day 0. The corrected
   value is therefore mechanically larger by construction. The rev-1 figure
   "BMO mean \|reaction\| +55.7%" is not an independent measurement.
2. **Shared cause with the outcome.** The tier is a function of the p75 of the
   as-measured target, and the label is a function of the same price series. The rev-1
   claim "94.4% of High Alert is AMC" was measuring a correlation between two functions
   of the same data.

The 59-ticker provider sample in rev-1 (`probe_announcement_hours.py`, 58/39/3) was
**only an aggregate cross-check** and was never joined to the 45k events. Rev-1 should
have said so explicitly; it did not.

### Ground truth now available

`audit/fetch_provider_timestamps.py` pulls `yf.Ticker(s).earnings_dates`, whose index is
tz-aware `America/New_York` **with the announcement time**. 500/503 tickers returned data;
12,269 unique (stock, date) timestamps, 2008-04-16 → 2026-12-10. Classification is a pure
function of the hour: `<09:30 → BMO`, `≥16:00 → AMC`, else `INTRADAY`.

**Label validation (price behavior used here ONLY to test the labels, never to make them):**

```
window      n      share day-0 dominant   med|ret_d0|   med|ret_d1|
BMO      6,640            0.726              3.10%         1.41%
AMC      4,958            0.208              1.18%         3.68%
INTRADAY    99            0.475              1.94%         1.71%
```

The two suspicious round hours are real times, not fill values — they behave like their
neighbours:

```
05:00 0.739 | 06:00 0.733 | 07:00 0.722 | 08:00 0.711 | 09:00 0.676  <- BMO cluster
16:00 0.205 | 17:00 0.193                                            <- AMC cluster
```

Caveat kept: ~27% of provider-BMO events are day-+1 dominant. Part is ordinary noise on
small moves; part may be genuine provider error. The labels are strong, not perfect.

---

## Q2 — Results using ONLY events with observed event-level timestamps

`audit/verified_timing_analysis.py`. No inference anywhere. Anchoring rule:
`AMC → close(D+3)/close(D)` (unchanged); `BMO`/`INTRADAY` → `close(D+2)/close(D−1)`.
Both span exactly three trading sessions beginning with the first post-announcement session.

### Sample

```
verified timestamp AND a scored row in full_df   11,496 events   496 tickers
date range                                       2008-04-16 -> 2026-09-01
window mix                                       BMO 6,570 | AMC 4,829 | INTRADAY 97
coverage of all 45,190 scored events             25.4%
```

**No result below is extrapolated to the other 74.6%.**

### Market baseline P(|reaction| ≥ 8%) — n = 11,496

| | rate [95% Wilson CI] |
|---|---|
| as measured today | **0.128** [0.122, 0.134] |
| timestamp-anchored | **0.204** [0.197, 0.211] |

### Tier hit rates

| tier | n | as measured [95% CI] | timestamp-anchored [95% CI] |
|---|---:|---|---|
| High Alert | 1,354 | 0.428 [0.402, 0.454] | **0.455** [0.429, 0.482] |
| Elevated | 1,022 | 0.267 [0.241, 0.295] | **0.293** [0.265, 0.321] |
| Normal | 9,120 | 0.068 [0.063, 0.074] | **0.157** [0.149, 0.164] |
| High Conviction | 162 | 0.562 [0.485, 0.636] | **0.586** [0.509, 0.659] |

### Lift vs the corresponding baseline

| tier | as measured | anchored (crude) | anchored, **stratified by window** |
|---|---:|---:|---:|
| High Alert | 3.34x | 2.23x | **1.91x** |
| Elevated | 2.08x | 1.43x | **1.23x** |
| Normal | 0.53x | 0.77x | 0.80x |
| High Conviction | 4.38x | 2.88x | **2.46x** |

*Stratified = observed rate ÷ the rate the tier's events would show from their BMO/AMC mix
alone with zero within-window skill (`audit/stratified_lift.py`). The crude-vs-stratified
gap is pure composition.*

### Tier composition by verified window (row %)

| window | Normal | Elevated | High Alert |
|---|---:|---:|---:|
| AMC (n=4,829) | 56.5 | 18.7 | 24.8 |
| BMO (n=6,570) | **96.0** | 1.7 | 2.3 |
| INTRADAY (n=97) | 87.6 | 5.2 | 7.2 |

Share of each tier drawn from each window: High Alert 88.6% AMC / 10.9% BMO;
Elevated 88.4% / 11.2%; Normal 29.9% / 69.2%.

### Rates within each window

**BMO — n = 6,570.** Baseline 0.041 [0.037, 0.046] as measured → **0.173** [0.164, 0.183] anchored.

| tier | n | as measured | anchored | anchored lift |
|---|---:|---|---|---:|
| High Alert | 148 | 0.155 [0.106, 0.222] | 0.419 [0.342, 0.499] | 2.42x |
| Elevated | 114 | 0.167 [0.109, 0.246] | 0.386 [0.302, 0.478] | 2.23x |
| Normal | 6,308 | 0.036 [0.032, 0.041] | 0.164 [0.155, 0.173] | **0.94x** |

**AMC — n = 4,829.** Baseline 0.246 [0.234, 0.258], identical under both (this half was
never mismeasured).

| tier | n | rate | anchored lift |
|---|---:|---|---:|
| High Alert | 1,199 | 0.460 [0.432, 0.488] | 1.87x |
| Elevated | 903 | 0.281 [0.253, 0.311] | 1.14x |
| Normal | 2,727 | 0.140 [0.128, 0.154] | 0.57x |

**INTRADAY — n = 97.** Baseline 0.186 [0.121, 0.274]. Tier cells are 5-85 events; CIs span
most of the unit interval. **Reported for completeness only — nothing is inferable here.**

### Capture of anchored ≥8% moves (High Alert or Elevated)

| | rate [95% CI] | n |
|---|---|---:|
| overall | 0.390 [0.371, 0.410] | 2,344 |
| AMC | 0.678 [0.650, 0.704] | 1,188 |
| BMO | **0.093** [0.078, 0.111] | 1,138 |

### Rev-1 vs rev-2 on the same quantities

| quantity | rev-1 (price-inferred, 45k) | rev-2 (verified, 11.5k) |
|---|---|---|
| anchored market baseline | 0.165 | 0.204 |
| High Alert anchored hit | 0.409 | 0.455 |
| High Alert crude lift | 2.48x | 2.23x |
| BMO Normal anchored hit | 0.132 | 0.164 |
| BMO capture | 4.7% | 9.3% |
| BMO tiered Normal | 98.1% | 96.0% |

Direction and order of magnitude survive. The rev-1 numbers were biased toward a cleaner
story than the data supports, exactly as the circularity predicts.

---

## Q3 — What is actually proven, and what is not

### Established (verified timestamps, no inference)

1. **The target is mismeasured for BMO events.** BMO baseline moves 0.041 → 0.173 under
   correct anchoring, on 6,570 timestamped events. Not in dispute.
2. **The tier assignment is near-degenerate on BMO.** 96.0% of BMO events are `Normal`;
   88.6% of `High Alert` is AMC. Established on observed timestamps.
3. **Capture on BMO is 9.3%** [7.8, 11.1] vs 67.8% on AMC. Established.
4. **Every published lift figure is overstated.** The `CLAUDE.md` 3.70x becomes 3.34x on
   this subsample as measured, 2.23x anchored, **1.91x stratified**. High-conviction
   4.83x → 2.46x stratified.
5. **A meaningful share of the apparent edge is composition, not discrimination.**
   Knowing only "this company reports AMC" is worth **1.42x** (0.246 vs 0.173) with no model
   at all. Within AMC — the half that was always measured correctly — High Alert lift is
   only **1.87x**.

### **WITHDRAWN from rev-1**

> ~~"The signal itself is not broken. Within BMO the tier ordering still holds under the
> corrected target (Normal 0.132 → Elevated 0.255 → High Alert 0.426)."~~

On verified timestamps the BMO ordering **does not hold in any useful sense**:

```
BMO Normal   0.164 [0.155, 0.173]   lift 0.94x   n=6,308  <- indistinguishable from baseline
BMO Elevated 0.386 [0.302, 0.478]   lift 2.23x   n=  114
BMO High Alert 0.419 [0.342, 0.499] lift 2.42x   n=  148
```

`Normal` is **0.94x** — no discrimination whatsoever across 96% of BMO events. And
Elevated vs High Alert CIs overlap almost entirely, so those two are not separable.
The honest statement is: *on BMO events the model makes almost no calls, and where it
does the two non-Normal tiers cannot be told apart.* Rev-1's monotone-and-healthy reading
was an artifact of the circular labels.

### What is NOT yet established, and cannot be until the rebuild

**The BMO tiers evaluated above were produced by a corrupted feature.** Every BMO event's
`abs_reaction_p75_rolling`, `abs_reaction_p75`, `abs_reaction_median`, `reaction_entropy`,
`reaction_std`, `directional_bias`, `is_extreme_reaction`, `stock_bucket_lift` and hence
its score, structural bucket and lift-promoted bucket were computed from mismeasured
prior outcomes. So the 148 BMO events labelled High Alert are a **sample selected by a
broken instrument**. Their 0.419 anchored hit rate says something about that selection —
it does not establish that the model's *logic* works on BMO.

**Nothing about the corrected model can be claimed until the full historical chain is
rebuilt from anchored outcomes**, in this order:

1. anchored `reaction_{1,3,5}d` and `abs_reaction_3d` for every event
2. → `is_large_reaction` / `is_extreme_reaction`
3. → `abs_reaction_p75_rolling`, `_p75`, `_median`, `_p90_rolling`, `reaction_entropy`,
   `reaction_std`, `directional_bias`
4. → `earnings_explosiveness_score` and its structural bucket
5. → `stock_bucket_lift` (needs both the corrected outcomes **and** the corrected buckets)
6. → the lift-adjusted bucket and `is_high_conviction`
7. → **re-fit** `BUCKET_ELEVATED_FLOOR` / `BUCKET_HIGH_ALERT_FLOOR` (73/79), `LIFT_TO_*`,
   `LIFT_PRIOR_STRENGTH`, the 0.12 p75 ceiling and the 0.85/0.15 weights — every one of
   which was selected against the distorted distribution
8. → re-run calibration **stratified by announcement window**, with the stratified lift as
   the headline, so composition can never again be read as skill

Until step 8 completes, the correct posture is: **the model's demonstrated edge is
1.87x within AMC and unestablished within BMO.** Whether correcting the target rescues
BMO is the open empirical question of Phase 1 — plausible (the mechanism is a pure
measurement/threshold artifact) but unproven.

### Coverage constraint on that rebuild

Verified timestamps cover **25.4% of scored events (11,496 of 45,190)** and only reach
~2020 for most tickers. Steps 1-8 on the verified subsample alone give ~11.5k events —
enough to re-fit thresholds, not enough for the 2011-2025 walk-forward the current
thresholds claim. **Do not re-fit on inferred labels.** Either accept a shorter,
honestly-scoped window, or source historical times from a paid provider before re-fitting.

---

## Q4 — Staleness effect on the FINAL shipped tier

`audit/staleness_final_tier.py`. Every history-dependent quantity recomputed through the
most recently completed event, replicating `scoring/scoring_features.py` exactly
(shrinkage `LIFT_PRIOR_STRENGTH=20`, gates 1.5/3.0, cuts 73/79). **Issue 1 is deliberately
NOT applied here** so the two defects stay separable. 495 upcoming events.

| quantity | events changed | mean \|Δ\| | max \|Δ\| |
|---|---:|---:|---:|
| `abs_reaction_p75` (effective) | 251 (51%) | 0.0020 | 0.0232 |
| `reaction_entropy` | 492 (99%) | 0.0109 | 0.4766 |
| `earnings_explosiveness_score` | 271 (55%) | 1.28 | 15.48 |
| `stock_bucket_lift` | **495 (100%)** | 0.044 | **2.519** |
| structural tier | 11 (2.2%) | — | — |
| **final lift-adjusted tier** | **7 (1.4%)** | — | — |
| `is_high_conviction` | **0 (0.0%)** | — | — |

**Final shipped tier** (rows = shipped today, cols = correct):

```
              Elevated  High Alert  Normal
Elevated            41           3       2
High Alert           1          65       0
Normal               1           0     382
```

**High conviction:** 5 True / 490 False, unchanged. Not structural luck — the three events
that gain `High Alert` (ISRG, CMG, WDC) all carry an empty `pre_earnings_drift_flag`, so
the HC gate never fires. A single drift flag on any of them would have flipped it.

### The seven wrong tiers

| stock | date | score old→new | lift old→new | tier old→new | last reaction |
|---|---|---|---|---|---|
| ISRG | 2026-10-20 | 73.55 → 80.65 | 1.10 → **3.62** | Elevated → **High Alert** | 13.0% |
| CMG | 2026-11-04 | 74.96 → 82.07 | 1.87 → **3.10** | Elevated → **High Alert** | 9.4% |
| WDC | 2026-11-05 | 76.27 → 86.09 | 2.27 → 1.45 | Elevated → **High Alert** | 15.6% |
| NXPI | 2026-10-26 | 63.60 → 63.60 | 1.40 → **1.54** | Normal → **Elevated** | 11.6% |
| ADBE | 2026-09-10 | 73.57 → 70.10 | 1.38 → 1.39 | Elevated → Normal | 5.2% |
| PTC | 2026-11-11 | 66.88 → 63.92 | 1.51 → 1.49 | Elevated → Normal | 5.6% |
| WSM | 2026-11-18 | 81.85 → 77.52 | 1.82 → 1.55 | High Alert → Elevated | 3.9% |

**4 under-called, 3 over-called — and the split is not random.** All four under-calls
follow a large recent reaction (9.4–15.6%); all three over-calls follow a small one
(3.9–5.6%). The staleness bias runs precisely against the product's purpose.

**Two findings the structural-only rev-1 analysis missed:**

1. **`stock_bucket_lift` is stale on 100% of rows**, with max |Δ| 2.52 — an order of
   magnitude larger than the score drift, because the lift moves on *three* inputs at once
   (the stock's prior count in bucket, its prior extreme count, and the global baseline).
2. **The lift can flip a tier with a frozen score.** NXPI's score is byte-identical
   (63.596) yet it moves Normal → Elevated purely because the stale lift 1.400 sits just
   under the 1.5 gate while the correct 1.535 clears it. Auditing only the score would
   have declared NXPI unaffected.

Net: **1.4% of shipped tiers are wrong per run**, less than the 2.2% structural rate
(lift promotion masks two structural changes), but the errors are systematically
anti-correlated with recent realized risk.

---

## Q5 — Revised architecture (replaces the rev-1 "pending row" proposal)

Rev-1 proposed adding a pending row and dropping `shift(1)`. **Both parts were wrong.**
Injecting a future-dated row into the daily frame corrupts `merge_asof`, every
`groupby("stock").rolling()`, and the `groupby("date")` cross-sectional ranks in
`engineer_vol_stress` / `engineer_momentum_pressure` / `engineer_sector_vol_stress`.
And dropping `shift(1)` would silently relax the leakage guard on 45k historical rows.

### The key observation

`shift(1)` already means *"aggregate over every event strictly before this one."* For a
pending event with a NaN outcome, that is **exactly** the correct as-of-today statistic —
including the most recently completed event. So the correct fix adds **no new statistical
logic at all**: it appends one row per stock to the frame those aggregations already run
on, and changes nothing else.

Every event-level statistic in the codebase already runs on an event-indexed frame built
by `build_earnings_df()` or a `df.loc[earnings_mask]` slice. The daily frame is only ever
used as a *carrier*. That carrier is the accident to remove.

### Smallest sufficient change: `output/events_df.parquet`

```
stage3  price features on the daily frame            UNCHANGED
        (daily_ret, drift, vol, momentum, earnings windows, sector cross-sections,
         reactions) — never sees a future row

stage3b build_event_frame(daily_df)   <-- NEW, ~40 lines
        one row per (stock, event):
          completed: the anchor row's price features + realized reaction
          pending:   the LAST price row's price features, stamped with the next
                     earnings_date, abs_reaction_3d = NaN, is_pending = 1
        columns: stock, event_id, earnings_date, anchor_date, announce_window,
                 is_pending, score_asof_date, <price features>, <reaction cols>

stage3c event-level features on that frame           MOVED, logic unchanged
stage4  scoring on that frame                        MOVED, logic unchanged
        every .shift(1) kept verbatim

stage5  writes output/events_df.parquet (~46k rows, ~5 MB)
        merges completed rows back onto full_df for continuity
```

### Why this is the smallest option that satisfies the constraints

| requirement | how it is met |
|---|---|
| historical and upcoming use the same as-of logic | literally the same function call over one frame; the pending row is just the last row of each stock's group |
| no `groupby().last()` | exporters select `is_pending == 1` |
| no future row in daily price calculations | the daily frame is never modified; the event frame is built *after* stage 3 and read by nobody upstream |
| leakage guard preserved | every `shift(1)` untouched; a pending row has a NaN outcome so it can contribute to nothing |
| historical values unchanged | completed rows are a reordering of rows that already exist — assertable byte-for-byte |

### Call-site changes (4 lines of deletion each)

```
streamlit_dash/streamlit_export.py:73   groupby("stock").last()  ->  events[events.is_pending == 1]
analysis/save_predictions.py:43         same
report/report_builder.py:56             same
cron/cron_weekly_digest.py              unchanged (reads upcoming_df)
```

### Three things this deletes for free

1. **`utilities/scoring_slice.py` (140 lines) becomes unnecessary.** It exists solely
   because `full_df.parquet` is 323 MB and the droplet has ~590 MB. `events_df.parquet`
   is ~5 MB. The streaming reader, the batch-size tuning and the non-contiguity hazard
   documented in its module docstring all go away.
2. **`INCREMENTAL_CACHED_COLS` (14 entries) becomes unnecessary.** It exists to carry
   forward event statistics the incremental window cannot recompute. With an event frame
   they are always recomputable from ~46k rows.
3. **The `pd.cut` / `groupby.ffill` / `.loc[mask]` assignment dance** in
   `engineer_high_conviction`, `engineer_surprise_momentum_flag` and
   `engineer_pre_earnings_drift_flag` — all of which exist to paper over "this column is
   NaN off earnings days" — collapses, because on an event frame every row is an event.

### Guardrails to add with it

- `assert` that completed-event scores are unchanged vs the pre-change parquet (golden file).
- `assert` exactly one pending row per stock with a future `earnings_date`.
- `assert` no pending row has a non-NaN outcome column.
- `score_asof_date` stamped on every row, exported and persisted.
- A test that `is_pending == 0` is what calibration reads and `is_pending == 1` is what the
  exporters read — the train/serve skew made explicit and enforced.

### Sequencing note

Do **Q5 before Q1's fix.** The event frame is a pure refactor with a byte-identical golden
test, and it is the substrate the anchoring change then edits in one place
(`build_event_frame` picks the anchor row). Doing them in the other order means writing
the anchoring logic twice.

---

## Q6 — Non-trading-day earnings dates: do NOT auto-roll

Rev-1 recommended "roll to the next session." **Withdrawn.** The evidence says these dates
are not a calendar-alignment problem; they are a data-integrity problem with at least two
distinct causes.

**318 unlabelled events fall inside the stock's own price history.** Splitting by whether
the *market* traded that day:

| | events | interpretation |
|---|---:|---|
| market closed that day (Sat 144, Sun 99, holiday-Mon 34, other 21) | **298** | genuine non-session date — either a real off-hours filing or a wrong date |
| **market open that day** | **20** | **not a calendar issue at all — a missing price row for that ticker** |

The 20 are a price-ingestion gap, clustered on 2026-05-20/21 (NDSN, HAS, INTU, WMT, RL,
ROST, DECK, DE, CPRT, TTWO, WDAY, WSM). Rolling these forward would paper over a bug in
`ingestion/fetch_prices.py` and silently mis-anchor 20 events.

**Only 44 of 318 have a provider timestamp to verify against at all (13.8%).** And where
the provider does corroborate the date, the times argue the dates are unreliable rather
than confirming them — of 65 such matches:

```
Saturday 08:00  x17     Thursday 16:00  x9      Monday 16:00  x4
Saturday 12:00  x1      Thursday 07:00  x4      Monday 17:00  x3
Sunday   16:00  x1      Wednesday 16:00 x5      ...
```

A Saturday 08:00 announcement is not a plausible release time; a Thursday 16:00 event that
lands on a "non-trading day" is self-contradictory. The provider timestamp here corroborates
the *unreliability*, not the date.

### Recommended policy

1. Add `anchor_status ∈ {resolved, unresolved_no_session, unresolved_price_gap, unresolved_no_timestamp}`
   to the event frame. Default `unresolved_*`, never silently rolled.
2. **Exclude `unresolved_*` events from training, calibration and published counts**, and
   print the excluded count every run. Silence is what let 318 events vanish.
3. Resolve an event **only** on a verified announcement timestamp that is consistent with a
   real session — then anchor by the ordinary rule, no special case.
4. Route the 20 market-open cases to a **separate price-gap investigation**; they are not a
   timing bug.
5. Fail the run if `unresolved_*` exceeds a threshold (say 1.5% of events) — a rising count
   means an ingestion regression.

---

## Revised implementation plan

**Phase 0.5 — instrumentation (no behavior change)**
- P0.5.1 Freeze `testing/calibration.py` output to `audit/baseline_calibration_<commit>.csv`.
- P0.5.2 Persist `audit/provider_timestamps.parquet` as a tracked artifact; add a refresh script.
- P0.5.3 Add the failing tests from rev-1 §3, plus:
  `test_lift_recomputed_for_pending_event`, `test_pending_row_has_null_outcome`,
  `test_completed_scores_unchanged_golden`.

**Phase 1 — event frame (Q5). Pure refactor, byte-identical golden test.**
- P1.1 `pipeline/events.py: build_event_frame()`; `is_pending`, `anchor_date`, `score_asof_date`.
- P1.2 Move event-level feature + scoring functions onto it. **Keep every `shift(1)`.**
- P1.3 Stage 5 writes `events_df.parquet`; four call sites switch to `is_pending == 1`.
- P1.4 Assert completed-event scores unchanged. **Fixes Issue 2 entirely.**
- P1.5 Delete `utilities/scoring_slice.py` and `INCREMENTAL_CACHED_COLS` once green.

**Phase 2 — announcement timing (Q1/Q2/Q6)**
- P2.1 Schema: `earnings.announce_ts_ny TIMESTAMP`, `announce_window VARCHAR`.
- P2.2 Stop `.dt.date` in `fetch_one_earnings_dates`; store the timestamp. Backfill 2020+.
- P2.3 `build_event_frame` picks `anchor_date` from `announce_window`
  (AMC → D, BMO/INTRADAY → prior session), sets `anchor_status`.
- P2.4 Assert AMC reactions are **bit-identical** to today's.
- P2.5 Events without a verified window → `anchor_status = unresolved_no_timestamp`,
  excluded and counted. **No price-behavior inference in production, ever.**
- P2.6 Fix the 20-event price gap separately.

**Phase 3 — rebuild and re-fit (Q3). Nothing about the corrected model is claimable before this.**
- P3.1 Rebuild the whole chain (§Q3 steps 1-6) on verified events.
- P3.2 Re-fit 73/79, `LIFT_TO_*`, `LIFT_PRIOR_STRENGTH`, the 0.12 ceiling, the 0.85/0.15 weights.
- P3.3 Calibration **stratified by announcement window**; stratified lift as the headline.
- P3.4 Decide, on evidence, whether BMO is modellable — including the option of a separate
  BMO threshold set, or of scoping the product to AMC reporters and saying so.
- P3.5 Decide the pre-2020 policy: shorter honest window vs paid historical timestamps.
  **Do not re-fit on inferred labels.**

**Phase 4 — claims**
- P4.1 Retire every published lift figure until P3.3.
- P4.2 Mark the 10 archived predictions `0.3.1-preaudit`, void for track-record purposes.
- P4.3 Pause `marketing/generate_public_track_record.py`.

---

## Corrections to `audit/PHASE0_AUDIT.md`

| rev-1 statement | status |
|---|---|
| "timing-aware" 45k analysis | **superseded** — 100% price-inferred, circular for its purpose |
| "The signal itself is not broken… BMO ordering holds monotonically" | **withdrawn** — BMO Normal lift is 0.94x on verified data |
| "a BMO High Alert is as accurate as an AMC one (0.426 vs 0.408)" | directionally survives (0.419 vs 0.460) but the sample was selected by the broken instrument; **not evidence the logic works** |
| lift 3.70x → 2.48x | refined: **3.34x → 2.23x crude → 1.91x stratified** on verified events |
| "the largest available upside in the product" | **downgraded to a hypothesis** pending Phase 3 |
| "roll [non-trading-day events] to the next session" | **withdrawn** — see Q6 |
| Issue 2 severity "2% of tiers wrong" | refined: **1.4% of FINAL tiers**, 100% of lifts, and the errors are anti-correlated with recent realized risk |
| pending-row-in-daily-frame proposal | **replaced** by the event frame (Q5) |
