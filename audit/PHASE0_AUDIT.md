> **SUPERSEDED IN PART — see `audit/PHASE0_AUDIT_REV2.md` (Phase 0 revision 2).**
> The "timing-aware" figures in Issue 1 classified all 45,380 events by price behavior,
> which is circular for the claim they support. Rev 2 redoes the analysis on 11,496 events
> with observed provider timestamps. Both defects remain real; the conclusion
> "the signal survives for BMO" is **withdrawn**, the non-trading-day roll-forward
> recommendation is **withdrawn**, and the pending-row design is **replaced**.

# Breakwater Phase 0 — Methodology Audit

Scope: two suspected correctness issues. No pipeline code changed. All probe scripts
live in `audit/` and read the existing DB / parquet read-only.

| # | Suspicion | Verdict |
|---|---|---|
| 1 | Announcement time discarded; BMO and AMC events measured inconsistently | **REAL — severe.** Worse than suspected: for a BMO event the entire earnings move is excluded from the target, not merely folded into the denominator |
| 2 | Upcoming event inherits the previous event's score via `groupby("stock").last()` | **REAL — confirmed at 100% of shipped rows.** Exactly one earnings event stale |

---

## Issue 1 — Earnings-event return alignment

### 1a. Exact current behavior

**Ingestion.** `ingestion/fetch_earnings_dates.py:fetch_one_earnings_dates` reads
`yf.Ticker(s).earnings_dates`, whose index is tz-aware `America/New_York` **with the
announcement time in it**. Verified live:

```
JPM   2026-07-14 06:00:00-04:00     <- BMO
AAPL  2026-07-30 16:00:00-04:00     <- AMC
```

Line 45-49 destroys it:

```python
earnings_dates_df["earnings_date"] = (
    pd.to_datetime(earnings_dates_df["earnings_date"])
      .dt.tz_localize(None)     # keeps NY wall time, drops the tz
      .dt.date                  # <-- announcement time discarded here
)
```

**DB.** `earnings.earnings_date` is typed `DATE`. There is no time column anywhere in the
schema. The legacy AlphaVantage path (`ingest_all_earnings_dates`) never had a time to
begin with — `quarterlyEarnings.reportedDate` is date-only.

**Stage 2.** `utilities/data_utilities.py:merge_prices_earnings_dates` does a
`merge_asof(..., direction="forward")`, so every price row carries the *next* earnings
calendar date.

**Labeling.** `engineer_earnings_windows` sets
`is_earnings_day = (earnings_date - date).days == 0` — true on the trading row whose
**calendar date equals the announcement date**, regardless of the hour.

**Reaction.** `feature_engineering/post_earnings_stock_features.py:engineer_earnings_reactions`:

```python
df["reaction_3d"] = (group.shift(-3) / df["price"]) - 1     # price == close of the earnings row
```

so `abs_reaction_3d = |close(D+3) / close(D) - 1|` where `D` is the announcement calendar day.

### 1b. Why that is inconsistent

| | announcement | `close(D)` is… | what `reaction_3d` measures |
|---|---|---|---|
| **AMC** (16:00) | after the close of D | the last **pre**-announcement print | the full event move (gap on D+1, plus 2 days) — **correct** |
| **BMO** (06:00–08:00) | before the open of D | the **post**-announcement close, gap already in it | days D+1…D+3 drift only — **the event move is entirely excluded** |
| **Intraday** (~3% of events) | during the session | partially post-announcement | partially excluded |

So the report-day jump is in the *denominator* for BMO — and, because the numerator starts
at D+3, it is in **neither** the numerator's start nor its end differential. BMO events are
not merely shifted; their earnings reaction is measured on the wrong three days.

### 1c. Concrete evidence

`audit/probe_announcement_timing.py` — for every confirmed event, the close-to-close return
on the stored `earnings_date` (day 0) vs the next trading day (day +1). 45,376 events:

```
known BMO reporters              known AMC reporters
        med|r_d0|  med|r_d1|             med|r_d0|  med|r_d1|
JPM       1.84%      0.93%      AAPL       1.33%      4.34%
PG        2.46%      0.67%      MSFT       0.88%      3.31%
CAT       3.41%      1.18%      NVDA       1.76%      5.44%
WMT       2.24%      0.95%      AMZN       2.02%      7.38%
MMM       2.94%      0.93%      NFLX       1.53%      9.71%
```

Per-stock verdict over the universe: **280 stocks day-0-dominant (BMO-like), 220 day-+1-dominant
(AMC-like)** — i.e. the two conventions are mixed roughly 57/43 inside one target column.

Independently confirmed against the provider's own timestamps
(`audit/probe_announcement_hours.py`, 59 sampled tickers, 1,454 events):
**58.0% BMO / 39.0% AMC / 3.0% intraday**, and **25 of 59 tickers mix more than one window
within their own history** — a per-ticker static label is not sufficient.

### 1d. Magnitude (`audit/quantify_bmo_bias.py`)

Rebuilding the BMO reaction from the last **pre**-announcement close
(`close(D+2)/close(D−1) − 1`, still a 3-trading-day span):

```
BMO events (n=25,963)         current      timing-aware
  mean |reaction_3d|           2.76%         4.29%     (+55.7%)
  p75                          3.68%         5.81%
  P(>= 5%)                     0.144         0.309
  P(>= 8%)                     0.049         0.140
  Spearman(current, correct)               0.304        <-- the BMO target is
AMC events (n=19,399)          unchanged (rank corr 1.000)     mostly noise today
```

`abs_reaction_p75_rolling` is the dominant term of `earnings_explosiveness_score`
(weight 0.85). It is therefore ~37% too low for every BMO reporter, every quarter.

### 1e. Methodological consequence — this is the product, not a rounding error

Tier composition, 2015–2025 (`audit/tier_by_timing.py`):

```
                Normal  Elevated  High Alert          share of tier that is AMC
  AMC             59.0%    19.3%      21.7%             Elevated   94.4% AMC
  BMO             98.1%     0.9%       1.0%             High Alert 94.4% AMC
```

**Breakwater is currently, in effect, an AMC-reporter detector.** 57% of the universe is
tiered `Normal` 98% of the time.

Capture of genuine (timing-aware) ≥8% moves:

```
  AMC events:  66.9% captured in High Alert/Elevated
  BMO events:   4.7% captured                          <-- 1,624 real tail events, 76 flagged
```

The signal itself is **not** broken. Within BMO the tier ordering still holds under the
corrected target (Normal 0.132 → Elevated 0.255 → High Alert 0.426), and a BMO `High Alert`
is *as accurate as* an AMC one (0.426 vs 0.408). What is broken is that the compressed BMO
distribution almost never clears the fixed 73/79 cut points. **This is a fixable
measurement/threshold problem, not a dead model** — arguably the single largest available
upside in the product.

### 1f. Secondary defects found on the same path

- **Non-trading-day events are silently dropped.** 1,285 confirmed events (2.75%) have no
  `is_earnings_day` row at all; 318 of those fall inside the stock's own price history
  (144 Sat, 99 Sun, plus holidays). `merge_asof(direction="forward")` hands those rows to a
  Friday label the reaction functions never fire on. No warning is raised.
- **Cross-ticker forward fill.** `engineer_earnings_explosiveness_score` uses
  `df["reaction_entropy"].ffill()` — ungrouped. On a frame sorted by `[stock, date]` this
  carries the *previous ticker's* entropy across the boundary: **4,007 of 45,701 event rows
  (8.8%), 502 tickers** are scored with an `e4` term borrowed from another company. Weight
  is only 0.15 and entropy is clipped to 1.0, so the practical error is bounded at ~15
  score points on a stock's earliest events — real but second-order.

### 1g. Files / functions

| File | Function | Role |
|---|---|---|
| `ingestion/fetch_earnings_dates.py` | `fetch_one_earnings_dates` (L45-49) | **where the time is destroyed** |
| `ingestion/fetch_earnings_dates.py` | `ingest_all_earnings_dates`, `validate_upcoming_earnings_dates` | date-only paths |
| `utilities/db_utilities.py` / DuckDB | `earnings.earnings_date DATE` | no column exists to hold it |
| `utilities/data_utilities.py` | `merge_prices_earnings_dates` | forward `merge_asof` |
| `feature_engineering/pre_earnings_stock_features.py` | `engineer_earnings_windows` | `is_earnings_day` |
| `feature_engineering/post_earnings_stock_features.py` | `engineer_earnings_reactions`, `engineer_abs_reaction_3d` | **the misaligned target** |
| `feature_engineering/pre_earnings_stock_features.py` | `engineer_abs_reaction_p75_rolling`, `_p75`, `_median`, `_p90_rolling` | consume the biased target |
| `scoring/scoring_features.py` | `engineer_earnings_explosiveness_score`, `engineer_stock_bucket_lift` | 0.85 weight on the biased p75 |

### 1h. Recommended correction (design level, not implemented)

1. **Store the time.** Add `announce_time TIMESTAMP` (or `announce_window VARCHAR` ∈
   `{BMO, AMC, INTRADAY, UNKNOWN}`) to `earnings`. Stop calling `.dt.date` in
   `fetch_one_earnings_dates`; keep the NY-localized timestamp.
2. **Define an event anchor, not a date.** Introduce `reaction_anchor_date` = the last
   trading day whose close *precedes* the announcement:
   `AMC → D`, `BMO → previous trading day`, `INTRADAY → previous trading day` (conservative).
   Label `is_earnings_day` on the anchor row and compute all reactions from it. This makes
   `reaction_kd` mean the same thing for every event and leaves AMC values bit-identical, so
   the AMC half of history stays comparable.
3. **Backfill.** yfinance only returns ~25 quarters, covering **28% of confirmed events
   (13,100 of 46,665)**. For pre-2020 either (a) accept `UNKNOWN` and infer the window from
   the empirical day-0/day-+1 move ratio per stock-era, flagging low-confidence events, or
   (b) source times from a paid provider. **Do not** apply a single static per-ticker label —
   42% of sampled tickers change window within their own history.
4. **Re-fit the cut points after the fix.** 73/79 were selected on the distorted
   distribution and will be wrong once the BMO half expands ~56%.
5. Handle non-trading-day announcements explicitly (roll to the next session; count them).
6. Group the `reaction_entropy` ffill by stock.

---

## Issue 2 — Upcoming-event structural score staleness

### 2a. Exact current behavior

Four consumers build "the latest row per stock" the same way:

```
streamlit_dash/streamlit_export.py:73   latest = df.sort_values("date").groupby("stock").last()
analysis/save_predictions.py:43         latest = df.sort_values("date").groupby("stock").last()
report/report_builder.py:56             latest_per_stock = df.sort_values("date").groupby("stock").last()
cron/cron_weekly_digest.py              reads output/upcoming_df.parquet (written by the first)
```

`GroupBy.last()` is **skipna per column**. The row it returns is a composite:
`date` / `earnings_date` come from the true final price row, but `earnings_explosiveness_score`,
`earnings_explosiveness_bucket`, `..._structural`, `stock_bucket_lift`, `risk_score`,
`abs_reaction_p75_rolling`, `abs_reaction_p75`, `abs_reaction_median`, `reaction_entropy`
and `reaction_std` are NaN on every non-earnings row (they are only ever written into
`df.loc[earnings_mask]`), so `last()` reaches back to **the stock's most recent completed
earnings event**.

This is acknowledged in comments (`utilities/scoring_slice.py:119`,
`scoring/scoring_features.py:204`) as a mechanism, but its methodological cost does not
appear to have been priced.

The score at that event was itself built with `.shift(1)` — correct *for that event*. Used
for the next event it is therefore **exactly one earnings event stale**: the 28-event rolling
p75 window ends one quarter early.

### 2b. Concrete evidence (`audit/quantify_score_staleness.py`)

Raw rows from `full_df.parquet` — the last six price rows for ORCL are all-NaN in every
score column, and the value that ships is the June event's:

```
ORCL 2026-06-10  is_earnings_day=1  p75_roll 0.117590  score 98.292840   <-- last completed
ORCL 2026-09-04  is_earnings_day=0  p75_roll NaN       score NaN         <-- true last row
upcoming_df.parquet: ORCL, earnings_date 2026-09-10, score 98.292840     <-- inherited
```

Same for NKE (100.000000, from 2026-06-30) and COO (74.029239, from 2026-06-04).

Across the whole export:

```
upcoming events exported                                          495
  score identical to the stock's last COMPLETED event         495 (100.0%)
  median age of the event the score came from                  36 days
recomputing with the most recent completed event included:
  score changes for                                           53.9% of events
  mean |delta| 1.26 pts, max 15.48 pts
  STRUCTURAL TIER CHANGES                                       10 (2.0%)
  upcoming events whose latest reaction was >= 8% and is
    invisible to the shipped score                              70
```

Tier changes that the staleness is currently suppressing:

```
stock  earnings_date  shipped  correct   last reaction   tier
WDC     2026-11-05     76.27    86.09       15.57%       Elevated   -> High Alert
ISRG    2026-10-20     73.55    80.65       12.99%       Elevated   -> High Alert
CMG     2026-11-04     74.96    82.07        9.40%       Elevated   -> High Alert
CTVA    2026-11-03     65.82    73.41       12.35%       Normal     -> Elevated
UHS     2026-10-26     77.82    69.20        2.50%       Elevated   -> Normal
```

WDC printed a 15.6% earnings move last quarter and is being published as `Elevated`
because the score backing it was frozen before that move happened.

### 2c. What is *not* stale

`pre_earnings_drift_flag` and `surprise_momentum_flag` are string columns filled with `""`,
not NaN, and are deliberately computed on pre-earnings rows
(`engineer_pre_earnings_drift_flag`, L322-337), so `last()` does return their current value.
`days_to_earnings` is recomputed from `earnings_date - today` in `export_upcoming_df`.
`expected_move_pct` / `atm_iv` come from `join_iv` and are current-snapshot. So the
freshness of the shipped row is genuinely mixed: current drift/IV, one-quarter-stale
structure. That mixture is itself a hazard — nothing in the row records an as-of date for
the score.

### 2d. Severity

Medium-high, but bounded and much narrower than Issue 1. Mean drift is ~1.3 score points and
only 2% of events cross a structural boundary per run. It matters because:

- the misses are **anti-correlated with what the product is for**: a stock that just printed
  a violent reaction is precisely the one whose tier should rise, and it is exactly the one
  the staleness holds down (70 upcoming events currently carry an invisible ≥8% print);
- it is a **train/serve skew**. `testing/calibration.py` and `testing/backtesting.py`
  evaluate `earnings_explosiveness_bucket` on `is_earnings_day == 1` rows — the *fresh*
  as-of-event score. **No test anywhere measures the score the product actually publishes.**

### 2e. Files / functions

`streamlit_dash/streamlit_export.py:73,86` · `analysis/save_predictions.py:43` ·
`report/report_builder.py:56` · `cron/cron_weekly_digest.py` (consumer of `upcoming_df`) ·
`utilities/scoring_slice.py:attach_earnings_history` (relies on the same skipna behavior) ·
`scoring/scoring_features.py:engineer_earnings_explosiveness_score`,
`engineer_stock_bucket_lift`, `engineer_lift_adjusted_bucket` ·
`feature_engineering/pre_earnings_stock_features.py:engineer_abs_reaction_p75_rolling`.

### 2f. Recommended correction (design level, not implemented)

Score the upcoming event as a first-class row rather than harvesting a stale one.

1. Add an explicit **pending-event row** per stock: `is_upcoming_event = 1`, keyed on the
   next `earnings_date`, and let the same p75 / entropy / median / lift functions write into
   it using **all** completed events (no `shift(1)` — there is no outcome to leak from a
   report that has not happened).
2. Have the exporters select `is_upcoming_event == 1` explicitly and **ban
   `groupby(...).last()`** on score columns — it is a silent NaN-skipping join.
3. Stamp `score_asof_date` on every exported/persisted row so staleness is observable
   instead of inferred.
4. Backtesting must then evaluate the **pending-event** score, not the earnings-day score,
   or it continues to measure something the product never ships.

---

## 3. Tests that should exist before either fix

**Alignment (Issue 1)**
1. `test_announcement_time_survives_ingestion` — a fixture with 06:00 and 16:00 ET rows
   round-trips through the DB with the window intact.
2. `test_reaction_anchor_precedes_announcement` — for every event, the anchor close is
   timestamped strictly before the announcement.
3. `test_bmo_and_amc_span_equal_sessions` — both conventions cover 3 trading sessions.
4. `test_amc_reactions_unchanged_by_the_fix` — golden-file guard: AMC values must be
   bit-identical pre/post, so any distribution shift is provably the BMO half.
5. `test_no_event_silently_dropped` — count of confirmed events == count of labeled event
   rows, else fail with the offenders (would fail today on 318 rows).
6. `test_reaction_uses_no_future_price_beyond_k` and existing shift(1) leakage guards, re-run.
7. `test_entropy_ffill_does_not_cross_stock` (would fail today on 4,007 rows).

**Staleness (Issue 2)**
8. `test_upcoming_score_differs_from_last_event_when_history_changed` — synthetic stock whose
   last reaction is an outlier; the upcoming score must move. **Fails today.**
9. `test_upcoming_p75_window_includes_most_recent_completed_event`.
10. `test_no_groupby_last_on_score_columns` — static/AST guard over `streamlit_dash/`,
    `analysis/`, `report/`.
11. `test_exported_row_is_internally_consistent` — every field in an exported upcoming row
    carries the same `score_asof_date`.
12. `test_backtest_scores_the_pending_row` — the column calibration reads is the column the
    exporter ships.

**Baseline capture (do first, before touching anything)**
13. Freeze `testing/calibration.py` output for the current model to
    `audit/baseline_calibration_<commit>.csv`, so every later change is a diff against a
    known artifact rather than against memory.

---

## 4. Does either issue invalidate existing backtest conclusions?

**Issue 1 — yes, materially, though not in the direction of "the model doesn't work".**

Recomputed on 2015–2025 OOS events with a timing-aware target
(`audit/tier_by_timing.py`):

```
                      hit rate               lift vs market baseline
                  measured -> aware        measured -> aware
  market baseline   0.1086 -> 0.1650
  High Alert         0.403 -> 0.409          3.71x -> 2.48x
  Elevated           0.237 -> 0.243          2.18x -> 1.48x
  Normal             0.058 -> 0.126          0.53x -> 0.76x
  High Conviction    0.525 -> 0.525          4.83x -> 3.18x
```

- **Tier *hit rates* survive** almost unchanged — High Alert is 94% AMC and AMC is measured
  correctly, so "High Alert ≈ 41%" holds up.
- **Every *lift* claim is inflated by ~50%**, because the market baseline is understated
  (10.9% vs 16.5%) by the BMO half of the denominator. The documented **3.70x top-decile lift
  is ~2.48x**; high-conviction 4.83x → 3.18x. These numbers are in `CLAUDE.md`, the public
  track record and the marketing copy and should not be repeated until re-measured.
- **Capture is badly overstated as a description of the universe** — 37.9% overall, but 66.9%
  on AMC and 4.7% on BMO. The claim "captures 57.0% of ≥8% moves" describes AMC reporters.
- The `Normal` tier's real extreme rate is **12.6%**, not 5.8% — the "safe" tier is more than
  twice as dangerous as the tables say.
- Anything selected on the distorted distribution — 73/79 cut points, `LIFT_TO_*` gates,
  the 0.12 p75 ceiling, the 0.85/0.15 weights — is fitted to a mismeasured target and must
  be re-fit, not carried over.

**Issue 2 — no, it does not invalidate the backtests; it makes them unrepresentative.**
Calibration/backtesting read earnings-day rows, whose score is fresh and leakage-free, so the
published historical numbers are internally valid. But the product ships the *stale* score,
which no test measures. The backtest is measuring a model adjacent to the one in production.
Expect the shipped model to be slightly *worse* than the tables (2% of tiers wrong per run,
biased against stocks that just moved).

## 5. Does either issue affect the prospective prediction archive?

**`db/predictions.duckdb` holds 10 rows**, all written `2026-08-31`, model_version `0.3.1`,
commit `f3dd1e2`, for events 2026-09-01…09-03. So the exposure is one week.

- **Issue 2 affects all 10** — every `tier` / `risk_score` there was inherited from the
  stock's previous earnings event (`save_predictions.py:43`).
- **Issue 1 affects them structurally and will affect their scoring.** The tier skew is
  visible even in this tiny sample: all four `High Alert` calls (DELL, PANW, NTAP, LULU) and
  both `Elevated` (AVGO, HPE) are AMC reporters; the three `Normal` calls include the BMO
  names (MDT, CPB, BF-B). When these are graded, the realized `abs_reaction_3d` for any BMO
  name in the set will be measured on the wrong three sessions.

**Recommendation:** do not discard the archive — 10 rows is not a track record worth
defending, and the cost of restarting is one week. Stamp the existing rows
`model_version = 0.3.1-preaudit`, treat them as void for any published track record, and
begin the real prospective archive after the fixes land. Any public claim built on
`marketing/generate_public_track_record.py` should be paused until then.

---

## 6. Suggested order of work

1. Freeze the baseline calibration artifact (test #13) — nothing else is measurable without it.
2. Write the failing tests (#1–#5, #8, #10) so both issues are pinned before any fix.
3. Fix Issue 2 (contained: one pending-event row + four call sites). Cheap, and it removes a
   confound from the Issue 1 measurement.
4. Fix Issue 1: schema + ingestion + anchor, then backfill 2020+, then decide the pre-2020
   inference policy.
5. **Re-fit** 73/79, the lift gates and the p75 ceiling on the corrected target. Only then
   re-open GTM claims.
