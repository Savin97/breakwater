# SEC-derived features vs Model C — results

Research only. No production module, score, threshold, report, prediction archive or
`MODEL_VERSION` was touched; nothing here is in live scoring. Rules: `PREREGISTRATION.md`
(+ Amendment 1, made before any outcome was read). Sections 1–2 were written **before**
`evaluate.py` was first run.

```bash
PYTHONPATH=. .venv/bin/python -m research.sec_features.frame          # current corrected frame (~3 min)
PYTHONPATH=. .venv/bin/python -m research.sec_filings_pilot.build     # pilot re-run on it
PYTHONPATH=. .venv/bin/python -m research.sec_features.refresh        # old vs new SEC association
export SEC_USER_AGENT="<name> <email>"
PYTHONPATH=. .venv/bin/python -m research.sec_features.build --fetch       # ~23k releases (~2 h)
PYTHONPATH=. .venv/bin/python -m research.sec_features.build --fetch-ex99  # ~14k 8-K indexes
PYTHONPATH=. .venv/bin/python -m research.sec_features.build               # features (~15 min)
PYTHONPATH=. .venv/bin/python -m research.sec_features.audit [--samples --seed N]
PYTHONPATH=. .venv/bin/python -m research.sec_features.evaluate            # 2017-2025
PYTHONPATH=. .venv/bin/python -m research.sec_features.evaluate --holdout  # 2026, once
.venv/bin/python -m pytest testing/test_sec_features.py -q
```

Outputs: `output/sec_features/` (gitignored). Raw filings stay in `data/vendor/sec/`
(gitignored). The run added ~23,000 release exhibits and ~37,000 index pages to the SEC
cache, all fetched at ≤ 8 requests/s (SEC's limit is 10); 503s were retried, none failed.

---

## 1. SEC mapping refresh after the 2026-09-29 date corrections

Phase 3's frame (and the SEC pilot) predated 103 earnings-date corrections. `frame.py`
rebuilt the frame with Phase 3's own code and Benzinga snapshot on the corrected
`full_df`; the pilot was re-run on it with the same SEC snapshot, identity chains and
timing rule. Comparison (`refresh_summary.json`; identity/clock/filing columns only):

| | |
|---|---|
| SEC-population events before → after | 22,837 → 22,947 |
| existing events whose cutoff, 10-Q/10-K, 8-K set or identity changed | **0** |
| earnings date moved | 1 (AES 2026-08-03 → 08-04; same cutoff, same filings) |
| events added | 110 = the 103 corrected dates (their old wrong dates had failed the timing match, so they were never in the population) + 7 newly timed |
| events dropped / new unresolved | 0 / 1 (DASH 2021-02-25, first report after listing) |

Small — the corrections only add events. The pilot's RESULTS.md §12 records it.

## 2. Features: coverage and extraction reliability (before outcomes)

### Previous-quarter release

For each event, the release 8-K of the stock's **previous** Breakwater event (Item 2.02,
−1…+4 days of its announcement), eligible at this event's cutoff under the pilot rule,
≤ 140 days old; its lowest-numbered EX-99 exhibit. The event's own release can never
qualify (filed after the cutoff; tested).

Denominator: 22,896 SEC-mapped events, 2014–2026, BMO/AMC.

| step | share |
|---|---|
| previous release 8-K matched and eligible | 98.9% |
| previous release usable (EX-99 found, ≥ 100 prose words) | 98.4% |
| second-prior release usable | 98.3% |
| **both usable** | **97.3%** |

Missing previous release: no Item 2.02 8-K near the previous event 203, no EX-99 78,
older than 140 days 47 (a Breakwater event missing in between), too little prose 22, first
event 9. Coverage is flat across years (96.0–98.6%) and BMO/AMC (97.3/97.4%). Previous
release age at cutoff: median 86 days.

### Reliability gates (`manual_audit.csv`, `gates.json`)

Audited by reading the extracted text for stratified samples (four year buckets).

| gate | round 1 | round 2 (after Amendment 1) | decision |
|---|---|---|---|
| release matching (≥ 90% right release) | 40/40 | — | **pass** |
| `guidance_present` (≥ 80% correct both ways) | 14/20 and 13/20 | 16/20 and **14/20** | **dropped** |
| EPS / revenue guidance width (coverage ≥ 40% and ≥ 90% correct) | 19/20 correct, but EPS ranges in only ~20% of events | 24.6% after amendment | **dropped** |
| uncertainty dictionary (≤ 50% non-business hits) | 42/61 = 69% | **43/91 = 47%** | **pass (barely)** |

What went wrong with guidance: releases phrase outlook in too many ways — ranges in tables
under an "Outlook" heading, "±" ranges, targets lists, "raises fiscal-year outlook" next
to "exceeded the high end of guidance". A fixed rule either over-reads past-tense
sentences ("lower revenues", "accounting guidance") or misses real outlook sections.
Getting this right needs either a much larger hand-built parser or a learned classifier —
exactly what the brief said not to build. Widths are accurate when found but exist for a
quarter of events (EPS ranges: Utilities 58%, Industrials/Health Care ~31%, Energy and
Communication Services ~0%).

The uncertainty count is still noisy after the amendment: about half its hits are
safe-harbour lists, risk-management headings, "uncertain tax positions", "blood pressure".
It is kept because it passed the pre-set bar, and `uncertainty_change` (same company,
consecutive quarters) cancels much of the house-style boilerplate.

### Feature coverage (share of the 22,896 events)

| feature | available |
|---|---|
| `uncertainty_rate` | 98.4% |
| `uncertainty_change`, `release_text_change` | 97.3% |
| every 8-K feature, `num_8k_ex99_last_30d` | 100% |
| (dropped) `guidance_present` 98.4%, EPS width 24.2%, revenue width 12.9% | |

Distributions: `release_text_change` median 0.093 (IQR 0.058–0.140); 8-Ks since the
periodic report median 2; in the last 30 days 0 (mean 0.62); recent-14-day flags are rare
(2.02: 1.7%, 7.01: 6.2%, 8.01: 5.6%).

### Models, as fixed by the gates

| spec | inputs beyond C |
|---|---|
| C | — (`log hist_mean_abs`, `log vol_30d`) |
| C+uncertainty | `log1p uncertainty_rate`, `uncertainty_change` |
| C+novelty | `release_text_change` |
| C+8K | 10 8-K metadata features |
| C+ex99 | `log1p num_8k_ex99_last_30d` |
| **C+compact** | `uncertainty_change`, `release_text_change`, `log1p num_8k_last_30d`, `has_recent_2_02_14d` (guidance removed by its gate) |

C+guidance and the width block are not run.

### Direction source data (kept, not evaluated)

Per release: raise/lower/reaffirm-guidance phrase counts (15.2% / 2.6% / 9.7% of releases
have one), demand (45%), margin (75%) and inventory (56%) mentions, and up to five
guidance passages — in `output/sec_features/releases.parquet`.

---

*Sections 3–7 were written after the outcomes were read.*

## 3. Main comparison, 2017–2025 walk-forward (identical rows)

Phase 3 population on the current frame (completed, BMO/AMC, target available, ≥ 8 prior
outcomes, C inputs) ∩ SEC-mapped ∩ both releases usable: **15,777 out-of-fold events, 473
stocks, base rate 0.178** (event-id hash `f6ca329e93425c78`, identical for every spec). C's
AUC here, 0.723, matches Phase 3's 0.723.

| model | top-10% hit | top-10% capture | top-10% lift | top-20% hit | top-20% capture | top-20% lift | AUC | PR-AUC | Brier | log loss | BMO AUC | AMC AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C** | 0.429 | 0.241 | 2.41 | 0.366 | 0.411 | 2.06 | 0.7226 | 0.3526 | 0.1330 | 0.4224 | 0.6901 | 0.7380 |
| C+uncertainty | 0.430 | 0.242 | 2.42 | 0.365 | 0.410 | 2.05 | 0.7224 | 0.3524 | 0.1330 | 0.4225 | 0.6898 | 0.7379 |
| C+novelty | 0.428 | 0.241 | 2.41 | 0.366 | 0.412 | 2.06 | 0.7225 | 0.3525 | 0.1330 | 0.4225 | 0.6900 | 0.7378 |
| C+8K | 0.433 | 0.243 | 2.43 | 0.364 | 0.409 | 2.05 | 0.7235 | 0.3529 | 0.1329 | 0.4221 | 0.6914 | 0.7385 |
| C+ex99 | 0.428 | 0.241 | 2.40 | 0.365 | 0.410 | 2.05 | 0.7230 | 0.3531 | 0.1329 | 0.4223 | 0.6906 | 0.7383 |
| C+compact | 0.428 | 0.241 | 2.40 | 0.362 | 0.407 | 2.03 | 0.7231 | 0.3529 | 0.1330 | 0.4223 | 0.6909 | 0.7381 |

## 4. Deltas vs C, stock-clustered bootstrap (500 reps), points

| vs C | Δ top-10% capture [95% CI] | Δ top-20% capture [95% CI] | Δ top-10% hit | Δ AUC [95% CI] |
|---|---|---|---|---|
| C+uncertainty | +0.07 [−0.30, +0.28] | −0.07 [−0.39, +0.23] | +0.13 | −0.0002 [−0.0005, +0.0002] |
| C+novelty | −0.04 [−0.26, +0.23] | +0.04 [−0.28, +0.27] | −0.06 | −0.0001 [−0.0004, +0.0002] |
| C+8K | +0.21 [−0.67, +0.85] | −0.18 [−0.73, +0.89] | +0.38 | +0.0009 [−0.0011, +0.0032] |
| C+ex99 | −0.07 [−0.31, +0.46] | −0.07 [−0.60, +0.39] | −0.13 | +0.0004 [−0.0004, +0.0012] |
| C+compact | −0.07 [−0.55, +0.46] | −0.43 [−0.88, +0.52] | −0.13 | +0.0005 [−0.0008, +0.0019] |

No delta comes near the +1.0-point bar; every CI contains zero; no AUC CI excludes zero.

**Broad 8-K block** (no release requirement, 16,115 OOF events): C+8K vs C top-10% capture
+0.35 pt [−0.58, +0.87], top-20% −0.03 pt [−0.75, +0.88], AUC +0.0009 [−0.0010, +0.0031].

## 5. BMO / AMC and yearly stability

| | C | C+uncertainty | C+novelty | C+8K | C+ex99 | C+compact |
|---|---|---|---|---|---|---|
| BMO AUC (9,108) | 0.6901 | 0.6898 | 0.6900 | 0.6914 | 0.6906 | 0.6909 |
| BMO top-10% capture | 0.215 | 0.213 | 0.215 | 0.208 | 0.213 | 0.212 |
| AMC AUC (6,669) | 0.7380 | 0.7379 | 0.7378 | 0.7385 | 0.7383 | 0.7381 |
| AMC top-10% capture | 0.224 | 0.222 | 0.222 | 0.217 | 0.226 | 0.226 |

Yearly Δ top-10% capture vs C (points): C+8K +1.1, +0.3, −1.1, −0.6, −0.4, −0.3, −0.3,
+0.2, +1.5 (2017→2025); C+compact 0.0, +0.7, −1.5, −0.3, +1.3, −0.5, 0.0, 0.0, +0.5. Signs
flip year to year; no model gains in a majority of years. Yearly Δ AUC is within ±0.005 for
every model and year (`main_yearly.csv`).

## 6. 2026 holdout (scored once)

Frozen before scoring (`selection.json`): **C+compact** (pre-registered) and **C+8K** (largest
pre-2026 Δ top-10% capture). Trained on every row with an endpoint before 2026-01-01.
1,357 events, 475 stocks, base rate 0.248.

| | top-10% capture | top-20% capture | AUC | BMO AUC | AMC AUC |
|---|---|---|---|---|---|
| C | 0.217 | 0.380 | 0.7184 | 0.6909 | 0.7381 |
| C+compact | 0.217 (Δ 0.0 [−1.5, +0.9]) | 0.383 (Δ +0.3 [−1.2, +1.2]) | 0.7189 | 0.6920 | 0.7392 |
| C+8K | 0.217 (Δ 0.0 [−1.8, +1.2]) | 0.383 (Δ +0.3 [−1.5, +1.8]) | 0.7198 | 0.6984 | 0.7336 |

Both SEC models select exactly the same top 10% as C.

## 7. Why (POST HOC, not part of the decision)

`posthoc_univariate.csv`: within each decile of C's out-of-fold ranking, how well does each
feature alone separate ≥ 8% moves? AUC 0.475–0.513 for every feature. The largest
departures are inverse and small: more 8-Ks since the last 10-Q (0.475), any Item 8.01
(0.482) — firms that file a lot are large and move less, which C's two inputs already
carry (Spearman with C −0.11 / −0.13). Release text change, uncertainty level and change sit
at 0.50 ± 0.01. Even the dropped `guidance_present` is 0.513.

## 8. Conclusion

- **Worth keeping:** none. Nothing improves the top of the ranking pre-2026 or in 2026.
- **8-K metadata** is the only family with any signal, and it is a company-size echo
  already in C.
- **Release text** — uncertainty words, quarter-to-quarter change — adds nothing measurable.
  The previous release is ~86 days old at the call and has been fully absorbed by the
  market and by the stock's own volatility.
- **Guidance** could not be extracted reliably by fixed rules (non-detections 70% correct
  after one revision); guidance ranges exist for only a quarter of events.
- **Deeper NLP?** Not justified on this evidence for the magnitude target. The cheap
  features found nothing within C's deciles, and the strongest text item (guidance) is not
  even reliably detectable without a learned parser. A learned parser would be justified
  only by a different question — direction (guidance raised/cut; the source data is kept) —
  or a fresher source than last quarter's release, which SEC filings before the call do
  not provide.

**SEC FEATURES ADD NO USEFUL VALUE**
