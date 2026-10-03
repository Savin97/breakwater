# Point-in-time accounting fundamentals vs Model C — results

Research only. No production module, score, threshold, report, prediction archive or
`MODEL_VERSION` was touched. Rules: `PREREGISTRATION.md`. Sections 1–3 were written **before**
`evaluate.py` was first run; no outcome had been read.

```bash
export SEC_USER_AGENT="<name> <email>"
PYTHONPATH=. .venv/bin/python -m research.fundamentals.acquire --finish   # Company Facts snapshot (515 requests, ~6 min)
PYTHONPATH=. .venv/bin/python -m research.fundamentals.build              # quarter records + features (~8 min)
PYTHONPATH=. .venv/bin/python -m research.fundamentals.audit --coverage
PYTHONPATH=. .venv/bin/python -m research.fundamentals.audit --verify     # 40 events vs EDGAR instance documents
PYTHONPATH=. .venv/bin/python -m research.fundamentals.audit --gates
PYTHONPATH=. .venv/bin/python -m research.fundamentals.evaluate           # 2017-2025
PYTHONPATH=. .venv/bin/python -m research.fundamentals.evaluate --holdout # 2026, once
.venv/bin/python -m pytest testing/test_fundamentals.py -q
```

Outputs: `output/fundamentals/` (gitignored). Raw SEC data: `data/vendor/sec/xbrl_snapshots/`
(gitignored) and the pilot's archive (instance documents fetched by the audit).

---

## 1. How a point-in-time accounting value is built

**Source.** SEC XBRL Company Facts for the 515 CIKs of the SEC pilot's identity chains,
snapshot `sec_20261002T213221Z` (sha `b2d13efd…a254`; sealed, read-only). Company Facts lists
every value **once per filing that reported it**, with that filing's accession and filing
date. Apple's 2017-12-30 quarter revenue, for example, appears under four accessions (its own
10-Q and three later filings that repeated it).

**The rule: one value, one accession.** Each event uses the pilot's selected filing S, the
latest original 10-Q/10-K eligible at the call cutoff by the pilot's rule. Each quarter's
record is built **only** from facts carrying that quarter's original filing's accession. AES
shows why. Its April–June 2016 revenue was **$3,229M** as filed. The 2018 10-K restated it to
**$2,452M** (discontinued operations). Company Facts serves both today. A 2016 event sees only
$3,229M, because the $2,452M entry carries a 2018 accession.

| step | rule |
|---|---|
| filings | the stock's original 10-Q/10-K/10-QT/10-KT, through the pilot's date-bounded CIK chain (GOOGL → Google Inc. before 2015-10-02, DD → E.I. du Pont before 2017-09-01, …); amendments never |
| quarter value | the filing's 3-month fact (70–120 days, ends at the report period ±10 d). 10-K: its own 3-month fact, else FY − 9M YTD of the same fiscal year's original Q3 10-Q (same tag, same FY start ±7 d). A YTD fact is never a quarter |
| year-over-year | the **same filing's** prior-year comparative column (same tag, same length, ending 358–372 d earlier): one accession, one accounting basis |
| balance sheet | instants at the report period; liabilities = `Liabilities`, else `LiabilitiesAndStockholdersEquity` − total equity |
| TTM net income / CFO | 10-K: FY. 10-Q: YTD + prior original 10-K FY − prior-year YTD comparative, one tag across all three |
| units | USD only; conflicting values for one (accession, tag, period) → missing |
| known_at | later of SEC's submissions filing date and Company Facts' `filed` (they differ by one business day for 7 filings); eligible at the cutoff — asserted for all 22,847 rows |

**Size of the build.** 1,070,340 facts on the 12 concept keys → 1,047,058 on original periodic
filings → 31,137 quarter records. XBRL exists for 95% of 2011 filings and ≥ 99% from 2012.
22,847 of 22,947 SEC-population events have a record for S. The other 100 have no selected
filing (51: new listings, NXPI's foreign-filer years) or no XBRL under S's CIK (49: mostly MAA
and REG, whose combined REIT/partnership filings sit under the partnership's CIK in Company
Facts). They are left missing.

## 2. Coverage (population: Phase 3's on the current frame, 21,761 events; 2017–2025: 16,127)

| feature | 2014–2026 | 2017–2025 | BMO | AMC |
|---|---|---|---|---|
| `log_assets` (size control) | 99.5% | 99.6% | 99.7% | 99.2% |
| A `revenue_growth_volatility` | 90.9% | 92.3% | 89.1% | 93.3% |
| B `operating_margin_volatility` | 71.4% | 72.6% | 69.4% | 74.0% |
| C `abs_operating_margin_change_yoy` | 72.3% | 73.5% | 70.5% | 74.8% |
| D `leverage` | 99.4% | 99.6% | 99.6% | 99.2% |
| E `abs_accruals` | 91.4% | 92.1% | 91.2% | 91.6% |
| F `wc_to_assets` | 82.3% | 82.5% | 84.0% | 80.0% |
| F `abs_wc_change_yoy` | 82.0% | 82.3% | 83.7% | 79.7% |
| (not used) `inventory_to_assets` | 56.5% | 56.4% | 59.5% | 52.3% |

**By year:** flat 2014–2019, then up in 2020. Revenue volatility 82–84% → 94–98%; margin
features 65–67% → 74–77%. ASC 606 (2018) moved many filers onto
`RevenueFromContractWithCustomer…`, and the trailing window needs two years of quarters
on it. Every year clears 60% except margin volatility (65–66%, still above).

**By sector** (the real limitation). Operating income is reported by only 31% of Financials
events and 62% of Real Estate, and by 51–56% of Energy (many have no operating-income
subtotal). Working capital covers 28% of Financials and Real Estate (no classified balance
sheet), against 90–100% elsewhere. Revenue, leverage and accruals cover 81–100% in every
sector. **So the margin and working-capital blocks are mostly non-financial companies.**

**Age of the information at the call.** S was filed a median **84 days** before the cutoff
(p10 60, p90 93). Its quarter ended a median **115 days** before the cutoff (p10 104, p90 126).
It describes the quarter *before* the one being reported, like the SEC pilot's text.
Quarter records up to S per stock: median 41.5 (p10 28).

**Missing reasons** (`missing_reasons.csv`, whole population):

| feature | main reasons |
|---|---|
| revenue volatility | < 6 of the 8 trailing quarters with revenue and its comparative: 1,900; no selected XBRL: 60 |
| margin volatility | < 6 quarters with operating income: 6,150 |
| margin change | no operating-income line: 4,898; no quarter revenue: 988; no comparative / revenue ≤ 0: 84 |
| leverage | no assets/liabilities: 70 |
| accruals | no matching TTM net income: 1,167; CFO: 654 (incl. a different tag in the 10-K than the 10-Q, e.g. BR 2015) |
| working capital | no classified balance sheet: 3,787 |

Routes at S: revenue tag `Revenues` 10,099 / `RevenueFromContract…Excl.` 6,399 /
`SalesRevenueNet` 2,747 / `…Incl.` 709 / `RevenuesNetOfInterestExpense` 161; Q4 derived as
FY − 9M for 2,688 events; liabilities by the balance identity for 6,711.

## 3. Reliability audit (before outcomes)

**Re-derived from the original filings, not Company Facts.** 40 events, stratified (4 year
buckets × 10-Q/10-K × financial/other) plus forced cases (3 derived Q4, 3 liabilities by
identity, 3 predecessor-CIK stocks). For every filing behind every feature (S, each trailing
quarter, the Q3 10-Q behind a derived Q4, the prior 10-K behind a TTM, the year-earlier
balance sheet), `audit.py` fetched the EDGAR index page and the filing's own XBRL instance
document. It parsed contexts and units itself and re-read each value. **325 filings, 7,924
checks:**

| check | agree |
|---|---|
| registrant (`dei:EntityRegistrantName` vs SEC name history of the chain's CIKs) | 452 / 452 |
| document type = form | 452 / 452 |
| period end (`DocumentPeriodEndDate`) = our report period | 451 / 452 |
| index-page filing date ≤ our known_at | 452 / 452 |
| every value (quarter, comparative, 9M, FY, YTD, instants), same tag, USD, no dimensions | all |
| features recomputed from the verified numbers | **360 / 360** (all 9 features × 40 events) |
| known_at before the event's cutoff | 40 / 40 |

The one period mismatch: XOM's 2012 Q3 10-Q, where SEC's `reportDate` says 2012-11-06 and the
filing says 2012-09-30. The record finds no facts at that date and is **missing**, not wrong.
120 of 31,137 records (0.4%) have no balance sheet at SEC's period. Those are the same kind of
error, and they are left missing.

Two problems the first audit run found were **in the checker, not the data**. Combined filings
(EQR + its operating partnership, Alliant + its utilities) carry several registrant names; the
filer's own is the undimensioned one. And the checker did not enforce the build's
one-tag-per-TTM rule (BR). Both checker rules were aligned with the build and re-run on the
same sample. The build was not changed.

**By eye**, three filings against their face statements:
- DRI 10-Q of 2024-02-25 (a 52/53-week year): sales 2,974.8 vs 2,786.2, assets 11,358.2,
  liabilities 9,177.3, 39-week CFO 1,195.7 ($M). All exact.
- JPM 10-K 2013: derived Q4 revenue $23.16B, which is JPM's reported 4Q13.
- DD 10-Q Q3 2017: this one exposes a real limitation. DowDuPont's comparative column is
  **Dow's** prior year (Dow was the accounting acquirer), so DD's YoY revenue growth that
  quarter is +23% from the merger. The figure is correct as filed and public at the time,
  but mergers make YoY growth measure deals, not volatility. No adjustment was made; this is
  noise, and it is the same for every merger in the sample.

### Gates (`gates.json`)

| feature | coverage 2017–2025 (≥ 60%) | verified present / agree (≥ 90%) | gate |
|---|---|---|---|
| A revenue volatility | 92.3% | 34 / 100% | **pass** |
| B margin volatility | 72.6% | 22 / 100% | **pass** |
| C abs margin change | 73.5% | 24 / 100% | **pass** |
| D leverage | 99.6% | 40 / 100% | **pass** |
| E abs accruals | 92.1% | 38 / 100% | **pass** |
| F working capital level / change | 82.5% / 82.3% | 29 / 100% | **pass** |
| size `log_assets` | 99.6% | 40 / 100% | **pass** |

Every pre-registered model is run; nothing was dropped. Inventory was set aside in the
pre-registration on coverage (56%), not on any result.

---

*Sections 4 onward were written after the outcomes were read.*

## 4. Main comparison, 2017–2025 walk-forward

Each block is fitted on identical rows: those with every input of its four models.
Blocks differ, so compare a model only with the C beside it. C's AUC on the broadest
block (leverage, 16,060 events) is 0.723, the same as Phase 3's 0.723.

| block | OOF events / stocks | base | model | top-10% hit | top-10% capture | top-10% lift | top-20% hit | top-20% capture | top-20% lift | AUC | PR-AUC | Brier | log loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A revenue | 14,875 / 469 | 0.182 | C | 0.429 | 0.237 | 2.36 | 0.369 | 0.406 | 2.03 | 0.7211 | 0.3553 | 0.1353 | 0.4283 |
| | | | C+revenue | 0.427 | 0.235 | 2.35 | 0.369 | 0.407 | 2.03 | 0.7208 | 0.3549 | 0.1353 | 0.4284 |
| B/C margin | 11,496 / 376 | 0.196 | C | 0.431 | 0.220 | 2.20 | 0.387 | 0.396 | 1.98 | 0.7200 | 0.3646 | 0.1434 | 0.4474 |
| | | | C+margin | 0.437 | 0.223 | 2.23 | 0.386 | 0.394 | 1.97 | 0.7197 | 0.3628 | 0.1435 | 0.4476 |
| D leverage | 16,060 / 473 | 0.178 | C | 0.430 | 0.241 | 2.41 | 0.366 | 0.411 | 2.05 | 0.7232 | 0.3529 | 0.1330 | 0.4224 |
| | | | C+leverage | 0.430 | 0.241 | 2.41 | 0.365 | 0.410 | 2.05 | 0.7234 | 0.3530 | 0.1330 | 0.4223 |
| E accruals | 14,850 / 472 | 0.176 | C | 0.434 | 0.246 | 2.46 | 0.368 | 0.418 | 2.09 | 0.7280 | 0.3562 | 0.1315 | 0.4182 |
| | | | C+accruals | 0.432 | 0.245 | 2.45 | 0.371 | 0.421 | 2.10 | 0.7277 | 0.3568 | 0.1315 | 0.4183 |
| F working capital | 13,266 / 394 | 0.195 | C | 0.433 | 0.222 | 2.22 | 0.384 | 0.393 | 1.97 | 0.7138 | 0.3641 | 0.1435 | 0.4486 |
| | | | C+workingcap | 0.435 | 0.223 | 2.23 | 0.383 | 0.392 | 1.96 | 0.7137 | 0.3641 | 0.1435 | 0.4488 |
| compact | 10,093 / 354 | 0.201 | C | 0.439 | 0.218 | 2.18 | 0.396 | 0.394 | 1.97 | 0.7189 | 0.3701 | 0.1461 | 0.4542 |
| | | | C+compact | 0.440 | 0.219 | 2.19 | 0.399 | 0.397 | 1.98 | 0.7176 | 0.3682 | 0.1464 | 0.4550 |

Size-controlled models (C+size, C+size+X) are in `main_models.csv`.

## 5. Deltas vs C, stock-clustered bootstrap (500 reps), percentage points

| comparison | Δ top-10% capture [95% CI] | Δ top-20% capture [95% CI] | Δ top-10% hit | Δ AUC [95% CI] |
|---|---|---|---|---|
| C+revenue vs C | −0.15 [−0.46, +0.25] | +0.04 [−0.36, +0.28] | −0.27 | −0.0003 [−0.0006, +0.0001] |
| C+margin vs C | +0.31 [−0.60, +0.61] | −0.18 [−0.65, +0.60] | +0.61 | −0.0003 [−0.0010, +0.0004] |
| C+leverage vs C | 0.00 [−0.19, +0.35] | −0.10 [−0.39, +0.38] | 0.00 | +0.0002 [−0.0002, +0.0006] |
| C+accruals vs C | −0.08 [−0.43, +0.49] | +0.31 [−0.39, +0.70] | −0.13 | −0.0004 [−0.0012, +0.0006] |
| C+workingcap vs C | +0.08 [−0.62, +0.64] | −0.08 [−0.57, +0.46] | +0.15 | −0.0001 [−0.0014, +0.0011] |
| C+compact vs C | +0.05 [−1.00, +0.82] | +0.30 [−0.95, +0.69] | +0.10 | −0.0013 [−0.0032, +0.0006] |
| *size-controlled:* C+size+X vs C+size | revenue −0.04, margin +0.13, leverage +0.17, accruals −0.08, workingcap −0.15, compact +0.30; every CI crosses 0 | | | |
| C+size vs C (size alone) | −0.07 to +0.27 by block; every CI crosses 0 | | | |

No model is worth keeping: the largest Δ top-10% capture is +0.31 pt (bar: +1.0 with CI > 0).
Every capture CI and every AUC CI contains zero. Size itself adds nothing beyond C either.
C's two inputs already carry it (`log_assets` vs C's probability: Spearman −0.31).

## 6. Stability

**BMO / AMC** (`main_models.csv`): BMO AUC moves by −0.002…+0.000 and AMC AUC by
−0.002…+0.000 across the six family models. Top-10% capture by window moves by at most
±0.4 pt (compact AMC +0.4, BMO +0.1). No window gains.

**By year**, Δ top-10% capture vs C (points, 2017→2025):

| model | per year | years > 0 |
|---|---|---|
| C+revenue | 0.0, 0.0, 0.0, −0.3, −0.4, 0.0, 0.0, −0.7, +0.7 | 1/9 |
| C+margin | 0.0, 0.0, 0.0, 0.0, 0.0, +0.7, 0.0, −0.3, −0.9 | 1/9 |
| C+leverage | 0.0, +0.3, 0.0, 0.0, +0.9, 0.0, 0.0, −0.2, +0.7 | 3/9 |
| C+accruals | 0.0, +0.4, 0.0, +0.3, −0.5, 0.0, −0.3, 0.0, 0.0 | 2/9 |
| C+workingcap | −0.6, −1.4, +0.8, −0.4, −2.4, 0.0, 0.0, +0.5, +0.8 | 3/9 |
| C+compact | −1.6, −0.5, +0.6, −1.4, −0.6, 0.0, −0.4, −0.3, −0.3 | 1/9 |

Most years are exactly 0.0: the model picks the same top 10% as C. Yearly Δ AUC is within
±0.004 for every family model; compact's worst year is −0.010 (2021).

## 7. 2026 holdout (scored once)

Frozen before scoring (`selection.json`): **C+compact** (pre-registered) and **C+margin**
(largest pre-2026 Δ top-10% capture, +0.31 pt). Trained on every block row with an endpoint
before 2026-01-01.

| block (2026 events / stocks, base) | model | top-10% capture | top-20% capture | AUC | BMO AUC | AMC AUC |
|---|---|---|---|---|---|---|
| compact (948 / 345, 0.296) | C | 0.174 | 0.335 | 0.6759 | 0.6649 | 0.6776 |
| | C+compact | 0.171 (Δ −0.4 [−1.5, +1.8]) | 0.338 (Δ +0.4 [−1.5, +1.8]) | 0.6744 (Δ −0.0015 [−0.0060, +0.0030]) | 0.6619 | 0.6756 |
| | C+size+compact vs C+size | Δ +0.4 [−1.6, +1.9] | Δ +1.1 [−1.6, +2.2] | Δ −0.0017 | | |
| margin (1,032 / 360, 0.288) | C | 0.182 | 0.337 | 0.6864 | 0.6646 | 0.7012 |
| | C+margin | 0.185 (Δ +0.3 [−1.0, +1.1]) | 0.340 (Δ +0.3 [−1.4, +1.3]) | 0.6887 (Δ +0.0024 [+0.0004, +0.0046]) | 0.6683 | 0.7018 |

The only CI that excludes zero anywhere in this test is C+margin's 2026 AUC (+0.002). That is
an AUC-only gain of the kind the pre-registration says does not count, and the same model's
pre-2026 AUC delta was −0.0003.

## 8. Is anything here just size or sector? (POST HOC, not part of the decision)

`posthoc_univariate.csv`. Within each decile of C's out-of-fold probability (leverage block),
the AUC of each feature alone for ≥ 8% moves:

| feature | within-C-decile AUC | Spearman with C | Spearman with log assets |
|---|---|---|---|
| revenue growth volatility | 0.483 | +0.16 | +0.09 |
| margin volatility | 0.486 | +0.18 | +0.12 |
| abs margin change | 0.490 | +0.16 | +0.04 |
| leverage | 0.480 | −0.15 | +0.29 |
| abs accruals | 0.508 | +0.27 | −0.19 |
| working capital / assets | 0.527 | +0.35 | −0.39 |
| abs WC change | 0.522 | +0.18 | −0.19 |
| log assets (size) | 0.474 | −0.31 | 1 |

- The instability measures **do** relate to C (unstable businesses already have violent
  earnings histories). Once inside a C decile, they lean slightly the **wrong** way
  (0.48–0.49).
- The largest departure is working-capital level (0.527). It is a company-type echo: high
  working capital means small, cash-rich tech and health-care names (−0.39 with size, +0.35
  with C). It did not move top-10% capture (+0.08 pt).
- The compact block is 55% Industrials + IT + Health Care + Consumer Discretionary and only
  8% Financials + Real Estate. The test answers the question mainly for non-financial
  companies, where accounting data is best. A financial-sector version would need different
  line items (bank revenue, no operating income, no classified balance sheet), which is the
  bespoke work the brief excluded.

## 9. Conclusion

- **Worth keeping:** none. No family and not the compact set moves top-10% or top-20% capture
  pre-2026 (largest +0.31 pt, all CIs across zero) or in 2026.
- **Size is not hiding a result:** with `log_assets` in both models every delta is still
  within ±0.3 pt. Size alone adds nothing beyond C.
- **Why, in plain terms:** the newest accounting data at the call is a median 84 days old and
  describes a quarter that ended 115 days earlier. Whatever instability it shows has already
  produced violent past reactions and elevated recent volatility, which C measures directly.
- **The data was not the problem.** Every value behind every feature was re-derived from the
  original filings with 100% agreement, and coverage clears the bar for every family. A null
  here is a real null, not a construction failure.

**FUNDAMENTALS ADD NO USEFUL VALUE**
