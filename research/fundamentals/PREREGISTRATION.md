# Point-in-time accounting fundamentals vs Model C — pre-registration

Written 2026-10-03, **before any outcome of this test was read.** As the brief ordered, the
point-in-time feasibility audit (construction, coverage, filing-level verification) was run
first; it reads no outcome — the population is selected with the target's *availability*
only — and its results are in `RESULTS.md` §1–3, written before `evaluate.py` exists.
Amendments, if any, are appended at the end with their date and whether they were made
before or after outcomes were seen. Nothing in production changes because of this test.

## Question

Do a handful of point-in-time accounting measures of operating instability and financial
fragility, public before Breakwater's weekly call cutoff, concentrate corrected ≥ 8% earnings
moves near the top of the ranking beyond Model C?

Not asked: direction, valuation, size as a discovery, any feature search. If this fails, no
further accounting features are built (brief §18).

## Data

- **Events**: Phase 3's population on the current corrected frame
  (`output/sec_features/feature_frame_current.parquet`): completed, BMO/AMC, report year
  2014–2026, target available, ≥ 8 prior corrected outcomes at the call cutoff, both C inputs.
  21,761 events (16,127 in 2017–2025).
- **Target**: `y = abs_reaction_3d_anchored >= 0.08` (the frame's `y_extreme`). Joined only in
  `evaluate.py`. `acquire.py` and `build.py` read no outcome column (tested).
- **Accounting**: SEC XBRL Company Facts, snapshot `sec_20261002T213221Z`
  (`snapshot_sha256 b2d13efd…a254`, 515 CIKs = exactly the SEC pilot's identity chains).
- **Identity / eligibility**: the SEC pilot's date-bounded CIK chains and its rule
  (`timing.eligible`: filed before the cutoff session, or on it with a verified index-page
  acceptance at or before the close). No second ticker→CIK system.

## 1. Point-in-time construction (`build.py`, fixed)

1. **Selected filing S** = the pilot's latest original 10-Q/10-K eligible at the event's call
   cutoff. Amendments are never used.
2. A Company Facts value is used only if its accession **is** the filing being read. The
   original filing of each quarter supplies that quarter's record; a later restatement,
   amendment or comparative under another accession cannot reach it.
3. **Quarter values** (revenue, operating income): the filing's own 3-month fact (70–120 days,
   ending at the report period ±10 days). For a 10-K without one: Q4 = FY (10-K) − 9-month
   YTD of the same fiscal year's original Q3 10-Q (same tag, same fiscal-year start ±7 days,
   filed earlier). A YTD value is never treated as a quarter.
4. **Year-over-year comparisons** use the same filing's prior-year comparative column (same
   tag, same period length, ending 358–372 days earlier) — both numbers from one accession
   on one accounting basis, both known when that filing was. For a derived Q4 the prior is
   FY-prior − 9M-prior from the same two filings.
5. **Balance sheet**: instants at the report period. Total liabilities = `Liabilities`, else
   `LiabilitiesAndStockholdersEquity` − total equity (incl. NCI, else parent) — the balance
   identity, recorded as its own route.
6. **TTM** (net income, CFO): 10-K → FY. 10-Q → YTD + prior FY (original 10-K whose period
   ends the day before the YTD starts) − prior-year YTD comparative of the same 10-Q; the same
   tag across all three parts, else missing.
7. History = the stock's quarter records filed on or before S and ending on or before S's
   period; one record per period (first original filed). `known_at` = the later of SEC's
   submissions filing date and Company Facts' `filed` for S; `known_at` eligible at the
   cutoff is asserted for every row.
8. Tags: one economic quantity per key, tried in order per filing — revenue: `Revenues`,
   `RevenueFromContractWithCustomerExcludingAssessedTax`, `…IncludingAssessedTax`,
   `SalesRevenueNet`, `RevenuesNetOfInterestExpense`; operating income: `OperatingIncomeLoss`
   only; net income: `ProfitLoss`, `NetIncomeLoss`; CFO: `NetCashProvidedByUsedInOperatingActivities`,
   `…ContinuingOperations`; plus `Assets`, `Liabilities`, `LiabilitiesAndStockholdersEquity`,
   equity (2 tags), `AssetsCurrent`, `LiabilitiesCurrent`, `InventoryNet`. USD only. Two
   different values for one (accession, tag, period) → missing. Nothing else is mapped.

## 2. Features (fixed)

Trailing window = S's quarter and the quarters ending up to 700 days before it (at most 8).

| family | feature | definition | model input |
|---|---|---|---|
| size control | `log_assets` | ln total assets at S | raw |
| A revenue instability | `revenue_growth_volatility` | SD (ddof 1) of quarterly YoY log revenue growth ln(rev / rev prior-year comparative) over the trailing window; ≥ 6 values | `l_rev_vol` = ln(x + 0.01) |
| B margin instability | `operating_margin_volatility` | SD of quarterly YoY operating-margin change (margin = OI / revenue, clipped to ±1, revenue > 0) over the trailing window; ≥ 6 values | `l_margin_vol` = ln(x + 0.01) |
| C recent operating change | `abs_operating_margin_change_yoy` | \|margin(S's quarter) − margin(its prior-year comparative)\| | `l_abs_margin_chg` = ln(x + 0.01) |
| D leverage | `leverage` | total liabilities / total assets at S | clip to [0, 2] |
| E accrual quality | `abs_accruals` | \|TTM net income − TTM CFO\| / total assets at S | `l_abs_accruals` = ln(x + 0.01) |
| F working capital | `wc_to_assets` | (current assets − current liabilities) / total assets at S | clip to [−1, 1] |
| F working capital | `abs_wc_change_yoy` | \|wc_to_assets(S) − wc_to_assets(original filing one year earlier)\| | `l_abs_wc_chg` = ln(x + 0.01) |

Choices made on reliability, not results:

- **Leverage = total liabilities / assets.** A debt measure has no single total tag (long-term
  debt current/noncurrent, short-term borrowings, commercial paper, finance leases…) and
  would need exactly the synonym list the brief rules out. Liabilities/assets covers 99.6%.
- **F = working capital, not inventory.** Working capital covers 82.5% of test-year events,
  inventory 56.4% (absent for financials, REITs, most services).
- **Accruals (E) is kept** because its construction (step 6) is a proper TTM on matching
  periods and passed the gates below; the signed value is stored, not evaluated. Absolute
  value because the target is magnitude.

## 3. Reliability gates (checked before outcomes; results in RESULTS.md §3)

| gate | rule | if it fails |
|---|---|---|
| coverage | feature available for ≥ 60% of the 2017–2025 population (the SEC experiment's bar) | feature dropped from every model |
| filing verification | 40 events (stratified by year bucket × 10-Q/10-K × financial/other, plus 3 derived-Q4, 3 liabilities-by-identity, 3 predecessor-CIK cases): every value behind every feature re-derived from the original filing's own XBRL instance on EDGAR (company, accession, document type, period, units, quarter vs YTD vs annual, filing date ≤ cutoff), features recomputed; ≥ 90% of present feature values must agree | feature dropped |
| too unreliable | every family fails | verdict POINT-IN-TIME FUNDAMENTALS ARE TOO UNRELIABLE TO ANSWER |

## 4. Models and samples

Phase 3's logistic (`calibration.fit_model`: standardised on the training fold, L2 C=1), via
`research.options_pilot.evaluate.walk_forward`. Walk-forward test years **2017–2025**; for test
year Y the training rows are those whose 3-session endpoint is before 1 January Y. **2026 is
excluded from every choice.** No imputation.

Each comparison is fitted on identical rows: the population rows where every input of every
model in that block exists.

| block (rows) | models |
|---|---|
| A: `l_rev_vol` present | C · **C+revenue** · C+size · C+size+revenue |
| B/C: `l_margin_vol`, `l_abs_margin_chg` present | C · **C+margin** (both) · C+size · C+size+margin |
| D: `leverage` present | C · **C+leverage** · C+size · C+size+leverage |
| E: `l_abs_accruals` present | C · **C+accruals** · C+size · C+size+accruals |
| F: `wc_to_assets`, `l_abs_wc_chg` present | C · **C+workingcap** (both) · C+size · C+size+workingcap |
| compact: all compact inputs present | C · **C+compact** · C+size · C+size+compact |

C = `log_hist_mean_abs`, `log_vol_30d` (fixed). Size = `log_assets`.
**C+compact** = C + `l_rev_vol`, `l_abs_margin_chg`, `leverage`, `l_abs_accruals`,
`l_abs_wc_chg` — one input per family (the within-family choice: higher coverage for B/C; the
instability reading for F). A feature that fails a gate in §3 is removed from its family spec
and from C+compact; nothing is added or removed after outcomes.

## 5. Metrics

Primary (pooled out-of-fold, Phase 3's `disc_metrics`): top-10% and top-20% hit rate, capture,
lift. Secondary: ROC-AUC, PR-AUC, Brier, log loss, BMO AUC, AMC AUC, each by year. Deltas vs
the block's C (and C+size+X vs C+size) with Phase 3's 500-rep stock-clustered bootstrap
(fixed seed), on identical event ids (asserted).

## 6. Decision rules

- A spec is **worth keeping** if, vs C on its block: Δ top-10% capture ≥ +1.0 pt, 95% CI lower
  bound > 0, and Δ top-20% capture point estimate > 0.
- It is **not merely size** if C+size+X vs C+size has Δ top-10% capture CI lower bound > 0.
- **FUNDAMENTALS SHOW MATERIAL INCREMENTAL VALUE**: some spec is worth keeping and not merely
  size pre-2026, and it is a frozen holdout model whose 2026 Δ top-10% capture is ≥ 0.
- **FUNDAMENTALS SHOW WEAK / UNCERTAIN VALUE**: not material, but some spec (vs C) has Δ top-10%
  or Δ top-20% capture with CI lower bound > 0, or a point estimate ≥ +1.0 pt.
- **FUNDAMENTALS ADD NO USEFUL VALUE**: neither — including an AUC-only gain.
- **POINT-IN-TIME FUNDAMENTALS ARE TOO UNRELIABLE TO ANSWER**: §3.

## 7. 2026 holdout

After the pre-2026 run, recorded in `selection.json` before 2026 is scored: **C+compact**
(always) and the family spec with the largest pre-2026 Δ top-10% capture point estimate.
Each is trained on its block's rows with an endpoint before 2026-01-01 and scored once on its
block's 2026 rows, against C (and C+size) on the same rows. Nothing changes afterwards.

## 8. After the decision (POST HOC, labelled)

Allowed and nothing more: within-C-decile univariate AUC of each feature; Spearman of each
feature with C's out-of-fold probability and with `log_assets`; the sector composition of
any apparent signal. Not used to rescue a failed test.

## Amendments

None.
