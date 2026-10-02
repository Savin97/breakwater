# SEC-derived features vs Model C — pre-registration

Written 2026-10-02, **before any SEC feature was computed and before any outcome of this
test was read.** Amendments, if any, are appended at the end with their date and whether
they were made before or after outcomes were seen. Nothing in production changes because
of this test, whatever the result.

## Question

Does a small set of causal, transparent SEC-derived measures of uncertainty and change,
available at Breakwater's weekly call cutoff, improve the concentration of corrected ≥ 8%
earnings moves near the top of the ranking beyond Model C?

Not asked: direction, broad text mining, embeddings, LLM labels, B vs C.

## Data

- **Events**: `output/sec_features/feature_frame_current.parquet` — Phase 3's frame rebuilt
  by the same code on the date-corrected data (`frame.py`; refresh audit in
  `research/sec_filings_pilot/RESULTS.md` §12).
- **SEC**: snapshot `sec_20261002T165536Z`, the pilot's identity chains (`mapping.py`) and
  its information sets (`output/sec_filings_pilot/`). Documents from the pilot's
  fetch-once archive.
- **Eligibility** is the pilot's rule and only that (`timing.eligible`): filing date before
  the cutoff session, or on it with a verified index-page acceptance at or before the NYSE
  close. The submissions-JSON `acceptanceDateTime` is never used.
- **Target**: `y = abs_reaction_3d_anchored >= 0.08` (the frame's `y_extreme`). Legacy target
  never used. Features are built in `build.py` without reading any outcome column; the
  target is joined only in `evaluate.py`.

## Population

Phase 3's population on the current frame: completed, BMO or AMC, report year 2014–2026,
target available, ≥ 8 prior corrected outcomes at the call cutoff, both C inputs present;
plus SEC `identity_status == mapped`. Test years **2017–2025** walk-forward (train on events
whose 3-session endpoint is before 1 January of the test year — Phase 3's
`calibration.training_mask`). **2026 is excluded from every choice** and scored once at the end.

## 1. Previous-quarter earnings release (source of families A–E)

For event *e* of stock *s*:

1. **Previous event** = the stock's immediately preceding completed event in the current
   frame (any window, any year), ordered by report date. **Second-prior** = the one before.
2. **Its release 8-K** = an original `8-K` (not 8-K/A) of the stock's CIK chain listing Item
   2.02, filed −1…+4 calendar days around that event's announcement date (Benzinga date, else
   report date). Several → smallest |days|, then earliest filing date, then accession.
3. **Usable at e** only if eligible at e's cutoff by the pilot rule, filed ≤ **140 days**
   before e's cutoff (older means a Breakwater event is missing in between), and — for the
   second-prior — filed strictly before the previous release.
4. **Exhibit** = the `EX-99*` document with the lowest sequence number in the filing index
   (normally EX-99.1, the press release). None → missing.
5. Text: `documents.html_to_text`; paragraphs matching forward-looking-statement / safe-
   harbor / PSLRA / undue-reliance / non-GAAP / conference-call / webcast / contact
   boilerplate are dropped. **Prose** = remaining lines of ≥ 12 words with ≤ 2 table-cell
   separators. Fewer than 100 prose words → missing.

The release of the event being predicted can never qualify: it is filed on or after the
announcement, which is after the cutoff, and step 3 rejects it. A test asserts this.

Missing reasons are recorded: `no_previous_event`, `no_release_8k`, `not_eligible`,
`too_old`, `no_ex99`, `fetch_failed`, `too_little_prose`.

## 2. Features (exact definitions in `features.py`, fixed with this document)

| family | feature | definition | model transform |
|---|---|---|---|
| A | `guidance_present` | ≥ 1 sentence of the previous release (boilerplate removed, tables kept) that contains a guidance word (guidance, outlook, expect(s/ed/ing), anticipate(s/d), forecast(s), project(s/ed), reaffirm*, reiterat*, raise(s/d), lower(s/ed), narrow(s/ed), target(s)), a forward period (fiscal 20XX, FY, full-year, first…fourth quarter, Q1–Q4, next quarter, 20XX) and a number ($, %, or decimal) | 0/1 |
| B | `eps_guidance_width` | median over EPS ranges "$a to $b" found in those sentences of (b−a)/((a+b)/2); valid only if 0 < a < b and width < 1 | log |
| B | `revenue_guidance_width` | same for revenue / net sales "$a (million/billion) to $b million/billion" | log |
| C | `uncertainty_rate` | hits of a fixed exploratory dictionary per 1,000 prose words: uncertain*, volatil*, unpredictab*, challeng*, headwind(s), disrupt*, pressure(s/d), cautio*, difficult*, softness, softer/softening, weak*, slowdown, slowing/slowed, turbulen*, instabilit*, fluctuat*, visibility, risk(s). Not the Loughran–McDonald list (none available locally; its licence is academic); **exploratory** | log1p |
| D | `uncertainty_change` | `uncertainty_rate`(previous release) − `uncertainty_rate`(second-prior release) | raw |
| E | `release_text_change` | 1 − cosine similarity of raw term counts of the two releases' prose (lower-case words ≥ 3 letters, scikit-learn's fixed English stop-word list removed). Pairwise, nothing fitted across companies or years | raw |
| 8-K | `num_8k_since_periodic` | original 8-Ks in the event's pilot information set (after the latest 10-Q/10-K, ≤ cutoff) | log1p |
| 8-K | `num_8k_last_30d` | eligible original 8-Ks filed in the 30 calendar days ending on the cutoff date | log1p |
| 8-K | `has_2_02`, `has_7_01`, `has_8_01`, `has_5_02` | item listed on any information-set 8-K | 0/1 |
| 8-K | `num_material_item_types` | distinct items on information-set 8-Ks, excluding 9.01 | raw |
| 8-K | `has_recent_{2_02,7_01,8_01}_14d` | item on an eligible original 8-K filed in the **14** calendar days ending on the cutoff date. No other window is tried | 0/1 |
| EX-99 (optional) | `num_8k_ex99_last_30d` | of the `num_8k_last_30d` 8-Ks, those whose index lists an EX-99 document | log1p |

Direction source data is kept beside the features and **not evaluated**: counts of
raise/lower/reaffirm-guidance phrases, demand / margin / inventory mentions, and up to five
guidance sentences per release.

## 3. Reliability gates — checked before any outcome is read

Audited by reading extracted text for a stratified sample (years × size × sector), written
up in RESULTS.md §2 before `evaluate.py` is run:

| gate | rule | if it fails |
|---|---|---|
| release matching | 40 matched exhibits; ≥ 90% are the right company's earnings release for the previous quarter | families A–E not evaluated; verdict "FEATURE EXTRACTION IS TOO UNRELIABLE TO ANSWER" if also the 8-K family is unusable |
| release coverage | previous AND second-prior release usable for ≥ 60% of the population | same |
| guidance_present | 20 detected + 20 not detected; ≥ 80% correct in each group | A dropped from every model |
| guidance widths | coverage ≥ 40% of the common sample **and** 20 audited extractions ≥ 90% correct | B dropped from every model (reported as coverage only) |
| uncertainty dictionary | 20 releases; share of hits that are not about business uncertainty (e.g. "risk-based capital", "pressure" in a physical sense) | reported; feature dropped only if > 50% |

## 4. Samples and models

**Main block** (identical rows): the population rows with previous and second-prior
release features present. Every spec is fit on exactly these rows.

| spec | inputs |
|---|---|
| **C** | `log_hist_mean_abs`, `log_vol_30d` (fixed; not redesigned) |
| C+guidance | C + `guidance_present` |
| C+uncertainty | C + `log1p uncertainty_rate`, `uncertainty_change` |
| C+novelty | C + `release_text_change` |
| C+8K | C + all 10 8-K features above |
| **C+compact** | C + `guidance_present`, `uncertainty_change`, `release_text_change`, `log1p num_8k_last_30d`, `has_recent_2_02_14d` |
| C+ex99 (if built) | C + `log1p num_8k_ex99_last_30d` |

If `guidance_present` fails its gate it is removed from C+guidance (spec dropped) and from
C+compact. Nothing is added to C+compact after outcomes.

**Width block** (only if widths pass their gate): C vs C + `guidance_present` + available
`log` widths, on rows where the EPS width exists.

**Broad 8-K block**: C vs C+8K on every population row (no release requirement).

Model: Phase 3's logistic (`calibration.fit_model`: standardised on the training fold,
L2 C=1), walk-forward 2017–2025. No imputation in the main block; no tree models.

## 5. Metrics

Primary: top-10% and top-20% hit rate, capture, lift (pooled OOF, Phase 3's
`disc_metrics`). Also ROC-AUC, PR-AUC, Brier, log loss, BMO / AMC AUC, every metric by year.
Deltas vs C with a 500-rep stock-clustered bootstrap (Phase 3's `bootstrap`, seed fixed).

## 6. Decision rules

- A model is **worth keeping** if its Δ top-10% capture vs C is ≥ +1.0 pt with the 95% CI
  lower bound > 0, **and** its Δ top-20% capture point estimate is > 0.
- **SEC FEATURES SHOW MATERIAL INCREMENTAL VALUE**: some model is worth keeping pre-2026 and
  the frozen holdout model's 2026 Δ top-10% capture is ≥ 0.
- **WEAK / UNCERTAIN VALUE**: no model is worth keeping, but some model has a top-10% or
  top-20% capture delta with CI lower bound > 0, or a point estimate ≥ +1.0 pt.
- **NO USEFUL VALUE**: neither — including an AUC-only gain.
- **TOO UNRELIABLE TO ANSWER**: the release gates fail and the 8-K block cannot stand alone.

An AUC gain without a top-of-ranking gain is never called product-relevant.

## 7. 2026 holdout

After the pre-2026 run, two models are frozen and recorded in RESULTS.md before 2026 is
scored: **C+compact** (always) and the family spec with the largest pre-2026 Δ top-10%
capture point estimate. Each is trained on every population row whose endpoint is before
2026-01-01 and scored once on 2026, against C on the same rows. Nothing changes afterwards.

## Amendments

### Amendment 1 — 2026-10-02, BEFORE any outcome was read

The first reliability audit (round 1, `manual_audit.csv`) failed two gates:

- `guidance_present`: 14/20 detections and 13/20 non-detections correct (< 80%). False
  hits came from "lower revenues", "growth projects", "accounting guidance"; misses from
  ranges printed on the line after the guidance word.
- uncertainty dictionary: 42 of 61 audited hits (69%) were safe-harbor risk lists the
  boilerplate filter did not catch (it matched only lines naming "forward-looking
  statements").

Release matching passed (40/40) and widths were accurate (19/20) but covered only ~20% of
events, so B is dropped as pre-registered.

Changes, made once:

1. Boilerplate filter also drops lines naming risk factors, "risks and uncertainties",
   "uncertainties", "actual results", "could cause", "cautionary", "no obligation to update",
   "undertake(s) no".
2. `guidance_present` uses passages instead of sentences: from a guidance word (guidance,
   outlook, forecast(s), expect(s), expected to, expecting, anticipate(s/d), reaffirm*,
   reiterat*; raise/lower/narrow/target/project dropped) to 300 characters after it, line
   breaks flattened, containing a forward period ("quarter/year ending" added) and a number.
   It is skipped near "accounting/tax/regulatory/IRS/supervisory/new guidance", "economic
   outlook", "than expected", "not/unable to provide/forecast/give", "high/low end of …
   guidance", "above/below/exceeded/beat/in line with/within … guidance/outlook/expectations".
3. Widths are read from those passages (still excluded unless their gate passes).

Re-audited ONCE on a fresh sample (round 2, different seed), same thresholds. A feature
that fails round 2 is dropped from every model. No further revision is allowed.
