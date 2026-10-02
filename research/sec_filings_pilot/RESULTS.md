# SEC filings feasibility pilot — results

Written 2026-10-02. **Data audit only.** No outcome column was read, no feature or score
was computed, and nothing in production, `MODEL_VERSION`, the reports or the prediction
archive was touched. The question was only:

> Can Breakwater reliably obtain, timestamp and map historical SEC filings that were
> publicly available before each weekly call cutoff?

**Verdict: SEC FILING DATA IS USABLE WITH IMPORTANT LIMITATIONS** (§11). The filings exist,
map to the right company and download cleanly for 99.8% of events. The one serious trap is
SEC's own timestamp field, which is wrong by 4–5 hours for about a fifth of filers and
would leak information if used naively. It is not used.

```bash
export SEC_USER_AGENT="<name> <contact email>"     # required; never committed
PYTHONPATH=. .venv/bin/python -m research.sec_filings_pilot.acquire [--finish]   # ticker map + submissions snapshot
PYTHONPATH=. .venv/bin/python -m research.sec_filings_pilot.sample --clock        # timestamp audit (300 index pages)
PYTHONPATH=. .venv/bin/python -m research.sec_filings_pilot.sample --cutoff-day   # verified times for cutoff-day filings
PYTHONPATH=. .venv/bin/python -m research.sec_filings_pilot.build                 # index + per-event selection (~5 min)
PYTHONPATH=. .venv/bin/python -m research.sec_filings_pilot.sample --documents    # 160-filing document sample
PYTHONPATH=. .venv/bin/python -m research.sec_filings_pilot.audit                 # tables below
.venv/bin/python -m pytest testing/test_sec_filings.py -q                         # 29 tests, no network
```

Folder is `research/sec_filings_pilot/` (the brief's `sec_filings+pilot` is not a valid
Python package name). Modules: `acquire.py` (the only network code), `mapping.py`,
`timing.py`, `build.py`, `documents.py` (HTML → text, sections, exhibit list),
`sample.py`, `audit.py`. Raw SEC data: `data/vendor/sec/` (gitignored). Derived tables:
`output/sec_filings_pilot/` (gitignored).

---

## 1. What one event looks like

The information set each event would have had at its call cutoff (last close before the
Monday of the report week):

| event | window | cutoff | latest 10-Q/10-K before cutoff | 8-Ks after it, up to the cutoff [items] |
|---|---|---|---|---|
| GOOGL 2015-07-16 | AMC | 2015-07-10 | 10-Q of **Google Inc.** (CIK 1288776), filed 2015-04-29, 72 days old | 2015-06-04 [5.02, 5.07, 9.01] |
| DD 2016-04-26 | BMO | 2016-04-22 | 10-K of **E.I. du Pont** (CIK 30554), filed 2016-02-04, 78 days | 2016-03-25 [2.03], 2016-03-31 [8.01] |
| JPM 2024-01-12 | BMO | 2024-01-05 | 10-Q, filed 2023-11-01, 65 days | none |
| NVDA 2024-02-21 | AMC | 2024-02-16 | 10-Q, filed 2023-11-21, 87 days | none |
| XOM 2026-07-31 | BMO | 2026-07-24 | 10-Q of **Exxon Mobil Corp** (CIK 34088), filed 2026-05-04, 81 days | 2026-05-29 [5.07] (old CIK), 2026-07-07 [7.01] (**new holding company**, CIK 2115436) |

Three of these five needed a company that today's SEC ticker map does not point to.
That is the mapping problem in one table (§3).

---

## 2. Sources (all official SEC; nothing third-party)

| what | endpoint | mutable? |
|---|---|---|
| ticker → CIK today | `https://www.sec.gov/files/company_tickers.json` | yes → snapshot |
| filing history | `https://data.sec.gov/submissions/CIK##########.json` + every `CIK…-submissions-NNN.json` page it lists | yes → snapshot |
| filing index (documents, exhibit types, **verified "Accepted" time**) | `https://www.sec.gov/Archives/edgar/data/<cik>/<acc>/<acc>-index.htm` | no → archive |
| header (cross-check of "Accepted") | `…/<acc>.hdr.sgml` | no → archive |
| documents | `…/<acc>/<file>` | no → archive |

- **Snapshot** `sec_20261002T165536Z`, 1,425 files (515 CIKs), `snapshot_sha256`
  `73ff61a5896af38f45073ae69574ac35ba809420768a0baf5fb8723d851b220c`. Built in a
  `.partial` folder (resumable), sealed by rename, files read-only, never reopened.
- **Archive**: fetched once, never refetched, read-only. `Archive.verify()` re-downloads
  and logs a changed digest to `archive_changes.jsonl` without overwriting.
- **Access rules**: User-Agent from `SEC_USER_AGENT` (refuses to run without one; a test
  checks no address is in the code), one request at a time, ≥ 0.2 s apart (SEC allows 10/s).
  Measured throughput with latency was ~2 requests/s. Total for this pilot: ~2,900 requests.

Every filing row in `filings_index.parquet` keeps CIK, Breakwater stock, accession, form,
filing date, the raw SEC timestamp, report period, primary document, source URL, index
URL, snapshot id (retrieval time is in the snapshot manifest), mapping provenance and
reason. `events.parquet` / `event_8k.parquet` carry the event id per selected filing.

---

## 3. Mapping Breakwater stocks to SEC CIKs

**Population:** Phase 3's events (`output/phase3_refit/feature_frame.parquet`, clock and
identity columns only): completed, BMO or AMC, report year 2014–2026, call cutoff known.
**22,837 events, 482 stocks.** No outcome-based filter (unlike Phase 3, no "target
available" or "≥ 8 prior outcomes" rule). The 21 missing stocks of the 503 are exactly the
Benzinga identity-hazard tickers (APP, BF-B, BRK-B, CBOE, CPAY, CPB, DELL, DOC, DOW, ES,
FISV, GE, GEN, HON, IR, LIN, MAR, ORCL, SMCI, SNDK, WBD): the timing seed skipped them, so
they have no BMO/AMC and no call-time construction. All 503 stocks map to a confident CIK
at the stock level (GE, SMCI via the manual brand-name list), but those 21 were **not**
checked event by event. LIN (Praxair → Linde plc 2018) and DOW (Dow Chemical before 2019)
would need predecessor chains; DOC's pre-2024 dates may belong to Physicians Realty, not
Healthpeak.

### Stocks

| | stocks |
|---|---|
| today's SEC ticker map, one CIK, SEC name agrees with Breakwater's | 459 |
| manually verified, single CIK (brand ≠ legal name: BK, DECK, IBM; no longer in SEC's map: AVB, CTRA, DAY, EA, EQR, HOLX) | 9 |
| manually verified **chain of CIKs** (predecessor companies) | 14 |
| ambiguous / unresolved | **0** |

Every manual link is checked on each build against SEC's own name history for that CIK;
one failed during the work (a misspelt expected name for E.I. du Pont) and was dropped,
not guessed, until fixed. Nothing is ever matched on company name alone.

**Predecessor chains** — the price series carried on under the same ticker while filings
moved to a new CIK. Boundary = filing date of the successor's completion filing (8-K12B
etc.) in SEC's data; each CIK's filings outside its date range are excluded (E.I. du Pont
kept filing as a subsidiary until 2023, Alphabet filed before it took over):

| stock | chain |
|---|---|
| GOOG, GOOGL | Google Inc. → Alphabet (2015-10-02) |
| MDT | Medtronic Inc → Medtronic plc (2015-01-27) |
| AVGO | Avago → Broadcom Ltd (2016-02-02) → Broadcom Inc (2018-04-04) |
| STE | STERIS Corp → STERIS plc UK (2015-11-06) → STERIS plc Ireland (2019-03-28) |
| DD | E.I. du Pont → DowDuPont/DuPont (2017-09-01) |
| CI | Cigna Corp → Cigna Group (2018-12-20) |
| DIS | Walt Disney Co (old) → new Disney holding co (2019-03-20) |
| TPL | Texas Pacific Land Trust → Corp (2021-01-11) |
| APA | Apache → APA Corp (2021-03-01) |
| APO | Apollo Global Mgmt (old) → new holding co (2022-01-03) |
| BG | Bunge Ltd → Bunge Global SA (2023-11-01) |
| BLK | BlackRock (old) → new holding co (2024-10-01) |
| XOM | Exxon Mobil Corp → ExxonMobil Holdings (2026-07-01) |

Without these, 294 events would have had no filings at all (XOM's whole history, since
SEC's map now points at a CIK created three months ago).

Share classes: GOOG/GOOGL, FOX/FOXA, NWS/NWSA each share one CIK — same filings, counted
once in the corpus estimate. Dot/dash spellings (BRK-B, BF-B) are handled; a spelling is
never allowed to reach a different share class.

### Events

| identity status | events |
|---|---|
| mapped | **22,787 (99.8%)** |
| before the company's first 10-Q/10-K (26 new listings — ABNB, CARR, GEHC, UBER, …; one event each, the first report after IPO/spin-off; genuinely nothing to find) | 26 |
| foreign-filer era (NXPI before 2019: 20-F/6-K instead of 10-K/10-Q/8-K) | 24 |
| unmapped | 0 |

**Identity check.** For 99.4% of mapped events the mapped CIK filed an Item 2.02 8-K
within −1…+4 days of the announcement date (≥ 99.0% every year). This confirms the link
and the event date together; it uses the release only as evidence, never as an input.
Lowest: TROW 45% (T. Rowe Price filed releases without Item 2.02 in early years — a filing
habit, not a wrong company), STLD 80%, then 89–94% for a handful. CB — where Benzinga
mixes in the old Chubb Corp — confirms on ACE/Chubb Ltd's CIK back to 2014.

---

## 4. Timing: SEC's timestamp field is wrong for ~22% of filers

This is the main finding and the reason the rule below exists.

The submissions JSON gives `acceptanceDateTime` like `2025-04-11T14:45:50.000Z`. Checked
against SEC's filing index page ("Accepted") and the `.hdr.sgml` header — which agree with
each other to the second in every case, both New York time — on 300 random filings across
all CIKs and years:

| JSON value is | filings | example |
|---|---|---|
| true UTC | 227 (76%) | NVIDIA 8-K: 20:27Z = 16:27 EDT, as the index page says |
| **UTC shifted by one more offset (+8 h EDT / +10 h EST vs NY)** | **66 (22%)** | JPMorgan earnings 8-K accepted 06:45 ET is stored as 14:45Z; Apple 16:31 ET as 02:31Z next day |
| New York time with a "Z" | 7 (2%) | all old, inactive predecessor CIKs |

It is filer-specific and present in every year. A morning filing shifted this way still
looks plausible, so it **cannot be detected per filing**. Corpus-wide, 22.0% of 10-K/10-Q/
8-K JSON timestamps provably break EDGAR's own rules when read as UTC; the true share
wrong is higher. Read literally, the field would place JPMorgan's 06:45 release 8-K at
10:45 — harmless there, but on a cutoff Friday a 16:30 filing would read 12:30 and be
admitted before a 16:00 close. **The JSON field is recorded and never used to decide
eligibility.**

### The rule (fixed in `timing.py` before any event was mapped)

| filing date vs cutoff session | eligible? |
|---|---|
| earlier day | **yes** — EDGAR accepts 06:00–22:00 ET and never dates a filing earlier than its acceptance, so it was public by 22:00 the day before, before the close |
| same day | only with a **verified** acceptance time (index page) ≤ that day's NYSE close (13:00 on early-close days) |
| later day | never |

No time of day is ever assumed. The verified index time obeyed EDGAR's filing-date rule in
300/300 checks (13 were accepted after 17:30 and dated the next business day, as the rule
says).

- Filings dated exactly on some event's cutoff day: 769 (792 event–filing pairs: 786
  8-K, 6 10-K/10-Q); all 769 verified from index pages. Of the pairs, **331 were accepted
  before the close (admitted), 461 after (excluded)**. Without the index pages all would
  simply be excluded.
- In the final information sets: 0 filings dated after the cutoff; 329 cutoff-day 8-Ks, all
  verified before the close.
- "After the 10-Q/10-K" also uses the filing date: an 8-K counts only on a strictly later
  day. Same-day 8-Ks (usually the release accompanying the 10-Q) are counted separately
  (≈ 0.3–0.5 per event) and left out.

### Earnings-release leakage

| check | result |
|---|---|
| Item 2.02 8-Ks filed −1…+4 days around an event's announcement (the release being predicted) | 22,975, covering 22,642 events |
| of those, inside any event's information set | **0** |
| of those, filed before the cutoff | 0 |
| Item 2.02 8-Ks *in* information sets | 1,266 — 444 within 14 days of the cutoff |

The 444 are separate, genuinely public disclosures (preliminary results, conference
updates): 441 of those events also have their own release 8-K at the announcement. They
stay eligible. Three events have a pre-cutoff 2.02 but no release 8-K at the announcement
— CCL 2021-01-26, VST 2021-05-04, VTRS 2021-03-01 — which may mean Breakwater's date is
not the release date (Carnival reported Q4 2020 on 2021-01-11). All 444 are in
`info_set_2_02_within_14d_of_cutoff.csv`; nothing was changed.

---

## 5. Periodic filing coverage (latest 10-Q/10-K at the cutoff)

Denominator = all 22,837 population events.

| year | events | mapped | % with prior 10-Q/10-K | % 10-Q | % 10-K | median age (days) | p90 age | median age (sessions) | missing |
|---|---|---|---|---|---|---|---|---|---|
| 2014 | 1,617 | 1,608 | 99.4 | 74.6 | 24.8 | 79 | 92 | 55 | 9 |
| 2015 | 1,677 | 1,671 | 99.6 | 74.8 | 24.9 | 79 | 92 | 55 | 6 |
| 2016 | 1,714 | 1,709 | 99.7 | 74.8 | 24.9 | 80 | 93 | 56 | 5 |
| 2017 | 1,736 | 1,730 | 99.7 | 74.9 | 24.8 | 80 | 93 | 56 | 6 |
| 2018 | 1,748 | 1,742 | 99.7 | 74.8 | 24.9 | 81 | 93 | 57 | 6 |
| 2019 | 1,792 | 1,784 | 99.6 | 74.5 | 25.1 | 84 | 93 | 57 | 8 |
| 2020 | 1,827 | 1,825 | 99.9 | 75.1 | 24.8 | 84 | 93 | 58 | 2 |
| 2021 | 1,857 | 1,854 | 99.8 | 74.9 | 25.0 | 84 | 93 | 58 | 3 |
| 2022 | 1,868 | 1,868 | 100.0 | 75.1 | 24.9 | 84 | 93 | 58 | 0 |
| 2023 | 1,875 | 1,873 | 99.9 | 74.8 | 25.1 | 84 | 94 | 58 | 2 |
| 2024 | 1,901 | 1,899 | 99.9 | 75.0 | 24.9 | 85 | 94 | 58 | 2 |
| 2025 | 1,861 | 1,860 | 99.9 | 74.7 | 25.3 | 85 | 93 | 58 | 1 |
| 2026 | 1,364 | 1,364 | 100.0 | 68.5 | 31.5 | 84 | 94 | 57 | 0 |
| **all** | **22,837** | **22,787** | **99.8** | 74.5 | 25.3 | **84** | 93 | 57 | **50** |

BMO 99.9%, AMC 99.6%, same median age (84 days). The six stocks that left SEC's map this
year (AVB, CTRA, DAY, EA, EQR, HOLX): 247 of 248 events covered. The 50 missing are exactly
the 26 new listings and 24 NXPI foreign-filer events. A 10-Q/A or 10-K/A sits between the
selected filing and the cutoff for 1.1% of events (recorded, never selected).

**Age matters more than coverage.** The latest periodic report is a median **84 days**
(57 sessions) old at the call — it describes the quarter *before* the one being
reported, and the market has had about three months with it.

## 6. 8-K coverage (filed after that 10-Q/10-K, up to the cutoff)

| year | mapped events | % ≥ 1 8-K | mean | median | p90 | % with an 8-K/A | % with item 2.02 | 7.01 | 8.01 | 5.02 | 1.01 | 5.07 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2014 | 1,608 | 80.7 | 2.23 | 2 | 5 | 4.5 | 6.2 | 32.2 | 37.1 | 37.7 | 24.1 | 20.5 |
| 2016 | 1,709 | 81.5 | 2.30 | 2 | 5 | 5.1 | 5.9 | 31.5 | 40.7 | 38.0 | 23.5 | 21.5 |
| 2018 | 1,742 | 78.8 | 2.17 | 2 | 5 | 5.6 | 4.9 | 32.2 | 36.8 | 39.4 | 23.6 | 20.9 |
| 2020 | 1,825 | 82.9 | 2.44 | 2 | 5 | 4.1 | 7.5 | 36.8 | 44.2 | 38.9 | 28.2 | 21.0 |
| 2022 | 1,868 | 78.5 | 2.07 | 1 | 4 | 4.0 | 4.2 | 30.7 | 32.7 | 40.7 | 20.0 | 21.2 |
| 2024 | 1,899 | 76.8 | 2.01 | 1 | 4 | 4.7 | 4.5 | 30.8 | 33.7 | 38.4 | 19.5 | 21.8 |
| 2026 | 1,364 | 82.3 | 2.19 | 2 | 4 | 4.6 | 5.9 | 34.5 | 36.1 | 43.0 | 23.5 | 30.1 |
| (all years in `coverage_8k_by_year.csv`) | | | | | | | | | | | | |

BMO 80.4% / AMC 79.6% with at least one. 47,874 distinct 8-Ks sit in some information set.
Items across them: 9.01 (exhibits) 69%, **8.01 other events 32%, 7.01 Reg FD 25%**, 5.02
officer changes 25%, 1.01 material agreements 15%, 5.07 shareholder votes 11%, 2.03 debt
9%, **2.02 results 2.6%**. SEC lists items for every 8-K in the period; no 8-K lacked them.
Most are routine (votes, debt, executive changes); none was judged relevant here.

---

## 7. Can the documents be downloaded and read?

160 filings, stratified: 4 periods (2014–15, 2018–19, 2022–23, 2025–26) × 5 kinds (10-K,
10-Q, 8-K with 2.02, other 8-K, amendment) × 2 sizes (100 largest issuers by SEC's own
ordering vs the smaller half + stocks that left the index), spread over all 11 sectors.
241 documents (primary + up to 4 EX-99 exhibits each).

| check | result |
|---|---|
| index page fetched / primary document listed in it | 160/160 / 160/160 |
| primary document downloaded | 160/160, all HTML (none plain text, even 2014) |
| exhibits downloaded | 81/81 |
| encoding | all UTF-8 (declared or default); 0 replacement characters, 0 mojibake |
| parser errors | 0 |
| inline-XBRL hidden facts | stripped (`display:none` / `ix:header`) |
| text size, median | 10-K 490k chars, 10-Q 172k, 8-K shell 4.6k, EX-99 40k |
| share of text inside tables, median | 10-K 26% (13% numeric tables), 10-Q 36% (22%) — prose and data tables separate cleanly; some filers lay out prose in tables (counted as table text) |
| 10-K Risk Factors located | 94% by "Item 1A" heading, 100% with title fallback |
| 10-K MD&A located | 69% by "Item 7" heading, **94%** with title fallback; the 2 misses (WFC 2015, DE 2015) put MD&A in the annual report exhibit (EX-13), not the 10-K body |
| 10-Q MD&A located | 97% |
| 10-Q Risk Factors | heading in 97%, but **47% are under 1,000 chars** — "no material change since the 10-K" |
| 8-K items in text | match SEC's metadata |

Amendments are mostly short: 10-K/A median 9k chars (Part III / exhibits only).

## 8. Earnings-release exhibits

| | |
|---|---|
| 2.02 8-Ks with an EX-99 exhibit | **32/32** |
| …where the exhibit was the release (metadata or opening text) | 32/32 (two needed the text: PLD "Earnings Release and Supplemental", BRK "News Release") |
| other 8-Ks with an EX-99 exhibit | 41% |
| exhibits downloaded and converted to text | 81/81 |
| 2.02 release exhibits mentioning "outlook" or "guidance" | 73% |

The 8-K itself is a ~4,600-character shell; the useful text is in EX-99.1 (median 40k
characters, often with an EX-99.2 CFO commentary or supplement). The index page names the
exhibit type reliably, so release text is easy to reach.

## 9. Full-corpus size

Window 2013-06-17 → 2026-09-04, mapped CIKs. Sizes from the sample (primary ~4.9 MB per
10-K, 2.7 MB per 10-Q — mostly inline-XBRL markup; 35 KB per 8-K; 554 KB per EX-99;
0.87 EX-99 per 8-K).

| | everything | only what events use |
|---|---|---|
| 10-K/10-Q | 25,620 (383 amendments) | 22,663 selected |
| 8-K | 87,593 | 47,874 |
| EX-99 exhibits | ~75,800 | ~41,400 |
| raw HTML (primary + EX-99) | ~128 GB | ~98 GB |
| gzip-compressed | ~7.3 GB | ~5.6 GB |
| extracted text | ~12.6 GB | ~9.6 GB |
| requests (index + primary + EX-99) | ~302,000 | ~182,000 |
| time at 5 req/s / at the measured ~2 req/s | 17 h / 42 h | 10 h / 25 h |

SEC's own "size" field (every file of every submission, incl. XBRL and images) is 585 GB —
not needed. Keeping the text locally (≈ 10 GB) or the compressed HTML (≈ 6 GB) is
practical; a full download is an overnight-to-weekend job, resumable from the archive
cache. Concurrency of 3–4 would stay under SEC's 10/s and cut the time accordingly.
Not estimated: EX-13 annual-report exhibits (needed for the few issuers whose MD&A lives
there).

---

## 10. Problems found (largest first)

1. **SEC's JSON timestamp is wrong for ~22% of filers** by one UTC offset (and NY-local
   for some old CIKs). Undetectable per filing. Handled by the filing-date rule plus
   verified index-page times on cutoff days. Any future code that uses
   `acceptanceDateTime` from the submissions JSON reintroduces a leak.
2. **Today's ticker map is wrong for history** for 14 stocks (13 reorganisations + the 2026
   XOM holding company): 294 events would have no filings, and GOOGL/DD/XOM-style events
   would silently read the wrong company's filings without date-bounded chains. The chains
   are hand-curated, verified against SEC's name history on every build, and need upkeep
   when the next reorganisation happens.
3. **The latest 10-Q/10-K is old by the call** (median 84 days). Text from it describes the
   previous quarter. Fresh information between reports arrives only through 8-Ks
   (≈ 2 per event, most routine).
4. **The population is today's index members** (plus six that left in 2026), and the 21
   Benzinga identity-hazard tickers are not in it at all (no timing). Companies that left
   earlier are not in Breakwater, so "historical constituents" were barely tested, and the
   hardest identity cases (LIN, DOW, DOC) are untested. A Breakwater limitation, not an
   SEC one.
5. **Section finding is approximate**: MD&A sometimes lives in EX-13; 10-Q risk factors are
   usually a pointer; section boundaries come from regular expressions (title fallback
   needed for ~25% of 10-K MD&As).
6. **Foreign filers** (NXPI to 2019; CRH to 2023, outside this population window) file
   20-F/6-K — different documents, not covered.
7. **Small date issues**: 3 events where a 2.02 8-K precedes the cutoff with no release at
   Breakwater's announcement date (possible date errors); TROW's early releases lack Item
   2.02. The 103 Breakwater date corrections of 2026-09-29 are not in this frame (it
   predates them).

## 11. Answers

1. **Map to CIKs historically?** Yes — 482/482 population stocks (503/503 at stock level),
   22,787/22,837 events, 0 ambiguous,
   with 14 hand-verified predecessor chains; 99.4% of events confirmed by the company's own
   release 8-K.
2. **Usable prior 10-Q/10-K?** 99.8% of 2014–2026 events (≥ 99.4% every year); the rest are
   first reports after IPO/spin and NXPI's foreign-filer years.
3. **Intervening 8-Ks?** ~80% of events have at least one (median 1–2, p90 4–5).
4. **Timestamps precise enough?** Not SEC's JSON field. The filing date plus verified
   index-page times on cutoff days are, and give 0 post-cutoff filings and 0 release
   8-Ks in any information set.
5. **Download and parse?** Yes: 241/241 documents downloaded, decoded and converted with no
   errors; tables separable; standard sections found in 94–100% of 10-Ks.
6. **Release exhibits?** Yes: every sampled 2.02 8-K carries an EX-99 release, identified by
   type and readable.
7. **Main problems:** §10.
8. **Corpus size:** ~6 GB compressed / ~10 GB text for what events use; ~25 h of polite
   downloading at the measured rate.
9. **Good enough for a Phase 2 text experiment?** Yes, as data.
10. **Most promising documents:**
    - *For magnitude:* the **previous quarter's earnings release (EX-99.1)** — guidance
      ranges and their width, "outlook" language (73% mention it), novelty vs the release
      before; then the **count and type of intervening 8-Ks** (7.01/8.01 disclosures,
      2.02 preliminary results — 444 events have one within 14 days of the cutoff, 5.02
      management changes); then **10-K Risk Factors change year over year** (located 100%)
      and **10-Q MD&A** (97%). 10-Q risk factors are mostly pointers and not worth much.
    - *For direction (later):* guidance raised/cut and KPI changes in the previous release
      and in any pre-cutoff 2.02/7.01 exhibit; MD&A demand/inventory language. All of this
      is old news by the call except pre-announcements, so expect little.

Inventory only — none of these was computed. Constraints any experiment inherits: the
filing-date rule; the predecessor chains; text from the latest 10-Q is ~3 months old;
exhibits must be fetched separately from the 8-K shell.

**SEC FILING DATA IS USABLE WITH IMPORTANT LIMITATIONS**

---

## 12. Refresh after the 2026-09-29 date corrections (2026-10-02)

The tables above were built on Phase 3's frame of 2026-09-12, before 103 Breakwater
earnings dates were corrected. `research/sec_features/frame.py` rebuilds that frame with
the same Phase 3 code and Benzinga snapshot on today's corrected `full_df`;
`paths.FEATURE_FRAME` now points at it, and the pilot was re-run with the same SEC
snapshot, identity chains and timing rule. `research/sec_features/refresh.py` compares the
two (identity/clock/filing columns only; old outputs kept in
`output/sec_filings_pilot/pre_refresh_20261002/`):

| | |
|---|---|
| events before → after | 22,837 → **22,947** |
| events whose cutoff, selected 10-Q/10-K, 8-K set or identity status changed | **0** |
| events whose earnings date moved | 1 (AES 2026-08-03 → 08-04, same cutoff, same filings) |
| events added | 110: the 103 corrected dates (their old wrong dates had failed the Benzinga timing match, so they were never in the population) + 7 events newly timed; by year 2021–24: 15, 2025: 52, 2026: 43 |
| events dropped | 0 |
| new missing/unresolved | 1 (DASH 2021-02-25, first report after listing → `pre_first_periodic`) |
| cutoff-day filings needing verified times | 772 (3 new, fetched) |

Nothing above needs restating: the corrections only added events.
