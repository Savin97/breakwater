# Independent audit of 0.4.0 (30d6675) — data findings and remediation, 2026-10-14

The independent review found four high-severity defects (H1 prospective eligibility, H2
same-day overwrite, H3 dashboard freshness, H4 adjusted-price ingestion) and medium items
(report-day price gaps dropping events, previous-event IV, the research rebuild, publication
provenance). This file records what was **measured on the production data** before anything
was changed, and what changed. It reports no performance figure.

Data: local copy of the droplet DB pulled 2026-10-12 (prices to 2026-10-12, 503 stocks).
Reference: a fresh yfinance download (auto_adjust=True, the ingestion's own mode, with
split/dividend actions) of every stock on 2026-10-14. Diagnostic scripts and outputs are in
the gitignored `output/audit_2026-10-14/`. Nothing in the DB was rewritten.

## 1. How the stored prices were built

`prices.ingested_at` shows it directly: every stock's history up to 2026-02-11/13 came from
ONE Alpha Vantage adjusted-close full load (2026-02-12 or -14; checked: exactly one load per
stock), then yfinance appended dates after the DB's global max — daily, except one catch-up
batch on 2026-05-19 covering 2026-02-23..05-18.

## 2. H4 — adjustment-basis breaks (stored / fresh ratio steps)

| period | 1–5% steps | 5–20% | >20% | cause |
|---|---|---|---|---|
| after 2026-02-11 (yfinance appends) | 159 | 1 | 9 | **verified**: each large step is a corporate action after the earlier rows were ingested — APH 2:1, BKNG 25:1, CRWD 4:1, CTVA 6.665, CVNA 5:1, DD 1:3, FDX 1.241, KLAC 10:1, MNST 2:1, SPGI 1.057; the small ones are dividends (64 at the 2026-02-23 batch seam) |
| inside the Alpha Vantage loads | 3,186 + 1,156 one-day spikes | 223 + 99 | 26 + 5 | **provider disagreement**, not this mechanism; 2000–2015 almost entirely (2016: 23 steps, 2018–2025: 23) — unverified which source is right |

Outcome impact (all 25,999 available corrected 3-session outcomes, same anchor/endpoint):

- **Verified contamination: MNST 2026-08-06 (AMC)** stored −51.65%, available and extreme;
  coherent prices give −3.29%. Six more 2026 outcomes differ by 0.5–1.2% (dividend seams).
- Effect of fixing the verified cases on Model B: 17 raw signals change (7 stocks, max
  0.0085); **one pending tier changes: MNST Elevated → Normal**; 2026 completed scored events
  (n 1,450) AUC 0.7166 → 0.7161; the 2014–2025 reference sample is untouched.
- Treating yfinance as correct everywhere (upper bound for the provider disagreement):
  16 raw signals change by > 0.005; reference p80/p90 0.05557/0.06960 → 0.05558/0.06957;
  label flips at 8%: MET 2017-08-02 (Brighthouse spin-off; stored +10.8% looks wrong, fresh
  −1.1%) and DOW 2026-07-23 (−8.00% both ways, rounding). Not material; not changed.
- 9 inactive/renamed tickers (AVB, BK, CTRA, DAY, EA, EQR, HOLX, PSKY, WBD) have no fresh
  history and are unverifiable; their only large 2026 move is PSKY +20.8% on 2026-02-27.

**The frozen 2014–2025 reference cannot contain H4 contamination** (each stock's pre-2026
history is one consistent load); the 2026 evaluation contains MNST.

## 3. Market-wide hole 2026-05-19..21

Only AAPL, NVDA and TSLA have prices on those three sessions (rows written 2026-05-25
12:22, which moved the global max to 05-21; every later run started at 05-22). 497 of 500
live stocks miss all three. The session grid itself is complete (6,734 sessions = the
exchange history 2000-01-03..2026-10-12, none extra). Cost: 19 computable 2026 outcomes
(18 events vanished from the frame, 1 `unavailable_endpoint_price_gap`). Elsewhere only 28
single-stock missing sessions (13 in 2006).

## 4. Catalog events missing from the event frame (pre-fix)

| catalog event (after stage-2 dedup) | count | kind |
|---|---|---|
| in frame (completed / pending) | 45,725 / 482 | — |
| before the stock's price history | 967 | intentional |
| after the stock's last price (future beyond the pending one, delisted) | 22 | intentional |
| report date not a market session (weekend/holiday) | 297 (53 timestamped) | **silently dropped** |
| market session, ticker has no price row | 20 (18 = the May hole, 2006, 2008) | **silently dropped** |

None of the 317 dropped events had a computable corrected outcome on the current prices.

## 5. What changed

- Event identities from the catalog; the 317 events are back as completed rows with explicit
  statuses (297 `unresolved_no_session`, 20 `unresolved_price_gap`). **Every Model B value of
  the 46,207 existing events is unchanged; available outcomes stay 25,999.**
- IV context only beside its own event: hidden on **40 of 482 pending calls** whose snapshot
  predates the stock's previous report (e.g. NWSA 2026-11-05 beside a 2026-06-26 snapshot).
- Ingestion reconciles a window per stock (section 2 mechanism and the May hole cannot
  recur). Dry run of the one-time repair `since="2026-02-11"` in memory: 495 stocks, 408
  basis changes, 1,883 rows filled (May 19–21 → 494 stocks); pre-2026-02-11 returns change
  by ≤ 9e-16 and 0 of 24,812 outcomes ending before then change; all 1,187 later outcomes
  then match the fresh history (32 were off by > 0.1%); MNST → −3.29%. **Not run on any DB.**
- Publication generations, append-only archive with publication time and revisions,
  announcement-time eligibility, generation-keyed dashboard (CLAUDE.md, *Scoring ≠
  publishing ≠ archival*). Archive migrated: 63 legacy rows identical, all void, no
  publication time (never eligible).

## 6. Repair validated on a disposable copy (2026-10-15)

Two copies of the production DB (`763649d4…`, prices to 2026-10-12) under the gitignored
`output/audit_2026-10-15/repair/`: one untouched (baseline), one repaired with
`incremental_ingest_all_prices_yf(con, since="2026-02-11", today="2026-10-12")`, which
fetched live yfinance data for the 482 active priced stocks. Stages 2–4 of the current code
ran on each (cwd = the copy; the live DB hash was checked unchanged before and after).

| check | result |
|---|---|
| failures / basis changes | 0 / 399 stocks, each logged once in `price_basis_log` |
| MNST | rows to 2026-08-07 × 0.5 (94.16 → 47.08); 2026-08-06 reaction −51.65% → **−3.29%** |
| 2026-05-19..21 | 3 → 482 stocks per session; rows 2,926,454 → 2,927,893 |
| before 2026-02-11 | same 2,844,472 rows; one factor per stock = the logged factor (spread ≤ 4e-16); daily returns change ≤ 9e-16; 104 stocks untouched |
| rerun (×3) | no rescale, no duplicate (stock, date), row count unchanged; window rows differ by ≤ 3.4e-7 relative (yfinance float noise, below the 1e-6 tolerance) |
| outcomes | 19 newly available (the May hole: 17 `unresolved_price_gap`, 2 endpoint gaps; 7 of them ≥ 8%), 0 lost, 897 2026 values move, 30 by > 0.1%; label flips: MNST, DOW (−8.00% both ways) |
| Model B | raw signal moves > 1e-12 on 1,034 events (all 2026+), > 0.001 on 8; pending tiers: **MNST Elevated (86.9) → Normal (80.0)** only; pre-2026 signals differ ≤ 2.2e-16 — which flips one retro tier sitting exactly on the boundary (TRGP 2022-11-03, percentile 80.0, Elevated → Normal; internal history only) |
| live 2014–2025 reference sample | identical identities (n 20,493); signals ≤ 1e-16, quantile grid ≤ 7e-17; `scoring/model_b_reference.json` and the pinned snapshot untouched (hashes, `--check` MATCHES) |

Live 2026 scored events with a corrected outcome (retro-scored, internal):

| | n | base rate | ROC-AUC | PR-AUC | top-10% capture | top-20% capture | High Alert hit |
|---|---|---|---|---|---|---|---|
| before repair | 1,450 | 0.2462 | 0.7166 | 0.4258 | 0.2017 | 0.3838 | 0.5167 |
| after repair | 1,469 | 0.2478 | 0.7158 | 0.4278 | 0.2005 | 0.3874 | 0.5220 |

2014–2025 is identical to four decimals (n 20,493, AUC 0.7230). The frozen validation on
the snapshot (2026 n 1,436, AUC 0.7174) does not change: it reads the pinned inputs.

Against the 2026-10-14 dry run: the same MNST fix and 19 added outcomes; 30 rather than 32
corrections > 0.1% — the two not repaired are **POOL 2026-02-19** (−15.56% stored vs −14.42%;
extreme either way) and **TAP 2026-02-18** (−3.65% vs −2.66%), inactive stocks the
active-only ingestion does not fetch. Not material; listed as open.

## 7. Open, for the maintainer

1. Run the price repair on the droplet DB (backup first), then re-run the pipeline: expected
   changes as in §6 — MNST's outcome and tier, 30 small 2026 outcome corrections, 19 added
   2026 outcomes. The frozen reference and the pinned snapshot do not change. Decide whether
   to also reconcile the inactive POOL and TAP (the function takes only active stocks).
2. Deploy dashboard code and the first generation together (the workflow refuses unless the
   droplet runs the same commit) and restart the dashboard process after pulling; remove or
   disable old landing-page track-record files.
