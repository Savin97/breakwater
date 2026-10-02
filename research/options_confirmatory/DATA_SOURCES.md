# Confirmatory options test — what data we need, and who might have it

Written 2026-10-01 from public vendor pages; nothing was bought or downloaded. Anything a
public page does not state is marked **unknown** — ask the vendor before paying.

## 1. Minimum data needed

**One options snapshot per event**, on the Friday (call-cutoff session) before the report
week, for the reporting stock only. Not the full market, not every day.

| | 2014–2025 (development) | 2026 YTD (holdout) |
|---|---|---|
| eligible events (stock × Friday) | ~20,300 | 1,339 |
| distinct Fridays | ~600 | 35 |
| stocks | 473 | — |
| requests if 10 tickers per call (ORATS limit) | ~2,500 total | |
| rows if filtered to expiries ≤ ~45 days | a few million (tens of MB) | |

Of these, 2014 events are training only. The ≥ 8-prior-outcome rule leaves 2013 with 50
events, so **history before 2014 is worthless to this test**. A source starting in 2016
still works; it loses ~2,600 events (2014–15, mostly training) and ~5–10 points of power
(`PREREGISTRATION.md` §Power).

## 2. Requirements checklist (a source must pass every "must")

| # | requirement | must / nice |
|---|---|---|
| R1 | Contract-level quotes: ticker, expiration, strike, call/put, **bid, ask** | must |
| R2 | Snapshot at or shortly before the **Friday close** (e.g. 15:45–16:00 ET), capture time stated | must |
| R3 | **Point-in-time**: the historical bid/ask are the quotes as archived that day, not recomputed, smoothed or backfilled later — **stated in writing** | must |
| R4 | Covers 2014 (or at least 2016) → 2026 for S&P 500 names, including names since removed or renamed | must |
| R5 | Retrieval by (ticker, date) — no need to buy the whole market | strongly preferred |
| R6 | Underlying spot captured **at the same snapshot**, nominal (not split-adjusted) | nice (else fallback definition) |
| R7 | Weekly expirations present (so the covering expiry is days, not weeks, after the event) | nice |
| R8 | Delta / IV for diagnostics | nice |
| R9 | License allows research use; data stays out of the public repo | must |

## 3. Candidate sources

| | ORATS | Cboe DataShop (EOD Summary / with Calcs) | OptionMetrics IvyDB (WRDS) | ThetaData | Alpha Vantage HISTORICAL_OPTIONS | historicaloptiondata.com |
|---|---|---|---|---|---|---|
| earliest year | 2007 | 2012 | 1996 | 2012 (Pro), 2016 (Standard), 2020 (Value), 2023 (Free) | "15+ years" (≈2008) | 2002 |
| snapshot timing | 14 min before close (15:46 ET) | 15:45 ET and close | closing best bid/offer | tick-level; any time requested (e.g. 15:45) | end of day (time **unknown**) | end of day (time **unknown**) |
| bid / ask (R1) | yes (`callBidPrice` … / `cBidPx` …) | NBBO bid/ask + sizes | best bid / best offer | yes (NBBO quotes) | **unknown** from public pages | yes |
| spot at snapshot (R6) | `stockPrice` / `spotPrice` — capture time of the historical value **unknown** | underlying bid/ask and active price at 15:45 | underlying close (separate table) | via stock data; tier requirement **unknown** | **unknown** | underlying last (**unknown** timing) |
| point-in-time (R3) | raw quotes plus smoothed "SMV" values; whether 2014–19 raw quotes were captured live **not stated** | exchange-archived OPRA NBBO — point-in-time by construction | end-of-day records compiled daily; academic standard | raw OPRA tick archive — point-in-time by construction | **not stated** | **not stated** |
| weekly expiries (R7) | yes (full chain) | yes | yes | yes | likely, **unknown** | yes |
| symbol coverage | 5,000+ | all listed | all US equity options | all OPRA | per-symbol | all US equities |
| access | S3 bulk, or API (≤ 10 tickers per call, `dte`/`delta` filters) | web cart; pick symbols and dates | WRDS login (institution) | local terminal + REST | REST, one symbol-date per call | ZIP by email |
| targeted retrieval (R5) | API yes; bulk no | yes (symbols + date ranges) | yes (SQL on WRDS) | yes | yes | per symbol-month only |
| minimum cost (public) | **$599 one-time**, full 2007→ near-EOD archive; or Delayed Data API **$199/month** (20,000 requests; "EOD history since 2007 comes with every plan"). No trial; free samples via ORATS University | **not published** (priced in cart) | $0 with university WRDS access; commercial licence **unknown** (typically far above this budget) | **not published** on docs page | premium key; price **unknown** here | $2.89–4.05 per symbol-month → ~20,000 symbol-months = **$58k+**; bulk yearly price **unknown** |
| main concern | point-in-time status of older raw quotes and of `stockPrice`; "cleaned using put-call parity" must not apply to the bid/ask we use | cost at ~20k symbol-days unknown | access only via an institution | price unknown; spot may need a second subscription | same provenance problem as DoltHub | cost; timing undocumented |

Massive (ex-Polygon, where Breakwater already holds a Benzinga account) was not assessed:
public pages did not establish its historical options quote depth or price. Worth one
question to the existing account ("options quotes history start, and price for one month").

Sources: [ORATS near-EOD](https://orats.com/near-eod-data), [ORATS Data API](https://orats.com/data-api),
[ORATS delayed API docs](https://orats.com/docs/delayed-data-api),
[Cboe EOD with calcs](https://datashop.cboe.com/option-quotes-end-of-day-with-calcs),
[Cboe EOD summary](https://datashop.cboe.com/end-of-day-options-summary),
[OptionMetrics on WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/optionmetrics),
[IvyDB US](https://optionmetrics.com/wp-content/uploads/2023/03/IvyDB-US-Brochure.pdf),
[ThetaData subscriptions](https://thetadata.net/docs/Articles/Getting-Started/Subscriptions.html),
[Alpha Vantage premium](https://alphavantage.co/premium),
[historicaloptiondata.com](https://historicaloptiondata.com/historical-options-prices/).

## 4. Cheapest credible routes, in order

1. **University / WRDS access to OptionMetrics** — $0 if available; the academic standard
   for exactly this (closing best bid/offer, underlying close, 1996→). Needs an affiliation.
2. **Free samples first** — ORATS University historical samples, Cboe sample files,
   ThetaData free tier (EOD from 2023): check R1–R3, R6 field by field before paying.
3. **One month of a targeted API** — ORATS Delayed Data API ($199): ~2,500 requests of
   the 20,000 allowed, if its historical-strikes endpoint is included in that plan and
   takes a trade date (confirm first). ThetaData Standard/Pro for one month is the
   alternative (raw OPRA, 2016/2012), price to be quoted.
4. **Bulk archive** — ORATS $599 one-time for 2007→. Cheaper than three months of API
   and removes request limits, but buys ~12 years of data we will not use.

## 5. Recommendation

**Investigate ORATS first, unless you have WRDS access** (in which case OptionMetrics
first — free and the most defensible).

* ORATS is the only source whose public prices fit the budget and that covers 2014 with
  per-ticker/per-date retrieval, a near-close snapshot (15:46, before our 16:00 cutoff)
  and a spot field.
* Its open risk is R3: it advertises cleaning and smoothing. The raw bid/ask must be the
  archived quotes. Settle that from a free sample and in writing before paying.
* Cboe DataShop is the cleanest point-in-time source (archived exchange NBBO, 15:45
  snapshot with underlying bid/ask). Get a quote for ~20,300 symbol-days as the
  comparison; it may be affordable if priced per symbol-day.

## 6. Questions to answer before acquisition

For the user:
1. Do you have university or employer access to WRDS / OptionMetrics? It changes the
   answer and costs nothing.
2. Maximum you are willing to spend on this one test ($199 API month, $599 archive, or a
   Cboe quote)?
3. Given the power estimate (~40% chance of passing the primary criterion if the true
   top-10% effect is the pilot's +0.7 pt), is a likely "real but too small" answer still
   worth that spend?

For ORATS, in writing:
4. Are the historical raw bid/ask for 2014–2019 the quotes captured at 15:46 on that day,
   or reconstructed/cleaned later? Does put-call-parity cleaning touch the raw bid/ask
   fields, or only the smoothed ones?
5. Is `stockPrice` in the historical strikes data the underlying price at the same 15:46
   capture, unadjusted for later splits?
6. Does the $199 Delayed Data API include the historical strikes endpoint by trade date,
   with delisted/renamed tickers, and what is the request cost of one ticker-date?
