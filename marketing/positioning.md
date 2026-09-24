# Breakwater Positioning

## Audience

Primary audience: options traders who care about earnings volatility and gap risk.

Secondary audiences to validate:
- Active stock traders who avoid or size around earnings events.
- Portfolio managers who need a weekly earnings risk screen.
- Financial analysts who want a structured event-risk watchlist.

Do not optimize the first message for generic retail investors. The product is most valuable when the user already understands that earnings weeks are discontinuous risk events.

## Core Promise

Breakwater is a weekly earnings risk radar for S&P 500 stocks.

It flags companies with elevated risk of a large post-earnings move using only information available before the announcement.

## Proof Points

**Source of truth: `audit/PHASE0_AUDIT_REV2.md`. Do not publish a performance number that is not traceable to a line in it.** The figures published before that audit were derived from a target that mismeasured before-open (BMO) announcements and were overstated by roughly 3x. They are retracted.

Verified figures, measured on 11,496 earnings events with provider announcement timestamps (audit §Q2):

- High Alert events moved at least 8% **45.5%** of the time [95% CI 42.9, 48.2].
- The base rate for an 8%+ move across those same events is **20.4%** [19.7, 21.1].
- That is a **1.91x lift, stratified by announcement window** (audit §Q2, "Lift vs the corresponding baseline"). Always use the stratified figure. The crude 2.23x is inflated by composition, not model skill.
- High Conviction events moved at least 8% **58.6%** of the time [50.9, 65.9], n=162 — a **2.46x** stratified lift.
- High Alert and Elevated together capture **39.0%** [37.1, 41.0] of all 8%+ moves (audit §Q2, "Capture of anchored ≥8% moves").

Scope and limits that must travel with those numbers:

- Sample: n=11,496 verified events, dated 2008-2026, with coverage reaching back only to about 2020 for most tickers (audit §Q2 sample block; §Q3 "Coverage constraint"). **There is no 15-year or 2015-2025 out-of-sample record. Do not claim one.**
- These are not out-of-sample figures. The tier cut points were selected on this distribution and are being re-fit (audit §Q3 steps 7-8).
- The edge is concentrated in after-close reporters. Capture is **67.8%** on AMC events and **9.3%** [7.8, 11.1] on BMO events (audit §Q2). Within AMC, High Alert lift is **1.87x**; on BMO the model makes almost no calls and its value is **unestablished** pending the Phase 3 rebuild (audit §Q3).
- Knowing only that a company reports after the close is worth **1.42x** with no model at all (audit §Q3). Any headline lift must be stratified so that composition is never read as skill.

Use these as directional product claims. When publishing externally, keep the wording tied to historical outcomes, carry the scope limits above, and avoid implying guaranteed future performance.

## Preferred Language

- Earnings risk radar
- Earnings move risk
- Tail-risk screen
- Flagged before the announcement
- Elevated probability of a large move
- Historical base rate
- Weekly earnings risk digest

The **delayed public track record is paused** and must not be used as a proof point until it returns. Tiers published before the audit were assigned against the mismeasured target and are void for track-record purposes (audit remediation plan P4.2, P4.3).

## Avoid

- Buy, sell, calls, puts, or trade this
- Guaranteed, sure thing, lock, free money
- "Options edge" as the primary public claim
- Anything that sounds like personalized financial advice
- Any lift, hit rate or base rate not traceable to `audit/PHASE0_AUDIT_REV2.md`
- Crude (unstratified) lift figures
- Any claim of a 15-year or 2015-2025 out-of-sample record

## Compliance Footer

Breakwater is for informational and research purposes only. It is not financial advice, investment advice, or a recommendation to buy or sell securities or options.
