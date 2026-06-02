---
name: next-to-build
description: Prioritized build list as of 2026-06-01 — what to work on next to make Breakwater monetizable
metadata:
  type: project
---

As of 2026-06-01, the model is proven and calibration is complete. The gaps are product delivery and monetisation. Build in this order:

**~~2. Historical calibration tables~~** ✅ done 2026-06-01
- Script: `testing/calibration.py`, outputs to `testing/testing_results/`
- Key results: High Alert 40.2% P(≥8%) vs 6.9% base rate (5.8x lift), HC tier 52.4% (7.6x lift), consistent 2015–2025
- Capture rate: 12% of events selected → 42% of ≥8% moves, 65% of ≥15% moves
- False-negative rate at ≥8% is 58% — accepted as irreducible (Normal bucket big moves are one-off surprises, not structurally predictable from history)
- **Product framing decision:** sell "which 15-20 events actually matter this week," not "catch everything." Lead with ≥8% lift, not ≥5%. Use "High Alert stocks blow up 40% of the time vs 7% for Normal" as the pitch.
- IV as future false-negative reducer: Normal stocks with high implied move vs historical p75 could recover some missed moves once IV data accumulates

**1. Dashboard upcoming events view** ← makes the dashboard forward-looking
- `streamlit_export.py` currently filters to `is_earnings_day == 1` — no upcoming events exported
- Fix: also export latest row per stock where `earnings_date` is in the future (same logic as digest `_select_stocks`)
- Add "Upcoming Events" tab to `streamlit_dash/app.py` showing flagged stocks for next 2-4 weeks
- Without this, the dashboard is purely historical and not useful as a live product

**2. Uncapped percentile ranking** ✅ done 2026-06-01
- Compute percentile from `abs_reaction_p75_rolling` directly (pre-clip), not from capped `earnings_explosiveness_score`
- Fixes ties-at-97th problem; also fixes empty 99–100th band in calibration tables
- Requires updating `_select_stocks` in `cron/cron_weekly_digest.py` and optionally the reports

**3. ~~Integrate IV into reports~~** ✅ done
**4. ~~Coverage automation~~** ✅ done
**5. ~~Weekly email digest~~** ✅ done

---

**1. ~~Integrate IV into reports~~ was next thing to build**
- Join latest `iv_snapshots` row per stock into stage2 (left join on stock, latest snapshot_date)
- Add `iv_vs_hist_ratio = expected_move_pct / abs_reaction_p75_rolling` in stage3/4
- Add "Options Market Signal" block to per-stock HTML reports: implied move %, ATM IV, vs. historical p75 — plain-language interpretation ("options appear to underprize the tail risk")
- Handle missing IV gracefully (omit block if NaN)
- Files: `pipeline/stage2.py`, `pipeline/stage3.py` or `stage4.py`, `pipeline/stage5.py`, `report/report_builder.py`, `report/styles.css`

**2. Coverage automation**
- Replace hardcoded `stocks_to_report_for` in `pipeline/stage5.py` with auto-selection: High Alert + Elevated stocks, earnings within 14 days, sorted by risk_score desc, capped at 30

**3. Weekly email digest cron**
- New file: `data_ingestion/cron_weekly_digest.py`
- Runs Monday 07:00 UTC, reads `output/full_df.parquet`, sends HTML email with High Alert stocks for coming week
- Subscriber list: `data/subscribers.txt`
- Add crontab entry on droplet

**4. Dual-window p75 + IV bucket bump** ← false negative reducer for Normal bucket
- Some stocks (e.g. A/Agilent) sit at Normal while having 9-11% actual moves, because the 28-event window includes a quiet period that suppresses rolling p75
- Fix 1: `max(rolling_28, rolling_8)` in `engineer_abs_reaction_p75_rolling()` — one-line change in `feature_engineering/pre_earnings_stock_features.py`
- Fix 2: IV-based bucket bump — if `iv_vs_hist_ratio` is high (options pricing well above historical p75), bump Normal → Elevated. Wire into `risk_scoring/scoring_features.py` ~line 217
- Note: AAPL Normal is correct (genuinely low earnings vol now). Problem is specifically stocks with recent volatility returning after a quiet period.

**What NOT to build yet:** SHAP/explainability, sector-specific models, API, portfolio-level aggregation — all institutional features, not needed for retail market.

**Why:** Model is already strong enough to sell. These three steps complete the product and create a delivery mechanism. Revenue path after: payment gate on dashboard (Stripe) in the website repo.
