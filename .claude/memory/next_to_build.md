---
name: next-to-build
description: Prioritized build list as of 2026-05-30 — what to work on next to make Breakwater monetizable
metadata:
  type: project
---

As of 2026-05-30, the model is proven (4.5x OOS lift). The gaps are product completeness and delivery. Build in this order:

**All three original items shipped 2026-05-30.** Current next priorities:

**1. Uncapped percentile ranking** ← next model improvement
- Compute percentile from `abs_reaction_p75_rolling` directly (pre-clip), not from capped `earnings_explosiveness_score`
- Fixes the 13-stocks-tied-at-97th problem for good
- Requires updating `_select_stocks` in `cron/cron_weekly_digest.py` and optionally the reports

**2. Historical calibration tables**
- Run digest logic across past 20-30 earnings weeks, include all Normal stocks
- Build: P(≥8%) and P(≥10%) by percentile band; capture rate of large moves by tier
- Validates the product claim statistically

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

**What NOT to build yet:** SHAP/explainability, sector-specific models, API, portfolio-level aggregation — all institutional features, not needed for retail market.

**Why:** Model is already strong enough to sell. These three steps complete the product and create a delivery mechanism. Revenue path after: payment gate on dashboard (Stripe) in the website repo.
