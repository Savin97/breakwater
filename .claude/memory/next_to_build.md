---
name: next-to-build
description: Prioritized build list — what to work on next to make Breakwater monetizable
metadata:
  type: project
---

Model is proven, calibration is complete, and core product delivery is shipped. Remaining gaps are monetisation and polish.

## Still to build

**1. Payment gate (Stripe)** ← highest-value next step
- Add Stripe payment link to the dashboard in the `harbor_webpage` repo
- Revenue path: $50–200/month per user retail tier

**2. Weekly chart visual polish** (`report/chart_weekly.py`)
- Functional but aesthetics are weak on dark background
- Options: stronger tier color contrast, larger ticker font, tighter layout, possibly light/cream background

**3. IV signal validation** (`testing/test_iv_signal.py`) ← deferred, small sample
- Before wiring `iv_vs_hist_ratio` into the score, validate it lifts accuracy
- Time-aware join: latest IV snapshot per stock WHERE snapshot_date < earnings_date
- Run `forward_eval_onefactor(df, "iv_vs_hist_ratio")` from `testing/testing_functions.py`
- Sample too small until ~late 2026 (IV collection started May 2026)
- `scripts/sync_iv.sh` handles pulling IV data from droplet first

## Done

- ✅ Dashboard upcoming events view (`streamlit_dash/app.py` Upcoming tab, `streamlit_export.py`) — 2026-06-03
- ✅ Uncapped percentile ranking (rank by `abs_reaction_p75_rolling` pre-clip) — 2026-06-01
- ✅ Historical calibration tables (`testing/calibration.py`) — 2026-06-01
- ✅ IV into per-stock reports ("Options Market Signal" block) — 2026-05-30
- ✅ Coverage automation (stage5 auto-selects HA + Elevated in 14-day window) — 2026-05-30
- ✅ Weekly email digest (`cron/cron_weekly_digest.py`, Monday 07:00 UTC) — 2026-05-30
- ✅ Dual-window p75 (`max(rolling_28, rolling_8)`) — 2026-06-04
- ✅ Incremental pipeline (`pipeline/incremental.py`, 0.8s vs 80s) — 2026-06-07

## What NOT to build yet
SHAP/explainability, sector-specific models, API, portfolio-level aggregation — institutional features, not needed for retail market. IV bucket bump — invalid for backtesting until IV data accumulates.
