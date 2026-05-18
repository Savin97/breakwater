# Session Memory Index

This file is read by Claude Code at the start of each session to restore context.
Entries are updated at the end of each session. Most recent first.

---

## 2026-05-18 — Memory setup + context recovery

Set up `.claude/memory/` inside the repo so session notes sync via git across machines.

---

## 2026-05-18 — Backtesting high_conviction + Report recommendation block

**Backtesting:**
- Validated `is_high_conviction` (High Alert + drift flag): 4.93x OOS lift, 12 events/yr
- Tested `surprise_momentum_flag` sub-categories: Beat/Miss Streak and Erratic add modest lift (~4.2x); Overdue Miss is below High Alert baseline (3.64x) — noise
- Tested "HA + Drift OR (Surprise ex-OM)": 4.24x lift, 75 events/yr — better coverage but lower precision
- Decision: `is_high_conviction` stays as drift-only (4.93x); broader definition has no clear home in the 3-tier system
- 3-tier system: Normal = calm, Elevated = risky, High Alert = very dangerous; High Conviction is a highlight within High Alert

**Report recommendation block:**
- Created `report/recommendations_builder.py` — `build_recommendation()` returns headline, body, action, flag_lines
- Wired into `stage5.py`, `report_builder.py`, HTML template, CSS
- 4 tiers: Normal (no action), Elevated (light caution), High Alert (reduce/hedge), High Alert + HC (strongest language)
- Flag explanations (drift + surprise) shown for Elevated and High Alert; suppressed for Normal
- `risk_score` = `earnings_explosiveness_score` (they are the same thing now)

**Report testing:**
- Fixed `stage5.py` to use `data/sp500_full_info.csv` (was incorrectly referencing `sp500_data.csv`)
- Successfully generated reports for AAPL, NVDA, TSLA, MSFT
- Reports confirmed working end-to-end with recommendation block rendering correctly

**Next:** Reports need further work — content and design TBD. Dashboard is lower priority.

---

## 2026-05-17 — Streamlit dashboard overhaul (other machine)

- Created `pipeline/streamlit_export.py`: generates `streamlit_df.csv` with Bayesian bucket stats; now called automatically at end of stage 5 (replaces manual `prep_for_streamlit.py` step)
- Dashboard: added `pre_earnings_drift_flag`, `surprise_momentum_flag`, `is_high_conviction` columns to Overview and Bucket Stats tabs
- Added "High Conviction only" sidebar filter and metric card
- Weekly Calendar tab: removed `momentum_fragility_score` / "Positioning"; replaced with timing flags and `is_high_conviction`
- `is_high_conviction` = "High Alert" bucket AND non-empty `pre_earnings_drift_flag`

**Next:** Unknown — pick up from here.

---
