---
name: codebase-audit-2026-06-07
description: Full codebase audit and refactor on 2026-06-07 — bugs fixed, memory/perf improvements, dead code deleted. Reference this if scores or pipeline behavior changes unexpectedly.
metadata:
  type: project
---

# Codebase Audit & Refactor — 2026-06-07

Backtesting results verified **bit-for-bit identical** before and after all changes.
Pipeline runs clean.

**Measured outcome:**
- Execution time: ~120s → ~80s (~33% faster). Speedup from eliminating 34 DataFrame copy/alloc/GC cycles and the stage5 full-scan loop.
- Peak memory: unchanged at ~6 GB. The per-function copies were short-lived and GC'd immediately — they were never all in memory simultaneously. Peak is dominated by the DataFrame itself (2.8M rows × ~60 columns) plus pandas internals during groupby/merge/transform operations. Memory reduction would require either reducing column count significantly or processing stocks in batches.

---

## Bug Fixes

### 1. Hardcoded ticker in `data_ingestion/api_functions.py`
- **What:** Line 76 had `t = yf.Ticker("wmb")` immediately after `t = yf.Ticker(ticker)`, silently overwriting the parameter so the function always fetched WMB regardless of input.
- **Fix:** Deleted the hardcoded line. Function now uses the `ticker` argument correctly.
- **Note:** `get_earnings_dates_yf` is not currently called in the active pipeline — but would have been a silent data-integrity bomb if wired in.

### 2. `.to_numpy()` positional alignment risk — three locations
Using `.to_numpy()` strips the pandas index and maps values positionally (1st array element → 1st True position in mask). If the sub-DataFrame used for computation was sorted differently from the boolean mask's True positions, values would go to wrong rows silently. Fixed all three to use index-aligned Series assignment instead.

- **`feature_engineering/pre_earnings_stock_features.py`** (`engineer_abs_reaction_p75`):
  - Was: `df.loc[earnings_mask, "abs_reaction_p75"] = earnings_df["abs_reaction_p75"].to_numpy()`
  - Now: `df.loc[earnings_mask, "abs_reaction_p75"] = earnings_df["abs_reaction_p75"]`
  - Also removed stale TODO comments on lines ~150 and ~156 (one said "FIX -1 back to 1" — the shift(1) was already correct; one warned about .to_numpy danger — now resolved).

- **`feature_engineering/post_earnings_stock_features.py`** (`engineer_reaction_entropy`):
  - Was: `df.loc[earnings_mask, "reaction_entropy"] = earnings_df["reaction_entropy"].to_numpy()`
  - Now: `df.loc[earnings_mask, "reaction_entropy"] = earnings_df["reaction_entropy"]`

- **`feature_engineering/post_earnings_stock_features.py`** (`engineer_directional_bias`):
  - Was: `df.loc[earnings_mask, "directional_bias"] = earnings_df["directional_bias"].to_numpy()`
  - Now: `df.loc[earnings_mask, "directional_bias"] = earnings_df["directional_bias"]`
  - Also removed a stale dead comment block (~5 lines) that described a merge approach that wasn't used.

---

## Memory / Performance

### 3. Removed per-function `df = input_df.copy()` from ~34 functions
Every feature engineering and scoring function started with a full copy of the 2.8M-row DataFrame (~150–200 MB each). The stage-entry copies in `stage3.py:54` (`stage3_df = stage2_df.copy()`) and `stage4.py:23` (`stage4_df = stage3_df.copy()`) are the correct isolation boundary — per-function copies were redundant.

Changed `df = input_df.copy()` → `df = input_df` in:

- **`feature_engineering/pre_earnings_stock_features.py`**: `engineer_daily_ret`, `engineer_drift`, `engineer_volatility`, `engineer_momentum`, `engineer_earnings_windows`, `engineer_abs_reaction_median`, `engineer_abs_reaction_p75`, `engineer_surprise_features`, `engineer_pre_earnings_drift_z`
- **`feature_engineering/scoring_features.py`**: `engineer_large_reaction`, `engineer_extreme_reaction`, `engineer_vol_stress`, `engineer_momentum_pressure`, `engineer_sector_vol_stress`, `engineer_proximity_score`, `engineer_vol_expansion_score`, `engineer_momentum_fragility_score`, `engineer_earnings_explosiveness_score`, `engineer_surprise_momentum_flag`, `engineer_pre_earnings_drift_flag`, `engineer_total_risk_score`, `classify_large_relative_earnings_move_bucket`
- **`feature_engineering/scoring_features.py`** (`engineer_earnings_explosiveness`): Was `df = input_df.sort_values(["stock", "date"]).copy()` — the sort was redundant (stage3 sorts at entry), changed to `df = input_df`.

**`engineer_earnings_reactions` in `post_earnings_stock_features.py`**: This function's sort was kept because `group.shift(-k)` inside the groupby depends on within-group date order. Changed `df = df.copy().sort_values(["stock", "date"])` → `df = df.sort_values(["stock", "date"])`. The `sort_values` itself creates a new object so no explicit `.copy()` is needed, and the sort ensures correctness.

Also cleaned up stale comment block in `engineer_abs_reaction_median` (the commented `.to_numpy()` line and its warning, now superseded by the fix in item 2).

### 4. Stage5 loop: pre-group by stock
- **File:** `pipeline/stage5.py`
- **What:** The report-generation loop was doing `df[df["stock"] == stock]` on every iteration — an O(N) scan of 2.8M rows per stock, 30 stocks = 60 full scans per pipeline run.
- **Fix:** Added `df_by_stock = {s: grp for s, grp in df.groupby("stock")}` before the loop. Loop now uses `df_by_stock.get(stock)` — O(1) lookup.

### 5. Streamlit: removed `.copy()` from sidebar filter call
- **File:** `streamlit_dash/app.py`
- **What:** `sidebar_filters(raw_df.copy(), raw_upcoming.copy())` was copying the full 303 MB cached DataFrame on every sidebar widget interaction. `sidebar_filters` only does boolean filtering (never mutates in-place), so no copy is needed.
- **Fix:** Changed to `sidebar_filters(raw_df, raw_upcoming)`.

---

## Dead Code Deleted

### 6. `engineer_timing_danger` removed from `feature_engineering/scoring_features.py`
- Defined but never imported or called in `stage4.py`. The weighted combination of proximity/vol/momentum/explosiveness scores was entirely commented out; the active body was broken (bitwise AND on floats). Safe to delete.

### 7. Files deleted
- `u_deprecated/temp_prep_for_streamlit.py` (entire `u_deprecated/` folder) — superseded by `streamlit_dash/streamlit_export.py`
- `testing/GPT_GENERATED_FILES/` (3 files: `CHATGENERATED_stage5.py`, `year_by_year_regime_eval.py`, `year_by_year_regular.py`) — unintegrated scratch code, never imported
- `feature_engineering/post_earnings_sector_features.py` — empty stub with only docstring comments, never imported

### 8. Minor comment cleanup
- `feature_engineering/scoring_features.py` (`engineer_momentum_pressure`): removed stale `#TODO: CHANGE ADDED` comment

---

## What Was NOT Changed (and why)

- **`reaction_1d`, `reaction_5d`, `is_up`, `is_down`, `is_nochange`**: Kept — useful for future analysis and backtesting alternative windows.
- **`earnings_explosiveness_z`, `earnings_tail_z`**: Kept — deprecated but may be useful for model experimentation.
- **`abs_reaction_p90_rolling`, `earnings_move_bucket`**: Kept — `classify_large_relative_earnings_move_bucket` chain retained for potential future use.
- **AlphaVantage ingestion functions** (`ingest_all_stocks`, `ingest_all_earnings_dates`, `fetch_daily_adjusted_prices`, `get_earnings_data_from_api`): Kept — may be re-activated if yfinance proves unreliable.
- **`testing/testing.py`**: Kept — blocked by `exit()` on line 13, but left for reference.

**Why:** User explicitly asked to keep these for potential future use.
