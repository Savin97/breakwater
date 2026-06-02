---
name: window-sensitivity
description: Rolling window grid search — window=28 confirmed optimal for earnings_explosiveness_score
metadata:
  type: project
---

## Grid search results (run 2026-05-27, OOS 2011–2025)

Script: `testing/window_sensitivity.py`
Results: `output/window_sensitivity.txt`

| window | avg_lift | median_lift | pct_years_≥3x | avg_n_regime/yr |
|--------|----------|-------------|----------------|-----------------|
| 8      | 4.41     | 4.50        | 0.87           | 133             |
| 10     | 4.32     | 4.17        | 0.87           | 141             |
| 15     | 4.28     | 4.17        | 0.93           | 156             |
| 20     | 4.18     | 4.08        | 0.93           | 154             |
| **28** | **4.49** | **4.77**    | **1.00**       | 128             |

**Window 28 wins on avg_lift (4.49x) and is the only window with 100% of years ≥3x lift.** Shorter windows fail in high-vol years (2020, 2022). Production already uses window=28 — no code change needed.
