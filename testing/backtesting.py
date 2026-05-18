# testing/backtesting.py
import pandas as pd, warnings
from testing.testing_functions import (
    forward_eval_onefactor,
    yearly_oos_report,
    high_conviction_regime_test)

def backtesting_suite(input_df):
    df = input_df.copy()
    cols = ["year", "n_regime", "regime_extreme_rate", "lift", "regime_capture_of_extremes"]

    print("-------------------------------\nearnings_explosiveness_score (top 10% regime)")
    e_stats, _ = forward_eval_onefactor(df, "earnings_explosiveness_score", q=0.90)
    print(e_stats[e_stats["split"] == "TEST"][cols].to_string(index=False))

    print("-------------------------------\nyearly_oos_report:")
    yearly_oos_report(df, date_col="date", score_feature="earnings_explosiveness_score", target_col="abs_reaction_3d")

    print("-------------------------------\nhigh_conviction_regime_test:")
    high_conviction_regime_test(df)

    return

if __name__ == "__main__":
    print("Running Backtesting Stage")
    warnings.filterwarnings('ignore')
    df = pd.read_parquet("output/full_df.parquet")
    backtesting_suite(df)
    print("-------------------------------\nBacktesting Done.")