# testing/backtesting.py
#
# Backtesting suite for Model B (0.4.0). INTERNAL ONLY — retro-scored history, not a
# track record. Reads output/events_df.parquet; every check uses completed events with an
# available corrected outcome (testing_functions.prepare_events) and the target
# abs_reaction_3d_anchored >= 8%.
#
#   1. Train-year threshold regime: the raw Model B signal's top-10% threshold set on
#      2013-2016 only, then held fixed through 2017-2025 — hit rate, capture, share.
#   2. Walk-forward yearly report: scaling and bucket edges from all prior years.
#   3. Overlay test: does a second condition on top of High Alert sharpen it?
#
# Run with: .venv/bin/python -m testing.backtesting
import pandas as pd, warnings
from testing.testing_functions import (
    forward_eval_onefactor,
    yearly_oos_report,
    high_conviction_regime_test)

EVENTS_PATH = "output/events_df.parquet"
SCORE = "model_b_hist_mean_abs"
# Candidate overlays tested on top of the High Alert tier: (column, quantile) conditions.
OVERLAYS = {
    "vol_30d top 20%":           [("vol_30d", 0.8)],
    "vol expanding (10d/30d top 20%)": [("vol_ratio_10_to_30", 0.8)],
}


def backtesting_suite(input_df):
    # Model B only scores events with >= 8 usable priors; Unscored rows carry a raw mean
    # from fewer priors and are not part of the model being backtested.
    df = input_df[input_df["risk_tier"] != "Unscored"].copy()
    cols = ["year", "n_regime", "regime_extreme_rate", "ratio", "regime_capture_of_extremes"]

    print(f"-------------------------------\n{SCORE} (top 10% regime, threshold from 2013-2016)")
    e_stats, thr = forward_eval_onefactor(df, SCORE, q=0.90)
    print(f"threshold: {thr:.4f}")
    print(e_stats[e_stats["split"] == "TEST"][cols].to_string(index=False))

    print("-------------------------------\nyearly_oos_report:")
    yearly_oos_report(df, score_feature=SCORE)

    print("-------------------------------\nHigh Alert + overlay test:")
    high_conviction_regime_test(df, overlays=OVERLAYS)

    return


if __name__ == "__main__":
    print("Running Backtesting Stage (Model B, internal)")
    warnings.filterwarnings('ignore')
    df = pd.read_parquet(EVENTS_PATH)
    backtesting_suite(df)
    print("-------------------------------\nBacktesting Done.")
