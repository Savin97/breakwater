# prepf ro streamlit.py
import pandas as pd

df = pd.read_parquet("output/full_df.parquet")
stock_list = pd.read_csv("data/stock_list.csv")

global_earnings_df = df[df["is_earnings_day"] == 1].copy()

P_extreme_global  = global_earnings_df["is_extreme_reaction"].mean()
P_extreme_given_bucket = (
    global_earnings_df.groupby("earnings_explosiveness_bucket")["is_extreme_reaction"]
    .mean()
)

bucket_stats = pd.DataFrame({
    "global_hist_prob": P_extreme_given_bucket,
    "global_risk_lift_vs_baseline": P_extreme_given_bucket / P_extreme_global
})

earnings_explosiveness_buckets= (
    global_earnings_df.groupby("earnings_explosiveness_bucket")["is_extreme_reaction"]
    .agg(extreme_count="sum", event_count="count")
)

prior_strength = 20
earnings_explosiveness_buckets["shrunk_prob"] = (
    earnings_explosiveness_buckets["extreme_count"] +
    prior_strength * P_extreme_global
) / (
    earnings_explosiveness_buckets["event_count"] + prior_strength
)
earnings_explosiveness_buckets["global_hist_prob"] = bucket_stats.loc[earnings_explosiveness_buckets.index, "global_hist_prob"]
# Lift relative to global baseline
earnings_explosiveness_buckets["lift_vs_baseline"] = (
    earnings_explosiveness_buckets["shrunk_prob"] / P_extreme_global
)
# Lift relative to global same-bucket risk
earnings_explosiveness_buckets["lift_vs_same_bucket_global"] = (
    earnings_explosiveness_buckets["shrunk_prob"] / earnings_explosiveness_buckets["global_hist_prob"]
)
P_extreme_global = round(P_extreme_global, 3)

scoring_df = global_earnings_df[['stock', 'sector', 'sub_sector', 'earnings_date', 'is_large_reaction', 'is_extreme_reaction',
    'vol_ratio_cross_sectional_pct', 'vol_stress_elevated',
    'vol_stress_extreme', 'sector_vol_ratio_pct', 'sector_vol_stress_high',
    'momentum_pressure_regime', 'earnings_explosiveness_z',
    'earnings_tail_z', 'proximity_score', 'vol_expansion_score',
    'momentum_fragility_score', 'earnings_explosiveness_score',
    'earnings_explosiveness_bucket', 'earnings_move_bucket']]

# Make bucket stats mergeable
bucket_report_stats = earnings_explosiveness_buckets.reset_index()

bucket_report_stats = bucket_report_stats.rename(columns={
    "shrunk_prob": "hist_extreme_prob",
    "lift_vs_baseline": "current_lift_vs_baseline",
    "lift_vs_same_bucket_global": "current_lift_vs_same_bucket_global",
})

# Merge bucket-level probabilities into every earnings row
scoring_df = scoring_df.merge(
    bucket_report_stats[
        [
            "earnings_explosiveness_bucket",
            "hist_extreme_prob",
            "global_hist_prob",
            "current_lift_vs_baseline",
            "current_lift_vs_same_bucket_global",
            "extreme_count",
            "event_count",
        ]
    ],
    on="earnings_explosiveness_bucket",
    how="left"
)

# Report-friendly columns
scoring_df["risk_level"] = scoring_df["earnings_explosiveness_bucket"]
scoring_df["risk_score"] = scoring_df["earnings_explosiveness_score"]
scoring_df["base_extreme_prob"] = P_extreme_global

# Optional formatting, only if this df is for display/reporting
scoring_df["risk_score"] = scoring_df["risk_score"].round(0)
scoring_df["hist_extreme_prob"] = scoring_df["hist_extreme_prob"].round(3)
scoring_df["global_hist_prob"] = scoring_df["global_hist_prob"].round(3)
scoring_df["current_lift_vs_baseline"] = scoring_df["current_lift_vs_baseline"].round(3)
scoring_df["current_lift_vs_same_bucket_global"] = scoring_df["current_lift_vs_same_bucket_global"].round(3)

scoring_df = scoring_df[ ['stock', 'sector', 'sub_sector', 'earnings_date', 'is_large_reaction',
       'is_extreme_reaction', 'hist_extreme_prob', 'global_hist_prob',
       'current_lift_vs_baseline', 'current_lift_vs_same_bucket_global',
       'extreme_count', 'risk_level', 'risk_score',
       'base_extreme_prob'] ]
scoring_df = scoring_df[(scoring_df['earnings_date'] >= '2025-01-01') & (scoring_df['earnings_date'] <= '2026-01-01')]

scoring_df.to_csv("df_for_streamlit.csv", index=False)
