import pandas as pd

def export_streamlit_df(df: pd.DataFrame, output_path: str = "output/streamlit_df.parquet") -> None:
    earnings_df = df[df["is_earnings_day"] == 1].copy()

    P_extreme_global = earnings_df["is_extreme_reaction"].mean()
    P_extreme_given_bucket = (
        earnings_df.groupby("earnings_explosiveness_bucket")["is_extreme_reaction"].mean()
    )

    bucket_stats = (
        earnings_df.groupby("earnings_explosiveness_bucket")["is_extreme_reaction"]
        .agg(extreme_count="sum", event_count="count")
    )
    prior_strength = 20
    bucket_stats["hist_extreme_prob"] = (
        bucket_stats["extreme_count"] + prior_strength * P_extreme_global
    ) / (bucket_stats["event_count"] + prior_strength)
    bucket_stats["global_hist_prob"] = P_extreme_given_bucket
    bucket_stats["current_lift_vs_baseline"] = (
        bucket_stats["hist_extreme_prob"] / P_extreme_global
    )
    bucket_stats["current_lift_vs_same_bucket_global"] = (
        bucket_stats["hist_extreme_prob"] / bucket_stats["global_hist_prob"]
    )

    out = earnings_df[[
        "stock", "sector", "sub_sector", "earnings_date",
        "is_large_reaction", "is_extreme_reaction",
        "earnings_explosiveness_bucket", "earnings_explosiveness_score",
        "momentum_fragility_score",
        "pre_earnings_drift_flag", "surprise_momentum_flag",
    ]].copy()

    out = out.merge(
        bucket_stats[[
            "hist_extreme_prob", "global_hist_prob",
            "current_lift_vs_baseline", "current_lift_vs_same_bucket_global",
            "extreme_count",
        ]].reset_index(),
        on="earnings_explosiveness_bucket",
        how="left",
    )

    out["risk_level"] = out["earnings_explosiveness_bucket"]
    out["risk_score"] = out["earnings_explosiveness_score"].round(0)
    out["base_extreme_prob"] = round(P_extreme_global, 3)
    out["is_high_conviction"] = (
        (out["risk_level"] == "High Alert") &
        (out["pre_earnings_drift_flag"].fillna("") != "")
    )

    out["hist_extreme_prob"] = out["hist_extreme_prob"].round(3)
    out["global_hist_prob"] = out["global_hist_prob"].round(3)
    out["current_lift_vs_baseline"] = out["current_lift_vs_baseline"].round(3)
    out["current_lift_vs_same_bucket_global"] = out["current_lift_vs_same_bucket_global"].round(3)

    out.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(out)} rows)\n--------------------")
