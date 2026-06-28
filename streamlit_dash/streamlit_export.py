# streamlit_dash/streamlit_export.py
import pandas as pd
from datetime import date


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
        "momentum_fragility_score", "abs_reaction_3d",
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

    cutoff = pd.Timestamp.today() - pd.DateOffset(years=5)
    out = out[out["earnings_date"] >= cutoff]

    out.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(out)} rows)\n--------------------")

    export_upcoming_df(df)


def export_upcoming_df(df: pd.DataFrame, output_path: str = "output/upcoming_df.parquet") -> None:
    today = pd.Timestamp(date.today())

    latest = df.sort_values("date").groupby("stock").last().reset_index()
    upcoming = latest[latest["earnings_date"] >= today].copy()

    if upcoming.empty:
        print(f"No upcoming earnings events (today={today.date()})")
        upcoming.to_parquet(output_path, index=False)
        return

    _rank_key = upcoming["abs_reaction_p75_rolling"].fillna(upcoming["abs_reaction_p75"])
    upcoming["peer_percentile"] = (_rank_key.rank(pct=True) * 100).fillna(0).astype(int)

    upcoming["is_high_conviction"] = (
        (upcoming["earnings_explosiveness_bucket"] == "High Alert") &
        (upcoming["pre_earnings_drift_flag"].fillna("") != "")
    )

    upcoming["days_to_earnings"] = (upcoming["earnings_date"] - today).dt.days

    # IV vs historical p75 ratio — only valid for upcoming events (current IV snapshot)
    p75_for_ratio = upcoming["abs_reaction_p75_rolling"].fillna(upcoming["abs_reaction_p75"])
    upcoming["iv_vs_hist_ratio"] = (
        upcoming["expected_move_pct"]
        .div(p75_for_ratio)
        .where(p75_for_ratio > 0)
        .round(2)
    )

    cols = [
        "stock", "sector", "sub_sector", "earnings_date", "days_to_earnings",
        "earnings_explosiveness_bucket", "earnings_explosiveness_score",
        "peer_percentile", "pre_earnings_drift_flag", "surprise_momentum_flag",
        "is_high_conviction", "expected_move_pct", "iv_vs_hist_ratio",
        "eps_avg", "eps_high", "eps_low", "eps_num_analysts",
        "eps_dispersion", "eps_revision_momentum",
        "revenue_avg", "revenue_high", "revenue_low", "eps_snapshot_date",
    ]
    out = upcoming[[c for c in cols if c in upcoming.columns]].copy()
    out = out.sort_values("earnings_date")

    out.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(out)} rows)\n--------------------")
