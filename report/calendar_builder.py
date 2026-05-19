# report/calendar_builder.py
#
# Generates the weekly earnings risk calendar — the primary recurring deliverable.
# Two outputs: HTML file (output/weekly_calendar.html) and structured data for Streamlit.
#
# Architecture:
#   earnings_explosiveness_score  → primary risk ranking (structural, changes slowly)
#   momentum_fragility_score      → positioning overlay flag (dynamic, changes quarterly)
#   These are kept separate intentionally — blending them into one score hurts both signals.

from itertools import groupby
from pathlib import Path
import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

# Fixed percentile thresholds calibrated on full earnings history.
# Using fixed thresholds (not within-week percentiles) means the flag
# has the same meaning every week — "top 25% of all historical fragility readings."
FRAG_ELEVATED_PCTL  = 0.75   # top 25% of historical momentum_fragility_score → "Elevated"
FRAG_STRETCHED_PCTL = 0.90   # top 10%                                         → "Stretched"


def _fragility_label(score, elevated_thr, stretched_thr):
    if pd.isna(score):
        return ""
    if score >= stretched_thr:
        return "Stretched"
    if score >= elevated_thr:
        return "Elevated"
    return ""


def _bucket_stats(df):
    """
    Computes Bayesian-shrunk extreme-move probability per earnings_explosiveness_bucket.
    Same prior (20 events) and method as stage5/prep_for_streamlit — keeps numbers consistent
    across the calendar, per-stock reports, and the Streamlit dashboard.
    Returns (bucket_stats_df, p_global_rounded).
    """
    earn = df[df["is_earnings_day"] == 1].copy() if "is_earnings_day" in df.columns else df.copy()
    p_global = earn["is_extreme_reaction"].mean()
    prior = 20  # shrink toward global mean; 20 events ≈ ~5 years of quarterly data per stock

    per_bucket = (
        earn.groupby("earnings_explosiveness_bucket")["is_extreme_reaction"]
        .agg(extreme_count="sum", event_count="count")
    )
    per_bucket["hist_extreme_prob"] = (
        (per_bucket["extreme_count"] + prior * p_global)
        / (per_bucket["event_count"] + prior)
    )
    per_bucket["lift_vs_baseline"] = per_bucket["hist_extreme_prob"] / p_global
    return per_bucket[["hist_extreme_prob", "lift_vs_baseline"]], round(p_global, 3)


def build_calendar_data(df, reference_date=None, window_days=14):
    """
    Prepares calendar data for a window of earnings events.

    reference_date : start of the display window.
                     Defaults to (latest earnings_date in data - 7 days) so the window
                     is always centered on the most recent available data.
    window_days    : how many days forward from reference_date to include (default 14).

    Returns (events_list, summary_dict, grouped_by_date_list).
    Stocks without a scored earnings_explosiveness_bucket are excluded — they lack
    sufficient earnings history for a meaningful risk score.
    """
    df = df.copy()
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])

    if reference_date is None:
        # Center window on latest available date so there's always something to show,
        # even when running against stale/historical data.
        reference_date = pd.Timestamp(df["earnings_date"].max()) - pd.Timedelta(days=7)
    reference_date = pd.Timestamp(reference_date)
    end_date = reference_date + pd.Timedelta(days=window_days)

    # Thresholds from full earnings history — stable across weeks
    earn_all = df[df["is_earnings_day"] == 1].copy() if "is_earnings_day" in df.columns else df.copy()
    all_frag = earn_all["momentum_fragility_score"].dropna()
    frag_elevated_thr  = all_frag.quantile(FRAG_ELEVATED_PCTL)
    frag_stretched_thr = all_frag.quantile(FRAG_STRETCHED_PCTL)

    bucket_stats, p_global = _bucket_stats(df)

    earn = earn_all
    window = earn[
        (earn["earnings_date"] >= reference_date) &
        (earn["earnings_date"] <= end_date)
    ].copy()

    # Exclude stocks with no scored bucket (< ~8 historical earnings events)
    window = window[window["earnings_explosiveness_bucket"].notna()].copy()

    if window.empty:
        return [], {}, []

    window = window.merge(bucket_stats, on="earnings_explosiveness_bucket", how="left")
    window["fragility_flag"] = window["momentum_fragility_score"].apply(
        lambda s: _fragility_label(s, frag_elevated_thr, frag_stretched_thr)
    )
    # Sort by date first, then highest risk score within each day
    window = window.sort_values(
        ["earnings_date", "earnings_explosiveness_score"],
        ascending=[True, False]
    )

    events = []
    for _, row in window.iterrows():
        bucket       = str(row.get("earnings_explosiveness_bucket", ""))
        surprise_flag = str(row.get("surprise_momentum_flag", "") or "")
        drift_flag    = str(row.get("pre_earnings_drift_flag", "") or "")

        # High Conviction: High Alert + any drift flag (4.78x OOS lift).
        # Drift flag compounds cleanly with the structural score; surprise flags add less.
        high_conviction = (bucket == "High Alert") and bool(drift_flag)

        events.append({
            "stock":             row["stock"],
            "earnings_date_str": pd.Timestamp(row["earnings_date"]).strftime("%a %b %d"),
            "earnings_date_raw": pd.Timestamp(row["earnings_date"]),
            "sector":            row.get("sector", ""),
            "sub_sector":        row.get("sub_sector", ""),
            "risk_score":        int(round(row["earnings_explosiveness_score"])) if pd.notna(row["earnings_explosiveness_score"]) else 0,
            "risk_level":        bucket,
            "risk_level_css":    bucket.lower().replace(" ", "-"),
            "high_conviction":   high_conviction,
            "hist_extreme_prob": f"{row['hist_extreme_prob'] * 100:.1f}%" if pd.notna(row.get("hist_extreme_prob")) else "—",
            "lift_vs_baseline":  f"{row['lift_vs_baseline']:.1f}x"  if pd.notna(row.get("lift_vs_baseline")) else "—",
            "fragility_flag":    row["fragility_flag"],
            "surprise_flag":     surprise_flag,
            "drift_flag":        drift_flag,
        })

    # Risk level summary counts, ordered high-to-low for the summary bar pills
    risk_order   = ["High Alert", "Elevated", "Normal"]
    level_counts = {
        lvl: int(window["earnings_explosiveness_bucket"].eq(lvl).sum())
        for lvl in risk_order
        if window["earnings_explosiveness_bucket"].eq(lvl).any()
    }
    sector_counts = window.groupby("sector").size().sort_values(ascending=False)

    n_high_conviction = sum(1 for e in events if e["high_conviction"])

    summary = {
        "reference_date":      reference_date.strftime("%B %d"),
        "end_date":            end_date.strftime("%B %d, %Y"),
        "n_total":             len(window),
        "n_elevated_frag":     int((window["fragility_flag"] != "").sum()),
        "n_high_conviction":   n_high_conviction,
        "avg_risk_score":      round(window["earnings_explosiveness_score"].mean(), 1),
        "hottest_sector":      sector_counts.index[0] if not sector_counts.empty else "—",
        "level_counts":        level_counts,
        "base_extreme_prob":   f"{p_global * 100:.1f}%",
    }

    # Group events by date string for template rendering (already sorted above)
    grouped = [
        {"date": date_str, "events": list(grp)}
        for date_str, grp in groupby(events, key=lambda e: e["earnings_date_str"])
    ]

    return events, summary, grouped


def generate_calendar(df, reference_date=None, window_days=14):
    """
    Renders the weekly calendar HTML and writes to output/weekly_calendar.html.
    Called automatically by stage5 at the end of every pipeline run.
    Can also be called directly (e.g. from Streamlit sidebar export button).
    """
    events, summary, grouped = build_calendar_data(df, reference_date, window_days)
    if not events:
        print("Weekly calendar: no scored earnings events in window.")
        return

    env = Environment(
        loader=FileSystemLoader("report/templates"),
        undefined=StrictUndefined,
        autoescape=True,
    )
    template  = env.get_template("weekly_calendar.html")
    html_out  = template.render(summary=summary, grouped=grouped)

    output_path = "output/weekly_calendar.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"Weekly calendar -> {output_path}\n--------------------")
