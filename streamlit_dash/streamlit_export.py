# streamlit_dash/streamlit_export.py
import pandas as pd
from datetime import date

from pipeline.events import pending_events
from scoring.model_b import sort_by_risk

# The Model B fields every dashboard view reads. Nothing legacy (explosiveness score,
# lift, High Conviction, drift/surprise flags, the legacy abs_reaction_3d) is exported.
MODEL_B_FIELDS = ["risk_score", "risk_tier", "model_b_hist_mean_abs", "model_b_n_prior",
                  "score_asof_date", "model_version"]

# The history parquet carries past corrected reactions ONLY — no reconstructed Model B
# score or tier. Pairing retro-scores with realized outcomes on the public dashboard would
# read as a track record, which is paused (marketing/generate_public_track_record.py).
# Retro-scored history is an internal maintainer view: analysis/predictions_range.py.
HISTORY_COLS = ["stock", "sector", "sub_sector", "earnings_date", "announce_window",
                "reaction_3d_anchored", "abs_reaction_3d_anchored",
                "reaction_3d_anchored_status", "is_extreme_move_anchored"]

UPCOMING_COLS = ["stock", "sector", "sub_sector", "earnings_date", "days_to_earnings",
                 *MODEL_B_FIELDS,
                 # options and analyst snapshots — market context only, never scored
                 "expected_move_pct", "atm_iv", "iv_snapshot_date",
                 "eps_avg", "eps_high", "eps_low", "eps_num_analysts",
                 "eps_dispersion", "eps_revision_momentum",
                 "revenue_avg", "revenue_high", "revenue_low", "eps_snapshot_date"]


def generate_streamlit_df(events_df: pd.DataFrame,
                          output_path: str = "output/streamlit_df.parquet",
                          upcoming_path: str = "output/upcoming_df.parquet") -> None:
    """The dashboard's history parquet — completed events from the last 5 years with their
    corrected 3-session reaction, no score — then the upcoming parquet."""
    completed = events_df[~events_df["is_pending"]]
    out = completed[[c for c in HISTORY_COLS if c in completed.columns]].copy()
    cutoff = pd.Timestamp.today() - pd.DateOffset(years=5)
    out = out[out["earnings_date"] >= cutoff]
    out.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(out)} rows)\n--------------------")

    export_upcoming_df(events_df, upcoming_path)


def export_upcoming_df(events_df: pd.DataFrame,
                       output_path: str = "output/upcoming_df.parquet") -> None:
    """One row per upcoming earnings event, from the PENDING rows of the event frame.
    Sorted by date, then Model B risk (scoring.model_b.sort_by_risk)."""
    today = pd.Timestamp(date.today())
    upcoming = pending_events(events_df, on_or_after=today)

    if upcoming.empty:
        print(f"No upcoming earnings events (today={today.date()})")
        upcoming.reindex(columns=UPCOMING_COLS).to_parquet(output_path, index=False)
        return

    upcoming["days_to_earnings"] = (upcoming["earnings_date"] - today).dt.days
    suspicious = upcoming[upcoming["days_to_earnings"] > 90]["stock"].tolist()
    if suspicious:
        print(f"WARNING: {len(suspicious)} stocks have earnings_date >90 days out (possible bad data): {suspicious}")

    out = upcoming[[c for c in UPCOMING_COLS if c in upcoming.columns]].copy()
    out = sort_by_risk(out, by=["earnings_date"])
    out.to_parquet(output_path, index=False)
    print(f"Wrote {output_path} ({len(out)} rows)\n--------------------")
