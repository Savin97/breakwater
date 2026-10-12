# report/calendar_builder.py
#
# The weekly earnings risk calendar — the primary recurring deliverable. Writes
# output/output_<date>/weekly_calendar.html. Every number on it is a Model B field from
# the event frame: risk_tier, risk_score (Breakwater Risk Percentile),
# model_b_hist_mean_abs and model_b_n_prior.

from itertools import groupby
import os
import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from report.recommendations_builder import PERCENTILE_EXPLAINER, TIER_BANDS, UNSCORED_NOTE
from scoring.model_b import TIERS, display_score, sort_by_risk
from utilities.output_utilities import get_run_output_dir


def build_calendar_data(rows, reference_date=None, window_days=14):
    """
    Calendar data for the events in `rows` dated within
    [reference_date, reference_date + window_days].

    rows : event-frame rows carrying the Model B fields. The pipeline passes the PENDING
           rows (`pending_events(events_df)`); the dashboard's "Export calendar HTML"
           button passes its own frame for the window the user picked.

    Unscored events are included, labelled "insufficient history" (not low risk) and
    sorted after the scored events of the same day. Returns (events_list, summary_dict, grouped_by_date_list).
    """
    df = rows.copy()
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    if reference_date is None:
        reference_date = pd.Timestamp.today().normalize()
    reference_date = pd.Timestamp(reference_date)
    end_date = reference_date + pd.Timedelta(days=window_days)

    window = df[(df["earnings_date"] >= reference_date) & (df["earnings_date"] <= end_date)].copy()
    if window.empty:
        return [], {}, []
    window = sort_by_risk(window, by=["earnings_date"])

    events = []
    for _, row in window.iterrows():
        tier = str(row["risk_tier"])
        hist = row.get("model_b_hist_mean_abs")
        events.append({
            "stock":             row["stock"],
            "earnings_date_str": pd.Timestamp(row["earnings_date"]).strftime("%a %b %d"),
            "earnings_date_raw": pd.Timestamp(row["earnings_date"]),
            "sector":            row.get("sector", "") or "",
            "sub_sector":        row.get("sub_sector", "") or "",
            "risk_score":        display_score(row.get("risk_score")),
            "risk_tier":         tier,
            "risk_tier_css":     tier.lower().replace(" ", "-"),
            "hist_mean_abs":     f"{hist * 100:.1f}%" if pd.notna(hist) else "—",
            "n_prior":           int(row["model_b_n_prior"]),
        })

    tier_counts = {t: int(window["risk_tier"].eq(t).sum()) for t in TIERS
                   if window["risk_tier"].eq(t).any()}
    sector_counts = window.groupby("sector").size().sort_values(ascending=False)
    summary = {
        "reference_date": reference_date.strftime("%B %d"),
        "end_date":       end_date.strftime("%B %d, %Y"),
        "n_total":        len(window),
        "n_high_alert":   int(window["risk_tier"].eq("High Alert").sum()),
        "n_elevated":     int(window["risk_tier"].eq("Elevated").sum()),
        "n_unscored":     int(window["risk_tier"].eq("Unscored").sum()),
        "hottest_sector": sector_counts.index[0] if not sector_counts.empty else "—",
        "tier_counts":    tier_counts,
        "percentile_explainer": PERCENTILE_EXPLAINER,
        "tier_bands":     TIER_BANDS,
        "unscored_note":  UNSCORED_NOTE,
    }
    grouped = [
        {"date": date_str, "events": list(grp)}
        for date_str, grp in groupby(events, key=lambda e: e["earnings_date_str"])
    ]
    return events, summary, grouped


def generate_calendar(rows, reference_date=None, window_days=14):
    """
    Renders the weekly calendar HTML into this run's output folder
    (output/output_<date>/weekly_calendar.html). Called by stage5 with the pending
    event rows; also callable from the dashboard's export button.
    Returns the path written to, or None if there was nothing to render.
    """
    events, summary, grouped = build_calendar_data(rows, reference_date, window_days)
    if not events:
        print("Weekly calendar: no earnings events in window.")
        return None

    env = Environment(
        loader=FileSystemLoader("report/templates"),
        undefined=StrictUndefined,
        autoescape=True,
    )
    html_out = env.get_template("weekly_calendar.html").render(summary=summary, grouped=grouped)

    output_path = os.path.join(get_run_output_dir(), "weekly_calendar.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"Weekly calendar -> {output_path}\n--------------------")
    return output_path
