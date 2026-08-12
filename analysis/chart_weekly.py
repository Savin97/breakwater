"""
Weekly earnings risk chart for social sharing.
Reads output/upcoming_df.parquet, generates a Twitter-ready PNG.

Usage (CLI):
    python -m analysis.chart_weekly              # this week (Mon–Fri containing today)
    python -m analysis.chart_weekly 20/07/2026   # week starting DD/MM/YYYY
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from datetime import date, datetime, timedelta
import numpy as np
from pathlib import Path

from utilities.output_utilities import get_run_output_dir
from analysis.chart_layout import layout_and_fit


TIER_COLORS = {
    "High Alert": "#c9a84c",   # gold — loud
    "Elevated":   "#1e1a08",   # dark amber bg
    "Normal":     "#ffffff",   # white bg
}
TIER_TEXT_COLORS = {
    "High Alert": "#0a0a0a",   # black on gold
    "Elevated":   "#c8a035",   # amber text — clearly visible
    "Normal":     "#141820",   # dark navy text on white
}
BG       = "#0a0a0a"
BORDER   = "#2a2a28"
TEXT_DIM = "#e8e8e8"
TEXT_MUT = "#b0aca8"
ACCENT   = "#c9a84c"


def generate_weekly_earnings_chart(
    output_path: str | None = None,
    parquet_path: str = "output/upcoming_df.parquet",
    start_date: str | None = None,
) -> str:
    """
    start_date: optional DD/MM/YYYY string; defaults to Monday of the current week.
    output_path: defaults to this run's timestamped output subfolder.
    """
    print("Generating Weekly Chart...")
    if output_path is None:
        output_path = os.path.join(get_run_output_dir(), "weekly_chart.png")
    df = pd.read_parquet(parquet_path)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])

    if start_date is not None:
        week_start = pd.Timestamp(datetime.strptime(start_date, "%d/%m/%Y"))
    else:
        today      = pd.Timestamp(date.today())
        week_start = today - timedelta(days=today.weekday())  # Monday of current week
    week_end = week_start + timedelta(days=4)  # Friday
    week = df[(df["earnings_date"] >= week_start) & (df["earnings_date"] <= week_end)].copy()

    if week.empty:
        print("No upcoming events in window.")
        return ""

    week = week.sort_values(["earnings_date", "peer_percentile"], ascending=[True, False])

    # Always show Mon-Fri regardless of whether events fall on every day
    all_days = [week_start.date() + timedelta(days=i) for i in range(5)]
    n_days = 5

    fig_w = max(10, n_days * 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ── Background tier zones ──────────────────────────────────────────────
    ax.axhspan(0,  75,  color="#0a0a0a", zorder=0)
    ax.axhspan(75, 90,  color="#0e0d0b", zorder=0)
    ax.axhspan(90, 100, color="#121008", zorder=0)

    # ── Horizontal gridlines ───────────────────────────────────────────────
    for y in [25, 50, 75, 90]:
        ax.axhline(y, color=BORDER, linewidth=0.6, zorder=1)

    # ── Zone labels (right margin) ─────────────────────────────────────────
    ax.text(4.45, 95,  "HIGH ALERT", va="center", ha="right",
            color=ACCENT,     fontsize=7, fontfamily="monospace", alpha=0.80)
    ax.text(4.45, 82,  "ELEVATED",   va="center", ha="right",
            color="#c8a035",  fontsize=7, fontfamily="monospace", alpha=0.80)
    ax.text(4.45, 38,  "NORMAL",     va="center", ha="right",
            color="#ffffff",  fontsize=7, fontfamily="monospace", alpha=0.80)

    # ── X-axis: day labels ─────────────────────────────────────────────────
    ax.set_xticks(range(n_days))
    ax.set_xticklabels(
        [d.strftime("%a %b %-d") for d in all_days],
        color=TEXT_DIM, fontsize=10, fontfamily="monospace",
    )
    ax.tick_params(axis="x", colors=TEXT_DIM, length=0, pad=10)

    # ── Y-axis ─────────────────────────────────────────────────────────────
    ax.set_ylim(0, 104)
    ax.set_xlim(-0.55, n_days - 0.45)
    ax.set_yticks([0, 25, 50, 75, 90, 100])
    ax.set_yticklabels(
        ["0th", "25th", "50th", "75th", "90th", "100th"],
        color=TEXT_MUT, fontsize=8, fontfamily="monospace",
    )
    ax.tick_params(axis="y", colors=TEXT_MUT, length=0)
    ax.set_ylabel("Peer Percentile", color=TEXT_MUT, fontsize=9,
                  fontfamily="monospace", labelpad=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # ── Title ──────────────────────────────────────────────────────────────
    week_label = f"Week of {week_start.strftime('%B %-d')}"
    fig.text(0.5, 0.96, "Earnings Risk", ha="center", va="top",
             color="#e8e4dc", fontsize=16, fontfamily="serif", fontstyle="italic")
    fig.text(0.5, 0.91, week_label, ha="center", va="top",
             color=ACCENT, fontsize=10, fontfamily="monospace",
             fontweight="500")

    # ── Legend ─────────────────────────────────────────────────────────────
    # Placed on the figure (not the axes) and pinned below the plotted data
    # region, so it can never collide with a stock label — unlike an
    # axes-anchored corner legend, which a dense low-percentile day can crowd.
    patches = [
        mpatches.Patch(facecolor=TIER_COLORS["High Alert"], label="High Alert"),
        mpatches.Patch(facecolor=TIER_COLORS["Elevated"],   label="Elevated"),
        mpatches.Patch(facecolor=TIER_COLORS["Normal"],     label="Normal"),
    ]
    fig.legend(
        handles=patches, loc="center", bbox_to_anchor=(0.5, 0.045), ncol=3,
        frameon=False, labelcolor=TEXT_DIM,
        prop={"family": "monospace", "size": 8},
        columnspacing=1.2, handletextpad=0.5,
    )

    # ── Footer ─────────────────────────────────────────────────────────────
    fig.text(0.5, 0.012, "harbor-markets.com  ·  Breakwater",
             ha="center", color=TEXT_MUT, fontsize=8, fontfamily="monospace")

    # Finalize axes layout before placing stock labels, so the pixel/data
    # scale used for decluttering matches what's actually saved.
    fig.tight_layout(rect=(0, 0.065, 1, 0.90))

    # ── Plot stocks ────────────────────────────────────────────────────────
    date_to_x = {d: i for i, d in enumerate(all_days)}
    day_texts: dict[date, list] = {d: [] for d in all_days}

    for _, row in week.iterrows():
        d     = row["earnings_date"].date()
        x_raw = date_to_x[d]
        y     = float(row["peer_percentile"])
        tier  = row["earnings_explosiveness_bucket"]
        hc    = row.get("is_high_conviction", False)
        label = ("★ " if hc else "") + row["stock"]

        fc = TIER_COLORS.get(tier, TIER_COLORS["Normal"])
        tc = TIER_TEXT_COLORS.get(tier, TIER_TEXT_COLORS["Normal"])
        fw = "bold" if tier == "High Alert" else "normal"
        fs = 9 if tier in ("High Alert", "Elevated") else 8
        box = dict(boxstyle="round,pad=0.35", facecolor=fc, edgecolor="none")
        text = ax.text(float(x_raw), y, label, ha="center", va="center",
                        fontsize=fs, fontweight=fw,
                        color=tc, bbox=box, zorder=3, fontfamily="monospace",
                        clip_on=True)
        text._true_xy = (float(x_raw), y)
        day_texts[d].append(text)

    # Spread out same-day labels that would otherwise overlap (growing the
    # figure if a single day is packed enough that even max lanes can't fit
    # everyone), and draw a thin leader line back to the true percentile for
    # any label that had to move to make room. clip_on=True on everything
    # here is a safety net: if a pathologically dense day still doesn't fully
    # converge within layout_and_fit's growth cap, residual overflow is
    # clipped at the axes edge instead of escaping onto the canvas margin,
    # where it would throw off every fixed-fraction fig.text() position once
    # savefig's bbox_inches="tight" re-crops around it.
    all_moved = layout_and_fit(fig, ax, day_texts)
    for moved in all_moved.values():
        for text, true_x, true_y in moved:
            fx, fy = text.get_position()
            ax.plot([true_x, fx], [true_y, fy], color=TEXT_MUT,
                    linewidth=0.6, alpha=0.45, zorder=2, clip_on=True)
            ax.plot([true_x], [true_y], marker="o", markersize=2.2,
                    color=TEXT_MUT, alpha=0.7, zorder=2, clip_on=True)

    # No bbox_inches="tight": the figure height was deliberately computed by
    # layout_and_fit, and letting savefig re-crop to rendered ink risks
    # throwing off the fixed-fraction title/legend/footer placement again.
    Path(output_path).parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved chart → {output_path}")

    table_path = os.path.join(get_run_output_dir(), "weekly_earnings_risk.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(_build_weekly_table_text(week, week_start))
    print(f"Saved earnings risk table → {table_path}")
    return output_path


TIER_ORDER = {"High Alert": 0, "Elevated": 1, "Normal": 2}


def _build_weekly_table_text(week: pd.DataFrame, week_start: pd.Timestamp) -> str:
    week_label = f"Week of {week_start.strftime('%B %-d')}"
    lines = [f"  Earnings Risk — {week_label}", ""]

    col_w = {"date": 11, "ticker": 8, "tier": 12, "pct": 6, "hc": 4}
    header = (
        f"  {'DATE':<{col_w['date']}}"
        f"{'TICKER':<{col_w['ticker']}}"
        f"{'TIER':<{col_w['tier']}}"
        f"{'PCT':>{col_w['pct']}}"
        f"  {'HC':<{col_w['hc']}}"
    )
    divider = "  " + "-" * (sum(col_w.values()) + 4)
    lines.append(header)
    lines.append(divider)

    week = week.copy()
    # .astype(str) first: earnings_explosiveness_bucket is an ordered Categorical
    # (Normal < Elevated < High Alert), and .map() on a Categorical keeps that
    # category order for the result rather than the numeric order of TIER_ORDER.
    week["_tier_rank"] = week["earnings_explosiveness_bucket"].astype(str).map(TIER_ORDER).fillna(9)
    week = week.sort_values(["_tier_rank", "peer_percentile"], ascending=[True, False])

    for _, row in week.iterrows():
        date_str = row["earnings_date"].strftime("%a %b %-d")
        ticker   = row["stock"]
        tier     = row["earnings_explosiveness_bucket"]
        pct      = f"{row['peer_percentile']:.0f}th"
        hc       = "★" if row.get("is_high_conviction", False) else ""
        lines.append(
            f"  {date_str:<{col_w['date']}}"
            f"{ticker:<{col_w['ticker']}}"
            f"{tier:<{col_w['tier']}}"
            f"{pct:>{col_w['pct']}}"
            f"  {hc:<{col_w['hc']}}"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sd = sys.argv[1] if len(sys.argv) > 1 else None
    generate_weekly_earnings_chart(start_date=sd)
