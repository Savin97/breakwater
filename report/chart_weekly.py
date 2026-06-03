"""
Weekly earnings risk chart for social sharing.
Reads output/upcoming_df.parquet, generates a Twitter-ready PNG.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from datetime import date, timedelta
import numpy as np
from pathlib import Path


TIER_COLORS = {
    "High Alert": "#c9a84c",
    "Elevated":   "#2e2c28",
    "Normal":     None,
}
TIER_TEXT_COLORS = {
    "High Alert": "#0a0a0a",
    "Elevated":   "#ffffff",
    "Normal":     "#e8e8e8",
}
BG       = "#0a0a0a"
BORDER   = "#2a2a28"
TEXT_DIM = "#e8e8e8"
TEXT_MUT = "#b0aca8"
ACCENT   = "#c9a84c"


def generate_weekly_earnings_chart(
    days_ahead: int = 7,
    output_path: str = "output/weekly_chart.png",
    parquet_path: str = "output/upcoming_df.parquet",
) -> str:
    df = pd.read_parquet(parquet_path)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])

    today     = pd.Timestamp(date.today())
    week_end  = today + timedelta(days=days_ahead)
    week = df[(df["earnings_date"] >= today) & (df["earnings_date"] <= week_end)].copy()

    if week.empty:
        print("No upcoming events in window.")
        return ""

    week = week.sort_values(["earnings_date", "peer_percentile"], ascending=[True, False])
    dates = sorted(week["earnings_date"].dt.date.unique())
    n_days = len(dates)

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
    ax.text(n_days - 0.45, 95,  "HIGH ALERT", va="center", ha="right",
            color=ACCENT, fontsize=7, fontfamily="monospace", alpha=0.55)
    ax.text(n_days - 0.45, 82,  "ELEVATED",   va="center", ha="right",
            color="#7a6a42", fontsize=7, fontfamily="monospace", alpha=0.55)
    ax.text(n_days - 0.45, 38,  "NORMAL",     va="center", ha="right",
            color=TEXT_MUT, fontsize=7, fontfamily="monospace", alpha=0.45)

    # ── Plot stocks ────────────────────────────────────────────────────────
    date_to_x = {d: i for i, d in enumerate(dates)}

    # Track placed positions to avoid overlap within same day
    placed: dict[date, list[float]] = {d: [] for d in dates}

    for _, row in week.iterrows():
        d     = row["earnings_date"].date()
        x_raw = date_to_x[d]
        y     = float(row["peer_percentile"])
        tier  = row["earnings_explosiveness_bucket"]
        hc    = row.get("is_high_conviction", False)
        label = ("★ " if hc else "") + row["stock"]

        # Nudge x if too close to an already-placed item on same day
        x = float(x_raw)
        for py in placed[d]:
            if abs(y - py) < 6:
                x += 0.18
                break
        placed[d].append(y)

        fc = TIER_COLORS.get(tier)
        tc = TIER_TEXT_COLORS.get(tier, TIER_TEXT_COLORS["Normal"])

        if fc is not None:
            box = dict(boxstyle="round,pad=0.35", facecolor=fc, edgecolor="none")
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=9, fontweight="bold" if tier == "High Alert" else "normal",
                    color=tc, bbox=box, zorder=3, fontfamily="monospace")
        else:
            ax.text(x, y, label, ha="center", va="center",
                    fontsize=8, fontweight="normal",
                    color=tc, zorder=3, fontfamily="monospace")

    # ── X-axis: day labels ─────────────────────────────────────────────────
    ax.set_xticks(range(n_days))
    ax.set_xticklabels(
        [d.strftime("%a %b %-d") for d in dates],
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
    week_label = f"Week of {today.strftime('%B %-d')}"
    fig.text(0.5, 0.96, "Earnings Risk", ha="center", va="top",
             color="#e8e4dc", fontsize=16, fontfamily="serif", fontstyle="italic")
    fig.text(0.5, 0.91, week_label, ha="center", va="top",
             color=ACCENT, fontsize=10, fontfamily="monospace",
             fontweight="500")

    # ── Legend ─────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=TIER_COLORS["High Alert"], label="High Alert"),
        mpatches.Patch(color=TIER_COLORS["Elevated"],   label="Elevated"),
        mpatches.Patch(color="#3a3835",                 label="Normal"),
    ]
    leg = ax.legend(
        handles=patches, loc="lower right", frameon=False,
        labelcolor=TEXT_DIM, prop={"family": "monospace", "size": 8},
    )

    # ── Footer ─────────────────────────────────────────────────────────────
    fig.text(0.5, 0.02, "harbor-markets.com  ·  Breakwater",
             ha="center", color=TEXT_MUT, fontsize=8, fontfamily="monospace")

    plt.tight_layout(rect=[0, 0.05, 1, 0.90])
    Path(output_path).parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved chart → {output_path}")
    return output_path


if __name__ == "__main__":
    generate_weekly_earnings_chart()
