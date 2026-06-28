"""
Last week's earnings results chart for social sharing.
Called by analysis/last_week_results.py; can also run standalone.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import date, timedelta
from pathlib import Path


TIER_COLORS = {
    "High Alert": "#c9a84c",
    "Elevated":   "#1e1a08",
    "Normal":     "#ffffff",
}
TIER_TEXT_COLORS = {
    "High Alert": "#0a0a0a",
    "Elevated":   "#c8a035",
    "Normal":     "#141820",
}
BG       = "#0a0a0a"
BORDER   = "#2a2a28"
TEXT_DIM = "#e8e8e8"
TEXT_MUT = "#b0aca8"
ACCENT   = "#c9a84c"

ELEVATED_FLOOR   = 73
HIGH_ALERT_FLOOR = 79


def generate_results_chart(
    earnings: pd.DataFrame,
    week_start: pd.Timestamp,
    output_path: str = "output/results_chart.png",
) -> str:
    """
    earnings: rows with is_earnings_day==1 for last week, already filtered.
              Must have: stock, earnings_date, earnings_explosiveness_bucket,
              earnings_explosiveness_score, move (decimal), is_high_conviction.
    week_start: Monday of the week (used for x-axis and title).
    """
    all_days = [week_start.date() + timedelta(days=i) for i in range(5)]

    fig, ax = plt.subplots(figsize=(max(10, 5 * 2.2), 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.axhspan(40,                ELEVATED_FLOOR,   color="#0a0a0a", zorder=0)
    ax.axhspan(ELEVATED_FLOOR,   HIGH_ALERT_FLOOR, color="#0e0d0b", zorder=0)
    ax.axhspan(HIGH_ALERT_FLOOR, 103,              color="#121008", zorder=0)

    for y in [ELEVATED_FLOOR, HIGH_ALERT_FLOOR]:
        ax.axhline(y, color=BORDER, linewidth=0.6, zorder=1)

    ax.text(4.45, 91, "HIGH ALERT", va="center", ha="right",
            color=ACCENT,    fontsize=7, fontfamily="monospace", alpha=0.80)
    ax.text(4.45, 76, "ELEVATED",   va="center", ha="right",
            color="#c8a035", fontsize=7, fontfamily="monospace", alpha=0.80)
    ax.text(4.45, 56, "NORMAL",     va="center", ha="right",
            color="#ffffff", fontsize=7, fontfamily="monospace", alpha=0.80)

    date_to_x = {d: i for i, d in enumerate(all_days)}
    placed: dict = {d: [] for d in all_days}

    for _, row in earnings.iterrows():
        d = row["earnings_date"].date()
        if d not in date_to_x:
            continue

        x_raw = date_to_x[d]
        y     = float(row["earnings_explosiveness_score"])
        tier  = row["earnings_explosiveness_bucket"]
        hc    = bool(row.get("is_high_conviction", False))
        move  = float(row["move"])

        sign     = "+" if move >= 0 else "−"
        move_str = f"{sign}{abs(move) * 100:.1f}%"
        prefix   = "★ " if hc else ""
        label    = f"{prefix}{row['stock']}\n{move_str}"

        x = float(x_raw)
        for py in placed[d]:
            if abs(y - py) < 5:
                x += 0.20
                break
        placed[d].append(y)

        fc = TIER_COLORS.get(tier, TIER_COLORS["Normal"])
        tc = TIER_TEXT_COLORS.get(tier, TIER_TEXT_COLORS["Normal"])
        fw = "bold" if tier == "High Alert" else "normal"
        fs = 8.5 if tier in ("High Alert", "Elevated") else 7.5
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=tc,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=fc, edgecolor="none"),
                zorder=3, fontfamily="monospace", linespacing=1.4)

    ax.set_xticks(range(5))
    ax.set_xticklabels(
        [d.strftime("%a %b %-d") for d in all_days],
        color=TEXT_DIM, fontsize=10, fontfamily="monospace",
    )
    ax.tick_params(axis="x", colors=TEXT_DIM, length=0, pad=10)

    ax.set_ylim(40, 103)
    ax.set_xlim(-0.55, 4.45)
    ax.set_yticks([ELEVATED_FLOOR, HIGH_ALERT_FLOOR])
    ax.set_yticklabels(["73", "79"], color=TEXT_MUT, fontsize=8, fontfamily="monospace")
    ax.tick_params(axis="y", colors=TEXT_MUT, length=0)
    ax.set_ylabel("Risk Score", color=TEXT_MUT, fontsize=9,
                  fontfamily="monospace", labelpad=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    week_label = f"Week of {week_start.strftime('%B %-d')}"
    fig.text(0.5, 0.96, "Last Week's Results", ha="center", va="top",
             color="#e8e4dc", fontsize=16, fontfamily="serif", fontstyle="italic")
    fig.text(0.5, 0.91, week_label, ha="center", va="top",
             color=ACCENT, fontsize=10, fontfamily="monospace", fontweight="500")

    patches = [
        mpatches.Patch(facecolor=TIER_COLORS["High Alert"], label="High Alert"),
        mpatches.Patch(facecolor=TIER_COLORS["Elevated"],   label="Elevated"),
        mpatches.Patch(facecolor=TIER_COLORS["Normal"],     label="Normal"),
    ]
    ax.legend(handles=patches, loc="lower right", frameon=False,
              labelcolor=TEXT_DIM, prop={"family": "monospace", "size": 8})

    fig.text(0.5, 0.02, "harbor-markets.com  ·  Breakwater",
             ha="center", color=TEXT_MUT, fontsize=8, fontfamily="monospace")

    plt.tight_layout(rect=(0, 0.05, 1, 0.90))
    Path(output_path).parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved results chart → {output_path}")
    return output_path


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from analysis.last_week_results import print_last_week_results
    print_last_week_results()
