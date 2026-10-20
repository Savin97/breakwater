"""
Last week's results chart for social sharing (X): the calls we PUBLISHED for last week's
events, each placed at the size of its corrected move.

Drawn by analysis/last_week_results.py from its results table (archived 0.4.0 calls —
the first eligible publication per event — joined to the event frame's corrected
outcome). It never re-scores the past: a week with no archived 0.4.0 call draws nothing.

Y axis = size of the corrected 3-session move, so every day's column reads largest move
first; the label carries the signed move. The box colour is the tier as published,
before the report. Events whose move cannot be measured yet (no verified announcement
time, or the window has not closed) have no position: they are named under the chart,
never drawn as a small move.

This chart shows outcomes only. It carries no hit count, rate or other summary figure.

    python -m analysis.chart_last_week              # last week
    python -m analysis.chart_last_week --lookback 2
"""
import math
import os
from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from analysis.chart_layout import layout_and_fit
from analysis.chart_weekly import (TIER_COLORS, TIER_TEXT_COLORS, BG, BORDER, TEXT_DIM,
                                   TEXT_MUT, ACCENT)
from feature_engineering.announcement_timing import TARGET_AVAILABLE
from scoring.model_b import HIGH_ALERT, ELEVATED, NORMAL, UNSCORED
from utilities.output_utilities import get_run_output_dir

UNSCORED_COLOR = "#3a3a38"
UNSCORED_TEXT = "#e8e8e8"
MIN_Y_TOP = 12.0          # percent; a quiet week still shows a readable scale
MOVE_NOTE = ("Move = 3 trading sessions from the last close before the announcement. "
             "Tier as published before the report.")
NOT_PLOTTED_NOTE = "Move not measurable yet: "
UNSCORED_LEGEND = "Unscored (insufficient history — not low risk)"


def chart_rows(tbl: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Split the results table into (plotted rows, unmeasured tickers).

    Plotted rows have a corrected move (`reaction_3d_anchored_status == available`) and
    are ordered by day, then by the SIZE of the move, largest first, then ticker. `y` is
    that size in percent. The legacy report-date reaction is never read.
    """
    measured = tbl["reaction_3d_anchored_status"].eq(TARGET_AVAILABLE) & \
        tbl["reaction_3d_anchored"].notna()
    plotted = tbl[measured].copy()
    plotted["earnings_date"] = pd.to_datetime(plotted["earnings_date"])
    plotted["y"] = plotted["reaction_3d_anchored"].abs() * 100
    plotted = plotted.sort_values(["earnings_date", "y", "stock"],
                                  ascending=[True, False, True]).reset_index(drop=True)
    unmeasured = sorted(tbl.loc[~measured, "stock"])
    return plotted, unmeasured


def _signed(move: float) -> str:
    return f"{'+' if move >= 0 else '−'}{abs(move) * 100:.1f}%"


def generate_last_week_chart(tbl: pd.DataFrame, week_start, output_path: str | None = None) -> str:
    """tbl: analysis.last_week_results.results_table output for one Mon-Fri week.
    Returns the PNG path, or "" when there is no archived call to show."""
    if tbl.empty:
        print("Last week's chart: no archived 0.4.0 call for last week — nothing drawn.")
        return ""
    if output_path is None:
        output_path = os.path.join(get_run_output_dir(), "results_chart.png")

    week_start = pd.Timestamp(week_start)
    plotted, unmeasured = chart_rows(tbl)
    all_days = [week_start.date() + timedelta(days=i) for i in range(5)]
    y_top = max(MIN_Y_TOP, math.ceil(plotted["y"].max() * 1.08) if len(plotted) else 0)

    fig, ax = plt.subplots(figsize=(max(10, 5 * 2.2), 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    step = 5 if y_top <= 30 else 10
    ticks = list(range(0, int(y_top) + 1, step))
    for y in ticks[1:]:
        ax.axhline(y, color=BORDER, linewidth=0.6, zorder=1)

    ax.set_xticks(range(5))
    ax.set_xticklabels([d.strftime("%a %b %-d") for d in all_days],
                       color=TEXT_DIM, fontsize=10, fontfamily="monospace")
    ax.tick_params(axis="x", colors=TEXT_DIM, length=0, pad=10)
    ax.set_ylim(0, y_top)
    ax.set_xlim(-0.55, 4.45)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t}%" for t in ticks], color=TEXT_MUT, fontsize=8,
                       fontfamily="monospace")
    ax.tick_params(axis="y", colors=TEXT_MUT, length=0)
    ax.set_ylabel("Size of earnings move", color=TEXT_MUT, fontsize=9,
                  fontfamily="monospace", labelpad=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(0.5, 0.96, "Last Week's Results", ha="center", va="top",
             color="#e8e4dc", fontsize=16, fontfamily="serif", fontstyle="italic")
    fig.text(0.5, 0.91, f"Week of {week_start.strftime('%B %-d')}", ha="center", va="top",
             color=ACCENT, fontsize=10, fontfamily="monospace", fontweight="500")

    tiers = [HIGH_ALERT, ELEVATED, NORMAL]
    patches = [mpatches.Patch(facecolor=TIER_COLORS[t], label=t) for t in tiers]
    if tbl["tier"].eq(UNSCORED).any():
        patches.append(mpatches.Patch(facecolor=UNSCORED_COLOR, label=UNSCORED_LEGEND))
    fig.legend(handles=patches, loc="center", bbox_to_anchor=(0.5, 0.100), ncol=len(patches),
               frameon=False, labelcolor=TEXT_DIM, prop={"family": "monospace", "size": 8},
               columnspacing=1.2, handletextpad=0.5)

    notes = [MOVE_NOTE]
    if unmeasured:
        notes.append(NOT_PLOTTED_NOTE + ", ".join(unmeasured))
    fig.text(0.5, 0.074, "\n".join(notes), ha="center", va="top", color=TEXT_MUT,
             fontsize=7, fontfamily="monospace", linespacing=1.5)
    fig.text(0.5, 0.010, "harbor-markets.com  ·  Breakwater",
             ha="center", color=TEXT_MUT, fontsize=8, fontfamily="monospace")

    # Fix the axes' position before placing labels (see chart_layout.layout_and_fit).
    fig.tight_layout(rect=(0, 0.125, 1, 0.90))

    date_to_x = {d: i for i, d in enumerate(all_days)}
    day_texts: dict = {d: [] for d in all_days}
    for _, row in plotted.iterrows():
        d = row["earnings_date"].date()
        if d not in date_to_x:
            continue
        tier = row["tier"]
        if tier == UNSCORED:
            fc, tc = UNSCORED_COLOR, UNSCORED_TEXT
        else:
            fc = TIER_COLORS.get(tier, TIER_COLORS[NORMAL])
            tc = TIER_TEXT_COLORS.get(tier, TIER_TEXT_COLORS[NORMAL])
        x, y = float(date_to_x[d]), float(row["y"])
        text = ax.text(x, y, f"{row['stock']}\n{_signed(row['reaction_3d_anchored'])}",
                       ha="center", va="center",
                       fontsize=8.5 if tier in (HIGH_ALERT, ELEVATED) else 7.5,
                       fontweight="bold" if tier == HIGH_ALERT else "normal", color=tc,
                       bbox=dict(boxstyle="round,pad=0.35", facecolor=fc, edgecolor="none"),
                       zorder=3, fontfamily="monospace", linespacing=1.4, clip_on=True)
        text._true_xy = (x, y)
        day_texts[d].append(text)

    # A crowded day spreads into lanes; any label moved off its true height gets a
    # leader line back to it, so the height stays the measured move.
    for moved in layout_and_fit(fig, ax, day_texts).values():
        for text, true_x, true_y in moved:
            fx, fy = text.get_position()
            ax.plot([true_x, fx], [true_y, fy], color=TEXT_MUT,
                    linewidth=0.6, alpha=0.45, zorder=2, clip_on=True)
            ax.plot([true_x], [true_y], marker="o", markersize=2.2,
                    color=TEXT_MUT, alpha=0.7, zorder=2, clip_on=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"Saved last week's chart → {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    from analysis.last_week_results import generate_last_week_results
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=1)
    generate_last_week_results(lookback_weeks=parser.parse_args().lookback)
