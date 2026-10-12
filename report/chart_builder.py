# report/chart_builder.py
import io
import re
import warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import EXTREME_EARNINGS_REACTION_THRESHOLD
from feature_engineering.announcement_timing import TARGET_AVAILABLE

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

_COLOR_POS      = "#27ae60"
_COLOR_NEG      = "#c0392b"
_COLOR_POS_EXT  = "#1a7a40"
_COLOR_NEG_EXT  = "#7b241c"
_COLOR_THRESH   = "#e74c3c"


def corrected_reactions(history: pd.DataFrame) -> pd.DataFrame:
    """The stock's completed events with an AVAILABLE corrected 3-session reaction,
    oldest first. Events without verified timing (or with an unavailable endpoint) are
    left out — never plotted as zero, never filled from the legacy reaction."""
    h = history[~history["is_pending"].astype(bool)]
    h = h[h["reaction_3d_anchored_status"].eq(TARGET_AVAILABLE)
          & h["reaction_3d_anchored"].notna()]
    return h.sort_values("earnings_date")


def reactions_chart_summary(history: pd.DataFrame, n: int = 16) -> dict:
    """What the chart shows: how many corrected reactions are plotted, how many exist,
    and how many completed events inside the plotted period were omitted."""
    usable = corrected_reactions(history)
    shown = usable.tail(n)
    omitted = 0
    if len(shown):
        completed = history[~history["is_pending"].astype(bool)]
        in_span = completed["earnings_date"] >= shown["earnings_date"].min()
        omitted = int(in_span.sum()) - len(shown)
    return {"n_shown": len(shown), "n_usable": len(usable), "n_omitted_in_span": omitted}


def generate_reactions_chart(history: pd.DataFrame, n: int = 16) -> str:
    """SVG bar chart of the stock's last `n` corrected (timestamp-anchored) 3-session
    earnings reactions. Empty string when there are none."""
    recent = corrected_reactions(history).tail(n)
    if recent.empty:
        return ""
    reactions = recent["reaction_3d_anchored"].to_numpy() * 100
    dates = [pd.Timestamp(d).strftime("%b '%y") for d in recent["earnings_date"]]
    thresh_pct = EXTREME_EARNINGS_REACTION_THRESHOLD * 100

    colors = []
    for r in reactions:
        if abs(r) >= thresh_pct:
            colors.append(_COLOR_POS_EXT if r >= 0 else _COLOR_NEG_EXT)
        else:
            colors.append(_COLOR_POS if r >= 0 else _COLOR_NEG)

    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.bar(range(len(reactions)), reactions, color=colors, width=0.65, zorder=3)

    ax.axhline(y= thresh_pct, color=_COLOR_THRESH, linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)
    ax.axhline(y=-thresh_pct, color=_COLOR_THRESH, linestyle="--", linewidth=0.8, alpha=0.6, zorder=2)
    ax.axhline(y=0, color="#333", linewidth=0.5, zorder=2)

    xmax = len(reactions) - 0.5
    ax.text(xmax + 0.05, thresh_pct,  "+8%", va="center", ha="left", fontsize=6.5,
            color=_COLOR_THRESH, alpha=0.8)
    ax.text(xmax + 0.05, -thresh_pct, "−8%", va="center", ha="left", fontsize=6.5,
            color=_COLOR_THRESH, alpha=0.8)
    ax.set_xlim(-0.6, xmax + 0.8)

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=45, ha="right", fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="x", length=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#ccc")
    ax.spines["bottom"].set_color("#ccc")
    ax.grid(axis="y", color="#ebebeb", linewidth=0.6, zorder=1)

    fig.tight_layout(pad=0.4)

    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)

    svg = buf.getvalue()
    # Make the SVG width-responsive so it fills the report column
    svg = re.sub(r'(<svg[^>]*)\swidth="[^"]*"', r'\1 width="100%"', svg)
    svg = re.sub(r'(<svg[^>]*)\sheight="[^"]*"', r'\1', svg)
    return svg
