"""
Shared collision-free label layout for the weekly/results earnings charts.

Both charts plot short stock-ticker labels at (day, value) positions, and it's
common for many stocks to share a day and land within a few points of each
other on the value axis. The old approach nudged a label's x position once,
the first time it landed within a fixed distance of an already-placed label
on the same day, and never checked again — under real crowding (e.g. a
30-stock Tuesday) that just stacks everything into the same unreadable spot.

This module lays labels for one day out on up to a few vertical lanes,
assigning by rank round-robin (so labels close in value end up in different
lanes instead of competing for the same lane), then runs a top-down greedy
cascade within each lane as a fallback for values that are truly identical
or near-identical. Box sizes are measured from the real rendered text extent
(so it already accounts for ticker length and 1- vs 2-line labels) rather
than guessed. A label that had to move off its true value gets a thin leader
line back to that value, so the chart's value axis stays truthful even under
crowding.
"""
from __future__ import annotations

from matplotlib.text import Text


def declutter_day(ax, texts: list[Text], max_lanes: int = 4) -> list[tuple[Text, float, float]]:
    """
    texts: Text artists for a single day/x-slot, each already placed at its
           true (x, y) position (x = the day's base x-coordinate, y = the
           true data value).

    Repositions each Text in place (via set_position) so their bounding
    boxes don't overlap. Returns (text, true_x, true_y) for every label whose
    y had to move, so the caller can draw a leader line back to the truth.
    """
    if not texts:
        return []

    fig = ax.figure
    renderer = fig.canvas.get_renderer()

    # ax.transData is affine, so a single unit sample gives an exact px/unit scale.
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, 1))
    px_per_x = abs(p1[0] - p0[0])
    px_per_y = abs(p1[1] - p0[1])

    fontsize_pt = texts[0].get_fontsize()
    dpi = fig.dpi
    # boxstyle pad is a fraction of fontsize; account for it on both sides
    # since get_window_extent measures glyph ink only, not the box patch.
    pad_px = 0.35 * fontsize_pt * dpi / 72.0 * 2

    items = []
    for text in texts:
        true_x, true_y = text.get_position()
        bbox = text.get_window_extent(renderer=renderer)
        items.append({
            "text": text,
            "true_x": true_x,
            "true_y": true_y,
            "h_units": (bbox.height + pad_px) / px_per_y,
            "w_units": (bbox.width + pad_px) / px_per_x,
        })

    items.sort(key=lambda it: it["true_y"], reverse=True)

    y_lo, y_hi = ax.get_ylim()
    y_span = y_hi - y_lo
    tallest = max(it["h_units"] for it in items)
    capacity = max(1, int(y_span // tallest))
    lanes = min(max_lanes, max(1, -(-len(items) // capacity)))  # ceil div

    widest = max(it["w_units"] for it in items)
    offsets = _lane_offsets(lanes, widest)

    buckets: list[list[dict]] = [[] for _ in range(lanes)]
    for rank, it in enumerate(items):
        buckets[rank % lanes].append(it)

    moved = []
    for lane_idx, bucket in enumerate(buckets):
        for it, y in _stack_lane(bucket, y_lo, y_hi):
            x = it["true_x"] + offsets[lane_idx]
            it["text"].set_position((x, y))
            if abs(y - it["true_y"]) > it["h_units"] * 0.15:
                moved.append((it["text"], it["true_x"], it["true_y"]))

    return moved


def _stack_lane(bucket: list[dict], y_lo: float, y_hi: float) -> list[tuple[dict, float]]:
    """
    Places one lane's items with the minimum gap enforced between neighbors,
    preserving true-value order. Spreads a tight cluster out from its own
    center rather than cascading strictly downward from the topmost item —
    a downward-only cascade runs out of room fast for a cluster sitting near
    the axis floor (the common case: most "Normal" tier scores/percentiles
    bunch up near the bottom of the range), overflowing well before the
    figure is anywhere near actually full.
    """
    n = len(bucket)
    if n == 0:
        return []
    ordered = sorted(bucket, key=lambda it: it["true_y"])
    if n == 1:
        return [(ordered[0], ordered[0]["true_y"])]

    gaps = [max(ordered[i]["h_units"], ordered[i + 1]["h_units"]) for i in range(n - 1)]
    true_ys = [it["true_y"] for it in ordered]
    positions = _isotonic_min_gap(true_ys, gaps)

    # Clamp the whole block (spacing intact) into the axis range if it fits.
    # `positions` is non-decreasing (same order as `ordered`), so the extreme
    # items are positions[0]/positions[-1] — clamp to y_lo/y_hi plus half
    # that item's own box height, not the bare limits, otherwise a box whose
    # *center* sits exactly on the axis floor still has its bottom half
    # hanging off the edge (invisible once clipped) no matter how much the
    # figure grows, since growth changes data-units-per-pixel, not the box's
    # fixed pixel size.
    lo_margin = ordered[0]["h_units"] / 2
    hi_margin = ordered[-1]["h_units"] / 2
    if positions[-1] > y_hi - hi_margin:
        shift = (y_hi - hi_margin) - positions[-1]
        positions = [p + shift for p in positions]
    if positions[0] < y_lo + lo_margin:
        shift = (y_lo + lo_margin) - positions[0]
        positions = [p + shift for p in positions]

    return list(zip(ordered, positions))


def _isotonic_min_gap(y_sorted: list[float], gaps: list[float]) -> list[float]:
    """
    Given ascending true values y_sorted and required minimum gaps between
    consecutive positions, returns positions p with p[i+1] - p[i] >= gaps[i]
    that minimize total squared displacement from y_sorted — via pool-
    adjacent-violators on the gap-subtracted sequence. Critically, wherever
    the true values are already spaced far enough apart, this leaves them
    exactly where they are: a naive "spread evenly around the cluster
    center" approach (tried first, and wrong) redistributed *any* pair
    needing separation, even two labels that were 50+ points apart already,
    visibly dragging a far-away label in just because it shared a lane.
    """
    n = len(y_sorted)
    cum = [0.0] * n
    for i in range(1, n):
        cum[i] = cum[i - 1] + gaps[i - 1]
    w = [y_sorted[i] - cum[i] for i in range(n)]

    values: list[float] = []
    counts: list[int] = []
    for wi in w:
        values.append(wi)
        counts.append(1)
        while len(values) > 1 and values[-2] > values[-1]:
            v2, c2 = values.pop(), counts.pop()
            v1, c1 = values.pop(), counts.pop()
            merged_count = c1 + c2
            values.append((v1 * c1 + v2 * c2) / merged_count)
            counts.append(merged_count)

    q = []
    for v, c in zip(values, counts):
        q.extend([v] * c)

    return [q[i] + cum[i] for i in range(n)]


def layout_and_fit(
    fig,
    ax,
    day_texts: dict,
    max_iter: int = 5,
    max_height_scale: float = 3.0,
    max_lanes_cap: int = 7,
) -> dict:
    """
    Runs declutter_day() for every day, and if labels still can't fit inside
    the axes' y-limits even at max_lanes lanes (an extreme same-day pile-up,
    e.g. 70+ stocks reporting on one day), grows the figure's height and the
    lane count together and retries. Caps growth at `max_height_scale` times
    the figure's original height so a single bad week can't produce an
    absurdly tall image; a few iterations is normally enough to converge.

    Call fig.tight_layout(...) yourself before this, once, to fix the axes'
    fractional position — growing the figure here only calls set_size_inches,
    never tight_layout again, since the axes' (left, bottom, width, height)
    fractions are invariant under a figure resize and re-running tight_layout
    mid-loop was found to shift the axes to an inconsistent position (day
    labels ending up floating mid-chart with a stray gap below them).

    day_texts: {day: [Text, ...]}, each Text carrying a `_true_xy` attribute
               set to its original (x, y) at creation time.

    Returns {day: moved_list} — see declutter_day's return value — for
    drawing leader lines.
    """
    fig_w, base_h = fig.get_size_inches()
    lanes = 4
    all_moved: dict = {}

    for attempt in range(max_iter):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        axes_bbox = ax.get_window_extent(renderer=renderer)

        all_moved = {}
        worst_overflow_px = 0.0
        for d, texts in day_texts.items():
            all_moved[d] = declutter_day(ax, texts, max_lanes=lanes)
            for text in texts:
                # Compare the *rendered box*, not just its center point — a
                # label whose center sits right at the axis limit still has
                # roughly half its box hanging off the edge, and a
                # center-only check missed that entirely (labels were being
                # silently clipped in half at the top/bottom of dense charts
                # while this loop reported zero overflow).
                tb = text.get_window_extent(renderer=renderer)
                if tb.ymin < axes_bbox.ymin:
                    worst_overflow_px = max(worst_overflow_px, axes_bbox.ymin - tb.ymin)
                if tb.ymax > axes_bbox.ymax:
                    worst_overflow_px = max(worst_overflow_px, tb.ymax - axes_bbox.ymax)

        if worst_overflow_px <= 1.0 or attempt == max_iter - 1:
            return all_moved

        axes_h_px = axes_bbox.height
        cur_h = fig.get_size_inches()[1]
        needed_h = cur_h * (axes_h_px + worst_overflow_px * 1.15) / axes_h_px
        new_h = min(needed_h, base_h * max_height_scale)
        lanes = min(max_lanes_cap, lanes + 1)
        fig.set_size_inches(fig_w, new_h)
        for texts in day_texts.values():
            for text in texts:
                text.set_position(text._true_xy)

    return all_moved


def _lane_offsets(n: int, unit: float) -> list[float]:
    if n <= 1:
        return [0.0]
    step = unit * 1.15
    return [(i - (n - 1) / 2) * step for i in range(n)]
