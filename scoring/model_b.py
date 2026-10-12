# scoring/model_b.py
"""
Model B — Breakwater's production earnings-magnitude model (MODEL_VERSION 0.4.0).

What it says
------------
For each earnings event, the raw signal is the stock's own track record of large moves:

    model_b_hist_mean_abs = mean(abs_reaction_3d_anchored) over the stock's PRIOR events
                            whose corrected 3-session outcome was complete on or before
                            this event's call cutoff

and an event needs at least MIN_PRIOR (8) such outcomes to be scored. Fewer, and the
event is `Unscored` — never zero risk, never filled in from anything else.

The model predicts MAGNITUDE risk only, never direction. It was validated against
`abs_reaction_3d_anchored >= 0.08` (analysis/model_c_refit/, SPEC.md and RESULTS.md);
that validation is research and production does not import it.

The rules that must not be relaxed
----------------------------------
* Only corrected, anchored, completed prior outcomes count
  (`reaction_3d_anchored_status == "available"`). The legacy `abs_reaction_3d`, which
  mismeasures before-open announcements, is never read here. Unresolved prior events are
  skipped and counted nowhere — not as zero, not as a guess.
* A prior outcome counts only once its endpoint session (anchor + 3 market sessions on
  the canonical grid) is ON OR BEFORE the call cutoff. An event never sees its own
  outcome.
* The call cutoff is the last market session strictly before the Monday of the report
  week — the close the weekend run scores from. A run later in the report week keeps
  that cutoff. An event further out, whose cutoff has not happened yet, uses the latest
  session actually in the data; a future cutoff is impossible by construction.
* The event being scored does not need its own announcement timestamp. Its timing is
  not part of Model B; only its PRIOR events' corrected outcomes are.

Score and tier
--------------
`risk_score` is the **Breakwater Risk Percentile**: where the raw signal falls in a
FROZEN reference distribution (`model_b_reference.json`, the 2014–2025 production-fit
sample). It is a rank, not a probability. The reference is a presentation scale only:
the raw signal of every event — historical or upcoming — is computed causally from its
own call cutoff, and the reference never feeds back into it.

    High Alert  risk_score >= 90   top decile of the reference distribution
    Elevated    80 <= score < 90   80th-90th percentile
    Normal      score < 80
    Unscored    fewer than 8 usable prior corrected reactions

The bands are descriptive percentile bands, not cut points fitted on outcomes. Nothing
else — drift, surprise, lift, volatility, options — alters the score or the tier.

Scope — what kind of model this is
----------------------------------
Model B is primarily a CROSS-SECTIONAL structural earnings-move-risk model. It
distinguishes stocks that historically have larger corrected earnings reactions from
stocks that have smaller ones. It has little demonstrated ability to identify which
quarter of the same stock will be unusually large (testing/metric_testing.py, section
"between- vs within-stock decomposition"). That is the design, not a defect: describe it
as structural earnings-move risk for upcoming earnings events, never as a quarter-timing
or event-specific forecast.

Unscored is coverage, not risk: an event with fewer than MIN_PRIOR usable prior corrected
reactions has too little verified history to evaluate. Short-history events are not
calmer — do not present Unscored as, or sort it as if it were, a level below Normal.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.announcement_timing import TARGET_AVAILABLE, market_session_grid

MODEL_NAME = "Model B"
TARGET = "abs_reaction_3d_anchored"
TARGET_STATUS = "reaction_3d_anchored_status"
OUTCOME_SESSIONS = 3           # the corrected target spans 3 post-announcement sessions
MIN_PRIOR = 8

HIGH_ALERT = "High Alert"
ELEVATED = "Elevated"
NORMAL = "Normal"
UNSCORED = "Unscored"
TIERS = (HIGH_ALERT, ELEVATED, NORMAL, UNSCORED)    # display order, riskiest first
HIGH_ALERT_FLOOR = 90.0
ELEVATED_FLOOR = 80.0

REFERENCE_PATH = Path(__file__).with_name("model_b_reference.json")

# Columns score_events() adds to the event frame.
MODEL_B_COLS = ["score_asof_date", "model_b_n_prior", "model_b_hist_mean_abs",
                "risk_score", "risk_tier", "model_version"]

SIGNAL_DEFINITION = (
    "mean of abs_reaction_3d_anchored over the stock's prior events with "
    "reaction_3d_anchored_status == 'available' whose endpoint session (anchor + 3 market "
    "sessions) is on or before the call cutoff (last market session strictly before the "
    "Monday of the report week); at least 8 such events")


# ───────────────────────────────────────── clocks ───────────────────────────────────────
def _grid_index(grid: np.ndarray, dates) -> np.ndarray:
    """Grid position of each date that is a session, else -1."""
    d = pd.to_datetime(pd.Series(dates)).to_numpy(dtype="datetime64[ns]")
    if len(grid) == 0:
        return np.full(len(d), -1, dtype=np.int64)
    pos = np.searchsorted(grid, d, side="left")
    hit = ~pd.isna(d) & (pos < len(grid))
    hit &= grid[np.minimum(pos, len(grid) - 1)] == d
    return np.where(hit, pos, -1).astype(np.int64)


def call_monday(report_dates) -> pd.Series:
    """Monday of the week each event reports in."""
    rd = pd.to_datetime(pd.Series(report_dates)).dt.normalize()
    return rd - pd.to_timedelta(rd.dt.weekday, unit="D")


def call_cutoff_index(grid: np.ndarray, report_dates) -> np.ndarray:
    """Grid position of each event's call cutoff, or -1 where none exists.

    The cutoff is the last session strictly before the Monday of the report week. When
    that Monday lies beyond the end of the grid (an event weeks away), the result is the
    last session in the grid: the latest close actually available, never a future one.
    A run on Wednesday of the report week gets the same answer as the weekend run.
    """
    monday = call_monday(report_dates).to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(grid, monday, side="left") - 1
    return np.where(pd.isna(monday), -1, idx).astype(np.int64)


def outcome_endpoint_index(grid: np.ndarray, anchor_dates) -> np.ndarray:
    """Grid position of each event's outcome endpoint — the session whose close ends
    `reaction_3d_anchored` — or -1 where the anchor is not a session or the endpoint is
    past the grid."""
    anchor = _grid_index(grid, anchor_dates)
    end = np.where(anchor >= 0, anchor + OUTCOME_SESSIONS, -1)
    return np.where(end < len(grid), end, -1).astype(np.int64)


# ──────────────────────────────────────── outcomes ──────────────────────────────────────
def usable_outcome(events: pd.DataFrame) -> pd.Series:
    """The corrected 3-session |reaction| where it may enter a later event's history:
    a completed event whose anchored target is genuinely available. NaN everywhere else
    — pending, unresolved timing, endpoint unavailable — and NaN is skipped, not zero."""
    ok = (~events["is_pending"].astype(bool)
          & events[TARGET_STATUS].eq(TARGET_AVAILABLE)
          & events[TARGET].notna())
    return pd.to_numeric(events[TARGET], errors="coerce").where(ok)


# ───────────────────────────────────── scoring steps ────────────────────────────────────
# Each step takes the event frame, adds its columns and returns it — the same shape as the
# engineer_* functions in feature_engineering/. Stage 4 calls them in order.

def engineer_score_asof_date(events: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """`score_asof_date`: each event's call cutoff — the last market session before the
    Monday of its report week, or the latest session in the data if that has not
    happened yet. NaT only when the data holds no session before it."""
    grid = market_session_grid(daily_df)
    cut = call_cutoff_index(grid, events["earnings_date"].values)
    asof = np.full(len(cut), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    asof[cut >= 0] = grid[cut[cut >= 0]]
    events["score_asof_date"] = asof
    return events


def engineer_prior_history(events: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """`model_b_n_prior` / `model_b_hist_mean_abs`: the stock's usable corrected prior
    outcomes whose endpoint session is on or before `score_asof_date`, and their mean.

    Prior outcomes are ordered by the session they became known (endpoint), ties by report
    date, so the result does not depend on the order of the input rows.
    """
    grid = market_session_grid(daily_df)
    n = len(events)
    cut = _grid_index(grid, events["score_asof_date"].values)
    end = outcome_endpoint_index(grid, events["anchor_date"].values)
    val = usable_outcome(events).to_numpy(float)
    ed = pd.to_datetime(events["earnings_date"]).to_numpy(dtype="datetime64[ns]")
    usable = np.isfinite(val) & (end >= 0)

    n_prior = np.zeros(n, dtype=np.int64)
    total = np.zeros(n)
    for _stock, rows in events.groupby("stock", sort=False).indices.items():
        have = rows[usable[rows]]
        order = have[np.lexsort((ed[have], end[have]))]
        e_sorted = end[order]
        csum = np.concatenate([[0.0], np.cumsum(val[order])])
        c = cut[rows]
        k = np.where(c >= 0, np.searchsorted(e_sorted, c, side="right"), 0)
        n_prior[rows] = k
        total[rows] = csum[k]

    # An event's own outcome ends at least two sessions after its report date, so it can
    # never precede its own cutoff. Excluded by identity regardless.
    own = usable & (cut >= 0) & (end <= cut)
    n_prior = n_prior - own
    total = total - np.where(own, val, 0.0)

    events["model_b_n_prior"] = n_prior.astype("int64")
    events["model_b_hist_mean_abs"] = np.where(n_prior > 0, total / np.maximum(n_prior, 1),
                                               np.nan)
    return events


def raw_signal(events: pd.DataFrame, daily_df: pd.DataFrame,
               min_prior: int = MIN_PRIOR) -> pd.DataFrame:
    """The causal Model B signal per event without touching `events`: `score_asof_date`,
    `model_b_n_prior`, `model_b_hist_mean_abs` and `model_b_scored` (at least
    `min_prior` usable priors). Used by the reference builder and the validation."""
    cols = ["stock", "earnings_date", "anchor_date", "is_pending", TARGET, TARGET_STATUS]
    sig = events[cols].copy()
    sig = engineer_score_asof_date(sig, daily_df)
    sig = engineer_prior_history(sig, daily_df)
    sig = sig[["model_b_n_prior", "model_b_hist_mean_abs", "score_asof_date"]]
    sig["model_b_scored"] = sig["model_b_n_prior"].ge(min_prior) & \
        sig["model_b_hist_mean_abs"].notna()
    return sig


# ─────────────────────────────────────── reference ──────────────────────────────────────
@dataclass(frozen=True)
class Reference:
    """The frozen presentation scale: raw signal → Breakwater Risk Percentile."""
    values: np.ndarray        # strictly increasing raw-signal quantiles
    percentiles: np.ndarray   # percentile (0-100) at each value
    meta: dict


def reference_checksum(payload: dict) -> str:
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def reference_from_payload(payload: dict) -> Reference:
    if payload.get("sha256") != reference_checksum(payload):
        raise ValueError("Model B reference artifact failed its checksum — it was edited "
                         "by hand or corrupted. Regenerate it with "
                         "scoring/build_model_b_reference.py; never patch it.")
    if payload.get("min_prior") != MIN_PRIOR:
        raise ValueError(f"reference built with min_prior={payload.get('min_prior')}, "
                         f"production uses {MIN_PRIOR}")
    q = payload["quantiles"]
    pct = np.asarray(q["percentiles"], float)
    val = np.asarray(q["values"], float)
    if len(pct) != len(val) or len(val) < 2:
        raise ValueError("malformed reference quantile grid")
    if np.any(np.diff(pct) <= 0) or np.any(np.diff(val) < 0):
        raise ValueError("reference quantile grid is not monotonic")
    # Where several percentiles share one raw value, that value sits at the highest of
    # them (the share of the reference at or below it) — makes the map a function.
    uniq, first = np.unique(val[::-1], return_index=True)
    top = pct[::-1][first]
    return Reference(values=uniq, percentiles=top, meta=payload)


@lru_cache(maxsize=4)
def _load(path: str) -> Reference:
    return reference_from_payload(json.loads(Path(path).read_text()))


def load_reference(path: Path | str = REFERENCE_PATH) -> Reference:
    return _load(str(path))


def to_percentile(raw, reference: Reference) -> np.ndarray:
    """Monotonic map of the raw signal onto the reference distribution, 0-100. Values
    outside the reference range clip to 0 / 100; NaN stays NaN."""
    x = np.asarray(raw, float)
    out = np.interp(x, reference.values, reference.percentiles)
    return np.where(np.isnan(x), np.nan, out)


def assign_tier(risk_score) -> pd.Series:
    s = pd.Series(risk_score, dtype=float)
    tier = pd.Series(NORMAL, index=s.index, dtype=object)
    tier[s >= ELEVATED_FLOOR] = ELEVATED
    tier[s >= HIGH_ALERT_FLOOR] = HIGH_ALERT
    tier[s.isna()] = UNSCORED
    return tier


def display_score(risk_score) -> int | None:
    """The whole-number percentile shown to users. Floored, not rounded, so the number
    shown always sits in the band of its tier (89.6 shows as 89, Elevated)."""
    if risk_score is None or pd.isna(risk_score):
        return None
    return int(math.floor(float(risk_score)))


def sort_by_risk(df: pd.DataFrame, by=()) -> pd.DataFrame:
    """THE ordering of every customer list and maintainer sheet: within the `by` columns
    (e.g. earnings_date), scored events before Unscored, then the RAW signal
    `model_b_hist_mean_abs` descending, then ticker.

    Never sorted on `risk_score`: the percentile is a presentation transform and clips to
    0 / 100 outside the reference range, where distinct raw signals would tie. Sorting on
    the raw signal reproduces Model B's ranking exactly."""
    by = list(by)
    key = df.assign(_unscored=df["risk_tier"].eq(UNSCORED),
                    _raw=pd.to_numeric(df["model_b_hist_mean_abs"], errors="coerce"))
    return (key.sort_values(by + ["_unscored", "_raw", "stock"],
                            ascending=[True] * len(by) + [True, False, True],
                            na_position="last", kind="mergesort")
               .drop(columns=["_unscored", "_raw"]))


def engineer_risk_score(events: pd.DataFrame, reference: Reference | None = None) -> pd.DataFrame:
    """`risk_score`: the Breakwater Risk Percentile of `model_b_hist_mean_abs` on the frozen
    reference scale; NaN (Unscored) with fewer than MIN_PRIOR usable priors."""
    reference = reference or load_reference()
    scored = (events["model_b_n_prior"].ge(MIN_PRIOR)
              & events["model_b_hist_mean_abs"].notna()).to_numpy()
    pct = to_percentile(events["model_b_hist_mean_abs"].to_numpy(float), reference)
    events["risk_score"] = np.where(scored, pct, np.nan)
    return events


def engineer_risk_tier(events: pd.DataFrame) -> pd.DataFrame:
    """`risk_tier` from `risk_score`: High Alert >= 90, Elevated 80-90, Normal, Unscored."""
    events["risk_tier"] = assign_tier(events["risk_score"].to_numpy()).to_numpy()
    return events


def engineer_model_version(events: pd.DataFrame) -> pd.DataFrame:
    from config import MODEL_VERSION
    events["model_version"] = MODEL_VERSION
    return events


def print_call_summary(events: pd.DataFrame) -> None:
    """One line per run: how the pending calls split across tiers, and who is Unscored."""
    pend = events[events["is_pending"]]
    counts = pend["risk_tier"].value_counts()
    print(f"  pending calls: {len(pend)} — "
          + ", ".join(f"{t} {int(counts.get(t, 0))}" for t in TIERS))
    unscored = sorted(pend.loc[pend["risk_tier"] == UNSCORED, "stock"])
    if unscored:
        print(f"  Unscored (fewer than {MIN_PRIOR} usable prior corrected reactions): "
              f"{', '.join(unscored)}")


def score_events(events: pd.DataFrame, daily_df: pd.DataFrame,
                 reference: Reference | None = None) -> pd.DataFrame:
    """Model B on a copy of `events` — the same steps stage 4 runs, for callers outside
    the pipeline (the reference builder, the validation, tests)."""
    out = events.drop(columns=[c for c in MODEL_B_COLS if c in events.columns]).copy()
    out = engineer_score_asof_date(out, daily_df)
    out = engineer_prior_history(out, daily_df)
    out = engineer_risk_score(out, reference)
    out = engineer_risk_tier(out)
    out = engineer_model_version(out)
    return out
