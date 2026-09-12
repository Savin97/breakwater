"""Phase 3: rebuild Breakwater's earnings target on verified announcement timing.

Research-only. Nothing in this module changes production scoring, thresholds, ingestion,
or the production database. It answers the narrow first Phase-3 question:

    If model 0.3.1 is evaluated on correctly anchored earnings reactions, and its own
    historical reaction inputs are rebuilt from those corrected reactions, does the
    structural signal survive?

The Benzinga/Massive snapshot remains under gitignored ``vendor/``. This module reads it
as research input, matches it to Breakwater events without using prices/returns to infer
timing, computes timing-aware outcomes in parallel, and compares three variants on the
same rows:

    A. legacy predictor  -> legacy target
    B. legacy predictor  -> corrected target
    C. corrected-history predictor -> corrected target

A vs B isolates label/target damage. B vs C isolates the effect of rebuilding the model's
historical reaction statistics on the corrected target.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import (
    BUCKET_ELEVATED_FLOOR,
    BUCKET_HIGH_ALERT_FLOOR,
    EXTREME_EARNINGS_REACTION_THRESHOLD,
    LIFT_PRIOR_STRENGTH,
    LIFT_TO_ELEVATED,
    LIFT_TO_HIGH_ALERT,
    MODEL_VERSION,
)
from feature_engineering.announcement_timing import (
    AMC,
    BMO,
    ANCHORED_OUTCOME_COLS,
    ANCHORED_STATUS_COLS,
    TARGET_AVAILABLE,
    market_session_grid,
    resolve_event_anchors,
)
from pipeline.events import build_event_frame, score_event_frame
from research.massive import paths as massive_paths
from research.massive.normalize import normalize_snapshot

FULL_DF_PATH = Path("output/full_df.parquet")
RESULTS_DIR = Path("output/phase3_target_rebuild")
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2025-12-31"

PHASE3_PREFIX = "phase3_"
CORRECTED_TARGET = "abs_reaction_3d_anchored"
CORRECTED_REACTION = "reaction_3d_anchored"
IDENTITY_GAP_DAYS = 400


def latest_snapshot(root: Path = massive_paths.EARNINGS_SNAPSHOT_ROOT) -> Path:
    """Newest completed immutable snapshot. ``.partial`` directories are never inputs."""
    candidates = sorted(
        p for p in Path(root).glob("earnings_*")
        if p.is_dir() and not p.name.endswith(".partial") and (p / "manifest.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"no completed Benzinga snapshot under {root}; run research.massive.acquire first"
        )
    return candidates[-1]


def _snapshot_observed_at(snapshot_name: str | None) -> pd.Timestamp:
    """Actual observation time encoded in ``earnings_YYYYMMDDTHHMMSSZ`` when available."""
    if not snapshot_name:
        return pd.NaT
    m = re.search(r"earnings_(\d{8}T\d{6}Z)", snapshot_name)
    if not m:
        return pd.NaT
    ts = pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%SZ", utc=True, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    return ts.tz_convert("America/New_York").tz_localize(None)


def _symbol_key(value: object) -> str:
    """Punctuation-insensitive class-share key, used only as a spelling fallback."""
    return str(value).strip().upper().replace("-", "").replace(".", "").replace(" ", "")


def event_keys_from_daily(daily: pd.DataFrame) -> pd.DataFrame:
    """Completed Breakwater earnings-event identities, one row per stock/date."""
    keys = daily.loc[daily["is_earnings_day"].eq(1), ["stock", "earnings_date"]].copy()
    keys = keys.dropna(subset=["stock", "earnings_date"])
    keys["stock"] = keys["stock"].astype(str).str.strip().str.upper()
    keys["earnings_date"] = pd.to_datetime(keys["earnings_date"]).dt.normalize()
    return (
        keys.drop_duplicates()
        .sort_values(["stock", "earnings_date"], kind="mergesort")
        .reset_index(drop=True)
    )


def _map_vendor_tickers(vendor: pd.DataFrame, event_stocks: pd.Index) -> pd.Series:
    """Exact ticker first; punctuation alias only if it maps to one BW ticker."""
    stocks = pd.Index(
        pd.Series(event_stocks, dtype="string").dropna().astype(str).str.upper().unique()
    )
    exact = set(stocks)
    key_to_stocks: dict[str, list[str]] = {}
    for stock in stocks:
        key_to_stocks.setdefault(_symbol_key(stock), []).append(stock)

    mapped: list[object] = []
    for raw in vendor["ticker_norm"].astype("string"):
        if pd.isna(raw):
            mapped.append(pd.NA)
            continue
        ticker = str(raw).strip().upper()
        if ticker in exact:
            mapped.append(ticker)
            continue
        choices = key_to_stocks.get(_symbol_key(ticker), [])
        mapped.append(choices[0] if len(choices) == 1 else pd.NA)
    return pd.Series(mapped, index=vendor.index, dtype="string")


def identity_hazards(
    vendor: pd.DataFrame,
    event_stocks: pd.Index,
    *,
    min_gap_days: int = IDENTITY_GAP_DAYS,
) -> pd.DataFrame:
    """Conservative ticker histories that Phase 3 must not join automatically.

    The source audit proved that today's ticker string is not a point-in-time identifier.
    We therefore *exclude* rather than repair a current-universe ticker when either:

    * the vendor associates it with multiple normalized company names; or
    * its earnings history has a >= ``min_gap_days`` break.

    Company names are not trusted to resolve identity; they are used only as a hazard
    signal. A future point-in-time identity layer can recover these rows deliberately.
    """
    v = vendor.copy()
    v["mapped_stock"] = _map_vendor_tickers(v, event_stocks)
    v = v.dropna(subset=["mapped_stock", "report_date"]).copy()
    v["report_date"] = pd.to_datetime(v["report_date"]).dt.normalize()

    rows: list[dict] = []
    for stock, sub in v.groupby("mapped_stock", sort=False):
        names = sub.get("company_name_norm", pd.Series(dtype="string")).dropna().astype(str)
        n_names = int(names.nunique())
        dates = pd.Series(pd.unique(sub["report_date"].dropna())).sort_values()
        gaps = dates.diff().dt.days.dropna()
        max_gap = int(gaps.max()) if len(gaps) else 0
        multi_name = n_names > 1
        history_gap = max_gap >= min_gap_days
        if not (multi_name or history_gap):
            continue
        reasons = []
        if multi_name:
            reasons.append("multiple_company_names")
        if history_gap:
            reasons.append(f"history_gap_ge_{min_gap_days}d")
        rows.append({
            "stock": str(stock),
            "reason": "+".join(reasons),
            "distinct_company_names": n_names,
            "max_history_gap_days": max_gap,
            "vendor_records": len(sub),
        })
    return pd.DataFrame(rows).sort_values("stock").reset_index(drop=True) if rows else pd.DataFrame(
        columns=["stock", "reason", "distinct_company_names", "max_history_gap_days", "vendor_records"]
    )


def _pick_unambiguous(candidates: pd.DataFrame) -> tuple[pd.Series | None, str]:
    """Pick a row only when all equally-close candidates imply the same timestamp."""
    if candidates.empty:
        return None, "no_candidate"
    timestamps = pd.to_datetime(candidates["announce_ts_vendor"], errors="coerce").dropna().unique()
    if len(timestamps) != 1:
        return None, "ambiguous_timestamp"
    ranked = candidates.sort_values(
        ["last_updated_utc", "benzinga_id"],
        ascending=[False, True],
        na_position="last",
        kind="mergesort",
    )
    return ranked.iloc[0], "matched"


def match_benzinga_timing(
    vendor: pd.DataFrame,
    event_keys: pd.DataFrame,
    *,
    tolerance_days: int = 1,
    confirmed_only: bool = True,
    snapshot_id: str | None = None,
    excluded_stocks: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match vendor clocks to Breakwater events without using prices or returns.

    Exact date wins. With no exact date, a unique nearest record within ``tolerance_days``
    may match. The vendor timestamp's OWN calendar date survives the match and later drives
    the event clock. Conflicting equally-close timestamps remain unresolved.
    """
    excluded = {str(x).upper() for x in (excluded_stocks or set())}
    v = vendor[vendor["time_usable"].fillna(False)].copy()
    if confirmed_only:
        v = v[v["is_confirmed"].fillna(False)].copy()
    v["mapped_stock"] = _map_vendor_tickers(v, pd.Index(event_keys["stock"].unique()))
    v = v.dropna(subset=["mapped_stock", "report_date", "announce_ts_vendor"]).copy()
    v["report_date"] = pd.to_datetime(v["report_date"]).dt.normalize()

    by_stock = {
        str(stock): sub.sort_values("report_date", kind="mergesort")
        for stock, sub in v.groupby("mapped_stock", sort=False)
    }
    observed_at = _snapshot_observed_at(snapshot_id)
    timing_rows: list[dict] = []
    audit_rows: list[dict] = []

    for ev in event_keys.itertuples(index=False):
        stock = str(ev.stock).upper()
        earnings_date = pd.Timestamp(ev.earnings_date).normalize()
        if stock in excluded:
            audit_rows.append({
                "stock": stock,
                "earnings_date": earnings_date,
                "match_status": "identity_hazard",
                "date_match": "none",
                "date_delta_days": np.nan,
            })
            continue

        sub = by_stock.get(stock)
        match = None
        status = "no_record"
        date_match = "none"
        delta_days = np.nan

        if sub is not None and not sub.empty:
            exact = sub[sub["report_date"].eq(earnings_date)]
            if not exact.empty:
                match, status = _pick_unambiguous(exact)
                date_match = "exact"
                delta_days = 0
            elif tolerance_days > 0:
                distance = (sub["report_date"] - earnings_date).dt.days.abs()
                near = sub[distance.le(tolerance_days)].copy()
                if not near.empty:
                    min_dist = int((near["report_date"] - earnings_date).dt.days.abs().min())
                    nearest = near[(near["report_date"] - earnings_date).dt.days.abs().eq(min_dist)]
                    match, status = _pick_unambiguous(nearest)
                    date_match = f"within_{min_dist}d"
                    if match is not None:
                        delta_days = int((pd.Timestamp(match["report_date"]) - earnings_date).days)

        base = {
            "stock": stock,
            "earnings_date": earnings_date,
            "match_status": status,
            "date_match": date_match,
            "date_delta_days": delta_days,
        }
        if match is None:
            audit_rows.append(base)
            continue

        source = f"massive_benzinga:{snapshot_id or 'snapshot'}:{match['benzinga_id']}"
        timing_rows.append({
            "stock": stock,
            "earnings_date": earnings_date,
            "announce_ts_ny": pd.Timestamp(match["announce_ts_vendor"]),
            "announce_ts_source": source,
            "announce_ts_observed_at": observed_at,
        })
        audit_rows.append({
            **base,
            "vendor_ticker": match["ticker_norm"],
            "vendor_report_date": pd.Timestamp(match["report_date"]),
            "announce_ts_ny": pd.Timestamp(match["announce_ts_vendor"]),
            "announce_window": match["announce_window"],
            "benzinga_id": match["benzinga_id"],
            "date_status": match["date_status_norm"],
        })

    timing = pd.DataFrame(timing_rows)
    audit = pd.DataFrame(audit_rows)
    if not timing.empty:
        timing = timing.drop_duplicates(["stock", "earnings_date"], keep="first")
    return timing, audit


def _phase3_proxy_session_dates(events: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """Translate each observed timestamp into Phase-2's market-session date convention.

    The economically meaningful definition is **last market close before announcement**.
    A real timestamp can be on a weekend/holiday, while Phase 2's resolver expects its D
    to be a market session. A proxy D derived only from the timestamp and market-session
    grid makes the existing resolver implement the correct clock:

      AMC on session    -> D = that session
      AMC off-session   -> D = prior session
      BMO on session    -> D = that session
      BMO off-session   -> D = next session

    In all four cases the resolver's anchor is the last market close before the timestamp.
    INTRADAY/UNKNOWN remain unresolved; nothing is inferred from price behaviour.
    """
    grid = market_session_grid(daily)
    result = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns]")
    if len(grid) == 0:
        return result

    ts = pd.to_datetime(events["announce_ts_ny"], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    dates = ts.dt.normalize().to_numpy(dtype="datetime64[ns]")
    windows = events["announce_window"].to_numpy(dtype=object)

    finite = ~pd.isna(dates)
    left = np.searchsorted(grid, dates, side="left")
    exact = finite & (left < len(grid)) & (grid[np.minimum(left, len(grid) - 1)] == dates)
    proxy = np.full(len(events), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

    amc = finite & (windows == AMC)
    amc_idx = np.where(exact, left, left - 1)
    amc_ok = amc & (amc_idx >= 0) & (amc_idx < len(grid))
    proxy[amc_ok] = grid[amc_idx[amc_ok]]

    bmo = finite & (windows == BMO)
    bmo_idx = left  # exact session, or the next session after an off-session timestamp
    bmo_ok = bmo & (bmo_idx >= 0) & (bmo_idx < len(grid))
    proxy[bmo_ok] = grid[bmo_idx[bmo_ok]]

    result.iloc[:] = proxy
    return result


def reanchor_to_observed_timestamp(events: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Overwrite Phase-2 parallel outcomes using the vendor timestamp's actual event clock."""
    out = events.copy()
    ts = pd.to_datetime(out["announce_ts_ny"], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
    out["phase3_announce_date"] = ts.dt.normalize()
    out["phase3_proxy_session_date"] = _phase3_proxy_session_dates(out, daily)

    resolver_input = out.copy()
    has_timestamp = out["phase3_announce_date"].notna()
    resolver_input.loc[has_timestamp, "earnings_date"] = out.loc[
        has_timestamp, "phase3_proxy_session_date"
    ]
    resolved = resolve_event_anchors(resolver_input, daily)

    replace_cols = [
        "anchor_date",
        "anchor_status",
        "anchor_session_status",
        *ANCHORED_OUTCOME_COLS,
        *ANCHORED_STATUS_COLS,
    ]
    for col in replace_cols:
        out[col] = resolved[col]
    return out


def build_phase3_events(daily: pd.DataFrame, timing: pd.DataFrame) -> pd.DataFrame:
    """Legacy-scored event frame with Benzinga timing-aware outcomes alongside it."""
    events = build_event_frame(daily, timing)
    events = reanchor_to_observed_timestamp(events, daily)
    return score_event_frame(events)


def _entropy(values: list[float], bins: int = 8) -> float:
    if len(values) < bins:
        return np.nan
    hist, _ = np.histogram(np.asarray(values, dtype=float), bins=bins)
    probs = hist / hist.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def _target_history_features(events: pd.DataFrame) -> pd.DataFrame:
    """Rebuild 0.3.1 target-derived state from prior resolved corrected outcomes.

    The same estimators are retained. Missing timing is missing information, not a zero
    return and not a failed trial, so unresolved events do not enter the observation
    history. The model's main history uses corrected 3-session reactions; entropy keeps
    0.3.1's 3-session-with-1-session-fallback semantics.
    """
    out = pd.DataFrame(index=events.index)
    cols = [
        "reaction_std",
        "reaction_entropy",
        "directional_bias",
        "abs_reaction_median",
        "abs_reaction_p75",
        "abs_reaction_p75_rolling",
        "abs_reaction_p90_rolling",
        "n_prior_resolved_reactions",
        "n_prior_entropy_reactions",
    ]
    for col in cols:
        out[PHASE3_PREFIX + col] = np.nan

    ordered = events.sort_values(
        ["stock", "earnings_date", "is_pending"], kind="mergesort"
    )
    for _stock, sub in ordered.groupby("stock", sort=False):
        signed3: list[float] = []
        absolute3: list[float] = []
        entropy_abs: list[float] = []

        for idx in sub.index:
            n = len(signed3)
            out.at[idx, PHASE3_PREFIX + "n_prior_resolved_reactions"] = n
            out.at[idx, PHASE3_PREFIX + "n_prior_entropy_reactions"] = len(entropy_abs)

            if n >= 3:
                out.at[idx, PHASE3_PREFIX + "reaction_std"] = float(
                    np.std(absolute3[-8:], ddof=1)
                )
            if len(entropy_abs) >= 8:
                out.at[idx, PHASE3_PREFIX + "reaction_entropy"] = _entropy(entropy_abs)
            if n:
                out.at[idx, PHASE3_PREFIX + "directional_bias"] = float(np.mean(signed3))
                out.at[idx, PHASE3_PREFIX + "abs_reaction_median"] = float(np.median(absolute3))
                out.at[idx, PHASE3_PREFIX + "abs_reaction_p75"] = float(
                    np.quantile(absolute3, 0.75)
                )
            if n >= 28:
                recent = np.asarray(absolute3[-28:], dtype=float)
                out.at[idx, PHASE3_PREFIX + "abs_reaction_p75_rolling"] = float(
                    np.quantile(recent, 0.75)
                )
                out.at[idx, PHASE3_PREFIX + "abs_reaction_p90_rolling"] = float(
                    np.quantile(recent, 0.90)
                )

            r3 = events.at[idx, CORRECTED_REACTION]
            r1 = events.at[idx, "reaction_1d_anchored"]
            best = r3 if pd.notna(r3) else r1
            if pd.notna(r3):
                r3 = float(r3)
                signed3.append(r3)
                absolute3.append(abs(r3))
            if pd.notna(best):
                entropy_abs.append(abs(float(best)))
    return out


def _missing_aware_lift(events: pd.DataFrame, bucket_col: str, extreme_col: str) -> pd.Series:
    """0.3.1 stock/bucket lift, excluding unresolved outcomes from sum and count."""
    sort_date = events["phase3_announce_date"].fillna(events["earnings_date"])
    ev = events.assign(_sort_date=sort_date).sort_values(
        ["_sort_date", "is_pending"], kind="mergesort"
    )
    y = pd.to_numeric(ev[extreme_col], errors="coerce")
    global_prior = y.expanding().mean().shift(1)

    valid = y.notna().astype(float)
    y0 = y.fillna(0.0)
    keys = [ev["stock"], ev[bucket_col].astype(object)]
    n_prior = valid.groupby(keys, sort=False, dropna=False).cumsum() - valid
    sum_prior = y0.groupby(keys, sort=False, dropna=False).cumsum() - y0

    shrunk = (
        sum_prior + LIFT_PRIOR_STRENGTH * global_prior
    ) / (n_prior + LIFT_PRIOR_STRENGTH)
    lift = (shrunk / global_prior).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return lift.reindex(events.index)


def score_corrected_shadow(events: pd.DataFrame) -> pd.DataFrame:
    """Parallel structural score with 0.3.1 weights/cuts held fixed.

    Deliberately preserved: 85/15 weights, 12% p75 ceiling, 73/79 cuts, lift prior 20,
    1.5x/3x promotions, and the known cross-stock entropy ffill quirk. High Conviction is
    intentionally NOT re-evaluated here: historical BMO drift needs its own pre-event
    anchoring before it can be treated as leakage-free.
    """
    out = pd.concat([events.copy(), _target_history_features(events)], axis=1)

    p75 = out[PHASE3_PREFIX + "abs_reaction_p75_rolling"].fillna(
        out[PHASE3_PREFIX + "abs_reaction_p75"]
    )
    e3 = (p75 / 0.12).clip(0, 1)

    entropy = out[PHASE3_PREFIX + "reaction_entropy"]
    chain = entropy.mask(out["is_pending"]).ffill()  # preserve known 0.3.1 quirk
    e4 = np.clip(entropy.fillna(chain).fillna(0), 0, 1)
    score = 100 * np.clip(0.85 * e3 + 0.15 * e4, 0, 1)
    out[PHASE3_PREFIX + "risk_score"] = score

    structural = pd.cut(
        score,
        bins=[-np.inf, BUCKET_ELEVATED_FLOOR, BUCKET_HIGH_ALERT_FLOOR, np.inf],
        labels=["Normal", "Elevated", "High Alert"],
    )
    out[PHASE3_PREFIX + "bucket_structural"] = structural

    target = out[CORRECTED_TARGET]
    extreme = pd.Series(np.nan, index=out.index, dtype=float)
    extreme.loc[target.notna()] = target.loc[target.notna()].ge(
        EXTREME_EARNINGS_REACTION_THRESHOLD
    ).astype(float)
    extreme.loc[out["is_pending"]] = np.nan
    out[PHASE3_PREFIX + "is_extreme_reaction"] = extreme

    lift = _missing_aware_lift(
        out,
        PHASE3_PREFIX + "bucket_structural",
        PHASE3_PREFIX + "is_extreme_reaction",
    )
    out[PHASE3_PREFIX + "stock_bucket_lift"] = lift

    adjusted = structural.astype(object)
    to_high = adjusted.isin(["Normal", "Elevated"]) & lift.ge(LIFT_TO_HIGH_ALERT)
    to_elevated = adjusted.eq("Normal") & lift.ge(LIFT_TO_ELEVATED) & ~to_high
    adjusted.loc[to_elevated] = "Elevated"
    adjusted.loc[to_high] = "High Alert"
    out[PHASE3_PREFIX + "bucket"] = pd.Categorical(
        adjusted,
        categories=["Normal", "Elevated", "High Alert"],
        ordered=True,
    )
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return centre - half, centre + half


def variant_metrics(
    sample: pd.DataFrame,
    *,
    name: str,
    cohort: str,
    target_col: str,
    score_col: str,
    bucket_col: str,
) -> dict:
    d = sample.dropna(subset=[target_col, score_col, bucket_col]).copy()
    y = d[target_col].ge(EXTREME_EARNINGS_REACTION_THRESHOLD)
    baseline = float(y.mean()) if len(d) else np.nan
    bucket_text = d[bucket_col].astype(str)
    selected = bucket_text.isin(["Elevated", "High Alert"])
    total_hits = int(y.sum())
    captured = int((y & selected).sum())
    tier_score = bucket_text.map({"Normal": 0, "Elevated": 1, "High Alert": 2})

    result = {
        "variant": name,
        "cohort": cohort,
        "n": len(d),
        "baseline_p_ge_8": baseline,
        "selected_share": float(selected.mean()) if len(d) else np.nan,
        "high_alert_share": float(bucket_text.eq("High Alert").mean()) if len(d) else np.nan,
        "capture_ge_8": captured / total_hits if total_hits else np.nan,
        "score_auc_ge_8": roc_auc_score(y.astype(int), d[score_col]) if y.nunique() == 2 else np.nan,
        "tier_auc_ge_8": roc_auc_score(y.astype(int), tier_score) if y.nunique() == 2 else np.nan,
    }
    for tier in ("Normal", "Elevated", "High Alert"):
        sub = d[bucket_text.eq(tier)]
        hits = int(sub[target_col].ge(EXTREME_EARNINGS_REACTION_THRESHOLD).sum())
        n = len(sub)
        lo, hi = _wilson(hits, n)
        result[f"{tier}_n"] = n
        result[f"{tier}_p_ge_8"] = hits / n if n else np.nan
        result[f"{tier}_ci_lo"] = lo
        result[f"{tier}_ci_hi"] = hi
        result[f"{tier}_lift"] = (hits / n) / baseline if n and baseline else np.nan
    return result


def target_shift_metrics(sample: pd.DataFrame) -> dict:
    """How much the label itself moves when timing is corrected."""
    d = sample.dropna(subset=["abs_reaction_3d", CORRECTED_TARGET]).copy()
    old = d["abs_reaction_3d"]
    new = d[CORRECTED_TARGET]
    old8 = old.ge(0.08)
    new8 = new.ge(0.08)
    result = {
        "n_common_targets": len(d),
        "median_abs_target_change": float((new - old).abs().median()) if len(d) else np.nan,
        "mean_abs_target_change": float((new - old).abs().mean()) if len(d) else np.nan,
        "extreme_label_flip_share": float(old8.ne(new8).mean()) if len(d) else np.nan,
        "legacy_p_ge_8": float(old8.mean()) if len(d) else np.nan,
        "corrected_p_ge_8": float(new8.mean()) if len(d) else np.nan,
    }
    for window in (BMO, AMC):
        sub = d[d["announce_window"].eq(window)]
        result[f"{window.lower()}_n"] = len(sub)
        result[f"{window.lower()}_median_abs_change"] = float(
            (sub[CORRECTED_TARGET] - sub["abs_reaction_3d"]).abs().median()
        ) if len(sub) else np.nan
        result[f"{window.lower()}_label_flip_share"] = float(
            sub["abs_reaction_3d"].ge(0.08).ne(sub[CORRECTED_TARGET].ge(0.08)).mean()
        ) if len(sub) else np.nan
    return result


def maturity_by_year(events: pd.DataFrame) -> pd.DataFrame:
    """Coverage/maturity census so early-year results cannot hide thin history."""
    d = events[~events["is_pending"]].copy()
    date = d["phase3_announce_date"].fillna(d["earnings_date"])
    d["year"] = pd.to_datetime(date).dt.year
    d["target_available"] = (
        d[CORRECTED_TARGET].notna()
        & d["reaction_3d_anchored_status"].eq(TARGET_AVAILABLE)
    )
    d["mature_8"] = d[PHASE3_PREFIX + "n_prior_resolved_reactions"].ge(8)
    d["mature_28"] = d[PHASE3_PREFIX + "n_prior_resolved_reactions"].ge(28)
    rows = []
    for year, sub in d.dropna(subset=["year"]).groupby("year"):
        rows.append({
            "year": int(year),
            "events": len(sub),
            "target_available_share": float(sub["target_available"].mean()),
            "mature_8_share": float(sub["mature_8"].mean()),
            "mature_28_share": float(sub["mature_28"].mean()),
        })
    return pd.DataFrame(rows)


def evaluate(
    events: pd.DataFrame,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """A/B/C comparison on an identical event sample, plus a mature-28 cohort."""
    completed = ~events["is_pending"]
    target_ok = (
        events[CORRECTED_TARGET].notna()
        & events["reaction_3d_anchored_status"].eq(TARGET_AVAILABLE)
    )
    dates = pd.to_datetime(events["phase3_announce_date"].fillna(events["earnings_date"]))
    in_window = dates.between(pd.Timestamp(start), pd.Timestamp(end))
    target_common = events[completed & target_ok & in_window].copy()

    comparison = target_common.dropna(subset=[
        "abs_reaction_3d",
        "risk_score",
        "earnings_explosiveness_bucket",
        PHASE3_PREFIX + "risk_score",
        PHASE3_PREFIX + "bucket",
    ]).copy()

    cohorts = {
        "all_resolved": comparison,
        "mature_28": comparison[
            comparison[PHASE3_PREFIX + "n_prior_resolved_reactions"].ge(28)
        ],
    }
    variants: list[dict] = []
    for cohort, sample in cohorts.items():
        variants.extend([
            variant_metrics(
                sample,
                name="legacy_predictor__legacy_target",
                cohort=cohort,
                target_col="abs_reaction_3d",
                score_col="risk_score",
                bucket_col="earnings_explosiveness_bucket",
            ),
            variant_metrics(
                sample,
                name="legacy_predictor__corrected_target",
                cohort=cohort,
                target_col=CORRECTED_TARGET,
                score_col="risk_score",
                bucket_col="earnings_explosiveness_bucket",
            ),
            variant_metrics(
                sample,
                name="corrected_history_predictor__corrected_target",
                cohort=cohort,
                target_col=CORRECTED_TARGET,
                score_col=PHASE3_PREFIX + "risk_score",
                bucket_col=PHASE3_PREFIX + "bucket",
            ),
        ])

    shift = target_shift_metrics(target_common)
    shift["n_model_comparison_common"] = len(comparison)
    shift["n_model_comparison_mature_28"] = len(cohorts["mature_28"])
    return pd.DataFrame(variants), shift, maturity_by_year(events)


def run(
    *,
    snapshot: Path,
    full_df_path: Path = FULL_DF_PATH,
    results_dir: Path = RESULTS_DIR,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    tolerance_days: int = 1,
    confirmed_only: bool = True,
    exclude_identity_hazards: bool = True,
) -> dict:
    daily = pd.read_parquet(full_df_path)
    vendor = normalize_snapshot(snapshot)
    keys = event_keys_from_daily(daily)

    hazards = identity_hazards(vendor, pd.Index(keys["stock"].unique()))
    excluded = set(hazards["stock"]) if exclude_identity_hazards else set()
    timing, match_audit = match_benzinga_timing(
        vendor,
        keys,
        tolerance_days=tolerance_days,
        confirmed_only=confirmed_only,
        snapshot_id=snapshot.name,
        excluded_stocks=excluded,
    )

    events = build_phase3_events(daily, timing)
    events = score_corrected_shadow(events)
    metrics, target_shift, maturity = evaluate(events, start=start, end=end)

    results_dir.mkdir(parents=True, exist_ok=True)
    hazards.to_csv(results_dir / "identity_hazards.csv", index=False)
    match_audit.to_parquet(results_dir / "benzinga_event_matches.parquet", index=False)
    events.to_parquet(results_dir / "phase3_events.parquet", index=False)
    metrics.to_csv(results_dir / "model_comparison.csv", index=False)
    maturity.to_csv(results_dir / "history_maturity.csv", index=False)

    match_counts = match_audit["match_status"].value_counts(dropna=False).to_dict()
    summary = {
        "phase": "phase3_target_rebuild",
        "model_version_held_fixed": MODEL_VERSION,
        "snapshot": snapshot.name,
        "confirmed_only": confirmed_only,
        "tolerance_days": tolerance_days,
        "exclude_identity_hazards": exclude_identity_hazards,
        "evaluation_start": start,
        "evaluation_end": end,
        "event_keys": len(keys),
        "identity_hazard_tickers": len(hazards),
        "timing_matches": len(timing),
        "match_status_counts": {str(k): int(v) for k, v in match_counts.items()},
        "target_shift": target_shift,
        "model_comparison": metrics.to_dict(orient="records"),
        "guardrails": [
            "unresolved timing is excluded, never treated as a zero/non-extreme event",
            "unsafe ticker identities are excluded by default rather than heuristically repaired",
            "High Conviction is not evaluated because historical BMO drift needs pre-event anchoring",
            "73/79 were selected on overlapping historical data; this is a controlled variant comparison, not pristine absolute OOS certification",
        ],
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )

    print(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="immutable vendor snapshot directory; defaults to newest completed snapshot",
    )
    ap.add_argument("--full-df", type=Path, default=FULL_DF_PATH)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--tolerance-days", type=int, default=1)
    ap.add_argument(
        "--include-projected",
        action="store_true",
        help="sensitivity only: allow usable vendor rows not marked confirmed",
    )
    ap.add_argument(
        "--allow-identity-hazards",
        action="store_true",
        help="sensitivity only: do not exclude ticker histories flagged as identity hazards",
    )
    args = ap.parse_args()

    snapshot = args.snapshot or latest_snapshot()
    run(
        snapshot=snapshot,
        full_df_path=args.full_df,
        results_dir=args.results_dir,
        start=args.start,
        end=args.end,
        tolerance_days=args.tolerance_days,
        confirmed_only=not args.include_projected,
        exclude_identity_hazards=not args.allow_identity_hazards,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
