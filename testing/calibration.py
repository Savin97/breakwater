"""
Historical calibration tables for Model B (0.4.0).

Reads output/events_df.parquet. Unit of analysis: completed earnings events in the
out-of-sample years whose corrected 3-session outcome (`abs_reaction_3d_anchored`) is
available. Events without a verified announcement time have no outcome and are excluded,
never counted as misses. Produces six tables, each followed by its BMO / AMC split where
it is pooled (before-open and after-close reporters differ in base rate and tier mix):

  1. By tier        — event counts + P(<1%), P(≥3/5/8/10/15%) for each Model B tier,
                      with a baseline row, an Unscored row (coverage, not a tier) and
                      95% Wilson CIs on the headline P(≥8%)
  2. Capture rate   — share of all large moves that fell in High Alert / Elevated
  3. Percentile band — P(≥3/8/10%) across fixed Breakwater Risk Percentile bands
  4. Year-by-year   — P(≥8%) for High Alert by calendar year
  5. Tier stability — P(≥8%) for EVERY tier by calendar year
  6. Consistency    — every tier's risk_score and raw-signal range, and any overlap
                      between adjacent tiers (must be zero: the tier is a pure function
                      of the percentile, which is monotonic in the raw signal)

INTERNAL ONLY. These are RETRO-SCORED events: each event's Model B call is computed from
prior outcomes only (the raw signal is causal), but nothing here was published at the
time, and the percentile scale behind the tiers was built from the 2014-2025 events these
tables cover. They describe how the fixed model behaves; they are not a track record and
must never be published as one. No multiple of the base rate ("lift") is printed: compare
tier rates within a window instead (CLAUDE.md, Performance Claims).

Outputs: prints tables to stdout and saves CSVs to testing/testing_results/ (gitignored).

Run with: .venv/bin/python -m testing.calibration
"""
import os
import math
import pandas as pd

from feature_engineering.announcement_timing import TARGET_AVAILABLE
from scoring.model_b import HIGH_ALERT, ELEVATED, NORMAL, UNSCORED

EVENTS_PATH = "output/events_df.parquet"
RESULTS_DIR = "testing/testing_results"
OOS_START   = "2017-01-01"     # the Model B refit's out-of-sample years
OOS_END     = "2025-12-31"
TARGET      = "abs_reaction_3d_anchored"

THRESHOLDS = [0.03, 0.05, 0.08, 0.10, 0.15]
LABELS     = ["≥3%", "≥5%", "≥8%", "≥10%", "≥15%"]

# Tier order, high → low risk. The consistency check relies on this direction:
# everything after a tier in this list is strictly lower risk than it.
TIER_ORDER = [HIGH_ALERT, ELEVATED, NORMAL]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because some slices (a tier inside one
    year or one window) have small n, where the normal interval misbehaves.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half   = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (centre - half, centre + half)


def load_oos_events(path: str = EVENTS_PATH) -> pd.DataFrame:
    df = pd.read_parquet(path)
    ev = df[
        (~df["is_pending"].astype(bool)) &
        (df["earnings_date"] >= OOS_START) &
        (df["earnings_date"] <= OOS_END) &
        (df["reaction_3d_anchored_status"] == TARGET_AVAILABLE) &
        (df[TARGET].notna())
    ].copy()
    ev["year"] = pd.to_datetime(ev["earnings_date"]).dt.year
    return ev


def hit_rates(sub: pd.DataFrame) -> dict:
    n = len(sub)
    row: dict = {"N Events": n}
    calm = (sub[TARGET] < 0.01).sum()
    row["P(<1%)"] = f"{calm / n * 100:.1f}%" if n > 0 else "—"
    for thresh, label in zip(THRESHOLDS, LABELS):
        hits = (sub[TARGET] >= thresh).sum()
        row[f"P({label})"] = f"{hits / n * 100:.1f}%" if n > 0 else "—"
        row[f"_n_{label}"] = hits

    # 95% CI on the headline metric. Tier n varies by an order of magnitude, so a bare
    # point estimate invites reading noise as signal.
    hits_8 = int((sub[TARGET] >= 0.08).sum())
    lo, hi = wilson(hits_8, n)
    row["P(≥8%) 95% CI"] = f"{lo * 100:.1f}–{hi * 100:.1f}%" if n > 0 else "—"
    row["_ci_lo_8"], row["_ci_hi_8"] = lo, hi
    return row


def table_by_bucket(ev: pd.DataFrame) -> pd.DataFrame:
    # Baseline = every scored event. Unscored is listed as coverage (not enough corrected
    # history to rank), never as a lower tier.
    scored = ev["risk_tier"] != UNSCORED
    tiers = [
        ("ALL scored (baseline)", scored),
        (NORMAL,                  ev["risk_tier"] == NORMAL),
        (ELEVATED,                ev["risk_tier"] == ELEVATED),
        (HIGH_ALERT,              ev["risk_tier"] == HIGH_ALERT),
        ("Unscored (coverage)",   ev["risk_tier"] == UNSCORED),
    ]
    rows = []
    for label, mask in tiers:
        row = {"Bucket": label}
        row.update(hit_rates(ev[mask]))
        rows.append(row)
    return pd.DataFrame(rows)


def table_capture_rate(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev[ev["risk_tier"] != UNSCORED]
    selected = ev["risk_tier"].isin([HIGH_ALERT, ELEVATED])
    total_n  = len(ev)
    sel_n    = selected.sum()
    rows = []
    for thresh, label in zip(THRESHOLDS, LABELS):
        all_large = (ev[TARGET] >= thresh).sum()
        sel_large = (ev[selected][TARGET] >= thresh).sum()
        capture   = sel_large / all_large * 100 if all_large else 0
        fn_rate   = 100 - capture
        rows.append({
            "Move threshold":    label,
            "Total moves":       all_large,
            "Captured (HA+El)":  sel_large,
            "Capture rate":      f"{capture:.1f}%",
            "False-neg rate":    f"{fn_rate:.1f}%",
        })
    header = pd.DataFrame([{
        "Move threshold": f"Universe: {sel_n}/{total_n} scored events selected ({sel_n/total_n*100:.1f}%)",
        "Total moves": "", "Captured (HA+El)": "", "Capture rate": "", "False-neg rate": "",
    }])
    return pd.concat([header, pd.DataFrame(rows)], ignore_index=True)


def table_by_percentile(ev: pd.DataFrame) -> pd.DataFrame:
    # Fixed bands of the Breakwater Risk Percentile. The percentile is a frozen reference
    # scale (scoring/model_b_reference.json), so no ranking happens here and no band
    # depends on another event's data.
    ev = ev[ev["risk_tier"] != UNSCORED]
    bands = [(0, 50), (50, 75), (75, 80), (80, 90), (90, 95), (95, 99), (99, 100)]
    rows = []
    for lo, hi in bands:
        mask = (ev["risk_score"] >= lo) & ((ev["risk_score"] < hi) if hi < 100 else (ev["risk_score"] <= hi))
        sub  = ev[mask]
        n    = len(sub)
        row  = {"Percentile band": f"{lo}–{hi}", "N Events": n}
        calm = (sub[TARGET] < 0.01).sum()
        row["P(<1%)"] = f"{calm / n * 100:.1f}%" if n > 0 else "—"
        for thresh, label in zip([0.03, 0.08, 0.10], ["≥3%", "≥8%", "≥10%"]):
            hits = (sub[TARGET] >= thresh).sum()
            row[f"P({label})"] = f"{hits / n * 100:.1f}%" if n > 0 else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def table_year_by_year(ev: pd.DataFrame) -> pd.DataFrame:
    ha = ev[ev["risk_tier"] == HIGH_ALERT]
    rows = []
    for year in sorted(ha["year"].unique()):
        sub  = ha[ha["year"] == year]
        n    = len(sub)
        hits = (sub[TARGET] >= 0.08).sum()
        rows.append({
            "Year":          year,
            "HA Events":     n,
            "Moves ≥8%":     hits,
            "P(≥8%)":        f"{hits / n * 100:.1f}%" if n > 0 else "—",
        })
    return pd.DataFrame(rows)


def table_tier_stability(ev: pd.DataFrame) -> pd.DataFrame:
    """P(≥8%) for every tier, year by year.

    table_year_by_year covers High Alert only. The tier boundaries are fixed percentile
    bands, so the share of events in each tier and its rate both drift with the market;
    a tier that holds in the pooled average while collapsing in individual years is not
    stable.
    """
    rows = []
    for year in sorted(ev["year"].unique()):
        yr  = ev[ev["year"] == year]
        row = {"Year": year, "N": len(yr)}
        for tier in TIER_ORDER:
            sub = yr[yr["risk_tier"] == tier]
            n   = len(sub)
            hits = int((sub[TARGET] >= 0.08).sum())
            row[f"{tier} n"]      = n
            row[f"{tier} P(≥8%)"] = f"{hits / n * 100:.1f}%" if n > 0 else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def table_consistency(ev: pd.DataFrame) -> pd.DataFrame:
    """Score/tier monotonicity.

    Guards the defect where an event carries a higher tier label than another while
    scoring lower. Reports each tier's risk_score and raw-signal range and counts overlaps
    between adjacent tiers, on both columns. Any non-zero overlap means a ranked list will
    disagree with the tier labels shown next to it.
    """
    rows = []
    for tier in TIER_ORDER:
        sub = ev[ev["risk_tier"] == tier]
        rows.append({
            "Tier":       tier,
            "N":          len(sub),
            "Min score":  round(sub["risk_score"].min(), 2) if len(sub) else "—",
            "Max score":  round(sub["risk_score"].max(), 2) if len(sub) else "—",
            "Min raw":    round(sub["model_b_hist_mean_abs"].min(), 5) if len(sub) else "—",
            "Max raw":    round(sub["model_b_hist_mean_abs"].max(), 5) if len(sub) else "—",
        })

    # Adjacent boundaries, walking high → low risk. Clean tiers never overlap: every
    # lower-tier value should sit below every higher-tier value.
    for hi_tier, lo_tier in zip(TIER_ORDER, TIER_ORDER[1:]):
        for col in ("risk_score", "model_b_hist_mean_abs"):
            hi = ev.loc[ev["risk_tier"] == hi_tier, col].dropna()
            lo = ev.loc[ev["risk_tier"] == lo_tier, col].dropna()
            if not len(hi) or not len(lo):
                continue
            lo_above = int((lo > hi.min()).sum())   # lower-tier events outscoring the higher tier
            hi_below = int((hi < lo.max()).sum())   # higher-tier events undercutting the lower tier
            rows.append({
                "Tier":      f"OVERLAP {lo_tier} / {hi_tier} ({col})",
                "N":         lo_above + hi_below,
                "Min score": f"{lo_above} {lo_tier} > {hi_tier} min",
                "Max score": f"{hi_below} {hi_tier} < {lo_tier} max",
                "Min raw": "", "Max raw": "",
            })
    return pd.DataFrame(rows)


def _by_window(ev, fn):
    """The same table for before-open and after-close reporters separately."""
    parts = []
    for w, g in ev.groupby("announce_window"):
        t = fn(g)
        t.insert(0, "Window", w)
        parts.append(t)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def run(path: str = EVENTS_PATH, results_dir: str = RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    ev = load_oos_events(path)
    n_uns = int((ev["risk_tier"] == UNSCORED).sum())
    print(f"Completed events with a corrected outcome: {len(ev)} ({n_uns} Unscored)  "
          f"({OOS_START} → {OOS_END}). INTERNAL, retro-scored, not a track record.\n")

    def save(t, name):
        t.to_csv(os.path.join(results_dir, f"calibration_{name}.csv"), index=False)

    display_cols = (["Bucket", "N Events", "P(<1%)"] + [f"P({l})" for l in LABELS]
                    + ["P(≥8%) 95% CI"])
    print("=== 1. By Tier ===")
    t1 = table_by_bucket(ev)
    print(t1[display_cols].to_string(index=False))
    t1w = _by_window(ev, table_by_bucket)
    print(t1w[["Window"] + display_cols].to_string(index=False))
    save(t1, "by_bucket"); save(t1w, "by_bucket_by_window")

    print("\n=== 2. Capture Rate (High Alert + Elevated selected) ===")
    t2 = table_capture_rate(ev)
    print(t2.to_string(index=False))
    t2w = _by_window(ev, table_capture_rate)
    print(t2w.to_string(index=False))
    save(t2, "capture_rate"); save(t2w, "capture_rate_by_window")

    print("\n=== 3. By Percentile Band ===")
    t3 = table_by_percentile(ev)
    print(t3.to_string(index=False))
    t3w = _by_window(ev, table_by_percentile)
    print(t3w.to_string(index=False))
    save(t3, "by_percentile"); save(t3w, "by_percentile_by_window")

    print("\n=== 4. Year-by-Year (High Alert, P(≥8%)) ===")
    t4 = table_year_by_year(ev)
    print(t4.to_string(index=False))
    save(t4, "year_by_year")

    print("\n=== 5. Tier Stability by Year (P(≥8%) per tier) ===")
    t5 = table_tier_stability(ev)
    print(t5.to_string(index=False))
    save(t5, "tier_stability")

    print("\n=== 6. Score/Tier Consistency ===")
    t6 = table_consistency(ev)
    print(t6.to_string(index=False))
    save(t6, "consistency")

    print(f"\nCSVs saved to {results_dir}/")
    return {"by_bucket": t1, "capture_rate": t2, "by_percentile": t3,
            "year_by_year": t4, "tier_stability": t5, "consistency": t6}


if __name__ == "__main__":
    run()
