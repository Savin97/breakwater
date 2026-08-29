"""
Historical calibration tables for the earnings_explosiveness model.

Reads output/full_df.parquet and uses earnings-day rows (is_earnings_day == 1)
as the unit of analysis. Produces six tables:

  1. By bucket      — event counts + P(<1%), P(≥3/5/8/10/15%) for each risk tier,
                      with a baseline row and 95% Wilson CIs on the headline P(≥8%)
  2. Capture rate   — share of all large moves that fell in High Alert / Elevated
  3. Percentile band — P(≥8/10%) across six percentile bands
  4. Year-by-year   — P(≥8%) for High Alert by calendar year
  5. Tier stability — P(≥8%) for EVERY tier by calendar year (not just High Alert)
  6. Consistency    — score/bucket monotonicity check (guards the defect where a
                      lower-scoring event carries a higher tier label)

LEAKAGE NOTE: this script computes no features. It reads precomputed columns and
is only as leakage-free as the pipeline that wrote them; the shift(1) guards live
in feature_engineering/ and scoring/, not here. Table 3 is the one place this file
itself introduces mild lookahead — percentile bands are ranked within each year to
limit it, but ranking is still cross-sectional over a full year of events.

IN-SAMPLE CAVEAT: the 73/79 bucket thresholds were themselves selected on this same
2011-2025 window (see scoring/scoring_features.py), so absolute tier rates here are
optimistically biased. Use this script to COMPARE two model variants on equal terms,
not to certify an absolute hit rate.

Outputs: prints tables to stdout and saves CSVs to testing/testing_results/.

Run with: .venv/bin/python -m testing.calibration
"""
import os
import math
import pandas as pd

PARQUET_PATH = "output/full_df.parquet"
RESULTS_DIR  = "testing/testing_results"
OOS_START    = "2015-01-01"
OOS_END      = "2025-12-31"

THRESHOLDS = [0.03, 0.05, 0.08, 0.10, 0.15]
LABELS     = ["≥3%", "≥5%", "≥8%", "≥10%", "≥15%"]

# Tier order, high → low risk — matches report/calendar_builder.py and
# analysis/results_check.py. The consistency check below relies on this direction:
# everything after a tier in this list is strictly lower risk than it.
TIER_ORDER = ["High Alert", "Elevated", "Normal"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because several tiers have small n
    (Elevated is ~600 events, HC ~170) where the normal interval misbehaves.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half   = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (centre - half, centre + half)


def load_oos_events() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    ev = df[
        (df["is_earnings_day"] == 1) &
        (df["date"] >= OOS_START) &
        (df["date"] <= OOS_END) &
        (df["abs_reaction_3d"].notna()) &
        (df["earnings_explosiveness_bucket"].notna())
    ].copy()
    ev["year"] = ev["date"].dt.year
    ev["is_hc"] = (
        (ev["earnings_explosiveness_bucket"] == "High Alert") &
        (ev["pre_earnings_drift_flag"].notna()) &
        (ev["pre_earnings_drift_flag"].str.strip() != "")
    )
    return ev


def hit_rates(sub: pd.DataFrame, baseline_8pct: float | None = None) -> dict:
    n = len(sub)
    row: dict = {"N Events": n}
    calm = (sub["abs_reaction_3d"] < 0.01).sum()
    row["P(<1%)"] = f"{calm / n * 100:.1f}%" if n > 0 else "—"
    for thresh, label in zip(THRESHOLDS, LABELS):
        hits = (sub["abs_reaction_3d"] >= thresh).sum()
        row[f"P({label})"] = f"{hits / n * 100:.1f}%" if n > 0 else "—"
        row[f"_n_{label}"] = hits

    # 95% CI on the headline metric. Tier n varies by two orders of magnitude
    # (Normal ~18k vs HC ~170), so a bare point estimate invites reading noise as
    # signal when comparing two model variants.
    hits_8 = int((sub["abs_reaction_3d"] >= 0.08).sum())
    lo, hi = wilson(hits_8, n)
    row["P(≥8%) 95% CI"] = f"{lo * 100:.1f}–{hi * 100:.1f}%" if n > 0 else "—"
    row["_ci_lo_8"], row["_ci_hi_8"] = lo, hi
    if baseline_8pct:
        row["Lift ≥8%"] = f"{(hits_8 / n) / baseline_8pct:.2f}x" if n > 0 else "—"
    return row


def table_by_bucket(ev: pd.DataFrame) -> pd.DataFrame:
    # Baseline = every event in the universe. Without it the tier rates below have
    # nothing to be judged against.
    baseline_8 = (ev["abs_reaction_3d"] >= 0.08).mean()

    tiers = [
        ("ALL (baseline)",  pd.Series(True, index=ev.index)),
        ("Normal",          ev["earnings_explosiveness_bucket"] == "Normal"),
        ("Elevated",        ev["earnings_explosiveness_bucket"] == "Elevated"),
        ("High Alert",      ev["earnings_explosiveness_bucket"] == "High Alert"),
        ("HC (HA + Drift)", ev["is_hc"]),
    ]
    rows = []
    for label, mask in tiers:
        row = {"Bucket": label}
        row.update(hit_rates(ev[mask], baseline_8pct=baseline_8))
        rows.append(row)
    return pd.DataFrame(rows)


def table_capture_rate(ev: pd.DataFrame) -> pd.DataFrame:
    selected = ev["earnings_explosiveness_bucket"].isin(["High Alert", "Elevated"])
    total_n  = len(ev)
    sel_n    = selected.sum()
    rows = []
    for thresh, label in zip(THRESHOLDS, LABELS):
        all_large = (ev["abs_reaction_3d"] >= thresh).sum()
        sel_large = (ev[selected]["abs_reaction_3d"] >= thresh).sum()
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
        "Move threshold": f"Universe: {sel_n}/{total_n} events selected ({sel_n/total_n*100:.1f}%)",
        "Total moves": "", "Captured (HA+El)": "", "Capture rate": "", "False-neg rate": "",
    }])
    return pd.concat([header, pd.DataFrame(rows)], ignore_index=True)


def table_by_percentile(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    rank_key = ev["abs_reaction_p75_rolling"].fillna(ev["abs_reaction_p75"])
    # Rank WITHIN year, not across the whole window. Ranking over 2015-2025 at once
    # would let a 2015 event's band depend on 2024 data; within-year still looks
    # across a year of events, so this table stays diagnostic, not a clean OOS claim.
    ev["pct"] = rank_key.groupby(ev["year"]).rank(pct=True) * 100
    bands = [(0, 50), (50, 75), (75, 90), (90, 95), (95, 99), (99, 100)]
    rows = []
    for lo, hi in bands:
        mask = (ev["pct"] >= lo) & (ev["pct"] <= hi)
        sub  = ev[mask]
        n    = len(sub)
        row  = {"Percentile band": f"{lo}–{hi}th", "N Events": n}
        calm = (sub["abs_reaction_3d"] < 0.01).sum()
        row["P(<1%)"] = f"{calm / n * 100:.1f}%" if n > 0 else "—"
        for thresh, label in zip([0.03, 0.08, 0.10], ["≥3%", "≥8%", "≥10%"]):
            hits = (sub["abs_reaction_3d"] >= thresh).sum()
            row[f"P({label})"] = f"{hits / n * 100:.1f}%" if n > 0 else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def table_year_by_year(ev: pd.DataFrame) -> pd.DataFrame:
    ha = ev[ev["earnings_explosiveness_bucket"] == "High Alert"]
    rows = []
    for year in sorted(ha["year"].unique()):
        sub  = ha[ha["year"] == year]
        n    = len(sub)
        hits = (sub["abs_reaction_3d"] >= 0.08).sum()
        rows.append({
            "Year":          year,
            "HA Events":     n,
            "Moves ≥8%":     hits,
            "P(≥8%)":        f"{hits / n * 100:.1f}%" if n > 0 else "—",
        })
    return pd.DataFrame(rows)


def table_tier_stability(ev: pd.DataFrame) -> pd.DataFrame:
    """P(≥8%) for every tier, year by year.

    table_year_by_year covers High Alert only. Any change that reclassifies events
    between tiers moves Elevated most, so that tier needs its own stability check —
    a variant that lifts the pooled average while collapsing in individual years is
    not an improvement.
    """
    rows = []
    for year in sorted(ev["year"].unique()):
        yr  = ev[ev["year"] == year]
        row = {"Year": year, "N": len(yr)}
        for tier in TIER_ORDER:
            sub = yr[yr["earnings_explosiveness_bucket"] == tier]
            n   = len(sub)
            hits = int((sub["abs_reaction_3d"] >= 0.08).sum())
            row[f"{tier} n"]      = n
            row[f"{tier} P(≥8%)"] = f"{hits / n * 100:.1f}%" if n > 0 else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def table_consistency(ev: pd.DataFrame) -> pd.DataFrame:
    """Score/bucket monotonicity.

    Guards the defect this script exists to catch: an event carrying a higher tier
    label than another while scoring lower. Reports each tier's risk_score range and
    counts overlaps between adjacent tiers. Any non-zero overlap means a score-sorted
    list will disagree with the tier labels shown next to it.
    """
    score_col = "risk_score" if "risk_score" in ev.columns else "earnings_explosiveness_score"
    rows = []
    for tier in TIER_ORDER:
        sub = ev[ev["earnings_explosiveness_bucket"] == tier][score_col].dropna()
        rows.append({
            "Tier":      tier,
            "Score col": score_col,
            "N":         len(sub),
            "Min score": round(sub.min(), 2) if len(sub) else "—",
            "Max score": round(sub.max(), 2) if len(sub) else "—",
        })

    # Adjacent boundaries, walking high → low risk. Clean tiers never overlap: every
    # lower-tier score should sit below every higher-tier score. Any bleed means a
    # score-sorted list will disagree with the tier labels printed beside it.
    for hi_tier, lo_tier in zip(TIER_ORDER, TIER_ORDER[1:]):
        hi = ev[ev["earnings_explosiveness_bucket"] == hi_tier][score_col].dropna()
        lo = ev[ev["earnings_explosiveness_bucket"] == lo_tier][score_col].dropna()
        if not len(hi) or not len(lo):
            continue
        lo_above = int((lo > hi.min()).sum())   # lower-tier events outscoring the higher tier
        hi_below = int((hi < lo.max()).sum())   # higher-tier events undercutting the lower tier
        rows.append({
            "Tier":      f"OVERLAP {lo_tier} / {hi_tier}",
            "Score col": "",
            "N":         lo_above + hi_below,
            "Min score": f"{lo_above} {lo_tier} > {hi_tier} min",
            "Max score": f"{hi_below} {hi_tier} < {lo_tier} max",
        })
    return pd.DataFrame(rows)


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ev = load_oos_events()
    print(f"OOS earnings events: {len(ev)}  ({OOS_START} → {OOS_END})\n")

    t1 = table_by_bucket(ev)
    display_cols = (["Bucket", "N Events", "P(<1%)"] + [f"P({l})" for l in LABELS]
                    + ["P(≥8%) 95% CI", "Lift ≥8%"])
    print("=== 1. By Bucket ===")
    print(t1[display_cols].to_string(index=False))
    t1.to_csv("testing/testing_results/calibration_by_bucket.csv", index=False)

    print("\n=== 2. Capture Rate (High Alert + Elevated selected) ===")
    t2 = table_capture_rate(ev)
    print(t2.to_string(index=False))
    t2.to_csv("testing/testing_results/calibration_capture_rate.csv", index=False)

    print("\n=== 3. By Percentile Band ===")
    t3 = table_by_percentile(ev)
    print(t3.to_string(index=False))
    t3.to_csv("testing/testing_results/calibration_by_percentile.csv", index=False)

    print("\n=== 4. Year-by-Year (High Alert, P(≥8%)) ===")
    t4 = table_year_by_year(ev)
    print(t4.to_string(index=False))
    t4.to_csv("testing/testing_results/calibration_year_by_year.csv", index=False)

    print("\n=== 5. Tier Stability by Year (P(≥8%) per tier) ===")
    t5 = table_tier_stability(ev)
    print(t5.to_string(index=False))
    t5.to_csv("testing/testing_results/calibration_tier_stability.csv", index=False)

    print("\n=== 6. Score/Bucket Consistency ===")
    t6 = table_consistency(ev)
    print(t6.to_string(index=False))
    t6.to_csv("testing/testing_results/calibration_consistency.csv", index=False)

    print(f"\nCSVs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
