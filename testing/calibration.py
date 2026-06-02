"""
Historical calibration tables for the earnings_explosiveness model.

Reads output/full_df.parquet and uses earnings-day rows (is_earnings_day == 1)
as the unit of analysis. Scores are computed with shift(1) so there is no
lookahead. Produces four tables:

  1. By bucket      — event counts + P(<1%), P(≥3/5/8/10/15%) for each risk tier
  2. Capture rate   — share of all large moves that fell in High Alert / Elevated
  3. Percentile band — P(≥8/10%) across six percentile bands
  4. Year-by-year   — P(≥8%) for High Alert by calendar year

Outputs: prints tables to stdout and saves CSVs to testing/testing_results/.

Run with: .venv/bin/python -m testing.calibration
"""
import os
import pandas as pd

PARQUET_PATH = "output/full_df.parquet"
RESULTS_DIR  = "testing/testing_results"
OOS_START    = "2015-01-01"
OOS_END      = "2025-12-31"

THRESHOLDS = [0.03, 0.05, 0.08, 0.10, 0.15]
LABELS     = ["≥3%", "≥5%", "≥8%", "≥10%", "≥15%"]


def _load_oos_events() -> pd.DataFrame:
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


def _hit_rates(sub: pd.DataFrame) -> dict:
    n = len(sub)
    row: dict = {"N Events": n}
    calm = (sub["abs_reaction_3d"] < 0.01).sum()
    row["P(<1%)"] = f"{calm / n * 100:.1f}%" if n > 0 else "—"
    for thresh, label in zip(THRESHOLDS, LABELS):
        hits = (sub["abs_reaction_3d"] >= thresh).sum()
        row[f"P({label})"] = f"{hits / n * 100:.1f}%" if n > 0 else "—"
        row[f"_n_{label}"] = hits
    return row


def table_by_bucket(ev: pd.DataFrame) -> pd.DataFrame:
    tiers = [
        ("Normal",          ev["earnings_explosiveness_bucket"] == "Normal"),
        ("Elevated",        ev["earnings_explosiveness_bucket"] == "Elevated"),
        ("High Alert",      ev["earnings_explosiveness_bucket"] == "High Alert"),
        ("HC (HA + Drift)", ev["is_hc"]),
    ]
    rows = []
    for label, mask in tiers:
        row = {"Bucket": label}
        row.update(_hit_rates(ev[mask]))
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
    _rank_key = ev["abs_reaction_p75_rolling"].fillna(ev["abs_reaction_p75"])
    ev["pct"] = _rank_key.rank(pct=True) * 100
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


def run():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ev = _load_oos_events()
    print(f"OOS earnings events: {len(ev)}  ({OOS_START} → {OOS_END})\n")

    t1 = table_by_bucket(ev)
    display_cols = ["Bucket", "N Events", "P(<1%)"] + [f"P({l})" for l in LABELS]
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

    print(f"\nCSVs saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run()
