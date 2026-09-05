"""Phase 2 diagnostics — properties of the corrected outcome dataset.

Describes the timing-aware target and nothing else. **No model performance is
interpreted here**: no tier hit rates, no lift, no capture. Those need the whole
historical chain rebuilt from anchored outcomes and the thresholds re-fit, which is
Phase 3 (audit/PHASE0_AUDIT_REV2.md Q3). Reading a tier number off this dataset today
would be reading a tier that a broken instrument produced.

Everything below is computed on events with an INDEPENDENTLY OBSERVED announcement
timestamp. Nothing is inferred from price behavior.

Usage:
    PYTHONPATH=. .venv/bin/python -m audit.phase2_diagnostics
"""
import numpy as np
import pandas as pd
import duckdb

from config import DB_PATH, EXTREME_EARNINGS_REACTION_THRESHOLD as X
from feature_engineering.announcement_timing import (
    AMC, BMO, INTRADAY, ANNOUNCE_WINDOWS,
    RESOLVED, UNRESOLVED_REASONS,
    UNRESOLVED_NO_SESSION, UNRESOLVED_PRICE_GAP,
    resolved_events,
)

EVENTS_PATH = "output/events_df.parquet"
FULL_PATH = "output/full_df.parquet"
RULE = "=" * 78


def h(title):
    print(f"\n{RULE}\n{title}\n{RULE}")


def missing_event_census(con, daily_df) -> pd.DataFrame:
    """Earnings events the DB holds that never reach the event frame.

    The event frame is built from `is_earnings_day == 1` rows, and a row only exists
    where the ticker has a price row on the report date. So an event on a date the
    ticker has no price for is not "unresolved" — it is ABSENT, which is worse, because
    absence is silent. Q6 found 318 of these and they are why nothing is auto-rolled:
    a date the market did not trade and a date the market traded but we failed to
    ingest have different causes and different fixes.
    """
    earnings = con.execute("""
        SELECT DISTINCT stock, earnings_date FROM earnings
        WHERE earnings_date IS NOT NULL
    """).fetch_df()
    earnings["earnings_date"] = pd.to_datetime(earnings["earnings_date"])

    have = daily_df.loc[daily_df["is_earnings_day"] == 1, ["stock", "earnings_date"]]
    have = set(map(tuple, have.to_numpy()))
    lo, hi = daily_df["date"].min(), daily_df["date"].max()
    sessions = set(pd.unique(daily_df["date"]))
    stock_span = daily_df.groupby("stock")["date"].agg(["min", "max"])

    rows = []
    for stock, ed in earnings.to_numpy():
        if (stock, ed) in have:
            continue
        if not (lo <= ed <= hi):
            reason = "outside the loaded price window (future date, or pre-history)"
        elif stock not in stock_span.index:
            reason = "ticker has no price history at all"
        elif not (stock_span.at[stock, "min"] <= ed <= stock_span.at[stock, "max"]):
            reason = "outside this ticker's own price history"
        elif ed in sessions:
            reason = UNRESOLVED_PRICE_GAP
        else:
            reason = UNRESOLVED_NO_SESSION
        rows.append({"stock": stock, "earnings_date": ed, "reason": reason})
    return pd.DataFrame(rows, columns=["stock", "earnings_date", "reason"])


def main():
    events = pd.read_parquet(EVENTS_PATH)
    daily = pd.read_parquet(FULL_PATH, columns=["stock", "date", "earnings_date",
                                                "is_earnings_day"])
    completed = events[~events["is_pending"]]
    res = resolved_events(events)

    h("1. ANNOUNCEMENT WINDOWS — observed timestamps only, never price behavior")
    counts = completed["announce_window"].value_counts()
    for w in ANNOUNCE_WINDOWS:
        n = int(counts.get(w, 0))
        print(f"  {w:<10} {n:>7}  {n / len(completed):>6.1%}")
    print(f"  {'TOTAL':<10} {len(completed):>7}")
    ts = completed.loc[completed["announce_ts_ny"].notna(), "announce_ts_ny"]
    print(f"\n  timestamped events: {len(ts)} ({len(ts)/len(completed):.1%})   "
          f"range {ts.min():%Y-%m-%d} -> {ts.max():%Y-%m-%d}")
    print(f"  provenance: " + ", ".join(
        f"{k}={v}" for k, v in completed["announce_ts_source"].value_counts().items()))

    h("2. RESOLVED COVERAGE BY YEAR")
    completed = completed.assign(year=completed["earnings_date"].dt.year)
    by_year = completed.groupby("year").apply(
        lambda d: pd.Series({
            "events": len(d),
            "resolved": int((d["anchor_status"] == RESOLVED).sum()),
            "BMO": int((d["announce_window"] == BMO).sum()),
            "AMC": int((d["announce_window"] == AMC).sum()),
            "INTRADAY": int((d["announce_window"] == INTRADAY).sum()),
        }), include_groups=False)
    by_year["coverage"] = (by_year["resolved"] / by_year["events"]).map("{:.1%}".format)
    print(by_year.to_string())
    print("\n  Coverage is thin before ~2020 and zero early on. That is the binding "
          "constraint\n  on Phase 3: a walk-forward re-fit cannot claim a window the "
          "timestamps do not cover.")

    h("3. LEGACY vs ANCHORED  P(|reaction_3d| >= 8%)  — resolved events only")
    print(f"  n = {len(res)} resolved events   "
          f"{res['earnings_date'].min():%Y-%m-%d} -> {res['earnings_date'].max():%Y-%m-%d}")
    print(f"\n  {'slice':<12}{'n':>7}{'legacy':>12}{'anchored':>12}{'delta':>10}")
    for label, sub in [("ALL", res), (BMO, res[res["announce_window"] == BMO]),
                       (AMC, res[res["announce_window"] == AMC])]:
        s = sub.dropna(subset=["abs_reaction_3d", "abs_reaction_3d_anchored"])
        if not len(s):
            continue
        a, b = (s["abs_reaction_3d"] >= X).mean(), (s["abs_reaction_3d_anchored"] >= X).mean()
        print(f"  {label:<12}{len(s):>7}{a:>12.4f}{b:>12.4f}{b - a:>+10.4f}")
    print("\n  Read this as a measurement correction, not a result. The legacy BMO rate is"
          "\n  low because the legacy window starts one session AFTER the news.")

    h("4. BMO DISTRIBUTION — legacy vs anchored |reaction_3d|")
    b = res[res["announce_window"] == BMO].dropna(
        subset=["abs_reaction_3d", "abs_reaction_3d_anchored"])
    qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    dist = pd.DataFrame({
        "legacy": b["abs_reaction_3d"].quantile(qs).values,
        "anchored": b["abs_reaction_3d_anchored"].quantile(qs).values,
    }, index=[f"p{int(q*100)}" for q in qs])
    dist["ratio"] = dist["anchored"] / dist["legacy"]
    print(dist.round(4).to_string())
    print(f"\n  mean   legacy {b['abs_reaction_3d'].mean():.4f}   "
          f"anchored {b['abs_reaction_3d_anchored'].mean():.4f}   "
          f"({b['abs_reaction_3d_anchored'].mean() / b['abs_reaction_3d'].mean():.2f}x)")
    print(f"  n = {len(b)} BMO events")

    h("5. AMC EQUALITY CHECK — anchored must be BIT-IDENTICAL to legacy")
    a = res[res["announce_window"] == AMC]
    allok = True
    for k in (1, 3, 5):
        x = a[f"reaction_{k}d_anchored"].to_numpy(dtype="float64")
        y = a[f"reaction_{k}d"].to_numpy(dtype="float64")
        same = np.array_equal(x, y, equal_nan=True)
        allok &= same
        print(f"  reaction_{k}d : {'IDENTICAL' if same else 'MISMATCH'}   n={len(a)}")
    x = a["abs_reaction_3d_anchored"].to_numpy(dtype="float64")
    y = a["abs_reaction_3d"].to_numpy(dtype="float64")
    print(f"  abs_reaction_3d: {'IDENTICAL' if np.array_equal(x, y, equal_nan=True) else 'MISMATCH'}")
    print(f"\n  {'PASS' if allok else 'FAIL'} — AMC was never mismeasured, so the "
          f"correction must be a no-op there.\n  This is the control that proves the "
          f"anchoring code is not moving things on its own.")

    h("6. UNRESOLVED EVENTS — counts and reasons")
    st = completed["anchor_status"].value_counts()
    for status, n in st.items():
        if status == RESOLVED:
            continue
        print(f"  {n:>7}  {status:<30} {UNRESOLVED_REASONS.get(status, '')}")
    n_unres = len(completed) - int(st.get(RESOLVED, 0))
    print(f"  {n_unres:>7}  TOTAL UNRESOLVED ({n_unres / len(completed):.1%} of completed events)")
    inc = completed[(completed["anchor_status"] == RESOLVED)
                    & completed["reaction_3d_anchored"].isna()]
    print(f"\n  resolved but with an incomplete forward window: {len(inc)} "
          f"(the 3 sessions after the anchor have not all closed yet)")
    print("  None of these carry an anchored target, and resolved_events() is the only "
          "gate\n  into corrected calibration — invariant 7.")

    h("7. MISSING-PRICE / NON-SESSION CASES — counted, never rolled")
    con = duckdb.connect(DB_PATH, read_only=True)
    try:
        missing = missing_event_census(con, daily)
    finally:
        con.close()
    print("  Events in the `earnings` table with no row in the event frame:")
    for reason, n in missing["reason"].value_counts().items():
        print(f"    {n:>6}  {reason}")
    print(f"    {len(missing):>6}  TOTAL")
    gap = missing[missing["reason"] == UNRESOLVED_PRICE_GAP]
    if len(gap):
        print(f"\n  The {len(gap)} price-gap cases are an INGESTION bug, not a calendar "
              f"problem —\n  the market traded that day and we have no row. Rolling them "
              f"forward would hide it.")
        print("  " + ", ".join(f"{r.stock}@{r.earnings_date:%Y-%m-%d}"
                               for r in gap.head(25).itertuples()))
    print("\n  In-frame report-date session check (anchor_session_status):")
    for s, n in completed["anchor_session_status"].value_counts().items():
        print(f"    {n:>6}  {s}")

    print(f"\n{RULE}\nEND — dataset properties only. No model claim is made or implied.\n{RULE}")


if __name__ == "__main__":
    main()
