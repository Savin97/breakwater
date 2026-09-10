"""How much history is actually there, and from when.

"Benzinga history starts in 2010" is a statement about the earliest row, not about
coverage. This module answers the only question Phase 3 cares about: at each point in
history, how many of the stocks Breakwater actually scores have enough PRIOR usable timed
events to build an event-level statistic from.

The maturity thresholds (8 / 12 / 20 / 28 prior events) are quarterly reporters at 2, 3, 5
and 7 years of history. 28 is the one that matters for the expanding per-stock reaction
statistics; anything shorter and the per-stock distribution is mostly the shrinkage prior.

A "usable timed event" here is one the vendor gives a real clock for — `time_usable`, so
the midnight filler is excluded — on or before the as-of date. Records dated after the
as-of date are invisible by construction, which is what makes these counts a point-in-time
answer rather than a hindsight one.
"""
import pandas as pd

from feature_engineering.announcement_timing import AMC, BMO, INTRADAY, UNKNOWN

MATURITY_THRESHOLDS = (8, 12, 20, 28)
# "A meaningful fraction of the universe is mature" — stated once, here, so the report's
# headline date is not a number picked to flatter the answer.
MEANINGFUL_FRACTION = 0.80


def breakwater_universe(path) -> list[str]:
    """Today's Breakwater tickers, from `data/stock_list.csv`. Read-only, no database."""
    col = pd.read_csv(path)
    name = next(c for c in col.columns if c.lower() in {"stock", "ticker", "symbol"})
    return sorted(col[name].astype(str).str.strip().str.upper().dropna().unique())


def yearly_profile(bz: pd.DataFrame, universe: list[str] | None = None) -> pd.DataFrame:
    """Records, tickers, timing quality and window mix per report year."""
    d = bz[bz["report_year"].notna()].copy()
    d["year"] = d["report_year"].astype(int)
    g = d.groupby("year")
    out = pd.DataFrame({
        "records": g.size(),
        "unique_tickers": g["ticker_norm"].nunique(),
        "confirmed": g["is_confirmed"].sum(),
        "usable_time": g["time_usable"].sum(),
    })
    for w in (BMO, AMC, INTRADAY, UNKNOWN):
        out[w.lower()] = g["announce_window"].apply(lambda s, w=w: (s == w).sum())
    out["usable_share"] = out["usable_time"] / out["records"]
    out["confirmed_share"] = out["confirmed"] / out["records"]
    if universe:
        uni = set(universe)
        cov = d[d["ticker_norm"].isin(uni)].groupby("year")["ticker_norm"].nunique()
        covt = (d[d["ticker_norm"].isin(uni) & d["time_usable"]]
                .groupby("year")["ticker_norm"].nunique())
        out["breakwater_tickers"] = cov.reindex(out.index).fillna(0).astype(int)
        out["breakwater_tickers_timed"] = covt.reindex(out.index).fillna(0).astype(int)
        out["breakwater_coverage"] = out["breakwater_tickers"] / len(uni)
        out["breakwater_coverage_timed"] = out["breakwater_tickers_timed"] / len(uni)
    return out.reset_index()


def maturity_curve(bz: pd.DataFrame, universe: list[str],
                   thresholds=MATURITY_THRESHOLDS,
                   start_year: int = 2010, end_year: int | None = None) -> pd.DataFrame:
    """For each year end: how many universe stocks have >= N PRIOR usable timed events.

    Counted per ticker on the vendor's own history, so a stock the vendor simply does not
    cover contributes zeros — which is the honest answer for a Phase 3 walk-forward.
    """
    uni = sorted(set(universe))
    d = bz[bz["time_usable"] & bz["ticker_norm"].isin(uni)][["ticker_norm", "report_date"]]
    d = d.dropna().sort_values("report_date")
    end_year = end_year or int(d["report_date"].dt.year.max())
    rows = []
    for year in range(start_year, end_year + 1):
        asof = pd.Timestamp(year=year, month=12, day=31)
        counts = (d[d["report_date"] <= asof].groupby("ticker_norm").size()
                  .reindex(uni, fill_value=0))
        row = {"asof_year": year, "universe": len(uni),
               "covered_at_all": int((counts > 0).sum())}
        for n in thresholds:
            row[f"n_ge_{n}"] = int((counts >= n).sum())
            row[f"frac_ge_{n}"] = float((counts >= n).mean())
        row["median_prior_events"] = float(counts.median())
        rows.append(row)
    return pd.DataFrame(rows)


def first_mature_year(curve: pd.DataFrame, threshold: int = 28,
                      fraction: float = MEANINGFUL_FRACTION) -> int | None:
    """Earliest year end at which `fraction` of the universe has >= `threshold` prior events."""
    col = f"frac_ge_{threshold}"
    hit = curve[curve[col] >= fraction]
    return int(hit["asof_year"].iloc[0]) if len(hit) else None


def universe_coverage(bz: pd.DataFrame, universe: list[str]) -> dict:
    """How much of today's Breakwater universe the vendor knows about at all."""
    uni = set(universe)
    present = set(bz["ticker_norm"].dropna().unique())
    timed = set(bz.loc[bz["time_usable"], "ticker_norm"].dropna().unique())
    return {
        "universe": len(uni),
        "present_in_vendor": len(uni & present),
        "with_any_usable_time": len(uni & timed),
        "absent": sorted(uni - present),
        "present_but_never_timed": sorted((uni & present) - timed),
    }
