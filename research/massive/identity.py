"""Ticker and company identity hazards in the vendor history.

Joining 16 years of vendor rows onto today's ticker is only safe if a ticker meant the
same company for all 16 years. It frequently did not: symbols are recycled after a
delisting, companies rename, and a merger moves a history onto a new symbol. Every one of
those makes a naive join attach one company's earnings clock to another company's prices.

This module only IDENTIFIES the hazard cases. It does not repair them and it deliberately
does not attempt a point-in-time index reconstruction — that is a separate, larger piece of
work, and doing half of it silently would be worse than not doing it.
"""
import pandas as pd


def _name_table(bz: pd.DataFrame) -> pd.DataFrame:
    d = bz[["ticker_norm", "company_name", "company_name_norm", "report_date"]].dropna(
        subset=["ticker_norm", "company_name_norm"])
    return d


def tickers_with_multiple_companies(bz: pd.DataFrame, universe=None) -> pd.DataFrame:
    """One ticker, several company identities — a rename, or a recycled symbol.

    The two are told apart by the gap: a rename shows the names interleaved or adjacent in
    time, a recycled symbol shows a clean break, usually of years, between the last event
    under one name and the first under the next.
    """
    d = _name_table(bz)
    g = d.groupby("ticker_norm")["company_name_norm"].nunique()
    multi = g[g > 1].index
    rows = []
    for tkr, sub in d[d["ticker_norm"].isin(multi)].groupby("ticker_norm"):
        spans = (sub.groupby("company_name_norm")
                 .agg(first=("report_date", "min"), last=("report_date", "max"),
                      records=("report_date", "size"),
                      display=("company_name", "first"))
                 .sort_values("first"))
        # Largest gap between consecutive name spans. NEGATIVE means the spans OVERLAP,
        # which is itself a finding: the vendor backfills today's company name onto old
        # rows, so `company_name` is not a point-in-time field and cannot be used on its
        # own to date a rename.
        gap_days = (spans["first"].shift(-1) - spans["last"]).dt.days.max()
        rows.append({
            "ticker": tkr,
            "distinct_companies": len(spans),
            "names": " / ".join(spans["display"].astype(str)),
            "first": spans["first"].min(),
            "last": spans["last"].max(),
            "gap_between_name_spans_days": gap_days if pd.notna(gap_days) else 0,
            "records": int(spans["records"].sum()),
            "in_breakwater_universe": bool(universe and tkr in set(universe)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["in_breakwater_universe", "gap_between_name_spans_days"],
                           ascending=False)


def companies_under_multiple_tickers(bz: pd.DataFrame, universe=None) -> pd.DataFrame:
    """One company identity, several tickers — a symbol change or a dual listing."""
    d = _name_table(bz)
    g = d.groupby("company_name_norm")["ticker_norm"].nunique()
    multi = g[g > 1].index
    rows = []
    for name, sub in d[d["company_name_norm"].isin(multi)].groupby("company_name_norm"):
        spans = (sub.groupby("ticker_norm")
                 .agg(first=("report_date", "min"), last=("report_date", "max"),
                      records=("report_date", "size")).sort_values("first"))
        rows.append({
            "company": sub["company_name"].iloc[0],
            "tickers": " / ".join(spans.index),
            "n_tickers": len(spans),
            "first": spans["first"].min(),
            "last": spans["last"].max(),
            "records": int(spans["records"].sum()),
            "touches_breakwater_universe": bool(universe and set(spans.index) & set(universe)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["touches_breakwater_universe", "n_tickers", "records"],
                           ascending=False)


def history_gaps(bz: pd.DataFrame, universe, min_gap_days: int = 400) -> pd.DataFrame:
    """Universe tickers whose vendor history stops and restarts.

    A quarterly reporter should never go 400 days between events. When one does, the rows
    on either side of the hole are usually not the same company.
    """
    uni = set(universe)
    d = (bz[bz["ticker_norm"].isin(uni)][["ticker_norm", "report_date", "company_name"]]
         .dropna(subset=["report_date"]).sort_values(["ticker_norm", "report_date"]))
    d["gap"] = d.groupby("ticker_norm")["report_date"].diff().dt.days
    d["prev_company"] = d.groupby("ticker_norm")["company_name"].shift(1)
    d["prev_date"] = d.groupby("ticker_norm")["report_date"].shift(1)
    big = d[d["gap"] >= min_gap_days].copy()
    big["name_changed"] = (big["company_name"].astype(str).str.casefold()
                           != big["prev_company"].astype(str).str.casefold())
    return big[["ticker_norm", "prev_date", "report_date", "gap", "prev_company",
                "company_name", "name_changed"]].sort_values("gap", ascending=False)


def summary(bz: pd.DataFrame, universe) -> dict:
    multi_c = tickers_with_multiple_companies(bz, universe)
    multi_t = companies_under_multiple_tickers(bz, universe)
    gaps = history_gaps(bz, universe)
    return {
        "tickers_with_multiple_companies": len(multi_c),
        "of_which_in_universe": int(multi_c["in_breakwater_universe"].sum()) if len(multi_c) else 0,
        "companies_under_multiple_tickers": len(multi_t),
        "of_which_touch_universe": int(multi_t["touches_breakwater_universe"].sum()) if len(multi_t) else 0,
        "universe_tickers_with_history_gaps": int(gaps["ticker_norm"].nunique()) if len(gaps) else 0,
        "universe_gap_events_with_name_change": int(gaps["name_changed"].sum()) if len(gaps) else 0,
    }


def _punctuation_variants(ticker: str) -> list[str]:
    """The spellings a class share can plausibly take: BRK-B, BRK.B, BRKB, BRK B."""
    base = ticker.strip().upper()
    plain = base.replace("-", "").replace(".", "").replace(" ", "")
    seen, out = set(), []
    for cand in (base, base.replace("-", "."), base.replace(".", "-"),
                 base.replace("-", " ").replace(".", " "), plain):
        if cand and cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def ticker_spelling_hazards(bz: pd.DataFrame, universe) -> pd.DataFrame:
    """Universe tickers the vendor spells differently, or spells more than one way.

    Class shares are where a symbol join quietly loses a company. Breakwater writes
    `BF-B`; the vendor writes `BF.B` for 54 records and `BFB` for 4 more. Joining on the
    literal string drops the ticker entirely and, worse, would drop it SILENTLY — the
    stock simply has no vendor history and looks like a coverage gap rather than a bug.
    """
    present = set(bz["ticker_norm"].dropna().astype(str).unique())
    counts = bz["ticker_norm"].value_counts()
    rows = []
    for tkr in sorted(set(universe)):
        variants = [v for v in _punctuation_variants(tkr) if v in present]
        exact = tkr in present
        if exact and len(variants) <= 1:
            continue
        rows.append({
            "breakwater_ticker": tkr,
            "exact_match_in_vendor": exact,
            "vendor_spellings": " / ".join(variants) or "—",
            "vendor_records": int(sum(counts.get(v, 0) for v in variants)),
            "records_under_exact": int(counts.get(tkr, 0)),
        })
    return pd.DataFrame(rows)
