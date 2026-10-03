"""Point-in-time accounting values and features for every SEC-mapped Breakwater event.

    PYTHONPATH=. .venv/bin/python -m research.fundamentals.build

Reads the sealed Company Facts snapshot (`acquire.py`), the SEC pilot's filing index and
per-event selection (`output/sec_filings_pilot/{filings_index,events}.parquet` — the pilot's
CIK chains and eligibility rule, reused, not rebuilt). **No outcome column is read**: the
pilot's `events.parquet` carries clock and identity columns only, and this module never
opens the feature frame. Writes `output/fundamentals/`.

The point-in-time construction (fixed before any outcome was read; PREREGISTRATION.md §2)
---------------------------------------------------------------------------------------
1. **Filings.** For each stock, its ORIGINAL 10-Q / 10-K (and 10-QT / 10-KT) filings as the
   pilot's date-bounded CIK chain attributes them (`filings_index.parquet`). Amendments
   (10-Q/A, 10-K/A) are never used.
2. **Facts.** A Company Facts entry is used only if its `accn` IS that filing's accession. A
   value the SEC exposes today under a later accession (a restatement, an amendment, a
   later filing's comparative column) therefore cannot reach the quarter's record.
3. **Quarter record** = one per original periodic filing, describing the fiscal period ending
   at the filing's `report_period` (P):
   * duration facts (revenue, operating income): the 3-month value — start/end with end
     within ±`END_TOL` days of P and 70–120 days long (12- and 16-week quarters included). A
     10-K reports the year: its Q4 is the 10-K's own 3-month fact if it tags one, else
     FY (340–390 days, from the 10-K) − 9-month YTD (250–290 days, from the same fiscal
     year's ORIGINAL Q3 10-Q, same start ±`START_TOL`). Never a YTD value as a quarter.
     Year-over-year comparisons use the SAME filing's prior-year comparative column (same
     tag, same period length, ending 358-372 days before P): one accession, one basis.
   * instant facts (balance sheet): the instant at P (±`END_TOL`).
   * cash-flow / net income: YTD (longest duration ending at P, ≤ 390 days) and the
     prior-year YTD comparative in the same filing (same length ±`START_TOL`, ending
     358–372 days before P). These are only ever combined into a trailing-twelve-month
     figure (`ttm`), never treated as a quarter.
   Conflicting values for one (accession, concept, start, end) → missing, not picked.
   Units: `USD` only; anything else is missing.
4. **Event** = the pilot's selected filing S (latest 10-Q/10-K eligible at the call cutoff
   by `timing.eligible`). History = the stock's quarter records whose filing date is on or
   before S's and whose period ends on or before S's. Every value used carries the
   accession and filing date that supplied it; `known_at` = the latest of them, and
   `known_at <= call_cutoff_date` is asserted for every feature row.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.fundamentals import acquire
from research.sec_filings_pilot import paths as pilot_paths
from research.sec_filings_pilot import timing

OUT = Path("output/fundamentals")
PERIODIC = ("10-Q", "10-K", "10-QT", "10-KT")
ANNUAL = ("10-K", "10-KT")

# One economic quantity per key. Alternatives only where they are the same quantity under a
# different taxonomy era / presentation, tried in this order PER FILING.
CONCEPTS: dict[str, list[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                "RevenuesNetOfInterestExpense"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["ProfitLoss", "NetIncomeLoss"],      # consolidated first, like CFO
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "assets": ["Assets"],
    "liabilities": ["Liabilities"],
    "liabilities_and_equity": ["LiabilitiesAndStockholdersEquity"],
    "equity": ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
               "StockholdersEquity"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "inventory": ["InventoryNet"],
}
DURATION = ("revenue", "operating_income")
YTD = ("net_income", "cfo")
INSTANT = ("assets", "liabilities", "liabilities_and_equity", "equity", "current_assets",
           "current_liabilities", "cash", "inventory")
ALL_TAGS = {t: k for k, ts in CONCEPTS.items() for t in ts}

END_TOL = 10          # days between a fact's end and the filing's report period
START_TOL = 7         # days between two starts that must be the same fiscal-year start
QUARTER_DAYS = (70, 120)
NINE_MONTH_DAYS = (250, 290)
YEAR_DAYS = (340, 390)
YOY_DAYS = (358, 372)  # a comparable period ends this many days earlier

# Feature windows (PREREGISTRATION.md §3), fixed before outcomes.
TRAIL_QUARTERS = 8          # quarters ending within TRAIL_DAYS of S's period
TRAIL_DAYS = 2 * 365 - 30   # 8 quarter-ends: E0 and the 7 before it
MIN_TRAIL = 6               # of the 8, at least this many YoY values
MARGIN_CLIP = 1.0


# ─────────────────────────────────────── loading ────────────────────────────────────────
def snapshot_dir() -> Path:
    return acquire.acquire.latest_snapshot(acquire.XBRL_SNAPSHOT_ROOT)


def facts_from_json(d: dict) -> pd.DataFrame:
    """Long table of the CONCEPTS tags of one Company Facts file (us-gaap only)."""
    rows = []
    g = d.get("facts", {}).get("us-gaap", {})
    for tag in ALL_TAGS:
        if tag not in g:
            continue
        for unit, entries in g[tag]["units"].items():
            for e in entries:
                rows.append((tag, unit, e.get("start"), e["end"], e["val"], e["accn"],
                             e.get("form"), e.get("filed"), e.get("fp"), e.get("fy")))
    f = pd.DataFrame(rows, columns=["tag", "unit", "start", "end", "val", "accession", "fact_form",
                                    "fact_filed", "fp", "fy"])
    f["cik"] = int(d["cik"])
    return f


def load_facts(snap: Path | None = None) -> pd.DataFrame:
    snap = snap or snapshot_dir()
    frames = [facts_from_json(json.loads(gzip.decompress(p.read_bytes())))
              for p in sorted((snap / "companyfacts").glob("*.json.gz"))]
    f = pd.concat(frames, ignore_index=True)
    f["concept"] = f["tag"].map(ALL_TAGS)
    f["start"] = pd.to_datetime(f["start"])
    f["end"] = pd.to_datetime(f["end"])
    f["fact_filed"] = pd.to_datetime(f["fact_filed"])
    f["snapshot_id"] = snap.name
    return f


def periodic_filings() -> pd.DataFrame:
    """Original periodic filings per stock, from the pilot's chain-bounded filing index."""
    fi = pd.read_parquet(pilot_paths.OUT / "filings_index.parquet",
                         columns=["stock", "cik", "accession", "form", "filing_date",
                                  "report_period", "mapping_provenance"])
    return fi[fi["form"].isin(PERIODIC)].reset_index(drop=True)


def events() -> pd.DataFrame:
    """The pilot's events: identity, clock, selected filing. Outcome-free by construction."""
    cols = ["event_id", "stock", "sector", "earnings_date", "year", "announce_window",
            "call_cutoff_date", "call_cutoff_ts", "identity_status", "periodic_accession",
            "periodic_form", "periodic_cik", "periodic_filing_date", "periodic_report_period"]
    return pd.read_parquet(pilot_paths.OUT / "events.parquet", columns=cols)


# ───────────────────────────────────── per filing ───────────────────────────────────────
def filing_facts(facts: pd.DataFrame, filings: pd.DataFrame) -> pd.DataFrame:
    """Facts of each stock's original periodic filings: exact accession match, USD only.

    Joined on (cik, accession) so a predecessor's accession reaches a stock only through
    the chain segment the pilot assigned it. Adds `dur` (days) and the filing's P.
    """
    m = filings.merge(facts, on=["cik", "accession"], how="inner")
    # Company Facts' `filed` is a business day later than the submissions filing date for a
    # handful of filings (7 in the 2026-10-02 snapshot). Known-at is the LATER of the two.
    m["filing_date_submissions"] = m["filing_date"]
    m["filing_date"] = np.maximum(m["filing_date"],
                                  m.groupby("accession")["fact_filed"].transform("max"))
    m = m[m["unit"].eq("USD")].copy()
    m["dur"] = (m["end"] - m["start"]).dt.days
    return m


def _distinct(vals: pd.Series):
    u = pd.unique(vals)
    return float(u[0]) if len(u) == 1 else np.nan      # conflict -> missing


def _pick(ff: pd.DataFrame, concept: str, sel) -> tuple[float, str | None, pd.Series | None]:
    """First tag of `concept` (priority order) with exactly one value among rows `sel`."""
    for tag in CONCEPTS[concept]:
        r = ff[(ff["tag"] == tag) & sel(ff)]
        if len(r):
            grp = r.groupby(["start", "end"], dropna=False)["val"].agg(_distinct)
            if len(grp) == 1 and np.isfinite(grp.iloc[0]):
                return grp.iloc[0], tag, r.iloc[0]
            return np.nan, f"{tag}:conflict", None
    return np.nan, None, None


def _near(a: pd.Series, b, tol: int) -> pd.Series:
    return (a - b).abs().dt.days <= tol


def _prior(g: pd.DataFrame, tag, P, dur_range) -> float:
    """The SAME filing's prior-year comparative for `tag`: same tag, a period of the same
    kind (`dur_range`) ending 358-372 days before P. One distinct value, else missing."""
    if not isinstance(tag, str) or tag.endswith(":conflict"):
        return np.nan
    r = g[g["tag"].eq(tag) & (P - g["end"]).dt.days.between(*YOY_DAYS)
          & g["dur"].between(*dur_range)]
    if r.empty:
        return np.nan
    grp = r.groupby(["start", "end"])["val"].agg(_distinct)
    return float(grp.iloc[0]) if len(grp) == 1 and np.isfinite(grp.iloc[0]) else np.nan


def quarter_records(ff: pd.DataFrame, workers: int = 8) -> pd.DataFrame:
    """One row per original periodic filing (module docstring, step 3), parallel by stock."""
    from concurrent.futures import ProcessPoolExecutor
    parts = [g for _, g in ff.groupby("stock", sort=True)]
    if workers <= 1:
        return pd.concat([_stock_records(g) for g in parts], ignore_index=True)
    with ProcessPoolExecutor(workers) as ex:
        return pd.concat(list(ex.map(_stock_records, parts, chunksize=4)), ignore_index=True)


def _stock_records(sf: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (stock, acc), g in sf.groupby(["stock", "accession"], sort=False):
        f0 = g.iloc[0]
        P = f0["report_period"]
        rec = {"stock": stock, "accession": acc, "cik": f0["cik"], "form": f0["form"],
               "filing_date": f0["filing_date"], "period_end": P}
        if pd.isna(P):
            rec["record_status"] = "no_report_period"
            out.append(rec)
            continue
        at_P = _near(g["end"], P, END_TOL)
        for k in INSTANT:
            v, tag, _ = _pick(g, k, lambda x: at_P.loc[x.index] & x["start"].isna())
            rec[k], rec[f"{k}_tag"] = v, tag
        for k in DURATION:
            v, tag, row = _pick(g, k, lambda x: at_P.loc[x.index]
                                & x["dur"].between(*QUARTER_DAYS))
            rec[f"{k}_q"], rec[f"{k}_q_tag"] = v, tag
            rec[f"{k}_q_prior"] = _prior(g, tag, P, QUARTER_DAYS)
            rec[f"{k}_q_route"] = "own_quarter" if np.isfinite(v) else None
            if f0["form"] in ANNUAL:
                fy, fytag, fyrow = _pick(g, k, lambda x: at_P.loc[x.index]
                                         & x["dur"].between(*YEAR_DAYS))
                rec[f"{k}_fy"], rec[f"{k}_fy_tag"] = fy, fytag
                rec[f"{k}_fy_prior"] = _prior(g, fytag, P, YEAR_DAYS)
                rec[f"{k}_fy_start"] = fyrow["start"] if fyrow is not None else pd.NaT
        for k in YTD:
            ytd = g[at_P & g["dur"].between(QUARTER_DAYS[0], YEAR_DAYS[1])
                    & g["concept"].eq(k)]
            v, tag, row = (np.nan, None, None)
            if len(ytd):
                longest = ytd["dur"].max()
                v, tag, row = _pick(g, k, lambda x: at_P.loc[x.index] & x["dur"].eq(longest))
            rec[f"{k}_ytd"], rec[f"{k}_ytd_tag"] = v, tag
            rec[f"{k}_ytd_start"] = row["start"] if row is not None else pd.NaT
            rec[f"{k}_ytd_dur"] = row["dur"] if row is not None else np.nan
            pv = np.nan
            if row is not None:
                prior_end = (P - g["end"]).dt.days.between(*YOY_DAYS)
                d0 = row["dur"]
                pv, _, _ = _pick(g[g["tag"].eq(tag)], k, lambda x: prior_end.loc[x.index]
                                 & (x["dur"] - d0).abs().le(START_TOL))
            rec[f"{k}_ytd_prior"] = pv
        nine = {}
        if f0["form"] in ("10-Q", "10-QT"):
            for k in DURATION:
                v, tag, row = _pick(g, k, lambda x: at_P.loc[x.index]
                                    & x["dur"].between(*NINE_MONTH_DAYS))
                nine[k] = (v, tag, row["start"] if row is not None else pd.NaT,
                           _prior(g, tag, P, NINE_MONTH_DAYS))
        for k in DURATION:
            (rec[f"{k}_9m"], rec[f"{k}_9m_tag"], rec[f"{k}_9m_start"],
             rec[f"{k}_9m_prior"]) = nine.get(k, (np.nan, None, pd.NaT, np.nan))
        rec["record_status"] = "ok"
        out.append(rec)
    return pd.DataFrame(out)


def derive_q4(qr: pd.DataFrame) -> pd.DataFrame:
    """Annual filings without their own 3-month fact: Q4 = FY − 9M YTD of the same fiscal
    year's original Q3 10-Q (same tag, same start ±START_TOL, filed before the 10-K)."""
    qr = qr.sort_values(["stock", "period_end", "filing_date"]).reset_index(drop=True)
    for k in DURATION:
        qr[f"{k}_q_q4_source_accession"] = None
        if f"{k}_fy" not in qr:
            continue
        need = qr[qr["form"].isin(ANNUAL) & qr[f"{k}_q"].isna() & qr[f"{k}_fy"].notna()]
        pool = qr.loc[qr[f"{k}_9m"].notna(), ["stock", "accession", "filing_date", "period_end",
                                               f"{k}_9m", f"{k}_9m_tag", f"{k}_9m_start",
                                               f"{k}_9m_prior"]]
        m = need[["stock", "filing_date", "period_end", f"{k}_fy", f"{k}_fy_tag",
                  f"{k}_fy_start", f"{k}_fy_prior"]].reset_index().merge(
            pool, on="stock", suffixes=("", "_q3"))
        m = m[(m["filing_date_q3"] <= m["filing_date"])
              & m[f"{k}_9m_tag"].eq(m[f"{k}_fy_tag"])
              & ((m[f"{k}_9m_start"] - m[f"{k}_fy_start"]).abs().dt.days <= START_TOL)
              & (m["period_end"] - m["period_end_q3"]).dt.days.between(*QUARTER_DAYS)]
        m = m[~m["index"].duplicated(keep=False)]          # exactly one Q3 10-Q, else missing
        i = m["index"].to_numpy()
        qr.loc[i, f"{k}_q"] = (m[f"{k}_fy"] - m[f"{k}_9m"]).to_numpy()
        # Prior-year Q4 on the same two filings' comparative columns.
        qr.loc[i, f"{k}_q_prior"] = (m[f"{k}_fy_prior"] - m[f"{k}_9m_prior"]).to_numpy()
        qr.loc[i, f"{k}_q_tag"] = m[f"{k}_fy_tag"].to_numpy()
        qr.loc[i, f"{k}_q_route"] = "fy_minus_9m"
        qr.loc[i, f"{k}_q_q4_source_accession"] = m["accession"].to_numpy()
    # Leverage: total liabilities, else total liabilities-and-equity minus total equity.
    qr["liabilities_route"] = np.where(qr["liabilities"].notna(), "Liabilities", None)
    alt = qr["liabilities"].isna() & qr["liabilities_and_equity"].notna() & qr["equity"].notna()
    qr.loc[alt, "liabilities"] = qr.loc[alt, "liabilities_and_equity"] - qr.loc[alt, "equity"]
    qr.loc[alt, "liabilities_route"] = "LiabilitiesAndStockholdersEquity-equity"
    return qr


# ───────────────────────────────────── per event ────────────────────────────────────────
def _yoy_index(ends: np.ndarray, j: int) -> int | None:
    d = (ends[j] - ends[:j]).astype("timedelta64[D]").astype(int)
    hit = np.flatnonzero((d >= YOY_DAYS[0]) & (d <= YOY_DAYS[1]))
    return int(hit[-1]) if len(hit) else None


def _margin(oi, rev):
    if not (np.isfinite(oi) and np.isfinite(rev)) or rev <= 0:
        return np.nan
    return float(np.clip(oi / rev, -MARGIN_CLIP, MARGIN_CLIP))


def _margin_change(rec) -> float:
    """Operating margin of the quarter minus that of the prior-year quarter, both from the
    same filing(s) — the quarter's own figures and their comparative column."""
    m1 = _margin(rec["operating_income_q"], rec["revenue_q"])
    m0 = _margin(rec["operating_income_q_prior"], rec["revenue_q_prior"])
    return m1 - m0 if np.isfinite(m1) and np.isfinite(m0) else np.nan


def event_features(ev: pd.DataFrame, qr: pd.DataFrame) -> pd.DataFrame:
    """One row per mapped event. `known_at` = latest filing date of any record used."""
    by_stock = {s: g.sort_values(["period_end", "filing_date"]).reset_index(drop=True)
                for s, g in qr.groupby("stock")}
    rows = []
    for e in ev.itertuples():
        r = {"event_id": e.event_id, "selected_accession": e.periodic_accession,
             "selected_form": e.periodic_form, "selected_filing_date": e.periodic_filing_date,
             "call_cutoff_date": e.call_cutoff_date}
        if e.identity_status != "mapped" or pd.isna(e.periodic_accession):
            r["status"] = "no_selected_filing"
            rows.append(r)
            continue
        g = by_stock.get(e.stock)
        if g is None or not g["accession"].eq(e.periodic_accession).any():
            r["status"] = "selected_filing_no_xbrl"
            rows.append(r)
            continue
        s_row = g[g["accession"].eq(e.periodic_accession)].iloc[0]
        if s_row["filing_date"] > e.periodic_filing_date and s_row["filing_date"] >= e.call_cutoff_date:
            r["status"] = "xbrl_filed_date_after_cutoff"     # Company Facts dates it later
            rows.append(r)
            continue
        # Earlier-filed records only; one record per period (the first original filed),
        # except that S itself always stands for its own period.
        h = g[(g["filing_date"] <= s_row["filing_date"]) & (g["period_end"] <= s_row["period_end"])
              & (g["accession"].eq(e.periodic_accession) | g["period_end"].ne(s_row["period_end"]))]
        h = h.drop_duplicates("period_end", keep="first").reset_index(drop=True)
        j = int(np.flatnonzero(h["accession"].eq(e.periodic_accession))[-1])
        h = h.iloc[: j + 1]
        ends = h["period_end"].to_numpy("datetime64[D]")
        r.update(status="ok", period_end=s_row["period_end"], known_at=s_row["filing_date"],
                 info_age_days=(e.call_cutoff_date - s_row["filing_date"]).days,
                 period_age_days=(e.call_cutoff_date - s_row["period_end"]).days)
        # Size control and balance sheet at S.
        A = s_row["assets"]
        r["assets"] = A
        r["log_assets"] = np.log(A) if np.isfinite(A) and A > 0 else np.nan
        ok_A = np.isfinite(A) and A > 0
        r["leverage"] = s_row["liabilities"] / A if ok_A and np.isfinite(s_row["liabilities"]) else np.nan
        r["leverage_route"] = s_row["liabilities_route"]
        wc = (s_row["current_assets"] - s_row["current_liabilities"]) / A if ok_A else np.nan
        r["wc_to_assets"] = wc
        r["inventory_to_assets"] = s_row["inventory"] / A if ok_A else np.nan
        jy = _yoy_index(ends, j)
        r["yoy_accession"] = h.loc[jy, "accession"] if jy is not None else None
        if jy is not None:
            y = h.loc[jy]
            Ay = y["assets"]
            wcy = (y["current_assets"] - y["current_liabilities"]) / Ay if np.isfinite(Ay) and Ay > 0 else np.nan
            r["abs_wc_change_yoy"] = abs(wc - wcy) if np.isfinite(wc) and np.isfinite(wcy) else np.nan
            invy = y["inventory"] / Ay if np.isfinite(Ay) and Ay > 0 else np.nan
            r["abs_inventory_change_yoy"] = abs(r["inventory_to_assets"] - invy)
        # Trailing quarters: YoY revenue growth and margin change.
        trail = [i for i in range(j + 1)
                 if (ends[j] - ends[i]).astype(int) <= TRAIL_DAYS][-TRAIL_QUARTERS:]
        g_rev, d_m = [], []
        for i in trail:
            a, b = h.loc[i, "revenue_q"], h.loc[i, "revenue_q_prior"]
            if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
                g_rev.append(np.log(a / b))
            dm = _margin_change(h.loc[i])
            if np.isfinite(dm):
                d_m.append(dm)
        r["n_trailing_quarters"] = len(trail)
        r["n_rev_growth"] = len(g_rev)
        r["n_margin_change"] = len(d_m)
        r["revenue_growth_volatility"] = float(np.std(g_rev, ddof=1)) if len(g_rev) >= MIN_TRAIL else np.nan
        r["operating_margin_volatility"] = float(np.std(d_m, ddof=1)) if len(d_m) >= MIN_TRAIL else np.nan
        r["abs_operating_margin_change_yoy"] = abs(_margin_change(s_row))
        r["revenue_q"] = s_row["revenue_q"]
        r["revenue_q_route"] = s_row["revenue_q_route"]
        r["operating_income_q"] = s_row["operating_income_q"]
        # Accruals: TTM net income and CFO; a YTD figure is never used as a quarter.
        ttm, ttm_src = {}, None
        for k in YTD:
            if s_row["form"] in ANNUAL:
                v = s_row[f"{k}_ytd"] if s_row[f"{k}_ytd_dur"] >= YEAR_DAYS[0] else np.nan
                ttm[k] = v
            else:
                fy = h[h["form"].isin(ANNUAL) & h[f"{k}_ytd_dur"].ge(YEAR_DAYS[0])
                       & _near(h["period_end"], s_row[f"{k}_ytd_start"] - pd.Timedelta(days=1), END_TOL)
                       & h[f"{k}_ytd_tag"].eq(s_row[f"{k}_ytd_tag"])]
                if len(fy) and np.isfinite(s_row[f"{k}_ytd_prior"]):
                    ttm[k] = s_row[f"{k}_ytd"] + fy.iloc[-1][f"{k}_ytd"] - s_row[f"{k}_ytd_prior"]
                    ttm_src = fy.iloc[-1]["accession"]
                else:
                    ttm[k] = np.nan
        r["net_income_ttm"], r["cfo_ttm"] = ttm["net_income"], ttm["cfo"]
        r["ttm_prior_10k_accession"] = ttm_src
        r["accruals_signed"] = (ttm["net_income"] - ttm["cfo"]) / A if ok_A else np.nan
        r["abs_accruals"] = abs(r["accruals_signed"])
        # Every value above comes from filings in h (filed <= S's filing date).
        r["history_first_filing"] = h["filing_date"].min()
        r["n_quarters_available"] = len(h)
        rows.append(r)
    return pd.DataFrame(rows)


def check_eligibility(feat: pd.DataFrame, ev: pd.DataFrame) -> int:
    """The invariant: known_at <= call cutoff, by the pilot's rule. Returns rows checked."""
    m = feat[feat["status"].eq("ok")].merge(ev[["event_id", "call_cutoff_ts"]], on="event_id")
    va = pd.read_parquet(pilot_paths.OUT / "verified_acceptance.parquet")
    va = dict(zip(va["accession"], va["acceptance_verified_ny"]))
    ok = timing.eligible(m["known_at"], m["call_cutoff_date"], m["call_cutoff_ts"],
                         pd.to_datetime(m["selected_accession"].map(va)))
    if not ok.all():
        raise AssertionError(f"{int((~ok).sum())} feature rows known after the cutoff")
    return len(m)


# ───────────────────────────────────────── main ─────────────────────────────────────────
def run() -> dict:
    snap = snapshot_dir()
    facts = load_facts(snap)
    fil = periodic_filings()
    ff = filing_facts(facts, fil)
    # Company Facts' own filing date must equal SEC's submissions filing date.
    bad_filed = int((ff["fact_filed"] != ff["filing_date_submissions"]).sum())
    qr = derive_q4(quarter_records(ff))
    ev = events()
    feat = event_features(ev, qr)
    n_checked = check_eligibility(feat, ev)
    OUT.mkdir(parents=True, exist_ok=True)
    qr.to_parquet(OUT / "quarter_records.parquet")
    feat = feat.merge(ev[["event_id", "stock", "sector", "year", "announce_window",
                          "earnings_date"]], on="event_id", how="left")
    feat.to_parquet(OUT / "fundamentals.parquet")
    info = {"xbrl_snapshot": snap.name,
            "snapshot_sha256": json.loads((snap / "manifest.json").read_text())["snapshot_sha256"],
            "facts_rows": len(facts), "facts_on_original_periodic_filings": len(ff),
            "facts_filed_date_mismatch": bad_filed, "quarter_records": len(qr),
            "periodic_filings": len(fil), "events": len(feat),
            "status": feat["status"].value_counts().to_dict(), "eligibility_checked": n_checked}
    (OUT / "build_stats.json").write_text(json.dumps(info, indent=1, default=str))
    return info


def main(argv=None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    print(json.dumps(run(), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
