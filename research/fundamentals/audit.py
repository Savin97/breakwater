"""Coverage and reliability audit of the point-in-time fundamentals — run BEFORE outcomes.

    PYTHONPATH=. .venv/bin/python -m research.fundamentals.audit --coverage
    export SEC_USER_AGENT="<name> <contact email>"
    PYTHONPATH=. .venv/bin/python -m research.fundamentals.audit --verify [--n 40 --seed 11]

`--coverage` writes coverage tables (feature, year, BMO/AMC, sector, information age,
quarters per stock, missing reasons, construction routes) to `output/fundamentals/`.

`--verify` draws a stratified sample of event rows and re-derives every number behind
their features from the ORIGINAL filings' XBRL instance documents on EDGAR — not from
Company Facts. For every filing a row used (the selected 10-Q/10-K, each trailing quarter's
filing, the Q3 10-Q behind a derived Q4, the prior 10-K behind a TTM figure, the filing a
year earlier) it checks: the accession's index page (filing date, period of report), the
registrant name in the instance (`dei:EntityRegistrantName`) against SEC's name history of
the CIK, the document type and period end, and each value by its own context (no
dimensions; quarter = 70-120 days, YTD/annual by length) and unit (USD). It then recomputes
the features from those verified numbers. Results: `verify_rows.csv` (one row per checked
value) and `verify_events.csv` (one row per event-feature).

The population is Phase 3's on the current corrected frame. Only the target's
AVAILABILITY is read (`y_extreme.notna()`), never its value.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from lxml import etree

from research.fundamentals import build
from research.sec_filings_pilot import acquire, documents, mapping
from research.sec_filings_pilot import paths as pilot_paths

OUT = build.OUT
FRAME = Path("output/sec_features/feature_frame_current.parquet")
POP_YEARS = (2014, 2026)
MIN_PRIOR = 8
FEATURES = ["revenue_growth_volatility", "operating_margin_volatility",
            "abs_operating_margin_change_yoy", "leverage", "abs_accruals", "wc_to_assets",
            "abs_wc_change_yoy", "inventory_to_assets", "abs_inventory_change_yoy", "log_assets"]


# ──────────────────────────────────────── population ────────────────────────────────────
def population() -> pd.DataFrame:
    """Phase 3 population (completed, BMO/AMC, 2014-2026, target available, >= 8 prior
    outcomes, both C inputs). Reads the target's availability only."""
    cols = ["stock", "earnings_date", "is_pending", "window_ok", "year", "announce_window",
            "sector", "n_prior_call", "hist_mean_abs_call", "vol_30d_call", "y_extreme"]
    fr = pd.read_parquet(FRAME, columns=cols)
    fr["target_available"] = fr.pop("y_extreme").notna()
    fr["event_id"] = fr["stock"] + "|" + pd.to_datetime(fr["earnings_date"]).dt.strftime("%Y-%m-%d")
    keep = (~fr["is_pending"].astype(bool) & fr["window_ok"] & fr["target_available"]
            & fr["year"].between(*POP_YEARS) & fr["n_prior_call"].ge(MIN_PRIOR)
            & fr["hist_mean_abs_call"].notna() & fr["vol_30d_call"].notna())
    return fr.loc[keep, ["event_id", "stock", "year", "announce_window", "sector"]].reset_index(drop=True)


def panel() -> pd.DataFrame:
    f = pd.read_parquet(OUT / "fundamentals.parquet").drop(
        columns=["stock", "sector", "year", "announce_window", "earnings_date"], errors="ignore")
    return population().merge(f, on="event_id", how="left")


# ───────────────────────────────────────── coverage ─────────────────────────────────────
def missing_reason(p: pd.DataFrame, feat: str) -> pd.Series:
    st = p["status"].fillna("not_in_sec_population")
    r = pd.Series(np.where(st.ne("ok"), st, None), index=p.index, dtype=object)
    ok = st.eq("ok") & p[feat].isna()
    if feat in ("revenue_growth_volatility", "operating_margin_volatility"):
        n = p["n_rev_growth"] if feat.startswith("revenue") else p["n_margin_change"]
        r[ok & p["n_trailing_quarters"].lt(build.MIN_TRAIL)] = "fewer_than_6_quarter_filings_in_2y"
        r[ok & r.isna() & n.lt(build.MIN_TRAIL)] = "fewer_than_6_quarters_with_values"
    elif feat == "abs_operating_margin_change_yoy":
        r[ok & p["operating_income_q"].isna()] = "no_operating_income_line"
        r[ok & r.isna() & p["revenue_q"].isna()] = "no_quarter_revenue"
        r[ok & r.isna()] = "no_prior_year_comparative_or_nonpositive_revenue"
    elif feat in ("leverage", "log_assets"):
        r[ok] = "no_assets_or_liabilities"
    elif feat == "abs_accruals":
        r[ok & p["net_income_ttm"].isna()] = "no_ttm_net_income"
        r[ok & r.isna()] = "no_ttm_cfo"
    elif feat.startswith(("wc", "abs_wc")):
        r[ok & p["wc_to_assets"].isna()] = "no_classified_balance_sheet"
        r[ok & r.isna()] = "no_filing_a_year_earlier"
    else:
        r[ok] = "not_reported"
    return r


def coverage() -> dict:
    p = panel()
    OUT.mkdir(parents=True, exist_ok=True)
    pre = p[p["year"].lt(2026)]
    out = {"population": len(p), "population_2014_2025": len(pre),
           "test_years_2017_2025": int(p["year"].between(2017, 2025).sum())}
    cov = pd.DataFrame({"all_2014_2026": p[FEATURES].notna().mean(),
                        "test_2017_2025": p.loc[p["year"].between(2017, 2025), FEATURES].notna().mean()})
    cov.to_csv(OUT / "coverage_feature.csv")
    p.groupby("year")[FEATURES].apply(lambda d: d.notna().mean()).to_csv(OUT / "coverage_year.csv")
    p.groupby("announce_window")[FEATURES].apply(lambda d: d.notna().mean()).to_csv(OUT / "coverage_window.csv")
    p.groupby("sector")[FEATURES].apply(lambda d: d.notna().mean()).to_csv(OUT / "coverage_sector.csv")
    reasons = pd.concat({f: missing_reason(p, f).value_counts() for f in FEATURES}, axis=1)
    reasons.to_csv(OUT / "missing_reasons.csv")
    ok = p[p["status"].eq("ok")]
    age = ok[["info_age_days", "period_age_days"]].describe(percentiles=[.1, .5, .9]).T
    age.to_csv(OUT / "information_age.csv")
    qps = ok.groupby("stock")["n_quarters_available"].median()
    routes = {"revenue_q_route": ok["revenue_q_route"].value_counts(dropna=False).to_dict(),
              "leverage_route": ok["leverage_route"].value_counts(dropna=False).to_dict(),
              "selected_form": ok["selected_form"].value_counts().to_dict()}
    qr = pd.read_parquet(OUT / "quarter_records.parquet")
    tags = qr.loc[qr["accession"].isin(ok["selected_accession"]), "revenue_q_tag"]
    routes["revenue_tag_at_selected_filing"] = tags.value_counts(dropna=False).to_dict()
    out.update(coverage=cov.round(4).to_dict(), info_age_days_median=float(age.loc["info_age_days", "50%"]),
               period_age_days_median=float(age.loc["period_age_days", "50%"]),
               quarters_per_stock_median=float(qps.median()), quarters_per_stock_p10=float(qps.quantile(.1)),
               routes=routes, status=p["status"].fillna("not_in_sec_population").value_counts().to_dict())
    (OUT / "coverage.json").write_text(json.dumps(out, indent=1, default=str))
    return out


# ───────────────────────────────── verification sample ──────────────────────────────────
def sample(p: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified: 4 year buckets x selected form (10-Q / 10-K) x financial-or-real-estate
    vs other; plus forced cases (derived Q4, liabilities by identity, predecessor CIK)."""
    ok = p[p["status"].eq("ok")].copy()
    ok["bucket"] = pd.cut(ok["year"], [2013, 2016, 2019, 2022, 2026], labels=["14-16", "17-19", "20-22", "23-26"])
    ok["fin"] = ok["sector"].isin(["Financials", "Real Estate"])
    ok["annual"] = ok["selected_form"].isin(build.ANNUAL)
    rng = np.random.default_rng(seed)
    forced = []
    seg = pd.read_csv(pilot_paths.OUT / "cik_segments.csv")
    chain = seg.groupby("stock")["cik"].nunique()
    chain = chain[chain > 1].index
    for mask in [ok["revenue_q_route"].eq("fy_minus_9m"),
                 ok["leverage_route"].eq("LiabilitiesAndStockholdersEquity-equity"),
                 ok["stock"].isin(chain) & ok["year"].le(2018)]:
        c = ok[mask & ~ok["event_id"].isin(forced)]
        forced += list(rng.choice(c["event_id"], size=min(3, len(c)), replace=False))
    rest = ok[~ok["event_id"].isin(forced)]
    strata = list(rest.groupby(["bucket", "annual", "fin"], observed=True))
    per = max(1, (n - len(forced)) // len(strata))
    picks = []
    for _, g in strata:
        picks += list(rng.choice(g["event_id"], size=min(per, len(g)), replace=False))
    left = n - len(forced) - len(picks)
    if left > 0:
        picks += list(rng.choice(rest.loc[~rest["event_id"].isin(picks), "event_id"], size=left, replace=False))
    return ok[ok["event_id"].isin(forced + picks)].reset_index(drop=True)


_DATA_TABLE = 'summary="Data Files"'


def instance_filename(index_page: bytes) -> str | None:
    """The XBRL instance listed in the index page's Data Files table: EX-101.INS (classic
    XBRL) or the EXTRACTED XBRL INSTANCE (inline XBRL, 2019 on)."""
    t = index_page.decode("utf-8", errors="replace")
    s = t.find(_DATA_TABLE)
    if s < 0:
        return None
    e = t.find("</table>", s)
    for row in documents._ROW.findall(t[s:e]):
        cells = documents._CELL.findall(row)
        if len(cells) < 4:
            continue
        typ = documents._TAG.sub("", cells[3]).strip()
        desc = documents._TAG.sub("", cells[1]).strip().upper()
        href = documents._HREF.search(cells[2])
        if href and (typ == "EX-101.INS" or "INSTANCE" in desc):
            return href.group(1).rsplit("/", 1)[-1]
    return None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_instance(b: bytes) -> tuple[pd.DataFrame, dict]:
    """(facts without dimensions: tag, prefix-namespace, start, end, instant, unit, value;
    dei values). Independent of Company Facts."""
    root = etree.fromstring(b, parser=etree.XMLParser(huge_tree=True, recover=True))
    ctx = {}
    for c in root.iter():
        if not isinstance(c.tag, str) or _local(c.tag) != "context":
            continue
        seg = any(_local(x.tag) in ("segment", "scenario") for x in c.iter() if isinstance(x.tag, str))
        per = {(_local(x.tag)): (x.text or "").strip() for x in c.iter()
               if isinstance(x.tag, str) and _local(x.tag) in ("startDate", "endDate", "instant")}
        ctx[c.get("id")] = (seg, per.get("startDate"), per.get("endDate") or per.get("instant"),
                            "instant" in per)
    units = {}
    for u in root.iter():
        if isinstance(u.tag, str) and _local(u.tag) == "unit":
            measures = [(m.text or "").split(":")[-1] for m in u.iter()
                        if isinstance(m.tag, str) and _local(m.tag) == "measure"]
            units[u.get("id")] = "/".join(measures)
    rows, dei = [], {}
    for el in root:
        if not isinstance(el.tag, str) or el.get("contextRef") is None:
            continue
        ns = el.tag[1:].split("}")[0] if el.tag.startswith("{") else ""
        name = _local(el.tag)
        if "/dei/" in ns:
            # The filer's own dei facts sit in an undimensioned context; a combined filing
            # (REIT + operating partnership, utility + subsidiaries) adds the others' under
            # LegalEntityAxis.
            if not ctx.get(el.get("contextRef"), (True,))[0]:
                dei[name] = (el.text or "").strip()
            continue
        if "us-gaap" not in ns or el.get("{http://www.w3.org/2001/XMLSchema-instance}nil") == "true":
            continue
        seg, start, end, inst = ctx.get(el.get("contextRef"), (True, None, None, False))
        if seg:
            continue
        try:
            v = float((el.text or "").strip())
        except ValueError:
            continue
        rows.append((name, start, end, inst, units.get(el.get("unitRef")), v))
    f = pd.DataFrame(rows, columns=["tag", "start", "end", "instant", "unit", "val"])
    f["start"] = pd.to_datetime(f["start"])
    f["end"] = pd.to_datetime(f["end"])
    f["dur"] = (f["end"] - f["start"]).dt.days
    return f.drop_duplicates(), dei


def _one(f: pd.DataFrame, tag, end, dur=None, instant=False):
    """Value of `tag` at `end` (±END_TOL): an instant, or a duration whose length is in
    `dur`. NaN if absent or ambiguous."""
    if not isinstance(tag, str):
        return np.nan
    r = f[f["tag"].eq(tag) & f["unit"].eq("USD") & (f["end"] - end).abs().dt.days.le(build.END_TOL)]
    r = r[r["instant"]] if instant else r[~r["instant"] & r["dur"].between(*dur)]
    v = r["val"].unique()
    return float(v[0]) if len(v) == 1 else np.nan


class Verifier:
    def __init__(self, archive: acquire.Archive):
        self.ar = archive
        self.cache: dict[str, tuple] = {}

    def filing(self, cik: int, acc: str):
        if acc not in self.cache:
            page, _ = self.ar.get(cik, acc, f"{acc}-index.htm")
            idx = documents.parse_index(page) if page else {}
            fn = instance_filename(page) if page else None
            b = self.ar.get(cik, acc, fn)[0] if fn else None
            facts, dei = parse_instance(b) if b else (pd.DataFrame(), {})
            self.cache[acc] = (idx, fn, facts, dei)
        return self.cache[acc]


Y = build.YEAR_DAYS
Q = build.QUARTER_DAYS
N = build.NINE_MONTH_DAYS


def _prior_end(P):
    return P - pd.Timedelta(days=364)


def verify_record(v: Verifier, rec: pd.Series, names: list[str], checks: list, event_id: str,
                  role: str) -> dict:
    """Re-derive one quarter record's numbers from its own instance document."""
    idx, fn, f, dei = v.filing(int(rec["cik"]), rec["accession"])
    P = rec["period_end"]
    base = {"event_id": event_id, "role": role, "accession": rec["accession"], "instance": fn}
    reg = dei.get("EntityRegistrantName", "")
    company_ok = any(mapping.names_agree(reg, n) for n in names) if reg else False
    checks.append({**base, "item": "registrant", "ours": "|".join(names[:1]), "filing": reg,
                   "ok": company_ok})
    checks.append({**base, "item": "document_type", "ours": rec["form"],
                   "filing": dei.get("DocumentType"), "ok": dei.get("DocumentType") == rec["form"]})
    dped = pd.to_datetime(dei.get("DocumentPeriodEndDate"), errors="coerce")
    checks.append({**base, "item": "period_end", "ours": str(P.date()), "filing": str(dped)[:10],
                   "ok": pd.notna(dped) and abs((dped - P).days) <= build.END_TOL})
    checks.append({**base, "item": "index_filing_date", "ours": str(rec["filing_date"].date()),
                   "filing": idx.get("filing_date"),
                   "ok": idx.get("filing_date") is not None
                   and pd.Timestamp(idx["filing_date"]) <= rec["filing_date"]})
    got = {}
    pe = _prior_end(P)
    spec = {"assets": (rec.get("assets_tag"), P, None, True),
            "current_assets": (rec.get("current_assets_tag"), P, None, True),
            "current_liabilities": (rec.get("current_liabilities_tag"), P, None, True),
            "inventory": (rec.get("inventory_tag"), P, None, True)}
    if rec.get("liabilities_route") == "Liabilities":
        spec["liabilities"] = ("Liabilities", P, None, True)
    else:
        spec["liabilities_and_equity"] = (rec.get("liabilities_and_equity_tag"), P, None, True)
        spec["equity"] = (rec.get("equity_tag"), P, None, True)
    if rec.get("revenue_q_route") == "own_quarter":
        spec["revenue_q"] = (rec["revenue_q_tag"], P, Q, False)
        spec["revenue_q_prior"] = (rec["revenue_q_tag"], pe, Q, False)
    if rec.get("operating_income_q_route") == "own_quarter":
        spec["operating_income_q"] = (rec["operating_income_q_tag"], P, Q, False)
        spec["operating_income_q_prior"] = (rec["operating_income_q_tag"], pe, Q, False)
    for k in ("revenue", "operating_income"):
        if rec["form"] in build.ANNUAL and isinstance(rec.get(f"{k}_fy_tag"), str):
            spec[f"{k}_fy"] = (rec[f"{k}_fy_tag"], P, Y, False)
            spec[f"{k}_fy_prior"] = (rec[f"{k}_fy_tag"], pe, Y, False)
        if rec["form"] not in build.ANNUAL and isinstance(rec.get(f"{k}_9m_tag"), str):
            spec[f"{k}_9m"] = (rec[f"{k}_9m_tag"], P, N, False)
            spec[f"{k}_9m_prior"] = (rec[f"{k}_9m_tag"], pe, N, False)
    for k in ("net_income", "cfo"):
        if isinstance(rec.get(f"{k}_ytd_tag"), str) and np.isfinite(rec.get(f"{k}_ytd_dur", np.nan)):
            d = rec[f"{k}_ytd_dur"]
            spec[f"{k}_ytd"] = (rec[f"{k}_ytd_tag"], P, (d - 1, d + 1), False)
            spec[f"{k}_ytd_prior"] = (rec[f"{k}_ytd_tag"], pe, (d - build.START_TOL, d + build.START_TOL), False)
    for item, (tag, end, dur, inst) in spec.items():
        val = _one(f, tag, end, dur, inst)
        ours = rec.get(item, np.nan)
        if item == "liabilities_and_equity" or item == "equity":
            ours = rec.get(item, np.nan)
        same = (np.isnan(val) and (ours is None or pd.isna(ours))) or \
               (np.isfinite(val) and pd.notna(ours) and abs(val - float(ours)) <= 0.5)
        checks.append({**base, "item": item, "tag": tag, "ours": ours, "filing": val, "ok": bool(same)})
        got[item] = val
    return got


def _margin(oi, rev):
    return build._margin(oi, rev)


def verify(n: int, seed: int) -> pd.DataFrame:
    p = panel()
    s = sample(p, n, seed)
    qr = pd.read_parquet(OUT / "quarter_records.parquet")
    ev = build.events().set_index("event_id")
    seg = pd.read_csv(pilot_paths.OUT / "cik_segments.csv")
    snap = acquire.latest_snapshot()
    subs = {}
    v = Verifier(acquire.Archive(acquire.SECClient(min_interval=0.125)))
    checks, ev_rows = [], []
    for e in s.itertuples():
        names = []
        for cik in seg.loc[seg["stock"].eq(e.stock), "cik"]:
            if cik not in subs:
                subs[cik] = acquire.read_snapshot_file(snap / "submissions", f"CIK{int(cik):010d}.json") or {}
            names += mapping.sec_names(subs[cik])
        g = qr[qr["stock"].eq(e.stock)].sort_values(["period_end", "filing_date"])
        S = g[g["accession"].eq(e.selected_accession)].iloc[0]
        h = g[(g["filing_date"] <= S["filing_date"]) & (g["period_end"] <= S["period_end"])
              & (g["accession"].eq(S["accession"]) | g["period_end"].ne(S["period_end"]))]
        h = h.drop_duplicates("period_end", keep="first")
        trail = h[(S["period_end"] - h["period_end"]).dt.days.le(build.TRAIL_DAYS)].tail(build.TRAIL_QUARTERS)
        got = {}
        for _, rec in trail.iterrows():
            role = "selected" if rec["accession"] == S["accession"] else "trailing"
            gv = verify_record(v, rec, names, checks, e.event_id, role)
            for k in ("revenue", "operating_income"):          # derived Q4 from its parts
                if rec.get(f"{k}_q_route") == "fy_minus_9m":
                    q3 = qr[qr["accession"].eq(rec[f"{k}_q_q4_source_accession"])].iloc[0]
                    g3 = verify_record(v, q3, names, checks, e.event_id, f"q3_for_{k}_q4")
                    gv[f"{k}_q"] = gv.get(f"{k}_fy", np.nan) - g3.get(f"{k}_9m", np.nan)
                    gv[f"{k}_q_prior"] = gv.get(f"{k}_fy_prior", np.nan) - g3.get(f"{k}_9m_prior", np.nan)
            got[rec["accession"]] = gv
        gS = got[S["accession"]]
        # Recompute every feature from verified numbers.
        A = gS.get("assets", np.nan)
        L = gS["liabilities"] if "liabilities" in gS else gS.get("liabilities_and_equity", np.nan) - gS.get("equity", np.nan)
        wc = (gS.get("current_assets", np.nan) - gS.get("current_liabilities", np.nan)) / A
        rec_feats = {"leverage": L / A, "wc_to_assets": wc, "log_assets": np.log(A),
                     "inventory_to_assets": gS.get("inventory", np.nan) / A}
        yoy = ev.loc[e.event_id]
        fe = p[p["event_id"].eq(e.event_id)].iloc[0]
        if isinstance(fe["yoy_accession"], str):
            y = qr[qr["accession"].eq(fe["yoy_accession"])].iloc[0]
            gy = verify_record(v, y, names, checks, e.event_id, "year_earlier_balance_sheet")
            wcy = (gy.get("current_assets", np.nan) - gy.get("current_liabilities", np.nan)) / gy.get("assets", np.nan)
            rec_feats["abs_wc_change_yoy"] = abs(wc - wcy)
        gr, dm = [], []
        for acc, gv in got.items():
            a, b = gv.get("revenue_q", np.nan), gv.get("revenue_q_prior", np.nan)
            if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
                gr.append(np.log(a / b))
            m1 = _margin(gv.get("operating_income_q", np.nan), a)
            m0 = _margin(gv.get("operating_income_q_prior", np.nan), b)
            if np.isfinite(m1) and np.isfinite(m0):
                dm.append(m1 - m0)
        rec_feats["revenue_growth_volatility"] = np.std(gr, ddof=1) if len(gr) >= build.MIN_TRAIL else np.nan
        rec_feats["operating_margin_volatility"] = np.std(dm, ddof=1) if len(dm) >= build.MIN_TRAIL else np.nan
        m1 = _margin(gS.get("operating_income_q", np.nan), gS.get("revenue_q", np.nan))
        m0 = _margin(gS.get("operating_income_q_prior", np.nan), gS.get("revenue_q_prior", np.nan))
        rec_feats["abs_operating_margin_change_yoy"] = abs(m1 - m0)
        # TTM accruals.
        if S["form"] in build.ANNUAL:
            ni, cfo = gS.get("net_income_ytd", np.nan), gS.get("cfo_ytd", np.nan)
        else:
            src = fe["ttm_prior_10k_accession"]
            if isinstance(src, str):
                k10 = qr[qr["accession"].eq(src)].iloc[0]
                gk = verify_record(v, k10, names, checks, e.event_id, "prior_10k_for_ttm")
                for k in ("net_income", "cfo"):     # same rule as build: one tag across parts
                    if k10[f"{k}_ytd_tag"] != S[f"{k}_ytd_tag"]:
                        gk[f"{k}_ytd"] = np.nan
                ni = gS.get("net_income_ytd", np.nan) + gk.get("net_income_ytd", np.nan) - gS.get("net_income_ytd_prior", np.nan)
                cfo = gS.get("cfo_ytd", np.nan) + gk.get("cfo_ytd", np.nan) - gS.get("cfo_ytd_prior", np.nan)
            else:
                ni = cfo = np.nan
        rec_feats["abs_accruals"] = abs(ni - cfo) / A
        cutoff_ok = bool(fe["known_at"] < yoy["call_cutoff_date"]) or bool(
            fe["known_at"] == yoy["call_cutoff_date"])
        for feat in FEATURES:
            if feat == "abs_inventory_change_yoy":
                continue
            ours, theirs = fe[feat], rec_feats.get(feat, np.nan)
            same = (pd.isna(ours) and pd.isna(theirs)) or (
                pd.notna(ours) and pd.notna(theirs) and abs(ours - theirs) <= 1e-6 * max(1, abs(theirs)))
            ev_rows.append({"event_id": e.event_id, "stock": e.stock, "year": e.year,
                            "sector": e.sector, "selected_form": e.selected_form,
                            "selected_accession": e.selected_accession,
                            "period_end": fe["period_end"], "known_at": fe["known_at"],
                            "call_cutoff_date": yoy["call_cutoff_date"], "cutoff_ok": cutoff_ok,
                            "feature": feat, "ours": ours, "recomputed_from_filings": theirs,
                            "ok": bool(same)})
    c = pd.DataFrame(checks)
    er = pd.DataFrame(ev_rows)
    c.to_csv(OUT / "verify_rows.csv", index=False)
    er.to_csv(OUT / "verify_events.csv", index=False)
    return er


COVERAGE_MIN = 0.60       # share of the 2017-2025 population (PREREGISTRATION.md §3)
VERIFY_MIN = 0.90         # share of present, verified feature values that agree
MODEL_FEATURES = ["revenue_growth_volatility", "operating_margin_volatility",
                  "abs_operating_margin_change_yoy", "leverage", "abs_accruals",
                  "wc_to_assets", "abs_wc_change_yoy", "log_assets"]


def write_gates() -> dict:
    """Pass/fail per feature from the coverage table and the filing verification."""
    cov = pd.read_csv(OUT / "coverage_feature.csv", index_col=0)["test_2017_2025"]
    er = pd.read_csv(OUT / "verify_events.csv")
    present = er[er["ours"].notna() | er["recomputed_from_filings"].notna()]
    acc = present.groupby("feature")["ok"].mean()
    n = present.groupby("feature").size()
    g = {f: {"coverage_2017_2025": float(cov[f]), "verified_present": int(n.get(f, 0)),
             "verified_agree": float(acc.get(f, np.nan)),
             "pass": bool(cov[f] >= COVERAGE_MIN and n.get(f, 0) > 0 and acc.get(f, 0) >= VERIFY_MIN)}
         for f in MODEL_FEATURES}
    vr = pd.read_csv(OUT / "verify_rows.csv")
    out = {"features": g, "verify_value_checks": int(len(vr)), "verify_value_agree": int(vr["ok"].sum()),
           "verify_filings": int(vr["accession"].nunique()),
           "verify_events": int(er["event_id"].nunique()),
           "cutoff_ok_events": int(er.drop_duplicates("event_id")["cutoff_ok"].sum())}
    (OUT / "gates.json").write_text(json.dumps(out, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    if a.coverage:
        print(json.dumps(coverage(), indent=1, default=str))
    if a.verify:
        er = verify(a.n, a.seed)
        print(er.groupby("feature")["ok"].agg(["sum", "count"]))
    if a.gates:
        print(json.dumps(write_gates(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
