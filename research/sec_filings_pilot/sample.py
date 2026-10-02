"""Download and inspect a stratified sample of real filings; verify SEC's acceptance clock.

Research-only. Network access goes through `acquire.Archive` (fetch-once, immutable cache).
Writes only `output/sec_filings_pilot/`.

    PYTHONPATH=. .venv/bin/python -m research.sec_filings_pilot.sample --clock    # clock audit
    PYTHONPATH=. .venv/bin/python -m research.sec_filings_pilot.sample --documents
    PYTHONPATH=. .venv/bin/python -m research.sec_filings_pilot.sample --cutoff-day

Three jobs
----------
--clock       index pages of a random sample of 10-K/10-Q/8-K filings across every mapped
              CIK and year: SEC's verified "Accepted" time against the submissions-JSON
              `acceptanceDateTime`, read literally as UTC.
--documents   the stratified document sample: years x form class x company size x sector,
              plus amendments and exhibit-bearing 8-Ks. Primary document + every EX-99
              exhibit (max 4) per filing. Decoding, text extraction, sections, exhibits.
--cutoff-day  index pages for every filing dated ON some event's cutoff date — the only
              filings whose eligibility needs a time of day (timing.py). Their verified
              acceptance goes to `verified_acceptance.parquet`, which `build.py` reads.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re

import numpy as np
import pandas as pd

from research.sec_filings_pilot import acquire, documents, paths, timing

log = logging.getLogger(__name__)
SEED = 20261002
MAX_EXHIBITS = 4
YEAR_BUCKETS = {"2014-2015": (2014, 2015), "2018-2019": (2018, 2019),
                "2022-2023": (2022, 2023), "2025-2026": (2025, 2026)}


def _load():
    sf = pd.read_parquet(paths.OUT / "filings_index.parquet")
    audit = pd.read_csv(paths.OUT / "mapping_audit.csv")
    return sf, audit


def _index(ar: acquire.Archive, cik: int, acc: str) -> dict | None:
    b, rec = ar.get(cik, acc, f"{acc}-index.htm")
    if b is None:
        return None
    out = documents.parse_index(b)
    out["status"] = rec["status"]
    return out


# ───────────────────────────────────────── clock ────────────────────────────────────────
def clock_sample(sf: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    f = sf[sf["form"].isin(["10-K", "10-Q", "8-K"]) & sf["filing_date"].dt.year.between(2014, 2026)]
    f = f.drop_duplicates("accession")
    rng = np.random.default_rng(SEED)
    # One filing per CIK first (spread across filers), then random fill.
    per_cik = f.groupby("cik").sample(1, random_state=SEED)
    picks = per_cik.sample(min(n, len(per_cik)), random_state=SEED)
    return picks.assign(_r=rng.random(len(picks))).sort_values("_r").drop(columns="_r")


def run_clock(ar: acquire.Archive, n: int = 300) -> pd.DataFrame:
    sf, _ = _load()
    rows = []
    for r in clock_sample(sf, n).itertuples():
        idx = _index(ar, r.cik, r.accession)
        acc = timing.parse_verified_acceptance(idx["accepted"]) if idx and idx["accepted"] else pd.NaT
        lit = r.acceptance_json_as_utc_ny
        rows.append({"cik": r.cik, "stock": r.stock, "accession": r.accession, "form": r.form,
                     "filing_date": r.filing_date, "json_raw": r.acceptance_json_raw,
                     "json_literal_utc_ny": lit, "index_accepted_ny": acc,
                     "index_filing_date": idx["filing_date"] if idx else None})
    out = pd.DataFrame(rows)
    naive_json = pd.to_datetime(out["json_raw"].str[:-1])
    out["json_minus_index_h"] = (naive_json - out["index_accepted_ny"]).dt.total_seconds() / 3600
    off = out["index_accepted_ny"].dt.tz_localize(timing.NY_TZ, ambiguous="NaT",
                                                   nonexistent="NaT").map(
        lambda t: t.utcoffset().total_seconds() / -3600 if pd.notna(t) else np.nan)
    out["utc_offset_h"] = off
    d = out["json_minus_index_h"]
    out["json_class"] = np.select(
        [d.round(3).eq(off), d.round(3).eq(2 * off), d.round(3).eq(0)],
        ["true_utc", "double_shifted", "ny_wall_clock"], "other")
    out["index_vs_filing_date_ok"] = _filing_date_consistent(out["index_accepted_ny"],
                                                             out["filing_date"], out["form"])
    out.to_csv(paths.OUT / "clock_audit.csv", index=False)
    return out


def _filing_date_consistent(acc: pd.Series, fd: pd.Series, form: pd.Series) -> pd.Series:
    """Verified acceptance obeys EDGAR's rule: same day if by 17:30 (22:00 for some forms),
    else the next business day."""
    import pandas_market_calendars as mcal
    days = mcal.get_calendar("NYSE").valid_days("2000-01-01", "2030-12-31").tz_localize(None)
    ok = timing.json_acceptance_consistent(acc, fd, pd.DatetimeIndex(days), form)
    return ok


# ─────────────────────────────────────── documents ──────────────────────────────────────
def form_class(r) -> str:
    if r.is_amendment:
        return "amendment"
    if r.base_form == "8-K":
        return "8-K 2.02" if "2.02" in r.items else "8-K other"
    return r.base_form


def document_sample(sf: pd.DataFrame, audit: pd.DataFrame, per_cell: int = 4) -> pd.DataFrame:
    """Stratified: year bucket x form class x size, each cell `per_cell` filings, spread
    across sectors. Size proxy = SEC's ticker-map order (roughly market value): `large` =
    the 100 highest-ranked Breakwater issuers, `small` = everything ranked below the
    universe median plus the stocks that have left the index."""
    f = sf[sf["form"].isin(["10-K", "10-Q", "8-K", "10-K/A", "10-Q/A", "8-K/A"])].copy()
    f = f.drop_duplicates("accession")
    rank = audit.set_index("stock")["sec_rank"]
    r = f["stock"].map(rank)
    med = rank.median()
    big = rank.nsmallest(100).index
    f["size_class"] = np.where(f["stock"].isin(big), "large",
                               np.where(r.isna() | (r > med), "small", "mid"))
    f["form_class"] = [form_class(x) for x in f.itertuples()]
    f["year"] = f["filing_date"].dt.year
    f["year_bucket"] = None
    for name, (a, b) in YEAR_BUCKETS.items():
        f.loc[f["year"].between(a, b), "year_bucket"] = name
    f = f[f["year_bucket"].notna() & f["size_class"].isin(["large", "small"])]
    sector = f["stock"].map(audit.set_index("stock")["sector"])
    f["sector"] = sector
    picks = []
    rng = np.random.default_rng(SEED)
    for _, g in f.groupby(["year_bucket", "form_class", "size_class"]):
        # Round-robin over sectors so no cell is one industry.
        g = g.assign(_r=rng.random(len(g))).sort_values("_r")
        g["_k"] = g.groupby("sector").cumcount()
        picks.append(g.sort_values(["_k", "_r"]).head(per_cell))
    return pd.concat(picks).drop(columns=["_r", "_k"])


def _doc_record(ar, r, doc: dict, role: str) -> dict:
    b, rec = ar.get(r.cik, r.accession, doc["filename"])
    out = {"accession": r.accession, "stock": r.stock, "form": r.form, "role": role,
           "filename": doc["filename"], "doc_type": doc.get("type"),
           "description": doc.get("description"), "status": rec["status"],
           "bytes": len(b) if b else 0}
    if not b:
        return out
    out["gzip_bytes"] = len(gzip.compress(b))
    if not doc["filename"].lower().endswith((".htm", ".html", ".txt", ".xml")):
        out["kind"] = "binary"
        return out
    dec = documents.decode(b)
    st = documents.html_to_text(dec.text)
    out.update(encoding=dec.encoding, declared_charset=dec.declared,
               replacement_chars=dec.replacement_chars, mojibake_hits=dec.mojibake_hits,
               is_html=st.is_html, text_chars=st.chars, table_chars=st.table_chars,
               numeric_table_chars=st.numeric_table_chars, n_tables=st.n_tables,
               prose_blocks=st.prose_blocks, prose_chars=st.prose_chars,
               hidden_chars_dropped=st.hidden_chars_dropped, parse_errors=";".join(st.errors),
               text_gzip_bytes=len(gzip.compress(st.text.encode())))
    if role == "primary":
        out.update(documents.locate_sections(st.text, r.form))
    if role == "exhibit":
        out["earnings_release"] = documents.is_earnings_release(doc, st.text)
        out["mentions_outlook"] = bool(re.search(r"\b(outlook|guidance)\b", st.text, re.I))
    return out


def run_documents(ar: acquire.Archive, per_cell: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    sf, audit = _load()
    smp = document_sample(sf, audit, per_cell)
    frows, drows = [], []
    for i, r in enumerate(smp.itertuples()):
        idx = _index(ar, r.cik, r.accession)
        docs = idx["documents"] if idx else []
        prim = next((d for d in docs if d["filename"] == r.primary_document), None) or \
            next((d for d in docs if d["type"] == r.form), None)
        ex = [d for d in docs if d["type"].upper().startswith("EX-")]
        ex99 = [d for d in ex if d["type"].upper().startswith("EX-99")]
        frow = {"accession": r.accession, "cik": r.cik, "stock": r.stock, "form": r.form,
                "form_class": r.form_class, "year_bucket": r.year_bucket,
                "size_class": r.size_class, "sector": r.sector, "items": r.items,
                "filing_date": r.filing_date, "index_ok": idx is not None,
                "accepted_verified": idx["accepted"] if idx else None,
                "n_documents": len(docs), "n_exhibits": len(ex), "n_ex99": len(ex99),
                "exhibit_types": ";".join(sorted({d["type"] for d in ex})),
                "filing_size_bytes": r.size, "primary_found_in_index": prim is not None}
        if prim:
            drows.append(_doc_record(ar, r, prim, "primary"))
        for d in ex99[:MAX_EXHIBITS]:
            drows.append(_doc_record(ar, r, d, "exhibit"))
        frows.append(frow)
        if (i + 1) % 25 == 0:
            log.info("documents: %d/%d filings", i + 1, len(smp))
    fdf, ddf = pd.DataFrame(frows), pd.DataFrame(drows)
    fdf.to_csv(paths.OUT / "sample_filings.csv", index=False)
    ddf.to_csv(paths.OUT / "sample_documents.csv", index=False)
    return fdf, ddf


# ─────────────────────────────────────── cutoff day ─────────────────────────────────────
def cutoff_day_filings() -> pd.DataFrame:
    """Every (stock, filing) dated on the cutoff date of one of that stock's mapped events."""
    sf = pd.read_parquet(paths.OUT / "filings_index.parquet")
    ev = pd.read_parquet(paths.OUT / "events.parquet",
                         columns=["event_id", "stock", "call_cutoff_date", "identity_status"])
    ev = ev[ev["identity_status"].eq("mapped")]
    f = sf[sf["base_form"].isin(["10-K", "10-Q", "10-KT", "10-QT", "8-K"])]
    m = ev.merge(f, left_on=["stock", "call_cutoff_date"], right_on=["stock", "filing_date"])
    return m


def run_cutoff_day(ar: acquire.Archive) -> pd.DataFrame:
    m = cutoff_day_filings()
    u = m.drop_duplicates("accession")
    rows = []
    for i, r in enumerate(u.itertuples()):
        idx = _index(ar, r.cik, r.accession)
        rows.append({"accession": r.accession, "cik": r.cik,
                     "acceptance_verified_ny": timing.parse_verified_acceptance(idx["accepted"])
                     if idx and idx["accepted"] else pd.NaT})
        if (i + 1) % 200 == 0:
            log.info("cutoff-day index pages %d/%d", i + 1, len(u))
    v = pd.DataFrame(rows)
    vp = paths.OUT / "verified_acceptance.parquet"
    if vp.exists():                      # keep earlier verifications (cached pages anyway)
        v = pd.concat([pd.read_parquet(vp), v]).drop_duplicates("accession", keep="last")
    v.to_parquet(vp)
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clock", action="store_true")
    ap.add_argument("--documents", action="store_true")
    ap.add_argument("--cutoff-day", action="store_true")
    ap.add_argument("--per-cell", type=int, default=4)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ar = acquire.Archive(acquire.SECClient())
    if args.clock:
        out = run_clock(ar)
        print(out["json_class"].value_counts().to_string())
    if args.cutoff_day:
        print(len(run_cutoff_day(ar)), "cutoff-day filings verified")
    if args.documents:
        f, d = run_documents(ar, args.per_cell)
        print(len(f), "filings,", len(d), "documents")
    print("requests:", ar.client.n_requests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
