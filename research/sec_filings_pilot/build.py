"""Build the filing metadata index and map filings to every Breakwater event.

Research-only. Reads a sealed SEC snapshot (`data/vendor/sec/snapshots/`), the Phase 3 feature
frame's clock columns (no outcome column is read: `pd.read_parquet(columns=...)` names
only identity and clock fields), the Breakwater universe CSVs and the session grid. Writes
only `output/sec_filings_pilot/`.

    PYTHONPATH=. .venv/bin/python -m research.sec_filings_pilot.build

Outputs
-------
    filings_index.parquet      every 10-K/10-Q/8-K (+ amendments, + transition forms) of
                               every mapped CIK, one row per (stock, accession)
    events.parquet             the event population with its identity status, cutoff, and
                               the selected periodic filing
    event_8k.parquet           one row per (event, intervening 8-K)
    release_8k.parquet         earnings-release 8-Ks around each event's announcement (for
                               the leakage audit and identity confirmation)
    mapping_audit.csv, cik_segments.csv
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.sec_filings_pilot import acquire, mapping, paths, timing

POP_YEARS = (2014, 2026)
PERIODIC = {"10-K", "10-Q", "10-KT", "10-QT"}
PERIODIC_AMEND = {f + "/A" for f in PERIODIC}
CURRENT = {"8-K"}
CURRENT_AMEND = {"8-K/A"}
# Foreign private issuers file these instead (NXP until 2019, CRH until 2023). Indexed so
# their eras can be counted; never selected as a 10-K/10-Q/8-K.
FOREIGN = {"20-F", "40-F", "6-K", "20-F/A", "40-F/A", "6-K/A"}
KEEP_FORMS = PERIODIC | PERIODIC_AMEND | CURRENT | CURRENT_AMEND | FOREIGN
# Clock and identity columns only. Nothing outcome-derived is read.
EVENT_COLS = ["stock", "sector", "sub_sector", "earnings_date", "is_pending", "announce_window",
              "phase3_announce_date", "announce_ts_source", "report_date", "call_monday",
              "call_cutoff_idx", "call_cutoff_date", "year", "window_ok"]
RELEASE_WINDOW = (-1, 4)       # calendar days around the announcement date
RELEASE_ITEM = "2.02"
SEC_FIELDS = ["accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "form",
              "items", "size", "isXBRL", "isInlineXBRL", "primaryDocument",
              "primaryDocDescription", "fileNumber"]


# ─────────────────────────────────────── loading ────────────────────────────────────────
def load_snapshot(snapshot_dir: Path) -> tuple[dict, dict[int, dict], pd.DataFrame]:
    """(tickers json, {cik: main submissions json}, all filings of every CIK, all forms)."""
    tickers = acquire.read_snapshot_file(snapshot_dir, "company_tickers.json")
    subs, frames = {}, []
    sub_dir = snapshot_dir / "submissions"
    for p in sorted(sub_dir.glob("CIK??????????.json.gz")):
        main = json.loads(gzip.decompress(p.read_bytes()))
        cik = int(main["cik"])
        subs[cik] = main
        pages = [("recent", main["filings"]["recent"])]
        for extra in main["filings"].get("files", []):
            pages.append((extra["name"], acquire.read_snapshot_file(sub_dir, extra["name"])))
        for name, cols in pages:
            if cols is None:
                raise FileNotFoundError(f"snapshot is missing {name}")
            d = pd.DataFrame({k: cols.get(k, [None] * len(cols["accessionNumber"]))
                              for k in SEC_FIELDS})
            d["cik"] = cik
            d["source_page"] = name
            frames.append(d)
    return tickers, subs, pd.concat(frames, ignore_index=True)


def tidy_filings(raw: pd.DataFrame, snapshot_id: str) -> tuple[pd.DataFrame, dict]:
    """Typed filing table for the forms of interest, deduplicated per (cik, accession)."""
    stats = {"rows_all_forms": len(raw)}
    first_any = raw.groupby("cik")["filingDate"].min()
    f = raw[raw["form"].isin(KEEP_FORMS)].copy()
    stats["rows_kept_forms"] = len(f)
    dup = f.duplicated(["cik", "accessionNumber"], keep="first")
    # A duplicate must be the same record; anything else would be an SEC data conflict.
    conflict = f[f.duplicated(["cik", "accessionNumber"], keep=False)].groupby(
        ["cik", "accessionNumber"])[["form", "filingDate"]].nunique().gt(1).any(axis=1)
    stats["duplicate_accession_rows_within_cik"] = int(dup.sum())
    stats["duplicate_accession_conflicts"] = int(conflict.sum())
    f = f[~dup].copy()
    f["filing_date"] = pd.to_datetime(f["filingDate"])
    f["report_period"] = pd.to_datetime(f["reportDate"].replace("", None))
    f["acceptance_json_raw"] = f["acceptanceDateTime"]
    f["acceptance_json_as_utc_ny"] = f["acceptanceDateTime"].map(
        timing.parse_json_acceptance_as_utc)
    f["is_amendment"] = f["form"].str.endswith("/A")
    f["base_form"] = f["form"].str.removesuffix("/A")
    f["items"] = f["items"].fillna("")
    f["source_url"] = [acquire.archive_url(c, a, d) if d else acquire.index_url(c, a)
                       for c, a, d in zip(f["cik"], f["accessionNumber"], f["primaryDocument"])]
    f["index_url"] = [acquire.index_url(c, a) for c, a in zip(f["cik"], f["accessionNumber"])]
    f["snapshot_id"] = snapshot_id
    f["cik_first_filing_any_form"] = f["cik"].map(pd.to_datetime(first_any))
    f = f.rename(columns={"accessionNumber": "accession", "primaryDocument": "primary_document",
                          "primaryDocDescription": "primary_doc_description"})
    keep = ["cik", "accession", "form", "base_form", "is_amendment", "filing_date",
            "report_period", "acceptance_json_raw", "acceptance_json_as_utc_ny", "items",
            "size", "isXBRL", "isInlineXBRL", "primary_document", "primary_doc_description",
            "fileNumber", "source_url", "index_url", "source_page", "snapshot_id",
            "cik_first_filing_any_form"]
    return f[keep].reset_index(drop=True), stats


def load_events() -> pd.DataFrame:
    """Completed BMO/AMC events 2014-2026 with a call cutoff. No outcome column is read."""
    f = pd.read_parquet(paths.FEATURE_FRAME, columns=EVENT_COLS)
    ev = f[~f["is_pending"].astype(bool) & f["window_ok"] & f["year"].between(*POP_YEARS)
           & f["call_cutoff_date"].notna()].copy()
    ev["event_id"] = ev["stock"] + "|" + pd.to_datetime(ev["earnings_date"]).dt.strftime("%Y-%m-%d")
    assert ev["event_id"].is_unique
    ts = pd.read_parquet(paths.EVENTS_DF, columns=["stock", "earnings_date", "announce_ts_ny"])
    ts["earnings_date"] = pd.to_datetime(ts["earnings_date"])
    ev["earnings_date"] = pd.to_datetime(ev["earnings_date"])
    ev = ev.merge(ts.drop_duplicates(["stock", "earnings_date"]), on=["stock", "earnings_date"],
                  how="left")
    ev["call_cutoff_ts"] = timing.cutoff_instant(ev["call_cutoff_date"])
    assert (ev["call_cutoff_date"] < ev["call_monday"]).all()
    assert (ev["call_cutoff_date"] < ev["report_date"]).all()
    return ev.reset_index(drop=True)


def session_grid() -> np.ndarray:
    d = pd.read_parquet(paths.FULL_DF, columns=["date"])
    return np.sort(pd.unique(pd.to_datetime(d["date"]).to_numpy(dtype="datetime64[ns]")))


# ───────────────────────────────────── stock filings ────────────────────────────────────
def stock_filings(filings: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """Each filing attributed to each stock whose CIK chain covers its filing date.

    A segment [valid_from, valid_to) is applied to the filing date. One accession reaching
    a stock through two CIKs of its chain (a co-registrant filing) is kept once.
    """
    m = segments.merge(filings, on="cik", how="inner")
    m = m[(m["filing_date"] >= m["valid_from"]) & (m["filing_date"] < m["valid_to"])]
    m = m.sort_values(["stock", "filing_date", "accession", "cik"])
    m["dup_in_chain"] = m.duplicated(["stock", "accession"], keep="first")
    m = m[~m["dup_in_chain"]].drop(columns="dup_in_chain")
    m = m.rename(columns={"provenance": "mapping_provenance", "reason": "mapping_reason"})
    return m.reset_index(drop=True)


# ──────────────────────────────────── event identity ────────────────────────────────────
def event_identity(ev: pd.DataFrame, map_audit: pd.DataFrame, segments: pd.DataFrame,
                   sf: pd.DataFrame) -> pd.DataFrame:
    """identity_status per event:

    unmapped            the stock has no confident CIK
    no_segment          no CIK segment of the stock's chain covers the cutoff
    foreign_filer_era   the issuer's latest annual report before the cutoff is a 20-F/40-F:
                        it filed as a foreign private issuer (6-K, not 8-K/10-Q) — mapped,
                        but outside the 10-K/10-Q/8-K family this pilot covers
    pre_first_periodic  the covering CIK had filed no 10-K/10-Q (or 20-F/40-F) by the
                        cutoff. Either a new listing (IPO / spin-off: there is genuinely no
                        prior report) or a predecessor issuer not linked in
                        `mapping.MANUAL`; `audit.py` tells the two apart per stock.
    mapped              otherwise
    """
    st = ev["stock"].map(map_audit.set_index("stock")["status"])
    out = pd.Series("mapped", index=ev.index, dtype=object)
    out[st.ne("confident")] = "unmapped"
    seg_cik = pd.Series(pd.NA, index=ev.index, dtype="Int64")
    annual_like = sf[sf["form"].isin(PERIODIC | {"20-F", "40-F"})]
    for stock, rows in ev[st.eq("confident")].groupby("stock").groups.items():
        sg = segments[segments["stock"].eq(stock)]
        g = annual_like[annual_like["stock"].eq(stock)]
        for i, c in ev.loc[rows, "call_cutoff_date"].items():
            cover = sg[(sg["valid_from"] <= c) & (c < sg["valid_to"])]
            if cover.empty:
                out[i] = "no_segment"
                continue
            seg_cik[i] = int(cover["cik"].iloc[-1])
            prior = g[g["filing_date"] < c]
            if prior.empty:
                out[i] = "pre_first_periodic"
            elif prior.sort_values("filing_date")["form"].iloc[-1] in ("20-F", "40-F"):
                out[i] = "foreign_filer_era"
    res = ev.copy()
    res["identity_status"] = out
    res["cik_at_cutoff"] = seg_cik
    res["stock_mapping_status"] = st
    return res


# ─────────────────────────────────────── selection ──────────────────────────────────────
def select(ev: pd.DataFrame, sf: pd.DataFrame, grid: np.ndarray,
           verified_acceptance: dict[str, pd.Timestamp] | None = None
           ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(per-event periodic selection, event x 8-K rows, event x periodic-amendment rows).

    Only events with identity_status == mapped are selected for; the rest keep NaN.
    `verified_acceptance`: {accession: NY acceptance} from index pages, used ONLY for
    filings dated on the cutoff date itself.
    """
    va = verified_acceptance or {}
    per_rows, k8_rows, am_rows = [], [], []
    mapped = ev[ev["identity_status"].eq("mapped")]
    by_stock = {s: g for s, g in sf.groupby("stock")}
    for stock, evs in mapped.groupby("stock"):
        g = by_stock.get(stock)
        if g is None:
            continue
        fd = g["filing_date"].to_numpy("datetime64[ns]")
        gva = pd.to_datetime(g["accession"].map(va)).to_numpy("datetime64[ns]")
        for e in evs.itertuples():
            ok = timing.eligible(fd, e.call_cutoff_date, e.call_cutoff_ts, gva)
            avail = g[ok]
            per = avail[avail["form"].isin(PERIODIC)]
            sel = None
            if len(per):
                last = per["filing_date"].max()
                cands = per[per["filing_date"].eq(last)].sort_values("accession")
                sel = cands.iloc[-1]
                per_rows.append({"event_id": e.event_id, "periodic_accession": sel["accession"],
                                 "periodic_form": sel["form"], "periodic_cik": sel["cik"],
                                 "periodic_filing_date": sel["filing_date"],
                                 "periodic_report_period": sel["report_period"],
                                 "periodic_same_day_ties": len(cands),
                                 "periodic_timing": timing.EXACT if sel["accession"] in va
                                 and sel["filing_date"] == e.call_cutoff_date else timing.DATE_ONLY})
            lo = sel["filing_date"] if sel is not None else pd.Timestamp.min
            k8 = avail[avail["base_form"].isin(CURRENT)]
            after = k8[k8["filing_date"] > lo]
            same = k8[k8["filing_date"].eq(lo)] if sel is not None else k8.iloc[0:0]
            for r in after.itertuples():
                k8_rows.append({"event_id": e.event_id, "accession": r.accession, "form": r.form,
                                "cik": r.cik, "filing_date": r.filing_date, "items": r.items,
                                "relation": "after_periodic" if sel is not None else "no_periodic"})
            for r in same.itertuples():
                k8_rows.append({"event_id": e.event_id, "accession": r.accession, "form": r.form,
                                "cik": r.cik, "filing_date": r.filing_date, "items": r.items,
                                "relation": "same_day_as_periodic_excluded"})
            am = avail[avail["form"].isin(PERIODIC_AMEND) & (avail["filing_date"] > lo)]
            for r in am.itertuples():
                am_rows.append({"event_id": e.event_id, "accession": r.accession, "form": r.form,
                                "filing_date": r.filing_date})
    per = pd.DataFrame(per_rows)
    k8 = pd.DataFrame(k8_rows, columns=["event_id", "accession", "form", "cik", "filing_date",
                                        "items", "relation"])
    am = pd.DataFrame(am_rows, columns=["event_id", "accession", "form", "filing_date"])
    out = ev.merge(per, on="event_id", how="left")
    out["periodic_age_days"] = (out["call_cutoff_date"] - out["periodic_filing_date"]).dt.days
    fpos = np.searchsorted(grid, out["periodic_filing_date"].to_numpy("datetime64[ns]"), "left")
    out["periodic_age_sessions"] = np.where(out["periodic_filing_date"].notna(),
                                            out["call_cutoff_idx"] - fpos, np.nan)
    return out, k8, am


def release_8ks(ev: pd.DataFrame, sf: pd.DataFrame) -> pd.DataFrame:
    """Item 2.02 8-Ks filed within RELEASE_WINDOW calendar days of each event's announcement
    date — the earnings release being predicted. Used for the leakage audit and to confirm
    the event belongs to the mapped issuer. Never an input to an information set."""
    k = sf[sf["base_form"].eq("8-K") & sf["items"].str.contains(RELEASE_ITEM, regex=False)]
    ann = pd.to_datetime(ev["phase3_announce_date"]).fillna(ev["report_date"])
    e = ev[["event_id", "stock", "call_cutoff_date", "call_cutoff_ts"]].assign(ann=ann.values)
    m = e.merge(k[["stock", "accession", "form", "filing_date", "items", "cik"]], on="stock")
    d = (m["filing_date"] - m["ann"]).dt.days
    m = m[d.between(*RELEASE_WINDOW)].copy()
    m["days_from_announcement"] = d[d.between(*RELEASE_WINDOW)]
    return m.drop(columns="ann").reset_index(drop=True)


# ───────────────────────────────────────── main ─────────────────────────────────────────
def run(snapshot_dir: Path | None = None, verified: dict | None = None) -> dict:
    snapshot_dir = snapshot_dir or acquire.latest_snapshot()
    tickers_json, subs, raw = load_snapshot(snapshot_dir)
    filings, stats = tidy_filings(raw, snapshot_dir.name)
    u = mapping.universe()
    map_audit, segments = mapping.resolve(u, mapping.tickers_frame(tickers_json), subs)
    sf = stock_filings(filings, segments)
    ev = event_identity(load_events(), map_audit, segments, sf)
    grid = session_grid()
    sel, k8, am = select(ev, sf, grid, verified)
    rel = release_8ks(ev, sf)
    paths.OUT.mkdir(parents=True, exist_ok=True)
    sf.to_parquet(paths.OUT / "filings_index.parquet")
    sel.to_parquet(paths.OUT / "events.parquet")
    k8.to_parquet(paths.OUT / "event_8k.parquet")
    am.to_parquet(paths.OUT / "event_periodic_amendments.parquet")
    rel.to_parquet(paths.OUT / "release_8k.parquet")
    map_audit.to_csv(paths.OUT / "mapping_audit.csv", index=False)
    segments.to_csv(paths.OUT / "cik_segments.csv", index=False)
    stats.update(snapshot_id=snapshot_dir.name, n_events=len(sel),
                 n_stock_filings=len(sf), n_ciks=int(filings["cik"].nunique()))
    (paths.OUT / "build_stats.json").write_text(json.dumps(stats, indent=1, default=str))
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=None)
    args = ap.parse_args(argv)
    verified = None
    vp = paths.OUT / "verified_acceptance.parquet"
    if vp.exists():
        v = pd.read_parquet(vp)
        verified = dict(zip(v["accession"], v["acceptance_verified_ny"]))
    print(json.dumps(run(args.snapshot, verified), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
