"""Build every SEC feature for every event — no outcome is read here.

    PYTHONPATH=. .venv/bin/python -m research.sec_features.build --plan     # counts only
    PYTHONPATH=. .venv/bin/python -m research.sec_features.build --fetch    # SEC downloads
    PYTHONPATH=. .venv/bin/python -m research.sec_features.build            # features

Inputs: the current frame's identity and clock columns (`EVENT_COLS`), the SEC pilot's
outputs (filings index, information sets, verified cutoff-day times) and its fetch-once
archive. Output: `output/sec_features/sec_features.parquet` (one row per pilot event) and
`releases.parquet` (one row per release exhibit). Rules: PREREGISTRATION.md.
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from research.sec_features import features as F
from research.sec_filings_pilot import acquire, documents, paths as sp, timing

log = logging.getLogger(__name__)
OUT = Path("output/sec_features")
FRAME = OUT / "feature_frame_current.parquet"
EVENT_COLS = ["stock", "earnings_date", "is_pending", "announce_window", "phase3_announce_date",
              "report_date", "call_cutoff_date", "year"]
RELEASE_WINDOW = (-1, 4)
MAX_RELEASE_AGE_DAYS = 140
MIN_PROSE_WORDS = 100
FETCH_INTERVAL_S = 0.125          # 8 request starts per second across all threads (SEC: 10)
WORKERS = 6
PARSE_WORKERS = 8


# ─────────────────────────────────────────── inputs ─────────────────────────────────────
def frame_events() -> pd.DataFrame:
    f = pd.read_parquet(FRAME, columns=EVENT_COLS)
    f = f[~f["is_pending"].astype(bool)].copy()
    f["event_id"] = f["stock"] + "|" + pd.to_datetime(f["earnings_date"]).dt.strftime("%Y-%m-%d")
    f["ann_date"] = pd.to_datetime(f["phase3_announce_date"]).fillna(f["report_date"]).dt.normalize()
    return f.reset_index(drop=True)


def pilot() -> dict:
    v = pd.read_parquet(sp.OUT / "verified_acceptance.parquet")
    return {"sf": pd.read_parquet(sp.OUT / "filings_index.parquet"),
            "ev": pd.read_parquet(sp.OUT / "events.parquet"),
            "k8": pd.read_parquet(sp.OUT / "event_8k.parquet"),
            "verified": dict(zip(v["accession"], v["acceptance_verified_ny"]))}


# ───────────────────────────────────────── releases ─────────────────────────────────────
def release_8k_per_event(fe: pd.DataFrame, sf: pd.DataFrame) -> pd.DataFrame:
    """Each completed frame event's own release 8-K (any window/year), by the §1.2 rule."""
    k = sf[sf["form"].eq("8-K") & sf["items"].str.split(",").map(lambda x: "2.02" in x)]
    m = fe[["event_id", "stock", "ann_date"]].merge(
        k[["stock", "cik", "accession", "filing_date", "items"]], on="stock")
    m["days"] = (m["filing_date"] - m["ann_date"]).dt.days
    m = m[m["days"].between(*RELEASE_WINDOW)]
    m = m.assign(absd=m["days"].abs()).sort_values(["event_id", "absd", "filing_date", "accession"])
    return m.drop_duplicates("event_id").drop(columns="absd")


def previous_releases(fe: pd.DataFrame, rel: pd.DataFrame, target: pd.DataFrame,
                      verified: dict) -> pd.DataFrame:
    """For every target event: previous and second-prior release, with usability checks."""
    fe = fe.sort_values(["stock", "report_date", "earnings_date"])
    fe["prev_event"] = fe.groupby("stock")["event_id"].shift(1)
    fe["prev2_event"] = fe.groupby("stock")["event_id"].shift(2)
    t = target[["event_id", "stock", "call_cutoff_date", "call_cutoff_ts"]].merge(
        fe[["event_id", "prev_event", "prev2_event"]], on="event_id", how="left")
    r = rel.set_index("event_id")
    out = t.copy()
    for tag, col in (("prev", "prev_event"), ("prev2", "prev2_event")):
        j = t[[col]].join(r[["accession", "cik", "filing_date", "items"]], on=col)
        out[f"{tag}_accession"] = j["accession"].to_numpy()
        out[f"{tag}_cik"] = j["cik"].to_numpy()
        out[f"{tag}_filing_date"] = pd.to_datetime(j["filing_date"]).to_numpy()
        ok_elig = timing.eligible(out[f"{tag}_filing_date"], out["call_cutoff_date"],
                                  out["call_cutoff_ts"],
                                  out[f"{tag}_accession"].map(verified))
        age = (out["call_cutoff_date"] - out[f"{tag}_filing_date"]).dt.days
        out[f"{tag}_age_days"] = age
        reason = np.select(
            [t[col].isna(), out[f"{tag}_accession"].isna(), ~ok_elig,
             age > MAX_RELEASE_AGE_DAYS * (2 if tag == "prev2" else 1)],
            ["no_previous_event", "no_release_8k", "not_eligible", "too_old"], "ok")
        out[f"{tag}_status"] = reason
    order_bad = out["prev_status"].eq("ok") & out["prev2_status"].eq("ok") & ~(
        out["prev2_filing_date"] < out["prev_filing_date"])
    out.loc[order_bad, "prev2_status"] = "not_before_prev"
    return out


# ────────────────────────────────────────── fetching ────────────────────────────────────
def _exhibit_for(ar: acquire.Archive, cik: int, acc: str) -> dict:
    b, rec = ar.get(cik, acc, f"{acc}-index.htm")
    if b is None:
        return {"accession": acc, "exhibit_status": "fetch_failed"}
    docs = documents.parse_index(b)["documents"]
    ex = [d for d in docs if d["type"].upper().startswith("EX-99")]
    if not ex:
        return {"accession": acc, "exhibit_status": "no_ex99", "n_ex99": 0}
    ex.sort(key=lambda d: (int(d["seq"]) if d["seq"].isdigit() else 10 ** 6))
    d = ex[0]
    body, rec = ar.get(cik, acc, d["filename"])
    return {"accession": acc, "exhibit_status": "ok" if body else "fetch_failed",
            "exhibit_filename": d["filename"], "exhibit_type": d["type"],
            "exhibit_description": d["description"], "n_ex99": len(ex)}


def _index_ex99_count(ar: acquire.Archive, cik: int, acc: str) -> dict:
    b, _ = ar.get(cik, acc, f"{acc}-index.htm")
    if b is None:
        return {"accession": acc, "index_ok": False}
    docs = documents.parse_index(b)["documents"]
    return {"accession": acc, "index_ok": True,
            "n_ex99": sum(d["type"].upper().startswith("EX-99") for d in docs)}


def _parallel(fn, items, label):
    client = acquire.SECClient(min_interval=FETCH_INTERVAL_S)
    ar = acquire.Archive(client)
    out = []
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, r in enumerate(pool.map(lambda x: fn(ar, *x), items)):
            out.append(r)
            if (i + 1) % 1000 == 0:
                log.info("%s %d/%d (requests %d)", label, i + 1, len(items), client.n_requests)
    return pd.DataFrame(out)


# ────────────────────────────────────────── features ────────────────────────────────────
def _release_row(cik: int, acc: str, filename: str) -> tuple[dict, dict]:
    raw = acquire.archive_path(cik, acc, filename).read_bytes()
    html = documents.decode(raw).text
    feats = F.release_features(html)
    _, prose = F.release_text(html)
    return feats, dict(F.term_counts(prose))


def _release_row_star(a):
    return _release_row(*a)


def release_table(exhibits: pd.DataFrame, accs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    ok = exhibits[exhibits["exhibit_status"].eq("ok")].merge(accs, on="accession")
    rows, terms = [], {}
    args = list(zip(ok["cik"], ok["accession"], ok["exhibit_filename"]))
    with ProcessPoolExecutor(PARSE_WORKERS) as pool:
        results = list(pool.map(_release_row_star, args, chunksize=64))
    for (_cik, acc, _fn), (f, tc) in zip(args, results):
        f["accession"] = acc
        f["release_status"] = "ok" if f["prose_words"] >= MIN_PROSE_WORDS else "too_little_prose"
        rows.append(f)
        terms[acc] = tc
    rt = pd.DataFrame(rows).merge(exhibits, on="accession", how="right")
    rt["release_status"] = rt["release_status"].fillna(rt["exhibit_status"])
    return rt, terms


def eightk_table(pv: dict, ev: pd.DataFrame, ex99: pd.DataFrame | None) -> pd.DataFrame:
    sf, k8, verified = pv["sf"], pv["k8"], pv["verified"]
    eightk = sf[sf["base_form"].eq("8-K")]
    by_stock = {s: g for s, g in eightk.groupby("stock")}
    inset = k8[k8["relation"].isin(["after_periodic", "no_periodic"])]
    inset_by = {e: g for e, g in inset.groupby("event_id")}
    exmap = dict(zip(ex99["accession"], ex99["n_ex99"])) if ex99 is not None else {}
    rows = []
    empty = inset.iloc[0:0]
    for e in ev.itertuples():
        g = by_stock.get(e.stock, eightk.iloc[0:0])
        ok = timing.eligible(g["filing_date"].to_numpy("datetime64[ns]"), e.call_cutoff_date,
                             e.call_cutoff_ts,
                             pd.to_datetime(g["accession"].map(verified)).to_numpy("datetime64[ns]"))
        elig = g[ok]
        f = F.eightk_features(inset_by.get(e.event_id, empty), elig, e.call_cutoff_date)
        if exmap:
            days = (e.call_cutoff_date - elig["filing_date"]).dt.days
            l30 = elig[elig["form"].eq("8-K") & days.between(0, F.WINDOW_30D - 1)]
            n = l30["accession"].map(exmap)
            f["num_8k_ex99_last_30d"] = int((n > 0).sum()) if n.notna().all() else np.nan
        f["event_id"] = e.event_id
        rows.append(f)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────── driver ─────────────────────────────────────
def plan(pv: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fe = frame_events()
    ev = pv["ev"][pv["ev"]["identity_status"].eq("mapped")]
    rel = release_8k_per_event(fe, pv["sf"])
    pr = previous_releases(fe, rel, ev, pv["verified"])
    need = pd.concat([pr.loc[pr["prev_status"].eq("ok"), ["prev_cik", "prev_accession"]]
                      .set_axis(["cik", "accession"], axis=1),
                      pr.loc[pr["prev2_status"].eq("ok"), ["prev2_cik", "prev2_accession"]]
                      .set_axis(["cik", "accession"], axis=1)]).drop_duplicates("accession")
    return pr, need, rel


def last30_8ks(pv: dict) -> pd.DataFrame:
    ev = pv["ev"][pv["ev"]["identity_status"].eq("mapped")]
    sf = pv["sf"][pv["sf"]["form"].eq("8-K")]
    m = ev[["event_id", "stock", "call_cutoff_date"]].merge(sf[["stock", "cik", "accession",
                                                                 "filing_date"]], on="stock")
    d = (m["call_cutoff_date"] - m["filing_date"]).dt.days
    return m[d.between(0, F.WINDOW_30D - 1)].drop_duplicates("accession")[["cik", "accession"]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--fetch-ex99", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    OUT.mkdir(parents=True, exist_ok=True)
    pv = pilot()
    pr, need, rel = plan(pv)
    if args.plan:
        print({"events": len(pr), "prev_status": pr["prev_status"].value_counts().to_dict(),
               "prev2_status": pr["prev2_status"].value_counts().to_dict(),
               "release_accessions_to_fetch": len(need),
               "last30_8k_index_pages": len(last30_8ks(pv))})
        return 0
    if args.fetch:
        ex = _parallel(_exhibit_for, list(need.itertuples(index=False, name=None)), "releases")
        ex.to_parquet(OUT / "release_exhibits.parquet")
    if args.fetch_ex99:
        items = list(last30_8ks(pv).itertuples(index=False, name=None))
        _parallel(_index_ex99_count, items, "8-K indexes").to_parquet(OUT / "ex99_last30.parquet")
    if args.fetch or args.fetch_ex99:
        return 0
    ex = pd.read_parquet(OUT / "release_exhibits.parquet")
    rt, terms = release_table(ex, need)
    rt.to_parquet(OUT / "releases.parquet")
    pairs = pr.merge(rt.add_prefix("prev_"), on="prev_accession", how="left").merge(
        rt[["accession", "release_status", "uncertainty_rate"]].add_prefix("prev2_"),
        on="prev2_accession", how="left")
    pairs["prev_ok"] = pairs["prev_status"].eq("ok") & pairs["prev_release_status"].eq("ok")
    pairs["prev2_ok"] = pairs["prev2_status"].eq("ok") & pairs["prev2_release_status"].eq("ok")
    pairs["prev_missing_reason"] = np.where(pairs["prev_status"].ne("ok"), pairs["prev_status"],
                                            pairs["prev_release_status"].fillna("not_fetched"))
    pairs["prev2_missing_reason"] = np.where(pairs["prev2_status"].ne("ok"), pairs["prev2_status"],
                                             pairs["prev2_release_status"].fillna("not_fetched"))
    both = pairs["prev_ok"] & pairs["prev2_ok"]
    pairs["uncertainty_change"] = np.where(
        both, pairs["prev_uncertainty_rate"] - pairs["prev2_uncertainty_rate"], np.nan)
    pairs["release_text_change"] = [
        F.cosine_distance(F.Counter(terms[a]), F.Counter(terms[b])) if ok else np.nan
        for a, b, ok in zip(pairs["prev_accession"], pairs["prev2_accession"], both)]
    for c in ["guidance_present", "eps_guidance_width", "revenue_guidance_width",
              "uncertainty_rate", "n_guidance_sentences"]:
        pairs[c] = pairs[f"prev_{c}"].where(pairs["prev_ok"])
    ex99p = OUT / "ex99_last30.parquet"
    ev = pv["ev"][pv["ev"]["identity_status"].eq("mapped")]
    k8f = eightk_table(pv, ev, pd.read_parquet(ex99p) if ex99p.exists() else None)
    out = pairs.merge(k8f, on="event_id", how="left")
    out.to_parquet(OUT / "sec_features.parquet")
    print(json.dumps({"events": len(out), "prev_ok": int(pairs["prev_ok"].sum()),
                      "both_ok": int(both.sum())}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
