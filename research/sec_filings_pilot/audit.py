"""Coverage, identity, timing/leakage, retrieval and corpus-size tables for RESULTS.md.

Reads only `output/sec_filings_pilot/` (built by build.py and sample.py) and the sealed SEC
snapshot. No outcome is read; no model, score or feature is computed.

    PYTHONPATH=. .venv/bin/python -m research.sec_filings_pilot.audit
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.sec_filings_pilot import acquire, build, paths, timing

O = paths.OUT
ITEMS_OF_NOTE = ["2.02", "7.01", "8.01", "5.02", "1.01", "2.01", "5.07", "9.01"]
REQ_PER_S = 5          # SEC allows 10; this pilot uses half


def _w(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df.to_csv(O / name, index=True)
    return df


def load():
    ev = pd.read_parquet(O / "events.parquet")
    k8 = pd.read_parquet(O / "event_8k.parquet")
    am = pd.read_parquet(O / "event_periodic_amendments.parquet")
    rel = pd.read_parquet(O / "release_8k.parquet")
    sf = pd.read_parquet(O / "filings_index.parquet")
    ma = pd.read_csv(O / "mapping_audit.csv")
    seg = pd.read_csv(O / "cik_segments.csv")
    return ev, k8, am, rel, sf, ma, seg


# ───────────────────────────────────────── mapping ──────────────────────────────────────
def mapping_tables(ev, ma, seg, rel) -> dict:
    stocks_in_pop = set(ev["stock"])
    m = ma[ma["stock"].isin(stocks_in_pop)].copy()
    chain = seg.groupby("stock")["cik"].nunique()
    m["n_ciks_in_chain"] = m["stock"].map(chain).fillna(0).astype(int)
    m["chain_provenance"] = np.where(m["n_ciks_in_chain"] > 1, "manual_verified_chain",
                                     m["provenance"])
    ev_status = ev.groupby("stock")["identity_status"].agg(lambda s: s.value_counts().to_dict())
    m["event_statuses"] = m["stock"].map(ev_status)
    _w(m.set_index("stock"), "mapping_audit_population.csv")
    shared = seg.groupby("cik")["stock"].unique()
    shared = shared[shared.map(len) > 1]
    ev = ev.assign(confirmed=ev["event_id"].isin(rel["event_id"]))
    mapped = ev[ev["identity_status"].eq("mapped")]
    by_stock = mapped.groupby("stock")["confirmed"].agg(["mean", "size"])
    newlist = ev[ev["identity_status"].eq("pre_first_periodic")]
    summary = {
        "stocks_in_population": len(m),
        "stock_status": m["status"].value_counts().to_dict(),
        "stock_provenance": m["chain_provenance"].value_counts().to_dict(),
        "predecessor_chains": {s: int(n) for s, n in chain[chain > 1].items()},
        "classes_sharing_a_cik": {int(c): list(s) for c, s in shared.items()},
        "event_identity_status": ev["identity_status"].value_counts().to_dict(),
        "pre_first_periodic_stocks": sorted(newlist["stock"].unique()),
        "issuer_confirmation_rate_mapped": round(float(mapped["confirmed"].mean()), 4),
        "issuer_confirmation_by_year": mapped.groupby("year")["confirmed"].mean().round(3).to_dict(),
        "lowest_confirmation_stocks": by_stock.sort_values("mean").head(10).round(3).to_dict("index"),
    }
    return summary


# ───────────────────────────────────────── coverage ─────────────────────────────────────
def periodic_coverage(ev: pd.DataFrame, by: str = "year") -> pd.DataFrame:
    g = ev.groupby(by)
    out = pd.DataFrame({
        "events": g.size(),
        "mapped": g["identity_status"].apply(lambda s: s.eq("mapped").sum()),
        "pct_prior_periodic": g["periodic_accession"].apply(lambda s: s.notna().mean() * 100),
        "pct_prior_10Q": g["periodic_form"].apply(lambda s: s.isin(["10-Q", "10-QT"]).mean() * 100),
        "pct_prior_10K": g["periodic_form"].apply(lambda s: s.isin(["10-K", "10-KT"]).mean() * 100),
        "median_age_days": g["periodic_age_days"].median(),
        "p90_age_days": g["periodic_age_days"].quantile(0.9),
        "median_age_sessions": g["periodic_age_sessions"].median(),
        "missing": g["periodic_accession"].apply(lambda s: s.isna().sum()),
    })
    out.loc["all"] = [len(ev), ev["identity_status"].eq("mapped").sum(),
                      ev["periodic_accession"].notna().mean() * 100,
                      ev["periodic_form"].isin(["10-Q", "10-QT"]).mean() * 100,
                      ev["periodic_form"].isin(["10-K", "10-KT"]).mean() * 100,
                      ev["periodic_age_days"].median(), ev["periodic_age_days"].quantile(0.9),
                      ev["periodic_age_sessions"].median(), ev["periodic_accession"].isna().sum()]
    return out.round(1)


def k8_coverage(ev: pd.DataFrame, k8: pd.DataFrame, by: str = "year") -> pd.DataFrame:
    inset = k8[k8["relation"].isin(["after_periodic", "no_periodic"])]
    n = inset.groupby("event_id").size()
    e = ev[ev["identity_status"].eq("mapped")].copy()
    e["n_8k"] = e["event_id"].map(n).fillna(0)
    e["n_8kA"] = e["event_id"].map(inset[inset["form"].eq("8-K/A")].groupby("event_id").size()).fillna(0)
    e["n_same_day_excluded"] = e["event_id"].map(
        k8[k8["relation"].eq("same_day_as_periodic_excluded")].groupby("event_id").size()).fillna(0)
    for it in ITEMS_OF_NOTE:
        has = inset[inset["items"].str.split(",").map(lambda x, it=it: it in x)]
        e[f"has_{it}"] = e["event_id"].isin(has["event_id"])
    g = e.groupby(by)
    out = pd.DataFrame({"mapped_events": g.size(),
                        "pct_ge1_8k": g["n_8k"].apply(lambda s: (s > 0).mean() * 100),
                        "mean_8k": g["n_8k"].mean(), "median_8k": g["n_8k"].median(),
                        "p90_8k": g["n_8k"].quantile(0.9),
                        "pct_with_8kA": g["n_8kA"].apply(lambda s: (s > 0).mean() * 100),
                        "mean_same_day_excluded": g["n_same_day_excluded"].mean()})
    for it in ITEMS_OF_NOTE:
        out[f"pct_item_{it}"] = g[f"has_{it}"].mean() * 100
    return out.round(2)


def item_distribution(k8: pd.DataFrame) -> pd.DataFrame:
    inset = k8[k8["relation"].isin(["after_periodic", "no_periodic"])].drop_duplicates(
        ["accession"])
    items = inset["items"].str.split(",").explode().replace("", "(none listed)")
    d = items.value_counts().to_frame("unique_8ks_carrying_item")
    d["pct_of_unique_8ks"] = (d.iloc[:, 0] / len(inset) * 100).round(2)
    return d


# ─────────────────────────────────────── timing / leakage ───────────────────────────────
def timing_tables(ev, k8, rel, sf) -> dict:
    out = {}
    clock = pd.read_csv(O / "clock_audit.csv", parse_dates=["filing_date", "index_accepted_ny"])
    out["clock_sample_n"] = len(clock)
    out["json_acceptance_class"] = clock["json_class"].value_counts().to_dict()
    out["json_class_by_year"] = pd.crosstab(clock["filing_date"].dt.year,
                                            clock["json_class"]).to_dict("index")
    out["index_time_consistent_with_filing_date_rule"] = \
        clock["index_vs_filing_date_ok"].value_counts().to_dict()
    out["index_acceptance_day_differs_from_filing_date"] = int(
        (clock["index_accepted_ny"].dt.normalize() != clock["filing_date"]).sum())

    # Corpus-wide: how often does the literal UTC reading of the JSON break EDGAR's rules?
    import pandas_market_calendars as mcal
    days = pd.DatetimeIndex(mcal.get_calendar("NYSE").valid_days("1995-01-01", "2030-12-31")
                            .tz_localize(None))
    f = sf.drop_duplicates("accession")
    f = f[f["form"].isin(["10-K", "10-Q", "8-K"])]
    ok = timing.json_acceptance_consistent(f["acceptance_json_as_utc_ny"], f["filing_date"],
                                           days, f["form"])
    out["json_literal_utc_provably_wrong_pct"] = round(float((~ok.astype(bool)).mean() * 100), 2)
    out["json_literal_utc_provably_wrong_pct_by_year"] = (
        (~ok.astype(bool)).groupby(f["filing_date"].dt.year).mean().mul(100).round(1)
        .loc[2014:].to_dict())
    out["json_acceptance_missing"] = int(f["acceptance_json_raw"].isin(["", None]).sum())

    # Cutoff-day filings: the only ones whose eligibility needs a time of day.
    vp = O / "verified_acceptance.parquet"
    if vp.exists():
        v = pd.read_parquet(vp)
        out["cutoff_day_filings_verified"] = int(v["acceptance_verified_ny"].notna().sum())
        out["cutoff_day_filings_total"] = len(v)
    inset = k8[k8["relation"].isin(["after_periodic", "no_periodic"])].merge(
        ev[["event_id", "call_cutoff_date", "call_cutoff_ts"]], on="event_id")
    out["info_set_8ks_on_cutoff_day"] = int(inset["filing_date"].eq(inset["call_cutoff_date"]).sum())
    out["info_set_8ks_after_cutoff_day"] = int((inset["filing_date"] > inset["call_cutoff_date"]).sum())
    out["periodic_after_cutoff_day"] = int(
        (ev["periodic_filing_date"] > ev["call_cutoff_date"]).sum())
    out["periodic_on_cutoff_day"] = int((ev["periodic_filing_date"] == ev["call_cutoff_date"]).sum())

    # Earnings-release 8-Ks (Item 2.02 within -1..+4 days of the announcement).
    mapped = ev[ev["identity_status"].eq("mapped")]
    r = rel[rel["event_id"].isin(mapped["event_id"])]
    leaked = r.merge(inset[["event_id", "accession"]], on=["event_id", "accession"])
    pre = r[r["filing_date"] < r["call_cutoff_date"]]
    out["release_8ks"] = len(r)
    out["events_with_release_8k"] = int(r["event_id"].nunique())
    out["release_8ks_in_an_information_set"] = len(leaked)
    out["release_8ks_filed_before_cutoff"] = len(pre)
    out["release_8ks_filed_before_cutoff_examples"] = pre.head(10)[
        ["event_id", "accession", "filing_date", "call_cutoff_date",
         "days_from_announcement"]].astype(str).to_dict("records")
    _w(pre.set_index("event_id"), "release_8k_before_cutoff.csv")
    # Item 2.02 8-Ks that ARE in information sets, by distance to the cutoff: earlier
    # quarters' releases (normal) versus anything close to the event (pre-announcements).
    i202 = inset[inset["items"].str.contains("2.02", regex=False)]
    gap = (i202["call_cutoff_date"] - i202["filing_date"]).dt.days
    near = i202[gap <= 14].assign(has_release_8k=lambda x: x["event_id"].isin(r["event_id"]))
    _w(near.set_index("event_id"), "info_set_2_02_within_14d_of_cutoff.csv")
    out["info_set_2_02_near_cutoff_without_release_8k"] = near.loc[
        ~near["has_release_8k"], "event_id"].tolist()
    out["info_set_item_2_02_8ks"] = len(i202)
    out["info_set_item_2_02_within_14d_of_cutoff"] = int((gap <= 14).sum())
    return out


# ───────────────────────────────────────── retrieval ────────────────────────────────────
def retrieval_tables() -> dict:
    fp, dp = O / "sample_filings.csv", O / "sample_documents.csv"
    if not (fp.exists() and dp.exists()):
        return {}
    f, d = pd.read_csv(fp), pd.read_csv(dp)
    prim = d[d["role"].eq("primary")]
    ex = d[d["role"].eq("exhibit")]
    k8 = f[f["form"].str.startswith("8-K")]
    out = {
        "filings": len(f), "documents": len(d),
        "filings_by_class": f["form_class"].value_counts().to_dict(),
        "filings_by_year_bucket": f["year_bucket"].value_counts().sort_index().to_dict(),
        "filings_by_size": f["size_class"].value_counts().to_dict(),
        "sectors": int(f["sector"].nunique()),
        "index_ok_pct": round(f["index_ok"].mean() * 100, 1),
        "primary_in_index_pct": round(f["primary_found_in_index"].mean() * 100, 1),
        "primary_download_ok_pct": round(prim["status"].eq(200).mean() * 100, 1),
        "primary_html_pct": round(prim["is_html"].mean() * 100, 1),
        "primary_text_chars_median": float(prim["text_chars"].median()),
        "docs_with_replacement_chars": int((d["replacement_chars"].fillna(0) > 0).sum()),
        "docs_with_mojibake": int((d["mojibake_hits"].fillna(0) > 0).sum()),
        "encodings": d["encoding"].value_counts().to_dict(),
        "parse_errors": int(d["parse_errors"].fillna("").ne("").sum()),
        "8k_with_ex99_pct": round((k8["n_ex99"] > 0).mean() * 100, 1),
        "8k_202_with_ex99_pct": round((k8[k8["items"].fillna("").str.contains("2.02")]["n_ex99"] > 0)
                                      .mean() * 100, 1),
        "exhibit_download_ok_pct": round(ex["status"].eq(200).mean() * 100, 1),
        "exhibit_text_chars_median": float(ex["text_chars"].median()),
        "exhibit_mentions_outlook_or_guidance_pct": round(ex["mentions_outlook"].fillna(False).mean() * 100, 1),
    }
    for form in ["10-K", "10-Q"]:
        p = prim[prim["form"].eq(form)]
        for sec in ["risk_factors", "mdna"]:
            for suffix in ["heading", "found", "found_with_title_fallback"]:
                col = f"{sec}_{suffix}"
                if col in p:
                    out[f"{form}_{sec}_{suffix}_pct"] = round(
                        p[col].fillna(False).astype(bool).mean() * 100, 1)
        out[f"{form}_text_chars_median"] = float(p["text_chars"].median())
        out[f"{form}_table_share_median"] = round(float((p["table_chars"] / p["text_chars"]).median()), 3)
        out[f"{form}_numeric_table_share_median"] = round(
            float((p["numeric_table_chars"] / p["text_chars"]).median()), 3)
    q = prim[prim["form"].eq("10-Q")]
    out["10-Q_risk_factors_chars_quartiles"] = q["risk_factors_chars"].quantile([.25, .5, .75]).tolist()
    k = prim[prim["form"].eq("10-K")].merge(f[["accession", "exhibit_types"]], on="accession")
    miss = k[~k["mdna_found_with_title_fallback"].fillna(False).astype(bool)]
    out["10-K_mdna_missing_stocks"] = miss["stock"].tolist()
    out["10-K_with_EX-13_annual_report_exhibit"] = int(k["exhibit_types"].fillna("").str.contains("EX-13").sum())
    out["8k_primary_text_chars_median"] = float(prim[prim["form"].str.startswith("8-K")]["text_chars"].median())
    out["8k_other_with_ex99_pct"] = round((f[f["form_class"].eq("8-K other")]["n_ex99"] > 0).mean() * 100, 1)
    rel = f[f["form_class"].eq("8-K 2.02")]
    rel_ex = ex[ex["accession"].isin(rel["accession"])]
    out["8k_202_filings_with_identified_release_pct"] = round(
        rel["accession"].isin(rel_ex.loc[rel_ex["earnings_release"].fillna(False).astype(bool),
                                         "accession"]).mean() * 100, 1)
    by_bucket = prim.merge(f[["accession", "year_bucket"]], on="accession").groupby(
        ["year_bucket", "form"])[["text_chars"]].median()
    _w(by_bucket, "sample_text_chars_by_bucket.csv")
    return out


# ──────────────────────────────────────── corpus size ───────────────────────────────────
def corpus_estimate(ev, k8, sf) -> dict:
    """Full 2014-2026 corpus from the metadata index, scaled by sampled document sizes."""
    mapped = ev[ev["identity_status"].eq("mapped")]
    lo = mapped["call_cutoff_date"].min() - pd.Timedelta(days=200)
    f = sf.drop_duplicates("accession")
    f = f[(f["filing_date"] >= lo) & (f["filing_date"] <= mapped["call_cutoff_date"].max())]
    periodic = f[f["base_form"].isin(["10-K", "10-Q", "10-KT", "10-QT"])]
    eightk_all = f[f["base_form"].eq("8-K")]
    inset = k8[k8["relation"].isin(["after_periodic", "no_periodic"])]
    eightk_rel = eightk_all[eightk_all["accession"].isin(inset["accession"])]
    out = {"window": [str(lo.date()), str(mapped["call_cutoff_date"].max().date())],
           "periodic_filings": len(periodic),
           "periodic_originals": int((~periodic["is_amendment"]).sum()),
           "periodic_amendments": int(periodic["is_amendment"].sum()),
           "selected_periodic_unique": int(mapped["periodic_accession"].nunique()),
           "8k_all": len(eightk_all), "8k_in_some_information_set": len(eightk_rel),
           "full_submission_bytes_periodic": int(periodic["size"].sum()),
           "full_submission_bytes_8k": int(eightk_all["size"].sum())}
    dp, fp = O / "sample_documents.csv", O / "sample_filings.csv"
    if dp.exists():
        d, fs = pd.read_csv(dp), pd.read_csv(fp)
        prim = d[d["role"].eq("primary")].merge(fs[["accession", "form_class"]], on="accession")
        ex = d[d["role"].eq("exhibit")]
        k8s = fs[fs["form"].str.startswith("8-K")]
        per_10k = prim[prim["form_class"].eq("10-K")]
        per_10q = prim[prim["form_class"].eq("10-Q")]
        per_8k = prim[prim["form_class"].isin(["8-K 2.02", "8-K other"])]
        n10k = int(periodic["base_form"].isin(["10-K", "10-KT"]).sum())
        n10q = len(periodic) - n10k
        ex_per_8k = k8s["n_ex99"].clip(upper=4).mean()
        est = {
            "mean_primary_bytes_10K": per_10k["bytes"].mean(),
            "mean_primary_bytes_10Q": per_10q["bytes"].mean(),
            "mean_primary_bytes_8K": per_8k["bytes"].mean(),
            "mean_ex99_bytes": ex["bytes"].mean(),
            "mean_ex99_per_8k": ex_per_8k,
            "gzip_ratio_raw": (d["gzip_bytes"].sum() / d["bytes"].sum()),
            "text_bytes_per_raw_byte": (d["text_chars"].sum() / d["bytes"].sum()),
        }
        raw = (n10k * est["mean_primary_bytes_10K"] + n10q * est["mean_primary_bytes_10Q"]
               + len(eightk_all) * (est["mean_primary_bytes_8K"]
                                    + ex_per_8k * est["mean_ex99_bytes"]))
        n_ex = len(eightk_all) * ex_per_8k
        n_req = len(periodic) + len(eightk_all) + len(periodic) + len(eightk_all) + n_ex
        out.update({k: round(float(v), 3) for k, v in est.items()})
        # Event-relevant subset only: the selected 10-Q/10-K per event and the 8-Ks that
        # sit in some information set (what a text experiment would actually read).
        sel = f[f["accession"].isin(mapped["periodic_accession"].dropna())]
        s10k = int(sel["base_form"].isin(["10-K", "10-KT"]).sum())
        s10q = len(sel) - s10k
        n8 = len(eightk_rel)
        rel_raw = (s10k * est["mean_primary_bytes_10K"] + s10q * est["mean_primary_bytes_10Q"]
                   + n8 * (est["mean_primary_bytes_8K"] + ex_per_8k * est["mean_ex99_bytes"]))
        rel_req = 2 * len(sel) + 2 * n8 + n8 * ex_per_8k
        out.update({"relevant_selected_periodic": len(sel), "relevant_8k": n8,
                    "relevant_est_ex99": int(n8 * ex_per_8k),
                    "relevant_est_raw_bytes": int(rel_raw),
                    "relevant_est_gzip_bytes": int(rel_raw * est["gzip_ratio_raw"]),
                    "relevant_est_text_bytes": int(rel_raw * est["text_bytes_per_raw_byte"]),
                    "relevant_est_requests": int(rel_req),
                    "relevant_est_hours_at_5_req_s": round(rel_req / REQ_PER_S / 3600, 1),
                    "relevant_est_hours_at_measured_2_req_s": round(rel_req / 2 / 3600, 1)})
        out.update({"est_exhibits_ex99": int(n_ex),
                    "est_raw_bytes_primary_plus_ex99": int(raw),
                    "est_gzip_bytes": int(raw * est["gzip_ratio_raw"]),
                    "est_text_bytes": int(raw * est["text_bytes_per_raw_byte"]),
                    "est_requests_index_primary_ex99": int(n_req),
                    "est_hours_at_5_req_s": round(n_req / REQ_PER_S / 3600, 1),
                    "est_hours_at_measured_rate_2_req_s": round(n_req / 2 / 3600, 1)})
    return out


# ───────────────────────────────────────── main ─────────────────────────────────────────
def main(argv=None) -> int:
    ev, k8, am, rel, sf, ma, seg = load()
    ev["window"] = ev["announce_window"]
    ev["inactive"] = ev["stock"].isin(
        ma.loc[ma["provenance"].eq("manual_verified") & ma["sec_ciks"].isna(), "stock"])
    report = {"build": json.loads((O / "build_stats.json").read_text())}
    report["mapping"] = mapping_tables(ev, ma, seg, rel)
    pc = _w(periodic_coverage(ev), "coverage_periodic_by_year.csv")
    pcw = _w(periodic_coverage(ev, "window"), "coverage_periodic_by_window.csv")
    kc = _w(k8_coverage(ev, k8), "coverage_8k_by_year.csv")
    kcw = _w(k8_coverage(ev, k8, "window"), "coverage_8k_by_window.csv")
    _w(item_distribution(k8), "8k_item_distribution.csv")
    ina = ev[ev["inactive"]]
    _w(periodic_coverage(ina, "stock"), "coverage_inactive_stocks.csv")
    am_n = am.groupby("event_id").size()
    report["periodic_amendments_in_window_pct_events"] = round(
        float(ev["event_id"].isin(am_n.index).mean() * 100), 2)
    report["timing"] = timing_tables(ev, k8, rel, sf)
    report["retrieval"] = retrieval_tables()
    report["corpus"] = corpus_estimate(ev, k8, sf)
    report["snapshot"] = acquire.latest_snapshot().name if any(
        p for p in paths.SNAPSHOT_ROOT.glob("sec_*") if not p.name.endswith(".partial")) else None
    (O / "audit_summary.json").write_text(json.dumps(report, indent=1, default=str))
    pd.set_option("display.width", 220)
    print(pc.to_string(), "\n", pcw.to_string(), "\n", kc.to_string(), "\n", kcw.to_string())
    print(json.dumps({k: v for k, v in report.items() if k != "build"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
