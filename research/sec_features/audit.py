"""Feature coverage and extraction-reliability audit — run BEFORE any outcome is read.

    PYTHONPATH=. .venv/bin/python -m research.sec_features.audit --samples   # texts to read
    PYTHONPATH=. .venv/bin/python -m research.sec_features.audit             # tables + gates

Denominator: SEC-mapped events of the current frame, completed, BMO/AMC, 2014-2026. No
outcome column is read (the Phase 3 population filter needs the target, so it is applied
only in evaluate.py). Writes `output/sec_features/coverage_*.csv`, the reading samples, and
`gates.json` from the human judgments in `research/sec_features/manual_audit.csv`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from research.sec_features import features as F
from research.sec_features.build import OUT
from research.sec_filings_pilot import acquire, documents, paths as sp

MANUAL = Path(__file__).with_name("manual_audit.csv")
SEED = 7
FEATURES = ["guidance_present", "eps_guidance_width", "revenue_guidance_width",
            "uncertainty_rate", "uncertainty_change", "release_text_change",
            "num_8k_since_periodic", "num_8k_last_30d", "has_recent_2_02_14d",
            "num_8k_ex99_last_30d"]


def load() -> pd.DataFrame:
    s = pd.read_parquet(OUT / "sec_features.parquet")
    ev = pd.read_parquet(sp.OUT / "events.parquet",
                         columns=["event_id", "year", "announce_window", "sector"])
    return s.merge(ev, on="event_id", how="left")


def coverage(s: pd.DataFrame) -> dict[str, pd.DataFrame]:
    feats = [f for f in FEATURES if f in s]
    out = {}
    for by in ["year", "announce_window", "sector"]:
        g = s.groupby(by)
        t = pd.DataFrame({"eligible_events": g.size()})
        for f in feats:
            t[f] = g[f].apply(lambda x: x.notna().mean() * 100).round(1)
        t.loc["all"] = [len(s)] + [round(s[f].notna().mean() * 100, 1) for f in feats]
        out[by] = t
    stats = pd.DataFrame({
        "available": [int(s[f].notna().sum()) for f in feats],
        "pct": [round(s[f].notna().mean() * 100, 1) for f in feats]}, index=feats)
    out["features"] = stats
    steps = {"previous release found (8-K matched)": s["prev_status"].eq("ok").mean(),
             "previous release usable (exhibit + prose)": s["prev_ok"].mean(),
             "second-prior release usable": s["prev2_ok"].mean(),
             "both usable": (s["prev_ok"] & s["prev2_ok"]).mean(),
             "guidance detected (of usable previous)": s.loc[s["prev_ok"], "guidance_present"].mean(),
             "EPS range extracted (of usable previous)": s.loc[s["prev_ok"], "eps_guidance_width"].notna().mean(),
             "revenue range extracted (of usable previous)": s.loc[s["prev_ok"], "revenue_guidance_width"].notna().mean()}
    out["steps"] = pd.DataFrame({"share": pd.Series(steps).round(4)})
    out["missing_prev"] = s["prev_missing_reason"].value_counts().to_frame("events")
    out["missing_prev2"] = s["prev2_missing_reason"].value_counts().to_frame("events")
    return out


# ───────────────────────────────────────── samples ──────────────────────────────────────
def _text(r) -> tuple[str, str]:
    raw = acquire.archive_path(r.prev_cik, r.prev_accession, r.prev_exhibit_filename).read_bytes()
    return F.release_text(documents.decode(raw).text)


def _strat(df: pd.DataFrame, n: int, seed: int = SEED) -> pd.DataFrame:
    df = df.assign(bucket=pd.cut(df["year"], [2013, 2016, 2019, 2022, 2026]).astype(str))
    k = max(1, n // df["bucket"].nunique())
    return df.groupby("bucket", group_keys=False).apply(
        lambda g: g.sample(min(k, len(g)), random_state=seed))


def samples(s: pd.DataFrame, seed: int = SEED) -> str:
    ok = s[s["prev_ok"]].drop_duplicates("prev_accession")
    lines = []
    lines.append("=== RELEASE MATCHING (40) ===")
    for r in _strat(ok, 40, seed).itertuples():
        full, prose = _text(r)
        lines.append(f"[R] {r.event_id} prev={r.prev_accession} filed={str(r.prev_filing_date)[:10]} "
                     f"{r.prev_exhibit_type} '{r.prev_exhibit_description}' :: {full[:350]!r}")
    lines.append("=== GUIDANCE DETECTED (20) ===")
    for r in _strat(ok[ok["guidance_present"].eq(1)], 20, seed).itertuples():
        full, _ = _text(r)
        gs = F.guidance_sentences(full)
        lines.append(f"[G+] {r.event_id} n={len(gs)} :: " + " || ".join(x[:250] for x in gs[:2]))
    lines.append("=== GUIDANCE NOT DETECTED (20) ===")
    for r in _strat(ok[ok["guidance_present"].eq(0)], 20, seed).itertuples():
        full, _ = _text(r)
        ctx = [full[max(0, m.start() - 120):m.end() + 200].replace("\n", " / ")
               for m in re.finditer(r"(?i)\b(outlook|guidance|expects?)\b", full)][:2]
        lines.append(f"[G-] {r.event_id} :: " + (" || ".join(ctx) if ctx else "(no outlook/guidance/expect word)"))
    lines.append("=== EPS / REVENUE WIDTHS (20) ===")
    w = ok[ok["eps_guidance_width"].notna() | ok["revenue_guidance_width"].notna()]
    for r in _strat(w, 20, seed).itertuples():
        full, _ = _text(r)
        gs = F.guidance_sentences(full)
        hit = [x for x in gs if any(rx.search(x) for rx in F.EPS_RANGE) or F.REV_RANGE.search(x)]
        lines.append(f"[W] {r.event_id} eps_w={r.eps_guidance_width} rev_w={r.revenue_guidance_width} :: "
                     + " || ".join(x[:300] for x in hit[:2]))
    lines.append("=== UNCERTAINTY HITS (20 releases) ===")
    for r in _strat(ok, 20, seed).itertuples():
        _, prose = _text(r)
        ctx = [prose[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
               for m in F.UNCERTAINTY.finditer(prose)]
        lines.append(f"[U] {r.event_id} rate={r.uncertainty_rate:.2f} hits={len(ctx)} :: " + " || ".join(ctx[:12]))
    return "\n".join(lines)


def write_gates(s: pd.DataFrame, cov: dict) -> dict:
    m = pd.read_csv(MANUAL)
    # The latest audit round of each kind decides (Amendment 1: one re-audit allowed).
    m = m[m["round"].eq(m.groupby("kind")["round"].transform("max"))]

    def acc(kind):
        x = m[m["kind"].eq(kind)]
        return float(x["correct"].mean()) if len(x) else np.nan, int(len(x))
    rel, n_rel = acc("release")
    gp, n_gp = acc("guidance_pos")
    gn, n_gn = acc("guidance_neg")
    wd, n_wd = acc("width")
    unc = m[m["kind"].eq("uncertainty")]
    false_share = float(unc["false_hits"].sum() / unc["hits"].sum()) if len(unc) else np.nan
    both = float((s["prev_ok"] & s["prev2_ok"]).mean())
    eps_cov = float(s.loc[s["prev_ok"] & s["prev2_ok"], "eps_guidance_width"].notna().mean())
    g = {"release_match_accuracy": rel, "release_audited": n_rel, "release_coverage_both": both,
         "release_ok": bool(rel >= 0.90 and both >= 0.60),
         "guidance_pos_accuracy": gp, "guidance_neg_accuracy": gn,
         "guidance_present_ok": bool(gp >= 0.80 and gn >= 0.80),
         "eps_width_coverage_common": eps_cov, "width_accuracy": wd,
         "width_ok": bool(eps_cov >= 0.40 and wd >= 0.90),
         "uncertainty_false_hit_share": false_share,
         "uncertainty_ok": bool(not false_share > 0.50)}
    (OUT / "gates.json").write_text(json.dumps(g, indent=1))
    return g


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    s = load()
    if a.samples:
        (OUT / f"manual_audit_samples_seed{a.seed}.txt").write_text(samples(s, a.seed))
        print("wrote samples for seed", a.seed)
        return 0
    cov = coverage(s)
    for k, t in cov.items():
        t.to_csv(OUT / f"coverage_{k}.csv")
        print(f"\n## {k}\n", t.to_string())
    if MANUAL.exists():
        print(json.dumps(write_gates(s, cov), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
