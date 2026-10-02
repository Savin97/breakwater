"""Walk-forward test of SEC features against Model C. Rules: PREREGISTRATION.md.

    PYTHONPATH=. .venv/bin/python -m research.sec_features.evaluate              # 2017-2025
    PYTHONPATH=. .venv/bin/python -m research.sec_features.evaluate --holdout    # 2026, once

Reads `output/sec_features/{feature_frame_current,sec_features}.parquet` and `gates.json`
(the reliability gates, written by `audit.py` before this module was first run). The
target is joined here and nowhere earlier. Model, metrics and bootstrap are Phase 3's,
via `research.options_pilot.evaluate` (identical rows asserted per comparison).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.announcement_timing import AMC, BMO
from research.options_pilot.evaluate import (compare_on_common_sample, delta_boot,
                                             metrics_by, walk_forward, _ids_sha)
from research.phase3_refit.candidates import add_model_inputs

OUT = Path("output/sec_features")
POP_YEARS = (2014, 2026)
MIN_PRIOR = 8
TEST_YEARS = list(range(2017, 2026))
HOLDOUT_YEAR = 2026
BOOT_REPS = 500

C = ["log_hist_mean_abs", "log_vol_30d"]
K8 = ["l_num_8k_since_periodic", "l_num_8k_last_30d", "has_2_02", "has_7_01", "has_8_01",
      "has_5_02", "num_material_item_types", "has_recent_2_02_14d", "has_recent_7_01_14d",
      "has_recent_8_01_14d"]
FAMILIES = {
    "C+guidance": ["guidance_present"],
    "C+uncertainty": ["l_uncertainty_rate", "uncertainty_change"],
    "C+novelty": ["release_text_change"],
    "C+8K": K8,
}
COMPACT = ["guidance_present", "uncertainty_change", "release_text_change",
           "l_num_8k_last_30d", "has_recent_2_02_14d"]
PRIMARY_KEYS = ["top10_hit", "top10_capture", "top10_lift", "top20_hit", "top20_capture",
                "top20_lift", "roc_auc", "pr_auc", "brier", "log_loss"]


# ─────────────────────────────────────────── panel ──────────────────────────────────────
def panel() -> pd.DataFrame:
    f = pd.read_parquet(OUT / "feature_frame_current.parquet")
    f["event_id"] = f["stock"] + "|" + pd.to_datetime(f["earnings_date"]).dt.strftime("%Y-%m-%d")
    f = add_model_inputs(f, "call")
    pop = f[~f["is_pending"].astype(bool) & f["window_ok"] & f["y_extreme"].notna()
            & f["year"].between(*POP_YEARS) & f["n_prior_call"].ge(MIN_PRIOR)
            & f["log_hist_mean_abs"].notna() & f["log_vol_30d"].notna()]
    s = pd.read_parquet(OUT / "sec_features.parquet")
    p = pop.merge(s, on="event_id", how="inner", suffixes=("", "_sec"))
    p["l_uncertainty_rate"] = np.log1p(p["uncertainty_rate"])
    p["l_eps_width"] = np.log(p["eps_guidance_width"])
    p["l_rev_width"] = np.log(p["revenue_guidance_width"])
    for c in ["num_8k_since_periodic", "num_8k_last_30d", "num_8k_ex99_last_30d"]:
        if c in p:
            p[f"l_{c}"] = np.log1p(p[c])
    return p.reset_index(drop=True)


def gates() -> dict:
    return json.loads((OUT / "gates.json").read_text())


def specs(g: dict, p: pd.DataFrame) -> tuple[dict, list]:
    fam = {k: list(v) for k, v in FAMILIES.items()}
    compact = list(COMPACT)
    if not g["guidance_present_ok"]:
        fam.pop("C+guidance")
        compact.remove("guidance_present")
    if not g["release_ok"]:
        for k in ("C+guidance", "C+uncertainty", "C+novelty"):
            fam.pop(k, None)
        compact = [c for c in compact if c not in
                   ("guidance_present", "uncertainty_change", "release_text_change")]
    if not g["uncertainty_ok"]:
        fam.pop("C+uncertainty", None)
        compact = [c for c in compact if c != "uncertainty_change"]
    out = {"C": C, **{k: C + v for k, v in fam.items()}, "C+compact": C + compact}
    if "l_num_8k_ex99_last_30d" in p and p["l_num_8k_ex99_last_30d"].notna().mean() > 0.95:
        out["C+ex99"] = C + ["l_num_8k_ex99_last_30d"]
    return out, compact


def main_sample(p: pd.DataFrame, g: dict) -> pd.DataFrame:
    """Rows with previous AND second-prior release features (identical rows for every spec)."""
    need = ["guidance_present", "uncertainty_rate", "uncertainty_change", "release_text_change"]
    if not g["release_ok"]:
        need = []
    return p.dropna(subset=need + K8 + C)


# ──────────────────────────────────────────── run ───────────────────────────────────────
def block(frame: pd.DataFrame, sp: dict, years: list[int], reps: int) -> tuple[dict, dict]:
    preds = compare_on_common_sample(frame, sp, years)
    ref = preds["C"]
    res = {"n_oof": len(ref), "n_stocks": int(ref["stock"].nunique()),
           "base_rate": float(ref["y"].mean()), "event_ids_sha": _ids_sha(ref["event_id"]),
           "metrics": {k: metrics_by(v) for k, v in preds.items()},
           "deltas": {f"{k}_vs_C": delta_boot(v, ref, reps) for k, v in preds.items() if k != "C"}}
    return res, preds


def table(res: dict, keys=PRIMARY_KEYS) -> pd.DataFrame:
    rows = []
    for name, m in res["metrics"].items():
        r = {"model": name, **{k: m["ALL"].get(k) for k in keys},
             "bmo_auc": m[BMO].get("roc_auc"), "amc_auc": m[AMC].get("roc_auc")}
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")


def delta_table(res: dict) -> pd.DataFrame:
    rows = []
    for name, d in res["deltas"].items():
        r = {"comparison": name}
        for k in ["top10_capture", "top20_capture", "top10_hit", "top20_hit", "top10_lift",
                  "top20_lift", "roc_auc", "pr_auc", "brier", "log_loss"]:
            r[k] = d.get(f"d_{k}")
            r[f"{k}_lo"] = d.get(f"d_{k}_lo")
            r[f"{k}_hi"] = d.get(f"d_{k}_hi")
        rows.append(r)
    return pd.DataFrame(rows).set_index("comparison")


def yearly(res: dict) -> pd.DataFrame:
    rows = []
    for name, m in res["metrics"].items():
        for y in TEST_YEARS + [HOLDOUT_YEAR]:
            if y in m:
                rows.append({"model": name, "year": y, **{k: m[y].get(k) for k in
                             ["n", "base_rate", "roc_auc", "top10_capture", "top20_capture"]}})
    return pd.DataFrame(rows)


def worth_keeping(d: dict) -> bool:
    return (d["d_top10_capture"] >= 0.01 and d["d_top10_capture_lo"] > 0
            and d["d_top20_capture"] > 0)


def verdict(res: dict, holdout_deltas: dict | None) -> str:
    """Pre-registered rule (§6). `holdout_deltas`: {"<model>_vs_C": delta dict} for 2026."""
    ds = res["deltas"]
    keep = [k for k, d in ds.items() if worth_keeping(d)]
    hold_ok = holdout_deltas is not None and any(
        holdout_deltas[k]["d_top10_capture"] >= 0 for k in keep if k in holdout_deltas)
    if keep and hold_ok:
        return "SEC FEATURES SHOW MATERIAL INCREMENTAL VALUE"
    weak = any(d["d_top10_capture_lo"] > 0 or d["d_top20_capture_lo"] > 0
               or d["d_top10_capture"] >= 0.01 or d["d_top20_capture"] >= 0.01
               for d in ds.values())
    if keep or weak:
        return "SEC FEATURES SHOW WEAK / UNCERTAIN VALUE"
    return "SEC FEATURES ADD NO USEFUL VALUE"


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def run_selection(reps: int = BOOT_REPS) -> dict:
    p = panel()
    p = p[p["year"].lt(HOLDOUT_YEAR)]                      # 2026 never enters selection
    assert not p["year"].eq(HOLDOUT_YEAR).any()
    g = gates()
    sp, compact = specs(g, p)
    ms = main_sample(p, g)
    main, preds = block(ms, sp, TEST_YEARS, reps)
    broad, _ = block(p.dropna(subset=K8 + C), {"C": C, "C+8K": C + K8}, TEST_YEARS, reps)
    out = {"gates": g, "specs": sp, "compact": compact, "population_rows": len(p),
           "main": main, "broad_8k": broad}
    if g.get("width_ok"):
        ws = ms.dropna(subset=["l_eps_width"])
        wsp = {"C": C, "C+guidance+width": C + ["guidance_present", "l_eps_width"]}
        out["width"], _ = block(ws, wsp, TEST_YEARS, reps)
    fam = {k: v for k, v in main["deltas"].items()
           if k not in ("C+compact_vs_C", "C+ex99_vs_C")}
    best = max(fam, key=lambda k: fam[k]["d_top10_capture"]).removesuffix("_vs_C")
    out["holdout_models"] = ["C+compact", best]
    OUT.joinpath("selection.json").write_text(json.dumps(out, indent=1, default=_json))
    table(main).to_csv(OUT / "main_models.csv")
    delta_table(main).to_csv(OUT / "main_deltas.csv")
    yearly(main).to_csv(OUT / "main_yearly.csv", index=False)
    table(broad).to_csv(OUT / "broad_8k_models.csv")
    delta_table(broad).to_csv(OUT / "broad_8k_deltas.csv")
    pd.concat(preds.values(), keys=preds.keys(), names=["model"]).to_parquet(OUT / "oof_main.parquet")
    return out


def run_holdout(reps: int = BOOT_REPS) -> dict:
    sel = json.loads((OUT / "selection.json").read_text())
    p = panel()
    g = sel["gates"]
    ms = main_sample(p, g)
    sp = {"C": sel["specs"]["C"], **{m: sel["specs"][m] for m in sel["holdout_models"]}}
    res, _ = block(ms, sp, [HOLDOUT_YEAR], reps)
    assert res["n_oof"] == int(ms["year"].eq(HOLDOUT_YEAR).sum())
    res["frozen_models"] = sel["holdout_models"]
    res["verdict"] = verdict(sel["main"], res["deltas"])
    OUT.joinpath("holdout_2026.json").write_text(json.dumps(res, indent=1, default=_json))
    table(res).to_csv(OUT / "holdout_models.csv")
    delta_table(res).to_csv(OUT / "holdout_deltas.csv")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--reps", type=int, default=BOOT_REPS)
    a = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    if a.holdout:
        r = run_holdout(a.reps)
        print(table(r).round(4).to_string(), "\n", delta_table(r).round(4).to_string())
        print(r["verdict"])
    else:
        r = run_selection(a.reps)
        print(table(r["main"]).round(4).to_string(), "\n",
              delta_table(r["main"]).round(4).to_string(), "\n", r["holdout_models"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
