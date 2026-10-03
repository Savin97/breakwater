"""Walk-forward test of point-in-time fundamentals against Model C. Rules: PREREGISTRATION.md.

    PYTHONPATH=. .venv/bin/python -m research.fundamentals.evaluate              # 2017-2025
    PYTHONPATH=. .venv/bin/python -m research.fundamentals.evaluate --holdout    # 2026, once

Reads `output/fundamentals/{fundamentals.parquet,gates.json}` and the current corrected frame.
The target is joined here and nowhere earlier. Model, metrics and bootstrap are Phase 3's via
`research.options_pilot.evaluate` (identical event ids asserted for every comparison).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.announcement_timing import AMC, BMO
from research.options_pilot.evaluate import (compare_on_common_sample, delta_boot,
                                             metrics_by, _ids_sha)
from research.phase3_refit.candidates import add_model_inputs

OUT = Path("output/fundamentals")
FRAME = Path("output/sec_features/feature_frame_current.parquet")
POP_YEARS = (2014, 2026)
MIN_PRIOR = 8
TEST_YEARS = list(range(2017, 2026))
HOLDOUT_YEAR = 2026
BOOT_REPS = 500
EPS = 0.01

C = ["log_hist_mean_abs", "log_vol_30d"]
SIZE = ["log_assets"]
# family name -> (model inputs, the gated features they come from)
FAMILIES = {
    "C+revenue": (["l_rev_vol"], ["revenue_growth_volatility"]),
    "C+margin": (["l_margin_vol", "l_abs_margin_chg"],
                 ["operating_margin_volatility", "abs_operating_margin_change_yoy"]),
    "C+leverage": (["leverage_c"], ["leverage"]),
    "C+accruals": (["l_abs_accruals"], ["abs_accruals"]),
    "C+workingcap": (["wc_to_assets_c", "l_abs_wc_chg"], ["wc_to_assets", "abs_wc_change_yoy"]),
}
COMPACT = ["l_rev_vol", "l_abs_margin_chg", "leverage_c", "l_abs_accruals", "l_abs_wc_chg"]
SOURCE = {i: f for inp, fs in FAMILIES.values() for i, f in zip(inp, fs)}
DELTA_KEYS = ["top10_capture", "top20_capture", "top10_hit", "top20_hit", "top10_lift",
              "top20_lift", "roc_auc", "pr_auc", "brier", "log_loss"]
PRIMARY_KEYS = ["top10_hit", "top10_capture", "top10_lift", "top20_hit", "top20_capture",
                "top20_lift", "roc_auc", "pr_auc", "brier", "log_loss"]


# ─────────────────────────────────────────── panel ──────────────────────────────────────
def transform(f: pd.DataFrame) -> pd.DataFrame:
    """Fixed transforms (PREREGISTRATION.md §2). Standardisation happens in the fold."""
    f = f.copy()
    f["l_rev_vol"] = np.log(f["revenue_growth_volatility"] + EPS)
    f["l_margin_vol"] = np.log(f["operating_margin_volatility"] + EPS)
    f["l_abs_margin_chg"] = np.log(f["abs_operating_margin_change_yoy"] + EPS)
    f["leverage_c"] = f["leverage"].clip(0, 2)
    f["l_abs_accruals"] = np.log(f["abs_accruals"] + EPS)
    f["wc_to_assets_c"] = f["wc_to_assets"].clip(-1, 1)
    f["l_abs_wc_chg"] = np.log(f["abs_wc_change_yoy"] + EPS)
    return f


def panel() -> pd.DataFrame:
    f = pd.read_parquet(FRAME)
    f["event_id"] = f["stock"] + "|" + pd.to_datetime(f["earnings_date"]).dt.strftime("%Y-%m-%d")
    f = add_model_inputs(f, "call")
    pop = f[~f["is_pending"].astype(bool) & f["window_ok"] & f["y_extreme"].notna()
            & f["year"].between(*POP_YEARS) & f["n_prior_call"].ge(MIN_PRIOR)
            & f["log_hist_mean_abs"].notna() & f["log_vol_30d"].notna()]
    fu = pd.read_parquet(OUT / "fundamentals.parquet")
    fu = fu[fu["status"].eq("ok")].drop(columns=["stock", "sector", "year", "announce_window",
                                                 "earnings_date"])
    p = pop.merge(fu, on="event_id", how="left", validate="one_to_one")
    return transform(p).reset_index(drop=True)


def gates() -> dict:
    return json.loads((OUT / "gates.json").read_text())["features"]


def blocks(g: dict) -> tuple[dict, list]:
    """{block name: specs}, with gate-failed features removed (PREREGISTRATION.md §4)."""
    ok = lambda i: g[SOURCE[i]]["pass"]
    size_ok = g["log_assets"]["pass"]
    out = {}
    for name, (inp, _) in FAMILIES.items():
        inp = [i for i in inp if ok(i)]
        if inp:
            out[name] = inp
    compact = [i for i in COMPACT if ok(i)]
    if compact:
        out["C+compact"] = compact
    res = {}
    for name, inp in out.items():
        sp = {"C": C, name: C + inp}
        if size_ok:
            sp.update({"C+size": C + SIZE, f"C+size+{name[2:]}": C + SIZE + inp})
        res[name] = sp
    return res, compact


# ──────────────────────────────────────────── run ───────────────────────────────────────
def run_block(frame: pd.DataFrame, sp: dict, years: list[int], reps: int) -> tuple[dict, dict]:
    preds = compare_on_common_sample(frame, sp, years)
    ref = preds["C"]
    res = {"n_oof": len(ref), "n_stocks": int(ref["stock"].nunique()),
           "base_rate": float(ref["y"].mean()), "event_ids_sha": _ids_sha(ref["event_id"]),
           "n_bmo": int(ref["announce_window"].eq(BMO).sum()),
           "metrics": {k: metrics_by(v) for k, v in preds.items()}, "deltas": {}}
    names = list(sp)
    fam = names[1]
    res["deltas"][f"{fam}_vs_C"] = delta_boot(preds[fam], ref, reps)
    if "C+size" in preds:
        res["deltas"]["C+size_vs_C"] = delta_boot(preds["C+size"], ref, reps)
        res["deltas"][f"{names[3]}_vs_C+size"] = delta_boot(preds[names[3]], preds["C+size"], reps)
    return res, preds


def table(res: dict) -> pd.DataFrame:
    rows = []
    for name, m in res["metrics"].items():
        rows.append({"model": name, **{k: m["ALL"].get(k) for k in PRIMARY_KEYS},
                     "bmo_auc": m[BMO].get("roc_auc"), "amc_auc": m[AMC].get("roc_auc"),
                     "bmo_top10_capture": m[BMO].get("top10_capture"),
                     "amc_top10_capture": m[AMC].get("top10_capture")})
    return pd.DataFrame(rows).set_index("model")


def delta_table(res: dict) -> pd.DataFrame:
    rows = []
    for name, d in res["deltas"].items():
        r = {"comparison": name}
        for k in DELTA_KEYS:
            r[k], r[f"{k}_lo"], r[f"{k}_hi"] = d.get(f"d_{k}"), d.get(f"d_{k}_lo"), d.get(f"d_{k}_hi")
        rows.append(r)
    return pd.DataFrame(rows).set_index("comparison")


def yearly(res: dict, years) -> pd.DataFrame:
    rows = []
    for name, m in res["metrics"].items():
        for y in years:
            if y in m:
                rows.append({"model": name, "year": y, **{k: m[y].get(k) for k in
                             ["n", "base_rate", "roc_auc", "top10_capture", "top20_capture"]}})
    return pd.DataFrame(rows)


def worth_keeping(d: dict) -> bool:
    return d["d_top10_capture"] >= 0.01 and d["d_top10_capture_lo"] > 0 and d["d_top20_capture"] > 0


def not_size(res_block: dict) -> bool:
    k = [x for x in res_block["deltas"] if x.endswith("_vs_C+size")]
    return bool(k) and res_block["deltas"][k[0]]["d_top10_capture_lo"] > 0


def verdict(sel: dict, holdout: dict | None) -> str:
    """PREREGISTRATION.md §6. `sel`: {block: result}; `holdout`: {block: result} for 2026."""
    if not sel:
        return "POINT-IN-TIME FUNDAMENTALS ARE TOO UNRELIABLE TO ANSWER"
    fam_d = {b: r["deltas"][f"{b}_vs_C"] for b, r in sel.items()}
    material = [b for b, d in fam_d.items() if worth_keeping(d) and not_size(sel[b])
                and holdout is not None and b in holdout
                and holdout[b]["deltas"][f"{b}_vs_C"]["d_top10_capture"] >= 0]
    if material:
        return "FUNDAMENTALS SHOW MATERIAL INCREMENTAL VALUE"
    weak = any(d["d_top10_capture_lo"] > 0 or d["d_top20_capture_lo"] > 0
               or d["d_top10_capture"] >= 0.01 or d["d_top20_capture"] >= 0.01
               for d in fam_d.values())
    return "FUNDAMENTALS SHOW WEAK / UNCERTAIN VALUE" if weak else "FUNDAMENTALS ADD NO USEFUL VALUE"


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


def run_selection(reps: int = BOOT_REPS) -> dict:
    p = panel()
    p = p[p["year"].lt(HOLDOUT_YEAR)].reset_index(drop=True)      # 2026 never enters selection
    assert not p["year"].eq(HOLDOUT_YEAR).any()
    g = gates()
    bl, compact = blocks(g)
    out = {"gates": g, "blocks": bl, "compact": compact, "population_rows": len(p), "results": {}}
    tabs, dts, yrs, oof = [], [], [], {}
    for name, sp in bl.items():
        need = sorted({c for v in sp.values() for c in v})
        frame = p.dropna(subset=need)
        res, preds = run_block(frame, sp, TEST_YEARS, reps)
        res["block_rows_all_years"] = len(frame)
        out["results"][name] = res
        tabs.append(table(res).assign(block=name))
        dts.append(delta_table(res).assign(block=name))
        yrs.append(yearly(res, TEST_YEARS).assign(block=name))
        for k, v in preds.items():
            oof[(name, k)] = v
    fam = {b: out["results"][b]["deltas"][f"{b}_vs_C"]["d_top10_capture"]
           for b in out["results"] if b != "C+compact"}
    out["holdout_models"] = ["C+compact", max(fam, key=fam.get)]
    out["verdict_pre_holdout_rules"] = {b: {"worth_keeping": worth_keeping(r["deltas"][f"{b}_vs_C"]),
                                            "not_merely_size": not_size(r)}
                                        for b, r in out["results"].items()}
    OUT.joinpath("selection.json").write_text(json.dumps(out, indent=1, default=_json))
    pd.concat(tabs).to_csv(OUT / "main_models.csv")
    pd.concat(dts).to_csv(OUT / "main_deltas.csv")
    pd.concat(yrs).to_csv(OUT / "main_yearly.csv", index=False)
    pd.concat(oof.values(), keys=oof.keys(), names=["block", "model"]).to_parquet(OUT / "oof_main.parquet")
    return out


def run_holdout(reps: int = BOOT_REPS) -> dict:
    sel = json.loads((OUT / "selection.json").read_text())
    if (OUT / "holdout_2026.json").exists():
        raise SystemExit("2026 has already been scored once; refusing to score it again")
    p = panel()
    res_all = {}
    for name in sel["holdout_models"]:
        sp = sel["blocks"][name]
        need = sorted({c for v in sp.values() for c in v})
        frame = p.dropna(subset=need)
        res, _ = run_block(frame, sp, [HOLDOUT_YEAR], reps)
        assert res["n_oof"] == int(frame["year"].eq(HOLDOUT_YEAR).sum())
        res_all[name] = res
    out = {"frozen_models": sel["holdout_models"], "results": res_all,
           "verdict": verdict(sel["results"], res_all)}
    OUT.joinpath("holdout_2026.json").write_text(json.dumps(out, indent=1, default=_json))
    pd.concat([table(r).assign(block=b) for b, r in res_all.items()]).to_csv(OUT / "holdout_models.csv")
    pd.concat([delta_table(r).assign(block=b) for b, r in res_all.items()]).to_csv(OUT / "holdout_deltas.csv")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--reps", type=int, default=BOOT_REPS)
    a = ap.parse_args(argv)
    pd.set_option("display.width", 250)
    if a.holdout:
        r = run_holdout(a.reps)
        for b, res in r["results"].items():
            print(b, "\n", table(res).round(4).to_string(), "\n", delta_table(res).round(4).to_string())
        print(r["verdict"])
    else:
        r = run_selection(a.reps)
        for b, res in r["results"].items():
            print(b, res["n_oof"], "\n", table(res).round(4).to_string(), "\n",
                  delta_table(res).round(4).to_string())
        print(r["holdout_models"], r["verdict_pre_holdout_rules"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
