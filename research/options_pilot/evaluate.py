"""Phase B — does options information at the call cutoff add to Model C?

Research-only. Reads `output/options_pilot/{event_options,vol_history}.parquet` (built by
`build.py`) and Phase 3's out-of-fold C predictions; writes `output/options_pilot/`.
Every rule used here is fixed in PREREGISTRATION.md.

    PYTHONPATH=. .venv/bin/python -m research.options_pilot.evaluate [--holdout]

`--holdout` scores 2026 YTD. It is run exactly once, after every choice above it is final.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.announcement_timing import AMC, BMO
from research.phase3_refit import calibration as cal
from research.phase3_refit.evaluate import bootstrap, disc_metrics

OUT = Path("output/options_pilot")
PHASE3_OOF = Path("output/phase3_refit/oof_predictions.parquet")
TARGET = "y_extreme"             # abs_reaction_3d_anchored >= 0.08, from the Phase 3 frame
LOG_FLOOR = 1e-4
STALENESS = {"strict": 1, "medium": 3, "broad": 5}
HOLDOUT_YEAR = 2026
LAST_TEST_YEAR = 2025
MIN_TRAIN_ROWS = 1000            # a test year needs this many covered training rows ...
MIN_TRAIN_POS = 150              # ... and this many positives among them
USEFUL_TEST_ROWS = 4000          # a staleness tier is "useful" with this many OOF rows ...
USEFUL_SHARE_OF_BROAD = 0.70     # ... and this share of the broad tier's OOF rows
SECONDARY_IN_COMPACT_MIN_COVERAGE = 0.80
BOOT_REPS = 500

C_INPUTS = ["log_hist_mean_abs", "log_vol_30d"]
PRIMARY = {"em": "log_expected_move", "iv": "log_atm_iv"}
SECONDARY = {"skew": "put_skew", "term": "log_term_ratio", "ivrel": "log_iv_rel_hist"}
DELTA_KEYS = ["roc_auc", "pr_auc", "brier", "log_loss", "top10_hit", "top10_capture",
              "top10_lift", "top20_hit", "top20_capture", "top20_lift"]


def _log(s):
    return np.log(pd.to_numeric(s, errors="coerce").clip(lower=LOG_FLOOR))


# ─────────────────────────────────────────── data ───────────────────────────────────────
def load_panel() -> pd.DataFrame:
    from research.options_pilot.build import iv_relative_history
    ev = pd.read_parquet(OUT / "event_options.parquet")
    vh_path = OUT / "vol_history.parquet"
    if vh_path.exists():
        vh = pd.read_parquet(vh_path)
        rel = [iv_relative_history(s, d, vh) if isinstance(s, str) and pd.notna(d) else {}
               for s, d in zip(ev["source_symbol"], ev["snapshot_date"])]
        ev = pd.concat([ev, pd.DataFrame(rel, index=ev.index)], axis=1)
    ev["log_expected_move"] = _log(ev.get("expected_move"))
    ev["log_atm_iv"] = _log(ev.get("atm_iv"))
    ev["put_skew"] = pd.to_numeric(ev.get("put_skew"), errors="coerce")
    ev["log_term_ratio"] = _log(ev.get("term_ratio"))
    ev["log_iv_rel_hist"] = _log(ev["iv_rel_hist"]) if "iv_rel_hist" in ev else np.nan
    ev["primary_ok"] = ev["status"].eq("ok") & ev["log_expected_move"].notna() \
        & ev["log_atm_iv"].notna()
    # robustness: the live collector's strict-after-report-date expiry rule
    ev["sa_log_expected_move"] = _log(ev.get("sa_expected_move"))
    ev["sa_log_atm_iv"] = _log(ev.get("sa_atm_iv"))
    ev["sa_primary_ok"] = ev.get("sa_status", pd.Series(index=ev.index)).eq("ok") \
        & ev["sa_log_expected_move"].notna() & ev["sa_log_atm_iv"].notna()
    return ev


IV_GATE_MIN_SAME_DAY = 0.80
IV_GATE_MIN_SPEARMAN = 0.70


def iv_history_gate(panel: pd.DataFrame, max_age: int) -> dict:
    """Amendment 2 point 4. Reads no outcome."""
    c = covered(selection_panel(panel), max_age)
    c = c[c["year"] >= 2020]
    if "vh_iv_current" not in c:
        return {"pass": False, "reason": "volatility_history not built"}
    same_day = float(c["vh_iv_current"].notna().mean())
    both = c[["vh_iv_current", "far_atm_iv"]].dropna()
    rho = float(both["vh_iv_current"].rank().corr(both["far_atm_iv"].rank())) if len(both) else np.nan
    ok = same_day >= IV_GATE_MIN_SAME_DAY and rho >= IV_GATE_MIN_SPEARMAN
    return {"pass": bool(ok), "same_day_share": same_day, "spearman_vs_far_atm_iv": rho,
            "n": len(c), "iv_rel_hist_available": float(c["log_iv_rel_hist"].notna().mean())}


def apply_iv_gate(panel: pd.DataFrame, gate: dict) -> pd.DataFrame:
    if not gate["pass"]:
        panel = panel.assign(log_iv_rel_hist=np.nan)
    return panel


def selection_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Everything model/feature selection may see: 2026 rows removed outright."""
    return panel[panel["year"] < HOLDOUT_YEAR].copy()


def covered(panel: pd.DataFrame, max_age: int, flag: str = "primary_ok") -> pd.DataFrame:
    """The options-covered sample at one staleness limit: primary features present."""
    return panel[panel[flag] & panel["snapshot_age"].le(max_age)].copy()


def choose_test_years(cov: pd.DataFrame) -> list[int]:
    """First test year whose strictly-earlier training block is big enough, to 2025."""
    years = []
    for y in range(int(cov["year"].min()) + 1, LAST_TEST_YEAR + 1):
        tr = cov[cal.training_mask(cov, y)]
        if len(tr) >= MIN_TRAIN_ROWS and tr[TARGET].sum() >= MIN_TRAIN_POS or years:
            years.append(y)
    return years


def choose_staleness(panel: pd.DataFrame) -> tuple[str, dict]:
    """Tightest limit that retains a useful sample (counts only — no outcome is read)."""
    info = {}
    broad = covered(panel, STALENESS["broad"])
    yrs = choose_test_years(broad)
    n_broad = int(broad["year"].isin(yrs).sum())
    for name in ("strict", "medium", "broad"):
        cov = covered(panel, STALENESS[name])
        n = int(cov["year"].isin(choose_test_years(cov)).sum())
        info[name] = {"oof_rows": n, "share_of_broad": n / n_broad if n_broad else np.nan,
                      "test_years": choose_test_years(cov)}
    for name in ("strict", "medium", "broad"):
        if info[name]["oof_rows"] >= USEFUL_TEST_ROWS and \
                info[name]["share_of_broad"] >= USEFUL_SHARE_OF_BROAD:
            return name, info
    return "broad", info


# ──────────────────────────────────────── modelling ─────────────────────────────────────
def walk_forward(panel: pd.DataFrame, inputs: list[str], years) -> tuple[pd.DataFrame, dict]:
    """Phase 3's logistic (standardise on training fold, L2 C=1). For test year Y the
    training rows are those whose 3-session endpoint is before 1 January Y."""
    rows, folds = [], {}
    for y in years:
        tr = panel[cal.training_mask(panel, y)]
        te = panel[panel["year"].eq(y)]
        if len(te) == 0 or tr[TARGET].nunique() < 2:
            continue
        fold = cal.fit_model(tr, list(inputs), y)
        p = cal.raw_probability(fold, te)
        rows.append(pd.DataFrame({"event_id": te["event_id"].to_numpy(),
                                  "stock": te["stock"].to_numpy(),
                                  "announce_window": te["announce_window"].to_numpy(),
                                  "test_year": y, "y": te[TARGET].to_numpy(int), "p": p}))
        folds[y] = {"train_n": len(tr), "train_max_endpoint": tr["endpoint3_date"].max(),
                    "coef": dict(zip(inputs, fold.model.coef_[0].round(4).tolist())),
                    "intercept": float(fold.model.intercept_[0])}
    return pd.concat(rows, ignore_index=True), folds


def compare_on_common_sample(panel: pd.DataFrame, specs: dict[str, list[str]], years):
    """Every spec on exactly the rows where the union of their inputs exists."""
    need = sorted({c for v in specs.values() for c in v})
    common = panel.dropna(subset=need)
    out = {name: walk_forward(common, inp, years)[0] for name, inp in specs.items()}
    ref = out[next(iter(out))]["event_id"].to_numpy()
    for p in out.values():
        assert np.array_equal(p["event_id"].to_numpy(), ref)
    return out


def metrics_by(pred: pd.DataFrame) -> dict:
    m = {"ALL": disc_metrics(pred["y"].to_numpy(), pred["p"].to_numpy())}
    for w in (BMO, AMC):
        s = pred[pred["announce_window"] == w]
        m[w] = disc_metrics(s["y"].to_numpy(), s["p"].to_numpy())
    for y, s in pred.groupby("test_year"):
        m[int(y)] = disc_metrics(s["y"].to_numpy(), s["p"].to_numpy())
    return m


def delta_boot(a: pd.DataFrame, b: pd.DataFrame, reps: int = BOOT_REPS) -> dict:
    """Stock-clustered bootstrap of metric(a) - metric(b) on identical rows."""
    assert np.array_equal(a["event_id"].to_numpy(), b["event_id"].to_numpy())
    y, pa, pb = a["y"].to_numpy(), a["p"].to_numpy(), b["p"].to_numpy()

    def stat(idx):
        ma, mb = disc_metrics(y[idx], pa[idx]), disc_metrics(y[idx], pb[idx])
        return {f"d_{k}": ma[k] - mb[k] for k in DELTA_KEYS if k in ma and k in mb}
    return bootstrap(a["stock"].to_numpy(), stat, reps=reps)


# ──────────────────────────────────────── the run ───────────────────────────────────────
def model_specs(cov: pd.DataFrame) -> tuple[dict, dict]:
    """Primary specs on the covered sample; secondary specs on their own sub-samples."""
    em, iv = PRIMARY["em"], PRIMARY["iv"]
    primary = {"C": C_INPUTS, "C+em": C_INPUTS + [em], "C+iv": C_INPUTS + [iv],
               "C+em+iv": C_INPUTS + [em, iv]}
    compact = C_INPUTS + [em, iv]
    coverage = {k: float(cov[c].notna().mean()) if c in cov else 0.0 for k, c in SECONDARY.items()}
    for k, c in SECONDARY.items():
        if coverage[k] >= SECONDARY_IN_COMPACT_MIN_COVERAGE:
            compact.append(c)
    secondary = {k: C_INPUTS + [c] for k, c in SECONDARY.items() if coverage[k] > 0}
    return primary, {"secondary": secondary, "coverage": coverage, "compact": compact}


def flat(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in metrics.items()}


def _block(frame, specs, years, reps, boot, ref="C", extra_pairs=()):
    preds = compare_on_common_sample(frame, specs, years)
    r = {"n_oof": len(preds[ref]), "n_stocks": int(preds[ref]["stock"].nunique()),
         "event_ids_sha": _ids_sha(preds[ref]["event_id"]),
         "metrics": {k: metrics_by(v) for k, v in preds.items()}}
    if boot:
        r["deltas"] = {f"{k}_vs_{ref}": delta_boot(v, preds[ref], reps)
                       for k, v in preds.items() if k != ref}
        for a_, b_ in extra_pairs:
            r["deltas"][f"{a_}_vs_{b_}"] = delta_boot(preds[a_], preds[b_], reps)
    return r, preds


def _ids_sha(ids) -> str:
    import hashlib
    return hashlib.sha256("|".join(sorted(map(str, ids))).encode()).hexdigest()[:16]


def run_tier(panel: pd.DataFrame, tier: str, reps: int, boot: bool = True) -> dict:
    panel = selection_panel(panel)
    cov = covered(panel, STALENESS[tier])
    years = choose_test_years(cov)
    specs, sec = model_specs(cov)
    res = {"tier": tier, "test_years": years, "secondary_coverage": sec["coverage"],
           "compact_inputs": sec["compact"], "covered_rows_all_years": len(cov)}
    res["primary"], preds = _block(cov, specs, years, reps, boot,
                                   extra_pairs=[("C+em+iv", "C+em")])
    res["n_oof"] = res["primary"]["n_oof"]
    # compact set on the rows where all its inputs exist, with C and C+em+iv refit there
    comp_specs = {"C": C_INPUTS, "C+em+iv": specs["C+em+iv"], "C+compact": sec["compact"]}
    res["compact"], cpreds = _block(cov.dropna(subset=sec["compact"]), comp_specs,
                                    choose_test_years(cov.dropna(subset=sec["compact"])),
                                    reps, boot, extra_pairs=[("C+compact", "C+em+iv")])
    preds["C+compact"] = cpreds["C+compact"]
    # each secondary feature on its own rows, C / C+em+iv refit on exactly those rows
    res["secondary"] = {}
    for k, inp in sec["secondary"].items():
        sub = cov.dropna(subset=inp)
        yrs = choose_test_years(sub)
        if len(sub) < 500 or not yrs:
            res["secondary"][k] = {"n": len(sub), "skipped": "too few rows"}
            continue
        s_specs = {"C": C_INPUTS, f"C+{k}": inp, "C+em+iv": specs["C+em+iv"],
                   f"C+em+iv+{k}": specs["C+em+iv"] + [inp[-1]]}
        res["secondary"][k], _ = _block(sub, s_specs, yrs, reps, boot,
                                        extra_pairs=[(f"C+em+iv+{k}", "C+em+iv")])
        res["secondary"][k]["rows_share_of_covered"] = len(sub) / len(cov)
    res["_preds"] = preds
    return res


def run_strict_after(panel: pd.DataFrame, tier: str, reps: int) -> dict:
    """Robustness: expected move / ATM IV under the live collector's expiry rule."""
    panel = selection_panel(panel)
    cov = covered(panel, STALENESS[tier], flag="sa_primary_ok")
    cov = cov.assign(log_expected_move=cov["sa_log_expected_move"], log_atm_iv=cov["sa_log_atm_iv"])
    specs, _ = model_specs(cov)
    r, _ = _block(cov, specs, choose_test_years(cov), reps, True)
    r["rows_differing_near_expiry"] = int((panel["near_expiration"] != panel["sa_near_expiration"])
                                          .where(panel["primary_ok"]).sum())
    return r


WINNER_CANDIDATES = ("C+em", "C+iv", "C+em+iv", "C+compact")


def select_winner(res: dict) -> dict:
    """Amendment 2's pre-registered rule, applied to pre-2026 results only."""
    rows = []
    for m in WINNER_CANDIDATES:
        blk = res["compact"] if m == "C+compact" else res["primary"]
        d = blk["deltas"][f"{m}_vs_C"]
        n_inputs = len(res["compact_inputs"]) if m == "C+compact" else \
            {"C+em": 3, "C+iv": 3, "C+em+iv": 4}[m]
        rows.append({"model": m, "n_inputs": n_inputs,
                     "auc_ci_pos": d["d_roc_auc_lo"] > 0,
                     "capture_ci_pos": d["d_top10_capture_lo"] > 0 or d["d_top20_capture_lo"] > 0,
                     "mean_capture_delta": (d["d_top10_capture"] + d["d_top20_capture"]) / 2,
                     "d_auc": d["d_roc_auc"]})
    t = pd.DataFrame(rows)
    q = t[t["auc_ci_pos"] & t["capture_ci_pos"]]
    if len(q):
        best = q["mean_capture_delta"].max()
        pick = q[q["mean_capture_delta"] >= best - 0.002].sort_values("n_inputs").iloc[0]
        basis = "qualified: AUC CI > 0 and a capture CI > 0"
    else:
        q = t[t["auc_ci_pos"]]
        if len(q):
            pick = q.sort_values(["d_auc", "n_inputs"], ascending=[False, True]).iloc[0]
            basis = "no capture gain: best AUC among AUC CI > 0 (weak)"
        else:
            pick = t[t["model"] == "C+em+iv"].iloc[0]
            basis = "nothing qualified: default primary pair C+em+iv, for information"
    return {"winner": pick["model"], "basis": basis, "table": t.to_dict(orient="records")}


def phase3_reference(cov: pd.DataFrame, years) -> dict:
    """Coverage bias: Phase 3's own OOF C (trained on full Benzinga history) on all its
    rows in these years vs on the covered rows."""
    oof = pd.read_parquet(PHASE3_OOF)
    oof["event_id"] = oof["stock"] + "|" + pd.to_datetime(oof["earnings_date"]).dt.strftime("%Y-%m-%d")
    oof = oof[oof["year"].isin(years)]
    col = "p__C_structural_vol__W0__call"
    ins = oof["event_id"].isin(cov["event_id"])
    out = {}
    for name, s in (("phase3_all", oof), ("phase3_covered", oof[ins]),
                    ("phase3_uncovered", oof[~ins])):
        out[name] = disc_metrics(s["y_extreme"].to_numpy(int), s[col].to_numpy())
        out[name]["bmo_share"] = float((s["announce_window"] == BMO).mean())
    return out


# Stocks with stock_data.status <> 'active' in the local DB on 2026-09-30, in this
# population. Frozen here because research/ may not import a database driver
# (testing/test_massive_earnings.py guards that statically).
INACTIVE_2026_09_30 = frozenset({
    "AVB", "BK", "BLDR", "CAG", "CTRA", "DAY", "EA", "EPAM", "EQR", "HOLX", "LW", "MOH",
    "MTCH", "PAYC", "POOL", "TAP", "TTD"})


def inactive_stocks() -> set[str]:
    return set(INACTIVE_2026_09_30)


def composition(panel_all: pd.DataFrame, cov: pd.DataFrame, years) -> pd.DataFrame:
    a = panel_all[panel_all["year"].isin(years)]
    c = cov[cov["year"].isin(years)]
    u = a[~a["event_id"].isin(c["event_id"])]
    inactive = inactive_stocks()
    rows = []
    for name, s in (("all_events", a), ("covered", c), ("not_covered", u)):
        share = s["stock"].value_counts(normalize=True)
        row = {"sample": name, "n": len(s), "stocks": s["stock"].nunique(),
               "base_rate": s[TARGET].mean(), "bmo_share": (s["announce_window"] == BMO).mean(),
               "top10_stock_share": share.head(10).sum(), "stock_hhi": float((share ** 2).sum()),
               "inactive_name_share": s["stock"].isin(inactive).mean(),
               "median_hist_mean_abs": s["hist_mean_abs_call"].median(),
               "median_vol_30d": s["vol_30d_call"].median()}
        for y, v in s["year"].value_counts(normalize=True).sort_index().items():
            row[f"year_{int(y)}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def sector_mix(panel_all, cov, years) -> pd.DataFrame:
    a = panel_all[panel_all["year"].isin(years)]
    c = cov[cov["year"].isin(years)]
    return pd.DataFrame({"all": a["sector"].value_counts(normalize=True),
                         "covered": c["sector"].value_counts(normalize=True)}).fillna(0)


def holdout(panel: pd.DataFrame, tier: str) -> dict:
    """2026 scored once: models trained on every covered row with endpoint < 2026-01-01.
    Refuses to run until the winner has been frozen from pre-2026 results."""
    frozen = json.loads((OUT / "winner.json").read_text())
    cov = covered(panel, STALENESS[tier])
    specs, sec = model_specs(cov[cov["year"] < HOLDOUT_YEAR])
    assert sec["compact"] == frozen["compact_inputs"], "compact set changed after freezing"
    out = {"frozen": frozen}
    r, _ = _block(cov, specs, [HOLDOUT_YEAR], BOOT_REPS, True, extra_pairs=[("C+em+iv", "C+em")])
    out["primary"] = r
    comp = {"C": C_INPUTS, "C+em+iv": specs["C+em+iv"], "C+compact": frozen["compact_inputs"]}
    out["compact"], _ = _block(cov.dropna(subset=frozen["compact_inputs"]), comp,
                               [HOLDOUT_YEAR], BOOT_REPS, True)
    return out


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(type(o))


def tables(summary: dict) -> None:
    """Flat, machine-readable tables of every block."""
    pooled, yearly, deltas = [], [], []
    def add(tier, block, blk):
        for model, m in blk["metrics"].items():
            for key, v in m.items():
                row = {"tier": tier, "block": block, "model": model, "slice": str(key), **v}
                (yearly if isinstance(key, int) else pooled).append(row)
        for name, d in blk.get("deltas", {}).items():
            deltas.append({"tier": tier, "block": block, "comparison": name, **d})
    for tier in STALENESS:
        r = summary[tier]
        add(tier, "primary", r["primary"])
        add(tier, "compact", r["compact"])
        for k, v in r["secondary"].items():
            if "metrics" in v:
                add(tier, f"secondary_{k}", v)
    add(summary["main_tier"], "strict_after_robustness", summary["strict_after"])
    pd.DataFrame(pooled).to_csv(OUT / "metrics_pooled.csv", index=False)
    pd.DataFrame(yearly).to_csv(OUT / "metrics_yearly.csv", index=False)
    pd.DataFrame(deltas).to_csv(OUT / "deltas_bootstrap.csv", index=False)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--reps", type=int, default=BOOT_REPS)
    a = ap.parse_args(argv)
    panel = load_panel()
    tier, info = choose_staleness(selection_panel(panel))
    gate = iv_history_gate(panel, STALENESS[tier])
    panel = apply_iv_gate(panel, gate)
    if a.holdout:
        res = holdout(panel, tier)
        (OUT / "holdout_2026.json").write_text(json.dumps(res, indent=1, default=_json))
        return 0
    summary = {"main_tier": tier, "staleness_info": info, "iv_history_gate": gate}
    print("iv gate", gate, flush=True)
    for t in ("strict", "medium", "broad"):
        r = run_tier(panel, t, a.reps, boot=True)
        preds = r.pop("_preds")
        pd.concat([v.assign(model=k) for k, v in preds.items()]).to_parquet(OUT / f"oof_{t}.parquet")
        summary[t] = r
        print(t, r["n_oof"], flush=True)
    summary["strict_after"] = run_strict_after(panel, tier, a.reps)
    sel = selection_panel(panel)
    cov = covered(sel, STALENESS[tier])
    yrs = summary[tier]["test_years"]
    summary["phase3_reference"] = phase3_reference(cov, yrs)
    composition(sel, cov, yrs).to_csv(OUT / "composition.csv", index=False)
    sector_mix(sel, cov, yrs).to_csv(OUT / "sector_mix.csv")
    summary["winner"] = select_winner(summary[tier])
    frozen = {"winner": summary["winner"]["winner"], "basis": summary["winner"]["basis"],
              "main_tier": tier, "max_snapshot_age": STALENESS[tier],
              "compact_inputs": summary[tier]["compact_inputs"],
              "inputs": {**model_specs(cov)[0], "C+compact": summary[tier]["compact_inputs"]},
              "transforms": "log for expected move, ATM IV, term ratio, IV-rel-hist; skew raw",
              "missing": "rows lacking any input of a model are not scored by it (no imputation)",
              "model": "logistic, standardised on training fold, L2 C=1, yearly refit"}
    (OUT / "winner.json").write_text(json.dumps(frozen, indent=1, default=_json))
    tables(summary)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
