"""Model C production-frame refit. Rules: SPEC.md.

    PYTHONPATH=. .venv/bin/python -m analysis.model_c_refit.evaluate build
    PYTHONPATH=. .venv/bin/python -m analysis.model_c_refit.evaluate pre2026
    PYTHONPATH=. .venv/bin/python -m analysis.model_c_refit.evaluate holdout   # needs FROZEN.json
    PYTHONPATH=. .venv/bin/python -m analysis.model_c_refit.evaluate compare

Writes only under output/model_c_refit/ (gitignored). Touches no production table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.model_c_refit import frame as F
from analysis.model_c_refit import model as M

OUT = Path("output/model_c_refit")
FRAME = OUT / "frame.parquet"
SPEC = Path(__file__).with_name("SPEC.md")
FROZEN = OUT / "FROZEN.json"
ARCHIVED_FRAME = Path("output/phase3_refit/feature_frame.parquet")
ARCHIVED_OOF = Path("output/phase3_refit/oof_predictions.parquet")
ARCHIVED_CURRENT = Path("output/sec_features/feature_frame_current.parquet")


def _spec_sha() -> str:
    return hashlib.sha256(SPEC.read_bytes()).hexdigest()


def _dump(obj, name):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))


# ────────────────────────────────────────── build ───────────────────────────────────────
def build() -> pd.DataFrame:
    events, daily = F.load_production_inputs()
    frame = F.build_frame(events, daily)
    exact = F.vol_at_cutoff(daily, events, F.event_clocks(events, F.market_session_grid(daily)),
                            tolerance=None)["vol_30d"]
    frame["vol_30d_exact"] = exact
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(FRAME)
    funnel = F.exclusion_counts(frame)
    funnel.to_csv(OUT / "funnel.csv", index=False)
    print(funnel.to_string(index=False))
    return frame


def load() -> pd.DataFrame:
    return pd.read_parquet(FRAME)


# ───────────────────────────────────────── pre-2026 ─────────────────────────────────────
def _pre2026(frame: pd.DataFrame) -> pd.DataFrame:
    """The only rows the pre-2026 stage may see: nothing reported in 2026 or later."""
    return frame[frame["year"] < M.HOLDOUT_YEAR]


def run_models(sample: pd.DataFrame) -> pd.DataFrame:
    preds = pd.DataFrame(index=sample.index)
    for name in M.MODELS:
        oof, _ = M.walk_forward(sample, name)
        preds[f"p_{name}"] = oof["p"]
        preds["test_year"] = oof["test_year"]
    preds["p_A0"] = M.platt_walk_forward(sample, "risk_score")
    preds = preds.dropna(subset=["test_year"])
    return preds.join(sample[["stock", "y", "year", "announce_window", "call_monday"]])


def table(preds: pd.DataFrame, by: str | None = None, models=("A0", "B", "C")) -> pd.DataFrame:
    rows = []
    groups = [("ALL", preds)] if by is None else list(preds.groupby(by))
    for key, g in groups:
        for m in models:
            rows.append({"group": key, "model": m, **M.metrics(g["y"], g[f"p_{m}"])})
    return pd.DataFrame(rows)


def week_table(preds: pd.DataFrame, models=("A0", "B", "C")) -> pd.DataFrame:
    rows = []
    for m in models:
        for f in (0.10, 0.20):
            r = M.within_week_top(preds["y"], preds[f"p_{m}"], preds["call_monday"], f)
            rows.append({"model": m, "frac": f, **r})
    return pd.DataFrame(rows)


def pre2026() -> dict:
    frame = load()
    pre = _pre2026(frame)
    sample = pre[F.common_mask(pre)]
    preds = run_models(sample)
    preds.to_parquet(OUT / "oof_predictions.parquet")

    pooled = table(preds)
    yearly = table(preds, "test_year")
    window = table(preds, "announce_window")
    weekly = week_table(preds)
    for name, t in [("pooled", pooled), ("yearly", yearly), ("by_window", window),
                    ("within_week", weekly)]:
        t.to_csv(OUT / f"{name}.csv", index=False)

    boot = {"C_minus_B": M.bootstrap_delta(preds["y"], preds["p_C"], preds["p_B"], preds["stock"]),
            "C_minus_A0": M.bootstrap_delta(preds["y"], preds["p_C"], preds["p_A0"], preds["stock"])}
    for w, g in preds.groupby("announce_window"):
        boot[f"C_minus_B_{w}"] = M.bootstrap_delta(g["y"], g["p_C"], g["p_B"], g["stock"])
    _dump(boot, "bootstrap.json")

    cal = {"reliability_C": M.reliability(preds["y"], preds["p_C"]).to_dict("records"),
           "reliability_B": M.reliability(preds["y"], preds["p_B"]).to_dict("records"),
           "line_C": M.calibration_line(preds["y"], preds["p_C"]),
           "line_B": M.calibration_line(preds["y"], preds["p_B"]),
           "by_year_C": [{"year": int(y), **M.calibration_line(g["y"], g["p_C"])}
                         for y, g in preds.groupby("test_year")],
           "by_window_C": [{"window": w, **M.calibration_line(g["y"], g["p_C"])}
                           for w, g in preds.groupby("announce_window")]}
    _dump(cal, "calibration_oof.json")

    # sensitivities — reported, never used to change the specification
    sens = []
    for label, mask in [("MIN_PRIOR=4", F.common_mask(pre, 4)), ("MIN_PRIOR=12", F.common_mask(pre, 12)),
                        ("vol exact-date only", F.common_mask(pre) & pre["vol_30d_exact"].notna())]:
        s = pre[mask].copy()
        if label.startswith("vol"):
            s["log_vol_30d"] = np.log(s["vol_30d_exact"].clip(lower=F.LOG_FLOOR))
        p = run_models(s)
        b = M.bootstrap_delta(p["y"], p["p_C"], p["p_B"], p["stock"])
        for m in ("B", "C"):
            sens.append({"variant": label, "model": m, **M.metrics(p["y"], p[f"p_{m}"])})
        sens.append({"variant": label, "model": "C-B AUC [CI]",
                     "roc_auc": b["roc_auc"]["delta"], "pr_auc": b["roc_auc"]["lo"],
                     "brier": b["roc_auc"]["hi"]})
    # B on its own larger sample (vol not required)
    b_only = pre[F.population_mask(pre)]
    oof_b, _ = M.walk_forward(b_only, "B")
    yb = b_only.loc[oof_b.index, "y"]
    sens.append({"variant": "B without the vol requirement", "model": "B", **M.metrics(yb, oof_b["p"])})
    pd.DataFrame(sens).to_csv(OUT / "sensitivity.csv", index=False)

    summary = {"common_sample_pre2026": int(len(sample)), "oof_rows": int(len(preds)),
               "oof_years": sorted(int(y) for y in preds["test_year"].unique()),
               "oof_by_window": preds["announce_window"].value_counts().to_dict(),
               "stocks": int(preds["stock"].nunique())}
    _dump(summary, "summary_pre2026.json")
    FROZEN.write_text(json.dumps({"spec_sha256": _spec_sha(),
                                  "frozen_after": "pre-2026 walk-forward written",
                                  "holdout_year": M.HOLDOUT_YEAR}, indent=1))
    print(pooled.to_string(index=False))
    print(json.dumps({k: v["roc_auc"] for k, v in boot.items()}, indent=1))
    return summary


# ────────────────────────────────────────── holdout ─────────────────────────────────────
def final_fits(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The candidate production fit: every common-sample row with endpoint < holdout year."""
    sample = frame[F.common_mask(frame)]
    train = sample[M.training_mask(sample, M.HOLDOUT_YEAR)]
    assert (train["year"] < M.HOLDOUT_YEAR).all()
    end = pd.Timestamp(year=M.HOLDOUT_YEAR, month=1, day=1)
    return train, {name: M.fit(train, M.MODELS[name], end) for name in M.MODELS}


def holdout() -> dict:
    if not FROZEN.exists():
        raise SystemExit("FROZEN.json missing: run `pre2026` first; the holdout is scored once.")
    frozen = json.loads(FROZEN.read_text())
    if frozen["spec_sha256"] != _spec_sha():
        raise SystemExit("SPEC.md changed after the specification was frozen.")
    frame = load()
    train, fits = final_fits(frame)
    sample = frame[F.common_mask(frame)]
    test = sample[sample["year"].eq(M.HOLDOUT_YEAR)].copy()
    for name, f in fits.items():
        test[f"p_{name}"] = f.predict(test)
    # A0 reference: Platt map on the same training rows
    from sklearn.linear_model import LogisticRegression
    a0 = LogisticRegression(C=1e6, max_iter=M.MAX_ITER).fit(
        train[["risk_score"]].to_numpy(float), train["y"].to_numpy(int))
    test["p_A0"] = a0.predict_proba(test[["risk_score"]].fillna(train["risk_score"].median())
                                    .to_numpy(float))[:, 1]
    test.to_parquet(OUT / "holdout_2026_predictions.parquet")
    res = {"n": int(len(test)), "date_range": [str(test["report_date"].min().date()),
                                               str(test["report_date"].max().date())],
           "pooled": table(test).to_dict("records"),
           "by_window": table(test, "announce_window").to_dict("records"),
           "within_week": week_table(test).to_dict("records"),
           "C_minus_B": M.bootstrap_delta(test["y"], test["p_C"], test["p_B"], test["stock"]),
           "calibration_C": M.calibration_line(test["y"], test["p_C"]),
           "reliability_C": M.reliability(test["y"], test["p_C"], bins=5).to_dict("records"),
           "fit_C": fits["C"].coefficients(), "fit_B": fits["B"].coefficients(),
           "train_report_years": [int(train["year"].min()), int(train["year"].max())]}
    _dump(res, "holdout_2026.json")
    print(table(test).to_string(index=False))
    print(json.dumps(res["fit_C"], indent=1))
    return res


# ────────────────────────────────────────── compare ─────────────────────────────────────
def compare() -> dict:
    """Old research sample vs the production sample, and why they differ."""
    new = load()
    new_pop = new[F.common_mask(new)]
    out = {}
    for label, path in [("archived_2026_09_28", ARCHIVED_FRAME),
                        ("archived_rules_on_current_data", ARCHIVED_CURRENT)]:
        if not path.exists():
            continue
        old = pd.read_parquet(path)
        old_pop = old[~old["is_pending"].astype(bool) & old["window_ok"] & old["y_extreme"].notna()
                      & old["year"].between(*F.POP_YEARS) & old["n_prior_call"].ge(F.MIN_PRIOR)
                      & old["hist_mean_abs_call"].notna() & old["vol_30d_call"].notna()]
        key = ["stock", "earnings_date"]
        o = old_pop[key + ["y_extreme", "hist_mean_abs_call", "n_prior_call", "call_cutoff_date",
                           "announce_window", "year"]].copy()
        n = new_pop[key + ["y", "hist_mean_abs", "n_prior", "cutoff_date", "announce_window"]].copy()
        o["earnings_date"] = pd.to_datetime(o["earnings_date"])
        n["earnings_date"] = pd.to_datetime(n["earnings_date"])
        both = o.merge(n, on=key, how="outer", indicator=True, suffixes=("_old", "_new"))
        only_old = both[both["_merge"] == "left_only"][key]
        only_new = both[both["_merge"] == "right_only"][key]
        shared = both[both["_merge"] == "both"]

        # Why each old-only event is missing from the production sample.
        nf = new.set_index(key)
        reasons = []
        for s, d in only_old.itertuples(index=False):
            if (s, d) not in nf.index:
                reasons.append("event not in production frame (date or identity differs)")
                continue
            r = nf.loc[(s, d)]
            r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            st = r[F.TARGET_STATUS]
            if st != "available":
                reasons.append(f"production target: {st}")
            elif r["n_prior"] < F.MIN_PRIOR:
                reasons.append("production history < 8 usable prior outcomes")
            elif not (r["vol_30d"] > 0):
                reasons.append("production vol_30d unavailable")
            elif not (F.POP_YEARS[0] <= r["year"] <= F.POP_YEARS[1]):
                reasons.append("outside population years")
            else:
                reasons.append("other")
        of = old.set_index(key)
        new_reasons = []
        for s, d in only_new.itertuples(index=False):
            if (s, d) not in of.index:
                new_reasons.append("event not in archived frame")
                continue
            r = of.loc[(s, d)]
            r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
            if pd.isna(r["y_extreme"]):
                new_reasons.append("archived target unavailable (no vendor match / unresolved)")
            elif not r["window_ok"]:
                new_reasons.append("archived window not BMO/AMC")
            elif r["n_prior_call"] < F.MIN_PRIOR:
                new_reasons.append("archived history < 8")
            elif pd.isna(r["vol_30d_call"]):
                new_reasons.append("archived vol_30d unavailable")
            else:
                new_reasons.append("other")
        out[label] = {
            "archived_sample": int(len(o)), "production_sample": int(len(n)),
            "shared": int(len(shared)), "archived_only": int(len(only_old)),
            "production_only": int(len(only_new)),
            "archived_only_reasons": pd.Series(reasons).value_counts().to_dict() if reasons else {},
            "production_only_reasons": pd.Series(new_reasons).value_counts().to_dict() if new_reasons else {},
            "shared_label_disagree": int((shared["y_extreme"] != shared["y"]).sum()),
            "shared_window_disagree": int((shared["announce_window_old"] != shared["announce_window_new"]).sum()),
            "shared_cutoff_disagree": int((pd.to_datetime(shared["call_cutoff_date"])
                                           != pd.to_datetime(shared["cutoff_date"])).sum()),
            "shared_hist_abs_diff_median": float((shared["hist_mean_abs_call"] - shared["hist_mean_abs"]).abs().median()),
            "shared_hist_abs_diff_p99": float((shared["hist_mean_abs_call"] - shared["hist_mean_abs"]).abs().quantile(0.99)),
            "shared_n_prior_disagree": int((shared["n_prior_call"] != shared["n_prior"]).sum()),
        }
    if ARCHIVED_OOF.exists():
        oof = pd.read_parquet(ARCHIVED_OOF)
        out["archived_oof_metrics"] = {
            m: M.metrics(oof["y_extreme"], oof[f"p__{c}__W0__call"])
            for m, c in [("B", "B_structural"), ("C", "C_structural_vol")]}
        # the archived OOF rows restricted to events also in the production sample
        preds = pd.read_parquet(OUT / "oof_predictions.parquet")
        oof["earnings_date"] = pd.to_datetime(oof["earnings_date"])
        pk = preds.join(new[["earnings_date"]], rsuffix="_e")
        pk["earnings_date"] = pd.to_datetime(new.loc[pk.index, "earnings_date"])
        j = oof.merge(pk[["stock", "earnings_date", "p_B", "p_C", "y"]], on=["stock", "earnings_date"])
        out["shared_oof_rows"] = int(len(j))
        out["shared_oof_metrics"] = {
            "archived_B": M.metrics(j["y_extreme"], j["p__B_structural__W0__call"]),
            "archived_C": M.metrics(j["y_extreme"], j["p__C_structural_vol__W0__call"]),
            "production_B": M.metrics(j["y"], j["p_B"]),
            "production_C": M.metrics(j["y"], j["p_C"])}
    _dump(out, "compare.json")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict)}
                      if isinstance(v, dict) else v for k, v in out.items()}, indent=1, default=str))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "pre2026", "holdout", "compare"])
    a = ap.parse_args(argv)
    {"build": build, "pre2026": pre2026, "holdout": holdout, "compare": compare}[a.stage]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
