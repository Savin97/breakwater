"""Tests for the point-in-time fundamentals test (`research/fundamentals/`).
Synthetic fixtures only; no network, no SEC data on disk needed."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.fundamentals import acquire, audit, build, evaluate
from research.sec_filings_pilot import build as pilot_build
from research.sec_filings_pilot import paths as pilot_paths

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "research" / "fundamentals"
T = pd.Timestamp
QEND = {1: "03-31", 2: "06-30", 3: "09-30", 4: "12-31"}
QSTART = {1: "01-01", 2: "04-01", 3: "07-01", 4: "10-01"}


# ───────────────────────────────────── synthetic issuer ─────────────────────────────────
def rev(y, q):
    return 1000.0 + 100 * (y - 2019) + 10 * q + (q == 4) * 50 + (y % 2) * 7 * q


def oi(y, q):
    return rev(y, q) * (0.10 + 0.01 * q + 0.005 * (y - 2019))


def cfo_q(y, q):
    return 80.0 + 3 * q + (y - 2019)


def ni_q(y, q):
    return 60.0 + 2 * q + 2 * (y - 2019)


def assets(y, q):
    return 5000.0 + 100 * (y - 2019) + 10 * q


def _f(rows, tag, start, end, val, acc, filed, unit="USD", cik=1):
    rows.append({"tag": tag, "unit": unit, "start": T(start) if start else pd.NaT, "end": T(end),
                 "val": float(val), "accession": acc, "fact_form": None, "fact_filed": T(filed),
                 "fp": None, "fy": None, "cik": cik, "concept": build.ALL_TAGS.get(tag),
                 "snapshot_id": "test"})


def acc_of(y, q):
    return f"0000000001-{y % 100:02d}-0000{q}"


def filed_of(y, q):
    return T(f"{y + 1}-02-20") if q == 4 else T(f"{y}-{QEND[q]}") + pd.Timedelta(days=40)


def issuer(years=range(2019, 2024), cik=1, stock="TST", skip=(), extra=None):
    """Facts + filings of a calendar-year filer: Q1-Q3 10-Q, 10-K; every filing carries its
    prior-year comparatives, 10-Qs report 3-month and YTD, cash flow YTD only."""
    facts, fil = [], []
    for y in years:
        for q in (1, 2, 3, 4):
            if (y, q) in skip:
                continue
            a, fd, P = acc_of(y, q), filed_of(y, q), T(f"{y}-{QEND[q]}")
            form = "10-K" if q == 4 else "10-Q"
            fil.append({"stock": stock, "cik": cik, "accession": a, "form": form,
                        "filing_date": fd, "report_period": P, "mapping_provenance": "test"})
            for yy in (y, y - 1):               # current and prior-year comparative
                s3, e3 = f"{yy}-{QSTART[q]}", f"{yy}-{QEND[q]}"
                if q < 4:
                    _f(facts, "Revenues", s3, e3, rev(yy, q), a, fd, cik=cik)
                    _f(facts, "OperatingIncomeLoss", s3, e3, oi(yy, q), a, fd, cik=cik)
                if q in (2, 3):                 # YTD revenue (6M / 9M)
                    _f(facts, "Revenues", f"{yy}-01-01", e3, sum(rev(yy, k) for k in range(1, q + 1)), a, fd, cik=cik)
                    _f(facts, "OperatingIncomeLoss", f"{yy}-01-01", e3,
                       sum(oi(yy, k) for k in range(1, q + 1)), a, fd, cik=cik)
                if q == 4:                      # annual only
                    _f(facts, "Revenues", f"{yy}-01-01", e3, sum(rev(yy, k) for k in range(1, 5)), a, fd, cik=cik)
                    _f(facts, "OperatingIncomeLoss", f"{yy}-01-01", e3,
                       sum(oi(yy, k) for k in range(1, 5)), a, fd, cik=cik)
                _f(facts, "NetCashProvidedByUsedInOperatingActivities", f"{yy}-01-01", e3,
                   sum(cfo_q(yy, k) for k in range(1, q + 1)), a, fd, cik=cik)
                _f(facts, "NetIncomeLoss", f"{yy}-01-01", e3, sum(ni_q(yy, k) for k in range(1, q + 1)),
                   a, fd, cik=cik)
            _f(facts, "Assets", None, P, assets(y, q), a, fd, cik=cik)
            _f(facts, "Liabilities", None, P, 0.6 * assets(y, q), a, fd, cik=cik)
            _f(facts, "AssetsCurrent", None, P, 0.3 * assets(y, q), a, fd, cik=cik)
            _f(facts, "LiabilitiesCurrent", None, P, (0.2 + 0.01 * q) * assets(y, q), a, fd, cik=cik)
    facts = pd.DataFrame(facts)
    if extra is not None:
        facts = pd.concat([facts, pd.DataFrame(extra)], ignore_index=True)
    return facts, pd.DataFrame(fil)


def records(facts, fil):
    return build.derive_q4(build.quarter_records(build.filing_facts(facts, fil), workers=1))


def event(stock, acc, filed, cutoff, form="10-Q", eid=None):
    return pd.DataFrame([{"event_id": eid or f"{stock}|{cutoff}", "stock": stock,
                          "identity_status": "mapped", "periodic_accession": acc,
                          "periodic_form": form, "periodic_filing_date": T(filed),
                          "call_cutoff_date": T(cutoff)}])


def rec(qr, y, q):
    return qr[qr["accession"].eq(acc_of(y, q))].iloc[0]


# ───────────────────────────────────── period selection ─────────────────────────────────
def test_quarter_value_is_the_three_month_fact_not_ytd():
    qr = records(*issuer())
    r = rec(qr, 2022, 2)
    assert r["revenue_q"] == rev(2022, 2) and r["revenue_q_route"] == "own_quarter"
    assert r["revenue_q_prior"] == rev(2021, 2)             # same filing's comparative
    assert r["revenue_q"] != rev(2022, 1) + rev(2022, 2)    # never the 6-month YTD


def test_q4_is_fy_minus_9m_of_the_same_years_q3_10q():
    qr = records(*issuer())
    r = rec(qr, 2022, 4)
    assert r["revenue_q_route"] == "fy_minus_9m"
    assert r["revenue_q"] == pytest.approx(rev(2022, 4))
    assert r["revenue_q_prior"] == pytest.approx(rev(2021, 4))
    assert r["revenue_q_q4_source_accession"] == acc_of(2022, 3)


def test_annual_value_never_enters_as_a_quarter():
    qr = records(*issuer(skip={(2022, 3)}))           # no Q3 10-Q -> no Q4 derivation
    r = rec(qr, 2022, 4)
    assert np.isnan(r["revenue_q"]) and r["revenue_fy"] == pytest.approx(sum(rev(2022, k) for k in range(1, 5)))


def test_ytd_cash_flow_is_never_a_quarter_and_ttm_is_built_correctly():
    facts, fil = issuer()
    qr = records(facts, fil)
    assert not any(c.startswith("cfo_q") for c in qr.columns)
    r = rec(qr, 2022, 2)
    assert r["cfo_ytd"] == cfo_q(2022, 1) + cfo_q(2022, 2) and r["cfo_ytd_dur"] > 150
    ev = event("TST", acc_of(2022, 2), filed_of(2022, 2), "2022-10-21")
    f = build.event_features(ev, qr).iloc[0]
    ttm = sum(cfo_q(2022, k) for k in (1, 2)) + sum(cfo_q(2021, k) for k in (3, 4))
    assert f["cfo_ttm"] == pytest.approx(ttm)
    ni = sum(ni_q(2022, k) for k in (1, 2)) + sum(ni_q(2021, k) for k in (3, 4))
    assert f["abs_accruals"] == pytest.approx(abs(ni - ttm) / assets(2022, 2))
    assert f["ttm_prior_10k_accession"] == acc_of(2021, 4)


def test_ttm_refuses_to_mix_cash_flow_tags():
    facts, fil = issuer()
    k = facts["accession"].eq(acc_of(2021, 4)) & facts["tag"].eq("NetCashProvidedByUsedInOperatingActivities")
    facts.loc[k, "tag"] = "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"
    facts.loc[k, "concept"] = "cfo"
    qr = records(facts, fil)
    f = build.event_features(event("TST", acc_of(2022, 2), filed_of(2022, 2), "2022-10-21"), qr).iloc[0]
    assert np.isnan(f["cfo_ttm"]) and np.isnan(f["abs_accruals"])


def test_units_other_than_usd_are_ignored():
    facts, fil = issuer()
    eur = facts[facts["accession"].eq(acc_of(2022, 2)) & facts["tag"].eq("Revenues")].copy()
    facts = facts[~(facts["accession"].eq(acc_of(2022, 2)) & facts["tag"].eq("Revenues"))]
    eur["unit"] = "EUR"
    qr = records(pd.concat([facts, eur]), fil)
    assert np.isnan(rec(qr, 2022, 2)["revenue_q"])


def test_conflicting_duplicate_context_is_missing_not_picked():
    extra = []
    _f(extra, "Assets", None, "2022-06-30", 1.0, acc_of(2022, 2), filed_of(2022, 2))
    qr = records(*issuer(extra=extra))
    assert np.isnan(rec(qr, 2022, 2)["assets"])


# ───────────────────────────────────── point in time ────────────────────────────────────
def test_exact_accession_only():
    """A value for the same period under another accession (not an original periodic
    filing of this stock) never reaches the record."""
    extra = []
    _f(extra, "Revenues", "2022-04-01", "2022-06-30", 9e9, "0000000001-25-00099", "2025-03-01")
    qr = records(*issuer(extra=extra))
    assert rec(qr, 2022, 2)["revenue_q"] == rev(2022, 2)
    assert "0000000001-25-00099" not in set(qr["accession"])


def test_later_restatement_cannot_leak_backward():
    """The 2023 Q2 10-Q restates 2022 Q2 revenue in its comparative column. The 2022 Q2
    record keeps the value as filed; only the 2023 record's YoY uses the restated basis."""
    facts, fil = issuer()
    k = (facts["accession"].eq(acc_of(2023, 2)) & facts["tag"].eq("Revenues")
         & facts["start"].eq(T("2022-04-01")) & facts["end"].eq(T("2022-06-30")))
    facts.loc[k, "val"] = 777.0
    qr = records(facts, fil)
    assert rec(qr, 2022, 2)["revenue_q"] == rev(2022, 2)
    assert rec(qr, 2023, 2)["revenue_q_prior"] == 777.0
    f = build.event_features(event("TST", acc_of(2022, 2), filed_of(2022, 2), "2022-10-21"), qr).iloc[0]
    assert f["revenue_q"] == rev(2022, 2)


def test_history_never_contains_a_filing_after_the_selected_one():
    qr = records(*issuer())
    f = build.event_features(event("TST", acc_of(2021, 3), filed_of(2021, 3), "2022-01-21"), qr).iloc[0]
    assert f["known_at"] == filed_of(2021, 3)
    assert f["n_quarters_available"] == int((qr["filing_date"] <= filed_of(2021, 3)).sum())
    assert f["period_end"] == T("2021-09-30")


def test_xbrl_dated_after_the_cutoff_cannot_enter():
    facts, fil = issuer()
    a = acc_of(2022, 2)
    facts.loc[facts["accession"].eq(a), "fact_filed"] = T("2022-10-21")   # CF says later
    qr = records(facts, fil)
    f = build.event_features(event("TST", a, filed_of(2022, 2), "2022-10-21"), qr).iloc[0]
    assert f["status"] == "xbrl_filed_date_after_cutoff"


def test_eligibility_check_refuses_rows_known_after_the_cutoff(tmp_path, monkeypatch):
    pd.DataFrame({"accession": ["x"], "acceptance_verified_ny": [pd.NaT]}).to_parquet(
        tmp_path / "verified_acceptance.parquet")
    monkeypatch.setattr(build.pilot_paths, "OUT", tmp_path)
    feat = pd.DataFrame({"event_id": ["e"], "status": ["ok"], "known_at": [T("2022-10-21")],
                         "call_cutoff_date": [T("2022-10-21")], "selected_accession": ["a"]})
    ev = pd.DataFrame({"event_id": ["e"], "call_cutoff_ts": [T("2022-10-21 16:00")]})
    with pytest.raises(AssertionError):
        build.check_eligibility(feat, ev)
    feat["known_at"] = T("2022-10-20")
    assert build.check_eligibility(feat, ev) == 1


def test_predecessor_chain_is_respected():
    """Old issuer (CIK 1) until 2021-07-01, successor (CIK 2) after. The pilot's
    `stock_filings` decides which accessions belong to the stock; the old CIK's later
    filings (it keeps filing as a subsidiary) never reach it."""
    old_f, old_fil = issuer(years=range(2019, 2023), cik=1, stock="OLD")
    new_f, new_fil = issuer(years=range(2021, 2024), cik=2, stock="NEW")
    new_f["accession"] = new_f["accession"].str.replace("0000000001", "0000000002")
    new_fil["accession"] = new_fil["accession"].str.replace("0000000001", "0000000002")
    allfil = pd.concat([old_fil, new_fil]).drop(columns=["stock", "mapping_provenance"])
    allfil["base_form"] = allfil["form"]
    seg = pd.DataFrame({"stock": ["TST", "TST"], "cik": [1, 2],
                        "valid_from": [T("1990-01-01"), T("2021-07-01")],
                        "valid_to": [T("2021-07-01"), T("2099-12-31")],
                        "provenance": ["manual_verified"] * 2, "sec_name": ["Old", "New"],
                        "reason": ["pred", "succ"]})
    sf = pilot_build.stock_filings(allfil, seg)[["stock", "cik", "accession", "form", "filing_date",
                                                 "report_period", "mapping_provenance"]]
    qr = records(pd.concat([old_f, new_f]), sf)
    assert set(qr.loc[qr["filing_date"] < T("2021-07-01"), "cik"]) == {1}
    assert set(qr.loc[qr["filing_date"] >= T("2021-07-01"), "cik"]) == {2}
    assert qr["period_end"].is_unique


def test_trailing_revenue_volatility_definition():
    qr = records(*issuer())
    f = build.event_features(event("TST", acc_of(2022, 4), filed_of(2022, 4), "2023-04-21", "10-K"),
                             qr).iloc[0]
    g = [np.log(rev(y, q) / rev(y - 1, q)) for y in (2021, 2022) for q in (1, 2, 3, 4)]
    assert f["n_rev_growth"] == 8
    assert f["revenue_growth_volatility"] == pytest.approx(np.std(g, ddof=1))
    m = lambda y, q: oi(y, q) / rev(y, q)
    assert f["abs_operating_margin_change_yoy"] == pytest.approx(abs(m(2022, 4) - m(2021, 4)))
    assert f["leverage"] == pytest.approx(0.6)


def test_fewer_than_six_quarters_is_missing():
    qr = records(*issuer(years=range(2021, 2023)))
    f = build.event_features(event("TST", acc_of(2022, 1), filed_of(2022, 1), "2022-07-15"), qr).iloc[0]
    assert np.isnan(f["revenue_growth_volatility"])


# ───────────────────────────────────── static guards ────────────────────────────────────
def _strings(path: Path) -> list[str]:
    return [n.value for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


@pytest.mark.parametrize("mod", ["acquire.py", "build.py"])
def test_no_target_in_acquisition_or_feature_building(mod):
    src = (PKG / mod).read_text()
    for bad in ("y_extreme", "abs_r3", "abs_r1", "abs_reaction", "reaction_3d", "feature_frame"):
        assert bad not in src, f"{mod} mentions {bad}"
    assert "evaluate" not in {n.module.split(".")[-1] for n in ast.walk(ast.parse(src))
                              if isinstance(n, ast.ImportFrom) and n.module}


def test_audit_reads_target_availability_only():
    src = (PKG / "audit.py").read_text()
    assert _strings(PKG / "audit.py").count("y_extreme") == 2   # the column list and `.notna()`
    assert 'fr.pop("y_extreme").notna()' in src
    assert "abs_r3" not in src and "reaction" not in src


def test_raw_sec_data_is_under_gitignored_vendor():
    assert acquire.XBRL_SNAPSHOT_ROOT.is_relative_to(pilot_paths.SEC_ROOT)
    assert "vendor" in acquire.XBRL_SNAPSHOT_ROOT.parts
    gi = (REPO / ".gitignore").read_text()
    assert "**/vendor/" in gi and "**/output/" in gi


def test_reuses_the_pilots_identity_chain():
    assert acquire.SEGMENTS == pilot_paths.OUT / "cik_segments.csv"
    src = (PKG / "build.py").read_text()
    assert "filings_index.parquet" in src and "company_tickers" not in src


# ───────────────────────────────────── evaluation ───────────────────────────────────────
def _panel(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(2014, 2027):
        for i in range(400):
            h, v = rng.normal(), rng.normal()
            rows.append({"event_id": f"S{i % 60}|{y}-{i}", "stock": f"S{i % 60}", "year": y,
                         "announce_window": "BMO" if i % 2 else "AMC",
                         "endpoint3_date": T(f"{y}-01-10") + pd.Timedelta(days=int(i % 340)),
                         "log_hist_mean_abs": h, "log_vol_30d": v, "log_assets": rng.normal(),
                         **{c: rng.normal() for c in evaluate.COMPACT + ["l_margin_vol", "wc_to_assets_c"]},
                         "y_extreme": int(rng.random() < 1 / (1 + np.exp(-(h + 0.5 * v - 1.5))))})
    p = pd.DataFrame(rows)
    p.loc[p.sample(frac=0.1, random_state=1).index, "l_abs_margin_chg"] = np.nan
    return p


def test_training_rows_precede_the_test_year():
    from research.options_pilot.evaluate import walk_forward
    p = _panel()
    _, folds = walk_forward(p, evaluate.C, [2017, 2020])
    for y, f in folds.items():
        assert f["train_max_endpoint"] < T(f"{y}-01-01")


def test_selection_excludes_2026_and_pairs_identical_events(tmp_path, monkeypatch):
    p = _panel()
    monkeypatch.setattr(evaluate, "OUT", tmp_path)
    monkeypatch.setattr(evaluate, "panel", lambda: p)
    g = {f: {"pass": True} for f in set(evaluate.SOURCE.values()) | {"log_assets"}}
    g["wc_to_assets"]["pass"] = False                         # a failed gate drops its input
    monkeypatch.setattr(evaluate, "gates", lambda: g)
    out = evaluate.run_selection(reps=3)
    assert "wc_to_assets_c" not in out["blocks"]["C+workingcap"]["C+workingcap"]
    oof = pd.read_parquet(tmp_path / "oof_main.parquet")
    assert not oof["test_year"].eq(2026).any() and oof["test_year"].max() == 2025
    assert not oof["event_id"].isin(p.loc[p["year"].eq(2026), "event_id"]).any()
    for block, g_ in oof.groupby(level="block"):
        ids = [tuple(m["event_id"]) for _, m in g_.groupby(level="model")]
        assert all(i == ids[0] for i in ids), block
    sel = json.loads((tmp_path / "selection.json").read_text())
    assert sel["holdout_models"][0] == "C+compact" and len(sel["holdout_models"]) == 2


def test_holdout_scores_2026_only_once(tmp_path, monkeypatch):
    p = _panel()
    monkeypatch.setattr(evaluate, "OUT", tmp_path)
    monkeypatch.setattr(evaluate, "panel", lambda: p)
    monkeypatch.setattr(evaluate, "gates", lambda: {f: {"pass": True} for f in
                                                    set(evaluate.SOURCE.values()) | {"log_assets"}})
    evaluate.run_selection(reps=3)
    r = evaluate.run_holdout(reps=3)
    assert all(v["n_oof"] == int(p.dropna(subset=evaluate.COMPACT)["year"].eq(2026).sum())
               for k, v in r["results"].items() if k == "C+compact")
    with pytest.raises(SystemExit):
        evaluate.run_holdout(reps=3)


def test_decision_rule():
    d = lambda t10, lo, t20: {"d_top10_capture": t10, "d_top10_capture_lo": lo,
                              "d_top20_capture": t20, "d_top20_capture_lo": -1}
    assert evaluate.worth_keeping(d(0.012, 0.001, 0.001))
    assert not evaluate.worth_keeping(d(0.012, -0.001, 0.01))
    assert not evaluate.worth_keeping(d(0.009, 0.001, 0.01))
    assert not evaluate.worth_keeping(d(0.02, 0.001, -0.001))
    sel = {"C+revenue": {"deltas": {"C+revenue_vs_C": d(0.002, -0.004, 0.0),
                                    "C+size+revenue_vs_C+size": d(0, -1, 0)}}}
    assert evaluate.verdict(sel, None) == "FUNDAMENTALS ADD NO USEFUL VALUE"
    sel["C+revenue"]["deltas"]["C+revenue_vs_C"] = d(0.015, 0.002, 0.01)
    assert evaluate.verdict(sel, None) == "FUNDAMENTALS SHOW WEAK / UNCERTAIN VALUE"   # size proxy
    assert evaluate.verdict({}, None) == "POINT-IN-TIME FUNDAMENTALS ARE TOO UNRELIABLE TO ANSWER"
