"""Phase A coverage report — no outcome is read here.

    PYTHONPATH=. .venv/bin/python -m research.options_pilot.coverage

Reads `output/options_pilot/event_options.parquet`, writes `coverage_*.csv` and
`identity_audit.parquet` next to it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.options_pilot.build import OUT, identity_audit
from research.options_pilot.features import FILTERS

AGE_BANDS = [("same_day", 0, 0), ("le1", 0, 1), ("le3", 0, 3), ("le5", 0, 5)]


def age_band(age: pd.Series, status: pd.Series) -> pd.Series:
    b = pd.Series("gt5_or_missing", index=age.index)
    for lab, lo, hi in (("0", 0, 0), ("1", 1, 1), ("2-3", 2, 3), ("4-5", 4, 5)):
        b[age.between(lo, hi) & status.eq("matched")] = lab
    return b


def summarise(ev: pd.DataFrame, by: str | None) -> pd.DataFrame:
    g = ev.groupby(by) if by else ev.assign(_all="ALL").groupby("_all")
    matched = ev["match_status"].eq("matched")
    ok = ev["status"].eq("ok")
    rows = {}
    for key, s in g:
        m, o, a = matched[s.index], ok[s.index], s["snapshot_age"]
        rows[key] = {
            "events": len(s),
            "matched_any_age": m.mean(),
            "same_day": (m & a.eq(0)).mean(),
            "le1": (m & a.le(1)).mean(),
            "le3": (m & a.le(3)).mean(),
            "le5": (m & a.le(5)).mean(),
            "gt5_or_missing": 1 - (m & a.le(5)).mean(),
            "covering_expiry": (m & s["has_covering_expiry"].eq(True)).mean(),
            "features_ok_any_age": o.mean(),
            "features_ok_le1": (o & a.le(1)).mean(),
            "features_ok_le5": (o & a.le(5)).mean(),
        }
    return pd.DataFrame(rows).T


def main() -> int:
    ev = pd.read_parquet(OUT / "event_options.parquet")
    ev["year"] = ev["year"].astype(int)
    tabs = {"overall": summarise(ev, None), "year": summarise(ev, "year"),
            "window": summarise(ev, "announce_window")}
    for k, t in tabs.items():
        t.to_csv(OUT / f"coverage_{k}.csv")
        print(f"\n== coverage by {k} ==\n{t.round(3).to_string()}")

    ev["age_band"] = age_band(ev["snapshot_age"], ev["match_status"])
    print("\n== snapshot age (sessions) ==\n",
          pd.crosstab(ev["year"], ev["age_band"], margins=True).to_string())

    m_ = ev[ev["match_status"].eq("matched")]
    exp = m_.groupby("year").agg(
        matched=("event_id", "size"),
        covering_expiry=("has_covering_expiry", "sum"),
        n_covering_median=("n_covering_expiries", "median"),
        usable_covering_1=("n_usable_covering_expiries", lambda x: int((x == 1).sum())),
        usable_covering_2plus=("n_usable_covering_expiries", lambda x: int((x >= 2).sum())),
        usable_covering_0=("n_usable_covering_expiries", lambda x: int((x == 0).sum())),
        near_pair_ok=("status", lambda x: int((x == "ok").sum())),
        second_usable_later=("has_second_usable_expiry", lambda x: int(x.fillna(False).astype(bool).sum())),
        strict_after_ok=("sa_status", lambda x: int((x == "ok").sum())))
    exp.loc["ALL"] = exp.sum()
    exp.loc["ALL", "n_covering_median"] = m_["n_covering_expiries"].median()
    exp.to_csv(OUT / "coverage_expiry.csv")
    print("\n== expiry availability (matched events) ==\n", exp.to_string())

    st = ev.loc[ev["match_status"].eq("matched"), "status"].value_counts()
    print("\n== feature status among matched ==\n", st.to_string())

    tk = summarise(ev, "stock")
    tk.to_csv(OUT / "coverage_ticker.csv")
    print("\n== tickers: share of events with features at <=5 sessions ==")
    print(tk["features_ok_le5"].describe().round(3).to_string())
    print("zero coverage:", sorted(tk.index[tk["features_ok_le5"].eq(0)]))

    m = ev[ev["match_status"].eq("matched")]
    q = {"contracts_raw_median": m["n_contracts_raw"].median(),
         "expirations_median": m["n_expirations_raw"].median(),
         "duplicate_rows": int(m["n_duplicate_rows"].sum()),
         "contracts_raw_total": int(m["n_contracts_raw"].sum())}
    for f in FILTERS:
        q[f"removed_{f}"] = int(m[f"removed_{f}"].sum())
    ok = ev[ev["status"].eq("ok")]
    q.update({
        "near_is_report_day_share": ok["near_is_report_day"].mean(),
        "near_days_to_expiry_median": ok["near_days_to_expiry"].median(),
        "atm_abs_call_delta_minus_half_median": (ok["atm_call_delta"] - 0.5).abs().median(),
        "strike_vs_parity_spot_abs_median": ok["strike_vs_parity_spot"].abs().median(),
        "strike_vs_parity_spot_abs_p95": ok["strike_vs_parity_spot"].abs().quantile(0.95),
        "delta_sum_gap_abs_p95": ok["delta_sum_gap"].abs().quantile(0.95),
        "atm_rel_spread_median": ok["atm_rel_spread"].median(),
        "atm_rel_spread_p95": ok["atm_rel_spread"].quantile(0.95),
        "expected_move_median": ok["expected_move"].median(),
        "expected_move_p99": ok["expected_move"].quantile(0.99),
        "atm_iv_median": ok["atm_iv"].median(),
        "atm_iv_max": ok["atm_iv"].max(),
        "put_skew_coverage": ok["put_skew"].notna().mean(),
        "term_ratio_coverage": ok["term_ratio"].notna().mean(),
        "term_ratio_median": ok["term_ratio"].median(),
    })
    pd.Series(q).to_csv(OUT / "coverage_quality.csv")
    print("\n== quote quality ==\n", pd.Series(q).to_string())

    audit = identity_audit(ev)
    audit.to_parquet(OUT / "identity_audit.parquet")
    print("\n== identity audit ==")
    print("events audited", len(audit), "| missing adjusted close", int(audit["price"].isna().sum()))
    print("split-like jumps", int(audit["split_like"].sum()),
          "| unexplained jumps", int(audit["jump_flag"].sum()),
          "in", audit.loc[audit["jump_flag"], "stock"].nunique(), "stocks")
    flagged = audit[audit["jump_flag"]][["stock", "source_symbol", "snapshot_date", "ratio", "log_jump"]]
    print(flagged.assign(jump=np.exp(flagged["log_jump"])).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
