"""Options features from ONE option-chain snapshot. Definitions: PREREGISTRATION.md.

Pure functions of a single `(snapshot date, symbol)` chain plus the event's report date
and announcement window. Nothing here reads a price, an outcome or another snapshot, so
a feature can only ever carry the information in the snapshot it names.

Price semantics: Breakwater's stored prices are split- AND dividend-adjusted (yfinance
`auto_adjust=True`; AAPL 2019-05-10 is 47.21 in the DB vs a nominal close of ~197), while
option strikes are nominal as of the snapshot. The two are never mixed. The straddle is
normalised by its own strike, and the pair is required to be at the money by its own
deltas. The put-call-parity spot `K + C - P` is kept only as a diagnostic (identity and
split checks), never as a model input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from feature_engineering.announcement_timing import AMC, BMO

# ── quote filters (applied per contract, in this order; each one's removals reported) ──
MAX_IV = 5.0            # the source's `vol` is decimal(5,4); >500% is not a quote
# ── ATM pair ───────────────────────────────────────────────────────────────────────────
ATM_DELTA_BAND = (0.30, 0.70)   # |call delta| and |put delta| must both lie in here
# ── 25-delta put ───────────────────────────────────────────────────────────────────────
PUT25_TARGET = -0.25
PUT25_MAX_DIST = 0.10           # accept a put with delta in [-0.35, -0.15]; no interpolation
# Amendment 1 (pre-outcome, after the coverage audit): the pair's strike must also be within
# 10% of its own put-call-parity spot K + C_mid - P_mid — the live collector's 10% rule.
# The provider's greeks are occasionally computed off a wrong spot (KEY 2024-07-12: strike
# 20, call 1.08, put 4.80 -> parity spot 16.3, yet both deltas ~0.5 and both IVs 3.07).
MAX_STRIKE_VS_PARITY = 0.10

RULE_ANNOUNCEMENT = "announcement_aware"
RULE_STRICT_AFTER = "strict_after_report_date"

FILTERS = ("bid_ask_present", "nonnegative", "ask_ge_bid", "ask_positive",
           "iv_valid", "delta_valid")


def normalise_chain(rows) -> pd.DataFrame:
    """The API's string rows -> typed frame. No row is dropped here."""
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame(columns=["date", "act_symbol", "expiration", "strike", "call_put",
                                     "bid", "ask", "vol", "delta"])
    for c in ("strike", "bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])
    df["call_put"] = df["call_put"].str.strip().str.title()
    return df


def quote_filter_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Each filter's pass mask, evaluated cumulatively in FILTERS order."""
    call = df["call_put"].eq("Call")
    put = df["call_put"].eq("Put")
    m = {}
    m["bid_ask_present"] = df["bid"].notna() & df["ask"].notna()
    m["nonnegative"] = (df["bid"] >= 0) & (df["ask"] >= 0)
    m["ask_ge_bid"] = df["ask"] >= df["bid"]
    m["ask_positive"] = df["ask"] > 0
    m["iv_valid"] = df["vol"].notna() & (df["vol"] > 0) & (df["vol"] <= MAX_IV)
    m["delta_valid"] = df["delta"].notna() & (
        (call & df["delta"].between(0, 1)) | (put & df["delta"].between(-1, 0)))
    return m


def clean_quotes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rows passing every filter, and how many rows each filter removed (cumulative)."""
    keep = pd.Series(True, index=df.index)
    removed = {}
    for name, mask in quote_filter_masks(df).items():
        new = keep & mask.fillna(False)
        removed[name] = int((keep & ~new).sum())
        keep = new
    out = df[keep].copy()
    out["mid"] = (out["bid"] + out["ask"]) / 2.0
    return out, removed


def covers_announcement(expiration, report_date, window: str) -> bool:
    """Does an option expiring on `expiration` still exist when the news lands?

    * expiry after the report date: yes.
    * expiry ON the report date: BMO yes (the news is out before that day's open, and the
      option trades through that session); AMC no (it expires at 16:00, the news comes
      after). Any other window on the same day: no — the timing is not known well enough.
    * expiry before the report date: never.
    """
    e = pd.Timestamp(expiration).normalize()
    d = pd.Timestamp(report_date).normalize()
    if e > d:
        return True
    if e == d:
        return window == BMO
    return False


def _atm_pair(q: pd.DataFrame, expiration) -> dict | None:
    """Same-strike call/put pair minimising |Δc − 0.5| + |Δp + 0.5| in one expiry.

    Both legs must pass the quote filters and both |deltas| must lie in ATM_DELTA_BAND.
    Ties: smaller |C_mid − P_mid|, then lower strike. None if no pair qualifies.
    """
    ex = q[q["expiration"] == expiration]
    c = ex[ex["call_put"] == "Call"].set_index("strike")
    p = ex[ex["call_put"] == "Put"].set_index("strike")
    strikes = c.index.intersection(p.index)
    if len(strikes) == 0:
        return None
    c, p = c.loc[strikes], p.loc[strikes]
    lo, hi = ATM_DELTA_BAND
    ok = c["delta"].abs().between(lo, hi) & p["delta"].abs().between(lo, hi)
    if not ok.any():
        return None
    c, p = c[ok], p[ok]
    cand = pd.DataFrame({
        "score": ((c["delta"] - 0.5).abs() + (p["delta"] + 0.5).abs()).to_numpy(),
        "gap": (c["mid"] - p["mid"]).abs().to_numpy(),
        "k": c.index.to_numpy(),
    }).sort_values(["score", "gap", "k"], kind="mergesort")
    k = cand["k"].iloc[0]
    cr, pr = c.loc[k], p.loc[k]
    return {
        "expiration": pd.Timestamp(expiration), "strike": float(k),
        "call_bid": float(cr["bid"]), "call_ask": float(cr["ask"]), "call_mid": float(cr["mid"]),
        "put_bid": float(pr["bid"]), "put_ask": float(pr["ask"]), "put_mid": float(pr["mid"]),
        "call_iv": float(cr["vol"]), "put_iv": float(pr["vol"]),
        "call_delta": float(cr["delta"]), "put_delta": float(pr["delta"]),
    }


def _put25(q: pd.DataFrame, expiration) -> dict | None:
    ex = q[(q["expiration"] == expiration) & (q["call_put"] == "Put")]
    if ex.empty:
        return None
    dist = (ex["delta"] - PUT25_TARGET).abs()
    ex = ex.assign(_d=dist, _k=ex["strike"]).sort_values(["_d", "_k"], kind="mergesort")
    best = ex.iloc[0]
    if best["_d"] > PUT25_MAX_DIST:
        return None
    return {"put25_strike": float(best["strike"]), "put25_iv": float(best["vol"]),
            "put25_delta": float(best["delta"])}


def snapshot_features(chain: pd.DataFrame, report_date, window: str) -> dict:
    """Every pre-registered feature and diagnostic from one cleaned-or-raw chain."""
    out: dict = {"n_contracts_raw": int(len(chain))}
    if chain.empty:
        out["status"] = "no_chain"
        return out
    snap = pd.Timestamp(chain["date"].iloc[0]).normalize()
    if chain["date"].nunique() != 1:
        raise ValueError("a chain must come from exactly one snapshot date")
    exps = sorted(chain["expiration"].unique())
    out["n_expirations_raw"] = len(exps)
    out["expirations"] = ",".join(pd.Timestamp(e).strftime("%Y-%m-%d") for e in exps)
    dup = chain.duplicated(["expiration", "strike", "call_put"]).sum()
    out["n_duplicate_rows"] = int(dup)
    q, removed = clean_quotes(chain)
    for k, v in removed.items():
        out[f"removed_{k}"] = v
    out["n_contracts_clean"] = int(len(q))

    ann = _rule_features(q, exps, snap, report_date, window, rule=RULE_ANNOUNCEMENT)
    out.update(ann)
    strict = _rule_features(q, exps, snap, report_date, window, rule=RULE_STRICT_AFTER)
    keep = ("status", "near_expiration", "expected_move", "atm_iv", "put_skew", "term_ratio",
            "atm_strike", "n_covering_expiries", "n_usable_covering_expiries")
    out.update({f"sa_{k}": strict[k] for k in keep if k in strict})
    return out


def is_covering(expiration, snap, report_date, window: str, rule: str) -> bool:
    """RULE_ANNOUNCEMENT: the pre-registered announcement-aware rule (covers_announcement).
    RULE_STRICT_AFTER: the live collector's rule, expiry strictly after the report date,
    whatever the window (robustness only)."""
    e = pd.Timestamp(expiration).normalize()
    if e < pd.Timestamp(snap).normalize():
        return False
    if rule == RULE_ANNOUNCEMENT:
        return covers_announcement(e, report_date, window)
    if rule == RULE_STRICT_AFTER:
        return e > pd.Timestamp(report_date).normalize()
    raise ValueError(rule)


def _valid_pair(q, expiration) -> dict | None:
    """A pre-registered ATM pair that also passes Amendment 1's parity-spot check."""
    pair = _atm_pair(q, expiration)
    if pair is None:
        return None
    spot = pair["strike"] + pair["call_mid"] - pair["put_mid"]
    if not (spot > 0 and abs(pair["strike"] / spot - 1) <= MAX_STRIKE_VS_PARITY):
        return None
    return pair


def _rule_features(q, exps, snap, report_date, window: str, rule: str) -> dict:
    out: dict = {}
    covering = [e for e in exps if is_covering(e, snap, report_date, window, rule)]
    out["has_covering_expiry"] = bool(covering)
    out["n_covering_expiries"] = len(covering)
    usable = [e for e in covering if _valid_pair(q, e) is not None]
    out["n_usable_covering_expiries"] = len(usable)
    same_day = [e for e in exps if pd.Timestamp(e).normalize() == pd.Timestamp(report_date).normalize()]
    out["has_same_day_expiry"] = bool(same_day)
    if not covering:
        out["status"] = "no_covering_expiry"
        return out
    near = covering[0]
    out["near_expiration"] = pd.Timestamp(near)
    out["near_is_report_day"] = pd.Timestamp(near).normalize() == pd.Timestamp(report_date).normalize()
    out["near_days_to_expiry"] = int((pd.Timestamp(near) - snap).days)
    pair = _atm_pair(q, near)
    if pair is None:
        out["status"] = "no_atm_pair"
        return out
    out.update({f"atm_{k}": v for k, v in pair.items() if k != "expiration"})
    K = pair["strike"]
    out["straddle_mid"] = pair["call_mid"] + pair["put_mid"]
    out["expected_move"] = out["straddle_mid"] / K
    out["atm_iv"] = (pair["call_iv"] + pair["put_iv"]) / 2.0
    out["parity_spot"] = K + pair["call_mid"] - pair["put_mid"]
    out["strike_vs_parity_spot"] = K / out["parity_spot"] - 1 if out["parity_spot"] > 0 else np.nan
    out["delta_sum_gap"] = pair["call_delta"] - pair["put_delta"] - 1.0
    if not abs(out["strike_vs_parity_spot"]) <= MAX_STRIKE_VS_PARITY:
        for k in ("expected_move", "atm_iv"):
            out[f"rejected_{k}"] = out.pop(k)
        out["status"] = "atm_far_from_parity_spot"
        return out
    out["atm_rel_spread"] = ((pair["call_ask"] - pair["call_bid"]) + (pair["put_ask"] - pair["put_bid"])) \
        / max(out["straddle_mid"], 1e-9)

    p25 = _put25(q, near)
    if p25 is not None:
        out.update(p25)
        out["put_skew"] = p25["put25_iv"] - out["atm_iv"]

    # Term structure: the first LATER expiry with a valid pair (Amendment 2: the parity
    # check applies to it too). None -> the feature stays missing; no farther substitute.
    later = [e for e in exps if pd.Timestamp(e) > pd.Timestamp(near)]
    out["has_second_usable_expiry"] = False
    for far in later:
        fp = _valid_pair(q, far)
        if fp is not None:
            out["has_second_usable_expiry"] = True
            out["far_expiration"] = pd.Timestamp(far)
            out["far_atm_iv"] = (fp["call_iv"] + fp["put_iv"]) / 2.0
            out["far_strike"] = fp["strike"]
            if out["far_atm_iv"] > 0:
                out["term_ratio"] = out["atm_iv"] / out["far_atm_iv"]
            break
    out["status"] = "ok"
    return out
