"""Phase 5B feature registry — the pre-registered set, and its audit against Phase 5.

Mirrors PREREGISTRATION.md. `feature_inventory()` is the "does this already exist?" table
requested before implementation; it is written out verbatim as feature_inventory.csv.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_PRICE = "phase_5b_new_feature_testing/price_features.py"
_PEER = "phase_5b_new_feature_testing/peer_features.py"
_EVAL = "phase_5b_new_feature_testing/evaluate.py"
_PRICES = "full_df.parquet prices on the market-session grid"
_PRICES_LOO = _PRICES + " + LOO(issuer) market/sector benchmarks"
_PEERS = "phase3_events.parquet reaction_1d_anchored + anchor_date; stock_data sector/sub_sector"
_CUT_PRICE = "close of last session strictly before min(earnings_date, phase3 proxy session)"
_CUT_PEER = "peer 1-session endpoint (anchor+1) <= that cutoff session"


@dataclass(frozen=True)
class Feature:
    name: str
    family: str
    primary: bool
    definition: str
    phase5_relation: str          # nearest Phase 5 / repo feature, and how this differs
    exists_already: bool
    tested_in_phase5: bool
    exact_duplicate: bool
    data_source: str
    cutoff: str
    location: str
    safe: bool = True


F1, F2, F3, F4 = "1_idio_vol", "2_jumps", "3_vol_accel", "4_residual_move"
F5, F6, F7 = "5_peer_earnings", "6_peer_vol", "7_decoupling"

FEATURES: tuple[Feature, ...] = (
    # ── family 1 ──
    Feature("idio_vol_30d", F1, True, "std of 2-factor (LOO mkt, LOO sector-mkt) residuals, 30 sessions",
            "vol_30d (raw total vol) — this removes market/sector co-movement", False, False, False,
            _PRICES_LOO, _CUT_PRICE, _PRICE),
    Feature("idio_vol_10d", F1, True, "same residuals, 10 sessions",
            "vol_10d (raw) — residualised", False, False, False, _PRICES_LOO, _CUT_PRICE, _PRICE),
    Feature("idio_share_30d", F1, True, "var(resid)/var(r), 30 sessions",
            "none; stock_vs_sector_vol is a vol RATIO, not a variance share", False, False, False,
            _PRICES_LOO, _CUT_PRICE, _PRICE),
    Feature("idio_vol_30d_z_own", F1, True,
            "idio_vol_30d z-scored vs stock's own PRIOR event rows (Phase 5 _z_own construction)",
            "vol_30d_z_own (Phase 5 survivor-grade) — same transform on idiosyncratic vol",
            False, False, False, _PRICES_LOO, _CUT_PRICE + "; prior event rows only", _EVAL),
    # ── family 2 ──
    Feature("max_abs_ret_20d", F2, True, "max |r| over 20 sessions", "none", False, False, False,
            _PRICES, _CUT_PRICE, _PRICE),
    Feature("jump_share_20d", F2, True, "max r^2 / sum r^2 over 20 sessions", "none", False, False,
            False, _PRICES, _CUT_PRICE, _PRICE),
    Feature("semivar_balance_20d", F2, True, "(down - up realized semivariance) / total, 20 sessions",
            "none (drift/momentum are first moments)", False, False, False, _PRICES, _CUT_PRICE, _PRICE),
    # ── family 3 ──
    Feature("vol_log_ratio_5_30", F3, True, "log(std 5 / std 30)",
            "vol_ratio_10_to_30 (tested, null) — shorter fast leg, log scale; NEAR-duplicate family",
            False, False, False, _PRICES, _CUT_PRICE, _PRICE),
    Feature("vol_log_ratio_10_60", F3, True, "log(std 10 / std 60)",
            "vol_ratio_10_to_30 (tested, null) — longer slow leg, log scale; NEAR-duplicate family",
            False, False, False, _PRICES, _CUT_PRICE, _PRICE),
    # ── family 4 ──
    Feature("abs_idio_ret_5d", F4, True, "|cumulative 2-factor residual|, 5 sessions",
            "abs_mom_5d (raw, tested, null) — residualised against LOO market/sector",
            False, False, False, _PRICES_LOO, _CUT_PRICE, _PRICE),
    Feature("abs_idio_ret_20d", F4, True, "|cumulative 2-factor residual|, 20 sessions",
            "abs_mom_20d (raw, tested, null) — residualised", False, False, False,
            _PRICES_LOO, _CUT_PRICE, _PRICE),
    # ── family 6 ──
    Feature("sub_peer_med_vol_10d", F6, True, "median 10-session vol of LOO sub-sector peers",
            "sector_vol_10d (sector MEAN incl. self, tested) — sub-sector, LOO, median",
            False, False, False, _PRICES, _CUT_PRICE, _PRICE),
    Feature("sub_peer_med_vol_30d", F6, True, "median 30-session vol of LOO sub-sector peers",
            "sector_vol_30d (sector MEAN incl. self, tested) — sub-sector, LOO, median",
            False, False, False, _PRICES, _CUT_PRICE, _PRICE),
    Feature("stock_vs_sub_vol_30d", F6, True, "log(own 30-session vol / sub_peer_med_vol_30d)",
            "stock_vs_sector_vol (tested, weak) — sub-sector LOO reference; granularity change only",
            False, False, False, _PRICES, _CUT_PRICE, _PRICE),
    Feature("sub_peer_frac_high_vol", F6, True,
            "share of LOO sub-sector peers with 30-session vol above their own trailing-252 p80",
            "none", False, False, False, _PRICES, _CUT_PRICE, _PRICE),
    # ── family 7 ──
    Feature("sector_corr_60d", F7, True, "corr(r, LOO sector return), 60 sessions", "none",
            False, False, False, _PRICES_LOO, _CUT_PRICE, _PRICE),
    Feature("sector_corr_change_20_120", F7, True, "corr 20 sessions minus corr 120 sessions",
            "none", False, False, False, _PRICES_LOO, _CUT_PRICE, _PRICE),
    # ── secondary price ──
    Feature("signed_idio_ret_20d", F4, False, "signed cumulative residual, 20 sessions",
            "mom_20d (raw, tested) — residualised; diagnostic only", False, False, False,
            _PRICES_LOO, _CUT_PRICE, _PRICE),
    Feature("std_abs_idio_ret_20d", F4, False, "abs_idio_ret_20d / (idio_vol_30d*sqrt(20))",
            "none", False, False, False, _PRICES_LOO, _CUT_PRICE, _PRICE),
    Feature("max_abs_idio_ret_20d", F2, False, "max |residual| over 20 sessions", "none",
            False, False, False, _PRICES_LOO, _CUT_PRICE, _PRICE),
    Feature("vol_change_20_20", F3, False, "log(std last 20 / std of the 20 before)",
            "vol_ratio_10_to_30 (tested) — non-overlapping windows", False, False, False,
            _PRICES, _CUT_PRICE, _PRICE),
    Feature("vol_30d_cut", F1, False, "plain 30-session std at the Phase 5B cutoff",
            "EXACT concept duplicate of vol_30d; kept only as a leak/cutoff control",
            True, True, True, _PRICES, _CUT_PRICE, _PRICE),
    Feature("sec_peer_med_vol_30d", F6, False, "median 30-session vol of LOO sector peers",
            "sector_vol_30d (tested) — LOO + median; control for sub-sector", False, False, False,
            _PRICES, _CUT_PRICE, _PRICE),
)

_PEER_DEF = {
    "n": ("count of usable distinct-issuer peer events", "sector_earnings_density counts "
          "SCHEDULED reports (calendar), not completed usable outcomes"),
    "mean_abs_1d": ("mean |anchored 1-session peer reaction|", "sector_prior_extreme_rate "
                    "(Phase 4) is an all-history sector base rate, not a recent window"),
    "max_abs_1d": ("max |anchored 1-session peer reaction|", "none"),
    "frac_large_1d": ("share of usable peers with |r1d| >= 0.05", "none"),
    "rw_mean_abs_1d": ("recency-weighted mean |r1d|, half-life 10 sessions", "none"),
    "shock": ("mean log((|r1d|+.005)/(peer own prior mean |r1d|+.005))", "none"),
    "frac_extreme_1d": ("share of usable peers with |r1d| >= 0.08", "none"),
    "n_unresolved": ("peer reports in window WITHOUT an available reaction (diagnostic)", "none"),
}
_PRIMARY_PEER_STATS = ("n", "mean_abs_1d", "max_abs_1d", "frac_large_1d", "rw_mean_abs_1d", "shock")

_peer: list[Feature] = []
for _lvl, _lname in (("sub", "sub-sector"), ("sec", "sector")):
    for _stat in _PRIMARY_PEER_STATS:
        d, rel = _PEER_DEF[_stat]
        _peer.append(Feature(f"peer_{_lvl}_{_stat}_20", F5, True, f"{_lname}, 20 sessions: {d}",
                             rel, False, False, False, _PEERS, _CUT_PEER, _PEER))
    for _w in (10, 40):
        for _stat in ("mean_abs_1d", "shock"):
            d, rel = _PEER_DEF[_stat]
            _peer.append(Feature(f"peer_{_lvl}_{_stat}_{_w}", F5, False,
                                 f"{_lname}, {_w} sessions: {d} (window sensitivity)",
                                 rel, False, False, False, _PEERS, _CUT_PEER, _PEER))
    for _stat in ("frac_extreme_1d", "n_unresolved"):
        d, rel = _PEER_DEF[_stat]
        _peer.append(Feature(f"peer_{_lvl}_{_stat}_20", F5, False, f"{_lname}, 20 sessions: {d}",
                             rel, False, False, False, _PEERS, _CUT_PEER, _PEER))
for _stat in ("n", "mean_abs_1d", "shock", "frac_large_1d"):
    d, rel = _PEER_DEF[_stat]
    _peer.append(Feature(f"peer_local_{_stat}_20", F5, False,
                         f"sub-sector if >=2 usable sub-sector peers else sector: {d}",
                         rel, False, False, False, _PEERS, _CUT_PEER, _PEER))
_peer.append(Feature("peer_mkt_exsec_mean_abs_1d_20", F5, False,
                     "all OTHER sectors, 20 sessions: mean |r1d| (market earnings-season control)",
                     "none", False, False, False, _PEERS, _CUT_PEER, _PEER))

FEATURES = FEATURES + tuple(_peer)
BY_NAME = {f.name: f for f in FEATURES}
PRIMARY = [f.name for f in FEATURES if f.primary]
SECONDARY = [f.name for f in FEATURES if not f.primary]
FAMILIES = sorted({f.family for f in FEATURES})


def family_members(family: str, *, primary_only: bool = True) -> list[str]:
    return [f.name for f in FEATURES if f.family == family and (f.primary or not primary_only)]


def feature_inventory() -> pd.DataFrame:
    return pd.DataFrame([{
        "proposed_feature": f.name,
        "family": f.family,
        "role": "primary" if f.primary else "secondary",
        "definition": f.definition,
        "already_exists": f.exists_already,
        "already_tested_in_phase5": f.tested_in_phase5,
        "exact_duplicate": f.exact_duplicate,
        "nearest_existing_feature_and_difference": f.phase5_relation,
        "safe_to_build": f.safe,
        "data_source": f.data_source,
        "causal_cutoff": f.cutoff,
        "implementation_location": f.location,
    } for f in FEATURES])
