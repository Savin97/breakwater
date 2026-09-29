"""Score candidates for the P3.2/P3.3 refit — fixed in PREREGISTRATION.md.

Nothing here is fitted. A candidate is either a deterministic formula (the 0.3.1 family)
or a list of model inputs that `calibration.py` fits walk-forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    BUCKET_ELEVATED_FLOOR,
    BUCKET_HIGH_ALERT_FLOOR,
    LIFT_TO_ELEVATED,
    LIFT_TO_HIGH_ALERT,
)

TIERS = ["Normal", "Elevated", "High Alert"]
LOG_FLOOR = 1e-4

# 0.3.1 constants, spelled out so the reconstruction cannot drift from them silently.
V031_P75_CEILING = 0.12
V031_W_P75 = 0.85
V031_W_ENTROPY = 0.15


def _log(s: pd.Series) -> pd.Series:
    return np.log(pd.to_numeric(s, errors="coerce").clip(lower=LOG_FLOOR))


def add_model_inputs(frame: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Deterministic transforms of the `_{cutoff}` features into model inputs.

    Written to cutoff-free column names so one model spec serves both cutoffs; the caller
    decides which cutoff a frame represents.
    """
    out = frame.copy()
    out["log_hist_mean_abs"] = _log(out[f"hist_mean_abs_{cutoff}"])
    out["log_vol_30d"] = _log(out[f"vol_30d_{cutoff}"])
    out["log_peer_sec_rw"] = _log(out[f"peer_sec_rw_mean_abs_1d_20_{cutoff}"])
    out["log_idio_vol_30d"] = _log(out[f"idio_vol_30d_{cutoff}"])
    out["a1_score"] = v031_score(out[f"hist_p75_{cutoff}"], out[f"hist_entropy_{cutoff}"])
    out["a1_nocap"] = v031_score(out[f"hist_p75_{cutoff}"], out[f"hist_entropy_{cutoff}"],
                                 cap=False)
    out["a1_noentropy"] = v031_score(out[f"hist_p75_{cutoff}"], out[f"hist_entropy_{cutoff}"],
                                     use_entropy=False)
    out["p75_uncapped"] = pd.to_numeric(out[f"hist_p75_{cutoff}"], errors="coerce")
    out["a0_shipped"] = pd.to_numeric(out["risk_score"], errors="coerce")
    return out


# Logistic model inputs per candidate. A-family candidates are single-input Platt models,
# so their ranking is exactly the raw score's.
MODEL_INPUTS: dict[str, list[str]] = {
    "A0_v031_shipped": ["a0_shipped"],
    "A1_v031_corrected": ["a1_score"],
    "B_structural": ["log_hist_mean_abs"],
    "C_structural_vol": ["log_hist_mean_abs", "log_vol_30d"],
    "D_phase5b_pair": ["log_hist_mean_abs", "log_vol_30d", "log_peer_sec_rw",
                       "log_idio_vol_30d"],
}
# Inputs that may be missing and are median-imputed inside the training fold.
IMPUTABLE = {"log_peer_sec_rw"}
MECHANICS = ["a1_score", "a1_nocap", "a1_noentropy", "p75_uncapped"]


def v031_score(p75: pd.Series, hist_entropy: pd.Series, *, cap: bool = True,
               use_entropy: bool = True) -> pd.Series:
    """0.3.1's structural score. `cap=False` drops the 12% ceiling, `use_entropy=False`
    the 15% entropy term. Missing entropy counts 0 (0.3.1's cross-ticker ffill is a defect
    and is not reproduced)."""
    e3 = (pd.to_numeric(p75, errors="coerce") / V031_P75_CEILING).clip(lower=0)
    if cap:
        e3 = e3.clip(upper=1)
    if not use_entropy:
        return 100 * e3
    e4 = pd.to_numeric(hist_entropy, errors="coerce").fillna(0).clip(0, 1)
    raw = V031_W_P75 * e3 + V031_W_ENTROPY * e4
    return 100 * (raw.clip(0, 1) if cap else raw)


def v031_structural_tier(score: pd.Series) -> pd.Series:
    return pd.cut(score, bins=[-np.inf, BUCKET_ELEVATED_FLOOR, BUCKET_HIGH_ALERT_FLOOR, np.inf],
                  labels=TIERS).astype(object)


def promote(bucket: pd.Series, lift: pd.Series, *, to_elevated: float = LIFT_TO_ELEVATED,
            to_high_alert: float = LIFT_TO_HIGH_ALERT) -> pd.Series:
    """0.3.1's lift promotion, verbatim logic."""
    b = bucket.astype(object).copy()
    to_high = b.isin(["Normal", "Elevated"]) & lift.ge(to_high_alert)
    to_elev = b.eq("Normal") & lift.ge(to_elevated) & ~to_high
    b[to_elev] = "Elevated"
    b[to_high] = "High Alert"
    return b
