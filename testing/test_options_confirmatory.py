"""Tests for the vendor-neutral rules of the confirmatory options test
(`research/options_confirmatory/core.py`). Synthetic data only."""
import subprocess

import numpy as np
import pandas as pd
import pytest

from feature_engineering.announcement_timing import AMC, BMO
from research.options_confirmatory import core as K
from research.options_pilot.features import RULE_STRICT_AFTER

GRID = pd.bdate_range("2024-01-01", "2024-12-31").to_numpy(dtype="datetime64[ns]")
CUT = pd.Timestamp("2024-03-08")                     # a Friday


def _snaps(*stamps, spot=100.0):
    return pd.DataFrame({"snapshot_ts": [pd.Timestamp(s).tz_localize(K.NY) for s in stamps],
                         "spot": spot, "spot_field": "stkPx", "source": "test"})


def _chain(exps=("2024-03-15", "2024-03-22"), strikes=(95, 100, 105)):
    rows = []
    for e in exps:
        for k in strikes:
            rows.append({"expiration": pd.Timestamp(e), "strike": float(k), "call_put": "Call",
                         "bid": 2.0 + (100 - k) / 5, "ask": 2.2 + (100 - k) / 5})
            rows.append({"expiration": pd.Timestamp(e), "strike": float(k), "call_put": "Put",
                         "bid": 2.0 - (100 - k) / 5, "ask": 2.2 - (100 - k) / 5})
    return pd.DataFrame(rows)


def test_snapshot_never_after_prediction_cutoff():
    s = _snaps("2024-03-08 15:46", "2024-03-08 16:30", "2024-03-11 15:46")
    got = K.select_snapshot(s, CUT, GRID)
    assert got["snapshot_ts"] == pd.Timestamp("2024-03-08 15:46").tz_localize(K.NY)
    assert got["snapshot_ts"] <= K.prediction_cutoff_ts(CUT)


def test_prediction_cutoff_is_1600_new_york():
    assert K.prediction_cutoff_ts(CUT) == pd.Timestamp("2024-03-08 16:00").tz_localize(K.NY)


def test_snapshot_older_than_one_session_is_rejected():
    assert K.select_snapshot(_snaps("2024-03-07 15:46"), CUT, GRID)["snapshot_age"] == 1
    assert K.select_snapshot(_snaps("2024-03-06 15:46"), CUT, GRID) is None


def test_amc_same_day_expiry_excluded_bmo_included():
    snap = {"snapshot_ts": pd.Timestamp("2024-03-08 15:46").tz_localize(K.NY), "spot": 100.0}
    ch = _chain(exps=("2024-03-15", "2024-03-22"))
    assert K.expected_move(ch, snap, "2024-03-15", AMC)["expiration"] == pd.Timestamp("2024-03-22")
    assert K.expected_move(ch, snap, "2024-03-15", BMO)["expiration"] == pd.Timestamp("2024-03-15")


def test_strict_after_rule_is_separate():
    snap = {"snapshot_ts": pd.Timestamp("2024-03-08 15:46").tz_localize(K.NY), "spot": 100.0}
    got = K.expected_move(_chain(), snap, "2024-03-15", BMO, rule=RULE_STRICT_AFTER)
    assert got["expiration"] == pd.Timestamp("2024-03-22") and got["rule"] == RULE_STRICT_AFTER


def test_pair_same_strike_same_expiry_and_spot_denominator():
    snap = {"snapshot_ts": pd.Timestamp("2024-03-08 15:46").tz_localize(K.NY), "spot": 101.0}
    ch = pd.concat([_chain(), pd.DataFrame([{"expiration": pd.Timestamp("2024-03-15"),
                                             "strike": 101.0, "call_put": "Call",
                                             "bid": 1.5, "ask": 1.6}])])
    got = K.expected_move(ch, snap, "2024-03-12", AMC)
    assert got["strike"] == 100.0                      # 101 has no put -> not a pair
    assert got["call_mid"] == pytest.approx((got["call_bid"] + got["call_ask"]) / 2)
    assert got["put_mid"] == pytest.approx((got["put_bid"] + got["put_ask"]) / 2)
    assert got["expected_move_pct"] == pytest.approx((got["call_mid"] + got["put_mid"]) / 101.0)


def test_strike_far_from_spot_gives_no_feature():
    snap = {"snapshot_ts": pd.Timestamp("2024-03-08 15:46").tz_localize(K.NY), "spot": 120.0}
    assert K.expected_move(_chain(), snap, "2024-03-12", AMC)["status"] == "no_strike_near_spot"


def test_missing_spot_is_not_filled():
    snap = {"snapshot_ts": pd.Timestamp("2024-03-08 15:46").tz_localize(K.NY), "spot": np.nan}
    assert K.expected_move(_chain(), snap, "2024-03-12", AMC)["status"] == "no_spot"


def test_event_row_carries_every_provenance_field():
    row = K.event_row("AAA|2024-03-12", CUT, "2024-03-12", AMC, _snaps("2024-03-08 15:46"),
                      lambda s: _chain(), GRID)
    missing = [f for f in K.REQUIRED_FIELDS if f not in row]
    assert row["status"] == "ok" and not missing, missing


def test_nearest_covering_expiry_with_no_valid_quotes_is_not_skipped():
    snap = {"snapshot_ts": pd.Timestamp("2024-03-08 15:46").tz_localize(K.NY), "spot": 100.0}
    ch = _chain()
    ch.loc[ch["expiration"] == pd.Timestamp("2024-03-15"), "bid"] = np.nan
    got = K.expected_move(ch, snap, "2024-03-12", AMC)
    assert got["status"] == "no_pair" and got["expiration"] == pd.Timestamp("2024-03-15")


def test_duplicate_contracts_are_refused():
    snap = {"snapshot_ts": pd.Timestamp("2024-03-08 15:46").tz_localize(K.NY), "spot": 100.0}
    ch = _chain()
    with pytest.raises(ValueError, match="duplicate"):
        K.expected_move(pd.concat([ch, ch.iloc[[0]]]), snap, "2024-03-12", AMC)


def test_no_vendor_data_tracked():
    out = subprocess.run(["git", "ls-files", "data/vendor"], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == ""
