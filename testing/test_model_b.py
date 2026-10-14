"""
Model B production invariants (scoring/model_b.py).

Causality, eligibility, the target, the percentile scale and the tiers — each pinned on a
small hand-built event frame whose answer is known, plus static guards that the legacy
target cannot reach the score.

Equivalence with the frozen research implementation is checked separately, against the
refit's saved frame: analysis/model_b_validation/ (run explicitly).

Run with:  pytest testing/test_model_b.py -v
"""
import ast
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from scoring import model_b as B

GRID = pd.bdate_range("2018-01-01", "2020-12-31")   # no holidays: a known session grid


def _daily(end=None):
    dates = GRID if end is None else GRID[GRID <= pd.Timestamp(end)]
    return pd.DataFrame({"stock": "S", "date": dates})


def _event(stock, report, value, window="AMC", status="available", pending=False):
    """One event row. AMC anchors on the report date, BMO on the session before."""
    report = pd.Timestamp(report)
    if pending or status != "available":
        anchor = pd.NaT if pending else report
    else:
        i = GRID.get_loc(report)
        anchor = report if window == "AMC" else GRID[i - 1]
    return {"stock": stock, "earnings_date": report, "is_pending": pending,
            "announce_window": "UNKNOWN" if pending else window,
            "anchor_date": anchor,
            "abs_reaction_3d_anchored": np.nan if pending else value,
            "reaction_3d_anchored_status": "pending" if pending else status,
            # the LEGACY column, deliberately wrong: Model B must never read it
            "abs_reaction_3d": 0.99}


def _quarterly(n, start="2018-01-09", step_weeks=13, values=None, stock="S"):
    """n completed AMC events on Tuesdays, `step_weeks` apart, with known outcomes."""
    values = values if values is not None else [0.01 * (k + 1) for k in range(n)]
    dates = [pd.Timestamp(start) + pd.Timedelta(weeks=step_weeks * k) for k in range(n)]
    return [_event(stock, d, v) for d, v in zip(dates, values)]


def _frame(rows):
    return pd.DataFrame(rows).sort_values(["stock", "earnings_date"]).reset_index(drop=True)


def _sig(events, daily=None):
    return B.raw_signal(events, _daily() if daily is None else daily)


# ── Clocks ────────────────────────────────────────────────────────────────────

def test_cutoff_is_the_last_session_before_the_report_week_monday():
    grid = _daily()["date"].to_numpy(dtype="datetime64[ns]")
    for report in ["2019-03-04", "2019-03-06", "2019-03-08"]:          # Mon, Wed, Fri
        i = B.call_cutoff_index(grid, [report])[0]
        assert pd.Timestamp(grid[i]) == pd.Timestamp("2019-03-01")      # previous Friday


def test_holiday_friday_falls_back_to_thursday():
    grid = _daily()["date"]
    grid = grid[grid != pd.Timestamp("2019-03-01")].to_numpy(dtype="datetime64[ns]")
    i = B.call_cutoff_index(grid, ["2019-03-05"])[0]
    assert pd.Timestamp(grid[i]) == pd.Timestamp("2019-02-28")


def test_midweek_rerun_keeps_the_monday_cutoff():
    """Data ingested through Wednesday of the report week changes nothing."""
    ev = _frame(_quarterly(9) + [_event("S", "2020-05-07", None, pending=True)])
    weekend = _sig(ev, _daily("2020-05-01"))
    midweek = _sig(ev, _daily("2020-05-06"))
    pd.testing.assert_frame_equal(weekend, midweek)
    assert weekend["score_asof_date"].iloc[-1] == pd.Timestamp("2020-05-01")


def test_a_far_future_event_uses_the_latest_session_never_a_future_one():
    ev = _frame(_quarterly(9) + [_event("S", "2020-06-25", None, pending=True)])
    sig = _sig(ev, _daily("2020-05-15"))
    assert sig["score_asof_date"].iloc[-1] == pd.Timestamp("2020-05-15")
    assert (sig["score_asof_date"].dropna() <= pd.Timestamp("2020-05-15")).all()


# ── Causality ─────────────────────────────────────────────────────────────────

def test_own_reaction_never_enters_own_score():
    ev = _frame(_quarterly(10))
    before = _sig(ev)
    ev.loc[9, "abs_reaction_3d_anchored"] = 5.0
    after = _sig(ev)
    assert after.loc[9, "model_b_hist_mean_abs"] == before.loc[9, "model_b_hist_mean_abs"]
    assert after.loc[9, "model_b_n_prior"] == 9


def test_prior_outcome_enters_only_once_its_endpoint_has_closed():
    """AMC report Tuesday 2019-03-05: anchor Tue, endpoint Fri 2019-03-08. A report the same
    week (cutoff Fri 03-01) cannot see it; one the next week (cutoff Fri 03-08) can."""
    prior = _event("S", "2019-03-05", 0.10)
    same_week = _event("S", "2019-03-07", 0.02)
    next_week = _event("S", "2019-03-12", 0.02)
    sig = _sig(_frame([prior, same_week, next_week]))
    assert sig["model_b_n_prior"].tolist() == [0, 0, 1]
    assert sig["model_b_hist_mean_abs"].iloc[2] == 0.10


def test_a_bmo_endpoint_is_one_session_earlier():
    """BMO Wednesday 2019-03-06: anchor Tue, endpoint Fri 03-08 — same as AMC Tuesday."""
    prior = _event("S", "2019-03-06", 0.10, window="BMO")
    nxt = _event("S", "2019-03-11", 0.02)
    sig = _sig(_frame([prior, nxt]))
    assert sig["model_b_n_prior"].tolist() == [0, 1]


def test_unresolved_prior_outcomes_do_not_count():
    rows = _quarterly(9)
    rows[3]["reaction_3d_anchored_status"] = "unresolved_no_timestamp"      # value present!
    rows[5]["abs_reaction_3d_anchored"] = np.nan                            # NaN, not zero
    rows[5]["reaction_3d_anchored_status"] = "unresolved_intraday"
    ev = _frame(rows + [_event("S", "2020-05-07", None, pending=True)])
    sig = _sig(ev)
    expected = [r["abs_reaction_3d_anchored"] for i, r in enumerate(rows) if i not in (3, 5)]
    assert sig["model_b_n_prior"].iloc[-1] == 7
    assert sig["model_b_hist_mean_abs"].iloc[-1] == pytest.approx(np.mean(expected), abs=1e-15)


def test_event_order_does_not_matter():
    ev = _frame(_quarterly(10) + _quarterly(10, stock="T", start="2018-01-10"))
    a = _sig(ev)
    shuffled = ev.sample(frac=1, random_state=3)
    b = _sig(shuffled).loc[ev.index]
    pd.testing.assert_frame_equal(a, b)


def test_later_price_rows_do_not_alter_a_frozen_historical_call():
    ev = _frame(_quarterly(10))
    short = _sig(ev, _daily("2020-08-31"))
    full = _sig(ev, _daily())
    pd.testing.assert_frame_equal(short, full)


def test_a_newly_completed_outcome_cannot_reach_back_into_an_earlier_call():
    """Event 9's outcome becomes available only once prices reach its endpoint; the
    calls before it never change when it does."""
    ev = _frame(_quarterly(10))
    end9 = B.outcome_endpoint_index(GRID.to_numpy(dtype="datetime64[ns]"),
                                    [ev.loc[9, "anchor_date"]])[0]
    unavailable = ev.copy()
    unavailable.loc[9, "reaction_3d_anchored_status"] = "unavailable_endpoint_beyond_market_grid"
    unavailable.loc[9, "abs_reaction_3d_anchored"] = np.nan
    a = _sig(unavailable, _daily(GRID[end9 - 1]))
    b = _sig(ev, _daily())
    pd.testing.assert_frame_equal(a.iloc[:10], b.iloc[:10])


# ── Eligibility ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,scored", [(7, False), (8, True)])
def test_minimum_history(n, scored):
    ev = _frame(_quarterly(n) + [_event("S", "2020-05-07", None, pending=True)])
    out = B.score_events(ev, _daily())
    row = out.iloc[-1]
    assert row["model_b_n_prior"] == n
    assert row["model_b_hist_mean_abs"] == pytest.approx(np.mean([0.01 * (k + 1) for k in range(n)]))
    if scored:
        assert pd.notna(row["risk_score"]) and row["risk_tier"] != B.UNSCORED
    else:
        assert pd.isna(row["risk_score"]) and row["risk_tier"] == B.UNSCORED


def test_unscored_with_no_history_is_nan_not_zero():
    out = B.score_events(_frame([_event("S", "2020-05-07", None, pending=True)]), _daily())
    row = out.iloc[0]
    assert row["model_b_n_prior"] == 0
    assert pd.isna(row["model_b_hist_mean_abs"]) and pd.isna(row["risk_score"])
    assert row["risk_tier"] == B.UNSCORED


def test_pending_event_is_scored_without_its_own_timestamp():
    """The upcoming event has no announce_ts / anchor; only its PRIOR outcomes matter."""
    ev = _frame(_quarterly(9) + [_event("S", "2020-05-07", None, pending=True)])
    assert ev.iloc[-1]["announce_window"] == "UNKNOWN" and pd.isna(ev.iloc[-1]["anchor_date"])
    out = B.score_events(ev, _daily())
    assert out.iloc[-1]["risk_tier"] != B.UNSCORED


def test_model_version_and_cutoff_are_stamped():
    from config import MODEL_VERSION
    out = B.score_events(_frame(_quarterly(9)), _daily())
    assert MODEL_VERSION == "0.4.0"
    assert (out["model_version"] == MODEL_VERSION).all()
    assert out["score_asof_date"].notna().all()
    assert (out["score_asof_date"] < out["earnings_date"]).all()


# ── Target ────────────────────────────────────────────────────────────────────

def test_legacy_reaction_cannot_affect_the_signal():
    ev = _frame(_quarterly(10))
    a = _sig(ev)
    ev["abs_reaction_3d"] = 123.0
    ev["reaction_3d"] = -7.0
    pd.testing.assert_frame_equal(a, _sig(ev))


def test_model_b_source_names_no_legacy_target():
    """Static: the only reaction column scoring/model_b.py may name is the anchored one."""
    tree = ast.parse(open("scoring/model_b.py").read())
    strings = {n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    legacy = {"abs_reaction_3d", "reaction_3d", "reaction_1d", "reaction_5d",
              "is_extreme_reaction"}
    assert not (strings & legacy)
    assert B.TARGET == "abs_reaction_3d_anchored"
    assert B.TARGET_STATUS == "reaction_3d_anchored_status"


@pytest.mark.parametrize("path", ["report/report_builder.py", "report/chart_builder.py",
                                  "report/calendar_builder.py", "streamlit_dash/app.py",
                                  "streamlit_dash/streamlit_export.py",
                                  "cron/cron_weekly_digest.py", "analysis/chart_weekly.py",
                                  "publishing/archive_calls.py",
                                  "analysis/last_week_results.py",
                                  "analysis/predictions_range.py",
                                  "marketing/generate_public_track_record.py"])
def test_no_output_reads_the_legacy_target(path):
    tree = ast.parse(open(path).read())
    strings = {n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert not (strings & {"abs_reaction_3d", "reaction_3d", "reaction_1d", "reaction_5d"}), path


# ── Percentile scale ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ref():
    return B.load_reference()


def test_reference_artifact_checksum_and_metadata(ref):
    m = ref.meta
    assert m["sha256"] == B.reference_checksum(m)
    assert m["model"] == "Model B" and m["model_version"] == "0.4.0"
    assert m["min_prior"] == B.MIN_PRIOR == 8
    assert m["reference_n"] == 20491
    assert m["reference_period"].startswith("report years 2014-2025")
    assert m["raw_signal"] == "model_b_hist_mean_abs"
    assert m["validation_target"] == "abs_reaction_3d_anchored >= 0.08"
    assert len(m["source"]["refit_spec_sha256"]) == 64
    assert m["quantiles"]["percentiles"][0] == 0 and m["quantiles"]["percentiles"][-1] == 100


def test_a_tampered_reference_is_refused():
    payload = json.loads(B.REFERENCE_PATH.read_text())
    payload["quantiles"]["values"][500] *= 1.01
    with pytest.raises(ValueError, match="checksum"):
        B.reference_from_payload(payload)


def test_mapping_is_monotonic_and_clips(ref):
    x = np.linspace(-0.05, 0.5, 5001)
    p = B.to_percentile(x, ref)
    assert np.all(np.diff(p) >= 0)
    assert p[0] == 0 and p[-1] == 100
    assert B.to_percentile([ref.values[0] / 2], ref)[0] == 0
    assert B.to_percentile([ref.values[-1] * 2], ref)[0] == 100
    assert np.isnan(B.to_percentile([np.nan], ref)[0])


@pytest.mark.parametrize("p", [10, 25, 50, 80, 90, 99])
def test_reference_quantiles_map_to_their_percentile(ref, p):
    q = ref.meta["quantiles"]
    v = q["values"][q["percentiles"].index(float(p))]
    assert B.to_percentile([v], ref)[0] == pytest.approx(p, abs=0.1)


def test_tier_floors_sit_at_the_reference_p80_and_p90(ref):
    m = ref.meta
    assert B.to_percentile([m["raw_signal_at_p90"]], ref)[0] == pytest.approx(90, abs=0.05)
    assert B.to_percentile([m["raw_signal_at_p80"]], ref)[0] == pytest.approx(80, abs=0.05)


@pytest.mark.parametrize("score,tier", [
    (100.0, "High Alert"), (90.0, "High Alert"), (89.999, "Elevated"), (80.0, "Elevated"),
    (79.999, "Normal"), (0.0, "Normal"), (np.nan, "Unscored")])
def test_tier_bands(score, tier):
    assert B.assign_tier([score]).iloc[0] == tier


@pytest.mark.parametrize("score,shown", [(89.6, 89), (90.0, 90), (79.99, 79), (100.0, 100)])
def test_displayed_number_always_sits_in_its_tier_band(score, shown):
    assert B.display_score(score) == shown
    assert B.assign_tier([shown]).iloc[0] == B.assign_tier([score]).iloc[0]
    assert B.display_score(np.nan) is None


def test_tier_uses_nothing_but_the_percentile():
    """No drift, surprise, lift, volatility or options input can move a tier."""
    ev = _frame(_quarterly(9) + [_event("S", "2020-05-07", None, pending=True)])
    a = B.score_events(ev, _daily())
    for col, val in [("pre_earnings_drift_z", 9.0), ("surprise_percentage", 99.0),
                     ("stock_bucket_lift", 9.0), ("vol_30d", 0.5),
                     ("expected_move_pct", 0.5), ("atm_iv", 3.0)]:
        ev[col] = val
    b = B.score_events(ev, _daily())
    pd.testing.assert_frame_equal(a[B.MODEL_B_COLS], b[B.MODEL_B_COLS])


# ── Real data ─────────────────────────────────────────────────────────────────
# Today's output/, refreshed by every pipeline run: only invariants that hold on any data
# here. Exact counts and bit-identity (the reference sample of 20,491, the approved
# scores) are pinned to the immutable snapshot in testing/test_model_b_snapshot.py —
# ingestion revises history, so they cannot hold on live data.

EVENTS_PATH = "output/events_df.parquet"


@pytest.fixture(scope="module")
def real_events():
    if not os.path.exists(EVENTS_PATH):
        pytest.skip(f"{EVENTS_PATH} not present — run the pipeline first")
    ev = pd.read_parquet(EVENTS_PATH)
    daily = pd.read_parquet("output/full_df.parquet", columns=["stock", "date"])
    return B.score_events(ev, daily), daily


def test_real_reference_sample_is_built_on_today_s_data(real_events):
    """The reference builder runs on refreshed data and finds a plausible sample. Its
    exact size is NOT asserted here: it drifts as history is revised (the snapshot test
    pins 20,491)."""
    from scoring.build_model_b_reference import reference_sample
    ev, daily = real_events
    sample = reference_sample(ev, daily)
    assert len(sample) > 15000 and sample["model_b_hist_mean_abs"].notna().all()
    assert (pd.to_datetime(sample["earnings_date"]).dt.year >= 2014).all()


def test_real_every_pending_event_with_enough_history_is_scored(real_events):
    ev, _ = real_events
    pend = ev[ev["is_pending"]]
    assert len(pend) > 400
    enough = pend["model_b_n_prior"] >= B.MIN_PRIOR
    assert pend.loc[enough, "risk_score"].notna().all()
    assert (pend.loc[~enough, "risk_tier"] == B.UNSCORED).all()
    assert enough.mean() > 0.9


def test_real_cutoffs_are_never_after_the_data_or_the_report(real_events):
    ev, daily = real_events
    assert (ev["score_asof_date"] <= daily["date"].max()).all()
    assert (ev["score_asof_date"] < ev["earnings_date"]).all()


# ── The retired 0.3.1 scorer stays off the pipeline path ──────────────────────

LEGACY_LIBRARIES = ["scoring.scoring_features", "feature_engineering.pre_earnings_sector_features",
                    "analysis.chart_results"]


def test_pipeline_and_outputs_never_load_the_legacy_scorer():
    """Import the pipeline and every output entry point in a fresh interpreter: none of the
    kept-but-retired legacy modules may end up loaded."""
    import subprocess
    entry = ["pipeline.pipeline", "cron.cron_weekly_digest", "analysis.predictions_range",
             "analysis.last_week_results", "marketing.generate_public_track_record",
             "streamlit_dash.streamlit_export", "scoring.build_model_b_reference"]
    code = ("import sys, importlib\n"
            f"for m in {entry!r}: importlib.import_module(m)\n"
            f"print([m for m in {LEGACY_LIBRARIES!r} if m in sys.modules])")
    out = subprocess.check_output([sys.executable, "-c", code], text=True,
                                  cwd=os.path.join(os.path.dirname(__file__), ".."))
    assert out.strip().splitlines()[-1] == "[]", out


@pytest.mark.parametrize("path", ["pipeline", "scoring", "report", "streamlit_dash", "cron",
                                  "analysis", "marketing", "feature_engineering", "utilities"])
def test_no_production_module_imports_the_legacy_scorer(path):
    """Static version, covering modules the import test cannot load (e.g. the Streamlit app)."""
    from pathlib import Path
    for f in Path(path).rglob("*.py"):
        if "__pycache__" in f.parts or "model_c_refit" in f.parts or f.name == "chart_results.py":
            continue
        if str(f) == "scoring/scoring_features.py":
            continue
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                    else [n.module or ""] if isinstance(n, ast.ImportFrom) else [])
            for m in mods:
                assert not any(m == lib or m.startswith(lib + ".") for lib in LEGACY_LIBRARIES), \
                    f"{f} imports {m}"


# ── List ordering = raw Model B ordering ──────────────────────────────────────

def test_lists_are_ordered_by_the_raw_signal_not_the_percentile(ref):
    """Two raw signals above the reference maximum both clip to percentile 100; the list
    must still put the larger raw signal first. Unscored goes last even with a large raw
    mean; genuinely equal raw signals fall back to ticker."""
    top = ref.values[-1]
    df = pd.DataFrame({"stock": ["LOW", "BIG", "BIGGER", "UNS", "TIE_B", "TIE_A"],
                       "model_b_hist_mean_abs": [0.01, top * 1.5, top * 2, top * 9, 0.05, 0.05],
                       "model_b_n_prior": [20, 20, 20, 3, 20, 20]})
    df = B.engineer_risk_tier(B.engineer_risk_score(df, ref))
    assert df.loc[df["stock"].isin(["BIG", "BIGGER"]), "risk_score"].eq(100).all()  # tied pct
    assert B.sort_by_risk(df)["stock"].tolist() == ["BIGGER", "BIG", "TIE_A", "TIE_B", "LOW", "UNS"]


def test_real_pending_order_equals_raw_order(real_events):
    ev, _ = real_events
    pend = ev[ev["is_pending"]]
    got = B.sort_by_risk(pend)
    scored = got[got["risk_tier"] != B.UNSCORED]
    assert scored["model_b_hist_mean_abs"].is_monotonic_decreasing
    assert (got["risk_tier"] == B.UNSCORED).tolist() == sorted((got["risk_tier"] == B.UNSCORED).tolist())


@pytest.mark.parametrize("path", ["report/report_builder.py", "report/calendar_builder.py",
                                  "streamlit_dash/streamlit_export.py", "streamlit_dash/app.py",
                                  "cron/cron_weekly_digest.py", "analysis/chart_weekly.py",
                                  "analysis/predictions_range.py", "analysis/last_week_results.py"])
def test_no_list_is_sorted_on_the_percentile(path):
    """Ordering goes through scoring.model_b.sort_by_risk; risk_score is never a sort key."""
    tree = ast.parse(open(path).read())
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "sort_values":
            keys = [c.value for c in ast.walk(n) if isinstance(c, ast.Constant)]
            assert "risk_score" not in keys, f"{path}:{n.lineno} sorts on risk_score"
