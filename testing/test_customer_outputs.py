"""What a customer can see — Model B only, said honestly.

Every live surface (PDF report, weekly calendar, dashboard parquets and app, weekly
digest email, weekly chart table) must:

  * show the same Model B score and tier for the same event;
  * describe the score as a percentile rank — never a probability or chance, a multiple
    of a market rate, an expected move, or a direction;
  * carry nothing from the retired 0.3.1 model: no explosiveness score, no lift, no High
    Conviction, no drift/surprise flags, no legacy bucket probabilities, no 73/79 cuts;
  * render Unscored events without crashing and without implying low risk.
"""
import ast
import os
import re

import numpy as np
import pandas as pd
import pytest

from scoring import model_b as B

TODAY = pd.Timestamp("2026-10-05")           # a Monday

# A number followed by a multiplier sign: "1.9x", "2.46×", "3&times;".
MULTIPLE = re.compile(r"\d(?:\.\d+)?\s*(?:x\b|×|&times;)", re.IGNORECASE)
ALLOWED = ["It is a rank, not a probability.", "a rank, not a probability"]
FORBIDDEN = [
    (re.compile(r"probabilit", re.I), "probability"),
    (re.compile(r"\bchance", re.I), "chance"),
    (re.compile(r"\blikel(y|ihood)", re.I), "likelihood"),
    (re.compile(r"expected (to )?move", re.I), "expected move"),
    (re.compile(r"\blift\b", re.I), "lift"),
    (re.compile(r"High Conviction", re.I), "High Conviction"),
    (re.compile(r"\b(upside|downside|bullish|bearish|rally|plunge|beat streak|miss streak)\b", re.I),
     "direction or legacy flag"),
    (re.compile(r"\b(drift|surprise)\b", re.I), "legacy drift/surprise signal"),
    (re.compile(r"market average|global baseline|bucket", re.I), "legacy probability table"),
]

# Fields of the retired 0.3.1 model. No output module may name one.
LEGACY_FIELDS = ["earnings_explosiveness", "stock_bucket_lift", "is_high_conviction",
                 "peer_percentile", "pre_earnings_drift_flag", "surprise_momentum_flag",
                 "momentum_fragility", "hist_extreme_prob", "global_hist_prob",
                 "iv_vs_hist_ratio", "abs_reaction_p75", "is_extreme_reaction",
                 "is_large_reaction", "lift_vs"]

OUTPUT_MODULES = ["report/report_builder.py", "report/recommendations_builder.py",
                  "report/calendar_builder.py", "report/chart_builder.py",
                  "streamlit_dash/app.py", "streamlit_dash/streamlit_export.py",
                  "cron/cron_weekly_digest.py", "analysis/chart_weekly.py",
                  "analysis/last_week_results.py", "analysis/chart_last_week.py",
                  "analysis/predictions_range.py",
                  "marketing/generate_public_track_record.py", "pipeline/stage5.py"]
TEMPLATES = ["report/templates/earnings_report.html", "report/templates/weekly_calendar.html"]

# Tier hit rates and lifts from the audit's re-measurement of the legacy tiers
# (audit/PHASE0_AUDIT_REV2.md §Q2). Historical record, not current product evidence.
AUDIT_ERA_FIGURES = ["58.6", "45.5", "29.3", "1.91", "2.46", "2.23", "2.88", "1.87", "39.0%"]


def assert_honest(text, where):
    for ok in ALLOWED:
        text = text.replace(ok, "")
    for rx, what in FORBIDDEN:
        m = rx.search(text)
        assert not m, f"{where} says {what}: {text[max(0, m.start() - 60):m.end() + 30]!r}"
    m = MULTIPLE.search(text)
    assert not m, f"{where} shows a multiple: {text[max(0, m.start() - 60):m.end() + 20]!r}"


# ── A scored week, one event per tier ─────────────────────────────────────────

def _events():
    """Pending calls for the week of TODAY (one per tier) plus a short corrected history."""
    pend = pd.DataFrame({
        "stock":  ["HHH", "EEE", "NNN", "UUU"],
        "earnings_date": pd.to_datetime(["2026-10-06", "2026-10-06", "2026-10-07", "2026-10-08"]),
        "risk_score": [93.7, 84.2, 41.0, np.nan],
        "model_b_hist_mean_abs": [0.081, 0.061, 0.032, 0.05],
        "model_b_n_prior": [30, 22, 41, 5],
        "expected_move_pct": [0.091, np.nan, 0.04, np.nan],
        "atm_iv": [0.62, np.nan, 0.3, np.nan],
        "iv_snapshot_date": pd.to_datetime(["2026-10-02", None, "2026-10-02", None]),
    })
    pend["risk_tier"] = B.assign_tier(pend["risk_score"]).to_numpy()
    pend["is_pending"] = True
    hist_rows = []
    for s in pend["stock"]:
        for k, d in enumerate(pd.date_range("2024-01-10", periods=8, freq="91D")):
            hist_rows.append({"stock": s, "earnings_date": d, "is_pending": False,
                              "reaction_3d_anchored": (-1) ** k * 0.02 * (k + 1),
                              "reaction_3d_anchored_status":
                                  "available" if k != 3 else "unresolved_no_timestamp"})
    hist = pd.DataFrame(hist_rows)
    hist["abs_reaction_3d_anchored"] = hist["reaction_3d_anchored"].abs()
    hist.loc[hist["reaction_3d_anchored_status"] != "available",
             ["reaction_3d_anchored", "abs_reaction_3d_anchored"]] = np.nan
    ev = pd.concat([hist, pend], ignore_index=True)
    ev["sector"], ev["sub_sector"] = "Technology", "Software"
    ev["score_asof_date"] = pd.Timestamp("2026-10-02")
    ev["model_version"] = "0.4.0"
    ev["announce_window"] = "AMC"
    ev["is_extreme_move_anchored"] = (ev["abs_reaction_3d_anchored"] >= 0.08).where(
        ev["abs_reaction_3d_anchored"].notna()).astype(float)
    return ev


@pytest.fixture
def events():
    return _events()


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """Point get_run_output_dir at a temp folder so no test can wipe today's real run."""
    import utilities.output_utilities as ou
    monkeypatch.setattr(ou, "_run_output_dir", str(tmp_path))
    return tmp_path


# ── PDF report ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("stock", ["HHH", "EEE", "NNN", "UUU"])
def test_rendered_report_is_model_b_only_and_honest(events, stock):
    from report.report_builder import build_report_data, render_report_html
    row = events[events["is_pending"] & (events["stock"] == stock)].iloc[0]
    hist = events[~events["is_pending"] & (events["stock"] == stock)]
    data = build_report_data(row, hist, "Zed Corp", TODAY)
    html = render_report_html(stock, data)
    text = re.sub(r"<svg.*?</svg>", "", html, flags=re.S)
    assert_honest(text, f"report {stock}")
    for f in AUDIT_ERA_FIGURES:
        assert f not in text, f"report {stock} shows {f}"
    assert "Breakwater Risk Percentile" in text and "It is a rank, not a probability." in text
    assert row["risk_tier"] in text and "Zed Corp" in text
    if row["risk_tier"] == B.UNSCORED:
        assert "fewer than 8 usable prior corrected earnings reactions" in text
        assert data["risk_score"] is None
    else:
        assert f"{B.display_score(row['risk_score'])}<span" in text
        assert f"{row['model_b_hist_mean_abs'] * 100:.1f}%" in text
    if pd.notna(row["expected_move_pct"]):
        assert "Options Market Context" in text and "does not enter the Breakwater score" in text
    else:
        assert "Options Market Context" not in text


def test_report_tier_explanations():
    from report.recommendations_builder import TIER_BANDS
    assert TIER_BANDS["High Alert"].startswith("top decile of the reference distribution")
    assert TIER_BANDS["Elevated"].startswith("80th–90th percentile")
    assert TIER_BANDS["Normal"].startswith("below the 80th percentile")
    assert TIER_BANDS["Unscored"] == "fewer than 8 usable prior corrected earnings reactions"


def test_reaction_chart_uses_corrected_reactions_and_omits_unresolved(events):
    from report.chart_builder import corrected_reactions, reactions_chart_summary, generate_reactions_chart
    hist = events[~events["is_pending"] & (events["stock"] == "HHH")]
    used = corrected_reactions(hist)
    assert len(used) == 7 and used["reaction_3d_anchored_status"].eq("available").all()
    s = reactions_chart_summary(hist)
    assert s == {"n_shown": 7, "n_usable": 7, "n_omitted_in_span": 1}
    assert generate_reactions_chart(hist).startswith("<?xml") or "<svg" in generate_reactions_chart(hist)
    assert generate_reactions_chart(hist.iloc[0:0]) == ""


# ── Every surface shows the same score and tier ───────────────────────────────

def test_pdf_calendar_dashboard_digest_and_chart_agree(events, tmp_path, monkeypatch):
    from report.report_builder import build_report_data
    from report.calendar_builder import build_calendar_data
    from streamlit_dash.streamlit_export import export_upcoming_df
    import streamlit_dash.streamlit_export as se
    import cron.cron_weekly_digest as dg
    from analysis.chart_weekly import _build_weekly_table_df
    from pipeline.events import pending_events

    monkeypatch.setattr(se, "date", type("D", (), {"today": staticmethod(lambda: TODAY.date())}))
    pend = pending_events(events)
    want = {r.stock: (B.display_score(r.risk_score), r.risk_tier) for r in pend.itertuples()}

    pdf = {r.stock: (build_report_data(pd.Series(r._asdict()), events.iloc[0:0], "", TODAY)["risk_score"],
                     r.risk_tier) for r in pend.itertuples()}
    cal, _, _ = build_calendar_data(pend, reference_date=TODAY, window_days=7)
    calendar = {e["stock"]: (e["risk_score"], e["risk_tier"]) for e in cal}
    up_path = tmp_path / "upcoming.parquet"
    export_upcoming_df(events, str(up_path))
    up = pd.read_parquet(up_path)
    dashboard = {r.stock: (B.display_score(r.risk_score), r.risk_tier) for r in up.itertuples()}
    chart = {r.stock: (r.risk_score if pd.notna(r.risk_score) else None, r.risk_tier)
             for r in _build_weekly_table_df(up).itertuples()}
    monkeypatch.setattr(dg, "work_week_window", lambda **k: (TODAY, TODAY + pd.Timedelta(days=4)))
    digest = {r.stock: (B.display_score(r.risk_score), r.risk_tier)
              for r in dg._select_stocks(up, {}).itertuples()}

    assert pdf == calendar == dashboard == chart == want
    assert digest == {s: v for s, v in want.items() if v[1] in ("High Alert", "Elevated")}


def test_surfaces_rank_highest_first_and_unscored_last(events, tmp_path, monkeypatch):
    from report.calendar_builder import build_calendar_data
    from streamlit_dash.streamlit_export import export_upcoming_df
    import streamlit_dash.streamlit_export as se
    from pipeline.events import pending_events
    ev = events.copy()
    ev.loc[ev["is_pending"], "earnings_date"] = pd.Timestamp("2026-10-06")   # one day
    monkeypatch.setattr(se, "date", type("D", (), {"today": staticmethod(lambda: TODAY.date())}))
    cal, _, _ = build_calendar_data(pending_events(ev), reference_date=TODAY, window_days=7)
    assert [e["stock"] for e in cal] == ["HHH", "EEE", "NNN", "UUU"]
    export_upcoming_df(ev, str(tmp_path / "u.parquet"))
    assert pd.read_parquet(tmp_path / "u.parquet")["stock"].tolist() == ["HHH", "EEE", "NNN", "UUU"]


# ── Calendar, digest, chart text ──────────────────────────────────────────────

def test_calendar_html_is_honest_and_handles_unscored(events, run_dir):
    from report.calendar_builder import generate_calendar
    from pipeline.events import pending_events
    path = generate_calendar(pending_events(events), reference_date=TODAY, window_days=7)
    html = open(path).read()
    assert_honest(html, "weekly calendar")
    assert "Unscored" in html and "Risk Percentile" in html


def test_digest_html_is_honest(events, monkeypatch):
    import cron.cron_weekly_digest as dg
    from pipeline.events import pending_events
    monkeypatch.setattr(dg, "work_week_window", lambda **k: (TODAY, TODAY + pd.Timedelta(days=4)))
    sel = dg._select_stocks(pending_events(events), {"HHH": "H Corp"})
    html = dg._build_html(sel, "October 05, 2026")
    assert_honest(html, "weekly digest")
    assert sel["stock"].tolist() == ["HHH", "EEE"]
    assert "does not enter the Breakwater score" in html


def test_weekly_chart_table_is_honest(events):
    from analysis.chart_weekly import _build_weekly_table_text
    from pipeline.events import pending_events
    text = _build_weekly_table_text(pending_events(events), TODAY)
    assert_honest(text, "weekly chart table")
    assert "UUU" in text and "Unscored" in text


def test_recommendations_are_honest_for_every_tier():
    from report.recommendations_builder import build_recommendation
    for tier in B.TIERS:
        rec = build_recommendation(tier, "ZZZ", "October 07, 2026", 12, 0.071)
        assert_honest(" ".join(rec.values()), f"recommendation {tier}")


# ── No legacy field anywhere a customer can reach ─────────────────────────────

@pytest.mark.parametrize("path", OUTPUT_MODULES + TEMPLATES)
def test_output_code_names_no_legacy_field(path):
    src = open(path).read()
    if path.endswith(".py"):
        tree = ast.parse(src)
        src = " ".join(n.value for n in ast.walk(tree)
                       if isinstance(n, ast.Constant) and isinstance(n.value, str))
    found = [f for f in LEGACY_FIELDS if f in src]
    assert not found, f"{path} names {found}"


@pytest.mark.parametrize("path", OUTPUT_MODULES + TEMPLATES)
def test_output_code_carries_no_legacy_cut_points(path):
    """The 73/79 structural cuts belonged to the retired scorer."""
    src = open(path).read()
    assert not re.search(r"\b(73|79)\b", src), path


def test_dashboard_parquets_carry_only_model_b_fields(events, tmp_path):
    from streamlit_dash.streamlit_export import generate_streamlit_df
    hist_path, up_path = tmp_path / "s.parquet", tmp_path / "u.parquet"
    ev = events.copy()
    ev.loc[ev["is_pending"], "earnings_date"] = pd.Timestamp.today().normalize() + pd.Timedelta(days=2)
    ev.loc[~ev["is_pending"], "earnings_date"] = pd.Timestamp.today().normalize() - pd.Timedelta(days=30)
    # legacy columns present on the input must not leak through
    ev["earnings_explosiveness_score"], ev["is_high_conviction"] = 99.0, True
    ev["risk_score"], ev["risk_tier"] = ev["risk_score"].fillna(50.0), ev["risk_tier"].fillna("Normal")
    generate_streamlit_df(ev, str(hist_path), str(up_path))
    up = list(pd.read_parquet(up_path).columns)
    assert {"risk_score", "risk_tier", "model_b_hist_mean_abs", "model_b_n_prior"} <= set(up)
    hist = list(pd.read_parquet(hist_path).columns)
    for cols in (up, hist):
        assert not [c for c in cols if any(f in c for f in LEGACY_FIELDS)], cols
        assert "abs_reaction_3d" not in cols and "reaction_3d" not in cols


def test_public_dashboard_shows_no_reconstructed_history(events, tmp_path):
    """The public track record is paused: past events may show their corrected reaction,
    never a reconstructed Model B score or tier next to it."""
    from streamlit_dash.streamlit_export import generate_streamlit_df
    hist_path = tmp_path / "s.parquet"
    ev = events.copy()
    ev.loc[~ev["is_pending"], "earnings_date"] = pd.Timestamp.today().normalize() - pd.Timedelta(days=30)
    ev.loc[~ev["is_pending"], ["risk_score", "risk_tier"]] = [77.0, "Normal"]   # retro-scores
    generate_streamlit_df(ev, str(hist_path), str(tmp_path / "u.parquet"))
    hist = pd.read_parquet(hist_path)
    assert len(hist) and "reaction_3d_anchored" in hist.columns
    assert not {"risk_score", "risk_tier", "model_b_hist_mean_abs", "model_b_n_prior",
                "score_asof_date"} & set(hist.columns)
    app = open("streamlit_dash/app.py").read()
    assert not re.search(r"retro", app.split("# Scores appear ONLY")[0] + app.split("(Internal retro view")[1],
                         re.I)
    tree = ast.parse(app)
    tabs = [c for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "tabs"
            for c in ast.walk(n.args[0]) if isinstance(c, ast.Constant)]
    assert [t.value for t in tabs] == ["Upcoming Events", "Past Earnings Reactions", "Weekly Calendar"]


# ── README and marketing docs ─────────────────────────────────────────────────

@pytest.mark.parametrize("path", ["readme.md", "marketing/positioning.md",
                                  "marketing/post_templates.md", "marketing/reddit_playbook.md",
                                  "marketing/content_calendar.md"])
def test_public_docs_present_no_audit_era_performance_figures(path):
    """Nothing may present a tier hit rate or lift as current evidence: Model B has no
    prospective record yet and its research figures are not live performance."""
    text = open(path).read()
    shown = [f for f in AUDIT_ERA_FIGURES if f in text]
    assert not shown, f"{path} presents {shown}"


# ── Unscored is coverage, never a low-risk level ──────────────────────────────

def test_unscored_is_labelled_wherever_it_appears(events, run_dir):
    from report.calendar_builder import generate_calendar
    from report.recommendations_builder import UNSCORED_NOTE
    from analysis.chart_weekly import _build_weekly_table_text
    from pipeline.events import pending_events
    assert "not a low-risk rating" in UNSCORED_NOTE
    html = open(generate_calendar(pending_events(events), reference_date=TODAY, window_days=7)).read()
    assert "insufficient history — not low risk" in html and "not a low-risk rating" in html
    assert UNSCORED_NOTE in _build_weekly_table_text(pending_events(events), TODAY)
    assert "UNSCORED_NOTE" in open("streamlit_dash/app.py").read()
    chart = open("analysis/chart_weekly.py").read()
    assert "Not ranked (insufficient history — not low risk)" in chart
