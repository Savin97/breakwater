"""No lift figure reaches a user.

audit/PHASE0_AUDIT_REV2.md P4.1 retires every published lift figure: the per-stock and
per-tier lifts are computed from the legacy target, which mismeasures before-open
announcements, and from the lift-promoted tier. The legacy scorer still computes them
internally until Model C replaces it; these tests pin that none is DISPLAYED on any
customer-facing surface — the PDF report (tables and recommendation text), the weekly
calendar, the dashboard and the parquet the dashboard reads.
"""
import ast
import os
import re

import numpy as np
import pandas as pd
import pytest

FULL_DF_PATH = "output/full_df.parquet"

# A number followed by a multiplier sign: "1.9x", "2.46×", "3&times;".
MULTIPLE = re.compile(r"\d(?:\.\d+)?\s*(?:x\b|×|&times;)", re.IGNORECASE)
LIFT_WORDS = re.compile(r"\blift\b|market average of \S+ x|current_lift|lift_vs", re.IGNORECASE)

TIERS = ["Normal", "Elevated", "High Alert"]

# Tier hit rates and lifts from the audit's re-measurement of the legacy tiers
# (audit/PHASE0_AUDIT_REV2.md §Q2). Historical record, not current product evidence: none
# may appear in the PDF, the README or the marketing docs.
AUDIT_ERA_FIGURES = ["58.6", "45.5", "29.3", "1.91", "2.46", "2.23", "2.88", "1.87", "39.0%"]


def _assert_no_lift(text, where):
    assert not LIFT_WORDS.search(text), f"{where} shows lift: {LIFT_WORDS.search(text).group(0)!r}"
    m = MULTIPLE.search(text)
    assert not m, f"{where} shows a multiple: {text[max(0, m.start() - 60):m.end() + 20]!r}"


def _bucket_frame():
    """The per-bucket frame report_builder builds, lift columns included."""
    eb = pd.DataFrame({
        "extreme_count": [3.0, 2.0, 1.0],
        "event_count":   [20.0, 8.0, 3.0],
        "shrunk_prob":   [0.15, 0.22, 0.31],
        "global_hist_prob": [0.12, 0.25, 0.38],
    }, index=pd.Index(TIERS, name="earnings_explosiveness_bucket"))
    eb["lift_vs_baseline"] = eb["shrunk_prob"] / 0.2
    eb["lift_vs_same_bucket_global"] = eb["shrunk_prob"] / eb["global_hist_prob"]
    return eb


# ── PDF report ────────────────────────────────────────────────────────────────

def test_bucket_table_has_no_lift_columns():
    from report.report_builder import bucket_table_html_for
    html = bucket_table_html_for(_bucket_frame())
    _assert_no_lift(html, "bucket table")
    assert "Hist. Prob." in html and "Events" in html          # the table still renders


@pytest.mark.parametrize("risk_level,high_conviction", [
    ("Normal", False), ("Elevated", False), ("High Alert", False), ("High Alert", True)])
@pytest.mark.parametrize("lift", ["0.900", "3.400"])
def test_recommendation_text_never_states_a_multiple(risk_level, high_conviction, lift):
    from report.recommendations_builder import build_recommendation
    rec = build_recommendation(risk_level=risk_level, hist_extreme_prob="0.420",
                               base_extreme_prob=0.204, lift=lift, surprise_flag="Beat Streak",
                               drift_flag="Extended", high_conviction=high_conviction,
                               stock="ZZZ", earnings_date="October 07, 2026")
    text = " ".join([rec["headline"], rec["body"], rec["action"], *rec["flag_lines"]])
    _assert_no_lift(text, f"recommendation ({risk_level}, HC={high_conviction}, lift={lift})")
    assert "42.0%" in text and "20.4%" in text                  # probabilities still shown


@pytest.mark.parametrize("risk_level,high_conviction", [
    ("Normal", False), ("Elevated", False), ("High Alert", True)])
def test_rendered_report_shows_no_lift(risk_level, high_conviction):
    from report.recommendations_builder import build_recommendation
    from report.report_builder import bucket_table_html_for, render_report_html
    data = {
        "earnings_date": "October 07, 2026", "company_name": "Zed Corp",
        "generated_date": "October 03, 2026", "risk_level": risk_level, "risk_score": "81",
        "hist_extreme_prob": "0.420", "base_extreme_prob": 0.204,
        "bucket_table": bucket_table_html_for(_bucket_frame()),
        "sector": "Technology", "sub_sector": "Software", "surprise_flag": "Beat Streak",
        "drift_flag": "Extended", "high_conviction": high_conviction,
        "recommendation": build_recommendation(
            risk_level=risk_level, hist_extreme_prob="0.420", base_extreme_prob=0.204,
            lift="3.400", surprise_flag="Beat Streak", drift_flag="Extended",
            high_conviction=high_conviction, stock="ZZZ", earnings_date="October 07, 2026"),
        "peer_percentile": 90, "days_to_earnings": 4, "reactions_chart_svg": "",
        "iv_implied_move_pct": None, "atm_iv_pct": None, "iv_vs_hist_ratio": None,
    }
    html = render_report_html("ZZZ", data)
    _assert_no_lift(html, f"rendered report ({risk_level})")
    for figure in AUDIT_ERA_FIGURES:
        assert figure not in html, f"rendered report ({risk_level}) shows {figure}"
    assert "Zed Corp" in html                                     # it really rendered
    if high_conviction:
        assert "High Conviction" in html                          # the label itself stays


def test_report_builder_no_longer_hands_lift_to_the_template():
    tree = ast.parse(open("report/report_builder.py").read())
    render = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "render_report_html")
    kwargs = {k.arg for n in ast.walk(render) if isinstance(n, ast.Call) for k in n.keywords}
    assert not {k for k in kwargs if k and "lift" in k}


# ── Weekly calendar ───────────────────────────────────────────────────────────

def test_weekly_calendar_template_has_no_lift_column():
    html = open("report/templates/weekly_calendar.html").read()
    assert not re.search(r"lift", html, re.IGNORECASE)


def test_weekly_calendar_events_carry_no_lift():
    from report.calendar_builder import build_calendar_data
    df = pd.DataFrame({
        "stock": ["AAA", "BBB"], "date": pd.to_datetime(["2026-07-01"] * 2),
        "earnings_date": pd.to_datetime(["2026-07-01", "2026-07-02"]),
        "is_earnings_day": [1, 1], "is_extreme_reaction": [1, 0],
        "earnings_explosiveness_bucket": ["High Alert", "Normal"],
        "earnings_explosiveness_score": [85.0, 40.0], "momentum_fragility_score": [0.2, 0.4],
        "sector": ["Tech", "Energy"], "sub_sector": ["a", "b"],
        "pre_earnings_drift_flag": ["", ""], "surprise_momentum_flag": ["", ""],
    })
    events, _, _ = build_calendar_data(df, reference_date="2026-07-01", window_days=3)
    assert len(events) == 2
    for e in events:
        assert not [k for k in e if "lift" in k.lower()], e
        _assert_no_lift(" ".join(str(v) for v in e.values()), f"calendar event {e['stock']}")


# ── Dashboard ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path", ["streamlit_dash/app.py", "streamlit_dash/streamlit_export.py"])
def test_dashboard_code_names_no_lift_column(path):
    tree = ast.parse(open(path).read())
    strings = [n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    docstrings = {ast.get_docstring(n, clean=False) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))}
    shown = [s for s in strings if s not in docstrings
             and re.search(r"current_lift|lift_vs|\blift\b", s, re.IGNORECASE)]
    assert not shown, f"{path} still names a lift column: {shown}"


def test_dashboard_parquet_has_no_lift_columns(tmp_path):
    """Real data: the parquet the public dashboard reads carries no lift column."""
    if not os.path.exists(FULL_DF_PATH):
        pytest.skip(f"{FULL_DF_PATH} not present — run the pipeline first")
    from streamlit_dash.streamlit_export import generate_streamlit_df
    import streamlit_dash.streamlit_export as se
    df = pd.read_parquet(FULL_DF_PATH)
    out = tmp_path / "streamlit_df.parquet"
    # export_upcoming_df is exercised elsewhere; here only the historical export matters.
    orig = se.export_upcoming_df
    se.export_upcoming_df = lambda *a, **k: None
    try:
        generate_streamlit_df(df, None, output_path=str(out))
    finally:
        se.export_upcoming_df = orig
    cols = pd.read_parquet(out).columns
    assert len(cols) and not [c for c in cols if "lift" in c.lower()]


# ── README and marketing docs ─────────────────────────────────────────────────

@pytest.mark.parametrize("path", ["readme.md", "marketing/positioning.md",
                                  "marketing/post_templates.md", "marketing/reddit_playbook.md",
                                  "marketing/content_calendar.md"])
def test_public_docs_present_no_audit_era_performance_figures(path):
    """Retraction notes may name the retracted pre-audit figures; nothing may present a
    tier hit rate or lift as current evidence until the replacement model is audited."""
    text = open(path).read()
    shown = [f for f in AUDIT_ERA_FIGURES if f in text]
    assert not shown, f"{path} presents {shown}"
