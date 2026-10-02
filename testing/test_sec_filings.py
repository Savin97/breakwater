"""Tests for the SEC filings feasibility pilot (`research/sec_filings_pilot/`).

Synthetic fixtures only; nothing here touches the network or depends on SEC being up.
"""
from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.sec_filings_pilot import acquire, build, documents, mapping, timing

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "research" / "sec_filings_pilot"
T = pd.Timestamp


# ─────────────────────────────────────── fixtures ───────────────────────────────────────
def _filings(rows):
    """rows: (accession, form, filing_date, items)"""
    df = pd.DataFrame(rows, columns=["accession", "form", "filing_date", "items"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["base_form"] = df["form"].str.removesuffix("/A")
    df["is_amendment"] = df["form"].str.endswith("/A")
    df["report_period"] = pd.NaT
    df["cik"] = 1
    df["stock"] = "AAA"
    return df


def _event(cutoff="2024-01-05", event_id="AAA|2024-01-10"):
    cd = T(cutoff)
    return pd.DataFrame([{"event_id": event_id, "stock": "AAA", "call_cutoff_date": cd,
                          "call_cutoff_ts": timing.cutoff_instant(pd.Series([cd])).iloc[0],
                          "call_cutoff_idx": 10, "identity_status": "mapped"}])


GRID = pd.bdate_range("2023-01-02", "2024-12-31").to_numpy("datetime64[ns]")


def _select(filings, ev=None, verified=None):
    return build.select(ev if ev is not None else _event(), filings, GRID, verified)


# ──────────────────────────────────── point in time ─────────────────────────────────────
def test_no_filing_after_the_cutoff_is_selected():
    f = _filings([("a1", "10-Q", "2023-11-01", ""), ("a2", "10-K", "2024-01-08", ""),
                  ("a3", "8-K", "2024-01-09", "2.02,9.01"), ("a4", "8-K", "2024-01-03", "8.01")])
    per, k8, _ = _select(f)
    assert per.loc[0, "periodic_accession"] == "a1"
    assert set(k8["accession"]) == {"a4"}


def test_latest_eligible_periodic_is_selected_and_amendments_are_not():
    f = _filings([("q2", "10-Q", "2023-08-01", ""), ("q3", "10-Q", "2023-11-01", ""),
                  ("q3a", "10-Q/A", "2023-12-01", ""), ("k", "10-K", "2023-02-20", "")])
    per, _, am = _select(f)
    assert per.loc[0, "periodic_accession"] == "q3"
    assert per.loc[0, "periodic_form"] == "10-Q"
    assert list(am["accession"]) == ["q3a"]          # recorded separately, not selected


def test_8k_interval_is_strictly_after_periodic_and_not_after_cutoff():
    f = _filings([("q", "10-Q", "2023-11-01", ""),
                  ("before", "8-K", "2023-10-25", "2.02"),     # before the 10-Q: out
                  ("same", "8-K", "2023-11-01", "2.02"),       # same day: ambiguous, excluded
                  ("in1", "8-K", "2023-11-02", "8.01"),
                  ("in2", "8-K/A", "2024-01-04", "5.02"),
                  ("onday", "8-K", "2024-01-05", "7.01"),      # cutoff day, no verified time
                  ("late", "8-K", "2024-01-08", "2.02")])
    _, k8, _ = _select(f)
    inset = k8[k8["relation"].eq("after_periodic")]
    assert set(inset["accession"]) == {"in1", "in2"}
    assert set(k8.loc[k8["relation"].eq("same_day_as_periodic_excluded"), "accession"]) == {"same"}
    assert (inset["filing_date"] > T("2023-11-01")).all()
    assert (inset["filing_date"] < T("2024-01-05")).all()


def test_amended_8k_is_kept_but_distinguished():
    f = _filings([("q", "10-Q", "2023-11-01", ""), ("x", "8-K/A", "2023-12-01", "9.01")])
    _, k8, _ = _select(f)
    assert k8.loc[0, "form"] == "8-K/A"


def test_cutoff_day_filing_needs_a_verified_time_before_the_close():
    f = _filings([("q", "10-Q", "2023-11-01", ""), ("early", "8-K", "2024-01-05", "7.01"),
                  ("late", "8-K", "2024-01-05", "8.01"), ("unk", "8-K", "2024-01-05", "8.01")])
    verified = {"early": T("2024-01-05 15:59:00"), "late": T("2024-01-05 16:00:01")}
    _, k8, _ = _select(f, verified=verified)
    assert set(k8["accession"]) == {"early"}


def test_early_close_cutoff_uses_the_real_close():
    # 2023-11-24, day after Thanksgiving: NYSE closes 13:00.
    ts = timing.cutoff_instant(pd.Series([T("2023-11-24")])).iloc[0]
    assert ts == T("2023-11-24 13:00")
    ok = timing.eligible(pd.Series([T("2023-11-24")] * 2), T("2023-11-24"), ts,
                         pd.Series([T("2023-11-24 12:59"), T("2023-11-24 14:00")]))
    assert ok.tolist() == [True, False]


def test_cutoff_that_is_not_a_session_raises():
    with pytest.raises(ValueError):
        timing.cutoff_instant(pd.Series([T("2024-01-06")]))       # a Saturday


def test_date_only_rule_is_strictly_earlier_day():
    fd = pd.Series([T("2024-01-04"), T("2024-01-05"), pd.NaT])
    ok = timing.eligible(fd, T("2024-01-05"), T("2024-01-05 16:00"))
    assert ok.tolist() == [True, False, False]


def test_earnings_release_after_cutoff_cannot_leak():
    f = _filings([("q", "10-Q", "2023-11-01", ""), ("rel", "8-K", "2024-01-10", "2.02,9.01")])
    ev = _event()
    ev["phase3_announce_date"] = T("2024-01-10")
    ev["report_date"] = T("2024-01-10")
    per, k8, _ = _select(f, ev)
    rel = build.release_8ks(ev, f)
    assert list(rel["accession"]) == ["rel"]          # found by the leakage audit ...
    assert "rel" not in set(k8["accession"])           # ... and absent from the info set


def test_genuine_pre_cutoff_preannouncement_stays_eligible():
    f = _filings([("q", "10-Q", "2023-11-01", ""), ("pre", "8-K", "2023-12-15", "2.02,7.01")])
    _, k8, _ = _select(f)
    assert "pre" in set(k8["accession"])


# ───────────────────────────────────────── clocks ───────────────────────────────────────
def test_json_acceptance_is_read_as_utc_only_for_audit():
    # EDT: 20:27Z -> 16:27 NY; EST: 21:05Z -> 16:05 NY.
    assert timing.parse_json_acceptance_as_utc("2026-06-30T20:27:26.000Z") == T("2026-06-30 16:27:26")
    assert timing.parse_json_acceptance_as_utc("2025-01-15T21:05:00.000Z") == T("2025-01-15 16:05:00")
    with pytest.raises(ValueError):
        timing.parse_json_acceptance_as_utc("2025-01-15 21:05:00")


def test_verified_acceptance_is_ny_wall_clock_and_tz_aware_input_is_converted():
    assert timing.parse_verified_acceptance("2025-04-11 06:45:50") == T("2025-04-11 06:45:50")
    assert timing.parse_verified_acceptance("20250411064550") == T("2025-04-11 06:45:50")
    aware = T("2025-04-11 10:45:50", tz="UTC")
    assert timing.parse_verified_acceptance(aware) == T("2025-04-11 06:45:50")
    with pytest.raises(ValueError):
        timing.parse_verified_acceptance(pd.Timestamp("2025-04-11 06:45:50").to_pydatetime())


def test_json_consistency_check_catches_the_double_shift():
    # JPMorgan-style record: true 16:31 ET stored as 02:31Z next day; same-day filing date.
    lit = pd.Series([timing.parse_json_acceptance_as_utc("2025-12-06T02:31:42.000Z"),
                     timing.parse_json_acceptance_as_utc("2025-12-05T21:31:42.000Z")])
    days = pd.bdate_range("2025-12-01", "2025-12-31")
    ok = timing.json_acceptance_consistent(lit, pd.Series([T("2025-12-05")] * 2), days)
    assert ok.tolist() == [False, True]


# ───────────────────────────────────────── mapping ──────────────────────────────────────
def _tickers(rows):
    return mapping.tickers_frame({str(i): {"cik_str": c, "ticker": t, "title": n}
                                  for i, (c, t, n) in enumerate(rows)})


def _universe(rows):
    return pd.DataFrame(rows, columns=["stock", "bw_name", "sector"]).assign(
        renamed_to=None, benzinga_identity_hazard=None)


def _sub(cik, name, former=()):
    return {"cik": f"{cik:010d}", "name": name,
            "formerNames": [{"name": n} for n in former]}


def test_ambiguous_ticker_is_not_guessed(monkeypatch):
    monkeypatch.setattr(mapping, "MANUAL", {})
    t = _tickers([(1, "XYZ.B", "Foo Corp"), (2, "XYZ-B", "Bar Corp")])
    a, seg = mapping.resolve(_universe([("XYZ-B", "Foo", "X")]), t,
                             {1: _sub(1, "Foo Corp"), 2: _sub(2, "Bar Corp")})
    assert a.loc[0, "status"] == "ambiguous"
    assert seg.empty


def test_name_mismatch_is_not_linked(monkeypatch):
    monkeypatch.setattr(mapping, "MANUAL", {})
    t = _tickers([(1, "DOC", "Healthpeak Properties")])
    a, seg = mapping.resolve(_universe([("DOC", "Physicians Realty Trust", "RE")]), t,
                             {1: _sub(1, "HEALTHPEAK PROPERTIES, INC.")})
    assert a.loc[0, "status"] == "name_mismatch"
    assert seg.empty


def test_dot_dash_spellings_reach_the_same_class_only():
    assert mapping.ticker_variants("BRK-B") == ["BRK-B", "BRK.B", "BRKB"]
    assert "BRK-A" not in mapping.ticker_variants("BRK-B")


def test_manual_link_failing_name_verification_is_dropped(monkeypatch):
    monkeypatch.setattr(mapping, "MANUAL", {"OLD": [mapping.ManualLink(
        9, None, None, "Some Name", "test")]})
    a, seg = mapping.resolve(_universe([("OLD", "Old Co", "X")]), _tickers([]),
                             {9: _sub(9, "Completely Different Inc")})
    assert a.loc[0, "status"] == "unresolved"
    assert a.loc[0, "manual_failed_verification"] == "9"
    assert seg.empty


def test_predecessor_chain_splits_filings_at_the_boundary():
    seg = pd.DataFrame([
        {"stock": "GOOGL", "cik": 1, "valid_from": T("1990-01-01"), "valid_to": T("2015-10-02"),
         "provenance": "manual_verified", "sec_name": "GOOGLE INC.", "reason": "pred"},
        {"stock": "GOOGL", "cik": 2, "valid_from": T("2015-10-02"), "valid_to": T("2099-12-31"),
         "provenance": "manual_verified", "sec_name": "Alphabet", "reason": "succ"}])
    fil = pd.DataFrame({"cik": [1, 1, 2, 2], "accession": ["g1", "g2", "a1", "a0"],
                        "filing_date": pd.to_datetime(["2015-07-23", "2016-02-01",
                                                       "2015-10-29", "2015-08-11"])})
    sf = build.stock_filings(fil, seg)
    # Google Inc.'s post-reorganisation filing and Alphabet's pre-completion one are both out.
    assert set(sf["accession"]) == {"g1", "a1"}


def test_duplicate_accession_across_a_chain_is_kept_once():
    seg = pd.DataFrame([
        {"stock": "S", "cik": c, "valid_from": T("1990-01-01"), "valid_to": T("2099-12-31"),
         "provenance": "p", "sec_name": "n", "reason": "r"} for c in (1, 2)])
    fil = pd.DataFrame({"cik": [1, 2], "accession": ["same", "same"],
                        "filing_date": pd.to_datetime(["2020-01-01", "2020-01-01"])})
    assert len(build.stock_filings(fil, seg)) == 1


def test_duplicate_accession_within_a_cik_is_deduplicated_and_counted():
    raw = pd.DataFrame({
        "accessionNumber": ["0001-24-1", "0001-24-1", "0001-24-2"], "filingDate": ["2024-01-02"] * 3,
        "reportDate": ["", "", ""], "acceptanceDateTime": ["2024-01-02T21:00:00.000Z"] * 3,
        "form": ["8-K", "8-K", "10-Q"], "items": ["", "", ""], "size": [1, 1, 1],
        "isXBRL": [0] * 3, "isInlineXBRL": [0] * 3, "primaryDocument": ["a.htm"] * 3,
        "primaryDocDescription": [""] * 3, "fileNumber": [""] * 3, "cik": [5] * 3,
        "source_page": ["recent", "p1", "recent"]})
    f, stats = build.tidy_filings(raw, "snap")
    assert stats["duplicate_accession_rows_within_cik"] == 1
    assert stats["duplicate_accession_conflicts"] == 0
    assert f["accession"].is_unique
    assert f.set_index("accession").loc["0001-24-1", "is_amendment"] == False  # noqa: E712


# ──────────────────────────────────────── documents ─────────────────────────────────────
INDEX = b"""<div class="infoHead">Filing Date</div>
<div class="info">2017-05-09</div><div class="infoHead">Accepted</div>
<div class="info">2017-05-09 16:21:40</div>
<table class="tableFile" summary="Document Format Files">
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>1</td><td>FORM 8-K</td><td><a href="/ix?doc=/Archives/edgar/data/1/0001/f8k.htm">f8k.htm</a></td><td>8-K</td><td>29546</td></tr>
<tr><td>2</td><td>Q1 PRESS RELEASE</td><td><a href="/Archives/edgar/data/1/0001/pr.htm">pr.htm</a></td><td>EX-99.1</td><td>330002</td></tr>
<tr><td>&nbsp;</td><td>Complete submission text file</td><td><a href="/x.txt">x.txt</a></td><td>&nbsp;</td><td>1</td></tr>
</table>"""


def test_index_page_parse_and_exhibit_discovery():
    p = documents.parse_index(INDEX)
    assert p["accepted"] == "2017-05-09 16:21:40"
    types = {d["filename"]: d["type"] for d in p["documents"]}
    assert types == {"f8k.htm": "8-K", "pr.htm": "EX-99.1", "x.txt": ""}
    assert documents.is_earnings_release(p["documents"][1])
    assert not documents.is_earnings_release(p["documents"][0])


def test_html_to_text_drops_hidden_ixbrl_and_counts_tables():
    raw = ("<html><body><div style='display:none'><ix:header>hidden facts</ix:header></div>"
           "<p>" + "word " * 40 + "</p><table><tr><td>Revenue</td><td>1,234</td>"
           "<td>5.6%</td></tr></table></body></html>")
    st = documents.html_to_text(raw)
    assert "hidden" not in st.text
    assert st.prose_blocks == 1 and st.n_tables == 1 and st.numeric_table_chars > 0


def test_decode_respects_declared_charset():
    b = '<meta charset="windows-1252"><p>Caf\xe9 \x93quoted\x94</p>'.encode("latin-1")
    d = documents.decode(b)
    assert d.encoding == "windows-1252" and "“quoted”" in d.text


def test_section_locator_prefers_body_over_table_of_contents():
    body = "x " * 3000
    text = ("Item 1A. Risk Factors 12\nItem 1B. Unresolved\n...\n"
            f"Item 1A. Risk Factors\n{body}\nItem 1B. Unresolved Staff Comments")
    s = documents.locate_sections(text, "10-K")
    assert s["risk_factors_found"] and s["risk_factors_chars"] > 5000


# ──────────────────────────────────── caching / storage ─────────────────────────────────
def test_raw_sec_cache_is_gitignored():
    for p in ["data/vendor/sec/archives/1/2/x.htm", "data/vendor/sec/snapshots/s/a.json.gz",
              "output/sec_filings_pilot/events.parquet"]:
        r = subprocess.run(["git", "check-ignore", "-q", p], cwd=REPO)
        assert r.returncode == 0, f"{p} is not gitignored"


class _Resp:
    def __init__(self, content, status=200):
        self.content, self.status_code, self.headers = content, status, {}


class _Client:
    def __init__(self, content):
        self.content, self.calls = content, 0

    def get(self, url):
        self.calls += 1
        return _Resp(self.content)


def test_archive_is_fetch_once_and_never_overwritten(tmp_path):
    c = _Client(b"v1")
    ar = acquire.Archive(c, root=tmp_path / "a", manifest=tmp_path / "m.jsonl",
                         changes=tmp_path / "c.jsonl")
    assert ar.get(1, "0001-24-1", "f.htm")[0] == b"v1"
    assert ar.get(1, "0001-24-1", "f.htm")[0] == b"v1"
    assert c.calls == 1
    c.content = b"v2"                                # SEC now serves something else
    assert ar.verify(1, "0001-24-1", "f.htm") is False
    assert acquire.archive_path(1, "0001-24-1", "f.htm", tmp_path / "a").read_bytes() == b"v1"
    change = json.loads((tmp_path / "c.jsonl").read_text())
    assert change["old_sha256"] != change["new_sha256"]


def test_snapshot_is_sealed_read_only_and_never_reopened(tmp_path):
    snap = acquire.Snapshot("sec_test", root=tmp_path, client=_Client(b'{"a": 1}'))
    snap.fetch("company_tickers.json", "u")
    final = snap.finish()
    p = final / "company_tickers.json.gz"
    assert p.exists() and not (p.stat().st_mode & 0o222)
    assert json.loads((final / "manifest.json").read_text())["n_files"] == 1
    with pytest.raises(FileExistsError):
        acquire.Snapshot("sec_test", root=tmp_path)


def test_user_agent_is_required_and_not_hardcoded(monkeypatch):
    monkeypatch.delenv(acquire.USER_AGENT_ENV, raising=False)
    with pytest.raises(acquire.SECAccessError):
        acquire.user_agent()
    for f in PKG.glob("*.py"):
        assert "@" not in "".join(c for c in _string_constants(f) if "." in c and " " not in c), f


def _string_constants(path):
    tree = ast.parse(path.read_text())
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


# ─────────────────────────────────────── no outcomes ────────────────────────────────────
def test_pilot_reads_no_outcome_column():
    forbidden = {"abs_reaction_3d_anchored", "abs_r3", "abs_r1", "y_extreme", "abs_reaction_3d",
                 "reaction_3d_anchored", "reaction_1d_anchored", "is_extreme_reaction"}
    for f in PKG.glob("*.py"):
        assert not forbidden & set(_string_constants(f)), f
    assert not forbidden & set(build.EVENT_COLS)
