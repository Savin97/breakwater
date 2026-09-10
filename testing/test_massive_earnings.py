"""Tests for the Massive/Benzinga source-evaluation tooling (`research/massive/`).

The tooling is research scaffolding, but it is scaffolding that decides whether a vendor
becomes Breakwater's historical timing source, so the invariants that matter are the ones
that would let it lie: a short acquisition that looks complete, a snapshot that changed
after the analysis ran, a filler timestamp read as a real one, a key on disk, and — the two
structural ones — any path by which this code could touch production data, or by which
production could come to depend on a vendor file.

Nothing here needs the network, a key, or a database.
"""
import ast
import gzip
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
import requests

from feature_engineering.announcement_timing import AMC, BMO, INTRADAY, UNKNOWN
from research.massive import acquire, client, completeness, identity, normalize, paths, validate

REPO = Path(__file__).resolve().parents[1]


# ───────────────────────────── a fake vendor, so no test needs a key ─────────────────────
class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class FakeVendor:
    """A cursor-paginated API with a fixed corpus. Records every URL and header it sees."""

    def __init__(self, records, page_size=2, fail_on_page=None, status_sequence=None):
        self.records = list(records)
        self.page_size = page_size
        self.fail_on_page = fail_on_page
        self.status_sequence = list(status_sequence or [])
        self.calls = []
        self.headers_seen = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        self.headers_seen.append(dict(headers or {}))
        if self.status_sequence:
            status = self.status_sequence.pop(0)
            if status != 200:
                return _Resp({"status": "ERROR"}, status=status)
        offset = int(params.get("cursor", 0)) if params else 0
        if "cursor=" in url:
            offset = int(url.split("cursor=")[1].split("&")[0])
        page_no = offset // self.page_size + 1
        if self.fail_on_page and page_no == self.fail_on_page:
            raise requests.ConnectionError("vendor went away mid-walk")
        chunk = self.records[offset:offset + self.page_size]
        nxt = (f"https://fake/earnings?cursor={offset + self.page_size}"
               if offset + self.page_size < len(self.records) else None)
        return _Resp({"status": "OK", "request_id": f"r{page_no}",
                      "results": chunk, "next_url": nxt})


def _record(i, ticker="AAA", date="2024-05-01", time="16:05:00", **kw):
    rec = {"benzinga_id": f"id{i}", "ticker": ticker, "company_name": "Fake Co",
           "date": date, "time": time, "date_status": "confirmed", "currency": "USD",
           "importance": 1, "fiscal_year": 2024, "fiscal_period": "Q1",
           "last_updated": "2024-05-02T10:00:00Z"}
    rec.update(kw)
    return rec


def _client(vendor, **kw):
    return client.MassiveClient(base_url="https://fake", session=vendor, sleep=lambda _s: None,
                                auth_header=lambda: {"Authorization": "Bearer TESTKEY"}, **kw)


# ───────────────────────────────────────── pagination ────────────────────────────────────
def test_pagination_walks_every_page_to_exhaustion():
    vendor = FakeVendor([_record(i) for i in range(7)], page_size=2)
    got = [r for _p, _u, pl in _client(vendor).iter_pages() for r in pl["results"]]
    assert [r["benzinga_id"] for r in got] == [f"id{i}" for i in range(7)]
    assert len(vendor.calls) == 4          # 2+2+2+1, then the vendor stops offering a cursor


def test_pagination_sends_the_query_only_on_the_first_request():
    """The cursor carries the query. Re-sending `sort` alongside it is how a walk restarts."""
    vendor = FakeVendor([_record(i) for i in range(5)], page_size=2)
    list(_client(vendor).iter_pages(params={"sort": "date.asc", "limit": 50000}))
    assert vendor.calls[0][1] == {"sort": "date.asc", "limit": 50000}
    assert all(params == {} for _url, params in vendor.calls[1:])


def test_pagination_stops_instead_of_looping_when_the_cursor_repeats():
    class Stuck(FakeVendor):
        def get(self, url, params=None, headers=None, timeout=None):
            self.calls.append((url, params))
            return _Resp({"status": "OK", "results": [_record(0)],
                          "next_url": "https://fake/earnings?cursor=0"})

    pages = list(_client(Stuck([])).iter_pages())
    assert len(pages) <= 2                 # it terminates rather than spinning forever


def test_pagination_stops_on_an_empty_page_even_if_a_cursor_comes_back():
    class Empty(FakeVendor):
        def get(self, url, params=None, headers=None, timeout=None):
            self.calls.append((url, params))
            return _Resp({"status": "OK", "results": [], "next_url": "https://fake/x?cursor=9"})

    assert len(list(_client(Empty([])).iter_pages())) == 1


def test_a_retryable_status_is_retried_and_a_fatal_one_is_not():
    vendor = FakeVendor([_record(0)], page_size=2, status_sequence=[503, 429, 200])
    assert list(_client(vendor).iter_pages())          # succeeds on the third attempt
    fatal = FakeVendor([_record(0)], page_size=2, status_sequence=[404])
    with pytest.raises(client.MassiveAPIError):
        list(_client(fatal).iter_pages())
    assert len(fatal.calls) == 1                        # 404 is not retried


def test_a_vendor_error_status_is_raised_not_returned_as_data():
    """A swallowed error produces a snapshot that is silently short — the one failure mode
    a completeness audit cannot detect after the fact."""
    class Bad(FakeVendor):
        def get(self, url, params=None, headers=None, timeout=None):
            return _Resp({"status": "NOT_AUTHORIZED", "results": []})

    with pytest.raises(client.MassiveAPIError):
        list(_client(Bad([])).iter_pages())


# ──────────────────────────────── acquisition: interruption & immutability ───────────────
def test_an_interrupted_acquisition_leaves_no_snapshot_and_resumes_where_it_stopped(tmp_path):
    records = [_record(i) for i in range(6)]
    vendor = FakeVendor(records, page_size=2, fail_on_page=3)
    with pytest.raises(client.MassiveAPIError):
        acquire.acquire(root=tmp_path, client=_client(vendor, max_retries=2),
                        snapshot_name="snap", params={})

    assert not (tmp_path / "snap").exists()             # nothing finished, nothing published
    partial = tmp_path / "snap.partial"
    assert partial.exists() and (partial / acquire.PROGRESS_NAME).exists()
    progress = json.loads((partial / acquire.PROGRESS_NAME).read_text())
    assert len(progress["pages"]) == 2                  # the two pages that did land

    resumed = acquire.acquire(root=tmp_path, client=_client(FakeVendor(records, page_size=2)),
                              snapshot_name="snap", params={})
    ids = [r["benzinga_id"] for r in acquire.iter_records(resumed)]
    assert ids == [f"id{i}" for i in range(6)]          # resumed, not restarted, not short
    assert acquire.read_manifest(resumed)["record_count"] == 6


def test_a_partial_directory_is_never_mistaken_for_a_snapshot(tmp_path):
    (tmp_path / "snap.partial" / "pages").mkdir(parents=True)
    assert acquire.latest_snapshot(tmp_path) is None


def test_a_finished_snapshot_is_never_overwritten(tmp_path):
    vendor = FakeVendor([_record(0)], page_size=2)
    acquire.acquire(root=tmp_path, client=_client(vendor), snapshot_name="snap", params={})
    with pytest.raises(FileExistsError):
        acquire.acquire(root=tmp_path, client=_client(FakeVendor([_record(9)])),
                        snapshot_name="snap", params={})


def test_a_finished_snapshot_is_read_only_on_disk(tmp_path):
    out = acquire.acquire(root=tmp_path, client=_client(FakeVendor([_record(0)])),
                          snapshot_name="snap", params={})
    for f in out.rglob("*"):
        assert not (f.stat().st_mode & 0o222), f"{f} is writable"


def test_the_manifest_checksums_detect_a_tampered_page(tmp_path):
    out = acquire.acquire(root=tmp_path, client=_client(FakeVendor([_record(i) for i in range(3)],
                                                                  page_size=2)),
                          snapshot_name="snap", params={})
    assert acquire.verify_snapshot(out)["ok"] is True
    page = out / acquire.read_manifest(out)["pages"][0]["file"]
    os.chmod(page, 0o644)
    with gzip.open(page, "wt") as fh:
        json.dump({"status": "OK", "results": [_record(99)]}, fh)
    report = acquire.verify_snapshot(out)
    assert report["ok"] is False and report["corrupt"] == [str(page.relative_to(out))]


# ─────────────────────────────────────── key secrecy ─────────────────────────────────────
def test_the_api_key_never_reaches_the_snapshot_or_a_url(tmp_path):
    secret = "sk-do-not-write-me"
    vendor = FakeVendor([_record(0)], page_size=2)
    c = client.MassiveClient(base_url="https://fake", session=vendor, sleep=lambda _s: None,
                             auth_header=lambda: {"Authorization": f"Bearer {secret}"})
    out = acquire.acquire(root=tmp_path, client=c, snapshot_name="snap", params={})
    assert vendor.headers_seen and all(secret in h["Authorization"] for h in vendor.headers_seen)
    for f in out.rglob("*"):
        if f.is_file():
            assert secret.encode() not in f.read_bytes(), f"key leaked into {f}"
    assert all(secret not in url for url, _p in vendor.calls)


def test_scrub_strips_credential_shaped_query_parameters():
    assert client.scrub("https://x/y?cursor=abc&apikey=SECRET&token=T&limit=5") == \
        "https://x/y?cursor=abc&limit=5"


def test_the_key_is_read_from_the_environment_and_not_from_an_argument(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.setattr(client, "load_dotenv", None)
    with pytest.raises(client.MassiveAPIError):
        client._auth_header()
    monkeypatch.setenv("MASSIVE_API_KEY", "abc")
    assert client._auth_header() == {"Authorization": "Bearer abc"}


# ───────────────────────────── normalization: schema and clocks ──────────────────────────
def test_a_record_missing_every_optional_field_still_normalizes():
    thin = {"benzinga_id": "x", "ticker": "aaa", "date": "2024-05-01", "time": "07:00:00"}
    out = normalize.normalize([thin])
    assert len(out) == 1
    assert out.loc[0, "ticker_norm"] == "AAA"
    assert out.loc[0, "announce_window"] == BMO
    for col in normalize.VENDOR_FIELDS:
        assert col in out.columns             # absent optional fields become empty columns


def test_an_unexpected_vendor_field_is_preserved_rather_than_dropped():
    out = normalize.normalize([_record(0, brand_new_field="keep me")])
    assert out.loc[0, "brand_new_field"] == "keep me"


def test_every_vendor_field_survives_normalization():
    rec = _record(0, notes="n", actual_eps=1.0, eps_method="adj")
    out = normalize.normalize([rec])
    for key, value in rec.items():
        assert out.loc[0, key] == value


@pytest.mark.parametrize("clock, window", [
    ("09:29:59", BMO),        # one second before the open
    ("09:30:00", INTRADAY),   # the open itself is NOT before the open
    ("09:30:01", INTRADAY),
    ("15:59:59", INTRADAY),   # one second before the close
    ("16:00:00", AMC),        # the close itself IS after the close
    ("16:00:01", AMC),
    ("06:30:00", BMO),
    ("20:00:00", AMC),
])
def test_the_classification_boundaries_are_exactly_0930_and_1600(clock, window):
    out = normalize.normalize([_record(0, time=clock)])
    assert out.loc[0, "announce_window"] == window


@pytest.mark.parametrize("bad", ["", None, "nan", "noon", "25:00:00", "16:61:00", "7pm",
                                 "2024-05-01T16:00:00"])
def test_a_malformed_or_missing_time_is_unknown_and_never_guessed(bad):
    out = normalize.normalize([_record(0, time=bad)])
    assert out.loc[0, "announce_window"] == UNKNOWN
    assert not out.loc[0, "time_usable"]
    assert pd.isna(out.loc[0, "announce_ts_vendor"])
    assert out.loc[0, "time_quality"] in (normalize.TIME_MISSING, normalize.TIME_MALFORMED)


def test_midnight_is_filler_not_a_bmo_announcement():
    """00:00 < 09:30, so the literal rule would call it BMO. That would fabricate a
    confident label out of the vendor's placeholder for 'time unknown'."""
    out = normalize.normalize([_record(0, time="00:00:00")])
    assert out.loc[0, "time_quality"] == normalize.TIME_MIDNIGHT
    assert out.loc[0, "announce_window"] == UNKNOWN
    assert not out.loc[0, "time_usable"]

    literal = normalize.normalize([_record(0, time="00:00:00")], midnight_is_real=True)
    assert literal.loc[0, "announce_window"] == BMO   # the sensitivity is measurable


def test_the_vendor_clock_is_taken_as_published_with_no_dst_shift():
    """§2 of the audit establishes the field is New York local, not fixed EST. A silent
    +1h during EDT would move 16:xx events across the AMC cut."""
    summer = normalize.normalize([_record(0, date="2024-07-01", time="15:30:00")])
    winter = normalize.normalize([_record(0, date="2024-01-08", time="15:30:00")])
    assert summer.loc[0, "announce_ts_vendor"] == pd.Timestamp("2024-07-01 15:30")
    assert winter.loc[0, "announce_ts_vendor"] == pd.Timestamp("2024-01-08 15:30")
    assert summer.loc[0, "announce_window"] == winter.loc[0, "announce_window"] == INTRADAY


# ────────────────────────────────────── duplicates ───────────────────────────────────────
def test_duplicate_reporting_is_flagged_and_never_silently_dropped():
    recs = [_record(1), _record(2), _record(3, date="2024-08-01")]
    out = normalize.normalize(recs)
    assert len(out) == 3                                  # nothing dropped
    same_day = out[out["report_date"] == pd.Timestamp("2024-05-01")]
    assert (same_day["n_records_ticker_date"] == 2).all()
    assert not out["is_duplicate_benzinga_id"].any()

    repeated_id = normalize.normalize([_record(1), _record(1)])
    assert repeated_id["is_duplicate_benzinga_id"].all()

    report = validate.duplicate_report(out)
    assert report["records_sharing_ticker_date"] == 2
    assert report["ticker_date_pairs_affected"] == 1


def test_the_join_view_collapses_a_duplicated_date_but_keeps_the_count_visible():
    out = normalize.normalize([_record(1, time="00:00:00", date_status="projected"),
                               _record(2, time="16:05:00")])
    view = validate.vendor_event_view(out)
    assert len(view) == 1
    assert view.iloc[0]["benzinga_id"] == "id2"           # confirmed + usable time wins
    assert view.iloc[0]["vendor_records_on_date"] == 2


# ─────────────────────────────── anchoring / validation helpers ──────────────────────────
def test_amc_on_friday_and_bmo_on_monday_anchor_to_the_same_session():
    """Two descriptions of one instant. A window disagreement that is really a date
    convention difference must not be counted as a timing error."""
    windows = pd.Series([AMC, BMO, INTRADAY])
    dates = pd.Series(pd.to_datetime(["2024-05-03", "2024-05-06", "2024-05-06"]))
    anchors = validate.anchor_session(windows, dates)
    assert anchors[0] == anchors[1] == pd.Timestamp("2024-05-03")
    assert pd.isna(anchors[2])                            # INTRADAY has no anchor


def test_a_one_day_date_disagreement_is_not_hidden_by_the_tolerant_join():
    """`merge_asof` keeps the LEFT frame's date; without carrying the vendor's own date
    through, a +/-1 day mismatch would score as agreement."""
    bz = normalize.normalize([_record(1, ticker="AAA", date="2024-05-02", time="06:30:00")])
    ref = pd.DataFrame({"ticker_norm": pd.Series(["AAA"], dtype="string"),
                        "report_date": pd.to_datetime(["2024-05-01"]),
                        "ref_ts_ny": pd.to_datetime(["2024-05-01 16:00"]),
                        "ref_is_edt": [True], "ref_window": [AMC]})
    matched = validate.match_events(bz, ref)
    assert matched.loc[0, "date_match"] == "within_1d"
    assert matched.loc[0, "vendor_report_date"] == pd.Timestamp("2024-05-02")
    assert matched.loc[0, "date_delta_days"] == 1


# ──────────────────────────────── completeness / identity ────────────────────────────────
def test_maturity_counts_only_prior_usable_events():
    recs = [_record(i, ticker="AAA", date=f"20{y}-05-01") for i, y in enumerate(range(15, 25))]
    recs.append(_record(99, ticker="AAA", date="2016-08-01", time="00:00:00"))  # filler
    bz = normalize.normalize(recs)
    curve = completeness.maturity_curve(bz, ["AAA"], thresholds=(8,), start_year=2015,
                                        end_year=2024)
    at_2016 = curve.set_index("asof_year").loc[2016]
    assert at_2016["n_ge_8"] == 0                    # only two usable events by end of 2016
    assert curve.set_index("asof_year").loc[2024, "n_ge_8"] == 1


def test_class_share_spelling_hazards_are_detected():
    bz = normalize.normalize([_record(1, ticker="BF.B"), _record(2, ticker="BFB"),
                              _record(3, ticker="AAPL")])
    haz = identity.ticker_spelling_hazards(bz, ["BF-B", "AAPL"])
    row = haz[haz["breakwater_ticker"] == "BF-B"].iloc[0]
    assert not row["exact_match_in_vendor"] and row["records_under_exact"] == 0
    assert row["vendor_records"] == 2
    assert "AAPL" not in set(haz["breakwater_ticker"])   # a clean ticker raises nothing


def test_a_reused_ticker_is_caught_by_the_gap_even_when_the_name_never_changes():
    recs = [_record(1, ticker="SNDK", date="2016-04-27", company_name="SanDisk"),
            _record(2, ticker="SNDK", date="2025-05-07", company_name="SanDisk")]
    gaps = identity.history_gaps(normalize.normalize(recs), ["SNDK"])
    assert len(gaps) == 1
    assert gaps.iloc[0]["gap"] > 3000
    assert not gaps.iloc[0]["name_changed"]              # the name detector sees nothing


# ─────────────────────── structural guards: production must stay untouched ───────────────
RESEARCH_DIR = REPO / "research"
PRODUCTION_DIRS = ["pipeline", "feature_engineering", "scoring", "ingestion", "utilities",
                   "analysis", "streamlit_dash", "report", "cron"]


def _python_files(*dirs):
    for d in dirs:
        for f in sorted(Path(d).rglob("*.py")):
            if "__pycache__" not in f.parts:
                yield f


def _code_only(path: Path) -> str:
    """The file's CODE, with comments and docstrings removed.

    The static guards below must fail on `duckdb.connect(...)`, not on a comment saying the
    module deliberately never calls it. Scanning raw source cannot tell those apart; this
    strips every docstring and, via `ast.unparse`, every comment.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


def _string_constants(tree):
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def test_this_tooling_cannot_modify_the_breakwater_database():
    """No database driver is imported and no database path is named — statically, so the
    capability cannot be acquired by a future edit either. `research/` is evaluation
    scaffolding; production data is not its business.

    Checked on the code with docstrings and comments stripped, so prose *about* the
    database (the audit report says a great deal about it) cannot trip or mask the guard.
    """
    forbidden_imports = {"duckdb", "sqlite3", "sqlalchemy", "psycopg2"}
    for f in _python_files(RESEARCH_DIR):
        tree = ast.parse(_code_only(f), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
                if (node.module or "") == "config":
                    assert not any(a.name.endswith("DB_PATH") for a in node.names), \
                        f"{f} imports a database path"
            else:
                continue
            assert not (names & forbidden_imports), f"{f} imports a database driver"
        for literal in _string_constants(tree):
            assert not literal.endswith(".duckdb"), f"{f} names a database file"
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert not node.attr.endswith("DB_PATH"), f"{f} reaches for a database path"


def test_this_tooling_writes_only_under_vendor_and_audit():
    """Every output root in the package resolves under `vendor/` (gitignored raw data),
    except the single committed audit report. No module hardcodes a production directory."""
    for name in ("MASSIVE_ROOT", "EARNINGS_SNAPSHOT_ROOT", "NORMALIZED_ROOT",
                 "REPORT_DATA_ROOT"):
        assert getattr(paths, name).is_relative_to(paths.VENDOR_ROOT)
    assert paths.VENDOR_ROOT.parent == REPO
    assert paths.AUDIT_REPORT_PATH.parent == REPO / "audit"
    for f in _python_files(RESEARCH_DIR):
        tree = ast.parse(_code_only(f), filename=str(f))
        for literal in _string_constants(tree):
            assert not literal.startswith(("db/", "output/", "data/", "/home", "~")), \
                f"{f} hardcodes the path {literal!r}"


def test_no_production_module_depends_on_a_raw_vendor_file():
    """The whole point of keeping `vendor/` outside the repo is that production cannot come
    to depend on it. This is the same guard `test_announcement_timing` puts on
    `audit/provider_timestamps.parquet`."""
    needles = ["vendor/", "research.massive", "research/massive", "massive.com",
               "benzinga", "MASSIVE_API_KEY"]
    for f in _python_files(*(REPO / d for d in PRODUCTION_DIRS if (REPO / d).is_dir())):
        source = _code_only(f).lower()
        for needle in needles:
            assert needle.lower() not in source, f"{f} references {needle!r}"


def test_the_vendor_directory_is_gitignored():
    ignored = (REPO / ".gitignore").read_text()
    assert "vendor/" in ignored


def test_the_window_classifier_is_the_reviewed_production_one():
    """The vendor evaluation must not grow a second, divergent definition of BMO/AMC."""
    source = _code_only(RESEARCH_DIR / "massive" / "normalize.py")
    assert "from feature_engineering.announcement_timing import classify_announce_window" in source
    assert "9.5" not in source and "16.0" not in source   # no re-implemented cut points


def test_nothing_in_the_package_infers_timing_from_price():
    """Rev-1 of the Phase 0 audit labelled BMO/AMC from realized returns and every
    'corrected' number it produced was circular. Enforced statically here too."""
    price_names = {"price", "close", "reaction", "return", "daily_ret", "abs_reaction_3d",
                   "risk_score", "vol_10d", "vol_30d"}
    for f in _python_files(RESEARCH_DIR):
        tree = ast.parse(_code_only(f), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in price_names, f"{f} names the price column {node.value!r}"
            if isinstance(node, ast.Name):
                assert node.id not in price_names, f"{f} binds {node.id!r}"
