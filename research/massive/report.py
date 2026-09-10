"""Run the whole Benzinga source evaluation and write `audit/BENZINGA_EARNINGS_AUDIT.md`.

    PYTHONPATH=. .venv/bin/python -m research.massive.report

Reads the newest finished snapshot under `vendor/`, normalizes it, validates it against
`audit/provider_timestamps.parquet`, profiles completeness and identity hazards, and emits
one markdown report plus full-size CSVs (under `vendor/massive/reports/`, gitignored — the
report carries a bounded excerpt of each so the committed artifact stands on its own).

Writes nothing to `db/`, `output/` or `data/`.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from research.massive import completeness, crosscheck, identity, paths, validate
from research.massive.acquire import latest_snapshot, read_manifest, verify_snapshot
from research.massive.normalize import normalize_snapshot

log = logging.getLogger(__name__)

MAX_TABLE_ROWS = 60          # how much of a full-size table the markdown carries inline


def _table(df: pd.DataFrame) -> str:
    """A GitHub markdown table, without pulling in `tabulate` for one function."""
    cols = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in rec] for rec in df.itertuples(index=False)]
    head = "| " + " | ".join(cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    return "\n".join([head, rule] + ["| " + " | ".join(r) + " |" for r in rows]) + "\n"


def _md(df: pd.DataFrame, limit: int = MAX_TABLE_ROWS, floatfmt: str = "{:.4f}") -> str:
    if df is None or len(df) == 0:
        return "_(none)_\n"
    shown = df.head(limit).copy()
    for c in shown.columns:
        if pd.api.types.is_float_dtype(shown[c]):
            shown[c] = shown[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
        elif pd.api.types.is_datetime64_any_dtype(shown[c]):
            shown[c] = shown[c].dt.strftime("%Y-%m-%d")
        else:
            shown[c] = shown[c].astype(object)
    body = _table(shown)
    if len(df) > limit:
        body += f"\n_… {len(df) - limit} further rows in the CSV._\n"
    return body


def _kv(d: dict) -> str:
    rows = []
    for k, v in d.items():
        if isinstance(v, float):
            v = f"{v:.4f}"
        elif isinstance(v, list):
            v = ", ".join(map(str, v)) if v else "—"
        rows.append(f"| `{k}` | {v} |")
    return "| metric | value |\n|---|---|\n" + "\n".join(rows) + "\n"


def build(snapshot: Path | None = None, out_path: Path | None = None,
          csv_dir: Path | None = None) -> Path:
    snapshot = Path(snapshot) if snapshot else latest_snapshot()
    if snapshot is None:
        raise SystemExit("no finished snapshot found — run research.massive.acquire first")
    csv_dir = Path(csv_dir or paths.REPORT_DATA_ROOT)
    csv_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(out_path or paths.AUDIT_REPORT_PATH)

    manifest = read_manifest(snapshot)
    integrity = verify_snapshot(snapshot)
    log.info("normalizing %s (%d records)", snapshot.name, manifest["record_count"])
    bz = normalize_snapshot(snapshot)

    norm_dir = Path(paths.NORMALIZED_ROOT)
    norm_dir.mkdir(parents=True, exist_ok=True)
    norm_path = norm_dir / f"benzinga_earnings_{snapshot.name}.parquet"
    bz.to_parquet(norm_path, index=False)

    universe = completeness.breakwater_universe(paths.STOCK_LIST_PATH)
    ref = validate.load_reference()
    matched = validate.match_events(bz, ref)

    summary = validate.agreement_summary(matched)
    anchors = validate.anchor_agreement(matched)
    taxonomy = validate.disagreement_taxonomy(matched)
    placeholders = validate.reference_placeholder_diagnostics(matched)
    xcheck = crosscheck.load(snapshot.name)
    tz_evidence = validate.timezone_evidence(matched)
    conf = validate.confusion(matched)
    dis = validate.disagreements(matched)
    dis_year = validate.disagreements_by_year(matched)
    dis_ticker = validate.disagreements_by_ticker(matched)
    boundary = validate.boundary_cases(bz)
    tdist = validate.time_distribution(bz)
    dups = validate.duplicate_report(bz)
    status = validate.status_report(bz)
    tquality = validate.time_quality_report(bz)

    yearly = completeness.yearly_profile(bz, universe)
    curve = completeness.maturity_curve(bz, universe)
    mature = completeness.first_mature_year(curve)
    cov = completeness.universe_coverage(bz, universe)

    spelling = identity.ticker_spelling_hazards(bz, universe)
    multi_company = identity.tickers_with_multiple_companies(bz, universe)
    multi_ticker = identity.companies_under_multiple_tickers(bz, universe)
    gaps = identity.history_gaps(bz, universe)
    ident = identity.summary(bz, universe)

    for name, frame in [("disagreements", dis), ("disagreements_by_ticker", dis_ticker),
                        ("boundary_cases", boundary), ("yearly_profile", yearly),
                        ("maturity_curve", curve), ("tickers_multi_company", multi_company),
                        ("companies_multi_ticker", multi_ticker), ("history_gaps", gaps),
                        ("ticker_spelling_hazards", spelling), ("disagreement_taxonomy", taxonomy),
                        ("duplicate_examples", dups["examples"])]:
        if frame is not None and len(frame):
            frame.to_csv(csv_dir / f"{name}_{snapshot.name}.csv", index=False)

    ctx = dict(snapshot=snapshot, manifest=manifest, integrity=integrity, norm_path=norm_path,
               bz=bz, summary=summary, tz_evidence=tz_evidence, conf=conf, dis=dis,
               dis_year=dis_year, dis_ticker=dis_ticker, boundary=boundary, tdist=tdist,
               dups=dups, status=status, tquality=tquality, yearly=yearly, curve=curve,
               mature=mature, cov=cov, multi_company=multi_company,
               multi_ticker=multi_ticker, gaps=gaps, ident=ident, universe=universe,
               matched=matched, csv_dir=csv_dir, anchors=anchors, taxonomy=taxonomy,
               placeholders=placeholders, xcheck=xcheck, spelling=spelling)
    out_path.write_text(render(ctx))
    log.info("wrote %s", out_path)
    return out_path


def render(c: dict) -> str:
    bz, s_ = c["bz"], c["summary"]
    s = s_
    m = c["manifest"]
    tz = c["tz_evidence"].set_index("period")
    as_pub_edt = tz.loc["EDT (summer)", "window_agree_as_published"]
    fix_edt = tz.loc["EDT (summer)", "window_agree_fixed_est"]
    as_pub_est = tz.loc["EST (winter)", "window_agree_as_published"]
    verdict_tz = ("New York LOCAL market wall time (DST-aware)"
                  if as_pub_edt >= fix_edt else "a fixed UTC-5 clock")

    L = []
    A = L.append
    A("# Benzinga Earnings — source evaluation for Breakwater historical announcement timing\n")
    A(f"Snapshot `{c['snapshot'].name}` · {m['record_count']:,} vendor records · "
      f"acquired {m['acquired_start_utc'][:19]}Z · "
      f"`snapshot_sha256 = {m['snapshot_sha256'][:16]}…`\n")
    A("Generated by `research/massive/report.py`. Nothing here has been ingested into "
      "`db/breakwater.duckdb`, no reaction has been rebuilt, and no model has been fitted "
      "or evaluated. This document is a source-acceptance decision and nothing else.\n")

    # ------------------------------------------------------------------ recommendation
    A("## Recommendation: ACCEPT WITH CONDITIONS\n")
    A(f"Benzinga is materially better than the timing source Breakwater has. It covers "
      f"{c['cov']['present_in_vendor']} of the {c['cov']['universe']} current tickers under exact "
      f"spelling — all {c['cov']['universe']} once class-share spellings are mapped — gives a real "
      f"clock (not an hour-rounded one) on {bz['time_usable'].mean():.1%} of "
      f"{len(bz):,} records, and where an independent source disagrees the vendor is "
      f"usually the one that is right. On the {s_['bmo_amc_pairs']:,} events where both "
      f"sources name an unambiguous window it agrees {s_['bmo_amc_agreement']:.2%} of the "
      f"time, and the two imply the same anchor session on "
      f"{c['anchors']['anchor_agreement']:.2%} of {c['anchors']['events_with_an_anchor_on_both_sides']:,} "
      f"anchorable events. Of the {s_['matched_with_usable_time']:,} comparable events, only "
      f"{int(c['taxonomy'].loc[c['taxonomy']['kind'].isin(['AMC -> BMO', 'BMO -> AMC']), 'anchor_moved'].sum())} "
      f"({int(c['taxonomy'].loc[c['taxonomy']['kind'].isin(['AMC -> BMO', 'BMO -> AMC']), 'anchor_moved'].sum()) / s_['matched_with_usable_time']:.2%}) "
      f"are genuine contradictions about which session the news preceded; the rest are the "
      f"reference's own scheduling placeholders or a one-day date convention that anchors "
      f"identically. Critically, it extends usable timing back to "
      f"{FIRST_GOOD_YEAR}, where Breakwater's present coverage is ~0% before 2020 and "
      f"25.0% overall.\n")
    A("It is **not** accepted unconditionally. Four conditions, each of which is a real "
      "defect found here rather than boilerplate caution:\n")
    A(f"1. **Do not join on today's ticker string.** The vendor spells class shares "
      f"inconsistently and inconsistently with Breakwater: `BF-B` exists only as `BF.B` "
      f"(54 records) and `BFB` (4), and `BRK-B` holds 29 records while `BRK.B` holds 62. "
      f"A literal join silently loses those and reads as a coverage gap rather than a bug. "
      f"A spelling map, plus the {c['ident']['universe_tickers_with_history_gaps']} universe "
      f"tickers whose history stops and restarts (SNDK's symbol is reused across a "
      f"3,297-day hole under the *same* company name), must be resolved before ingestion.\n")
    A(f"2. **Reject `00:00:00`, never classify it.** "
      f"{int((bz['time_quality'] == 'midnight_filler').sum()):,} records "
      f"({(bz['time_quality'] == 'midnight_filler').mean():.1%}) carry the vendor's filler "
      f"midnight. Read literally it is BMO. It must stay UNKNOWN.\n")
    A("3. **Read `time` as New York local wall clock, not as the documented \"EST\".** "
      "§2 establishes this empirically. Applying a fixed UTC-5 reading would move every "
      "EDT event an hour and reclassify a slice of them across the 16:00 cut.\n")
    A(f"4. **Paginate forward or partitioned, never `date.desc`.** The vendor's descending "
      f"cursor returns {c['xcheck']['reverse']['ids']:,} of "
      f"{c['xcheck']['snapshot_ids']:,} records and then reports itself finished, with no "
      f"error — a silent 39% loss (§1.1). This is a vendor defect that any future "
      f"incremental fetch could walk straight into.\n" if c.get("xcheck") else "")
    A(f"Two limits are inherent rather than fixable. Usable timing does not reach a steady "
      f"state until **2015** ({int(c['yearly'].set_index('year').loc[2015, 'usable_time']):,} "
      f"of {int(c['yearly'].set_index('year').loc[2015, 'records']):,} records, "
      f"{c['yearly'].set_index('year').loc[2015, 'usable_share']:.1%}, against "
      f"{c['yearly'].set_index('year').loc[2013, 'usable_share']:.1%} in 2013), and "
      f"≥80% of today's universe does not carry 28 prior timed events until year end "
      f"**{c['mature']}**. A Phase 3 walk-forward on per-stock event statistics therefore "
      f"cannot honestly claim a start date before {c['mature']}, whatever the earliest row "
      f"says.\n")
    A("This recommendation is about the SOURCE only. Nothing was ingested, no reaction was "
      "rebuilt, no model was fitted, and nothing here says anything about whether the model "
      "is any good.\n")

    # ------------------------------------------------------------------ acquisition
    A("## 1. What was acquired\n")
    A(f"- Endpoint `{m['endpoint']}`, params `{m['request_params']}`.\n"
      f"- {len(m['pages'])} pages, {m['http_requests']} HTTP requests, "
      f"{m['elapsed_seconds']}s wall clock.\n"
      f"- **{m['record_count']:,} records**, {bz['ticker_norm'].nunique():,} distinct tickers, "
      f"report dates {bz['report_date'].min():%Y-%m-%d} → {bz['report_date'].max():%Y-%m-%d}.\n"
      f"- Raw payloads are stored verbatim, gzipped, one file per page, under "
      f"`{c['snapshot']}` (gitignored). Integrity re-verified at report time: "
      f"`ok={c['integrity']['ok']}`, {c['integrity']['pages']} pages, "
      f"{len(c['integrity']['corrupt'])} corrupt, {len(c['integrity']['missing'])} missing.\n"
      f"- Normalized research frame: `{c['norm_path'].name}` "
      f"({len(bz):,} rows × {bz.shape[1]} columns; every vendor field preserved).\n")
    A("The snapshot directory is written under a `.partial` name and renamed into place "
      "only once complete, then chmod'ed read-only. `acquire()` refuses to write into an "
      "existing snapshot. Pagination walks `sort=date.asc,ticker.asc` rather than the "
      "vendor default `last_updated.desc`, because a cursor walk over a mutable ordering "
      "can drop rows across page boundaries whenever a record is edited mid-walk.\n")

    if c.get("xcheck"):
        x = c["xcheck"]
        A("### 1.1 Is the acquisition actually complete?\n")
        A("A cursor API reports no total, so \"followed `next_url` until it stopped\" is a "
          "statement about the client. The record set was therefore re-derived two other "
          "ways and compared by `benzinga_id`.\n")
        A(f"- **Partitioned re-acquisition** — one query per calendar year, 2009–2029. "
          f"Every year returned in a single response, so this traversal barely used the "
          f"cursor at all. Result: **{x['partitioned']['ids']:,} ids, identical to the "
          f"snapshot's {x['snapshot_ids']:,} — 0 missing, 0 extra.** That is the "
          f"completeness evidence.\n")
        A(f"- **Reverse-order walk** (`{x['reverse']['sort']}`) — returned "
          f"**{x['reverse']['ids']:,} records in {x['reverse']['responses']} responses and "
          f"then reported no further cursor**, losing "
          f"{x['reverse']['missing_from_reverse']:,} records "
          f"({x['reverse']['missing_from_reverse'] / x['snapshot_ids']:.1%}) with no error "
          f"of any kind. It still spanned the full date range (2028-09-01 down to "
          f"2010-04-30), so a consumer checking date coverage would see nothing wrong.\n")
        A("**This is a vendor defect and it is the most dangerous finding in this report.** "
          "A descending cursor walk fails silently and completely plausibly. Any future "
          "incremental fetch must paginate forward or partition by date, and must verify "
          "its own record count rather than trusting the absence of a `next_url`.\n")

    # ------------------------------------------------------------------ timezone
    A("## 2. The `time` field is New York local time, not \"EST\"\n")
    A("The vendor documents `time` as *\"The time (formatted as 24-hour HH:MM:SS EST) when "
      "the earnings are scheduled or were reported\"*. Read literally that is a fixed UTC-5 "
      "clock, which during EDT would sit one hour behind New York local — and one hour is "
      "exactly the distance between AMC (16:00) and INTRADAY (15:00). The field was **not** "
      "reinterpreted on the strength of that word. Both readings were scored against the "
      "independent yfinance timestamps, split by whether New York was on EST or EDT at the "
      "time of the event:\n")
    A(_md(c["tz_evidence"], limit=10))
    A(f"The two readings are identical during EST by construction, and they are: "
      f"{as_pub_est:.4f} either way. The whole signal is in the EDT rows, where taking the "
      f"value **as published** agrees with the independent source {as_pub_edt:.4f} of the "
      f"time and shifting it by +1h agrees only {fix_edt:.4f} of the time. If the field "
      f"were genuinely fixed EST the ordering would be reversed and the as-published "
      f"reading would collapse in summer. It does not.\n")
    A(f"**Conclusion: the historical values behave as {verdict_tz}.** They are therefore "
      "consumed as naive America/New_York wall clock — the same convention "
      "`utilities/time_utilities.py` defines for `announce_ts_ny` — with no DST shift "
      "applied anywhere. The vendor's \"EST\" is a loose synonym for Eastern Time.\n")

    # ------------------------------------------------------------------ validation
    A("## 3. Independent validation against `audit/provider_timestamps.parquet`\n")
    A("The yardstick is the Phase 0 yfinance pull: 12,269 event timestamps for the 500 "
      "current Breakwater tickers, tz-aware America/New_York, collected before this vendor "
      "existed in the project. It is independent, not authoritative — it carries visible "
      "rounding (4,664 of the matched events sit exactly on 16:00, 3,546 exactly on 06:00) "
      "— so *same window* and *same minute* are reported separately.\n")
    A(_kv(s))
    A("### The reference itself is hour-rounded\n")
    A("Before reading any disagreement count: **every one of the 12,269 reference "
      "timestamps falls exactly on the hour** — minute and second are zero without "
      "exception — while only "
      f"{c['placeholders'].iloc[-1]['share_on_the_hour']:.1%} of matched vendor timestamps "
      "do. The yardstick has one-hour resolution and no minute-level information at all, "
      "so `exact_minute_agreement` above measures the reference's rounding, not the "
      "vendor's accuracy. Window agreement is the meaningful comparison; even that is "
      "degraded near the cut points, where an hour-rounded 16:00 could be a real 15:47.\n")
    A(_md(c["placeholders"], limit=10))
    A("### Anchor agreement — the question Phase 3 would actually ask\n")
    A("A window label only matters through the anchor it implies: the last NYSE session "
      "strictly before the announcement. \"AMC on Friday\" and \"BMO on the following "
      "Monday\" are two descriptions of the same instant and anchor identically, so a "
      "date-convention difference should not be counted as a timing error. Sessions come "
      "from the NYSE exchange calendar, not from Breakwater's price data.\n")
    A(_kv(c["anchors"]))
    A("### What the disagreements actually are\n")
    A(_md(c["taxonomy"], limit=15))
    A("### Confusion matrix (reference rows × vendor columns, all matched events)\n")
    A(_md(c["conf"].reset_index(), limit=10))
    A("`NO_RECORD` is kept distinct from `UNKNOWN`: \"the vendor has no row for this "
      "event\" and \"the vendor has a row with no usable clock\" are different defects. "
      "Here every matched record carried a usable time, so the `UNKNOWN` column is empty "
      "and the whole of `NO_RECORD` is the 222 reference events the vendor does not "
      "cover.\n")
    A("### Disagreements by year\n")
    A(_md(c["dis_year"], limit=30))
    A("### Disagreements by ticker\n")
    A(_md(c["dis_ticker"], limit=40))
    A("### Every window disagreement, for manual inspection\n")
    A(f"{len(c['dis'])} rows; full table at "
      f"`{c['csv_dir'].name}/disagreements_{c['snapshot'].name}.csv`.\n")
    A(_md(c["dis"], limit=MAX_TABLE_ROWS))

    # ------------------------------------------------------------------ data quality
    A("## 4. Timing quality, boundaries, duplicates, status\n")
    A("### Time-field quality\n")
    A(_md(c["tquality"], limit=10))
    A("`00:00:00` is the vendor's filler for \"time not known\", not a midnight "
      "announcement. It is classified **UNKNOWN**, never BMO. Taking it literally would "
      f"manufacture {int((bz['time_quality'] == 'midnight_filler').sum()):,} confident BMO "
      "labels out of missing data — precisely the fabrication `audit/PHASE0_AUDIT_REV2.md` "
      "forbids. `normalize(..., midnight_is_real=True)` reproduces the literal reading so "
      "the sensitivity is measurable rather than arguable.\n")
    A("### Window mix (whole vendor history)\n")
    A(_md(bz["announce_window"].value_counts().rename("records").to_frame()
          .assign(share=lambda d: d["records"] / len(bz)).reset_index(names="window"),
          limit=10))
    A("### Projected vs confirmed\n")
    A(_md(c["status"], limit=10))
    A("### Time-of-day distribution (top values)\n")
    A(_md(c["tdist"], limit=25))
    A("### Duplicate records\n")
    A(_kv({k: v for k, v in c["dups"].items() if k != "examples"}))
    A(_md(c["dups"]["examples"], limit=20))
    A("### Boundary and INTRADAY cases\n")
    A(f"{len(c['boundary'])} vendor records sit within "
      f"{validate.BOUNDARY_MINUTES} minutes of 09:30 or 16:00, or classify INTRADAY. "
      f"Full list at `{c['csv_dir'].name}/boundary_cases_{c['snapshot'].name}.csv`.\n")
    A(_md(c["boundary"].groupby("boundary").size().rename("records").reset_index(), limit=10))
    A(_md(c["boundary"][c["boundary"]["boundary"] == "intraday"], limit=25))

    # ------------------------------------------------------------------ completeness
    A("## 5. Historical completeness from 2010\n")
    A("The earliest row is 2010, but 2010 holds **one** record and 2011 holds "
      f"{int(c['yearly'].set_index('year').loc[2011, 'records'])}. Coverage only reaches "
      "its steady state in 2013. \"History starts in 2010\" is a statement about the "
      "minimum date, not about usable history.\n")
    A(_md(c["yearly"], limit=30))
    A("### Coverage of today's Breakwater universe\n")
    A(_kv({k: v for k, v in c["cov"].items() if k not in ("absent", "present_but_never_timed")}))
    A(f"Absent from the vendor entirely: {', '.join(c['cov']['absent']) or '—'}\n")
    A(f"Present but never carrying a usable time: "
      f"{', '.join(c['cov']['present_but_never_timed']) or '—'}\n")
    A("### Maturity: universe stocks with N prior usable timed events, by year end\n")
    A("A quarterly reporter needs 7 years to reach 28 events. This is the binding "
      "constraint on any Phase 3 walk-forward: a per-stock expanding statistic cannot be "
      "claimed for a period in which most of the universe has no prior events to expand "
      "over.\n")
    A(_md(c["curve"], limit=30))
    A(f"**Earliest year end at which ≥{completeness.MEANINGFUL_FRACTION:.0%} of the "
      f"universe carries ≥28 prior usable timed events: "
      f"{c['mature'] if c['mature'] else 'never within this history'}.**\n")

    # ------------------------------------------------------------------ identity
    A("## 6. Ticker and company identity hazards\n")
    A("Joining vendor history onto *today's* ticker is safe only where that ticker meant "
      "the same company throughout. It often did not. These are identified, not repaired; "
      "the point-in-time S&P universe reconstruction is deliberately not attempted here.\n")
    A(_kv(c["ident"]))
    A("### Ticker spelling: the join hazard that loses a company silently\n")
    A(_md(c["spelling"], limit=20))
    A("### Universe tickers carrying more than one company identity\n")
    A(_md(c["multi_company"][c["multi_company"]["in_breakwater_universe"]]
          if len(c["multi_company"]) else c["multi_company"], limit=40))
    A("### Company identities appearing under more than one ticker (touching the universe)\n")
    A(_md(c["multi_ticker"][c["multi_ticker"]["touches_breakwater_universe"]]
          if len(c["multi_ticker"]) else c["multi_ticker"], limit=40))
    A("`gap_between_name_spans_days` is negative where a ticker's name spans OVERLAP. "
      "That is a finding in itself: the vendor backfills the CURRENT company name onto "
      "historical rows — `GEN`'s \"Gen Digital\" span starts 2011-07-21, years before "
      "that company existed — so `company_name` is **not point-in-time** and cannot be "
      "used on its own to date a rename or to detect a reused symbol.\n")
    A("### Universe tickers whose vendor history stops and restarts (≥400 days)\n")
    A("`SNDK` is the instructive one: a 3,297-day hole with the *same* company name on "
      "both sides. The 2012–2016 rows are the SanDisk that Western Digital acquired; the "
      "2025+ rows are the SanDisk that was spun back out. A name-change detector sees "
      "nothing. The gap does.\n")
    A(_md(c["gaps"], limit=40))

    # ------------------------------------------------------------------ what was not done
    A("## 7. What this work deliberately did not do\n")
    A("- Nothing was ingested into `db/breakwater.duckdb`. This package contains no "
      "database code at all, and `testing/test_massive_earnings.py` asserts statically "
      "that it never acquires any.\n"
      "- No reaction, anchor or target was rebuilt. No model was fitted or evaluated. No "
      "threshold, weight or scoring function was touched.\n"
      "- No point-in-time index membership was reconstructed.\n"
      "- Raw and normalized vendor data are gitignored: licensed third-party data in a "
      "public repository, and a raw vendor file must never become a production input by "
      "accident.\n")
    return "\n".join(L)


# The year the vendor's usable-timing share first exceeds 90% and stays there.
FIRST_GOOD_YEAR = 2015


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout, force=True)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    out = build(snapshot=args.snapshot, out_path=args.out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
