"""Breakwater stock -> SEC CIK, point in time, with provenance for every link.

A Breakwater "stock" is a price series under today's ticker. SEC files by CIK, which is
stable for an *issuer*, not for a ticker or a business: when a company reorganises under
a new holding company (Google -> Alphabet, Praxair -> Linde plc) the price series carries
on under the same ticker while filings move to a NEW CIK. So a stock maps to a *chain* of
CIK segments, each valid for a date range:

    stock -> [(cik, valid_from, valid_to, provenance), ...]

Provenance values
-----------------
* `sec_ticker_map`        today's SEC `company_tickers.json` lists the ticker (or its
                          dot/dash spelling) under exactly one CIK, and the SEC entity name
                          agrees with Breakwater's company name.
* `breakwater_rename`     as above, reached through `data/ticker_renames.csv` (BK -> BNY).
* `manual_verified`       not derivable from today's map (delisted, or a predecessor
                          issuer). Every manual link is listed in `MANUAL` with its reason,
                          and is accepted only if SEC's own submissions file for that CIK
                          carries the expected name. Verification is re-run on every build.
* `unresolved`            anything else. Never guessed.

What is deliberately NOT done: matching on company name alone, fuzzy-joining on today's
ticker string for a historical date, or inferring identity from prices or outcomes.
Whether each event's report date is actually backed by an earnings filing from the
mapped CIK is checked separately (`build.issuer_confirmation`) — it validates the link, it
does not create one.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

from research.sec_filings_pilot import paths

SEC_TICKER_MAP = "sec_ticker_map"
BREAKWATER_RENAME = "breakwater_rename"
MANUAL_VERIFIED = "manual_verified"
UNRESOLVED = "unresolved"

OPEN_START = pd.Timestamp("1990-01-01")
OPEN_END = pd.Timestamp("2099-12-31")


@dataclass(frozen=True)
class ManualLink:
    cik: int
    valid_from: str | None      # None = open
    valid_to: str | None        # None = open (exclusive end: filings known BEFORE this date)
    expected_name: str          # must occur in SEC's name or formerNames for this CIK
    reason: str


# Every manual link, with the reason it exists. Found from the stocks the ticker map could
# not cover and from events falling before their CIK's first 10-Q/10-K (`build.event_identity`);
# each is checked against SEC's submissions file (name history) on every build.
_BRAND = "Breakwater carries a brand name; SEC's ticker map gives this one CIK and the legal name is"
_GONE = "ticker no longer in SEC's map (left the index / delisted in 2026); CIK of"
MANUAL: dict[str, list[ManualLink]] = {
    # Brand name vs legal name — today's SEC map already links the ticker to this CIK.
    "BK": [ManualLink(1390777, None, None, "Bank of New York Mellon", f"{_BRAND} Bank of New York Mellon Corp (BK -> BNY 2026)")],
    "DECK": [ManualLink(910521, None, None, "DECKERS OUTDOOR", f"{_BRAND} Deckers Outdoor Corp")],
    "GE": [ManualLink(40545, None, None, "GENERAL ELECTRIC", f"{_BRAND} General Electric Co")],
    "IBM": [ManualLink(51143, None, None, "INTERNATIONAL BUSINESS MACHINES", f"{_BRAND} International Business Machines Corp")],
    "SMCI": [ManualLink(1375365, None, None, "Super Micro Computer", f"{_BRAND} Super Micro Computer, Inc.")],
    # No longer in today's map.
    "AVB": [ManualLink(915912, None, None, "AVALONBAY", f"{_GONE} AvalonBay Communities")],
    "CTRA": [ManualLink(858470, None, None, "CABOT OIL", f"{_GONE} Coterra Energy (formerly Cabot Oil & Gas)")],
    "DAY": [ManualLink(1725057, None, None, "Ceridian", f"{_GONE} Dayforce (formerly Ceridian HCM)")],
    "EA": [ManualLink(712515, None, None, "ELECTRONIC ARTS", f"{_GONE} Electronic Arts")],
    "EQR": [ManualLink(906107, None, None, "EQUITY RESIDENTIAL", f"{_GONE} Equity Residential")],
    "HOLX": [ManualLink(859737, None, None, "HOLOGIC", f"{_GONE} Hologic")],
}

# Holding-company reorganisations and mergers-of-record: the price series continues under
# the same ticker while SEC filings move to a new CIK. The boundary is the filing date of
# the successor's completion filing (8-K12B / 8-K12G3 / completion 8-K) as it appears in
# SEC's own submissions file; filings dated before it belong to the predecessor, filings
# dated on or after it to the successor. The predecessor's later filings (it often keeps
# filing as a subsidiary — E.I. du Pont did until 2023) are thereby excluded.
_SUCC = "successor issuer; completion filing"
_PRED = "predecessor issuer of the same price series until"
_CHAINS: dict[str, list[tuple[int, str | None, str | None, str, str]]] = {
    # stock: [(cik, valid_from, valid_to, expected SEC name fragment, reason), ...]
    "APA": [(6769, None, "2021-03-01", "APACHE", f"{_PRED} APA Corp holdco 2021-03-01"),
            (1841666, "2021-03-01", None, "APA", f"{_SUCC} 8-K12B 2021-03-01")],
    "APO": [(1411494, None, "2022-01-03", "Apollo", f"{_PRED} Apollo/Athene holdco 2022-01-03"),
            (1858681, "2022-01-03", None, "Apollo", f"{_SUCC} 8-K12B 2022-01-03")],
    "AVGO": [(1441634, None, "2016-02-02", "Avago", f"{_PRED} Broadcom Ltd 8-K12B 2016-02-02"),
             (1649338, "2016-02-02", "2018-04-04", "Broadcom", f"{_PRED} Broadcom Inc redomicile 2018-04-04"),
             (1730168, "2018-04-04", None, "Broadcom", f"{_SUCC} 8-K12B 2018-04-04")],
    "BG": [(1144519, None, "2023-11-01", "BUNGE", f"{_PRED} Bunge Global SA redomicile 2023-11-01"),
           (1996862, "2023-11-01", None, "Bunge", f"{_SUCC} 8-K12G3 2023-11-01")],
    "BLK": [(1364742, None, "2024-10-01", "BlackRock", f"{_PRED} BlackRock holdco (GIP) 2024-10-01"),
            (2012383, "2024-10-01", None, "BlackRock", f"{_SUCC} 8-K12B 2024-10-01")],
    "CI": [(701221, None, "2018-12-20", "CIGNA", f"{_PRED} Cigna/Express Scripts holdco 2018-12-20"),
           (1739940, "2018-12-20", None, "Cigna", f"{_SUCC} 8-K12B 2018-12-20")],
    "DD": [(30554, None, "2017-09-01", "DUPONT E I", f"{_PRED} DowDuPont merger 2017-09-01"),
           (1666700, "2017-09-01", None, "DuPont", f"{_SUCC} 8-K12B 2017-09-01")],
    "DIS": [(1001039, None, "2019-03-20", "DISNEY", f"{_PRED} Disney/21CF holdco 2019-03-20"),
            (1744489, "2019-03-20", None, "Disney", f"{_SUCC} 8-K12B 2019-03-20")],
    "GOOG": [(1288776, None, "2015-10-02", "Google", f"{_PRED} Alphabet holdco 2015-10-02"),
             (1652044, "2015-10-02", None, "Alphabet", f"{_SUCC} 8-K12B 2015-10-02")],
    "GOOGL": [(1288776, None, "2015-10-02", "Google", f"{_PRED} Alphabet holdco 2015-10-02"),
              (1652044, "2015-10-02", None, "Alphabet", f"{_SUCC} 8-K12B 2015-10-02")],
    "MDT": [(64670, None, "2015-01-27", "MEDTRONIC", f"{_PRED} Medtronic plc (Covidien) 2015-01-27"),
            (1613103, "2015-01-27", None, "Medtronic", f"{_SUCC} 8-K12B 2015-01-27")],
    "STE": [(815065, None, "2015-11-06", "STERIS", f"{_PRED} STERIS plc (UK, Synergy) completion 8-K 2015-11-06"),
            (1624899, "2015-11-06", "2019-03-28", "STERIS", f"{_PRED} STERIS plc (Ireland) 2019-03-28"),
            (1757898, "2019-03-28", None, "STERIS", f"{_SUCC} completion 8-K 2019-03-28")],
    "TPL": [(97517, None, "2021-01-11", "TEXAS PACIFIC LAND", f"{_PRED} trust -> corporation 2021-01-11"),
            (1811074, "2021-01-11", None, "Texas Pacific Land", f"{_SUCC} completion 8-K 2021-01-11")],
    "XOM": [(34088, None, "2026-07-01", "EXXON MOBIL", f"{_PRED} ExxonMobil Holdings 2026-07-01"),
            (2115436, "2026-07-01", None, "ExxonMobil", f"{_SUCC} 8-K12B 2026-07-01")],
}
for _stock, _links in _CHAINS.items():
    MANUAL[_stock] = [ManualLink(*link) for link in _links]


# ──────────────────────────────────── Breakwater side ───────────────────────────────────
def universe() -> pd.DataFrame:
    info = pd.read_csv(paths.STOCK_INFO, usecols=["ticker", "name", "sector", "sub_sector"])
    info = info.rename(columns={"ticker": "stock", "name": "bw_name"})
    d = pd.read_parquet(paths.FULL_DF, columns=["stock", "date"])
    span = d.groupby("stock")["date"].agg(first_price="min", last_price="max").reset_index()
    u = info.merge(span, on="stock", how="left")
    ren = pd.read_csv(paths.TICKER_RENAMES)
    u["renamed_to"] = u["stock"].map(dict(zip(ren["old_ticker"], ren["new_ticker"])))
    haz = pd.read_csv(paths.IDENTITY_HAZARDS) if paths.IDENTITY_HAZARDS.exists() else \
        pd.DataFrame(columns=["stock", "reason"])
    u["benzinga_identity_hazard"] = u["stock"].map(dict(zip(haz["stock"], haz["reason"])))
    return u


# ───────────────────────────────────────── SEC side ─────────────────────────────────────
def ticker_variants(stock: str) -> list[str]:
    """Spellings of one share class: BRK-B, BRK.B, BRKB. Never a different class."""
    s = stock.upper()
    out = [s, s.replace("-", "."), s.replace(".", "-"), s.replace("-", "").replace(".", "")]
    return list(dict.fromkeys(out))


def tickers_frame(tickers_json: dict) -> pd.DataFrame:
    """SEC `company_tickers.json` as a frame. `sec_rank` is SEC's own ordering, which runs
    roughly by market value — used only as a size proxy for sampling, never for identity."""
    df = pd.DataFrame(list(tickers_json.values()), columns=["cik_str", "ticker", "title"])
    df = df.rename(columns={"cik_str": "cik", "title": "sec_title"})
    df["sec_rank"] = [int(k) for k in tickers_json.keys()]
    return df


def sec_ticker_matches(stock: str, renamed_to, tickers: pd.DataFrame) -> list[dict]:
    hits = []
    for via, t in [(SEC_TICKER_MAP, stock)] + ([(BREAKWATER_RENAME, renamed_to)]
                                               if isinstance(renamed_to, str) else []):
        for v in ticker_variants(t):
            for r in tickers[tickers["ticker"].eq(v)].itertuples():
                hits.append({"cik": int(r.cik), "sec_ticker": r.ticker, "via": via,
                             "sec_title": r.sec_title, "sec_rank": r.sec_rank})
    return hits


def candidate_ciks(tickers_json: dict) -> list[int]:
    t = tickers_frame(tickers_json)
    u = universe()
    ciks = {h["cik"] for r in u.itertuples() for h in sec_ticker_matches(r.stock, r.renamed_to, t)}
    ciks |= {m.cik for links in MANUAL.values() for m in links}
    return sorted(ciks)


_SUFFIX = {"inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
           "plc", "the", "holdings", "holding", "group", "sa", "nv", "llc", "lp", "class",
           "de", "new", "a", "b", "cos", "companies", "trust", "international", "intl",
           "and", "of"}


def name_tokens(name: str) -> set[str]:
    s = re.sub(r"[\u2010-\u2015-]", " ", str(name))           # Brown–Forman -> Brown Forman
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"\(.*?\)", " ", s.lower())
    s = re.sub(r"\b([a-z])\.([a-z])\.", r"\1\2", s)       # U.S. -> us
    s = s.replace("&", " and ").replace("'", "")
    toks = set(re.findall(r"[a-z0-9]+", s))
    toks = {t for t in toks if t not in _SUFFIX}
    # Initials ("J.B.", "C H") are spelled inconsistently; keep only longer words when any.
    return {t for t in toks if len(t) > 2} or toks


def names_agree(a: str, b: str) -> bool:
    """Conservative: one name's significant tokens contained in the other's, or a Jaccard of
    at least 0.5. Used only to CONFIRM a ticker link SEC already made — never to make one."""
    ta, tb = name_tokens(a), name_tokens(b)
    if not ta or not tb:
        return False
    if ta <= tb or tb <= ta:
        return True
    return len(ta & tb) / len(ta | tb) >= 0.5


def sec_names(sub: dict) -> list[str]:
    return [sub.get("name", "")] + [f.get("name", "") for f in sub.get("formerNames", []) or []]


# ───────────────────────────────────────── resolve ──────────────────────────────────────
def resolve(u: pd.DataFrame, tickers: pd.DataFrame, submissions: dict[int, dict]) -> tuple[
        pd.DataFrame, pd.DataFrame]:
    """(stock-level mapping audit, CIK segment table).

    Stock statuses: `confident`, `ambiguous` (today's map gives >1 CIK for the class —
    left unresolved, not picked), `name_mismatch` (one CIK, but SEC's name history does
    not contain Breakwater's company — left unresolved), `unresolved` (no link at all).
    """
    rows, segs = [], []
    for r in u.itertuples():
        hits = sec_ticker_matches(r.stock, r.renamed_to, tickers)
        ciks = sorted({h["cik"] for h in hits})
        rec = {"stock": r.stock, "bw_name": r.bw_name, "sector": r.sector,
               "renamed_to": r.renamed_to, "benzinga_identity_hazard": r.benzinga_identity_hazard,
               "sec_ciks": ";".join(map(str, ciks)),
               "sec_tickers": ";".join(sorted({h["sec_ticker"] for h in hits})),
               "sec_rank": min((h["sec_rank"] for h in hits), default=None)}
        manual = MANUAL.get(r.stock, [])
        if len(ciks) > 1:
            rec.update(status="ambiguous", provenance=UNRESOLVED,
                       note="today's SEC map lists more than one CIK for this class")
        elif len(ciks) == 1:
            cik = ciks[0]
            sub = submissions.get(cik)
            names = sec_names(sub) if sub else []
            ok = any(names_agree(r.bw_name, n) for n in names)
            via = hits[0]["via"]
            rec.update(sec_name=sub.get("name") if sub else None)
            if sub is None:
                rec.update(status="unresolved", provenance=UNRESOLVED,
                           note="CIK in ticker map but no submissions file")
            elif not ok and not any(m.cik == cik for m in manual):
                rec.update(status="name_mismatch", provenance=UNRESOLVED,
                           note=f"SEC names {names} vs Breakwater {r.bw_name!r}")
            else:
                rec.update(status="confident", provenance=via if ok else MANUAL_VERIFIED)
                if not any(m.cik == cik for m in manual):
                    segs.append({"stock": r.stock, "cik": cik, "valid_from": OPEN_START,
                                 "valid_to": OPEN_END, "provenance": via,
                                 "sec_name": sub.get("name"), "reason": "current SEC ticker map"})
        else:
            rec.update(status="unresolved", provenance=UNRESOLVED,
                       note="ticker not in today's SEC map (delisted / renamed)")
        # Manual links: verified against SEC's name history, else dropped (and said so).
        bad = []
        for m in manual:
            sub = submissions.get(m.cik)
            names = sec_names(sub) if sub else []
            if sub is None or not any(m.expected_name.lower() in n.lower() for n in names):
                bad.append(m.cik)
                continue
            segs.append({"stock": r.stock, "cik": m.cik,
                         "valid_from": pd.Timestamp(m.valid_from) if m.valid_from else OPEN_START,
                         "valid_to": pd.Timestamp(m.valid_to) if m.valid_to else OPEN_END,
                         "provenance": MANUAL_VERIFIED, "sec_name": sub.get("name"),
                         "reason": m.reason})
        if manual:
            rec["manual_links"] = len(manual)
            rec["manual_failed_verification"] = ";".join(map(str, bad))
            if rec["status"] == "unresolved" and not bad:
                rec.update(status="confident", provenance=MANUAL_VERIFIED,
                           note="manual link verified against SEC name history")
        rows.append(rec)
    audit = pd.DataFrame(rows)
    seg = pd.DataFrame(segs, columns=["stock", "cik", "valid_from", "valid_to", "provenance",
                                      "sec_name", "reason"])
    # A stock is only as good as its status; drop segments of non-confident stocks.
    seg = seg[seg["stock"].isin(audit.loc[audit["status"].eq("confident"), "stock"])]
    return audit, seg.reset_index(drop=True)
