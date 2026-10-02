"""Deterministic SEC-derived features. Pure functions; no I/O, no outcome, no fitting.

Every rule here is fixed in PREREGISTRATION.md. Text features are computed from ONE
document (or one pair of documents of the same company), so nothing is fitted across
companies or years: no vocabulary, no IDF, no scaling. That makes them causal by
construction — a release's features depend only on that release.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from research.sec_filings_pilot import documents

# ───────────────────────────────────── text preparation ─────────────────────────────────
# Paragraphs dropped before ANY text feature: legal boilerplate and logistics. Safe-harbor
# language is dense in "risks" and "uncertainties" and identical quarter to quarter, so it
# would swamp both the uncertainty rate and the change measures.
BOILERPLATE = re.compile(
    r"forward[- ]looking|safe\s+harbor|private\s+securities\s+litigation|"
    r"undue\s+reliance|non[- ]gaap|conference\s+call|webcast|investor\s+relations\s+contact|"
    r"media\s+contact|"
    # Amendment 1 (before outcomes): risk-factor lists of the safe-harbor paragraph.
    r"risk\s+factors|risks\s+and\s+uncertainties|uncertainties|actual\s+results|"
    r"could\s+cause|cautionary|no\s+obligation\s+to\s+update|undertakes?\s+no", re.I)
PROSE_MIN_WORDS = 12
PROSE_MAX_CELL_SEPARATORS = 2


def release_text(raw_html: str) -> tuple[str, str]:
    """(full text without boilerplate paragraphs, prose-only text). Tables stay in the full
    text (guidance is often a table); prose = lines of >= 12 words that are not table rows."""
    st = documents.html_to_text(raw_html)
    full, prose = [], []
    for line in st.text.split("\n"):
        if BOILERPLATE.search(line):
            continue
        full.append(line)
        if len(line.split()) >= PROSE_MIN_WORDS and line.count("|") <= PROSE_MAX_CELL_SEPARATORS:
            prose.append(line)
    return "\n".join(full), "\n".join(prose)


WORD = re.compile(r"[a-z][a-z'\-]+")


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


# ─────────────────────────────────────── A. guidance ────────────────────────────────────
# Amendment 1 (before outcomes; PREREGISTRATION.md): a passage rule replaces the sentence
# rule, which counted "lower revenues" / "growth projects" / "accounting guidance" and
# missed ranges printed on the next line.
G_TERMS = re.compile(
    r"\b(guidance|outlook|forecasts?|expects?|expected\s+to|expecting|anticipates?|"
    r"anticipated|reaffirm\w*|reiterat\w*)\b", re.I)
G_EXCLUDE = re.compile(
    r"(accounting|tax|regulatory|irs|supervisory|new)\s+guidance|economic\s+outlook|"
    r"than\s+expected|(not|unable\s+to)\s+(provid|forecast|giv)\w*|"
    r"(high|low)[- ]end\s+of\s+(the\s+|our\s+|its\s+)?(\w+\s+)?(guidance|outlook)|"
    r"(above|below|exceeded|beat|in\s+line\s+with|within)\s+(the\s+|our\s+|its\s+)?"
    r"(\w+\s+)?(guidance|outlook|expectations)", re.I)
T_TERMS = re.compile(
    r"\b(fiscal\s+(year\s+)?20\d\d|fiscal\s+\d{4}|fy\s?'?\d{2,4}|full[- ]year|"
    r"(first|second|third|fourth)[- ]quarter|q[1-4]|next\s+quarter|20\d\d|"
    r"(quarter|year)\s+ending)\b", re.I)
NUMBER = re.compile(r"\$\s?\d|\d+(\.\d+)?\s?%|\d+\.\d+")
PASSAGE_CHARS = 300


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.;!?])\s+(?=[A-Z(\"“$])|\n+", text)
    return [p.strip() for p in parts if len(p.split()) >= 3]


def guidance_sentences(full_text: str) -> list[str]:
    """Guidance passages: from a forward-looking guidance word to PASSAGE_CHARS after it
    (line breaks flattened), containing a forward period and a number, and not one of the
    excluded phrasings. Passages overlapping an earlier one are skipped."""
    flat = re.sub(r"\s*\n\s*", " ", full_text)
    out, last_end = [], -1
    for m in G_TERMS.finditer(flat):
        if m.start() < last_end:
            continue
        ctx = flat[max(0, m.start() - 60):m.start() + PASSAGE_CHARS]
        passage = flat[m.start():m.start() + PASSAGE_CHARS]
        if G_EXCLUDE.search(ctx):
            continue
        if T_TERMS.search(passage) and NUMBER.search(passage):
            out.append(passage)
            last_end = m.start() + PASSAGE_CHARS
    return out


# ───────────────────────────────────── B. guidance ranges ───────────────────────────────
_DASH = r"\s*(?:to|-|–|—|and)\s*"
EPS_RANGE = [
    re.compile(r"(?:\beps\b|earnings per (?:diluted |basic )?share|per diluted share|"
               r"diluted earnings per share)[^.;]{0,120}?\$\s?(\d{1,3}\.\d{2})" + _DASH +
               r"\$?\s?(\d{1,3}\.\d{2})", re.I),
    re.compile(r"\$\s?(\d{1,3}\.\d{2})" + _DASH + r"\$?\s?(\d{1,3}\.\d{2})\s*"
               r"(?:of |in )?(?:adjusted |gaap |diluted |non-gaap )*"
               r"(?:eps\b|earnings per|per (?:diluted )?share)", re.I),
]
REV_RANGE = re.compile(
    r"(?:revenues?|net sales|sales)[^.;]{0,120}?\$\s?(\d[\d,]*(?:\.\d+)?)\s*(billion|million)?"
    + _DASH + r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s*(billion|million)\b", re.I)
_SCALE = {"billion": 1e9, "million": 1e6, None: None, "": None}


def _width(lo: float, hi: float) -> float | None:
    if not (0 < lo < hi):
        return None
    w = (hi - lo) / ((hi + lo) / 2)
    return w if w < 1 else None


def eps_widths(gs: list[str]) -> list[float]:
    out = []
    for s in gs:
        for rx in EPS_RANGE:
            for m in rx.finditer(s):
                w = _width(float(m.group(1)), float(m.group(2)))
                if w is not None:
                    out.append(w)
    return out


def revenue_widths(gs: list[str]) -> list[float]:
    out = []
    for s in gs:
        for m in REV_RANGE.finditer(s):
            lo, s1, hi, s2 = m.group(1), m.group(2), m.group(3), m.group(4)
            scale_hi = _SCALE[s2.lower()]
            scale_lo = _SCALE[s1.lower()] if s1 else scale_hi
            w = _width(float(lo.replace(",", "")) * scale_lo, float(hi.replace(",", "")) * scale_hi)
            if w is not None:
                out.append(w)
    return out


# ─────────────────────────────────────── C. uncertainty ─────────────────────────────────
# Exploratory dictionary, fixed before any outcome was read (PREREGISTRATION.md §5C). In
# the spirit of Loughran-McDonald's "uncertainty" category plus common earnings-release
# words for adverse conditions; not the LM list itself.
UNCERTAINTY = re.compile(
    r"\b(uncertain\w*|volatil\w*|unpredictab\w*|challeng\w*|headwinds?|disrupt\w*|"
    r"pressures?|pressured|cautio\w*|difficult\w*|softness|soft(?:er|ening)|weak\w*|"
    r"slowdown|slow(?:ing|ed)|turbulen\w*|instabilit\w*|fluctuat\w*|visibility|risks?)\b", re.I)
PER = 1000.0


def uncertainty_rate(prose: str) -> tuple[float, int, int]:
    """(hits per 1,000 prose words, hits, prose words). NaN when there is no prose."""
    n = len(words(prose))
    hits = len(UNCERTAINTY.findall(prose))
    return (hits / n * PER if n else np.nan), hits, n


# ───────────────────────────────────────── E. novelty ───────────────────────────────────
STOP = frozenset(ENGLISH_STOP_WORDS)


def term_counts(prose: str) -> Counter:
    return Counter(w for w in words(prose) if len(w) >= 3 and w not in STOP)


def cosine_distance(a: Counter, b: Counter) -> float:
    """1 - cosine of raw term-frequency vectors of two documents. Nothing is fitted."""
    if not a or not b:
        return np.nan
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return float(1 - dot / (na * nb))


# ─────────────────────────────── direction source data (kept, not used) ─────────────────
DIRECTION = {
    "dir_raise": re.compile(r"\b(rais(?:es|ed|ing)|increas(?:es|ed|ing))\b[^.;]{0,60}"
                            r"\b(guidance|outlook|forecast)", re.I),
    "dir_lower": re.compile(r"\b(lower(?:s|ed|ing)?|reduc(?:es|ed|ing)|cut)\b[^.;]{0,60}"
                            r"\b(guidance|outlook|forecast)", re.I),
    "dir_reaffirm": re.compile(r"\b(reaffirm\w*|reiterat\w*|maintain\w*)\b[^.;]{0,60}"
                               r"\b(guidance|outlook|forecast)", re.I),
    "mentions_demand": re.compile(r"\bdemand\b", re.I),
    "mentions_margin": re.compile(r"\bmargins?\b", re.I),
    "mentions_inventory": re.compile(r"\binventor(?:y|ies)\b", re.I),
}


def release_features(raw_html: str) -> dict:
    """Every per-release quantity. The pairwise ones (change, novelty) are made from two of
    these by `pair_features`."""
    full, prose = release_text(raw_html)
    gs = guidance_sentences(full)
    eps, rev = eps_widths(gs), revenue_widths(gs)
    rate, hits, n = uncertainty_rate(prose)
    out = {"full_chars": len(full), "prose_words": n, "uncertainty_hits": hits,
           "uncertainty_rate": rate, "n_guidance_sentences": len(gs),
           "guidance_present": int(len(gs) > 0),
           "eps_guidance_width": float(np.median(eps)) if eps else np.nan,
           "revenue_guidance_width": float(np.median(rev)) if rev else np.nan,
           "n_eps_ranges": len(eps), "n_revenue_ranges": len(rev),
           "guidance_sentences_sample": " || ".join(gs[:5])[:2000]}
    for k, rx in DIRECTION.items():
        out[k] = len(rx.findall(full))
    return out


def pair_features(prev: dict, prev2: dict, prev_terms: Counter, prev2_terms: Counter) -> dict:
    return {"uncertainty_change": prev["uncertainty_rate"] - prev2["uncertainty_rate"],
            "release_text_change": cosine_distance(prev_terms, prev2_terms)}


# ─────────────────────────────────────────── 8-K ────────────────────────────────────────
RECENT_DAYS = 14          # fixed in PREREGISTRATION.md; no other window is tried
WINDOW_30D = 30
ITEMS = ("2.02", "7.01", "8.01", "5.02")
NON_MATERIAL_ITEMS = {"9.01", ""}


def _items(s: str) -> set[str]:
    return {i.strip() for i in str(s).split(",") if i.strip()}


def eightk_features(since_periodic: pd.DataFrame, eligible_all: pd.DataFrame,
                    cutoff_date: pd.Timestamp) -> dict:
    """since_periodic: the event's information-set 8-Ks (pilot rule; originals and /A).
    eligible_all: every eligible 8-K of the stock (originals and /A), any date."""
    orig = since_periodic[since_periodic["form"].eq("8-K")]
    items = set().union(*[_items(s) for s in orig["items"]]) if len(orig) else set()
    days = (cutoff_date - eligible_all["filing_date"]).dt.days
    e_orig = eligible_all["form"].eq("8-K")
    last30 = eligible_all[e_orig & days.between(0, WINDOW_30D - 1)]
    recent = eligible_all[e_orig & days.between(0, RECENT_DAYS - 1)]
    r_items = set().union(*[_items(s) for s in recent["items"]]) if len(recent) else set()
    out = {"num_8k_since_periodic": len(orig), "num_8k_last_30d": len(last30),
           "num_material_item_types": len(items - NON_MATERIAL_ITEMS)}
    for it in ITEMS:
        out[f"has_{it.replace('.', '_')}"] = int(it in items)
    for it in ("2.02", "7.01", "8.01"):
        out[f"has_recent_{it.replace('.', '_')}_14d"] = int(it in r_items)
    return out
