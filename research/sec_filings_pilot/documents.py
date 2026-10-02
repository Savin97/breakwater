"""Turn EDGAR documents into text and check what can be found in them. No network here.

Nothing in this module scores, labels or summarises a document. It answers only: does the
file decode, does HTML become readable text, how much of it is tables, can the standard
sections be located, and what exhibits does a filing carry.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from charset_normalizer import from_bytes
from lxml import etree, html as lhtml

_META_CHARSET = re.compile(rb'charset\s*=\s*["\']?([A-Za-z0-9_\-]+)', re.I)
_MOJIBAKE = re.compile("Ã.|â€|Â |�")
_BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "center",
          "title", "pre"}
_NUM_TOKEN = re.compile(r"^[\(\$\-–—]*[\d,.]+%?\)?$")


# ─────────────────────────────────────────── index ──────────────────────────────────────
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S | re.I)
_HREF = re.compile(r'href="([^"]+)"', re.I)
_TAG = re.compile(r"<[^>]+>")


def parse_index(page: bytes) -> dict:
    """The filing index page: accepted time, filing date, period, and the document table."""
    t = page.decode("utf-8", errors="replace")

    def info(label):
        m = re.search(rf">{label}</div>\s*<div class=\"info\">([^<]+)<", t)
        return m.group(1).strip() if m else None

    docs = []
    start = t.find('summary="Document Format Files"')
    end = t.find("</table>", start)
    for row in _ROW.findall(t[start:end] if start >= 0 else ""):
        cells = _CELL.findall(row)
        if len(cells) < 5:
            continue
        href = _HREF.search(cells[2])
        if not href:
            continue
        url = href.group(1)
        if "ix?doc=" in url:                      # inline XBRL viewer link -> raw document
            url = url.split("ix?doc=", 1)[1]
        filename = url.rsplit("/", 1)[-1]
        docs.append({"seq": _TAG.sub("", cells[0]).replace("&nbsp;", "").strip(),
                     "description": _TAG.sub("", cells[1]).strip(),
                     "filename": filename,
                     "type": _TAG.sub("", cells[3]).replace("&nbsp;", "").strip(),
                     "size": int(re.sub(r"\D", "", _TAG.sub("", cells[4])) or 0)})
    return {"accepted": info("Accepted"), "filing_date": info("Filing Date"),
            "period": info("Period of Report"), "documents": docs}


def is_earnings_release(doc: dict, text_head: str = "") -> bool:
    """An EX-99 exhibit that presents itself as an earnings/results release. Metadata first,
    then the first 3,000 characters of its text. Deliberately narrow — counted, not used."""
    if not doc.get("type", "").upper().startswith("EX-99"):
        return False
    meta = f"{doc.get('description', '')} {doc.get('filename', '')}".lower()
    if re.search(r"press|release|earnings|results|\bpr\b|financial", meta):
        return True
    head = text_head[:3000].lower()
    return bool(re.search(r"(reports?|announces?)\s.{0,80}(results|earnings)|"
                          r"(first|second|third|fourth)[- ]quarter.{0,40}(results|earnings)|"
                          r"(earnings|news|press)\s+release|operating\s+results", head))


# ──────────────────────────────────────── decoding ──────────────────────────────────────
@dataclass
class Decoded:
    text: str
    encoding: str
    declared: str | None
    replacement_chars: int
    mojibake_hits: int


def decode(b: bytes) -> Decoded:
    m = _META_CHARSET.search(b[:4096])
    declared = m.group(1).decode().lower() if m else None
    for enc in [declared, "utf-8"] if declared else ["utf-8"]:
        try:
            s = b.decode(enc)
            return Decoded(s, enc, declared, s.count("�"), len(_MOJIBAKE.findall(s)))
        except (LookupError, UnicodeDecodeError):
            continue
    best = from_bytes(b).best()
    enc = best.encoding if best else "latin-1"
    s = b.decode(enc, errors="replace")
    return Decoded(s, enc, declared, s.count("�"), len(_MOJIBAKE.findall(s)))


# ─────────────────────────────────────── HTML -> text ───────────────────────────────────
@dataclass
class TextStats:
    text: str
    chars: int = 0
    table_chars: int = 0
    numeric_table_chars: int = 0
    n_tables: int = 0
    prose_blocks: int = 0
    prose_chars: int = 0
    hidden_chars_dropped: int = 0
    is_html: bool = True
    errors: list = field(default_factory=list)


def _norm_ws(s: str) -> str:
    s = s.replace(" ", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


def html_to_text(raw: str) -> TextStats:
    """Readable text with block structure kept. Drops script/style, the inline-XBRL header
    and `display:none` content (iXBRL hidden facts). Counts table vs prose characters."""
    if not re.search(r"<(html|body|div|p|table|font)\b", raw[:20000], re.I):
        t = _norm_ws(raw)
        return TextStats(text=t, chars=len(t), prose_chars=len(t), is_html=False,
                         prose_blocks=len([b for b in t.split("\n\n") if len(b.split()) >= 30]))
    st = TextStats(text="")
    try:
        doc = lhtml.document_fromstring(raw.encode("utf-8", errors="replace"),
                                        parser=lhtml.HTMLParser(encoding="utf-8", recover=True,
                                                                huge_tree=True))
    except (etree.ParserError, ValueError) as e:
        st.errors.append(f"parse:{e}")
        return st
    for el in list(doc.iter("script", "style", "{*}header", "ix:header")):
        el.drop_tree()
    for el in doc.xpath('//*[contains(translate(@style," ",""),"display:none")]'):
        st.hidden_chars_dropped += len(el.text_content())
        el.drop_tree()
    for el in doc.iter():
        if isinstance(el.tag, str) and el.tag.lower() in _BLOCK:
            el.tail = ("\n" + el.tail) if el.tail else "\n"
    for el in doc.iter("td", "th"):
        el.tail = " | " + (el.tail or "")
    for tbl in doc.iter("table"):
        txt = _norm_ws(tbl.text_content())
        st.n_tables += 1
        st.table_chars += len(txt)
        toks = [t for t in txt.split() if t != "|"]
        if toks and sum(bool(_NUM_TOKEN.match(t)) for t in toks) / len(toks) >= 0.25:
            st.numeric_table_chars += len(txt)
    text = _norm_ws(doc.text_content())
    st.text, st.chars = text, len(text)
    for block in text.split("\n"):
        if len(block.split()) >= 30 and block.count("|") < 3:
            st.prose_blocks += 1
            st.prose_chars += len(block)
    return st


# ─────────────────────────────────────── sections ───────────────────────────────────────
_G = r"[\s|.:\-\u2013\u2014\u201c\u201d\"]*"      # cell separators, dashes, quotes between number and title
_SECTIONS = {
    "10-K": {
        "risk_factors": (rf"item\s*1a{_G}risk\s+factors", r"item\s*1b|item\s*1c|item\s*2\b"),
        "mdna": (rf"item\s*7{_G}management.{{0,3}}s\s+discussion", r"item\s*7a|item\s*8\b"),
    },
    "10-Q": {
        "mdna": (rf"item\s*2{_G}management.{{0,3}}s\s+discussion", r"item\s*3\b|item\s*4\b"),
        "risk_factors": (rf"item\s*1a{_G}risk\s+factors", r"item\s*2\b|item\s*5\b|item\s*6\b"),
    },
}
# Fallback when the body heading drops the item number (running headers, "Item 7." in a
# separate table cell on the TOC only): title to the next section's title.
_TITLE_FALLBACK = {
    "mdna": (r"management.{0,3}s\s+discussion\s+and\s+analysis\s+of\s+(the\s+)?(consolidated\s+)?"
             r"(financial\s+condition|results)",
             r"quantitative\s+and\s+qualitative\s+disclosures?\s+(about|of)\s+market\s+risk|"
             r"controls\s+and\s+procedures|financial\s+statements\s+and\s+supplementary\s+data"),
    "risk_factors": (r"\brisk\s+factors\b", r"unresolved\s+staff\s+comments|"
                     r"unregistered\s+sales\s+of\s+equity|\bproperties\b"),
}
_MIN_SECTION = {"risk_factors": 2000, "mdna": 3000}
_8K_ITEM = re.compile(r"item\s*(\d\.\d\d)", re.I)


def locate_sections(text: str, form: str) -> dict:
    """Length of each standard section's BODY (not its table-of-contents line).

    For every heading match, the span to the next end-marker is measured; the longest span
    is taken as the body. A 10-Q risk-factor section that only refers back to the 10-K is
    real but short — its length is reported either way, `found` needs a minimum length.
    """
    out = {}
    base = form.removesuffix("/A")
    low = text.lower()
    for name, (start, stop) in _SECTIONS.get(base, {}).items():
        best = 0
        for m in re.finditer(start, low):
            nxt = re.search(stop, low[m.end():])
            span = nxt.start() if nxt else min(len(low) - m.end(), 400_000)
            best = max(best, span)
        out[f"{name}_heading"] = bool(re.search(start, low))
        out[f"{name}_chars"] = best
        out[f"{name}_found"] = best >= _MIN_SECTION[name]
        fb = best
        if best < _MIN_SECTION[name]:
            fstart, fstop = _TITLE_FALLBACK[name]
            for m in re.finditer(fstart, low):
                nxt = re.search(fstop, low[m.end():])
                fb = max(fb, nxt.start() if nxt else 0)
        out[f"{name}_chars_with_title_fallback"] = fb
        out[f"{name}_found_with_title_fallback"] = fb >= _MIN_SECTION[name]
    if base == "8-K":
        out["items_in_text"] = ",".join(sorted(set(_8K_ITEM.findall(text))))
    return out
