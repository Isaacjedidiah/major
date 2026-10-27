"""Rule-based structuring of ai_parse_document output into composite-named
claims — NO LLM cost. ai_parse returns tables as HTML (``<table><tr><th>...``)
and text as blobs; both are structured enough to parse with code, so we build
the ``<section> | <entity/row> | <column> = value`` metric names deterministically
instead of paying for an LLM to structure them.

Two extractors:
  * ``html_table_to_claims`` — parses an HTML table (headers + rows, incl.
    multi-level headers and colspan/rowspan) into one claim per data cell, with
    a composite name combining the row identity and the full column-header path.
  * ``numbers_from_text`` — pulls monetary/numeric values embedded in prose
    (e.g. "X Account purchased £125m" -> "X Account purchased" = £125m) with the
    surrounding phrase as the metric name.

Titles/sections linked upstream are passed in and prefixed onto the names.
Everything returns standard Claim objects for the existing storage path.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Optional


# ---- HTML table parsing ----

class _TableParser(HTMLParser):
    """Minimal HTML table parser: collects rows as lists of (text, is_header,
    colspan, rowspan). Tolerant of the ai_parse table markup."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[dict]] = []
        self._cur_row = None
        self._cur_cell = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "tr":
            self._cur_row = []
        elif tag in ("td", "th"):
            self._cur_cell = {"text": "", "header": tag == "th",
                              "colspan": int(a.get("colspan", 1) or 1),
                              "rowspan": int(a.get("rowspan", 1) or 1)}

    def handle_endtag(self, tag):
        if tag == "tr" and self._cur_row is not None:
            self.rows.append(self._cur_row)
            self._cur_row = None
        elif tag in ("td", "th") and self._cur_cell is not None:
            if self._cur_row is not None:
                self._cur_row.append(self._cur_cell)
            self._cur_cell = None

    def handle_data(self, data):
        if self._cur_cell is not None:
            self._cur_cell["text"] += data


_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# unit tokens that may accompany a value (£m, %, bps, etc.)
_UNIT_RE = re.compile(r"(£|\$|€|%|bps|bn|m\b|k\b|million|billion)", re.I)


# $mm / £mm / mm = millions notation; recognise it for the currency/unit split.
_CUR_RE = re.compile(r"[£$€]")
_SCALE_RE = re.compile(r"(?<![a-zA-Z])(mm|bn|billion|million|m|k)(?![a-zA-Z])", re.I)
_PCT_RE = re.compile(r"(%|bps)")


def _split_currency_unit(text):
    """Split a value or label into (currency, unit). currency is the symbol
    (£/$/€) if present; unit is the scale/measure (mm, bn, m, k, %, bps). Handles
    $mm, £mm, mm (millions) that the old single-field logic missed. Either may be
    None. e.g. '$125mm' -> ('$', 'mm'); '15%' -> (None, '%'); '£m' -> ('£', 'm')."""
    if not text:
        return None, None
    s = str(text)
    cur_m = _CUR_RE.search(s)
    currency = cur_m.group(0) if cur_m else None
    pct_m = _PCT_RE.search(s)
    scale_m = _SCALE_RE.search(s)
    if pct_m:
        unit = pct_m.group(1)
    elif scale_m:
        unit = scale_m.group(1).lower().replace("million", "m").replace("billion", "bn")
    else:
        unit = None
    return currency, unit


def _extract_unit(value: str) -> Optional[str]:
    """Pull a unit/currency token from a value string (e.g. '£125m' -> '£m',
    '15%' -> '%'). Returns None if no unit is present."""
    if not value:
        return None
    tokens = _UNIT_RE.findall(value)
    if not tokens:
        return None
    # normalise: currency symbol + scale (e.g. £ + m -> £m)
    cur = next((t for t in tokens if t in ("£", "$", "€")), "")
    scale = next((t.lower() for t in tokens
                  if t.lower() in ("m", "bn", "k", "million", "billion")), "")
    pct = next((t for t in tokens if t in ("%", "bps")), "")
    unit = (cur + scale) if (cur or scale) else pct
    return unit or (tokens[0] if tokens else None)


def _unit_from_label(text):
    """Extract a unit from a HEADER/TITLE label and return (unit, cleaned_text).
    e.g. 'Revenue ($m)' -> ('$m', 'Revenue'); 'Report in £m' -> ('£m', 'Report').
    If no unit, returns (None, text unchanged). The unit is stripped from the
    text so the field_name / canonical_metric stays clean."""
    if not text:
        return None, text
    s = str(text)
    # find a parenthesised or 'in <unit>' unit token
    m = re.search(r"\((?:\s*in\s+)?\s*([£$€]\s*(?:mm|bn|billion|million|m|k)?|%|bps)\s*\)"
                  r"|\bin\s+([£$€]\s*(?:mm|bn|billion|million|m|k)?|%|bps)\b",
                  s, re.I)
    if not m:
        # also handle a trailing bare unit like 'Revenue £m'
        m2 = re.search(r"([£$€]\s*(?:mm|bn|billion|million|m|k)|%|bps)\s*$", s, re.I)
        if not m2:
            return None, text
        raw = m2.group(1)
        cleaned = _clean(s[:m2.start()])
        return _norm_label_unit(raw), (cleaned or s)
    raw = m.group(1) or m.group(2)
    cleaned = _clean((s[:m.start()] + " " + s[m.end():]))
    return _norm_label_unit(raw), (cleaned or s)


def _norm_label_unit(raw):
    """Normalise a label unit token to a clean form: '$ m' -> '$m', '£' -> '£',
    '%'/'bps' as-is."""
    if not raw:
        return None
    t = re.sub(r"\s+", "", str(raw).lower())
    # keep currency symbol case as-is, lowercase the scale
    t = t.replace("million", "m").replace("billion", "bn")
    return t


def _ai_parse_tier(from_figure: bool = False):
    """Citation tier for ai_parse-derived claims, honest about the source:
      * TABLES and TEXT -> PARSED. ai_parse returns tables as structured HTML and
        text verbatim; our rule-based parser reads them DETERMINISTICALLY, so
        these are 'deterministic structural output' — the PARSED definition.
      * FIGURES -> LLM_ESTIMATED. A figure's values come from ai_parse's
        multimodal DESCRIPTION (an LLM interpreting the chart image), which is an
        estimate, not a deterministic read.
    """
    try:
        from ..shared.schema import CitationTier
        return CitationTier.LLM_ESTIMATED if from_figure else CitationTier.PARSED
    except Exception:
        return None


def _has_number(s: str) -> bool:
    return bool(_NUM_RE.search(s or ""))


# a GENUINE numeric value: the whole cell is a number (optional parentheses for
# negative, optional sign, optional currency, digits with separators/decimals,
# optional unit/%), NOT a text+number mix like 'UTS-8976', 'Delta2', '89+',
# 'S&C'. Only genuine numbers become VALUES; everything else is an identifier.
_GENUINE_NUM_RE = re.compile(
    r"^\s*\(?[-+]?[£$€]?\s?\d[\d,]*(?:\.\d+)?\s?"
    r"(?:%|bps|bn|m|k|million|billion)?\)?\s*$", re.I)


def _is_genuine_number(s: str) -> bool:
    """True only if the cell is a standalone number (optionally signed/paren/
    currency/unit). Rejects text+digit mixes (IDs, codes, 'Delta2', '89+')."""
    return bool(_GENUINE_NUM_RE.match(str(s or "").strip()))


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def html_table_to_markdown(html: str) -> Optional[str]:
    """Render an ai_parse HTML table as a clean MARKDOWN table for the index —
    readable AND structure-preserving (unlike a raw-HTML blob or flat lines).

    Reuses the SAME grid model as ``html_table_to_claims`` so complex tables are
    handled faithfully: colspan/rowspan headers are expanded into a full grid,
    multi-level headers are flattened per column, merged cells repeat. This means
    the index gets a table the model can actually read and reason over — e.g.
    'what was Japan's WWE THS' — at no LLM cost.
    """
    if not html or "<" not in html:
        return None
    try:
        p = _TableParser()
        p.feed(html)
    except Exception:
        return None
    rows = [r for r in p.rows if r]
    if not rows:
        return None

    # build the grid honouring colspan + rowspan (same as the claim parser)
    grid: dict = {}
    occupied: set = set()
    n_cols = 0
    for ri, r in enumerate(rows):
        c = 0
        for cell in r:
            while (ri, c) in occupied:
                c += 1
            cs = max(1, cell.get("colspan", 1))
            rs = max(1, cell.get("rowspan", 1))
            for dr in range(rs):
                for dc in range(cs):
                    grid[(ri + dr, c + dc)] = _clean(cell["text"])
                    occupied.add((ri + dr, c + dc))
            c += cs
            n_cols = max(n_cols, c)
    n_rows = len(rows)
    if n_cols == 0:
        return None

    # detect how many leading rows are HEADER rows (mostly non-numeric),
    # so multi-level headers merge into ONE markdown header row per column.
    def _row_cells(ri):
        return [grid.get((ri, c), "") for c in range(n_cols)]

    def _is_num(s):
        return bool(re.search(r"\d", s or ""))

    n_header = 0
    for ri in range(min(n_rows, 3)):
        cells = _row_cells(ri)
        if any(_is_num(c) for c in cells):
            break
        n_header += 1
    n_header = max(1, n_header)

    # merge the header rows per column (dedup repeated span text)
    header = []
    for c in range(n_cols):
        parts = []
        for ri in range(n_header):
            t = grid.get((ri, c), "")
            if t and t not in parts:
                parts.append(t)
        header.append(" ".join(parts))

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in range(n_cols)) + " |")
    for ri in range(n_header, n_rows):
        lines.append("| " + " | ".join(_row_cells(ri)) + " |")
    return "\n".join(lines)


def html_table_to_claims(html: str, filename: str, page: Optional[int] = None,
                         title: Optional[str] = None, team=None,
                         report_type=None, entity_ref=None, claim_cls=None) -> list:
    """Parse an ai_parse HTML table into one claim per data cell, handling:
      * MULTI-LEVEL column headers (nested <th> rows joined by column) ;
      * SECTION rows (a full-width row spanning all columns with no numbers,
        e.g. 'CASH - OUTFLOWS') used as a prefix for the rows beneath ;
      * NESTED row labels (several leading non-numeric cells, e.g.
        portfolio->ty->position) combined into the row identity.
    Each numeric data cell -> '<title> | <section> | <row path> | <column> =
    value'. Deterministic, no LLM.
    """
    if claim_cls is None:
        from ..shared.schema import Claim as claim_cls
    if not html or "<" not in html:
        return []
    try:
        p = _TableParser()
        p.feed(html)
    except Exception:
        return []
    rows = [r for r in p.rows if r]
    if not rows:
        return []

    # Build a proper GRID that honours BOTH colspan and rowspan, so header cells
    # like 'Region' (rowspan=2) occupy their column across both header rows, and
    # 'date' (colspan=2) headers occupy two columns. grid[r][c] = (text, header).
    # This is what makes multi-level headers WITH rowspan row-labels align.
    grid: dict = {}          # (row, col) -> {"text":, "header":}
    occupied: set = set()
    n_cols = 0
    for ri, r in enumerate(rows):
        c = 0
        for cell in r:
            while (ri, c) in occupied:      # skip cells already filled by a rowspan
                c += 1
            cs = max(1, cell.get("colspan", 1))
            rs = max(1, cell.get("rowspan", 1))
            for dr in range(rs):
                for dc in range(cs):
                    grid[(ri + dr, c + dc)] = {"text": cell["text"],
                                               "header": cell["header"]}
                    occupied.add((ri + dr, c + dc))
            c += cs
            n_cols = max(n_cols, c)
    n_rows = len(rows)

    # which rows are HEADER rows (mostly header cells, no data numbers)?
    def _row_cells(ri):
        return [grid.get((ri, c), {"text": "", "header": False})
                for c in range(n_cols)]
    header_row_idx = []
    for ri in range(n_rows):
        cells = _row_cells(ri)
        hdr = sum(1 for c in cells if c["header"]) >= max(1, n_cols // 2) \
            and not any(_has_number(c["text"]) for c in cells if not c["header"])
        if hdr and (not header_row_idx or ri == header_row_idx[-1] + 1):
            header_row_idx.append(ri)
        elif header_row_idx:
            break
    body_row_idx = [ri for ri in range(n_rows) if ri not in header_row_idx]

    # column-header path per column: join the header rows' cells for that column
    # (rowspan cells repeat, so a spanned label like 'Region' appears once).
    # a header row that is a single full-width text (same text across all cols,
    # e.g. a 'Customer Report' title spanning the table) is a TITLE row, not
    # column headers -- exclude it so it does not merge into every col header.
    def _is_title_row(ri):
        vals = [_clean(grid.get((ri, c), {}).get("text", "")) for c in range(n_cols)]
        nonblank = [v for v in vals if v]
        return len(set(nonblank)) == 1 and n_cols > 1
    header_cols_rows = [ri for ri in header_row_idx if not _is_title_row(ri)]

    col_headers: dict[int, str] = {}
    for c in range(n_cols):
        parts = []
        for ri in header_cols_rows:
            t = _clean(grid.get((ri, c), {}).get("text", ""))
            if t and t not in parts:        # dedup repeated rowspan text
                parts.append(t)
        col_headers[c] = " ".join(parts)

    # UNIT resolution from TITLE / HEADER / SUB-HEADER. ai_parse often puts the
    # unit ($m, £m, %, bps) in the column header ('Revenue ($m)'), a sub-header,
    # or the table title ('... in £m') rather than on each (bare) value. Extract
    # it from there, CLEAN it out of the header/title text (so names stay clean),
    # and apply it to bare values with precedence: value > column > title.
    col_unit: dict = {}
    for _c in range(n_cols):
        _u, _cleaned = _unit_from_label(col_headers.get(_c, ""))
        if _u:
            col_unit[_c] = _u
            col_headers[_c] = _cleaned        # strip the unit from the header name
    title_unit = None
    if title:
        title_unit, _title_clean = _unit_from_label(title)
        if title_unit:
            title = _title_clean              # strip the unit from the title name

    # -----------------------------------------------------------------------
    # GRID-INTEGRITY check (agnostic): the HTML's span instructions can ask for
    # MORE logical columns than the raw cells actually delivered (e.g. a colspan
    # claims 21 columns but the body rows only hold 19 raw cells, one blank). We
    # trust the RAW CELLS, not the span claim. If a body row's own cell count is
    # smaller than the span-grid width, we align by the grid positions that the
    # body actually fills -- blanks included -- rather than the phantom span
    # width. (The grid already places each real cell at its true position; we
    # simply never require more columns than the data provides.)
    # -----------------------------------------------------------------------
    # the real column count = the widest BODY row's filled extent in the grid
    def _row_last_filled(ri):
        last = -1
        for c in range(n_cols):
            if _clean(grid.get((ri, c), {}).get("text", "")):
                last = c
        return last
    body_extent = max((_row_last_filled(ri) for ri in body_row_idx), default=-1)
    eff_cols = min(n_cols, body_extent + 1) if body_extent >= 0 else n_cols

    # -----------------------------------------------------------------------
    # CELL-CLASSIFICATION extraction (universal; no label-column detection).
    # For each body row:
    #   * TEXT cells become IDENTIFIERS in the canonical_metric, each tagged with
    #     its column header -> '<header>: <text>' (full text, never truncated).
    #   * NUMERIC cells each become a VALUE claim: field_name = the column header;
    #     canonical_metric = title | section | <all row identifiers> | <header>.
    # This removes the fragile 'which columns are labels' detection entirely --
    # a number is always a value (so a table that starts with a number is fine),
    # and text is always an identifier. Everything is copied for traceability;
    # nothing is cut off.
    # -----------------------------------------------------------------------
    claims = []
    eid = f"{filename}-p{page}-aiparsetbl"
    section = None
    carry_text: dict = {}                   # carry blank/merged row identifiers down

    for ri in body_row_idx:
        texts = [_clean(grid.get((ri, c), {}).get("text", "")) for c in range(eff_cols)]
        non_empty = [t for t in texts if t]

        # SECTION row: one distinct label spanning the width, no numbers.
        distinct = list(dict.fromkeys(non_empty))
        if len(distinct) == 1 and not _has_number(distinct[0]) and eff_cols > 1:
            section = distinct[0]
            continue
        if not non_empty:
            continue

        # gather this row's TEXT-cell identifiers as '<header>: <text>' (carry
        # down a blank cell's previous identifier so merged rows keep context).
        # Also build a structured {header: text} dict for the query_gold map.
        identifiers = []
        ident_map: dict = {}
        for c in range(eff_cols):
            t = texts[c]
            hdr = col_headers.get(c, "")
            if t and not _is_genuine_number(t):
                carry_text[c] = (hdr, t)
                identifiers.append(f"{hdr}: {t}" if hdr else t)
                if hdr:
                    ident_map[hdr] = t
            elif not t and c in carry_text:
                hdr0, t0 = carry_text[c]
                identifiers.append(f"{hdr0}: {t0}" if hdr0 else t0)
                if hdr0:
                    ident_map[hdr0] = t0

        # emit a VALUE claim for each NUMERIC cell in the row
        for c in range(eff_cols):
            val = texts[c]
            if val and _is_genuine_number(val):
                header = col_headers.get(c, "")
                parts = [x for x in ([title] + ([section] if section else [])
                                     + identifiers + [header]) if x]
                name = " | ".join(parts) if parts else (header or "value")
                # unit precedence: value's own unit > column header unit > title
                # unit (title/sub-heading $mm now captured). The resolved token is
                # then SPLIT into currency (£/$/€) and unit (mm/bn/m/k/%/bps) so the
                # two live in separate columns.
                _resolved = (_extract_unit(val) or col_unit.get(c) or title_unit)
                _currency, _unit = _split_currency_unit(_resolved) if _resolved \
                    else (None, None)
                claims.append(claim_cls(
                    field_name=header or "value", canonical_metric=name,
                    value=val, unit=_unit, currency=_currency, element_type="table",
                    source_element_id=eid,
                    page=page, entity_ref=entity_ref, confidence=0.9,
                    citation_tier=_ai_parse_tier(),
                    model_used="ai_parse_rule_structured",
                    identifiers=dict(ident_map) if ident_map else None))
    return claims


# ---- numbers embedded in text ----

# a value = a currency/number token, optionally with a unit suffix.
_VALUE_RE = re.compile(
    r"(?:[£$€]\s?\d[\d,]*(?:\.\d+)?\s?(?:m|bn|k|bps|%)?"
    r"|\d[\d,]*(?:\.\d+)?\s?(?:m|bn|k|bps|%|million|billion))",
    re.I)


def chart_description_to_claims(description: str, filename: str,
                                page: Optional[int] = None, title: Optional[str] = None,
                                team=None, report_type=None, entity_ref=None,
                                claim_cls=None) -> list:
    """Parse an ai_parse figure DESCRIPTION (multimodal, often contains the
    actual chart values) into label->value claims. E.g.

      "A vertical bar chart titled 'Quarterly Revenue 2025' ... Q1 at
       approximately $12M, Q2 at $15M, Q3 at $14M, and Q4 peaking at $22M."

    -> 'Quarterly Revenue 2025 | Q1 = $12M', '... | Q2 = $15M', etc.

    Rule-based, no LLM. Pairs each value with the LABEL that precedes it,
    stripping filler words ('at', 'approximately', 'peaking at'). Uses the
    chart's own title (parsed from the description) as the name prefix when
    present.
    """
    if claim_cls is None:
        from ..shared.schema import Claim as claim_cls
    desc = _clean(description)
    if not desc:
        return []

    # chart title, if the description states one: titled '...' or "..."
    chart_title = None
    m = re.search(r"titled\s+['\"]([^'\"]+)['\"]", desc, re.I)
    if m:
        chart_title = m.group(1).strip()
    prefix = " | ".join(x for x in (title, chart_title) if x) or None

    # filler tokens between a label and its value.
    _FILLER = re.compile(
        r"\b(?:at|of|is|was|are|reached|reaching|peaking|approximately|approx|"
        r"about|around|roughly|circa|=|:|to)\b", re.I)

    claims = []
    eid = f"{filename}-p{page}-fig"
    # find each value, then take the label = the token(s) just before it,
    # stripped of filler.
    for m in _VALUE_RE.finditer(desc):
        val = _clean(m.group(0))
        before = desc[:m.start()]
        # last clause before the value (split on comma/'and'/semicolon)
        clause = re.split(r",|;|\band\b", before, flags=re.I)[-1]
        clause = _FILLER.sub(" ", clause)          # drop filler words
        # strip leading narrative phrases ("the data shows", "the chart shows")
        clause = re.sub(r".*\b(?:shows?|shown|following|below|are)\b[:\s]*", "",
                        clause, flags=re.I)
        label = _clean(clause).split()
        label = " ".join(label[-4:])               # last few words = the label
        label = _clean(label)
        if not label:
            continue
        name = " | ".join(x for x in (prefix, label) if x)
        claims.append(claim_cls(
            field_name=label, canonical_metric=name, value=val,
            unit=_extract_unit(val), element_type="text", source_element_id=eid, page=page,
            entity_ref=entity_ref, confidence=0.75,
            citation_tier=_ai_parse_tier(from_figure=True),
            model_used="ai_parse_desc_structured"))
    return claims


def numbers_from_text(text: str, filename: str, page: Optional[int] = None,
                      title: Optional[str] = None, team=None, report_type=None,
                      entity_ref=None, claim_cls=None) -> list:
    """Extract numeric/monetary values embedded in prose, with the surrounding
    phrase as the metric name. E.g. 'X Account purchased £125m' ->
    name 'X Account purchased', value '£125m'. Rule-based, no LLM.

    Only fires on values carrying a unit or currency (£125m, 15%, 30bps) — a
    bare integer in prose is too ambiguous to name reliably and is skipped.
    """
    if claim_cls is None:
        from ..shared.schema import Claim as claim_cls
    t = _clean(text)
    if not t:
        return []
    claims = []
    eid = f"{filename}-p{page}-aiparsetxt"
    for m in _VALUE_RE.finditer(t):
        val = _clean(m.group(0))
        # the metric name = the few words immediately BEFORE the value.
        before = t[:m.start()].rstrip()
        # take up to the last ~6 words before the number as the name
        name_words = before.split()[-6:]
        name = _clean(" ".join(name_words))
        if not name:
            continue                          # a number with no context -> skip
        if title:
            name = f"{title} | {name}"
        claims.append(claim_cls(
            field_name=name, canonical_metric=name, value=val,
            unit=_extract_unit(val), element_type="text", source_element_id=eid, page=page,
            entity_ref=entity_ref, confidence=0.7,
            citation_tier=_ai_parse_tier(),
            model_used="ai_parse_rule_structured"))
    return claims
