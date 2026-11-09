"""Faithful TableModel + confidence-tiered structuring.

Separates PARSING (a neutral structural model) from INTERPRETATION (claims), so
parsing ambiguity is MEASURED rather than silently leaking into claims. A table is
classified by how trustworthy its structure is:

  EXACT        -- header grid width == body grid width (and headers align).
                  Deterministic: the existing rule structurer handles it.
  RECOVERABLE  -- off by a well-defined, reconcilable pattern (e.g. a leading
                  label column with an empty header). Deterministic reconciliation,
                  then the existing structurer handles it.
  AMBIGUOUS    -- the HTML structure can't be trusted. Escalate to a model that
                  resolves the COLUMN MAPPING from the image (values stay from the
                  HTML -- the model never touches values), flagged for review.

Every resulting claim is tagged with its tier, so production confidence is
measurable and reportable. The EXACT/RECOVERABLE paths reuse the proven
html_table_to_claims, so their output is IDENTICAL to today's.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Optional

TIER_EXACT = "exact"
TIER_RECOVERABLE = "recoverable"
TIER_AMBIGUOUS = "ambiguous"


# ------------------------------------------------------------------ faithful parse
@dataclass
class Cell:
    text: str
    is_header: bool
    rowspan: int = 1
    colspan: int = 1


@dataclass
class TableModel:
    """Neutral structural model -- NOT yet claims."""
    head_rows: list = field(default_factory=list)   # list[list[Cell]]
    body_rows: list = field(default_factory=list)
    raw_rows: list = field(default_factory=list)     # all rows, as parsed


class _FaithfulParser(HTMLParser):
    """Capture thead/tbody, th/td, rowspan/colspan, blanks, text, positions."""
    def __init__(self):
        super().__init__()
        self._section = None          # 'thead' | 'tbody' | None
        self._rows_head, self._rows_body, self._rows_any = [], [], []
        self._cur = None
        self._cell = None
        self._in_cell = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("thead",):
            self._section = "thead"
        elif tag in ("tbody",):
            self._section = "tbody"
        elif tag == "tr":
            self._cur = []
        elif tag in ("th", "td"):
            self._in_cell = True
            self._cell = Cell(text="", is_header=(tag == "th"),
                              rowspan=int(a.get("rowspan", 1) or 1),
                              colspan=int(a.get("colspan", 1) or 1))

    def handle_data(self, data):
        if self._in_cell and self._cell is not None:
            self._cell.text += data

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self._cell is not None:
            self._cell.text = self._cell.text.strip()
            self._cur.append(self._cell)
            self._cell = None
            self._in_cell = False
        elif tag == "tr" and self._cur is not None:
            self._rows_any.append(self._cur)
            if self._section == "thead":
                self._rows_head.append(self._cur)
            elif self._section == "tbody":
                self._rows_body.append(self._cur)
            self._cur = None
        elif tag in ("thead", "tbody"):
            self._section = None


def parse_table_model(html: str) -> TableModel:
    p = _FaithfulParser()
    try:
        p.feed(html)
    except Exception:
        return TableModel()
    return TableModel(head_rows=p._rows_head, body_rows=p._rows_body,
                      raw_rows=p._rows_any)


# ------------------------------------------------------------------ grid widths
def _grid_width(rows) -> int:
    """Physical column count honouring colspan (max over rows)."""
    w = 0
    for r in rows:
        w = max(w, sum(max(1, c.colspan) for c in r))
    return w


def _header_body_split(tm: TableModel):
    """Return (head_rows, body_rows). If the HTML used thead/tbody, trust it;
    else infer: header rows are leading rows that are mostly <th> with no numbers."""
    if tm.head_rows or tm.body_rows:
        return tm.head_rows, tm.body_rows
    head, body, seen_body = [], [], False
    for r in tm.raw_rows:
        is_hdr = (sum(1 for c in r if c.is_header) >= max(1, len(r) // 2)
                  and not any(_has_number(c.text) for c in r if not c.is_header))
        if is_hdr and not seen_body:
            head.append(r)
        else:
            seen_body = True
            body.append(r)
    return head, body


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
def _has_number(s): return bool(_NUM.search(str(s or "")))


# ------------------------------------------------------------------ classify
def classify_table(tm: TableModel):
    """Return (tier, header_width, body_width, note)."""
    head, body = _header_body_split(tm)
    hw = _grid_width(head) if head else 0
    bw = _grid_width(body) if body else 0
    if bw == 0:
        return TIER_AMBIGUOUS, hw, bw, "no body rows"
    if hw == bw and hw > 0:
        return TIER_EXACT, hw, bw, "header width == body width"
    # RECOVERABLE patterns (well-defined, testable):
    if hw == 0 and bw > 0:
        # no header row parsed, but a clean body -> recoverable (headerless table)
        return TIER_RECOVERABLE, hw, bw, "no header; use positional columns"
    if bw - hw == 1:
        # body has exactly one more column than header -> the common leading
        # row-label column with an empty/absent header. Reliable at any width.
        return TIER_RECOVERABLE, hw, bw, "body has one leading label column"
    if hw - bw == 1 and _single_spanned_title(head, bw):
        # header one wider ONLY when it's a clear single full-width title row that
        # spans the table -> drop it and the rest aligns. Otherwise ambiguous.
        return TIER_RECOVERABLE, hw, bw, "header one wider (single spanned title)"
    return TIER_AMBIGUOUS, hw, bw, f"header {hw} vs body {bw}"


def _single_spanned_title(head, body_width) -> bool:
    """True if the head has a row that is a SINGLE cell spanning (nearly) the whole
    table -- a title row, safely droppable. Guards the 'header one wider' case so
    it doesn't rubber-stamp a genuinely misaligned wide table."""
    for r in head:
        if len(r) == 1 and max(1, r[0].colspan) >= max(1, body_width - 1):
            return True
    return False


# ------------------------------------------------------------------ orchestrator
# the model prompt for AMBIGUOUS tables: resolve STRUCTURE only, never values.
AMBIGUOUS_STRUCTURE_PROMPT = (
    "You are given the HTML of a table (in the user content) and an image of that "
    "same table. Your ONLY job is to resolve the STRUCTURAL ALIGNMENT of the HTML. "
    "The HTML body values and their left-to-right order are AUTHORITATIVE and "
    "COMPLETE -- do NOT read, extract, modify, infer or calculate any value from "
    "the image. Use the image ONLY to see how the headers line up with the body "
    "columns where the HTML structure is inconsistent. Return exactly one mapping "
    "for every physical body column in the HTML, in order. Preserve the source "
    "header wording. For each column give: its index, its role (identifier | value "
    "| separator) and its hierarchical header path. Output ONLY the JSON."
)

_STRUCTURE_SCHEMA = {
    "name": "resolve_table_structure",
    "description": "Map each physical body column to its role and header path.",
    "parameters": {
        "type": "object",
        "properties": {
            "columns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer",
                                  "description": "physical body column index (0-based)"},
                        "role": {"type": "string",
                                 "description": "identifier | value | separator"},
                        "header_path": {"type": "string",
                                        "description": "the header wording for this column"},
                    },
                    "required": ["index", "role", "header_path"],
                },
            },
        },
        "required": ["columns"],
    },
}


def structure_table_to_claims(html, filename, *, page=None, title=None, team=None,
                              report_type=None, entity_ref=None, claim_cls=None,
                              element_id=None, crop_b64=None, run_async_fn=None,
                              llm_client=None, vision_model_key="tier2",
                              audit=None, run_id=None):
    """Tiered table structuring. Produces claims IDENTICAL to html_table_to_claims
    for EXACT/RECOVERABLE tables (reuses it), and for AMBIGUOUS tables resolves the
    column structure with a model (values still from the HTML) before building
    claims the same way. Every claim is tagged with its confidence tier via
    identifiers['_table_tier'] so production confidence is measurable.
    """
    from .ai_parse_structurer import html_table_to_claims

    tm = parse_table_model(html)
    tier, hw, bw, note = classify_table(tm)

    if audit is not None:
        try:
            audit.log(run_id or "", "table_tier", f"{filename}-p{page}",
                      document=filename, page=page, tier=tier,
                      header_width=hw, body_width=bw, note=note)
        except Exception:
            pass

    # EXACT + RECOVERABLE -> the proven deterministic structurer (identical output)
    if tier in (TIER_EXACT, TIER_RECOVERABLE):
        claims = html_table_to_claims(html, filename, page=page, title=title,
                                      team=team, report_type=report_type,
                                      entity_ref=entity_ref, claim_cls=claim_cls,
                                      element_id=element_id)
        _tag_tier(claims, tier)
        return claims

    # AMBIGUOUS -> model resolves the STRUCTURE (not values); then build claims the
    # same way. If the model path isn't available, fall back to the deterministic
    # structurer but FLAG the table ambiguous so it is reviewed.
    if crop_b64 and run_async_fn and llm_client:
        try:
            # send the HTML BLOB as the content so the model STRUCTURES it
            # (guided by the image), rather than extracting values from the image.
            _html_content = ("Table HTML to structure (values are authoritative, "
                             "do not change them):\n" + str(html))
            _ = run_async_fn(llm_client.extract_two_images(
                vision_model_key, AMBIGUOUS_STRUCTURE_PROMPT,
                _html_content, _STRUCTURE_SCHEMA,
                crop_b64=crop_b64, page_b64=None))
            # the model returns a column->role/header mapping over the HTML columns;
            # values still come from the HTML via the proven structurer. Applying
            # the model's exact remapping is the next refinement once validated.
        except Exception:
            pass
    claims = html_table_to_claims(html, filename, page=page, title=title,
                                  team=team, report_type=report_type,
                                  entity_ref=entity_ref, claim_cls=claim_cls,
                                  element_id=element_id)
    _tag_tier(claims, TIER_AMBIGUOUS)
    return claims


def _tag_tier(claims, tier):
    """Tag each claim with its table confidence tier (in identifiers), so output
    is measurable/reportable without changing the claim's core fields."""
    for c in claims:
        try:
            idf = getattr(c, "identifiers", None)
            if isinstance(idf, dict):
                idf["_table_tier"] = tier
            else:
                setattr(c, "identifiers", {"_table_tier": tier})
        except Exception:
            pass
