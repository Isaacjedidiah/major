"""Analytical Gold layer.

Gold stores each metric with a single composite name, e.g.

    COMPREHENSIVE STATEMENT ($MM) | Asset | Current | June 25   = 4000

That is precise but not easy to read or pivot. This module explodes the
composite name into separate analytical dimensions so the data can be filtered
and grouped by heading:

    level_1  = COMPREHENSIVE STATEMENT   (the '($MM)' is stripped out)
    level_2  = Asset
    level_3  = Current
    metric   = (the measured thing, when distinct from the levels)
    period   = June 25                   (the date/time segment)
    value    = 4000
    numeric_value = 4000.0
    currency = $                         (symbol; from the heading or the value)
    unit     = MM                        (scale/unit; heading or value, else None)

plus governance / traceability carried straight through:
    entity_ref, team, report_type, source_document, page,
    citation_tier, confidence, content_hash, extraction_date.

The split is deterministic and rule-based — no LLM. Currency/unit are looked
for in the value first, then in any heading parenthetical like '($MM)'; if
neither has them, they are None (never guessed).
"""

from __future__ import annotations

import re
from typing import Optional

# currency symbols -> kept as the symbol (no risky ISO mapping).
_CUR = {"£": "£", "$": "$", "€": "€"}
_CUR_RE = re.compile(r"[£$€]")

# scale / unit tokens, normalised lightly so they group (M and MM -> MM).
# Two forms: standalone in a heading ('($MM)') OR attached to a number ('4844m',
# '15%', '30bps').
_UNIT_STANDALONE = re.compile(r"\b(bps|MM|BN|M|B|K)\b", re.I)
_UNIT_ATTACHED = re.compile(r"\d\s*(bps|MM|BN|M|B|K|%)", re.I)
_PCT = re.compile(r"%")
_UNIT_NORM = {"M": "MM", "MM": "MM", "B": "BN", "BN": "BN", "K": "K",
              "BPS": "bps", "%": "%"}

# a parenthetical carrying currency/unit, e.g. '($MM)', '(£m)', '(%)'
_PAREN_RE = re.compile(r"\(([^)]*)\)")

# the date/period segment — reuse the pipeline's date sense where possible.
try:
    from .dates import looks_like_date
except Exception:                       # pragma: no cover
    def looks_like_date(s):             # minimal fallback
        return bool(re.search(r"\b(19|20)\d{2}\b", s or ""))


def _extract_currency(text: str) -> Optional[str]:
    if not text:
        return None
    m = _CUR_RE.search(text)
    return _CUR.get(m.group(0)) if m else None


def _extract_unit(text: str) -> Optional[str]:
    if not text:
        return None
    # percent first (attaches directly to a number: '15%')
    if _PCT.search(text):
        return "%"
    # a unit attached to a number: '£4844m', '30bps'
    m = _UNIT_ATTACHED.search(text)
    if m:
        return _UNIT_NORM.get(m.group(1).upper(), m.group(1))
    # a standalone unit token in a heading: '($MM)'
    m = _UNIT_STANDALONE.search(text)
    if m:
        return _UNIT_NORM.get(m.group(1).upper(), m.group(1))
    return None


def _clean_label(seg: str) -> str:
    """Drop a trailing/inline currency-unit parenthetical from a heading label,
    e.g. 'COMPREHENSIVE STATEMENT ($MM)' -> 'COMPREHENSIVE STATEMENT'."""
    if not seg:
        return seg
    def _drop(m):
        inside = m.group(1)
        # only strip the parenthetical if it's purely currency/unit noise
        if _CUR_RE.search(inside) or _UNIT_STANDALONE.search(inside) \
                or _PCT.search(inside):
            return ""
        return m.group(0)
    return re.sub(r"\s+", " ", _PAREN_RE.sub(_drop, seg)).strip()


def split_metric_row(row: dict) -> dict:
    """Explode one Gold row's composite canonical_metric into semantic heading
    columns, keeping value/currency/unit and governance/traceability.

    Headings (positional, left to right):
        statement     -> 1st segment  (e.g. 'COMPREHENSIVE STATEMENT')
        classification-> 2nd segment  (e.g. 'Asset')
        category      -> 3rd segment  (e.g. 'Current')
        sub_category  -> everything remaining, joined (incl. any inline period
                         text like 'June 25') — so nothing is lost when a name
                         has more segments than named columns.

    We deliberately DO NOT try to detect a period inside the name (not every
    table has one, and that detection is fragile). The authoritative time
    dimension is ``report_date``, derived at document level and carried through.

    Currency/unit are read from the value first, then any heading parenthetical
    like '($MM)'; if neither has them, they are None (never guessed). Heading
    labels have that currency/unit parenthetical stripped so they read cleanly.
    """
    name = str(row.get("canonical_metric") or "")
    segments = [s.strip() for s in name.split("|") if s.strip()]

    value = str(row.get("value") or "")
    currency = _extract_currency(value)
    unit = row.get("unit") or _extract_unit(value)

    cleaned = []
    for seg in segments:
        # currency/unit may hide in a heading like 'COMPREHENSIVE STATEMENT ($MM)'
        if currency is None:
            currency = _extract_currency(seg)
        if not unit:
            unit = _extract_unit(seg)
        c = _clean_label(seg)
        if c:
            cleaned.append(c)

    # pull any date-like segments OUT of the hierarchy into `period` (e.g.
    # 'June 25', '09/12/2024', '31-Mar-26'). report_date stays the authoritative
    # document-level period; `period` just surfaces a date found in the name.
    period_parts = [c for c in cleaned if looks_like_date(c)]
    non_date = [c for c in cleaned if not looks_like_date(c)]
    period = " | ".join(period_parts) if period_parts else None

    statement      = non_date[0] if len(non_date) > 0 else None
    classification = non_date[1] if len(non_date) > 1 else None
    category       = non_date[2] if len(non_date) > 2 else None
    # sub_category absorbs any remaining non-date segments (4th onward)
    sub_category   = " | ".join(non_date[3:]) if len(non_date) > 3 else None

    return {
        "statement": statement,
        "classification": classification,
        "category": category,
        "sub_category": sub_category,
        "period": period,              # date found IN the name (if any)
        "field_name": row.get("field_name"),   # the measure / column header
        "value": row.get("value"),
        "numeric_value": row.get("numeric_value"),
        "currency": currency,          # symbol or None (never guessed)
        "unit": unit or None,          # scale/unit or None
        # authoritative period + governance + traceability carried through
        "report_date": row.get("report_date"),
        "entity_ref": row.get("entity_ref"),
        "team": row.get("team"),
        "report_type": row.get("report_type"),
        "source_document": row.get("source_document"),
        "page": row.get("page"),
        "citation_tier": row.get("citation_tier"),
        "confidence": row.get("confidence"),
        "content_hash": row.get("content_hash"),
        "extraction_date": row.get("extraction_date"),
    }


def build_analytical_rows(gold_rows: list[dict]) -> list[dict]:
    """Explode a list of Gold rows into analytical rows."""
    return [split_metric_row(r) for r in gold_rows]
