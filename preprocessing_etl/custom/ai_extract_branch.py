"""ai_extract-based extraction branch — composes ai_parse_document with
ai_extract (Databricks Foundation Model APIs) for structured extraction, as an
alternative to the rule-based ai_parse_structurer.

ai_extract accepts ai_parse_document's VARIANT output directly and returns
structured fields per a schema you supply, so the whole flow is SQL:

    ai_parse_document(content) -> ai_extract(<parsed>, <schema>, <options>)

This branch trades cost (ai_extract is an LLM/DBU call) for robustness (the LLM
handles messy tables/prose the rule-based parser struggles with). Use it only
where the rule-based path is inadequate and ai_extract is approved for use.

This module provides:
  * the extraction SCHEMA (what metric fields to pull), tuned for regulatory
    tables/charts — a list of {label, value, unit, entity, period} records;
  * ``build_ai_extract_sql`` — the SQL that chains ai_parse -> ai_extract;
  * ``ai_extract_result_to_claims`` — maps the returned VARIANT into Claim
    objects for the existing storage path.

The actual spark.sql call is left to the caller (notebook), since it is
environment-specific; this module builds the SQL and maps the result, and is
unit-testable without a cluster.
"""

from __future__ import annotations

import json
from typing import Any, Optional


# The extraction schema: ai_extract v2 takes a JSON schema describing the fields
# to pull. We ask for an ARRAY of metric records, each carrying the coordinates
# needed to build a composite name plus the value/unit. Descriptions guide the
# model (v2 supports per-field descriptions).
METRIC_SCHEMA = {
    "type": "object",
    "properties": {
        "metrics": {
            "type": "array",
            "description": (
                "Every quantitative data point in the document — from tables, "
                "charts, and prose. One record per value."),
            "items": {
                "type": "object",
                "properties": {
                    "entity": {"type": "string", "description":
                               "The row entity/organisation the value belongs "
                               "to (e.g. 'X Soverign'), or empty if none."},
                    "section": {"type": "string", "description":
                                "The section/title/heading the value sits under "
                                "(e.g. 'CASH - OUTFLOWS', 'FINANCIAL RISK')."},
                    "label": {"type": "string", "description":
                              "What the value measures — the column header(s) or "
                              "the metric name (e.g. 'Exposure', 'Q1 Revenue', "
                              "'Tier 1 Capital')."},
                    "value": {"type": "string", "description":
                              "The numeric value exactly as shown, with any "
                              "currency/unit symbol (e.g. '£4844m', '15%', "
                              "'1790')."},
                    "unit": {"type": "string", "description":
                             "The unit if shown (£m, %, bps, count), else empty."},
                    "period": {"type": "string", "description":
                               "The time period/date for the value if any "
                               "(e.g. '31-Mar-26', 'Q1 2024'), else empty."},
                },
                "required": ["label", "value"],
            },
        }
    },
    "required": ["metrics"],
}


# Domain guidance passed to ai_extract's 'instructions' option.
EXTRACT_INSTRUCTIONS = (
    "You are extracting quantitative metrics from a regulatory/financial report. "
    "Capture every data value from tables, charts, and prose. For each value, "
    "give its entity (row), section (heading), label (column/metric), value "
    "(with unit), unit, and period. For multi-level table headers combine the "
    "levels into the label. For charts, read each series/category and its value. "
    "Never invent values; omit anything not clearly present."
)


def build_ai_extract_sql(view_name: str = "_doc",
                         mode: str = "precision") -> str:
    """Build the SQL that chains ai_parse_document -> ai_extract over a temp
    view holding a ``content`` (binary) column. Returns a VARIANT column
    ``extracted`` with the metrics array.
    """
    schema_json = json.dumps(METRIC_SCHEMA).replace("'", "\\'")
    instr = EXTRACT_INSTRUCTIONS.replace("'", "\\'")
    return (
        "SELECT ai_extract("
        "ai_parse_document(content), "
        f"'{schema_json}', "
        f"map('mode', '{mode}', 'instructions', '{instr}', 'version', '2.0')"
        f") AS extracted FROM {view_name}"
    )


def ai_extract_result_to_claims(extracted: Any, filename: str,
                                team=None, report_type=None, entity_ref=None,
                                claim_cls=None) -> list:
    """Map ai_extract's returned VARIANT (a {metrics: [...]} structure) into
    Claim objects, building the composite name section|entity|label and carrying
    value/unit/period. Tolerant of VariantVal / dict / JSON string input.
    """
    if claim_cls is None:
        from ..shared.schema import Claim as claim_cls
    data = _to_dict(extracted)
    if not isinstance(data, dict):
        return []
    metrics = data.get("metrics") or []
    if not isinstance(metrics, list):
        return []

    from ..shared.schema import CitationTier
    tier = CitationTier.LLM_ESTIMATED

    claims = []
    eid = f"{filename}-aiextract"
    for m in metrics:
        if not isinstance(m, dict):
            continue
        value = str(m.get("value", "")).strip()
        label = str(m.get("label", "")).strip()
        if not value or not label:
            continue
        parts = [str(m.get(k, "")).strip() for k in ("section", "entity", "label")]
        parts = [p for p in parts if p]
        name = " | ".join(parts) if parts else label
        period = str(m.get("period", "")).strip() or None
        if period:
            name = f"{name} | {period}"
        unit = str(m.get("unit", "")).strip() or None
        claims.append(claim_cls(
            field_name=label, canonical_metric=name, value=value, unit=unit,
            as_at_date=period, source_element_id=eid, entity_ref=entity_ref,
            confidence=0.8, citation_tier=tier, model_used="ai_extract"))
    return claims


def _to_dict(v: Any):
    """Normalise VariantVal / JSON string / dict to a dict."""
    if v is None:
        return None
    if isinstance(v, dict):
        return v
    if type(v).__name__ == "VariantVal":
        for meth in ("toJson", "to_json"):
            if hasattr(v, meth):
                try:
                    return json.loads(getattr(v, meth)())
                except Exception:
                    return None
        try:
            return json.loads(str(v))
        except Exception:
            return None
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return None
    if hasattr(v, "asDict"):
        try:
            return v.asDict(recursive=True)
        except Exception:
            return None
    return None
