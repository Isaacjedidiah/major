"""Parse the Unity Catalog volume path into recognisable entity columns.

Documents live under a fixed folder structure:

    /Volumes/<catalog>/<schema>/submissions/<team>/<report_type>/
             <entity>/<entity_id>/<entity_special_name>/<file>

so the identifying attributes (team, report_type, entity, entity_id,
entity_special_name) are encoded in the path. This module pulls them out so
they can become COLUMNS on every claim — making it easy to recognise which
entity a metric belongs to, without opening the document.

Parse once at ingestion and attach the fields to each submission; they then
flow through Gold, the analytical layer and the index automatically.
"""

from __future__ import annotations

from typing import Optional


def parse_entity_path(path: str) -> dict:
    """Extract entity attributes from a document's volume path.

    Expected layout (segments after '.../submissions/'):
        <team>/<report_type>/<entity>/<entity_id>/<entity_special_name>/<file>

    Returns team, report_type, entity, entity_id, entity_name (the special
    name). Missing segments come back as None — never guessed. Tolerant of a
    leading 'dbfs:' and of extra or missing trailing segments.
    """
    if not path:
        return _empty()

    p = path.replace("dbfs:", "").strip()
    parts = [seg for seg in p.split("/") if seg]

    # anchor on the 'submissions' folder; the entity structure follows it.
    try:
        si = parts.index("submissions")
        seg = parts[si + 1:]
    except ValueError:
        seg = parts                 # no 'submissions' anchor — parse positionally

    def _at(i):
        return seg[i] if i < len(seg) else None

    return {
        "team": _at(0),
        "report_type": _at(1),
        "entity": _at(2),
        "entity_id": _at(3),
        "entity_name": _at(4),      # the 'entity_special_name' segment
    }


def enrich_submission(sub: dict) -> dict:
    """Add the parsed entity columns onto a submission dict (from its path),
    without overwriting any value the submission already carries explicitly."""
    parsed = parse_entity_path(sub.get("path", ""))
    for k, v in parsed.items():
        if v is not None and not sub.get(k):
            sub[k] = v
    return sub


def _empty() -> dict:
    return {"team": None, "report_type": None, "entity": None,
            "entity_id": None, "entity_name": None}
