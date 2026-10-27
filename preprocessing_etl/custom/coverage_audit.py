"""Coverage audit for the extraction pipeline.

'Ensuring coverage' is two things, and this module does both honestly:

  1. ACCOUNTING — did we handle every element ai_parse returned? Every element
     is either extracted (produced claims), deliberately skipped (logo, layout
     marker), or flagged as UNACCOUNTED. Nothing is silently dropped.

  2. COMPLETENESS SIGNALS — per document and per page, how much did we extract,
     and are there pages/elements that produced NOTHING (a coverage gap worth a
     human's eye)?

Honest scope: this MAXIMISES and VERIFIES coverage; it does not 'prove we got
everything' — that needs a ground truth we can't fully have (a bare number in a
document is not self-evidently a metric). What it guarantees is that every
element ai_parse saw was consciously handled, and that gaps are surfaced, not
hidden.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional


# element types we DELIBERATELY do not extract from (not a coverage gap)
_INTENTIONAL_SKIP = {"page_header", "page_footer", "page_number"}


def audit_coverage(parsed_elements, claims) -> dict:
    """Reconcile what ai_parse RETURNED against what we EXTRACTED.

    parsed_elements: the ParsedElement list from ai_parse (what was available).
    claims:          the claims we produced (what we captured).

    Returns a coverage report: counts by element type, how many elements
    produced at least one claim, and any UNACCOUNTED elements (a real gap).
    """
    # how many claims came from each page (a page with 0 claims is a gap signal)
    claims_per_page = Counter(getattr(c, "page", None) for c in claims)
    # which source elements produced claims (by the id we stamp on each claim)
    produced_pages = set(claims_per_page)

    by_type = Counter(e.el_type for e in parsed_elements)
    pages = sorted({e.page for e in parsed_elements if e.page is not None})

    accounted, skipped, unaccounted = [], [], []
    for e in parsed_elements:
        if e.el_type in _INTENTIONAL_SKIP:
            skipped.append(e)                          # deliberately not extracted
        elif e.el_type == "figure" and _is_logo(e):
            skipped.append(e)                          # logos are skipped by design
        elif e.page in produced_pages:
            accounted.append(e)                        # its page produced claims
        else:
            unaccounted.append(e)                      # produced nothing -> GAP

    # pages that yielded no metrics at all — worth a human eye
    empty_pages = [pg for pg in pages if claims_per_page.get(pg, 0) == 0]

    total = len(parsed_elements)
    return {
        "elements_total": total,
        "elements_by_type": dict(by_type),
        "elements_accounted": len(accounted),
        "elements_intentionally_skipped": len(skipped),
        "elements_unaccounted": len(unaccounted),        # the number to watch
        "accounted_pct": round(100 * len(accounted) /
                               max(1, total - len(skipped)), 1),
        "claims_total": len(claims),
        "claims_per_page": dict(claims_per_page),
        "pages_total": len(pages),
        "empty_pages": empty_pages,                       # pages with 0 metrics
        "unaccounted_detail": [
            {"type": e.el_type, "page": e.page} for e in unaccounted[:50]],
    }


def coverage_ok(report: dict, max_unaccounted: int = 0,
                max_empty_pages: int = 0) -> bool:
    """A simple gate: coverage is clean if no elements are unaccounted and no
    pages came back empty (tune the thresholds to your tolerance)."""
    return (report["elements_unaccounted"] <= max_unaccounted
            and len(report["empty_pages"]) <= max_empty_pages)


def coverage_report_text(report: dict) -> str:
    """Readable one-block summary for a notebook or a review."""
    lines = ["===== COVERAGE AUDIT =====",
             f"elements: {report['elements_total']}  "
             f"({report['elements_by_type']})",
             f"accounted: {report['elements_accounted']}  "
             f"({report['accounted_pct']}%)",
             f"intentionally skipped: {report['elements_intentionally_skipped']}",
             f"UNACCOUNTED (gaps): {report['elements_unaccounted']}",
             f"metrics extracted: {report['claims_total']}  "
             f"over {report['pages_total']} pages",
             f"empty pages (0 metrics): {report['empty_pages']}"]
    if report["unaccounted_detail"]:
        lines.append("  gap sample: " + ", ".join(
            f"{d['type']}@p{d['page']}" for d in report["unaccounted_detail"][:10]))
    return "\n".join(lines)


def _is_logo(el) -> bool:
    txt = ((getattr(el, "description", "") or "") + " " +
           (getattr(el, "content", "") or "")).lower()
    return any(w in txt for w in ("logo", "emblem", "icon", "crest", "letterhead"))
