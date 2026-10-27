"""Quarantine-driven recovery fallback for slides the main pipeline couldn't read.

The problem it solves: some PowerPoint slides contain images the direct pipeline
can't rasterise on serverless (EMF/WMF vector charts). Those fail with "could
not process image" and land in bronze_quarantine, with a source_element_id like
``report.pptx-s2-img7`` (s2 = slide 2). This module runs a SECOND pass that:

  1. reads the quarantine to learn EXACTLY which slides failed, per document;
  2. renders those documents to slide images via ai_parse_document (which the
     direct pipeline can't do on serverless);
  3. sends ONLY the failed slides' images to Sonnet to recover the values;
  4. tags the recovered claims as fallback-sourced (provenance + safe re-runs).

Duplicate safety: ai_parse renders the WHOLE document, but we only process the
slides that actually failed (from the quarantine), so slides the first pass
already extracted are never re-stored. The failed-slide filter is the dedup.

Pure-logic where possible (quarantine parsing, slide filtering, mapping); the
live ai_parse call, image reading, and Sonnet call are injected so this is
unit-testable without a cluster.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from .hybrid_extraction import (parse_sonnet_chart_points, sonnet_points_to_claims,
                                CHART_DIGITISATION_PROMPT, ParsedElement)


# source_element_id looks like "<file>-s<slide>-img<n>" (PPTX) or
# "<file>-p<page>-fig<n>" (PDF). Pull the document and the slide/page number.
_SLIDE_RE = re.compile(r"-s(\d+)-", re.I)
_PAGE_RE = re.compile(r"-p(\d+)-", re.I)


def failed_slides_from_quarantine(rows: list[dict]) -> dict[str, set]:
    """From quarantine rows, build {source_document: {failed slide numbers}}.

    Only rows that are image/figure failures are considered (those are the ones
    a render+vision retry can recover). A row is keyed by its source_document;
    the slide number comes from the source_element_id (-s<N>- for PPTX, -p<N>-
    for PDF). Rows without a locatable slide/page are skipped.
    """
    out: dict[str, set] = {}
    for r in rows:
        sid = str(r.get("source_element_id", "") or "")
        doc = r.get("source_document") or _doc_from_sid(sid)
        if not doc:
            continue
        # only image/figure failures are recoverable by render+vision
        err = str(r.get("error", "") or "").lower()
        is_image_failure = ("image" in err or "img" in sid or "fig" in sid)
        if not is_image_failure:
            continue
        m = _SLIDE_RE.search(sid) or _PAGE_RE.search(sid)
        if not m:
            continue
        out.setdefault(doc, set()).add(int(m.group(1)))
    return out


def _doc_from_sid(sid: str) -> Optional[str]:
    """Best-effort document name from a source_element_id prefix."""
    if not sid:
        return None
    m = re.split(r"-(?:s\d+-|p\d+-)?(?:img|fig|tbl)\d+", sid)
    return m[0] if m and m[0] else None


def recover_document(document_path: str,
                     failed_slides: set,
                     render_slides: Callable[[str], dict],
                     read_slide_image: Callable[[Any], str],
                     entity_ref=None, team=None, report_type=None,
                     claim_cls=None) -> list:
    """Recover claims for the FAILED slides of one document.

    Injected callables (environment-specific, wired on the cluster):
      * render_slides(document_path) -> {slide_number: <image handle>} — runs
        ai_parse_document with imageOutputPath and returns the rendered slide
        images keyed by slide number. Renders ALL slides (ai_parse works on the
        whole document); we only USE the failed ones.
      * read_slide_image(image_handle) -> raw Sonnet text — sends the slide image
        to Sonnet (with the chart-digitisation prompt) and returns its output.

    Only slides in ``failed_slides`` are processed, so slides the first pass
    already handled are never re-extracted (dedup by construction).
    """
    if claim_cls is None:
        from ..shared.schema import Claim as claim_cls
    import os
    filename = os.path.basename(document_path)

    rendered = render_slides(document_path) or {}
    claims: list = []
    for slide_no in sorted(failed_slides):
        img = rendered.get(slide_no)
        if img is None:
            continue                      # ai_parse didn't render this slide
        raw = read_slide_image(img)
        points = parse_sonnet_chart_points(raw or "")
        # synthesise a ParsedElement so the shared mapper can build claims with
        # the right page + source id.
        el = ParsedElement(el_type="figure", content=None, confidence=None,
                           page=slide_no, bbox=None)
        recovered = sonnet_points_to_claims(points, el, filename, entity_ref,
                                            claim_cls=claim_cls)
        # tag as fallback-recovered for provenance + safe dedup on re-runs
        for c in recovered:
            c.model_used = "ai_parse_sonnet_fallback"
            c.source_element_id = f"{filename}-s{slide_no}-recovered"
            if team is not None:
                c.team = team
            if report_type is not None:
                c.report_type = report_type
        claims.extend(recovered)
    return claims


def run_quarantine_fallback(quarantine_rows: list[dict],
                            document_paths: dict[str, str],
                            render_slides: Callable[[str], dict],
                            read_slide_image: Callable[[Any], str],
                            submissions_meta: Optional[dict] = None,
                            claim_cls=None) -> list:
    """Top-level: recover every document that has image failures in quarantine.

    ``document_paths`` maps source_document -> full path to the file (so we can
    render it). Attribution (team/report_type/entity_ref) is taken from the
    QUARANTINE ROWS themselves (self-sufficient — the extractor now stamps them
    on each quarantined element), falling back to ``submissions_meta`` if a row
    lacks them. Returns all recovered claims, ready for the existing storage path.
    """
    failed = failed_slides_from_quarantine(quarantine_rows)

    # build per-document attribution from the quarantine rows (preferred) or the
    # supplied submissions_meta (fallback).
    attribution: dict[str, dict] = {}
    for r in quarantine_rows:
        doc = r.get("source_document") or _doc_from_sid(str(r.get("source_element_id", "")))
        if not doc or doc in attribution:
            continue
        meta = {}
        if r.get("team") is not None:
            meta = {"team": r.get("team"), "report_type": r.get("report_type"),
                    "entity_ref": r.get("entity_ref")}
        elif submissions_meta and doc in submissions_meta:
            meta = submissions_meta[doc]
        attribution[doc] = meta

    all_claims: list = []
    for doc, slides in failed.items():
        path = document_paths.get(doc)
        if not path:
            continue
        meta = attribution.get(doc) or (submissions_meta or {}).get(doc, {})
        all_claims.extend(recover_document(
            path, slides, render_slides, read_slide_image,
            entity_ref=meta.get("entity_ref"), team=meta.get("team"),
            report_type=meta.get("report_type"), claim_cls=claim_cls))
    return all_claims
