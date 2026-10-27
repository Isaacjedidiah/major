"""Source citations for the chat app.

Purpose: answer the question every reviewer asks of a RAG/LLM system — "how do
we know the app is showing real content from the documents, and not inventing
it?" Every answer the app gives can carry a citation back to the exact source:
which document, and (for metrics) which page/slide. A reviewer can then open
that page and verify the figure themselves.

Two answer paths, two levels of citation:
  * METRIC answers (a value pulled from Gold) cite document + page/slide, because
    Gold now carries `source_document` and `page`. This is the strongest proof —
    a specific number tied to a specific page is hard to fake.
  * TEXT answers (a narrative chunk retrieved from the vector index) cite the
    source document. Page-level text citations would need `page` promoted to a
    chunk column (an index rebuild); document-level works today.

Nothing here writes to storage or the index — it only formats what the pipeline
already produced, so it's safe to wire into the app without schema changes.
"""

from __future__ import annotations

from typing import Any, Optional


def _locator_kind(document: str) -> str:
    """'slide' for PowerPoint sources, else 'page'. PPTX 'pages' are slides, and
    labelling them 'page 4' confuses reviewers who see slides."""
    d = (document or "").lower()
    return "slide" if d.endswith((".pptx", ".ppt")) else "page"


def format_metric_citation(row: dict) -> str:
    """One inline citation for a Gold metric row, e.g.
    '[q1_fx.pptx, slide 4]'. Falls back gracefully when fields are missing:
    document only, or a neutral marker if even that is absent."""
    doc = row.get("source_document") or row.get("source_document_id")
    page = row.get("page")
    if not doc:
        return "[source unavailable]"
    if page is None:
        return f"[{doc}]"
    return f"[{doc}, {_locator_kind(doc)} {page}]"


def format_chunk_citation(result: dict) -> str:
    """One inline citation for a retrieved text chunk. Chunks carry the source
    document; page isn't a returned column yet, so this cites the document.
    Accepts either a raw search result ({text, metadata, score}) or a flat
    dict."""
    meta = result.get("metadata", result)
    doc = meta.get("source_document_id") or meta.get("source_document")
    if not doc:
        return "[source unavailable]"
    return f"[{doc}]"


def cite_metric(row: dict) -> str:
    """A metric answer line WITH its citation, ready to show in chat, e.g.
    'FX exposure: £2.3bn  [q1_fx.pptx, slide 4]'."""
    metric = row.get("canonical_metric") or row.get("field_name") or "metric"
    value = row.get("value", "")
    return f"{metric}: {value}  {format_metric_citation(row)}"


def build_sources_block(metric_rows: Optional[list[dict]] = None,
                        chunk_results: Optional[list[dict]] = None) -> str:
    """Assemble a 'Sources' footer for a chat answer, de-duplicated and ordered
    metrics-first (page-level, strongest) then text (document-level).

    Wire this under the model's answer text:
        answer + "\\n\\n" + build_sources_block(metrics, chunks)

    Returns "" when there's nothing to cite, so callers can append unconditionally.
    """
    lines: list[str] = []
    seen: set[str] = set()

    for row in (metric_rows or []):
        c = format_metric_citation(row)
        if c not in seen:
            seen.add(c)
            metric = row.get("canonical_metric") or row.get("field_name") or ""
            lines.append(f"- {metric}: {row.get('value','')} {c}")

    for res in (chunk_results or []):
        c = format_chunk_citation(res)
        if c not in seen:
            seen.add(c)
            lines.append(f"- {c}")

    if not lines:
        return ""
    return "Sources (verify in the original document):\n" + "\n".join(lines)


def attach_citations(answer_text: str,
                     metric_rows: Optional[list[dict]] = None,
                     chunk_results: Optional[list[dict]] = None) -> str:
    """Convenience: the model's answer plus a Sources footer. If there's nothing
    to cite, returns the answer unchanged."""
    block = build_sources_block(metric_rows, chunk_results)
    return f"{answer_text}\n\n{block}" if block else answer_text
