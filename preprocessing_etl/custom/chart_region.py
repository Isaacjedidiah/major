"""Combine the elements that make up ONE chart into a single crop region, using
ELEMENT ORDER (sequence adjacency) rather than geometric position — because
chart/legend placement varies by report, team and entity, but ai_parse returns
related elements ADJACENT in the sequence: a header just before the figure, and
the legend (a footnote/caption) just after it.

So for a chart figure we look at its immediate neighbours in the element list:
  * the element right BEFORE it, if it's a header/title/caption/text -> include
  * the element right AFTER it, if it's a footnote/caption/text      -> include
and union their coords with the figure's. This is layout-agnostic: it doesn't
matter WHERE the legend sits on the slide, only that ai_parse emitted it next to
the figure.

Rule-based, deterministic, no LLM.
"""

from __future__ import annotations

from typing import Optional

# a neighbour of these types, adjacent to a figure, is treated as part of the
# chart (its header above, or its legend/caption below).
_HEADER_TYPES = {"title", "section_header", "caption", "text"}
_FOOTER_TYPES = {"footnote", "caption", "text"}


def _coord(el):
    bb = el.get("bbox") if isinstance(el, dict) else getattr(el, "bbox", None)
    if bb and isinstance(bb, list) and isinstance(bb[0], dict):
        return bb[0].get("coord")
    return None


def _page(el):
    bb = el.get("bbox") if isinstance(el, dict) else getattr(el, "bbox", None)
    if bb and isinstance(bb, list) and isinstance(bb[0], dict):
        return bb[0].get("page_id", 1)
    return 1


def _etype(el):
    return (el.get("type") if isinstance(el, dict) else getattr(el, "el_type", "")) or ""


def chart_crop_region(figure_el, all_elements) -> Optional[dict]:
    """Union a chart figure with the header immediately BEFORE it and the
    legend/footnote immediately AFTER it (in element order), on the same page.
    Layout-agnostic — uses sequence adjacency, not pixel position.

    Returns {"page_id":, "coord":[x0,y0,x1,y1]} or None.
    """
    fig_c = _coord(figure_el)
    if not fig_c:
        return None
    page_id = _page(figure_el)

    # find the figure's index in the sequence
    try:
        idx = next(i for i, e in enumerate(all_elements) if e is figure_el)
    except StopIteration:
        return {"page_id": page_id, "coord": list(fig_c)}

    x0, y0, x1, y1 = fig_c

    def _maybe_union(el, allowed_types):
        nonlocal x0, y0, x1, y1
        if el is None or _page(el) != page_id:
            return
        if _etype(el) not in allowed_types:
            return
        c = _coord(el)
        if not c:
            return
        x0 = min(x0, c[0]); y0 = min(y0, c[1])
        x1 = max(x1, c[2]); y1 = max(y1, c[3])

    # header right BEFORE the figure
    if idx > 0:
        _maybe_union(all_elements[idx - 1], _HEADER_TYPES)
    # legend/footnote right AFTER the figure
    if idx + 1 < len(all_elements):
        _maybe_union(all_elements[idx + 1], _FOOTER_TYPES)

    return {"page_id": page_id, "coord": [x0, y0, x1, y1]}
