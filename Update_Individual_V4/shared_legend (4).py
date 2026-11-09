"""SHARED LEGEND via COMBINED crop+page read (one call, crop primary).

The problem: a chart whose legend is DETACHED (shared) comes out of the normal
crop read with unnamed series (a colour tag, or "Series 1"), because the legend
isn't in the crop. Reading the legend separately and joining by colour is
unreliable (the model doesn't always emit a colour).

This approach instead does ONE read with BOTH images:
  * image 1 = the CROP (primary -- all values come from here),
  * image 2 = the WHOLE PAGE (reference ONLY -- used to find the legend if the
    crop's series are unnamed).
The model matches colour->series INTERNALLY and returns NAMED series directly, so
there is no fragile join key. Values come from the crop (focused -> less
hallucination than full-page extraction); the page is only a legend reference.

Used ONLY for charts that came out with UNNAMED series (colour tag or "Series N").
Charts that already have named series are NOT re-read.

TEST BEFORE RELYING ON IT: sending the whole page back in risks the hallucination
that cropping was meant to avoid. Verify on real charts that values still come
from the crop and the page is only used for the legend.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

# a series is UNNAMED if it is a bare colour, "Series N", "unlabeled", or empty.
_COLOUR_WORDS = None


def _colour_words():
    global _COLOUR_WORDS
    if _COLOUR_WORDS is None:
        try:
            from matplotlib.colors import CSS4_COLORS
            _COLOUR_WORDS = set(CSS4_COLORS.keys())
        except Exception:
            _COLOUR_WORDS = {"red", "green", "blue", "orange", "purple", "teal",
                             "navy", "maroon", "cyan", "magenta", "yellow", "pink",
                             "brown", "grey", "gray", "black", "olive", "lime",
                             "indigo", "violet", "gold", "lightblue", "darkblue"}
    return _COLOUR_WORDS


def _is_unnamed_series(name):
    """True if the metric's series part looks UNNAMED: ends in a colour tag, is a
    'Series N'/'unlabeled' placeholder, etc. These are the charts needing the
    page legend."""
    s = str(name or "").lower()
    tail = s.rsplit(" - ", 1)[-1].rsplit(" | ", 1)[-1].strip()
    tail = tail.strip("()").strip()
    if not tail:
        return True
    if re.match(r"^series\s*\d+$", tail):
        return True
    if "unlabeled" in tail or "unlabelled" in tail:
        return True
    words = re.findall(r"[a-zA-Z]+", tail)
    shapes = {"bar", "segment", "point", "line", "wedge", "slice", "series"}
    # tail is only colour/shape words -> unnamed
    cols = _colour_words()
    if words and all(w in cols or w in shapes for w in words):
        return True
    return False


# ---- combined-read prompt + schema ----
COMBINED_PROMPT = (
    "You are given TWO images. IMAGE 1 is a single cropped chart. IMAGE 2 is the "
    "whole page it came from (reference only). Read ONLY the chart in IMAGE 1 and "
    "return its series with proper NAMES. If a series in IMAGE 1 has NO legend/name "
    "you can read from the crop itself, look at IMAGE 2's LEGEND to find the name "
    "for that colour, and use it. Do NOT read any other chart from IMAGE 2; use it "
    "ONLY to resolve a missing legend. Return every bar/point with its series name "
    "and value. Output ONLY the JSON."
)

_COMBINED_SCHEMA = {
    "name": "record_chart_named",
    "description": "Record the cropped chart's points with resolved series names.",
    "parameters": {
        "type": "object",
        "properties": {
            "chart_title": {"type": "string"},
            "points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "series": {"type": "string",
                                   "description": "the series NAME (resolved from "
                                                  "the page legend if the crop has "
                                                  "none)"},
                        "label": {"type": "string",
                                  "description": "x-axis / category label"},
                        "value": {"type": "string"},
                        "unit": {"type": "string"},
                    },
                    "required": ["series", "value"],
                },
            },
        },
        "required": ["points"],
    },
}


def _parse_json(resp):
    ta = getattr(resp, "tool_arguments", None)
    if isinstance(ta, dict):
        return ta
    txt = getattr(resp, "text", "") or ""
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def read_crop_with_page(run_async_fn, llm_client, model_key, crop_b64, page_b64):
    """ONE combined read: crop (primary) + page (legend reference). Returns
    (title, points) where points carry RESOLVED series names. [] on empty/error."""
    try:
        resp = run_async_fn(llm_client.extract_two_images(
            model_key, COMBINED_PROMPT, COMBINED_PROMPT, _COMBINED_SCHEMA,
            crop_b64=crop_b64, page_b64=page_b64))
    except Exception:
        return "", []
    data = _parse_json(resp)
    title = str(data.get("chart_title") or "").strip()
    out = []
    for p in (data.get("points") or []):
        if isinstance(p, dict) and p.get("value") not in (None, ""):
            out.append({"series": str(p.get("series") or "").strip(),
                        "label": str(p.get("label") or "").strip(),
                        "value": p.get("value"),
                        "unit": p.get("unit")})
    return title, out


def resolve_unnamed_charts(crops, run_async_fn, llm_client, model_key,
                           page_image_fn, on_resolved=None, audit=None,
                           run_id=None, document=None):
    """For each crop whose claims were UNNAMED (colour/Series N), do a combined
    crop+page read to get named series, and hand the resolved points to
    on_resolved(crop, title, points). Only unnamed crops are re-read; named ones
    are skipped. Returns {resolved, checked}.

    crops: list of {crop_b64, page, figure_index, el, was_unnamed(bool)}.
    """
    resolved = 0
    checked = 0
    for crop in crops:
        if not crop.get("was_unnamed"):
            continue
        checked += 1
        page = crop.get("page")
        page_b64 = None
        if page_image_fn is not None and page is not None:
            try:
                page_b64 = page_image_fn(page)
            except Exception:
                page_b64 = None
        title, pts = read_crop_with_page(
            run_async_fn, llm_client, model_key, crop.get("crop_b64"), page_b64)
        if pts and on_resolved:
            on_resolved(crop, title, pts)
            resolved += 1
        if audit is not None:
            try:
                audit.log(run_id or "", "shared_legend_combined",
                          f"{document}-p{page}-fig{crop.get('figure_index')}",
                          document=document, page=page,
                          figure_index=crop.get("figure_index"),
                          page_present=bool(page_b64), n_points=len(pts))
            except Exception:
                pass
    return {"resolved": resolved, "checked": checked}
