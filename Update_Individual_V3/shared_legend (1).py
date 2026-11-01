"""SHARED LEGEND resolution — a SEPARATE feature from the chart retry loop.

Problem: ~half of charts have their legend ATTACHED (series readable in the crop)
and half have a SHARED legend sitting apart from the chart (so a crop read can't
name the series). This module resolves series names for the shared-legend case,
WITHOUT affecting charts whose legend is attached.

Agnostic design — it does not assume which charts are shared. It is called per
page; it reads the page legend ONCE (colour -> series) and, for each chart's
points, fills in the series name BY COLOUR only where the point has no series of
its own. Points that already have a series (attached legend) are left untouched.

This is deliberately independent of chart_loop (skips) and of the inline read.
The caller runs it after extraction when it wants shared-legend resolution.

Make-or-break: the vision model must report COLOURS consistently in both the
legend read and the per-point reads for the join to work. Where it doesn't, the
point keeps its colour as the series and is flagged (series_source=colour_only),
so nothing is silently mislabelled.
"""
from __future__ import annotations

import json
import re
from typing import Optional

LEGEND_PROMPT = (
    "Look ONLY at the LEGEND(S) on this page image. Return the mapping of each "
    "series' COLOUR to its NAME as JSON, ignoring all chart data: "
    "{\"legend\": [{\"colour\": \"<colour word, e.g. blue>\", "
    "\"series\": \"<series name, e.g. Revenue>\"}]}. If there is no legend, "
    "return {\"legend\": []}."
)


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


def read_page_legend(run_async_fn, llm_client, model_key, page_img_b64):
    """Read the page ONCE for the legend colour->series map. {colour_lower: series}.
    Empty dict if no legend / on error."""
    try:
        from ..shared.llm_client import EXTRACTION_TOOL_SCHEMA
    except Exception:
        EXTRACTION_TOOL_SCHEMA = {"name": "extract"}
    try:
        resp = run_async_fn(llm_client.extract(
            model_key, LEGEND_PROMPT, "", EXTRACTION_TOOL_SCHEMA,
            image_base64=page_img_b64))
    except Exception:
        return {}
    data = _parse_json(resp)
    out = {}
    for item in (data.get("legend") or []):
        if isinstance(item, dict):
            col = str(item.get("colour") or item.get("color") or "").strip().lower()
            ser = str(item.get("series") or "").strip()
            if col and ser:
                out[col] = ser
    return out


# ===========================================================================
# COLOUR-BASED series augmentation (post-step; no chart re-read, no duplicates).
#
# When a chart has no legend, the chart prompt already names each series with its
# COLOUR (e.g. "Revenue - teal bar", "Revenue - Teal segment"). This step:
#   1. reads the page legend ONCE (colour -> series),
#   2. finds claims whose name contains a known colour word (via matplotlib's
#      CSS4 colour names -- robust to the LLM's varying phrasing, unlike a regex
#      pattern),
#   3. if that colour IS in the page legend -> AUGMENTS the name with the legend's
#      series (keeps the original, appends the resolved name),
#   4. if the colour is NOT in the legend (no legend exists, or colour unmapped)
#      -> MAINTAINS the name as-is (the genuine no-legend case).
# Claims with no colour word are never touched. Nothing is re-read.
# ===========================================================================
def _colour_words():
    try:
        from matplotlib.colors import CSS4_COLORS
        return set(CSS4_COLORS.keys())
    except Exception:
        # minimal fallback if matplotlib is unavailable
        return {"red", "green", "blue", "orange", "purple", "teal", "navy",
                "maroon", "cyan", "magenta", "yellow", "pink", "brown", "grey",
                "gray", "black", "olive", "lime", "indigo", "violet", "gold"}


_COLOUR_SET = None


def _find_colour(text):
    """Return the first known colour word in text (lowercased), else None.
    Uses CSS4 colour names so it is robust to the LLM's phrasing."""
    global _COLOUR_SET
    if _COLOUR_SET is None:
        _COLOUR_SET = _colour_words()
    if not text:
        return None
    s = str(text)
    # the chart prompt appends the colour as a trailing tag after " - "
    # ("Revenue - teal bar"). REQUIRE that separator: a name with no " - " has no
    # colour tag, so a colour word inside the real name ("Green Energy Revenue")
    # is NOT treated as a tag. Only the tail after the last " - " is searched.
    if " - " not in s:
        return None
    tail = s.rsplit(" - ", 1)[-1]
    for w in re.findall(r"[a-zA-Z]+", tail.lower()):
        if w in _COLOUR_SET:
            return w
    return None


def augment_claims_with_legend(claims, legend_map, set_field=None):
    """For each claim whose name contains a colour word that IS in legend_map,
    AUGMENT the name with the legend's series (keep original + append resolved).
    Colours not in the legend are left untouched (genuine no-legend case).
    Claims with no colour word are untouched. Operates on existing claims only --
    no chart is re-read, so no duplicates are created.

    legend_map : {colour_lower: series}
    set_field  : optional callable (claim, new_name) to write the augmented name;
                 by default sets .canonical_metric (and .field_name if present).
    Returns the count of claims augmented.
    """
    if not legend_map:
        return 0
    n = 0
    for c in claims:
        name = getattr(c, "canonical_metric", None) or getattr(c, "field_name", None) or ""
        col = _find_colour(name)
        if not col or col not in legend_map:
            continue                      # no colour, or colour not in legend -> leave as-is
        series = legend_map[col]
        if not series or series.lower() in name.lower():
            continue                      # already carries the resolved name
        new_name = f"{name} | {series}"   # AUGMENT: append resolved, matching the
                                         # pipeline's normal ' | ' separator
        if set_field:
            set_field(c, new_name)
        else:
            try:
                setattr(c, "canonical_metric", new_name)
            except Exception:
                pass
        n += 1
    return n


def augment_page_claims(claims, run_async_fn, llm_client, model_key, page_img_b64):
    """Convenience: read the page legend ONCE, then augment the given claims by
    colour. Returns (n_augmented, legend_size). No chart re-read."""
    legend_map = read_page_legend(run_async_fn, llm_client, model_key, page_img_b64) \
        if page_img_b64 else {}
    n = augment_claims_with_legend(claims, legend_map)
    return n, len(legend_map)


def augment_claims_per_page(claims, run_async_fn, llm_client, model_key,
                            page_image_fn, audit=None, run_id=None, document=None):
    """Augment colour-tagged claims PER PAGE — each page's claims are matched ONLY
    against THAT page's legend, so a legend is never assumed across pages.

    For every page that has at least one colour-tagged claim, reads that page's
    legend ONCE (via page_image_fn(page) -> base64) and augments only that page's
    claims. Pages with no legend, or claims whose colour isn't in their page's
    legend, are left untouched. No chart is re-read -> no duplicates.

    Returns {pages_checked, augmented, legends_found}.
    """
    from collections import defaultdict
    by_page = defaultdict(list)
    for c in claims:
        name = getattr(c, "canonical_metric", None) or getattr(c, "field_name", None) or ""
        if _find_colour(name):               # only pages with colour-tagged claims
            by_page[getattr(c, "page", None)].append(c)

    pages_checked = 0
    total_aug = 0
    legends_found = 0
    for page, page_claims in by_page.items():
        pages_checked += 1
        page_b64 = None
        if page_image_fn is not None and page is not None:
            try:
                page_b64 = page_image_fn(page)
            except Exception:
                page_b64 = None
        legend_map = read_page_legend(run_async_fn, llm_client, model_key, page_b64) \
            if page_b64 else {}
        if legend_map:
            legends_found += 1
        n = augment_claims_with_legend(page_claims, legend_map)   # match within page
        total_aug += n
        if audit is not None:
            try:
                audit.log(run_id or "", "shared_legend_page", f"{document}-p{page}",
                          document=document, page=page,
                          colour_tagged=len(page_claims), augmented=n,
                          legend_size=len(legend_map))
            except Exception:
                pass
    return {"pages_checked": pages_checked, "augmented": total_aug,
            "legends_found": legends_found}
