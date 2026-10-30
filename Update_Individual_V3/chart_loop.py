"""Chart extraction via a RETRY LOOP with a once-per-page LEGEND read (Option B).

Two problems this addresses:
  1. Intermittent SKIPS — Sonnet returns nothing for a crop on one run but reads
     it on another. We retry each crop (fresh reads) until it extracts OR a pass
     limit is hit; whatever never reads goes to human review. No self-review:
     each read is an independent fresh attempt, success judged OBJECTIVELY (did
     it produce points), never by Sonnet's opinion of its own work.
  2. SHARED LEGENDS — the legend sits apart from the charts. We read the WHOLE
     PAGE ONCE per page for ONLY the colour->series map, then join each crop's
     bars (read with their colour) to that map. The legend is read once (cheap,
     reliable), not re-associated per chart.

Flow per page:
    legend_map = read_page_legend(page_image)          # colour -> series, ONCE
    crops = [all chart crops on the page]
    for pass in 1..MAX_PASSES:
        for crop still pending:
            points = read_crop(crop)                    # chart data + colours
            if points:
                apply legend_map (colour -> series) to points
                -> emit claims, move crop to COMPLETED
            # else: leave pending for the next pass
        if none pending: break
    pending after the loop -> human_in_the_loop
"""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

MAX_PASSES = 4          # retry a crop across at most this many passes


# --- prompts / schema -----------------------------------------------------
LEGEND_PROMPT = (
    "Look ONLY at the chart LEGEND(S) on this page. Return the mapping of each "
    "series' COLOUR to its NAME, as JSON: {\"legend\": [{\"colour\": \"<colour "
    "word, e.g. blue>\", \"series\": \"<series name, e.g. Revenue>\"}]}. Ignore "
    "the chart data itself. If there is no legend, return {\"legend\": []}."
)

CROP_PROMPT = (
    "Read this ONE chart. Return every bar/point as JSON: {\"points\": "
    "[{\"series\": \"<series name IF this chart has its OWN legend, else empty>\", "
    "\"colour\": \"<the bar/point colour, e.g. blue>\", \"value\": <number "
    "printed on it, else your estimate>, \"label\": \"<x-axis label if printed, "
    "else empty>\", \"unit\": \"<%/m/bn/etc, else empty>\"}]}. If the chart has "
    "its OWN legend, give the series name directly. Always give the colour too. "
    "Return every bar/point; do not skip any."
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
    """Read the page ONCE for the legend colour->series map. Returns a dict
    {colour_lower: series}. Empty dict if no legend / on error."""
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


def read_crop_once(run_async_fn, llm_client, model_key, crop_b64):
    """One fresh read of a crop. Returns a list of point dicts (colour/value/
    label/unit) or [] if empty/error. No self-review -- a plain fresh attempt."""
    try:
        from ..shared.llm_client import EXTRACTION_TOOL_SCHEMA
    except Exception:
        EXTRACTION_TOOL_SCHEMA = {"name": "extract"}
    try:
        resp = run_async_fn(llm_client.extract(
            model_key, CROP_PROMPT, "", EXTRACTION_TOOL_SCHEMA,
            image_base64=crop_b64))
    except Exception:
        return []
    data = _parse_json(resp)
    pts = data.get("points") or data.get("claims") or []
    out = []
    for p in (pts if isinstance(pts, list) else []):
        if isinstance(p, dict) and p.get("value") not in (None, ""):
            out.append({"series": str(p.get("series") or "").strip(),
                        "colour": str(p.get("colour") or p.get("color") or "").strip().lower(),
                        "value": p.get("value"),
                        "label": str(p.get("label") or "").strip(),
                        "unit": p.get("unit")})
    return out


def apply_legend(points, legend_map):
    """Resolve each point's series name. Precedence:
      1. the series the crop read itself gave (chart has its OWN attached legend),
      2. else the page legend map keyed by colour (SHARED legend),
      3. else the colour (flagged) as a last resort.
    So charts with their own legend use it directly -- the page map is only used
    when the crop has no series of its own (shared-legend case)."""
    for p in points:
        own = (p.get("series") or "").strip()
        col = p.get("colour", "")
        if own:
            p["series"] = own
            p["series_source"] = "own_legend"
        elif col in legend_map:
            p["series"] = legend_map[col]
            p["series_source"] = "page_legend"
        else:
            p["series"] = col or "series"
            p["series_source"] = "colour_only"
    return points


def run_chart_loop(page_crops, run_async_fn, llm_client, model_key,
                   page_img_b64=None, max_passes=MAX_PASSES, on_extracted=None,
                   on_unresolved=None, audit=None, run_id=None, document=None):
    """Process one page's chart crops with the retry loop + once-read legend.

    page_crops: list of {crop_id, page, figure_index, crop_b64}.
    page_img_b64: the whole-page image (for the once-per-page legend read); may
      be None (then series come from colour only).
    on_extracted(crop_meta, points): called when a crop reads -> emit claims.
    on_unresolved(crop_meta, reason): called for crops still empty after the loop
      -> route to human_in_the_loop.
    Returns {extracted: n, unresolved: n, passes_used: n, legend_size: n}.
    """
    # 1. read the legend ONCE for the page
    legend_map = {}
    if page_img_b64:
        legend_map = read_page_legend(run_async_fn, llm_client, model_key, page_img_b64)

    pending = list(page_crops)
    extracted = 0
    passes_used = 0
    for _pass in range(1, max_passes + 1):
        if not pending:
            break
        passes_used = _pass
        still = []
        for crop in pending:
            pts = read_crop_once(run_async_fn, llm_client, model_key,
                                 crop.get("crop_b64"))
            if pts:
                pts = apply_legend(pts, legend_map)
                extracted += 1
                if on_extracted:
                    on_extracted(crop, pts)
                if audit is not None:
                    try:
                        audit.log(run_id or "", "chart_read",
                                  f"{document}-p{crop.get('page')}-fig{crop.get('figure_index')}",
                                  document=document, page=crop.get("page"),
                                  figure_index=crop.get("figure_index"),
                                  outcome="extracted", n_points=len(pts),
                                  pass_no=_pass)
                    except Exception:
                        pass
            else:
                still.append(crop)
        pending = still

    # 2. whatever never read -> unresolved (human review)
    for crop in pending:
        if on_unresolved:
            on_unresolved(crop, f"unresolved_after_{passes_used}_passes")
        if audit is not None:
            try:
                audit.log(run_id or "", "chart_read",
                          f"{document}-p{crop.get('page')}-fig{crop.get('figure_index')}",
                          document=document, page=crop.get("page"),
                          figure_index=crop.get("figure_index"),
                          outcome="unresolved", n_points=0,
                          reason=f"unresolved_after_{passes_used}_passes")
            except Exception:
                pass

    return {"extracted": extracted, "unresolved": len(pending),
            "passes_used": passes_used, "legend_size": len(legend_map)}
