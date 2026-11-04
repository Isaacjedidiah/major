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
    "Look at the LEGEND(S) and the chart categories on this page image. For each "
    "legend entry return: its COLOUR (a plain colour word, e.g. blue, teal), the "
    "SERIES name it maps to (e.g. 'Approved with no condition'), and the chart "
    "CATEGORY LABELS that series appears under (e.g. Revenue, Cost; the x-axis / "
    "segment labels). This lets a colour be matched to the right legend when a "
    "page has several. If there is no legend, return an empty list."
)

# a tool schema SHAPED FOR THE LEGEND (not the claims schema -- using the claims
# schema makes the model return claims and NO 'legend' array -> empty legend map).
_LEGEND_SCHEMA = {
    "name": "record_legend",
    "description": "Record each legend entry's colour and the series name it maps to.",
    "parameters": {
        "type": "object",
        "properties": {
            "legend": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "colour": {"type": "string",
                                   "description": "plain colour word, e.g. blue, teal"},
                        "series": {"type": "string",
                                   "description": "the series name for that colour"},
                        "labels": {"type": "array", "items": {"type": "string"},
                                   "description": "the chart category labels this "
                                                  "series appears under (x-axis / "
                                                  "segment labels), for disambiguation"},
                    },
                    "required": ["colour", "series"],
                },
            },
        },
        "required": ["legend"],
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


def read_page_legend(run_async_fn, llm_client, model_key, page_img_b64):
    """Read the page ONCE for the legend colour->series map. {colour_lower: series}.
    Empty dict if no legend / on error."""
    try:
        resp = run_async_fn(llm_client.extract(
            model_key, LEGEND_PROMPT, "", _LEGEND_SCHEMA,
            image_base64=page_img_b64))
    except Exception:
        return {}
    data = _parse_json(resp)
    out = []
    for item in (data.get("legend") or []):
        if isinstance(item, dict):
            col = str(item.get("colour") or item.get("color") or "").strip().lower()
            ser = str(item.get("series") or "").strip()
            labels = [str(x).strip().lower() for x in (item.get("labels") or [])
                      if str(x).strip()]
            if col and ser:
                out.append({"colour": col, "series": ser, "labels": labels})
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
    tail = s.rsplit(" - ", 1)[-1].strip()
    words = re.findall(r"[a-zA-Z]+", tail.lower())
    # the defined format is "<title> - <colour>" (optionally "<colour> bar/segment/
    # point/line/wedge"). Accept only a SHORT tail (<=3 words) whose words are all
    # colours or shape words -- so a real phrase ("quarterly view") is NOT taken
    # as a colour tag. Return the colour word found.
    _SHAPES = {"bar", "bars", "segment", "segments", "point", "points", "line",
               "lines", "wedge", "wedges", "slice", "slices", "series", "unlabeled",
               "unlabelled"}
    if not words or len(words) > 4:
        return None
    colour = None
    for w in words:
        if w in _COLOUR_SET:
            colour = w
        elif w not in _SHAPES:
            return None          # a non-colour, non-shape word -> not a colour tag
    return colour


_COLOUR_QUALIFIERS = {"dark", "light", "bright", "pale", "deep", "medium",
                      "royal", "navy"}  # modifiers that precede a base colour


def _base_colour(colour):
    """Reduce a colour name to its base colour word so the claim's simple colour
    ("navy") matches the legend's qualified colour ("dark navy"). Returns the LAST
    recognised colour word (the base), lowercased."""
    global _COLOUR_SET
    if _COLOUR_SET is None:
        _COLOUR_SET = _colour_words()
    words = [w for w in re.findall(r"[a-zA-Z]+", str(colour).lower())]
    # the base colour is the last word that is itself a known colour
    base = None
    for w in words:
        if w in _COLOUR_SET:
            base = w
    # "navy" etc. are themselves colours, so "dark navy" -> "navy"
    return base or (words[-1] if words else "")


def _normalise_legend(legend_map):
    """Key the legend by BASE colour word, so claims with simple colours match
    qualified legend colours. On conflict (same base colour -> different series),
    keep the first and drop later conflicting entries (ambiguous)."""
    out = {}
    for col, ser in (legend_map or {}).items():
        base = _base_colour(col)
        if not base:
            continue
        if base in out and out[base] != ser:
            continue           # conflict -> keep the first, skip the ambiguous dup
        out.setdefault(base, ser)
    return out


def _claim_label(name):
    """Extract the chart LABEL from a colour-tagged claim name. The claim looks
    like '<label> - <colour...>' or '<title> | <label> - <colour...>'. The label
    is the text segment just before the ' - <colour>' tag. Lowercased."""
    base = name.rsplit(" - ", 1)[0]          # drop the colour tag
    base = base.split(" | ")[-1]             # the last | segment is the label/series
    return base.strip().lower()


def _resolve(legend, base_colour, claim_label):
    """Resolve a colour-tagged claim to a series using (colour + label).
    legend: list of {colour, series, labels}. Precedence:
      1. an entry whose BASE colour matches AND whose labels include the claim
         label -> that series (disambiguates multiple legends),
      2. else the single entry whose base colour matches (if exactly one) -> its
         series,
      3. else None (ambiguous colour across legends with no label match, or no
         colour match) -> leave the claim untouched (safe).
    Never returns a bare colour as the series."""
    # candidate entries whose base colour matches
    cands = [e for e in legend if _base_colour(e.get("colour", "")) == base_colour]
    if not cands:
        return None
    # 1. label-disambiguated match
    if claim_label:
        for e in cands:
            labels = [l for l in (e.get("labels") or [])]
            if any(claim_label == l or claim_label in l or l in claim_label
                   for l in labels):
                return _clean_series(e.get("series"), base_colour)
    # 2. unambiguous single colour match
    if len(cands) == 1:
        return _clean_series(cands[0].get("series"), base_colour)
    # 3. ambiguous (colour in several legends, no label match) -> leave untouched
    return None


def _clean_series(series, base_colour):
    """Reject a 'series' that is empty or is just the colour word."""
    s = (series or "").strip()
    if not s:
        return None
    if s.lower() == base_colour or s.lower() in _colour_words():
        return None
    return s


def augment_claims_with_legend(claims, legend, set_field=None):
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
    if not legend:
        return 0
    n = 0
    for c in claims:
        name = getattr(c, "canonical_metric", None) or getattr(c, "field_name", None) or ""
        # AIRTIGHT SCOPING: only act on a claim that carries a COLOUR TAG (i.e. a
        # chart read WITHOUT its own legend). A chart WITH a legend has a clean
        # series name, no colour tag -> _find_colour returns None -> skipped.
        col = _find_colour(name)
        if not col:
            continue
        base = _base_colour(col)
        claim_label = _claim_label(name)        # the chart label on this claim
        series = _resolve(legend, base, claim_label)
        if not series:
            continue                            # colour not in legend -> leave as-is
        if not series:
            continue
        # skip meaningless mappings where the "series" is just the colour word
        # (e.g. teal->teal) -- appending the colour is not a resolution.
        if series.strip().lower() == col.strip().lower():
            continue
        if series.strip().lower() in _COLOUR_SET:
            continue                      # series is a bare colour -> not a real name
        if name.rstrip().endswith(f"| {series}"):
            continue                      # already augmented with THIS series
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
    legend = read_page_legend(run_async_fn, llm_client, model_key, page_img_b64) \
        if page_img_b64 else []
    n = augment_claims_with_legend(claims, legend)
    return n, len(legend)


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
        try:
            page_b64 = None
            if page_image_fn is not None and page is not None:
                try:
                    page_b64 = page_image_fn(page)
                except Exception:
                    page_b64 = None
            legend = read_page_legend(run_async_fn, llm_client, model_key,
                                      page_b64) if page_b64 else []
            if legend:
                legends_found += 1
            n = augment_claims_with_legend(page_claims, legend)
            total_aug += n
            if audit is not None:
                try:
                    # include the extracted legend (colour->series) so it can
                    # be inspected in the audit. Cap the serialised size.
                    import json as _json
                    _legend_str = _json.dumps(legend)[:1000]
                    audit.log(run_id or "", "shared_legend_page",
                              f"{document}-p{page}", document=document, page=page,
                              colour_tagged=len(page_claims), augmented=n,
                              legend_size=len(legend), legend=_legend_str)
                except Exception:
                    pass
        except Exception as _e:
            # one page failing must NOT abort the whole step
            if audit is not None:
                try:
                    audit.log(run_id or "", "shared_legend_page_error",
                              f"{document}-p{page}", document=document,
                              page=page, detail=str(_e))
                except Exception:
                    pass
            continue
    return {"pages_checked": pages_checked, "augmented": total_aug,
            "legends_found": legends_found}
