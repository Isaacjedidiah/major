"""Hybrid figure/table extraction: ai_parse_document first, Sonnet for hard charts.

The design (validated on real documents):
  * ai_parse_document is excellent at tables-in-images and MOST charts, and
    returns, per element, a ``type`` (text/table/figure/...), ``content``
    (NULL for figures it couldn't read), ``confidence`` (0-1), and ``bbox``.
  * So we route by the element ai_parse gives back:
      - table / text / title / ... -> use ai_parse's content directly.
      - figure with good confidence AND non-null content -> use it.
      - figure with LOW confidence OR null content -> escalate to Sonnet, which
        acts as a focused "chart digitisation engine" (structured output, no
        narrative), because those are the hard charts ai_parse couldn't do.

This keeps cost/rate-limit pressure low: only the residual hard charts reach
Sonnet, not every figure. Text and native tables stay on the existing direct
(pdfplumber) path upstream — ai_parse is used only for image content, per the
Databricks-recommended hybrid pattern.

This module is pure-logic where possible (parsing the ai_parse response,
deciding escalation) so it is unit-testable without a live endpoint; the actual
ai_parse and Sonnet calls are injected.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# Element types ai_parse returns that are NON-figure — used as-is from ai_parse.
_NON_FIGURE_TYPES = {"text", "table", "title", "caption", "section_header",
                     "page_header", "page_footer", "page_number", "footnote"}


# The focused prompt for reading a complex TABLE image that ai_parse couldn't
# reliably extract. Returns row-wise {metric, value} pairs, preserving the
# row/column meaning. Structured output only, no narrative.
TABLE_DIGITISATION_PROMPT = (
    "You are a table digitisation engine. Read this table image and return ONLY "
    "a JSON array of its data cells — no prose:\n"
    '[{"label": "<block | all row levels/entity | grouping band | sub-header>", '
    '"value": <number>, "unit": "<%/£m/etc if shown>"}]\n'
    "Rules:\n"
    "- ROWS may be NESTED across several left columns (e.g. portfolio->ty->"
    "position: 'Seg'->'ju'->'Able'); merged cells apply to every row they span. "
    "A first-column entity/organisation name (e.g. 'HTST YU') is also part of "
    "the row. Combine ALL row levels.\n"
    "- COLUMNS are often TWO levels: a grouping band (e.g. 'State', 'Aug TE') "
    "AND a per-column sub-header (e.g. 'Jun-28', 'Lit Sek'). Combine both.\n"
    "- SEPARATE table BLOCKS side by side (e.g. 'State' block, 'Pot' block) are "
    "distinct — prefix each value with its block; never mix blocks.\n"
    "- SECTION banner rows (no data) are context, not metrics.\n"
    "- Free-text columns (e.g. 'Comments') are context, not values.\n"
    "- Build each label combining EVERY coordinate, e.g. "
    "'Pot | Seg | ju | Able | Jun-28' = 809.\n"
    "- NEVER return an unnamed value or a bare number. A '-'/blank/empty cell "
    "means no value — skip it. If unreadable, OMIT it. Never invent values.\n"
    "- Output ONLY the JSON array."
)


# The focused prompt for the Sonnet escalation — a chart digitisation engine.
# Structured output only, no narrative (which also avoids the JSON-in-prose
# parsing issues), plus the grounding guardrail (omit what you can't read).
# For the firm-specific x-axis case: the SAME single call also returns each point's
# x-position and the printed x-axis labels + positions, so the geometry can
# resolve periods IN-MEMORY without a second Sonnet call (no doubled calls).
CHART_DIGITISATION_PROMPT_WITH_POS = (
    "You are a chart digitisation engine for bar, line AND dot/scatter charts. "
    "Read the chart LEFT TO RIGHT and record, via the tool:\n"
    "- chart_title: the title exactly as printed.\n"
    "- legend: the name of EVERY series.\n"
    "- slots: ONE entry per x-axis POSITION, in left-to-right ORDER (slot_index "
    "0,1,2,...). For each slot give: slot_index; label (the x-axis label printed "
    "under that position, or \"\" if that position has NO printed label); and "
    "values = a list of {series, value, value_source} for EVERY series at that "
    "position (bars, line vertices AND dots stacked/grouped at that x).\n"
    "- value: if a number is PRINTED on/beside the marker use it EXACTLY "
    "(value_source=\"printed\"); if not, ESTIMATE from the marker height against "
    "the y-axis (value_source=\"estimated\"). Never skip a marker.\n"
    "COUNTING RULES (critical):\n"
    "1. First COUNT every distinct x-axis position (bar group / point) from the "
    "LEFTMOST to the RIGHTMOST, including positions with NO printed label; put "
    "that count in total_positions.\n"
    "2. slot_index 0 is the LEFTMOST position (labelled or not); the last slot is "
    "the RIGHTMOST. Create EXACTLY total_positions slots, in order.\n"
    "3. For each slot set has_label (true/false) and label (the printed text or "
    "\"\"). Report first_label and last_label for the leftmost and rightmost "
    "positions.\n"
    "4. Do NOT skip leading or trailing positions that have no label -- they are "
    "still slots. Do NOT use pixel positions; the ORDER and COUNT are what matter.\n"
    "Output ONLY the JSON."
)


CHART_DIGITISATION_PROMPT = (
    "You are a chart digitisation engine. Read ONLY the CHARTS and GRAPHS in "
    "this image and return their data points as a JSON array \u2014 no prose:\n"
    '[{"label": "<chart title | series name from legend | category/x-axis label>", '
    '"value": <number>, "unit": "<%/\u00a3m/bps/count/etc>", '
    '"source_type": "chart"}]\n'
    "\n"
    "Every item MUST include \"source_type\". Set it to \"chart\" for a value read "
    "from a chart/graph/plot. If you (incorrectly) read something from a table "
    "set \"table\"; from text set \"text\" \u2014 but you should not be returning those "
    "at all (see the ABSOLUTE RULE below).\n"
    "*** ABSOLUTE RULE \u2014 CHARTS ONLY ***\n"
    "- Extract data ONLY from charts, graphs and plots: bar charts, line charts, "
    "pie/donut charts, radial charts, scatter/area plots \u2014 anything with bars, "
    "lines, wedges or plotted points read against an axis or legend.\n"
    "- NEVER extract from TABLES. If numbers sit in a grid of rows and columns, "
    "that is a TABLE \u2014 IGNORE it completely. Do not return any value from a table.\n"
    "- NEVER extract from TEXT, paragraphs, headings, bullet points or figures "
    "of prose. IGNORE all plain text and tabular numbers.\n"
    "- Tables and text are extracted separately by another system. Your ONLY job "
    "is the charts. If a value is not plotted in a chart, DO NOT return it.\n"
    "- If the image contains NO chart at all (only tables/text/logos), return an "
    "empty array [].\n"
    "\n"
    "Chart-reading rules:\n"
    "- The chart may COMBINE types (bars AND a line) \u2014 read BOTH series. Combo "
    "charts may have TWO y-axes; use the correct axis for each series.\n"
    "- RADIAL/circular bar charts: read each segment against its radial scale.\n"
    "- COLOUR LEGEND: bars/lines are distinguished by colour with a legend/"
    "footnote mapping colour -> series name (e.g. blue = \'DEF exposure\'). Map "
    "each coloured bar/line to its series name and put it in the label. Never "
    "report a value without its series name.\n"
    "- X-AXIS LABELS: map each bar/point to the x-axis label directly below "
    "its horizontal position. If a bar has NO label directly below it, use "
    "the previous or next position to find the best estimate rather than "
    "borrowing a neighbouring bar's label. Do NOT shift labels between bars.\n"
    "- ALWAYS prefix EVERY series label with the chart TITLE/heading, "
    "for EVERY value, without exception (e.g. 'Overlaid Limit Jan 2025'). "
    "Do not omit the title for any series.\n"
    "- Read every value across ALL series against the axis scale; estimate "
    "between gridlines if needed.\n"
    "- CRITICAL \u2014 read the DATA, not the axis. Do NOT return the numbers printed "
    "on the y-axis scale / gridlines (e.g. 0, 2000, 4000, 6000). Those are the "
    "SCALE, not data. For each BAR, estimate the value at the TOP of the bar by "
    "its height against the scale; for each LINE point, estimate its value by "
    "its vertical position. If a bar top sits between 4000 and 6000 nearer 6000, "
    "return about 5500 \u2014 its actual height \u2014 NOT 4000 or 6000.\n"
    "- If a data label is printed on or above a bar/point, use that exact number "
    "instead of estimating.\n"
    "- If you cannot read a value, OMIT it. Never invent values.\n"
    "- Output ONLY the JSON array."
)


@dataclass
class HybridConfig:
    """Tuning for the hybrid gate."""
    figure_confidence_threshold: float = 0.75  # below -> escalate to Sonnet
    escalate_null_content: bool = True         # figure with null content -> Sonnet
    table_confidence_threshold: float = 0.6    # below -> table escalates to Sonnet image
    escalate_low_conf_tables: bool = True      # gate tables too (image-tables can be unreliable)


@dataclass
class ParsedElement:
    """One normalised element out of ai_parse_document."""
    el_type: str
    content: Optional[str]
    confidence: Optional[float]
    page: Optional[int]
    bbox: Optional[list]
    description: Optional[str] = None


@dataclass
class HybridResult:
    """Outcome of hybrid extraction over one document."""
    used_ai_parse: list = field(default_factory=list)   # elements taken from ai_parse
    escalated: list = field(default_factory=list)       # elements sent to Sonnet
    sonnet_points: list = field(default_factory=list)    # data points Sonnet returned


def parse_ai_parse_response(resp: Any) -> list[ParsedElement]:
    """Normalise an ai_parse_document response into a flat list of ParsedElement.

    Accepts what ai_parse_document actually returns across surfaces:
      * a Spark VARIANT (VariantVal) — converted via its JSON form;
      * a JSON string;
      * a dict / Row.
    Reads document.elements, pulling type/content/confidence/bbox/description.
    Returns [] on anything unreadable.
    """
    if resp is None:
        return []
    data = resp
    # Spark VARIANT (VariantVal) -> get its JSON, then load. VariantVal exposes
    # toJson()/to_json() depending on version; fall back to str().
    if type(resp).__name__ == "VariantVal":
        try:
            js = None
            for meth in ("toJson", "to_json"):
                if hasattr(resp, meth):
                    js = getattr(resp, meth)()
                    break
            if js is None:
                js = str(resp)
            data = json.loads(js)
        except (ValueError, TypeError):
            return []
    elif isinstance(resp, str):
        try:
            data = json.loads(resp)
        except (ValueError, TypeError):
            return []
    elif hasattr(resp, "asDict"):
        try:
            data = resp.asDict(recursive=True)
        except Exception:
            return []
    if not isinstance(data, dict):
        return []
    doc = data.get("document") or {}
    elements = doc.get("elements") or []
    out: list[ParsedElement] = []
    for e in elements:
        if not isinstance(e, dict):
            continue
        page = None
        # SHARPENED PATH: each sharpened page is parsed as a standalone single-page
        # image, so ai_parse's bbox page_id is always 0. The real page number is
        # carried on the element as _sharpen_page (already 1-based from the file
        # name). Use it when present; else fall back to the bbox page_id (+1).
        _sp = e.get("_sharpen_page")
        if isinstance(_sp, int):
            page = _sp
        else:
            bbox = e.get("bbox")
            try:
                if bbox and isinstance(bbox, list) and isinstance(bbox[0], dict):
                    _pid = bbox[0].get("page_id")
                    page = (_pid + 1) if isinstance(_pid, int) else _pid
            except Exception:
                page = None
        bbox = e.get("bbox")
        _pe = ParsedElement(
            el_type=str(e.get("type", "")).lower(),
            content=e.get("content"),
            confidence=e.get("confidence"),
            page=page,
            bbox=bbox,
            description=e.get("description"),
        )
        # carry the sharpened page image path (if any) so the crop fn uses it
        _si = e.get("_sharpen_img")
        if _si:
            try:
                setattr(_pe, "sharpen_img", _si)
            except Exception:
                pass
        out.append(_pe)
    return out


def needs_sonnet_escalation(el: ParsedElement, cfg: HybridConfig) -> bool:
    """Decide whether a figure element should be escalated to Sonnet.

    Only figures are candidates. A figure escalates when ai_parse is not
    confident enough OR returned no content (both mean "ai_parse couldn't
    digitise this chart"). Non-figures never escalate — ai_parse handles them.
    """
    if el.el_type != "figure":
        return False
    if cfg.escalate_null_content and not (el.content or "").strip():
        return True
    if el.confidence is not None and el.confidence < cfg.figure_confidence_threshold:
        return True
    return False


def parse_sonnet_chart_points(raw: str) -> list[dict]:
    """Parse Sonnet's chart-digitisation output (a JSON array of points).
    Tolerant: strips prose around the array, returns [] if unparseable."""
    if not raw:
        return []
    import re
    s = raw.strip()
    # whole thing is JSON?
    try:
        val = json.loads(s)
        if isinstance(val, list):
            return [p for p in val if isinstance(p, dict)]
    except (ValueError, TypeError):
        pass
    # extract the first [...] array from prose
    m = re.search(r"\[.*\]", s, re.DOTALL)
    if m:
        try:
            val = json.loads(m.group(0))
            if isinstance(val, list):
                return [p for p in val if isinstance(p, dict)]
        except (ValueError, TypeError):
            pass
    return []


def run_hybrid_extraction(ai_parse_response: Any,
                          sonnet_read_figure: Callable[[ParsedElement], str],
                          cfg: Optional[HybridConfig] = None) -> HybridResult:
    """Orchestrate the hybrid over one document's ai_parse response.

    ``sonnet_read_figure`` is injected: given a figure ParsedElement (with its
    bbox/page so the caller can crop the image), it returns Sonnet's raw
    digitisation output. Kept as a callback so this function stays pure/testable
    and the actual Sonnet/image plumbing lives in the caller.
    """
    cfg = cfg or HybridConfig()
    result = HybridResult()
    for el in parse_ai_parse_response(ai_parse_response):
        if el.el_type in _NON_FIGURE_TYPES:
            result.used_ai_parse.append(el)          # ai_parse handles it
        elif el.el_type == "figure":
            if needs_sonnet_escalation(el, cfg):
                result.escalated.append(el)
                raw = sonnet_read_figure(el)          # hard chart -> Sonnet
                result.sonnet_points.extend(parse_sonnet_chart_points(raw))
            else:
                result.used_ai_parse.append(el)       # ai_parse got the chart
        else:
            result.used_ai_parse.append(el)           # unknown type -> keep
    return result


# ---- figure classification: logo / scale-only / data-blob / usable ----

# ratio of "round" numbers (multiples of a power of ten) above which a figure's
# content is treated as SCALE-ONLY (axis gridlines, not real data).
_SCALE_ROUND_RATIO = 0.8


def _numbers_in_text(text: str) -> list[float]:
    import re
    out = []
    for m in re.finditer(r"-?\d[\d,]*(?:\.\d+)?", text or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except (ValueError, TypeError):
            pass
    return out


def _is_round(n: float) -> bool:
    """True for 'round' gridline-style numbers: 0, 100, 1000, 2500, 5000 —
    i.e. a multiple of a sizable power of ten. Real data values are usually not
    all round."""
    n = abs(n)
    if n == 0:
        return True
    for base in (1000, 500, 100):
        if n % base == 0:
            return True
    return False


def classify_figure(el: ParsedElement) -> str:
    """Classify a figure element into how it should be handled:
      'logo'       — the description says logo/emblem, or there is no data at all
                     -> skip.
      'data'       — the description (or content) contains real numeric values
                     -> usable (structure it from the description).
      'no_content' — no description and no content -> nothing to read without an
                     image (needs vision if use_sonnet).

    IMPORTANT: for a figure, ai_parse puts the chart's data in ``description``
    (a multimodal description like "Q1 at $12M, Q2 at $15M..."); ``content`` is
    usually NULL. So we classify primarily on the DESCRIPTION, falling back to
    content.
    """
    desc = (el.description or "").strip()
    content = (el.content or "").strip()
    # For a figure, ai_parse usually puts the chart's data in DESCRIPTION and
    # leaves content NULL — but content can carry OCR'd text too. Use BOTH so
    # neither source is missed; description leads because it holds the
    # interpreted values, content is appended for any extra text.
    text = " ".join(x for x in (desc, content) if x).strip()
    low = text.lower()

    # explicit logo/emblem -> skip
    if any(w in low for w in ("logo", "emblem", "icon", "crest", "letterhead")):
        return "logo"

    if not text:
        return "no_content"                       # nothing to read without vision

    nums = _numbers_in_text(text)
    if not nums:
        return "no_content"                       # a described figure with no values

    return "data"                                 # description/content has values


def table_route(el: ParsedElement, cfg: Optional[HybridConfig] = None) -> str:
    """Decide the route for a TABLE (or text) element:
      'use_ai_parse' — ai_parse's content is reliable -> use directly.
      'sonnet_image' — low confidence (typically a complex IMAGE-table ai_parse
                       struggled with) -> send the table image to Sonnet to read.
    Only tables are image-escalated; plain text always uses ai_parse.
    """
    cfg = cfg or HybridConfig()
    if el.el_type != "table":
        return "use_ai_parse"
    if not cfg.escalate_low_conf_tables:
        return "use_ai_parse"
    if not (el.content or "").strip():
        return "sonnet_image"   # empty table content -> must read the image
    if el.confidence is not None and el.confidence < cfg.table_confidence_threshold:
        return "sonnet_image"   # low-confidence (complex image-table) -> Sonnet
    return "use_ai_parse"


def figure_route(el: ParsedElement, cfg: Optional[HybridConfig] = None) -> str:
    """Decide the ROUTE for a figure:
      'skip'          — logo / no data worth extracting.
      'sonnet_image'  — needs Sonnet to READ THE IMAGE (no_content or scale_only,
                        or low ai_parse confidence).
      'sonnet_text'   — ai_parse got real data as a blob; Sonnet STRUCTURES the
                        text (cheap, no image) into labelled pairs.
      'use_ai_parse'  — ai_parse's content is already usable as-is.
    """
    cfg = cfg or HybridConfig()
    kind = classify_figure(el)
    if kind == "logo":
        return "skip"
    if kind in ("no_content", "scale_only"):
        return "sonnet_image"
    # kind == "data": trust it if confident, else structure the text
    if el.confidence is not None and el.confidence < cfg.figure_confidence_threshold:
        return "sonnet_image"
    return "sonnet_text"


# ---- mapping ai_parse elements + Sonnet points into Claim objects ----

def ai_parse_table_to_claims(el: ParsedElement, filename: str,
                             team=None, report_type=None, entity_ref=None,
                             claim_cls=None) -> list:
    """A table/text element from ai_parse becomes claim(s). Tables keep their
    markdown content as a single TABLE-derived claim carrying the content;
    downstream metric parsing runs over it as with any table."""
    if claim_cls is None:
        from ..shared.schema import Claim as claim_cls
    content = (el.content or "").strip()
    if not content:
        return []
    eid = f"{filename}-p{el.page}-aiparse{el.el_type}"
    return [claim_cls(
        field_name=el.el_type, canonical_metric=None, value=content,
        source_element_id=eid, entity_ref=entity_ref, page=el.page,
        confidence=el.confidence or 1.0, model_used="ai_parse_document")]


def _build_chart_claims_with_geometry(chart_title, slots, el, filename,
                                      entity_ref, firm_xaxis_cfg, title=None):
    """ORDER-BASED chart claims. slots is an ordered list of
    {index, label, values:[{series,value,unit}]}. The geometry fills each slot's
    PERIOD by ORDER (resolve_periods_by_order) -- anchored on which slot carries a
    printed label, NOT on pixel positions (which Sonnet reads inconsistently). The
    LLM supplied the values (printed or estimated); the geometry supplies the
    periods. canonical_metric = <title> | <series> | <period>. All llm_estimated."""
    from .firm_xaxis import resolve_periods_by_order
    try:
        from ..shared.schema import Claim as _Claim
    except Exception:
        _Claim = None

    # resolve a period per slot_index, by ORDER
    slot_period, status = ({}, "normal")
    if any(s.get("label") for s in slots):
        try:
            slot_period, status = resolve_periods_by_order(
                [{"index": s["index"], "label": s.get("label", "")} for s in slots],
                monthly_step=firm_xaxis_cfg.get("monthly_step", 1))
        except Exception:
            slot_period, status = ({}, "normal")

    # if Sonnet's slot count disagreed with its own total_positions, the read
    # dropped/added a bar -> we cannot trust the period fill -> flag.
    if any(s.get("_count_mismatch") for s in slots):
        status = "flag"

    prefix = (chart_title or "").strip() or (title or "").strip()
    out = []
    eid = f"{filename}-p{getattr(el, 'page', None)}-fig"
    for s in slots:
        period = slot_period.get(s["index"])
        period = None if period == "FLAG_REVIEW" else period
        for v in s.get("values", []):
            series = str(v.get("series", "") or "").strip()
            value = v.get("value")
            if value is None or str(value).strip() == "":
                continue
            name = " | ".join(x for x in (prefix, series, period) if x) or series or "value"
            try:
                from .ai_parse_structurer import _split_currency_unit as _scu
                _cur, _un = _scu(v.get("unit"))
            except Exception:
                _cur, _un = (None, v.get("unit"))
            claim_kwargs = dict(
                field_name=series or "value", canonical_metric=name,
                value=str(value), unit=_un, currency=_cur, element_type="figure",
                source_element_id=eid, page=getattr(el, "page", None),
                entity_ref=entity_ref, confidence=0.75,
                model_used="sonnet_chart_order_geometry")
            if period:
                claim_kwargs["as_at_date"] = period
            try:
                from ..shared.schema import CitationTier
                claim_kwargs["citation_tier"] = CitationTier.LLM_ESTIMATED
            except Exception:
                pass
            if _Claim is not None:
                out.append(_Claim(**claim_kwargs))
    return out

def _sonnet_read_image_with_pos(run_async_fn, llm_client, model_key, content, img_b64):
    """ONE Sonnet call that returns chart points (value + x_center) AND the printed
    x-axis labels (text + x_center), for the firm-specific geometry. Returns
    (points, xlabels). No second call. Sends the ai_parse `content` (description)
    like the normal read, so context is preserved."""
    import json as _json, re as _re
    # CUSTOM schema matching the WITH_POS prompt (points + xlabels) -- NOT the
    # normal claims schema, so x-axis LABELS never leak into the claims/gold.
    _POS_SCHEMA = {
        "name": "record_chart_by_order",
        "description": "Record a chart's title, legend and its x-axis positions "
                       "(slots) in left-to-right order; each slot has its printed "
                       "label (or empty) and the values of every series there.",
        "parameters": {
            "type": "object",
            "properties": {
                "chart_title": {"type": "string"},
                "legend": {"type": "array", "items": {"type": "string"}},
                "total_positions": {"type": "integer",
                    "description": "The TOTAL number of distinct x-axis positions "
                                   "(bar groups / points) in the image, counting "
                                   "from the leftmost to the rightmost, including "
                                   "ones with NO printed label. slots MUST have "
                                   "exactly this many entries."},
                "first_label": {"type": "string",
                    "description": "The printed x-axis label of the LEFTMOST "
                                   "position, or empty if it has none."},
                "last_label": {"type": "string",
                    "description": "The printed x-axis label of the RIGHTMOST "
                                   "position, or empty if it has none."},
                "slots": {
                    "type": "array",
                    "description": "One per x-axis position, left-to-right order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "slot_index": {"type": "integer",
                                           "description": "0,1,2,... left to right; "
                                                          "0 is the LEFTMOST position."},
                            "has_label": {"type": "boolean",
                                          "description": "true if this position has a "
                                                         "printed x-axis label."},
                            "label": {"type": "string",
                                      "description": "x-axis label printed under this "
                                                     "position, or empty if none."},
                            "values": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "series": {"type": "string"},
                                        "value": {"type": "string"},
                                        "value_source": {"type": "string",
                                                         "enum": ["printed", "estimated"]},
                                    },
                                    "required": ["series", "value"],
                                },
                            },
                        },
                        "required": ["slot_index", "has_label", "label", "values"],
                    },
                },
            },
            "required": ["chart_title", "legend", "total_positions",
                         "first_label", "last_label", "slots"],
        },
    }
    try:
        resp = run_async_fn(llm_client.extract(
            model_key, CHART_DIGITISATION_PROMPT_WITH_POS, content or "",
            _POS_SCHEMA, image_base64=img_b64))
    except Exception:
        return [], []
    ta = getattr(resp, "tool_arguments", None)
    data = ta if isinstance(ta, dict) else None
    if data is None:
        txt = getattr(resp, "text", "") or ""
        m = _re.search(r"\{.*\}", txt, _re.S)
        try:
            data = _json.loads(m.group(0)) if m else {}
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    title = str(data.get("chart_title") or data.get("title") or "").strip()
    total_positions = data.get("total_positions")
    raw_slots = data.get("slots") or []
    # normalise slots -> [{index, label, values:[{series,value,unit}]}] in order.
    slots = []
    for i, s in enumerate(raw_slots if isinstance(raw_slots, list) else []):
        if not isinstance(s, dict):
            continue
        idx = s.get("slot_index")
        if idx is None:
            idx = i
        vals = []
        for v in (s.get("values") or []):
            if isinstance(v, dict) and v.get("value") not in (None, ""):
                vals.append({"series": str(v.get("series", "") or "").strip(),
                             "value": v.get("value"),
                             "unit": v.get("unit")})
        slots.append({"index": int(idx), "label": str(s.get("label", "") or "").strip(),
                      "values": vals})
    # CHECKSUM: if Sonnet's own total_positions disagrees with the slots it
    # returned, it dropped/added a position -> mark so the builder can flag.
    if isinstance(total_positions, int) and total_positions != len(slots):
        for s in slots:
            s["_count_mismatch"] = True
    return title, slots


def sonnet_points_to_claims(points: list[dict], el: ParsedElement, filename: str,
                            entity_ref=None, claim_cls=None) -> list:
    """Sonnet's chart claims (from EXTRACTION_TOOL_SCHEMA -> tool_arguments.claims,
    each carrying field_name / value / unit / scale / confidence / as_at_date)
    become one Claim each, tagged LLM_ESTIMATED (read by a vision model from the
    rendered figure image — estimates, not deterministic parses). Tolerant of the
    older {label, value} shape too."""
    if claim_cls is None:
        from ..shared.schema import Claim as claim_cls
    try:
        from ..shared.schema import CitationTier
        tier = CitationTier.LLM_ESTIMATED
    except Exception:
        tier = None
    eid = f"{filename}-p{el.page}-fig"
    out = []
    for p in points:
        if not isinstance(p, dict):
            continue
        # ENFORCE charts-only via source_type WHEN Sonnet provides it. Sonnet
        # does not always include source_type, so a MISSING tag defaults to
        # "chart" (kept) — being strict here dropped every value, because Sonnet
        # wasn't tagging at all. We drop only values Sonnet EXPLICITLY tags as
        # table/text.
        stype = str(p.get("source_type") or "chart").strip().lower()
        if stype not in ("chart", "graph", "plot", ""):
            continue
        name = str(p.get("field_name") or p.get("label") or "").strip()
        val = p.get("value")
        if val is None or not name:
            continue
        conf = p.get("confidence")
        try:
            conf = float(conf) if conf is not None else 0.7
        except (TypeError, ValueError):
            conf = 0.7
        try:
            from .ai_parse_structurer import _split_currency_unit as _scu
            _cur, _un = _scu(p.get("unit"))
        except Exception:
            _cur, _un = (None, (str(p.get("unit")) if p.get("unit") else None))
        out.append(claim_cls(
            field_name=name,
            canonical_metric=name,
            value=str(val),
            unit=_un,
            currency=_cur,
            scale=str(p.get("scale")) if p.get("scale") else None,
            as_at_date=str(p.get("as_at_date")) if p.get("as_at_date") else None,
            element_type="figure",        # the type column: chart values are figures
            source_element_id=eid, entity_ref=entity_ref, page=el.page,
            confidence=conf, citation_tier=tier,
            model_used="sonnet_chart_digitiser"))
    return out


def _sonnet_read_image(run_async_fn, llm_client, model_key, prompt, content, img_b64):
    """Call the vision extract() correctly — it needs the EXTRACTION_TOOL_SCHEMA
    (forced tool-use) and returns structured data in ``tool_arguments``. Returns
    a list of {label, value, unit} points parsed from either the tool arguments
    or the text, tolerant of both shapes."""
    try:
        from ..shared.llm_client import EXTRACTION_TOOL_SCHEMA
    except Exception:
        EXTRACTION_TOOL_SCHEMA = {"name": "extract"}
    resp = run_async_fn(llm_client.extract(
        model_key, prompt, content or "", EXTRACTION_TOOL_SCHEMA,
        image_base64=img_b64))
    # prefer structured tool_arguments; fall back to text JSON.
    ta = getattr(resp, "tool_arguments", None)
    if isinstance(ta, dict):
        # the schema records results under 'claims'; be tolerant of a few names
        for key in ("claims", "metrics", "points", "data", "values"):
            if isinstance(ta.get(key), list):
                return [p for p in ta[key] if isinstance(p, dict)]
        # or a flat list of fields -> wrap as one
        if ta:
            return [ta]
    return parse_sonnet_chart_points(getattr(resp, "text", "") or "")


def hybrid_extract_document(ai_parse_response: Any, filename: str,
                            run_async_fn: Callable = None,
                            llm_client: Any = None,
                            crop_figure_image: Optional[Callable] = None,
                            team=None, report_type=None, entity_ref=None,
                            cfg: Optional[HybridConfig] = None,
                            vision_model_key: str = "tier2",
                            use_sonnet: bool = False,
                            firm_xaxis_cfg=None,
                            missed_root=None, figure_index=0) -> list:
    """END-TO-END extraction for one document -> Claim objects for storage.

    ``use_sonnet`` controls the cost/quality trade-off:
      * False (default) — EVERYTHING is structured by RULE (no LLM cost):
        HTML tables -> composite-named claims, text-embedded numbers -> claims,
        figures -> ai_parse's own content/description parsed for values. This is
        the cheap path (ai_parse DBUs only). Use this first and inspect quality.
      * True — the hybrid: rule-based tables/text, but figures (and low-quality
        tables) escalate to Sonnet for vision reading. Flip this on only if the
        rule-based figure extraction proves inadequate.

    When use_sonnet=True, run_async_fn + llm_client must be provided.
    """
    cfg = cfg or HybridConfig()
    claims: list = []
    elements = parse_ai_parse_response(ai_parse_response)

    from .ai_parse_structurer import (html_table_to_claims, numbers_from_text,
                                      chart_description_to_claims)

    # link a nearby title (a preceding title/heading element) to each element,
    # in document order, so composite names carry it.
    current_title = None
    current_section = None
    current_page = None

    # layout-marker element types carry no metric data -> always skipped.
    _SKIP_TYPES = {"page_header", "page_footer", "page_number"}

    for el in elements:
        etype = el.el_type
        content = el.content or ""
        clean_content = re.sub(r"<[^>]+>", "", content).strip()

        # PAGE BOUNDARY: a heading only governs its own page. When the page
        # changes, clear the carried title/section so a heading on page 1 does
        # not wrongly prefix metrics on page 2.
        if el.page is not None and el.page != current_page:
            current_page = el.page
            current_title = None
            current_section = None

        # layout noise: page headers/footers/numbers -> skip entirely.
        if etype in _SKIP_TYPES:
            continue

        # title -> document-level prefix; section_header -> section prefix.
        if etype == "title":
            if clean_content:
                current_title = clean_content
            continue
        if etype == "section_header":
            if clean_content:
                current_section = clean_content
            continue
        # caption -> context for the nearby figure/table; also scan it for
        # numbers (captions sometimes state a figure's key value).
        if etype == "caption":
            claims.extend(numbers_from_text(
                clean_content, filename, page=el.page,
                title=_prefix(current_title, current_section),
                team=team, report_type=report_type, entity_ref=entity_ref))
            continue
        # footnote -> often carries meaningful caveats AND numbers (e.g.
        # "*restated to £4.2m"); extract any numeric values from it.
        if etype == "footnote":
            claims.extend(numbers_from_text(
                clean_content, filename, page=el.page,
                title=_prefix(current_title, current_section, suffix="footnote"),
                team=team, report_type=report_type, entity_ref=entity_ref))
            continue

        if etype == "table":
            # TABLES ALWAYS use the rule-based path — ai_parse structures tables
            # very well as HTML, so they never go to Sonnet (by design). Parse
            # the HTML; if that yields nothing, fall back to ai_parse's own
            # content — but never vision.
            rule_claims = html_table_to_claims(
                content, filename, page=el.page,
                title=_prefix(current_title, current_section),
                team=team, report_type=report_type, entity_ref=entity_ref)
            if rule_claims:
                claims.extend(rule_claims)
                continue
            claims.extend(ai_parse_table_to_claims(
                el, filename, team, report_type, entity_ref))
            continue

        if etype == "text":
            claims.extend(numbers_from_text(
                clean_content, filename, page=el.page,
                title=_prefix(current_title, current_section),
                team=team, report_type=report_type, entity_ref=entity_ref))
            continue

        if etype == "figure":
            kind = classify_figure(el)
            if kind == "logo":
                # a figure classified as a logo is skipped. If it might actually
                # be a chart (misclassified), record a miss so a human can check.
                if missed_root:
                    try:
                        from .missed_crops import save_missed_meta
                        save_missed_meta(missed_root, filename,
                                         getattr(el, "page", None),
                                         getattr(el, "figure_index", figure_index),
                                         reason="classified_logo_skipped")
                    except Exception:
                        pass
                continue
            # figure content is OFTEN NULL, but ai_parse's DESCRIPTION is
            # multimodal and typically contains the chart's actual values.
            # Combine description + any content and parse for values.
            fig_text = " ".join(x for x in (el.description, content) if x).strip()
            desc_claims = chart_description_to_claims(
                fig_text, filename, page=el.page,
                title=_prefix(current_title, current_section),
                team=team, report_type=report_type, entity_ref=entity_ref)
            if desc_claims and not use_sonnet:
                claims.extend(desc_claims)
                continue
            if not use_sonnet:
                claims.extend(desc_claims)
                continue
            if crop_figure_image is None:
                claims.extend(desc_claims)
                continue
            # SEQUENTIAL Sonnet read — one figure at a time. This is the version
            # that produces NO duplicates (concurrent dispatch introduced
            # duplicates, so we keep the sequential, correct behaviour).
            try:
                img_b64 = crop_figure_image(el)
                if not img_b64:
                    # crop failed -> no image to read. Record a metadata-only
                    # miss so the figure is visible for human review (the figure
                    # existed but produced no chart claims).
                    if missed_root:
                        try:
                            from .missed_crops import save_missed_meta
                            save_missed_meta(missed_root, filename,
                                             getattr(el, "page", None),
                                             getattr(el, "figure_index", figure_index),
                                             reason="crop_failed")
                        except Exception:
                            pass
                    claims.extend(desc_claims)
                    continue
                # For the FIRM-SPECIFIC x-axis case, one call returns values AND
                # positions; the geometry then resolves periods IN-MEMORY (no 2nd
                # Sonnet call -> no doubled calls / throttling). Otherwise the
                # normal single read.
                if firm_xaxis_cfg:
                    chart_title, chart_slots = _sonnet_read_image_with_pos(
                        run_async_fn, llm_client, vision_model_key, content, img_b64)
                    if chart_slots:
                        fig_claims = _build_chart_claims_with_geometry(
                            chart_title, chart_slots, el, filename, entity_ref,
                            firm_xaxis_cfg,
                            title=_prefix(current_title, current_section))
                        claims.extend(fig_claims)
                    else:
                        if missed_root and img_b64:
                            try:
                                from .missed_crops import save_missed_crop
                                save_missed_crop(img_b64, missed_root, filename,
                                                 getattr(el, "page", None),
                                                 getattr(el, "figure_index", figure_index),
                                                 reason="firm_xaxis_empty")
                            except Exception:
                                pass
                        claims.extend(desc_claims)
                else:
                    # RETRY empty reads (Sonnet is non-deterministic: a good crop
                    # can return nothing on a given run; a retry usually succeeds).
                    def _read_once():
                        return _sonnet_read_image(
                            run_async_fn, llm_client, vision_model_key,
                            CHART_DIGITISATION_PROMPT, content, img_b64)
                    try:
                        from .missed_crops import read_chart_with_retry, save_missed_crop
                        pts, _att = read_chart_with_retry(_read_once)
                    except Exception:
                        pts = _read_once()
                    if pts:
                        claims.extend(sonnet_points_to_claims(pts, el, filename, entity_ref))
                    else:
                        # STILL empty after retries -> a MISS. Save the crop to the
                        # recovery folder (if configured) so it can be re-run, and
                        # keep the description fallback so nothing is lost now.
                        if missed_root and img_b64:
                            try:
                                from .missed_crops import save_missed_crop
                                save_missed_crop(img_b64, missed_root, filename,
                                                 getattr(el, "page", None),
                                                 getattr(el, "figure_index", figure_index),
                                                 reason="empty_read_after_retry")
                            except Exception:
                                pass
                        claims.extend(desc_claims)   # keep desc fallback
            except Exception:
                # an error during the chart read -> record a miss so the figure
                # is not silently lost, then keep the description fallback.
                if missed_root:
                    try:
                        from .missed_crops import save_missed_crop, save_missed_meta
                        if "img_b64" in dir() and img_b64:
                            save_missed_crop(img_b64, missed_root, filename,
                                             getattr(el, "page", None),
                                             getattr(el, "figure_index", figure_index),
                                             reason="chart_read_error")
                        else:
                            save_missed_meta(missed_root, filename,
                                             getattr(el, "page", None),
                                             getattr(el, "figure_index", figure_index),
                                             reason="chart_read_error")
                    except Exception:
                        pass
                claims.extend(desc_claims)       # never lose the description fallback
            continue

        # any unrecognised type with content -> try number extraction.
        if clean_content:
            claims.extend(numbers_from_text(
                clean_content, filename, page=el.page,
                title=_prefix(current_title, current_section),
                team=team, report_type=report_type, entity_ref=entity_ref))

    # No value-based de-duplication: the chart prompt instructs Sonnet to read
    # CHARTS ONLY (never tables/text), so its output does not overlap with the
    # rule-based table/text claims. Separation is by SOURCE (Sonnet = charts,
    # rule-based = tables/text), which avoids the value-collision risk of
    # comparing raw values.
    # DROP the ai_parse text echoes: chart_description_to_claims turns ai_parse's
    # figure DESCRIPTION/CONTENT text into claims tagged (llm_estimated + text).
    # When Sonnet reads the same figure from the IMAGE it produces the real
    # values tagged (llm_estimated + figure). So a claim that is llm_estimated
    # AND element_type='text' is the ai_parse text echo — a duplicate of what
    # Sonnet read from the image — and is dropped. Rule-based (parsed) text and
    # real chart (figure) claims are kept.
    # NOTE: ai_parse's figure TEXT also becomes claims (llm_estimated + text),
    # which duplicate the chart values Sonnet reads from the image (llm_estimated
    # + figure). These echoes are removed by a deterministic SQL clean in the
    # notebook (DELETE WHERE citation_tier='llm_estimated' AND element_type='text')
    # rather than in-memory here — the SQL approach is observable and avoids the
    # ordering issues an in-memory drop introduced.
    # Figures are read SEQUENTIALLY inline in the loop above (one at a time).
    # A concurrent dispatch was trialled for speed but produced DUPLICATES, so
    # the correct, sequential behaviour is kept.
    #
    # NOTE: value-based dedup of Sonnet-vs-rule claims is NOT applied — a value
    # can legitimately appear in BOTH a table AND a chart (the same figure shown
    # two ways), so dropping the chart copy would lose genuine data. Table
    # mis-reads must be handled another way, not by value collision.
    return claims


def _dedup_sonnet_vs_rules(claims: list) -> list:
    """When figures are read from the WHOLE PAGE image, Sonnet may re-read values
    already captured from the page's tables/text by the rule-based path. Drop a
    Sonnet chart claim (model_used='sonnet_chart_digitiser') when a rule-based
    claim on the SAME page already has the same value — keeping the deterministic
    (parsed) one. Rule-based claims are always kept."""
    def _norm_val(v):
        import re as _re
        return _re.sub(r"[,£$€%\s]", "", str(v or "").lower())

    # index rule-based values per page
    rule_vals = {}
    for c in claims:
        if getattr(c, "model_used", "") != "sonnet_chart_digitiser":
            rule_vals.setdefault(getattr(c, "page", None), set()).add(
                _norm_val(getattr(c, "value", "")))

    out = []
    for c in claims:
        if getattr(c, "model_used", "") == "sonnet_chart_digitiser":
            page = getattr(c, "page", None)
            if _norm_val(getattr(c, "value", "")) in rule_vals.get(page, set()):
                continue    # duplicate of a rule-based value on this page -> drop
        out.append(c)
    return out


def _prefix(title, section, suffix=None):
    """Build the context prefix from the current title + section (+ optional
    marker like 'footnote')."""
    parts = [x for x in (title, section) if x]
    if suffix:
        parts.append(suffix)
    return " | ".join(parts) if parts else None
