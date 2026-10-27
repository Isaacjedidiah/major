"""Two-tier extraction cascade with content-aware routing and cost tracking.

Routing is now two-stage. A cheap, deterministic content check
(``shared.routing.decide_start_tier``) picks each element's STARTING tier
before any model call: known-hard content (tables, over-long or
flagged-dense elements) starts on Tier 2 (Sonnet) directly, skipping the
wasted Tier 1 pass. Everything else starts on Tier 1 (GPT-5 mini) and
escalates reactively when confidence is low. Still-low confidence after
Tier 2 flags for human review. The system flags but never rejects. Token
cost accumulates per document, and each routing decision is recorded for the
audit trail.

Fixes over the earlier version:
  * The tool schema now returns scale/netting/as_at_date, and those are
    threaded into every Claim (previously always None).
  * entity_ref is propagated from the source Element into each Claim
    (previously nothing populated it, so Gold was unattributable).
  * Malformed model rows are quarantined, not fatal (flag-never-reject).
"""
from __future__ import annotations

import asyncio

from dataclasses import dataclass, field
from typing import Optional

from ..shared.config import CONFIG
from ..shared.dictionary import canonical_metric
from ..shared.llm_client import EXTRACTION_TOOL_SCHEMA, LLMClient
from ..shared.metric_normaliser import classify_metric
from ..shared.prompts import build_prompt


# Appended to the extraction prompt for FIGURE elements (charts read as images).
# Explicit chart-reading steps improve accuracy over a generic instruction:
# the model is told to establish the axes/units first, then read each value
# against them, and to flag values it had to estimate between gridlines.
_CHART_READING_GUIDANCE = (
    "This element is a CHART image. Read it carefully.\n"
    "\n"
    "CHART TYPE — first identify the type(s). A single chart may COMBINE types "
    "(e.g. bars AND a line on the same axes — read BOTH the bar series and the "
    "line series). Types include: bar, grouped/stacked bar, line, bar+line "
    "combo, pie, and RADIAL/circular bar charts (values run around a circle "
    "rather than along a straight axis — read each segment against its radial "
    "scale).\n"
    "\n"
    "AXES & SCALE — read the axis labels and UNITS (%, £m, bps, count) and the "
    "scale (min, max, gridline interval) before reading any value. Some combo "
    "charts have TWO y-axes (e.g. bars on the left axis, the line on the right) "
    "— use the correct axis for each series.\n"
    "\n"
    "COLOUR LEGEND — bars/lines are often distinguished by COLOUR, with a legend "
    "or footnote mapping each colour to a series name (e.g. blue = 'DEF "
    "exposure', orange = 'Non-DEF'). USE THE LEGEND: map each coloured bar/line "
    "to its series name, and include that name in the label. Never report a "
    "value without the series name the colour represents.\n"
    "\n"
    "For EACH data point/bar/segment, report: <series name from legend> + "
    "<category/x-axis label> = <value> (<unit>). Read every clearly-readable "
    "value across ALL series, not just the largest.\n"
    "\n"
    "RULES: If a value falls between gridlines, estimate it from the scale and "
    "mark it approximate (e.g. '~120'). Do NOT invent values you cannot read; "
    "if a label or bar is illegible, omit it. Read numbers off the chart's own "
    "axes/scale — never fabricate."
)

# Guidance for TABLE images (a table rendered as a picture — common in scanned
# or image-heavy reports). Preserving the row/column structure is what makes the
# values usable; this is the table analogue of the chart guidance above.
_TABLE_READING_GUIDANCE = (
    "This element is a TABLE image. It may have any of: MULTI-LEVEL column "
    "headers, NESTED row labels, SECTION rows, an ENTITY column, or SEPARATE "
    "table blocks side by side. Read it carefully and name every value fully.\n"
    "\n"
    "ROWS — identify the full row identity, which may be NESTED across several "
    "left columns (e.g. portfolio -> ty -> position: 'Seg' -> 'ju' -> 'Able'). "
    "Merged cells mean a label applies to every row it spans. Also treat a "
    "first-column ENTITY/organisation name (e.g. 'HTST YU') or a code ('UK1a') "
    "as part of the row identity. Combine ALL row levels.\n"
    "\n"
    "COLUMNS — identify the full column path, often TWO levels: a grouping band "
    "spanning several columns (e.g. 'State', 'Aug TE', 'Total unweighted value') "
    "AND a per-column sub-header (e.g. 'Jun-28', 'Lit Sek', '31-Mar-26'). A "
    "value belongs to both.\n"
    "\n"
    "BLOCKS — if there are SEPARATE table blocks side by side (e.g. a 'State' "
    "block and a 'Pot' block), treat them as DISTINCT and prefix each value with "
    "its block name; never mix values across blocks.\n"
    "\n"
    "SECTION rows (shaded banners spanning the width, no data) are context, not "
    "metrics — use as a prefix, never emit as a value.\n"
    "\n"
    "For each DATA cell, build a name combining EVERY coordinate in order:\n"
    "<block> | <all row levels / entity> | <grouping band> | <sub-header>\n"
    "Example: the 'Able' position under 'Seg'->'ju', in the 'Pot' block, "
    "'Jun-28' column -> 'Pot | Seg | ju | Able | Jun-28' = 809.\n"
    "\n"
    "RULES: NEVER emit an unnamed metric or a bare value with no row/column. "
    "Include the unit (%, £m, bps). A '-' or blank/empty cell means NO value — "
    "skip it, never invent one. Free-text columns (e.g. 'Comments') are context, "
    "not numeric metrics. Read numbers exactly as shown; never fabricate."
)
from ..shared.routing import RoutingDecision, decide_start_tier
from ..shared.schema import (Claim, CitationTier, Element, Modality, ReviewTier,
                             build_claim)
from ..search.figure_preprocessor import (encode_image_base64,
                                          needs_image_read,
                                          prepare_figure_image)


@dataclass
class ExtractionResult:
    claims: list[Claim] = field(default_factory=list)
    quarantine: list[dict] = field(default_factory=list)
    total_cost_usd: float = 0.0
    # Per-element routing decisions, keyed by element_id, for the audit trail.
    routing: dict[str, RoutingDecision] = field(default_factory=dict)
    cropped_figures: int = 0

    def track_cost(self, model_key: str, in_tok: int, out_tok: int) -> None:
        spec = CONFIG.model(model_key)
        self.total_cost_usd += (in_tok / 1e6) * spec.input_cost_per_m
        self.total_cost_usd += (out_tok / 1e6) * spec.output_cost_per_m


class Extractor:
    def __init__(self, client: Optional[LLMClient] = None):
        self._client = client or LLMClient()
        self._th = CONFIG.thresholds

    async def extract(self, elements: list[Element],
                      document_bytes: Optional[bytes] = None,
                      filename: Optional[str] = None,
                      page_image_provider=None) -> ExtractionResult:
        """Extract claims from every element, concurrently.

        Each element is processed as its own task; ``asyncio.gather`` runs them
        overlapping, and the client's semaphore caps how many model calls are
        actually in flight (respecting the endpoint rate limit). This is the
        async payoff: 1000 elements no longer wait in a 1000-long line — they
        run in bounded concurrent waves.

        When ``document_bytes``, ``filename`` and ``page_image_provider`` are
        supplied, chart-like FIGURE elements are cropped-and-zoomed and read
        multimodally (see search.figure_preprocessor). Text-only extraction
        is unchanged when they are absent — crop-and-zoom is fully optional
        and degrades to text-only per element, never crashing a document.
        """
        result = ExtractionResult()
        tasks = []
        for el in elements:
            prompt = build_prompt(el.team or "", el.report_type or "")
            # Modality-specific image-reading guidance. A FIGURE image can be a
            # CHART or a TABLE-rendered-as-an-image (common in Word/PPTX), and we
            # can't reliably tell which before the model sees it — so a figure
            # gets BOTH the chart and table guidance and applies whichever fits.
            # A native/text TABLE (not an image) gets only the table guidance,
            # and only when it's complex enough to be image-read.
            if el.modality == Modality.FIGURE:
                prompt = (prompt + "\n\n" + _CHART_READING_GUIDANCE
                          + "\n\nIF THE IMAGE IS A TABLE (not a chart), instead "
                          "follow this:\n" + _TABLE_READING_GUIDANCE)
            elif el.modality == Modality.TABLE and needs_image_read(el):
                prompt = prompt + "\n\n" + _TABLE_READING_GUIDANCE
            # If a title was linked to this table/figure (a separate heading
            # element above it, e.g. 'FINANCIAL RISK'), pass it as context so the
            # metric names carry it: 'FINANCIAL RISK | <row> | AMOUNT = 500'.
            if getattr(el, "description", None) and el.modality in (
                    Modality.TABLE, Modality.FIGURE):
                prompt = (prompt + f"\n\nThis table/chart's TITLE is "
                          f"'{el.description}'. Prefix every metric name with "
                          "this title so each value carries its context.")
            tasks.append(self._extract_element(
                el, prompt, result, document_bytes, filename,
                page_image_provider))
        # return_exceptions=True so one element's failure can't cancel the
        # whole gather; a raised element is quarantined below.
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        for el, outcome in zip(elements, outcomes):
            if isinstance(outcome, Exception):
                result.quarantine.append(
                    {"kind": "element", "source_element_id": el.element_id,
                     "error": f"{type(outcome).__name__}: {outcome}",
                     # carry attribution so a recovery pass knows which team/
                     # report_type table to write recovered claims to, without
                     # needing the original submissions list.
                     "source_document": el.source_document,
                     "team": el.team, "report_type": el.report_type,
                     "entity_ref": el.entity_ref, "page": el.page})
        return result

    def _figure_image(self, el: Element, document_bytes, filename,
                      page_image_provider) -> Optional[str]:
        """Return a base64 image for an element that reads better visually — a
        chart-like FIGURE or a complex TABLE. None → text-only. Never raises.

        Two sources, in order:
          1. Directly-embedded image bytes on the element (e.g. a chart picture
             extracted straight from a .pptx/.docx) — used as-is, best quality,
             works for formats with no page renderer.
          2. Render-and-crop a page region via the provider (PDF path).
        """
        if not needs_image_read(el):
            return None
        # 1. directly-embedded image bytes (PPTX/DOCX pictures)
        if getattr(el, "image_bytes", None):
            try:
                # Normalise embedded images (PPTX/Word blobs) the same way as
                # cropped ones: open via PIL and re-encode, so odd modes/formats
                # (CMYK, RGBA, WMF/EMF metafiles) that the model rejects with
                # "Could not process image" are converted to clean RGB PNG.
                from PIL import Image
                from io import BytesIO
                img = Image.open(BytesIO(el.image_bytes))
                img.load()
                return encode_image_base64(img)
            except Exception:
                # If PIL can't open it (e.g. a vector metafile), fall back to
                # the raw base64 — better to try than to drop the figure.
                try:
                    import base64
                    return base64.b64encode(el.image_bytes).decode("ascii")
                except Exception:
                    pass  # fall through to render-and-crop
        # 2. render-and-crop (PDF)
        if not (document_bytes and filename and page_image_provider):
            return None
        try:
            cropped = prepare_figure_image(
                page_image_provider, document_bytes, filename, el.page, el.bbox)
        except Exception:
            return None
        if cropped is None:
            return None
        try:
            return encode_image_base64(cropped)
        except Exception:
            return None

    async def _extract_element(self, el: Element, prompt: str,
                         result: ExtractionResult,
                         document_bytes: Optional[bytes] = None,
                         filename: Optional[str] = None,
                         page_image_provider=None) -> None:
        # Content-aware routing decides the STARTING tier before any model
        # call. Hard content (tables, over-long, flagged-dense) starts on
        # Tier 2, skipping the wasted Tier 1 pass.
        decision = decide_start_tier(el)
        result.routing[el.element_id] = decision

        # Crop-and-zoom: compute once for a chart-like figure OR a complex
        # multi-header table, reuse across tier1/tier2 calls. None => text-only.
        image_b64 = self._figure_image(el, document_bytes, filename,
                                       page_image_provider)
        if image_b64 is not None:
            result.cropped_figures += 1
            prompt = prompt + (
                "\n\n(The attached image is this element, cropped and zoomed "
                "for legibility. If it is a table with multi-level column "
                "headers, attach EVERY header level to each value — the group "
                "header and the sub-column. If it is a chart, extract the "
                "salient values — peaks, troughs, notable levels and their "
                "period — not every plotted point.)")

        if decision.tier == "tier2":
            # Start on Tier 2 directly: one pass, no Tier 1 spend. A
            # low-confidence result here has no higher tier to escalate to,
            # so it flows to human review via the needs_review flag in
            # _build_claim (same as a Tier-2 result on the reactive path).
            resp = await self._client.extract("tier2", prompt, el.content,
                                        EXTRACTION_TOOL_SCHEMA, image_b64)
            result.track_cost("tier2", resp.input_tokens, resp.output_tokens)
            model_used = "tier2"
            raw_claims = (resp.tool_arguments or {}).get("claims", [])
        else:
            # Reactive cascade: Tier 1 first, escalate on low confidence.
            resp = await self._client.extract("tier1", prompt, el.content,
                                        EXTRACTION_TOOL_SCHEMA, image_b64)
            result.track_cost("tier1", resp.input_tokens, resp.output_tokens)
            model_used = "tier1"
            raw_claims = (resp.tool_arguments or {}).get("claims", [])

            # Escalate if nothing came back, or the lowest-confidence claim is
            # below the escalate threshold. The whole element is re-run on
            # Tier 2 (and both passes are billed — see CostTracker.project).
            if _min_conf(raw_claims) < self._th.tier1_escalate:
                resp = await self._client.extract("tier2", prompt, el.content,
                                            EXTRACTION_TOOL_SCHEMA, image_b64)
                result.track_cost("tier2", resp.input_tokens,
                                  resp.output_tokens)
                model_used = "tier2"
                raw_claims = (resp.tool_arguments or {}).get("claims", [])

        # Guard: raw_claims must be a list of dicts. If a model returned the
        # claims as a JSON string (not parsed), iterating it would yield single
        # characters — each failing as a claim row. Coerce defensively.
        if isinstance(raw_claims, str):
            try:
                import json as _json
                raw_claims = _json.loads(raw_claims)
            except (ValueError, TypeError):
                raw_claims = []
        if isinstance(raw_claims, dict):
            raw_claims = raw_claims.get("claims", []) if "claims" in raw_claims else [raw_claims]
        if not isinstance(raw_claims, list):
            raw_claims = []

        for rc in raw_claims:
            if not isinstance(rc, dict):
                continue  # skip stray non-dict rows rather than mis-parsing them
            claim = self._build_claim(rc, el, model_used, result)
            if claim is not None:
                result.claims.append(claim)

    def _build_claim(self, rc: dict, el: Element, model_used: str,
                     result: ExtractionResult) -> Optional[Claim]:
        """Assemble a Claim from a raw model row, quarantining bad rows.

        Required keys (field_name, value, confidence) missing => quarantine,
        never a KeyError that kills the element.
        """
        if not isinstance(rc, dict) or "field_name" not in rc or "value" not in rc:
            result.quarantine.append(
                {"kind": "claim_row", "raw": rc, "error": "missing field_name/value",
                 "source_element_id": el.element_id})
            return None

        conf = _as_float(rc.get("confidence"), default=0.0)
        canon, _mapped = canonical_metric(str(rc["field_name"]))
        # Provenance: a value read from a chart/figure is LLM_ESTIMATED (lower
        # real-world reliability at equal confidence); parsed text/tables are
        # PARSED. This does NOT force review on its own — a chart-derived
        # value only goes to a human if its own confidence is low, matching
        # the uploaded codebase's deliberate choice to keep review volume
        # manageable.
        # A value READ FROM AN IMAGE is an estimate (from pixels), whether it's
        # a chart FIGURE or a complex TABLE routed to image-reading. Tag both as
        # LLM_ESTIMATED so the gate treats their (often low) confidence the same
        # way — image reads carry inherent uncertainty and are labelled +
        # traceable rather than flooding review. Only genuinely text-parsed
        # values are PARSED.
        is_figure = el.modality == Modality.FIGURE
        was_image_read = is_figure or needs_image_read(el)
        citation = CitationTier.LLM_ESTIMATED if was_image_read else CitationTier.PARSED
        # Grounding check (hallucination guard): does the value actually appear
        # in the source element's text? Text/table values that don't are flagged
        # ungrounded and forced to review — a value the model reported but that
        # isn't in the source is a likely fabrication. Figures are not text-
        # groundable (image-derived) and come back not_applicable.
        from .grounding import check_grounding, is_suspected_hallucination
        grounding_status = check_grounding(
            str(rc["value"]), el.content or "", is_figure)
        # Structural classification (measure shape + period) from the raw name.
        cls = classify_metric(str(rc["field_name"]), rc.get("value"))
        payload = {
            "field_name": str(rc["field_name"]),
            "canonical_metric": canon,
            "value": str(rc["value"]),
            "unit": _clean(rc.get("unit")),
            "reporting_basis": _clean(rc.get("reporting_basis")),
            "netting": _clean(rc.get("netting")),
            "scale": _clean(rc.get("scale")),
            "as_at_date": _clean(rc.get("as_at_date")),
            "confidence": conf,
            "source_element_id": el.element_id,
            "entity_ref": el.entity_ref,
            "model_used": model_used,
            "needs_review": conf < self._th.review_threshold(canon)
                             or is_suspected_hallucination(grounding_status),
            "grounding": grounding_status,
            "citation_tier": citation,
            "review_tier": ReviewTier.MODEL_AUTO,
            "measure_type": cls.measure_type.value,
            "period": cls.period.value,
            "page": el.page,
            "bbox": el.bbox,
        }
        return build_claim(payload, result.quarantine)


def _clean(v) -> Optional[str]:
    """Empty strings from the model become None so structural comparison and
    Gold columns are null rather than ''."""
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _as_float(v, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _min_conf(raw_claims: list) -> float:
    if not raw_claims:
        return 0.0  # nothing extracted -> escalate
    return min(_as_float(c.get("confidence"), 0.0)
               for c in raw_claims if isinstance(c, dict))
