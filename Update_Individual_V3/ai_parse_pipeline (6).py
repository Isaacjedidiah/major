"""Production orchestrator for the ai_parse extraction branch.

Wraps the full operational flow into ONE entry point so the notebook stays thin
and non-technical operators don't hand-edit pipeline internals:

    ai_parse_document -> rule-based structuring -> gate for review
      -> write gold / silver / review / quarantine -> audit log -> search index

Call ``run_ai_parse_batch(spark, submissions, storage, ...)`` and it does the
lot, returning a small summary. All the moving parts (gating, quarantine on
failure, audit events, indexing) live here, not in the notebook.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from .hybrid_extraction import hybrid_extract_document
from .validator import gate_for_review


@dataclass
class BatchResult:
    run_id: str
    documents: int = 0
    claims_total: int = 0
    gold_total: int = 0
    review_total: int = 0
    quarantined: int = 0
    indexed_chunks: int = 0
    failures: list = field(default_factory=list)


def _run_ai_parse(spark, path: str, image_output_path: str = None):
    """Run ai_parse_document on one file. Uses the DEFAULT call (no
    descriptionElementTypes option): ai_parse handles figures automatically and
    populates their ``content`` with the read values (verified — the plain call
    returns the figure value blob). Adding descriptionElementTypes changed that
    behaviour (moved values to a generic description), suppressing figure
    extraction — so we use the default. The parser reads whatever ai_parse
    populates (content or description).

    When ``image_output_path`` is given (Option A), ai_parse also RENDERS each
    figure to an image there, so figures can be sent to Sonnet for accurate
    (colour-coded) reading.

    A UNIQUE temp view name is used per call so Spark cannot reuse a cached plan
    / result from a previous run — the lazy binaryFile+ai_parse_document query
    was being cached at the Spark level, so a second run reused the first run's
    rendering (and did NOT re-render into the new folder), which caused figures
    to be missing on re-run. Unique view + explicit read forces a fresh execute.
    """
    import uuid as _uuid
    _view = f"_aip_doc_{_uuid.uuid4().hex}"
    df = spark.read.format("binaryFile").load(path)
    df.createOrReplaceTempView(_view)
    if image_output_path:
        return spark.sql(
            f"SELECT ai_parse_document(content, "
            f"map('version','2.0','imageOutputPath','{image_output_path}')) AS p FROM {_view}"
        ).collect()[0]["p"]
    return spark.sql(
        "SELECT ai_parse_document(content, map('version','2.0')) AS p FROM " + _view
    ).collect()[0]["p"]


# optional debug: when set (via set_crop_debug_dir), each chart crop sent to
# Sonnet is also written here as a PNG for inspection. None = don't save.
_SAVE_CROPS_TO = None
_ENHANCE_CROPS = False   # crop enhancement OFF by default (page sharpening handles quality)


def set_crop_debug_dir(path):
    """Notebook helper: set a Volumes folder to save each chart crop to (for
    inspection), or None to turn it off."""
    global _SAVE_CROPS_TO
    _SAVE_CROPS_TO = path


def _enhance_for_vision(region, ImageEnhance, ImageFilter, Image, target_min=1400,
                        max_scale=3):
    """Make a chart crop more legible for the vision model: upscale small crops
    with LANCZOS, then apply light sharpening and a slight contrast lift. Moderate
    on purpose -- over-sharpening introduces artifacts. Returns the enhanced image;
    on any error returns the original (never fails the crop)."""
    try:
        w, h = region.size
        if w <= 0 or h <= 0:
            return region
        # upscale so the smaller dimension reaches ~target_min px (capped)
        if w < target_min:
            scale = min(max_scale, max(1, round(target_min / w)))
            if scale > 1:
                region = region.resize((w * scale, h * scale), Image.LANCZOS)
        # light unsharp mask (better than plain Sharpness for text/edges)
        # gentle sharpen only -- NO contrast lift (contrast was fading crops,
        # especially on already page-sharpened pages).
        region = region.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=3))
        return region
    except Exception:
        return region


def _build_figure_crop_fn(spark, image_output_path, dbutils_ref,
                          all_elements=None, save_crops_to=None, parsed=None):
    """Build a crop_figure_image(el) callable using ai_parse 2.0's coords.

    For each figure, crops the CHART region and MERGES its adjacent header
    (section_header / title / caption BEFORE it in document order) and legend
    (footnote / caption / legend AFTER it), using the 'coord' bbox that ai_parse
    2.0 provides. Never crosses into another figure (no double-capture). This
    isolates each chart (+ its header/legend) so Sonnet reads the chart WITHOUT
    the page's tables — removing the "tables read as charts" problem at source.

    Falls back to whole-page rendering only if coords/image_uri are unavailable.
    """
    from PIL import Image, ImageEnhance, ImageFilter
    import base64 as _b64, io as _io, json as _json

    # --- parse the response for pages(image_uri) + all raw elements w/ coords ---
    data = parsed
    try:
        if type(parsed).__name__ == "VariantVal" or hasattr(parsed, "toJson"):
            data = _json.loads(parsed.toJson())
        elif isinstance(parsed, str):
            data = _json.loads(parsed)
        elif hasattr(parsed, "asDict"):
            data = parsed.asDict(recursive=True)
    except Exception:
        data = None
    doc = (data or {}).get("document", data) if isinstance(data, dict) else {}
    raw_elems = doc.get("elements", []) if isinstance(doc, dict) else []
    pages = (data.get("pages") if isinstance(data, dict) else None) or \
            (doc.get("pages") if isinstance(doc, dict) else None) or []

    page_uri = {}
    for pg in pages:
        if isinstance(pg, dict):
            pid = pg.get("id") if pg.get("id") is not None else pg.get("page_id")
            uri = pg.get("image_uri")
            if pid is not None and uri:
                page_uri[pid] = uri

    def _norm(p):
        if not p:
            return None
        if p.startswith("dbfs:/Volumes"):
            return p.replace("dbfs:", "")
        if p.startswith("dbfs:"):
            return p.replace("dbfs:", "/dbfs")
        return p

    # fallback: modification-time page images (only if no image_uri)
    try:
        _imgs = sorted(dbutils_ref.fs.ls(image_output_path),
                       key=lambda f: f.modificationTime)
    except Exception:
        _imgs = []

    def _page_image(page_id):
        if page_id in page_uri:
            try:
                return Image.open(_norm(page_uri[page_id]))
            except Exception:
                pass
        idx = (page_id - 1) if page_id else 0
        if 0 <= idx < len(_imgs):
            try:
                return Image.open(_norm(_imgs[idx].path))
            except Exception:
                return None
        return None

    _HEADER_TYPES = {"section_header", "title", "caption"}
    _LEGEND_TYPES = {"footnote", "caption", "legend"}

    def _fig_coord(el):
        """coord for a ParsedElement, from its bbox[0]['coord']."""
        bb = getattr(el, "bbox", None)
        if bb and isinstance(bb, list) and isinstance(bb[0], dict):
            return bb[0].get("coord"), bb[0].get("page_id")
        return None, None

    _counter = {"n": 0}

    def crop(el):
        coord, page_id = _fig_coord(el)
        if page_id is None:
            bb = getattr(el, "bbox", None)
            page_id = (bb[0].get("page_id", 0) if bb and isinstance(bb[0], dict) else 0)
        # SHARPENED PATH: crop from the element's OWN sharpened page image (bad
        # pages affect charts too, so the chart must come from the sharpened page).
        page_img = None
        _simg = getattr(el, "sharpen_img", None) or getattr(el, "_sharpen_img", None)
        if _simg:
            try:
                page_img = Image.open(_norm(_simg))
            except Exception:
                page_img = None
        if page_img is None:
            page_img = _page_image(page_id)
        if page_img is None:
            return None
        try:
            img = page_img.convert("RGB")
            iw, ih = img.size

            # if we have a coord, do sequence-adjacency crop; else whole page
            if coord and len(coord) >= 4:
                fx0, fy0, fx1, fy1 = coord[0], coord[1], coord[2], coord[3]
                # find this figure's index among the RAW elements (by coord match)
                fig_idx = None
                for j, re_ in enumerate(raw_elems):
                    rb = re_.get("bbox")
                    if rb and isinstance(rb[0], dict) and rb[0].get("coord") == coord \
                       and rb[0].get("page_id") == page_id:
                        fig_idx = j; break
                box = {"top": fy0, "bottom": fy1, "left": fx0, "right": fx1}

                def _merge(idx, allowed):
                    if fig_idx is None or idx < 0 or idx >= len(raw_elems) or idx == fig_idx:
                        return
                    e = raw_elems[idx]
                    etype = str(e.get("type", "")).lower()
                    if etype in ("table", "figure") or etype not in allowed:
                        return
                    eb = e.get("bbox")
                    if not eb or not isinstance(eb[0], dict) or eb[0].get("page_id") != page_id:
                        return
                    cc = eb[0].get("coord")
                    if not cc or len(cc) < 4:
                        return
                    cx0, cy0, cx1, cy1 = cc
                    if not (cx1 < fx0 or cx0 > fx1) or not (cy1 < fy0 or cy0 > fy1):
                        box["top"] = min(box["top"], cy0); box["bottom"] = max(box["bottom"], cy1)
                        box["left"] = min(box["left"], cx0); box["right"] = max(box["right"], cx1)

                if fig_idx is not None:
                    _merge(fig_idx - 1, _HEADER_TYPES); _merge(fig_idx - 2, _HEADER_TYPES)
                    _merge(fig_idx + 1, _LEGEND_TYPES); _merge(fig_idx + 2, _LEGEND_TYPES)

                top, bottom = box["top"], box["bottom"]
                left, right = box["left"], box["right"]

                # clip so the crop never crosses into another FIGURE
                for oe in raw_elems:
                    if str(oe.get("type", "")).lower() != "figure":
                        continue
                    ob = oe.get("bbox")
                    if not ob or not isinstance(ob[0], dict) or ob[0].get("page_id") != page_id:
                        continue
                    oc = ob[0].get("coord")
                    if not oc or len(oc) < 4 or oc == coord:
                        continue
                    ox0, oy0, ox1, oy1 = oc
                    hh = not (ox1 < fx0 or ox0 > fx1)
                    vv = not (oy1 < fy0 or oy0 > fy1)
                    if oy1 <= fy0 and hh: top = max(top, oy1 + 5)
                    if oy0 >= fy1 and hh: bottom = min(bottom, oy0 - 5)
                    if ox1 <= fx0 and vv: left = max(left, ox1 + 5)
                    if ox0 >= fx1 and vv: right = min(right, ox0 - 5)

                x0, y0 = max(0, int(left)), max(0, int(top))
                x1, y1 = min(iw, int(right)), min(ih, int(bottom))
                if x1 <= x0 or y1 <= y0:
                    x0, y0, x1, y1 = max(0,int(fx0)), max(0,int(fy0)), min(iw,int(fx1)), min(ih,int(fy1))
                region = img.crop((x0, y0, x1, y1))
            else:
                region = img   # no coord -> whole page fallback

            if save_crops_to:
                try:
                    _counter["n"] += 1
                    region.save(f"{save_crops_to}/crop_{_counter['n']}_p{page_id}.png".replace("dbfs:", ""))
                except Exception:
                    pass

            # ENHANCE the crop for Sonnet: chart crops are often small/blurry
            # (the source page renders at limited resolution). Upscaling with a
            # high-quality filter + light sharpening/contrast makes the text and
            # data points more legible for the vision model. This improves
            # READABILITY of what is there; it cannot recover detail the source
            # render never captured (raise ai_parse's render resolution for that).
            if _ENHANCE_CROPS:
                region = _enhance_for_vision(region, ImageEnhance, ImageFilter, Image)

            buf = _io.BytesIO()
            region.save(buf, format="PNG")
            buf.seek(0)
            return _b64.b64encode(buf.read()).decode()
        except Exception:
            return None

    return crop

def _index_ai_parse_elements(parsed_response, sub, search_store, entity_ref,
                             claims=None):
    """Index ai_parse's content into the search store, formatted for
    READABILITY by type:
      * table  -> clean MARKDOWN table (not raw HTML)
      * text   -> the prose as-is
      * figure -> the EXTRACTED chart VALUES (from Sonnet/rule extraction) as
                  readable lines, not the generic description — so chart data is
                  actually searchable. Falls back to the description if no claims.
    Best-effort — never breaks the batch."""
    if search_store is None:
        return 0
    try:
        from .hybrid_extraction import parse_ai_parse_response
        from ..shared.schema import RegulatoryChunk, ChunkType, content_hash
    except Exception:
        return 0

    # group the extracted claims by the page they came from, so a figure's page
    # can be indexed with its chart values.
    figure_claims_by_page = {}
    for c in (claims or []):
        if getattr(c, "model_used", "") in ("sonnet_chart_digitiser",
                                            "ai_parse_desc_structured"):
            figure_claims_by_page.setdefault(getattr(c, "page", None), []).append(c)

    elements = parse_ai_parse_response(parsed_response)
    chunks = []
    for el in elements:
        if el.el_type in ("page_header", "page_footer", "page_number"):
            continue
        if el.el_type == "table":
            try:
                from .ai_parse_structurer import html_table_to_markdown
                text = html_table_to_markdown(el.content or "") or (el.content or "")
            except Exception:
                text = el.content or ""
        elif el.el_type == "figure":
            # prefer the EXTRACTED chart values (what Sonnet/rules read) as
            # readable lines — that's the searchable chart data. Fall back to the
            # description only if no values were extracted for this page.
            fcs = figure_claims_by_page.get(el.page, [])
            if fcs:
                lines = [f"{getattr(c, 'field_name', '') or c.canonical_metric} = "
                         f"{c.value}" + (f" {c.unit}" if getattr(c, 'unit', None) else "")
                         for c in fcs]
                head = (el.description or "Chart").strip()
                text = head + "\n" + "\n".join(lines)
            else:
                text = (el.description or "") or (el.content or "")
        else:  # text, caption, footnote — index as-is
            text = el.content or el.description or ""
        text = text.strip()
        if not text or len(text) < 3:
            continue
        try:
            chunks.append(RegulatoryChunk(
                chunk_id=content_hash(f"{sub['path']}-{el.page}-{text[:40]}"),
                chunk_type=ChunkType.RAW_TEXT,
                content=text,
                entity_ref=entity_ref or sub.get("entity_ref") or "",
                source_document_id=os.path.basename(sub["path"]),
                content_hash=content_hash(text),
                page=el.page,
                team=sub.get("team"),
                report_type=sub.get("report_type"),
                metrics={"page": el.page, "element_type": el.el_type,
                         "team": sub.get("team"),
                         "report_type": sub.get("report_type")}))
        except Exception:
            continue
    if chunks:
        try:
            # provision the vector index (ops_search_index) if it doesn't exist,
            # THEN write the chunks. The main pipeline does this in stage 4; the
            # ai_parse branch must do it too or only ops_search_chunks is created.
            if hasattr(search_store, "ensure_index"):
                search_store.ensure_index()
            search_store.index_many(chunks)
        except Exception:
            return 0
    return len(chunks)


def _build_page_url_map(parsed, sharpen_dir=None):
    """Build {page(1-based): page_image_url} so every metric from a page can be
    stamped with the SAME page_url. Prefers ai_parse's pages[].image_uri; for the
    sharpened path uses the known sharpened page images. Returns {} if none."""
    import json as _json, os as _os
    out = {}
    # 1. sharpened path: deterministic page_<NNN>.png in the sharpen dir
    if sharpen_dir:
        try:
            from .pdf_sharpen import iter_sharpened_pages, _vol
            for _sd, _page, _img in iter_sharpened_pages(sharpen_dir):
                out[_page] = _img            # already a Volume path, 1-based page
            if out:
                return out
        except Exception:
            pass
    # 2. normal path: ai_parse pages[].image_uri, keyed by page_id (+1 -> 1-based)
    try:
        data = parsed
        if type(parsed).__name__ == "VariantVal" or hasattr(parsed, "toJson"):
            data = _json.loads(parsed.toJson())
        elif isinstance(parsed, str):
            data = _json.loads(parsed)
        elif hasattr(parsed, "asDict"):
            data = parsed.asDict(recursive=True)
        doc = (data or {}).get("document", data) if isinstance(data, dict) else {}
        pages = (data.get("pages") if isinstance(data, dict) else None) or \
                (doc.get("pages") if isinstance(doc, dict) else None) or []
        for pg in pages:
            if isinstance(pg, dict):
                pid = pg.get("id") if pg.get("id") is not None else pg.get("page_id")
                uri = pg.get("image_uri")
                if pid is not None and uri:
                    u = uri.replace("dbfs:/Volumes", "/Volumes").replace("dbfs:", "/dbfs") \
                        if isinstance(uri, str) else uri
                    out[int(pid) + 1] = u     # ai_parse page_id is 0-based -> +1
    except Exception:
        pass
    return out


def _make_page_image_fn(parsed, image_dir, dbutils_ref):
    """Return a function page->base64 of the WHOLE page image, for the chart
    loop's once-per-page legend read. Uses the same page images ai_parse wrote.
    Returns None-producing fn on any failure (legend read is then skipped)."""
    import base64 as _b64, io as _io, json as _json, os as _os
    from PIL import Image
    # map 1-based page -> image path, from ai_parse pages[].image_uri
    page_uri = {}
    try:
        data = parsed
        if type(parsed).__name__ == "VariantVal" or hasattr(parsed, "toJson"):
            data = _json.loads(parsed.toJson())
        elif isinstance(parsed, str):
            data = _json.loads(parsed)
        doc = (data or {}).get("document", data) if isinstance(data, dict) else {}
        pages = (data.get("pages") if isinstance(data, dict) else None) or \
                (doc.get("pages") if isinstance(doc, dict) else None) or []
        for pg in pages:
            if isinstance(pg, dict):
                pid = pg.get("id") if pg.get("id") is not None else pg.get("page_id")
                uri = pg.get("image_uri")
                if pid is not None and uri:
                    u = uri.replace("dbfs:/Volumes", "/Volumes").replace("dbfs:", "/dbfs") \
                        if isinstance(uri, str) else uri
                    page_uri[int(pid) + 1] = u
    except Exception:
        pass

    def _fn(page):
        try:
            uri = page_uri.get(page)
            if not uri:
                return None
            img = Image.open(uri)
            buf = _io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
            return _b64.b64encode(buf.read()).decode()
        except Exception:
            return None
    return _fn


def run_ai_parse_batch(spark, submissions, storage, *, audit=None,
                       search_store=None, limit=None, use_sonnet=False,
                       llm_client=None, run_async_fn=None,
                       image_output_path=None, dbutils=None,
                       folder_map=None, sharpen_pdfs=False,
                       sharpen_root=None, missed_root=None,
                       use_chart_loop=False) -> BatchResult:
    """Run the ai_parse branch over a batch of submissions, end-to-end.

    Args:
      spark        — the Spark session.
      submissions  — list of {path, team, report_type, entity_ref}.
      storage      — a DeltaLakeStorage (already prefix-configured by caller).
      audit        — optional AuditLog; created if None.
      search_store — optional vector store for indexing (RAG). None -> no index.
      limit        — process only the first N submissions (None = all).
      use_sonnet   — send FIGURES (only) to Sonnet for accurate colour-coded
                     chart reading (Option A). Tables/text stay rule-based.
                     Needs llm_client + run_async_fn + image_output_path +
                     dbutils.
      image_output_path — a DEDICATED Unity Catalog volume folder. When set with
                     use_sonnet, ai_parse renders each figure there and Sonnet
                     reads it. Cleared PER DOCUMENT so the figure->image mapping
                     stays unambiguous.
      dbutils      — the notebook's dbutils (for clearing/listing the volume).

    Returns a BatchResult summary. Writes gold, silver, review, quarantine,
    audit, and (if search_store given) the index — the full operational set.
    """
    from ..shared.audit_log import AuditLog
    audit = audit or AuditLog()
    result = BatchResult(run_id=str(uuid.uuid4()))
    option_a = bool(use_sonnet and image_output_path and dbutils
                    and llm_client and run_async_fn)

    subs = submissions[:limit] if limit else submissions

    # SHARPENED-PDF support: when sharpen_pdfs is on, a PDF's pages are SHARPENED
    # (fixes OCR misreads from soft scans) into THIS run's own fresh folder --
    # the SAME per-run/per-document pattern as the figure images/crops, so the
    # run_id is internal (no hand-off) and folders are fresh (reproducible).
    _sharpen_on = bool(sharpen_pdfs)
    _sharpen_base = (sharpen_root or image_output_path)

    for sub in subs:
        path = sub["path"]
        fname = os.path.basename(path)
        try:
            crop_fn = None
            _file_img_dir = None
            _file_crop_dir = None
            _file_missed_dir = None
            if option_a:
                # PRODUCTION reproducibility: give THIS file, THIS run its own
                # fresh folders under one per-document base — images, crops AND
                # missed crops all live together, named by run_id/document, so
                # everything for a document-run is in one traceable place and
                # there is no stale-image / eventual-consistency contamination.
                import re as _re
                _safe_doc = _re.sub(r"[^A-Za-z0-9_.-]", "_", fname)[:80]
                _base = f"{image_output_path.rstrip('/')}/{result.run_id}/{_safe_doc}"
                _file_img_dir = f"{_base}/images"
                _file_crop_dir = f"{_base}/crops"
                _file_missed_dir = f"{_base}/missed"
                try:
                    dbutils.fs.mkdirs(_file_img_dir)
                    dbutils.fs.mkdirs(_file_crop_dir)
                    dbutils.fs.mkdirs(_file_missed_dir)   # pre-create so misses land
                except Exception:
                    pass

            # SHARPENED PDF path: when sharpen_pdfs is on and this is a PDF, we
            # SHARPEN each page (fixes OCR misreads from soft scans) into THIS
            # run's OWN fresh folder -- the SAME per-run/per-document pattern as
            # the figure images, so the run_id is internal (no hand-off) and the
            # folders are fresh (reproducible). We then ai_parse the SHARPENED
            # page images instead of the raw PDF. Non-PDFs / sharpen-off -> the
            # normal raw path (unchanged). Sharpening lives in its own module
            # (pdf_sharpen) so a sharpen issue never touches extraction/storage.
            if _sharpen_on and str(path).lower().endswith(".pdf"):
                from .pdf_sharpen import (sharpen_pdf_to_folder,
                                          iter_sharpened_pages,
                                          run_ai_parse_on_image, _safe_folder)
                import re as _re
                _safe_doc2 = _re.sub(r"[^A-Za-z0-9_.-]", "_", fname)[:80]
                _sharp_dir = (f"{_base}/sharpened" if _file_img_dir
                              else f"{(_sharpen_base or image_output_path).rstrip('/')}/"
                                   f"{result.run_id}/{_safe_doc2}/sharpened")
                try:
                    dbutils.fs.mkdirs(_sharp_dir)
                except Exception:
                    pass
                sharpen_pdf_to_folder(path, _sharp_dir)   # writes page_<NNN>.png here
                # ai_parse EACH sharpened page, merge the parsed pages' elements.
                # Tag each element with its sharpened image path + page so the crop
                # fn crops from the SHARPENED page image (bad pages affect charts
                # too, so charts must come from the sharpened page).
                _merged_elements = []
                for _sd, _page, _img in iter_sharpened_pages(_sharp_dir):
                    _pp = run_ai_parse_on_image(spark, _img, _file_img_dir)
                    import json as _json
                    _d = _json.loads(_pp) if isinstance(_pp, str) else (
                        _json.loads(_pp.toJson()) if hasattr(_pp, "toJson") else _pp)
                    for _el in _d.get("document", _d).get("elements", []):
                        _el["_sharpen_page"] = _page        # real page number
                        _el["_sharpen_img"] = _img          # the sharpened page image
                        _merged_elements.append(_el)
                parsed = {"document": {"elements": _merged_elements}}
            else:
                parsed = _run_ai_parse(spark, path, _file_img_dir) if option_a \
                    else _run_ai_parse(spark, path)
            audit.log(result.run_id, "ai_parse", path, detail=f"parsed {fname}")

            # tag each figure with its index (position among figures) so the
            # crop fn can map it to the correct rendered image.
            crop_fn = None
            if option_a:
                from .hybrid_extraction import parse_ai_parse_response
                _els = parse_ai_parse_response(parsed)
                fi = 0
                for e in _els:
                    if e.el_type == "figure":
                        e.figure_index = fi
                        fi += 1
                # build the whole-page image fn. Pass ai_parse's pages array so
                # the page->image mapping uses the reliable image_uri (not the
                # unstable modification-time sort that sent wrong pages/tables).
                crop_fn = _build_figure_crop_fn(
                    spark, _file_img_dir, dbutils,
                    all_elements=_els,
                    save_crops_to=(_file_crop_dir or _SAVE_CROPS_TO),
                    parsed=parsed)

            # firm-specific x-axis geometry config (None -> normal path, no-op).
            # Resolved here so the SINGLE-call position read + in-memory geometry
            # applies only to the configured firm.
            _fx_cfg = None
            try:
                from .firm_xaxis import _firm_config
                from .entity_path import parse_entity_path as _pep
                _ent = _pep(path)
                _fx_cfg = _firm_config(
                    firm_id=sub.get("entity_id") or _ent.get("entity_id"),
                    firm_name=sub.get("entity_name") or _ent.get("entity_name"))
            except Exception:
                _fx_cfg = None

            claims = hybrid_extract_document(
                ai_parse_response=parsed, filename=fname,
                team=sub["team"], report_type=sub["report_type"],
                entity_ref=sub.get("entity_ref"),
                use_sonnet=option_a, llm_client=llm_client,
                run_async_fn=run_async_fn, crop_figure_image=crop_fn,
                firm_xaxis_cfg=_fx_cfg,
                missed_root=_file_missed_dir,
                audit=audit, run_id=result.run_id,
                use_chart_loop=use_chart_loop,
                page_image_fn=(_make_page_image_fn(parsed, _file_img_dir, dbutils)
                               if use_chart_loop else None))
            audit.log(result.run_id, "extract", path,
                      detail=f"{len(claims)} claims")

            gated = gate_for_review(claims, [])
            clean, review = gated["clean"], gated["needs_review"]

            # derive the report date (best-effort; never break the doc over it)
            try:
                from .storage import derive_report_date
                from .hybrid_extraction import parse_ai_parse_response
                _title_els = parse_ai_parse_response(parsed)
                rdate = derive_report_date(claims, _title_els, path=path)
            except Exception:
                rdate = None

            # parse entity_id / entity_name from the document's volume path so
            # they become recognisable columns (submission may already carry them).
            from .entity_path import parse_entity_path
            ent = parse_entity_path(path)
            entity_id = sub.get("entity_id") or ent["entity_id"]
            entity_name = sub.get("entity_name") or ent["entity_name"]

            # page_url: one image url per page (SAME for every metric on a page)
            _purl_map = _build_page_url_map(
                parsed, sharpen_dir=(_sharp_dir if ("_sharp_dir" in dir()
                                                    and _sharpen_on) else None))

            storage.write_silver(claims, team=sub["team"],
                                 report_type=sub["report_type"],
                                 report_date=rdate, entity_id=entity_id,
                                 entity_name=entity_name, page_url_map=_purl_map)
            result.gold_total += storage.write_gold_metrics(
                clean, team=sub["team"], report_type=sub["report_type"],
                report_date=rdate, entity_id=entity_id, entity_name=entity_name,
                page_url_map=_purl_map)
            if review:
                storage.write_review_claims(
                    review, team=sub["team"], report_type=sub["report_type"],
                    report_date=rdate, entity_id=entity_id, entity_name=entity_name)

            # ADDITIVE outputs (best-effort; never fail the run):
            #  - bronze_original: the RAW ai_parse blob per element (traceability)
            #  - query_gold: all gold claims + identifiers map (tables only)
            #  - json: claims for this document serialised to a Volume
            try:
                _raw_els = []
                for _el in parse_ai_parse_response(parsed):
                    _raw_els.append({
                        "source_document": fname,
                        "page": getattr(_el, "page", None),
                        "element_type": getattr(_el, "el_type", None),
                        "raw_content": getattr(_el, "content", None),
                        "bbox": str(getattr(_el, "bbox", None)),
                    })
                if _raw_els:
                    storage.write_bronze_original(_raw_els)
            except Exception as e:
                audit.log(result.run_id, "bronze_skip", path, detail=str(e))
            try:
                _gold_rows = storage.build_gold_metric_rows(clean)
                storage.write_query_gold(_gold_rows, clean)
            except Exception as e:
                audit.log(result.run_id, "query_gold_skip", path, detail=str(e))
            try:
                storage.write_json_output(claims, fname, dbutils=dbutils)
            except Exception as e:
                audit.log(result.run_id, "json_skip", path, detail=str(e))

            audit.log(result.run_id, "store", path,
                      detail=f"gold={len(clean)} review={len(review)}")

            n_idx = _index_ai_parse_elements(
                parsed, sub, search_store, sub.get("entity_ref"), claims=claims)
            result.indexed_chunks += n_idx

            result.documents += 1
            result.claims_total += len(claims)
            result.review_total += len(review)

        except Exception as e:
            try:
                storage.write_quarantine([{
                    "kind": "document", "source_element_id": fname,
                    "error": str(e)[:200], "source_document": path,
                    "team": sub["team"], "report_type": sub["report_type"],
                    "entity_ref": sub.get("entity_ref")}])
            except Exception:
                pass
            audit.log(result.run_id, "error", path, error=str(e)[:200])
            result.quarantined += 1
            result.failures.append((fname, str(e)[:150]))

    # flush the audit events to ops_audit_events.
    try:
        storage.write_audit(audit.to_rows())
    except Exception:
        pass
    _log_run_to_mlflow(result, run_name="ai_parse_batch",
                       params={"use_sonnet": use_sonnet, "limit": limit})
    return result


def _log_run_to_mlflow(result, run_name="pipeline_run", params=None):
    """Log the run's scorecard metrics to MLflow — only when MLflow is present
    and enabled in config. Safe no-op otherwise (never breaks the run)."""
    try:
        from ..shared.config import CONFIG
        if not getattr(getattr(CONFIG, "observability", None), "mlflow_tracing", False):
            return
        import mlflow
        from .scorecard import build_scorecard
        sc = build_scorecard(result)
        with mlflow.start_run(run_name=run_name):
            if params:
                mlflow.log_params(params)
            for section in ("coverage", "quality", "robustness", "cost"):
                for k, v in sc[section].items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
    except Exception:
        pass


def run_ai_extract_batch(spark, submissions, storage, *, audit=None,
                         search_store=None, limit=None, mode="precision"):
    """Run the ai_parse -> ai_extract branch over a batch, end-to-end, with the
    full operational set (gate, gold/silver/review/quarantine, audit, index) —
    the LLM-structured alternative to run_ai_parse_batch. Requires ai_extract to
    be available/approved. Returns a BatchResult summary."""
    from ..shared.audit_log import AuditLog
    from .ai_extract_branch import build_ai_extract_sql, ai_extract_result_to_claims
    audit = audit or AuditLog()
    result = BatchResult(run_id=str(uuid.uuid4()))

    subs = submissions[:limit] if limit else submissions

    for sub in subs:
        path = sub["path"]
        fname = os.path.basename(path)
        try:
            (spark.read.format("binaryFile").load(path)
                .createOrReplaceTempView("_aix_doc"))
            sql = build_ai_extract_sql("_aix_doc", mode=mode)
            extracted = spark.sql(sql).collect()[0]["extracted"]
            audit.log(result.run_id, "ai_extract", path, detail=f"extracted {fname}")

            claims = ai_extract_result_to_claims(
                extracted, fname, team=sub["team"],
                report_type=sub["report_type"], entity_ref=sub.get("entity_ref"))
            audit.log(result.run_id, "map", path, detail=f"{len(claims)} claims")

            gated = gate_for_review(claims, [])
            clean, review = gated["clean"], gated["needs_review"]

            from .storage import derive_report_date
            from .hybrid_extraction import parse_ai_parse_response
            _title_els = parse_ai_parse_response(parsed)
            rdate = derive_report_date(claims, _title_els, path=path)

            storage.write_silver(claims, team=sub["team"],
                                 report_type=sub["report_type"],
                                 report_date=rdate)
            result.gold_total += storage.write_gold_metrics(
                clean, team=sub["team"], report_type=sub["report_type"],
                report_date=rdate)
            if review:
                storage.write_review_claims(
                    review, team=sub["team"], report_type=sub["report_type"],
                    report_date=rdate)
            audit.log(result.run_id, "store", path,
                      detail=f"gold={len(clean)} review={len(review)}")
            try:
                parsed = _run_ai_parse(spark, path)
                result.indexed_chunks += _index_ai_parse_elements(
                    parsed, sub, search_store, sub.get("entity_ref"), claims=claims)
            except Exception:
                pass

            result.documents += 1
            result.claims_total += len(claims)
            result.review_total += len(review)

        except Exception as e:
            try:
                storage.write_quarantine([{
                    "kind": "document", "source_element_id": fname,
                    "error": str(e)[:200], "source_document": path,
                    "team": sub["team"], "report_type": sub["report_type"],
                    "entity_ref": sub.get("entity_ref")}])
            except Exception:
                pass
            audit.log(result.run_id, "error", path, error=str(e)[:200])
            result.quarantined += 1
            result.failures.append((fname, str(e)[:150]))

    try:
        storage.write_audit(audit.to_rows())
    except Exception:
        pass
    return result
