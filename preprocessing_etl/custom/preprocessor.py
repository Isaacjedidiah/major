"""Format-agnostic preprocessing.

The only module that knows about file formats. Everything downstream works
on ``Element`` objects. Unknown formats fail loudly rather than silently
dropping a submission.

Real parsing uses pdfplumber (PDF text/tables), python-docx (Word),
openpyxl (Excel), python-pptx (PowerPoint), and the stdlib email package
(.eml; .msg needs the optional extract-msg library). Imports are lazy so the
module can be imported in environments lacking a given parser.

Email attachments are deliberately NOT unpacked here: teams place attachments
in the submission folder as their own files, so each flows through the
pipeline independently with its own provenance.

Scanned / image-only PDF pages have no text layer, so pdfplumber returns
nothing for them. Those pages fall back to ``ai_parse_document`` (the same
``databricks-ai-parse`` endpoint the native track uses) for OCR, keeping one
OCR path across both tracks. If the endpoint is unavailable or yields
nothing, the page keeps its flagged-empty element and goes to quarantine —
no silent data loss.

entity_ref resolution: the firm reference is threaded onto every Element from
the submission descriptor (resolved upstream from the volume path or a
sidecar manifest). This is what makes Gold attributable — previously nothing
populated it.
"""
from __future__ import annotations

import os
import re
from typing import Callable, Optional

from ..shared.schema import Element, Modality
from .ai_parse_client import AIParseClient


def preprocess_pdf(path: str, team: str, report_type: str,
                   entity_ref: Optional[str] = None,
                   ocr_client: Optional[AIParseClient] = None) -> list[Element]:
    import pdfplumber  # lazy

    # Instantiate the OCR fallback once per document. If it isn't configured
    # (e.g. local dev without a Databricks endpoint), ``available`` is False
    # and we skip straight to the flagged-empty behaviour.
    ocr = ocr_client if ocr_client is not None else AIParseClient()

    elements: list[Element] = []
    with pdfplumber.open(path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            paragraphs = _paragraphs_with_bbox(page)   # (text, bbox) pairs
            # find_tables() gives Table objects WITH a .bbox (needed to crop a
            # complex table for image-reading). BUT its default detection is
            # stricter than extract_tables() and can MISS borderless / whitespace
            # -aligned tables. So we take find_tables() for the bbox-carrying
            # ones, then add any extract_tables() tables that find_tables missed
            # (as bbox-less table elements) — so no table is lost.
            table_objs = page.find_tables() or []
            found_bboxes = []
            for t in table_objs:
                try:
                    found_bboxes.append(tuple(t.bbox))
                except Exception:
                    pass
            extract_only = []
            try:
                raw_tables = page.extract_tables() or []
                # extract_tables returns data only; keep those we can't match to
                # a find_tables bbox (i.e. the ones find_tables missed).
                if len(raw_tables) > len(table_objs):
                    extract_only = raw_tables[len(table_objs):]
            except Exception:
                extract_only = []

            if not paragraphs and not table_objs and not extract_only:
                # No text layer: likely scanned. Try OCR via ai_parse_document;
                # fall back to a flagged-empty element (-> quarantine) if that
                # is unavailable or returns nothing.
                ocr_elements = _ocr_page(page, page_no, path, team,
                                         report_type, entity_ref, ocr)
                elements.extend(ocr_elements)
                continue

            for para, bbox in paragraphs:
                elements.append(_text_element(
                    path, page_no, len(elements), para, team, report_type,
                    entity_ref, bbox=bbox))
            pw, ph = float(page.width or 1), float(page.height or 1)
            for tbl_no, tobj in enumerate(table_objs):
                # normalise the table bbox (x0,y0,x1,y1 in points) to 0-1 so the
                # crop provider can locate it on the rendered page.
                nbbox = None
                try:
                    x0, y0, x1, y1 = tobj.bbox
                    nbbox = (x0 / pw, y0 / ph, x1 / pw, y1 / ph)
                except Exception:
                    nbbox = None
                elements.append(_table_element(
                    path, page_no, tbl_no,
                    _table_to_markdown(tobj.extract()), team,
                    report_type, entity_ref, bbox=nbbox))
            # tables extract_tables() found but find_tables() missed (borderless
            # etc.) — emit them without a bbox (they extract as text; if complex
            # they still route by content, just can't be image-cropped).
            for k, raw in enumerate(extract_only):
                elements.append(_table_element(
                    path, page_no, len(table_objs) + k,
                    _table_to_markdown(raw), team, report_type, entity_ref,
                    bbox=None))
            # Charts/figures: embedded images AND dense vector-graphics regions
            # become FIGURE elements with a normalised bbox, so the crop-and-zoom
            # vision path reads them. Charts have no text layer, so without this
            # they would be silently skipped.
            for fig_no, fbbox in enumerate(_detect_figure_regions(page)):
                elements.append(Element(
                    element_id=f"{os.path.basename(path)}-p{page_no}-fig{fig_no}",
                    modality=Modality.FIGURE,
                    content="",
                    source_document=path,
                    page=page_no,
                    bbox=fbbox,
                    team=team,
                    report_type=report_type,
                    entity_ref=entity_ref))
    return elements


def _detect_figure_regions(page) -> list[tuple]:
    """Find chart/figure regions on a pdfplumber page, as normalised bboxes
    (x0, top, x1, bottom) in 0..1 page coordinates.

    Two sources, because charts arrive both ways:
      1. Embedded raster images (page.images) — the common case: a chart
         pasted in as a picture. Each image is its own region.
      2. Vector-drawn charts — lines/curves/rects rendered directly. We look
         for a region dense with vector primitives that isn't a table. As a
         robust proxy we take the bounding box of all curves+lines when there
         are enough of them (charts have many; plain rules have few).

    Errs toward catching charts (a missed chart is the failure mode we avoid);
    a false positive just sends a harmless region to the vision model.
    """
    pw = float(page.width) or 1.0
    ph = float(page.height) or 1.0
    regions: list[tuple] = []

    def _norm(x0, top, x1, bottom):
        # clamp + normalise; skip only genuinely tiny/degenerate boxes. Use a
        # LOOSE per-dimension floor so short/thin real charts (e.g. a wide but
        # short bar chart) are kept — the old "either dimension < 5%" test
        # wrongly discarded them. We intentionally do NOT gate on area: a small
        # but proportionate chart (e.g. 8% x 12%) is a real chart and must be
        # kept.
        x0, x1 = max(0.0, x0), min(pw, x1)
        top, bottom = max(0.0, top), min(ph, bottom)
        w = (x1 - x0) / pw if pw else 0
        h = (bottom - top) / ph if ph else 0
        # drop only a near-zero sliver in either dimension (noise/rules/borders).
        if w < 0.02 or h < 0.02:
            return None
        return (x0 / pw, top / ph, x1 / pw, bottom / ph)

    # 1. embedded raster images
    for img in (getattr(page, "images", None) or []):
        b = _norm(float(img.get("x0", 0)), float(img.get("top", 0)),
                  float(img.get("x1", 0)), float(img.get("bottom", 0)))
        if b is not None:
            regions.append(b)

    # 2. vector-drawn charts: if there are many curves/lines, bound them
    curves = (getattr(page, "curves", None) or [])
    lines = (getattr(page, "lines", None) or [])
    vectors = curves + lines
    # a table is mostly straight lines; a chart has curves or many segments.
    # Trigger when there are enough primitives OR any curves are present.
    if len(vectors) >= 12 or len(curves) >= 1:
        xs0 = [float(v.get("x0", 0)) for v in vectors]
        xs1 = [float(v.get("x1", 0)) for v in vectors]
        ts = [float(v.get("top", 0)) for v in vectors]
        bs = [float(v.get("bottom", 0)) for v in vectors]
        if xs0 and ts:
            b = _norm(min(xs0), min(ts), max(xs1), max(bs))
            # avoid duplicating a region already captured as an embedded image
            if b is not None and not _overlaps_any(b, regions):
                regions.append(b)

    return regions


def _overlaps_any(box, existing, thresh: float = 0.6) -> bool:
    """True if box substantially overlaps any box already in existing."""
    ax0, ay0, ax1, ay1 = box
    a_area = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
    for bx0, by0, bx1, by1 in existing:
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        if ix1 > ix0 and iy1 > iy0:
            inter = (ix1 - ix0) * (iy1 - iy0)
            if inter / a_area >= thresh:
                return True
    return False


def _ocr_page(page, page_no: int, path: str, team: str, report_type: str,
              entity_ref: Optional[str],
              ocr: AIParseClient) -> list[Element]:
    """OCR one text-less page, or return a single flagged-empty element.

    The flagged-empty element preserves the previous behaviour exactly, so a
    page we still can't read is never silently dropped — it goes to
    quarantine downstream.
    """
    flagged_empty = [Element(
        element_id=f"{os.path.basename(path)}-p{page_no}-empty",
        modality=Modality.IMAGE,
        content="",
        source_document=path,
        page=page_no,
        team=team,
        report_type=report_type,
        entity_ref=entity_ref,
    )]

    if not ocr.available:
        return flagged_empty

    try:
        image_bytes = _render_page_png(page)
        blocks = ocr.parse_page_image(image_bytes, mime="image/png")
    except Exception:
        # Any parse/transport failure: keep the page visible via quarantine
        # rather than losing it. (No silent fallback to a stub.)
        return flagged_empty

    if not blocks:
        return flagged_empty

    out: list[Element] = []
    for i, block in enumerate(blocks):
        if block.is_table:
            out.append(_table_element(
                path, page_no, i, block.content, team, report_type, entity_ref,
                suffix="ocr"))
        else:
            out.append(_text_element(
                path, page_no, i, block.content, team, report_type, entity_ref,
                suffix="ocr"))
    return out


def _render_page_png(page) -> bytes:
    """Render a pdfplumber page to PNG bytes for OCR.

    Uses pdfplumber's page.to_image(); the resolution is high enough for OCR
    without producing an oversized payload.
    """
    import io

    img = page.to_image(resolution=200)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _native_chart_to_markdown(chart) -> str:
    """Turn a native PPTX chart's embedded data into a markdown table of EXACT
    values (categories x series). These are the real numbers the chart was built
    from — no pixel-reading, no vision model, no misreads. Returns "" if the
    chart carries no usable data, so the caller can fall back to other paths.

    Layout: first column is the category (x-axis label); one column per series
    with its values. A chart title, if present, is included as a caption line.
    """
    try:
        plots = list(chart.plots)
        if not plots:
            return ""
        categories = [str(c) for c in plots[0].categories]
        series = []
        for plot in plots:
            for s in plot.series:
                name = getattr(s, "name", None) or f"series{len(series)+1}"
                values = list(s.values)
                series.append((str(name), values))
        if not series:
            return ""

        lines = []
        title = None
        try:
            if chart.has_title and chart.chart_title.text_frame.text.strip():
                title = chart.chart_title.text_frame.text.strip()
        except Exception:
            title = None
        if title:
            lines.append(f"Chart: {title}")

        header = ["category"] + [n for n, _ in series]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        n_rows = len(categories) if categories else max(len(v) for _, v in series)
        for i in range(n_rows):
            cat = categories[i] if i < len(categories) else str(i + 1)
            row = [cat]
            for _, values in series:
                v = values[i] if i < len(values) else ""
                row.append("" if v is None else str(v))
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)
    except Exception:
        return ""  # any structural surprise -> fall back, never raise


def _text_element(path: str, page_no: int, idx: int, content: str, team: str,
                  report_type: str, entity_ref: Optional[str],
                  suffix: str = "t",
                  bbox: Optional[tuple] = None) -> Element:
    return Element(
        element_id=f"{os.path.basename(path)}-p{page_no}-{suffix}{idx}",
        modality=Modality.TEXT,
        content=content,
        source_document=path,
        page=page_no,
        bbox=bbox,
        team=team,
        report_type=report_type,
        entity_ref=entity_ref,
    )


def _table_element(path: str, page_no: int, idx: int, content: str, team: str,
                   report_type: str, entity_ref: Optional[str],
                   suffix: str = "tbl", bbox: Optional[tuple] = None) -> Element:
    return Element(
        element_id=f"{os.path.basename(path)}-p{page_no}-{suffix}{idx}",
        modality=Modality.TABLE,
        content=content,
        source_document=path,
        page=page_no,
        bbox=bbox,
        team=team,
        report_type=report_type,
        entity_ref=entity_ref,
    )


def preprocess_docx(path: str, team: str, report_type: str,
                    entity_ref: Optional[str] = None) -> list[Element]:
    import docx  # python-docx, lazy

    doc = docx.Document(path)
    elements: list[Element] = []
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip():
            elements.append(Element(
                element_id=f"{os.path.basename(path)}-para{i}",
                modality=Modality.TEXT,
                content=para.text,
                source_document=path,
                team=team,
                report_type=report_type,
                entity_ref=entity_ref,
            ))
    for t, table in enumerate(doc.tables):
        rows = [[c.text for c in row.cells] for row in table.rows]
        elements.append(Element(
            element_id=f"{os.path.basename(path)}-tbl{t}",
            modality=Modality.TABLE,
            content=_rows_to_markdown(rows),
            source_document=path,
            team=team,
            report_type=report_type,
            entity_ref=entity_ref,
        ))
    # Embedded charts/images -> FIGURE elements carrying the image bytes, so the
    # vision path reads them (same approach as PPTX). Word stores images as
    # related parts; we pull each image part's blob directly. Errs toward
    # catching charts — a decorative image just returns "no metrics" cheaply.
    try:
        rels = doc.part.rels
        img_no = 0
        for rel in rels.values():
            if "image" in rel.reltype:
                try:
                    blob = rel.target_part.blob
                except Exception:
                    blob = None
                if blob:
                    elements.append(Element(
                        element_id=f"{os.path.basename(path)}-img{img_no}",
                        modality=Modality.FIGURE,
                        content="",
                        source_document=path,
                        bbox=(0.0, 0.0, 1.0, 1.0),  # whole-image; no layout bbox in docx
                        image_bytes=blob,
                        team=team,
                        report_type=report_type,
                        entity_ref=entity_ref,
                    ))
                    img_no += 1
    except Exception:
        pass  # no images / unreadable rels — text+tables still returned
    return elements


def preprocess_xlsx(path: str, team: str, report_type: str,
                    entity_ref: Optional[str] = None) -> list[Element]:
    from openpyxl import load_workbook  # lazy

    wb = load_workbook(path, data_only=True)
    elements: list[Element] = []
    for sheet in wb.worksheets:
        rows = [[("" if c is None else str(c)) for c in row]
                for row in sheet.iter_rows(values_only=True)]
        if rows:
            elements.append(Element(
                element_id=f"{os.path.basename(path)}-{sheet.title}",
                modality=Modality.TABLE,
                content=_rows_to_markdown(rows),
                source_document=path,
                team=team,
                report_type=report_type,
                entity_ref=entity_ref,
            ))
    return elements


def _flatten_shapes(shapes):
    """Yield every shape on a slide, recursing INTO group shapes so charts and
    pictures nested in a group aren't missed. PowerPoint groups a chart with its
    title/annotations very commonly; without this, grouped charts (shape_type 6
    GROUP) are invisible to a top-level-only loop — a frequent cause of 'some
    charts weren't processed'."""
    for shape in shapes:
        # MSO_SHAPE_TYPE.GROUP == 6; group shapes expose their children as
        # shape.shapes. Recurse; otherwise yield the shape itself.
        if getattr(shape, "shape_type", None) == 6 or hasattr(shape, "shapes"):
            try:
                yield from _flatten_shapes(shape.shapes)
                continue
            except Exception:
                pass
        yield shape


def preprocess_pptx(path: str, team: str, report_type: str,
                    entity_ref: Optional[str] = None) -> list[Element]:
    """Read slide text, tables and speaker notes from a .pptx.

    Slide TEXT is cheap (local, no model call). Slides also often carry
    meaning in images/diagrams — those are flagged as FIGURE elements with the
    slide as ``page`` so the downstream vision path can handle them, exactly as
    chart figures are handled in PDFs. Slide number is used as ``page`` so a
    retrieved chunk can be traced to its slide.
    """
    from pptx import Presentation  # python-pptx, lazy

    prs = Presentation(path)
    # slide dimensions (EMU) for normalising picture bboxes to 0-1
    sw = float(prs.slide_width or 1)
    sh = float(prs.slide_height or 1)

    elements: list[Element] = []
    for slide_no, slide in enumerate(prs.slides, start=1):
        for shp_no, shape in enumerate(_flatten_shapes(slide.shapes)):
            # tables
            if shape.has_table:
                rows = [[cell.text for cell in row.cells]
                        for row in shape.table.rows]
                elements.append(Element(
                    element_id=f"{os.path.basename(path)}-s{slide_no}-tbl{shp_no}",
                    modality=Modality.TABLE,
                    content=_rows_to_markdown(rows),
                    source_document=path, page=slide_no,
                    team=team, report_type=report_type, entity_ref=entity_ref))
                continue
            # NATIVE CHART: a real PowerPoint chart object carries its data as
            # structured numbers in the file — read them directly (exact, no
            # vision model, no pixel-reading). This beats reading the picture
            # for any chart that is a native object rather than a pasted image.
            if getattr(shape, "has_chart", False):
                md = _native_chart_to_markdown(shape.chart)
                if md:
                    elements.append(Element(
                        element_id=f"{os.path.basename(path)}-s{slide_no}-chart{shp_no}",
                        modality=Modality.TABLE,  # structured data reads best as a table
                        content=md,
                        source_document=path, page=slide_no,
                        team=team, report_type=report_type, entity_ref=entity_ref))
                    continue
            # text frames
            if shape.has_text_frame and shape.text_frame.text.strip():
                elements.append(_text_element(
                    path, slide_no, shp_no, shape.text_frame.text, team,
                    report_type, entity_ref, suffix="s"))
                continue
            # pictures -> FIGURE with a normalised bbox, for the vision path
            # Detect a PICTURE robustly: by the numeric enum (13), by the enum
            # NAME ("PICTURE"), or simply by the shape exposing an .image — some
            # picture-like shapes (placeholders holding an image) don't report
            # shape_type == 13 but do have .image.
            _stype = getattr(shape, "shape_type", None)
            _is_picture = (
                _stype == 13
                or str(getattr(_stype, "name", _stype)).upper() == "PICTURE"
                or hasattr(shape, "image"))
            if _is_picture:
                try:
                    bbox = (float(shape.left) / sw, float(shape.top) / sh,
                            float(shape.left + shape.width) / sw,
                            float(shape.top + shape.height) / sh)
                except (TypeError, ValueError):
                    bbox = None
                # Pull the embedded image bytes directly — no slide rendering
                # needed. This is what the vision path reads for PPTX charts.
                img_bytes = None
                try:
                    img_bytes = shape.image.blob
                except Exception:
                    img_bytes = None

                # Flag vector images (EMF/WMF) that PIL can't rasterise — these
                # can't be read on serverless (no renderer). We mark them so the
                # run reports exactly which slides need a one-off manual PDF
                # conversion + re-upload. Detection: an unreadable-by-PIL blob.
                needs_conversion = False
                if img_bytes:
                    try:
                        ext = ""
                        try:
                            ext = (shape.image.ext or "").lower()
                        except Exception:
                            ext = ""
                        if ext in ("emf", "wmf"):
                            needs_conversion = True
                        else:
                            from PIL import Image as _PImg
                            from io import BytesIO as _BIO
                            try:
                                _PImg.open(_BIO(img_bytes)).load()
                            except Exception:
                                needs_conversion = True  # PIL can't open it
                    except Exception:
                        needs_conversion = True

                content = ""
                if needs_conversion:
                    content = ("[vector image (EMF/WMF) — not auto-readable on "
                               f"serverless; convert slide {slide_no} to PDF and "
                               "re-upload to extract this chart]")
                elements.append(Element(
                    element_id=f"{os.path.basename(path)}-s{slide_no}-img{shp_no}",
                    modality=Modality.FIGURE,
                    content=content,
                    source_document=path, page=slide_no, bbox=bbox,
                    image_bytes=img_bytes,
                    team=team, report_type=report_type, entity_ref=entity_ref))

        # speaker notes as text (often carry the real narrative)
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                elements.append(_text_element(
                    path, slide_no, 999, notes, team, report_type,
                    entity_ref, suffix="notes"))
    return elements


def preprocess_email(path: str, team: str, report_type: str,
                     entity_ref: Optional[str] = None) -> list[Element]:
    """Read an email body (.eml / .msg), headers as metadata.

    ATTACHMENTS ARE DELIBERATELY IGNORED — teams place attachments in the
    submission folder as their own files, so they flow through the pipeline on
    their own (and get their own entity_ref / provenance) rather than being
    silently unpacked here. Only the message body and its headers are read.

    .eml is parsed with the stdlib ``email`` package. .msg (Outlook) needs the
    optional ``extract-msg`` library; if it's absent, the file is flagged-empty
    (-> quarantine) rather than crashing, mirroring the OCR fallback posture.
    """
    ext = os.path.splitext(path)[1].lower()
    headers: dict = {}
    body = ""

    if ext == ".eml":
        import email
        from email import policy
        with open(path, "rb") as fh:
            msg = email.message_from_binary_file(fh, policy=policy.default)
        headers = {k: str(msg[k]) for k in ("from", "to", "subject", "date")
                   if msg[k] is not None}
        part = msg.get_body(preferencelist=("plain", "html"))
        body = part.get_content().strip() if part is not None else ""
    elif ext == ".msg":
        try:
            import extract_msg  # optional, lazy
        except ImportError:
            return [Element(
                element_id=f"{os.path.basename(path)}-empty",
                modality=Modality.IMAGE, content="",
                source_document=path, page=1, team=team,
                report_type=report_type, entity_ref=entity_ref)]
        m = extract_msg.Message(path)
        headers = {"from": m.sender or "", "to": m.to or "",
                   "subject": m.subject or "", "date": str(m.date or "")}
        body = (m.body or "").strip()

    elements: list[Element] = []
    # header block as its own text element (searchable context: who/when/subject)
    header_text = "\n".join(f"{k.title()}: {v}" for k, v in headers.items() if v)
    if header_text:
        elements.append(_text_element(
            path, 1, 0, header_text, team, report_type, entity_ref,
            suffix="hdr"))
    # body as paragraphs (footnote-aware split reused; emails rarely have
    # footnotes but the whole-block behaviour is harmless)
    for i, para in enumerate(_split_paragraphs(body), start=1):
        elements.append(_text_element(
            path, 1, i, para, team, report_type, entity_ref, suffix="body"))
    return elements


HANDLERS: dict[str, Callable[..., list[Element]]] = {
    ".pdf": preprocess_pdf,
    ".docx": preprocess_docx,
    ".xlsx": preprocess_xlsx,
    ".xlsm": preprocess_xlsx,
    ".pptx": preprocess_pptx,
    ".eml": preprocess_email,
    ".msg": preprocess_email,
}


def preprocess(path: str, team: str, report_type: str,
               entity_ref: Optional[str] = None,
               ocr_client: Optional[AIParseClient] = None) -> list[Element]:
    ext = os.path.splitext(path)[1].lower()
    if ext not in HANDLERS:
        raise ValueError(
            f"No preprocessor for {ext!r}. Add a handler or quarantine the file."
        )
    # Only the PDF handler performs OCR fallback; other formats have a real
    # text layer by construction, so the OCR client is not threaded to them.
    if ext == ".pdf":
        elements = preprocess_pdf(path, team, report_type, entity_ref,
                                  ocr_client=ocr_client)
    else:
        elements = HANDLERS[ext](path, team, report_type, entity_ref)
    _link_titles_to_tables(elements)
    return elements


def _link_titles_to_tables(elements: list) -> None:
    """Attach a nearby TITLE to each table as context, so extraction can prefix
    its metrics with what the table is about (e.g. 'FINANCIAL RISK | <row> |
    AMOUNT = 500'). Titles are often a SEPARATE heading/text element above the
    table (and sometimes absent). We link the closest short, heading-like text
    element that sits DIRECTLY ABOVE a table on the same page. Mutates elements
    in place, setting each table's ``description`` to the linked title.

    Purely positional: uses page + bbox top-edge. If a table has no bbox or no
    heading above it, it is left as-is (no title — the table still extracts on
    its own headers).
    """
    from ..shared.schema import Modality

    def _mod(e):
        return e.modality.value if hasattr(e.modality, "value") else str(e.modality)

    def _is_titleish(text: str) -> bool:
        t = (text or "").strip()
        if not t or len(t) > 80:          # titles are short
            return False
        # a title is mostly words, few digits, not a full sentence of data
        return len(t.split()) <= 10

    def _has_real_bbox(e):
        # docx uses a placeholder (0,0,1,1); treat that as "no layout bbox".
        return e.bbox is not None and e.bbox != (0.0, 0.0, 1.0, 1.0)

    # POSITIONAL linking for tables with a real layout bbox (PDF/PPTX).
    texts = [e for e in elements if _mod(e) == Modality.TEXT.value
             and _is_titleish(e.content)]
    for tbl in elements:
        if _mod(tbl) != Modality.TABLE.value or tbl.description:
            continue
        if not _has_real_bbox(tbl):
            continue
        t_top = tbl.bbox[1]               # normalised top edge of the table
        best = None
        best_gap = 1.0
        for txt in texts:
            if not _has_real_bbox(txt) or txt.page != tbl.page:
                continue
            gap = t_top - txt.bbox[3]     # positive if the text is ABOVE the table
            if 0 <= gap < 0.15 and gap < best_gap:
                best, best_gap = txt, gap
        if best is not None:
            tbl.description = best.content.strip()

    # ORDER-BASED linking for tables without a real bbox (DOCX): the heading-like
    # text element immediately before the table in document order is its title.
    for i, tbl in enumerate(elements):
        if _mod(tbl) != Modality.TABLE.value or tbl.description:
            continue
        if _has_real_bbox(tbl):
            continue                       # already handled positionally
        for j in range(i - 1, -1, -1):
            prev = elements[j]
            pm = _mod(prev)
            if pm == Modality.TEXT.value and _is_titleish(prev.content):
                tbl.description = prev.content.strip()
                break
            if pm == Modality.TABLE.value:
                break                      # hit another table -> no title here


# --- helpers ---------------------------------------------------------------

# A footnote marker at the start of a line: (a), (1), (i) etc. Used to detect
# the footnote region so its parts are kept together as one chunk rather than
# fragmented by blank-line splitting.
_FOOTNOTE_MARKER = re.compile(r"^\s*\(([a-z0-9]{1,3})\)\s", re.IGNORECASE)


def _footnote_markers(text: str) -> list[str]:
    """Return the footnote marker ids present at line-starts in ``text`` — e.g.
    a table containing '(a)' and '(f)' references. Used to link a table chunk
    to its footnote chunk even if they end up apart (see production indexing)."""
    out = []
    for line in text.splitlines():
        m = _FOOTNOTE_MARKER.match(line)
        if m:
            mid = m.group(1).lower()
            if mid not in out:
                out.append(mid)
    return out


def _split_paragraphs(text: str) -> list[str]:
    """Footnote-aware paragraph split (text-only; no coordinates).

    Splits the body on blank lines as before, BUT once the footnote region
    begins (first line starting with a ``(x)`` marker) everything from there to
    the end is kept together as a SINGLE paragraph, so a multi-part footnote
    block (a)(b)(c)... survives whole as one chunk instead of fragmenting.
    """
    lines = text.split("\n")
    footnote_start = None
    for i, line in enumerate(lines):
        if _FOOTNOTE_MARKER.match(line):
            footnote_start = i
            break

    if footnote_start is None:
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    body = "\n".join(lines[:footnote_start])
    footnotes = "\n".join(lines[footnote_start:]).strip()
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    if footnotes:
        paras.append(footnotes)          # whole footnote region = one chunk
    return paras


def _paragraphs_with_bbox(page) -> list[tuple[str, Optional[tuple]]]:
    """Build (paragraph_text, normalised_bbox) pairs from a pdfplumber page.

    Uses word coordinates so each paragraph carries its position on the page
    (normalised 0-1, matching how figure bboxes are stored). Falls back to
    ``(text, None)`` if word extraction is unavailable, so behaviour degrades
    to the previous text-only path rather than failing.
    """
    text = page.extract_text() or ""
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    try:
        words = page.extract_words() or []
        pw, ph = float(page.width), float(page.height)
    except Exception:
        return [(p, None) for p in paragraphs]

    if not words or not pw or not ph:
        return [(p, None) for p in paragraphs]

    # For each paragraph, consume its token count from the word stream and take
    # the bounding box of that span. Approximate (word order ~ reading order),
    # but good enough to locate a paragraph/footnote on the page.
    out: list[tuple[str, Optional[tuple]]] = []
    cursor = 0
    for para in paragraphs:
        para_tokens = para.split()
        if not para_tokens:
            out.append((para, None))
            continue
        span = words[cursor:cursor + len(para_tokens)]
        cursor += len(para_tokens)
        if not span:
            out.append((para, None))
            continue
        x0 = min(float(w["x0"]) for w in span) / pw
        top = min(float(w["top"]) for w in span) / ph
        x1 = max(float(w["x1"]) for w in span) / pw
        bottom = max(float(w["bottom"]) for w in span) / ph
        out.append((para, (x0, top, x1, bottom)))
    return out


def _table_to_markdown(table: list[list]) -> str:
    rows = [[("" if c is None else str(c)) for c in row] for row in table]
    return _rows_to_markdown(rows)


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """Render rows as markdown, tolerating ragged rows.

    Real PDF tables from pdfplumber are frequently ragged; we pad short rows
    and never let a long row desync the column count.
    """
    if not rows:
        return ""
    width = max(len(r) for r in rows)

    def pad(r: list[str]) -> list[str]:
        return list(r) + [""] * (width - len(r))

    header = pad(rows[0])
    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * width) + "|"]
    for row in rows[1:]:
        md.append("| " + " | ".join(pad(row)) + " |")
    return "\n".join(md)
