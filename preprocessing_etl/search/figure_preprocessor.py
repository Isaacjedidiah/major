"""Crop-and-zoom preprocessing for chart/figure extraction.

Chart extraction is a known weak spot for vision models: a chart's axis
labels and gridlines are small relative to a full page, so the model must
both LOCATE and READ the chart in one pass. Cropping to the figure's
bounding box and upscaling turns that into just READ — the highest-leverage
fix before reaching for a different model or a chart-digitisation tool.
(Adopted from the uploaded figure_preprocessor; the design is unchanged.)

What differs here from the upload: the upload left its page-image source
(`DatabricksPageImageProvider`) as ``NotImplementedError``, so crop-and-zoom
never actually ran in their production. This package already rasterises PDF
pages with pdfplumber (in the OCR fallback) and depends on Pillow, so we
supply a WORKING provider — ``PdfPlumberPageImageProvider`` — and the
technique runs for real on PDF sources. An Azure/Databricks provider can be
added later for non-PDF page renders.

IMPORTANT: the image handed to the MODEL is clean — no highlight box drawn
on it (that would bias the model's own reading). Drawing a visible box for a
HUMAN reviewer is a separate concern (see reviewer/bbox rendering).
"""
from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional

DEFAULT_PADDING = 0.02            # fraction of page dim added around bbox each side
DEFAULT_MIN_OUTPUT_DIMENSION = 1024  # upscale smaller cropped dim to at least this
DEFAULT_MAX_OUTPUT_DIMENSION = 2000  # cap to bound image size / token cost


class PageImageProvider(ABC):
    """Abstraction over where a rendered page image comes from."""

    @abstractmethod
    def get_page_image(self, document_bytes: bytes, filename: str,
                       page_number: int):
        """Return a PIL.Image for the page, or None if unavailable."""


class PdfPlumberPageImageProvider(PageImageProvider):
    """Working provider for PDF sources, backed by pdfplumber's page renderer.

    This is the piece the uploaded codebase left unimplemented. pdfplumber is
    already a dependency (used by the preprocessor / OCR fallback), so no new
    requirement is introduced. Returns None for non-PDF inputs or if the page
    can't be rendered, so callers degrade to text-only rather than fail.
    """

    def __init__(self, resolution: int = 200):
        self._resolution = resolution

    def get_page_image(self, document_bytes: bytes, filename: str,
                       page_number: int):
        if not filename.lower().endswith(".pdf"):
            return None
        try:
            import pdfplumber  # lazy
            from PIL import Image  # noqa: F401 (ensures Pillow present)

            with pdfplumber.open(BytesIO(document_bytes)) as pdf:
                if page_number < 1 or page_number > len(pdf.pages):
                    return None
                page = pdf.pages[page_number - 1]
                pil = page.to_image(resolution=self._resolution).original
                return pil
        except Exception:
            return None


def crop_and_zoom(page_image, bbox, padding: float = DEFAULT_PADDING,
                  min_output_dimension: int = DEFAULT_MIN_OUTPUT_DIMENSION,
                  max_output_dimension: int = DEFAULT_MAX_OUTPUT_DIMENSION):
    """Crop ``page_image`` to ``bbox`` (normalised [x0,y0,x1,y1], 0-1) plus
    padding, then upscale so the smaller cropped dimension reaches
    ``min_output_dimension`` px — never past ``max_output_dimension``.

    Padding matters: charts often have axis labels or legends just outside
    the plot area, so cropping tight to the bbox risks cutting off the labels
    needed to read the chart.
    """
    from PIL import Image

    width, height = page_image.size
    x0, y0, x1, y1 = bbox

    px0 = int(max(0.0, x0 - padding) * width)
    py0 = int(max(0.0, y0 - padding) * height)
    px1 = int(min(1.0, x1 + padding) * width)
    py1 = int(min(1.0, y1 + padding) * height)
    px1, py1 = max(px1, px0 + 1), max(py1, py0 + 1)  # guard zero-area crop

    cropped = page_image.crop((px0, py0, px1, py1))
    crop_w, crop_h = cropped.size
    smaller = min(crop_w, crop_h)

    scale = 1.0
    if smaller < min_output_dimension:
        scale = min_output_dimension / smaller
    if max(crop_w, crop_h) * scale > max_output_dimension:
        scale = max_output_dimension / max(crop_w, crop_h)

    if scale != 1.0:
        new_size = (max(1, round(crop_w * scale)), max(1, round(crop_h * scale)))
        cropped = cropped.resize(new_size, Image.LANCZOS)
    return cropped


def enhance_for_legibility(image, is_table: bool = False):
    """Modestly improve an image's legibility for a vision model, WITHOUT
    destroying data. Calibrated to help scanned/small image-tables and charts:

      * upscale small images so digits are large enough to read reliably;
      * a gentle contrast + sharpness boost so faint numbers stand out.

    Deliberately conservative — aggressive processing (hard binarisation, heavy
    sharpening, grayscale on colour charts) can DESTROY faint values or the
    colour coding that distinguishes chart series, so we avoid those. Returns
    the original image unchanged on any error.
    """
    try:
        from PIL import Image, ImageEnhance
        w, h = image.size

        # 1. upscale small images — the biggest legibility win for tiny crops.
        #    Target a comfortable minimum long side; don't upscale huge images.
        _MIN_SIDE = 900
        longest = max(w, h)
        if 0 < longest < _MIN_SIDE:
            scale = _MIN_SIDE / float(longest)
            image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                                  Image.LANCZOS)   # high-quality upscaling

        # 2. gentle contrast + sharpness — helps faint digits, low risk.
        image = ImageEnhance.Contrast(image).enhance(1.15)
        image = ImageEnhance.Sharpness(image).enhance(1.3)
        return image
    except Exception:
        return image


def encode_image_base64(image, fmt: str = "PNG", enhance: bool = False,
                        is_table: bool = False) -> str:
    """Base64-encode a PIL image for a multimodal LLM image block.

    Normalises the image so the model can actually process it (avoids the
    "Could not process image" 400): converts to RGB (models reject some modes
    like CMYK/RGBA/P), and downsizes if either dimension exceeds a safe cap
    (very large crops — full pages at high DPI — are a common cause of the
    rejection). Small/oversized/odd-mode images are the usual culprits for PDF
    and Word chart crops.

    When ``enhance`` is True, applies modest legibility enhancement (upscale
    small images, gentle contrast/sharpen) before encoding — helps the model
    read faint or small image-tables/charts.
    """
    # 1. convert mode: models want RGB; RGBA/CMYK/P often fail
    try:
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
    except Exception:
        pass

    # 2. optional legibility enhancement (before the size cap, so upscaling of a
    #    small image isn't immediately undone).
    if enhance:
        image = enhance_for_legibility(image, is_table=is_table)

    # 3. downsize if too large. Keep the longest side within a safe bound so the
    #    encoded image stays well under model limits (both pixel and byte).
    _MAX_SIDE = 1568  # a safe long-side cap for vision models
    try:
        w, h = image.size
        if max(w, h) > _MAX_SIDE:
            scale = _MAX_SIDE / float(max(w, h))
            image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    except Exception:
        pass

    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def prepare_figure_image(provider: PageImageProvider, document_bytes: bytes,
                         filename: str, page_number: Optional[int],
                         bbox) -> Optional[object]:
    """Full crop-and-zoom for one figure element. Returns a PIL image, or None
    if there's no bbox/page to crop (caller falls back to text-only, never
    fails)."""
    if bbox is None or page_number is None:
        return None
    page_image = provider.get_page_image(document_bytes, filename, page_number)
    if page_image is None:
        return None
    return crop_and_zoom(page_image, bbox)


_CHART_KEYWORDS = None


def is_chart_like(element) -> bool:
    """True if a FIGURE element should be read as an image.

    A FIGURE is a picture/chart with no text layer (content is empty), so it
    inherently needs visual reading — we don't require a 'chart' keyword in a
    description (parsers often don't provide one). If a description IS present,
    a chart keyword still counts; but a bare FIGURE with a bbox is enough,
    because there's no text to extract from otherwise.
    """
    import re
    from ..shared.schema import Modality

    global _CHART_KEYWORDS
    if _CHART_KEYWORDS is None:
        _CHART_KEYWORDS = re.compile(
            r"\b(chart|graph|plot|trend line|bar chart|pie chart|line graph|"
            r"histogram|axis)\b", re.IGNORECASE)

    modality = element.modality
    modality = modality.value if isinstance(modality, Modality) else str(modality)
    if modality != Modality.FIGURE.value:
        return False
    # A FIGURE with a description mentioning a chart -> yes.
    if element.description and _CHART_KEYWORDS.search(element.description):
        return True
    # A FIGURE with no text content is a picture that can only be read visually
    # -> read it as an image. It qualifies if we have EITHER embedded image
    # bytes (PPTX/DOCX pictures — the image is right there, no crop needed) OR a
    # bbox to crop from a rendered page (PDF). Requiring a bbox alone wrongly
    # dropped embedded-image figures that carry image_bytes but no layout bbox.
    if not (element.content or "").strip() and (
            element.bbox is not None
            or getattr(element, "image_bytes", None)):
        return True
    return False


def is_complex_table(element) -> bool:
    """Heuristic: a TABLE element whose structure flattens badly to markdown.

    Multi-level / nested column headers (e.g. a group header spanning several
    sub-columns) collapse when pdfplumber flattens the grid to markdown, so
    the model loses which sub-column a value belongs to. For those we read the
    table AS AN IMAGE instead — the model then SEES the header hierarchy rather
    than parsing an ambiguous flattened grid.

    Signals (any is sufficient):
      * has_merged_cells — the parser flagged merged/spanning header cells.
      * The markdown has several leading rows that look like headers before the
        numeric body begins (a proxy for a multi-level header band).
      * The table is wide (many columns) AND has a header row with repeated
        sub-labels (Exposure/RWA repeated under each group) — the classic
        two-level financial table.
    """
    from ..shared.schema import Modality

    modality = element.modality
    modality = modality.value if isinstance(modality, Modality) else str(modality)
    if modality != Modality.TABLE.value:
        return False

    if getattr(element, "has_merged_cells", False):
        return True

    content = element.content or ""
    lines = [ln for ln in content.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False

    import re
    from collections import Counter

    # A row is "header-like" if it has FEW real data numbers. Crucially, DATE
    # tokens (2024, Q1 2024, 31-Mar-26, Apr-25) are headers, not data — many
    # regulatory/time-series tables have DATE column headers, which naive digit
    # counting mistakes for a data row (breaking multi-header detection). So we
    # strip date-like tokens before counting digits.
    _DATE_TOK = re.compile(
        r"\b(?:q[1-4]|h[12])[- ]?\d{2,4}\b"
        r"|\b\d{1,2}[-/ ](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/ ]?\d{0,4}\b"
        r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[-/ ]?\d{2,4}\b"
        r"|\b(19|20)\d{2}\b"
        r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.I)
    num_re = re.compile(r"\d")

    def _is_header_like(ln: str) -> bool:
        stripped = _DATE_TOK.sub("", ln)      # remove dates before counting
        return len(num_re.findall(stripped)) <= 1

    # 1. Multiple leading header-like rows = a multi-level header band. Check the
    #    first 5 rows (a deep header band is common in regulatory templates).
    header_band = 0
    for ln in lines[:5]:
        if _is_header_like(ln):
            header_band += 1
        else:
            break
    if header_band >= 2:
        return True

    # 2. WIDE table: many columns is itself a strong signal that a flat markdown
    #    grid will be ambiguous. Estimate column count from the header row's
    #    separators (| or multiple spaces / tabs).
    header = lines[0]
    cols = max(header.count("|"),
               len(re.split(r"\s{2,}|\t", header.strip())))
    if cols >= 6:
        return True

    # 3. Repeated header sub-labels across the header rows (two-level header:
    #    the same sub-column label repeated under each group). Check the first
    #    THREE lines (bands may push the repeated labels to line 2 or 3).
    header_text = " ".join(lines[:3]).lower()
    words = re.findall(r"[a-z]{3,}", header_text)
    if words:
        counts = Counter(words)
        _STOP = {"the", "and", "for", "per", "of", "to", "in", "by", "total",
                 "value", "average"}
        repeated = [w for w, n in counts.items() if n >= 2 and w not in _STOP]
        if repeated:
            return True

    # 4. Repeated DATE headers across a row (e.g. the same set of quarter dates
    #    appearing under two value-bands) — a hallmark of the wide two-band
    #    regulatory table. If a date token appears 2+ times in the header rows,
    #    the table has parallel period columns under grouping bands.
    date_tokens = _DATE_TOK.findall(" ".join(lines[:3]))
    if date_tokens:
        # findall returns tuples for grouped patterns; count distinct matches by
        # re-scanning for spans instead.
        spans = _DATE_TOK.findall(" ".join(lines[:3]))
        if len(spans) >= 2:
            return True

    return False


def needs_image_read(element) -> bool:
    """True if this element should be read multimodally (as a cropped image):
    chart-like figures OR structurally complex tables."""
    return is_chart_like(element) or is_complex_table(element)
