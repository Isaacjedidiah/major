"""PDF page SHARPENING pre-processing for ai_parse_document.

Poor page-image quality (soft / low-contrast scans) makes ai_parse_document
misread text (e.g. 'SSE' -> 'S&C') and can distort table-structure detection.
Applying an image SHARPNESS enhancement to each page BEFORE ai_parse reads it
fixes these at the SOURCE — deterministically — which is far better than any
downstream compensation.

This module is PDF-ONLY (PPTX stays on the normal native path). It works in two
steps so the brightening is separate from, and does not disturb, the existing
extraction:

  1. sharpen_pdf_to_folder(pdf_path, out_root)
        Renders each PDF page, applies SHARPNESS, and writes
        <out_root>/<document_name>/page_<NNN>.png  (page_001 = first page).
        The folder is the EXACT document name (only '/' -> '_', the one
        filesystem-forbidden char) for full traceability.

  2. run_ai_parse_on_image(spark, image_path, image_output_path)
        Runs ai_parse_document on ONE sharpened page image and returns the same
        VARIANT the normal pipeline consumes — so the existing
        parse_ai_parse_response / hybrid_extract_document path is unchanged.

  iter_sharpened_pages(out_root)
        Yields (source_document, page, image_path) for every sharpened page, so
        a batch loop can parse each image and tag claims with the document + page
        (document from the folder, page from the file name).

Tunables: SHARPNESS (the enhancement factor) and RENDER_DPI (page render
resolution). SHARPNESS is set to the value found to fix the misreads.
"""
from __future__ import annotations

import io
import os
import re
import uuid
import json
from typing import Iterator, Tuple, List


# --- tunables -------------------------------------------------------------
SHARPNESS = 35.0        # PIL Sharpness factor found to fix the misreads
RENDER_DPI = 200        # page render resolution before sharpening


def _local(path: str) -> str:
    """Map a path to a LOCAL filesystem path for PIL / PyMuPDF / open().

    Unity Catalog Volumes are accessed DIRECTLY at /Volumes/... with normal
    Python file I/O -- they are NOT prefixed with /dbfs (that is legacy DBFS and
    raises '/dbfs/Volumes is not supported'). Only old dbfs: URIs map to /dbfs.
    """
    if path.startswith("/Volumes"):
        return path                       # UC Volume: use the path AS-IS
    if path.startswith("dbfs:/Volumes"):
        return path.replace("dbfs:", "")  # dbfs:/Volumes -> /Volumes
    if path.startswith("dbfs:"):
        return path.replace("dbfs:", "/dbfs")   # legacy DBFS only
    return path


def _vol(path: str) -> str:
    """Return the Volume path for Spark READ_FILES. UC Volume paths are already
    /Volumes/...; strip a legacy /dbfs prefix if one is present."""
    if path.startswith("/dbfs/Volumes"):
        return path.replace("/dbfs/Volumes", "/Volumes")
    return path


def _safe_folder(document_name: str) -> str:
    """Keep the document name EXACTLY; only replace '/' (the single filesystem-
    forbidden char in a path segment) with '_'. Nothing else is altered."""
    return document_name.replace("/", "_")


def _page_no_from_name(filename: str):
    m = re.search(r"page_(\d+)\.png$", filename)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# STEP 1 — sharpen a PDF's pages into a per-document folder
# ---------------------------------------------------------------------------
def sharpen_pdf_to_folder(pdf_path: str, out_root: str,
                          sharpness: float = SHARPNESS,
                          dpi: int = RENDER_DPI) -> str:
    """Render each page of the PDF, apply SHARPNESS, and save to
    <out_root>/<document_name>/page_<NNN>.png (1-based). Returns the folder path.
    Requires PyMuPDF (fitz) and Pillow."""
    import fitz                                   # PyMuPDF
    from PIL import Image, ImageEnhance

    document_name = os.path.basename(pdf_path)    # EXACT name, with extension
    folder = os.path.join(out_root, _safe_folder(document_name))
    os.makedirs(_local(folder), exist_ok=True)

    doc = fitz.open(_local(pdf_path))
    try:
        for i in range(doc.page_count):
            pix = doc[i].get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img = ImageEnhance.Sharpness(img).enhance(sharpness)   # ONLY sharpness
            fname = f"page_{i + 1:03d}.png"                        # page_001...
            img.save(_local(os.path.join(folder, fname)), format="PNG")
        n = doc.page_count
    finally:
        doc.close()
    return folder


def sharpen_pdfs(pdf_paths: List[str], out_root: str,
                 sharpness: float = SHARPNESS, dpi: int = RENDER_DPI) -> List[str]:
    """Sharpen a list of PDFs. Returns the list of folders written."""
    return [sharpen_pdf_to_folder(p, out_root, sharpness, dpi) for p in pdf_paths]


# ---------------------------------------------------------------------------
# iterate the sharpened pages (document from folder, page from file name)
# ---------------------------------------------------------------------------
def iter_sharpened_pages(out_root: str) -> Iterator[Tuple[str, int, str]]:
    """Yield (source_document, page, image_volume_path) for every sharpened page.
    source_document = the folder name (the exact document name); page is 1-based
    from the file name. image path is returned as a Volume path for Spark."""
    root_local = _local(out_root)
    if not os.path.isdir(root_local):
        return
    for document_name in sorted(os.listdir(root_local)):
        folder_local = os.path.join(root_local, document_name)
        if not os.path.isdir(folder_local):
            continue
        for fname in sorted(f for f in os.listdir(folder_local) if f.endswith(".png")):
            page = _page_no_from_name(fname)
            if page is None:
                continue
            yield document_name, page, _vol(os.path.join(folder_local, fname))


# ---------------------------------------------------------------------------
# STEP 2 — ai_parse ONE sharpened image (returns the same VARIANT as normal)
# ---------------------------------------------------------------------------
def run_ai_parse_on_image(spark, image_path: str, image_output_path: str = None):
    """Run ai_parse_document on a single (sharpened) page image. Returns the
    same VARIANT value the normal pipeline consumes, so downstream parsing is
    unchanged. A unique temp view avoids Spark plan caching (same precaution as
    the main _run_ai_parse)."""
    view = f"_aip_sharp_{uuid.uuid4().hex}"
    df = spark.read.format("binaryFile").load(image_path)
    df.createOrReplaceTempView(view)
    if image_output_path:
        return spark.sql(
            f"SELECT ai_parse_document(content, "
            f"map('version','2.0','imageOutputPath','{image_output_path}')) AS p "
            f"FROM {view}"
        ).collect()[0]["p"]
    return spark.sql(
        f"SELECT ai_parse_document(content, map('version','2.0')) AS p FROM {view}"
    ).collect()[0]["p"]


# ---------------------------------------------------------------------------
# SHARPEN-ONLY batch (Design B): sharpen the PDFs in a submission list into a
# per-run folder of page images. NO extraction here -- ai_parse_pipeline then
# consumes these images. Keeping sharpening separate from extraction means a
# sharpen issue never touches the proven extraction/storage path.
# ---------------------------------------------------------------------------
def sharpen_submissions(submissions, sharpen_root, *, run_id=None,
                        sharpness=SHARPNESS, dpi=RENDER_DPI, dbutils=None):
    """Sharpen the PDF submissions into <sharpen_root>/<run_id>/<document>/page_<NNN>.png.
    Returns {run_id, root, documents: [{source_document, folder, pages}]}.
    Non-PDF submissions are ignored (they go through ai_parse_pipeline normally).
    The caller passes the SAME run_id to ai_parse_pipeline so it finds these images."""
    import os as _os
    rid = run_id or uuid.uuid4().hex
    run_root = f"{sharpen_root.rstrip('/')}/{rid}"
    out = {"run_id": rid, "root": run_root, "documents": []}
    for sub in submissions:
        path = sub["path"] if isinstance(sub, dict) else sub
        if not str(path).lower().endswith(".pdf"):
            continue                                  # PDFs only
        folder = sharpen_pdf_to_folder(path, run_root, sharpness, dpi)
        pages = sorted(f for f in _os.listdir(_local(folder)) if f.endswith(".png"))
        out["documents"].append({
            "source_document": _os.path.basename(path),
            "folder": _vol(folder) if folder.startswith("/dbfs") else folder,
            "pages": len(pages),
        })
    return out
