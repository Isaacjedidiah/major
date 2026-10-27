"""Missed-crop detection + recovery for chart reading.

The vision model (Sonnet) is non-deterministic: a crop it reads on one run can
return NOTHING on another, even when the crop is good. In production we cannot
eyeball this, so we:

  1. RETRY an empty chart read a few times (an intermittent empty usually
     succeeds on retry, since the chart IS readable).
  2. If it is STILL empty, record it as a MISS -- log it AND save the crop image
     to a recovery folder, tagged by document + page + figure, so nothing is
     silently lost.
  3. A separate RECOVERY run re-reads the saved crops (the non-determinism means
     most now succeed) and returns their claims; any still-missed crop is left in
     the folder for human review.

Honest stance: the model can miss; we detect it, log it, and re-run it.
"""
from __future__ import annotations

import os
import re
import json
import base64
import uuid
from typing import Optional, List


EMPTY_READ_RETRIES = 2          # retry an empty chart read this many times


def _local(path: str) -> str:
    if path.startswith("/Volumes"):
        return path.replace("/Volumes", "/dbfs/Volumes")
    if path.startswith("dbfs:"):
        return path.replace("dbfs:", "/dbfs")
    return path


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))[:80]


def read_chart_with_retry(read_fn, retries: int = EMPTY_READ_RETRIES):
    """Call read_fn() (which returns a points list or falsy). If it returns
    nothing, retry up to `retries` times -- an intermittent empty read usually
    succeeds on retry. Returns (points, attempts). points may still be empty."""
    attempts = 0
    pts = None
    while attempts <= retries:
        attempts += 1
        try:
            pts = read_fn()
        except Exception:
            pts = None
        if pts:
            return pts, attempts
    return (pts or []), attempts


def save_missed_crop(img_b64: str, missed_root: str, document: str,
                     page, figure_index, reason: str = "empty_read") -> Optional[str]:
    """Save a missed crop image to the recovery folder, named for traceability:
    <missed_root>/<document>/page_<page>_fig_<idx>.png. Returns the path, or None
    on failure (never raises -- recovery is best-effort)."""
    try:
        folder = f"{missed_root.rstrip('/')}/{_safe(document)}"
        os.makedirs(_local(folder), exist_ok=True)
        pg = page if page is not None else "x"
        fname = f"page_{pg}_fig_{figure_index}.png"
        path = f"{folder}/{fname}"
        with open(_local(path), "wb") as f:
            f.write(base64.b64decode(img_b64))
        # a small sidecar json records the reason + metadata for the log
        meta = {"document": document, "page": page, "figure_index": figure_index,
                "reason": reason}
        with open(_local(path + ".json"), "w") as f:
            json.dump(meta, f)
        return path
    except Exception:
        return None


def iter_missed_crops(missed_root: str):
    """Yield (document, page, figure_index, image_path) for each saved missed
    crop, reading the metadata from the sidecar json. image_path is a Volume
    path for Spark/base64 loading."""
    root = _local(missed_root)
    if not os.path.isdir(root):
        return
    for doc in sorted(os.listdir(root)):
        dfolder = os.path.join(root, doc)
        if not os.path.isdir(dfolder):
            continue
        for fn in sorted(f for f in os.listdir(dfolder) if f.endswith(".png")):
            meta_path = os.path.join(dfolder, fn + ".json")
            meta = {}
            if os.path.exists(meta_path):
                try:
                    meta = json.load(open(meta_path))
                except Exception:
                    meta = {}
            vol = os.path.join(dfolder, fn)
            if vol.startswith("/dbfs/Volumes"):
                vol = vol.replace("/dbfs/Volumes", "/Volumes")
            yield (meta.get("document", doc), meta.get("page"),
                   meta.get("figure_index"), vol)


def recover_missed_crops(missed_root, run_async_fn, llm_client, model_key,
                         chart_prompt, extraction_tool_schema,
                         claim_cls=None, retries: int = EMPTY_READ_RETRIES) -> List:
    """RECOVERY run: re-read every saved missed crop (the model's non-determinism
    means most now succeed). Returns recovered claims, tagged with document+page
    from the saved metadata. Crops still empty after retries are left in the
    folder for human review (not deleted). Best-effort; never raises per crop."""
    from .hybrid_extraction import sonnet_points_to_claims
    recovered = []
    for document, page, fig_idx, image_path in iter_missed_crops(missed_root):
        try:
            with open(_local(image_path), "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()
        except Exception:
            continue

        def _read():
            resp = run_async_fn(llm_client.extract(
                model_key, chart_prompt, "", extraction_tool_schema,
                image_base64=img_b64))
            ta = getattr(resp, "tool_arguments", None)
            if isinstance(ta, dict):
                return ta.get("claims") or ta.get("points") or []
            txt = getattr(resp, "text", "") or ""
            m = re.search(r"\{.*\}", txt, re.S)
            if m:
                try:
                    d = json.loads(m.group(0))
                    return d.get("claims") or d.get("points") or []
                except Exception:
                    return []
            return []

        pts, _ = read_chart_with_retry(_read, retries=retries)
        if pts:
            # build claims; tag with the saved document + page
            class _El:  # minimal element carrier for sonnet_points_to_claims
                pass
            el = _El(); el.page = page; el.el_type = "figure"
            el.content = ""; el.bbox = None; el.figure_index = fig_idx
            claims = sonnet_points_to_claims(pts, el, document, None)
            for c in claims:
                setattr(c, "source_document", document)
                setattr(c, "page", page)
                setattr(c, "model_used", "sonnet_recovery")
            recovered.extend(claims)
            # mark as recovered (rename the png so it isn't re-run next time)
            try:
                os.rename(_local(image_path), _local(image_path + ".recovered"))
            except Exception:
                pass
        else:
            # STILL empty after the recovery retries -> the automated levels
            # (extraction + missed-crop recovery) could not resolve this chart.
            # Move it to a human_in_the_loop sub-folder within the SAME run folder
            # so users can review it. Preserves the document/page metadata.
            try:
                base = os.path.dirname(_local(image_path))        # .../<document>
                hitl = os.path.join(os.path.dirname(base),
                                    "human_in_the_loop", os.path.basename(base))
                os.makedirs(hitl, exist_ok=True)
                dest = os.path.join(hitl, os.path.basename(_local(image_path)))
                os.rename(_local(image_path), dest)
                # move the sidecar metadata too
                meta_src = _local(image_path) + ".json"
                if os.path.exists(meta_src):
                    os.rename(meta_src, dest + ".json")
            except Exception:
                pass
    return recovered
