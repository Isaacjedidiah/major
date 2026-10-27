"""Batch ingestion entry point (custom track).

Discovers submissions under the volume and hands them to the orchestrator.
Invoked by a Databricks Job / Workflow.

Layout: <root>/<team>/<report_type>/<file>. The orchestration scans EVERY
team folder, then EVERY report_type folder under it, and processes every
document found. entity_ref is NOT in the path — it is resolved from the
filename convention (entityID_entityName_documentTitle_reportDate.ext) and stays
a column in the data. A filename that doesn't parse still processes; its
entity_ref is left None and handled by the existing quarantine/UNKNOWN path
rather than crashing the run.

Idempotency comes from content-hash dedup in storage, so re-scanning already
processed files is a no-op.
"""
from __future__ import annotations

import asyncio
import argparse
import os
import re

from ..search.ai_search_store import AzureAISearchStore
from ..shared.config import CONFIG
from .production import process_batch

# entityID_entityName_documentTitle_reportDate.ext  (reportDate = YYYY-MM-DD)
_FILENAME_RE = re.compile(
    r"^(?P<entity_ref>[A-Za-z0-9]+)_[A-Za-z0-9\-]+_[A-Za-z0-9\-]+_"
    r"\d{4}-\d{2}-\d{2}\.[A-Za-z0-9]+$")


def entity_ref_from_filename(filename: str) -> str | None:
    """Extract entity_ref from the filename convention, or None if it doesn't
    match (the document still processes; attribution routes to review)."""
    m = _FILENAME_RE.match(filename.strip())
    return m.group("entity_ref") if m else None


def _normalise_entity(folder: str) -> str:
    """Normalise an entity folder name so casing/spacing differences don't
    fragment one entity into several (ACME_CORP vs 'Acme Corp'). Trim, collapse
    whitespace, and keep the given form otherwise — teams file under a canonical
    entity folder; this only guards against trivial variance."""
    return re.sub(r"\s+", " ", folder.strip())


def discover_submissions(root: str) -> list[dict]:
    """Scan <root>/<team>/<report_type>/<entity>/<entity_id>/<entity_special_name>/<file>.

    Attribution comes from the PATH, not the filename: a file is only collected
    when it sits inside a complete team/report_type/entity/entity_id/
    entity_special_name/ path, so every discovered file is fully attributed by
    construction. The walk descends the full entity hierarchy to find work, so a
    file dropped at a shallower level (no full entity path) is simply not
    collected.

    team, report_type, entity, entity_id and entity_name (the special name) are
    all taken from the PATH folder names. The filename is free-form.
    """
    subs: list[dict] = []
    if not os.path.isdir(root):
        return subs

    def _subdirs(d):
        try:
            return sorted(x for x in os.listdir(d)
                          if os.path.isdir(os.path.join(d, x)) and not x.startswith("."))
        except OSError:
            return []

    for team in _subdirs(root):
        team_dir = os.path.join(root, team)
        for report_type in _subdirs(team_dir):
            rt_dir = os.path.join(team_dir, report_type)
            for entity in _subdirs(rt_dir):
                entity_dir = os.path.join(rt_dir, entity)
                for entity_id in _subdirs(entity_dir):
                    eid_dir = os.path.join(entity_dir, entity_id)
                    for entity_name in _subdirs(eid_dir):
                        leaf = os.path.join(eid_dir, entity_name)
                        for fname in sorted(os.listdir(leaf)):
                            fpath = os.path.join(leaf, fname)
                            if fname.startswith(".") or not os.path.isfile(fpath):
                                continue
                            subs.append({
                                "path": fpath,
                                "entity_ref": _normalise_entity(entity),
                                "entity": entity,
                                "entity_id": entity_id,
                                "entity_name": entity_name,
                                "team": team,
                                "report_type": report_type,
                            })
    return subs


def main() -> None:
    parser = argparse.ArgumentParser(description="the pipeline regdata ingestion")
    parser.add_argument("--root", default=CONFIG.storage.volume_root)
    parser.add_argument("--no-index", action="store_true",
                        help="Skip Databricks AI Search conflict indexing.")
    parser.add_argument("--crop-zoom", action="store_true",
                        help="Enable crop-and-zoom multimodal reading of "
                             "chart-like figures (PDF sources).")
    args = parser.parse_args()

    subs = discover_submissions(args.root)
    print(f"Discovered {len(subs)} submissions under {args.root}")

    search_store = None if args.no_index else AzureAISearchStore()
    if search_store is not None:
        search_store.ensure_index()

    # Crop-and-zoom is opt-in. The pdfplumber-backed provider renders figure
    # pages for legible re-reading; left off, extraction is text-only.
    page_image_provider = None
    if args.crop_zoom:
        from ..search.figure_preprocessor import PdfPlumberPageImageProvider
        page_image_provider = PdfPlumberPageImageProvider()

    result = asyncio.run(process_batch(subs, search_store=search_store,
                           page_image_provider=page_image_provider))
    print(
        f"run_id={result.run_id} documents={result.documents} "
        f"gold_rows={result.gold_metrics_total} "
        f"cost=${result.total_cost_usd:.4f} review={result.needs_review} "
        f"conflicts={result.conflicts} quarantined={result.quarantined} "
        f"cropped_figures={result.cropped_figures}"
    )


if __name__ == "__main__":
    main()
