"""Chart extraction RETRY LOOP — solves INTERMITTENT SKIPS only.

Sonnet is non-deterministic: it returns nothing for a crop on one run but reads
it on another. This loop retries each crop (fresh, independent reads) until it
produces points OR a pass limit is reached; whatever never reads goes to review.

It uses the SAME proven chart extraction as the inline path (the caller passes a
read function built on CHART_DIGITISATION_PROMPT). No self-review — each read is a
fresh attempt, success judged OBJECTIVELY (did it return points), never by
Sonnet's opinion of its own work.

This module is ONLY about the skip problem. Shared-legend / colour-join is a
SEPARATE concern and is deliberately NOT handled here.

Flow per document:
    for pass in 1..MAX_PASSES:
        for crop still pending:
            points = read_fn(crop)          # proven read (CHART_DIGITISATION_PROMPT)
            if points: emit claims, done    # move out of pending
            # else: keep for the next pass
        if none pending: break
    pending after the loop -> on_unresolved (human review)
"""
from __future__ import annotations

from typing import Callable, Optional

MAX_PASSES = 4          # retry a crop across at most this many passes


def run_chart_loop(page_crops, read_fn, on_extracted=None, on_unresolved=None,
                   max_passes=MAX_PASSES, audit=None, run_id=None, document=None):
    """Retry-until-read loop over chart crops. Pure skip handling.

    page_crops : list of crop dicts; each must carry what read_fn/on_extracted
                 need (e.g. crop_b64, page, figure_index, el).
    read_fn(crop) -> points list (truthy) or falsy. MUST use the proven chart
                 read (CHART_DIGITISATION_PROMPT) so loop extraction == inline
                 extraction. A fresh call each time (no memory, no self-review).
    on_extracted(crop, points) : emit claims for a crop that read.
    on_unresolved(crop, reason): a crop that never read after max_passes.
    Returns {extracted, unresolved, passes_used}.
    """
    pending = list(page_crops)
    extracted = 0
    passes_used = 0
    for _pass in range(1, max_passes + 1):
        if not pending:
            break
        passes_used = _pass
        still = []
        for crop in pending:
            try:
                pts = read_fn(crop)
            except Exception:
                pts = None
            if pts:
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

    for crop in pending:
        reason = f"unresolved_after_{passes_used}_passes"
        if on_unresolved:
            on_unresolved(crop, reason)
        if audit is not None:
            try:
                audit.log(run_id or "", "chart_read",
                          f"{document}-p{crop.get('page')}-fig{crop.get('figure_index')}",
                          document=document, page=crop.get("page"),
                          figure_index=crop.get("figure_index"),
                          outcome="unresolved", n_points=0, reason=reason)
            except Exception:
                pass

    return {"extracted": extracted, "unresolved": len(pending),
            "passes_used": passes_used}


def run_chart_loop_concurrent(page_crops, async_read_fn, run_async_fn,
                              on_extracted=None, on_unresolved=None,
                              max_passes=MAX_PASSES, concurrency=5,
                              audit=None, run_id=None, document=None):
    """Same retry-until-read loop, but each PASS reads its pending crops
    CONCURRENTLY (bounded by `concurrency`) instead of one at a time. Accuracy is
    unchanged (reads are independent); only the serial waiting is removed -- the
    win for documents with many charts.

    async_read_fn(crop) -> awaitable yielding points (truthy) or falsy. Must use
                 the proven chart read so output == the sequential loop.
    run_async_fn(coro) -> runs one coroutine to completion (the caller's runner).
    """
    from .chart_concurrency import read_crops_concurrently

    pending = list(page_crops)
    extracted = 0
    passes_used = 0
    for _pass in range(1, max_passes + 1):
        if not pending:
            break
        passes_used = _pass
        # read this pass's pending crops concurrently
        results = run_async_fn(
            read_crops_concurrently(pending, async_read_fn, concurrency=concurrency))
        still = []
        for crop, pts in zip(pending, results):
            if isinstance(pts, Exception):
                pts = None
            if pts:
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

    for crop in pending:
        reason = f"unresolved_after_{passes_used}_passes"
        if on_unresolved:
            on_unresolved(crop, reason)
        if audit is not None:
            try:
                audit.log(run_id or "", "chart_read",
                          f"{document}-p{crop.get('page')}-fig{crop.get('figure_index')}",
                          document=document, page=crop.get("page"),
                          figure_index=crop.get("figure_index"),
                          outcome="unresolved", n_points=0, reason=reason)
            except Exception:
                pass

    return {"extracted": extracted, "unresolved": len(pending),
            "passes_used": passes_used}
