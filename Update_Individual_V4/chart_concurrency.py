"""Bounded-concurrency chart reads.

Charts are read one crop at a time today, so 300 charts pay the API round-trip
300 times in series. Crops are small and INDEPENDENT (cropping removed the
large-payload contention that made parallel reads risky before), so they are safe
to read concurrently -- bounded by a semaphore so the model endpoint's rate limit
is respected.

This does NOT change accuracy: each crop read is independent, so parallel vs
sequential yields the same per-chart result. It only removes the serial waiting.

Usage: the caller builds an async read coroutine per crop and passes them here;
read_crops_concurrently gathers them under a semaphore and returns results in the
SAME order as the input crops. The caller's run_async_fn runs the single gather.
"""
from __future__ import annotations

import asyncio
from typing import Callable

DEFAULT_CONCURRENCY = 5     # simultaneous reads; tune to the endpoint's limit


async def _bounded(sem, coro):
    async with sem:
        try:
            return await coro
        except Exception as e:
            return e            # return the error; caller decides per-crop


async def read_crops_concurrently(crops, async_read_fn,
                                  concurrency=DEFAULT_CONCURRENCY):
    """Read every crop concurrently, bounded by `concurrency`.
    async_read_fn(crop) -> awaitable yielding that crop's points (or falsy).
    Returns a list of results in the SAME order as `crops`; a failed read is
    returned as the Exception (so one failure never kills the batch)."""
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    tasks = [ _bounded(sem, async_read_fn(c)) for c in crops ]
    return await asyncio.gather(*tasks)
