"""Query router: decide, cheaply and instantly, whether a question should be
answered from the STRUCTURED metric store (SQL / gold + analytical) or from the
narrative INDEX (vector search).

The point is speed: running BOTH paths for every question is slow and burns
tokens. A rules-based classifier (no model call) routes each question to the
one path that fits, and only falls back to BOTH when a question is genuinely
ambiguous.

Routing signal (keyword / pattern based — instant, deterministic, free):

  * STRUCTURED  -> the question wants numbers, a trend, an aggregate, a value
                  for a named entity/period. Answer from gold / analytical with
                  a fast key-lookup (WHERE entity=.. AND category=.. AND period..).
  * NARRATIVE   -> the question wants explanation, context, what a report says.
                  Answer from the index (vector search).
  * BOTH        -> ambiguous; the app may run both and merge, or default to one.

This is deliberately simple and transparent so it can be tuned by eye. It can
be replaced by a small classifier later without changing the callers.
"""

from __future__ import annotations

import re

STRUCTURED = "structured"      # -> SQL / gold / analytical
NARRATIVE = "narrative"        # -> index / vector search
BOTH = "both"

# --- signals that a question wants STRUCTURED data (numbers / trends / aggregates)
_STRUCTURED_PATTERNS = [
    r"\bhow much\b", r"\bhow many\b", r"\bwhat (?:was|is|are|were) the (?:value|amount|total|figure|number|level|ratio|balance)\b",
    r"\btotal\b", r"\bsum\b", r"\baverage\b", r"\bmean\b", r"\bcount\b",
    r"\btrend\b", r"\bover (?:the )?(?:last |past )?\d+ (?:year|quarter|month|period)s?\b",
    r"\bover time\b", r"\byear[- ]on[- ]year\b", r"\bcompare(?:d)?\b", r"\bcomparison\b",
    r"\bby (?:team|entity|report|quarter|year|period|month)\b",
    r"\bhighest\b", r"\blowest\b", r"\btop \d+\b", r"\bexceed(?:ed|ing|s)?\b",
    r"\bgreater than\b", r"\bless than\b", r"\babove\b", r"\bbelow\b", r"\bthreshold\b",
    r"\bposition\b", r"\bbreakdown\b", r"\bexposure\b", r"\brevenue\b", r"\bratio\b",
    r"\bgrowth\b", r"\bchange in\b", r"\bmovement\b", r"\bchart\b", r"\bgraph\b", r"\bplot\b",
]

# --- signals that a question wants NARRATIVE / interpretation
_NARRATIVE_PATTERNS = [
    r"\bwhat does .* say\b", r"\bexplain\b", r"\bdescribe\b", r"\bsummar(?:y|ise|ize)\b",
    r"\bwhy\b", r"\bhow (?:does|did|is|are|was|were) .* (?:manage|approach|handle|address)\b",
    r"\bcontext\b", r"\bcommentary\b", r"\bnarrative\b", r"\brationale\b", r"\boutlook\b",
    r"\bdiscuss\b", r"\bmention(?:ed|s)?\b", r"\bstate(?:d|s)?\b", r"\bnote(?:d|s)?\b",
    r"\bwhat (?:is|are) the (?:risk|reason|driver|factor|assumption|concern)s?\b",
    r"\bqualitative\b", r"\bopinion\b", r"\bview\b",
]

_STRUCT_RE = [re.compile(p, re.I) for p in _STRUCTURED_PATTERNS]
_NARR_RE = [re.compile(p, re.I) for p in _NARRATIVE_PATTERNS]


def route_query(question: str) -> str:
    """Return STRUCTURED, NARRATIVE, or BOTH for a question.

    Counts how many structured vs narrative signals the question hits; the
    stronger side wins. A tie (or no signal) routes to NARRATIVE by default —
    the index is the safer general-purpose fallback and never returns a wrong
    exact number. A question that hits BOTH sides strongly routes to BOTH.
    """
    if not question or not question.strip():
        return NARRATIVE

    s = sum(1 for r in _STRUCT_RE if r.search(question))
    n = sum(1 for r in _NARR_RE if r.search(question))

    if s and n and abs(s - n) <= 1:
        return BOTH                 # genuinely mixed -> let the app run both
    if s > n:
        return STRUCTURED
    if n > s:
        return NARRATIVE
    # no signal either way -> default to the index (safe general fallback)
    return NARRATIVE


def route_explain(question: str) -> dict:
    """Same as route_query but returns the reasoning (which signals fired), for
    logging / debugging / tuning the rules."""
    s_hits = [r.pattern for r in _STRUCT_RE if r.search(question or "")]
    n_hits = [r.pattern for r in _NARR_RE if r.search(question or "")]
    return {
        "route": route_query(question),
        "structured_signals": s_hits,
        "narrative_signals": n_hits,
    }
