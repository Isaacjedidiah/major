"""Grounding check: does an extracted value actually appear in its source?

This is a hallucination guard. Citations tell a reviewer *where to look*;
grounding checks that the value the model reported is *actually there*. A model
can produce a plausible number that is not in the source text — grounding
catches that class of fabrication before it reaches Gold.

Scope, and why it matters:
  * TEXT / TABLE claims — the value should appear (in some normalised form) in
    the element's own text. If it doesn't, the claim is "ungrounded": the model
    likely invented or mis-copied it. We can check this deterministically.
  * FIGURE / chart claims — the value is read from an *image*, so it will NOT be
    in any text. These cannot be text-grounded and are marked "not_applicable"
    (their reliability is handled by the LLM_ESTIMATED citation tier + confidence
    gating, not by text grounding). Marking them "ungrounded" would be wrong.

The check is intentionally lenient about format (15.6 vs 15.6% vs "15.60"),
because we want to catch *fabrication*, not punish *formatting*. It compares the
numeric core of the value against numbers present in the source.
"""

from __future__ import annotations

import re
from typing import Optional


# grounding status values (kept as plain strings for easy storage/filtering)
GROUNDED = "grounded"              # value found in source text
UNGROUNDED = "ungrounded"         # value NOT found — possible hallucination
NOT_APPLICABLE = "not_applicable"  # figure/image claim — can't be text-grounded


_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _numeric_core(text: str) -> Optional[str]:
    """The first number in a string, normalised (commas stripped, trailing
    zeros trimmed) so 15.6 / 15.60 / '15.6%' compare equal. None if no number."""
    if text is None:
        return None
    m = _NUM_RE.search(str(text))
    if not m:
        return None
    raw = m.group(0).replace(",", "")
    try:
        f = float(raw)
    except (ValueError, TypeError):
        return None
    # normalise: integers keep no decimal, floats trim trailing zeros
    return str(int(f)) if f == int(f) else ("%f" % f).rstrip("0").rstrip(".")


def _numbers_in(text: str) -> set[str]:
    """All numbers present in a body of text, normalised for comparison."""
    out: set[str] = set()
    for m in _NUM_RE.finditer(text or ""):
        raw = m.group(0).replace(",", "")
        try:
            f = float(raw)
        except (ValueError, TypeError):
            continue
        out.add(str(int(f)) if f == int(f) else ("%f" % f).rstrip("0").rstrip("."))
    return out


def check_grounding(value: str, source_content: str, is_figure: bool) -> str:
    """Return a grounding status for one claim value against its source element.

    is_figure=True  -> NOT_APPLICABLE (image-derived, can't text-ground).
    numeric value    -> GROUNDED if the same number appears in source_content.
    non-numeric value-> GROUNDED if the (case-insensitive) text appears in source.
    Anything checkable but not found -> UNGROUNDED.
    """
    if is_figure:
        return NOT_APPLICABLE

    if value is None or not str(value).strip():
        return UNGROUNDED

    content = source_content or ""

    # numeric values: compare the numeric core against numbers in the source
    core = _numeric_core(value)
    if core is not None:
        return GROUNDED if core in _numbers_in(content) else UNGROUNDED

    # non-numeric values (e.g. "N/A", a category label): substring match
    v = str(value).strip().lower()
    return GROUNDED if v in content.lower() else UNGROUNDED


def is_suspected_hallucination(status: str) -> bool:
    """True only for UNGROUNDED — the status that flags a possible fabrication.
    NOT_APPLICABLE (figures) is not a hallucination signal."""
    return status == UNGROUNDED
