"""Distinguish dates from metric values, and normalise dates.

Problem this solves: an extracted value like "Q1 2024", "9 March 2024", or
"31/03/2024" is a DATE, not a metric. Without a guard, the numeric parser turns
"Micro Report 2024 Q1" into numeric_value=2024 — polluting the metrics. Dates
must NOT land in numeric_value; a document-level report date belongs on
as_at_date instead.

CONTEXT-AWARE by design: a value is only treated as a date when it carries
temporal *context* — a month, quarter, half-year, an explicit date separator, or
a date-signalling field name ("report_date", "period", "as at"). A BARE number,
even one in the year range (e.g. "2024", "1434"), is NOT assumed to be a date,
because it could be a genuine metric (a count, an amount). This errs toward
keeping numbers as metrics unless there's real evidence they're dates —
`looks_like_date` takes the field name so it can use that evidence.
"""

from __future__ import annotations

import re
from typing import Optional


_MONTHS = ("january|february|march|april|may|june|july|august|september|"
           "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|"
           "oct|nov|dec")

# Patterns where the value ITSELF carries temporal context (unambiguous dates).
_DATE_PATTERNS = [
    re.compile(r"^\s*(q[1-4]|h[12])\s*[- ]?\s*\d{4}\s*$", re.I),   # "Q1 2024"
    re.compile(r"^\s*\d{4}\s*[- ]?\s*(q[1-4]|h[12])\s*$", re.I),   # "2024 Q1"
    re.compile(r"^\s*\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\s*$"),      # 31/03/2024
    re.compile(r"^\s*\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\s*$"),        # 2024-03-31
    re.compile(rf"^\s*\d{{1,2}}\s+({_MONTHS})\s+\d{{4}}\s*$", re.I),   # 9 March 2024
    re.compile(rf"^\s*({_MONTHS})\s+\d{{1,2}},?\s+\d{{4}}\s*$", re.I),  # March 9, 2024
    re.compile(rf"^\s*({_MONTHS})\s+\d{{4}}\s*$", re.I),           # March 2024
    re.compile(rf"^\s*\d{{4}}\s+({_MONTHS})\s*$", re.I),           # 2024 March
    re.compile(rf"^\s*({_MONTHS})\s*$", re.I),                     # a bare month name
    # abbreviated dash formats, very common in regulatory tables:
    re.compile(rf"^\s*\d{{1,2}}[- ]({_MONTHS})[- ]\d{{2,4}}\s*$", re.I),  # 31-Mar-26
    re.compile(rf"^\s*({_MONTHS})[- ]\d{{2,4}}\s*$", re.I),        # Apr-25 / Mar-2026
    re.compile(rf"^\s*\d{{2,4}}[- ]({_MONTHS})\s*$", re.I),        # 25-Apr
    re.compile(r"^\s*\d{4}[/\-.]\d{1,2}\s*$"),                     # 2025-04
    re.compile(r"^\s*\d{1,2}[/\-.]\d{4}\s*$"),                     # 04-2025
    # QUALIFIED month-year phrases ('End June 2025', 'As at June 2025',
    # 'Period ended 30 June 2025', 'Year end December 2025'). Dates because of
    # the MONTH-NAME context — bare numbers and categoricals never match. Kept
    # SIMPLE (shallow alternation) to avoid deep-nesting regex issues.
    re.compile(rf"^\s*(?:end|ended|ending|as at|as of|period end|period ended|"
               rf"period ending|quarter end|quarter ended|quarter ending|"
               rf"half year|year end|year ended|year ending|at)\s+"
               rf"(?:\d{{1,2}}\s+)?(?:{_MONTHS})\s+\d{{2,4}}\s*$", re.I),
    # month + year with an optional trailing qualifier ('June 2025 year end').
    re.compile(rf"^\s*(?:{_MONTHS})\s+\d{{4}}"
               rf"(?:\s+(?:year end|end|quarter|half year))?\s*$", re.I),
]

# A field NAME that signals the value is a date/period, letting a bare year like
# "2024" be read as a date ONLY when its metric name says so.
_DATE_FIELD_RE = re.compile(
    r"(date|period|quarter|half[-_ ]?year|financial[-_ ]?year|fy|as[-_ ]?at|"
    r"as[-_ ]?of|reporting[-_ ]?date|report[-_ ]?date|month|year[-_ ]?end|"
    r"period[-_ ]?end|effective[-_ ]?date|valuation[-_ ]?date)", re.I)

_BARE_YEAR_RE = re.compile(r"^\s*(19|20)\d{2}\s*$")


def looks_like_date(value, field_name: Optional[str] = None) -> bool:
    """True if the value is a date/period rather than a metric.

    Context-aware:
      * The value carries its own temporal context (month/quarter/separator) ->
        date, regardless of field name.
      * The value is a bare year (19xx/20xx) -> date ONLY IF the field name
        signals a date ("report_date", "period", "as at", ...). A bare year with
        a metric-like name (e.g. "transactions", "headcount") is treated as a
        METRIC, since a number alone isn't proof of a date.
      * Anything else -> not a date.
    """
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False

    # A DECIMAL number is never a date — including a float like "1252.0" (which a
    # numeric claim value stringifies to). "1252.0" otherwise matches the YYYY.MM
    # date pattern (\d{4}.\d{1,2}), so a stray metric value would be mistaken for
    # a report date. NOTE: bare integer years ("2024") are handled below (they
    # can be a date with the right field name), so only reject values that have a
    # decimal point / are non-integer numbers.
    if re.search(r"\d\.\d", s):
        try:
            float(s.replace(",", ""))
            return False      # a decimal number (e.g. 1252.0) -> not a date
        except (ValueError, TypeError):
            pass

    # 1. value carries unambiguous temporal context on its own
    for rx in _DATE_PATTERNS:
        if rx.match(s):
            return True

    # 2. bare year: only a date when the FIELD NAME says it's a date/period
    if _BARE_YEAR_RE.match(s):
        if field_name and _DATE_FIELD_RE.search(str(field_name)):
            return True
        return False  # bare year, non-date field -> treat as a metric

    return False


def normalise_report_date(value) -> Optional[str]:
    """Best-effort normalised report date string for as_at_date. Keeps the
    period form the document used (e.g. "Q1 2024") rather than forcing a
    calendar date — regulatory reporting is period-based. Returns the string if
    it carries temporal context or is a bare year; else None."""
    if value is None:
        return None
    s = str(value).strip()
    for rx in _DATE_PATTERNS:
        if rx.match(s):
            return s
    if _BARE_YEAR_RE.match(s):
        return s
    return None
