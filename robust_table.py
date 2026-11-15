"""Robust, NEVER-FAIL table structuring with confidence tagging.

ai_parse returns tables as HTML (version 2.0), and the HTML can be malformed in
ways a hand-rolled parser doesn't anticipate -> brittle in production. This module
makes table parsing ROBUST and SAFE:

  1. PRIMARY  : pandas.read_html with the html5lib backend -- a mature, FORGIVING
                HTML-table parser (html5lib repairs broken HTML like a browser).
                Handles far more malformed shapes than custom code. Deterministic.
  2. FALLBACK : the existing grid parser (html_table_to_claims) if read_html fails
                or yields nothing usable.
  3. LAST     : flat-extract -- every number + its nearest row/column label, so we
                NEVER return nothing from a table that has numbers.

Every claim is tagged with the tier that produced it (table_confidence:
clean | fallback | degraded) so production confidence is MEASURABLE and the
degraded ones can be routed to human review. No LLM -- fully deterministic.
"""
from __future__ import annotations

import re
from typing import Optional

CONF_CLEAN = "clean"         # pandas.read_html parsed it cleanly
CONF_FALLBACK = "fallback"   # the existing grid parser handled it
CONF_TATR = "tatr"           # TATR re-recognised the structure from the image
CONF_DEGRADED = "degraded"   # flat-extract only -- flag for review

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _has_number(s):
    return bool(_NUM.search(str(s or "")))


def _tag(claims, tier):
    for c in claims:
        try:
            idf = getattr(c, "identifiers", None)
            if isinstance(idf, dict):
                idf["table_confidence"] = tier
            else:
                setattr(c, "identifiers", {"table_confidence": tier})
        except Exception:
            pass
    return claims


def _df_looks_misaligned(df):
    """Heuristic: does the parsed DataFrame look structurally wrong? Signals:
    a column that is entirely empty/NaN, or a very high fraction of empty cells,
    or only one column (collapsed). Deterministic check -> route to TATR."""
    try:
        import pandas as pd
        if df is None or df.empty:
            return True
        if df.shape[1] <= 1 and df.shape[0] > 1:
            return True                    # collapsed to one column
        # fraction of empty/NaN cells
        total = df.size
        if total == 0:
            return True
        empt = int(df.isna().sum().sum()) + int((df.astype(str)
                   .apply(lambda s: s.str.strip() == "")).sum().sum())
        if empt / total > 0.5:             # >50% empty -> likely misaligned
            return True
        # a whole column empty
        for ci in range(df.shape[1]):
            col = df.iloc[:, ci].astype(str).str.strip()
            if (col == "").all() or df.iloc[:, ci].isna().all():
                return True
    except Exception:
        return False
    return False


def _read_html_df(html):
    """Parse the table HTML with pandas + html5lib (forgiving). Returns a list of
    DataFrames, or [] on failure. html5lib repairs malformed HTML before parsing."""
    try:
        import pandas as pd
        from io import StringIO
        # html5lib is the most forgiving backend; fall back to bs4 then lxml.
        for _flavor in ("html5lib", "bs4", "lxml"):
            try:
                dfs = pd.read_html(StringIO(html), flavor=_flavor)
                if dfs:
                    return dfs
            except Exception:
                continue
    except Exception:
        pass
    return []


def _df_to_claims(df, filename, page, title, team, report_type, entity_ref,
                  claim_cls, element_id):
    """Turn a parsed DataFrame into one claim per numeric cell, with the row label
    (first non-numeric column) + column header as the metric path. Deterministic."""
    import pandas as pd
    claims = []
    if df is None or df.empty:
        return claims
    # flatten multi-level columns into "a b" strings
    cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            cols.append(" ".join(str(x) for x in col if str(x) != "nan").strip())
        else:
            cols.append(str(col))
    # identify label columns (no numbers anywhere) vs value columns
    label_cols, value_cols = [], []
    for ci, col in enumerate(cols):
        series = df.iloc[:, ci]
        has_num = any(_has_number(v) for v in series.tolist())
        (value_cols if has_num else label_cols).append(ci)
    if not value_cols:                      # no numeric columns -> nothing to claim
        return claims
    for ri in range(len(df)):
        # build the row label from the label columns
        row_label = " ".join(str(df.iloc[ri, ci]).strip()
                             for ci in label_cols
                             if str(df.iloc[ri, ci]) not in ("nan", "None", "")).strip()
        for ci in value_cols:
            raw = df.iloc[ri, ci]
            if not _has_number(raw):
                continue
            col_label = cols[ci]
            parts = [p for p in (title, row_label, col_label) if p and p != "nan"]
            name = " | ".join(parts) if parts else col_label
            try:
                claims.append(claim_cls(
                    field_name=col_label or name,
                    canonical_metric=name,
                    value=str(raw).strip(),
                    source_document=filename,
                    page=page, team=team, report_type=report_type,
                    entity_ref=entity_ref, element_type="table",
                    element_id=element_id,
                    model_used="pandas_read_html"))
            except Exception:
                continue
    return claims


def _flat_extract(html, filename, page, title, team, report_type, entity_ref,
                  claim_cls, element_id):
    """LAST resort: strip tags, pull every number with its nearest preceding text
    as the label. Never returns nothing from a table that has numbers. Degraded."""
    claims = []
    # split into cells on tags, keep order
    cells = [re.sub(r"\s+", " ", c).strip()
             for c in re.split(r"<[^>]+>", html or "") if c.strip()]
    last_label = ""
    for cell in cells:
        if _has_number(cell):
            m = _NUM.search(cell)
            val = m.group(0) if m else cell
            parts = [p for p in (title, last_label) if p]
            name = " | ".join(parts) if parts else (last_label or "value")
            try:
                claims.append(claim_cls(
                    field_name=last_label or "value",
                    canonical_metric=name,
                    value=val,
                    source_document=filename, page=page, team=team,
                    report_type=report_type, entity_ref=entity_ref,
                    element_type="table", element_id=element_id,
                    model_used="flat_extract"))
            except Exception:
                continue
        else:
            last_label = cell            # text cell -> label for following numbers
    return claims


def robust_table_to_claims(html, filename, *, page=None, title=None, team=None,
                           report_type=None, entity_ref=None, claim_cls=None,
                           element_id=None, audit=None, run_id=None,
                           crop_b64=None):
    """NEVER-FAIL table structuring. Tries, in order: pandas.read_html (clean) ->
    the existing grid parser (fallback) -> flat-extract (degraded). Tags every
    claim with its table_confidence tier and logs the tier for measurability."""
    if claim_cls is None:
        from ..shared.schema import Claim as claim_cls_default
        claim_cls = claim_cls_default

    tier = None
    claims = []

    # 1. PRIMARY: pandas.read_html (forgiving html5lib). Accept ONLY if the
    #    parsed table does not look misaligned -- a misaligned parse (ragged /
    #    collapsed / mostly-empty) is rejected so the chain proceeds to TATR.
    try:
        dfs = _read_html_df(html)
        _aligned = [df for df in dfs if not _df_looks_misaligned(df)]
        if _aligned:
            for df in _aligned:
                claims.extend(_df_to_claims(df, filename, page, title, team,
                                            report_type, entity_ref, claim_cls,
                                            element_id))
            if claims:
                tier = CONF_CLEAN
    except Exception:
        claims = []

    # 2. FALLBACK: the existing grid parser
    if not claims:
        try:
            from .ai_parse_structurer import html_table_to_claims
            claims = html_table_to_claims(
                html, filename, page=page, title=title, team=team,
                report_type=report_type, entity_ref=entity_ref,
                claim_cls=claim_cls, element_id=element_id)
            if claims:
                tier = CONF_FALLBACK
        except Exception:
            claims = []

    # 3. LAST RESORT: flat-extract (never return nothing). Misaligned-structure
    #    tables land here (tier "degraded") and are flagged for human review.
    #    (A TATR image-recognition tier exists in tatr_table.py but is NOT wired --
    #    it needs an external model that isn't approved here. The deterministic
    #    chain + HITL review covers the misaligned tables instead.)
    if not claims:
        try:
            claims = _flat_extract(html, filename, page, title, team,
                                   report_type, entity_ref, claim_cls, element_id)
            tier = CONF_DEGRADED
        except Exception:
            claims = []

    _tag(claims, tier or CONF_DEGRADED)
    if audit is not None:
        try:
            audit.log(run_id or "", "table_confidence", f"{filename}-p{page}",
                      document=filename, page=page, tier=(tier or CONF_DEGRADED),
                      n_claims=len(claims))
        except Exception:
            pass
    return claims
