# Here are the exact changes to storage.py — two fixes.

# FIX 1 — Apply the GOLD_WHITELIST (this is what actually drops the columns).

# In build_gold_metric_rows, find:

            rows.append(gm.model_dump())
        return rows

#replace
            _full = gm.model_dump()
            # APPLY the whitelist: keep only the gold metric columns we want,
            # plus the structural columns gold always needs. Drops netting,
            # measure_type, period, reporting_basis, scale.
            _keep = set(GOLD_WHITELIST) | {
                "field_name", "element_type", "numeric_value",
                "source_document", "page", "domain_tag"}
            rows.append({k: v for k, v in _full.items() if k in _keep})
        return rows

# FIX 2 — Fix the table naming for query_gold/bronze (the privileges error). 
# Find
        table = getattr(CONFIG.storage, "bronze_original_table", None) \
            or f"{getattr(CONFIG.storage, 'table_prefix', '')}bronze_original"
        return self._write("bronze_original", table, raw_elements)

# replace
        table = getattr(CONFIG.storage, "bronze_original_table", None) \
            or CONFIG.storage.fqtn("bronze_original")
        return self._write("bronze_original", table, raw_elements)

# Then find:
        table = getattr(CONFIG.storage, "query_gold_table", None) \
            or f"{getattr(CONFIG.storage, 'table_prefix', '')}query_gold"
        return self._write("query_gold", table, rows)
    
# replace
        table = getattr(CONFIG.storage, "query_gold_table", None) \
            or CONFIG.storage.fqtn("query_gold")
        return self._write("query_gold", table, rows)
    

# UPDATE 5 — the identifiers MAP column (so query_gold writes cleanly).

# 5a — in _spark_schema_for. Find:
    from pyspark.sql.types import (StructType, StructField, StringType,
                                   DoubleType, LongType)
    # column -> Spark type. page is integer-like (Long); the rest of the
    # numeric metrics are Double. Everything not listed is String.
    numeric_types = {
        "numeric_value": DoubleType(),
        "confidence": DoubleType(),
        "page": LongType(),
    }
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    fields = [StructField(k, numeric_types.get(k, StringType()), True)
              for k in keys]
    return StructType(fields)

# replace
    from pyspark.sql.types import (StructType, StructField, StringType,
                                   DoubleType, LongType, MapType)
    # column -> Spark type. page is integer-like (Long); numeric metrics are
    # Double; identifiers is a MAP<string,string> (query_gold). Everything else
    # is String.
    explicit_types = {
        "numeric_value": DoubleType(),
        "confidence": DoubleType(),
        "page": LongType(),
        "identifiers": MapType(StringType(), StringType()),
    }
    keys = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    fields = [StructField(k, explicit_types.get(k, StringType()), True)
              for k in keys]
    return StructType(fields)

# 5b — in _rows_for_explicit_schema. Find:
    numeric_cols = {"numeric_value", "confidence", "page"}
    out = []
    for r in rows:
        clean = {}
        for k, v in r.items():
            if k in numeric_cols:
                if v is None or v == "":
                    clean[k] = None
                else:
                    try:
                        clean[k] = int(v) if k == "page" else float(v)
                    except (ValueError, TypeError):
                        clean[k] = None
            else:
                clean[k] = None if v is None else str(v)
        out.append(clean)
    return out

# replace
    numeric_cols = {"numeric_value", "confidence", "page"}
    out = []
    for r in rows:
        clean = {}
        for k, v in r.items():
            if k in numeric_cols:
                if v is None or v == "":
                    clean[k] = None
                else:
                    try:
                        clean[k] = int(v) if k == "page" else float(v)
                    except (ValueError, TypeError):
                        clean[k] = None
            elif k == "identifiers":
                # keep the MAP as a dict of string->string (never str() it)
                if isinstance(v, dict):
                    clean[k] = {str(kk): ("" if vv is None else str(vv))
                                for kk, vv in v.items()}
                else:
                    clean[k] = {}
            else:
                clean[k] = None if v is None else str(v)
        out.append(clean)
    return out

# FILE 1: schema.py — add the currency field (two places).

# In the Claim class, find:
    unit: Optional[str] = None
    reporting_basis: Optional[str] = None

#replace
    unit: Optional[str] = None
    currency: Optional[str] = None
    reporting_basis: Optional[str] = None

# In the GoldMetric class, find:
    unit: Optional[str] = None
    reporting_basis: Optional[str] = None
#replace
    unit: Optional[str] = None
    currency: Optional[str] = None
    reporting_basis: Optional[str] = None

# FILE 2: ai_parse_structurer.py — the split function + applying it.

# 2a — add the split function (near the top, before _extract_unit):
# $mm / £mm / mm = millions notation; recognise it for the currency/unit split.
_CUR_RE = re.compile(r"[£$€]")
_SCALE_RE = re.compile(r"(?<![a-zA-Z])(mm|bn|billion|million|m|k)(?![a-zA-Z])", re.I)
_PCT_RE = re.compile(r"(%|bps)")


def _split_currency_unit(text):
    """Split a value or label into (currency, unit). currency is the symbol
    (£/$/€) if present; unit is the scale/measure (mm, bn, m, k, %, bps). Handles
    $mm, £mm, mm (millions). Either may be None.
    e.g. '$125mm' -> ('$', 'mm'); '15%' -> (None, '%'); '£m' -> ('£', 'm')."""
    if not text:
        return None, None
    s = str(text)
    cur_m = _CUR_RE.search(s)
    currency = cur_m.group(0) if cur_m else None
    pct_m = _PCT_RE.search(s)
    scale_m = _SCALE_RE.search(s)
    if pct_m:
        unit = pct_m.group(1)
    elif scale_m:
        unit = scale_m.group(1).lower().replace("million", "m").replace("billion", "bn")
    else:
        unit = None
    return currency, unit

# 2b — _unit_from_label must recognise mm. Find the two regexes inside _unit_from_label and make them:
    m = re.search(r"\((?:\s*in\s+)?\s*([£$€]\s*(?:mm|bn|billion|million|m|k)?|%|bps)\s*\)"
                  r"|\bin\s+([£$€]\s*(?:mm|bn|billion|million|m|k)?|%|bps)\b",
                  s, re.I)

# and the trailing-bare one:
        m2 = re.search(r"([£$€]\s*(?:mm|bn|billion|million|m|k)|%|bps)\s*$", s, re.I)

# 2c — apply the split in the table-claim construction. Find:
                # unit precedence: value's own unit > column header unit > title unit
                _unit = _extract_unit(val) or col_unit.get(c) or title_unit
                claims.append(claim_cls(
                    field_name=header or "value", canonical_metric=name,
                    value=val, unit=_unit, element_type="table",
# replace
                # unit precedence: value's own unit > column header unit > title
                # unit. The resolved token is then SPLIT into currency (£/$/€) and
                # unit (mm/bn/m/k/%/bps) so the two live in separate columns.
                _resolved = (_extract_unit(val) or col_unit.get(c) or title_unit)
                _currency, _unit = _split_currency_unit(_resolved) if _resolved \
                    else (None, None)
                claims.append(claim_cls(
                    field_name=header or "value", canonical_metric=name,
                    value=val, unit=_unit, currency=_currency, element_type="table",

# FILE 3: storage.py — carry currency into the gold rows + whitelist.

# 3a — GOLD_WHITELIST includes currency (near line 25):
GOLD_WHITELIST = [
    "entity_ref", "canonical_metric", "value", "currency", "unit",
    "as_at_date",
    "citation_tier", "review_tier",
]

# 3b — pass currency into GoldMetric in build_gold_metric_rows. Find:
                page=c.page,
                unit=c.unit,
                reporting_basis=c.reporting_basis,
# replace

                page=c.page,
                unit=c.unit,
                currency=getattr(c, "currency", None),
                reporting_basis=c.reporting_basis,

# 3c — the whitelist filter (same as before — this keeps currency and drops the unwanted). Find:
            rows.append(gm.model_dump())
        return rows
# replace
            _full = gm.model_dump()
            _keep = set(GOLD_WHITELIST) | {
                "field_name", "element_type", "numeric_value",
                "source_document", "page", "domain_tag"}
            rows.append({k: v for k, v in _full.items() if k in _keep})
        return rows