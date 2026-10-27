"""Medallion storage: Bronze / Silver / Gold.

Local mode persists JSONL under a root dir; Databricks mode writes Delta
tables. Gold is a deliberate whitelist — ``normalisation_conflicts`` is
excluded from Gold (they go to Databricks AI Search for narrative search).

The build/persist split (pure ``build_gold_metric_rows`` + thin
``write_gold_metrics``) keeps row-building testable without I/O.

Fixes: entity_ref is read from the Claim (which now carries it) instead of a
sentinel; re-runs are idempotent via content_hash dedup on JSONL; quarantine,
review decisions and audit events have real sinks; reviewed/overridden claims
can be promoted into Gold.
"""
from __future__ import annotations

import json
import re
import os
from typing import Optional

from ..shared.config import CONFIG
from ..shared.schema import Claim, GoldMetric, content_hash

GOLD_WHITELIST = [
    "entity_ref", "canonical_metric", "value", "currency", "unit",
    "as_at_date",
    "citation_tier", "review_tier",
]


class StorageManager:
    """Local JSONL persistence for MVP / test runs."""

    def __init__(self, root: Optional[str] = None):
        self.root = root or CONFIG.storage.local_root
        os.makedirs(self.root, exist_ok=True)

    def _path(self, layer: str) -> str:
        return os.path.join(self.root, f"{layer}.jsonl")

    def append_jsonl(self, layer: str, rows: list[dict]) -> int:
        """Append rows, skipping any whose content hash was already written.

        Idempotency: each row gets a stable hash over its sorted JSON; hashes
        already present in the file are skipped, so re-scanning processed
        files never duplicates. (On Databricks, MERGE handles this instead.)
        """
        if not rows:
            return 0
        seen = self._seen_hashes(layer)
        written = 0
        with open(self._path(layer), "a") as fh:
            for r in rows:
                # dedup on the STABLE content_hash (metric+value+source) when the
                # row carries one — this ignores per-run fields like
                # extraction_date, keeping re-runs idempotent. Fall back to
                # hashing the whole row for rows without a content_hash.
                h = r.get("content_hash") or content_hash(
                    json.dumps(r, sort_keys=True, default=str))
                if h in seen:
                    continue
                seen.add(h)
                fh.write(json.dumps(r, default=str) + "\n")
                written += 1
        return written

    def _seen_hashes(self, layer: str) -> set[str]:
        p = self._path(layer)
        if not os.path.exists(p):
            return set()
        seen: set[str] = set()
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    # match the write-side dedup: prefer the stable content_hash
                    # (ignores per-run extraction_date), else hash the whole row.
                    seen.add(row.get("content_hash") or content_hash(
                        json.dumps(row, sort_keys=True, default=str)))
        return seen

    def read_jsonl(self, layer: str) -> list[dict]:
        p = self._path(layer)
        if not os.path.exists(p):
            return []
        with open(p) as fh:
            return [json.loads(line) for line in fh if line.strip()]


class DeltaLakeStorage(StorageManager):
    """Writes Delta tables on Databricks; falls back to JSONL locally.

    ``spark`` is injected on a cluster. When absent, all writes go to JSONL
    so the same code path works in tests.
    """

    def __init__(self, spark=None, root: Optional[str] = None):
        super().__init__(root=root)
        # If no spark session is passed, try to acquire the active one (on
        # Databricks it's always available). Without this, storage silently
        # falls back to writing JSONL files instead of Delta tables — the
        # caller sees "no tables created". Self-acquiring makes the Delta path
        # the default on-cluster; off-cluster (tests) it stays None and uses
        # the file fallback.
        if spark is None:
            try:
                from pyspark.sql import SparkSession
                spark = SparkSession.getActiveSession() or \
                    SparkSession.builder.getOrCreate()
            except Exception:
                spark = None
        self.spark = spark

    @staticmethod
    def build_gold_metric_rows(claims: list[Claim]) -> list[dict]:
        """PURE: build Gold rows from claims. No I/O.

        Filters to METRIC-like claims only — Gold is the structured metrics
        layer, so narrative sentences and headings that were over-extracted as
        claims are dropped here (they remain available via the search index).
        """
        from .dates import looks_like_date, normalise_report_date
        rows: list[dict] = []
        for c in claims:
            if not _is_metric_like(c):
                continue  # drop sentences/headings/noise — not a metric
            field = c.canonical_metric or c.field_name
            value_is_date = looks_like_date(c.value, field)
            # A date-valued claim carries a date, not a metric: surface it on
            # as_at_date and leave numeric_value empty. Prefer an explicit
            # as_at_date on the claim if it already has one.
            as_at = c.as_at_date or (normalise_report_date(c.value)
                                     if value_is_date else None)
            gm = GoldMetric(
                entity_ref=_entity_ref(c),
                canonical_metric=c.canonical_metric or c.field_name,
                field_name=c.field_name,
                element_type=getattr(c, "element_type", None),
                value=c.value,
                numeric_value=_parse_numeric(c.value, field),
                source_document=_source_doc(c),
                page=c.page,
                unit=c.unit,
                currency=getattr(c, "currency", None),
                reporting_basis=c.reporting_basis,
                scale=c.scale,
                as_at_date=as_at,
                netting=c.netting,
                measure_type=c.measure_type,
                period=c.period,
                citation_tier=c.citation_tier.value,
                review_tier=c.review_tier.value,
                domain_tag=c.domain_tag,
            )
            rows.append(gm.model_dump())
        return rows

    def write_gold_metrics(self, claims: list[Claim],
                           team: Optional[str] = None,
                           report_type: Optional[str] = None,
                           report_date: Optional[str] = None,
                           entity_id: Optional[str] = None,
                           entity_name: Optional[str] = None) -> int:
        """THIN: build THEN persist. Returns rows written.

        team/report_type/report_date + entity_id/entity_name are stamped onto
        EVERY row as columns (not just encoded in the table name), so chat/SQL
        can filter by them and recognise the entity.
        """
        rows = self.build_gold_metric_rows(claims)
        rows = _stamp(rows, team=team, report_type=report_type,
                      report_date=report_date, entity_id=entity_id,
                      entity_name=entity_name)
        return self._write_layer("gold", rows, team, report_type)

    def write_silver(self, claims: list[Claim], team: Optional[str] = None,
                     report_type: Optional[str] = None,
                     report_date: Optional[str] = None,
                     entity_id: Optional[str] = None,
                     entity_name: Optional[str] = None) -> int:
        rows = [c.model_dump(mode="json") for c in claims]
        rows = _stamp(rows, team=team, report_type=report_type,
                      report_date=report_date, entity_id=entity_id,
                      entity_name=entity_name)
        return self._write_layer("silver", rows, team, report_type)

    def write_bronze(self, elements: list[dict], team: Optional[str] = None,
                     report_type: Optional[str] = None) -> int:
        return self._write_layer("bronze", elements, team, report_type)

    # ------------------------------------------------------------------
    # ADDITIVE outputs (do not change silver/gold):
    #   * bronze_original  — the RAW ai_parse blob per element (traceability)
    #   * query_gold       — ALL gold claims + an identifiers MAP (tables only)
    #   * json output      — our claims serialised to a Volume as JSON
    # ------------------------------------------------------------------
    def write_bronze_original(self, raw_elements: list[dict]) -> int:
        """Store the RAW ai_parse output per element (HTML table blob / figure
        description / text), untouched, for source-of-truth traceability.
        raw_elements: [{source_document, page, element_type, raw_content, bbox}]."""
        table = getattr(CONFIG.storage, "bronze_original_table", None) \
            or f"{getattr(CONFIG.storage, 'table_prefix', '')}bronze_original"
        return self._write("bronze_original", table, raw_elements)

    @staticmethod
    def build_query_gold_rows(gold_rows: list[dict], claims: list) -> list[dict]:
        """PURE: build query_gold rows = ALL gold rows + an `identifiers` MAP.
        The map is populated for TABLE claims only (from claim.identifiers);
        charts/text get an empty map. Keyed to the claim by
        (canonical_metric, value, page) so the map lands on the right row."""
        # index table claims' identifiers by a stable key
        ident_by_key = {}
        for cl in claims:
            if getattr(cl, "element_type", None) == "table" and getattr(cl, "identifiers", None):
                key = (getattr(cl, "canonical_metric", None) or getattr(cl, "field_name", None),
                       str(getattr(cl, "value", None)), getattr(cl, "page", None))
                ident_by_key[key] = cl.identifiers
        out = []
        for r in gold_rows:
            key = (r.get("canonical_metric"), str(r.get("value")), r.get("page"))
            row = dict(r)
            row["identifiers"] = ident_by_key.get(key) or {}   # map (empty for non-tables)
            out.append(row)
        return out

    def write_query_gold(self, gold_rows: list[dict], claims: list) -> int:
        """Write query_gold: all gold claims + identifiers map (tables only)."""
        rows = self.build_query_gold_rows(gold_rows, claims)
        table = getattr(CONFIG.storage, "query_gold_table", None) \
            or f"{getattr(CONFIG.storage, 'table_prefix', '')}query_gold"
        return self._write("query_gold", table, rows)

    def write_json_output(self, claims: list, document: str,
                          json_dir: str = None, dbutils=None) -> str:
        """Serialise our claims for ONE document to a JSON file in a Volume.
        Returns the path written. Portable record of what we extracted."""
        import json, os
        json_dir = json_dir or getattr(CONFIG.storage, "json_dir", None) \
            or "/Volumes/main/default/docstore/json"
        payload = []
        for cl in claims:
            d = cl.model_dump() if hasattr(cl, "model_dump") else dict(getattr(cl, "__dict__", {}))
            # make enums/tuples JSON-safe
            payload.append({k: (v.value if hasattr(v, "value") else
                                list(v) if isinstance(v, tuple) else v)
                            for k, v in d.items()})
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(document)) if 're' in dir() else str(document)
        path = f"{json_dir.rstrip('/')}/{safe}.json"
        local = path.replace("/Volumes", "/dbfs/Volumes") if path.startswith("/Volumes") else path
        try:
            os.makedirs(os.path.dirname(local), exist_ok=True)
            with open(local, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            return ""
        return path

    def write_quarantine(self, rows: list[dict]) -> int:
        return self._write("quarantine", CONFIG.storage.quarantine_table, rows)

    def write_reviews(self, rows: list[dict]) -> int:
        return self._write("reviews", CONFIG.storage.review_table, rows)

    def write_review_claims(self, claims: list[Claim], team: Optional[str] = None,
                            report_type: Optional[str] = None,
                            report_date: Optional[str] = None,
                            entity_id: Optional[str] = None,
                            entity_name: Optional[str] = None) -> int:
        """Persist the claims flagged for human review to a queryable table, so
        reviewers can SEE exactly what needs checking (metric, value, why it was
        flagged, confidence, and the source page) — instead of it being just a
        count.

        IMPORTANT: review claims are written AS-IS (like Silver), NOT through the
        metric-like filter that Gold uses. Flagged claims are often the borderline
        / low-confidence / non-metric items — that is precisely WHY they need a
        human — so filtering them through the Gold metric gate would drop the very
        claims the review queue exists to hold (and, if all of them were dropped,
        the table would never be created)."""
        rows = [c.model_dump(mode="json") for c in claims]      # AS-IS, no filter
        # annotate WHY each is under review, from the claim
        for r, c in zip(rows, claims):
            r["review_reason"] = ("conflict" if getattr(c, "needs_review", False)
                                  and (c.canonical_metric or c.field_name) else
                                  "low_confidence")
            r["confidence"] = getattr(c, "confidence", None)
        rows = _stamp(rows, team=team, report_type=report_type,
                      report_date=report_date, entity_id=entity_id,
                      entity_name=entity_name)
        return self._write("review_claims", CONFIG.storage.review_claims_table, rows)

    def write_audit(self, rows: list[dict]) -> int:
        return self._write("audit", CONFIG.storage.audit_table, rows)

    def write_analytical_gold(self, gold_rows: list[dict], team: str,
                              report_type: str, entity_id=None,
                              entity_name=None) -> int:
        """Explode Gold rows into the analytical layer (statement / classification
        / category / sub_category + period + field_name + currency + unit +
        report_date + traceability) and write them to a PER-TEAM table
        ``analytical_gold_<team>_<report>`` — mirroring the Gold table structure.
        entity_id / entity_name (parsed from the volume path) are stamped on so
        people can recognise the entity. Rule-based, no LLM."""
        from .analytical_gold import build_analytical_rows
        rows = build_analytical_rows(gold_rows)
        rows = _stamp(rows, team=team, report_type=report_type,
                      entity_id=entity_id, entity_name=entity_name)
        table = CONFIG.storage.layer_table("analytical_gold", team, report_type)
        return self._write("analytical", table, rows)

    def _write_layer(self, layer: str, rows: list[dict],
                     team: Optional[str], report_type: Optional[str]) -> int:
        """Write a medallion layer, partitioned by team+report_type when both
        are supplied. Falls back to a single shared table/file otherwise (so
        existing callers and tests keep working)."""
        if team and report_type:
            table = CONFIG.storage.layer_table(layer, team, report_type)
            local = CONFIG.storage.layer_local(layer, team, report_type)
        else:
            table = getattr(CONFIG.storage, f"{layer}_table", None) \
                or CONFIG.storage.gold_metrics
            local = layer
        if self.spark is not None and rows:
            typed = _rows_for_explicit_schema(rows)
            df = self.spark.createDataFrame(typed, schema=_spark_schema_for(typed))
            df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table)
            return len(rows)
        return self.append_jsonl(local, rows)

    def _write(self, layer: str, table: str, rows: list[dict]) -> int:
        if self.spark is not None and rows:
            typed = _rows_for_explicit_schema(rows)
            df = self.spark.createDataFrame(typed, schema=_spark_schema_for(typed))
            df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table)
            return len(rows)
        return self.append_jsonl(layer, rows)


def _stamp(rows: list[dict], team=None, report_type=None,
          report_date=None, entity_id=None, entity_name=None) -> list[dict]:
    """Add team / report_type / report_date onto every row as columns, so the
    curated tables are filterable by them (WHERE team = ...) without relying on
    the table name.

    Also stamps entity_id and entity_name (parsed from the document's volume
    path) so people can recognise which entity a metric belongs to, plus two
    operational columns:
      * extraction_date — the UTC timestamp of THIS run.
      * content_hash — a stable key over the row's identifying content.

    Uses DIRECT assignment, not setdefault: the rows come from model_dump(),
    which already includes team/report_type keys set to None, so setdefault
    would be a no-op and leave them empty.
    """
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for r in rows:
        if team is not None:
            r["team"] = team
        if report_type is not None:
            r["report_type"] = report_type
        if report_date is not None:
            r["report_date"] = report_date
        if entity_id is not None:
            r["entity_id"] = entity_id
        if entity_name is not None:
            r["entity_name"] = entity_name
        r["extraction_date"] = now
        # stable export/promote key over the row's identity (metric + value +
        # source), independent of the run timestamp.
        key_src = "|".join(str(r.get(k, "")) for k in
                           ("canonical_metric", "field_name", "value",
                            "source_document", "page", "entity_ref"))
        r["content_hash"] = content_hash(key_src)
    return rows


def _spark_schema_for(rows: list[dict]):
    """Build an EXPLICIT Spark schema for the rows, so types are never inferred.

    This is the permanent fix for both failure modes of inference:
      * CANNOT_DETERMINE_TYPE — an all-None column can't be typed by inference;
      * CAST_INVALID_INPUT — inference guessing a column is string ("") when the
        existing table column is numeric.
    Known numeric columns are declared with their real types (nullable);
    everything else is a nullable string. Imported lazily so the module still
    imports off-cluster (tests), where pyspark isn't present.
    """
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


def _rows_for_explicit_schema(rows: list[dict]) -> list[dict]:
    """Coerce row values to match the explicit schema: numeric columns become
    real numbers or None (never ""), string columns become strings or None."""
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


def _typeable_rows(rows: list[dict]) -> list[dict]:
    """Make rows safe for Spark's schema inference WITHOUT flattening real
    numbers to strings.

    Spark infers column types from the data; a column that is None in every row
    can't be typed and raises CANNOT_DETERMINE_TYPE. We fix that per column:
      * columns KNOWN to be numeric (numeric_value) always stay numeric — None
        stays None (never ""), so a batch where every value is null still writes
        NULLs into the DOUBLE column instead of an empty string (which would
        fail the cast against an existing DOUBLE column);
      * a column that holds at least one number keeps its numbers, None -> None;
      * a column that is entirely None (and not known-numeric) is filled with ""
        so it types as string rather than failing inference;
      * everything else is coerced to its string form (None -> "").
    """
    if not rows:
        return rows
    # Columns that must ALWAYS be numeric, regardless of this batch's values —
    # so an all-null batch doesn't turn them into string columns (which then
    # fails the cast against an existing numeric column). Covers every numeric
    # column across gold + silver rows.
    KNOWN_NUMERIC = {"numeric_value", "page", "confidence"}

    keys = set()
    for r in rows:
        keys.update(r.keys())

    numeric_cols = set(KNOWN_NUMERIC)
    allnull_cols = set()
    for k in keys:
        if k in KNOWN_NUMERIC:
            continue  # already forced numeric
        vals = [r.get(k) for r in rows]
        non_null = [v for v in vals if v is not None]
        if not non_null:
            allnull_cols.add(k)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool)
                 for v in non_null):
            numeric_cols.add(k)

    out: list[dict] = []
    for r in rows:
        clean: dict = {}
        for k in keys:
            v = r.get(k)
            if k in numeric_cols:
                # keep as a real number; None stays None (NULL), never ""
                if v is None:
                    clean[k] = None
                else:
                    try:
                        # page is an integer column; keep it int, others float
                        clean[k] = int(v) if k == "page" else float(v)
                    except (ValueError, TypeError):
                        clean[k] = None  # unparseable -> NULL, not a bad cast
            elif k in allnull_cols:
                clean[k] = ""
            elif v is None:
                clean[k] = ""
            elif isinstance(v, (str, int, float, bool)):
                clean[k] = v if isinstance(v, str) else str(v)
            else:
                clean[k] = str(v)
        out.append(clean)
    return out


def _parse_numeric(value, field_name=None) -> Optional[float]:
    """Best-effort parse of a metric string into a float for dashboards.
    Handles percent signs, thousands separators, currency, surrounding text
    ("~750", "1,234.5%", "$12.4m"). Returns None when there's no parseable
    number OR when the value is a date (context-aware — a bare year is only a
    date if field_name signals one). Does NOT apply scale multipliers."""
    if value is None:
        return None
    # Dates are NOT metrics: context-aware so a genuine numeric metric (even a
    # 4-digit one like 1434, or a bare year used as a count) is not dropped.
    from .dates import looks_like_date
    if looks_like_date(value, field_name):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    import re
    s = str(value).strip()
    # NEGATIVE when wrapped in accounting parentheses "(123)" or a leading minus.
    negative = (s.startswith("(") and s.endswith(")")) or bool(re.match(r"\s*-", s))
    # find the first number, allowing thousands separators and a decimal part
    m = re.search(r"\d+(?:,\d{3})*(?:\.\d+)?", s)
    if not m:
        return None
    try:
        num = float(m.group(0).replace(",", ""))
        return -num if negative else num
    except (ValueError, TypeError):
        return None


def derive_report_date(claims: list, elements: list = None,
                       path: str = None) -> Optional[str]:
    """Find the document's report date so every stored row can be stamped with it.

    Sources, in order of reliability for THIS corpus:
      1. The FILENAME — the report date is encoded in the document name
         (e.g. 'LCR_BankABC_February_2026.pptx' -> 'February 2026'). This is the
         primary source: filenames are named consistently with the period.
      2. The report TITLE / heading text — a date embedded in the title
         (e.g. 'Micro Report 2024 Q1').
      3. A date-valued CLAIM — a claim whose value is a period/date.
    Returns None if no date is found — the column stays empty rather than guess.
    """
    import os
    from .dates import (looks_like_date, normalise_report_date, _DATE_FIELD_RE,
                        _DATE_PATTERNS, _BARE_YEAR_RE)

    # 1. FILENAME — extract a date from the document's file name (primary).
    if path:
        fname = os.path.basename(str(path))
        # strip the extension so "..._2026.pptx" doesn't confuse the parser
        stem = os.path.splitext(fname)[0]
        # filenames often use underscores/dashes as separators -> spaces
        stem_spaced = stem.replace("_", " ").replace("-", " ")
        found = _extract_date_from_text(stem_spaced)
        if found:
            return found

    # 2. TITLE / heading text — search the first several text elements for a
    #    date/period embedded anywhere in the heading.
    if elements:
        def _is_text_el(e):
            # support both shapes: ai_parse ParsedElement (.el_type) and the
            # older modality-based elements (.modality.value).
            et = getattr(e, "el_type", None)
            if et is not None:
                return str(et).lower() in ("text", "title", "section_header",
                                           "caption", "heading")
            mod = getattr(getattr(e, "modality", None), "value",
                          getattr(e, "modality", ""))
            return str(mod).lower() == "text"
        text_els = [e for e in elements if _is_text_el(e)]
        for e in text_els[:8]:  # titles/headings are near the top
            content = (getattr(e, "content", "") or "") or (getattr(e, "description", "") or "")
            found = _extract_date_from_text(content)
            if found:
                return found

    # 2. date-valued claim (fallback)
    dated = []
    for c in claims:
        field = getattr(c, "canonical_metric", None) or getattr(c, "field_name", "")
        val = getattr(c, "value", None)
        if looks_like_date(val, field):
            nd = normalise_report_date(val)
            if nd:
                dated.append((field or "", nd))
    if dated:
        for field, nd in dated:
            if _DATE_FIELD_RE.search(field):
                return nd
        return dated[0][1]
    return None


def _extract_date_from_text(text: str) -> Optional[str]:
    """Pull a date/period out of a heading/title string (searches WITHIN the
    text, e.g. 'Micro Report 2024 Q1' -> '2024 Q1'). Returns None if none."""
    import re
    from .dates import _MONTHS
    if not text:
        return None
    s = str(text)
    # quarter/half + year in any order, month+year, or a bare year — searched
    # anywhere in the string (not anchored), so titles with surrounding words hit
    patterns = [
        r"(q[1-4]|h[12])\s*[- ]?\s*\d{4}",
        r"\d{4}\s*[- ]?\s*(q[1-4]|h[12])",
        rf"({_MONTHS})\s+\d{{4}}",
        r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}",
        r"\b(19|20)\d{2}\b",
    ]
    for pat in patterns:
        m = re.search(pat, s, re.I)
        if m:
            return m.group(0).strip()
    return None


def _is_metric_like(c) -> bool:
    """True if a claim looks like a real METRIC (belongs in Gold), False if it's
    narrative noise — a sentence or heading that was over-extracted as a claim.

    Gold is the structured metrics layer; prose belongs in the search index, not
    here. A value is metric-like if it is short and either numeric, a date, or a
    recognised categorical status (N/A, nil, compliant, ...). Long multi-word
    prose values are treated as noise and dropped.
    """
    from .dates import looks_like_date

    value = (getattr(c, "value", None) or "").strip()
    if not value:
        return False

    # numeric or date values are metric-like
    if _parse_numeric(value, getattr(c, "canonical_metric", None)
                      or getattr(c, "field_name", "")) is not None:
        return True
    if looks_like_date(value):
        return True

    # recognised non-numeric metric values (status/flags) are metric-like
    _CATEGORICAL = {"n/a", "na", "nil", "none", "compliant", "non-compliant",
                    "not applicable", "not reported", "exempt", "yes", "no",
                    "pass", "fail", "true", "false", "-", "—"}
    if value.lower() in _CATEGORICAL:
        return True

    # otherwise: metric-like only if the value is SHORT (a value, not a
    # sentence). A long, many-word value is prose/heading noise -> drop.
    words = value.split()
    if len(words) > 4 or len(value) > 40:
        return False
    # a short value that is a Title-Case heading (2+ capitalised words, e.g.
    # "Risk Management Overview") is a heading, not a metric value -> drop.
    cap_words = [w for w in words if w[:1].isupper()]
    if len(words) >= 2 and len(cap_words) >= 2:
        return False
    # a short value containing a sentence verb/connective is prose, not a
    # metric value ("this report summarises ...") -> drop.
    _PROSE = {"the", "this", "that", "is", "are", "was", "were", "summarises",
              "summarizes", "shows", "includes", "provides", "and", "of", "for",
              "with", "report", "section", "overview", "summary"}
    if len(words) >= 3 and any(w.lower() in _PROSE for w in words):
        return False
    return True


def _source_doc(c: Claim) -> Optional[str]:
    """Best-effort source document name for a claim, for user verification.
    The source_element_id is built as '<filename>-<locator>' (e.g.
    'q1_report.pptx-s2-img0'), so the document is the part before the first
    locator suffix. Returns None if it can't be determined."""
    import re
    sid = getattr(c, "source_element_id", "") or ""
    if not sid:
        return None
    # strip known locator suffixes: -s{n}-img{n} / -p{n}-fig{n} / -tbl{n} / -img{n}
    m = re.split(r"-(?:s\d+-|p\d+-)?(?:img|fig|tbl|p)\d+", sid)
    return m[0] if m and m[0] else sid


def _entity_ref(c: Claim) -> str:
    """entity_ref now genuinely originates from the Claim (propagated from the
    source Element). The sentinel only appears if attribution truly failed
    upstream, which the pipeline routes to review rather than silently
    trusting."""
    return c.entity_ref or "UNKNOWN_ENTITY_REF"
