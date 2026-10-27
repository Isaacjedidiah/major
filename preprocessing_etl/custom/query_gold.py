"""Fast structured metric queries over the per-team Gold / analytical tables.

Gold is stored per team+report (``prod_gold_<team>_<report>``), so the fastest
query touches only the tables the filters actually need, rather than unioning
every table and then filtering:

  * team + report given  -> one table
  * team only            -> that team's tables
  * neither              -> union all prod_gold_* tables (cross-team query)

Sensitive tables use a different prefix (``limited_``) and therefore never match
``prod_gold_*`` — so they are structurally excluded from these shared queries.

This is the SQL side of the app's retrieval: the router sends structured /
metric / trend questions here; narrative questions go to the vector index.
"""

from __future__ import annotations

import re
from typing import Optional


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")


def _list_gold_tables(spark, catalog_schema: str, prefix: str,
                      team: Optional[str], report_type: Optional[str]) -> list:
    """Filter-FIRST: choose the smallest set of tables the filters require."""
    like = f"{prefix}gold_"
    if team and report_type:
        # one table — fastest path
        name = f"{prefix}gold_{_slug(team)}_{_slug(report_type)}"
        return [name]
    if team:
        like = f"{prefix}gold_{_slug(team)}_"
    rows = spark.sql(f"SHOW TABLES IN {catalog_schema} LIKE '{like}*'").collect()
    # never include the sensitive/limited tables or non-gold ops tables; and if a
    # team is given, keep only that team's gold tables (belt-and-braces on top of
    # the SQL LIKE, so a mismatch in LIKE behaviour can't leak other teams).
    team_slug = _slug(team) if team else None
    out = []
    for r in rows:
        t = r.tableName
        if not t.startswith(f"{prefix}gold_"):
            continue
        if "quarantine" in t or "needs_review" in t:
            continue
        if team_slug and not t.startswith(f"{prefix}gold_{team_slug}_"):
            continue
        out.append(t)
    return out


def query_gold(spark, *, catalog_schema="docextract.reporting", prefix="prod_",
               team=None, report_type=None, source_document=None,
               entity=None, entity_id=None, metric_like=None, limit=1000):
    """Return a Spark DataFrame of Gold metrics matching the filters.

    Filter-first table selection (only touches the tables the team/report filter
    needs) THEN a UNION of just those, THEN row filters (source_document /
    entity / metric). Sensitive ``limited_*`` tables are never included.

    Args:
      team / report_type  -> choose which tables to read (fast path).
      source_document     -> filter to one document (e.g. review a new upload).
      entity / entity_id  -> filter to an entity.
      metric_like         -> substring match on the metric name (e.g. 'revenue').
    """
    tables = _list_gold_tables(spark, catalog_schema, prefix, team, report_type)
    if not tables:
        return spark.createDataFrame([], schema="canonical_metric string")

    # build the UNION of only the needed tables
    parts = [f"SELECT * FROM {catalog_schema}.{t}" for t in tables]
    sql = " UNION ALL ".join(parts)

    df = spark.sql(sql)

    # row filters (Spark pushes these down into each table scan)
    if team:
        df = df.filter(df["team"] == team)
    if report_type:
        df = df.filter(df["report_type"] == report_type)
    if source_document:
        df = df.filter(df["source_document"].contains(source_document))
    if entity:
        df = df.filter(df["entity_name"] == entity)
    if entity_id:
        df = df.filter(df["entity_id"] == entity_id)
    if metric_like:
        df = df.filter(df["canonical_metric"].rlike("(?i)" + re.escape(metric_like)))

    return df.limit(limit)
