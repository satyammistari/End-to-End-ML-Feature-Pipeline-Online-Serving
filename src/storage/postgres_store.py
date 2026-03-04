"""
PostgreSQL + TimescaleDB offline feature store.
Handles historical feature storage, time-series queries, and training data extraction.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool

from ..core.schemas import FeatureGroup, FeatureRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL templates
# ---------------------------------------------------------------------------

_CREATE_HYPERTABLE_SQL = """
CREATE TABLE IF NOT EXISTS {table} (
    id          BIGSERIAL,
    entity_id   TEXT        NOT NULL,
    entity_type TEXT        NOT NULL,
    feature_group TEXT      NOT NULL,
    feature_version TEXT    NOT NULL,
    feature_name TEXT       NOT NULL,
    feature_value JSONB     NOT NULL,
    event_time  TIMESTAMPTZ NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, event_time)
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS {table}_entity_time_idx
    ON {table} (entity_id, feature_name, event_time DESC);
"""

_SELECT_TIMESCALEDB = """
SELECT to_regclass('timescaledb_information.hypertables') IS NOT NULL;
"""

_CONVERT_HYPERTABLE_SQL = """
SELECT create_hypertable('{table}', 'event_time', if_not_exists => TRUE);
"""


class PostgresFeatureStore:
    """
    Offline feature store backed by PostgreSQL (+ optional TimescaleDB).

    All feature values are stored as JSONB, keyed by
    (entity_id, feature_name, event_time), enabling efficient
    point-in-time queries and range scans.

    Parameters
    ----------
    host : str
    port : int
    database : str
    user : str
    password : str
    min_conn / max_conn : connection pool sizes
    table : str – hypertable name
    use_timescaledb : bool – set True if TimescaleDB is installed
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "features",
        user: str = "feature_user",
        password: str = "",
        min_conn: int = 2,
        max_conn: int = 20,
        table: str = "feature_values",
        use_timescaledb: bool = True,
    ) -> None:
        self._table = table
        self._use_timescale = use_timescaledb
        self._pool = ThreadedConnectionPool(
            minconn=min_conn,
            maxconn=max_conn,
            host=host,
            port=port,
            dbname=database,
            user=user,
            password=password,
            cursor_factory=psycopg2.extras.RealDictCursor,
        )
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE_HYPERTABLE_SQL.format(table=self._table))
                cur.execute(_CREATE_INDEX_SQL.format(table=self._table))
                if self._use_timescale:
                    try:
                        cur.execute(_CONVERT_HYPERTABLE_SQL.format(table=self._table))
                    except Exception as exc:
                        logger.warning("TimescaleDB convert failed (may already exist): %s", exc)
                        conn.rollback()
            conn.commit()
        logger.info("PostgreSQL schema ready (table=%s)", self._table)

    def create_feature_table(self, feature_group: FeatureGroup) -> None:
        """Create a dedicated table for a feature group (optional)."""
        table = f"features_{feature_group.name}"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(_CREATE_HYPERTABLE_SQL.format(table=table))
                cur.execute(_CREATE_INDEX_SQL.format(table=table))
                if self._use_timescale:
                    try:
                        cur.execute(_CONVERT_HYPERTABLE_SQL.format(table=table))
                    except Exception:
                        conn.rollback()
            conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_features_batch(
        self,
        records: List[FeatureRecord],
        chunk_size: int = 1000,
    ) -> int:
        """
        Bulk-insert feature records using execute_values for efficiency.
        Returns the number of rows inserted.
        """
        rows = []
        for rec in records:
            event_time = rec.event_time or rec.computed_at
            for fname, fval in rec.features.items():
                rows.append((
                    rec.entity_id,
                    rec.entity_type,
                    rec.feature_group,
                    rec.feature_version,
                    fname,
                    psycopg2.extras.Json(fval),
                    event_time,
                    rec.computed_at,
                ))

        if not rows:
            return 0

        sql = f"""
        INSERT INTO {self._table}
            (entity_id, entity_type, feature_group, feature_version,
             feature_name, feature_value, event_time, computed_at)
        VALUES %s
        ON CONFLICT DO NOTHING
        """
        total = 0
        with self._conn() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(rows), chunk_size):
                    chunk = rows[i: i + chunk_size]
                    psycopg2.extras.execute_values(cur, sql, chunk)
                    total += cur.rowcount
            conn.commit()
        return total

    # ------------------------------------------------------------------
    # Read – point-in-time
    # ------------------------------------------------------------------

    def read_features_point_in_time(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        timestamp: datetime,
        feature_group: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Return the most-recent feature value for each (entity, feature_name)
        that was computed *at or before* `timestamp`.

        This prevents data leakage in training pipelines.
        """
        placeholders_ids = ",".join(["%s"] * len(entity_ids))
        placeholders_fnames = ",".join(["%s"] * len(feature_names))

        group_filter = "AND feature_group = %s" if feature_group else ""
        group_args = [feature_group] if feature_group else []

        sql = f"""
        SELECT DISTINCT ON (entity_id, feature_name)
            entity_id,
            feature_name,
            feature_value
        FROM {self._table}
        WHERE entity_id IN ({placeholders_ids})
          AND feature_name IN ({placeholders_fnames})
          AND event_time <= %s
          {group_filter}
        ORDER BY entity_id, feature_name, event_time DESC
        """

        args = list(entity_ids) + list(feature_names) + [timestamp] + group_args

        result: Dict[str, Dict[str, Any]] = {eid: {} for eid in entity_ids}
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                for row in cur.fetchall():
                    eid = row["entity_id"]
                    fname = row["feature_name"]
                    val = row["feature_value"]
                    result[eid][fname] = val
        return result

    def read_training_data(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        start_time: datetime,
        end_time: datetime,
        feature_group: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return all feature rows in a time range (for building training datasets).
        """
        placeholders_ids = ",".join(["%s"] * len(entity_ids))
        placeholders_fnames = ",".join(["%s"] * len(feature_names))
        group_filter = "AND feature_group = %s" if feature_group else ""
        group_args = [feature_group] if feature_group else []

        sql = f"""
        SELECT entity_id, entity_type, feature_group, feature_version,
               feature_name, feature_value, event_time, computed_at
        FROM {self._table}
        WHERE entity_id IN ({placeholders_ids})
          AND feature_name IN ({placeholders_fnames})
          AND event_time >= %s
          AND event_time <= %s
          {group_filter}
        ORDER BY entity_id, feature_name, event_time
        """
        args = list(entity_ids) + list(feature_names) + [start_time, end_time] + group_args

        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                return [dict(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def save_feature_group(self, group_dict: Dict[str, Any]) -> None:
        sql = """
        INSERT INTO feature_registry (name, data, updated_at)
        VALUES (%(name)s, %(data)s, NOW())
        ON CONFLICT (name) DO UPDATE SET data = EXCLUDED.data, updated_at = NOW()
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS feature_registry (
                        name TEXT PRIMARY KEY,
                        data JSONB NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute(sql, {"name": group_dict["name"],
                                  "data": psycopg2.extras.Json(group_dict)})
            conn.commit()

    def get_feature_group_raw(self, name: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        "SELECT data FROM feature_registry WHERE name = %s", (name,)
                    )
                    row = cur.fetchone()
                    return row["data"] if row else None
                except psycopg2.errors.UndefinedTable:
                    return None

    def list_feature_groups(self) -> List[str]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("SELECT name FROM feature_registry ORDER BY name")
                    return [r["name"] for r in cur.fetchall()]
                except psycopg2.errors.UndefinedTable:
                    return []

    def delete_feature_group(self, name: str) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM feature_registry WHERE name = %s", (name,))
            conn.commit()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    @contextmanager
    def _conn(self) -> Generator:
        conn = self._pool.getconn()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)
