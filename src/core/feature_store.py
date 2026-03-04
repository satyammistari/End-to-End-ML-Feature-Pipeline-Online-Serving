"""
FeatureStore: top-level orchestrator that coordinates the online store (Redis),
offline store (PostgreSQL), registry, and validation layers.

Also houses FeatureVersion for semantic versioning / rollout management.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from .registry import FeatureRegistry, RegistryError
from .schemas import FeatureRecord, OnlineFeatureRequest, OnlineFeatureResponse
from .validators import FeatureValidator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature version lifecycle
# ---------------------------------------------------------------------------

class FeatureVersion:
    """
    Tracks rollout state for a specific feature group version.

    Parameters
    ----------
    group : str
        Feature group name.
    version : str
        Version label, e.g. "v2".
    rollout_pct : float
        Fraction of traffic served this version (0.0-1.0).
    """

    def __init__(
        self,
        group: str,
        version: str,
        rollout_pct: float = 1.0,
        deprecated: bool = False,
        deprecation_message: str = "",
    ) -> None:
        self.group = group
        self.version = version
        self.rollout_pct = rollout_pct
        self.deprecated = deprecated
        self.deprecation_message = deprecation_message
        self._created_at = datetime.utcnow()

    def should_serve(self, entity_id: str) -> bool:
        """Consistent-hash based traffic split."""
        if self.rollout_pct >= 1.0:
            return True
        if self.rollout_pct <= 0.0:
            return False
        bucket = hash(entity_id) % 100
        return bucket < int(self.rollout_pct * 100)

    def __repr__(self) -> str:
        return (
            f"FeatureVersion(group={self.group!r}, version={self.version!r}, "
            f"rollout={self.rollout_pct:.0%}, deprecated={self.deprecated})"
        )


# ---------------------------------------------------------------------------
# Main FeatureStore
# ---------------------------------------------------------------------------

class FeatureStore:
    """
    High-level API for reading and writing features.

    Responsibilities:
    - Route reads to the online store (Redis) with fallback to offline (Postgres)
    - Validate feature records on write
    - Consult the registry for schema and version information
    - Emit latency metrics for every operation

    Parameters
    ----------
    registry : FeatureRegistry
    online_store : RedisFeatureStore (optional)
    offline_store : PostgresFeatureStore (optional)
    metrics_collector : MetricsCollector (optional)
    """

    def __init__(
        self,
        registry: FeatureRegistry,
        online_store: Any = None,
        offline_store: Any = None,
        metrics_collector: Any = None,
    ) -> None:
        self._registry = registry
        self._online = online_store
        self._offline = offline_store
        self._metrics = metrics_collector
        self._validator = FeatureValidator()
        self._version_overrides: Dict[str, FeatureVersion] = {}

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def write_features(
        self,
        record: FeatureRecord,
        validate: bool = True,
        online: bool = True,
        offline: bool = True,
    ) -> None:
        """
        Persist a FeatureRecord to online and/or offline stores.

        Validates against the registered schema before writing (can be disabled
        for performance in bulk backfill jobs by setting validate=False).
        """
        t0 = time.monotonic()
        if validate:
            schema = self._registry.get_version(
                record.feature_group, record.feature_version
            )
            if schema:
                errors = self._validator.validate_record(record, schema)
                if errors:
                    if self._metrics:
                        self._metrics.feature_validation_errors.inc(len(errors))
                    raise ValueError(f"Validation failed: {errors}")

        if online and self._online:
            try:
                self._online.write_features(
                    entity_id=record.entity_id,
                    feature_group=record.feature_group,
                    features=record.features,
                    ttl=record.ttl_seconds,
                )
            except Exception as exc:
                logger.error("Online store write failed: %s", exc)

        if offline and self._offline:
            try:
                self._offline.write_features_batch([record])
            except Exception as exc:
                logger.error("Offline store write failed: %s", exc)

        elapsed = time.monotonic() - t0
        if self._metrics:
            self._metrics.feature_write_latency.observe(elapsed)

    async def write_features_async(
        self, record: FeatureRecord, **kwargs: Any
    ) -> None:
        """Async wrapper around write_features."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: self.write_features(record, **kwargs))

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_online_features(self, request: OnlineFeatureRequest) -> List[OnlineFeatureResponse]:
        """
        Serve features from the online store (Redis).

        Falls back to the offline store for any entity not found in Redis.
        Records cache hit/miss ratio.
        """
        t0 = time.monotonic()
        version = request.feature_version
        if version is None:
            group = self._registry.get_feature_group(request.feature_group)
            version = group.latest_version if group else "v1"

        results: List[OnlineFeatureResponse] = []
        cache_hits = 0

        if self._online:
            online_results = self._online.read_features(
                entity_ids=request.entity_ids,
                feature_group=request.feature_group,
                feature_names=request.feature_names,
            )
            hit_ids = set()
            for entity_id, features in online_results.items():
                results.append(
                    OnlineFeatureResponse(
                        entity_id=entity_id,
                        features=features,
                        version=version,
                        cache_hit=True,
                    )
                )
                hit_ids.add(entity_id)
                cache_hits += 1

            # Fallback for misses
            miss_ids = [eid for eid in request.entity_ids if eid not in hit_ids]
            if miss_ids and self._offline:
                offline_results = self._fallback_offline(
                    miss_ids, request.feature_group, request.feature_names, version
                )
                results.extend(offline_results)
        elif self._offline:
            results = self._fallback_offline(
                request.entity_ids,
                request.feature_group,
                request.feature_names,
                version,
            )

        elapsed = time.monotonic() - t0
        if self._metrics:
            self._metrics.feature_read_latency.observe(elapsed)
            total = len(request.entity_ids)
            if total > 0:
                self._metrics.feature_cache_hit_ratio.set(cache_hits / total)

        return results

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def set_version_rollout(
        self,
        group: str,
        version: str,
        rollout_pct: float,
    ) -> None:
        """Control what percentage of traffic gets a specific version."""
        self._version_overrides[f"{group}/{version}"] = FeatureVersion(
            group=group, version=version, rollout_pct=rollout_pct
        )

    def get_serving_version(self, group: str, entity_id: str) -> str:
        """
        Return the version to serve for a given entity, respecting rollout %.
        """
        fg = self._registry.get_feature_group(group)
        if not fg or not fg.latest_version:
            return "v1"

        # Check for non-100% rollout overrides
        for key, fv in self._version_overrides.items():
            if fv.group == group and fv.should_serve(entity_id):
                return fv.version

        return fg.latest_version

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fallback_offline(
        self,
        entity_ids: List[str],
        feature_group: str,
        feature_names: List[str],
        version: str,
    ) -> List[OnlineFeatureResponse]:
        results = []
        if not self._offline:
            return results
        try:
            data = self._offline.read_features_point_in_time(
                entity_ids=entity_ids,
                feature_names=feature_names,
                timestamp=datetime.utcnow(),
                feature_group=feature_group,
            )
            for entity_id, features in data.items():
                filtered = {k: v for k, v in features.items() if k in feature_names}
                results.append(
                    OnlineFeatureResponse(
                        entity_id=entity_id,
                        features=filtered,
                        version=version,
                        cache_hit=False,
                    )
                )
        except Exception as exc:
            logger.error("Offline fallback failed: %s", exc)
        return results


# ---------------------------------------------------------------------------
# FeatureStoreWithValidation  (test / validation helper)
# ---------------------------------------------------------------------------

try:
    import json
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore

    class FeatureStoreWithValidation:
        """
        Lightweight feature store used in integration tests.
        Validates computed feature values against pre-generated expected values,
        writes to Redis (online) and PostgreSQL (offline).
        """

        def __init__(self, redis_client, postgres_conn):
            self.redis = redis_client
            self.postgres = postgres_conn
            self.validation_mode = True
            self.validation_errors: list = []

            try:
                self.expected_features = pd.read_csv('/tmp/test_data/expected_features.csv')
                self.expected_features['timestamp'] = pd.to_datetime(
                    self.expected_features['timestamp']
                )
                logger.info("Loaded expected features for validation")
            except FileNotFoundError:
                self.expected_features = None
                logger.warning("No expected features found, validation disabled")

        def write_features(
            self,
            entity_id: str,
            features: dict,
            timestamp,
            transaction_id=None,
        ):
            if self.validation_mode and self.expected_features is not None:
                self._validate_features(entity_id, features, timestamp, transaction_id)
            self._write_to_redis(entity_id, features, timestamp)
            self._write_to_postgres(entity_id, features, timestamp)

        def _validate_features(self, entity_id, features, timestamp, transaction_id=None):
            if transaction_id:
                expected = self.expected_features[
                    self.expected_features['transaction_id'] == transaction_id
                ]
            else:
                expected = self.expected_features[
                    (self.expected_features['user_id'] == entity_id)
                    & (self.expected_features['timestamp'] == timestamp)
                ]

            if expected.empty:
                return

            expected_row = expected.iloc[0]
            mismatches = []

            for fname, computed in features.items():
                if fname not in expected_row:
                    continue
                exp = expected_row[fname]
                if pd.isna(exp) and pd.isna(computed):
                    continue
                if pd.isna(exp) or pd.isna(computed):
                    mismatches.append({'feature': fname, 'expected': exp, 'computed': computed})
                    continue
                if isinstance(computed, (int, float)) and isinstance(exp, (int, float)):
                    if not np.isclose(float(computed), float(exp), rtol=0.01, atol=0.01):
                        mismatches.append({
                            'feature': fname,
                            'expected': float(exp),
                            'computed': float(computed),
                            'diff': abs(float(computed) - float(exp)),
                        })
                elif computed != exp:
                    mismatches.append({'feature': fname, 'expected': exp, 'computed': computed})

            if mismatches:
                self.validation_errors.append({
                    'entity_id': entity_id,
                    'timestamp': timestamp,
                    'transaction_id': transaction_id,
                    'mismatches': mismatches,
                })
                raise ValueError(f"Feature validation failed: {len(mismatches)} mismatches")

        def _write_to_redis(self, entity_id, features, timestamp):
            key = f"features:user:{entity_id}"
            data = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                for k, v in features.items()
            }
            data['_timestamp'] = str(timestamp)
            pipe = self.redis.pipeline()
            pipe.hset(key, mapping=data)
            pipe.expire(key, 3600)
            pipe.execute()

        def _write_to_postgres(self, entity_id, features, timestamp):
            cursor = self.postgres.cursor()
            for name, value in features.items():
                cursor.execute(
                    """
                    INSERT INTO features (entity_id, feature_name, feature_value, timestamp)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (entity_id, feature_name, timestamp)
                    DO UPDATE SET feature_value = EXCLUDED.feature_value
                    """,
                    (entity_id, name, json.dumps(value), timestamp),
                )
            self.postgres.commit()
            cursor.close()

        def read_features(self, entity_id, feature_names):
            key = f"features:user:{entity_id}"
            cached = self.redis.hmget(key, feature_names)
            result = {}
            for name, value in zip(feature_names, cached):
                if value is not None:
                    try:
                        result[name] = json.loads(value)
                    except Exception:
                        try:
                            result[name] = float(value) if '.' in str(value) else int(value)
                        except Exception:
                            result[name] = value
            missing = set(feature_names) - set(result.keys())
            if missing:
                result.update(self._read_from_postgres(entity_id, list(missing)))
            return result

        def _read_from_postgres(self, entity_id, feature_names):
            cursor = self.postgres.cursor()
            cursor.execute(
                """
                SELECT DISTINCT ON (feature_name)
                    feature_name, feature_value
                FROM features
                WHERE entity_id = %s AND feature_name = ANY(%s)
                ORDER BY feature_name, timestamp DESC
                """,
                (entity_id, feature_names),
            )
            result = {}
            for row in cursor.fetchall():
                try:
                    result[row[0]] = json.loads(row[1])
                except Exception:
                    result[row[0]] = row[1]
            cursor.close()
            return result

except ImportError:
    pass  # pandas / numpy / redis not available in all environments
