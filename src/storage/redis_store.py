"""
Redis-based online feature store.
Provides low-latency (<10ms P99) feature reads and writes with TTL-based expiration.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import redis
import redis.exceptions
from redis.cluster import RedisCluster

logger = logging.getLogger(__name__)

_KEY_PREFIX = "features"


def _entity_key(entity_type: str, entity_id: str, feature_group: str) -> str:
    """Canonical Redis key: features:{entity_type}:{entity_id}:{feature_group}"""
    return f"{_KEY_PREFIX}:{entity_type}:{entity_id}:{feature_group}"


class RedisFeatureStore:
    """
    Online feature store backed by Redis.

    Supports both standalone Redis and Redis Cluster.
    Uses Redis Hashes — one hash per (entity, feature_group) — for O(1)
    field-level reads and minimal memory overhead.

    Parameters
    ----------
    host : str
    port : int
    db : int
    password : optional str
    cluster_enabled : bool
        When True, connects via RedisCluster (expects seed nodes at host:port).
    max_connections : int
        Connection pool size.
    default_ttl : int
        Default TTL in seconds (3600 = 1 hour).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        cluster_enabled: bool = False,
        max_connections: int = 50,
        default_ttl: int = 3600,
        socket_timeout: float = 0.1,
    ) -> None:
        self._default_ttl = default_ttl
        self._cluster_enabled = cluster_enabled

        if cluster_enabled:
            self._client: Any = RedisCluster(
                host=host,
                port=port,
                password=password,
                decode_responses=True,
                socket_timeout=socket_timeout,
            )
        else:
            pool = redis.ConnectionPool(
                host=host,
                port=port,
                db=db,
                password=password,
                max_connections=max_connections,
                decode_responses=True,
                socket_timeout=socket_timeout,
            )
            self._client = redis.Redis(connection_pool=pool)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_features(
        self,
        entity_id: str,
        feature_group: str,
        features: Dict[str, Any],
        entity_type: str = "entity",
        ttl: Optional[int] = None,
    ) -> None:
        """
        Write feature values for an entity.

        Serialises values to JSON strings inside a Redis Hash.
        Sets a TTL on the entire hash key.
        """
        ttl = ttl if ttl is not None else self._default_ttl
        key = _entity_key(entity_type, entity_id, feature_group)

        # Serialize each value
        serialised = {k: json.dumps(v) for k, v in features.items()}
        pipe = self._client.pipeline(transaction=False)
        pipe.hset(key, mapping=serialised)
        if ttl > 0:
            pipe.expire(key, ttl)
        pipe.execute()

    def write_features_batch(
        self,
        records: List[Dict[str, Any]],
        entity_type: str = "entity",
        ttl: Optional[int] = None,
    ) -> None:
        """
        Bulk write using a single Redis pipeline.

        Each record must have: entity_id, feature_group, features.
        """
        ttl = ttl if ttl is not None else self._default_ttl
        pipe = self._client.pipeline(transaction=False)
        for r in records:
            key = _entity_key(entity_type, r["entity_id"], r["feature_group"])
            serialised = {k: json.dumps(v) for k, v in r["features"].items()}
            pipe.hset(key, mapping=serialised)
            if ttl > 0:
                pipe.expire(key, ttl)
        pipe.execute()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_features(
        self,
        entity_ids: List[str],
        feature_group: str,
        feature_names: Optional[List[str]] = None,
        entity_type: str = "entity",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Batched read of features for a list of entities.

        Returns a dict: {entity_id: {feature_name: value}}.
        Missing entities are omitted from the result.
        """
        pipe = self._client.pipeline(transaction=False)
        keys = [_entity_key(entity_type, eid, feature_group) for eid in entity_ids]

        if feature_names:
            for k in keys:
                pipe.hmget(k, feature_names)
        else:
            for k in keys:
                pipe.hgetall(k)

        raw_results = pipe.execute()

        output: Dict[str, Dict[str, Any]] = {}
        for entity_id, raw in zip(entity_ids, raw_results):
            if feature_names:
                # hmget returns a list in same order as feature_names
                if any(v is not None for v in raw):
                    output[entity_id] = {
                        name: json.loads(val) if val is not None else None
                        for name, val in zip(feature_names, raw)
                    }
            else:
                # hgetall returns dict
                if raw:
                    output[entity_id] = {k: json.loads(v) for k, v in raw.items()}

        return output

    def read_single(
        self,
        entity_id: str,
        feature_group: str,
        feature_name: str,
        entity_type: str = "entity",
    ) -> Optional[Any]:
        """Read a single feature value."""
        key = _entity_key(entity_type, entity_id, feature_group)
        raw = self._client.hget(key, feature_name)
        return json.loads(raw) if raw is not None else None

    # ------------------------------------------------------------------
    # Delete / TTL
    # ------------------------------------------------------------------

    def delete_features(
        self,
        entity_id: str,
        feature_group: str,
        entity_type: str = "entity",
    ) -> bool:
        """Delete all features for an entity in a feature group."""
        key = _entity_key(entity_type, entity_id, feature_group)
        return bool(self._client.delete(key))

    def get_feature_freshness(
        self,
        entity_id: str,
        feature_group: str,
        entity_type: str = "entity",
    ) -> Optional[int]:
        """
        Return remaining TTL in seconds for an entity's feature key.
        Returns None if the key does not exist, -1 if no TTL is set.
        """
        key = _entity_key(entity_type, entity_id, feature_group)
        ttl = self._client.ttl(key)
        if ttl == -2:  # key does not exist
            return None
        return ttl

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        try:
            return self._client.ping()
        except redis.exceptions.RedisError:
            return False

    def get(self, key: str) -> Optional[str]:
        """Raw GET for registry cache use."""
        return self._client.get(key)

    def setex(self, key: str, ttl: int, value: str) -> None:
        """Raw SETEX for registry cache use."""
        self._client.setex(key, ttl, value)

    def delete(self, key: str) -> None:
        """Raw DELETE for registry cache use."""
        self._client.delete(key)
