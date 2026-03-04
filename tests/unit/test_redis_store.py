"""
Unit tests for RedisFeatureStore (mocked Redis client).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from src.storage.redis_store import RedisFeatureStore, _entity_key


@pytest.fixture
def mock_redis_client():
    return MagicMock()


@pytest.fixture
def store(mock_redis_client):
    with patch("src.storage.redis_store.redis.Redis", return_value=mock_redis_client):
        with patch("src.storage.redis_store.redis.ConnectionPool"):
            s = RedisFeatureStore(
                host="localhost",
                port=6379,
                cluster_enabled=False,
                default_ttl=3600,
            )
            s._client = mock_redis_client
            return s


class TestRedisFeatureStore:
    def test_write_features_uses_hset(self, store, mock_redis_client):
        pipe = MagicMock()
        mock_redis_client.pipeline.return_value = pipe

        store.write_features(
            entity_id="user_123",
            feature_group="user_features",
            features={"count": 5, "amount": 100.0},
            entity_type="user",
            ttl=3600,
        )

        pipe.hset.assert_called_once()
        pipe.expire.assert_called_once()
        pipe.execute.assert_called_once()

    def test_key_format(self):
        key = _entity_key("user", "u123", "user_features")
        assert key == "features:user:u123:user_features"

    def test_read_features_batched(self, store, mock_redis_client):
        pipe = MagicMock()
        mock_redis_client.pipeline.return_value = pipe
        # Simulate hmget returning values for two entities
        pipe.execute.return_value = [
            ['"5"', '"100.0"'],   # entity_1
            [None, None],          # entity_2 – cache miss
        ]

        result = store.read_features(
            entity_ids=["entity_1", "entity_2"],
            feature_group="user_features",
            feature_names=["count", "amount"],
        )

        assert "entity_1" in result
        assert result["entity_1"]["count"] == 5
        assert result["entity_1"]["amount"] == 100.0
        # entity_2 has all-None values → excluded
        assert "entity_2" not in result

    def test_read_single(self, store, mock_redis_client):
        mock_redis_client.hget.return_value = '"42"'
        val = store.read_single(
            entity_id="u1",
            feature_group="user_features",
            feature_name="count",
        )
        assert val == 42

    def test_read_single_missing(self, store, mock_redis_client):
        mock_redis_client.hget.return_value = None
        assert store.read_single("u1", "user_features", "nonexistent") is None

    def test_delete_features(self, store, mock_redis_client):
        mock_redis_client.delete.return_value = 1
        result = store.delete_features("u1", "user_features")
        assert result is True

    def test_get_feature_freshness_with_ttl(self, store, mock_redis_client):
        mock_redis_client.ttl.return_value = 1800
        ttl = store.get_feature_freshness("u1", "user_features")
        assert ttl == 1800

    def test_get_feature_freshness_missing_key(self, store, mock_redis_client):
        mock_redis_client.ttl.return_value = -2  # key does not exist
        ttl = store.get_feature_freshness("u1", "user_features")
        assert ttl is None

    def test_ping_success(self, store, mock_redis_client):
        mock_redis_client.ping.return_value = True
        assert store.ping() is True

    def test_ping_failure(self, store, mock_redis_client):
        import redis.exceptions
        mock_redis_client.ping.side_effect = redis.exceptions.ConnectionError()
        assert store.ping() is False

    def test_write_features_batch(self, store, mock_redis_client):
        pipe = MagicMock()
        mock_redis_client.pipeline.return_value = pipe
        records = [
            {"entity_id": "u1", "feature_group": "ug", "features": {"a": 1}},
            {"entity_id": "u2", "feature_group": "ug", "features": {"a": 2}},
        ]
        store.write_features_batch(records, ttl=60)
        assert pipe.hset.call_count == 2
        assert pipe.expire.call_count == 2
        pipe.execute.assert_called_once()
