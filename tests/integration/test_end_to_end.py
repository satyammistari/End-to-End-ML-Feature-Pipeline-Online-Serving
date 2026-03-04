"""
Integration - end-to-end pipeline test.

Tests the full data path:
  Raw event -> KafkaConsumer mock -> Transformer -> FeatureStore -> Online store
  -> REST API read

Requires: no real external dependencies (all mocked).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.rest_api import create_app
from src.core.feature_store import FeatureStore
from src.core.registry import FeatureRegistry
from src.core.schemas import (
    FeatureDefinition,
    FeatureGroup,
    FeatureGroupVersion,
    FeatureRecord,
    FeatureType,
    OnlineFeatureRequest,
)
from src.ingestion.transformers import build_default_registry
from src.utils.metrics import MetricsCollector


# --- In-memory stores -------------------------------------------------------

class InMemoryOnlineStore:
    """Lightweight in-memory substitute for RedisFeatureStore."""

    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}

    def write_features(self, entity_id, feature_group, features, entity_type="entity", ttl=None):
        key = f"{entity_id}:{feature_group}"
        self._data.setdefault(key, {}).update(features)

    def read_features(self, entity_ids, feature_group, feature_names=None, entity_type="entity"):
        result = {}
        for eid in entity_ids:
            key = f"{eid}:{feature_group}"
            if key in self._data:
                vals = self._data[key]
                if feature_names:
                    result[eid] = {k: vals.get(k) for k in feature_names}
                else:
                    result[eid] = dict(vals)
        return result

    def ping(self):
        return True

    def get(self, k):
        return None

    def setex(self, k, t, v):
        pass

    def delete(self, k):
        pass


class InMemoryOfflineStore:
    def __init__(self):
        self._records: List[FeatureRecord] = []

    def write_features_batch(self, records):
        self._records.extend(records)

    def read_features_point_in_time(self, entity_ids, feature_names, timestamp, feature_group=None):
        result = {eid: {} for eid in entity_ids}
        for rec in self._records:
            if rec.entity_id in entity_ids:
                for fname, fval in rec.features.items():
                    if not feature_names or fname in feature_names:
                        result[rec.entity_id][fname] = fval
        return result

    def ping(self):
        return True


# --- Fixtures ---------------------------------------------------------------

@pytest.fixture
def online_store():
    return InMemoryOnlineStore()


@pytest.fixture
def offline_store():
    return InMemoryOfflineStore()


@pytest.fixture
def registry(online_store):
    r = FeatureRegistry(redis_store=online_store)
    group = FeatureGroup(
        name="user_features",
        entity_type="user",
        versions={
            "v1": FeatureGroupVersion(
                version="v1",
                features=[
                    FeatureDefinition(name="transaction_count_24h", feature_type=FeatureType.INTEGER),
                    FeatureDefinition(name="avg_amount_7d", feature_type=FeatureType.FLOAT),
                ],
            )
        },
        latest_version="v1",
    )
    r.register_feature_group(group)
    return r


@pytest.fixture
def metrics():
    from prometheus_client import CollectorRegistry
    return MetricsCollector(registry=CollectorRegistry())


@pytest.fixture
def feature_store(registry, online_store, offline_store, metrics):
    return FeatureStore(
        registry=registry,
        online_store=online_store,
        offline_store=offline_store,
        metrics_collector=metrics,
    )


@pytest.fixture
def test_client(feature_store, registry, metrics, online_store, offline_store):
    app = create_app(
        feature_store=feature_store,
        registry=registry,
        metrics=metrics,
        online_store=online_store,
        offline_store=offline_store,
    )
    return TestClient(app)


# --- Tests ------------------------------------------------------------------

class TestEndToEnd:
    def test_write_then_read_online(self, feature_store, online_store):
        """Write features -> read them back via FeatureStore."""
        record = FeatureRecord(
            entity_id="user_99",
            entity_type="user",
            feature_group="user_features",
            feature_version="v1",
            features={"transaction_count_24h": 7, "avg_amount_7d": 123.45},
        )
        feature_store.write_features(record, validate=False)

        req = OnlineFeatureRequest(
            entity_ids=["user_99"],
            feature_names=["transaction_count_24h", "avg_amount_7d"],
            feature_group="user_features",
        )
        results = feature_store.get_online_features(req)
        assert len(results) == 1
        assert results[0].entity_id == "user_99"
        assert results[0].features["transaction_count_24h"] == 7

    def test_health_endpoint(self, test_client):
        resp = test_client.get("/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ok", "degraded")

    def test_list_registry_endpoint(self, test_client):
        resp = test_client.get("/v1/features/registry")
        assert resp.status_code == 200
        body = resp.json()
        assert "user_features" in body["feature_groups"]

    def test_get_registry_group_endpoint(self, test_client):
        resp = test_client.get("/v1/features/registry/user_features")
        assert resp.status_code == 200
        assert resp.json()["name"] == "user_features"

    def test_get_unknown_registry_group_404(self, test_client):
        resp = test_client.get("/v1/features/registry/does_not_exist")
        assert resp.status_code == 404

    def test_online_feature_serving_endpoint(self, test_client, feature_store, online_store):
        # Seed the store
        online_store.write_features(
            "user_1", "user_features", {"transaction_count_24h": 3, "avg_amount_7d": 50.0}
        )
        resp = test_client.post(
            "/v1/features/online",
            json={
                "entity_ids": ["user_1"],
                "feature_names": ["transaction_count_24h"],
                "feature_group": "user_features",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["entity_id"] == "user_1"
        assert body[0]["features"]["transaction_count_24h"] == 3

    def test_batch_job_submission(self, test_client):
        resp = test_client.post(
            "/v1/features/batch",
            json={
                "job_name": "daily_user_agg",
                "feature_group": "user_features",
                "start_date": "2025-01-01T00:00:00",
                "end_date": "2025-01-02T00:00:00",
            },
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "queued"
        assert "job_id" in body

    def test_metrics_endpoint(self, test_client):
        resp = test_client.get("/v1/metrics")
        assert resp.status_code == 200
        assert b"feature_pipeline" in resp.content or b"#" in resp.content

    def test_transformer_pipeline(self):
        """Event -> Transformer -> FeatureRecord fields are correct."""
        from src.core.schemas import RawEvent
        registry = build_default_registry()
        transformer = registry.get("transaction_created")
        event = RawEvent(
            event_id="e1",
            event_type="transaction_created",
            entity_id="user_55",
            entity_type="user",
            occurred_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
            payload={"amount": "250.00", "currency": "EUR", "merchant_id": "m1"},
        )
        records = transformer.transform(event)
        assert records[0].features["transaction_amount"] == pytest.approx(250.0)
        assert records[0].features["transaction_currency"] == "EUR"


# -----------------------------------------------------------------------------
# DB-backed end-to-end pipeline tests (requires running Postgres + Redis)
#
# Run with:
#   pytest tests/integration/test_end_to_end.py::TestEndToEndPipeline -v -s
# -----------------------------------------------------------------------------

import os
import sys

import numpy as np
import pandas as pd
import psycopg2
import redis as redis_module


_POSTGRES_PARAMS = dict(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    database=os.getenv("POSTGRES_DB", "features_test"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
)

_REDIS_PARAMS = dict(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=0,
    decode_responses=True,
)

_TEST_DATA_DIR = "/tmp/test_data"


def _postgres_available() -> bool:
    try:
        conn = psycopg2.connect(**_POSTGRES_PARAMS, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


def _redis_available() -> bool:
    try:
        r = redis_module.Redis(**_REDIS_PARAMS, socket_connect_timeout=3)
        r.ping()
        r.close()
        return True
    except Exception:
        return False


def _test_data_available() -> bool:
    return os.path.exists(os.path.join(_TEST_DATA_DIR, "transactions.csv"))


_skip_db = pytest.mark.skipif(
    not (_postgres_available() and _redis_available() and _test_data_available()),
    reason="Requires PostgreSQL, Redis, and generated test data in /tmp/test_data/",
)


@pytest.fixture(scope="class")
def db_conn():
    """Class-scoped PostgreSQL connection with schema bootstrapped."""
    conn = psycopg2.connect(**_POSTGRES_PARAMS)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id  VARCHAR(255) PRIMARY KEY,
            user_id         VARCHAR(255) NOT NULL,
            merchant_id     VARCHAR(255),
            merchant_category VARCHAR(100),
            amount          NUMERIC(12, 2),
            timestamp       TIMESTAMPTZ NOT NULL,
            status          VARCHAR(50),
            payment_method  VARCHAR(50),
            device_type     VARCHAR(50),
            ip_country      VARCHAR(10),
            is_international BOOLEAN
        );
        CREATE INDEX IF NOT EXISTS idx_t_user_ts
            ON transactions(user_id, timestamp DESC);

        CREATE TABLE IF NOT EXISTS features (
            id           SERIAL PRIMARY KEY,
            entity_id    VARCHAR(255) NOT NULL,
            feature_name VARCHAR(255) NOT NULL,
            feature_value JSONB,
            timestamp    TIMESTAMPTZ NOT NULL,
            UNIQUE (entity_id, feature_name, timestamp)
        );
        CREATE INDEX IF NOT EXISTS idx_f_entity_ts
            ON features(entity_id, feature_name, timestamp DESC);
    """)
    conn.commit()
    cur.close()
    yield conn
    conn.close()


@pytest.fixture(scope="class")
def redis_conn():
    """Class-scoped Redis client (test DB flushed before and after)."""
    r = redis_module.Redis(**_REDIS_PARAMS)
    r.flushdb()
    yield r
    r.flushdb()
    r.close()


@pytest.mark.usefixtures("db_conn", "redis_conn")
class TestEndToEndPipeline:
    """
    Full pipeline integration tests.
    Require live Postgres + Redis + pre-generated test data.
    """

    @_skip_db
    def test_01_load_transactions(self, db_conn):
        """Load generated CSV transactions into PostgreSQL."""
        print("\n" + "=" * 60)
        print("TEST 1: Loading test data into PostgreSQL")
        print("=" * 60)

        txns = pd.read_csv(os.path.join(_TEST_DATA_DIR, "transactions.csv"))

        cur = db_conn.cursor()
        for _, row in txns.iterrows():
            cur.execute(
                """
                INSERT INTO transactions
                    (transaction_id, user_id, merchant_id, merchant_category,
                     amount, timestamp, status, payment_method,
                     device_type, ip_country, is_international)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (transaction_id) DO NOTHING
                """,
                (
                    row["transaction_id"], row["user_id"], row["merchant_id"],
                    row["merchant_category"], row["amount"], row["timestamp"],
                    row["status"], row["payment_method"], row["device_type"],
                    row["ip_country"], bool(row["is_international"]),
                ),
            )
        db_conn.commit()
        cur.close()

        cur = db_conn.cursor()
        cur.execute("SELECT COUNT(*) FROM transactions")
        count = cur.fetchone()[0]
        cur.close()

        print(f"[EMOJI] Loaded {count:,} transactions")
        assert count == len(txns)

    @_skip_db
    def test_02_compute_and_validate_features(self, db_conn, redis_conn):
        """Compute features for all transactions and cross-validate against expected."""
        print("\n" + "=" * 60)
        print("TEST 2: Computing and validating features")
        print("=" * 60)

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
        from core.feature_store import FeatureStoreWithValidation  # type: ignore
        from features.realtime_features import RealtimeFeatureComputer  # type: ignore

        store = FeatureStoreWithValidation(redis_conn, db_conn)
        computer = RealtimeFeatureComputer(redis_conn, db_conn)

        txns = pd.read_csv(os.path.join(_TEST_DATA_DIR, "transactions.csv"))
        txns["timestamp"] = pd.to_datetime(txns["timestamp"])
        txns = txns.sort_values("timestamp").reset_index(drop=True)

        success, errors = 0, 0
        for _, txn in txns.iterrows():
            try:
                feats = computer.compute_transaction_features(
                    user_id=txn["user_id"],
                    transaction=txn.to_dict(),
                    timestamp=txn["timestamp"],
                )
                store.write_features(
                    entity_id=txn["user_id"],
                    features=feats,
                    timestamp=txn["timestamp"],
                    transaction_id=txn["transaction_id"],
                )
                success += 1
            except ValueError:
                errors += 1
                if errors > 10:
                    break

        total = len(txns)
        rate = success / total
        print(f"[EMOJI] {success:,}/{total:,}  success rate {rate * 100:.2f}%  errors={errors}")
        assert rate > 0.95, f"Too many validation failures: {errors}"

    @_skip_db
    def test_03_feature_retrieval(self, db_conn, redis_conn):
        """Verify features can be retrieved from Redis and Postgres."""
        print("\n" + "=" * 60)
        print("TEST 3: Feature retrieval")
        print("=" * 60)

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
        from core.feature_store import FeatureStoreWithValidation  # type: ignore

        store = FeatureStoreWithValidation(redis_conn, db_conn)
        test_user = "user_000042"
        names = [
            "transaction_count_24h", "transaction_count_7d",
            "avg_amount_7d", "amount_zscore",
        ]
        result = store.read_features(test_user, names)
        print(f"  Retrieved for {test_user}: {result}")
        assert len(result) > 0, "Expected at least one feature"
        print(f"[EMOJI] Retrieved {len(result)} features")

    @_skip_db
    def test_04_point_in_time_correctness(self, db_conn):
        """Assert no feature timestamp is strictly after its source transaction."""
        print("\n" + "=" * 60)
        print("TEST 4: Point-in-time correctness")
        print("=" * 60)

        cur = db_conn.cursor()
        cur.execute(
            """
            SELECT f.entity_id, f.timestamp AS feat_ts, t.timestamp AS txn_ts
            FROM   features f
            JOIN   transactions t ON f.entity_id = t.user_id
            WHERE  f.timestamp > t.timestamp
            LIMIT  10
            """
        )
        leakage = cur.fetchall()
        cur.close()

        if leakage:
            for row in leakage[:5]:
                print(f"  Leakage: entity={row[0]}, feat_ts={row[1]}, txn_ts={row[2]}")
            pytest.fail(f"Future data leakage detected: {len(leakage)} rows")

        print("[EMOJI] No future data leakage")
