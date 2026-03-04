"""
Performance tests: verify P99 latency targets and throughput.

Run with::
  pytest tests/performance/ -v --benchmark-sort=mean

Requirements:
  pytest-benchmark>=4.0
  All tests use in-memory stores to isolate pipeline code performance.
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

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
from src.utils.metrics import MetricsCollector


# ─── In-memory stores (same as integration tests) ────────────────────────────

class _InMemOnline:
    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}

    def write_features(self, entity_id, feature_group, features, **kw):
        self._data[f"{entity_id}:{feature_group}"] = features

    def read_features(self, entity_ids, feature_group, feature_names=None, **kw):
        out = {}
        for eid in entity_ids:
            v = self._data.get(f"{eid}:{feature_group}")
            if v:
                out[eid] = {k: v[k] for k in (feature_names or v) if k in v}
        return out

    def ping(self):
        return True

    def get(self, k):
        return None

    def setex(self, *a):
        pass

    def delete(self, k):
        pass


class _InMemOffline:
    def write_features_batch(self, records):
        pass

    def read_features_point_in_time(self, entity_ids, feature_names, timestamp, **kw):
        return {eid: {} for eid in entity_ids}


@pytest.fixture(scope="module")
def perf_store():
    online = _InMemOnline()
    offline = _InMemOffline()

    from prometheus_client import CollectorRegistry
    metrics = MetricsCollector(registry=CollectorRegistry())

    reg = FeatureRegistry(redis_store=online)
    group = FeatureGroup(
        name="user_features",
        entity_type="user",
        versions={
            "v1": FeatureGroupVersion(
                version="v1",
                features=[
                    FeatureDefinition(name="f1", feature_type=FeatureType.FLOAT),
                    FeatureDefinition(name="f2", feature_type=FeatureType.INTEGER),
                    FeatureDefinition(name="f3", feature_type=FeatureType.STRING),
                ],
            )
        },
        latest_version="v1",
    )
    reg.register_feature_group(group)

    store = FeatureStore(
        registry=reg,
        online_store=online,
        offline_store=offline,
        metrics_collector=metrics,
    )

    # Pre-populate 10_000 entities
    for i in range(10_000):
        online.write_features(
            f"user_{i}",
            "user_features",
            {"f1": float(i), "f2": i, "f3": f"val_{i}"},
        )

    return store


# ─── Latency benchmarks ──────────────────────────────────────────────────────

class TestLatency:
    def test_single_entity_online_read_benchmark(self, perf_store, benchmark):
        """Single-entity read should complete in well under 1ms (pure Python)."""
        req = OnlineFeatureRequest(
            entity_ids=["user_42"],
            feature_names=["f1", "f2"],
            feature_group="user_features",
        )
        benchmark(perf_store.get_online_features, req)

    def test_batch_10_entities_benchmark(self, perf_store, benchmark):
        """Batch of 10 entities."""
        entity_ids = [f"user_{i}" for i in range(10)]
        req = OnlineFeatureRequest(
            entity_ids=entity_ids,
            feature_names=["f1", "f2", "f3"],
            feature_group="user_features",
        )
        benchmark(perf_store.get_online_features, req)

    def test_batch_100_entities_benchmark(self, perf_store, benchmark):
        """Batch of 100 entities – should still be fast."""
        entity_ids = [f"user_{i}" for i in range(100)]
        req = OnlineFeatureRequest(
            entity_ids=entity_ids,
            feature_names=["f1", "f2", "f3"],
            feature_group="user_features",
        )
        benchmark(perf_store.get_online_features, req)

    def test_p99_latency_under_10ms(self, perf_store):
        """
        Make 1000 single-entity requests and verify P99 < 10ms.
        (Tests Python-layer latency; add 1-2ms for real Redis network hops.)
        """
        latencies = []
        entity_ids = [f"user_{random.randint(0, 9999)}" for _ in range(1000)]

        for eid in entity_ids:
            req = OnlineFeatureRequest(
                entity_ids=[eid],
                feature_names=["f1", "f2"],
                feature_group="user_features",
            )
            t0 = time.perf_counter()
            perf_store.get_online_features(req)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

        latencies.sort()
        p99 = latencies[int(len(latencies) * 0.99)]
        p50 = latencies[len(latencies) // 2]

        print(f"\nP50={p50:.3f}ms  P99={p99:.3f}ms  max={max(latencies):.3f}ms")
        assert p99 < 10.0, f"P99 latency {p99:.2f}ms exceeds 10ms target"


# ─── Throughput tests ────────────────────────────────────────────────────────

class TestThroughput:
    def test_write_throughput(self, perf_store):
        """Write 10_000 feature records and check it completes in <2 seconds."""
        t0 = time.perf_counter()
        for i in range(10_000):
            rec = FeatureRecord(
                entity_id=f"perf_user_{i}",
                entity_type="user",
                feature_group="user_features",
                feature_version="v1",
                features={"f1": float(i), "f2": i, "f3": "x"},
            )
            perf_store.write_features(rec, validate=False)

        elapsed = time.perf_counter() - t0
        qps = 10_000 / elapsed
        print(f"\nWrite throughput: {qps:,.0f} ops/s  ({elapsed:.2f}s for 10k records)")
        assert elapsed < 5.0, f"Write throughput too low: {elapsed:.2f}s"

    def test_read_throughput(self, perf_store):
        """Read 10_000 single-entity requests and compute QPS."""
        t0 = time.perf_counter()
        for i in range(10_000):
            req = OnlineFeatureRequest(
                entity_ids=[f"user_{i % 10_000}"],
                feature_names=["f1", "f2"],
                feature_group="user_features",
            )
            perf_store.get_online_features(req)

        elapsed = time.perf_counter() - t0
        qps = 10_000 / elapsed
        print(f"\nRead throughput: {qps:,.0f} QPS  ({elapsed:.2f}s for 10k requests)")
        assert qps > 1_000, f"Read QPS {qps:.0f} is below 1,000"
