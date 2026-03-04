"""
Prometheus metrics collector for the ML Feature Pipeline.

Tracks:
- Feature read / write latency (histograms)
- Cache hit ratio (gauge)
- Feature staleness (gauge)
- Validation errors (counter)
- API request counts (counter with labels)
- Active features count (gauge)
"""

from __future__ import annotations

import time
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    push_to_gateway,
)

# ---------------------------------------------------------------------------
# Latency buckets optimised for <10ms P99 target
# ---------------------------------------------------------------------------

_LATENCY_BUCKETS = (
    0.0005,  # 0.5ms
    0.001,   # 1ms
    0.002,   # 2ms
    0.005,   # 5ms
    0.010,   # 10ms  ← P99 target
    0.025,
    0.050,
    0.100,
    0.250,
    0.500,
    1.000,
)


class MetricsCollector:
    """
    Wraps all Prometheus metrics used by the feature pipeline.

    Parameters
    ----------
    registry : CollectorRegistry
        Use the default registry (None) for production.
        Pass a fresh CollectorRegistry() for tests to avoid conflicts.
    namespace : str
        Metric name prefix.
    """

    def __init__(
        self,
        registry: Any = None,
        namespace: str = "feature_pipeline",
    ) -> None:
        kwargs: dict = {"namespace": namespace}
        if registry is not None:
            kwargs["registry"] = registry

        # ------------------------------------------------------------------
        # Latency histograms
        # ------------------------------------------------------------------
        self.feature_read_latency = Histogram(
            "feature_read_latency_seconds",
            "Time to read features from the online store",
            buckets=_LATENCY_BUCKETS,
            **kwargs,
        )

        self.feature_write_latency = Histogram(
            "feature_write_latency_seconds",
            "Time to write features to the store",
            buckets=_LATENCY_BUCKETS,
            **kwargs,
        )

        # ------------------------------------------------------------------
        # Cache metrics
        # ------------------------------------------------------------------
        self.feature_cache_hit_ratio = Gauge(
            "feature_cache_hit_ratio",
            "Fraction of online feature reads served from Redis cache",
            **kwargs,
        )

        # ------------------------------------------------------------------
        # Freshness
        # ------------------------------------------------------------------
        self.feature_staleness_seconds = Gauge(
            "feature_staleness_seconds",
            "Age in seconds of the most recently written feature value",
            ["feature_group"],
            **kwargs,
        )

        # ------------------------------------------------------------------
        # Error counters
        # ------------------------------------------------------------------
        self.feature_validation_errors = Counter(
            "feature_validation_errors_total",
            "Total number of feature validation failures",
            ["feature_group"],
            **kwargs,
        )

        # ------------------------------------------------------------------
        # API counters
        # ------------------------------------------------------------------
        self.api_requests_total = Counter(
            "api_requests_total",
            "Total HTTP/gRPC requests received",
            ["method", "path", "status"],
            **kwargs,
        )

        # ------------------------------------------------------------------
        # Registry
        # ------------------------------------------------------------------
        self.active_features_count = Gauge(
            "active_features_count",
            "Total number of registered, non-deprecated features",
            **kwargs,
        )

        # ------------------------------------------------------------------
        # Throughput summary
        # ------------------------------------------------------------------
        self.events_processed_total = Counter(
            "events_processed_total",
            "Total Kafka events processed",
            ["event_type", "status"],  # status: success | dlq
            **kwargs,
        )

        self.batch_job_duration_seconds = Histogram(
            "batch_job_duration_seconds",
            "Duration of batch feature computation jobs",
            ["job_name"],
            buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def record_read(self, latency: float) -> None:
        """Record a single online read latency sample."""
        self.feature_read_latency.observe(latency)

    def record_write(self, latency: float) -> None:
        """Record a single write latency sample."""
        self.feature_write_latency.observe(latency)

    def update_cache_hit_ratio(self, hits: int, total: int) -> None:
        if total > 0:
            self.feature_cache_hit_ratio.set(hits / total)

    def record_validation_error(self, feature_group: str = "unknown") -> None:
        self.feature_validation_errors.labels(feature_group=feature_group).inc()

    def set_staleness(self, feature_group: str, age_seconds: float) -> None:
        self.feature_staleness_seconds.labels(feature_group=feature_group).set(age_seconds)


# ---------------------------------------------------------------------------
# Context manager for timing blocks
# ---------------------------------------------------------------------------

class timed:  # noqa: N801
    """
    Usage::

        with timed(metrics.feature_read_latency):
            result = store.read_features(...)
    """

    def __init__(self, histogram: Histogram) -> None:
        self._histogram = histogram
        self._start = 0.0

    def __enter__(self) -> "timed":
        self._start = time.monotonic()
        return self

    def __exit__(self, *args: Any) -> None:
        self._histogram.observe(time.monotonic() - self._start)
