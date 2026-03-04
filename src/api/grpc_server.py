"""
gRPC feature serving server.

Uses the generated stub from feature_service.proto (included as a pre-built
Python module for portability – run `generate_proto.sh` to regenerate).

Falls back to the offline store on cache miss.
Targets: >10,000 QPS, P99 latency <10ms.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent import futures
from typing import Any, Iterator, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Proto stubs (generated from feature_service.proto)
# We define lightweight stand-ins here so the module loads even without
# grpc_tools. Replace with real generated code after running generate_proto.sh.
# ---------------------------------------------------------------------------

try:
    from .proto import feature_service_pb2, feature_service_pb2_grpc  # type: ignore
    _PROTO_AVAILABLE = True
except ImportError:
    _PROTO_AVAILABLE = False
    logger.warning(
        "Proto stubs not found. Run scripts/generate_proto.sh first. "
        "gRPC server will not be available."
    )

import grpc


# ---------------------------------------------------------------------------
# Servicer implementation
# ---------------------------------------------------------------------------

class FeatureServicer:
    """
    gRPC servicer implementing the FeatureService contract.

    Parameters
    ----------
    feature_store : FeatureStore
    metrics : MetricsCollector
    """

    def __init__(self, feature_store: Any, metrics: Any = None) -> None:
        self._store = feature_store
        self._metrics = metrics

    def GetFeatures(self, request: Any, context: Any) -> Any:
        """
        Single-request online feature lookup.

        Reads from Redis; falls back to PostgreSQL on cache miss.
        """
        if not _PROTO_AVAILABLE:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details("Proto stubs not generated")
            return None

        t0 = time.monotonic()
        try:
            from ..core.schemas import OnlineFeatureRequest

            req = OnlineFeatureRequest(
                entity_ids=list(request.entity_ids),
                feature_names=list(request.feature_names),
                feature_group=request.feature_group,
            )
            results = self._store.get_online_features(req)

            response = feature_service_pb2.GetFeaturesResponse()
            for r in results:
                ef = response.entity_features.add()
                ef.entity_id = r.entity_id
                for name, val in r.features.items():
                    ef.features[name] = str(val) if val is not None else ""
                ef.cache_hit = r.cache_hit

            elapsed = time.monotonic() - t0
            if self._metrics:
                self._metrics.feature_read_latency.observe(elapsed)

            return response

        except Exception as exc:
            logger.exception("GetFeatures error: %s", exc)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return feature_service_pb2.GetFeaturesResponse()

    def GetFeaturesBatch(self, request: Any, context: Any) -> Iterator[Any]:
        """
        Server-streaming batch lookup.  Splits large requests into chunks
        and streams responses back to the client.
        """
        if not _PROTO_AVAILABLE:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            return

        CHUNK_SIZE = 100
        entity_ids = list(request.entity_ids)
        feature_names = list(request.feature_names)

        for i in range(0, len(entity_ids), CHUNK_SIZE):
            chunk = entity_ids[i: i + CHUNK_SIZE]
            from ..core.schemas import OnlineFeatureRequest

            req = OnlineFeatureRequest(
                entity_ids=chunk,
                feature_names=feature_names,
                feature_group=request.feature_group,
            )
            results = self._store.get_online_features(req)
            response = feature_service_pb2.GetFeaturesResponse()
            for r in results:
                ef = response.entity_features.add()
                ef.entity_id = r.entity_id
                for name, val in r.features.items():
                    ef.features[name] = str(val) if val is not None else ""
                ef.cache_hit = r.cache_hit
            yield response


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def create_grpc_server(
    feature_store: Any,
    metrics: Any = None,
    port: int = 50051,
    max_workers: int = 10,
) -> Any:
    """
    Create and return a configured gRPC server.

    Call .start() and .wait_for_termination() on the returned server.
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 10000),
            ("grpc.keepalive_timeout_ms", 5000),
        ],
    )

    servicer = FeatureServicer(feature_store=feature_store, metrics=metrics)

    if _PROTO_AVAILABLE:
        feature_service_pb2_grpc.add_FeatureServiceServicer_to_server(servicer, server)
    else:
        logger.warning("Skipping servicer registration – proto stubs missing")

    server.add_insecure_port(f"[::]:{port}")
    logger.info("gRPC server configured on port %d", port)
    return server


if __name__ == "__main__":  # pragma: no cover
    from pathlib import Path
    import yaml
    from ..core.registry import FeatureRegistry
    from ..core.feature_store import FeatureStore
    from ..utils.metrics import MetricsCollector

    cfg_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    registry = FeatureRegistry()
    metrics = MetricsCollector()
    store = FeatureStore(registry=registry, metrics_collector=metrics)

    server = create_grpc_server(store, metrics, port=cfg["api"]["grpc_port"])
    server.start()
    logger.info("gRPC server started on port %d", cfg["api"]["grpc_port"])
    server.wait_for_termination()
