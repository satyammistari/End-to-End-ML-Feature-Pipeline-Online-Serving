"""
FastAPI REST API for the feature pipeline.

Endpoints:
  POST /v1/features/online       – serve online features
  POST /v1/features/batch        – submit a batch job
  GET  /v1/features/registry     – list feature groups
  POST /v1/features/registry     – register a new feature group
  GET  /v1/features/registry/{name} – get feature group details
  GET  /v1/health                – health check
  GET  /v1/metrics               – Prometheus text format metrics
  POST /v1/experiments           – create experiment
  GET  /v1/experiments/{name}/variant – get variant for entity
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from ..core.feature_store import FeatureStore
from ..core.registry import FeatureRegistry, RegistryError
from ..core.schemas import (
    BatchFeatureJobRequest,
    BatchFeatureJobResponse,
    Experiment,
    FeatureGroup,
    HealthStatus,
    OnlineFeatureRequest,
    OnlineFeatureResponse,
)
from ..utils.metrics import MetricsCollector

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    feature_store: FeatureStore,
    registry: FeatureRegistry,
    metrics: MetricsCollector,
    online_store: Any = None,
    offline_store: Any = None,
    kafka_connected: bool = False,
    ab_manager: Any = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.start_time = time.time()
        yield

    app = FastAPI(
        title="ML Feature Pipeline API",
        version="1.0.0",
        description="Production-grade feature serving with <10ms P99 latency",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Inject dependencies via app state
    app.state.feature_store = feature_store
    app.state.registry = registry
    app.state.metrics = metrics
    app.state.online_store = online_store
    app.state.offline_store = offline_store
    app.state.kafka_connected = kafka_connected
    app.state.ab_manager = ab_manager

    # ------------------------------------------------------------------ #
    # Middleware – request tracing + latency metrics                       #
    # ------------------------------------------------------------------ #

    @app.middleware("http")
    async def record_request(request: Request, call_next: Any) -> Response:
        t0 = time.monotonic()
        response = await call_next(request)
        elapsed = time.monotonic() - t0
        metrics.api_requests_total.labels(
            method=request.method,
            path=request.url.path,
            status=str(response.status_code),
        ).inc()
        return response

    # ------------------------------------------------------------------ #
    # Health                                                               #
    # ------------------------------------------------------------------ #

    @app.get("/v1/health", response_model=HealthStatus, tags=["ops"])
    async def health_check(request: Request) -> HealthStatus:
        redis_ok = False
        pg_ok = False
        if request.app.state.online_store:
            redis_ok = request.app.state.online_store.ping()
        if request.app.state.offline_store:
            pg_ok = request.app.state.offline_store.ping()

        uptime = time.time() - request.app.state.start_time
        overall = "ok" if (redis_ok and pg_ok) else "degraded"
        return HealthStatus(
            status=overall,
            redis_connected=redis_ok,
            postgres_connected=pg_ok,
            kafka_connected=request.app.state.kafka_connected,
            uptime_seconds=uptime,
        )

    # ------------------------------------------------------------------ #
    # Prometheus metrics                                                   #
    # ------------------------------------------------------------------ #

    @app.get("/v1/metrics", tags=["ops"])
    async def prometheus_metrics() -> Response:
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
        )

    # ------------------------------------------------------------------ #
    # Feature registry                                                     #
    # ------------------------------------------------------------------ #

    @app.get("/v1/features/registry", tags=["registry"])
    async def list_feature_groups(request: Request) -> Dict[str, Any]:
        names = request.app.state.registry.list_feature_groups()
        return {"feature_groups": names, "count": len(names)}

    @app.post(
        "/v1/features/registry",
        status_code=status.HTTP_201_CREATED,
        tags=["registry"],
    )
    async def register_feature_group(
        group: FeatureGroup, request: Request
    ) -> Dict[str, Any]:
        try:
            created = request.app.state.registry.register_feature_group(group)
            return {"name": created.name, "status": "registered"}
        except RegistryError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.get("/v1/features/registry/{name}", tags=["registry"])
    async def get_feature_group(name: str, request: Request) -> FeatureGroup:
        group = request.app.state.registry.get_feature_group(name)
        if not group:
            raise HTTPException(status_code=404, detail=f"Feature group '{name}' not found")
        return group

    # ------------------------------------------------------------------ #
    # Online serving                                                       #
    # ------------------------------------------------------------------ #

    @app.post(
        "/v1/features/online",
        response_model=List[OnlineFeatureResponse],
        tags=["serving"],
    )
    async def get_online_features(
        req: OnlineFeatureRequest, request: Request
    ) -> List[OnlineFeatureResponse]:
        try:
            return request.app.state.feature_store.get_online_features(req)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------ #
    # Batch jobs                                                           #
    # ------------------------------------------------------------------ #

    @app.post(
        "/v1/features/batch",
        response_model=BatchFeatureJobResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["batch"],
    )
    async def submit_batch_job(
        req: BatchFeatureJobRequest, request: Request
    ) -> BatchFeatureJobResponse:
        job_id = str(uuid.uuid4())
        # In production this would enqueue to Airflow/Celery/Spark
        return BatchFeatureJobResponse(job_id=job_id, status="queued")

    # ------------------------------------------------------------------ #
    # A/B experiments                                                      #
    # ------------------------------------------------------------------ #

    @app.post("/v1/experiments", status_code=status.HTTP_201_CREATED, tags=["experiments"])
    async def create_experiment(
        experiment: Experiment, request: Request
    ) -> Dict[str, Any]:
        mgr = request.app.state.ab_manager
        if mgr is None:
            raise HTTPException(status_code=503, detail="A/B manager not configured")
        mgr.create_experiment(experiment)
        return {"name": experiment.name, "status": "created"}

    @app.get("/v1/experiments/{name}/variant", tags=["experiments"])
    async def get_variant(
        name: str, entity_id: str, request: Request
    ) -> Dict[str, str]:
        mgr = request.app.state.ab_manager
        if mgr is None:
            raise HTTPException(status_code=503, detail="A/B manager not configured")
        variant = mgr.get_variant(entity_id, name)
        return {"experiment": name, "entity_id": entity_id, "variant": variant}

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import yaml
    from pathlib import Path
    from ..core.registry import FeatureRegistry
    from ..core.feature_store import FeatureStore
    from ..utils.metrics import MetricsCollector

    cfg_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    registry = FeatureRegistry()
    metrics = MetricsCollector()
    store = FeatureStore(registry=registry, metrics_collector=metrics)

    app = create_app(
        feature_store=store,
        registry=registry,
        metrics=metrics,
    )
    uvicorn.run(app, host="0.0.0.0", port=cfg["api"]["rest_port"], workers=1)


# ---------------------------------------------------------------------------
# Default app instance for uvicorn import
# ---------------------------------------------------------------------------

# Create default app instance
registry = FeatureRegistry()
metrics = MetricsCollector()
store = FeatureStore(registry=registry, metrics_collector=metrics)
app = create_app(
    feature_store=store,
    registry=registry,
    metrics=metrics,
)
