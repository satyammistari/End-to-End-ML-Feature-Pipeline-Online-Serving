# ML Feature Pipeline

Production-grade ML feature engineering pipeline that computes real-time and batch features from raw events, stores them in a feature store, and serves them at **<10ms P99 latency** for model inference.

Implements the feature store pattern used at Uber (Michelangelo), LinkedIn (Feathr), and major ML platforms.

---

## Architecture
![HCgEQLxa8AAXUSI](https://github.com/user-attachments/assets/b0861f7c-7b01-4325-a977-1fc0fb99a525)
![HCklyUUawAoCWHb](https://github.com/user-attachments/assets/4a7cb132-a0c1-4e0b-94f6-f45a889d8914)
![HCkfwLha4AA7IaY](https://github.com/user-attachments/assets/afed5996-4d5a-4c81-8ac2-90b72599696b)
![HCkfwLha4AA7IaY](https://github.com/user-attachments/assets/5f886c10-7eaf-4767-8bc6-265954a13d29)


```
Raw Events (Kafka)
      │
      ▼
┌─────────────────────┐       ┌─────────────────────┐
│  Kafka Consumer     │──────▶│  Feature Validator  │
│  (ingestion/)       │       │  (core/validators)  │
└─────────────────────┘       └──────────┬──────────┘
                                          │
              ┌───────────────────────────┼──────────────────────────┐
              ▼                           ▼                          ▼
   ┌───────────────────┐      ┌───────────────────┐     ┌───────────────────┐
   │   Redis Online    │      │  PostgreSQL        │     │  Feature Registry │
   │   Store (<1ms)    │      │  Offline Store     │     │  (core/registry)  │
   └────────┬──────────┘      │  (TimescaleDB)    │     └───────────────────┘
            │                 └────────┬──────────┘
            │                          │
            └──────────┬───────────────┘
                       ▼
            ┌───────────────────┐      ┌───────────────────┐
            │   Feature Store   │◀─────│  Batch Pipeline   │
            │   (core/)         │      │  (Polars/Spark)   │
            └────────┬──────────┘      └───────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   ┌─────────────┐      ┌─────────────┐
   │  REST API   │      │  gRPC API   │
   │  (FastAPI)  │      │  (50051)    │
   └─────────────┘      └─────────────┘
```

---

## Quick Start

### Prerequisites
- Docker + Docker Compose
- Python 3.10+

### 1. Start Infrastructure

```bash
cd docker
docker compose up -d
```

This starts: Kafka, Zookeeper, Redis, PostgreSQL (TimescaleDB), Prometheus, Grafana.

### 2. Install Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set Up Kafka Topics

```bash
bash scripts/setup_kafka.sh
```

### 4. Start the API Server

```bash
uvicorn src.api.rest_api:app --reload --port 8000
```

### 5. Generate Test Events

```bash
python scripts/generate_test_data.py --events 10000 --topic transaction-events
```

---

## Project Structure

```
ml-feature-pipeline/
├── src/
│   ├── api/               # REST (FastAPI) + gRPC serving
│   │   ├── rest_api.py
│   │   ├── grpc_server.py
│   │   └── proto/
│   ├── core/              # Domain logic
│   │   ├── feature_store.py   # Main orchestrator
│   │   ├── registry.py        # Feature catalog
│   │   ├── schemas.py         # Pydantic models
│   │   ├── validators.py      # Schema validation
│   │   └── ab_testing.py      # A/B experiments
│   ├── ingestion/         # Kafka → Feature pipeline
│   │   ├── kafka_consumer.py
│   │   └── transformers.py
│   ├── features/          # Feature computation
│   │   ├── realtime_features.py
│   │   ├── batch_features.py
│   │   └── definitions/       # Feature schemas
│   ├── storage/           # Store implementations
│   │   ├── redis_store.py     # Online (<1ms reads)
│   │   ├── postgres_store.py  # Offline (TimescaleDB)
│   │   └── point_in_time.py   # PIT joins (no leakage)
│   ├── backfill/          # Historical recomputation
│   └── utils/             # Metrics + logging
├── tests/
│   ├── unit/              # Fast, mocked tests
│   ├── integration/       # Full pipeline (mocked stores)
│   └── performance/       # Latency + throughput
├── docker/                # Dockerfiles + docker-compose
├── k8s/                   # Kubernetes manifests
├── config/                # YAML configuration
└── scripts/               # Setup + data generation
```

---

## API Reference

### REST API (`http://localhost:8000`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/features/online` | Serve online features |
| `POST` | `/v1/features/batch` | Submit batch job |
| `GET` | `/v1/features/registry` | List feature groups |
| `POST` | `/v1/features/registry` | Register feature group |
| `GET` | `/v1/features/registry/{name}` | Get feature group |
| `GET` | `/v1/health` | Health check |
| `GET` | `/v1/metrics` | Prometheus metrics |
| `POST` | `/v1/experiments` | Create A/B experiment |
| `GET` | `/v1/experiments/{name}/variant` | Get variant for entity |

Interactive docs: `http://localhost:8000/docs`

### gRPC API (`:50051`)

```protobuf
service FeatureService {
  rpc GetFeatures(GetFeaturesRequest) returns (GetFeaturesResponse);
  rpc GetFeaturesBatch(GetFeaturesBatchRequest) returns (stream GetFeaturesResponse);
}
```

Generate stubs:
```bash
python -m grpc_tools.protoc \
  -I src/api/proto \
  --python_out=src/api/proto \
  --grpc_python_out=src/api/proto \
  src/api/proto/feature_service.proto
```

---

## Feature Definitions

All features are defined in `config/feature_definitions.yaml` and registered via the Python API or YAML loader.

### User Features
| Feature | Type | Computation | TTL |
|---------|------|-------------|-----|
| `transaction_count_24h` | int | realtime | 24h |
| `avg_transaction_amount_7d` | float | batch | 24h |
| `std_transaction_amount_7d` | float | batch | 24h |
| `transaction_velocity_1h` | float | realtime | 1h |
| `unique_merchants_30d` | int | batch | 24h |
| `user_lifetime_value` | float | batch | 7d |
| `account_age_days` | int | batch | 24h |
| `kyc_verified` | bool | realtime | — |
| `typical_active_hours` | list | batch | 7d |

### Transaction Features
| Feature | Type | Computation |
|---------|------|-------------|
| `transaction_amount` | float | realtime |
| `amount_zscore` | float | on-demand |
| `transaction_currency` | string | realtime |
| `merchant_category` | string | realtime |
| `is_online` | bool | realtime |
| `is_new_merchant` | bool | on-demand |
| `merchant_category_frequency` | float | batch |
| `is_high_risk_time` | bool | on-demand |

---

## Running Tests

```bash
# Unit tests
pytest tests/unit/ -v --cov=src

# Integration tests
pytest tests/integration/ -v

# Performance / latency tests
pytest tests/performance/ -v --benchmark-sort=mean

# Full suite with coverage
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace feature-pipeline

# Apply configs, secrets, deployments, services
kubectl apply -f k8s/configmaps/
kubectl apply -f k8s/deployments/
kubectl apply -f k8s/services/

# Check rollout
kubectl rollout status deployment/feature-api -n feature-pipeline

# Scale API manually
kubectl scale deployment feature-api --replicas=5 -n feature-pipeline
```

HPA (Horizontal Pod Autoscaler) is pre-configured: scales API from 3→20 pods at 60% CPU.

---

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| P99 read latency | < 10ms | Redis online store |
| P50 read latency | < 1ms | In-memory path |
| Write throughput | > 10,000 ops/s | Kafka + Redis |
| Cache hit rate | > 95% | Hot features |
| API QPS (per pod) | > 5,000 | async FastAPI |
| Batch job duration | < 1h | Daily aggregations |

---

## Monitoring

- **Prometheus**: `http://localhost:9090`
- **Grafana**: `http://localhost:3000` (admin / admin)

Key metrics exposed at `/v1/metrics`:
- `feature_pipeline_feature_read_latency_seconds` (histogram)
- `feature_pipeline_feature_write_latency_seconds` (histogram)
- `feature_pipeline_feature_cache_hit_ratio` (gauge)
- `feature_pipeline_feature_validation_errors_total` (counter)
- `feature_pipeline_api_requests_total` (counter)
- `feature_pipeline_events_processed_total` (counter)

---

## Configuration

Edit `config/config.yaml` or use environment variables:

| Env var | Default | Description |
|---------|---------|-------------|
| `REDIS_HOST` | localhost | Redis hostname |
| `REDIS_PORT` | 6379 | Redis port |
| `POSTGRES_HOST` | localhost | Postgres hostname |
| `POSTGRES_PASSWORD` | — | Postgres password |
| `KAFKA_BOOTSTRAP_SERVERS` | localhost:9092 | Kafka brokers |
| `LOG_LEVEL` | INFO | Logging level |
| `ENV` | local | Environment name |

---

## Key Design Decisions

1. **Point-in-Time Correctness**: `storage/point_in_time.py` ensures training features never include future data. Critical for model/serving consistency.

2. **Exactly-Once Processing**: Kafka consumer commits offsets only after successful batch processing + store writes. Failed events go to a dead-letter queue (DLQ).

3. **Schema Versioning**: Non-breaking changes (additive) are allowed automatically. Breaking changes (removals, type changes) require a new major version.

4. **A/B Testing via Consistent Hashing**: User assignment is deterministic (same user always in same bucket) across restarts and replicas, without shared state.

5. **Graceful Degradation**: If Redis is unavailable, serving falls back to PostgreSQL. Validation errors are logged + counted but don't crash the pipeline.

---

## License

MIT
