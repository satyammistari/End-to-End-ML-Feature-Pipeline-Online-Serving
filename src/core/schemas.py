"""
Core Pydantic schemas for the ML Feature Pipeline.
Defines all data models used across the system.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FeatureType(str, enum.Enum):
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    LIST = "list"
    MAP = "map"
    TIMESTAMP = "timestamp"


class FeatureComputationType(str, enum.Enum):
    REALTIME = "realtime"
    BATCH = "batch"
    ON_DEMAND = "on_demand"


class ChangeType(str, enum.Enum):
    BREAKING = "breaking"
    NON_BREAKING = "non_breaking"
    PATCH = "patch"


class ExperimentStatus(str, enum.Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


# ---------------------------------------------------------------------------
# Feature definition schemas
# ---------------------------------------------------------------------------

class ValidationRule(BaseModel):
    """A single validation constraint on a feature value."""

    min_value: Optional[float] = None
    max_value: Optional[float] = None
    regex_pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    not_null: bool = True
    custom_validator: Optional[str] = None  # dotted import path

    model_config = {"extra": "forbid"}


class FeatureDefinition(BaseModel):
    """Schema for a single feature."""

    name: str = Field(..., min_length=1, max_length=128)
    feature_type: FeatureType
    computation_type: FeatureComputationType = FeatureComputationType.BATCH
    description: str = ""
    owner: str = ""
    tags: List[str] = Field(default_factory=list)
    ttl_seconds: Optional[int] = Field(None, ge=0)
    validation_rules: Optional[ValidationRule] = None
    depends_on: List[str] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def name_snake_case(cls, v: str) -> str:
        if not v.replace("_", "").isalnum():
            raise ValueError("Feature name must be alphanumeric with underscores only")
        return v.lower()

    model_config = {"extra": "forbid"}


class FeatureGroupVersion(BaseModel):
    """Versioned snapshot of a FeatureGroup."""

    version: str = Field(..., pattern=r"^v\d+$")
    features: List[FeatureDefinition]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    change_type: ChangeType = ChangeType.NON_BREAKING
    changelog: str = ""
    deprecated: bool = False
    deprecation_message: Optional[str] = None


class FeatureGroup(BaseModel):
    """Collection of related features sharing an entity key."""

    name: str = Field(..., min_length=1, max_length=128)
    entity_type: str = Field(..., description="e.g. 'user', 'transaction'")
    description: str = ""
    owner: str = ""
    tags: List[str] = Field(default_factory=list)
    versions: Dict[str, FeatureGroupVersion] = Field(default_factory=dict)
    latest_version: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def get_latest(self) -> Optional[FeatureGroupVersion]:
        if self.latest_version:
            return self.versions.get(self.latest_version)
        return None


# ---------------------------------------------------------------------------
# Event schemas
# ---------------------------------------------------------------------------

class RawEvent(BaseModel):
    """Raw Kafka event before transformation."""

    event_id: str
    event_type: str
    entity_id: str
    entity_type: str
    occurred_at: datetime
    payload: Dict[str, Any]
    source: str = ""

    model_config = {"extra": "allow"}


class FeatureRecord(BaseModel):
    """Computed feature values for an entity at a point in time."""

    entity_id: str
    entity_type: str
    feature_group: str
    feature_version: str
    features: Dict[str, Any]
    computed_at: datetime = Field(default_factory=datetime.utcnow)
    event_time: Optional[datetime] = None  # original event timestamp
    ttl_seconds: Optional[int] = None


# ---------------------------------------------------------------------------
# API request / response schemas
# ---------------------------------------------------------------------------

class OnlineFeatureRequest(BaseModel):
    entity_ids: List[str] = Field(..., min_length=1, max_length=1000)
    feature_names: List[str] = Field(..., min_length=1)
    feature_group: str
    feature_version: Optional[str] = None  # defaults to latest
    include_metadata: bool = False


class OnlineFeatureResponse(BaseModel):
    entity_id: str
    features: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    served_at: datetime = Field(default_factory=datetime.utcnow)
    cache_hit: bool = False
    version: Optional[str] = None


class BatchFeatureJobRequest(BaseModel):
    job_name: str
    feature_group: str
    start_date: datetime
    end_date: datetime
    entity_ids: Optional[List[str]] = None  # None = all entities
    feature_names: Optional[List[str]] = None  # None = all features
    backfill: bool = False


class BatchFeatureJobResponse(BaseModel):
    job_id: str
    status: str
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    estimated_completion: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Experiment / A/B schemas
# ---------------------------------------------------------------------------

class ExperimentVariant(BaseModel):
    name: str
    description: str = ""
    traffic_fraction: float = Field(..., ge=0.0, le=1.0)
    feature_overrides: Dict[str, Any] = Field(default_factory=dict)


class Experiment(BaseModel):
    name: str
    description: str = ""
    variants: List[ExperimentVariant]
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    owner: str = ""
    tags: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def traffic_sums_to_one(self) -> "Experiment":
        total = sum(v.traffic_fraction for v in self.variants)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Variant traffic fractions must sum to 1.0, got {total:.4f}")
        return self


# ---------------------------------------------------------------------------
# Health / monitoring schemas
# ---------------------------------------------------------------------------

class HealthStatus(BaseModel):
    status: str  # "ok" | "degraded" | "down"
    redis_connected: bool
    postgres_connected: bool
    kafka_connected: bool
    uptime_seconds: float
    version: str = "1.0.0"


class FeatureFreshness(BaseModel):
    entity_id: str
    feature_group: str
    last_written_at: Optional[datetime]
    staleness_seconds: Optional[float]
    is_fresh: bool
