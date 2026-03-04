"""Kafka event ingestion and feature transformation."""
from src.ingestion.kafka_consumer import KafkaFeatureConsumer
from src.ingestion.transformers import (
    EventTransformer,
    EventTransformerRegistry,
    build_default_registry,
    TransactionCreatedTransformer,
    UserLoginTransformer,
    UserProfileUpdatedTransformer,
)

__all__ = [
    "KafkaFeatureConsumer",
    "EventTransformer",
    "EventTransformerRegistry",
    "build_default_registry",
    "TransactionCreatedTransformer",
    "UserLoginTransformer",
    "UserProfileUpdatedTransformer",
]
