"""
Event transformers: convert raw Kafka events into FeatureRecord(s).

Each transformer handles a specific event_type and produces one or more
FeatureRecord objects that can be written to the feature store.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..core.schemas import FeatureRecord, RawEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base transformer
# ---------------------------------------------------------------------------

class EventTransformer(ABC):
    """
    Abstract base class for event transformers.

    Each subclass handles exactly one event_type and is responsible for
    extracting feature values from the raw event payload.
    """

    @property
    @abstractmethod
    def event_type(self) -> str:
        """The event type this transformer handles (e.g. 'transaction_created')."""

    @abstractmethod
    def transform(self, event: RawEvent) -> List[FeatureRecord]:
        """
        Transform a raw event into feature records.

        Parameters
        ----------
        event : RawEvent

        Returns
        -------
        List[FeatureRecord] – may be empty if the event carries no useful features.
        """

    def _now(self) -> datetime:
        return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Transformer registry
# ---------------------------------------------------------------------------

class EventTransformerRegistry:
    """
    Registry mapping event_type strings to EventTransformer instances.

    Usage::

        registry = EventTransformerRegistry()
        registry.register(TransactionCreatedTransformer())
        transformer = registry.get("transaction_created")
    """

    def __init__(self) -> None:
        self._transformers: Dict[str, EventTransformer] = {}

    def register(self, transformer: EventTransformer) -> None:
        self._transformers[transformer.event_type] = transformer
        logger.debug("Registered transformer for event_type=%s", transformer.event_type)

    def get(self, event_type: str) -> Optional[EventTransformer]:
        return self._transformers.get(event_type)

    def list_types(self) -> List[str]:
        return list(self._transformers.keys())


# ---------------------------------------------------------------------------
# Concrete transformers
# ---------------------------------------------------------------------------

class TransactionCreatedTransformer(EventTransformer):
    """
    Handles 'transaction_created' events.

    Extracts raw transaction-level features:
    - transaction_amount
    - transaction_currency
    - merchant_id
    - merchant_category
    - is_online
    """

    event_type = "transaction_created"

    def transform(self, event: RawEvent) -> List[FeatureRecord]:
        p = event.payload
        features = {
            "transaction_amount": float(p.get("amount", 0.0)),
            "transaction_currency": p.get("currency", "USD"),
            "merchant_id": p.get("merchant_id", ""),
            "merchant_category": p.get("merchant_category", "unknown"),
            "is_online": bool(p.get("is_online", False)),
        }
        record = FeatureRecord(
            entity_id=event.entity_id,
            entity_type="user",
            feature_group="transaction_features",
            feature_version="v1",
            features=features,
            event_time=event.occurred_at,
        )
        return [record]


class UserLoginTransformer(EventTransformer):
    """
    Handles 'user_login' events.

    Extracts:
    - last_login_timestamp
    - login_device
    - login_ip_country
    """

    event_type = "user_login"

    def transform(self, event: RawEvent) -> List[FeatureRecord]:
        p = event.payload
        features = {
            "last_login_timestamp": event.occurred_at.isoformat(),
            "login_device": p.get("device", "unknown"),
            "login_ip_country": p.get("ip_country", "unknown"),
        }
        record = FeatureRecord(
            entity_id=event.entity_id,
            entity_type="user",
            feature_group="user_features",
            feature_version="v1",
            features=features,
            event_time=event.occurred_at,
        )
        return [record]


class UserProfileUpdatedTransformer(EventTransformer):
    """
    Handles 'user_profile_updated' events.

    Extracts account-level signals.
    """

    event_type = "user_profile_updated"

    def transform(self, event: RawEvent) -> List[FeatureRecord]:
        p = event.payload
        features = {
            "account_age_days": int(p.get("account_age_days", 0)),
            "kyc_verified": bool(p.get("kyc_verified", False)),
            "preferred_currency": p.get("preferred_currency", "USD"),
        }
        record = FeatureRecord(
            entity_id=event.entity_id,
            entity_type="user",
            feature_group="user_features",
            feature_version="v1",
            features=features,
            event_time=event.occurred_at,
        )
        return [record]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_default_registry() -> EventTransformerRegistry:
    """Create a registry pre-loaded with all default transformers."""
    registry = EventTransformerRegistry()
    registry.register(TransactionCreatedTransformer())
    registry.register(UserLoginTransformer())
    registry.register(UserProfileUpdatedTransformer())
    return registry
