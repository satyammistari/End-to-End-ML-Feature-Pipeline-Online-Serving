"""
Unit tests for event transformers.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.core.schemas import RawEvent
from src.ingestion.transformers import (
    TransactionCreatedTransformer,
    UserLoginTransformer,
    UserProfileUpdatedTransformer,
    EventTransformerRegistry,
    build_default_registry,
)


def _event(event_type: str, entity_id: str = "u1", payload: dict = None) -> RawEvent:
    return RawEvent(
        event_id="evt_001",
        event_type=event_type,
        entity_id=entity_id,
        entity_type="user",
        occurred_at=datetime(2025, 6, 1, 14, 0, 0, tzinfo=timezone.utc),
        payload=payload or {},
        source="test",
    )


class TestTransactionCreatedTransformer:
    def setup_method(self):
        self.t = TransactionCreatedTransformer()

    def test_event_type(self):
        assert self.t.event_type == "transaction_created"

    def test_transform_basic(self):
        event = _event(
            "transaction_created",
            entity_id="user_42",
            payload={
                "amount": "99.99",
                "currency": "GBP",
                "merchant_id": "m_001",
                "merchant_category": "retail",
                "is_online": True,
            },
        )
        records = self.t.transform(event)
        assert len(records) == 1
        r = records[0]
        assert r.entity_id == "user_42"
        assert r.features["transaction_amount"] == pytest.approx(99.99)
        assert r.features["transaction_currency"] == "GBP"
        assert r.features["merchant_id"] == "m_001"
        assert r.features["is_online"] is True
        assert r.feature_group == "transaction_features"

    def test_missing_amount_defaults_to_zero(self):
        records = self.t.transform(_event("transaction_created"))
        assert records[0].features["transaction_amount"] == 0.0

    def test_event_time_preserved(self):
        event = _event("transaction_created")
        records = self.t.transform(event)
        assert records[0].event_time == event.occurred_at


class TestUserLoginTransformer:
    def setup_method(self):
        self.t = UserLoginTransformer()

    def test_event_type(self):
        assert self.t.event_type == "user_login"

    def test_transform(self):
        event = _event(
            "user_login",
            entity_id="user_7",
            payload={"device": "mobile", "ip_country": "US"},
        )
        records = self.t.transform(event)
        assert len(records) == 1
        r = records[0]
        assert r.entity_id == "user_7"
        assert r.features["login_device"] == "mobile"
        assert r.features["login_ip_country"] == "US"
        assert "last_login_timestamp" in r.features


class TestUserProfileUpdatedTransformer:
    def setup_method(self):
        self.t = UserProfileUpdatedTransformer()

    def test_transform(self):
        event = _event(
            "user_profile_updated",
            payload={"account_age_days": 365, "kyc_verified": True},
        )
        records = self.t.transform(event)
        r = records[0]
        assert r.features["account_age_days"] == 365
        assert r.features["kyc_verified"] is True


class TestEventTransformerRegistry:
    def test_register_and_get(self):
        registry = EventTransformerRegistry()
        t = TransactionCreatedTransformer()
        registry.register(t)
        assert registry.get("transaction_created") is t

    def test_get_unknown_returns_none(self):
        registry = EventTransformerRegistry()
        assert registry.get("unknown_event") is None

    def test_list_types(self):
        registry = build_default_registry()
        types = registry.list_types()
        assert "transaction_created" in types
        assert "user_login" in types
        assert "user_profile_updated" in types

    def test_build_default_registry(self):
        registry = build_default_registry()
        assert registry.get("transaction_created") is not None
        assert registry.get("user_login") is not None
