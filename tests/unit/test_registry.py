"""
Unit tests for FeatureRegistry and FeatureValidator.
"""

from __future__ import annotations

import pytest
from datetime import datetime

from src.core.registry import FeatureRegistry, RegistryError
from src.core.schemas import (
    ChangeType,
    FeatureComputationType,
    FeatureDefinition,
    FeatureGroup,
    FeatureGroupVersion,
    FeatureType,
    ValidationRule,
)
from src.core.validators import FeatureValidator


# ─── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def simple_feature() -> FeatureDefinition:
    return FeatureDefinition(
        name="transaction_count_24h",
        feature_type=FeatureType.INTEGER,
        computation_type=FeatureComputationType.REALTIME,
        description="Test feature",
        validation_rules=ValidationRule(min_value=0, max_value=10000),
    )


@pytest.fixture
def feature_group(simple_feature) -> FeatureGroup:
    v1 = FeatureGroupVersion(version="v1", features=[simple_feature])
    return FeatureGroup(
        name="user_features",
        entity_type="user",
        versions={"v1": v1},
        latest_version="v1",
    )


@pytest.fixture
def registry() -> FeatureRegistry:
    return FeatureRegistry()  # In-memory (no external stores)


# ─── Registry CRUD ──────────────────────────────────────────────────────────

class TestFeatureRegistry:
    def test_register_new_group(self, registry, feature_group):
        result = registry.register_feature_group(feature_group)
        assert result.name == "user_features"

    def test_register_duplicate_raises(self, registry, feature_group):
        registry.register_feature_group(feature_group)
        with pytest.raises(RegistryError):
            registry.register_feature_group(feature_group)

    def test_get_registered_group(self, registry, feature_group):
        registry.register_feature_group(feature_group)
        retrieved = registry.get_feature_group("user_features")
        assert retrieved is not None
        assert retrieved.name == "user_features"

    def test_get_missing_group_returns_none(self, registry):
        assert registry.get_feature_group("nonexistent") is None

    def test_list_feature_groups(self, registry, feature_group):
        assert registry.list_feature_groups() == []
        registry.register_feature_group(feature_group)
        names = registry.list_feature_groups()
        assert "user_features" in names

    def test_delete_feature_group(self, registry, feature_group):
        registry.register_feature_group(feature_group)
        assert registry.delete_feature_group("user_features") is True
        assert registry.get_feature_group("user_features") is None

    def test_delete_nonexistent_returns_false(self, registry):
        assert registry.delete_feature_group("nope") is False

    def test_add_version(self, registry, feature_group, simple_feature):
        registry.register_feature_group(feature_group)
        new_feature = FeatureDefinition(
            name="avg_amount_7d",
            feature_type=FeatureType.FLOAT,
        )
        v2 = registry.add_version(
            "user_features",
            features=[simple_feature, new_feature],
            change_type=ChangeType.NON_BREAKING,
            changelog="Added avg_amount_7d",
        )
        assert v2.version == "v2"
        group = registry.get_feature_group("user_features")
        assert group.latest_version == "v2"
        assert len(group.versions) == 2

    def test_add_version_to_unknown_group_raises(self, registry):
        with pytest.raises(RegistryError):
            registry.add_version("unknown_group", features=[], change_type=ChangeType.PATCH)

    def test_deprecate_version(self, registry, feature_group):
        registry.register_feature_group(feature_group)
        result = registry.deprecate_version("user_features", "v1", "Use v2")
        assert result is True
        group = registry.get_feature_group("user_features")
        assert group.versions["v1"].deprecated is True
        assert group.versions["v1"].deprecation_message == "Use v2"


# ─── Validators ─────────────────────────────────────────────────────────────

class TestFeatureValidator:
    def setup_method(self):
        self.validator = FeatureValidator()

    def _make_schema(self, features) -> FeatureGroupVersion:
        return FeatureGroupVersion(version="v1", features=features)

    def test_valid_integer_feature(self):
        schema = self._make_schema([
            FeatureDefinition(
                name="count",
                feature_type=FeatureType.INTEGER,
                validation_rules=ValidationRule(min_value=0, max_value=100),
            )
        ])
        errors = self.validator.validate_schema({"count": 42}, schema)
        assert errors == []

    def test_type_mismatch(self):
        schema = self._make_schema([
            FeatureDefinition(name="count", feature_type=FeatureType.INTEGER)
        ])
        errors = self.validator.validate_schema({"count": "not-an-int"}, schema)
        assert any("integer" in e for e in errors)

    def test_min_value_violation(self):
        schema = self._make_schema([
            FeatureDefinition(
                name="amount",
                feature_type=FeatureType.FLOAT,
                validation_rules=ValidationRule(min_value=0.0),
            )
        ])
        errors = self.validator.validate_schema({"amount": -5.0}, schema)
        assert len(errors) == 1

    def test_max_value_violation(self):
        schema = self._make_schema([
            FeatureDefinition(
                name="score",
                feature_type=FeatureType.FLOAT,
                validation_rules=ValidationRule(max_value=1.0),
            )
        ])
        errors = self.validator.validate_schema({"score": 1.5}, schema)
        assert len(errors) == 1

    def test_null_not_allowed(self):
        schema = self._make_schema([
            FeatureDefinition(
                name="count",
                feature_type=FeatureType.INTEGER,
                validation_rules=ValidationRule(not_null=True),
            )
        ])
        errors = self.validator.validate_schema({"count": None}, schema)
        assert len(errors) == 1

    def test_regex_validation(self):
        schema = self._make_schema([
            FeatureDefinition(
                name="currency",
                feature_type=FeatureType.STRING,
                validation_rules=ValidationRule(regex_pattern=r"^[A-Z]{3}$"),
            )
        ])
        assert self.validator.validate_schema({"currency": "USD"}, schema) == []
        assert len(self.validator.validate_schema({"currency": "usd"}, schema)) == 1

    def test_unknown_feature_in_schema(self):
        schema = self._make_schema([
            FeatureDefinition(name="known_feature", feature_type=FeatureType.INTEGER)
        ])
        errors = self.validator.validate_schema(
            {"known_feature": 1, "unknown": "x"}, schema
        )
        assert any("Unknown feature" in e for e in errors)

    def test_version_compatibility_breaking_removal(self):
        old = self._make_schema([
            FeatureDefinition(name="feat_a", feature_type=FeatureType.INTEGER),
            FeatureDefinition(name="feat_b", feature_type=FeatureType.STRING),
        ])
        new = self._make_schema([
            FeatureDefinition(name="feat_a", feature_type=FeatureType.INTEGER),
            # feat_b removed
        ])
        ok, issues = self.validator.validate_version_compatibility(old, new)
        assert ok is False
        assert any("feat_b" in i and "BREAKING" in i for i in issues)

    def test_version_compatibility_type_change(self):
        old = self._make_schema([
            FeatureDefinition(name="count", feature_type=FeatureType.INTEGER)
        ])
        new = self._make_schema([
            FeatureDefinition(name="count", feature_type=FeatureType.FLOAT)
        ])
        ok, issues = self.validator.validate_version_compatibility(old, new)
        assert ok is False

    def test_version_compatibility_additive_is_ok(self):
        old = self._make_schema([
            FeatureDefinition(name="feat_a", feature_type=FeatureType.INTEGER)
        ])
        new = self._make_schema([
            FeatureDefinition(name="feat_a", feature_type=FeatureType.INTEGER),
            FeatureDefinition(name="feat_b", feature_type=FeatureType.STRING),
        ])
        ok, issues = self.validator.validate_version_compatibility(old, new)
        assert ok is True
        assert issues == []
