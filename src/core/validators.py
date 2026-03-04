"""
Feature schema validators.
Validates feature values against schemas, type constraints, and custom rules.
"""

from __future__ import annotations

import importlib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .schemas import (
    ChangeType,
    FeatureDefinition,
    FeatureGroupVersion,
    FeatureRecord,
    FeatureType,
    ValidationRule,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------

_TYPE_CHECKERS: Dict[FeatureType, type] = {
    FeatureType.INTEGER: int,
    FeatureType.FLOAT: float,
    FeatureType.STRING: str,
    FeatureType.BOOLEAN: bool,
    FeatureType.LIST: list,
    FeatureType.MAP: dict,
}


class ValidationError(Exception):
    """Raised when feature validation fails."""

    def __init__(self, message: str, field: Optional[str] = None) -> None:
        self.field = field
        super().__init__(f"[{field}] {message}" if field else message)


class FeatureValidator:
    """
    Validates feature records against their group schema.

    Usage::

        validator = FeatureValidator()
        errors = validator.validate_record(record, schema_version)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_record(
        self,
        record: FeatureRecord,
        schema: FeatureGroupVersion,
    ) -> List[str]:
        """
        Validate a FeatureRecord against a FeatureGroupVersion.

        Returns a list of error strings (empty = valid).
        """
        errors: List[str] = []
        feature_map: Dict[str, FeatureDefinition] = {
            f.name: f for f in schema.features
        }

        for name, value in record.features.items():
            if name not in feature_map:
                errors.append(f"Unknown feature '{name}' not in schema version {schema.version}")
                continue
            defn = feature_map[name]
            errs = self._validate_single(name, value, defn)
            errors.extend(errs)

        return errors

    def validate_schema(
        self,
        features: Dict[str, Any],
        schema: FeatureGroupVersion,
    ) -> List[str]:
        """Validate a raw feature dict against a schema version."""
        dummy = FeatureRecord(
            entity_id="__validation__",
            entity_type="__validation__",
            feature_group="__validation__",
            feature_version=schema.version,
            features=features,
        )
        return self.validate_record(dummy, schema)

    def validate_version_compatibility(
        self,
        old_schema: FeatureGroupVersion,
        new_schema: FeatureGroupVersion,
    ) -> Tuple[bool, List[str]]:
        """
        Check whether new_schema is backward-compatible with old_schema.

        Returns (is_compatible, list_of_issues).
        Breaking changes:
        - Removing an existing feature
        - Changing a feature's type
        - Tightening constraints (lower max, higher min)
        """
        issues: List[str] = []
        old_map = {f.name: f for f in old_schema.features}
        new_map = {f.name: f for f in new_schema.features}

        # Removed features → breaking
        for name in old_map:
            if name not in new_map:
                issues.append(f"BREAKING: Feature '{name}' removed")

        # Type changes → breaking
        for name, old_defn in old_map.items():
            if name in new_map:
                new_defn = new_map[name]
                if old_defn.feature_type != new_defn.feature_type:
                    issues.append(
                        f"BREAKING: Feature '{name}' type changed "
                        f"{old_defn.feature_type} → {new_defn.feature_type}"
                    )

        # Constraint tightening → breaking
        for name, old_defn in old_map.items():
            if name not in new_map:
                continue
            new_defn = new_map[name]
            old_r = old_defn.validation_rules
            new_r = new_defn.validation_rules
            if old_r and new_r:
                if new_r.min_value is not None and (
                    old_r.min_value is None or new_r.min_value > old_r.min_value
                ):
                    issues.append(f"BREAKING: Feature '{name}' min_value tightened")
                if new_r.max_value is not None and (
                    old_r.max_value is None or new_r.max_value < old_r.max_value
                ):
                    issues.append(f"BREAKING: Feature '{name}' max_value tightened")

        is_compatible = len(issues) == 0
        return is_compatible, issues

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_single(
        self, name: str, value: Any, defn: FeatureDefinition
    ) -> List[str]:
        errors: List[str] = []

        # Null check
        if value is None:
            rules = defn.validation_rules
            if rules and rules.not_null:
                errors.append(f"Feature '{name}' must not be null")
            return errors  # Can't validate further if null

        # Type check
        expected_type = _TYPE_CHECKERS.get(defn.feature_type)
        if expected_type and not isinstance(value, expected_type):
            # Allow int where float expected
            if defn.feature_type == FeatureType.FLOAT and isinstance(value, int):
                pass
            else:
                errors.append(
                    f"Feature '{name}' expected {defn.feature_type.value}, "
                    f"got {type(value).__name__}"
                )

        rules = defn.validation_rules
        if rules:
            errors.extend(self._validate_rules(name, value, rules))

        return errors

    def _validate_rules(
        self, name: str, value: Any, rules: ValidationRule
    ) -> List[str]:
        errors: List[str] = []

        if rules.min_value is not None and isinstance(value, (int, float)):
            if value < rules.min_value:
                errors.append(
                    f"Feature '{name}' value {value} < min {rules.min_value}"
                )

        if rules.max_value is not None and isinstance(value, (int, float)):
            if value > rules.max_value:
                errors.append(
                    f"Feature '{name}' value {value} > max {rules.max_value}"
                )

        if rules.regex_pattern and isinstance(value, str):
            if not re.fullmatch(rules.regex_pattern, value):
                errors.append(
                    f"Feature '{name}' value '{value}' does not match "
                    f"pattern '{rules.regex_pattern}'"
                )

        if rules.allowed_values is not None:
            if value not in rules.allowed_values:
                errors.append(
                    f"Feature '{name}' value '{value}' not in allowed set"
                )

        if rules.custom_validator:
            try:
                module_path, fn_name = rules.custom_validator.rsplit(".", 1)
                mod = importlib.import_module(module_path)
                fn = getattr(mod, fn_name)
                result = fn(value)
                if result is not True and result is not None:
                    errors.append(
                        f"Feature '{name}' failed custom validator: {result}"
                    )
            except Exception as exc:
                logger.warning("Custom validator error for '%s': %s", name, exc)

        return errors
