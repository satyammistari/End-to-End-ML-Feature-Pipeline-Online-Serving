"""
Feature Registry: central catalog for all feature groups and their versioned schemas.
Metadata persisted to PostgreSQL; hot schemas cached in Redis.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .schemas import (
    ChangeType,
    FeatureDefinition,
    FeatureGroup,
    FeatureGroupVersion,
    FeatureType,
)
from .validators import FeatureValidator

logger = logging.getLogger(__name__)


class RegistryError(Exception):
    """Raised on registry operation failures."""


class FeatureRegistry:
    """
    Central registry for feature group schemas.

    - CRUD operations for FeatureGroups and their versions
    - Schema validation & compatibility checks on version bumps
    - Lineage tracking (feature dependencies)
    - Metadata stored in PostgreSQL, hot-path cached in Redis

    Parameters
    ----------
    postgres_store : optional PostgresFeatureStore
        If provided, registry state is persisted to PostgreSQL.
    redis_store : optional RedisFeatureStore
        If provided, schema lookups are cached in Redis.
    """

    _CACHE_PREFIX = "registry:schema:"
    _CACHE_TTL = 300  # 5 minutes

    def __init__(
        self,
        postgres_store: Any = None,
        redis_store: Any = None,
    ) -> None:
        self._postgres = postgres_store
        self._redis = redis_store
        self._validator = FeatureValidator()
        # In-memory fallback when no external stores are configured
        self._memory: Dict[str, FeatureGroup] = {}

    # ------------------------------------------------------------------
    # FeatureGroup CRUD
    # ------------------------------------------------------------------

    def register_feature_group(self, group: FeatureGroup) -> FeatureGroup:
        """
        Register a new FeatureGroup.
        Raises RegistryError if a group with the same name already exists.
        """
        if self._load_group(group.name):
            raise RegistryError(
                f"Feature group '{group.name}' already exists. "
                "Use add_version() to bump the schema."
            )
        group.created_at = datetime.utcnow()
        group.updated_at = datetime.utcnow()
        self._save_group(group)
        logger.info("Registered feature group '%s'", group.name)
        return group

    def get_feature_group(self, name: str) -> Optional[FeatureGroup]:
        """Retrieve a FeatureGroup by name (with Redis cache)."""
        cached = self._cache_get(name)
        if cached:
            return FeatureGroup.model_validate_json(cached)
        group = self._load_group(name)
        if group:
            self._cache_set(name, group.model_dump_json())
        return group

    def list_feature_groups(self) -> List[str]:
        """Return names of all registered feature groups."""
        if self._postgres:
            try:
                return self._postgres.list_feature_groups()
            except Exception as exc:
                logger.warning("Postgres list failed, using memory: %s", exc)
        return list(self._memory.keys())

    def delete_feature_group(self, name: str) -> bool:
        """Delete a feature group and all its versions."""
        if name not in self._memory and not self._load_group(name):
            return False
        self._memory.pop(name, None)
        if self._postgres:
            try:
                self._postgres.delete_feature_group(name)
            except Exception as exc:
                logger.warning("Postgres delete failed: %s", exc)
        self._cache_delete(name)
        logger.info("Deleted feature group '%s'", name)
        return True

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------

    def add_version(
        self,
        group_name: str,
        features: List[FeatureDefinition],
        change_type: ChangeType = ChangeType.NON_BREAKING,
        changelog: str = "",
    ) -> FeatureGroupVersion:
        """
        Add a new version to an existing FeatureGroup.

        For BREAKING changes, bumps the major version number.
        For NON_BREAKING, bumps minor; PATCH keeps same major.
        Validates backward compatibility before persisting.
        """
        group = self._load_group(group_name)
        if not group:
            raise RegistryError(f"Feature group '{group_name}' not found")

        # Determine next version string
        next_ver = self._next_version(group, change_type)

        # Compatibility check for non-breaking claims
        if group.latest_version and change_type != ChangeType.BREAKING:
            latest = group.versions[group.latest_version]
            new_ver_obj = FeatureGroupVersion(
                version=next_ver,
                features=features,
                change_type=change_type,
                changelog=changelog,
            )
            ok, issues = self._validator.validate_version_compatibility(latest, new_ver_obj)
            if not ok:
                # Auto-upgrade change type
                logger.warning(
                    "Schema has breaking changes but change_type=%s. "
                    "Upgrading to BREAKING. Issues: %s",
                    change_type,
                    issues,
                )
                change_type = ChangeType.BREAKING
                next_ver = self._next_version(group, change_type)

        new_version = FeatureGroupVersion(
            version=next_ver,
            features=features,
            change_type=change_type,
            changelog=changelog,
        )
        group.versions[next_ver] = new_version
        group.latest_version = next_ver
        group.updated_at = datetime.utcnow()
        self._save_group(group)
        self._cache_delete(group_name)
        logger.info("Added version %s to group '%s'", next_ver, group_name)
        return new_version

    def get_version(
        self, group_name: str, version: Optional[str] = None
    ) -> Optional[FeatureGroupVersion]:
        """Return a specific version (or latest if version=None)."""
        group = self.get_feature_group(group_name)
        if not group:
            return None
        if version is None:
            return group.get_latest()
        return group.versions.get(version)

    def deprecate_version(
        self, group_name: str, version: str, message: str = ""
    ) -> bool:
        """Mark a version as deprecated."""
        group = self._load_group(group_name)
        if not group or version not in group.versions:
            return False
        group.versions[version].deprecated = True
        group.versions[version].deprecation_message = message
        group.updated_at = datetime.utcnow()
        self._save_group(group)
        self._cache_delete(group_name)
        logger.info("Deprecated version %s of group '%s'", version, group_name)
        return True

    # ------------------------------------------------------------------
    # Lineage helpers
    # ------------------------------------------------------------------

    def get_feature_lineage(self, group_name: str, feature_name: str) -> List[str]:
        """
        Return transitive dependencies of a feature.
        Walks depends_on fields recursively.
        """
        visited: List[str] = []
        self._walk_lineage(group_name, feature_name, visited, set())
        return visited

    def _walk_lineage(
        self,
        group_name: str,
        feature_name: str,
        result: List[str],
        seen: set,
    ) -> None:
        key = f"{group_name}.{feature_name}"
        if key in seen:
            return
        seen.add(key)
        group = self.get_feature_group(group_name)
        if not group:
            return
        latest = group.get_latest()
        if not latest:
            return
        for defn in latest.features:
            if defn.name == feature_name:
                for dep in defn.depends_on:
                    result.append(dep)
                    # deps may be qualified "group.feature" or just "feature"
                    if "." in dep:
                        g, f = dep.split(".", 1)
                        self._walk_lineage(g, f, result, seen)

    # ------------------------------------------------------------------
    # Internal persistence helpers
    # ------------------------------------------------------------------

    def _load_group(self, name: str) -> Optional[FeatureGroup]:
        if name in self._memory:
            return self._memory[name]
        if self._postgres:
            try:
                raw = self._postgres.get_feature_group_raw(name)
                if raw:
                    group = FeatureGroup.model_validate(raw)
                    self._memory[name] = group
                    return group
            except Exception as exc:
                logger.warning("Postgres load failed for '%s': %s", name, exc)
        return None

    def _save_group(self, group: FeatureGroup) -> None:
        self._memory[group.name] = group
        if self._postgres:
            try:
                self._postgres.save_feature_group(group.model_dump())
            except Exception as exc:
                logger.warning("Postgres save failed: %s", exc)

    def _cache_get(self, name: str) -> Optional[str]:
        if self._redis:
            try:
                return self._redis.get(f"{self._CACHE_PREFIX}{name}")
            except Exception:
                pass
        return None

    def _cache_set(self, name: str, value: str) -> None:
        if self._redis:
            try:
                self._redis.setex(
                    f"{self._CACHE_PREFIX}{name}", self._CACHE_TTL, value
                )
            except Exception:
                pass

    def _cache_delete(self, name: str) -> None:
        if self._redis:
            try:
                self._redis.delete(f"{self._CACHE_PREFIX}{name}")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Version numbering
    # ------------------------------------------------------------------

    @staticmethod
    def _next_version(group: FeatureGroup, change_type: ChangeType) -> str:
        if not group.latest_version:
            return "v1"
        current = int(group.latest_version[1:])
        if change_type == ChangeType.BREAKING:
            return f"v{current + 1}"
        # Non-breaking / patch: keep same major for now (simple integer versioning)
        return f"v{current + 1}"
