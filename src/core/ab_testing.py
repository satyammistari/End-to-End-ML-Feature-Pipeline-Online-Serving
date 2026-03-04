"""
A/B testing framework for feature experiments.

Supports:
- Consistent hash-based user assignment (deterministic, no state required)
- Multi-variant traffic splitting
- Metric tracking per variant
- Simple statistical significance checks (z-test for proportions)
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from .schemas import Experiment, ExperimentStatus, ExperimentVariant

logger = logging.getLogger(__name__)


class ABTestManager:
    """
    Manages feature experiments and traffic allocation.

    Parameters
    ----------
    store : optional Redis/Postgres store for persisting experiment state
    """

    def __init__(self, store: Any = None) -> None:
        self._experiments: Dict[str, Experiment] = {}
        self._metrics: Dict[str, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )  # {experiment: {variant: [values]}}
        self._store = store

    # ------------------------------------------------------------------
    # Experiment lifecycle
    # ------------------------------------------------------------------

    def create_experiment(self, experiment: Experiment) -> Experiment:
        """Register a new experiment. Raises if name already exists."""
        if experiment.name in self._experiments:
            raise ValueError(f"Experiment '{experiment.name}' already exists")
        self._experiments[experiment.name] = experiment
        logger.info("Created experiment '%s' with %d variants", experiment.name, len(experiment.variants))
        return experiment

    def start_experiment(self, name: str) -> None:
        exp = self._get_experiment(name)
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.utcnow()
        logger.info("Started experiment '%s'", name)

    def stop_experiment(self, name: str) -> None:
        exp = self._get_experiment(name)
        exp.status = ExperimentStatus.COMPLETED
        exp.ended_at = datetime.utcnow()
        logger.info("Stopped experiment '%s'", name)

    def list_experiments(self) -> List[str]:
        return list(self._experiments.keys())

    # ------------------------------------------------------------------
    # Traffic assignment
    # ------------------------------------------------------------------

    def get_variant(self, entity_id: str, experiment_name: str) -> str:
        """
        Assign an entity to a variant using consistent hashing.

        The assignment is deterministic: the same entity always gets the
        same variant for the lifetime of the experiment.  This prevents
        the novelty effect and ensures holdout integrity.

        Returns the variant name, or "control" if experiment not found.
        """
        exp = self._experiments.get(experiment_name)
        if exp is None or exp.status != ExperimentStatus.RUNNING:
            return "control"

        # Hash entity_id + experiment_name → integer bucket [0, 10000)
        raw = f"{experiment_name}:{entity_id}".encode()
        digest = hashlib.md5(raw).hexdigest()
        bucket = int(digest[:8], 16) % 10000  # 0–9999

        # Walk variants in order; assign to first one whose cumulative
        # traffic fraction covers the bucket
        cumulative = 0.0
        for variant in exp.variants:
            cumulative += variant.traffic_fraction
            if bucket < int(cumulative * 10000):
                return variant.name

        # Fallback: last variant
        return exp.variants[-1].name

    # ------------------------------------------------------------------
    # Metric collection
    # ------------------------------------------------------------------

    def track_metric(
        self,
        experiment_name: str,
        variant_name: str,
        metric: str,
        value: float,
    ) -> None:
        """Record a metric observation for a variant."""
        key = f"{experiment_name}:{metric}"
        self._metrics[key][variant_name].append(value)

    def get_metrics_summary(
        self, experiment_name: str
    ) -> Dict[str, Any]:
        """
        Return descriptive statistics for each metric/variant combination.
        """
        summary: Dict[str, Any] = {"experiment": experiment_name, "metrics": {}}
        for full_key, variant_data in self._metrics.items():
            if not full_key.startswith(f"{experiment_name}:"):
                continue
            metric_name = full_key[len(experiment_name) + 1:]
            summary["metrics"][metric_name] = {}
            for variant, values in variant_data.items():
                if not values:
                    continue
                n = len(values)
                mean = sum(values) / n
                variance = sum((x - mean) ** 2 for x in values) / max(n - 1, 1)
                summary["metrics"][metric_name][variant] = {
                    "n": n,
                    "mean": round(mean, 6),
                    "std": round(math.sqrt(variance), 6),
                    "min": round(min(values), 6),
                    "max": round(max(values), 6),
                }
        return summary

    # ------------------------------------------------------------------
    # Statistical significance (z-test for means)
    # ------------------------------------------------------------------

    def analyze_results(
        self,
        experiment_name: str,
        metric: str,
        control_variant: str = "control",
        significance_level: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Perform a two-sample z-test comparing each variant against control.

        Returns dict with p-values and significance flags per variant.
        """
        key = f"{experiment_name}:{metric}"
        data = self._metrics.get(key, {})
        control_vals = data.get(control_variant, [])
        if not control_vals:
            return {"error": "No control data found"}

        n_c = len(control_vals)
        mean_c = sum(control_vals) / n_c
        var_c = sum((x - mean_c) ** 2 for x in control_vals) / max(n_c - 1, 1)

        results: Dict[str, Any] = {"experiment": experiment_name, "metric": metric, "variants": {}}
        for variant, vals in data.items():
            if variant == control_variant or not vals:
                continue
            n_t = len(vals)
            mean_t = sum(vals) / n_t
            var_t = sum((x - mean_t) ** 2 for x in vals) / max(n_t - 1, 1)

            se = math.sqrt(var_c / max(n_c, 1) + var_t / max(n_t, 1))
            z = (mean_t - mean_c) / se if se > 0 else 0.0
            # Two-tailed p-value approximation using standard normal CDF
            p_val = 2 * (1 - _norm_cdf(abs(z)))

            results["variants"][variant] = {
                "n": n_t,
                "mean": round(mean_t, 6),
                "lift_pct": round((mean_t - mean_c) / max(abs(mean_c), 1e-9) * 100, 2),
                "z_score": round(z, 4),
                "p_value": round(p_val, 6),
                "significant": p_val < significance_level,
            }
        return results

    # ------------------------------------------------------------------
    # Multi-armed bandit allocation (ε-greedy)
    # ------------------------------------------------------------------

    def epsilon_greedy_allocation(
        self,
        experiment_name: str,
        metric: str,
        epsilon: float = 0.1,
    ) -> Dict[str, float]:
        """
        Compute updated traffic fractions using ε-greedy selection.

        Returns a dict of {variant_name: new_fraction}.
        """
        key = f"{experiment_name}:{metric}"
        data = self._metrics.get(key, {})
        if not data:
            return {}

        means = {
            v: sum(vals) / len(vals)
            for v, vals in data.items()
            if vals
        }
        if not means:
            return {}

        best_variant = max(means, key=means.__getitem__)
        n_variants = len(means)
        allocations: Dict[str, float] = {}
        for v in means:
            if v == best_variant:
                allocations[v] = 1.0 - epsilon + (epsilon / n_variants)
            else:
                allocations[v] = epsilon / n_variants
        return allocations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_experiment(self, name: str) -> Experiment:
        exp = self._experiments.get(name)
        if exp is None:
            raise ValueError(f"Experiment '{name}' not found")
        return exp


def _norm_cdf(x: float) -> float:
    """Approximate standard normal CDF using the error function."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
