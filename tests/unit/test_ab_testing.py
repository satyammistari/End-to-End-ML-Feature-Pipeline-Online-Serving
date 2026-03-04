"""
Unit tests for the A/B testing framework.
"""

from __future__ import annotations

import pytest

from src.core.ab_testing import ABTestManager
from src.core.schemas import Experiment, ExperimentStatus, ExperimentVariant


@pytest.fixture
def manager():
    return ABTestManager()


@pytest.fixture
def two_variant_experiment():
    return Experiment(
        name="checkout_flow_test",
        variants=[
            ExperimentVariant(name="control", traffic_fraction=0.5),
            ExperimentVariant(name="treatment", traffic_fraction=0.5),
        ],
    )


class TestABTestManager:
    def test_create_experiment(self, manager, two_variant_experiment):
        exp = manager.create_experiment(two_variant_experiment)
        assert exp.name == "checkout_flow_test"

    def test_duplicate_creates_raises(self, manager, two_variant_experiment):
        manager.create_experiment(two_variant_experiment)
        with pytest.raises(ValueError, match="already exists"):
            manager.create_experiment(two_variant_experiment)

    def test_get_variant_not_running_returns_control(self, manager, two_variant_experiment):
        manager.create_experiment(two_variant_experiment)
        # Not started → DRAFT, returns "control"
        variant = manager.get_variant("user_1", "checkout_flow_test")
        assert variant == "control"

    def test_get_variant_after_start(self, manager, two_variant_experiment):
        manager.create_experiment(two_variant_experiment)
        manager.start_experiment("checkout_flow_test")
        variant = manager.get_variant("user_1", "checkout_flow_test")
        assert variant in ("control", "treatment")

    def test_assignment_is_deterministic(self, manager, two_variant_experiment):
        manager.create_experiment(two_variant_experiment)
        manager.start_experiment("checkout_flow_test")
        v1 = manager.get_variant("user_abc", "checkout_flow_test")
        v2 = manager.get_variant("user_abc", "checkout_flow_test")
        assert v1 == v2

    def test_traffic_split_approximate(self, manager):
        """With 50/50 split, roughly half the users should get each variant."""
        exp = Experiment(
            name="split_test",
            variants=[
                ExperimentVariant(name="control", traffic_fraction=0.5),
                ExperimentVariant(name="treatment", traffic_fraction=0.5),
            ],
        )
        manager.create_experiment(exp)
        manager.start_experiment("split_test")

        counts = {"control": 0, "treatment": 0}
        for i in range(10000):
            v = manager.get_variant(f"user_{i}", "split_test")
            counts[v] += 1

        # Allow ±5% tolerance
        assert 4500 <= counts["control"] <= 5500
        assert 4500 <= counts["treatment"] <= 5500

    def test_stop_experiment(self, manager, two_variant_experiment):
        manager.create_experiment(two_variant_experiment)
        manager.start_experiment("checkout_flow_test")
        manager.stop_experiment("checkout_flow_test")
        exp = manager._experiments["checkout_flow_test"]
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.ended_at is not None

    def test_track_and_summarize_metrics(self, manager, two_variant_experiment):
        manager.create_experiment(two_variant_experiment)
        for _ in range(100):
            manager.track_metric("checkout_flow_test", "control", "conversion", 0.0)
            manager.track_metric("checkout_flow_test", "treatment", "conversion", 1.0)

        summary = manager.get_metrics_summary("checkout_flow_test")
        assert "conversion" in summary["metrics"]
        assert summary["metrics"]["conversion"]["control"]["mean"] == pytest.approx(0.0)
        assert summary["metrics"]["conversion"]["treatment"]["mean"] == pytest.approx(1.0)

    def test_analyze_results_significance(self, manager):
        exp = Experiment(
            name="revenue_test",
            variants=[
                ExperimentVariant(name="control", traffic_fraction=0.5),
                ExperimentVariant(name="treatment", traffic_fraction=0.5),
            ],
        )
        manager.create_experiment(exp)
        # Control: mean=10, Treatment: mean=15 (large effect → significant)
        import random
        rng = random.Random(42)
        for _ in range(500):
            manager.track_metric("revenue_test", "control", "revenue", rng.gauss(10, 1))
            manager.track_metric("revenue_test", "treatment", "revenue", rng.gauss(15, 1))

        results = manager.analyze_results("revenue_test", metric="revenue")
        assert results["variants"]["treatment"]["significant"] is True
        assert results["variants"]["treatment"]["lift_pct"] > 0

    def test_experiment_invalid_traffic_raises(self):
        with pytest.raises(ValueError):
            Experiment(
                name="bad_split",
                variants=[
                    ExperimentVariant(name="control", traffic_fraction=0.4),
                    ExperimentVariant(name="treatment", traffic_fraction=0.4),
                    # Total = 0.8, not 1.0
                ],
            )

    def test_list_experiments(self, manager, two_variant_experiment):
        manager.create_experiment(two_variant_experiment)
        assert "checkout_flow_test" in manager.list_experiments()
