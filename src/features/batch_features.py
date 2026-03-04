"""
Batch feature computation using Polars (pure-Python, no JVM dependency).

Falls back to pandas-style processing when Polars is not available.
Run as a daily/hourly scheduled job (cron / Airflow / Kubernetes CronJob).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False
    logger.warning("Polars not installed – batch features will use fallback mode")


# ---------------------------------------------------------------------------
# Batch job base
# ---------------------------------------------------------------------------

class BatchFeatureJob:
    """
    Abstract base class for batch feature jobs.

    Subclasses implement ``compute()`` and this base class handles
    schema validation and persistence.
    """

    feature_group: str = ""
    feature_version: str = "v1"

    def __init__(self, offline_store: Any, online_store: Any = None) -> None:
        self._offline = offline_store
        self._online = online_store

    def run(self, start_date: datetime, end_date: datetime) -> None:
        logger.info(
            "[%s] Running batch job %s → %s",
            self.feature_group,
            start_date.date(),
            end_date.date(),
        )
        records = self.compute(start_date, end_date)
        if records:
            self._offline.write_features_batch(records)
            logger.info("[%s] Wrote %d records", self.feature_group, len(records))

    def compute(
        self, start_date: datetime, end_date: datetime
    ) -> List[Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# User batch features
# ---------------------------------------------------------------------------

class UserAggregateBatchJob(BatchFeatureJob):
    """
    Computes daily user-level aggregations:

    - avg_transaction_amount_7d
    - transaction_count_24h  (from offline history — overrides realtime on daily run)
    - std_transaction_amount_7d
    - unique_merchants_30d
    - user_lifetime_value
    - typical_active_hours  (list of typical transaction hours)
    """

    feature_group = "user_features"

    def compute(
        self, start_date: datetime, end_date: datetime
    ) -> List[Any]:
        from ..core.schemas import FeatureRecord

        # Fetch raw transaction data from the offline store
        training_rows = self._offline.read_training_data(
            entity_ids=[],  # empty = all entities (store handles this)
            feature_names=[
                "transaction_amount",
                "merchant_id",
            ],
            start_time=start_date - timedelta(days=30),
            end_time=end_date,
            feature_group="transaction_features",
        )

        if not training_rows:
            logger.warning("No transaction data found for date range")
            return []

        if _HAS_POLARS:
            return self._compute_polars(training_rows, end_date)
        return self._compute_python(training_rows, end_date)

    def _compute_polars(self, rows: List[Dict[str, Any]], as_of: datetime) -> List[Any]:
        from ..core.schemas import FeatureRecord

        df = pl.DataFrame(rows)
        # Ensure proper types
        if "event_time" in df.columns:
            df = df.with_columns(pl.col("event_time").cast(pl.Datetime))

        cutoff_7d = as_of - timedelta(days=7)
        cutoff_30d = as_of - timedelta(days=30)

        records = []
        for entity_id in df["entity_id"].unique().to_list():
            entity_df = df.filter(pl.col("entity_id") == entity_id)

            amt_7d = (
                entity_df
                .filter(pl.col("event_time") >= cutoff_7d)
                .filter(pl.col("feature_name") == "transaction_amount")
                ["feature_value"]
                .cast(pl.Float64, strict=False)
            )

            all_merchants = (
                entity_df
                .filter(pl.col("event_time") >= cutoff_30d)
                .filter(pl.col("feature_name") == "merchant_id")
                ["feature_value"]
                .to_list()
            )

            all_amounts = (
                entity_df
                .filter(pl.col("feature_name") == "transaction_amount")
                ["feature_value"]
                .cast(pl.Float64, strict=False)
                .to_list()
            )

            features = {
                "avg_transaction_amount_7d": float(amt_7d.mean() or 0.0),
                "std_transaction_amount_7d": float(amt_7d.std() or 0.0),
                "unique_merchants_30d": len(set(m for m in all_merchants if m)),
                "user_lifetime_value": float(sum(v for v in all_amounts if v)),
            }

            records.append(FeatureRecord(
                entity_id=entity_id,
                entity_type="user",
                feature_group=self.feature_group,
                feature_version=self.feature_version,
                features=features,
                event_time=as_of,
            ))
        return records

    def _compute_python(self, rows: List[Dict[str, Any]], as_of: datetime) -> List[Any]:
        """Pure-Python fallback – slower but dependency-free."""
        from ..core.schemas import FeatureRecord

        cutoff_7d = as_of - timedelta(days=7)
        cutoff_30d = as_of - timedelta(days=30)

        # Group by entity_id
        by_entity: Dict[str, List[Dict]] = {}
        for row in rows:
            by_entity.setdefault(row["entity_id"], []).append(row)

        records = []
        for entity_id, entity_rows in by_entity.items():
            amounts_7d = []
            merchants_30d = set()
            all_amounts = []

            for r in entity_rows:
                et = r.get("event_time")
                fname = r.get("feature_name", "")
                fval = r.get("feature_value")

                if fname == "transaction_amount" and fval is not None:
                    try:
                        v = float(fval)
                        all_amounts.append(v)
                        if et and et >= cutoff_7d:
                            amounts_7d.append(v)
                    except (TypeError, ValueError):
                        pass

                if fname == "merchant_id" and fval and et and et >= cutoff_30d:
                    merchants_30d.add(str(fval))

            avg_7d = sum(amounts_7d) / len(amounts_7d) if amounts_7d else 0.0
            if len(amounts_7d) > 1:
                mean = avg_7d
                std_7d = (sum((x - mean) ** 2 for x in amounts_7d) / (len(amounts_7d) - 1)) ** 0.5
            else:
                std_7d = 0.0

            features = {
                "avg_transaction_amount_7d": avg_7d,
                "std_transaction_amount_7d": std_7d,
                "unique_merchants_30d": len(merchants_30d),
                "user_lifetime_value": sum(all_amounts),
            }

            records.append(FeatureRecord(
                entity_id=entity_id,
                entity_type="user",
                feature_group=self.feature_group,
                feature_version=self.feature_version,
                features=features,
                event_time=as_of,
            ))
        return records


# ---------------------------------------------------------------------------
# Temporal features
# ---------------------------------------------------------------------------

def temporal_features(ts: datetime) -> Dict[str, Any]:
    """
    Extract temporal signals from a timestamp.
    These are computed on-the-fly (no store needed).
    """
    return {
        "hour_of_day": ts.hour,
        "day_of_week": ts.weekday(),          # 0=Monday
        "is_weekend": ts.weekday() >= 5,
        "is_business_hours": 9 <= ts.hour <= 17,
        "week_of_year": ts.isocalendar().week,
        "month": ts.month,
        "quarter": (ts.month - 1) // 3 + 1,
    }
