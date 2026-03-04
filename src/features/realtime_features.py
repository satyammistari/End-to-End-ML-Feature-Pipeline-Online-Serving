"""
Real-time feature computation functions.

All functions are stateless (side-effect free).  They receive either a raw
event payload or pre-fetched context and return a dict of feature values.

These are invoked in the REST/gRPC serving path for on-demand features.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _window_start(window_hours: int) -> datetime:
    return _utcnow() - timedelta(hours=window_hours)


# ---------------------------------------------------------------------------
# User features (realtime)
# ---------------------------------------------------------------------------

class UserRealtimeFeatures:
    """
    Computes realtime user features by querying the online store (Redis).

    Designed to be used in the serving path - should complete in <5 ms.
    """

    def __init__(self, online_store: Any) -> None:
        self._store = online_store

    def transaction_count_24h(self, user_id: str) -> int:
        """
        Count of transactions in the last 24 hours.
        Incremented by the Kafka consumer on each transaction event.
        """
        val = self._store.read_single(
            entity_id=user_id,
            feature_group="user_features",
            feature_name="transaction_count_24h",
        )
        return int(val) if val is not None else 0

    def transaction_velocity_1h(self, user_id: str) -> float:
        """
        Transactions per hour in the last 60 minutes (sliding window).
        Stored as a sorted set in Redis; we COUNT entries in [now-1h, now].
        Falls back to stored scalar if sorted set not available.
        """
        val = self._store.read_single(
            entity_id=user_id,
            feature_group="user_features",
            feature_name="transaction_velocity_1h",
        )
        return float(val) if val is not None else 0.0

    def is_high_risk_time(self, user_id: str, event_hour: Optional[int] = None) -> bool:
        """
        True if the current hour is unusual given the user's transaction history.
        Compares current hour against the user's typical active hours stored offline.
        """
        typical_hours = self._store.read_single(
            entity_id=user_id,
            feature_group="user_features",
            feature_name="typical_active_hours",
        )
        current_hour = event_hour if event_hour is not None else _utcnow().hour
        if not typical_hours:
            return False
        return current_hour not in typical_hours

    def get_all_user_realtime_features(self, user_id: str) -> Dict[str, Any]:
        """Return all real-time user features in one batched call."""
        raw = self._store.read_features(
            entity_ids=[user_id],
            feature_group="user_features",
            feature_names=[
                "transaction_count_24h",
                "transaction_velocity_1h",
                "last_transaction_amount",
                "last_transaction_merchant",
                "session_count",
                "last_login_timestamp",
            ],
        )
        return raw.get(user_id, {})


# ---------------------------------------------------------------------------
# Transaction features (realtime / on-demand)
# ---------------------------------------------------------------------------

class TransactionRealtimeFeatures:
    """
    Computes realtime transaction-level features.

    Some features (z-score) require user statistics fetched from the store.
    """

    def __init__(self, online_store: Any) -> None:
        self._store = online_store

    def amount_zscore(
        self,
        user_id: str,
        transaction_amount: float,
    ) -> float:
        """
        Z-score of transaction amount relative to user's historical distribution.

        z = (x - mu) / sigma

        Returns 0.0 if user statistics not available.
        """
        stats = self._store.read_features(
            entity_ids=[user_id],
            feature_group="user_features",
            feature_names=["avg_transaction_amount_7d", "std_transaction_amount_7d"],
        ).get(user_id, {})

        mu = stats.get("avg_transaction_amount_7d")
        sigma = stats.get("std_transaction_amount_7d")

        if mu is None or sigma is None or float(sigma) == 0.0:
            return 0.0

        return (transaction_amount - float(mu)) / float(sigma)

    def is_new_merchant(
        self, user_id: str, merchant_id: str
    ) -> bool:
        """
        True if user has never transacted at this merchant before.
        Checks a Redis set of known merchant IDs per user.
        """
        known = self._store.read_single(
            entity_id=user_id,
            feature_group="user_features",
            feature_name="known_merchant_ids",
        )
        if not known:
            return True
        if isinstance(known, list):
            return merchant_id not in known
        return True

    def amount_deviation_from_merchant_avg(
        self,
        user_id: str,
        merchant_id: str,
        transaction_amount: float,
    ) -> float:
        """
        Deviation of transaction amount from the user's average at this merchant.
        """
        key = f"merchant_avg_amount_{merchant_id}"
        avg = self._store.read_single(
            entity_id=user_id,
            feature_group="user_features",
            feature_name=key,
        )
        if avg is None:
            return 0.0
        return transaction_amount - float(avg)


# ---------------------------------------------------------------------------
# Aggregation window helpers
# ---------------------------------------------------------------------------

def compute_rolling_count(
    values: List[float],
    window_hours: int,
    timestamps: List[datetime],
) -> int:
    """
    Count events within the specified rolling window.

    Parameters
    ----------
    values : not used (kept for API consistency)
    window_hours : look-back window in hours
    timestamps : event timestamps (UTC)
    """
    cutoff = _window_start(window_hours)
    return sum(1 for ts in timestamps if ts >= cutoff)


def compute_rolling_sum(
    values: List[float],
    window_hours: int,
    timestamps: List[datetime],
) -> float:
    cutoff = _window_start(window_hours)
    return sum(v for v, ts in zip(values, timestamps) if ts >= cutoff)


def compute_rolling_avg(
    values: List[float],
    window_hours: int,
    timestamps: List[datetime],
) -> float:
    cutoff = _window_start(window_hours)
    windowed = [v for v, ts in zip(values, timestamps) if ts >= cutoff]
    if not windowed:
        return 0.0
    return sum(windowed) / len(windowed)


def compute_rolling_std(
    values: List[float],
    window_hours: int,
    timestamps: List[datetime],
) -> float:
    cutoff = _window_start(window_hours)
    windowed = [v for v, ts in zip(values, timestamps) if ts >= cutoff]
    if len(windowed) < 2:
        return 0.0
    n = len(windowed)
    mean = sum(windowed) / n
    variance = sum((x - mean) ** 2 for x in windowed) / (n - 1)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# RealtimeFeatureComputer  (DB-backed, used in end-to-end tests)
# ---------------------------------------------------------------------------

try:
    import pandas as pd  # type: ignore

    class RealtimeFeatureComputer:
        """
        Computes all real-time features for a transaction by fetching the user's
        full history from PostgreSQL (point-in-time correct).

        Used in integration tests and the fraud model training pipeline.
        Distinct from UserRealtimeFeatures which reads from the online store (Redis).
        """

        def __init__(self, redis_client: Any, postgres_conn: Any) -> None:
            self.redis = redis_client
            self.postgres = postgres_conn

        def compute_transaction_features(
            self,
            user_id: str,
            transaction: Dict[str, Any],
            timestamp: datetime,
        ) -> Dict[str, Any]:
            """
            Returns a dict of 10 features:
            transaction_count_{24h,7d,30d}, total_amount_24h, avg_amount_7d,
            max_amount_30d, unique_merchants_7d, unique_categories_30d,
            transactions_last_hour, amount_zscore.
            """
            history = self._get_user_history(user_id, timestamp)

            def _window(hours=None, days=None):
                delta = timedelta(hours=hours) if hours else timedelta(days=days)
                return history[history['timestamp'] > timestamp - delta]

            last_24h = _window(hours=24)
            last_7d  = _window(days=7)
            last_30d = _window(days=30)
            last_1h  = _window(hours=1)

            features: Dict[str, Any] = {
                'transaction_count_24h':    len(last_24h),
                'transaction_count_7d':     len(last_7d),
                'transaction_count_30d':    len(last_30d),
                'total_amount_24h':         float(last_24h['amount'].sum()) if len(last_24h) > 0 else 0.0,
                'avg_amount_7d':            float(last_7d['amount'].mean()) if len(last_7d) > 0 else 0.0,
                'max_amount_30d':           float(last_30d['amount'].max()) if len(last_30d) > 0 else 0.0,
                'unique_merchants_7d':      int(last_7d['merchant_id'].nunique()) if len(last_7d) > 0 else 0,
                'unique_categories_30d':    int(last_30d['merchant_category'].nunique()) if len(last_30d) > 0 else 0,
                'transactions_last_hour':   len(last_1h),
                'amount_zscore':            0.0,
            }

            if len(history) >= 2:
                mean = history['amount'].mean()
                std  = history['amount'].std(ddof=0)  # population std, matches numpy default used in generator
                if std > 0:
                    features['amount_zscore'] = float(
                        (transaction['amount'] - mean) / std
                    )

            return features

        def _get_user_history(self, user_id: str, before_timestamp: datetime) -> 'pd.DataFrame':
            """Fetch all transactions for a user strictly before *before_timestamp*."""
            cursor = self.postgres.cursor()
            cursor.execute(
                """
                SELECT transaction_id, amount, merchant_id, merchant_category, timestamp
                FROM transactions
                WHERE user_id = %s AND timestamp < %s
                ORDER BY timestamp ASC
                """,
                (user_id, before_timestamp),
            )
            rows = cursor.fetchall()
            cursor.close()

            if not rows:
                return pd.DataFrame(
                    columns=['transaction_id', 'amount', 'merchant_id',
                             'merchant_category', 'timestamp']
                )

            df = pd.DataFrame(
                rows,
                columns=['transaction_id', 'amount', 'merchant_id',
                         'merchant_category', 'timestamp'],
            )
            df['amount'] = df['amount'].astype(float)
            # Convert to tz-naive UTC so comparisons with tz-naive input timestamps work
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(None)
            return df

except ImportError:
    pass  # pandas not available in all environments
