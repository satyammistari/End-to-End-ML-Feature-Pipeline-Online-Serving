#!/usr/bin/env python3
"""
Generate realistic test data for the entire ML Feature Pipeline.

OUTPUT:
  1. Raw transaction events (Kafka input)       -> /tmp/test_data/transaction_events.jsonl
  2. User profiles (reference data)             -> /tmp/test_data/users.csv
  3. Merchant data (reference data)             -> /tmp/test_data/merchants.csv
  4. Fraud labels (ground truth for model)      -> /tmp/test_data/fraud_labels.csv
  5. Expected feature values (for validation)   -> /tmp/test_data/expected_features.csv
  6. Dataset summary                            -> /tmp/test_data/summary.json

Legacy Kafka producer mode (original):
  python scripts/generate_test_data.py --events 10000 --topic transaction-events
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

try:
    import numpy as np
    import pandas as pd
    from faker import Faker
    _HAS_DATA_LIBS = True
except ImportError:
    _HAS_DATA_LIBS = False

# --- Optional Kafka import (legacy mode) --------------------------------------
try:
    from kafka import KafkaProducer
    _HAS_KAFKA = True
except ImportError:
    _HAS_KAFKA = False

# -----------------------------------------------------------------------------
# TestDataGenerator  (Phase 0)
# -----------------------------------------------------------------------------

if _HAS_DATA_LIBS:
    fake = Faker()
    Faker.seed(42)
    np.random.seed(42)
    random.seed(42)

    class TestDataGenerator:
        """Generate a complete, self-consistent dataset for pipeline validation."""

        def __init__(self, num_users: int = 1000, num_merchants: int = 200, days: int = 30):
            self.num_users = num_users
            self.num_merchants = num_merchants
            self.days = days
            self.start_date = datetime.now() - timedelta(days=days)

        # -- Reference data ----------------------------------------------------

        def generate_users(self) -> "pd.DataFrame":
            """Generate user profiles."""
            users = []
            for i in range(self.num_users):
                users.append({
                    'user_id': f'user_{i:06d}',
                    'name': fake.name(),
                    'email': fake.email(),
                    'signup_date': str(fake.date_between(start_date='-2y', end_date='-30d')),
                    'country': fake.country_code(),
                    'risk_score': random.choice(['low', 'medium', 'high']),
                    'is_verified': random.choice([True, False]),
                    'age': random.randint(18, 80),
                    'account_balance': round(random.uniform(0, 50000), 2),
                })
            df = pd.DataFrame(users)
            df.to_csv('/tmp/test_data/users.csv', index=False)
            print(f"[OK] Generated {len(df)} users")
            return df

        def generate_merchants(self) -> "pd.DataFrame":
            """Generate merchant profiles."""
            categories = [
                'electronics', 'groceries', 'clothing', 'restaurants',
                'travel', 'entertainment', 'utilities', 'healthcare',
            ]
            merchants = []
            for i in range(self.num_merchants):
                merchants.append({
                    'merchant_id': f'merchant_{i:04d}',
                    'name': fake.company(),
                    'category': random.choice(categories),
                    'country': fake.country_code(),
                    'risk_rating': random.choice(['A', 'B', 'C', 'D']),
                    'average_transaction': round(random.uniform(10, 500), 2),
                    'is_verified': random.choice([True, False]),
                })
            df = pd.DataFrame(merchants)
            df.to_csv('/tmp/test_data/merchants.csv', index=False)
            print(f"[OK] Generated {len(df)} merchants")
            return df

        # -- Transaction events ------------------------------------------------

        def generate_transactions(
            self,
            users_df: "pd.DataFrame",
            merchants_df: "pd.DataFrame",
        ) -> tuple["pd.DataFrame", "pd.DataFrame"]:
            """
            Generate transactions with three user archetypes:
              - normal   (85 %)  : 1-5 txns/day, $10-200
              - heavy    (10 %)  : 10-20 txns/day, $50-500
              - fraudster (5 %)  : normal until last 3 days, then spike
            """
            user_types: Dict[str, str] = {}
            for uid in users_df['user_id']:
                user_types[uid] = random.choices(
                    ['normal', 'heavy', 'fraudster'],
                    weights=[0.85, 0.10, 0.05],
                )[0]

            transactions = []
            fraud_labels = []
            transaction_id = 0

            for day in range(self.days):
                current_date = self.start_date + timedelta(days=day)

                for user_id in users_df['user_id']:
                    utype = user_types[user_id]

                    if utype == 'normal':
                        num_txns = random.randint(1, 5)
                        amount_range = (10, 200)
                    elif utype == 'heavy':
                        num_txns = random.randint(10, 20)
                        amount_range = (50, 500)
                    else:  # fraudster
                        if day < self.days - 3:
                            num_txns = random.randint(1, 3)
                            amount_range = (10, 150)
                        else:
                            num_txns = random.randint(30, 50)
                            amount_range = (500, 5000)

                    for _ in range(num_txns):
                        ts = current_date + timedelta(
                            hours=random.randint(0, 23),
                            minutes=random.randint(0, 59),
                            seconds=random.randint(0, 59),
                        )
                        merchant = merchants_df.sample(1).iloc[0]
                        amount = round(random.uniform(*amount_range), 2)
                        is_fraud = utype == 'fraudster' and day >= self.days - 3

                        txn: Dict[str, Any] = {
                            'transaction_id': f'txn_{transaction_id:010d}',
                            'user_id': user_id,
                            'merchant_id': merchant['merchant_id'],
                            'merchant_category': merchant['category'],
                            'amount': amount,
                            'currency': 'USD',
                            'timestamp': ts.isoformat(),
                            'status': 'completed',
                            'payment_method': random.choice(
                                ['credit_card', 'debit_card', 'bank_transfer']
                            ),
                            'device_type': random.choice(['mobile', 'desktop', 'tablet']),
                            'ip_country': random.choice(['US', 'GB', 'CA', 'AU']),
                            'is_international': random.choice([True, False]),
                        }
                        transactions.append(txn)
                        fraud_labels.append({
                            'transaction_id': txn['transaction_id'],
                            'user_id': user_id,
                            'timestamp': ts.isoformat(),
                            'is_fraud': is_fraud,
                            'fraud_type': 'account_takeover' if is_fraud else None,
                        })
                        transaction_id += 1

            txn_df = pd.DataFrame(transactions)
            txn_df.to_csv('/tmp/test_data/transactions.csv', index=False)

            with open('/tmp/test_data/transaction_events.jsonl', 'w') as fh:
                for t in transactions:
                    fh.write(json.dumps(t) + '\n')

            labels_df = pd.DataFrame(fraud_labels)
            labels_df.to_csv('/tmp/test_data/fraud_labels.csv', index=False)

            print(f"[OK] Generated {len(txn_df)} transactions")
            print(f"   - Fraud rate: {labels_df['is_fraud'].mean() * 100:.2f}%")
            print(f"   - Date range: {txn_df['timestamp'].min()} to {txn_df['timestamp'].max()}")
            return txn_df, labels_df

        # -- Expected features (ground truth) ----------------------------------

        def generate_expected_features(self, transactions_df: "pd.DataFrame") -> "pd.DataFrame":
            """Pre-compute expected feature values for validation (ground truth)."""
            transactions_df = transactions_df.copy()
            transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
            transactions_df = transactions_df.sort_values('timestamp').reset_index(drop=True)

            records = []
            for _, txn in transactions_df.iterrows():
                uid = txn['user_id']
                t = txn['timestamp']

                hist = transactions_df[
                    (transactions_df['user_id'] == uid) & (transactions_df['timestamp'] < t)
                ]

                def _w(h=None, d=None) -> "pd.DataFrame":
                    delta = timedelta(hours=h) if h is not None else timedelta(days=d)
                    return hist[hist['timestamp'] > t - delta]

                h24 = _w(h=24)
                h7d = _w(d=7)
                h30d = _w(d=30)
                h1 = _w(h=1)

                records.append({
                    'transaction_id': txn['transaction_id'],
                    'user_id': uid,
                    'timestamp': t.isoformat(),
                    'transaction_count_24h': len(h24),
                    'transaction_count_7d': len(h7d),
                    'transaction_count_30d': len(h30d),
                    'total_amount_24h': float(h24['amount'].sum()),
                    'avg_amount_7d': float(h7d['amount'].mean()) if len(h7d) > 0 else 0.0,
                    'max_amount_30d': float(h30d['amount'].max()) if len(h30d) > 0 else 0.0,
                    'unique_merchants_7d': int(h7d['merchant_id'].nunique()),
                    'unique_categories_30d': int(h30d['merchant_category'].nunique()),
                    'transactions_last_hour': len(h1),
                    'amount_zscore': self._zscore(txn['amount'], hist['amount']),
                })

            df = pd.DataFrame(records)
            df.to_csv('/tmp/test_data/expected_features.csv', index=False)
            print(f"[OK] Generated expected features for {len(df)} transactions")
            return df

        @staticmethod
        def _zscore(value: float, history: "pd.Series") -> float:
            if len(history) < 2:
                return 0.0
            std = history.std(ddof=0)  # population std - matches RealtimeFeatureComputer
            if std == 0:
                return 0.0
            return float((value - history.mean()) / std)

        # -- Orchestrator ------------------------------------------------------

        def generate_all(self) -> dict:
            """Generate the complete test dataset."""
            os.makedirs('/tmp/test_data', exist_ok=True)
            print("[START] Generating test data...\n")

            users_df = self.generate_users()
            merchants_df = self.generate_merchants()
            txn_df, labels_df = self.generate_transactions(users_df, merchants_df)
            self.generate_expected_features(txn_df)

            summary = {
                'generation_time': datetime.now().isoformat(),
                'num_users': len(users_df),
                'num_merchants': len(merchants_df),
                'num_transactions': len(txn_df),
                'fraud_rate': float(labels_df['is_fraud'].mean()),
                'date_range': {
                    'start': str(txn_df['timestamp'].min()),
                    'end': str(txn_df['timestamp'].max()),
                },
                'total_volume': float(txn_df['amount'].sum()),
                'avg_transaction': float(txn_df['amount'].mean()),
            }

            with open('/tmp/test_data/summary.json', 'w') as fh:
                json.dump(summary, fh, indent=2, default=str)

            print("\n" + "=" * 60)
            print("TEST DATA SUMMARY")
            print("=" * 60)
            print(f"Users:           {summary['num_users']:,}")
            print(f"Merchants:       {summary['num_merchants']:,}")
            print(f"Transactions:    {summary['num_transactions']:,}")
            print(f"Fraud Rate:      {summary['fraud_rate'] * 100:.2f}%")
            print(f"Total Volume:    ${summary['total_volume']:,.2f}")
            print(f"Avg Transaction: ${summary['avg_transaction']:.2f}")
            print("=" * 60)
            print("\n[DONE] All test data saved to /tmp/test_data/")
            return summary


# -----------------------------------------------------------------------------
# Legacy Kafka producer  (kept for backward-compat)
# -----------------------------------------------------------------------------

_TOPICS = {
    "transaction-events": "transaction_created",
    "user-login-events": "user_login",
    "user-events": "user_profile_updated",
}

_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"]
_MERCHANT_CATEGORIES = ["retail", "dining", "travel", "gas", "grocery", "entertainment", "health"]


def _random_tx_payload(user_id: str) -> Dict[str, Any]:
    return {
        "amount": round(random.uniform(1.0, 5000.0), 2),
        "currency": random.choice(_CURRENCIES),
        "merchant_id": f"merchant_{random.randint(1, 500)}",
        "merchant_category": random.choice(_MERCHANT_CATEGORIES),
        "is_online": random.random() > 0.4,
    }


def _random_login_payload(user_id: str) -> Dict[str, Any]:
    if _HAS_DATA_LIBS:
        ip_country = fake.country_code()
        user_agent = fake.user_agent()
    else:
        ip_country = "US"
        user_agent = "Mozilla/5.0"
    return {
        "device": random.choice(["mobile", "desktop", "tablet"]),
        "ip_country": ip_country,
        "user_agent": user_agent,
    }


def _random_profile_payload(user_id: str) -> Dict[str, Any]:
    return {
        "account_age_days": random.randint(0, 3650),
        "kyc_verified": random.random() > 0.2,
        "preferred_currency": random.choice(_CURRENCIES),
    }


_PAYLOAD_GENERATORS = {
    "transaction_created": _random_tx_payload,
    "user_login": _random_login_payload,
    "user_profile_updated": _random_profile_payload,
}


def generate_event(event_type: str, user_pool: int = 10000) -> Dict[str, Any]:
    user_id = f"user_{random.randint(1, user_pool)}"
    occurred_at = datetime.now(tz=timezone.utc) - timedelta(
        seconds=random.randint(0, 30 * 86400)
    )
    payload_fn = _PAYLOAD_GENERATORS.get(event_type, lambda uid: {})
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "entity_id": user_id,
        "entity_type": "user",
        "occurred_at": occurred_at.isoformat(),
        "payload": payload_fn(user_id),
        "source": "test-generator",
    }


def produce_events(
    bootstrap_servers: str,
    topic: str,
    n_events: int,
    batch_size: int = 200,
) -> None:
    if not _HAS_KAFKA:
        raise ImportError("Install kafka-python: pip install kafka-python")
    event_type = _TOPICS.get(topic, "user_login")
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks=1,
        linger_ms=10,
        batch_size=16384,
    )
    sent = 0
    t0 = time.time()
    try:
        while sent < n_events:
            batch = [
                generate_event(event_type)
                for _ in range(min(batch_size, n_events - sent))
            ]
            for event in batch:
                producer.send(topic, event)
            producer.flush()
            sent += len(batch)
            elapsed = time.time() - t0
            print(f"\r[{topic}] {sent}/{n_events}  ({int(sent/elapsed)} events/s)", end="", flush=True)
    finally:
        producer.close()
    print(f"\nDone. Produced {n_events} events to '{topic}' in {time.time()-t0:.1f}s")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate test data for the ML Feature Pipeline.\n"
            "With no --topic flag: generates a full offline dataset.\n"
            "With --topic flag:   streams events to Kafka (legacy mode)."
        )
    )
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", choices=list(_TOPICS), default=None,
                        help="If set, uses legacy Kafka producer mode.")
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--users", type=int, default=1000)
    parser.add_argument("--merchants", type=int, default=200)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    if args.topic:
        produce_events(
            bootstrap_servers=args.bootstrap_servers,
            topic=args.topic,
            n_events=args.events,
            batch_size=args.batch_size,
        )
    else:
        if not _HAS_DATA_LIBS:
            print("Install data libs: pip install pandas numpy faker")
            raise SystemExit(1)
        generator = TestDataGenerator(
            num_users=args.users,
            num_merchants=args.merchants,
            days=args.days,
        )
        generator.generate_all()


if __name__ == "__main__":
    main()
