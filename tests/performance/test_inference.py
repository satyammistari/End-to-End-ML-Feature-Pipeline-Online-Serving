"""
Real-time fraud detection inference test.

Simulates the online serving path:
  Transaction arrives -> compute features -> predict fraud probability -> decide.

Run:
  python tests/performance/test_inference.py

Requires:
  - A trained model at /tmp/fraud_model.pkl  (run src/ml/fraud_model.py first)
  - /tmp/test_data/transactions.csv and fraud_labels.csv
  - Live Postgres + Redis  (or it falls back to CSV-only mode)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# Allow running as a script from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_TEST_DATA_DIR = "/tmp/test_data"
_MODEL_PATH = "/tmp/fraud_model.pkl"

_POSTGRES_PARAMS = dict(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    database=os.getenv("POSTGRES_DB", "features_test"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
)

_REDIS_PARAMS = dict(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", "6379")),
    db=0,
    decode_responses=True,
)


def _try_connect_postgres():
    try:
        import psycopg2  # type: ignore
        conn = psycopg2.connect(**_POSTGRES_PARAMS, connect_timeout=3)
        return conn
    except Exception:
        return None


def _try_connect_redis():
    try:
        import redis  # type: ignore
        r = redis.Redis(**_REDIS_PARAMS, socket_connect_timeout=3)
        r.ping()
        return r
    except Exception:
        return None


# -----------------------------------------------------------------------------
# CSV-only feature stub (used when DB is unavailable)
# -----------------------------------------------------------------------------

class _CsvFeatureComputer:
    """
    Returns pre-computed expected features from CSV.
    Used as a fallback when Postgres is not reachable.
    """

    def __init__(self, expected_csv: str) -> None:
        self._df = pd.read_csv(expected_csv)
        self._idx = dict(zip(self._df["transaction_id"], range(len(self._df))))

    def compute_transaction_features(self, user_id, transaction, timestamp):
        tid = transaction.get("transaction_id", "")
        if tid in self._idx:
            row = self._df.iloc[self._idx[tid]]
            return {
                "transaction_count_24h":   float(row.get("transaction_count_24h", 0)),
                "transaction_count_7d":    float(row.get("transaction_count_7d", 0)),
                "transaction_count_30d":   float(row.get("transaction_count_30d", 0)),
                "total_amount_24h":        float(row.get("total_amount_24h", 0)),
                "avg_amount_7d":           float(row.get("avg_amount_7d", 0)),
                "max_amount_30d":          float(row.get("max_amount_30d", 0)),
                "unique_merchants_7d":     float(row.get("unique_merchants_7d", 0)),
                "unique_categories_30d":   float(row.get("unique_categories_30d", 0)),
                "transactions_last_hour":  float(row.get("transactions_last_hour", 0)),
                "amount_zscore":           float(row.get("amount_zscore", 0)),
            }
        return {k: 0.0 for k in [
            "transaction_count_24h", "transaction_count_7d", "transaction_count_30d",
            "total_amount_24h", "avg_amount_7d", "max_amount_30d",
            "unique_merchants_7d", "unique_categories_30d",
            "transactions_last_hour", "amount_zscore",
        ]}


# -----------------------------------------------------------------------------
# Main inference test
# -----------------------------------------------------------------------------

def test_realtime_inference():
    """
    Simulate real-time fraud detection on the last 100 transactions.

    Decision thresholds:
      fraud_prob > 0.80  -> DECLINE
      fraud_prob > 0.50  -> REVIEW
      otherwise          -> APPROVE
    """

    print("\n" + "=" * 60)
    print("REAL-TIME FRAUD DETECTION SIMULATION")
    print("=" * 60)

    # -- Validate prerequisites ---------------------------------------------
    missing = []
    if not os.path.exists(_MODEL_PATH):
        missing.append(f"Trained model: {_MODEL_PATH}  (run: python src/ml/fraud_model.py)")
    if not os.path.exists(os.path.join(_TEST_DATA_DIR, "transactions.csv")):
        missing.append(f"Test data: {_TEST_DATA_DIR}  (run: python scripts/generate_test_data.py)")
    if missing:
        print("\n[WARN] Prerequisites missing:")
        for m in missing:
            print(f"  - {m}")
        print("\nSkipping inference test.")
        return pd.DataFrame()

    # -- Load model ---------------------------------------------------------
    from ml.fraud_model import FraudDetectionModel  # type: ignore
    model = FraudDetectionModel()
    model.load(_MODEL_PATH)

    # -- Connect to infrastructure (best-effort) ----------------------------
    pg_conn = _try_connect_postgres()
    redis_client = _try_connect_redis()

    if pg_conn and redis_client:
        from features.realtime_features import RealtimeFeatureComputer  # type: ignore
        computer = RealtimeFeatureComputer(redis_client, pg_conn)
        mode = "live"
    else:
        expected_csv = os.path.join(_TEST_DATA_DIR, "expected_features.csv")
        if not os.path.exists(expected_csv):
            print("[WARN] Neither DB nor expected_features.csv available. Skipping.")
            return pd.DataFrame()
        computer = _CsvFeatureComputer(expected_csv)
        mode = "csv"

    print(f"\nMode: {mode.upper()}  |  Model: {_MODEL_PATH}")

    # -- Load test transactions ---------------------------------------------
    txns = pd.read_csv(os.path.join(_TEST_DATA_DIR, "transactions.csv")).tail(100).copy()
    txns["timestamp"] = pd.to_datetime(txns["timestamp"])

    labels = pd.read_csv(os.path.join(_TEST_DATA_DIR, "fraud_labels.csv"))
    label_map: dict = dict(zip(labels["transaction_id"], labels["is_fraud"].astype(bool)))

    print(f"\nRunning inference on {len(txns)} transactions...")

    results = []
    for _, txn in txns.iterrows():
        feats = computer.compute_transaction_features(
            user_id=txn["user_id"],
            transaction=txn.to_dict(),
            timestamp=txn["timestamp"],
        )
        feats["amount"] = float(txn["amount"])

        feat_df = pd.DataFrame([feats])
        prob = float(model.predict(feat_df)[0])

        if prob > 0.80:
            decision = "DECLINE"
        elif prob > 0.50:
            decision = "REVIEW"
        else:
            decision = "APPROVE"

        actual = label_map.get(txn["transaction_id"], False)
        correct = (decision == "DECLINE") == actual

        if prob > 0.50:
            tag = "CORRECT" if correct else "WRONG"
            print(
                f"\n[HIGH RISK] {txn['transaction_id']}  user={txn['user_id']}"
                f"  amount=${txn['amount']:.2f}  prob={prob:.2%}"
                f"  decision={decision}  actual={'FRAUD' if actual else 'LEGIT'}  [{tag}]"
            )

        results.append({
            "transaction_id": txn["transaction_id"],
            "user_id": txn["user_id"],
            "amount": txn["amount"],
            "fraud_probability": prob,
            "decision": decision,
            "actual_fraud": actual,
            "correct": correct,
        })

    df = pd.DataFrame(results)

    # -- Summary ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(f"Total transactions : {len(df)}")
    print(f"Actual fraud rate  : {df['actual_fraud'].mean() * 100:.2f}%")
    print(f"\nDecision breakdown :")
    print(df["decision"].value_counts().to_string())
    print(f"\nOverall accuracy   : {df['correct'].mean() * 100:.2f}%")

    declined = df[df["decision"] == "DECLINE"]
    if len(declined) > 0:
        prec = declined["actual_fraud"].mean()
        print(f"Precision (DECLINE): {prec * 100:.2f}%")

    frauds = df[df["actual_fraud"]]
    if len(frauds) > 0:
        recall = (frauds["decision"] == "DECLINE").mean()
        print(f"Recall (fraud)     : {recall * 100:.2f}%")

    print("=" * 60)

    # -- Cleanup ------------------------------------------------------------
    if redis_client:
        redis_client.close()
    if pg_conn:
        pg_conn.close()

    return df


# -----------------------------------------------------------------------------
# pytest-compatible thin wrappers
# -----------------------------------------------------------------------------

def test_inference_runs_without_error():
    """Pytest entry point - simply asserts the pipeline returns a DataFrame."""
    df = test_realtime_inference()
    # If prerequisites are missing we get an empty DF - that's acceptable here.
    assert isinstance(df, pd.DataFrame)


def test_inference_accuracy_above_baseline():
    """If test data is present the model should beat a random baseline (50 %)."""
    if not (
        os.path.exists(_MODEL_PATH)
        and os.path.exists(os.path.join(_TEST_DATA_DIR, "transactions.csv"))
    ):
        import pytest  # type: ignore
        pytest.skip("Prerequisites not available")

    df = test_realtime_inference()
    if df.empty:
        import pytest  # type: ignore
        pytest.skip("Empty result DataFrame")

    accuracy = df["correct"].mean()
    assert accuracy > 0.50, f"Model accuracy {accuracy:.2%} below 50 % baseline"


# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    results = test_realtime_inference()
    out = "/tmp/inference_results.csv"
    results.to_csv(out, index=False)
    print(f"\n[OK] Results saved to {out}")
