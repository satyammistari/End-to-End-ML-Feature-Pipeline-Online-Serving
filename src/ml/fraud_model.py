"""
Fraud Detection Model - trains on computed features and evaluates on held-out data.

Training pipeline:
  1. Load fraud labels + raw transactions from PostgreSQL
  2. Join with precomputed features from the feature store
  3. Temporal train/test split (80/20)
  4. Train a RandomForestClassifier
  5. Report classification metrics + feature importance

Usage (standalone):
  python src/ml/fraud_model.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# --- Feature columns used by the model ---------------------------------------

FEATURE_COLS: List[str] = [
    "transaction_count_24h",
    "transaction_count_7d",
    "transaction_count_30d",
    "total_amount_24h",
    "avg_amount_7d",
    "max_amount_30d",
    "unique_merchants_7d",
    "unique_categories_30d",
    "transactions_last_hour",
    "amount_zscore",
    "amount",  # current transaction amount
]


# -----------------------------------------------------------------------------

class FraudDetectionModel:
    """
    Wraps a RandomForestClassifier trained to detect fraudulent transactions.

    Attributes
    ----------
    model : RandomForestClassifier | None
    feature_names : list[str] | None
    """

    def __init__(self) -> None:
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names: Optional[List[str]] = None

    # -- Data preparation ------------------------------------------------------

    def prepare_training_data(self, postgres_conn: Any) -> pd.DataFrame:
        """
        Join fraud labels with transactions and precomputed features.

        Returns a DataFrame with one row per transaction that has both a label
        and at least one computed feature.
        """
        print("[DATA] Preparing training data...")

        labels_df = pd.read_sql(
            """
            SELECT transaction_id, user_id, timestamp, is_fraud
            FROM fraud_labels
            ORDER BY timestamp
            """,
            postgres_conn,
        )

        txn_df = pd.read_sql(
            """
            SELECT transaction_id, amount, merchant_category, device_type, is_international
            FROM transactions
            """,
            postgres_conn,
        )

        features_df = pd.read_sql(
            """
            SELECT entity_id, feature_name, feature_value, timestamp
            FROM features
            """,
            postgres_conn,
        )

        # Pivot feature rows into columns
        features_df["feature_value"] = features_df["feature_value"].apply(
            lambda v: json.loads(v) if isinstance(v, str) else v
        )
        features_pivot = (
            features_df.pivot_table(
                index=["entity_id", "timestamp"],
                columns="feature_name",
                values="feature_value",
                aggfunc="first",
            )
            .reset_index()
        )
        features_pivot.columns.name = None

        # Join labels -> transactions -> features
        dataset = labels_df.merge(txn_df, on="transaction_id", how="inner")
        dataset = dataset.merge(
            features_pivot,
            left_on=["user_id", "timestamp"],
            right_on=["entity_id", "timestamp"],
            how="inner",
        )

        print(f"[OK] Prepared dataset: {len(dataset):,} samples")
        print(f"   Fraud rate: {dataset['is_fraud'].mean() * 100:.2f}%")
        return dataset

    # -- Training --------------------------------------------------------------

    def train(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the fraud detection model and return evaluation metrics.

        Uses a strict temporal split to avoid future leakage:
        first 80 % of rows (sorted by timestamp) -> train,
        last 20 % -> test.
        """
        print("\n[TRAIN] Training fraud detection model...")

        # Ensure all feature columns exist
        for col in FEATURE_COLS:
            if col not in dataset.columns:
                dataset[col] = 0.0

        X = dataset[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = dataset["is_fraud"].astype(int)

        self.feature_names = FEATURE_COLS

        split = int(len(dataset) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(y_test, y_proba) if len(y_test.unique()) > 1 else float("nan")
        cm = confusion_matrix(y_test, y_pred)

        print("\n[RESULTS] MODEL PERFORMANCE:")
        print("=" * 60)
        print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))
        print(f"ROC-AUC: {roc:.4f}")
        print("\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"                Legit  Fraud")
        print(f"Actual  Legit  {cm[0][0]:6d} {cm[0][1]:6d}")
        print(f"        Fraud  {cm[1][0]:6d} {cm[1][1]:6d}")

        importances = pd.DataFrame(
            {"feature": FEATURE_COLS, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print("\nFeature Importances:")
        print(importances.to_string(index=False))

        return {"roc_auc": roc, "confusion_matrix": cm, "feature_importance": importances}

    # -- Inference -------------------------------------------------------------

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return fraud probability for each row in *features*."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        X = features.reindex(columns=self.feature_names, fill_value=0)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        return self.model.predict_proba(X)[:, 1]

    # -- Persistence -----------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize model + metadata to *path*."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)
        print(f"[OK] Model saved to {path}")

    def load(self, path: str) -> None:
        """Deserialize model + metadata from *path*."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        print(f"[OK] Model loaded from {path}")


# -----------------------------------------------------------------------------
# Standalone training script
# -----------------------------------------------------------------------------

def _load_labels_into_db(conn: Any) -> None:
    """Insert fraud labels from CSV into the database (idempotent)."""
    labels_path = "/tmp/test_data/fraud_labels.csv"
    if not os.path.exists(labels_path):
        print(f"[EMOJI]  Labels file not found: {labels_path}")
        return

    labels_df = pd.read_csv(labels_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS fraud_labels (
            transaction_id VARCHAR(255) PRIMARY KEY,
            user_id        VARCHAR(255),
            timestamp      TIMESTAMPTZ,
            is_fraud       BOOLEAN,
            fraud_type     VARCHAR(100)
        )
        """
    )
    conn.commit()

    for _, row in labels_df.iterrows():
        cursor.execute(
            """
            INSERT INTO fraud_labels (transaction_id, user_id, timestamp, is_fraud, fraud_type)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (transaction_id) DO NOTHING
            """,
            (
                row["transaction_id"],
                row["user_id"],
                row["timestamp"],
                bool(row["is_fraud"]),
                row["fraud_type"] if pd.notna(row["fraud_type"]) else None,
            ),
        )
    conn.commit()
    cursor.close()
    print(f"[OK] Loaded {len(labels_df):,} fraud labels into database")


def train_fraud_model() -> tuple["FraudDetectionModel", Dict[str, Any]]:
    """Complete standalone training pipeline."""
    import psycopg2  # type: ignore

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        database=os.getenv("POSTGRES_DB", "features_test"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
    )

    _load_labels_into_db(conn)

    model = FraudDetectionModel()
    dataset = model.prepare_training_data(conn)
    results = model.train(dataset)
    model.save("/tmp/fraud_model.pkl")

    conn.close()
    return model, results


if __name__ == "__main__":
    model, results = train_fraud_model()
    print(f"\nFinal ROC-AUC: {results['roc_auc']:.4f}")
