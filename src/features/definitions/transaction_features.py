"""
Concrete transaction feature definitions.
"""

from ...core.schemas import (
    FeatureComputationType,
    FeatureDefinition,
    FeatureGroup,
    FeatureGroupVersion,
    FeatureType,
    ValidationRule,
)

TRANSACTION_FEATURE_DEFINITIONS = [
    FeatureDefinition(
        name="transaction_amount",
        feature_type=FeatureType.FLOAT,
        computation_type=FeatureComputationType.REALTIME,
        description="Raw transaction amount in the original currency",
        owner="payments-team",
        tags=["transaction", "raw"],
        validation_rules=ValidationRule(min_value=0.0, max_value=1_000_000.0),
    ),
    FeatureDefinition(
        name="amount_zscore",
        feature_type=FeatureType.FLOAT,
        computation_type=FeatureComputationType.ON_DEMAND,
        description="Z-score of transaction amount vs user's 7d distribution",
        owner="data-team",
        tags=["transaction", "fraud", "on_demand"],
        depends_on=["user_features.avg_transaction_amount_7d",
                    "user_features.std_transaction_amount_7d"],
    ),
    FeatureDefinition(
        name="transaction_currency",
        feature_type=FeatureType.STRING,
        computation_type=FeatureComputationType.REALTIME,
        description="ISO-4217 currency code",
        owner="payments-team",
        tags=["transaction", "raw"],
        validation_rules=ValidationRule(regex_pattern=r"^[A-Z]{3}$"),
    ),
    FeatureDefinition(
        name="merchant_id",
        feature_type=FeatureType.STRING,
        computation_type=FeatureComputationType.REALTIME,
        description="Unique merchant identifier",
        owner="payments-team",
        tags=["transaction", "merchant"],
    ),
    FeatureDefinition(
        name="merchant_category",
        feature_type=FeatureType.STRING,
        computation_type=FeatureComputationType.REALTIME,
        description="Merchant Category Code (MCC) description",
        owner="payments-team",
        tags=["transaction", "merchant"],
    ),
    FeatureDefinition(
        name="is_online",
        feature_type=FeatureType.BOOLEAN,
        computation_type=FeatureComputationType.REALTIME,
        description="Whether the transaction was made online",
        owner="payments-team",
        tags=["transaction", "channel"],
    ),
    FeatureDefinition(
        name="is_new_merchant",
        feature_type=FeatureType.BOOLEAN,
        computation_type=FeatureComputationType.ON_DEMAND,
        description="True if user has never transacted at this merchant before",
        owner="data-team",
        tags=["transaction", "merchant", "fraud"],
        depends_on=["user_features.known_merchant_ids"],
    ),
    FeatureDefinition(
        name="merchant_category_frequency",
        feature_type=FeatureType.FLOAT,
        computation_type=FeatureComputationType.BATCH,
        description="Fraction of user's 30d transactions in this merchant category",
        owner="data-team",
        tags=["transaction", "merchant", "batch"],
        ttl_seconds=86400,
        validation_rules=ValidationRule(min_value=0.0, max_value=1.0),
    ),
    FeatureDefinition(
        name="is_high_risk_time",
        feature_type=FeatureType.BOOLEAN,
        computation_type=FeatureComputationType.ON_DEMAND,
        description="True if transaction hour is outside user's typical hours",
        owner="data-team",
        tags=["transaction", "temporal", "fraud"],
        depends_on=["user_features.typical_active_hours"],
    ),
]

TRANSACTION_FEATURE_GROUP = FeatureGroup(
    name="transaction_features",
    entity_type="transaction",
    description="Per-transaction raw and derived features",
    owner="payments-team",
    tags=["transaction"],
    versions={
        "v1": FeatureGroupVersion(
            version="v1",
            features=TRANSACTION_FEATURE_DEFINITIONS,
        )
    },
    latest_version="v1",
)
