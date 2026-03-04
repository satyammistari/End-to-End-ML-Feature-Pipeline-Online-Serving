#!/usr/bin/env bash
# setup_kafka.sh – Create required Kafka topics for the feature pipeline
set -euo pipefail

KAFKA_BOOTSTRAP="${KAFKA_BOOTSTRAP:-localhost:9092}"
REPLICATION="${REPLICATION_FACTOR:-1}"
PARTITIONS="${PARTITIONS:-4}"

TOPICS=(
  "user-events"
  "transaction-events"
  "user-login-events"
  "feature-pipeline-dlq"
)

echo "Creating Kafka topics on ${KAFKA_BOOTSTRAP}..."
for topic in "${TOPICS[@]}"; do
  kafka-topics.sh \
    --bootstrap-server "${KAFKA_BOOTSTRAP}" \
    --create \
    --if-not-exists \
    --topic "${topic}" \
    --partitions "${PARTITIONS}" \
    --replication-factor "${REPLICATION}"
  echo "  ✓ ${topic}"
done

echo ""
echo "Topic list:"
kafka-topics.sh --bootstrap-server "${KAFKA_BOOTSTRAP}" --list
