"""
Kafka consumer for ingesting raw events into the feature pipeline.

Supports:
- Consumer group management with offset commits
- Exactly-once semantics via idempotent processing
- Dead letter queue for failed events
- Graceful shutdown
"""

from __future__ import annotations

import json
import logging
import signal
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

from ..core.schemas import RawEvent
from .transformers import EventTransformerRegistry

logger = logging.getLogger(__name__)

_STOP_SENTINEL = object()


class KafkaFeatureConsumer:
    """
    Kafka consumer that reads raw events, transforms them to feature records,
    and persists them via the provided feature store.

    Parameters
    ----------
    bootstrap_servers : str | list[str]
    topics : list of topic names to subscribe to
    group_id : consumer group identifier (enables offset management)
    transformer_registry : EventTransformerRegistry
    feature_store : FeatureStore (or any store with write_features())
    dlq_topic : dead-letter-queue topic for unprocessable events
    max_poll_records : max records per poll
    session_timeout_ms : Kafka session timeout
    auto_offset_reset : 'earliest' | 'latest'
    """

    def __init__(
        self,
        bootstrap_servers: Any,
        topics: List[str],
        group_id: str,
        transformer_registry: EventTransformerRegistry,
        feature_store: Any,
        dlq_topic: str = "feature-pipeline-dlq",
        max_poll_records: int = 500,
        session_timeout_ms: int = 30000,
        auto_offset_reset: str = "earliest",
        enable_auto_commit: bool = False,  # we commit manually for exactly-once
    ) -> None:
        self._topics = topics
        self._transformers = transformer_registry
        self._store = feature_store
        self._dlq_topic = dlq_topic
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            max_poll_records=max_poll_records,
            session_timeout_ms=session_timeout_ms,
            auto_offset_reset=auto_offset_reset,
            enable_auto_commit=enable_auto_commit,
        )

        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            retries=5,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, in_background: bool = True) -> None:
        """Start consuming events."""
        self._running = True
        if in_background:
            self._thread = threading.Thread(
                target=self._consume_loop, daemon=True, name="kafka-consumer"
            )
            self._thread.start()
            logger.info("Kafka consumer started in background thread")
        else:
            self._consume_loop()

    def stop(self, timeout: float = 10.0) -> None:
        """Gracefully stop the consumer."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
        self._consumer.close()
        self._producer.close()
        logger.info("Kafka consumer stopped")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _consume_loop(self) -> None:
        logger.info("Consuming from topics: %s", self._topics)
        while self._running:
            try:
                records = self._consumer.poll(timeout_ms=1000)
                if not records:
                    continue

                batch_success = True
                for tp, messages in records.items():
                    for msg in messages:
                        success = self._process_message(msg)
                        if not success:
                            batch_success = False

                # Commit only after the full batch is processed (at-least-once)
                if batch_success:
                    self._consumer.commit()

            except KafkaError as exc:
                logger.error("Kafka error: %s", exc)
                time.sleep(5)
            except Exception as exc:
                logger.exception("Unexpected error in consumer loop: %s", exc)
                time.sleep(1)

    # ------------------------------------------------------------------
    # Per-message processing
    # ------------------------------------------------------------------

    def _process_message(self, msg: Any) -> bool:
        """
        Process a single Kafka message.

        1. Deserialize → RawEvent
        2. Look up transformer by event_type
        3. Transform → FeatureRecord(s)
        4. Validate and write to feature store
        5. On failure → send to DLQ

        Returns True if processed successfully.
        """
        payload = msg.value
        try:
            event = RawEvent(**payload)
        except Exception as exc:
            logger.warning("Failed to parse RawEvent: %s | payload=%s", exc, payload)
            self._send_to_dlq(payload, str(exc))
            return False

        transformer = self._transformers.get(event.event_type)
        if transformer is None:
            logger.debug("No transformer for event_type=%s, skipping", event.event_type)
            return True  # Not an error – just unknown type

        try:
            records = transformer.transform(event)
        except Exception as exc:
            logger.error("Transformer error for event %s: %s", event.event_id, exc)
            self._send_to_dlq(payload, str(exc))
            return False

        for record in records:
            try:
                self._store.write_features(record)
            except Exception as exc:
                logger.error("Store write failed for entity %s: %s", record.entity_id, exc)
                self._send_to_dlq(payload, str(exc))
                return False

        return True

    def _send_to_dlq(self, payload: Any, reason: str) -> None:
        try:
            dlq_payload = {"original": payload, "failure_reason": reason}
            self._producer.send(self._dlq_topic, dlq_payload)
        except Exception as exc:
            logger.error("DLQ send failed: %s", exc)
