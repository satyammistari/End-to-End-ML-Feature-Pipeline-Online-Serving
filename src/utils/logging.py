"""
Structured JSON logging configuration for the ML Feature Pipeline.

Features:
- JSON log format for ingestion by ELK / Datadog / Loki
- Correlation ID support (request tracing)
- Automatic service and environment labels
- Log level set via LOG_LEVEL env var
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger  # type: ignore[import]

# ---------------------------------------------------------------------------
# Correlation ID context variable
# ---------------------------------------------------------------------------

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    return _correlation_id.get()


def set_correlation_id(cid: Optional[str] = None) -> str:
    cid = cid or str(uuid.uuid4())
    _correlation_id.set(cid)
    return cid


# ---------------------------------------------------------------------------
# Custom JSON formatter
# ---------------------------------------------------------------------------

class PipelineJsonFormatter(jsonlogger.JsonFormatter):
    """
    Adds standard fields to every log record:
    - service, environment, version
    - correlation_id
    """

    SERVICE = os.getenv("SERVICE_NAME", "feature-pipeline")
    ENV = os.getenv("ENV", "local")
    VERSION = os.getenv("APP_VERSION", "1.0.0")

    def add_fields(
        self,
        log_record: Dict[str, Any],
        record: logging.LogRecord,
        message_dict: Dict[str, Any],
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record["service"] = self.SERVICE
        log_record["env"] = self.ENV
        log_record["version"] = self.VERSION
        log_record["correlation_id"] = get_correlation_id()
        log_record["logger"] = record.name
        log_record["level"] = record.levelname


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def configure_logging(
    level: Optional[str] = None,
    json_format: bool = True,
) -> None:
    """
    Configure root logger.

    Call once at application startup.

    Parameters
    ----------
    level : log level string (default: LOG_LEVEL env var or INFO)
    json_format : emit JSON (True) or plain text (False, useful in dev)
    """
    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove default handlers
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)

    if json_format:
        fmt = PipelineJsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )
    else:
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)-32s  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    handler.setFormatter(fmt)
    root_logger.addHandler(handler)

    # Silence noisy third-party loggers
    for noisy in ("kafka", "urllib3", "grpc", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug("Logging configured (level=%s, json=%s)", log_level, json_format)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Always use this instead of logging.getLogger() directly."""
    return logging.getLogger(name)
