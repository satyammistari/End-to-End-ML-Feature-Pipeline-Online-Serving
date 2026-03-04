"""
Backfill pipeline: recompute features for a historical date range.

Design goals:
- Idempotent: safe to re-run without duplicates
- Resumable: checkpoints progress to survive failures
- Efficient: processes in date-partitioned chunks
- Isolated: writes only to the offline store
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..features.batch_features import UserAggregateBatchJob

logger = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT_DIR = Path("/tmp/feature_pipeline_checkpoints")


class BackfillPipeline:
    """
    Orchestrates historical feature backfills.

    Parameters
    ----------
    offline_store : PostgresFeatureStore
    jobs : list of BatchFeatureJob subclasses to run
    checkpoint_dir : directory to store progress (one file per job+date_range)
    batch_days : number of days processed per chunk (default 1)
    """

    def __init__(
        self,
        offline_store: Any,
        jobs: Optional[List[Any]] = None,
        checkpoint_dir: Optional[Path] = None,
        batch_days: int = 1,
    ) -> None:
        self._offline = offline_store
        self._batch_days = batch_days
        self._checkpoint_dir = checkpoint_dir or _DEFAULT_CHECKPOINT_DIR
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Default to standard user aggregation job
        self._jobs: List[Any] = jobs or [UserAggregateBatchJob(offline_store=offline_store)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def backfill_features(
        self,
        feature_names: Optional[List[str]],
        start_date: datetime,
        end_date: datetime,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Recompute features for all configured jobs in [start_date, end_date].

        Parameters
        ----------
        feature_names : optional filter (None = all features)
        start_date : inclusive start
        end_date : inclusive end
        resume : if True, skip already-processed date ranges using checkpoints

        Returns a summary dict with counts and timings.
        """
        logger.info(
            "Starting backfill %s → %s  (resume=%s)",
            start_date.date(),
            end_date.date(),
            resume,
        )
        summary: Dict[str, Any] = {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "jobs": [],
        }

        for job in self._jobs:
            job_summary = self._run_job(job, start_date, end_date, resume)
            summary["jobs"].append(job_summary)

        return summary

    def validate_backfill(
        self,
        feature_group: str,
        entity_ids: List[str],
        expected_start: datetime,
        expected_end: datetime,
    ) -> Dict[str, Any]:
        """
        Validate that backfilled data is present and within expected range.

        Returns a report dict.
        """
        data = self._offline.read_training_data(
            entity_ids=entity_ids,
            feature_names=[],  # all features
            start_time=expected_start,
            end_time=expected_end,
            feature_group=feature_group,
        )
        entity_coverage: Dict[str, int] = {}
        for row in data:
            eid = row["entity_id"]
            entity_coverage[eid] = entity_coverage.get(eid, 0) + 1

        missing = [eid for eid in entity_ids if entity_coverage.get(eid, 0) == 0]
        return {
            "feature_group": feature_group,
            "total_rows": len(data),
            "entity_coverage": entity_coverage,
            "missing_entities": missing,
            "coverage_pct": (len(entity_ids) - len(missing)) / max(len(entity_ids), 1) * 100,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_job(
        self,
        job: Any,
        start_date: datetime,
        end_date: datetime,
        resume: bool,
    ) -> Dict[str, Any]:
        job_name = job.feature_group
        checkpoint_key = f"{job_name}_{start_date.date()}_{end_date.date()}"
        checkpoint_path = self._checkpoint_dir / f"{checkpoint_key}.json"

        processed_dates: set = set()
        if resume and checkpoint_path.exists():
            try:
                with open(checkpoint_path) as f:
                    processed_dates = set(json.load(f).get("processed", []))
                logger.info("[%s] Resuming – %d dates already done", job_name, len(processed_dates))
            except Exception as exc:
                logger.warning("Failed to read checkpoint: %s", exc)

        current = start_date
        total_records = 0
        failed_dates = []
        wall_start = time.monotonic()

        while current <= end_date:
            day_key = current.date().isoformat()
            next_day = current + timedelta(days=self._batch_days)

            if day_key in processed_dates:
                current = next_day
                continue

            try:
                records = job.compute(current, next_day)
                if records:
                    self._offline.write_features_batch(records)
                    total_records += len(records)

                processed_dates.add(day_key)
                self._save_checkpoint(checkpoint_path, list(processed_dates))
                logger.info("[%s] Processed %s (%d records)", job_name, day_key, len(records))

            except Exception as exc:
                logger.error("[%s] Failed for %s: %s", job_name, day_key, exc)
                failed_dates.append(day_key)

            current = next_day

        elapsed = time.monotonic() - wall_start
        return {
            "job": job_name,
            "total_records": total_records,
            "failed_dates": failed_dates,
            "elapsed_seconds": elapsed,
        }

    @staticmethod
    def _save_checkpoint(path: Path, processed: List[str]) -> None:
        try:
            with open(path, "w") as f:
                json.dump({"processed": processed}, f)
        except Exception as exc:
            logger.warning("Failed to save checkpoint: %s", exc)
