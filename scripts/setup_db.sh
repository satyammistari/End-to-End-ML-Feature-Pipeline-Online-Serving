#!/usr/bin/env bash
# setup_db.sh – Initialise PostgreSQL for the feature pipeline
# Runs automatically as a Docker entrypoint init script.
set -euo pipefail

PGUSER="${POSTGRES_USER:-feature_user}"
PGDB="${POSTGRES_DB:-features}"

psql -v ON_ERROR_STOP=1 --username "$PGUSER" --dbname "$PGDB" <<-EOSQL
  -- Enable TimescaleDB extension (no-op if not installed)
  CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

  -- Feature values hypertable
  CREATE TABLE IF NOT EXISTS feature_values (
      id              BIGSERIAL,
      entity_id       TEXT          NOT NULL,
      entity_type     TEXT          NOT NULL,
      feature_group   TEXT          NOT NULL,
      feature_version TEXT          NOT NULL,
      feature_name    TEXT          NOT NULL,
      feature_value   JSONB         NOT NULL,
      event_time      TIMESTAMPTZ   NOT NULL,
      computed_at     TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
      PRIMARY KEY (id, event_time)
  );

  SELECT create_hypertable('feature_values', 'event_time', if_not_exists => TRUE);

  CREATE INDEX IF NOT EXISTS feature_values_entity_time_idx
      ON feature_values (entity_id, feature_name, event_time DESC);

  -- Feature registry table
  CREATE TABLE IF NOT EXISTS feature_registry (
      name        TEXT PRIMARY KEY,
      data        JSONB NOT NULL,
      updated_at  TIMESTAMPTZ DEFAULT NOW()
  );

  GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO ${PGUSER};
  GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO ${PGUSER};
EOSQL

echo "Database initialisation complete."
