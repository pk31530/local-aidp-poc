-- Runs once, automatically, on first container start (mounted into
-- /docker-entrypoint-initdb.d/). Creates the extra databases this POC needs
-- beyond the default POSTGRES_DB (aidp):
--   mlflow     - MLflow's backend store (fix C5: persisted via the postgres
--                data volume, so the model registry survives a restart)
--   aidp_test  - isolated integration-test database (fix H4)
-- then applies the shared schema to both the demo db and the test db.

CREATE DATABASE mlflow;
CREATE DATABASE aidp_test;

\i /docker-entrypoint-initdb.d/lib/schema.sql

\connect aidp_test
\i /docker-entrypoint-initdb.d/lib/schema.sql
