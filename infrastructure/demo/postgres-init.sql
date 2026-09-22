-- Runs once, automatically, on first start of the ISOLATED demo Postgres
-- container (aidp-demo-postgres), mounted into /docker-entrypoint-initdb.d/.
--
-- Deliberately DIFFERENT from infrastructure/postgres/init.sql: that file
-- creates the shared POC's `aidp_test` database. This one never does. The
-- demo cluster must contain exactly two databases beyond the built-ins:
--
--   aidp_demo  - the demo database (created by the entrypoint itself from
--                POSTGRES_DB; see .env.demo.example)
--   mlflow     - the ISOLATED MLflow backend store for this stack only
--
-- The absence of `aidp` and `aidp_test` from this cluster is what the demo
-- script asserts as structural proof that neither can be contacted:
-- scripts/demo_debit_card_full_lifecycle.sh, stage A.
--
-- The shared, unmodified schema is applied to POSTGRES_DB (aidp_demo).
-- infrastructure/postgres/lib/ is mounted read-only at
-- /docker-entrypoint-initdb.d/lib/ (a directory, so the Postgres entrypoint
-- ignores it for direct execution -- the same pattern the shared stack uses).

CREATE DATABASE mlflow;

\i /docker-entrypoint-initdb.d/lib/schema.sql
