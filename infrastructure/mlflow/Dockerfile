FROM python:3.11-slim

RUN pip install --no-cache-dir \
    mlflow==2.17.2 \
    psycopg2-binary==2.9.10 \
    boto3==1.35.54

EXPOSE 5000

# Backend store (Postgres) and artifact store (MinIO/S3) are passed in via
# environment variables at `docker compose up` time — see docker-compose.yml.
# Backend store + artifact store both persist outside the container (fix C5):
# Postgres data lives on the postgres_data volume, MinIO objects on minio_data.
ENTRYPOINT ["sh", "-c", "mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" \
  --default-artifact-root \"$MLFLOW_DEFAULT_ARTIFACT_ROOT\" \
  --serve-artifacts"]
