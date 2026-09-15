#!/usr/bin/env bash
# Trains the fraud-detection model on data/output/features/transactions.parquet
# and registers it as fraud-detection-model in MLflow. Requires the batch
# pipeline to have been run first (python -m src.processing.pipeline).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -d .venv ]; then
  source .venv/bin/activate
fi

python -m src.ml.train "$@"
