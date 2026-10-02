#!/usr/bin/env bash
# Full pipeline: synthetic data -> trained model -> web app
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"

echo "==> [1/3] Generating synthetic sensor data"
python -m src.generate_data

echo "==> [2/3] Training the failure-risk model"
python -m src.train

echo "==> [3/3] Starting Streamlit on port 8501"
exec streamlit run app/streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true
