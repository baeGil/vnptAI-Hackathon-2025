#!/bin/bash
# Inference script for VNPT AI Hackathon
# Automatically handles Qdrant and Inference

PROJECT_ROOT="/code"
cd "$PROJECT_ROOT"

# Ensure output directory exists
mkdir -p output

# 1. Start Qdrant in background
echo "[INFO] Starting Qdrant..."
# Use environment variable to specify exact storage path
QDRANT__STORAGE__STORAGE_PATH=/code/data/qdrant_storage qdrant > output/qdrant.log 2>&1 &
QDRANT_PID=$!

# 2. Wait for Qdrant to be ready (Health Check)
echo "[INFO] Waiting for Qdrant to initialize..."
MAX_RETRIES=30
RETRY_COUNT=0

until curl -sf http://localhost:6333/readyz >/dev/null || [ $RETRY_COUNT -eq $MAX_RETRIES ]; do
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT+1))
    echo "[INFO] Retrying health check ($RETRY_COUNT/$MAX_RETRIES)..."
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "[ERROR] Qdrant failed to start. Logs:"
    cat output/qdrant.log
    kill $QDRANT_PID
    exit 1
fi

echo "[INFO] Qdrant is ready. Starting inference..."

# 3. Run Inference
uv run python3 predict.py --auto

# 4. Cleanup
echo "[INFO] Inference finished. Cleaning up..."
kill $QDRANT_PID
wait $QDRANT_PID 2>/dev/null

echo "[DONE] Output saved in output/submission.csv"