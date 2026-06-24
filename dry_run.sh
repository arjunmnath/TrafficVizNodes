#!/bin/bash

echo "====================================="
echo " Starting CCTV System Dry Run"
echo "====================================="

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Default ReID model (can be overridden via env vars)
REID_MODEL_NAME="${REID_MODEL_NAME:-resnet101_ibn_a}"
REID_MODEL_PATH="${REID_MODEL_PATH:-agent-working/trained_models/101a_384/v1/resnet101_ibn_a_2.pth}"

echo "[1/2] Starting Backend Services (Tracker + ReID Server)..."
echo "  ReID Model: ${REID_MODEL_NAME}"
echo "  Weights:    ${REID_MODEL_PATH}"
uv run python run_all.py \
    --videos dataset/test/S06/c041/vdo.avi dataset/test/S06/c042/vdo.avi \
    --reid_model_name "${REID_MODEL_NAME}" \
    --reid_model_path "${REID_MODEL_PATH}" &
BACKEND_PID=$!

echo "[2/2] Starting Next.js Dashboard UI..."
cd dashboard
npm run dev &
FRONTEND_PID=$!

echo ""
echo "====================================="
echo " System is LIVE! "
echo " Dashboard: http://localhost:3000"
echo "====================================="
echo "Press Ctrl+C to stop the simulation."

# Cleanup handler
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "Dry run stopped successfully."
    exit 0
}

# Trap Ctrl+C and call cleanup
trap cleanup SIGINT SIGTERM

# Wait indefinitely until interrupted
wait
