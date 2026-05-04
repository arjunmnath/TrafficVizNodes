#!/bin/bash

echo "====================================="
echo " Starting CCTV System Dry Run"
echo "====================================="

# Ensure we're in the right directory
cd "$(dirname "$0")"

echo "[1/2] Starting Backend Services (Tracker + ReID Server)..."
uv run python run_all.py --videos dataset/test/S06/c041/vdo.avi dataset/test/S06/c042/vdo.avi &
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
