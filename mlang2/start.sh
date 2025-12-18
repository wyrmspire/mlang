#!/bin/bash
# start.sh - Start mlang2 backend and frontend servers
# Usage: ./start.sh

set -e

echo "=============================================="
echo "  MLang2 - Starting Services"
echo "=============================================="

# Load environment variables
if [ -f src/.env.local ]; then
    echo "[1] Loading src/.env.local..."
    export $(cat src/.env.local | grep -v '^#' | xargs)
elif [ -f .env.local ]; then
    echo "[1] Loading .env.local..."
    export $(cat .env.local | grep -v '^#' | xargs)
else
    echo "[!] Warning: No .env.local found. GEMINI_API_KEY may not be set."
fi

# Kill existing processes
echo "[2] Stopping any existing servers..."
pkill -f "uvicorn src.server.main" 2>/dev/null || true
pkill -f "vite" 2>/dev/null || true
sleep 1

# Start backend
echo "[3] Starting FastAPI backend on port 8000..."
python -m uvicorn src.server.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Wait for backend to be ready
echo "    Waiting for backend..."
sleep 3

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "    Backend is ready!"
else
    echo "[!] Warning: Backend may not be ready yet."
fi

# Start frontend
echo "[4] Starting Vite frontend on port 5173..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=============================================="
echo "  Services Started!"
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo "=============================================="
echo ""
echo "Press Ctrl+C to stop all services."
echo ""

# Wait for Ctrl+C
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" INT
wait
