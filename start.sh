#!/bin/bash

echo "Starting Application Reset..."

# 1. Kill existing processes
echo "Killing existing instances..."
# Windows TaskKill (try/pass)
taskkill //F //IM uvicorn.exe > /dev/null 2>&1
taskkill //F //IM python.exe > /dev/null 2>&1
taskkill //F //IM node.exe > /dev/null 2>&1

# Give them a moment to die
sleep 2

# 2. Start Backend
echo "Starting Backend..."
# Run in background, output to logs
nohup python -m uvicorn src.api:app --reload --port 8000 > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# 3. Start Frontend
echo "Starting Frontend..."
cd frontend
nohup npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo "Frontend started (PID: $FRONTEND_PID)"

echo "Application restarted successfully!"
echo "Backend Logs: logs/backend.log"
echo "Frontend Logs: logs/frontend.log"
echo ""
echo "Access the App here: http://localhost:5173"
