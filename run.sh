#!/bin/bash

# ==============================================================================
# VISION-REWARD-OS LAUNCHER
# Description: Orchestrates the startup of both FastAPI backend and Gradio UI.
# Automatically handles graceful shutdown of all processes when Ctrl+C is pressed.
# ==============================================================================

echo "============================================================"
echo "🚀 INITIATING VISION-REWARD-OS"
echo "============================================================"

# Function to cleanly terminate background processes on exit
cleanup() {
    echo ""
    echo "[SYSTEM] Intercepted termination signal (Ctrl+C)."
    echo "[SYSTEM] Shutting down FastAPI (PID: $FASTAPI_PID)..."
    kill $FASTAPI_PID 2>/dev/null
    echo "[SYSTEM] Shutting down Gradio UI (PID: $GRADIO_PID)..."
    kill $GRADIO_PID 2>/dev/null
    echo "[SYSTEM] All processes terminated gracefully. Goodbye!"
    exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM to trigger the cleanup function
trap cleanup SIGINT SIGTERM

# 1. Boot up the FastAPI Backend in the background
echo "[SYSTEM] Launching FastAPI Backend (Port 8000)..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Give the backend a 3-second head start to bind to the port
sleep 3

# 2. Boot up the Gradio Web Interface in the background
echo "[SYSTEM] Launching Gradio Frontend (Port 8001)..."
python -m src.api.gradio_ui &
GRADIO_PID=$!

echo "============================================================"
echo "✅ SYSTEM IS LIVE AND ORCHESTRATING"
echo "🔗 Backend API: http://localhost:8000/docs"
echo "🎨 Web UI:      http://localhost:8001"
echo "⚠️  Press Ctrl+C at any time to safely shut down the entire OS."
echo "============================================================"

# Wait indefinitely, keeping the script alive while background tasks run
wait
