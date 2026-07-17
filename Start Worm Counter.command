#!/bin/bash
# ── Worm Counter Launcher ──────────────────────────────────────────────────────
# Double-click this file in Finder to start the app.

APP_DIR="$(cd "$(dirname "$0")/worm_app" && pwd)"

echo "============================================"
echo "  Worm Counter"
echo "============================================"

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Please install Python 3."
    read -p "Press Enter to close..."
    exit 1
fi

# Install dependencies if missing
echo "Checking dependencies..."
python3 -c "import cv2, flask, flask_cors, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages (one-time setup)..."
    pip3 install opencv-python-headless flask flask-cors numpy
fi

# Kill any existing instance on port 8080
lsof -ti:8080 | xargs kill -9 2>/dev/null

# Start Flask
echo "Starting server on http://localhost:8080 ..."
cd "$APP_DIR"
python3 app.py &
SERVER_PID=$!

# Wait for server to be ready
for i in {1..10}; do
    sleep 0.5
    if curl -s http://localhost:8080/ > /dev/null 2>&1; then
        break
    fi
done

# Get local IP for sharing
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "unknown")

# Open browser
echo "Opening browser..."
open http://localhost:8080

echo ""
echo "  Your address:      http://localhost:8080"
echo "  Share with others: http://$LOCAL_IP:8080"
echo ""
echo "App is running. Close this window to stop the server."
echo "(or press Ctrl+C)"

# Keep running until user closes window
wait $SERVER_PID
