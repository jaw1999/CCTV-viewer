#!/bin/bash
set -e

# Start backend API server
python backend/main.py &
BACKEND_PID=$!

# Start frontend HTTP server
cd /app/client
python -m http.server 8000 &
FRONTEND_PID=$!
cd /app

# Trap signals for graceful shutdown
cleanup() {
    echo "Stopping servers..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null
    echo "Servers stopped."
    exit 0
}

trap cleanup SIGTERM SIGINT

# Wait for either process to exit
wait -n
EXIT_CODE=$?
cleanup
exit $EXIT_CODE
