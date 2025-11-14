#!/bin/bash
# Start script for Taiwan CCTV Viewer

echo "==================================="
echo "Taiwan CCTV Viewer - Startup"
echo "==================================="
echo ""

cd "$(dirname "$0")"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies with increased timeout for large packages (torch, cuda)
echo "Installing/updating dependencies..."
echo "Note: First-time installation may take 10-15 minutes due to large ML packages"
pip install -r requirements.txt

# Check if installation was successful
if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to install dependencies"
    echo "This is often due to network timeouts with large packages (PyTorch, CUDA)"
    echo ""
    echo "You can try:"
    echo "1. Run the script again (it will resume from cache)"
    echo "2. Install manually: source venv/bin/activate && pip install --timeout 300 -r requirements.txt"
    echo ""
    deactivate 2>/dev/null
    exit 1
fi

echo ""
echo "Starting backend server on port 8001..."
echo "Starting frontend server on port 8000..."
echo ""

# Start backend in background
python backend/main.py &
BACKEND_PID=$!

# Check if backend started successfully
sleep 1
if ! ps -p $BACKEND_PID > /dev/null; then
    echo "ERROR: Backend failed to start"
    echo "Check backend/main.py for errors"
    exit 1
fi

# Wait for backend to start
sleep 2

# Start simple HTTP server for frontend
cd client
python -m http.server 8000 &
FRONTEND_PID=$!

echo ""
echo "==================================="
echo "Servers started successfully!"
echo "==================================="
echo "Backend API: http://0.0.0.0:8001"
echo "Frontend:    http://0.0.0.0:8000"
echo ""
echo "Access from:"
echo "  - Local:   http://localhost:8000"
echo "  - Network: http://$(hostname -I | awk '{print $1}'):8000"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "==================================="

# Handle Ctrl+C - kill all child processes
cleanup() {
    echo ''
    echo 'Stopping servers...'
    # Kill the specific PIDs
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
    # Also kill any processes on our ports (in case PIDs changed)
    lsof -ti:8001 2>/dev/null | xargs kill -9 2>/dev/null
    lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null
    # Deactivate virtual environment
    deactivate 2>/dev/null
    echo 'Servers stopped'
    exit 0
}

trap cleanup INT TERM

# Wait for processes
wait
