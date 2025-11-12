#!/bin/bash
# Setup script for Gemma 3 1B environment

echo "=================================="
echo "Gemma 3 1B Setup"
echo "=================================="

# Check if we're in the right directory
if [ ! -d "gemma_pytorch" ]; then
    echo "Error: Please run this from the gemma3-1b directory"
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/3] Virtual environment already exists"
fi

# Activate venv
echo "[2/3] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "[3/3] Installing dependencies..."
echo "  - Installing requirements.txt..."
pip install -r requirements.txt

echo "  - Installing gemma_pytorch package..."
pip install -e gemma_pytorch/

echo ""
echo "=================================="
echo "✅ Setup complete!"
echo "=================================="
echo ""
echo "To use:"
echo "  source venv/bin/activate"
echo "  python load_and_run.py"
echo ""
echo "To see the ACTUAL model code:"
echo "  cat gemma_pytorch/gemma/model.py"
echo ""
