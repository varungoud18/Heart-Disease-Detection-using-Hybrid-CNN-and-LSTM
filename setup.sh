#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=== Heart Disease Detection Project Environment Setup ==="

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 could not be found. Please install Python 3.10+ and try again."
    exit 1
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment '.venv'..."
    python3 -m venv .venv
else
    echo "Virtual environment '.venv' already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=========================================================="
echo "Setup completed successfully!"
echo "To activate the virtual environment in your terminal, run:"
echo "    source .venv/bin/activate"
echo "=========================================================="
