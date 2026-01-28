#!/bin/bash
# ChessBot Conda Environment Setup
# Creates a clean 'cb' environment with all dependencies

set -e

ENV_NAME="cb"
PYTHON_VERSION="3.10"

echo "============================================================"
echo "ChessBot Conda Environment Setup"
echo "============================================================"

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] Removing existing '${ENV_NAME}' environment..."
    conda env remove -n ${ENV_NAME} -y
fi

# Create new environment
echo "[INFO] Creating '${ENV_NAME}' environment with Python ${PYTHON_VERSION}..."
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate and install requirements
echo "[INFO] Installing pip requirements..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}
python -m pip install -r requirements.txt

echo ""
echo "[OK] Environment '${ENV_NAME}' created successfully!"
echo ""
echo "Activate with: conda activate ${ENV_NAME}"
