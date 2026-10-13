#!/usr/bin/env bash
# ==============================================================================
# Transcript Generator Launcher
# Optimized for NVIDIA GeForce RTX 3050 & CUDA GPU Acceleration
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "⚡ Initializing virtual environment with uv..."
    if command -v uv &> /dev/null; then
        uv venv --python 3.12 "$VENV_DIR"
        uv pip install --python "$VENV_DIR/bin/python" -r requirements.txt
    else
        python3 -m venv "$VENV_DIR"
        "$VENV_DIR/bin/pip" install -r requirements.txt
    fi
fi

# Locate CUDA libraries inside venv
SITE_PACKAGES=$(find "$VENV_DIR/lib" -name "site-packages" -type d | head -n 1)
if [ -d "$SITE_PACKAGES/nvidia" ]; then
    for sub in cublas cudnn cuda_nvrtc cuda_runtime; do
        if [ -d "$SITE_PACKAGES/nvidia/$sub/lib" ]; then
            export LD_LIBRARY_PATH="$SITE_PACKAGES/nvidia/$sub/lib:${LD_LIBRARY_PATH:-}"
        fi
    done
fi

# Launch the interactive transcription pipeline
exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/main.py" "$@"
