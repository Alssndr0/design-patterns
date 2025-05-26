#!/bin/bash

# Exit on error
set -e


# Detect OS and install uv if missing
if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."

    # Detect OS type
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        # Linux or macOS
        curl -LsSf https://astral.sh/uv/install.sh | sh
    else
        # Assume Windows (using PowerShell)
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    fi
fi

echo "uv is installed and available."
# 2. Create venv with specified Python version
uv venv --python 3.12.0

# 3. Activate venv
uv init

# 4. Install required packages
UV_TORCH_BACKEND=auto uv pip install torch
uv pip install -r pyproject.toml

# 5. Make sure Hugging Face cache is local (persistent, project folder)
export HF_HOME="$(pwd)/hf_cache"
mkdir -p "$HF_HOME"

echo "Setup complete. All Hugging Face models will be stored in $HF_HOME."
