# Project Setup

## 1. Install uv & create virtual environment
curl -LsSf https://astral.sh/uv/install.sh | sh   # Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv venv --python 3.12.0
uv init

## 2. Install dependencies (from pyproject.toml)
UV_TORCH_BACKEND=auto uv pip install torch
uv pip install -r pyproject.toml

## 3. Run any script with local Hugging Face cache:
uv run run_with_cache.py dir/script.py   # uv run run_with_cache.py creational_patterns/singleton.py 
uv tool install poethepoet

## 4. Run any script with Poe the Poet
poe script_name   # Example: poe singleton

## 5. To clean up all downloaded models and cache:
rm -rf hf_cache  # Windows: rmdir /s /q hf_cache
