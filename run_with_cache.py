import os
import subprocess
import sys
from pathlib import Path


def main():
    project_root = Path(__file__).parent.resolve()
    hf_cache = project_root / "hf_cache"
    hf_cache.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["HF_HOME"] = str(hf_cache)

    # Compose the command to run user's script (passed as args)
    if len(sys.argv) < 2:
        print("Usage: python run_with_cache.py <script.py> [args...]")
        sys.exit(1)
    cmd = [sys.executable] + sys.argv[1:]
    print(f"Running: {' '.join(cmd)} with HF_HOME={hf_cache}")
    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
