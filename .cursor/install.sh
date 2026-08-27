#!/usr/bin/env bash
# Idempotent bootstrap for the Kaggle notebooks in this repo.
# Creates an isolated virtualenv, installs pinned dependencies and registers a
# Jupyter kernel. Safe to run repeatedly (e.g. after dependency changes).
set -euo pipefail

cd "$(dirname "$0")/.."

# Runtime shared libraries needed by opencv-python (imported by the notebooks).
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-venv python3-pip libgl1 libglib2.0-0
fi

# Isolated virtualenv (kept out of git via .gitignore's .venv/).
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

python -m pip install --upgrade pip wheel
pip install -r requirements.txt

# Register a kernel so the notebooks run against this venv.
python -m ipykernel install --user \
  --name clustering-physical-activity \
  --display-name "Python (clustering-physical-activity)"

echo "install.sh: environment ready"
