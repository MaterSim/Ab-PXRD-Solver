#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_conda_env.sh [env_name]
# Example:
#   bash scripts/setup_conda_env.sh pxrd-agent

ENV_NAME="${1:-pxrd-agent}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda was not found in PATH. Install Miniconda/Anaconda first."
  exit 1
fi

# Enable 'conda activate' in non-interactive shells.
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "[INFO] Creating environment '${ENV_NAME}' with Python ${PYTHON_VERSION}"
conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
conda activate "${ENV_NAME}"

echo "[INFO] Installing core scientific stack from conda-forge"
conda install -y -c conda-forge \
  "numpy=1.26" pandas scipy matplotlib tqdm scikit-learn \
  pymatgen pyxtal ase monty spglib typing_extensions

echo "[INFO] Installing PyTorch"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] NVIDIA GPU detected. Installing CUDA-enabled PyTorch."
  conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
else
  echo "[INFO] No NVIDIA GPU detected. Installing CPU PyTorch."
  conda install -y -c pytorch pytorch torchvision torchaudio cpuonly
fi

echo "[INFO] Installing pip packages used by this project"
python -m pip install --upgrade pip
python -m pip install \
  "numpy<2" \
  google-genai \
  strands-agents \
  wandb \
  tensorboard \
  jarvis-tools \
  mace-torch

echo "[INFO] GSAS-II setup note"
echo "[INFO] GSAS-II is not reliably available as a conda/pip package on all platforms/channels."
echo "[INFO] Use the official gitstrap workflow documented by GSAS-II to install it."

echo "[INFO] Verifying key imports"
python - <<'PY'
import importlib

modules = [
    "numpy",
    "pandas",
    "scipy",
    "torch",
    "pyxtal",
    "pymatgen",
    "ase",
    "mace",
    "strands",
    "google.genai",
]

missing = []
for m in modules:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)

if missing:
    print("[WARN] Missing imports:", ", ".join(missing))
else:
    print("[OK] Environment verification passed.")
PY

echo ""
echo "[DONE] Environment '${ENV_NAME}' is ready."
echo "Activate it with: conda activate ${ENV_NAME}"
echo "Then run: python PXRD_agent.py"
