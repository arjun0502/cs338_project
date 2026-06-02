#!/usr/bin/env bash
# One-shot env setup for the GRIFT / BigMath reproduction.
# Creates a uv-managed venv at ./.venv with all deps pinned. Run from repo root.
#
# Usage:
#   bash setup.sh
#   source .venv/bin/activate
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PY_VER="${PY_VER:-3.10}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"   # H100 box reports driver cu127, cu124 wheels are compatible

# 1. Tooling
if ! command -v uv >/dev/null; then
  echo "uv not found on PATH. Install with: pip install --user uv" >&2
  exit 1
fi

# Redirect uv cache to scratch ($HOME has a tight quota).
export UV_CACHE_DIR="${UV_CACHE_DIR:-/scratch/m000122-pm05/agam/uv_cache}"
mkdir -p "${UV_CACHE_DIR}"

# 2. Venv
if [ ! -d "${VENV_DIR}" ]; then
  uv venv --python "${PY_VER}" "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Convenience alias for the rest of the script
PIP=(uv pip install)

# 2.5. setuptools (uv venvs don't ship it; triton's JIT imports it at runtime)
"${PIP[@]}" "setuptools>=70" wheel

# 3. Torch (CUDA build)
"${PIP[@]}" "torch==2.4.0" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# 4. vLLM (pulls its own compatible transformers; install BEFORE peft/trl)
"${PIP[@]}" "vllm==0.6.3"

# 5. HF stack + RL training libs
"${PIP[@]}" \
  "transformers==4.46.0" \
  "peft==0.13.0" \
  "trl==0.11.4" \
  "accelerate==1.0.1" \
  "datasets==3.0.1"

# 6. Analysis + utilities
"${PIP[@]}" \
  "scikit-learn==1.5.2" \
  "numpy<2" \
  "pandas==2.2.2" \
  "pyarrow==17.0.0" \
  tqdm tiktoken openai matplotlib

# 7. HF cache on scratch (avoids filling $HOME)
HF_CACHE="${HF_CACHE:-/scratch/m000122-pm05/agam/hf_cache}"
mkdir -p "${HF_CACHE}"
if ! grep -q "HF_HOME=${HF_CACHE}" "${VENV_DIR}/bin/activate"; then
  echo "export HF_HOME=${HF_CACHE}" >> "${VENV_DIR}/bin/activate"
fi
export HF_HOME="${HF_CACHE}"

# 8. PYTHONPATH so cross-dir imports resolve (icl.gradient.*, utils_, etc.)
if ! grep -q "PYTHONPATH=" "${VENV_DIR}/bin/activate"; then
  cat >> "${VENV_DIR}/bin/activate" <<EOF
export PYTHONPATH="${REPO_ROOT}/big_math:${REPO_ROOT}/arlsat:${REPO_ROOT}:\${PYTHONPATH:-}"
EOF
fi

# 9. Sanity check
python - <<'PY'
import torch, transformers, peft, trl, vllm, datasets, sklearn, numpy
print("torch        :", torch.__version__, "| cuda:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
print("transformers :", transformers.__version__)
print("peft         :", peft.__version__)
print("trl          :", trl.__version__)
print("vllm         :", vllm.__version__)
print("datasets     :", datasets.__version__)
print("sklearn      :", sklearn.__version__)
print("numpy        :", numpy.__version__)
PY

echo
echo "Env ready. Activate with:  source ${VENV_DIR}/bin/activate"
echo "HF cache: ${HF_CACHE}"
