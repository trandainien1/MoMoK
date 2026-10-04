#!/usr/bin/env bash
set -euo pipefail
set +x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/kaggle_env.sh"
KAGGLE="$ROOT_DIR/.venv-kaggle-cli/bin/kaggle"
PYTHON="$ROOT_DIR/.venv-kaggle-cli/bin/python"
export KAGGLE_MKGY_KERNEL_SLUG="momok-mkg-y-reproduction"
export KAGGLE_ACTIVE_KERNEL_REF="nientrandai1/momok-mkg-y-reproduction"

"$ROOT_DIR/scripts/kaggle_auth.sh"
"$PYTHON" "$ROOT_DIR/scripts/build_mkgy_notebook.py"
"$PYTHON" "$ROOT_DIR/scripts/validate_mkgy_notebook.py"

kernel_ref="nientrandai1/momok-mkg-y-reproduction"
for accelerator in NvidiaTeslaA100 NvidiaL4 NvidiaL4X1 NvidiaTeslaT4; do
  export KAGGLE_ACCELERATOR="$accelerator"
  "$PYTHON" "$ROOT_DIR/scripts/configure_mkgy_kernel.py"
  if "$KAGGLE" kernels push -p "$ROOT_DIR/kaggle/momok-mkg-y" --accelerator "$accelerator"; then
    echo "KAGGLE_PUSH=PASS"
    echo "KERNEL_REF=$kernel_ref"
    echo "ACCEPTED_ACCELERATOR=$accelerator"
    echo "PRIVATE=YES"
    echo "GPU=ON"
    echo "INTERNET=ON"
    exit 0
  fi
done
echo "KAGGLE_PUSH=FAIL" >&2
exit 1
