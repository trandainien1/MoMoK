#!/usr/bin/env bash
set -euo pipefail
set +x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/.venv-kaggle-cli/bin/python"
KAGGLE="$ROOT_DIR/.venv-kaggle-cli/bin/kaggle"

source "$ROOT_DIR/scripts/kaggle_env.sh"
"$ROOT_DIR/scripts/kaggle_auth.sh"
"$PYTHON" "$ROOT_DIR/scripts/build_kaggle_notebook.py"
"$PYTHON" "$ROOT_DIR/scripts/validate_notebook.py"

KERNEL_REF="${KAGGLE_USERNAME_SLUG}/${KAGGLE_KERNEL_SLUG}"
[[ "$KERNEL_REF" == "nientrandai1/"* ]] || { echo "KAGGLE_ACCOUNT_MISMATCH=FAIL" >&2; exit 1; }
selected=""
for accelerator in NvidiaTeslaA100 NvidiaL4 NvidiaL4X1 NvidiaTeslaT4; do
  export KAGGLE_ACCELERATOR="$accelerator"
  if "$PYTHON" "$ROOT_DIR/scripts/configure_kaggle_kernel.py" >/tmp/momok_metadata_attempt.txt 2>&1 && \
     "$KAGGLE" kernels push -p "$ROOT_DIR/kaggle/momok-db15k" --accelerator "$accelerator"; then
    selected="$accelerator"
    break
  fi
done

if [[ -z "$selected" ]]; then
  echo "KAGGLE_PUSH=FAIL" >&2
  exit 1
fi

echo "KAGGLE_PUSH=PASS"
echo "KERNEL_REF=$KERNEL_REF"
echo "ACCELERATOR=$selected"
echo "GPU=ON"
echo "INTERNET=ON"
echo "PRIVATE=YES"
echo "MODE=RESTART_WITH_CHECKPOINTING"

if tmux has-session -t momok-kaggle-monitor 2>/dev/null; then
  tmux kill-session -t momok-kaggle-monitor
fi
tmux new-session -d -s momok-kaggle-monitor "cd '$ROOT_DIR' && ./scripts/kaggle_monitor_detached.sh"
echo "MONITOR=tmux:momok-kaggle-monitor"
