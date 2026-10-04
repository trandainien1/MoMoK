#!/usr/bin/env bash
set -euo pipefail
set +x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/kaggle_env.sh"
KERNEL_REF="nientrandai1/momok-mkg-y-reproduction"
KAGGLE="$ROOT_DIR/.venv-kaggle-cli/bin/kaggle"
LOG_FILE="$ROOT_DIR/kaggle_outputs/mkgy_monitor.log"
mkdir -p "$ROOT_DIR/kaggle_outputs"

while true; do
  status="$(timeout 45 "$KAGGLE" kernels status "$KERNEL_REF" 2>&1 || true)"
  {
    date -Is
    printf '%s\n' "$status"
  } >> "$LOG_FILE"
  if printf '%s\n' "$status" | grep -Eiq 'KernelWorkerStatus\.(COMPLETE|SUCCESS)'; then
    exit 0
  fi
  if printf '%s\n' "$status" | grep -Eiq 'KernelWorkerStatus\.(ERROR|FAILED|CANCEL)'; then
    exit 1
  fi
  sleep 60
done
