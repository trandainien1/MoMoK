#!/usr/bin/env bash
set -euo pipefail
set +x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/kaggle_env.sh"
KERNEL_REF="${KAGGLE_ACTIVE_KERNEL_REF:-${KAGGLE_USERNAME_SLUG}/${KAGGLE_KERNEL_SLUG}}"
[[ "$KERNEL_REF" == "nientrandai1/"* ]] || { echo "KAGGLE_ACCOUNT_MISMATCH=FAIL" >&2; exit 1; }
LOG_FILE="$ROOT_DIR/kaggle_outputs/kaggle_monitor.log"
mkdir -p "$ROOT_DIR/kaggle_outputs"

while true; do
  status="$(timeout 45 "$ROOT_DIR/.venv-kaggle-cli/bin/kaggle" kernels status "$KERNEL_REF" 2>&1 || true)"
  {
    date -Is
    printf '%s\n' "$status"
  } >> "$LOG_FILE"
  if printf '%s\n' "$status" | grep -Eiq 'KernelWorkerStatus\.(COMPLETE|SUCCESS)'; then
    "$ROOT_DIR/scripts/kaggle_download_outputs.sh" "$KERNEL_REF" >> "$LOG_FILE" 2>&1
    exit 0
  fi
  if printf '%s\n' "$status" | grep -Eiq 'KernelWorkerStatus\.(ERROR|FAILED|CANCEL)'; then
    exit 1
  fi
  sleep 60
done
