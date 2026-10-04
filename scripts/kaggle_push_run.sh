#!/usr/bin/env bash
set -euo pipefail
set +x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/.venv-kaggle-cli/bin/python"
KAGGLE="$ROOT_DIR/.venv-kaggle-cli/bin/kaggle"

if [[ ! -x "$PYTHON" || ! -x "$KAGGLE" ]]; then
  echo "MATERIAL_BLOCKER: .venv-kaggle-cli is not ready" >&2
  exit 1
fi

source "$ROOT_DIR/scripts/kaggle_env.sh"
"$ROOT_DIR/scripts/kaggle_auth.sh"

"$PYTHON" "$ROOT_DIR/scripts/build_kaggle_notebook.py"
"$PYTHON" "$ROOT_DIR/scripts/validate_notebook.py"
"$PYTHON" "$ROOT_DIR/scripts/configure_kaggle_kernel.py"

KERNEL_REF="${KAGGLE_USERNAME_SLUG}/${KAGGLE_KERNEL_SLUG}"
[[ "$KERNEL_REF" == "nientrandai1/"* ]] || { echo "KAGGLE_ACCOUNT_MISMATCH=FAIL" >&2; exit 1; }
echo "KERNEL=$KERNEL_REF"
echo "ACCELERATOR=$KAGGLE_ACCELERATOR"
echo "INTERNET=ON"
echo "GPU=ON"
echo "PRIVATE=YES"

"$KAGGLE" kernels push -p "$ROOT_DIR/kaggle/momok-db15k" --accelerator "$KAGGLE_ACCELERATOR"
echo "KAGGLE_PUSH=PASS"
echo "KERNEL_REF=$KERNEL_REF"
echo "KAGGLE_RUN_STARTED=YES"

if "$KAGGLE" kernels logs "$KERNEL_REF" --follow --interval 20; then
  true
else
  echo "KAGGLE_LOG_FOLLOW=UNSUPPORTED_OR_ENDED"
fi

for attempt in $(seq 1 60); do
  status_output="$("$KAGGLE" kernels status "$KERNEL_REF" 2>&1 || true)"
  printf '%s\n' "$status_output"
  if printf '%s\n' "$status_output" | grep -Eiq 'complete|success'; then
    echo "KAGGLE_RUN=COMPLETE"
    exec "$ROOT_DIR/scripts/kaggle_download_outputs.sh" "$KERNEL_REF"
  fi
  if printf '%s\n' "$status_output" | grep -Eiq 'error|failed|cancel'; then
    echo "KAGGLE_RUN=FAIL" >&2
    exit 1
  fi
  sleep 20
done

echo "KAGGLE_RUN=TIMEOUT" >&2
exit 1
