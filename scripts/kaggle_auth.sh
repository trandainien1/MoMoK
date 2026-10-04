#!/usr/bin/env bash
set -euo pipefail
set +x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/kaggle_env.sh"
KAGGLE="$ROOT_DIR/.venv-kaggle-cli/bin/kaggle"
AUTH_METHOD="TOKEN"

if ! account_output="$($KAGGLE config view 2>/dev/null)"; then
  echo "KAGGLE_AUTH=FAIL"
  exit 1
fi
if ! printf '%s\n' "$account_output" | grep -Eq 'username:[[:space:]]*nientrandai1([[:space:]]|$)'; then
  echo "KAGGLE_AUTH=FAIL"
  echo "KAGGLE_ACCOUNT_MISMATCH=FAIL"
  exit 1
fi

echo "KAGGLE_AUTH=PASS"
echo "KAGGLE_ACCOUNT=$KAGGLE_EXPECTED_ACCOUNT"
echo "AUTH_METHOD=$AUTH_METHOD"
