#!/usr/bin/env bash
set -euo pipefail
set +x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/kaggle_env.sh"
KAGGLE="$ROOT_DIR/.venv-kaggle-cli/bin/kaggle"

[[ "$KAGGLE_EXPECTED_ACCOUNT" == "nientrandai1" ]]
[[ "$KAGGLE_CONFIG_CONTEXT_PERSISTENT" == "PASS" ]]

child_view="$(bash -c 'set +x; printf "%s\n" "$KAGGLE_CONFIG_DIR"; exec "$1" config view' -- "$KAGGLE")"
grep -Fxq "$KAGGLE_CONFIG_DIR" <<<"$child_view"
grep -Eq 'username:[[:space:]]*nientrandai1([[:space:]]|$)' <<<"$child_view"

echo "KAGGLE_EXPECTED_ACCOUNT=nientrandai1"
echo "KAGGLE_CONFIG_CONTEXT_PERSISTENT=PASS"
echo "GLOBAL_ACCOUNT_FALLBACK_PREVENTED=PASS"
