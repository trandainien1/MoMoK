#!/usr/bin/env bash
# Source this file in the current shell before any Kaggle CLI command.
set +x

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "MATERIAL_BLOCKER: source scripts/kaggle_env.sh; do not execute it" >&2
  exit 1
fi

KAGGLE_REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE_ENV_FILE="$KAGGLE_REPRO_ROOT/.secrets/kaggle.env"
if [[ ! -f "$KAGGLE_ENV_FILE" ]]; then
  echo "MATERIAL_BLOCKER: Kaggle credential file is missing" >&2
  return 1
fi

set -a
# shellcheck disable=SC1090
source "$KAGGLE_ENV_FILE"
set +a

if [[ "${KAGGLE_USERNAME_SLUG:-}" != "nientrandai1" ]]; then
  echo "MATERIAL_BLOCKER: expected Kaggle account nientrandai1" >&2
  return 1
fi
if [[ -z "${KAGGLE_API_TOKEN:-}" || "$KAGGLE_API_TOKEN" == "PASTE_TOKEN_HERE" ]]; then
  echo "MATERIAL_BLOCKER: Kaggle API token is unavailable" >&2
  return 1
fi

export KAGGLE_EXPECTED_ACCOUNT="nientrandai1"
export KAGGLE_KERNEL_SLUG="momok-mkg-w-reproduction"
export KAGGLE_ACTIVE_KERNEL_REF="nientrandai1/momok-mkg-w-reproduction"
export KAGGLE_CONFIG_DIR="$KAGGLE_REPRO_ROOT/.secrets/kaggle-config"
export KAGGLE_REPRO_ROOT
export KAGGLE_CONFIG_CONTEXT_PERSISTENT="PASS"
