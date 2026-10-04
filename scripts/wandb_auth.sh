#!/usr/bin/env bash
set -euo pipefail
set +x

set -a
source .secrets/wandb.env
set +a

: "${WANDB_API_KEY:?WANDB_API_KEY must be non-empty}"
: "${WANDB_ENTITY:?WANDB_ENTITY must be non-empty}"
: "${WANDB_PROJECT:?WANDB_PROJECT must be non-empty}"

echo "WANDB_CREDENTIAL_FILE=FOUND"
echo "WANDB_ENTITY=${WANDB_ENTITY}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
