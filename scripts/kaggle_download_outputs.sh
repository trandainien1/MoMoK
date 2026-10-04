#!/usr/bin/env bash
set -euo pipefail
set +x

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
KAGGLE="$ROOT_DIR/.venv-kaggle-cli/bin/kaggle"
source "$ROOT_DIR/scripts/kaggle_env.sh"
KERNEL_REF="${1:-${KAGGLE_ACTIVE_KERNEL_REF:-${KAGGLE_USERNAME_SLUG:?}/momok-mkg-w-reproduction}}"
[[ "$KERNEL_REF" == "nientrandai1/"* ]] || { echo "KAGGLE_ACCOUNT_MISMATCH=FAIL" >&2; exit 1; }
OUT_DIR="$ROOT_DIR/kaggle_outputs/latest"
mkdir -p "$OUT_DIR"

"$ROOT_DIR/scripts/kaggle_auth.sh" >/dev/null
"$KAGGLE" kernels output "$KERNEL_REF" -p "$OUT_DIR" -o

summary_path="$(find "$OUT_DIR" -type f -name repro_summary.json -print -quit)"
report_path="$(find "$OUT_DIR" -type f -name REPRODUCTION_REPORT.md -print -quit)"
log_path="$(find "$OUT_DIR" -type f -name db15k_full.log -print -quit)"
[[ -n "$summary_path" ]] && echo "REPRO_SUMMARY_FOUND=YES" || { echo "REPRO_SUMMARY_FOUND=NO"; exit 1; }
[[ -n "$report_path" ]] && echo "REPORT_FOUND=YES" || { echo "REPORT_FOUND=NO"; exit 1; }
[[ -n "$log_path" ]] && echo "FULL_LOG_FOUND=YES" || { echo "FULL_LOG_FOUND=NO"; exit 1; }

"$ROOT_DIR/.venv-kaggle-cli/bin/python" - "$summary_path" <<'PY'
import json
import sys
summary = json.load(open(sys.argv[1]))
print("VERDICT=" + str(summary.get("verdict")))
for key in ("MRR", "Hits@1", "Hits@3", "Hits@10"):
    print(f"REPRODUCED_{key}={summary.get('reproduced', {}).get(key)}")
PY
echo "DOWNLOAD=PASS"
