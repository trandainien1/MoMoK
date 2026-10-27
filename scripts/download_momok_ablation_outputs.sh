#!/usr/bin/env bash
set -euo pipefail
set +x

usage() {
  echo "usage: $0 <owner/kernel-slug> <destination>" >&2
  exit 2
}

[[ $# -eq 2 ]] || usage
kernel_ref=$1
destination=$2
[[ "$kernel_ref" =~ ^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$ ]] || usage
[[ -n "${KAGGLE_API_TOKEN:-}" ]] || {
  echo "KAGGLE_API_TOKEN must be provided by the caller's local auth context" >&2
  exit 2
}
command -v kaggle >/dev/null || {
  echo "kaggle CLI not found" >&2
  exit 127
}

case "${kernel_ref##*/}" in
  momok-mkg-w-no-gating-noise-full|momok-mkg-w-no-exid-full|momok-mkg-w-no-relation-temperature-full) ;;
  *) echo "kernel is not in the completed MKG-W ablation allowlist" >&2; exit 2 ;;
esac

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
target="$destination/${kernel_ref##*/}"
mkdir -p "$target"
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
pattern='(^|/)(ablation_smoke\.log|dataset_manifest\.json|output_inventory\.json|checkpoint_final\.pt\.sha256)$'
kaggle kernels output "$kernel_ref" --path "$stage" --file-pattern "$pattern"

for filename in ablation_smoke.log dataset_manifest.json output_inventory.json checkpoint_final.pt.sha256; do
  mapfile -t matches < <(find "$stage" -type f -name "$filename" -print)
  [[ ${#matches[@]} -eq 1 ]] || {
    echo "expected exactly one allowlisted $filename from Kaggle output" >&2
    exit 1
  }
  install -m 600 "${matches[0]}" "$target/$filename"
done

PYTHONPATH="$root_dir${PYTHONPATH:+:$PYTHONPATH}" python - "$kernel_ref" "$target" <<'PY'
import sys
from pathlib import Path
from ablations.wandb_publication import PUBLICATION_VARIANTS, validate_safe_output_dir

ref, directory = sys.argv[1], Path(sys.argv[2])
slug = ref.split("/", 1)[1]
item = next((entry for entry in PUBLICATION_VARIANTS.values() if entry.kernel_slug == slug), None)
if item is None:
    raise SystemExit("kernel does not map to a known ablation publication")
validate_safe_output_dir(directory, item)
print(f"SAFE_KAGGLE_OUTPUT=PASS VARIANT={item.variant_id}")
PY
