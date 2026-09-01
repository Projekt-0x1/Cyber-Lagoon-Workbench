#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$root"
mode="${1:-fast}"
case "$mode" in
  fast)
    cmd="python3 hardware_native/tools/foundry_workbench/pycuda_organism_scale_verify.py"
    ;;
  deep)
    cmd="python3 hardware_native/tools/foundry_workbench/pycuda_organism_scale_verify.py && python3 hardware_native/tools/foundry_workbench/pycuda_language_pressure_verify.py"
    ;;
  *)
    echo "usage: $0 [fast|deep]" >&2
    exit 2
    ;;
esac
if command -v gpu-exec >/dev/null 2>&1; then
  exec gpu-exec --exclusive "$cmd"
fi
exec bash -lc "$cmd"
