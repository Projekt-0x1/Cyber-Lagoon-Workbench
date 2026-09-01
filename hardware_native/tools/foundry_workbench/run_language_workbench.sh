#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$root"
start="$(date +%s%N)"

# Retired synchronization union. There is one developmental authority now:
# the canonical Life Function. Historical focused verifiers remain runnable by
# name for archaeology/falsification, but no top-level language entry point may
# rebirth and privately retrain a second Adult.
bash hardware_native/tools/foundry_workbench/run_life_function_curriculum_fast.sh

end="$(date +%s%N)";ms=$(( (end-start)/1000000 ))
echo "FOUNDRY_LANGUAGE_WORKBENCH GREEN elapsed_ms=$ms one_life_function=1 fixture_births=0 reference_only=1"
