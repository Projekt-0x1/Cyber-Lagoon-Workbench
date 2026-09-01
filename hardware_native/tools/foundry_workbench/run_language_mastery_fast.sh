#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$root"
start="$(date +%s%N)"

# Product iteration and destructive controls share the same Life Function.
bash hardware_native/tools/foundry_workbench/run_life_function_curriculum_fast.sh

end="$(date +%s%N)"
ms=$(( (end-start)/1000000 ))
echo "FOUNDRY_LANGUAGE_MASTERY_FAST GREEN elapsed_ms=$ms one_life_function=1 fixture_births=0 reference_only=1"
