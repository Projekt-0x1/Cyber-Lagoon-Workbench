#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$root"
start="$(date +%s%N)"

bash hardware_native/tools/foundry_workbench/run_life_function_curriculum_fast.sh

end="$(date +%s%N)";ms=$(( (end-start)/1000000 ))
echo "FOUNDRY_LANGUAGE_MASTERY_FACTORY_FAST GREEN elapsed_ms=$ms one_birth=1 one_training_chain=1 fixture_births=0 transformer=0"
