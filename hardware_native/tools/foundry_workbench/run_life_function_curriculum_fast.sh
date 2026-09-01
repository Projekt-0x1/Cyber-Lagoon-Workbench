#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$root"
start="$(date +%s%N)"

# One birth, one canonical chronology, all mechanic probes from derivative checkpoints.
python3 hardware_native/tools/foundry_workbench/reference_life_function_factory_verify.py

end="$(date +%s%N)";ms=$(( (end-start)/1000000 ))
echo "FOUNDRY_LIFE_FUNCTION_CURRICULUM_FAST_GREEN elapsed_ms=$ms canonical_curriculum=v3 one_birth=1 one_training_chain=1 fixture_births=0 transformer=0 direct_parity=RED graph_promotion=RED"
