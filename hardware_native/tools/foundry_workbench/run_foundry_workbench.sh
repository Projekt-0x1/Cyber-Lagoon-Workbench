#!/usr/bin/env bash
set -euo pipefail
if [[ "${FOUNDRY_BOUNDED:-0}" != 1 ]]; then
  exec env FOUNDRY_BOUNDED=1 timeout --signal=KILL 60s bash "$0" "$@"
fi
root="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$root"
start="$(date +%s%N)"
run() { echo "=== $1 ==="; shift; "$@"; }
# This named runner is the authoritative strict boundary only. Exploratory
# organism/language assays remain available through run_language_workbench.sh,
# but their semantic fixture APIs cannot certify this contract.
run legacy-absence python3 hardware_native/tools/foundry_workbench/legacy_absence_verify.py
run reference-1610 hardware_native/tools/foundry_workbench/run_reference_contract_1610.sh
run portability-boundary python3 hardware_native/tools/foundry_workbench/verify_direct_language_portability.py
end="$(date +%s%N)"
out="${FOUNDRY_WORKBENCH_OUT:-${TMPDIR:-/tmp}/0x1-foundry-workbench}"
mkdir -p "$out"
python3 - "$start" "$end" "$out" <<'PY'
import json,pathlib,sys
elapsed=(int(sys.argv[2])-int(sys.argv[1]))/1e6
receipt={
  'schema':'0x1.graph-neutral-foundry-workbench.v4',
  'workbench_status':'C0_RED',
  'strict_boundary_audit':'PASS',
  'reference_contract':'REFERENCE_ONLY_NON_CAPABILITY_PASS',
  'reference_only':True,
  'adult_attached':False,
  'legacy_modules':0,
  'portability_boundary_audit':'PASS_C0_RED',
  'semantic_fixture_suites_authoritative':False,
  'runtime_llm':False,
  'network':True,
  'network_reference_causal_recruitment':False,
  'network_math_host_contract':False,
  'compiler':False,
  'per_recipe_compile':False,
  'cuda':False,
  'experimental_ir':'ReferenceRecipeIrV1',
  'production_ir':'ResidentRecipeIrProgram.vcurrent',
  'translation_status':'UNDEFINED',
  'physical_direct_parity':'NOT_RUN/RED',
  'whole_adult_physical_parity':'NOT_RUN/RED',
  'graph_flip':False,
  'human_level_language_claim':False,
  'adult_language_claim':False,
  'phase_lowering':'ARBITRATE_FRONTIER_COMPOSE_PUBLISH_SETTLE_CONDENSE',
  'remaining_gaps':['authenticated causal-difference Network credit','production IR translation',
                    'physical Direct differential parity','production resident cognition ownership'],
  'elapsed_ms':round(elapsed,3),
}
path=pathlib.Path(sys.argv[3]);path.mkdir(parents=True,exist_ok=True)
(path/'closure_receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,sort_keys=True))
print(f"FOUNDRY_WORKBENCH C0_RED elapsed_ms={elapsed:.3f} reference_only=true adult_attached=false legacy_modules=0 physical_direct_parity=NOT_RUN/RED graph_flip=false")
PY
