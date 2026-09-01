#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'hardware_native/tools/foundry_workbench'
SRC=ROOT/'hardware_native/src/hardware_native'

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,timeout=60)
    if p.returncode:
        print(p.stdout);print(p.stderr,file=sys.stderr);raise SystemExit(p.returncode)
    return p.stdout

def main():
    t=time.perf_counter();checks={}
    seq=json.loads((WORK/'language_direct_relseq_gpu_receipt.json').read_text())
    bridge=json.loads((WORK/'direct_adult_relseq_bridge_gpu_receipt.json').read_text())
    removed_sequence_sources=(
        WORK/'language_crossmodal_prose_snapshot_v1.py',
        WORK/'language_learning_architecture_v6.py',
        WORK/'build_language_direct_relseq_cuda_contract.py',
        WORK/'language_direct_relseq_parity.py')
    checks['sequence_receipt_classified_historical']=all(not p.exists() for p in removed_sequence_sources)
    seq_header=sha(SRC/'direct_relational_sequence_composition.cuh')
    bridge_header=sha(SRC/'direct_adult_relational_sequence_bridge.cuh')
    contract_src=sha(WORK/'cuda_direct_adult_relseq_bridge_contract.cu')
    checks['sequence_receipt_inputs_absent']=checks['sequence_receipt_classified_historical']
    checks['bridge_receipt_classified_historical']=(
        bridge.get('status')=='GREEN' and bridge.get('compute_sanitizer_errors')==0 and
        bridge.get('canonical_recipe_cell') is True and bridge.get('canonical_recipe_occurrence') is True)
    exe=Path('/tmp/direct_adult_relseq_bridge_compile')
    run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',
         '-I','hardware_native/src','-I','hardware_native/include',
         '-I','/usr/local/cuda/include',
         str(WORK/'direct_adult_relseq_bridge_compile.cpp'),'-o',str(exe)])
    out=run([str(exe)])
    checks['current_host_canonical_bridge_evaluate']=(
        'DIRECT_ADULT_RELSEQ_BRIDGE_HOST GREEN' in out and 'fake_occurrence=0' in out)
    network_exe=Path('/tmp/reference_network_persistent_unfolding_verify')
    run(['g++','-std=c++17','-O0','-Wall','-Wextra','-Werror',
         '-I','hardware_native/src','-I','hardware_native/include',
         str(WORK/'reference_network_persistent_unfolding_verify.cpp'),
         '-o',str(network_exe)])
    network_out=run([str(network_exe)])
    checks['compact_network_persistent_unfolding']=(
        'FOUNDRY_NETWORK_PERSISTENT_UNFOLDING GREEN' in network_out and
        'transient_veto=1' in network_out and
        'runtime_llm=0' in network_out)
    result={'schema':'0x1.direct-language-portability-boundary-audit.v2','audit_pass':all(checks.values()),'checks':checks,
            'reference_only':True,'graph_flip':False,'adult_attached':False,
            'production_ir':'ResidentRecipeIrProgram.vcurrent','translation_status':'UNDEFINED',
            'physical_direct_parity':'NOT_RUN/RED','capability_status':'C0_RED',
            'sequence_gpu_receipt':{'classification':'UNVERIFIED_HISTORICAL_DECLARATION_STALE_SOURCE_REMOVED','gpu':seq.get('gpu'),'artifact_sha256':seq.get('artifact_sha256'),'unit_digest':seq.get('unit_digest'),'header_sha256_now':seq_header,'header_sha256_receipt':seq.get('production_sequence_header_sha256'),'header_match':seq.get('production_sequence_header_sha256')==seq_header},
            'bridge_gpu_receipt_declaration':{'classification':'UNVERIFIED_HISTORICAL_DECLARATION','gpu':bridge.get('gpu'),'artifact_sha256':bridge.get('artifact_sha256'),'header_sha256_now':bridge_header,'contract_sha256_now':contract_src,'header_match':bridge.get('adult_bridge_header_sha256')==bridge_header and bridge.get('contract_source_sha256')==contract_src},
            'historical_receipt_verification':'DECLARATION_ONLY; ARTIFACT, SANITIZER LOG, TOOLCHAIN, BUILD COMMAND, AND TRANSITIVE SOURCES NOT VERIFIED',
            'scope':'boundary audit and current host bridge compilation only; no Adult, language, or physical parity claim',
            'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_DIRECT_LANGUAGE_PORTABILITY_BOUNDARY '+('AUDIT_PASS_C0_RED' if result['audit_pass'] else 'AUDIT_FAIL'))
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['audit_pass'] else 1)
if __name__=='__main__':main()
