#!/usr/bin/env python3
import hashlib,json,time
from pathlib import Path
from reference_causal_program_chunk_v1 import CausalChunkBankV1,Q

def main():
    t=time.perf_counter();b=CausalChunkBankV1();seq=(11,22,33,44);chunk=None
    for n in range(5):chunk=b.observe(seq,n*10,n*10+6,Q//2,Q//2,successor=77)
    c=chunk;p=b.predictive.row(c.identity);checks={}
    checks['repetition_condenses_higher_order']=c is not None and c.members==seq and c.depth==1
    checks['chunk_reduces_decision_depth']=b.decision_depth(c.identity)==1 and len(seq)>1
    checks['chunk_has_lived_time_effort']=p.duration_mean_q16==6*Q and p.effort_mean_q16==Q//2
    entrench=p.accessibility_q16;b.devalue(c.identity,-2*Q)
    checks['devaluation_suppresses_without_erasing_chunk']=(not b.executable(c.identity) and c.identity in b.chunks and b.predictive.row(c.identity).accessibility_q16==entrench)
    before=b.predictive.row(c.identity).duration_mean_q16
    for n in range(5,10):b.observe(seq,n*20,n*20+14,Q//2,Q//2,successor=77)
    checks['duration_reversal_recalibrates']=b.predictive.row(c.identity).duration_mean_q16>before
    pe=b.predictive.row(c.identity).prediction_error_q16;b.predictive.observe_successor(c.identity,88,1)
    checks['successor_perturbation_prediction_error']=b.predictive.row(c.identity).prediction_error_q16>pe
    # Higher abstraction from already-condensed opaque chunks, no transcript replay.
    seq2=(55,66)
    for n in range(3):c2=b.observe(seq2,300+n*5,302+n*5,Q//4,Q//2)
    higher=None
    for n in range(3):higher=b.observe((c.identity,c2.identity),400+n*8,405+n*8,Q//3,Q//2)
    checks['chunks_compose_into_higher_chunk']=higher is not None and higher.depth>=2 and higher.members==(c.identity,c2.identity)
    pending=(91,92,93);b.observe(pending,500,502,Q//8,Q//4);b.observe(pending,503,505,Q//8,Q//4)
    checkpoint=json.loads(json.dumps(b.checkpoint()));restored=CausalChunkBankV1.restore(checkpoint)
    checks['learned_chunk_connectivity_survives_restart']=(restored.checkpoint()==checkpoint and
        restored.chunks==b.chunks and restored.predictive.snapshot()==b.predictive.snapshot())
    checks['preconsolidation_quorum_survives_restart']=restored.observe(pending,506,508,Q//8,Q//4) is not None
    checks['transient_factor_lowering_is_not_duplicate_chunk_authority']=not restored.factors
    checks['bounded_runtime']=time.perf_counter()-t<2
    failed=[k for k,v in checks.items() if not v]
    if failed:raise SystemExit('FOUNDRY_AGI_CAUSAL_CHUNK_RED '+','.join(failed))
    here=Path(__file__).parent;paths=[here/'reference_causal_program_chunk_v1.py',here/'reference_causal_program_chunk_verify.py']
    r={'contract':'FOUNDRY_AGI_CAUSAL_CHUNK_GREEN','reference_only':True,'language_data':False,'neuron_required':False,'recipe_required':False,'network_required':False,'checks':checks,'chunk_id':c.identity,'higher_id':higher.identity,'remaining_red':['DIRECT_PHYSICAL_ACTION_PARITY','CAPACITY_INTERFERENCE_SWEEP','CANDIDATE_A_B_C_DELETION_TOURNAMENT','LANGUAGE_CONSTRUCTION_TRANSFER'],'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    print(r['contract']);print(json.dumps(r,sort_keys=True,indent=2))
if __name__=='__main__':main()
