#!/usr/bin/env python3
import hashlib,json,time
from pathlib import Path
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q

def train(b,sid,n,d,e,out=None,som=0,ctx=1):
    t=10
    for _ in range(n):
        b.observe_use(sid,t,t+d,e,ctx)
        if out is not None:b.observe_return(sid,out,som,t+d+1,True,ctx)
        t+=d+3

def populate(b):
    hb,rg,fast,slow,easy,hard,sur=101,102,103,104,105,106,107
    train(b,hb,8,2,Q//8,-Q//2,-Q//4);train(b,rg,1,2,Q//8,Q,Q//4)
    train(b,fast,4,2,Q//8,Q//4);train(b,slow,4,9,Q//8,Q//4)
    train(b,easy,4,4,Q//16,Q//4);train(b,hard,4,4,Q//2,Q//4)
    train(b,sur,3,3,Q//8,Q//2)
    for _ in range(5):b.observe_successor(sur,9001,2)
    pe=b.row(sur).prediction_error_q16;b.observe_successor(sur,9002,2)
    b.observe_use(hb,200,202,Q//8,77);access_before=b.row(hb).accessibility_q16;rare_access_before=b.row(rg).accessibility_q16
    b.disuse(1000)
    b.row(555);b.observe_return(555,Q,Q,20,False,3)
    b.row(777)
    for x in range(100,120):b.observe_successor(777,x,1)
    return locals(),pe,access_before,rare_access_before

def main():
    started=time.perf_counter();b=PredictiveCreditBankV1(16);v,pe,access_before,rare_access_before=populate(b);checks={}
    hb,rg,fast,slow,easy,hard,sur=[b.row(x) for x in (101,102,103,104,105,106,107)]
    checks['habit_accessible_and_bad']=access_before>rare_access_before and hb.outcome_mean_q16<0
    checks['rare_valuable_without_entrenchment']=rg.outcome_mean_q16>0 and rare_access_before<access_before
    checks['duration_independent']=slow.duration_mean_q16>fast.duration_mean_q16 and slow.outcome_mean_q16==fast.outcome_mean_q16
    checks['effort_independent']=hard.effort_mean_q16>easy.effort_mean_q16 and hard.duration_mean_q16==easy.duration_mean_q16
    checks['surprise_not_negative_value']=sur.prediction_error_q16>pe and sur.outcome_mean_q16>0
    checks['context_shift_no_global_erase']=hb.last_context==77 and hb.outcome_mean_q16<0
    checks['disuse_only_access']=hb.accessibility_q16<access_before and hb.outcome_mean_q16<0 and hb.duration_mean_q16>0
    cold=b.row(555);checks['nonindependent_return_cannot_teach']=cold.outcome_samples==0 and cold.somatic_mean_q16==0
    checks['successor_sparse']=len(b.row(777).successors)<=8
    cap=PredictiveCreditBankV1(16)
    for x in range(1,9):cap.observe_successor(1,x,1)
    for _ in range(3):cap.observe_successor(1,1,1)
    cap.observe_successor(1,9,1)
    checks['successor_eviction_keeps_count_winner']=1 in cap.row(1).successors and 9 in cap.row(1).successors and 8 not in cap.row(1).successors
    checks['unique_successor_is_count_winner']=sur.expected_successor()==9001
    replay=PredictiveCreditBankV1(16);populate(replay);checks['deterministic_replay']=replay.snapshot()==b.snapshot()
    local=PredictiveCreditBankV1(2)
    local.observe_use(1,1,2,Q//8,11);local.observe_use(2,3,4,Q//8,22)
    checks['context_candidates_are_incidence_local']=local.candidates(11)=={1} and local.candidates(22)=={2} and local.candidates(33)==()
    local.observe_return(2,Q,0,5,True,33)
    checks['return_without_participation_cannot_nominate']=local.candidates(33)==()
    local.observe_use(3,5,6,Q//8,33)
    checks['eviction_removes_derived_candidate']=1 not in local.rows and local.candidates(11)==() and local.candidates(33)=={3}
    checks['candidates_are_live_incidence_set']=isinstance(local.candidates(33),set)
    checks['bounded_runtime']=time.perf_counter()-started<2
    failed=[k for k,v in checks.items() if not v]
    if failed:raise SystemExit('FOUNDRY_AGI_PREDICTIVE_CREDIT_RED '+','.join(failed))
    here=Path(__file__).parent;paths=[here/'reference_predictive_credit_profile_v1.py',here/'reference_predictive_credit_profile_verify.py']
    receipt={'contract':'FOUNDRY_AGI_PREDICTIVE_CREDIT_GREEN','reference_only':True,'language_phenotype_improved':True,'canonical_integrated':False,'language_data':False,'tokens':False,'transformer':False,'backprop':False,'expected_output':False,'checks':checks,'habitual_bad':{'access':hb.accessibility_q16,'outcome':hb.outcome_mean_q16,'duration':hb.duration_mean_q16,'effort':hb.effort_mean_q16,'somatic':hb.somatic_mean_q16},'rare_good':{'access':rg.accessibility_q16,'outcome':rg.outcome_mean_q16},'remaining_red':['NON_LANGUAGE_HIERARCHICAL_CHUNK_INTEGRATION','DIRECT_RECIPE_NETWORK_PARITY','CAUSAL_PROGRAM_DELETION_TOURNAMENT','LANGUAGE_TRANSFER'],'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    print(receipt['contract']);print(json.dumps(receipt,sort_keys=True,indent=2))
if __name__=='__main__':main()
