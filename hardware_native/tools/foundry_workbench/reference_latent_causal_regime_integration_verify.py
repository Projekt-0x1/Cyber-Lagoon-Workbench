#!/usr/bin/env python3
"""Whole-organism receipt for WORLD_CONTEXT versus CONTROL_CONTEXT category separation."""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_reason_action_recommendation_v1 import ReasonActionRecommendationV1
from reference_population_v1 import PopulationSpecV1

ACTION=61;OTHER=62;WORLD_SRC=1101;REASON_SRC=1102;REASON=211
TARGET=(901,);S0=(10,20,30,40);S1=(501,);BODY_A=(1001,1002,1003,1004,1005,1006,1007,1008);BODY_B=(2001,2002,2003,2004,2005,2006,2007,2008)

def organism():return ReferenceOrganismV2(PopulationSpecV1(32768,fanout=2,sites_per_feature=4,eligibility_horizon=8))
def set_state(o,state,body):
    o.contact(CONTACT_WORLD_STATE,tuple(state),WORLD_SRC,True,True);o.contact(CONTACT_BODY_TARGET,TARGET,1103,True,True);o.contact(CONTACT_BODY_STATE,tuple(body),1105,True,True);o.contact(CONTACT_AFFORDANCES,(ACTION,OTHER),1104,True,True)
def train_transition(o,state,nxt):
    o.cognition.observe(state,ACTION,nxt,1,2101,True);o.cognition.observe(state,ACTION,nxt,1,2102,True)
def motor(o,nxt,effect=1,independent=True):
    m=o._issue_motor(ACTION)
    if not isinstance(m,MotorActionV2):raise AssertionError(('category:motor',m))
    o.contact(CONTACT_MOTOR_CONSEQUENCE,(m.ticket,int(effect),len(nxt),*nxt),WORLD_SRC,True,bool(independent));return m

def main():
    started=time.perf_counter();checks={};o=organism();train_transition(o,S0,S1);set_state(o,S0,BODY_A)
    w1,c1,_wc,_cc,_r,_c=o._infer_contexts(S0,WORLD_SRC)
    recommendation=ReasonActionRecommendationV1(REASON,REASON_SRC,ACTION,0)
    tick=max(int(o.recursive_causal_experiment._tick)+1,int(getattr(o,'_developmental_curriculum_tick',0))+1)
    before_global_self=o.recursive_metacontrol.outcome_count
    iv=o.recursive_causal_experiment.begin_world_probe(ACTION,0,(recommendation,),(ACTION,OTHER),CTXQ//2,tick,w1,True)
    if not iv:raise AssertionError('category:intervention')
    o._metacontrol_pending_intervention=iv;motor(o,S1,1,True)
    checks['one_probe_writes_global_self_exactly_once']=o.recursive_metacontrol.outcome_count==before_global_self+1
    checks['probe_writes_world_context_recommendation_once']=o.contextual_recommendation_outcome_reliability_q16(REASON,REASON_SRC,ACTION,w1)>CTXQ//2
    checks['successful_probe_writes_control_self']=o.contextual_self_reliability_q16(ACTION,c1)>CTXQ//2
    checks['recommendation_is_not_proposition_truth']=recommendation.authority==0 and not hasattr(o.recursive_causal_experiment,'proposition_truth')

    set_state(o,S0,BODY_B);w_same,c2,_wc,_cc,_r,_c=o._infer_contexts(S0,WORLD_SRC)
    checks['body_change_does_not_change_world_context']=w_same==w1
    checks['body_change_can_change_control_context']=c2!=c1
    checks['recommendation_outcome_history_does_not_change_with_body_control_context']=(
        o.contextual_recommendation_outcome_reliability_q16(REASON,REASON_SRC,ACTION,w_same)==o.contextual_recommendation_outcome_reliability_q16(REASON,REASON_SRC,ACTION,w1))
    motor(o,S1,-1,True)
    checks['self_competence_can_differ_across_control_context']=(o.contextual_self_reliability_q16(ACTION,c1)>CTXQ//2 and o.contextual_self_reliability_q16(ACTION,c2)<CTXQ//2)

    w=organism();WA=(30,40,50,60);WB=(30,40,50,70);EXPECTED=(701,);ACTUAL=(702,)
    train_transition(w,WA,EXPECTED);train_transition(w,WB,EXPECTED);set_state(w,WA,BODY_A);motor(w,EXPECTED,1,True)
    set_state(w,WA,BODY_A);wa,_ca,*_=w._infer_contexts(WA,WORLD_SRC);before_contexts=w.recursive_context_partition.context_count
    set_state(w,WB,BODY_A);pre_w,_cb,*_=w._infer_contexts(WB,WORLD_SRC)
    checks['novel_world_cues_are_provisionally_classified_before_outcome']=pre_w==wa
    motor(w,ACTUAL,1,True);wb=w._world_context_current
    checks['supported_preoutcome_transition_mismatch_can_split_world_context']=(wb!=wa and w.recursive_context_partition.row(wb).parent==wa and w.recursive_context_partition.context_count>before_contexts)

    ident=organism();train_transition(ident,WA,EXPECTED);set_state(ident,WA,BODY_A);motor(ident,EXPECTED,1,True)
    set_state(ident,WA,BODY_A);wi,_ci,*_=ident._infer_contexts(WA,WORLD_SRC);before_ident=ident.recursive_context_partition.context_count;motor(ident,ACTUAL,1,True)
    checks['identical_world_cue_mismatch_does_not_spawn_context']=ident._world_context_current==wi and ident.recursive_context_partition.context_count==before_ident

    yoked=organism();train_transition(yoked,WA,EXPECTED);train_transition(yoked,WB,EXPECTED);set_state(yoked,WA,BODY_A);motor(yoked,EXPECTED,1,True)
    set_state(yoked,WA,BODY_A);yw,_yc,*_=yoked._infer_contexts(WA,WORLD_SRC);set_state(yoked,WB,BODY_A);yp,_yc2,*_=yoked._infer_contexts(WB,WORLD_SRC)
    before_yoked=yoked.recursive_context_partition.context_count;motor(yoked,ACTUAL,1,False)
    checks['nonindependent_transition_mismatch_cannot_split_world_context']=yp==yw and yoked.recursive_context_partition.context_count==before_yoked

    old_api_refused=False
    try:o.infer_causal_regime()
    except RuntimeError:old_api_refused=True
    checks['self_confidence_as_world_regime_api_is_disabled']=old_api_refused
    checks['context_rows_have_zero_truth_authority']=all(int(row.authority)==0 for row in o.recursive_context_partition._rows)

    cp=o.checkpoint();r=ReferenceOrganismV2.restore(copy.deepcopy(cp))
    checks['typed_context_and_evidence_survive_checkpoint']=(r.recursive_context_partition.checkpoint()==o.recursive_context_partition.checkpoint() and r.recursive_metacontrol.checkpoint()==o.recursive_metacontrol.checkpoint() and r.recursive_causal_experiment.checkpoint()==o.recursive_causal_experiment.checkpoint())
    failed=sorted(k for k,v in checks.items() if not v)
    result={'schema':'cyber-lagoon.typed-context-category-separation.v2','pass':not failed,'checks':checks,'failed':failed,
        'world_context_a':w1,'control_context_a':c1,'control_context_b':c2,
        'claim':'WORLD_TRANSITION_CONTROL_COMPETENCE_AND_RECOMMENDATION_OUTCOME_HISTORY_REMAIN_CAUSALLY_DISTINCT',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_TYPED_CONTEXT_CATEGORY_SEPARATION '+('GREEN' if not failed else 'RED'));print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if not failed else 1)
if __name__=='__main__':main()
