#!/usr/bin/env python3
"""Whole-organism receipt for learned experiment-strategy competence."""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_reason_action_recommendation_v1 import ReasonActionRecommendationV1
from reference_population_v1 import PopulationSpecV1

ACTION=71;OTHER=72;SRC=801;REASON_SRC=802;REASON=91

def organism():return ReferenceOrganismV2(PopulationSpecV1(32768,fanout=2,sites_per_feature=4,eligibility_horizon=8))
def run_probe(o,sequence,recommendations,independent=True):
    state=(100+sequence,);nxt=(200+sequence,)
    o.contact(CONTACT_WORLD_STATE,state,SRC,True,True);o.contact(CONTACT_BODY_TARGET,nxt,803,True,True);o.contact(CONTACT_AFFORDANCES,(ACTION,OTHER),804,True,True)
    branch=PolicyBranchV1(ACTION,STRATQ//2,STRATQ//2,STRATQ//2,STRATQ//8,3*STRATQ//4,0);other=PolicyBranchV1(OTHER,STRATQ//2,STRATQ//2,STRATQ//2,STRATQ//8,STRATQ//2,0)
    key=o.recursive_experiment_strategy.structural_key(branch,(branch,other),len(recommendations),STRATQ)
    before=o.recursive_causal_experiment.recommendation_calibration_world(ACTION,recommendations,0)[1];gain=o.recursive_causal_experiment.recommendation_information_gain_world_q16(branch,(branch,other),recommendations,0)
    live_tick=max(int(o.recursive_causal_experiment._tick),int(getattr(o,'_developmental_curriculum_tick',0)),int(o.tick_count))+1
    intervention=o.recursive_causal_experiment.begin(ACTION,0,recommendations,(ACTION,OTHER),gain,live_tick)
    if not intervention:raise AssertionError('intervention')
    o.recursive_experiment_strategy.begin(intervention,key,before,True);o._metacontrol_pending_intervention=intervention
    motor=o._issue_motor(ACTION)
    if not isinstance(motor,MotorActionV2):raise AssertionError(('motor',motor))
    o.contact(CONTACT_MOTOR_CONSEQUENCE,(motor.ticket,1,len(nxt),*nxt),SRC,True,bool(independent));return key,intervention

def main():
    started=time.perf_counter();checks={};o=organism()
    key1,_=run_probe(o,1,(),True);key2,_=run_probe(o,2,(),True)
    checks['two_real_no_recommendation_probes_train_strategy_failure']=o.recursive_experiment_strategy.evidence_count==2 and key1==key2
    checks['repeated_no_recommendation_geometry_is_suppressed']=not o.recursive_experiment_strategy.permits(key1)
    before_self=o.recursive_metacontrol.outcome_count;before_strategy=o.recursive_experiment_strategy.evidence_count
    _key_nonind,_=run_probe(o,3,(),False)
    checks['nonindependent_probe_trains_no_strategy']=o.recursive_experiment_strategy.evidence_count==before_strategy
    checks['nonindependent_probe_trains_no_self_model']=o.recursive_metacontrol.outcome_count==before_self
    recommendation=(ReasonActionRecommendationV1(REASON,REASON_SRC,ACTION,0),);key3,intervention3=run_probe(o,4,recommendation,True)
    checks['recommendation_bearing_probe_is_distinct_strategy_geometry']=key3!=key1 and key3.reason_count==1 and key1.reason_count==0
    checks['suppressed_no_recommendation_strategy_does_not_suppress_recommendation_probe']=o.recursive_experiment_strategy.permits(key3)
    checks['diagnostic_probe_adds_strategy_evidence']=o.recursive_experiment_strategy.evidence_count==before_strategy+1
    checks['diagnostic_probe_updates_exact_action_recommendation_history']=o.recommendation_outcome_reliability_q16(REASON,REASON_SRC,ACTION)>EXPQ//2
    checks['strategy_history_does_not_generalize_recommendation_to_other_action']=o.recommendation_outcome_reliability_q16(REASON,REASON_SRC,OTHER)==EXPQ//2
    checks['recommendation_object_has_zero_truth_authority']=recommendation[0].authority==0
    trace=o.recursive_experiment_strategy.trace(intervention3)
    checks['strategy_trace_is_zero_authority']=bool(trace and trace.settled and trace.independent and trace.realized_diagnosticity_q16>0 and trace.authority==0)
    cp=o.checkpoint();r=ReferenceOrganismV2.restore(copy.deepcopy(cp))
    checks['strategy_and_causal_state_survive_checkpoint']=(r.recursive_experiment_strategy.checkpoint()==o.recursive_experiment_strategy.checkpoint() and r.recursive_causal_experiment.checkpoint()==o.recursive_causal_experiment.checkpoint())
    failed=sorted(k for k,v in checks.items() if not v)
    result={'schema':'cyber-lagoon.experiment-strategy-integration.v2','pass':not failed,'checks':checks,'failed':failed,'strategy_evidence':o.recursive_experiment_strategy.evidence_count,'recommendation_evidence':o.recursive_causal_experiment.reason_outcome_count,'claim':'SELF_SELECTED_PROBE_GEOMETRY_LEARNS_FROM_REALIZED_RECOMMENDATION_DIAGNOSTICITY_WITHOUT_TRUTH_OR_SAFETY_LEAKS','elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_EXPERIMENT_STRATEGY_INTEGRATION '+('GREEN' if not failed else 'RED'));print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if not failed else 1)
if __name__=='__main__':main()
