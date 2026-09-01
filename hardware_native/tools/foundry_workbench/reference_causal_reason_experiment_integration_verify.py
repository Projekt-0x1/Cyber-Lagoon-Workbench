#!/usr/bin/env python3
"""Whole-organism separation of world, self, and recommendation-outcome evidence."""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_reason_action_recommendation_v1 import ReasonActionRecommendationV1
from reference_population_v1 import PopulationSpecV1

SRC=6001;REASON_SRC=6002;ACTION=201;OTHER=202;REASON=301;S0=(401,);S1=(402,)

def prepare(independent=True):
    o=ReferenceOrganismV2(PopulationSpecV1(32768,fanout=2,sites_per_feature=4,eligibility_horizon=8))
    o.contact(CONTACT_WORLD_STATE,S0,SRC,True,True);o.contact(CONTACT_BODY_TARGET,S1,6003,True,True);o.contact(CONTACT_AFFORDANCES,(ACTION,OTHER),6004,True,True)
    branch=PolicyBranchV1(ACTION,EXPQ//2,EXPQ//2,EXPQ//2,EXPQ//8,3*EXPQ//4,0)
    recommendation=ReasonActionRecommendationV1(REASON,REASON_SRC,ACTION,0)
    gain=o.recursive_causal_experiment.information_gain_q16(branch,(branch,),(recommendation,))
    tick=max(int(o.recursive_causal_experiment._tick)+1,int(getattr(o,'_developmental_curriculum_tick',0))+1)
    intervention=o.recursive_causal_experiment.begin(ACTION,0,(recommendation,),(ACTION,OTHER),gain,tick)
    if not intervention:raise AssertionError('recommendation:intervention')
    o._metacontrol_pending_intervention=intervention;motor=o._issue_motor(ACTION)
    if not isinstance(motor,MotorActionV2) or motor.action_id!=ACTION:raise AssertionError(('recommendation:motor',motor))
    before={'edges':len(o.cognition.edges()),'self':o.recursive_metacontrol.outcome_count,'recommendation':o.recursive_causal_experiment.reason_outcome_count}
    o.contact(CONTACT_MOTOR_CONSEQUENCE,(motor.ticket,1,len(S1),*S1),SRC,True,bool(independent))
    after={'edges':len(o.cognition.edges()),'self':o.recursive_metacontrol.outcome_count,'recommendation':o.recursive_causal_experiment.reason_outcome_count}
    return o,recommendation,before,after

def main():
    started=time.perf_counter();checks={};o,recommendation,before,after=prepare(True)
    checks['one_real_return_updates_world_transition_owner']=after['edges']>before['edges']
    checks['same_independent_probe_calibrates_global_self_exactly_once']=after['self']==before['self']+1
    checks['same_independent_return_calibrates_exposed_recommendation_once']=after['recommendation']==before['recommendation']+1
    checks['recommendation_outcome_history_is_action_local']=(o.recommendation_outcome_reliability_q16(REASON,REASON_SRC,ACTION)>EXPQ//2 and o.recommendation_outcome_reliability_q16(REASON,REASON_SRC,OTHER)==EXPQ//2)
    checks['recommendation_success_does_not_create_proposition_truth']=recommendation.authority==0 and not hasattr(o.recursive_causal_experiment,'proposition_truth') and not hasattr(o.recursive_causal_experiment,'explanation_truth')
    intervention=o.recursive_causal_experiment.intervention(1)
    checks['intervention_remains_zero_authority_after_settlement']=bool(intervention and intervention.authority==0 and intervention.settled and intervention.independent)
    counts=(o.recursive_metacontrol.outcome_count,o.recursive_causal_experiment.reason_outcome_count);_=o.constructive_self_futures(128)
    checks['constructive_replay_cannot_create_self_or_recommendation_evidence']=counts==(o.recursive_metacontrol.outcome_count,o.recursive_causal_experiment.reason_outcome_count)

    n,nrecommendation,nbefore,nafter=prepare(False)
    checks['nonindependent_return_creates_no_self_calibration']=nafter['self']==nbefore['self']
    checks['nonindependent_return_creates_no_recommendation_calibration']=nafter['recommendation']==nbefore['recommendation']
    nintervention=n.recursive_causal_experiment.intervention(1)
    checks['nonindependent_intervention_is_history_not_evidence']=bool(nintervention and nintervention.settled and not nintervention.independent and nintervention.authority==0)

    cp=o.checkpoint();restored=ReferenceOrganismV2.restore(copy.deepcopy(cp))
    checks['three_owner_state_survives_checkpoint']=(restored.recursive_metacontrol.checkpoint()==o.recursive_metacontrol.checkpoint() and restored.recursive_causal_experiment.checkpoint()==o.recursive_causal_experiment.checkpoint() and len(restored.cognition.edges())==len(o.cognition.edges()))
    failed=sorted(k for k,v in checks.items() if not v)
    result={'schema':'cyber-lagoon.recommendation-experiment-integration.v2','pass':not failed,'checks':checks,'failed':failed,'before':before,'after':after,
        'claim':'ONE_REAL_RETURN_SEPARATELY_UPDATES_WORLD_SELF_AND_ACTION_RECOMMENDATION_RELATIONS_WITHOUT_PROPOSITION_TRUTH',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_RECOMMENDATION_EXPERIMENT_INTEGRATION '+('GREEN' if not failed else 'RED'));print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if not failed else 1)
if __name__=='__main__':main()
