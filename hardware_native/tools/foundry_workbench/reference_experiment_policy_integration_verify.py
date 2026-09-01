#!/usr/bin/env python3
"""Whole-organism receipt for context-conditional adaptive experiment policies."""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_reason_action_recommendation_v1 import ReasonActionRecommendationV1
from reference_population_v1 import PopulationSpecV1

A=81;B=82;SRC=901;RA=101;RB=102;RSA=902;RSB=903

def organism():return ReferenceOrganismV2(PopulationSpecV1(32768,fanout=2,sites_per_feature=4,eligibility_horizon=8))
def branches():
    return (PolicyBranchV1(A,POLICYQ//3,POLICYQ//3,POLICYQ//2,POLICYQ//8,POLICYQ,0),PolicyBranchV1(B,3*POLICYQ//4,3*POLICYQ//4,POLICYQ//2,POLICYQ//8,POLICYQ//3,0))
def recommendations():
    return {A:(ReasonActionRecommendationV1(RA,RSA,A,0),),B:(ReasonActionRecommendationV1(RB,RSB,B,0),)}
def real_probe(o,sequence,action,recs,policy=0,context=None,independent=True):
    state=(300+sequence,);nxt=(400+sequence,)
    o.contact(CONTACT_WORLD_STATE,state,SRC,True,True);o.contact(CONTACT_BODY_TARGET,nxt,904,True,True);o.contact(CONTACT_AFFORDANCES,(A,B),905,True,True)
    before=o.recursive_causal_experiment.recommendation_calibration_world(action,recs,0)[1]
    tick=max(int(o.recursive_causal_experiment._tick)+1,int(getattr(o,'_developmental_curriculum_tick',0))+1)
    intervention=o.recursive_causal_experiment.begin(action,0,recs,(A,B),POLICYQ//2,tick)
    if not intervention:raise AssertionError('policy-integration:intervention')
    if policy:
        if context is None:raise AssertionError('policy-integration:context')
        o.recursive_experiment_policy.begin(intervention,policy,context,before,True)
    o._metacontrol_pending_intervention=intervention;motor=o._issue_motor(action)
    if not isinstance(motor,MotorActionV2) or int(motor.action_id)!=int(action):raise AssertionError(('policy-integration:motor',motor))
    o.contact(CONTACT_MOTOR_CONSEQUENCE,(motor.ticket,1,len(nxt),*nxt),SRC,True,bool(independent));return intervention

def main():
    started=time.perf_counter();checks={};o=organism();a,b=branches();recommendation_map=recommendations()
    for i in range(4):real_probe(o,10+i,A,recommendation_map[A],0,None,True)
    checks['pretraining_saturates_only_a_recommendation_certainty']=(o.recursive_causal_experiment.recommendation_calibration_world(A,recommendation_map[A],0)[1]==POLICYQ and o.recursive_causal_experiment.recommendation_calibration_world(B,recommendation_map[B],0)[1]==0)

    frontier=o.recursive_causal_experiment.eligible_probes((a,b),recommendation_map,POLICYQ,POLICYQ)
    context=o.recursive_experiment_policy.context(frontier,POLICYQ,POLICYQ)
    discrim=o.recursive_causal_experiment.policy_candidate(frontier,POLICY_DISCRIMINATE);confirm=o.recursive_causal_experiment.policy_candidate(frontier,POLICY_CONFIRM)
    checks['same_safe_frontier_supports_distinct_policy_choices']=bool(discrim and confirm and discrim.action==A and confirm.action==B)
    before_policy=o.recursive_experiment_policy.evidence_count;before_edges=len(o.cognition.edges());default_policy,_=o.recursive_experiment_policy.choose_policy(frontier,POLICYQ,POLICYQ)
    checks['policy_choice_itself_creates_no_evidence']=o.recursive_experiment_policy.evidence_count==before_policy and len(o.cognition.edges())==before_edges
    checks['high_ambiguity_untrained_context_starts_discriminative']=default_policy==POLICY_DISCRIMINATE

    before_self=o.recursive_metacontrol.outcome_count;before_recommendation=o.recursive_causal_experiment.reason_outcome_count
    real_probe(o,20,A,recommendation_map[A],POLICY_DISCRIMINATE,context,False)
    checks['nonindependent_policy_trial_trains_no_policy']=o.recursive_experiment_policy.evidence_count==before_policy
    checks['nonindependent_policy_trial_trains_no_self_or_recommendation']=(o.recursive_metacontrol.outcome_count==before_self and o.recursive_causal_experiment.reason_outcome_count==before_recommendation)

    real_probe(o,21,A,recommendation_map[A],POLICY_DISCRIMINATE,context,True);real_probe(o,22,A,recommendation_map[A],POLICY_DISCRIMINATE,context,True)
    real_probe(o,23,B,recommendation_map[B],POLICY_CONFIRM,context,True);real_probe(o,24,B,recommendation_map[B],POLICY_CONFIRM,context,True)
    shifted,_=o.recursive_experiment_policy.choose_policy(frontier,POLICYQ,POLICYQ)
    checks['real_motor_history_shifts_policy_from_discriminate_to_confirm']=shifted==POLICY_CONFIRM
    checks['discriminative_retests_were_realized_nondiagnostic']=o.recursive_experiment_policy.competence_q16(POLICY_DISCRIMINATE,context)<POLICYQ//2
    checks['confirmatory_tests_were_realized_diagnostic']=o.recursive_experiment_policy.competence_q16(POLICY_CONFIRM,context)>POLICYQ//2

    medium_context=o.recursive_experiment_policy.context(frontier,5*POLICYQ//8,POLICYQ)
    checks['policy_competence_does_not_leak_across_resource_context']=o.recursive_experiment_policy.evidence_for(POLICY_CONFIRM,medium_context)==0
    checks['learned_policy_cannot_override_current_resource_veto']=not o.recursive_causal_experiment.eligible_probes((a,b),recommendation_map,POLICYQ//8,POLICYQ)
    checks['policy_learning_does_not_generalize_recommendation_to_other_action']=(o.recommendation_outcome_reliability_q16(RA,RSA,B)==POLICYQ//2 and o.recommendation_outcome_reliability_q16(RB,RSB,A)==POLICYQ//2)
    checks['recommendations_have_zero_proposition_truth_authority']=all(rec.authority==0 for rows in recommendation_map.values() for rec in rows)
    checks['policy_traces_remain_zero_authority']=all(int(row.authority)==0 for row in o.recursive_experiment_policy._traces)

    cp=o.checkpoint();r=ReferenceOrganismV2.restore(copy.deepcopy(cp))
    checks['policy_causal_strategy_state_survives_checkpoint']=(r.recursive_experiment_policy.checkpoint()==o.recursive_experiment_policy.checkpoint() and r.recursive_experiment_strategy.checkpoint()==o.recursive_experiment_strategy.checkpoint() and r.recursive_causal_experiment.checkpoint()==o.recursive_causal_experiment.checkpoint())
    failed=sorted(k for k,v in checks.items() if not v)
    result={'schema':'cyber-lagoon.adaptive-experiment-policy-integration.v2','pass':not failed,'checks':checks,'failed':failed,'policy_evidence':o.recursive_experiment_policy.evidence_count,'default_policy':default_policy,'shifted_policy':shifted,'claim':'ONE_LIFE_LEARNS_CONTEXT_CONDITIONAL_EXPERIMENT_POLICY_FROM_REAL_RECOMMENDATION_DIAGNOSTICITY_WITH_HARD_CAUSAL_SAFETY','elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_ADAPTIVE_EXPERIMENT_POLICY_INTEGRATION '+('GREEN' if not failed else 'RED'));print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if not failed else 1)
if __name__=='__main__':main()
