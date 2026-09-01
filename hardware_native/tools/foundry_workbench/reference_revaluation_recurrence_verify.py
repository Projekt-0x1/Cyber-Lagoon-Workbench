#!/usr/bin/env python3
"""Actual discrepancy -> target-free recurrence -> later intervention assay."""
from __future__ import annotations

import copy
import json
import time

import reference_endogenous_prospection_verify as p
from reference_cognition_v1 import PlanV1
from reference_organism_v2 import MotorActionV2,ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1

START,A_MID,B_MID,GOAL,FAIL=4000,4100,4200,4999,4888
DECOY_START,DECOY_MID,DECOY_GOAL=100,101,102
ACTION_A,ACTION_B,FINISH,DECOY_FIRST,DECOY_FINISH=8001,8002,8003,8101,8102


def lived_edge(o,state,action,next_state,source,independent=True):
    p.stage(o,state,next_state,(action,),source)
    issued=o.tick()
    if not isinstance(issued,MotorActionV2) or issued.action_id!=action:
        raise AssertionError('revaluation:resident_action')
    p.settle(o,issued,source,next_state,1,independent)


def develop(a_mid=A_MID,b_mid=B_MID):
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    lived_edge(o,START,ACTION_A,a_mid,501)
    lived_edge(o,a_mid,FINISH,GOAL,502)
    lived_edge(o,START,ACTION_B,b_mid,503)
    lived_edge(o,b_mid,FINISH,GOAL,504)
    lived_edge(o,DECOY_START,DECOY_FIRST,DECOY_MID,505)
    lived_edge(o,DECOY_MID,DECOY_FINISH,DECOY_GOAL,506)

    # Before revaluation both one-shot routes are unresolved. The deterministic
    # ordinary fallback selects A; its yoked return changes no evidence.
    p.stage(o,START,GOAL,(ACTION_A,ACTION_B),600)
    baseline=o.tick()
    if not isinstance(baseline,MotorActionV2) or baseline.action_id!=ACTION_A:
        raise AssertionError('revaluation:baseline')
    p.settle(o,baseline,600,a_mid,1,False)

    # One real changed mapping makes A ambiguous. Two yoked B trials equalize
    # motor frequency without changing its lived transition evidence.
    lived_edge(o,START,ACTION_A,FAIL,601,True)
    lived_edge(o,START,ACTION_B,b_mid,602,False)
    lived_edge(o,START,ACTION_B,b_mid,603,False)
    return o


def lesion_on_demand(o):
    o.cognition._condense_prospective=(
        lambda start,goal,authority,max_depth=8,current_tick=0:
        PlanV1(0,(),(tuple(start),),0,0,()))


def choose(o,source):
    lesion_on_demand(o)
    p.stage(o,START,GOAL,(ACTION_A,ACTION_B),source)
    return o.tick()


def main():
    began=time.perf_counter();checks={};base=develop();checkpoint=copy.deepcopy(base.checkpoint())
    evidence=copy.deepcopy(base.cognition._evidence);credit=base.population.credit_events
    checks['real_changed_mapping_is_actual_but_does_not_majority_vote_truth']=(
        base.cognition.transition((START,),ACTION_A,1) is None
        and len(base.cognition._evidence)==7)
    checks['motor_exposure_is_equal_before_probe']=(
        base.exploration_trials[((START,),ACTION_A)]
        ==base.exploration_trials[((START,),ACTION_B)]==3)
    checks['discrepancy_prioritizes_unambiguous_local_alternative_without_target']=(
        base.cognition._reconcile_priority=={(B_MID,):1})

    prepared=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    built=prepared._reconcile_low_pressure(1);prepared_touches=prepared.cognition.last_reconcile_touches
    prepared_checkpoint=copy.deepcopy(prepared.checkpoint())
    target_key=((START,),(GOAL,))
    checks['budget_one_prepares_revalued_alternative']=(
        built==1 and prepared_touches==2
        and prepared.cognition._prospective_recipes[target_key].actions==(ACTION_B,FINISH))
    checks['recurrence_conserves_contact_evidence_and_credit']=(
        prepared.cognition._evidence==evidence
        and prepared.population.credit_events==credit)
    restored=ReferenceOrganismV2.restore(copy.deepcopy(prepared_checkpoint))
    checks['checkpoint_preserves_priority_consumption_and_prepared_recipe']=(
        restored.cognition.digest()==prepared.cognition.digest()
        and not restored.cognition._reconcile_priority)

    prioritized=ReferenceOrganismV2.restore(copy.deepcopy(prepared_checkpoint))
    prioritized_action=choose(prioritized,701)
    checks['prepared_recurrence_changes_later_real_intervention']=(
        isinstance(prioritized_action,MotorActionV2)
        and prioritized_action.action_id==ACTION_B
        and prioritized.last_prospective_recipe!=0)

    disabled=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    disabled_action=choose(disabled,702)
    checks['disabled_recurrence_retains_bad_tie_fallback']=(
        isinstance(disabled_action,MotorActionV2)
        and disabled_action.action_id==ACTION_A
        and disabled.last_prospective_recipe==0)

    focal=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    focal._reconcile_low_pressure(1);focal.cognition._prospective_recipes.pop(target_key)
    focal_action=choose(focal,703)
    checks['focal_recipe_lesion_removes_behavior_at_equal_reconcile_work']=(
        focal.cognition.last_reconcile_touches==prepared_touches
        and isinstance(focal_action,MotorActionV2) and focal_action.action_id==ACTION_A)

    remote=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    remote.cognition._reconcile_priority={(DECOY_MID,):1}
    remote.cognition._reconcile_frontier.add((DECOY_MID,))
    remote._reconcile_low_pressure(1);remote_action=choose(remote,704)
    checks['yoked_remote_recurrence_does_not_reproduce_choice']=(
        remote.cognition.last_reconcile_touches==prepared_touches
        and ((DECOY_START,),(DECOY_GOAL,)) in remote.cognition._prospective_recipes
        and isinstance(remote_action,MotorActionV2) and remote_action.action_id==ACTION_A)

    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    withdrawn.cognition.withdraw_source(503);withdrawn._reconcile_low_pressure(1)
    checks['source_withdrawal_prevents_preparation']=target_key not in withdrawn.cognition._prospective_recipes

    permuted=develop(a_mid=9400,b_mid=1200);permuted._reconcile_low_pressure(1)
    checks['opaque_state_permutation_preserves_discrepancy_priority']=(
        ((START,),(GOAL,)) in permuted.cognition._prospective_recipes
        and permuted.cognition._prospective_recipes[((START,),(GOAL,))].actions==(ACTION_B,FINISH))

    corrupt=copy.deepcopy(checkpoint)
    corrupt['cognition']['reconcile_priority'][0][1]=0
    try:
        ReferenceOrganismV2.restore(corrupt);priority_corruption_refused=False
    except ValueError:
        priority_corruption_refused=True
    checks['priority_checkpoint_corruption_refused']=priority_corruption_refused

    swapped=ReferenceOrganismV2.restore(copy.deepcopy(prepared_checkpoint))
    swapped_action=choose(swapped,705)
    p.settle(swapped,swapped_action,705,FAIL,1,True)
    lesion_on_demand(swapped);p.stage(swapped,START,GOAL,(ACTION_A,ACTION_B),706)
    after_swap=swapped.tick()
    checks['fresh_mapping_swap_defeats_imagined_route']=(
        swapped.cognition._prospective_recipes[target_key].shadow_counter==1
        and isinstance(after_swap,MotorActionV2) and after_swap.action_id!=ACTION_B)

    closed=ReferenceOrganismV2.restore(copy.deepcopy(prepared_checkpoint))
    closed_action=choose(closed,707);before_close=closed.population.credit_events
    p.settle(closed,closed_action,707,B_MID,1,True)
    checks['only_later_actual_attempt_and_return_close_new_credit']=(
        closed.population.credit_events>before_close
        and 707 in closed.cognition._evidence[((START,),ACTION_B,(B_MID,),1)])

    passed=all(checks.values())
    result={'schema':'agi.reference-revaluation-recurrence.v1',
        'contract_status':'CONTRACT' if passed else 'RED','checks':checks,
        'behavior':{'baseline_action':ACTION_A,'prepared_action':getattr(prioritized_action,'action_id',0),
                    'disabled_action':getattr(disabled_action,'action_id',0),
                    'remote_action':getattr(remote_action,'action_id',0)},
        'resource':{'resident_sites':base.population.spec.site_count,
                    'actual_evidence_rows':len(evidence),'reconcile_budget':1,
                    'reconcile_touches':prepared_touches,
                    'checkpoint_bytes':len(json.dumps(checkpoint,separators=(',',':')))},
        'evidence_delta_during_recurrence':0,'credit_delta_during_recurrence':0,
        'reference_only':True,'runtime_llm':False,'adult_attached':False,
        'graph_flip':False,'physical_direct_parity':'NOT_RUN/RED',
        'claim':'DISCREPANCY_PRIORITIZED_TARGET_FREE_RECURRENCE_CHANGES_LATER_REFERENCE_INTERVENTION',
        'remaining_red':['PHYSICAL_CONSEQUENCE_AUTHENTICATION','PRODUCTION_RESOURCE_SCHEDULE',
                         'MULTISTEP_REVALUATION_QUANTITY','DIRECT_ENDOGENOUS_HYPOTHESIS_TRANSLATION'],
        'elapsed_ms':round((time.perf_counter()-began)*1000,3)}
    print('FOUNDRY_REVALUATION_RECURRENCE '+result['contract_status']+
          ' prepared=B disabled=A evidence_delta=0 credit_delta=0 graph_flip=0')
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
