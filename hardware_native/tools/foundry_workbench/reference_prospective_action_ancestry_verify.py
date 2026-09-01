#!/usr/bin/env python3
"""Falsifier for structural custody of a prospective action selection.

This is a reference lower bound, not a belief, truth, trait, or Adult claim.  A
residently condensed prospective Recipe must shape the issued motor Occurrence.
The return remains host staged: this assay does not authenticate independence or
claim causal-difference credit.
"""
from __future__ import annotations

import copy
import json
import time

from reference_endogenous_prospection_verify import (
    DISTRACTOR, FIRST, GOAL, START, TEST_SOURCE, WRONG, one_shot_organism,
    settle, stage,
)
from reference_organism_v2 import MotorActionV2, ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1


def main():
    started=time.perf_counter();checks={}
    organism=one_shot_organism(PopulationSpecV1(32768,2,4,42,8))
    stage(organism,START,GOAL,(FIRST,DISTRACTOR),TEST_SOURCE)
    action=organism.tick()
    recipe=next(iter(organism.cognition._prospective_recipes.values()))
    occurrence=next((row for row in organism.population.occurrences
                     if row.identity==action.population_occurrence),None)
    expected=organism._motor_action_features(
        action.action_id,action.state_before,action.prospective_recipe,
        action.prospective_snapshot,action.prospective_context_signature)

    checks['resident_arbitration_recipe_structurally_shapes_action']=(
        isinstance(action,MotorActionV2)
        and action.prospective_recipe==recipe.identity
        and occurrence is not None
        and occurrence.feature_count==len(expected)
        and occurrence.sites==organism.population.signature(expected))

    pending=copy.deepcopy(organism.checkpoint())
    replay=ReferenceOrganismV2.restore(copy.deepcopy(pending))
    replay_action=replay.motor_actions[-1]
    checks['checkpoint_recomputes_pending_recipe_action_binding']=(
        replay_action.prospective_recipe==action.prospective_recipe
        and replay.digest()==organism.digest())

    corrupt=copy.deepcopy(pending)
    corrupt['motor_actions'][-1]['prospective_recipe']+=1
    try:
        ReferenceOrganismV2.restore(corrupt)
        corrupt_refused=False
    except ValueError as exc:
        corrupt_refused=str(exc)=='organism:checkpoint_motor_action_commitment'
    checks['corrupt_checkpoint_ancestry_is_refused']=corrupt_refused

    tampered=ReferenceOrganismV2.restore(copy.deepcopy(pending))
    tampered_action=tampered.motor_actions[-1]
    tampered_action.prospective_recipe+=1
    before=tampered.digest()
    try:
        settle(tampered,tampered_action,TEST_SOURCE,WRONG,1,True)
        tamper_refused=False
    except ValueError as exc:
        tamper_refused=str(exc)=='organism:motor_action_commitment'
    checks['live_ancestry_tamper_refuses_atomically']=(
        tamper_refused and tampered.digest()==before and not tampered_action.settled)

    learned=settle(organism,action,TEST_SOURCE,WRONG,1,True)
    checks['host_signed_settlement_can_coexist_with_wrong_prediction']=(
        learned.get('host_signed_prospective_settlement',0)>0
        and recipe.shadow_counter==1
        and recipe.counter_sources==(TEST_SOURCE,))
    checks['positive_action_effect_does_not_make_prediction_true']=(
        organism.world_state==(WRONG,)
        and organism.cognition.plan((START,),(GOAL,)).status==0)
    checks['settlement_does_not_self_confirm_prediction']=(
        recipe.shadow_credit==1 and recipe.shadow_counter==1)

    elapsed_ms=(time.perf_counter()-started)*1000
    checks['small_cpu_assay_completes_under_60_seconds']=elapsed_ms<60000
    contract_ok=all(checks.values())
    result={
        'schema':'0x1.reference-prospective-action-ancestry.v1',
        'contract_status':'CONTRACT' if contract_ok else 'RED',
        'checks':checks,
        'runtime_llm':False,'adult_attached':False,'reference_only':True,
        'graph_flip':False,'physical_direct_parity':'NOT_RUN/RED',
        'production_ir':'ResidentRecipeIrProgram.vcurrent',
        'translation_status':'UNDEFINED',
        'consequence_ingress':'HOST_STAGED_REFERENCE/PHYSICAL_AUTH_RED',
        'contract_scope':'SMALL_REFERENCE_PROSPECTIVE_RECIPE_ACTION_BINDING',
        'claim':'SMALL_DETERMINISTIC_REFERENCE_RESIDENT_ARBITRATION_BINDS_A_PROSPECTIVE_RECIPE_TO_A_MOTOR_SIGNATURE',
        'candidate_authority':'HOST_STAGED_STATE_TARGET_AFFORDANCES_AND_DEVELOPMENT',
        'credit_status':'HOST_SIGNED_ELIGIBILITY_SETTLEMENT/CAUSAL_DIFFERENCE_RED',
        'checkpoint_authentication':'STRUCTURAL_RECOMPUTE_ONLY/AUTHENTICATION_RED',
        'non_claims':['BELIEF','TRUTH','TRUST','TRAIT','VIRTUE','CHEMISTRY'],
        'remaining_red':['AUTONOMOUS_GOAL_FORMATION','RESIDENT_COMMIT_SUSPEND_REJECT',
                         'PHYSICAL_CONSEQUENCE_AUTHENTICATION','CAUSAL_DIFFERENCE_CREDIT',
                         'CHECKPOINT_AUTHENTICATION','GENERAL_CONSEQUENCE_TRANSACTIONALITY',
                         'CONTACT_AND_MOTOR_ACTION_RESOURCE_BOUNDS','PRODUCTION_TRANSLATION'],
        'elapsed_ms':round(elapsed_ms,3),
    }
    print('FOUNDRY_PROSPECTIVE_RECIPE_ACTION_BINDING '+result['contract_status']+
          ' reference_only=1 adult=0 graph_flip=0')
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if contract_ok else 1)


if __name__=='__main__':main()
