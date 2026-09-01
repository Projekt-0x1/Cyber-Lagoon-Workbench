#!/usr/bin/env python3
"""Whole-organism receipt separating prospective execution/prediction from consequence valence."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
from reference_endogenous_prospection_verify import (
    one_shot_organism,stage,settle,START,MID,GOAL,FIRST,SECOND,DISTRACTOR,TEST_SOURCE,
)

PRED_SOURCE=0xDD31
PRED_ACTION=0xDD32
PRED_ALT=0xDD33
PRED_STATE=(0xDD34,)
PRED_NEXT=(0xDD35,)
PRED_TARGET=(0xDD36,)


def main():
    started=time.perf_counter();checks={}
    spec=PopulationSpecV1(32768,2,4,42,8)

    # Compose the resident two-edge route from lived one-shot transitions, then execute it
    # under aversive consequence. Reaching the planned final state is route execution evidence;
    # negative valence must not erase that fact.
    o=one_shot_organism(spec)
    stage(o,START,GOAL,(FIRST,DISTRACTOR),TEST_SOURCE)
    first=o.tick()
    if not isinstance(first,MotorActionV2) or int(first.action_id)!=FIRST:
        raise AssertionError(('prospective-valence:first',first))
    first_result=settle(o,first,TEST_SOURCE,MID,-1,True)
    checks['negative_first_step_still_executes_planned_transition']=(
        tuple(first.state_after)==(MID,) and int(first.effect)<0 and bool(first.independent_consequence))

    stage(o,MID,GOAL,(SECOND,),TEST_SOURCE)
    second=o.tick()
    if not isinstance(second,MotorActionV2) or int(second.action_id)!=SECOND:
        raise AssertionError(('prospective-valence:second',second))
    second_result=settle(o,second,TEST_SOURCE,GOAL,-1,True)
    checks['negative_final_step_reaches_planned_goal']=tuple(second.state_after)==(GOAL,) and int(second.effect)<0
    checks['planned_route_completion_is_not_reward_gated']=(
        int(second_result.get('prospective_completion_observed',0))==1)
    checks['negative_valence_remains_available_separately']=(int(first.effect)<0 and int(second.effect)<0)

    # Neutral valence must still test an explicit grounded source prediction against next_state.
    p=ReferenceOrganismV2(spec)
    p.contact(CONTACT_WORLD_STATE,PRED_STATE,TEST_SOURCE,True,True)
    p.contact(CONTACT_BODY_TARGET,PRED_TARGET,TEST_SOURCE+1,True,True)
    p.contact(CONTACT_AFFORDANCES,(PRED_ACTION,PRED_ALT),TEST_SOURCE+2,True,True)
    ctx=p._source_context_signature()
    p._record_source_assertion(PRED_ACTION,PRED_SOURCE,predicted_state=PRED_NEXT)
    motor=p.tick()
    if not isinstance(motor,MotorActionV2) or int(motor.action_id)!=PRED_ACTION:
        raise AssertionError(('prospective-valence:prediction-motor',motor))
    neutral=p.contact(CONTACT_MOTOR_CONSEQUENCE,(motor.ticket,0,len(PRED_NEXT),*PRED_NEXT),TEST_SOURCE,True,True)
    checks['neutral_valence_explicit_prediction_still_calibrates']=(
        p._source_calibration(PRED_SOURCE,ctx)>0
        and int(neutral.get('typed_neutral_prediction_updates',0))==1)
    checks['neutral_prediction_calibration_does_not_require_reward']=int(motor.effect)==0

    failed=sorted(name for name,value in checks.items() if not value)
    result={
        'schema':'cyber-lagoon.prospective-execution-valence-separation.v1',
        'pass':not failed,'checks':checks,'failed':failed,
        'claim':'PROSPECTIVE_EXECUTION_GOAL_ATTAINMENT_PREDICTION_ACCURACY_AND_VALENCE_REMAIN_DISTINCT',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print('FOUNDRY_PROSPECTIVE_EXECUTION_VALENCE_SEPARATION '+('GREEN' if not failed else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if not failed else 1)

if __name__=='__main__':main()
