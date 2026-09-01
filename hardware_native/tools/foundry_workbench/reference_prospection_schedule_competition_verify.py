#!/usr/bin/env python3
"""Compare target-free rolling recurrence with active on-demand prospection."""
from __future__ import annotations

import copy
import inspect
import json
import time

from reference_organism_v2 import MotorActionV2,ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
import reference_endogenous_prospection_verify as p

DECOYS=8


def live_edge(o,state,target,action,source):
    p.stage(o,state,target,(action,),source)
    issued=o.tick()
    if not isinstance(issued,MotorActionV2) or issued.action_id!=action:
        raise AssertionError('schedule:resident_action')
    p.settle(o,issued,source,target)


def lived_population(target=(p.START,p.MID,p.GOAL),decoy_base=1000):
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    start,middle,goal=target
    live_edge(o,start,middle,p.FIRST,70001)
    live_edge(o,middle,goal,p.SECOND,70002)
    for index in range(DECOYS):
        s=decoy_base+index*3;m=s+1;g=s+2
        live_edge(o,s,m,72000+index*2,71000+index*2)
        live_edge(o,m,g,72001+index*2,71001+index*2)
    return o


def decide(o,target,source):
    start,_middle,goal=target
    p.stage(o,start,goal,(p.FIRST,p.DISTRACTOR),source)
    began=time.perf_counter_ns();action=o.tick();elapsed_us=(time.perf_counter_ns()-began)//1000
    return action,elapsed_us


def main():
    began=time.perf_counter();checks={};target=(p.START,p.MID,p.GOAL)
    base=lived_population(target);checkpoint=copy.deepcopy(base.checkpoint())
    evidence=copy.deepcopy(base.cognition._evidence);credit=base.population.credit_events

    awake=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    checks['awake_has_no_prebuilt_target_recipe']=not awake.cognition._prospective_recipes
    awake_action,awake_us=decide(awake,target,73001)
    awake_recipe=awake.last_prospective_recipe;awake_work=awake.last_prospective_touches

    rolling=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    rolling_started=time.perf_counter_ns();rolling_built=rolling._reconcile_low_pressure(1)
    rolling_reconcile_us=(time.perf_counter_ns()-rolling_started)//1000
    rolling_work=rolling.cognition.last_reconcile_touches
    rolling_action,rolling_decision_us=decide(rolling,target,73002)

    burst=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    burst_started=time.perf_counter_ns();burst_built=burst._reconcile_low_pressure(64)
    burst_reconcile_us=(time.perf_counter_ns()-burst_started)//1000
    burst_work=burst.cognition.last_reconcile_touches
    burst_action,burst_decision_us=decide(burst,target,73003)

    checks['all_schedules_use_same_actual_evidence']=(
        awake.cognition._evidence==rolling.cognition._evidence==burst.cognition._evidence
        and evidence==base.cognition._evidence)
    checks['schedule_work_mints_no_actual_credit']=(
        rolling.population.credit_events==burst.population.credit_events==credit)
    checks['rolling_target_free_budget_builds_one_recipe']=(
        rolling_built==1 and len(rolling.cognition._prospective_recipes)==1)
    checks['quiet_burst_builds_more_resident_hypotheses']=(
        burst_built>rolling_built and len(burst.cognition._prospective_recipes)==DECOYS+1)
    checks['all_schedules_nominate_same_real_intervention']=(
        all(isinstance(action,MotorActionV2) and action.action_id==p.FIRST
            for action in (awake_action,rolling_action,burst_action)))
    checks['on_demand_uses_less_total_touched_work_here']=(
        awake_work < rolling_work+rolling.last_prospective_touches
        and awake_work < burst_work+burst.last_prospective_touches)
    checks['schedule_hook_has_no_goal_or_target_argument']=(
        tuple(inspect.signature(rolling._reconcile_low_pressure).parameters)==('budget',))

    restored=ReferenceOrganismV2.restore(copy.deepcopy(rolling.checkpoint()))
    checks['checkpoint_preserves_schedule_result_and_frontier']=(
        restored.cognition.digest()==rolling.cognition.digest()
        and restored.cognition._reconcile_frontier==rolling.cognition._reconcile_frontier)

    high_target=(5000,5001,5002)
    permuted=lived_population(high_target,1000);permuted._reconcile_low_pressure(1)
    target_key=((high_target[0],),(high_target[2],))
    prebuilt=target_key in permuted.cognition._prospective_recipes
    permuted_action,_=decide(permuted,high_target,73004)
    checks['topology_permutation_prevents_rolling_oracle']=(
        not prebuilt and isinstance(permuted_action,MotorActionV2)
        and permuted_action.action_id==p.FIRST and permuted.last_prospective_recipe!=0)

    try:
        rolling.cognition._reconcile_local(object(),rolling.tick_count,1);host_refused=False
    except ValueError:
        host_refused=True
    checks['direct_host_schedule_authority_refused']=host_refused
    checks['no_sleep_or_dream_api']=not any(hasattr(base,name) for name in ('sleep','dream','replay'))

    passed=all(checks.values())
    result={
        'schema':'agi.reference-prospection-schedule-competition.v1',
        'contract_status':'CONTRACT' if passed else 'RED','checks':checks,
        'selected_policy_for_this_workload':'awake_on_demand',
        'selection_basis':'same_behavior_lower_deterministic_touched_work',
        'schedules':{
            'awake_on_demand':{'recipes':1,'decision_us':awake_us,'touched_work':awake_work},
            'rolling_local':{'recipes':rolling_built,'reconcile_us':rolling_reconcile_us,
                             'decision_us':rolling_decision_us,
                             'touched_work':rolling_work+rolling.last_prospective_touches},
            'quiet_burst':{'recipes':burst_built,'reconcile_us':burst_reconcile_us,
                           'decision_us':burst_decision_us,
                           'touched_work':burst_work+burst.last_prospective_touches}},
        'quantity':{'resident_sites':base.population.spec.site_count,'lived_edges':2*(DECOYS+1),
                    'schedule_budget_rolling':1,'schedule_budget_burst':64,
                    'checkpoint_bytes':len(json.dumps(checkpoint,separators=(',',':')))},
        'actual_credit_delta':0,'runtime_llm':False,'reference_only':True,
        'adult_attached':False,'graph_flip':False,'physical_direct_parity':'NOT_RUN/RED',
        'claim':'REFERENCE_SCHEDULE_ECONOMICS_NOT_SLEEP_OR_GENERAL_OPTIMALITY',
        'remaining_red':['PRODUCTION_SCHEDULE_SELECTION','RESOURCE_PRESSURE_TRANSFER',
                         'DIRECT_ENDOGENOUS_HYPOTHESIS_TRANSLATION'],
        'elapsed_ms':round((time.perf_counter()-began)*1000,3)}
    print('FOUNDRY_PROSPECTION_SCHEDULE_COMPETITION '+result['contract_status']+
          ' selected=awake_on_demand sleep=0 graph_flip=0')
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__=='__main__':main()
