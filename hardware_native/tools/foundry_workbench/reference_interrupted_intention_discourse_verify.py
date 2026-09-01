#!/usr/bin/env python3
"""Falsifier for cue-recalled unfinished intention entering live discourse.

The organism first acts on a learned three-step route.  An unrelated feature is
then added to the intermediate world state.  Exact-coordinate replanning cannot
use that enriched state.  The remaining route may reach speech only through a
resident unfinished intention, recalled by its learned focal cue without a host
retrieval request.  Exact executable action must still outrank narration.
"""
from __future__ import annotations

import copy
import json
import time

import reference_endogenous_prospection_verify as p
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1


INTERRUPTION_FEATURE=0x1A7E22
NONFOCAL_FEATURE=0x1A7E23
COMPETING_CUE_A=0x1A7E24
COMPETING_CUE_B=0x1A7E25


def interrupted_route(spec):
    o=ReferenceOrganismV2(spec)
    p.live_edge(o,p.START,p.MID,p.FIRST,0xFC01,1)
    p.live_edge(o,p.MID,p.DEEP_MID,p.FIFTH,0xFC02,1)
    p.live_edge(o,p.DEEP_MID,p.GOAL,p.SIXTH,0xFC03,1)
    p.teach_prospective_expression(o)
    o.contact(CONTACT_BODY_STATE,(p.BODY_MARKER,),0xFC04,True,True)
    p._partner(o,p.PARTNER_A,0xFC05)
    p.stage(o,p.START,p.GOAL,(p.FIRST,),0xFC06)
    first=o.tick()
    if not isinstance(first,MotorActionV2) or first.action_id!=p.FIRST:
        raise AssertionError('interrupted_intention:first_action')
    p.settle(o,first,first.source,p.MID,1,True)
    return o


def competing_routes(spec,winner_effect:int,runner_effect:int):
    """Live two independently started routes into one focal resumption cue."""
    o=ReferenceOrganismV2(spec)
    cue_a=(p.MID,COMPETING_CUE_A);cue_b=(p.MID,COMPETING_CUE_B);goal=(p.GOAL,)
    def live_edge(state,next_state,action,source,effect):
        o.contact(CONTACT_WORLD_STATE,tuple(state),source,True,True)
        o.contact(CONTACT_BODY_TARGET,tuple(next_state),source+0x100,True,True)
        o.contact(CONTACT_AFFORDANCES,(action,),source+0x200,True,True)
        issued=o.tick()
        if not isinstance(issued,MotorActionV2) or issued.action_id!=action:
            raise AssertionError('interrupted_intention:competing_live_edge')
        o.contact(CONTACT_MOTOR_CONSEQUENCE,
                  (issued.ticket,int(effect),len(next_state),*next_state),
                  issued.source,True,True)
    live_edge((p.START,),cue_a,p.FIRST,0xFD01,1)
    live_edge(cue_a,goal,p.SECOND,0xFD02,int(winner_effect))
    live_edge((p.WRONG,),cue_b,p.THIRD,0xFD03,1)
    live_edge(cue_b,goal,p.FOURTH,0xFD04,int(runner_effect))
    p.teach_prospective_expression(o)
    o.contact(CONTACT_BODY_STATE,(p.BODY_MARKER,),0xFD05,True,True)
    p._partner(o,p.PARTNER_A,0xFD06)
    for start,cue,action,source in (
            ((p.START,),cue_a,p.FIRST,0xFD10),
            ((p.WRONG,),cue_b,p.THIRD,0xFD20)):
        o.contact(CONTACT_WORLD_STATE,start,source,True,True)
        o.contact(CONTACT_BODY_TARGET,goal,source+1,True,True)
        o.contact(CONTACT_AFFORDANCES,(action,),source+2,True,True)
        first=o.tick()
        if not isinstance(first,MotorActionV2) or first.action_id!=action:
            raise AssertionError('interrupted_intention:competing_first_action')
        o.contact(CONTACT_MOTOR_CONSEQUENCE,
                  (first.ticket,1,len(cue),*cue),first.source,True,True)
    if len(o.cognition._prospective_intentions)!=2:
        raise AssertionError('interrupted_intention:competing_retention')
    return o


def main():
    started=time.perf_counter();checks={};spec=PopulationSpecV1(32768,2,4,42,8)

    o=interrupted_route(spec)
    checks['first_action_binds_sparse_unfinished_intention']=(
        bool(getattr(o.cognition,'_prospective_intentions',{}))
        and not hasattr(o,'transcript') and not hasattr(o,'context_window'))
    checkpoint=copy.deepcopy(o.checkpoint())

    # The extra feature defeats exact-state replanning.  A matching learned cue
    # may recall the unfinished route, but cannot authorize motor generalization.
    o.contact(CONTACT_WORLD_STATE,(p.MID,INTERRUPTION_FEATURE),0xFC10,True,True)
    o.contact(CONTACT_AFFORDANCES,(p.DISTRACTOR,),0xFC11,True,True)
    recalled=o.tick();surface=bytes(recalled.payload) if isinstance(recalled,ActionV2) else b''
    checks['focal_overlap_recalls_remaining_intention_into_public_discourse']=(
        isinstance(recalled,ActionV2)
        and surface==b'consider deeper then finish goal'
        and not o.cognition.satisfies((p.MID,INTERRUPTION_FEATURE),(p.GOAL,)))
    checks['partial_cue_cannot_generalize_motor_transition']=(
        isinstance(recalled,ActionV2) and not o.motor_actions[-1].state_before==(p.MID,INTERRUPTION_FEATURE))

    # No overlapping learned cue: the same retained intention must stay silent.
    nonfocal=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    nonfocal.contact(CONTACT_WORLD_STATE,(NONFOCAL_FEATURE,),0xFC20,True,True)
    nonfocal.contact(CONTACT_AFFORDANCES,(p.DISTRACTOR,),0xFC21,True,True)
    absent=nonfocal.tick()
    checks['nonfocal_interruption_does_not_recall_intention']=(
        not isinstance(absent,ActionV2) and not nonfocal.actions
        and nonfocal.cognition.last_intention_touches==0)

    # Exact intermediate state plus an available next action keeps embodied action
    # ahead of narrating the route.
    executable=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    executable.contact(CONTACT_WORLD_STATE,(p.MID,),0xFC30,True,True)
    executable.contact(CONTACT_AFFORDANCES,(p.FIFTH,),0xFC31,True,True)
    resumed=executable.tick()
    checks['exact_cue_reentry_executes_before_speaking']=(
        isinstance(resumed,MotorActionV2) and resumed.action_id==p.FIFTH
        and not executable.actions)

    # The pending occurrence, not an external reminder, survives checkpoint.
    restored=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    restored.contact(CONTACT_WORLD_STATE,(p.MID,INTERRUPTION_FEATURE),0xFC40,True,True)
    restored.contact(CONTACT_AFFORDANCES,(p.DISTRACTOR,),0xFC41,True,True)
    replay=restored.tick()
    checks['checkpoint_preserves_same_unfinished_individual']=(
        isinstance(replay,ActionV2) and replay.payload==recalled.payload
        and bool(restored.cognition.checkpoint().get('prospective_intentions')))

    # A recalled route remains source-qualified; removing any lived route source
    # prevents the retained occurrence from becoming an answer cache.
    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(0xFC02,),0xFC50,True,True)
    withdrawn.contact(CONTACT_WORLD_STATE,(p.MID,INTERRUPTION_FEATURE),0xFC51,True,True)
    withdrawn.contact(CONTACT_AFFORDANCES,(p.DISTRACTOR,),0xFC52,True,True)
    withdrawn_action=withdrawn.tick()
    checks['source_withdrawal_blocks_cue_retrieval']=(
        not isinstance(withdrawn_action,ActionV2) and not withdrawn.actions)

    # Executing both remaining bound edges spends the occurrence.  A completed
    # intention cannot leak back into later discourse as a stale commission error.
    completed=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    completed.contact(CONTACT_WORLD_STATE,(p.MID,),0xFC60,True,True)
    completed.contact(CONTACT_AFFORDANCES,(p.FIFTH,),0xFC61,True,True)
    second=completed.tick()
    if not isinstance(second,MotorActionV2):raise AssertionError('interrupted_intention:second_action')
    p.settle(completed,second,second.source,p.DEEP_MID,1,True)
    completed.contact(CONTACT_AFFORDANCES,(p.SIXTH,),0xFC62,True,True)
    third=completed.tick()
    if not isinstance(third,MotorActionV2):raise AssertionError('interrupted_intention:third_action')
    p.settle(completed,third,third.source,p.GOAL,1,True)
    checks['completion_retires_unfinished_occurrence']=(
        not completed.cognition._prospective_intentions
        and not completed.cognition.checkpoint()['prospective_intentions'])

    contradicted=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    contradicted.contact(CONTACT_WORLD_STATE,(p.MID,),0xFC70,True,True)
    contradicted.contact(CONTACT_AFFORDANCES,(p.FIFTH,),0xFC71,True,True)
    attempted=contradicted.tick()
    if not isinstance(attempted,MotorActionV2):raise AssertionError('interrupted_intention:contradiction_action')
    p.settle(contradicted,attempted,attempted.source,p.WRONG,-1,True)
    checks['contradictory_return_retires_bound_intention']=(
        not contradicted.cognition._prospective_intentions)

    # Two learned unfinished routes share the same focal cue. Independently
    # returned consequence strength, not insertion order, gates the unique route;
    # exact equality remains unresolved, and withdrawing the winner reveals the
    # still-lawful runner-up instead of deleting the ecology.
    competing=competing_routes(spec,3,1)
    competing_checkpoint=copy.deepcopy(competing.checkpoint())
    competing.contact(CONTACT_WORLD_STATE,(p.MID,COMPETING_CUE_A,COMPETING_CUE_B),0xFD30,True,True)
    competing.contact(CONTACT_AFFORDANCES,(p.DISTRACTOR,),0xFD31,True,True)
    preferred=competing.tick()
    checks['returned_consequence_gates_one_competing_intention']=(
        isinstance(preferred,ActionV2) and bytes(preferred.payload)==b'verify goal')

    tie=competing_routes(spec,2,2)
    tie.contact(CONTACT_WORLD_STATE,(p.MID,COMPETING_CUE_A,COMPETING_CUE_B),0xFD40,True,True)
    tie.contact(CONTACT_AFFORDANCES,(p.DISTRACTOR,),0xFD41,True,True)
    tied=tie.tick()
    checks['equal_competing_intentions_refuse_arbitrary_winner']=(
        tied is None and tie.information_need==(2,p.SECOND,p.FOURTH))

    fallback=ReferenceOrganismV2.restore(competing_checkpoint)
    fallback.contact(CONTACT_WITHDRAW_SOURCE,(0xFD02,),0xFD50,True,True)
    fallback.contact(CONTACT_WORLD_STATE,(p.MID,COMPETING_CUE_A,COMPETING_CUE_B),0xFD51,True,True)
    fallback.contact(CONTACT_AFFORDANCES,(p.DISTRACTOR,),0xFD52,True,True)
    runner_up=fallback.tick()
    checks['winner_source_withdrawal_reveals_runner_up']=(
        isinstance(runner_up,ActionV2) and bytes(runner_up.payload)==b'confirm goal')
    checks['visible_discussion_improvement']=(
        checks['focal_overlap_recalls_remaining_intention_into_public_discourse']
        and checks['partial_cue_cannot_generalize_motor_transition']
        and checks['nonfocal_interruption_does_not_recall_intention']
        and checks['returned_consequence_gates_one_competing_intention']
        and checks['winner_source_withdrawal_reveals_runner_up'])

    result={
        'schema':'cyber-lagoon.reference-interrupted-intention-discourse.v1',
        'pass':all(checks.values()),'checks':checks,'runtime_llm':False,
        'graph_flip':False,
        'observed':{'recalled_surface':surface.decode(errors='replace')},
        'claim':'UNFINISHED_RESIDENT_INTENTION_SURVIVES_INTERRUPTION_AND_CUE_REENTERS_LEARNED_DISCOURSE_WITHOUT_HOST_RETRIEVAL_REFERENCE_ONLY',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print('FOUNDRY_INTERRUPTED_INTENTION_DISCOURSE '+('GREEN' if result['pass'] else 'RED')+
          f" surface={result['observed']['recalled_surface']!r}")
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if result['pass'] else 1)


if __name__=='__main__':main()
