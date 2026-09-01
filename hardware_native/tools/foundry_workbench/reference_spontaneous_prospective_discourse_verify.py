#!/usr/bin/env python3
"""Falsifier for resident prospective thought entering ordinary public discourse.

The verifier never calls ``current_prospective_expression_plan``.  It stages
world/body/social conditions and observes ordinary ``ReferenceOrganismV2.tick``.
A learned prospective route may become speech only when its next motor action is
currently unavailable; ordinary executable action keeps priority.
"""
from __future__ import annotations

import copy
import json
import time

import reference_endogenous_prospection_verify as p
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

language_phenotype_improved = True
future_update_authority_preserved = True
visible_language_gain = 'NEGATIVE_ROUTE_COMPLETION_PRESERVES_SPONTANEOUS_PARTNER_FACING_PROSPECTIVE_SPEECH'


def single_route(spec):
    o=p.one_shot_organism(spec)
    p.teach_prospective_expression(o)
    o.contact(CONTACT_BODY_STATE,(p.BODY_MARKER,),0xE902,True,True)
    return o


def stage_partner(o,partner,affordances,source):
    p._partner(o,partner,source+1)
    p.stage(o,p.START,p.GOAL,tuple(affordances),source)


def settle_speech(o,a,effect=1,independent=True):
    return o.contact(CONTACT_CONSEQUENCE,(a.ticket,int(effect)),a.source,True,independent)


def body_competition(spec):
    o=ReferenceOrganismV2(spec)
    p.live_edge(o,p.START,p.MID,p.FIRST,0xF301,1)
    p.live_edge(o,p.MID,p.GOAL,p.SECOND,0xF302,1)
    p.live_edge(o,p.START,p.ALT_MID,p.THIRD,0xF303,1)
    p.live_edge(o,p.ALT_MID,p.GOAL,p.FOURTH,0xF304,1)
    p.teach_prospective_expression(o)
    p.learn_body_marker(o,p.FIRST,p.MID,p.BODY_A,p.BODY_A_SOURCE,1,0xF310)
    p.learn_body_marker(o,p.THIRD,p.ALT_MID,p.BODY_B,p.BODY_B_SOURCE,1,0xF320)
    return o


def main():
    started=time.perf_counter();checks={};spec=PopulationSpecV1(32768,2,4,42,8)

    # RED discriminator repaired: cognition already has a realizable two-hop route,
    # but FIRST is physically unavailable.  Ordinary tick must itself communicate.
    o=single_route(spec);stage_partner(o,p.PARTNER_A,(p.DISTRACTOR,),0xFA10)
    cp=copy.deepcopy(o.checkpoint())
    a=o.tick();surface=bytes(a.payload) if isinstance(a,ActionV2) else b''
    checks['ordinary_tick_spontaneously_expresses_blocked_prospective_route']=(
        isinstance(a,ActionV2) and surface==b'inspect middle then verify goal'
        and a.scene_identity==o.prospective_expression_announced
        and a.scene_identity not in o._scene_by_id
        and a.closure_identity>0 and a.span_identity>0
        and p.FIRST in next(iter(o.cognition._prospective_recipes.values())).actions)
    checks['prospective_speech_does_not_persist_transient_surface_tree']=(
        not hasattr(o,'hierarchy') and 'hierarchy' not in o.checkpoint())
    replay=ReferenceOrganismV2.restore(copy.deepcopy(cp));ra=replay.tick()
    checks['checkpoint_replays_exact_spontaneous_thought']=(
        isinstance(ra,ActionV2) and isinstance(a,ActionV2)
        and ra.payload==a.payload and ra.scene_identity==a.scene_identity
        and ra.template_identity==a.template_identity and ra.closure_identity==a.closure_identity
        and ra.contributors==a.contributors)
    learned=settle_speech(o,a,1,True) if isinstance(a,ActionV2) else {}
    after=o.tick()
    checks['speaking_does_not_mint_world_transition_or_repeat_itself']=(
        learned.get('credit',0)>0 and isinstance(after,MotorActionV2)
        and after.action_id==p.DISTRACTOR
        and o.prospective_expression_announced==a.scene_identity
        and not o.cognition.edges())

    # If the route can actually be executed, action has priority over narration.
    executable=single_route(spec);stage_partner(executable,p.PARTNER_A,(p.FIRST,p.DISTRACTOR),0xFA20)
    ea=executable.tick()
    checks['available_prospective_action_has_priority_over_narration']=(
        isinstance(ea,MotorActionV2) and ea.action_id==p.FIRST and not executable.actions)

    # Goal attainment is route evidence, not reward. After the same learned two-hop
    # route reaches GOAL under negative consequences, later blocking FIRST must still
    # expose the intact prospective content through ordinary partner-facing speech.
    negative=single_route(spec)
    p.stage(negative,p.START,p.GOAL,(p.FIRST,p.DISTRACTOR),0xFC10)
    negative_first=negative.tick()
    negative_first_result=(p.settle(negative,negative_first,0xFC10,p.MID,-1,True)
                           if isinstance(negative_first,MotorActionV2) else {})
    p.stage(negative,p.MID,p.GOAL,(p.SECOND,),0xFC11)
    negative_second=negative.tick()
    negative_second_result=(p.settle(negative,negative_second,0xFC11,p.GOAL,-1,True)
                            if isinstance(negative_second,MotorActionV2) else {})
    stage_partner(negative,p.PARTNER_A,(p.DISTRACTOR,),0xFC20)
    negative_speech=negative.tick()
    checks['negative_route_completion_preserves_spontaneous_prospective_language']=(
        isinstance(negative_first,MotorActionV2) and int(negative_first.effect)<0
        and isinstance(negative_second,MotorActionV2) and int(negative_second.effect)<0
        and int(negative_second_result.get('prospective_completion_observed',0))==1
        and isinstance(negative_speech,ActionV2)
        and bytes(negative_speech.payload)==b'inspect middle then verify goal')

    # No social recipient: same blocked route cannot leak into public language.
    alone=single_route(spec)
    alone.contact(CONTACT_PARTNER_CONTEXT,(0,0,0),0xFA2F,True,True)
    p.stage(alone,p.START,p.GOAL,(p.DISTRACTOR,),0xFA30)
    aa=alone.tick()
    checks['no_partner_keeps_blocked_thought_private']=(
        isinstance(aa,MotorActionV2) and aa.action_id==p.DISTRACTOR and not alone.actions)

    # Remove the developmental surface ecology while preserving prospective cognition.
    lesion=single_route(spec)
    lesion.contact(CONTACT_WITHDRAW_SOURCE,(p.PARTNER_A,),0xFA40,True,True)
    lesion.contact(CONTACT_WITHDRAW_SOURCE,(p.PARTNER_A+0x100,),0xFA41,True,True)
    stage_partner(lesion,p.PARTNER_A,(p.DISTRACTOR,),0xFA42)
    la=lesion.tick()
    checks['language_source_lesion_preserves_plan_but_abolishes_expression']=(
        lesion._resident_selected_prospective_plan() is not None
        and not isinstance(la,ActionV2)
        and isinstance(la,MotorActionV2) and la.action_id==p.DISTRACTOR)

    # Same nonlinguistic route, different learned partner ecology.
    pb=single_route(spec);stage_partner(pb,p.PARTNER_B,(p.DISTRACTOR,),0xFA50)
    ba=pb.tick()
    checks['partner_history_changes_surface_not_prospective_content']=(
        isinstance(ba,ActionV2) and bytes(ba.payload)==b'pruefen mitte danach testen ziel'
        and ba.scene_identity!=0)

    # Equal prospective population: body history alone reverses which resident thought
    # reaches public expression.  The verifier still invokes only ordinary tick().
    soma=body_competition(spec);soma_cp=copy.deepcopy(soma.checkpoint());body_outputs={}
    for body,body_source,label in ((p.BODY_A,p.BODY_A_SOURCE,'a'),(p.BODY_B,p.BODY_B_SOURCE,'b')):
        x=ReferenceOrganismV2.restore(copy.deepcopy(soma_cp))
        x.contact(CONTACT_BODY_STATE,tuple(body),int(body_source),True,True)
        stage_partner(x,p.PARTNER_A,(p.DISTRACTOR,),0xFB00+(0 if label=='a' else 16))
        act=x.tick();body_outputs[label]=b'' if not isinstance(act,ActionV2) else bytes(act.payload)
    checks['somatic_history_reverses_spontaneously_expressed_future']=(
        body_outputs['a']==b'inspect middle then verify goal'
        and body_outputs['b']==b'probe alternate then confirm goal')

    checks['no_host_speech_or_prompt_control_api']=all(not hasattr(o,name) for name in (
        'prompt','speak','enqueue_goal','say_plan','express_plan','context_window','transcript'))
    checks['observer_expression_helper_not_required_by_assay']=(
        'current_prospective_expression_plan' not in main.__code__.co_names)

    result={
        'schema':'0x1.reference-spontaneous-prospective-discourse.v1',
        'pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,
        'observed':{
            'blocked_partner_a':surface.decode(errors='replace'),
            'blocked_partner_b':bytes(ba.payload).decode(errors='replace') if isinstance(ba,ActionV2) else '',
            'body_a':body_outputs['a'].decode(errors='replace'),
            'body_b':body_outputs['b'].decode(errors='replace'),
        },
        'claim':'BLOCKED_RESIDENT_PROSPECTION_ENDOGENOUSLY_REACHES_LEARNED_PARTNER_SENSITIVE_PUBLIC_DISCOURSE_WITH_ACTION_PRIORITY_AND_SOMATIC_REVERSAL_REFERENCE_ONLY',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print('FOUNDRY_SPONTANEOUS_PROSPECTIVE_DISCOURSE '+('GREEN' if result['pass'] else 'RED')+
          f" spontaneous={result['observed']['blocked_partner_a']!r} body_b={result['observed']['body_b']!r}")
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if result['pass'] else 1)


if __name__=='__main__':main()
