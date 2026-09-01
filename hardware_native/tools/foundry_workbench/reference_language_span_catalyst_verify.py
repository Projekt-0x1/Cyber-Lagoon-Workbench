#!/usr/bin/env python3
"""Single higher-language closure -> nonlinguistic prospective cognition falsifier."""
from __future__ import annotations
import copy,json,time

import reference_language_guided_cognition_verify as base
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

REL_NEXT=62001
START,MID,GOAL=900,9900,9999
STEP_INSPECT,STEP_TEST=7001,7134
GROUND_INSPECT,GROUND_TEST=84001,84002
TEACH=86001
WORLD1,WORLD2,WORLD3=96001,96002,96003
FULL='zoe inspects the relay. then bob tests the target.'
CHILD1='zoe inspects the relay.'
CHILD2='bob tests the target.'

def u(text):return tuple(text.encode())

def scene(o,atoms,source):
    return o.contact(CONTACT_SCENE,(7,base.CTX,len(atoms),*atoms),source,True,True)

def add_mid(o):
    base.feat(o,MID,(65,66,67),8020)
    sources=[]
    for src in (12000+MID,13000+MID):sources.append(base.name(o,MID,'relay',src))
    return tuple(sources)

def teach_span(o):
    sources=[]
    pairs=(
      ((base.BOB,base.TEST,base.VALVE),(base.REMOTE,base.INSPECT,base.SENSOR),'bob tests the valve. then zoe inspects the sensor.'),
      ((base.REMOTE,base.INSPECT,base.SENSOR),(base.BOB,base.TEST,base.VALVE),'zoe inspects the sensor. then bob tests the valve.'),
    )
    for i,(left_atoms,right_atoms,text) in enumerate(pairs):
        src=87000+i
        left=scene(o,left_atoms,src+100);right=scene(o,right_atoms,src+200)
        o.contact(CONTACT_SCENE_LINK,(left,right,REL_NEXT),src,True,True)
        o.contact(CONTACT_DISCOURSE_SURFACE,u(text),src,True,True);sources.append(src)
    if o.language.span_template(REL_NEXT,2) is None:raise AssertionError('span')
    return tuple(sources)

def prepare():
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    sources=list(base.train_language(o));sources.extend(add_mid(o));sources.extend(teach_span(o))
    inspect_units=o.language.lexeme(base.INSPECT);test_units=o.language.lexeme(base.TEST)
    inspect_lid=o.language.lexeme_identity(base.INSPECT,inspect_units);test_lid=o.language.lexeme_identity(base.TEST,test_units)
    if not o._ground_language_action_recruitment(inspect_lid,STEP_INSPECT,GROUND_INSPECT,1,True):raise AssertionError('inspect grounding')
    if not o._ground_language_action_recruitment(test_lid,STEP_TEST,GROUND_TEST,1,True):raise AssertionError('test grounding')
    return o,tuple(sources)

def stage(o,state,source):
    o.contact(CONTACT_WORLD_STATE,(state,),source,True,True)
    o.contact(CONTACT_BODY_TARGET,(GOAL,),7001,True,True)
    o.contact(CONTACT_AFFORDANCES,(STEP_INSPECT,STEP_TEST),7002,True,True)

def settle(o,action,source,next_state,independent=True,effect=1):
    return o.contact(CONTACT_MOTOR_CONSEQUENCE,(action.ticket,effect,1,next_state),source,True,independent)

def main():
    started=time.perf_counter();checks={};o,language_sources=prepare();stage(o,START,WORLD1)
    span=o.language.invert_span(u(FULL));checks['higher_surface_unfolds_one_span']=len(span)==1 and tuple(bytes(x).decode() for x in span[0].children)==(CHILD1,CHILD2)
    children=[o.language.invert_surface(child) for child in span[0].children]
    checks['heldout_children_bind_existing_language_math']=(len(children[0])==1 and children[0][0].atoms==(base.REMOTE,base.INSPECT,MID) and len(children[1])==1 and children[1][0].atoms==(base.BOB,base.TEST,GOAL))
    checks['first_counterfactual_differs']=o._exploration_candidate()==STEP_TEST
    first_assert=o.contact(CONTACT_SOURCE_UTTERANCE,u(FULL),TEACH,True,True)
    checks['one_surface_creates_one_transient_prospective_closure']=first_assert>0 and len(o._prospective_source_closures)==1
    closure=next(iter(o._prospective_source_closures.values()))
    checks['closure_is_compact_math_not_surface_cache']=(closure.cursor==0 and tuple(h.action_id for h in closure.hypotheses)==(STEP_INSPECT,STEP_TEST) and not hasattr(closure,'surface') and not hasattr(closure,'text'))
    first=o.tick();checks['first_step_is_language_causal']=isinstance(first,MotorActionV2) and first.action_id==STEP_INSPECT and first.source_counterfactual_action==STEP_TEST and first_assert in first.source_assertion_ids
    first_learned=settle(o,first,WORLD1,MID,True,1)
    checks['actual_first_return_advances_transient_closure']=first_learned.get('prospective_next_assertion',0)>0 and next(iter(o._prospective_source_closures.values())).cursor==1
    checks['second_counterfactual_flips']=o._exploration_candidate()==STEP_INSPECT
    inflight=copy.deepcopy(o.checkpoint());state_text=json.dumps(inflight,sort_keys=True,separators=(',',':'))
    checks['checkpoint_keeps_current_computation_without_instruction_transcript']=FULL not in state_text and CHILD1 not in state_text and CHILD2 not in state_text and len(inflight.get('prospective_source_closures',()))==1
    restored=ReferenceOrganismV2.restore(copy.deepcopy(inflight));restored_second=restored.tick()
    checks['inflight_checkpoint_replays_second_step']=isinstance(restored_second,MotorActionV2) and restored_second.action_id==STEP_TEST
    second=o.tick();checks['second_step_waits_for_real_intermediate_state']=isinstance(second,MotorActionV2) and second.action_id==STEP_TEST and second.source_counterfactual_action==STEP_INSPECT
    settle(o,second,WORLD1,GOAL,True,1)
    checks['prospective_language_closure_dies_after_completion']=not o._prospective_source_closures
    checks['one_lived_route_is_not_authoritative_truth']=len(o.cognition._evidence)==2 and not o.cognition.edges()

    # Remove all language and cue->motor grounding. The lived one-shot route remains.
    o.contact(CONTACT_WITHDRAW_SOURCE,(TEACH,),88001,True,True)
    for i,source in enumerate(language_sources):o.contact(CONTACT_WITHDRAW_SOURCE,(source,),88100+i,True,True)
    o.contact(CONTACT_WITHDRAW_SOURCE,(GROUND_INSPECT,),88901,True,True);o.contact(CONTACT_WITHDRAW_SOURCE,(GROUND_TEST,),88902,True,True)
    checks['teaching_surface_is_unavailable']=o.language.invert_span(u(FULL))==() and o.language.invert_surface(u(CHILD1))==()

    # A new actual episode causes resident one-shot fragments to condense into a
    # prospective Recipe; no language assertion participates.
    stage(o,START,WORLD2);shadow_first=o.tick();recipe_id=o.last_prospective_recipe
    checks['one_shot_lived_route_condenses_prospective_recipe']=(isinstance(shadow_first,MotorActionV2) and shadow_first.action_id==STEP_INSPECT and not shadow_first.source_assertion_ids and recipe_id>0 and bool(o.last_prospective_occurrences))
    before_shadow_return=copy.deepcopy(o.checkpoint())
    settle(o,shadow_first,WORLD2,MID,True,1)
    continued=copy.deepcopy(o.checkpoint());continued_restored=ReferenceOrganismV2.restore(copy.deepcopy(continued))
    continued_second=continued_restored.tick()
    checks['checkpoint_preserves_selected_computation_after_recipe_invalidation']=(
        not continued_restored.cognition._prospective_recipes
        and isinstance(continued_second,MotorActionV2)
        and continued_second.action_id==STEP_TEST)
    retargeted=ReferenceOrganismV2.restore(copy.deepcopy(continued))
    retargeted.contact(CONTACT_BODY_TARGET,(START,),7001,True,True)
    retargeted_action=retargeted.tick()
    checks['changed_body_target_cancels_selected_computation_suffix']=(
        retargeted.last_prospective_recipe==0
        and (not isinstance(retargeted_action,MotorActionV2)
             or retargeted_action.action_id!=STEP_TEST))
    # The settled motor Occurrence is now retired after its return; the future-causal
    # continuation authority is the unfinished prospective intention. Corrupt that
    # resident identity rather than requiring obsolete settled motor snapshot bytes.
    corrupt_intention=copy.deepcopy(continued)
    intentions=corrupt_intention['cognition'].get('prospective_intentions',())
    intention_corruption_refused=False
    if len(intentions)==1:
        intentions[0]['identity']=int(intentions[0]['identity'])+1
        try:
            ReferenceOrganismV2.restore(corrupt_intention)
        except ValueError as exc:
            intention_corruption_refused=str(exc)=='cognition:prospective_intention_checkpoint'
    checks['selected_computation_intention_corruption_refused']=intention_corruption_refused
    yoked=ReferenceOrganismV2.restore(before_shadow_return);yoked_first=yoked.motor_actions[-1]
    settle(yoked,yoked_first,WORLD2,MID,False,1);yoked.cognition._prospective_recipes.clear()
    yoked_second=yoked.tick()
    checks['yoked_return_cannot_advance_selected_computation']=(
        not isinstance(yoked_second,MotorActionV2) or yoked_second.action_id!=STEP_TEST)
    shadow_second=o.tick()
    checks['prospective_recipe_unfolds_suffix_from_actual_mid_state']=(isinstance(shadow_second,MotorActionV2) and shadow_second.action_id==STEP_TEST and not shadow_second.source_assertion_ids and o.last_prospective_recipe==recipe_id)
    settle(o,shadow_second,WORLD2,GOAL,True,1)
    checks['second_lived_route_promotes_ordinary_nonlinguistic_plan']=(len(o.cognition.edges())==2 and o.cognition.plan((START,),(GOAL,)).actions==(STEP_INSPECT,STEP_TEST))

    # Once authoritative, the compact shadow Recipe is dispensable.
    o.cognition._prospective_recipes.clear();stage(o,START,WORLD3);auth_first=o.tick()
    checks['authoritative_plan_survives_shadow_and_language_lesion']=isinstance(auth_first,MotorActionV2) and auth_first.action_id==STEP_INSPECT and not auth_first.source_assertion_ids and o.last_prospective_recipe==0
    settle(o,auth_first,WORLD3,MID,True,1);auth_second=o.tick()
    checks['authoritative_second_step_survives']=isinstance(auth_second,MotorActionV2) and auth_second.action_id==STEP_TEST

    # Yoked/non-difference return must destroy, never advance, the current language closure.
    y,_=prepare();stage(y,START,WORLD1);y.contact(CONTACT_SOURCE_UTTERANCE,u(FULL),TEACH,True,True);ya=y.tick();settle(y,ya,WORLD1,MID,False,1)
    checks['yoked_return_cancels_prospective_language_closure']=not y._prospective_source_closures and not any(r.prospective_step==1 and r.active for r in y.source_assertions)

    out={'schema':'agi.reference-language-span-catalyst.v1','pass':all(checks.values()),'checks':checks,'reference_only':True,'runtime_llm':False,'graph_flip':False,'physical_direct_parity':'NOT_RUN/RED','claim':'ONE_HIGHER_LANGUAGE_CLOSURE_BOOTSTRAPS_NONLINGUISTIC_PROSPECTIVE_RECIPE_REFERENCE_ONLY','resource':{'span_inverse_touches':o.language.last_lookup_touches,'prospective_recipe':recipe_id,'retained_occurrences':len(o.population.occurrences),'resident_sites':o.population.spec.site_count},'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_LANGUAGE_SPAN_CATALYST '+('GREEN' if out['pass'] else 'RED')+' one_surface=1 transient_sequence=1 prospective_recipe=1 authoritative_after_replay=1')
    print(json.dumps(out,indent=2,sort_keys=True));raise SystemExit(0 if out['pass'] else 1)
if __name__=='__main__':main()
