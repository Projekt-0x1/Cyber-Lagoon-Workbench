#!/usr/bin/env python3
"""Language-mediated construction falsifier for the continuing reference Adult.

Raw learned utterance may nominate a nonlinguistic action, but cannot itself create
world truth.  Only later independent world consequences may create the durable
transition relation.  The learned transition must survive removal of both testimony
and the language construction that originally nominated it.
"""
from __future__ import annotations

import copy
import json
import time

from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CTX=8801
ALICE,ALICE2,BOB,REMOTE,TEST,VALVE,INSPECT,SENSOR=201,501,202,601,302,402,303,403
MOTOR_TEST,MOTOR_INSPECT=7302,8303
GROUND_TEST,GROUND_INSPECT=8051,8052
STATE=900
GOAL,WRONG=9999,9998
WORLD1,WORLD2,WORLD3=9101,9102,9103
TEACH1,TEACH2=8101,8102
BAD_TEACH,PRED_TEACH1,PRED_TEACH2=8200,8201,8202
PRED_WORLD0,PRED_WORLD1,PRED_WORLD2,PRED_WORLD3=9200,9201,9202,9203
SOMA_WORLD1,SOMA_WORLD2,SOMA_TEACH1,SOMA_TEACH2=9401,9402,8401,8402
INSTRUCTION='zoe inspects the valve.'
PREDICTION='zoe inspects the target.'


def u(text): return tuple(text.encode())


def feat(o,e,features,source):
    o.contact(CONTACT_ENTITY_FEATURES,(e,len(features),*features),source,True,True)


def name(o,e,text,source):
    o.contact(CONTACT_SCENE,(7,0,1,e),source,True,True)
    o.contact(CONTACT_SURFACE,u(text),source+1000,True,True)
    return source+1000


def clause(o,atoms,text,source):
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),source,True,True)
    o.contact(CONTACT_SURFACE,u(text),source+2000,True,True)
    return source+2000


def train_language(o):
    for e,features,source in (
        (ALICE,(11,12,13,14),8001),(ALICE2,(11,12,13,99),8003),
        (BOB,(21,22,23,24),8002),(REMOTE,(71,72,73,74),8004),
        (TEST,(33,34),8013),(VALVE,(43,44),8014),
        (INSPECT,(35,36),8015),(SENSOR,(45,46),8016),(GOAL,(55,56),8017),
    ):
        feat(o,e,features,source)
    language_sources=[]
    for e,text in ((ALICE,'alice'),(ALICE2,'alice'),(BOB,'bob'),(REMOTE,'zoe'),(TEST,'tests'),(VALVE,'valve'),(INSPECT,'inspects'),(SENSOR,'sensor'),(GOAL,'target')):
        language_sources.append(name(o,e,text,10000+e))
        language_sources.append(name(o,e,text,11000+e))
    language_sources.append(clause(o,(BOB,TEST,VALVE),'bob tests the valve.',30005))
    language_sources.append(clause(o,(BOB,TEST,VALVE),'bob tests the valve.',30006))
    language_sources.append(clause(o,(REMOTE,INSPECT,SENSOR),'zoe inspects the sensor.',30009))
    language_sources.append(clause(o,(REMOTE,INSPECT,SENSOR),'zoe inspects the sensor.',30010))
    if o.language.template(CTX,3) is None:
        raise AssertionError('template')
    test_surface=o.language.lexeme(TEST);inspect_surface=o.language.lexeme(INSPECT)
    if test_surface is None or inspect_surface is None:raise AssertionError('action lexeme')
    test_lexeme=o.language.lexeme_identity(TEST,test_surface);inspect_lexeme=o.language.lexeme_identity(INSPECT,inspect_surface)
    if not o._ground_language_action_recruitment(test_lexeme,MOTOR_TEST,GROUND_TEST,1,True):raise AssertionError('test grounding')
    if not o._ground_language_action_recruitment(inspect_lexeme,MOTOR_INSPECT,GROUND_INSPECT,1,True):raise AssertionError('inspect grounding')
    return tuple(language_sources)


def setup(o,world_source):
    o.contact(CONTACT_WORLD_STATE,(STATE,),world_source,True,True)
    o.contact(CONTACT_BODY_TARGET,(GOAL,),7001,True,True)
    o.contact(CONTACT_AFFORDANCES,(MOTOR_TEST,MOTOR_INSPECT),7002,True,True)


def teach(o,source):
    return o.contact(CONTACT_SOURCE_UTTERANCE,u(INSTRUCTION),source,True,True)


def settle(o,action,world_source,independent=True,next_state=None,effect=1):
    nxt=(STATE,GOAL) if next_state is None else tuple(next_state)
    return o.contact(CONTACT_MOTOR_CONSEQUENCE,(action.ticket,int(effect),len(nxt),*nxt),world_source,True,independent)


def main():
    started=time.perf_counter(); checks={}
    spec=PopulationSpecV1(32768,2,4,42,8)
    o=ReferenceOrganismV2(spec); language_sources=train_language(o); setup(o,WORLD1)
    checks['no_language_counterfactual_is_test']=o._exploration_candidate()==MOTOR_TEST
    demonstrated={u('bob tests the valve.'),u('zoe inspects the sensor.')}
    checks['instruction_is_heldout_recombination']=u(INSTRUCTION) not in demonstrated
    bindings=o.language.invert_surface(u(INSTRUCTION))
    checks['raw_surface_inverts_to_grounded_relation']=len(bindings)==1 and bindings[0].atoms==(REMOTE,INSPECT,VALVE)

    amb=ReferenceOrganismV2(spec);train_language(amb);setup(amb,WORLD1)
    ambiguous=amb.language.invert_surface(u('alice inspects the sensor.'))
    amb_id=amb.contact(CONTACT_SOURCE_UTTERANCE,u('alice inspects the sensor.'),TEACH1,True,True)
    amb_row=next(row for row in amb.source_assertions if row.identity==amb_id);amb_motor=amb.tick()
    checks['ambiguous_referent_preserved_as_multiple_bindings']=len(ambiguous)==2 and {row.atoms[0] for row in ambiguous}=={ALICE,ALICE2}
    checks['common_meaning_can_act_without_inventing_referent']=(
        isinstance(amb_motor,MotorActionV2) and amb_motor.action_id==MOTOR_INSPECT
        and ALICE not in amb_row.binding_atoms and ALICE2 not in amb_row.binding_atoms
        and set(amb_row.binding_atoms)=={INSPECT,SENSOR}
    )

    relations_before_instruction=len(o.recruitment.relations)
    assertion1=teach(o,TEACH1);language_recipe_touches=o.last_language_recipe_touches
    row1=next(row for row in o.source_assertions if row.identity==assertion1)
    checks['raw_testimony_derives_action_without_host_action_id']=row1.action_id==MOTOR_INSPECT and row1.language_binding!=0
    inspect_units=o.language.lexeme(INSPECT);inspect_lexeme=o.language.lexeme_identity(INSPECT,inspect_units)
    cue_morphology=o.recruitment.morphology_identity(o.population.signature((LANGUAGE_ACTION_CUE_TAG,inspect_lexeme)))
    action_morphology=o._action_recipe_morphology(MOTOR_INSPECT)
    checks['language_nominates_earned_action_recipe_family']=any(
        set(relation.morphologies)=={cue_morphology,action_morphology}
        for relation in o.recruitment.relations.values())
    checks['nomination_does_not_create_recipe_or_credit']=(
        len(o.recruitment.relations)==relations_before_instruction
        and not o.cognition.edges())
    checks['language_recipe_nomination_is_sparse']=(
        language_recipe_touches==1
        and language_recipe_touches/o.population.spec.site_count<0.001)
    checks['testimony_alone_is_not_world_relation']=not o.cognition.edges()
    first=o.tick()
    checks['language_changes_nonlinguistic_action']=isinstance(first,MotorActionV2) and first.action_id==MOTOR_INSPECT and first.source_counterfactual_action==MOTOR_TEST and first.source_assertion_ids==(assertion1,)
    learned1=settle(o,first,WORLD1,True)
    checks['first_world_return_not_enough_for_durable_transition']=not o.cognition.edges() and learned1.get('source_credit',0)>0
    o.contact(CONTACT_WITHDRAW_SOURCE,(TEACH1,),88001,True,True)

    setup(o,WORLD2); assertion2=teach(o,TEACH2); second=o.tick()
    checks['second_independent_language_guided_trial']=isinstance(second,MotorActionV2) and second.action_id==MOTOR_INSPECT and second.source_assertion_ids==(assertion2,)
    settle(o,second,WORLD2,True)
    edges=o.cognition.edges()
    checks['world_consequence_creates_nonlinguistic_relation']=len(edges)==1 and edges[0].action==MOTOR_INSPECT and edges[0].support==2 and set(edges[0].sources)=={WORLD1,WORLD2}

    # Remove the teacher and every source that supplied the learned expression.
    o.contact(CONTACT_WITHDRAW_SOURCE,(TEACH2,),88002,True,True)
    for offset,source in enumerate(language_sources,1):
        o.contact(CONTACT_WITHDRAW_SOURCE,(source,),89000+offset,True,True)
    checks['teaching_surface_is_gone']=o.language.realize(CTX,(REMOTE,INSPECT,VALVE)) is None and o.language.invert_surface(u(INSTRUCTION))==()

    setup(o,WORLD3)
    cp=copy.deepcopy(o.checkpoint())
    direct=o.tick(); restored=ReferenceOrganismV2.restore(copy.deepcopy(cp)); replay=restored.tick()
    checks['grounded_relation_survives_language_withdrawal']=(
        isinstance(direct,MotorActionV2) and direct.action_id==MOTOR_INSPECT and not direct.source_assertion_ids
        and isinstance(replay,MotorActionV2) and replay.action_id==MOTOR_INSPECT and not replay.source_assertion_ids
    )
    checks['checkpoint_preserves_nonlinguistic_result']=restored.cognition.digest()==o.cognition.digest()

    lesion=ReferenceOrganismV2.restore(copy.deepcopy(cp)); lesion.cognition._evidence.clear(); fallback=lesion.tick()
    checks['focal_nonlinguistic_lesion_removes_learned_behavior']=isinstance(fallback,MotorActionV2) and fallback.action_id==MOTOR_TEST and not fallback.source_assertion_ids

    yoked=ReferenceOrganismV2(spec); train_language(yoked); setup(yoked,WORLD1); teach(yoked,TEACH1); ya=yoked.tick(); settle(yoked,ya,WORLD1,False)
    yoked.contact(CONTACT_WITHDRAW_SOURCE,(TEACH1,),88101,True,True); setup(yoked,WORLD2)
    checks['yoked_language_guidance_does_not_create_world_relation']=not yoked.cognition.edges() and yoked._exploration_candidate()==MOTOR_TEST

    # Stronger relation-level teaching: the learned utterance names both an action
    # and the current grounded target.  The target is a prediction, not truth.
    p=ReferenceOrganismV2(spec); prediction_language_sources=train_language(p); setup(p,PRED_WORLD0)
    pred_bindings=p.language.invert_surface(u(PREDICTION))
    checks['prediction_surface_is_heldout_recombination']=u(PREDICTION) not in demonstrated and len(pred_bindings)==1 and pred_bindings[0].atoms==(REMOTE,INSPECT,GOAL)
    checks['prediction_counterfactual_is_test']=p._exploration_candidate()==MOTOR_TEST
    bad_id=p.contact(CONTACT_SOURCE_UTTERANCE,u(PREDICTION),BAD_TEACH,True,True)
    bad_row=next(row for row in p.source_assertions if row.identity==bad_id);pred_ctx=p._source_context_signature()
    checks['language_binding_carries_grounded_prediction']=bad_row.action_id==MOTOR_INSPECT and bad_row.predicted_state==(GOAL,) and bad_row.language_binding!=0
    bad_action=p.tick()
    checks['prediction_changes_action_before_truth']=isinstance(bad_action,MotorActionV2) and bad_action.action_id==MOTOR_INSPECT and not p.cognition.edges()
    bad_learned=settle(p,bad_action,PRED_WORLD0,True,(STATE,WRONG),1)
    checks['causal_credit_can_coexist_with_wrong_prediction']=bad_learned.get('source_credit',0)>0 and p._source_calibration(BAD_TEACH,pred_ctx)==-1
    checks['positive_effect_does_not_make_wrong_prediction_true']=not p.cognition.edges() and GOAL not in p.world_state
    p.contact(CONTACT_WITHDRAW_SOURCE,(BAD_TEACH,),88200,True,True)

    # A grounded predictive utterance may persist as defeasible prospective
    # support across a changed current state. It remains source-conditioned:
    # it neither writes the world model nor survives contradictory lived return.
    shadow=ReferenceOrganismV2(spec);train_language(shadow);setup(shadow,9250)
    shadow_assertion=shadow.contact(CONTACT_SOURCE_UTTERANCE,u(PREDICTION),8250,True,True)
    shadow.contact(CONTACT_WORLD_STATE,(STATE,7777),9251,True,True)
    shadow_before=copy.deepcopy(shadow.checkpoint())
    shadow_replay=ReferenceOrganismV2.restore(copy.deepcopy(shadow_before))
    shadow_action=shadow.tick();replay_action=shadow_replay.tick()
    shadow_recipe_touches=shadow.last_source_touches
    shadow_evidence_before_return=len(shadow.cognition._evidence)
    checks['instruction_recipe_unfolds_after_current_state_changes']=(
        isinstance(shadow_action,MotorActionV2) and shadow_action.action_id==MOTOR_INSPECT
        and shadow_action.source_assertion_ids==(shadow_assertion,)
        and shadow.last_source_touches==1 and not shadow.cognition._evidence
    )
    checks['shadow_recipe_index_is_rebuildable_checkpoint_exact']=(
        '_source_recipe_index' not in json.dumps(shadow_before,sort_keys=True)
        and isinstance(replay_action,MotorActionV2)
        and replay_action.action_id==shadow_action.action_id
        and replay_action.source_assertion_ids==shadow_action.source_assertion_ids
    )
    contradicted=settle(shadow,shadow_action,9251,True,(STATE,7777,WRONG),1)
    shadow.contact(CONTACT_WORLD_STATE,(STATE,8888),9252,True,True)
    shadow_fallback=shadow._exploration_candidate();after_contradiction=shadow.tick()
    checks['contradictory_actual_return_defeats_live_shadow_recipe']=(
        contradicted.get('source_credit',0)>0
        and shadow._source_calibration(8250,shadow._source_context_signature())<0
        and any(row.identity==shadow_assertion and row.active for row in shadow.source_assertions)
        and isinstance(after_contradiction,MotorActionV2)
        and after_contradiction.action_id==shadow_fallback
        and not after_contradiction.source_assertion_ids and not shadow.cognition.edges()
    )
    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(shadow_before))
    withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(8250,),88250,True,True)
    withdrawn_fallback=withdrawn._exploration_candidate();withdrawn_action=withdrawn.tick()
    checks['source_withdrawal_defeats_shadow_recipe']=(
        isinstance(withdrawn_action,MotorActionV2)
        and withdrawn_action.action_id==withdrawn_fallback
        and not withdrawn_action.source_assertion_ids
    )

    setup(p,PRED_WORLD1); good1=p.contact(CONTACT_SOURCE_UTTERANCE,u(PREDICTION),PRED_TEACH1,True,True); ga1=p.tick(); settle(p,ga1,PRED_WORLD1,True,(STATE,GOAL),1)
    checks['first_correct_prediction_calibrates_positive']=p._source_calibration(PRED_TEACH1,p._source_context_signature())==1 and not p.cognition.edges()
    p.contact(CONTACT_WITHDRAW_SOURCE,(PRED_TEACH1,),88201,True,True)
    setup(p,PRED_WORLD2); good2=p.contact(CONTACT_SOURCE_UTTERANCE,u(PREDICTION),PRED_TEACH2,True,True); ga2=p.tick(); settle(p,ga2,PRED_WORLD2,True,(STATE,GOAL),1)
    pred_edges=p.cognition.edges()
    checks['repeated_grounded_prediction_becomes_nonlinguistic_relation']=(
        len(pred_edges)==1 and pred_edges[0].action==MOTOR_INSPECT and pred_edges[0].next_state==(STATE,GOAL)
        and pred_edges[0].support==2 and set(pred_edges[0].sources)=={PRED_WORLD1,PRED_WORLD2}
    )
    p.contact(CONTACT_WITHDRAW_SOURCE,(PRED_TEACH2,),88202,True,True)
    for offset,source in enumerate(prediction_language_sources,1):p.contact(CONTACT_WITHDRAW_SOURCE,(source,),89200+offset,True,True)
    checks['prediction_language_can_disappear']=p.language.realize(CTX,(REMOTE,INSPECT,GOAL)) is None and p.language.invert_surface(u(PREDICTION))==()
    setup(p,PRED_WORLD3); pred_cp=copy.deepcopy(p.checkpoint()); autonomous=p.tick()
    checks['grounded_prediction_runs_without_language']=(isinstance(autonomous,MotorActionV2) and autonomous.action_id==MOTOR_INSPECT and not autonomous.source_assertion_ids)
    pred_lesion=ReferenceOrganismV2.restore(copy.deepcopy(pred_cp));pred_lesion.cognition._evidence.clear();pred_fallback=pred_lesion.tick()
    checks['prediction_relation_lesion_restores_baseline']=isinstance(pred_fallback,MotorActionV2) and pred_fallback.action_id==MOTOR_TEST

    # Multi-step developmental leverage. Two grounded utterances teach two
    # nonlinguistic transitions; later ordinary planning composes them with every
    # teaching surface and cue->motor recruitment source removed.
    START2,MID2,GOAL2=900,9900,9999
    STEP_INSPECT,STEP_TEST=7001,7134
    m=ReferenceOrganismV2(spec);multi_language_sources=list(train_language(m))
    feat(m,MID2,(65,66,67),8020);multi_language_sources.extend((name(m,MID2,'relay',12000+MID2),name(m,MID2,'relay',13000+MID2)))
    inspect_units=m.language.lexeme(INSPECT);test_units=m.language.lexeme(TEST)
    inspect_lid=m.language.lexeme_identity(INSPECT,inspect_units);test_lid=m.language.lexeme_identity(TEST,test_units)
    G_INSPECT,G_TEST=84001,84002
    m._ground_language_action_recruitment(inspect_lid,STEP_INSPECT,G_INSPECT,1,True)
    m._ground_language_action_recruitment(test_lid,STEP_TEST,G_TEST,1,True)
    instruction1=u('zoe inspects the relay.');instruction2=u('bob tests the target.')
    checks['multistep_surfaces_are_heldout']=instruction1 not in demonstrated and instruction2 not in demonstrated
    def stage_multi(state,target,world_source):
        m.contact(CONTACT_WORLD_STATE,(state,),world_source,True,True);m.contact(CONTACT_BODY_TARGET,(target,),7011,True,True);m.contact(CONTACT_AFFORDANCES,(STEP_INSPECT,STEP_TEST),7012,True,True)
    def learn_step(state,target,instruction,teacher,world_source,expected_action,next_state):
        stage_multi(state,target,world_source);baseline=m._exploration_candidate();assert baseline!=expected_action
        aid=m.contact(CONTACT_SOURCE_UTTERANCE,instruction,teacher,True,True);action=m.tick();assert aid and isinstance(action,MotorActionV2) and action.action_id==expected_action and action.source_counterfactual_action==baseline
        learned=settle(m,action,world_source,True,(next_state,),1);m.contact(CONTACT_WITHDRAW_SOURCE,(teacher,),teacher+70000,True,True);return learned
    for teacher,world in ((85001,95001),(85002,95002)):learn_step(START2,MID2,instruction1,teacher,world,STEP_INSPECT,MID2)
    for teacher,world in ((85003,95003),(85004,95004)):learn_step(MID2,GOAL2,instruction2,teacher,world,STEP_TEST,GOAL2)
    multiedges=m.cognition.edges();multi_plan=m.cognition.plan((START2,),(GOAL2,))
    checks['language_teaching_builds_two_nonlinguistic_edges']=(
        len(multiedges)==2 and {(edge.state,edge.action,edge.next_state) for edge in multiedges}=={
            ((START2,),STEP_INSPECT,(MID2,)),((MID2,),STEP_TEST,(GOAL2,))})
    checks['nonlinguistic_planner_composes_taught_edges']=multi_plan.status==1 and multi_plan.actions==(STEP_INSPECT,STEP_TEST)
    for offset,source in enumerate(multi_language_sources,1):m.contact(CONTACT_WITHDRAW_SOURCE,(source,),96000+offset,True,True)
    m.contact(CONTACT_WITHDRAW_SOURCE,(G_INSPECT,),97001,True,True);m.contact(CONTACT_WITHDRAW_SOURCE,(G_TEST,),97002,True,True)
    checks['multistep_teaching_surfaces_are_gone']=m.language.invert_surface(instruction1)==() and m.language.invert_surface(instruction2)==()
    stage_multi(START2,GOAL2,98001);multi_cp=copy.deepcopy(m.checkpoint());first_step=m.tick()
    checks['language_free_first_plan_step']=isinstance(first_step,MotorActionV2) and first_step.action_id==STEP_INSPECT and not first_step.source_assertion_ids
    settle(m,first_step,98001,True,(MID2,),1);second_step=m.tick()
    checks['language_free_second_plan_step']=isinstance(second_step,MotorActionV2) and second_step.action_id==STEP_TEST and not second_step.source_assertion_ids
    restored_multi=ReferenceOrganismV2.restore(copy.deepcopy(multi_cp));restored_first=restored_multi.tick()
    checks['multistep_checkpoint_replays']=isinstance(restored_first,MotorActionV2) and restored_first.action_id==STEP_INSPECT
    first_edge_lesion=ReferenceOrganismV2.restore(copy.deepcopy(multi_cp));first_edge_lesion.cognition._evidence={key:value for key,value in first_edge_lesion.cognition._evidence.items() if not (key[0]==(START2,) and key[1]==STEP_INSPECT)};lesioned_plan=first_edge_lesion.cognition.plan((START2,),(GOAL2,));lesioned_action=first_edge_lesion.tick()
    checks['multistep_focal_edge_lesion_breaks_composition']=lesioned_plan.status==0 and isinstance(lesioned_action,MotorActionV2) and lesioned_action.action_id!=STEP_INSPECT
    checks['no_persistent_language_plan_object']=not hasattr(m,'language_plan') and not hasattr(m,'instruction_plan')

    # Language can reinstate an earned nonlinguistic somatic relation, but cannot
    # write it. The named entity must occur in independent action reafference.
    soma=ReferenceOrganismV2(spec);train_language(soma);setup(soma,SOMA_WORLD1)
    soma.contact(CONTACT_SOURCE_UTTERANCE,u(PREDICTION),SOMA_TEACH1,True,True)
    soma_action=soma.tick()
    checks['language_prediction_alone_does_not_create_somatic_marker']=(
        isinstance(soma_action,MotorActionV2) and not soma_action.somatic_occurrences
        and REMOTE not in soma._somatic_marked_entities())
    soma_learned=settle(soma,soma_action,SOMA_WORLD1,True,(STATE,REMOTE),-1)
    checks['actual_reafferent_referent_creates_somatic_revision']=(
        soma_learned.get('somatic_updates',0)>0
        and REMOTE in soma._somatic_marked_entities())
    soma.contact(CONTACT_WITHDRAW_SOURCE,(SOMA_TEACH1,),89401,True,True)
    setup(soma,SOMA_WORLD2)
    soma.contact(CONTACT_SOURCE_UTTERANCE,u(INSTRUCTION),SOMA_TEACH2,True,True)
    soma_recalled=soma.tick();somatic_recipe_touches=soma.last_somatic_touches
    checks['later_language_recruits_earned_somatic_occurrence']=(
        isinstance(soma_recalled,MotorActionV2)
        and soma_recalled.action_id==MOTOR_INSPECT
        and len(soma_recalled.somatic_occurrences)==1)
    checks['somatic_reinstatement_is_sparse']=(
        somatic_recipe_touches==1
        and somatic_recipe_touches/soma.population.spec.site_count<0.001)
    soma_cp=copy.deepcopy(soma.checkpoint());soma_restored=ReferenceOrganismV2.restore(copy.deepcopy(soma_cp))
    checks['checkpoint_preserves_somatic_recipe_not_transient_network']=(
        REMOTE in soma_restored._somatic_marked_entities()
        and bool(soma_restored._source_somatic_occurrences(soma_recalled.source_assertion_ids)))
    soma_restored.contact(CONTACT_WITHDRAW_SOURCE,(SOMA_WORLD1,),89402,True,True)
    checks['body_consequence_withdrawal_removes_somatic_relation']=(
        REMOTE not in soma_restored._somatic_marked_entities()
        and not soma_restored._source_somatic_occurrences(soma_recalled.source_assertion_ids))
    checks['predicted_but_absent_referent_gets_no_somatic_revision']=(
        REMOTE not in p._somatic_marked_entities())

    yoked_soma=ReferenceOrganismV2(spec);train_language(yoked_soma);setup(yoked_soma,SOMA_WORLD1)
    yoked_soma.contact(CONTACT_SOURCE_UTTERANCE,u(PREDICTION),SOMA_TEACH1,True,True)
    yoked_soma_action=yoked_soma.tick();settle(yoked_soma,yoked_soma_action,SOMA_WORLD1,False,(STATE,REMOTE),-1)
    checks['yoked_reafference_cannot_create_somatic_revision']=(
        REMOTE not in yoked_soma._somatic_marked_entities())

    state=json.dumps(cp,sort_keys=True,separators=(',',':'))
    checks['checkpoint_has_no_instruction_transcript']=(INSTRUCTION not in state and 'source-language-binding-v1' not in state)
    checks['language_remains_outer_not_model']=not hasattr(o,'prompt') and not hasattr(o,'answer') and not hasattr(o,'semantic_summary')

    no_world=ReferenceOrganismV2(spec);train_language(no_world)
    no_world.contact(CONTACT_BODY_TARGET,(GOAL,),7001,True,True);no_world.contact(CONTACT_AFFORDANCES,(MOTOR_TEST,MOTOR_INSPECT),7002,True,True)
    before_no_world=no_world.digest()
    try:no_world.contact(CONTACT_SOURCE_UTTERANCE,u(PREDICTION),9300,True,True)
    except ValueError as exc:no_world_refused=str(exc)=='organism:source_utterance_context'
    else:no_world_refused=False
    checks['language_without_world_cannot_nominate_or_revise']=no_world_refused and no_world.digest()==before_no_world and not no_world.source_assertions and not no_world.cognition.edges()

    out={
        'schema':'agi.reference-language-guided-cognition.v2',
        'pass':all(checks.values()),
        'checks':checks,
        'language_binding':row1.language_binding,
        'grounded_transition':None if not edges else {'action':edges[0].action,'support':edges[0].support,'sources':list(edges[0].sources)},
        'resource':{'resident_sites':o.population.spec.site_count,
                    'materialized_sites':o.population.materialized_site_count(),
                    'retained_occurrences':len(o.population.occurrences),
                    'language_recipe_touches':language_recipe_touches,
                    'shadow_recipe_touches':shadow_recipe_touches,
                    'shadow_checkpoint_bytes':len(json.dumps(shadow_before,sort_keys=True,separators=(',',':'))),
                    'shadow_world_evidence_before_return':shadow_evidence_before_return,
                    'somatic_recipe_touches':somatic_recipe_touches,
                    'recruitment_relations':len(o.recruitment.relations),
                    'checkpoint_bytes':len(state)},
        'runtime_llm':False,
        'host_action_id_in_instruction':False,
        'physical_direct_parity':'NOT_RUN/RED',
        'graph_flip':False,
        'claim':'LANGUAGE_MEDIATED_NONLINGUISTIC_LEARNING_REFERENCE_ONLY_NOT_HUMAN_LANGUAGE_MASTERY',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print('FOUNDRY_LANGUAGE_GUIDED_COGNITION '+('GREEN' if out['pass'] else 'RED')+' raw_instruction=1 provisional=1 shadow_recipe=1 grounded_transition=1 language_withdrawn=1')
    print(json.dumps(out,indent=2,sort_keys=True))
    raise SystemExit(0 if out['pass'] else 1)


if __name__=='__main__':main()
