#!/usr/bin/env python3
"""Held-out surface transformation learned by one continuing reference organism."""
from __future__ import annotations

import copy
import json
import time

from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CTX,ADJ,BCTX,PLUR=8801,8802,8803,7002
P,N2,T2,CAT_FORM,DOG_FORM=9901,9902,9903,9911,9912
REMOTE,INSPECT,CAT,DOG,WUG,BIRD,FISH,LEFT,RIGHT=601,303,101,103,105,107,109,111,113
RAM,GLIP,SLIP,FLIP,CLIP,BLIP=115,117,119,121,123,125
MOTOR_TEST,MOTOR_INSPECT=7302,8303
BODY_MULTI=(771,772,773);BODY_OTHER=(781,782,783)


def u(text):return tuple(text.encode())


def partner(o,source=P):
    o.contact(CONTACT_PARTNER_CONTEXT,(1,7,source),70000+source,True,True)


def bind(o,entity,condition,source):
    tail=(condition,) if condition else ()
    o.contact(CONTACT_ENTITY_FEATURES,(entity,1,entity,*tail),source,True,True)


def name(o,entity,text,source):
    o.contact(CONTACT_SCENE,(7,0,1,entity),source,True,True)
    o.contact(CONTACT_SURFACE,u(text),source,True,True)


def clause(o,atoms,text,source,context=CTX):
    o.contact(CONTACT_SCENE,(7,context,len(atoms),*atoms),source,True,True)
    o.contact(CONTACT_SURFACE,u(text),source,True,True)


def conditioned(o,entity,text,source):
    bind(o,entity,PLUR,source)
    clause(o,(REMOTE,INSPECT,entity),f'zoe inspects {text}.',source)


def body_conditioned(o,entity,text,state,source):
    o.contact(CONTACT_BODY_TARGET,(entity,),source,True,True)
    o.contact(CONTACT_BODY_STATE,state,source,True,True)
    clause(o,(REMOTE,INSPECT,entity),f'zoe inspects {text}.',source)


def outward(o,source):
    partner(o)
    bind(o,WUG,PLUR,source)
    o.contact(CONTACT_SCENE,(7,CTX,3,REMOTE,INSPECT,WUG),source+1,True,True)
    action=o.tick()
    if isinstance(action,ActionV2):
        o.contact(CONTACT_CONSEQUENCE,(action.ticket,1),action.source,True,True)
    return action


def emit(o,partner_source,context,source,target=WUG):
    partner(o,partner_source);bind(o,target,PLUR,source)
    o.contact(CONTACT_SCENE,(7,context,3,REMOTE,INSPECT,target),source+1,True,True)
    return o.tick()


def stage(o,state,source,target=WUG):
    bind(o,target,PLUR,source)
    o.contact(CONTACT_WORLD_STATE,(state,),source+1,True,True)
    o.contact(CONTACT_BODY_TARGET,(target,),source+2,True,True)
    o.contact(CONTACT_AFFORDANCES,(MOTOR_TEST,MOTOR_INSPECT),source+3,True,True)
    for _ in range(2):
        if o._exploration_candidate()==MOTOR_TEST:break
        trial=o.tick()
        if not isinstance(trial,MotorActionV2):raise AssertionError('resident exploration')
        o.contact(CONTACT_MOTOR_CONSEQUENCE,(trial.ticket,0,1,state),source,True,True)
    return o._exploration_candidate()


def build():
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    for entity,text in ((REMOTE,'zoe'),(INSPECT,'inspects'),(CAT,'cat'),(DOG,'dog'),(WUG,'wug')):
        name(o,entity,text,P);name(o,entity,text,N2)
    clause(o,(REMOTE,INSPECT,CAT),'zoe inspects cat.',P)
    clause(o,(REMOTE,INSPECT,CAT),'zoe inspects cat.',T2)
    lexeme=o.language.lexeme_identity(INSPECT,u('inspects'))
    assert o._ground_language_action_recruitment(lexeme,MOTOR_INSPECT,60001,1,True)
    conditioned(o,CAT,'cats',P);conditioned(o,CAT,'cats',CAT_FORM)
    return o


def build_bilingual():
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    ecologies=(
        (12001,12002,('zoe','inspects','cat','dog','wug'),'zoe inspects cat.',
         ('zoe inspects cats.','zoe inspects dogs.')),
        (13001,13002,('lena','prueft','Katze','Blume','Daxe'),'lena prueft Katze.',
         ('lena prueft Katzen.','lena prueft Blumen.')),
    )
    for first,second,words,base_surface,_forms in ecologies:
        for source in (first,second):
            for entity,text in zip((REMOTE,INSPECT,CAT,DOG,WUG),words):name(o,entity,text,source)
            clause(o,(REMOTE,INSPECT,CAT),base_surface,source,BCTX)
        identity=o.language.lexeme_identity(INSPECT,u(words[1]))
        assert o._ground_language_action_recruitment(identity,MOTOR_INSPECT,60000+first,1,True)
    for first,second,_words,_base_surface,forms in ecologies:
        for entity,text in zip((CAT,DOG),forms):
            for source in (first,second):
                bind(o,entity,PLUR,source);clause(o,(REMOTE,INSPECT,entity),text,source,BCTX)
    return o


def build_body_conditioned():
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    for entity,text in ((REMOTE,'zoe'),(INSPECT,'inspects'),(CAT,'cat'),(DOG,'dog'),(WUG,'wug')):
        name(o,entity,text,P);name(o,entity,text,N2)
    clause(o,(REMOTE,INSPECT,CAT),'zoe inspects cat.',P)
    clause(o,(REMOTE,INSPECT,CAT),'zoe inspects cat.',T2)
    identity=o.language.lexeme_identity(INSPECT,u('inspects'))
    assert o._ground_language_action_recruitment(identity,MOTOR_INSPECT,60002,1,True)
    body_conditioned(o,CAT,'cats',BODY_MULTI,P);body_conditioned(o,CAT,'cats',BODY_MULTI,9011)
    return o


def body_stage(o,target,state,world,source):
    o.contact(CONTACT_BODY_TARGET,(target,),source,True,True)
    o.contact(CONTACT_BODY_STATE,state,source,True,True)
    o.contact(CONTACT_WORLD_STATE,(world,),source,True,True)
    o.contact(CONTACT_AFFORDANCES,(MOTOR_TEST,MOTOR_INSPECT),source,True,True)
    for _ in range(2):
        if o._exploration_candidate()==MOTOR_TEST:break
        trial=o.tick()
        if not isinstance(trial,MotorActionV2):raise AssertionError('resident body exploration')
        o.contact(CONTACT_MOTOR_CONSEQUENCE,(trial.ticket,0,1,world),source,True,True)
    return o._exploration_candidate()


def remove_form(o,entity):
    rows=[row for row in o.language._form_index.get(entity,()) if row[0]==(PLUR,)]
    for required,units in rows:
        o.language._form_sources.pop((entity,required,units),None)
        o.language._form_index[entity].discard((required,units))
    o.language._rebuild_indices()


def main():
    started=time.perf_counter();checks={};o=build()
    red_checkpoint=copy.deepcopy(o.checkpoint())
    red=ReferenceOrganismV2.restore(copy.deepcopy(red_checkpoint));red_action=outward(red,8100)
    checks['one_donor_feature_cannot_generalize']=(
        o.language.form(WUG,(PLUR,),True) is None and red_action is None)
    cat_sources=o.language._form_sources.get((CAT,(PLUR,),u('cats')),set())
    checks['conditioned_form_came_from_multiword_contacts']=(
        cat_sources=={P,CAT_FORM}
        and all(any(scene.source==source and scene.atoms==(REMOTE,INSPECT,CAT)
                    for scene in o._scene_by_id.values()) for source in cat_sources))

    conditioned(o,DOG,'dogs',P);conditioned(o,DOG,'dogs',DOG_FORM)
    derived=o.language.form(WUG,(PLUR,),True);rule_touches=o.language.last_rule_touches
    inverse=o.language.invert_form_candidates(u('wugs'),(PLUR,),True)
    inverse_touches=o.language.last_lookup_touches
    checks['two_distinct_donors_derive_heldout_form']=(
        derived==u('wugs') and {row[0] for row in inverse}=={WUG}
        and rule_touches==2)
    checks['transformation_is_not_a_persisted_target_form']=(
        not any(feature==WUG and required==(PLUR,) for feature,required,_units in o.language._form_sources))

    productive_checkpoint=copy.deepcopy(o.checkpoint());action=outward(o,8200)
    checks['resident_conditioned_construction_emits_heldout_form']=(
        isinstance(action,ActionV2) and action.payload==u('zoe inspects wugs.')
        and any(kind==PREF_FORM for kind,_slot,_candidate,_occ in action.selection_occurrences))

    understood=ReferenceOrganismV2.restore(copy.deepcopy(productive_checkpoint))
    baseline=stage(understood,8300,8301)
    assertion=understood.contact(CONTACT_SOURCE_UTTERANCE,u('zoe inspects wugs.'),P,True,True)
    motor=understood.tick()
    learned={} if not isinstance(motor,MotorActionV2) else understood.contact(
        CONTACT_MOTOR_CONSEQUENCE,(motor.ticket,1,2,8300,WUG),motor.source,True,True)
    row=next((item for item in understood.source_assertions if item.identity==assertion),None)
    checks['raw_derived_form_recruits_current_nonlinguistic_target']=(
        baseline==MOTOR_TEST and row is not None and row.binding_atoms==(REMOTE,INSPECT,WUG)
        and isinstance(motor,MotorActionV2) and motor.action_id==MOTOR_INSPECT
        and learned.get('source_credit',0)>0)
    checks['comprehension_does_not_memorize_derived_form']=(
        not any(feature==WUG and required==(PLUR,) for feature,required,_units in understood.language._form_sources))

    replay=ReferenceOrganismV2.restore(copy.deepcopy(productive_checkpoint))
    replay_form=replay.language.form(WUG,(PLUR,),True)
    checks['checkpoint_rebuilds_transient_rule_index_exactly']=(
        replay.digest()==ReferenceOrganismV2.restore(copy.deepcopy(productive_checkpoint)).digest()
        and replay_form==u('wugs') and 'form_rule' not in json.dumps(productive_checkpoint))

    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(productive_checkpoint))
    withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(CAT_FORM,),8400,True,True)
    checks['one_donor_source_withdrawal_removes_generalization']=(
        withdrawn.language.lexeme(WUG)==u('wug') and withdrawn.language.form(WUG,(PLUR,),True) is None)
    name_cut=ReferenceOrganismV2.restore(copy.deepcopy(productive_checkpoint))
    name_cut.contact(CONTACT_WITHDRAW_SOURCE,(N2,),8401,True,True)
    checks['target_name_support_is_required']=(
        name_cut.language.lexeme(WUG) is None and name_cut.language.form(WUG,(PLUR,),True) is None)

    lesion=ReferenceOrganismV2.restore(copy.deepcopy(productive_checkpoint));remove_form(lesion,DOG)
    checks['focal_donor_lesion_is_selective']=(
        lesion.language.lexeme(WUG)==u('wug') and lesion.language.form(CAT,(PLUR,),True)==u('cats')
        and lesion.language.form(WUG,(PLUR,),True) is None)

    exception=ReferenceOrganismV2.restore(copy.deepcopy(productive_checkpoint))
    conditioned(exception,WUG,'wugren',8801);conditioned(exception,WUG,'wugren',8802)
    explicit=exception.language.form(WUG,(PLUR,),True)
    exception.contact(CONTACT_WITHDRAW_SOURCE,(8802,),8803,True,True)
    unsettled=exception.language.form(WUG,(PLUR,),True)
    exception.contact(CONTACT_WITHDRAW_SOURCE,(8801,),8804,True,True)
    checks['observed_exception_wins_and_does_not_fall_back_when_unsettled']=(
        explicit==u('wugren') and unsettled is None
        and exception.language.form(WUG,(PLUR,),True)==u('wugs'))

    ambiguous=ReferenceOrganismV2.restore(copy.deepcopy(productive_checkpoint))
    for entity,text,changed in ((BIRD,'bird','prebird'),(FISH,'fish','prefish')):
        name(ambiguous,entity,text,P);name(ambiguous,entity,text,N2)
        conditioned(ambiguous,entity,changed,P);conditioned(ambiguous,entity,changed,8900+entity)
    checks['equal_support_transformations_remain_ambiguous']=(
        ambiguous.language.form(WUG,(PLUR,),True) is None and outward(ambiguous,8910) is None)

    embodied=build_body_conditioned();body_red=copy.deepcopy(embodied.checkpoint())
    body_conditioned(embodied,DOG,'dogs',BODY_MULTI,P);body_conditioned(embodied,DOG,'dogs',BODY_MULTI,9012)
    body_checkpoint=copy.deepcopy(embodied.checkpoint())
    body_condition=surface_conditions(embodied,DOG,BODY_STATE_TAG)
    checks['body_condition_has_no_authored_entity_tail']=(
        len(body_condition)==1 and not embodied.entity_conditions
        and body_condition[0] not in BODY_MULTI and 'body_condition' not in json.dumps(body_checkpoint))
    one_body=ReferenceOrganismV2.restore(copy.deepcopy(body_red))
    one_body.contact(CONTACT_BODY_TARGET,(WUG,),9020,True,True);one_body.contact(CONTACT_BODY_STATE,BODY_MULTI,9020,True,True)
    partner(one_body);one_body.contact(CONTACT_SCENE,(7,CTX,3,REMOTE,INSPECT,WUG),9021,True,True)
    checks['one_body_conditioned_donor_remains_red']=one_body.tick() is None
    body_out=ReferenceOrganismV2.restore(copy.deepcopy(body_checkpoint))
    body_out.contact(CONTACT_BODY_TARGET,(WUG,),9022,True,True);body_out.contact(CONTACT_BODY_STATE,BODY_MULTI,9022,True,True)
    partner(body_out);body_out.contact(CONTACT_SCENE,(7,CTX,3,REMOTE,INSPECT,WUG),9023,True,True);body_action=body_out.tick()
    checks['live_body_morphology_drives_heldout_form']=(
        isinstance(body_action,ActionV2) and body_action.payload==u('zoe inspects wugs.'))
    if isinstance(body_action,ActionV2):body_out.contact(CONTACT_CONSEQUENCE,(body_action.ticket,1),body_action.source,True,True)
    other_body=ReferenceOrganismV2.restore(copy.deepcopy(body_checkpoint))
    other_body.contact(CONTACT_BODY_TARGET,(WUG,),9024,True,True);other_body.contact(CONTACT_BODY_STATE,BODY_OTHER,9024,True,True)
    partner(other_body);other_body.contact(CONTACT_SCENE,(7,CTX,3,REMOTE,INSPECT,WUG),9025,True,True)
    checks['different_body_morphology_does_not_recruit_form']=other_body.tick() is None
    no_target=ReferenceOrganismV2.restore(copy.deepcopy(body_checkpoint))
    no_target.contact(CONTACT_BODY_TARGET,(WUG,),9026,True,True);no_target.contact(CONTACT_BODY_STATE,BODY_MULTI,9026,True,True)
    target_condition=surface_conditions(no_target,WUG,BODY_STATE_TAG);no_target.body_target=()
    partner(no_target);no_target.contact(CONTACT_SCENE,(7,CTX,3,REMOTE,INSPECT,WUG),9027,True,True);base_action=no_target.tick()
    checks['body_state_without_current_target_refuses']=(
        bool(target_condition) and not surface_conditions(no_target,WUG,BODY_STATE_TAG) and base_action is None)
    body_inverse=ReferenceOrganismV2.restore(copy.deepcopy(body_checkpoint));partner(body_inverse)
    body_baseline=body_stage(body_inverse,WUG,BODY_MULTI,9030,9031)
    body_assertion=body_inverse.contact(CONTACT_SOURCE_UTTERANCE,u('zoe inspects wugs.'),P,True,True)
    body_motor=body_inverse.tick();body_row=next((row for row in body_inverse.source_assertions if row.identity==body_assertion),None)
    body_learned={} if not isinstance(body_motor,MotorActionV2) else body_inverse.contact(
        CONTACT_MOTOR_CONSEQUENCE,(body_motor.ticket,1,2,9030,WUG),body_motor.source,True,True)
    checks['body_conditioned_form_is_bidirectional_and_causally_used']=(
        body_baseline==MOTOR_TEST and body_row is not None and body_row.binding_atoms==(REMOTE,INSPECT,WUG)
        and isinstance(body_motor,MotorActionV2) and body_motor.action_id==MOTOR_INSPECT
        and body_learned.get('source_credit',0)>0)
    body_cut=ReferenceOrganismV2.restore(copy.deepcopy(body_checkpoint))
    body_cut.contact(CONTACT_BODY_TARGET,(WUG,),9032,True,True);body_cut.contact(CONTACT_BODY_STATE,BODY_MULTI,9033,True,True)
    cut_condition=surface_conditions(body_cut,WUG,BODY_STATE_TAG)
    body_cut.contact(CONTACT_WITHDRAW_SOURCE,(9033,),9034,True,True)
    partner(body_cut);body_cut.contact(CONTACT_SCENE,(7,CTX,3,REMOTE,INSPECT,WUG),9035,True,True);cut_action=body_cut.tick()
    body_lesion=ReferenceOrganismV2.restore(copy.deepcopy(body_checkpoint))
    body_lesion.contact(CONTACT_BODY_TARGET,(WUG,),9036,True,True);body_lesion.contact(CONTACT_BODY_STATE,BODY_MULTI,9036,True,True)
    lesion_condition=surface_conditions(body_lesion,WUG,BODY_STATE_TAG);body_lesion.body_state_occurrence=0
    form_cut=ReferenceOrganismV2.restore(copy.deepcopy(body_checkpoint))
    form_cut.contact(CONTACT_BODY_TARGET,(WUG,),9037,True,True);form_cut.contact(CONTACT_BODY_STATE,BODY_MULTI,9037,True,True)
    supported_condition=surface_conditions(form_cut,WUG,BODY_STATE_TAG)
    for source in (P,9011,9012):form_cut.contact(CONTACT_WITHDRAW_SOURCE,(source,),9040+source,True,True)
    checks['body_source_withdrawal_and_occurrence_lesion_remove_condition']=(
        bool(cut_condition) and not surface_conditions(body_cut,WUG,BODY_STATE_TAG) and cut_action is None
        and bool(lesion_condition) and not surface_conditions(body_lesion,WUG,BODY_STATE_TAG))
    checks['body_condition_requires_live_independent_form_support']=(
        bool(supported_condition) and not surface_conditions(form_cut,WUG,BODY_STATE_TAG))
    body_replay=ReferenceOrganismV2.restore(copy.deepcopy(body_checkpoint))
    checks['body_condition_checkpoint_reconstructs_from_resident_state']=(
        body_replay.digest()==embodied.digest()
        and surface_conditions(body_replay,DOG,BODY_STATE_TAG)==body_condition)

    shaped=ReferenceOrganismV2.restore(copy.deepcopy(productive_checkpoint))
    for entity,text in ((RAM,'ram'),(GLIP,'glip'),(SLIP,'slip'),(FLIP,'flip'),(CLIP,'clip'),(BLIP,'blip')):
        name(shaped,entity,text,P);name(shaped,entity,text,N2)
    for entity,text,second in ((RAM,'rams',8991),(GLIP,'glipt',8992),(SLIP,'slipt',8993)):
        conditioned(shaped,entity,text,P);conditioned(shaped,entity,text,second)
    local_form=shaped.language.form(BLIP,(PLUR,),True);local_touches=shaped.language.last_rule_touches
    shape_checkpoint=copy.deepcopy(shaped.checkpoint())
    local_action=emit(shaped,P,CTX,8994,BLIP)
    if isinstance(local_action,ActionV2):shaped.contact(CONTACT_CONSEQUENCE,(local_action.ticket,1),local_action.source,True,True)
    checks['local_edge_reliability_beats_global_type_frequency']=(
        local_form==u('blipt') and isinstance(local_action,ActionV2)
        and local_action.payload==u('zoe inspects blipt.'))
    shape_inverse=ReferenceOrganismV2.restore(copy.deepcopy(shape_checkpoint));partner(shape_inverse,P)
    shape_baseline=stage(shape_inverse,8995,8996,BLIP)
    shape_assertion=shape_inverse.contact(CONTACT_SOURCE_UTTERANCE,u('zoe inspects blipt.'),P,True,True)
    shape_motor=shape_inverse.tick();shape_row=next((row for row in shape_inverse.source_assertions if row.identity==shape_assertion),None)
    shape_learned={} if not isinstance(shape_motor,MotorActionV2) else shape_inverse.contact(
        CONTACT_MOTOR_CONSEQUENCE,(shape_motor.ticket,1,2,8995,BLIP),shape_motor.source,True,True)
    checks['local_shape_transfer_is_bidirectional_and_causally_used']=(
        shape_baseline==MOTOR_TEST and shape_row is not None and shape_row.binding_atoms==(REMOTE,INSPECT,BLIP)
        and isinstance(shape_motor,MotorActionV2) and shape_motor.action_id==MOTOR_INSPECT
        and shape_learned.get('source_credit',0)>0)
    local_tie=ReferenceOrganismV2.restore(copy.deepcopy(shape_checkpoint))
    for entity,text,changed,second in ((FLIP,'flip','flipk',8997),(CLIP,'clip','clipk',8998)):
        conditioned(local_tie,entity,changed,P);conditioned(local_tie,entity,changed,second)
    checks['equal_local_reliability_preserves_ambiguity']=(
        local_tie.language.form(BLIP,(PLUR,),True) is None and emit(local_tie,P,CTX,8999,BLIP) is None)
    local_cut=ReferenceOrganismV2.restore(copy.deepcopy(shape_checkpoint))
    local_cut.contact(CONTACT_WITHDRAW_SOURCE,(8992,),9000,True,True)
    checks['local_evidence_withdrawal_reveals_broader_fallback']=(
        local_cut.language.form(BLIP,(PLUR,),True)==u('blips'))

    boundary=ReferenceOrganismV2(PopulationSpecV1(8192,2,4,42,8))
    for entity,text in ((LEFT,'a'),(RIGHT,'b')):
        name(boundary,entity,text,8920);name(boundary,entity,text,8921)
    clause(boundary,(LEFT,RIGHT),'ab',8920,ADJ);clause(boundary,(LEFT,RIGHT),'ab',8921,ADJ)
    for source in (8922,8923):
        bind(boundary,LEFT,PLUR,source);bind(boundary,RIGHT,PLUR,source)
        boundary.contact(CONTACT_SCENE,(7,ADJ,2,LEFT,RIGHT),source,True,True)
        boundary.contact(CONTACT_SURFACE,u('aabb'),source,True,True)
    checks['adjacent_unknown_ports_preserve_boundary_ambiguity']=(
        not boundary.language._form_index.get(LEFT) and not boundary.language._form_index.get(RIGHT))

    bilingual=build_bilingual();bilingual_checkpoint=copy.deepcopy(bilingual.checkpoint())
    english=ReferenceOrganismV2.restore(copy.deepcopy(bilingual_checkpoint));en_action=emit(english,12001,BCTX,8930)
    german=ReferenceOrganismV2.restore(copy.deepcopy(bilingual_checkpoint));de_action=emit(german,13001,BCTX,8940)
    neutral=ReferenceOrganismV2.restore(copy.deepcopy(bilingual_checkpoint));neutral_action=emit(neutral,14001,BCTX,8950)
    checks['partner_history_selects_bilingual_productive_form']=(
        isinstance(en_action,ActionV2) and en_action.payload==u('zoe inspects wugs.')
        and isinstance(de_action,ActionV2) and de_action.payload==u('lena prueft Daxen.')
        and neutral_action is None)
    checks['unpartnered_form_uses_local_shape_without_router']=(
        bilingual.language.form(WUG,(PLUR,),True)==u('Daxen'))
    inverse_rows=[]
    for partner_source,text,state,source in (
        (12001,'zoe inspects wugs.',8960,8961),(13001,'lena prueft Daxen.',8970,8971)):
        x=ReferenceOrganismV2.restore(copy.deepcopy(bilingual_checkpoint));partner(x,partner_source)
        baseline=stage(x,state,source);assertion=x.contact(CONTACT_SOURCE_UTTERANCE,u(text),partner_source,True,True)
        selected=x.tick();row=next((item for item in x.source_assertions if item.identity==assertion),None)
        learned={} if not isinstance(selected,MotorActionV2) else x.contact(
            CONTACT_MOTOR_CONSEQUENCE,(selected.ticket,1,2,state,WUG),selected.source,True,True)
        inverse_rows.append((baseline,row,selected,learned))
    checks['bilingual_derived_forms_bind_same_nonlinguistic_target']=all(
        baseline==MOTOR_TEST and row is not None and row.binding_atoms==(REMOTE,INSPECT,WUG)
        and isinstance(selected,MotorActionV2) and selected.action_id==MOTOR_INSPECT
        and learned.get('source_credit',0)>0
        for baseline,row,selected,learned in inverse_rows)
    english_cut=ReferenceOrganismV2.restore(copy.deepcopy(bilingual_checkpoint))
    english_cut.contact(CONTACT_WITHDRAW_SOURCE,(12002,),8980,True,True);cut_en=emit(english_cut,12001,BCTX,8981)
    german_spared=ReferenceOrganismV2.restore(copy.deepcopy(bilingual_checkpoint))
    german_spared.contact(CONTACT_WITHDRAW_SOURCE,(12002,),8982,True,True);cut_de=emit(german_spared,13001,BCTX,8983)
    checks['bilingual_source_withdrawal_is_ecology_local']=(
        cut_en is None and isinstance(cut_de,ActionV2) and cut_de.payload==u('lena prueft Daxen.'))
    checks['bilingual_rules_are_transient_source_provenance_not_router_state']=(
        'form_rule' not in json.dumps(bilingual_checkpoint)
        and not hasattr(bilingual,'language_id') and not hasattr(bilingual,'language_router'))

    quantity=ReferenceOrganismV2.restore(copy.deepcopy(shape_checkpoint))
    before=quantity.language.invert_form_candidates(u('blipt'),(PLUR,),True)
    before_lookup=quantity.language.last_lookup_touches;before_rule=quantity.language.last_rule_touches
    bytes_before=len(json.dumps(quantity.checkpoint(),separators=(',',':')).encode())
    for index in range(512):
        entity=20000+index;name(quantity,entity,f'~q{index:04d}',90000+index);name(quantity,entity,f'~q{index:04d}',91000+index)
    after=quantity.language.invert_form_candidates(u('blipt'),(PLUR,),True)
    after_lookup=quantity.language.last_lookup_touches;after_rule=quantity.language.last_rule_touches
    bytes_after=len(json.dumps(quantity.checkpoint(),separators=(',',':')).encode())
    checks['sparse_inverse_work_survives_512_lexical_decoys']=(
        before==after and before_lookup==after_lookup and before_rule==after_rule==5)

    body_quantity=ReferenceOrganismV2.restore(copy.deepcopy(body_checkpoint));condition_id=body_condition[0]
    condition_before=body_quantity.language.condition_supported(condition_id);condition_touches_before=body_quantity.language.last_lookup_touches
    for index in range(512):
        entity=30000+index;condition=40000+index;surface=u(f'~f{index:04d}')
        body_quantity.language.observe_form(entity,(condition,),surface,100000+index)
        body_quantity.language.observe_form(entity,(condition,),surface,101000+index)
    condition_after=body_quantity.language.condition_supported(condition_id);condition_touches_after=body_quantity.language.last_lookup_touches
    checks['sparse_body_condition_lookup_survives_512_form_decoys']=(
        condition_before and condition_after and condition_touches_before==condition_touches_after==2)

    metrics={
        'rule_evidence_touches':rule_touches,'inverse_touches':inverse_touches,
        'local_rule_evidence_touches':local_touches,
        'decoy_inverse_touches_before_after':[before_lookup,after_lookup],
        'decoy_rule_touches_before_after':[before_rule,after_rule],
        'body_condition_touches_before_after':[condition_touches_before,condition_touches_after],
        'checkpoint_bytes_before_decoys':bytes_before,'checkpoint_bytes_after_decoys':bytes_after,
        'decoys':512,
    }
    result={
        'schema':'agi.reference-organism-productive-morphology.v1',
        'pass':all(checks.values()),'checks':checks,'metrics':metrics,
        'runtime_llm':False,'graph_flip':False,
        'claim':'PRODUCTIVE_SURFACE_TRANSFORMATION_ON_CONTINUING_REFERENCE_ORGANISM_ONLY',
        'direct_parity':False,'human_language_mastery':False,
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print('FOUNDRY_REFERENCE_ORGANISM_PRODUCTIVE_MORPHOLOGY '+('GREEN' if result['pass'] else 'RED')+
          f" checks={sum(checks.values())}/{len(checks)} rule_touches={rule_touches} decoys=512")
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)


if __name__=='__main__':main()
