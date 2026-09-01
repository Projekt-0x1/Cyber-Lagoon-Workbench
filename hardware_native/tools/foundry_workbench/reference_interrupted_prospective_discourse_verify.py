#!/usr/bin/env python3
"""Falsifier for cue-reinstated expression of an unfinished resident route."""
from __future__ import annotations

import copy
import json
import time

import reference_endogenous_prospection_verify as p
from reference_organism_v2 import ActionV2,MotorActionV2,ReferenceOrganismV2,CONTACT_AFFORDANCES,CONTACT_BODY_STATE,CONTACT_SCENE,CONTACT_SURFACE,CONTACT_WITHDRAW_SOURCE,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1


def teach_heldout_prospective_expression(organism):
    """Learn lexemes and clause form without ever contacting the target clause."""
    ecologies=(
        (p.PARTNER_A,{
            p.FIRST:'inspect',p.MID:'middle',p.SECOND:'verify',p.GOAL:'goal',
            p.THIRD:'probe',p.ALT_MID:'alternate',p.FOURTH:'confirm'}),
        (p.PARTNER_B,{
            p.FIRST:'pruefen',p.MID:'mitte',p.SECOND:'testen',p.GOAL:'ziel',
            p.THIRD:'sondieren',p.ALT_MID:'alternative',p.FOURTH:'bestaetigen'}),
    )
    demonstrations=((p.FIRST,p.MID),(p.THIRD,p.ALT_MID),
                    (p.SECOND,p.ALT_MID),(p.FOURTH,p.GOAL))
    receipts={}
    for ecology_index,(partner,words) in enumerate(ecologies):
        lexical_sources={}
        for atom,text in words.items():
            sources=(partner,0x180001+ecology_index*0x10000+atom*2)
            lexical_sources[atom]=sources
            for source in sources:
                organism.contact(CONTACT_SCENE,(7,0,1,atom),source,True,True)
                p._surface(organism,text,source)
        construction_sources=(partner,partner+0x100+ecology_index)
        training_surfaces=set(words.values())
        for source in construction_sources:
            for left,right in demonstrations:
                organism.contact(CONTACT_SCENE,
                                 (7,p.EXPRESSION_CONTEXT,2,left,right),
                                 source,True,True)
                surface=words[left]+' '+words[right]
                p._surface(organism,surface,source);training_surfaces.add(surface)
        target=words[p.SECOND]+' '+words[p.GOAL]
        receipts[partner]={
            'target':target,'training_surfaces':training_surfaces,
            'construction_sources':construction_sources,
            'target_lexical_sources':lexical_sources[p.SECOND],
        }
    all_construction_sources=tuple(source for row in receipts.values()
                                   for source in row['construction_sources'])
    for receipt in receipts.values():
        receipt['all_construction_sources']=all_construction_sources
    return receipts


def prepared(spec,partner):
    organism=p.one_shot_organism(spec)
    receipts=teach_heldout_prospective_expression(organism)
    organism.contact(CONTACT_BODY_STATE,(p.BODY_MARKER,),0xFC01,True,True)
    p._partner(organism,partner,0xFC02+partner)
    p.stage(organism,p.START,p.GOAL,(p.FIRST,),0xFC10)
    first=organism.tick()
    if not isinstance(first,MotorActionV2) or first.action_id!=p.FIRST:
        raise AssertionError('interrupted-intention:first-action')
    p.settle(organism,first,first.source,p.MID,1,True)
    return organism,first,receipts[partner]


def cue(organism,state,affordances,source):
    organism.contact(CONTACT_WORLD_STATE,(int(state),),int(source),True,True)
    organism.contact(CONTACT_AFFORDANCES,tuple(map(int,affordances)),int(source)+1,True,True)


def main():
    started=time.perf_counter();checks={};spec=PopulationSpecV1(32768,2,4,42,8)
    organism,first,english_receipt=prepared(spec,p.PARTNER_A);after_first=copy.deepcopy(organism.checkpoint())
    lived_actions=organism._prospective_snapshot_parts(first.prospective_snapshot)[4]
    checks['first_action_carries_lived_multistep_route']=(
        first.prospective_recipe>0 and len(first.prospective_snapshot)>0)
    checks['exact_complete_suffix_was_never_demonstrated']=(
        english_receipt['target'] not in english_receipt['training_surfaces']
        and all(not (episode.context==p.EXPRESSION_CONTEXT
                    and episode.atoms==(p.SECOND,p.GOAL))
                for episode in organism.episodes))

    nonfocal=ReferenceOrganismV2.restore(copy.deepcopy(after_first))
    cue(nonfocal,p.WRONG,(p.DISTRACTOR,),0xFC20);nonfocal_action=nonfocal.tick()
    checks['nonfocal_current_situation_does_not_recall_speech']=(
        not isinstance(nonfocal_action,ActionV2))

    focal=ReferenceOrganismV2.restore(copy.deepcopy(after_first))
    cue(focal,p.MID,(p.DISTRACTOR,),0xFC30);focal_action=focal.tick()
    checks['focal_situation_recalls_only_unfinished_suffix']=(
        isinstance(focal_action,ActionV2)
        and bytes(focal_action.payload)==b'verify goal'
        and p.FIRST not in focal_action.contributors
        and lived_actions==(p.FIRST,p.SECOND))

    resumed=ReferenceOrganismV2.restore(copy.deepcopy(after_first))
    cue(resumed,p.WRONG,(p.DISTRACTOR,),0xFC40)
    interrupted_checkpoint=copy.deepcopy(resumed.checkpoint())
    resumed=ReferenceOrganismV2.restore(interrupted_checkpoint)
    cue(resumed,p.MID,(p.DISTRACTOR,),0xFC50);resumed_action=resumed.tick()
    checks['interruption_checkpoint_preserves_cue_reinstatement']=(
        isinstance(resumed_action,ActionV2)
        and bytes(resumed_action.payload)==b'verify goal')

    german,_,german_receipt=prepared(spec,p.PARTNER_B)
    cue(german,p.MID,(p.DISTRACTOR,),0xFC60);german_action=german.tick()
    checks['same_unfinished_route_recruits_partner_history_not_stored_sentence']=(
        isinstance(german_action,ActionV2)
        and bytes(german_action.payload)==b'testen ziel'
        and bytes(german_action.payload)!=bytes(getattr(focal_action,'payload',()))
        and german_receipt['target'] not in german_receipt['training_surfaces'])

    construction_cut=ReferenceOrganismV2.restore(copy.deepcopy(after_first))
    for offset,source in enumerate(english_receipt['all_construction_sources']):
        if source!=p.PARTNER_A:
            construction_cut.contact(CONTACT_WITHDRAW_SOURCE,(source,),
                                     0xFC61+offset,True,True)
    cue(construction_cut,p.MID,(p.DISTRACTOR,),0xFC62)
    construction_cut_action=construction_cut.tick()
    checks['construction_support_is_required_but_route_survives']=(
        not isinstance(construction_cut_action,ActionV2)
        and organism._prospective_snapshot_parts(first.prospective_snapshot)[4]
            ==(p.FIRST,p.SECOND))

    lexical_cut=ReferenceOrganismV2.restore(copy.deepcopy(after_first))
    lexical_cut.contact(CONTACT_WITHDRAW_SOURCE,
                        (english_receipt['target_lexical_sources'][1],),
                        0xFC65,True,True)
    cue(lexical_cut,p.MID,(p.DISTRACTOR,),0xFC66)
    checks['target_lexeme_support_is_required']=(
        not isinstance(lexical_cut.tick(),ActionV2))

    executable=ReferenceOrganismV2.restore(copy.deepcopy(after_first))
    cue(executable,p.MID,(p.SECOND,),0xFC70);executed=executable.tick()
    checks['available_next_action_executes_before_discourse']=(
        isinstance(executed,MotorActionV2) and executed.action_id==p.SECOND)
    checks['no_host_intention_or_speech_api']=all(not hasattr(organism,name) for name in (
        'prompt','speak','enqueue_goal','resume_intention','context_window','transcript'))
    checks['visible_discussion_improvement']=(
        checks['focal_situation_recalls_only_unfinished_suffix']
        and checks['same_unfinished_route_recruits_partner_history_not_stored_sentence']
        and checks['nonfocal_current_situation_does_not_recall_speech']
        and checks['exact_complete_suffix_was_never_demonstrated'])

    result={
        'schema':'agi.reference-interrupted-prospective-discourse.v2',
        'pass':all(checks.values()),'checks':checks,'runtime_llm':False,
        'claim':'REFERENCE_FOCAL_CUE_REINSTATES_UNFINISHED_ROUTE_SUFFIX_NOT_DIRECT_CAPABILITY',
        'observed':{
            'focal':'' if not isinstance(focal_action,ActionV2) else bytes(focal_action.payload).decode(),
            'nonfocal_type':type(nonfocal_action).__name__,
            'partner_b':'' if not isinstance(german_action,ActionV2) else bytes(german_action.payload).decode(),
            'complete_target_clause_demonstrations':0,
        },
        'remaining_red':['NONFOCAL_MONITORING_COST','ALLOSTATIC_CONTROLLABLE_YOKED_RECOVERY','DIRECT_CONTINUING_ADULT'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print('FOUNDRY_INTERRUPTED_PROSPECTIVE_DISCOURSE '+('GREEN' if result['pass'] else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True))
    raise SystemExit(0 if result['pass'] else 1)


if __name__=='__main__':main()
