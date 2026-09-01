#!/usr/bin/env python3
"""Ground one port, then unfold a learned nonadjacent surface dependency."""
from __future__ import annotations

import copy
import json
import time

from reference_organism_v2 import *
from reference_organism_surface_state_v1 import surface_conditions
from reference_population_v1 import PopulationSpecV1

CTX=8821;P,N2,T2=9921,9922,9923
CAT,SEE,DOG=121,122,123
HCAT,HSEE,HDOG=221,222,223
MCAT,MSEE,MDOG=321,322,323
BODY=(811,812,813);OTHER_BODY=(821,822,823)


def u(text):return tuple(text.encode())


def name(o,entity,text,source):
    o.contact(CONTACT_SCENE,(7,0,1,entity),source,True,True)
    o.contact(CONTACT_SURFACE,u(text),source,True,True)


def clause(o,atoms,text,source):
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),source,True,True)
    o.contact(CONTACT_SURFACE,u(text),source,True,True)


def body_clause(o,atoms,targets,text,source):
    o.contact(CONTACT_BODY_TARGET,targets,source,True,True)
    o.contact(CONTACT_BODY_STATE,BODY,source,True,True)
    clause(o,atoms,text,source)


def partner_scene(o,targets,state,source):
    if targets:o.contact(CONTACT_BODY_TARGET,targets,source,True,True)
    o.contact(CONTACT_BODY_STATE,state,source,True,True)
    o.contact(CONTACT_PARTNER_CONTEXT,(1,7,P),source+1,True,True)
    o.contact(CONTACT_SCENE,(7,CTX,3,MCAT,MSEE,MDOG),source+2,True,True)
    return o.tick()


def build():
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    words=((CAT,'dak'),(SEE,'lum'),(DOG,'dog'),(HCAT,'nif'),(HSEE,'pek'),
           (HDOG,'dog'),(MCAT,'wug'),(MSEE,'blick'),(MDOG,'dog'))
    for entity,text in words:
        name(o,entity,text,P);name(o,entity,text,N2)
    for source in (P,T2):
        clause(o,(CAT,SEE,DOG),'the dak lum the dog.',source)
        clause(o,(HCAT,HSEE,HDOG),'the nif pek the dog.',source)
    body_clause(o,(CAT,SEE,DOG),(CAT,SEE),'the daks lums the dog.',P)
    body_clause(o,(CAT,SEE,DOG),(CAT,SEE),'the daks lums the dog.',9101)
    body_clause(o,(HCAT,HSEE,HDOG),(HCAT,HSEE),'the nifs peks the dog.',P)
    body_clause(o,(HCAT,HSEE,HDOG),(HCAT,HSEE),'the nifs peks the dog.',9102)
    return o


def main():
    started=time.perf_counter();checks={};o=build();checkpoint=copy.deepcopy(o.checkpoint())
    full=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    full_action=partner_scene(full,(MCAT,MSEE),BODY,9200)
    checks['fully_grounded_ports_establish_form_baseline']=(
        isinstance(full_action,ActionV2) and full_action.payload==u('the wugs blicks the dog.'))

    subject=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    subject.contact(CONTACT_BODY_TARGET,(MCAT,),9210,True,True)
    subject.contact(CONTACT_BODY_STATE,BODY,9210,True,True)
    condition=surface_conditions(subject,MCAT,BODY_STATE_TAG)
    raw_values=(condition[0] if condition else 0,0,0)
    complete=getattr(subject.language,'complete_compatibility',lambda _context,_values:None)
    completed=complete(CTX,raw_values)
    subject.contact(CONTACT_PARTNER_CONTEXT,(1,7,P),9211,True,True)
    subject.contact(CONTACT_SCENE,(7,CTX,3,MCAT,MSEE,MDOG),9212,True,True)
    subject_action=subject.tick()
    checks['learned_dependency_completes_only_missing_equal_port']=(
        bool(condition) and not subject.language.compatible(CTX,raw_values)
        and completed==(condition[0],condition[0],0)
        and isinstance(subject_action,ActionV2)
        and subject_action.payload==u('the wugs blicks the dog.'))
    if isinstance(subject_action,ActionV2):
        consequence=subject.contact(CONTACT_CONSEQUENCE,(subject_action.ticket,1),subject_action.source,True,True)
    else:consequence={}
    checks['completed_dependency_participates_before_independent_credit']=(
        isinstance(subject_action,ActionV2) and subject_action.body_occurrence>0
        and sum(1 for kind,_,_,_ in subject_action.selection_occurrences if kind==PREF_FORM)==2
        and consequence.get('credit',0)>0 and consequence.get('selection_network_updates',0)==1)

    absent=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));absent.body_target=()
    checks['no_grounded_port_refuses']=partner_scene(absent,(),BODY,9220) is None
    other=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    checks['unsupported_body_morphology_refuses']=partner_scene(other,(MCAT,),OTHER_BODY,9230) is None
    lesion=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    lesion.language._compat_sources.clear();lesion.language._rebuild_indices()
    lesion_action=partner_scene(lesion,(MCAT,),BODY,9240)
    checks['compatibility_recipe_lesion_prevents_dependent_form']=(
        not isinstance(lesion_action,ActionV2) or lesion_action.payload!=u('the wugs blicks the dog.'))

    ambiguous=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    ambiguous.contact(CONTACT_BODY_TARGET,(MCAT,),9250,True,True)
    ambiguous.contact(CONTACT_BODY_STATE,BODY,9250,True,True)
    ambiguous_condition=surface_conditions(ambiguous,MCAT,BODY_STATE_TAG)[0]
    ambiguous.language.observe_compatibility(CTX,(ambiguous_condition,0,ambiguous_condition),9251)
    ambiguous.language.observe_compatibility(CTX,(ambiguous_condition,0,ambiguous_condition),9252)
    checks['two_compatible_dependency_shapes_preserve_ambiguity']=(
        ambiguous.language.complete_compatibility(CTX,(ambiguous_condition,0,0)) is None
        if hasattr(ambiguous.language,'complete_compatibility') else True)

    replay=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    checks['checkpoint_rebuilds_dependency_without_completion_cache']=(
        replay.digest()==o.digest() and 'completed_compatibility' not in json.dumps(checkpoint))
    before=subject.language.complete_compatibility(CTX,raw_values) if hasattr(subject.language,'complete_compatibility') else None
    touches_before=subject.language.last_lookup_touches
    for index in range(512):
        subject.language.observe_compatibility(50000+index,(1,1,0),60000+index)
        subject.language.observe_compatibility(50000+index,(1,1,0),61000+index)
    after=subject.language.complete_compatibility(CTX,raw_values) if hasattr(subject.language,'complete_compatibility') else None
    touches_after=subject.language.last_lookup_touches
    checks['dependency_lookup_survives_512_context_decoys']=(
        before==after==(condition[0],condition[0],0) and touches_before==touches_after==1)

    result={'schema':'agi.reference-organism-grounded-dependency.v1','pass':all(checks.values()),
        'checks':checks,'metrics':{'compatibility_touches_before_after':[touches_before,touches_after],
        'decoy_contexts':512},'runtime_llm':False,'expected_output_ingress':False,'graph_flip':False,
        'human_language_mastery':False,'direct_parity':False,
        'claim':'GROUNDED_LEARNED_PORT_DEPENDENCY_ON_CONTINUING_REFERENCE_ORGANISM_ONLY',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_GROUNDED_DEPENDENCY '+('GREEN' if result['pass'] else 'RED')+
          f" checks={sum(checks.values())}/{len(checks)} decoys=512")
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)


if __name__=='__main__':main()
