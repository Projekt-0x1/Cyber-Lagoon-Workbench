#!/usr/bin/env python3
"""Learn a directional remote form dependency from raw grounded clauses."""
from __future__ import annotations

import copy
import json
import time

from reference_organism_v2 import *
from reference_organism_surface_state_v1 import surface_conditions
from reference_population_v1 import PopulationSpecV1

CTX=8841;P,N2,T2=9941,9942,9943
CUE_A,DEP_A,OBJ_A=141,142,143
CUE_B,DEP_B,OBJ_B=241,242,243
HCUE,HDEP,HOBJ=341,342,343
BODY=(831,832,833)


def u(text):return tuple(text.encode())


def name(o,entity,text,source):
    o.contact(CONTACT_SCENE,(7,0,1,entity),source,True,True)
    o.contact(CONTACT_SURFACE,u(text),source,True,True)


def clause(o,atoms,text,source):
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),source,True,True)
    o.contact(CONTACT_SURFACE,u(text),source,True,True)


def body_clause(o,atoms,text,source):
    o.contact(CONTACT_BODY_TARGET,(atoms[0],),source,True,True)
    o.contact(CONTACT_BODY_STATE,BODY,source,True,True)
    clause(o,atoms,text,source)


def build():
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    words=((CUE_A,'dak'),(DEP_A,'lum'),(OBJ_A,'dog'),(CUE_B,'nif'),(DEP_B,'pek'),
           (OBJ_B,'dog'),(HCUE,'wug'),(HDEP,'blick'),(HOBJ,'dog'))
    for entity,text in words:
        name(o,entity,text,P);name(o,entity,text,N2)
    for source in (P,T2):
        clause(o,(CUE_A,DEP_A,OBJ_A),'the dak lum the dog.',source)
        clause(o,(CUE_B,DEP_B,OBJ_B),'the nif pek the dog.',source)
    body_clause(o,(CUE_A,DEP_A,OBJ_A),'the daks lums the dog.',P)
    body_clause(o,(CUE_A,DEP_A,OBJ_A),'the daks lums the dog.',9101)
    body_clause(o,(CUE_B,DEP_B,OBJ_B),'the nifs peks the dog.',P)
    body_clause(o,(CUE_B,DEP_B,OBJ_B),'the nifs peks the dog.',9102)
    return o


def stage(o,target,source):
    o.contact(CONTACT_BODY_TARGET,(target,),source,True,True)
    o.contact(CONTACT_BODY_STATE,BODY,source,True,True)
    o.contact(CONTACT_PARTNER_CONTEXT,(1,7,P),source+1,True,True)
    o.contact(CONTACT_SCENE,(7,CTX,3,HCUE,HDEP,HOBJ),source+2,True,True)
    return o.tick()


def main():
    started=time.perf_counter();checks={};o=build();checkpoint=copy.deepcopy(o.checkpoint())
    checks['raw_training_has_no_authored_condition_tail']=not o.entity_conditions
    dependency_supported=getattr(o.language,'dependency_supported',lambda *_:False)
    checks['directional_recipe_is_independently_supported']=dependency_supported(CTX,0,1)

    held=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));action=stage(held,HCUE,9200)
    condition=surface_conditions(held,HCUE,BODY_STATE_TAG)
    complete=getattr(held.language,'complete_dependencies',lambda *_:None)
    completed=complete(CTX,(condition[0] if condition else 0,0,0))
    checks['source_port_drives_heldout_dependent_form']=(
        bool(condition) and completed==(condition[0],condition[0],0)
        and isinstance(action,ActionV2) and action.payload==u('the wugs blicks the dog.'))
    learned={} if not isinstance(action,ActionV2) else held.contact(
        CONTACT_CONSEQUENCE,(action.ticket,1),action.source,True,True)
    checks['directional_forms_participate_before_independent_credit']=(
        isinstance(action,ActionV2) and action.body_occurrence>0
        and sum(1 for kind,_,_,_ in action.selection_occurrences if kind==PREF_FORM)==2
        and learned.get('credit',0)>0 and learned.get('selection_network_updates',0)==1)

    reverse=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));reverse_action=stage(reverse,HDEP,9210)
    checks['dependent_port_does_not_reverse_the_learned_edge']=(
        not isinstance(reverse_action,ActionV2) or reverse_action.payload!=u('the wugs blicks the dog.'))
    lesion=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    if hasattr(lesion.language,'_dependency_sources'):
        lesion.language._dependency_sources.clear();lesion.language._rebuild_indices()
    lesion_action=stage(lesion,HCUE,9220)
    checks['dependency_recipe_lesion_prevents_remote_form']=(
        not isinstance(lesion_action,ActionV2) or lesion_action.payload!=u('the wugs blicks the dog.'))
    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    for source in (P,9101):withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(source,),9300+source,True,True)
    withdrawn_action=stage(withdrawn,HCUE,9225)
    checks['dependency_requires_live_independent_sources']=(
        not withdrawn.language.dependency_supported(CTX,0,1)
        and (not isinstance(withdrawn_action,ActionV2) or withdrawn_action.payload!=u('the wugs blicks the dog.')))

    conflict=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    conflict.contact(CONTACT_BODY_TARGET,(HCUE,),9230,True,True);conflict.contact(CONTACT_BODY_STATE,BODY,9230,True,True)
    conflict_condition=surface_conditions(conflict,HCUE,BODY_STATE_TAG)
    conflict_result=(getattr(conflict.language,'complete_dependencies',lambda *_:None)(
        CTX,(conflict_condition[0],999999,0)) if conflict_condition else None)
    checks['contradictory_dependent_condition_refuses_completion']=conflict_result is None

    replay=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    checks['checkpoint_rebuilds_direction_without_completion_cache']=(
        replay.digest()==o.digest() and 'completed_dependencies' not in json.dumps(checkpoint))
    probe=(condition[0],0,0) if condition else (0,0,0)
    before=complete(CTX,probe);touches_before=held.language.last_lookup_touches
    for index in range(512):
        observe=getattr(held.language,'observe_dependency',lambda *_:None)
        observe(50000+index,0,1,60000+index);observe(50000+index,0,1,61000+index)
    after=complete(CTX,probe);touches_after=held.language.last_lookup_touches
    checks['direction_lookup_survives_512_context_decoys']=(
        before==after==(condition[0],condition[0],0) if condition else False)
    checks['direction_lookup_survives_512_context_decoys']=(
        checks['direction_lookup_survives_512_context_decoys'] and touches_before==touches_after==1)

    result={'schema':'agi.reference-organism-directional-dependency.v1','pass':all(checks.values()),
        'checks':checks,'metrics':{'dependency_touches_before_after':[touches_before,touches_after],
        'decoy_contexts':512},'runtime_llm':False,'expected_output_ingress':False,'graph_flip':False,
        'human_language_mastery':False,'direct_parity':False,
        'claim':'RAW_CONTACT_DIRECTIONAL_DEPENDENCY_ON_CONTINUING_REFERENCE_ORGANISM_ONLY',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_DIRECTIONAL_DEPENDENCY '+('GREEN' if result['pass'] else 'RED')+
          f" checks={sum(checks.values())}/{len(checks)} decoys=512")
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)


if __name__=='__main__':main()
