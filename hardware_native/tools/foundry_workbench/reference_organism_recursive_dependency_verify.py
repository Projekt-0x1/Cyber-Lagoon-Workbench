#!/usr/bin/env python3
"""Prior directed structure bootstraps the next resident dependency frontier."""
from __future__ import annotations

import copy
import json
import time

import reference_organism_directional_dependency_verify as base
from reference_organism_v2 import *
from reference_organism_surface_state_v1 import surface_conditions


def extended():
    o=base.build()
    rows=((base.CUE_A,base.DEP_A,base.OBJ_A,'the daks lums the dogs.',base.P),
          (base.CUE_A,base.DEP_A,base.OBJ_A,'the daks lums the dogs.',9111),
          (base.CUE_A,base.DEP_A,base.OBJ_A,'the daks lums the dogs.',9113),
          (base.CUE_B,base.DEP_B,base.OBJ_B,'the nifs peks the dogs.',base.P),
          (base.CUE_B,base.DEP_B,base.OBJ_B,'the nifs peks the dogs.',9112))
    for cue,dependent,obj,text,source in rows:
        o.contact(CONTACT_BODY_TARGET,(cue,),source,True,True)
        o.contact(CONTACT_BODY_STATE,base.BODY,source,True,True)
        base.clause(o,(cue,dependent,obj),text,source)
    return o


def stage(o,target,source):
    o.contact(CONTACT_BODY_TARGET,(target,),source,True,True)
    o.contact(CONTACT_BODY_STATE,base.BODY,source,True,True)
    o.contact(CONTACT_PARTNER_CONTEXT,(1,7,base.P),source+1,True,True)
    o.contact(CONTACT_SCENE,(7,base.CTX,3,base.HCUE,base.HDEP,base.HOBJ),source+2,True,True)
    return o.tick()


def main():
    started=time.perf_counter();checks={};o=extended();checkpoint=copy.deepcopy(o.checkpoint())
    checks['prior_edge_exposes_one_new_dependency_frontier']=(
        o.language.dependency_supported(base.CTX,0,1)
        and o.language.dependency_supported(base.CTX,1,2)
        and not o.language.dependency_supported(base.CTX,0,2))

    held=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    held.contact(CONTACT_BODY_TARGET,(base.HCUE,),9200,True,True)
    held.contact(CONTACT_BODY_STATE,base.BODY,9200,True,True)
    condition=surface_conditions(held,base.HCUE,BODY_STATE_TAG)
    completed=held.language.complete_dependencies(base.CTX,(condition[0] if condition else 0,0,0))
    held.contact(CONTACT_PARTNER_CONTEXT,(1,7,base.P),9201,True,True)
    held.contact(CONTACT_SCENE,(7,base.CTX,3,base.HCUE,base.HDEP,base.HOBJ),9202,True,True)
    action=held.tick()
    checks['heldout_condition_unfolds_through_two_edges']=(
        bool(condition) and completed==(condition[0],condition[0],condition[0])
        and isinstance(action,ActionV2) and action.payload==base.u('the wugs blicks the dogs.'))
    learned={} if not isinstance(action,ActionV2) else held.contact(
        CONTACT_CONSEQUENCE,(action.ticket,1),action.source,True,True)
    checks['recursive_form_network_participates_before_credit']=(
        isinstance(action,ActionV2) and action.body_occurrence>0
        and sum(1 for kind,_,_,_ in action.selection_occurrences if kind==PREF_FORM)==3
        and learned.get('credit',0)>0 and learned.get('selection_network_updates',0)==1)

    middle_cut=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    middle_cut.language._dependency_sources.pop((base.CTX,1,2),None);middle_cut.language._rebuild_indices()
    cut_action=stage(middle_cut,base.HCUE,9210)
    checks['intermediate_edge_lesion_blocks_only_deeper_form']=(
        isinstance(cut_action,ActionV2) and cut_action.payload==base.u('the wugs blicks the dog.'))
    source_cut=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    source_cut.contact(CONTACT_WITHDRAW_SOURCE,(9111,),9215,True,True)
    source_action=stage(source_cut,base.HCUE,9216)
    checks['deep_edge_requires_live_independent_source']=(
        source_cut.language.dependency_supported(base.CTX,0,1)
        and not source_cut.language.dependency_supported(base.CTX,1,2)
        and isinstance(source_action,ActionV2) and source_action.payload==base.u('the wugs blicks the dog.'))

    reverse=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));reverse_action=stage(reverse,base.HOBJ,9220)
    checks['deep_target_cannot_run_chain_backward']=(
        not isinstance(reverse_action,ActionV2) or reverse_action.payload!=base.u('the wugs blicks the dogs.'))
    cycle=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));cycle_refused=False
    try:cycle.language.observe_dependency(base.CTX,2,0,9225)
    except ValueError:cycle_refused=True
    checks['dependency_cycle_is_not_a_hierarchy']=cycle_refused

    replay=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    checks['checkpoint_rebuilds_recursive_edges_without_unfolded_state']=(
        replay.digest()==o.digest() and 'completed_dependencies' not in json.dumps(checkpoint))
    probe=(condition[0],0,0) if condition else (0,0,0)
    before=held.language.complete_dependencies(base.CTX,probe);touches_before=held.language.last_lookup_touches
    for index in range(512):
        context=50000+index
        for source in (60000+index,61000+index):
            held.language.observe_dependency(context,0,1,source);held.language.observe_dependency(context,1,2,source)
    after=held.language.complete_dependencies(base.CTX,probe);touches_after=held.language.last_lookup_touches
    checks['recursive_lookup_survives_512_dependency_networks']=(
        before==after==(condition[0],condition[0],condition[0]) if condition else False)
    checks['recursive_lookup_survives_512_dependency_networks']=(
        checks['recursive_lookup_survives_512_dependency_networks'] and touches_before==touches_after==2)

    result={'schema':'agi.reference-organism-recursive-dependency.v1','pass':all(checks.values()),
        'checks':checks,'metrics':{'dependency_touches_before_after':[touches_before,touches_after],
        'decoy_networks':512},'runtime_llm':False,'expected_output_ingress':False,'graph_flip':False,
        'human_language_mastery':False,'direct_parity':False,
        'claim':'RECURSIVE_DEPENDENCY_NETWORK_ON_CONTINUING_REFERENCE_ORGANISM_ONLY',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_RECURSIVE_DEPENDENCY '+('GREEN' if result['pass'] else 'RED')+
          f" checks={sum(checks.values())}/{len(checks)} decoys=512")
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)


if __name__=='__main__':main()
