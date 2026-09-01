#!/usr/bin/env python3
"""Resident-boundary audit for language as a developmental catalyst.

Raw learned surface may nominate a motor Recipe only through an earned population
recruitment relation.  Language concept ids and motor Recipe ids must remain
distinct; prediction/testimony alone cannot create nonlinguistic transition truth.
"""
from __future__ import annotations
import copy,json,time
import reference_language_guided_cognition_verify as candidate


def prepared():
    spec=candidate.PopulationSpecV1(32768,2,4,42,8)
    o=candidate.ReferenceOrganismV2(spec);candidate.train_language(o);candidate.setup(o,candidate.WORLD1)
    return o


def main():
    started=time.perf_counter();checks={};o=prepared()
    binding=o.language.invert_surface(candidate.u(candidate.INSTRUCTION))[0]
    intersection=tuple(sorted(set(binding.atoms)&set(o.affordances)))
    resolved=o._language_action_candidate(binding)
    checks['surface_contains_no_motor_recipe_identity']=candidate.MOTOR_INSPECT not in binding.atoms and candidate.MOTOR_TEST not in binding.atoms
    checks['surface_affordance_intersection_is_empty']=intersection==()
    checks['resident_candidate_recipe_search']=resolved==candidate.MOTOR_INSPECT and resolved not in binding.atoms
    checks['language_concept_and_motor_recipe_are_distinct']=candidate.INSPECT in binding.atoms and candidate.INSPECT!=candidate.MOTOR_INSPECT

    # Lesion only the learned inspect<->motor recruitment relation. Raw language
    # still inverts, but no motor Recipe may be nominated from it.
    lesion=prepared();lb=lesion.language.invert_surface(candidate.u(candidate.INSTRUCTION))[0]
    inspect_lid=lb.lexical_identities[1]
    cue=lesion._language_action_cue_occurrence(inspect_lid)
    cue_morph=lesion.recruitment.morphology_identity(cue.sites)
    motor_morph=lesion._action_recipe_morphology(candidate.MOTOR_INSPECT)
    relation_id=lesion.recruitment.relation_identity((cue_morph,motor_morph))
    relation=lesion.recruitment.relations.get(relation_id)
    checks['grounding_relation_exists_as_generic_morphology_relation']=relation is not None and relation.credit>0
    if relation is not None: relation.credit=0
    checks['grounding_relation_lesion_removes_language_guidance']=lesion._language_action_candidate(lb)==0
    try:lesion.contact(candidate.CONTACT_SOURCE_UTTERANCE,candidate.u(candidate.INSTRUCTION),candidate.TEACH1,True,True)
    except ValueError as exc:checks['lesioned_language_cannot_smuggle_motor_id']=str(exc)=='organism:source_utterance_ambiguous'
    else:checks['lesioned_language_cannot_smuggle_motor_id']=False

    # Testimony/prediction is nomination only. No motor consequence => no durable
    # nonlanguage transition, no world/somatic truth.
    provisional=prepared();before_world=provisional._world_revisions.row_count;before_soma=provisional._somatic_revisions.row_count
    provisional.contact(candidate.CONTACT_SOURCE_UTTERANCE,candidate.u(candidate.INSTRUCTION),candidate.TEACH1,True,True)
    checks['prediction_only_persistence_refusal']=not provisional.cognition.edges() and provisional._world_revisions.row_count==before_world and provisional._somatic_revisions.row_count==before_soma

    # Two independent actual consequences may create nonlanguage cognition. Once
    # that exists, both language testimony and its grounding recruitment can be
    # removed without erasing the nonlanguage transition.
    grounded=prepared()
    for world,teacher in ((candidate.WORLD1,candidate.TEACH1),(candidate.WORLD2,candidate.TEACH2)):
        candidate.setup(grounded,world);aid=grounded.contact(candidate.CONTACT_SOURCE_UTTERANCE,candidate.u(candidate.INSTRUCTION),teacher,True,True);act=grounded.tick();candidate.settle(grounded,act,world,True);grounded.contact(candidate.CONTACT_WITHDRAW_SOURCE,(teacher,),teacher+70000,True,True)
    checks['authenticated_causal_difference_settlement']=len(grounded.cognition.edges())==1 and grounded.cognition.edges()[0].action==candidate.MOTOR_INSPECT and grounded.cognition.edges()[0].support==2
    for source in candidate.train_language(grounded):
        grounded.contact(candidate.CONTACT_WITHDRAW_SOURCE,(source,),source+80000,True,True)
    # Disable the motor grounding morphology as well: cognition should already own
    # the learned transition and run before testimony/recruitment search.
    grounded.recruitment.lesion_morphology(grounded._action_recipe_morphology(candidate.MOTOR_INSPECT))
    candidate.setup(grounded,candidate.WORLD3);act=grounded.tick()
    checks['language_and_grounding_lesion_leave_nonlanguage_cognition']=isinstance(act,candidate.MotorActionV2) and act.action_id==candidate.MOTOR_INSPECT and not act.source_assertion_ids

    # Quantity pressure: unrelated learned matter may be huge, but hearing one
    # utterance must traverse only physically matching surface/trie incidence.
    scaled=prepared();raw=candidate.u(candidate.INSTRUCTION)
    base=scaled.language.invert_surface(raw);base_touches=scaled.language.last_lookup_touches
    for i in range(4000):
        feature=100000+i;surface=tuple(f'zoe_noise_{i:04d}'.encode())
        scaled.language.observe_naming(feature,surface,200000+i*2);scaled.language.observe_naming(feature,surface,200001+i*2)
    for i in range(512):
        surface=tuple(f'zoe inspectsx{i:04d} the valve.'.encode());context=50000+i
        scaled.language.observe_construction(context,(candidate.REMOTE,candidate.INSPECT,candidate.VALVE),surface,300000+i*2)
        scaled.language.observe_construction(context,(candidate.REMOTE,candidate.INSPECT,candidate.VALVE),surface,300001+i*2)
    expanded=scaled.language.invert_surface(raw);expanded_touches=scaled.language.last_lookup_touches
    restored=type(scaled.language).restore(copy.deepcopy(scaled.language.checkpoint()));replayed=restored.invert_surface(raw);restored_touches=restored.last_lookup_touches
    projection=lambda rows:tuple((row.context,row.atoms,row.lexical_identities) for row in rows)
    checks['large_language_population_preserves_binding']=projection(base)==projection(expanded)==projection(replayed)
    checks['inverse_work_independent_of_unrelated_language_quantity']=expanded_touches==base_touches==restored_touches and expanded_touches<128
    checks['inverse_index_is_rebuildable_not_authority']=scaled.language.checkpoint()==restored.checkpoint()

    all_green=all(checks.values())
    result={'schema':'agi.reference-language-catalyst-host-boundary.v3','pass':all_green,'checks':checks,'reference_only':True,'adult_attached':False,'physical_direct_parity':'NOT_RUN/RED','graph_flip':False,'runtime_llm':False,'resource':{'inverse_touches_base':base_touches,'inverse_touches_scaled':expanded_touches,'lexeme_rows':len(scaled.language._lexeme_sources),'template_rows':len(scaled.language._template_sources)},'claim':'RESIDENT_MORPHOLOGY_SEARCH_AND_SPARSE_LANGUAGE_INVERSE_REFERENCE_ONLY_NOT_DIRECT_PARITY','elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_LANGUAGE_CATALYST_HOST_BOUNDARY '+('GREEN' if all_green else 'RED')+' motor_id_leak=0 resident_search=1 prediction_self_credit=0 language_outer=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if all_green else 1)

if __name__=='__main__':main()
