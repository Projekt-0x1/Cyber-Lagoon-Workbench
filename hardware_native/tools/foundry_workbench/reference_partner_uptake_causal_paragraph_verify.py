#!/usr/bin/env python3
"""N+1: partner-grounded causal uptake changes later paragraph frontier, not truth."""
from __future__ import annotations
import copy,json,time
from reference_life_function_curriculum_v1 import (
    LifeCurriculumEventV2,ReferenceLifeFunctionRuntimeV2,
    canonical_life_function_curriculum_v2,canonical_species_program_v2,
)
from reference_predictive_credit_profile_v1 import Q

A=0xE901;B=0xE911;Y=0xE921

def developed():
    species=canonical_species_program_v2();runtime=ReferenceLifeFunctionRuntimeV2(species)
    runtime.run(canonical_life_function_curriculum_v2());return species,runtime

def target_scene(runtime):
    rows=[]
    for seq,identity in runtime.occurrences.items():
        scene=runtime.contact.scenes.get(int(identity))
        if scene is not None and int(scene.context)==100 and tuple(scene.atoms)==(0xA104,):rows.append(int(seq))
    if not rows:raise RuntimeError('paragraph:target-scene')
    return max(rows)

def leaf(runtime):return runtime.adult.language_adult.leaf(100,(0xA104,))

def relation_coordinates(adult,leaf_identity,programs):
    coordinates=[]
    for program in programs:
        factor=adult.language_adult.programs.factor(program)
        if factor is None:continue
        for row in adult.causal_message_rows(leaf_identity):
            if adult.causal_program_for_row(row,factor,False)==program:
                coordinates.append(tuple(map(int,row[2:5])));break
    return tuple(coordinates)

def apply(runtime,lane,source=0,payload=()):
    seq=int(runtime.cursor)+1;result=runtime.apply(LifeCurriculumEventV2(seq,lane,int(source),tuple(payload)));return seq,result

def teach_prefix(runtime,partner,authenticated=True):
    scene=target_scene(runtime);action,_=apply(runtime,'partner_causal_dialogue_opportunity',partner,(scene,))
    apply(runtime,'causal_dialogue_return',partner+0x40,(action,Q,0,1))
    current=leaf(runtime);surface,programs=runtime.adult.compose_causal_component(current.identity,channel=partner)
    if not programs:raise RuntimeError('paragraph:no-programs')
    first=bytes(runtime.adult.language_adult.public_surface(programs[0]) or b'')
    if not first:raise RuntimeError('paragraph:first-link')
    lane='authenticated_utterance' if authenticated else 'utterance'
    apply(runtime,lane,partner,tuple(first));apply(runtime,lane,partner,tuple(first))
    return first

def main():
    started=time.perf_counter();species,base=developed();root=leaf(base)
    full,full_programs=base.adult.compose_causal_component(root.identity,channel=B)
    full=bytes(full);full_programs=tuple(full_programs)
    before_history=base.adult.language_adult.completed_public_episode_history(B)
    _repeat,_=base.adult.compose_causal_component(root.identity,channel=B)
    after_history=base.adult.language_adult.completed_public_episode_history(B)

    uptake=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(base.checkpoint()));first=teach_prefix(uptake,A,True)
    uptake_cp=copy.deepcopy(uptake.checkpoint());uptake_root=leaf(uptake);a_surface,a_programs=uptake.adult.compose_causal_component(uptake_root.identity,channel=A)
    b_surface,b_programs=uptake.adult.compose_causal_component(uptake_root.identity,channel=B)
    full_coordinates=relation_coordinates(base.adult,root.identity,full_programs)
    uptake_coordinates=relation_coordinates(uptake.adult,uptake_root.identity,a_programs)

    resumed=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(uptake_cp));r_surface,r_programs=resumed.adult.compose_causal_component(leaf(resumed).identity,channel=A)

    yoked=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(base.checkpoint()));teach_prefix(yoked,Y,False);y_surface,y_programs=yoked.adult.compose_causal_component(leaf(yoked).identity,channel=Y)

    withdrawn=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(uptake_cp));apply(withdrawn,'language_source_withdrawal',A,());w_surface,w_programs=withdrawn.adult.compose_causal_component(leaf(withdrawn).identity,channel=A)

    loaded=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(uptake_cp))
    for offset in range(6):apply(loaded,'body_load',0xE980,(96+offset,Q//2))
    l_surface,l_programs=loaded.adult.compose_causal_component(leaf(loaded).identity,channel=A)
    apply(loaded,'quiet',0,(64,));rec_surface,rec_programs=loaded.adult.compose_causal_component(leaf(loaded).identity,channel=A)

    cp_text=json.dumps(uptake_cp,sort_keys=True)
    checks={
      'baseline_is_certified_multi_relation_message':(
          len(full_programs)>=3 and len(set(full_programs))==len(full_programs)
          and all(base.adult.language_adult.programs.factor(program) is not None
                  for program in full_programs)),
      'internal_planning_does_not_forge_public_episode_history':before_history==after_history,
      'authenticated_partner_uptake_shortens_same_causal_situation':0<len(a_programs)<len(full_programs) and len(a_surface)<len(full),
      'uptake_omits_acknowledged_relation_but_preserves_certified_remainder':(
          first not in bytes(a_surface) and bytes(a_surface)
          and 1<len(a_programs)<len(full_programs)
          and set(uptake_coordinates)<set(full_coordinates)),
      'other_partner_does_not_borrow_uptake':tuple(b_programs)==full_programs and bytes(b_surface)==full,
      'checkpoint_resume_preserves_partner_conditioned_frontier':tuple(r_programs)==tuple(a_programs) and bytes(r_surface)==bytes(a_surface),
      'yoked_non_authenticated_contact_cannot_author_uptake':len(y_programs)==len(full_programs) and first in bytes(y_surface),
      'withdrawing_uptake_source_reopens_explicitness':len(w_programs)==len(full_programs) and first in bytes(w_surface),
      'body_load_further_bounds_only_current_novel_frontier':0<len(l_programs)<len(a_programs),
      'quiet_recovery_restores_partner_frontier_without_reteaching':tuple(rec_programs)==tuple(a_programs) and bytes(rec_surface)==bytes(a_surface),
      'checkpoint_contains_no_complete_paragraph_surface':full.decode() not in cp_text,
      'one_life_run_plus_bounded_checkpoint_forks':(
          base.cursor==len(canonical_life_function_curriculum_v2().events)
          and len(full_coordinates)==len(full_programs)
          and len(uptake_coordinates)==len(a_programs)),
    }
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-partner-uptake-causal-paragraph.v1','contract':'FOUNDRY_PARTNER_UPTAKE_CAUSAL_PARAGRAPH_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'checks':checks,'failed':failed,'before':{'links':len(full_programs),'bytes':len(full),'sentences':full.count(b'.')},'after':{'partner_a_links':len(a_programs),'partner_a_bytes':len(a_surface),'partner_b_links':len(b_programs),'loaded_links':len(l_programs),'recovered_links':len(rec_programs)},'visible_language':{'full':full.decode(),'partner_a':bytes(a_surface).decode(),'loaded':bytes(l_surface).decode()},'remaining_red':['ADVERSE_UPTAKE_REVERSAL','PARAPHRASTIC_UPTAKE_BEYOND_LEARNED_CAUSAL_FACTORS','DIRECT_PARTNER_PARAGRAPH_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
