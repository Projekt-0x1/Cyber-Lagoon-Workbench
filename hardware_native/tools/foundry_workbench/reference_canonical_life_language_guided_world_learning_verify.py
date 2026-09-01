#!/usr/bin/env python3
"""One canonical Life lets reliable language guide consequence-owned learning."""
from __future__ import annotations

import json
import time

from reference_language_mastery_terminal_v1 import emit_choice
from reference_life_function_curriculum_v1 import (
    ALLOWED_LANES,
    LifeCurriculumEventV2,
    LifeFunctionCurriculumV2,
    ReferenceLifeFunctionRuntimeV2,
    canonical_life_function_curriculum_v2,
    canonical_species_program_v2,
)

REQUIRED_WORLD_LANES={
    'authenticated_utterance','causal_field','resident_world_step','source_withdrawal'}
TARGET_SOURCE=0xD101
TARGET_INSTRUCTION=b'warm air dries the soil since bright sunlight warms the greenhouse'

def _target_state(runtime):
    adult=runtime.adult.language_adult;learner=adult.world_causal_learning
    cause=adult.leaf(100,(0xA103,));effect=adult.leaf(100,(0xA104,))
    rows=tuple(row for row in learner.current_resolutions()
               if adult.leaf_equivalent(row[2],cause.identity) and adult.leaf_equivalent(row[3],effect.identity))
    receipt=int(rows[0][4]) if len(rows)==1 else 0
    factor=int(learner.preferred_factor(adult))
    materialized=learner.materialize_program(adult,receipt,factor) if receipt and factor else None
    program=0 if materialized is None else int(materialized.identity)
    return adult,learner,cause,effect,receipt,factor,program

def _authentication_lesion(curriculum):
    rows=[];lesions=0
    for event in curriculum.events:
        if (event.lane=='authenticated_utterance' and event.source==TARGET_SOURCE
                and bytes(event.payload)==TARGET_INSTRUCTION):
            event=LifeCurriculumEventV2(
                event.sequence,'utterance',event.source,event.payload)
            lesions+=1
        rows.append(event)
    if lesions!=1:raise RuntimeError('canonical-life:target-instruction')
    return LifeFunctionCurriculumV2(tuple(rows))

def main():
    started=time.perf_counter();checks={};failure='';visible=b''
    full_curriculum=canonical_life_function_curriculum_v2()
    curriculum=full_curriculum.prefix_at_mark('language_guided_world_learning')
    missing=REQUIRED_WORLD_LANES-set(ALLOWED_LANES)
    checks['one_runtime_exposes_required_world_contact_laws']=not missing
    checks['world_history_never_names_a_selected_intervention']=all(
        'selected' not in json.dumps(event.document(),sort_keys=True).lower()
        for event in curriculum.events)
    expected=b'Because bright sunlight warms the greenhouse, warm air dries the soil.'
    checks['heldout_public_wording_absent_from_lived_contacts']=all(
        expected!=bytes(event.payload)
        for event in curriculum.events
        if event.lane in {'surface','discourse_surface','utterance','authenticated_utterance'}
        and all(isinstance(value,int) and 0<=value<=255 for value in event.payload))

    runtime=None
    try:
        species=canonical_species_program_v2()
        runtime=ReferenceLifeFunctionRuntimeV2(species).run(curriculum)
        adult,learner,cause,effect,receipt,factor,public_program=_target_state(runtime)
        chunk=adult.programs.chunks.get(public_program) if public_program else None
        certificates=(() if chunk is None or len(chunk.members)!=2 or not factor else
                      (learner.language_relation_certificate(
                          adult,factor,*tuple(map(int,chunk.members))),))
        visible=emit_choice(adult,public_program)
        restored=ReferenceLifeFunctionRuntimeV2.restore(species,runtime.checkpoint())
        restored_state=_target_state(restored)
        restored_visible=emit_choice(restored_state[0],restored_state[-1])

        checks['one_continuing_adult_reaches_canonical_world_language_mark']=(
            runtime.marks.get('language_guided_world_learning',0)==runtime.cursor)
        checks['three_independent_consequence_blocks_settle_target_relation']=(
            receipt>0 and learner.complete_source_blocks(receipt)==3
            and learner.resolve(receipt)==(cause.identity,effect.identity))
        checks['withdrawing_testimony_does_not_withdraw_world_learning']=(
            learner.testimony_source(53505)
            in learner.testimony_withdrawn
            and learner.resolve(receipt)==(cause.identity,effect.identity))
        checks['public_program_is_derived_from_live_world_certificate']=(
            len(certificates)==1 and certificates[0][:2]==(1,factor)
            and tuple(certificates[0][2:4])==tuple(map(int,chunk.members))
            and int(certificates[0][4])==receipt)
        contact_surfaces=tuple(
            bytes(event.payload) for event in curriculum.events
            if event.lane in {'surface','discourse_surface','utterance','authenticated_utterance'}
            and all(isinstance(value,int) and 0<=value<=255 for value in event.payload))
        pieces=adult.language.historical_span_pieces(factor) or ()
        port_order=tuple(int(piece.port) for piece in pieces if int(piece.kind)==2)
        grounded_support={int(fid):sum(1 for source in rows if source not in learner.grounding.withdrawn)
                          for fid,rows in learner.grounding.rows.items()
                          if learner.grounding.orientation(fid)}
        checks['public_realization_uses_unique_strongest_grounded_form']=(
            factor in grounded_support and len(grounded_support)>=2
            and all(grounded_support[factor]>support for fid,support in grounded_support.items()
                    if int(fid)!=factor))
        checks['visible_output_is_complete_learned_causal_sentence']=(
            visible==expected and visible!=TARGET_INSTRUCTION and visible not in contact_surfaces)
        checks['learned_causal_realization_fronts_cause_and_closes_sentence']=(
            port_order==(0,1) and visible.startswith(b'Because ')
            and b', ' in visible and visible.endswith(b'.')
            and visible.index(bytes(cause.surface))<visible.index(bytes(effect.surface)))
        checks['checkpoint_resume_preserves_learned_relation_and_visible_output']=(
            restored_state[4:]==(receipt,factor,public_program)
            and restored_visible==visible)

        target_event=next(event for event in curriculum.events
                          if event.lane=='authenticated_utterance'
                          and event.source==TARGET_SOURCE
                          and bytes(event.payload)==TARGET_INSTRUCTION)
        pre=ReferenceLifeFunctionRuntimeV2(species).run(
            curriculum.prefix(target_event.sequence-1))
        pre_cp=pre.checkpoint()
        authenticated=ReferenceLifeFunctionRuntimeV2.restore(species,pre_cp)
        unauthenticated=ReferenceLifeFunctionRuntimeV2.restore(species,pre_cp)
        authenticated.apply(target_event)
        unauthenticated.apply(LifeCurriculumEventV2(
            target_event.sequence,'utterance',target_event.source,target_event.payload))
        ac=authenticated.adult.language_adult;uc=unauthenticated.adult.language_adult
        acause=ac.leaf(100,(0xA103,));aeffect=ac.leaf(100,(0xA104,))
        target_receipts=[rid for rid,binding in ac.world_causal_learning.bindings.items()
                         if ac.leaf_equivalent(binding.effect,aeffect.identity)
                         and any(ac.leaf_equivalent(cause_id,acause.identity)
                                 for cause_id in binding.causes)]
        target_pre_receipt=int(target_receipts[0]) if len(target_receipts)==1 else 0
        checks['authentication_adds_prospective_relation_authority_not_truth']=(
            target_pre_receipt>0
            and bool(ac.world_causal_learning.current_testimony_resolutions())
            and not uc.world_causal_learning.current_testimony_resolutions()
            and bool(ac.world_causal_learning._guided_coalition(target_pre_receipt))
            and not uc.world_causal_learning._guided_coalition(target_pre_receipt)
            and ac.world_causal_learning.resolve(target_pre_receipt) is None
            and uc.world_causal_learning.resolve(target_pre_receipt) is None)
    except Exception as exc:
        failure=f'{type(exc).__name__}:{exc}'

    failed=[name for name,passed in checks.items() if not passed]
    result={
        'schema':'cyber-lagoon.canonical-life-language-guided-world-learning.v2',
        'contract':'FOUNDRY_CANONICAL_LIFE_LANGUAGE_GUIDED_WORLD_LEARNING_'+
                   ('GREEN' if not failed and not failure else 'RED'),
        'pass':not failed and not failure,
        'reference_only':True,
        'language_phenotype_improved':bool(visible),
        'visible_language_gain':visible.decode(errors='replace') if visible else 'RED',
        'runtime_failure':failure,
        'checks':checks,
        'failed':failed,
        'remaining_red':['RESIDENT_CONVERSATION_SELECTION','DIRECT_PARITY'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if result['pass'] else 1

if __name__=='__main__':raise SystemExit(main())
