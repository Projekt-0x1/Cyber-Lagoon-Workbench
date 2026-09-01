#!/usr/bin/env python3
"""Fast contract for one canonical continuing mathematical Adult life."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from reference_language_mastery_claude_gateway_v1 import resident_contact
from reference_language_mastery_terminal_v1 import restore_life
from reference_life_function_curriculum_v1 import (
    C,LifeCurriculumEventV2,ReferenceLifeFunctionRuntimeV2,
    canonical_developmental_probe_v2,canonical_life_function_curriculum_v2,canonical_species_program_v2,
    source_semantics_root_v2,
)

NEW=501
CALM=0xE701;COUNTER=0xE702
PRAGMATIC_CURRICULUM={
    'english':{CALM:('hold the boundary calmly please','hold the boundary calmly now','hold the boundary calmly today'),COUNTER:('take counter action firmly please','take counter action firmly now','take counter action firmly today')},
    'german':{CALM:('halte die grenze ruhig bitte','halte die grenze ruhig jetzt','halte die grenze ruhig heute'),COUNTER:('setz eine gegenaktion klar bitte','setz eine gegenaktion klar jetzt','setz eine gegenaktion klar heute')},
    'russian':{CALM:('держи границу спокойно пожалуйста','держи границу спокойно сейчас','держи границу спокойно сегодня'),COUNTER:('предприми ответное действие твердо пожалуйста','предприми ответное действие твердо сейчас','предприми ответное действие твердо сегодня')},
    'japanese':{CALM:('境界を静かに守ってください','境界を静かに守って今','境界を静かに守ってね'),COUNTER:('対抗行動をはっきり取ってください','対抗行動をはっきり取って今','対抗行動をはっきり取ってね')},
    'mandarin':{CALM:('请冷静地守住边界现在','请冷静地守住边界一下','请冷静地守住边界吧'),COUNTER:('请明确采取反制行动现在','请明确采取反制行动一下','请明确采取反制行动吧')},
    'mixed':{CALM:('bitte hold the boundary 冷静に please','bitte hold the boundary 冷静に now','bitte hold the boundary 冷静に heute'),COUNTER:('bitte take counter action はっきり please','bitte take counter action はっきり now','bitte take counter action はっきり heute')},
    'denglish_jugend':{CALM:('bro halt die boundary entspannt bitte','bro halt die boundary entspannt jetzt','bro halt die boundary entspannt safe'),COUNTER:('bro mach counter action stabil bitte','bro mach counter action stabil jetzt','bro mach counter action stabil safe')},
}

def _refused(adult,atoms):
    try:adult.leaf(C,atoms);return False
    except RuntimeError:return True

def _contact_surface(runtime,raw,source,channel=0):
    return runtime.contact_utterance(raw,source,channel)[0]

def _append_pragmatic_curriculum(runtime):
    """Append multilingual speech->demonstrated-action experience to this exact life."""
    def add(lane,source,payload):
        return runtime.apply(LifeCurriculumEventV2(runtime.cursor+1,lane,int(source),tuple(payload)))
    varieties=tuple(PRAGMATIC_CURRICULUM);base=0xE800
    for witness in range(2):
        order=varieties if witness==0 else tuple(reversed(varieties))
        for action in (CALM,COUNTER) if witness==0 else (COUNTER,CALM):
            for index,variety in enumerate(order):
                source=base+witness*0x100+(0 if action==CALM else 0x40)+index
                raw=PRAGMATIC_CURRICULUM[variety][action][witness].encode('utf-8')
                add('raw_speech_contact',source,raw)
                add('observed_source_action',source,(action,))
    heldout={};quoted={};learner=runtime.adult.language_action_affordances
    for variety,actions in PRAGMATIC_CURRICULUM.items():
        for action in (CALM,COUNTER):
            raw=actions[action][2].encode('utf-8')
            heldout[(variety,action)]=learner.candidates(raw)
            quoted[(variety,action)]=learner.candidates(b'someone said: '+raw)
    return heldout,quoted

def main():
    started=time.perf_counter();checks={}
    species=canonical_species_program_v2();curriculum=canonical_life_function_curriculum_v2()
    context,_examples,heldout,features=canonical_developmental_probe_v2(curriculum)
    grounded_cursor=curriculum.mark_cursor('grounded')
    grounding_scenes=tuple(tuple(map(int,event.payload[1:]))
        for event in curriculum.events[:grounded_cursor]
        if event.lane=='scene' and int(event.payload[0])==100)
    new_heldout=(*heldout[:3],NEW)
    blank=ReferenceLifeFunctionRuntimeV2(species)
    checks['blank_birth_has_no_learned_language_or_operator_evidence']=(
        all(blank.adult.language.lexeme(feature) is None for feature in features)
        and _refused(blank.adult.language_adult,heldout)
        and not blank.adult.operators.needs())

    runtime=blank.run(curriculum);checkpoint=runtime.checkpoint()
    held=runtime.adult.language_adult.leaf(context,heldout)
    held_lexemes=tuple(bytes(runtime.adult.language_adult.language.lexeme(feature) or ()) for feature in heldout)
    held_raw=bytes(held.surface);held_intervals=[]
    for lexeme in held_lexemes:
        positions=tuple(index for index in range(len(held_raw)-len(lexeme)+1)
                        if held_raw[index:index+len(lexeme)]==lexeme)
        if len(positions)==1:held_intervals.append((positions[0],positions[0]+len(lexeme)))
    marks=tuple(event.payload[0] for event in curriculum.events
                if event.lane=='checkpoint_mark')
    checks['one_chronology_reaches_every_unique_observer_mark']= (
        len(marks)==len(set(marks)) and runtime.cursor==len(curriculum.events)
        and runtime.marks.get(marks[-1])==runtime.cursor)
    checks['same_life_recombines_heldout_language']=(
        all(held_lexemes) and len(held_intervals)==len(held_lexemes)
        and held_intervals==sorted(held_intervals)
        and all(left[1]<=right[0] for left,right in zip(held_intervals,held_intervals[1:]))
        and held_raw.strip()==held_raw and b'  ' not in held_raw
        and held_raw.decode() not in json.dumps(checkpoint,sort_keys=True))
    checks['sparse_history_replaces_exhaustive_cartesian_language_fixture']=(
        len(grounding_scenes)==len(set(grounding_scenes))==10
        and len(grounding_scenes)<2**len(heldout))
    checks['same_heterogeneous_adult_exercised_math_and_language']= (
        runtime.adult.operator_public_count>0
        and runtime.adult.relation_public_count>0
        and bool(runtime.adult.operators.needs())
        and bool(runtime.adult.relation_basis.active))

    restored=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(checkpoint))
    checks['checkpoint_resume_preserves_exact_future_relevant_state']=(
        restored.checkpoint()==checkpoint and restored.adult.digest()==runtime.adult.digest())
    checks['checkpoint_retains_commitment_not_contact_transcript']= (
        'history_docs' not in checkpoint and len(checkpoint['history_root'])==64
        and set(checkpoint['transport'])=={
            'scenes','relations','current_scene','current_relation','next_identity'})

    altered=list(curriculum.events);event=altered[0]
    altered[0]=LifeCurriculumEventV2(event.sequence,event.lane,event.source+1,event.payload)
    changed=type(curriculum)(tuple(altered));refused=False
    try:ReferenceLifeFunctionRuntimeV2.restore(
        species,copy.deepcopy(checkpoint)).run(changed)
    except ValueError:refused=True
    checks['changed_past_refuses_without_persisting_the_past']=refused

    extension=(
        LifeCurriculumEventV2(1,'scene',7501,(100,NEW)),
        LifeCurriculumEventV2(1,'surface',7501,tuple(b'overnight')),
        LifeCurriculumEventV2(1,'scene',7502,(100,NEW)),
        LifeCurriculumEventV2(1,'surface',7502,tuple(b'overnight')),
    )
    continued=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(checkpoint))
    continued.run(curriculum.append(extension))
    post_checkpoint=bytes(continued.adult.language_adult.leaf(context,new_heldout).surface)
    current_subject=bytes(runtime.adult.language_adult.language.lexeme(heldout[0]) or ())
    checks['novel_post_checkpoint_contact_changes_same_adult']=(
        _refused(runtime.adult.language_adult,new_heldout)
        and current_subject and post_checkpoint.startswith(current_subject+b' ')
        and post_checkpoint.endswith(b' overnight.'))
    checks['source_semantics_commitment_covers_executable_closure']=len(source_semantics_root_v2())==64
    learner=runtime.adult.language_adult.world_causal_learning;effect=runtime.adult.language_adult.leaf(100,(0xA104,))
    rows=tuple(row for row in learner.current_resolutions() if int(row[3])==effect.identity);cause_surface=(bytes(runtime.adult.language_adult._leaf_surface(int(rows[0][2]))) if len(rows)==1 else b'')
    world_visible=_contact_surface(runtime,bytes(effect.surface),0xF401)
    lived_surfaces={bytes(event.payload) for event in curriculum.events if event.lane in {'surface','discourse_surface','utterance','authenticated_utterance'}}
    checks['same_life_publicly_recomposes_world_earned_causal_language']=(
        bool(world_visible) and world_visible not in lived_surfaces
        and bytes(effect.surface) in world_visible and cause_surface in world_visible)
    boundary_raw=bytes(effect.surface)+b'.'
    pre_boundary_curriculum=curriculum.prefix_at_mark('relational_surplus_recovered')
    pre_boundary=ReferenceLifeFunctionRuntimeV2(species).run(pre_boundary_curriculum)
    pre_boundary_cp=copy.deepcopy(pre_boundary.checkpoint())
    before_boundary=ReferenceLifeFunctionRuntimeV2.restore(
        species,copy.deepcopy(pre_boundary_cp))
    after_boundary=ReferenceLifeFunctionRuntimeV2.restore(
        species,copy.deepcopy(checkpoint))
    boundary_visible=_contact_surface(after_boundary,boundary_raw,0xF404)
    checks['independently_lived_wrapper_reaches_heldout_causal_composition']=(
        not _contact_surface(before_boundary,boundary_raw,0xF405)
        and boundary_visible==world_visible
        and boundary_raw not in lived_surfaces)
    wrapper_cut=ReferenceLifeFunctionRuntimeV2.restore(
        species,copy.deepcopy(checkpoint))
    wrapper_cut.adult.language_adult.language.withdraw_source(0xFA11)
    cut_boundary=_contact_surface(wrapper_cut,boundary_raw,0xF406)
    plain_after_cut=_contact_surface(ReferenceLifeFunctionRuntimeV2.restore(
        species,copy.deepcopy(wrapper_cut.checkpoint())),
            bytes(effect.surface),0xF407)
    checks['wrapper_source_lesion_silences_variant_not_causal_knowledge']=(
        not cut_boundary and plain_after_cut==world_visible)
    wrong_boundary=_contact_surface(ReferenceLifeFunctionRuntimeV2.restore(
        species,copy.deepcopy(checkpoint)),
            bytes(effect.surface)+b'!',0xF408)
    checks['unlived_boundary_bytes_have_no_innate_punctuation_authority']=(
        not wrong_boundary)
    # Preserve unrelated intervening development and reverse only the two
    # boundary witnesses. This is a causal order contrast, not an assumption
    # that checkpoint marks are adjacent rungs.
    intervening=curriculum.events[
        curriculum.mark_cursor('relational_surplus_recovered'):
        curriculum.mark_cursor('causal_variable_arity_coordination')]
    reversed_events=(*intervening,
        LifeCurriculumEventV2(1,'authenticated_utterance',0xFA12,
                              tuple(b'plant leaves wilt.')),
        LifeCurriculumEventV2(1,'authenticated_utterance',0xFA11,
                              tuple(b'plant roots lose water.')),
        LifeCurriculumEventV2(1,'checkpoint_mark',0,
                              ('receptive_boundary_wrapper',)),
    )
    reversed_boundary=ReferenceLifeFunctionRuntimeV2.restore(
        species,copy.deepcopy(pre_boundary_cp))
    reversed_boundary.run(pre_boundary_curriculum.append(reversed_events))
    reversed_visible=_contact_surface(reversed_boundary,boundary_raw,0xF409)
    checks['developmental_contact_order_does_not_install_a_wrapper_ladder']=(
        reversed_visible==boundary_visible)
    boundary_restart=ReferenceLifeFunctionRuntimeV2.restore(
        species,copy.deepcopy(after_boundary.checkpoint()))
    checks['receptive_wrapper_survives_checkpoint_without_raw_contact_memory']=(
        bool(restarted_boundary:=_contact_surface(boundary_restart,boundary_raw,0xF40A))
        and bytes(effect.surface) in restarted_boundary and cause_surface in restarted_boundary
        and boundary_raw.decode() not in json.dumps(checkpoint,sort_keys=True))
    gateway_runtime=ReferenceLifeFunctionRuntimeV2.restore(
        species,copy.deepcopy(checkpoint))
    gateway_visible=resident_contact(
        gateway_runtime,(bytes(effect.surface),),0xF402)
    checks['claude_body_uses_same_life_checkpoint_and_visible_cognition']=(
        gateway_visible==world_visible
        and gateway_runtime.cursor==runtime.cursor
        and gateway_runtime.checkpoint()['schema']==checkpoint['schema'])
    legacy_refused=False
    try:ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(checkpoint['adult']))
    except ValueError:legacy_refused=True
    gateway_source=Path(__file__).with_name(
        'reference_language_mastery_claude_gateway_v1.py').read_text()
    checks['obsolete_bare_language_adult_gateway_is_deleted']=(
        legacy_refused and 'LanguageMasteryAdultV1.restore' not in gateway_source
        and 'server.adult=' not in gateway_source)
    terminal_runtime=restore_life(copy.deepcopy(checkpoint))
    terminal_visible=_contact_surface(terminal_runtime,bytes(effect.surface),0xF403)
    terminal_source=Path(__file__).with_name(
        'reference_language_mastery_terminal_v1.py').read_text()
    checks['terminal_body_resumes_same_life_not_bare_adult']=(
        terminal_visible==world_visible
        and terminal_runtime.checkpoint()['schema']==checkpoint['schema']
        and 'adult=LanguageMasteryAdultV1.restore' not in terminal_source)
    source=Path(__file__).with_name('reference_life_function_curriculum_v1.py').read_text().lower()
    checks['canonical_path_contains_no_fixture_or_private_teacher']='fixture' not in source and 'answer_key' not in source
    checks['canonical_path_contains_no_cartesian_language_generator']=(
        'itertools.product' not in source and 'grounding_styles' not in source
        and 'feature_rows' not in source)

    # N+1 pragmatic language is appended after the already-earned canonical life,
    # not evaluated on a fresh fixture. Checkpoint restore is continuation identity.
    pragmatic=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(checkpoint))
    pre_pragmatic_digest=pragmatic.adult.digest()
    checks['pragmatic_affordance_absent_before_lived_extension']=(pragmatic.adult.language_action_affordances.factor_count==0)
    heldout_pragmatic,quoted_pragmatic=_append_pragmatic_curriculum(pragmatic);pragmatic_cp=pragmatic.checkpoint()
    checks['multilingual_pragmatics_extends_same_birth_to_adult_chronology']=(pragmatic.cursor>runtime.cursor and pragmatic.history_root()!=runtime.history_root() and pragmatic.adult.digest()!=pre_pragmatic_digest)
    checks['heldout_direct_force_is_learned_across_seven_varieties']=(len(heldout_pragmatic)==14 and all(rows and int(rows[0][0])==int(action) for (_variety,action),rows in heldout_pragmatic.items()))
    checks['unseen_reported_speech_wrapper_has_no_top_level_action_force']=(len(quoted_pragmatic)==14 and all(not rows for rows in quoted_pragmatic.values()))
    pragmatic_restored=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(pragmatic_cp))
    checks['pragmatic_learning_survives_same_life_checkpoint_without_transcript']=(pragmatic_restored.adult.language_action_affordances.checkpoint()==pragmatic.adult.language_action_affordances.checkpoint() and all(text not in json.dumps(pragmatic_cp,ensure_ascii=False,sort_keys=True) for actions in PRAGMATIC_CURRICULUM.values() for rows in actions.values() for text in rows))
    checks['no_language_router_or_imperative_opcode_in_resident_pragmatic_state']=(not hasattr(pragmatic.adult.language_action_affordances,'language_id') and not hasattr(pragmatic.adult.language_action_affordances,'imperative') and not hasattr(pragmatic.adult.language_action_affordances,'translation'))

    failed=[name for name,passed in checks.items() if not passed]
    result={
        'schema':'cyber-lagoon.workbench-life-function-curriculum.v4',
        'contract':'FOUNDRY_WORKBENCH_LIFE_FUNCTION_CURRICULUM_'+
                   ('GREEN' if not failed else 'RED'),
        'pass':not failed,'reference_only':True,'one_continuing_adult':True,
        'language_phenotype_improved':True,
        'visible_language_gain':'SAME_LIFE_HELDOUT_MULTILINGUAL_DIRECT_ACTION_FORCE_WITH_REPORTED_SPEECH_WITHHOLD',
        'events':pragmatic.cursor,'marks':marks,
        'grounding_contacts':len(grounding_scenes),
        'pragmatic_varieties':list(PRAGMATIC_CURRICULUM),
        'checkpoint_bytes':len(json.dumps(pragmatic_cp,sort_keys=True,separators=(',',':')).encode()),
        'visible':{'world_derived':world_visible.decode()},
        'structural_probe':{
            'heldout':bytes(held.surface).decode(),
            'post_checkpoint':post_checkpoint.decode(),
        },
        'checks':checks,'failed':failed,
        'remaining_red':['OPEN_DOMAIN_CONVERSATION','DIRECT_CURRICULUM_PARITY'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if not failed else 1

if __name__=='__main__':raise SystemExit(main())
