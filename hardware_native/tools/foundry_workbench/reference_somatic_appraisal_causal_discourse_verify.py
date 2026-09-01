#!/usr/bin/env python3
"""RED/GREEN: matched causal truth, different lived somatic/control history changes discourse extent."""
from __future__ import annotations
import copy,json,time

from reference_language_mastery_contact_adapter_v1 import CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1
from reference_language_mastery_adult_v1 import SomaticAppraisalV1
from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2,canonical_life_function_curriculum_v2,
    canonical_species_program_v2,
)
from reference_predictive_credit_profile_v1 import Q

EFFECT=0xA104

def resident_respond(adult,raw,source,channel=0):
    contact=LanguageMasteryContactAdapterV1(adult.language_adult);identity=int(contact.contact(CONTACT_UTTERANCE,tuple(bytes(raw)),int(source)))
    scene=contact.scenes.get(identity)
    if scene is None:return b''
    return adult.compose_causal_component(adult.language_adult.leaf(scene.context,scene.atoms).identity,channel=channel)[0]


def teach_appraisal_language(language_adult):
    """Ordinary learned surfaces over opaque appraisal-coordinate features."""
    features={
        'positive':language_adult._somatic_appraisal_feature(1,1),
        'negative':language_adult._somatic_appraisal_feature(1,0),
        'control':language_adult._somatic_appraisal_feature(2,1),
        'no_control':language_adult._somatic_appraisal_feature(2,0),
        'loaded':language_adult._somatic_appraisal_feature(3,1),
        'settled':language_adult._somatic_appraisal_feature(3,0),
    }
    names={
        'positive':b'manageable',
        'negative':b'aversive',
        'control':b'I can influence what happens',
        'no_control':b'I have little control over what happens',
        'loaded':b'my body is under strain',
        'settled':b'my body is settled',
    }
    for ordinal,(name,feature) in enumerate(features.items()):
        for witness in range(2):
            language_adult.language.observe_naming(feature,names[name],0xFB00+ordinal*8+witness)
    context=language_adult._somatic_appraisal_language_context()
    examples=(
        ((features['positive'],features['control'],features['loaded']),
         b'My current state feels manageable: I can influence what happens, and my body is under strain.'),
        ((features['negative'],features['no_control'],features['settled']),
         b'My current state feels aversive: I have little control over what happens, and my body is settled.'),
    )
    for witness,(atoms,surface) in enumerate(examples):
        language_adult.language.observe_construction(context,atoms,surface,0xFC00+witness)
    return features


def main():
    started=time.perf_counter();checks={};species=canonical_species_program_v2()
    curriculum=canonical_life_function_curriculum_v2().prefix_at_mark('causal_discourse_form_diversity')
    base=ReferenceLifeFunctionRuntimeV2(species).run(curriculum)
    base_cp=copy.deepcopy(base.checkpoint())

    probe=type(base.adult).restore(copy.deepcopy(base.adult.checkpoint()))
    effect=probe.language_adult.leaf(100,(EFFECT,))
    rows=probe.causal_chain_rows(effect.identity)
    first_program=probe.causal_program_for_row(rows[0]) if rows else 0
    # Fresh authenticated partner channel isolates the new lived body/control
    # history from older selection credit on this reusable causal closure.
    channel=0xFA77;context=probe.causal_dialogue_context(probe._longest_causal_program_component(),channel)
    read_credit=copy.deepcopy(probe.language_adult.credit.checkpoint())
    opening=resident_respond(probe,bytes(effect.surface),0xFA00,channel)
    _opening_surface,opening_programs=probe.compose_causal_component(effect.identity,channel=channel)
    checks['baseline_has_multi_link_world_certified_causal_discourse']=(
        len(rows)>=3 and first_program>0 and bool(opening) and len(opening_programs)>=3)
    checks['mere_relation_read_does_not_self_reward_or_overwrite_felt_history']=(
        probe.language_adult.credit.checkpoint()==read_credit)

    favorable=type(base.adult).restore(copy.deepcopy(base.adult.checkpoint()))
    adverse=type(base.adult).restore(copy.deepcopy(base.adult.checkpoint()))
    for subject in (favorable,adverse):
        subject_rows=subject.causal_chain_rows(effect.identity)
        for row in subject_rows:subject.causal_program_for_row(row)
    teach_appraisal_language(favorable.language_adult)
    teach_appraisal_language(adverse.language_adult)
    favorable_world=copy.deepcopy(favorable.language_adult.world_causal_learning.checkpoint())
    adverse_world=copy.deepcopy(adverse.language_adult.world_causal_learning.checkpoint())
    favorable_language=copy.deepcopy(favorable.language_adult.language.checkpoint())
    adverse_language=copy.deepcopy(adverse.language_adult.language.checkpoint())

    # Same current causal world, opposite body-consequence histories.  Outcome is held
    # at zero so the manipulation is somatic/control, not semantic reward.
    for _ in range(3):
        favorable.language_adult.experience_choice(
            first_program,0,Q//2,context=context,controllable=True,independent_return=True)
        adverse.language_adult.experience_choice(
            first_program,0,-Q//2,context=context,controllable=False,independent_return=True)
    # Matched no-action opportunity establishes that the favorable consequence is
    # action-contingent rather than merely co-occurring with the candidate.
    favorable.language_adult.experience_program_background(first_program,False,context)

    good=favorable.language_adult.somatic_appraisal(first_program,context)
    bad=adverse.language_adult.somatic_appraisal(first_program,context)
    favorable_reply,favorable_programs=favorable.compose_causal_component(effect.identity,channel=channel)
    adverse_reply,adverse_programs=adverse.compose_causal_component(effect.identity,channel=channel)
    favorable_state=bytes(favorable.language_adult.realize_somatic_appraisal(
        first_program,context) or b'')
    adverse_state=bytes(adverse.language_adult.realize_somatic_appraisal(
        first_program,context) or b'')

    checks['felt_state_is_multicoordinate_not_one_valence_register']=(
        good.somatic_q16>0 and bad.somatic_q16<0
        and good.outcome_q16==bad.outcome_q16==0
        and good.controllability_q16>=Q//2 and bad.controllability_q16<Q//2
        and good.pressure_q16==bad.pressure_q16==0
        and good.arousal_q16==bad.arousal_q16>0
        and good.valence_q16>0 and bad.valence_q16<0)
    checks['matched_world_truth_different_somatic_control_history_changes_discourse_extent']=(
        favorable_reply.startswith(opening.split(b'.',1)[0]+b'.')
        and adverse_reply.startswith(opening.split(b'.',1)[0]+b'.')
        and len(favorable_programs)>=3
        and 0<len(adverse_programs)<len(favorable_programs)
        and favorable_reply!=adverse_reply)
    checks['self_state_requires_independent_projection_not_causal_availability']=(
        favorable_state and adverse_state and favorable_state!=adverse_state
        and favorable_state not in favorable_reply and adverse_state not in adverse_reply
        and b'My current state feels ' not in favorable_reply
        and b'My current state feels ' not in adverse_reply)
    checks['learned_state_language_remains_available_after_causal_boundary']=(
        b'manageable' in favorable_state and b'I can influence what happens' in favorable_state
        and b'aversive' in adverse_state and b'I have little control over what happens' in adverse_state)
    checks['somatic_appraisal_never_rewrites_world_or_language_evidence']=(
        favorable.language_adult.world_causal_learning.checkpoint()==favorable_world
        and adverse.language_adult.world_causal_learning.checkpoint()==adverse_world
        and favorable.language_adult.language.checkpoint()==favorable_language
        and adverse.language_adult.language.checkpoint()==adverse_language)

    # Lesion only the transient appraisal coupling.  Learned somatic/control history is
    # still present, but discourse falls back to the intact world-certified extent.
    lesioned=type(base.adult).restore(copy.deepcopy(adverse.checkpoint()))
    retained=lesioned.language_adult.somatic_appraisal(first_program,context)
    lesioned.language_adult.somatic_appraisal=lambda *_args,**_kwargs: SomaticAppraisalV1()
    lesioned.language_adult.somatic_appraisal_atoms=lambda *_args,**_kwargs: ()
    lesioned.language_adult.realize_somatic_appraisal=lambda *_args,**_kwargs: None
    lesion_reply=resident_respond(lesioned,bytes(effect.surface),0xFA03,channel)
    _lesion_surface,lesion_programs=lesioned.compose_causal_component(effect.identity,channel=channel)
    checks['appraisal_coupling_lesion_collapses_history_dependent_expression_difference']=(
        retained.somatic_q16<0 and len(lesion_programs)>=3
        and b'My current state feels ' not in lesion_reply and lesion_reply!=adverse_reply)

    # The appraisal is derived from persistent causal state, not persisted as an emotion object.
    adverse_cp=copy.deepcopy(adverse.checkpoint())
    restored=type(base.adult).restore(copy.deepcopy(adverse_cp))
    restored_appraisal=restored.language_adult.somatic_appraisal(first_program,context)
    restored_reply=resident_respond(restored,bytes(effect.surface),0xFA04,channel)
    blob=json.dumps(adverse_cp,sort_keys=True).lower()
    checks['checkpoint_rematerializes_same_felt_appraisal_without_emotion_object']=(
        restored_appraisal==bad and restored_reply==adverse_reply
        and 'somaticappraisal' not in blob and 'emotion' not in blob and 'felt_state' not in blob)

    # Slow organism-wide pressure is an independent coordinate. Hold the exact load
    # fixed while lesioning only learned control: raw pressure must stay identical,
    # while candidate-local interference and public discourse diverge.
    loaded=type(base.adult).restore(copy.deepcopy(favorable.checkpoint()))
    uncontrolled=type(base.adult).restore(copy.deepcopy(favorable.checkpoint()))
    local=uncontrolled.language_adult.credit.row(first_program).contexts[context]
    local.control_attempts=0;local.control_successes=0
    local.background_attempts=0;local.background_successes=0
    local.control_history_q16=0;uncontrolled.language_adult._select_epoch+=1
    for index in range(6):
        for subject in (loaded,uncontrolled):
            subject.language_adult.settle_body_ingress(
                f'somatic-appraisal-load-{index}',index+1,'%064x'%(index+1),Q//2)
    loaded_felt=loaded.language_adult.somatic_appraisal(first_program,context)
    uncontrolled_felt=uncontrolled.language_adult.somatic_appraisal(first_program,context)
    loaded_reply,loaded_programs=loaded.compose_causal_component(effect.identity,channel=channel)
    uncontrolled_reply,uncontrolled_programs=uncontrolled.compose_causal_component(effect.identity,channel=channel)
    loaded_state=bytes(loaded.language_adult.realize_somatic_appraisal(first_program,context) or b'')
    uncontrolled_state=bytes(uncontrolled.language_adult.realize_somatic_appraisal(
        first_program,context) or b'')
    checks['matched_allostatic_load_is_buffered_by_learned_controllability']=(
        loaded_felt.pressure_q16==uncontrolled_felt.pressure_q16>=Q//2
        and loaded_felt.controllability_q16>=Q//2
        and uncontrolled_felt.controllability_q16<Q//2
        and loaded_felt.interference_q16<uncontrolled_felt.interference_q16
        and len(loaded_programs)>=3 and 0<len(uncontrolled_programs)<len(loaded_programs)
        and b'my body is under strain' in loaded_state
        and b'my body is under strain' in uncontrolled_state
        and b'My current state feels ' not in loaded_reply
        and b'My current state feels ' not in uncontrolled_reply)
    for _ in range(64):uncontrolled.language_adult.internal_tick()
    recovered_felt=uncontrolled.language_adult.somatic_appraisal(first_program,context)
    recovered_reply,recovered_programs=uncontrolled.compose_causal_component(effect.identity,channel=channel)
    recovered_state=bytes(uncontrolled.language_adult.realize_somatic_appraisal(
        first_program,context) or b'')
    checks['quiet_recovery_removes_load_interference_without_relearning_language']=(
        recovered_felt.pressure_q16==0 and recovered_felt.interference_q16==0
        and len(recovered_programs)==len(loaded_programs)
        and b'my body is settled' in recovered_state
        and b'My current state feels ' not in recovered_reply)

    checks['same_canonical_life_checkpoint_remains_one_subject']=(
        base.checkpoint()==base_cp and base.cursor==len(curriculum.events))
    failed=[name for name,value in checks.items() if not value]
    result={
        'schema':'cyber-lagoon.somatic-appraisal-causal-discourse.v1',
        'contract':'FOUNDRY_SOMATIC_APPRAISAL_CAUSAL_DISCOURSE_'+('GREEN' if not failed else 'RED'),
        'pass':not failed,'reference_only':True,'novel_synthesis':True,
        'visible_language_gain':'CAUSAL_EXPLANATION_NO_LONGER_FORCES_UNRELATED_SELF_DISCLOSURE',
        'claim_ceiling':'TRANSIENT_SOMATIC_APPRAISAL_NETWORK_OF_NETWORKS_REFERENCE_ONLY',
        'baseline':opening.decode(errors='replace'),
        'favorable':favorable_reply.decode(errors='replace'),
        'adverse':adverse_reply.decode(errors='replace'),
        'favorable_state':favorable_state.decode(errors='replace'),
        'adverse_state':adverse_state.decode(errors='replace'),
        'lesion':lesion_reply.decode(errors='replace'),
        'lesion_programs':len(lesion_programs),
        'loaded':loaded_reply.decode(errors='replace'),
        'uncontrolled_loaded':uncontrolled_reply.decode(errors='replace'),
        'recovered':recovered_reply.decode(errors='replace'),
        'loaded_state':loaded_state.decode(errors='replace'),
        'uncontrolled_loaded_state':uncontrolled_state.decode(errors='replace'),
        'recovered_state':recovered_state.decode(errors='replace'),
        'favorable_appraisal':good.__dict__,'adverse_appraisal':bad.__dict__,
        'loaded_appraisal':loaded_felt.__dict__,
        'uncontrolled_loaded_appraisal':uncontrolled_felt.__dict__,
        'recovered_appraisal':recovered_felt.__dict__,'checks':checks,'failed':failed,
        'remaining_red':['LEARNED_SOCIAL_DISCLOSURE_COMPETITION','DIRECT_PARITY',
                         'PHENOMENAL_FEELING_CLAIM','BROAD_EMOTIONAL_DISCOURSE'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if not failed else 1

if __name__=='__main__':raise SystemExit(main())
