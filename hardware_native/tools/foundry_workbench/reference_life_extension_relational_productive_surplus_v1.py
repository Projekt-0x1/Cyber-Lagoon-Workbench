#!/usr/bin/env python3
"""Canonical append after certified causal discourse becomes available.

The historical filename is repository archaeology. Later developmental contacts grow from
one continuing causal Adult checkpoint. The tail exposes recursive self/culture,
metacognition, source-qualified recommendations, causal testing, learn-to-learn experiment
structure, adaptive experiment-policy pressure, external-world cue recurrence/change,
body/control contrasts, and partner-local shared-history gaps using only ordinary canonical
life lanes. No observer semantic opcode names a hidden context, belief, truth, or policy.
"""
from __future__ import annotations
from dataclasses import dataclass
from reference_life_extension_causal_depth_plus_v1 import A_GROWTH,A_NEED,A_ROOTS,A_STOMATA
from reference_life_extension_endogenous_state_inquiry_v1 import TESTIMONY_ALARM
LIFE_AFTER=('reference_life_extension_endogenous_state_inquiry_v1',)
GROUP_CONTEXT=0xA135
@dataclass(frozen=True)
class ExtensionEventV1:
    sequence:int;lane:str;source:int=0;payload:tuple=()

def build_extension(start):
    rows=[]
    def add(lane,source=0,payload=()):rows.append(ExtensionEventV1(int(start)+len(rows)+1,lane,int(source),tuple(payload)))
    add('body_load',909020,(8,1<<12));add('scene',0xFA01,(100,A_ROOTS));add('checkpoint_mark',0,('relational_surplus_recovered',))
    root_effect_scenes=[]
    for offset,atom in enumerate((A_STOMATA,TESTIMONY_ALARM,A_GROWTH)):
        sequence=int(start)+len(rows)+1;add('scene',0xFDC0+offset,(100,atom));root_effect_scenes.append(sequence)
    for source in (0xFE01,0xFE02):
        add('relation',source,(GROUP_CONTEXT,*root_effect_scenes));add('discourse_surface',source,tuple(b'plant stomata close, the irrigation alarm sounds, and plant growth slows'))
    add('checkpoint_mark',0,('causal_variable_arity_coordination',));add('authenticated_utterance',0xFA11,tuple(b'plant roots lose water.'));add('authenticated_utterance',0xFA12,tuple(b'plant leaves wilt.'))
    for atom,source in ((A_GROWTH,0xFA31),(A_NEED,0xFA32)):
        scene=int(start)+len(rows)+1;add('scene',source,(100,atom));action=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',source,(scene,));add('causal_dialogue_return',source,(action,0,0,1));add('authenticated_utterance',source,tuple(b'What else happens because of that?'))
    add('authenticated_utterance',0xFA41,tuple(b'Please continue: that changed what happened'));add('authenticated_utterance',0xFA42,tuple(b'Please continue: that worked again'));add('checkpoint_mark',0,('receptive_boundary_wrapper',));add('authenticated_utterance',0xFA11,tuple(b'why is it the case that plant roots lose water?'));add('authenticated_utterance',0xFA12,tuple(b'why is it the case that plant leaves wilt?'));add('checkpoint_mark',0,('natural_causal_question_experience',))

    for atom,source in ((A_ROOTS,0xF510),(A_STOMATA,0xF511)):
        scene=int(start)+len(rows)+1;add('scene',source,(100,atom));action=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',source,(scene,));add('causal_dialogue_return',0xF520+(source&1),(action,1<<16,0,1))
    add('authenticated_utterance',0xF501,tuple(b'First inspect the roots, then check the stomata because the second tells us whether the response propagated.'));add('checkpoint_mark',0,('cultural_first_trial_instruction_exposure',))
    for source,text in ((0xF531,b'Check both because either observation alone can mislead us.'),(0xF532,b'Use both observations because the pair separates water loss from downstream response.')):add('authenticated_utterance',source,tuple(text))
    add('checkpoint_mark',0,('shared_reason_ecology',))

    revision_scene=int(start)+len(rows)+1;add('scene',0xF541,(100,A_NEED));revision_action=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',0xF541,(revision_scene,));add('quiet',0,(4,));add('causal_dialogue_return',0xF542,(revision_action,1<<16,0,1));add('checkpoint_mark',0,('post_decision_revision_window',))
    for offset in range(4):add('body_load',0xF550+offset,(70+offset,1<<15))
    add('authenticated_utterance',0xF501,tuple(b'First inspect the roots, then check the stomata because the second tells us whether the response propagated.'));add('checkpoint_mark',0,('cultural_instruction_loaded',));add('quiet',0,(48,));add('authenticated_utterance',0xF501,tuple(b'First inspect the roots, then check the stomata because the second tells us whether the response propagated.'));add('checkpoint_mark',0,('cultural_instruction_recovered',))
    add('authenticated_utterance',0xF561,tuple(b'First inspect the roots, then check the stomata because the second tells us whether the response propagated.'));add('source_withdrawal',0xF501);add('authenticated_utterance',0xF561,tuple(b'First inspect the roots, then check the stomata because the second tells us whether the response propagated.'));add('checkpoint_mark',0,('cultural_retransmission_after_teacher_withdrawal',))

    add('authenticated_utterance',0xF571,tuple(b'Act now because the roots already show the failure.'));add('authenticated_utterance',0xF572,tuple(b'Wait and inspect again because one observation can be misleading.'))
    meta_scene=int(start)+len(rows)+1;add('scene',0xF573,(100,A_ROOTS,A_NEED));meta_action=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',0xF573,(meta_scene,));add('quiet',0,(6,));add('causal_dialogue_return',0xF574,(meta_action,1<<16,0,1));add('checkpoint_mark',0,('competing_reason_postdecision_evidence_window',))
    for offset in range(3):add('body_load',0xF580+offset,(90+offset,1<<15))
    add('authenticated_utterance',0xF571,tuple(b'Act now because the roots already show the failure.'));add('authenticated_utterance',0xF572,tuple(b'Wait and inspect again because one observation can be misleading.'));add('checkpoint_mark',0,('metacontrol_same_reasons_high_load',));add('quiet',0,(64,));add('authenticated_utterance',0xF571,tuple(b'Act now because the roots already show the failure.'));add('authenticated_utterance',0xF572,tuple(b'Wait and inspect again because one observation can be misleading.'));add('checkpoint_mark',0,('metacontrol_same_reasons_recovered',))

    # These utterances support action recommendations. Their later action returns can teach
    # recommendation-outcome history, never proposition or explanation truth.
    add('authenticated_utterance',0xF591,tuple(b'Inspect the roots first; that usually exposes the failure early.'))
    recommendation_scene_a=int(start)+len(rows)+1;add('scene',0xF593,(100,A_ROOTS,A_NEED));recommendation_action_a=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',0xF593,(recommendation_scene_a,));add('causal_dialogue_return',0xF594,(recommendation_action_a,1<<16,0,1));add('checkpoint_mark',0,('recommendation_source_a_lived_action_return',))
    add('authenticated_utterance',0xF592,tuple(b'Ignore the roots and inspect stomata first; roots are usually noise here.'))
    recommendation_scene_b=int(start)+len(rows)+1;add('scene',0xF595,(100,A_STOMATA,A_NEED));recommendation_action_b=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',0xF595,(recommendation_scene_b,));add('causal_dialogue_return',0xF596,(recommendation_action_b,0,0,1));add('checkpoint_mark',0,('recommendation_source_b_lived_action_return',));add('authenticated_utterance',0xF591,tuple(b'Inspect the roots first; that usually exposes the failure early.'));add('authenticated_utterance',0xF592,tuple(b'Ignore the roots and inspect stomata first; roots are usually noise here.'));add('checkpoint_mark',0,('recommendation_sources_recur_after_lived_returns',))

    for offset in range(2):
        scene=int(start)+len(rows)+1;add('scene',0xF5A0+offset,(100,A_ROOTS,A_NEED));action=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',0xF5A0+offset,(scene,));add('causal_dialogue_return',0xF5A8+offset,(action,1<<16,0,1))
    add('checkpoint_mark',0,('repeated_action_without_live_recommendation',));add('authenticated_utterance',0xF5B1,tuple(b'Inspect the roots now; if the failure is active this observation should change.'))
    scene=int(start)+len(rows)+1;add('scene',0xF5B2,(100,A_ROOTS,A_NEED));action=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',0xF5B2,(scene,));add('causal_dialogue_return',0xF5B3,(action,1<<16,0,1));add('checkpoint_mark',0,('similar_action_with_live_recommendation',))

    add('authenticated_utterance',0xF5C1,tuple(b'Roots should reveal the failure if it is active.'))
    for offset in range(3):
        scene=int(start)+len(rows)+1;add('scene',0xF5C2+offset,(100,A_ROOTS,A_NEED));action=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',0xF5C2+offset,(scene,));add('causal_dialogue_return',0xF5C8+offset,(action,1<<16,0,1))
    add('authenticated_utterance',0xF5D1,tuple(b'Stomata should change too if the response propagated.'));add('authenticated_utterance',0xF5C1,tuple(b'Roots should reveal the failure if it is active.'))
    fresh_scene=int(start)+len(rows)+1;add('scene',0xF5D2,(100,A_STOMATA,A_NEED));fresh_action=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',0xF5D2,(fresh_scene,));add('causal_dialogue_return',0xF5D3,(fresh_action,1<<16,0,1));add('checkpoint_mark',0,('repeated_recommendation_history_competes_with_fresh_recommendation',))
    for offset in range(3):add('body_load',0xF5E0+offset,(110+offset,1<<15))
    add('authenticated_utterance',0xF5C1,tuple(b'Roots should reveal the failure if it is active.'));add('authenticated_utterance',0xF5D1,tuple(b'Stomata should change too if the response propagated.'));add('checkpoint_mark',0,('same_experiment_choice_ecology_under_pressure',));add('quiet',0,(72,));add('authenticated_utterance',0xF5C1,tuple(b'Roots should reveal the failure if it is active.'));add('authenticated_utterance',0xF5D1,tuple(b'Stomata should change too if the response propagated.'));add('checkpoint_mark',0,('same_experiment_choice_ecology_after_recovery',))

    # Stable external cue history, a cue-shifted contradictory return, then recurrence of the
    # prior cues. Nothing tells the Adult whether to merge/split any memory.
    for offset in range(3):
        add('authenticated_utterance',0xF5F1,tuple(b'When the roots are pale and the soil is dry, inspecting roots should expose the problem.'))
        scene=int(start)+len(rows)+1;add('scene',0xF5F2+offset,(100,A_ROOTS,A_NEED,A_GROWTH));action=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',0xF5F2+offset,(scene,));add('causal_dialogue_return',0xF5F8+offset,(action,1<<16,0,1))
    add('checkpoint_mark',0,('recurring_world_cues_with_stable_action_returns',))

    add('authenticated_utterance',0xF601,tuple(b'When the alarm is active and the leaves are closed, the same roots inspection should still expose the problem.'))
    shifted_scene=int(start)+len(rows)+1;add('scene',0xF602,(100,A_ROOTS,TESTIMONY_ALARM,A_STOMATA));shifted_action=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',0xF602,(shifted_scene,));add('causal_dialogue_return',0xF603,(shifted_action,0,0,1));add('checkpoint_mark',0,('world_cue_shift_with_different_action_return',))

    add('authenticated_utterance',0xF5F1,tuple(b'When the roots are pale and the soil is dry, inspecting roots should expose the problem.'))
    return_scene=int(start)+len(rows)+1;add('scene',0xF604,(100,A_ROOTS,A_NEED,A_GROWTH));return_action=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',0xF604,(return_scene,));add('causal_dialogue_return',0xF605,(return_action,1<<16,0,1));add('checkpoint_mark',0,('return_to_prior_world_cues',))

    # Same external cultural content under body/control pressure and after recovery. Any
    # control-memory change must not rewrite external recommendation/world evidence.
    for offset in range(3):add('body_load',0xF610+offset,(130+offset,1<<15))
    add('authenticated_utterance',0xF5F1,tuple(b'When the roots are pale and the soil is dry, inspecting roots should expose the problem.'));add('checkpoint_mark',0,('same_external_content_under_body_control_shift',));add('quiet',0,(80,));add('authenticated_utterance',0xF5F1,tuple(b'When the roots are pale and the soil is dry, inspecting roots should expose the problem.'));add('checkpoint_mark',0,('same_external_content_after_body_recovery',))

    # Partner-local shared-history ecology. P participates in an earlier public episode. A
    # later world update is carried by other sources before P returns; only a later public
    # P episode re-establishes current shared history. No belief/perspective label is supplied.
    p_scene=int(start)+len(rows)+1;add('scene',0xF701,(100,A_ROOTS,A_NEED,A_GROWTH));p_action=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',0xF701,(p_scene,));add('causal_dialogue_return',0xF702,(p_action,1<<16,0,1));add('authenticated_utterance',0xF701,tuple(b'I can see the pale roots and dry soil too.'));add('checkpoint_mark',0,('partner_p_prior_public_episode',))
    update_scene=int(start)+len(rows)+1;add('scene',0xF711,(100,A_STOMATA,TESTIMONY_ALARM,A_NEED));update_action=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',0xF711,(update_scene,));add('causal_dialogue_return',0xF712,(update_action,1<<16,0,1));add('authenticated_utterance',0xF713,tuple(b'The alarm is active now and the stomata have changed.'));add('checkpoint_mark',0,('later_public_update_from_other_sources',))
    add('authenticated_utterance',0xF701,tuple(b'The roots were pale and the soil was dry.'));add('authenticated_utterance',0xF713,tuple(b'The alarm is active now and the stomata have changed.'));add('checkpoint_mark',0,('partner_p_returns_after_other_source_update',))
    p_new_scene=int(start)+len(rows)+1;add('scene',0xF701,(100,A_STOMATA,TESTIMONY_ALARM,A_NEED));p_new_action=int(start)+len(rows)+1;add('partner_causal_dialogue_opportunity',0xF701,(p_new_scene,));add('causal_dialogue_return',0xF714,(p_new_action,1<<16,0,1));add('checkpoint_mark',0,('partner_p_updated_public_episode',))
    return tuple(rows)