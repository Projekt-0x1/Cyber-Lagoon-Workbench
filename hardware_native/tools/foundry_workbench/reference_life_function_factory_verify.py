#!/usr/bin/env python3
"""Primary Workbench factory: one canonical birth/life pass, all probes from its marks."""
from __future__ import annotations
import copy,hashlib,json,tempfile,time
from pathlib import Path
from life_function_factory_v1 import build_cache,load_mark
from life_function_emergence_ledger_v1 import ledger as build_emergence_ledger
from reference_single_growing_curriculum_policy_verify import verify as verify_runner_policy
from reference_canonical_life_language_guided_world_learning_verify import TARGET_INSTRUCTION
from reference_language_mastery_terminal_v1 import emit_choice
from reference_language_mastery_contact_adapter_v1 import CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1
from reference_language_mastery_claude_gateway_v1 import resident_contact
from reference_relational_productive_surplus_verify import verify_loaded as verify_certified_causal_discourse
from reference_natural_causal_question_retrieval_verify import verify_runtime as verify_natural_causal_question
from reference_partner_adverse_uptake_reopening_verify import verify_loaded as verify_partner_adverse_uptake
from reference_variable_arity_causal_message_planning_verify import GROUP_SOURCES,verify_loaded as verify_variable_arity_message_planning
from reference_life_function_curriculum_v1 import *
from reference_life_extension_causal_depth_plus_v1 import A_CONSERVE,A_GROWTH,A_HARDEN,A_HUMIDITY,A_NEED,A_ROOTS
from reference_life_extension_history_matrix_v1 import A as HISTORY_A,B as HISTORY_B,C as HISTORY_C
from reference_life_extension_open_state_prompt_v1 import PROMPT as OPEN_STATE_PROMPT
from reference_life_extension_endogenous_state_inquiry_v1 import (
    CAUSAL_FANIN_SOURCES,TESTIMONY_ALARM,TESTIMONY_MOTOR_SOURCE,TESTIMONY_PARTNER,
)

NEW=501
_probe_context,_probe_examples,_heldout,_probe_features=canonical_developmental_probe_v2(canonical_life_function_curriculum_v2())
_first_example=_probe_examples[0];_new_heldout=(*_heldout[:3],NEW)

def _checkpoint_leaf_count(value):
    if isinstance(value,dict):return sum(_checkpoint_leaf_count(row) for row in value.values())
    if isinstance(value,(list,tuple)):return sum(_checkpoint_leaf_count(row) for row in value)
    return 1

def resident_respond(adult,raw,source,channel=0):
    raw=bytes(raw);hits=list(adult.relation_surface_matches(raw))
    if len(hits)>=2 and len({identity for _start,_end,identity in hits})==len(hits):
        hits.sort();cursor=0;exact=True
        for start,end,_identity in hits:
            if start<cursor or raw[cursor:start].strip():exact=False;break
            cursor=end
        if exact and not raw[cursor:].strip():
            surface,_receipt=adult.externalize_relation_frontier(tuple(identity for _start,_end,identity in hits));return bytes(surface or b'')
    contact=LanguageMasteryContactAdapterV1(adult.language_adult);identity=int(contact.contact(CONTACT_UTTERANCE,tuple(raw),int(source),max(0,int(channel))))
    scene=contact.scenes.get(identity)
    if scene is None:return b''
    try:leaf=adult.language_adult.leaf(scene.context,scene.atoms)
    except RuntimeError:return b''
    contextual=adult.respond_context_affordance(scene.context,scene.atoms,channel)
    if contextual:return bytes(contextual)
    return adult.compose_causal_component(leaf.identity,channel=channel)[0]

def _causal_chain_response(adult,leaf_identity,with_trace=False,with_expression_trace=False):
    return adult.compose_causal_component(leaf_identity,with_expression_trace)

def refused(adult,atoms):
    try:adult.leaf(_probe_context,atoms);return False
    except RuntimeError:return True

def current_world_state(runtime,effect_atom=0xA104):
    adult=runtime.adult.language_adult;learner=adult.world_causal_learning;effect=adult.leaf(100,(int(effect_atom),))
    rows=tuple(row for row in learner.current_resolutions() if int(row[3])==effect.identity)
    if len(rows)!=1:return adult,learner,0,effect,0,0,0,b''
    row=rows[0];cause=int(row[2]);receipt=int(row[4]);factor=int(learner.preferred_factor(adult));program=adult.programs.ident((effect.identity,cause),factor) if factor else 0
    if adult.programs.factor(program)!=factor:
        chunk=learner.materialize_program(adult,receipt,factor);program=0 if chunk is None else int(chunk.identity)
    return adult,learner,cause,effect,receipt,factor,program,bytes(adult._leaf_surface(cause))

def main():
    started=time.perf_counter();checks={};species=canonical_species_program_v2();curriculum=canonical_life_function_curriculum_v2();marks=tuple(e.payload[0] for e in curriculum.events if e.lane=='checkpoint_mark')
    with tempfile.TemporaryDirectory(prefix='foundry-life-factory-') as directory:
        manifest=build_cache(directory)
        # No training below this line: every probe restores derivative checkpoints.
        loaded={mark:load_mark(directory,mark) for mark in marks}
        final=loaded[marks[-1]];final_cp=copy.deepcopy(final.checkpoint())
        variable_arity=verify_variable_arity_message_planning(loaded,curriculum)
        checks.update({f'variable_arity_{name}':passed
                       for name,passed in variable_arity['checks'].items()})
        checks['variable_arity_message_planning_is_in_one_life_factory']=bool(variable_arity['pass'])
        partner_adverse=verify_partner_adverse_uptake(loaded['relational_surplus_recovered'].adult)
        checks.update({f'partner_adverse_{name}':passed for name,passed in partner_adverse['checks'].items()})
        checks['partner_adverse_uptake_reopening_is_in_one_life_factory']=bool(partner_adverse['pass'])
        emergence=build_emergence_ledger(tuple((mark,loaded[mark]) for mark in marks))
        # Evaluation imports must not participate in global Life-extension
        # registration before birth or emergence measurement has completed.
        from reference_settled_intention_causal_continuation_verify import verify_loaded as verify_cross_action_continuity
        cross_action=verify_cross_action_continuity(final)
        checks.update({f'cross_action_{name}':passed
                       for name,passed in cross_action['checks'].items()})
        checks['cross_action_structural_continuity_is_in_one_life_factory']=bool(cross_action['pass'])
        question_mark='natural_causal_question_experience';question_index=marks.index(question_mark)
        natural_question_checks,natural_question=verify_natural_causal_question(
            loaded[question_mark],loaded[marks[question_index-1]])
        checks.update(natural_question_checks)
        # Top-level language/math runners must all delegate here; the visible gain
        # below is therefore measured once on this same continuing Adult.
        runner_checks,_runner_rows=verify_runner_policy();checks.update(runner_checks)
        checks['emergence_ledger_covers_every_current_mark']=(tuple(row['mark'] for row in emergence['marks'])==marks and emergence['marks'][-1]['cursor']==len(curriculum.events))
        checks['exactly_one_canonical_training_pass_materializes_all_marks']=(manifest['events']==len(curriculum.events) and tuple(row['mark'] for row in manifest['checkpoints'])==marks and final.cursor==len(curriculum.events))
        checks['all_probes_share_species_curriculum_and_source_semantics_roots']=(manifest['species_root']==species.root() and manifest['curriculum_root']==curriculum.root() and manifest['source_semantics_root']==source_semantics_root_v2())
        checks['canonical_species_has_no_scalar_constructor_law']='generic_constructor_metaplasticity' not in {law.law for law in species.laws}
        blob=json.dumps(final_cp,sort_keys=True).lower();checks['adult_checkpoint_has_no_scalar_constructor_or_fixture_state']=all(token not in blob for token in ('constructor_metaplasticity','pending_constructor','constructor_horizon','fixture'))
        unknown_contact=ReferenceLifeFunctionRuntimeV2.restore(
            species,copy.deepcopy(final_cp))
        unknown_surface,unknown_action=unknown_contact.contact_utterance(
            b'\x00\xff\x01',0xF00D,0xF00D)
        checks['unknown_contact_refuses_without_unowned_public_action_router']=(
            not unknown_surface and unknown_action==0)

        grounded=loaded['grounded'].adult.language_adult;productive=loaded['productive'].adult.language_adult;discourse=loaded['discourse'].adult.language_adult
        construction_cursors=[];construction_sources=set()
        grounding_scenes=tuple(event for event in curriculum.events[:curriculum.mark_cursor('grounded')]
                               if event.lane=='scene' and int(event.payload[0])==100)
        checks['canonical_lexicon_is_cross_situational_not_single_item_word_list']=(
            len(grounding_scenes)>=8 and all(len(event.payload)>2 for event in grounding_scenes))
        for event in curriculum.events[curriculum.mark_cursor('grounded'):curriculum.mark_cursor('productive')]:
            if event.lane=='scene' and int(event.payload[0])==_probe_context and len(event.payload)==5:construction_sources.add(int(event.source))
            elif event.lane=='surface' and int(event.source) in construction_sources:construction_cursors.append(int(event.sequence))
        if len(construction_cursors)<2:raise RuntimeError('life-factory:developmental-contact-incidence')
        c1=ReferenceLifeFunctionRuntimeV2(species).run(curriculum.prefix(construction_cursors[0])).adult.language_adult
        c2=ReferenceLifeFunctionRuntimeV2(species).run(curriculum.prefix(construction_cursors[1])).adult.language_adult
        learned_heldout=b' '.join(bytes(c2.language.lexeme(feature) or ()) for feature in _heldout)+b'.'
        checks['developmental_language_ratchet_is_same_lineage']=(grounded.language.lexeme(_heldout[0]) is not None and refused(grounded,_heldout) and refused(c1,_heldout) and bytes(c2.leaf(_probe_context,_heldout).surface)==learned_heldout and bytes(productive.leaf(_probe_context,_heldout).surface)==learned_heldout)
        checks['developmental_acquisition_is_contact_incidence_not_numbered_checkpoint_ladder']=(len(construction_cursors)==4 and not any(str(mark).startswith('construction_') for mark in marks))
        first_scene=next(event for event in curriculum.events[curriculum.mark_cursor('grounded'):curriculum.mark_cursor('productive')] if event.lane=='scene' and int(event.payload[0])==_probe_context)
        checks['raw_surface_positions_not_observer_tuple_order_own_ports']=(tuple(map(int,first_scene.payload[1:]))!=tuple(_probe_examples[0]))
        checks['settled_lexical_math_discards_rematerializable_raw_scene_history']=(
            not productive.language.checkpoint().get('scene_streams'))
        pair=discourse.compose(JOIN,discourse.leaf(_probe_context,_heldout),discourse.leaf(_probe_context,_first_example));checks['learned_discourse_relation_composes_heldout_propositions']=pair.depth==1 and len(pair.surface)>len(discourse.leaf(_probe_context,_heldout).surface)

        # Historical marks prove acquisition order; the latest final Adult owns the capability bar.
        op_mark=loaded['operator_ready'].adult;mark_winner=op_mark.operator_run_until_settled();mark_surface,_=op_mark.externalize_operator()
        op=type(final.adult).restore(copy.deepcopy(final.adult.checkpoint()));winner=op.operator_run_until_settled();operator_surface,_=op.externalize_operator()
        checks['final_adult_inherits_or_exceeds_earlier_operator_frontier']=(mark_winner is not None and winner is not None and winner.operations>=mark_winner.operations and operator_surface is not None and mark_surface is not None and operator_surface.count(b'.')>=mark_surface.count(b'.') and len(operator_surface)>=len(mark_surface)>250)

        cold=loaded['relation_primitives'].adult;hot=loaded['self_derived_basis'].adult;hot_best=hot.relation_basis.best_derived();hot_spaces=hot.relation_basis.expand_spaces(hot_best)
        if hot_spaces and len(hot_spaces)>=2:
            start,goal=hot_spaces[0],hot_spaces[-1];raw=cold.relation_basis.resolve(start,goal);fast=hot.relation_basis.resolve(start,goal)
        else:raw=fast=None
        historical_scrambled=(tuple(hot_spaces[i] for i in (4,0,8,2,6,1,7,3,5)) if hot_spaces and len(hot_spaces)>=9 else ())
        final_rel=type(final.adult).restore(copy.deepcopy(final.adult.checkpoint()));best=final_rel.relation_basis.best_derived();spaces=final_rel.relation_basis.expand_spaces(best)
        if spaces and len(spaces)>=9:
            scrambled=tuple(spaces[i] for i in (4,0,8,2,6,1,7,3,5))+tuple(spaces[9:]);organized=final_rel.organize_relation_frontier(scrambled)
        else:organized=None;scrambled=()
        checks['final_adult_inherits_or_exceeds_self_derived_relation_frontier']=(hot_best>0 and best>0 and final_rel.relation_basis.generation(best)>=hot.relation_basis.generation(hot_best)>=7 and spaces is not None and hot_spaces is not None and len(spaces)>=len(hot_spaces)>=9)
        checks['retained_abstraction_reduces_exact_relation_work']=(raw is not None and fast is not None and raw.boundary_q16==fast.boundary_q16 and fast.relation_touches<raw.relation_touches)
        checks['final_self_derived_math_organizes_unseen_scrambled_language_frontier']=(organized is not None and bytes(organized).count(b'.')==len(spaces) and tuple(scrambled)!=tuple(spaces))
        public=loaded['self_derived_public'].adult;checks['canonical_scrambled_frontier_was_publicly_reafferent']=(public.relation_public_count>=1 and len(public.pending_relation_actions)>=1 and final_rel.relation_public_count>=public.relation_public_count)
        top_cut=type(final_rel).restore(copy.deepcopy(final_rel.checkpoint()));top_cut.relation_basis.remove_derived(best);lower=top_cut.organize_relation_frontier(scrambled);checks['abstraction_lesion_deoptimizes_but_preserves_exact_language_order']=(lower is not None and bytes(lower)==bytes(organized))
        premise_cut=type(final_rel).restore(copy.deepcopy(final_rel.checkpoint()));primitive=next(row for row in premise_cut.relation_basis.relations.values() if row.direct_evidence and row.left_space==spaces[0] and row.right_space==spaces[1]);primitive_sources=tuple(primitive.direct_evidence)
        for evidence_source in primitive_sources:premise_cut.relation_basis.withdraw_evidence(evidence_source)
        checks['primitive_evidence_lesion_destroys_dependent_language_organization']=bool(primitive_sources) and premise_cut.organize_relation_frontier(scrambled) is None

        target_runtime=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(final_cp));target=current_world_state(target_runtime);world_visible=emit_choice(target[0],target[6]);checks['reliable_language_guides_world_experiment_but_world_owns_truth']=(target[4]>0 and target[1].complete_source_blocks(target[4])==3 and not target[1].current_testimony_resolutions())
        checks['world_earned_language_recomposes_never_taught_form']=(bool(world_visible) and world_visible!=TARGET_INSTRUCTION and bytes(target[3].surface) in world_visible and target[7] in world_visible)
        # A later current-world topology is acquired after every earlier language
        # mechanism. The evaluator derives its question from the learned effect;
        # only independent intervention consequences may license the answer.
        pre_topic=loaded['pre_unseen_causal_topic'];learned_topic=loaded['unseen_causal_topic']
        topic_effect=learned_topic.adult.language_adult.leaf(100,(0xA105,))
        topic_question=(b'why is it the case that '+bytes(topic_effect.surface).rstrip(b'.?').lower()+b'?')
        topic_cause=bytes(learned_topic.adult.language_adult.leaf(100,(0xA103,)).surface)
        topic_pre=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(pre_topic.checkpoint()))
        topic_pre_reply,_=topic_pre.contact_utterance(topic_question,0xF330,0xF330)
        topic_live=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(learned_topic.checkpoint()))
        topic_reply,topic_action=topic_live.contact_utterance(topic_question,0xF331,0xF331)
        topic_receipt=topic_live.adult.pending_causal_dialogue_actions.get(topic_action)
        topic_certificate=(() if topic_receipt is None else
                           topic_live.adult._causal_action_coordinates(topic_receipt))
        topic_restart=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(learned_topic.checkpoint()))
        topic_restart_reply,_=topic_restart.contact_utterance(topic_question,0xF331,0xF331)
        yoked_topic=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(pre_topic.checkpoint()))
        topic_tail=curriculum.events[pre_topic.cursor:curriculum.mark_cursor('unseen_causal_topic')]
        drying_sequence=next(event.sequence for event in topic_tail
                             if event.lane=='scene' and event.source==0xE102)
        yoked_step=0
        for event in topic_tail:
            if event.lane=='resident_world_step':
                yoked_step+=1
                # Equal independent contact quantity, but the actual cause changes
                # across trials. No stable controllable topology may settle.
                if yoked_step%2==0:
                    event=LifeCurriculumEventV2(
                        event.sequence,event.lane,event.source,
                        (event.payload[0],drying_sequence,event.payload[2]))
            yoked_topic.apply(event)
        yoked_reply,_=yoked_topic.contact_utterance(topic_question,0xF332,0xF332)
        withdrawn_topic=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(learned_topic.checkpoint()))
        for evidence_source in (0xE111,0xE112,0xE113):
            withdrawn_topic.adult.language_adult.world_causal_learning.withdraw_source(evidence_source)
        withdrawn_reply,_=withdrawn_topic.contact_utterance(topic_question,0xF333,0xF333)
        checks['later_world_consequence_acquires_unseen_causal_topic_in_same_adult']=(
            not topic_pre_reply and bool(topic_reply) and topic_cause in topic_reply
            and bytes(topic_effect.surface) in topic_reply
            and any(int(row[2])==int(topic_effect.identity) for row in topic_certificate))
        checks['unseen_topic_requires_independent_consequence_and_live_sources']=(
            not yoked_reply and not withdrawn_reply)
        checks['unseen_topic_survives_checkpoint_without_question_or_answer']=(
            topic_restart_reply==topic_reply
            and topic_question.decode() not in json.dumps(learned_topic.checkpoint(),sort_keys=True)
            and topic_reply.decode() not in json.dumps(learned_topic.checkpoint(),sort_keys=True))

        # Later consequences to resident lexical actions must revise public realization without
        # severing the already learned nonlinguistic causal coordinates.
        pre_lexical_causal=loaded['pre_lexical_causal_integration']
        pending_lexical_causal=loaded['lexical_causal_hypotheses']
        integrated_lexical_causal=loaded['lexical_causal_integration']
        pre_lc_adult=pre_lexical_causal.adult.language_adult
        integrated_lc_adult=integrated_lexical_causal.adult.language_adult
        pre_lc_cause=pre_lc_adult.leaf(100,(0xA103,));pre_lc_effect=pre_lc_adult.leaf(100,(0xA105,))
        integrated_lc_cause=integrated_lc_adult.leaf(100,(0xA103,));integrated_lc_effect=integrated_lc_adult.leaf(100,(0xA105,))
        pre_lc_question=b'why is it the case that '+bytes(pre_lc_effect.surface).lower()+b'?'
        integrated_lc_question=b'why is it the case that '+bytes(integrated_lc_effect.surface).lower()+b'?'
        pre_lc_runtime=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(pre_lexical_causal.checkpoint()))
        pre_lc_reply,_=pre_lc_runtime.contact_utterance(pre_lc_question,0xF341,0xF341)
        pending_lc_runtime=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(pending_lexical_causal.checkpoint()))
        pending_lc_reply,_=pending_lc_runtime.contact_utterance(pre_lc_question,0xF342,0xF342)
        integrated_lc_runtime=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(integrated_lexical_causal.checkpoint()))
        integrated_lc_reply,integrated_lc_action=integrated_lc_runtime.contact_utterance(
            integrated_lc_question,0xF343,0xF343)
        integrated_lc_receipt=integrated_lc_runtime.adult.pending_causal_dialogue_actions.get(integrated_lc_action)
        integrated_lc_certificate=(() if integrated_lc_receipt is None else
                                   integrated_lc_runtime.adult._causal_action_coordinates(integrated_lc_receipt))
        restarted_lc=ReferenceLifeFunctionRuntimeV2.restore(
            species,copy.deepcopy(integrated_lexical_causal.checkpoint()))
        restarted_lc_reply,_=restarted_lc.contact_utterance(integrated_lc_question,0xF343,0xF343)

        withdrawn_lc=ReferenceLifeFunctionRuntimeV2.restore(
            species,copy.deepcopy(integrated_lexical_causal.checkpoint()))
        withdrawn_lc.adult.language_adult.language.withdraw_source(0xE221)
        withdrawn_lc.adult.language_adult.language.withdraw_source(0xE222)
        withdrawn_lc_reply,_=withdrawn_lc.contact_utterance(pre_lc_question,0xF344,0xF344)
        counter_lc=ReferenceLifeFunctionRuntimeV2.restore(
            species,copy.deepcopy(pending_lexical_causal.checkpoint()))
        counter_actions=tuple(sorted(
            counter_lc.adult.pending_endogenous_inquiry_actions))
        for offset,identity in enumerate(counter_actions):
            counter_lc.settle_contact_consequence(
                identity,0xE251+offset,-1,0,True)
        counter_lc_reply,_=counter_lc.contact_utterance(pre_lc_question,0xF345,0xF345)

        checks['lexical_revision_preserves_world_coordinates_while_revising_both_surfaces']=(
            pre_lc_cause.identity==integrated_lc_cause.identity
            and pre_lc_effect.identity==integrated_lc_effect.identity
            and bytes(pre_lc_cause.surface)!=bytes(integrated_lc_cause.surface)
            and bytes(pre_lc_effect.surface)!=bytes(integrated_lc_effect.surface))
        checks['resident_action_consequences_recompose_existing_causal_discourse']=(
            bool(pre_lc_reply) and pending_lc_reply==pre_lc_reply and bool(integrated_lc_reply)
            and bytes(integrated_lc_cause.surface) in integrated_lc_reply
            and bytes(integrated_lc_effect.surface) in integrated_lc_reply
            and integrated_lc_reply not in (
                b'morning sunlight warms the greenhouse aka sunbeams heat the glasshouse',
                b'steady wind closes the vent aka airflow seals the vent')
            and any(int(row[2])==int(integrated_lc_effect.identity)
                    for row in integrated_lc_certificate))
        checks['lexical_causal_revision_varies_with_source_and_counterhistory']=(
            withdrawn_lc_reply==pre_lc_reply and counter_lc_reply==pre_lc_reply)
        checks['lexical_causal_revision_survives_restart_without_transcript']= (
            restarted_lc_reply==integrated_lc_reply
            and integrated_lc_question.decode() not in json.dumps(
                integrated_lexical_causal.checkpoint(),sort_keys=True)
            and integrated_lc_reply.decode() not in json.dumps(
                integrated_lexical_causal.checkpoint(),sort_keys=True))
        pre_lexical=loaded['language_guided_world_learning'].adult;revised_lexical=loaded['lexical_revision'].adult
        pre_surface=bytes(pre_lexical.language_adult.leaf(_probe_context,_first_example).surface);revised_surface=bytes(revised_lexical.language_adult.leaf(_probe_context,_first_example).surface)
        revised_order=revised_lexical.organize_relation_frontier(historical_scrambled)
        pre_subject=bytes(pre_lexical.language_adult.language.lexeme(_heldout[0]) or ());revised_subject=bytes(revised_lexical.language_adult.language.lexeme(_heldout[0]) or ())
        checks['later_lexical_revision_changes_language_without_retraining_relation_basis']=(pre_subject and revised_subject and pre_subject!=revised_subject and pre_surface.startswith(pre_subject+b' ') and revised_surface.startswith(revised_subject+b' ') and pre_lexical.relation_basis.checkpoint()==revised_lexical.relation_basis.checkpoint() and pre_lexical.language.checkpoint()!=revised_lexical.language.checkpoint())
        checks['old_self_derived_relation_cognition_immediately_uses_new_lexicon']=(revised_order is not None and revised_subject in bytes(revised_order) and pre_subject not in bytes(revised_order))

        # Correction is a causal-history contrast in this same life, not an
        # evaluator instruction. Testimony may guide sampling but public revision
        # waits for independently settled replacement evidence.
        before_repair=loaded['world_lexical_revision'];heard_only=loaded['correction_contact'];repaired=loaded['correction_repair']
        # Hold later-acquired expression grounding fixed while intervening on causal
        # history, so early inability to speak is not confused with correction.
        expression_owner=loaded['causal_discourse_form_diversity'].adult
        def correction_projection(runtime):
            subject=type(expression_owner).restore(copy.deepcopy(expression_owner.checkpoint()))
            grounding=copy.deepcopy(subject.language_adult.world_causal_learning.checkpoint()['grounding'])
            causal_state=copy.deepcopy(runtime.adult.language_adult.world_causal_learning.checkpoint());causal_state['grounding']=grounding
            subject.language_adult.world_causal_learning=type(subject.language_adult.world_causal_learning).restore(causal_state)
            return subject
        before_adult=correction_projection(before_repair);heard_adult=correction_projection(heard_only);repaired_adult=correction_projection(repaired)
        correction_query=b'warm air dries the soil';correction_contact=b'warm air dries the soil since the heater warms the greenhouse'
        before_reply=resident_respond(before_adult,correction_query,0xF280)
        heard_reply=resident_respond(heard_adult,correction_query,0xF281)
        repaired_reply=resident_respond(repaired_adult,correction_query,0xF282)
        obsolete_reply=resident_respond(repaired_adult,b'morning sunlight warms the greenhouse',0xF283)
        checks['correction_contact_waits_for_consequence_before_public_revision']=(bool(before_reply) and not heard_reply)
        checks['independent_consequence_revises_visible_explanation_without_sentence_replay']=(bool(repaired_reply) and repaired_reply!=before_reply and repaired_reply!=correction_contact and b'warm air dries the soil' in repaired_reply and not obsolete_reply)
        heard_learner=heard_only.adult.language_adult.world_causal_learning;repaired_learner=repaired.adult.language_adult.world_causal_learning
        checks['spoken_alternative_never_becomes_truth_without_world_consequence']=(bool(heard_learner.current_testimony_resolutions()) and len(heard_learner.current_resolutions())<len(repaired_learner.current_resolutions())==2)
        repaired_restart=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(repaired.checkpoint()))
        checks['corrected_individual_restarts_with_same_changed_explanation']=(repaired_restart.checkpoint()==repaired.checkpoint() and repaired_reply and b'the heater warms the greenhouse' in repaired_reply)
        testimony_cut=correction_projection(repaired);tl=testimony_cut.language_adult.world_causal_learning;tl.withdraw_source(tl.testimony_source(53506))
        evidence_cut=correction_projection(repaired);evidence_cut.language_adult.world_causal_learning.withdraw_source(0xC311)
        checks['settled_world_relation_survives_testimony_withdrawal_but_not_evidence_lesion']=(resident_respond(testimony_cut,correction_query,0xF285)==repaired_reply and not resident_respond(evidence_cut,correction_query,0xF286))

        # Continuing resident conversation from the canonical checkpoint: current raw
        # matter activates only relations already licensed by live Adult state.
        conversation=type(final.adult).restore(copy.deepcopy(final.adult.checkpoint()))
        checks['canonical_adult_owns_authenticated_causal_contact_uptake']=callable(getattr(conversation,'observe_authenticated_causal_dialogue_contact',None))
        common_ground=type(final.adult).restore(copy.deepcopy(final.adult.checkpoint()));common_leaf=common_ground.language_adult.leaf(100,(0xA104,));common_ground_before,common_action=common_ground.externalize_causal_component(common_leaf.identity,0xEE11,0xEE01)
        common_return=bool(common_action and common_ground.settle_causal_dialogue_return(common_action,0xEE21,Q,0,True));common_row=common_ground.causal_chain_rows(common_leaf.identity)[0];common_program=common_action.programs[0] if common_action else 0;common_factor=common_ground.language_adult.programs.factor(common_program);common_acceptance=bytes(common_ground._causal_self_contained_surface(common_row,common_factor) or b'')
        common_alternatives=tuple(factor for factor in common_ground._causal_self_contained_factors() if factor!=common_factor);common_paraphrase_factor=common_alternatives[0] if len(common_alternatives)==1 else 0;common_paraphrase=bytes(common_ground._causal_self_contained_surface(common_row,common_paraphrase_factor) or b'') if common_paraphrase_factor else b'';common_paraphrase_orientation=common_ground.language_adult.world_causal_learning.grounding.orientation(common_paraphrase_factor) if common_paraphrase_factor else 0;common_cause,common_effect=int(common_row[2]),int(common_row[3]);common_paraphrase_children=((common_effect,common_cause) if common_paraphrase_orientation>0 else (common_cause,common_effect));common_pieces=common_ground.language.historical_span_pieces(common_paraphrase_factor) if common_paraphrase_factor else ();common_reversal=bytes(common_ground.language_adult._render_pieces(common_pieces,tuple(tuple(common_ground.language_adult._leaf_surface(child)) for child in reversed(common_paraphrase_children)))) if common_pieces else b''
        common_wrong=(common_ground.observe_authenticated_causal_dialogue_contact(common_acceptance,0xFE02) if common_acceptance else 0);common_once=(common_ground.observe_authenticated_causal_dialogue_contact(common_acceptance,0xEE01) if common_acceptance else 0);common_once_surface=common_ground.compose_causal_component(common_leaf.identity,channel=0xEE01)[0];common_cp=copy.deepcopy(common_ground.checkpoint());common_reversal_subject=type(common_ground).restore(copy.deepcopy(common_cp));common_reversal_first=common_reversal_subject.observe_authenticated_causal_dialogue_contact(common_reversal,0xEE01) if common_reversal else 0;common_reversal_second=common_reversal_subject.observe_authenticated_causal_dialogue_contact(common_reversal,0xEE01) if common_reversal else 0;common_restart=type(common_ground).restore(copy.deepcopy(common_cp));common_twice=common_restart.observe_authenticated_causal_dialogue_contact(common_paraphrase,0xEE01) if common_paraphrase else 0;common_ground_after,common_after_programs=common_restart.compose_causal_component(common_leaf.identity,channel=0xEE01)
        checks['authenticated_partner_acceptance_focuses_later_certified_discourse']=(common_return and common_wrong==0 and common_once==common_twice==1 and common_acceptance in common_once_surface and common_acceptance not in common_ground_after and len(common_after_programs)==len(common_action.programs)-1)
        checks['heldout_structural_paraphrase_and_reversed_causality_control_share_one_plan']=(common_paraphrase and common_paraphrase!=common_acceptance and common_reversal_first==common_reversal_second==0 and common_reversal_subject.causal_dialogue_uptake_support(0xEE01,int(common_row[4]))==1 and common_paraphrase not in common_ground_before)
        common_checkpoint_text=json.dumps(common_cp,sort_keys=True);checks['partner_uptake_survives_exact_restart_without_transcript']=(common_restart.checkpoint()!=common_cp and type(common_ground).restore(copy.deepcopy(common_cp)).checkpoint()==common_cp and common_acceptance.decode(errors='replace') not in common_checkpoint_text and common_paraphrase.decode(errors='replace') not in common_checkpoint_text)
        common_repair_question,common_repair_receipt=common_reversal_subject.externalize_endogenous_inquiry(0xEE12,0xEE01);common_repair_motor=bool(common_repair_receipt and common_reversal_subject.settle_endogenous_inquiry_motor_return(common_repair_receipt,0xEE01,True));common_repair_cp=copy.deepcopy(common_reversal_subject.checkpoint());common_repair_text=json.dumps(common_repair_cp,sort_keys=True)
        common_reasserted=type(common_ground).restore(copy.deepcopy(common_repair_cp));common_reasserted.observe_authenticated_causal_dialogue_contact(common_reversal,0xEE01);common_reasserted_closed=common_reasserted.settle_endogenous_inquiry_resolution(common_repair_receipt,0xEE01);common_reasserted_question,_=common_reasserted.externalize_endogenous_inquiry(0xEE13,0xEE01)
        common_answered=type(common_ground).restore(copy.deepcopy(common_repair_cp));common_answered_change=common_answered.observe_authenticated_causal_dialogue_contact(common_paraphrase,0xEE01);common_answered_closed=common_answered.settle_endogenous_inquiry_resolution(common_repair_receipt,0xEE01);common_answered_question,_=common_answered.externalize_endogenous_inquiry(0xEE14,0xEE01);common_answered_surface,common_answered_programs=common_answered.compose_causal_component(common_leaf.identity,channel=0xEE01)
        checks['causal_reversal_originates_specific_repair_and_answer_changes_later_plan']=(bool(common_repair_question and common_repair_receipt and common_repair_motor and common_repair_question.endswith(b'?') and common_repair_question not in common_ground_before and common_repair_question.decode(errors='replace') not in common_repair_text and not common_reasserted_closed and not common_reasserted_question and common_answered_change==1 and common_answered_closed and not common_answered_question and common_acceptance not in common_answered_surface and len(common_answered_programs)==len(common_action.programs)-1))

        # Same canonical Life: two source-qualified alternatives compete for
        # public inquiry, then separate interventions revise both relations.
        revision_old=type(loaded['testimony_revision_old_world'].adult).restore(copy.deepcopy(loaded['testimony_revision_old_world'].adult.checkpoint()));revision_old_alarm=revision_old.language_adult.leaf(100,(TESTIMONY_ALARM,));testimony_before,_=revision_old.compose_causal_component(revision_old_alarm.identity,channel=TESTIMONY_PARTNER)
        revision_need=type(loaded['testimony_revision_need'].adult).restore(copy.deepcopy(loaded['testimony_revision_need'].adult.checkpoint()));testimony_winner=revision_need._unique_inquiry_bid(revision_need._causal_repair_inquiry_candidates(TESTIMONY_PARTNER));testimony_question,testimony_question_receipt=revision_need.externalize_endogenous_inquiry(TESTIMONY_MOTOR_SOURCE,TESTIMONY_PARTNER)
        second_subject=type(loaded['testimony_revision_second_conflict'].adult).restore(copy.deepcopy(loaded['testimony_revision_second_conflict'].adult.checkpoint()));second_pending=tuple(second_subject.pending_endogenous_inquiry_actions.values());second_probe=type(second_subject).restore(copy.deepcopy(second_subject.checkpoint()));second_probe.pending_endogenous_inquiry_actions.clear();second_probe.reafferenced_endogenous_inquiry_actions.clear();second_winner=second_probe._unique_inquiry_bid(second_probe._causal_inquiry_candidates(TESTIMONY_PARTNER));second_question=(bytes(second_winner[0]) if second_winner else b'')
        contrast_cut=type(second_probe).restore(copy.deepcopy(second_probe.checkpoint()));contrast_permuted=type(second_probe).restore(copy.deepcopy(second_probe.checkpoint()))
        for relation_source in CAUSAL_FANIN_SOURCES:contrast_cut.language.withdraw_source(relation_source)
        contrast_cut_candidates=contrast_cut._causal_inquiry_candidates(TESTIMONY_PARTNER);contrast_cut_winner=contrast_cut._unique_inquiry_bid(contrast_cut_candidates);contrast_permuted.language_adult.world_causal_learning.bindings=dict(reversed(tuple(contrast_permuted.language_adult.world_causal_learning.bindings.items())));contrast_permuted._causal_dialogue_dispute_evidence=dict(reversed(tuple(contrast_permuted._causal_dialogue_dispute_evidence.items())));contrast_permuted_winner=contrast_permuted._unique_inquiry_bid(contrast_permuted._causal_inquiry_candidates(TESTIMONY_PARTNER))
        open_probe=type(loaded['testimony_revision_need'].adult).restore(copy.deepcopy(loaded['testimony_revision_need'].adult.checkpoint()));removed_pair=tuple(map(int,testimony_winner[2:])) if testimony_winner else ();open_probe._causal_dialogue_dispute_evidence={key:value for key,value in open_probe._causal_dialogue_dispute_evidence.items() if (int(key[3]),int(key[4]))!=removed_pair};open_question,open_receipt=open_probe.externalize_endogenous_inquiry(TESTIMONY_MOTOR_SOURCE+26,TESTIMONY_PARTNER)
        allocation_loaded=type(loaded['testimony_revision_competing_loaded'].adult).restore(copy.deepcopy(loaded['testimony_revision_competing_loaded'].adult.checkpoint()));allocation_loaded_state=allocation_loaded.endogenous_state_inquiry(TESTIMONY_PARTNER);allocation_loaded_causal=allocation_loaded.endogenous_causal_repair_inquiry(TESTIMONY_PARTNER);allocation_loaded_question,allocation_loaded_receipt=allocation_loaded.externalize_endogenous_inquiry(TESTIMONY_MOTOR_SOURCE+24,TESTIMONY_PARTNER)
        allocation_recovered=type(loaded['testimony_revision_competing_recovered'].adult).restore(copy.deepcopy(loaded['testimony_revision_competing_recovered'].adult.checkpoint()));allocation_recovered_state=allocation_recovered.endogenous_state_inquiry(TESTIMONY_PARTNER);allocation_recovered_causal=allocation_recovered.endogenous_causal_repair_inquiry(TESTIMONY_PARTNER);allocation_recovered_question,allocation_recovered_receipt=allocation_recovered.externalize_endogenous_inquiry(TESTIMONY_MOTOR_SOURCE+25,TESTIMONY_PARTNER)
        revision_settled=loaded['testimony_revision_settled'].adult;revision_root=revision_settled.language_adult.leaf(100,(0xA104,));revision_alarm=revision_settled.language_adult.leaf(100,(TESTIMONY_ALARM,));revision_growth=revision_settled.language_adult.leaf(100,(A_GROWTH,));testimony_after,testimony_after_programs=revision_settled.compose_causal_component(revision_root.identity,channel=TESTIMONY_PARTNER);testimony_alarm_after,_=revision_settled.compose_causal_component(revision_alarm.identity,channel=TESTIMONY_PARTNER);testimony_growth_after,_=revision_settled.compose_causal_component(revision_growth.identity,channel=TESTIMONY_PARTNER);revision_learner=revision_settled.language_adult.world_causal_learning;revision_rows=tuple(row for row in revision_learner.current_resolutions() if int(row[3])==revision_alarm.identity);revision_growth_rows=tuple(row for row in revision_learner.current_resolutions() if int(row[3])==revision_growth.identity);revision_source=revision_learner.testimony_source(TESTIMONY_PARTNER)
        checks['canonical_resource_recovery_reorders_somatic_then_source_causal_inquiry']=(bool(allocation_loaded_state and allocation_loaded_causal and allocation_loaded_state!=allocation_loaded_causal and allocation_loaded_receipt and allocation_loaded_question==allocation_loaded_state and allocation_recovered_state is None and allocation_recovered_causal==allocation_loaded_causal and allocation_recovered_receipt and allocation_recovered_question==allocation_recovered_causal))
        checks['unresolved_topology_can_recruit_learned_inquiry_without_authored_question']=(bool(open_receipt and open_question.endswith(b'?')) and len(open_probe.language_adult.world_causal_learning.current_open_fields())==2 and (open_receipt.obligation_candidates,open_receipt.obligation_effect)==((removed_pair[0],),removed_pair[1]) and open_question.decode(errors='replace') not in json.dumps(loaded['testimony_revision_need'].adult.checkpoint(),sort_keys=True))
        checks['canonical_first_settlement_exposes_second_dispute_question']=(len(second_pending)==1 and second_winner is not None and second_pending[0].surface_digest==hashlib.sha256(bytes(second_winner[0])).hexdigest() and (second_pending[0].obligation_candidates,second_pending[0].obligation_effect)==(tuple(map(int,second_winner[2])),int(second_winner[3])) and len(second_pending[0].obligation_candidates)==2)
        checks['same_field_fanin_not_surface_or_owns_contrastive_question']=(second_winner is not None and len(second_winner[2])==2 and second_question.count(b'because')==2 and second_question.decode(errors='replace') not in json.dumps(second_probe.checkpoint(),sort_keys=True))
        checks['contrastive_construction_lesion_preserves_competing_polar_needs_without_arbitrary_choice']=(contrast_cut_winner is None and len(contrast_cut_candidates)==2 and all(len(candidate[2])==1 and bytes(candidate[0])!=second_question and int(candidate[3])==int(second_winner[3]) for candidate in contrast_cut_candidates) and {int(candidate[2][0]) for candidate in contrast_cut_candidates}==set(map(int,second_winner[2])))
        checks['causal_field_and_dispute_storage_permutation_cannot_order_alternatives']=(contrast_permuted_winner==second_winner)
        checks['canonical_testimony_conflict_recruits_inquiry_then_revises_source_relation_and_discourse']=(bool(testimony_winner and testimony_question_receipt and testimony_question.endswith(b'?')) and testimony_question.decode(errors='replace') not in json.dumps(loaded['testimony_revision_need'].adult.checkpoint(),sort_keys=True) and (testimony_question_receipt.obligation_candidates,testimony_question_receipt.obligation_effect)==((int(testimony_winner[2]),),int(testimony_winner[3])) and len(revision_rows)==len(revision_growth_rows)==1 and int(revision_rows[0][2])==int(revision_growth_rows[0][2])==revision_settled.language_adult.leaf(100,(A_ROOTS,)).identity and revision_learner.testimony_accuracy.get(revision_source)==(2,1) and revision_learner.testimony_reliable(revision_source) and not revision_settled._causal_dialogue_dispute_evidence and b'the heater warms the greenhouse' in testimony_before and b'the irrigation alarm sounds' in testimony_before and b'plant roots lose water' in testimony_alarm_after and b'the irrigation alarm sounds' in testimony_alarm_after and b'plant stomata close' not in testimony_alarm_after and b'plant roots lose water' in testimony_growth_after and b'plant growth slows' in testimony_growth_after and len(testimony_after_programs)==len(revision_settled.causal_message_rows(revision_root.identity)))
        effect_raw=bytes(target[3].surface);cause_raw=target[7]
        effect_reply=resident_respond(conversation,effect_raw,0xF301)
        gateway_runtime=ReferenceLifeFunctionRuntimeV2.restore(
            species,copy.deepcopy(final_cp))
        gateway_reply=resident_contact(
            gateway_runtime,(effect_raw,),0xF314)
        gateway_source=Path(__file__).with_name(
            'reference_language_mastery_claude_gateway_v1.py').read_text()
        checks['claude_body_resumes_same_life_and_exposes_current_composition']=(
            gateway_reply==effect_reply
            and gateway_runtime.cursor==final.cursor
            and gateway_runtime.checkpoint()['schema']==final_cp['schema']
            and 'LanguageMasteryAdultV1.restore' not in gateway_source
            and 'server.adult=' not in gateway_source)
        _planned_surface,_planned_programs,expression_factors=_causal_chain_response(
            conversation,conversation.language_adult.leaf(100,(0xA104,)).identity,
            with_trace=True,with_expression_trace=True)
        unrelated_raw=bytes(conversation.language_adult.leaf(
            _probe_context,_heldout).surface)
        unrelated_reply=resident_respond(conversation,unrelated_raw,0xF302)
        cause_reply=resident_respond(conversation,cause_raw,0xF303)
        full_raw=b' '.join(bytes(conversation.relation_space_surface(x) or ()) for x in scrambled);full_reply=resident_respond(conversation,full_raw,0xF304)
        relation_rows=[]
        for rid in sorted(conversation.relation_basis.active):
            seq=conversation.relation_basis.expand_spaces(rid)
            if seq:relation_rows.append((len(seq),conversation.relation_basis.generation(rid),rid,tuple(seq)))
        intermediate=next((row for row in relation_rows if row[0]==5),None)
        if intermediate:
            isequence=intermediate[3];order=(2,0,4,1,3);subset=tuple(isequence[i] for i in order);subset_raw=b' '.join(bytes(conversation.relation_space_surface(x) or ()) for x in subset);subset_reply=resident_respond(conversation,subset_raw,0xF305);subset_expected=conversation.project_relation_basis(intermediate[2])
        else:subset_reply=b'';subset_expected=None
        primitive=next((row for row in relation_rows if row[0]==2),None)
        if primitive:
            pair=tuple(reversed(primitive[3]));pair_raw=b' '.join(bytes(conversation.relation_space_surface(x) or ()) for x in pair);pair_reply=resident_respond(conversation,pair_raw,0xF306)
        else:pair_reply=b''
        gap=next(((left,right) for left in spaces for right in spaces if left!=right and conversation.relation_basis_for_frontier((left,right))==0),())
        gap_raw=b' '.join(bytes(conversation.relation_space_surface(x) or ()) for x in gap) if gap else b'';gap_reply=resident_respond(conversation,gap_raw,0xF307) if gap else b''
        causal_expected=world_visible
        mid_raw=bytes(final.adult.language_adult.leaf(100,(A_ROOTS,)).surface);terminal_raw=bytes(final.adult.language_adult.leaf(100,(A_GROWTH,)).surface)
        mid_reply=resident_respond(conversation,mid_raw,0xF308);terminal_reply=resident_respond(conversation,terminal_raw,0xF309)
        focus_surfaces={atom:bytes(conversation.language_adult.leaf(100,(atom,)).surface) for atom in (A_NEED,A_HUMIDITY,A_HARDEN,A_CONSERVE,A_GROWTH)}
        own_focus_replies={atom:resident_respond(conversation,surface,0xF320+index) for index,(atom,surface) in enumerate(focus_surfaces.items())}
        checks['canonical_checkpoint_supports_continuing_relation_grounded_conversation']=(effect_reply!=cause_reply and cause_raw in effect_reply and effect_raw in effect_reply and effect_reply.count(b'.')>=2 and cause_raw in cause_reply and effect_raw in cause_reply and mid_reply and terminal_reply and not unrelated_reply and organized is not None and full_reply==bytes(organized) and intermediate is not None and subset_expected is not None and subset_reply==bytes(subset_expected) and primitive is not None and pair_reply.count(b'.')==2 and not gap_reply)
        checks['authenticated_contact_selects_local_causal_message_not_component_dump']=(
            len({effect_reply,mid_reply,terminal_reply,own_focus_replies[A_NEED]})==4 and
            all(surface in own_focus_replies[atom] for atom,surface in focus_surfaces.items()) and
            focus_surfaces[A_GROWTH] in mid_reply and focus_surfaces[A_CONSERVE] not in mid_reply and
            focus_surfaces[A_GROWTH] in terminal_reply and all(focus_surfaces[atom] not in terminal_reply for atom in (A_NEED,A_HUMIDITY,A_HARDEN,A_CONSERVE)) and
            focus_surfaces[A_HUMIDITY] not in own_focus_replies[A_NEED] and b'plant leaves wilt' not in effect_reply)
        branch_leaf=conversation.language_adult.leaf(100,(A_NEED,));sibling_leaf=conversation.language_adult.leaf(100,(A_HUMIDITY,));message_leaf=conversation.language_adult.leaf(100,(0xA104,));branch_rows=conversation.causal_message_rows(message_leaf.identity)
        branch_cause=bytes(conversation.language_adult.leaf(100,(0xA104,)).surface);branch_effect=bytes(branch_leaf.surface);sibling_effect=bytes(sibling_leaf.surface)
        public_factors=tuple(map(int,expression_factors[:len(_planned_programs)]));self_contained=set(conversation._causal_self_contained_factors());continuations=set(conversation._causal_continuation_factors())
        role_fit=[];certified=[]
        for index,(row,factor) in enumerate(zip(branch_rows,public_factors)):
            contiguous=index>0 and int(row[2])==int(branch_rows[index-1][3])
            role_fit.append(factor in (continuations if contiguous else self_contained))
            certified.append(conversation.causal_program_for_row(
                row,int(factor),False)==int(_planned_programs[index]))
        planned_rows=tuple((row,program,factor) for row,program,factor in zip(branch_rows,_planned_programs,public_factors))
        by_cause={}
        for item in planned_rows:by_cause.setdefault(int(item[0][2]),[]).append(item)
        coordinated_groups=[];coordination_factors=[]
        for siblings in by_cause.values():
            if len(siblings)<2:continue
            relation=conversation.language_adult.common_cause_span_expression(
                *(int(item[0][3]) for item in siblings))
            if not relation:
                relation=conversation.language_adult.common_cause_span_expression(
                    int(siblings[0][0][3]),int(siblings[1][0][3]))
            surface=(conversation._causal_sibling_surface(tuple(siblings),int(relation[0])) if relation else None)
            if surface is not None:coordinated_groups.append(bytes(surface));coordination_factors.append(int(relation[0]))
        motor_moves=[];index=0
        while index<len(planned_rows):
            siblings=[planned_rows[index]];cause=int(planned_rows[index][0][2])
            while index+len(siblings)<len(planned_rows) and int(planned_rows[index+len(siblings)][0][2])==cause:
                siblings.append(planned_rows[index+len(siblings)])
            following=index+len(siblings);following_cause=(int(planned_rows[following][0][2]) if following<len(planned_rows) else 0)
            projected=conversation._causal_motor_siblings(tuple(siblings),following_cause)
            if projected!=tuple(siblings):motor_moves.append((tuple(siblings),projected,following_cause))
            index=following
        distinct_parent_pair=next(((left,right) for left in planned_rows for right in planned_rows if int(left[0][2])!=int(right[0][2])),())
        wrong_parent_group=(conversation._causal_sibling_surface(distinct_parent_pair,coordination_factors[0]) if distinct_parent_pair and coordination_factors else None)
        action_probe=type(conversation).restore(copy.deepcopy(conversation.checkpoint()));action_surface,action_receipt=action_probe.externalize_causal_component(action_probe.language_adult.leaf(100,(0xA104,)).identity,0xF30B)
        checks['public_causal_action_credits_exact_evidence_certified_expression_programs']=(
            _planned_surface==effect_reply==bytes(action_surface) and len(_planned_programs)==len(branch_rows)>=2 and
            all(factor in expression_factors for factor in public_factors) and all(role_fit) and all(certified) and
            action_receipt is not None and tuple(action_receipt.programs)==tuple(_planned_programs) and
            tuple(action_receipt.factors)==tuple(expression_factors))
        checks['resident_message_boundary_visibly_composes_every_selected_certified_relation']=(
            len(branch_rows)==len(_planned_programs)>=2 and branch_effect in effect_reply and sibling_effect in effect_reply and
            branch_cause in effect_reply and all(certified) and len(branch_rows)<len(conversation.causal_focus_rows(message_leaf.identity)))
        checks['hierarchy_sensitive_plan_groups_selected_fanout_and_reduces_repetition']=(
            len(coordinated_groups)>=1 and all(bytes(conversation.language_adult._leaf_surface(int(row[3]))) in effect_reply for row in branch_rows)
            and effect_reply.count(b'Because ')==1 and effect_reply.count(b'.')<len(branch_rows)+1)
        checks['motor_message_plan_preserves_certified_resident_action']=(
            all(int(projected[-1][0][3])==following for _canonical,projected,following in motor_moves)
            and tuple(conversation.causal_program_for_row(row,factor,False) for row,factor in zip(branch_rows,public_factors))==tuple(_planned_programs)
            and action_receipt is not None and tuple(action_receipt.programs)==tuple(_planned_programs))
        checks['same_words_with_different_causal_parents_refuse_false_coordination']=(wrong_parent_group is None)
        coordination=conversation.language_adult.common_cause_span_expression(
            branch_leaf.identity,sibling_leaf.identity)
        coordination_factor=int(coordination[0]) if coordination else 0
        coordination_surface=bytes(coordination[1]) if coordination else b''
        coordination_examples=tuple(conversation.language_adult._template_exemplars.get(coordination_factor,()))
        coordination_certificates=tuple(conversation.language_adult.world_causal_learning.common_cause_certificate(*pair) for pair in coordination_examples)
        target_effects=tuple(int(row[3]) for row in branch_rows
                             if int(row[2])==int(message_leaf.identity))
        exact_coordination=conversation.language_adult.common_cause_span_expression(*target_effects)
        lived_language=tuple(bytes(event.payload) for event in curriculum.events if event.lane in {'surface','discourse_surface','utterance','authenticated_utterance'})
        checks['heldout_sibling_coordination_requires_two_distinct_common_cause_histories']=(
            coordination_factor>0 and len(coordination_examples)>=2 and all(coordination_certificates) and
            len({row[1] for row in coordination_certificates})>=2 and
            (branch_leaf.identity,sibling_leaf.identity) not in coordination_examples and
            exact_coordination and bytes(exact_coordination[1]) in effect_reply and
            all(bytes(exact_coordination[1]) not in raw for raw in lived_language))
        coordination_cut=type(conversation).restore(copy.deepcopy(conversation.checkpoint()))
        for source in GROUP_SOURCES:coordination_cut.language.withdraw_source(source)
        coordination_cut_reply=resident_respond(coordination_cut,effect_raw,0xF30E)
        checks['coordination_source_lesion_deoptimizes_wording_not_causal_truth']=(
            variable_arity['checks']['construction_source_withdrawal_deoptimizes_without_erasing_causal_truth']
            and not coordination_cut.language_adult.common_cause_span_expression(*target_effects)
            and branch_effect in coordination_cut_reply and sibling_effect in coordination_cut_reply)
        branch_cut=type(conversation).restore(copy.deepcopy(conversation.checkpoint()));branch_cut.language_adult.world_causal_learning.withdraw_source(0xF810)
        cut_reply=resident_respond(branch_cut,effect_raw,0xF30A);cut_rows=branch_cut.causal_message_rows(branch_cut.language_adult.leaf(100,(0xA104,)).identity)
        checks['branch_evidence_lesion_removes_only_its_claim_and_preserves_trunk']=(
            len(cut_rows)==len(branch_rows)-1 and branch_effect not in cut_reply and
            sibling_effect in cut_reply and cause_raw in cut_reply and bool(cut_reply))
        sibling_cut=type(conversation).restore(copy.deepcopy(conversation.checkpoint()));sibling_cut.language_adult.world_causal_learning.withdraw_source(0xF910)
        sibling_cut_reply=resident_respond(sibling_cut,effect_raw,0xF30C);sibling_cut_rows=sibling_cut.causal_message_rows(sibling_cut.language_adult.leaf(100,(0xA104,)).identity)
        checks['sibling_evidence_lesion_removes_only_its_claim_and_preserves_peer']=(
            len(sibling_cut_rows)==len(branch_rows)-1 and sibling_effect not in sibling_cut_reply and
            branch_effect in sibling_cut_reply and cause_raw in sibling_cut_reply)
        history_reordered=type(conversation).restore(copy.deepcopy(conversation.checkpoint()));history_learner=history_reordered.language_adult.world_causal_learning
        sibling_rows={int(row[3]):row for row in history_learner.current_resolutions() if int(row[3]) in (branch_leaf.identity,sibling_leaf.identity)}
        if set(sibling_rows)=={branch_leaf.identity,sibling_leaf.identity}:
            left=history_learner.ecology.pending[int(sibling_rows[branch_leaf.identity][4])];right=history_learner.ecology.pending[int(sibling_rows[sibling_leaf.identity][4])]
            left.opened_tick,right.opened_tick=right.opened_tick,left.opened_tick
        history_reply=resident_respond(history_reordered,effect_raw,0xF30D)
        checks['developmental_history_changes_only_equal_depth_motor_order']=(
            branch_effect in history_reply and sibling_effect in history_reply and
            effect_reply.index(branch_effect)<effect_reply.index(sibling_effect) and
            history_reply.index(sibling_effect)<history_reply.index(branch_effect) and
            len(history_reordered.causal_message_rows(message_leaf.identity))==len(branch_rows) and
            resident_respond(history_reordered,terminal_raw,0xF30F)==terminal_reply)
        reordered=type(conversation).restore(copy.deepcopy(conversation.checkpoint()));reordered_learner=reordered.language_adult.world_causal_learning
        reordered_learner.bindings=dict(reversed(tuple(reordered_learner.bindings.items())))
        reordered_learner.ecology.pending=dict(reversed(tuple(reordered_learner.ecology.pending.items())))
        checks['causal_structure_not_mapping_storage_order_owns_public_plan']=(
            resident_respond(reordered,effect_raw,0xF30B)==effect_reply and
            resident_respond(reordered,terminal_raw,0xF30C)==terminal_reply and
            resident_respond(reordered,focus_surfaces[A_NEED],0xF30D)==own_focus_replies[A_NEED])
        checks['obsolete_fixture_authorities_are_deleted']=all(
            not Path(__file__).with_name(name).exists() for name in (
                'reference_canonical_nested_composite_port_verify.py',
                'reference_discourse_quantity_interaction_verify.py'))
        certified_discourse=verify_certified_causal_discourse(
            loaded,curriculum,final.adult,effect_reply)
        checks.update(certified_discourse['checks'])
        def causal_reply_at(mark,raw,source):
            subject=type(loaded[mark].adult).restore(copy.deepcopy(loaded[mark].adult.checkpoint()))
            return resident_respond(subject,raw,source),subject
        pre_correction=before_reply;during_correction=heard_reply;repaired=repaired_reply
        numeric_causal_marks=tuple(mark for mark in marks if mark.startswith('causal_depth_') and mark[len('causal_depth_'):].isdigit())
        numeric_causal_rows=[]
        for index,mark in enumerate(numeric_causal_marks):
            subject=correction_projection(loaded[mark]);reply=resident_respond(subject,effect_raw,0xF340+index);leaf=subject.language_adult.leaf(100,(0xA104,));numeric_causal_rows.append((mark,reply,subject,len(subject.causal_focus_rows(leaf.identity)),len(subject.causal_message_rows(leaf.identity))))
        causal_depth_counts=tuple(reply.count(b'.') if reply else 0 for _mark,reply,_subject,_internal,_message in numeric_causal_rows)
        internal_depth_counts=tuple(row[3] for row in numeric_causal_rows);message_widths=tuple(row[4] for row in numeric_causal_rows)
        resource_rows=[]
        for index,mark in enumerate(mark for mark in marks if mark.startswith('causal_depth') and mark.endswith('_loaded')):
            recovered_mark=mark[:-len('_loaded')]+'_recovered'
            if recovered_mark not in loaded:continue
            loaded_row=causal_reply_at(mark,effect_raw,0xF360+index*2);recovered_row=causal_reply_at(recovered_mark,effect_raw,0xF361+index*2);resource_rows.append((mark,recovered_mark,*loaded_row,*recovered_row))
        loaded_reply=resource_rows[-1][2] if resource_rows else b'';recovered=resource_rows[-1][4] if resource_rows else b''
        checks['lived_correction_suppresses_old_claim_until_independent_repair']=bool(pre_correction and not during_correction and repaired and repaired!=pre_correction)
        checks['world_evidence_grows_internal_causal_depth_without_public_component_dump']=(len(causal_depth_counts)>=2 and all(right>left for left,right in zip(internal_depth_counts,internal_depth_counts[1:])) and len(set(message_widths))==1 and message_widths[0]>0 and max(causal_depth_counts)<=message_widths[0])
        control_learned=loaded['causal_control_history_learned'].adult;control_challenged=loaded['causal_control_history_challenged'].adult;control_loaded=loaded['causal_control_resilience_loaded'].adult;control_recovered=loaded['causal_control_resilience_recovered'].adult
        control_leaf=control_challenged.language_adult.leaf(100,(0xA104,));control_rows=control_challenged.causal_chain_rows(control_leaf.identity);control_programs=tuple(control_challenged.causal_program_for_row(row) for row in control_rows)
        control_credit_programs=tuple(sorted({credited for pid in control_programs for credited in control_challenged._causal_expression_credit_programs(pid)}))
        learned_histories=[];challenged_pairs=[]
        for pid in control_credit_programs:
            learned_profile=control_learned.language_adult.credit.rows.get(pid);challenged_profile=control_challenged.language_adult.credit.rows.get(pid)
            if learned_profile is None or challenged_profile is None:continue
            learned_histories.extend(local.control_history_q16 for local in learned_profile.contexts.values() if local.control_history_q16>0)
            challenged_pairs.extend((local.control_history_q16,local.controllability_q16,local.control_attempts,local.control_successes,local.background_attempts,local.background_successes) for local in challenged_profile.contexts.values() if local.control_history_q16)
        trained_loaded=type(control_loaded).restore(copy.deepcopy(control_loaded.checkpoint()));trained_leaf=trained_loaded.language_adult.leaf(100,(0xA104,));trained_loaded_reply,trained_loaded_programs=trained_loaded.compose_causal_component(trained_leaf.identity)
        yoked=type(control_loaded).restore(copy.deepcopy(control_loaded.checkpoint()))
        before_counts=[]
        for pid in control_credit_programs:
            profile=yoked.language_adult.credit.row(pid);before_counts.append((profile.control_attempts,profile.control_successes,profile.background_attempts,profile.background_successes,tuple((c,local.control_attempts,local.control_successes,local.background_attempts,local.background_successes) for c,local in sorted(profile.contexts.items()))))
            profile.control_history_q16=0
            for local in profile.contexts.values():local.control_history_q16=0
        after_counts=[]
        for pid in control_credit_programs:
            profile=yoked.language_adult.credit.row(pid);after_counts.append((profile.control_attempts,profile.control_successes,profile.background_attempts,profile.background_successes,tuple((c,local.control_attempts,local.control_successes,local.background_attempts,local.background_successes) for c,local in sorted(profile.contexts.items()))))
        yoked_leaf=yoked.language_adult.leaf(100,(0xA104,));yoked_reply,yoked_programs=yoked.compose_causal_component(yoked_leaf.identity);control_recovered_reply=resident_respond(type(control_recovered).restore(copy.deepcopy(control_recovered.checkpoint())),effect_raw,0xF342)
        checks['public_action_return_and_background_earn_slow_control_history']=(control_programs and min(learned_histories or (0,))>=Q and challenged_pairs and all(history>current and attempts==3 and successes==2 and backgrounds==1 and background_successes==0 for history,current,attempts,successes,backgrounds,background_successes in challenged_pairs))
        checks['learned_control_history_buffers_later_load_against_yoked_current_contingency']=(before_counts==after_counts and control_loaded.language_adult.slow_resource_history.pressure_q16()>=Q//2 and len(trained_loaded_programs)==len(trained_loaded.causal_message_rows(trained_leaf.identity)) and 0<len(yoked_programs)<len(trained_loaded_programs)<len(control_programs) and control_loaded.language_adult.world_causal_learning.checkpoint()==yoked.language_adult.world_causal_learning.checkpoint())
        checks['control_resilience_recovery_preserves_full_causal_depth']=(control_recovered.language_adult.slow_resource_history.pressure_q16()==0 and control_recovered_reply==trained_loaded_reply)
        visible_resource_rows=tuple(row for row in resource_rows if row[2] and row[4])
        # Scene-stream episodes are rematerializable acquisition evidence. Compare
        # future-relevant learned language after applying the same resident
        # consolidation law, not incidental timing of that cleanup.
        for _loaded_mark,_recovered_mark,_loaded_reply_row,loaded_subject,_recovered_reply_row,recovered_subject in visible_resource_rows:
            loaded_subject.language_adult.language.consolidate_scene_streams()
            recovered_subject.language_adult.language.consolidate_scene_streams()
        checks['somatic_pressure_is_not_a_global_discourse_opcode']=bool(visible_resource_rows) and all(loaded_subject.language_adult.slow_resource_history.pressure_q16()>0 and recovered_subject.language_adult.slow_resource_history.pressure_q16()==0 and loaded_subject.language_adult.world_causal_learning.checkpoint()==recovered_subject.language_adult.world_causal_learning.checkpoint() and loaded_subject.language_adult.language.checkpoint()==recovered_subject.language_adult.language.checkpoint() for _loaded_mark,_recovered_mark,_loaded_reply_row,loaded_subject,_recovered_reply_row,recovered_subject in visible_resource_rows)

        # One Adult, one world, three partner-local lived histories. No authored state taxonomy.
        matrix=loaded['open_state_prompt_loaded'].adult;matrix_effect=matrix.language_adult.leaf(100,(0xA104,));matrix_programs=matrix._longest_causal_program_component()
        prompt_bindings=matrix.language.invert_surface(tuple(OPEN_STATE_PROMPT));prompt_binding=prompt_bindings[0] if len(prompt_bindings)==1 else None
        partner_rows={}
        for partner in (HISTORY_A,HISTORY_B,HISTORY_C):
            state_surface=(bytes(matrix.respond_context_affordance(prompt_binding.context,prompt_binding.atoms,partner) or b'') if prompt_binding else b'')
            causal_surface,used_programs=matrix.compose_causal_component(matrix_effect.identity,channel=partner)
            context=matrix._causal_dialogue_appraisal_context(matrix_programs,partner);credited=matrix._causal_expression_credit_program(matrix_programs[0],context);local=matrix.language_adult.credit.row(credited).contexts.get(context) if credited else None
            partner_rows[partner]=(state_surface,bytes(causal_surface),tuple(used_programs),local)
        a_state,a_causal,a_programs,a_local=partner_rows[HISTORY_A];b_state,b_causal,b_programs,b_local=partner_rows[HISTORY_B];c_state,c_causal,c_programs,c_local=partner_rows[HISTORY_C]
        checks['partner_history_matrix_creates_three_local_control_regimes']=(all(row is not None for row in (a_local,b_local,c_local)) and a_local.control_history_q16>b_local.control_history_q16>c_local.control_history_q16==0 and a_local.controllability_q16>b_local.controllability_q16>c_local.controllability_q16==0)
        checks['same_open_state_prompt_exposes_only_language_distinctions_the_adult_has_earned']=(bool(a_state) and a_state==b_state and c_state!=a_state and b'I can influence what happens' in a_state and b'I have little control over what happens' in c_state and all(b'my body is under strain' in row for row in (a_state,b_state,c_state)))
        matrix_message_width=len(matrix.causal_message_rows(matrix_effect.identity))
        checks['same_world_and_load_partner_history_changes_downstream_causal_action_depth']=(len(matrix_programs)>matrix_message_width and len(a_programs)==matrix_message_width>len(b_programs)>len(c_programs)>0 and matrix.language_adult.slow_resource_history.pressure_q16()>0)
        recovered_matrix=loaded['open_state_prompt_recovered'].adult;recovered_effect=recovered_matrix.language_adult.leaf(100,(0xA104,));recovered_programs=recovered_matrix._longest_causal_program_component();recovered_binding=recovered_matrix.language.invert_surface(tuple(OPEN_STATE_PROMPT))[0]
        recovered_rows=[]
        for partner in (HISTORY_A,HISTORY_B,HISTORY_C):
            state=bytes(recovered_matrix.respond_context_affordance(recovered_binding.context,recovered_binding.atoms,partner) or b'');surface,used=recovered_matrix.compose_causal_component(recovered_effect.identity,channel=partner);recovered_rows.append((partner,state,bytes(surface),tuple(used)))
        recovered_message_width=len(recovered_matrix.causal_message_rows(recovered_effect.identity))
        checks['quiet_recovery_restores_action_depth_without_erasing_partner_control_history']=(recovered_matrix.language_adult.slow_resource_history.pressure_q16()==0 and len(recovered_programs)>recovered_message_width and all(len(row[3])==recovered_message_width and b'my body is settled' in row[1] for row in recovered_rows) and b'I have little control over what happens' in recovered_rows[2][1] and recovered_rows[0][1]==recovered_rows[1][1]!=recovered_rows[2][1])
        # Lesion prompt relation evidence only: open answer disappears, causal policy survives.
        prompt_cut=type(matrix).restore(copy.deepcopy(matrix.checkpoint()))
        for source_id in (0xDD20,0xDD21):prompt_cut.language.withdraw_source(source_id)
        prompt_cut_state=(prompt_cut.respond_context_affordance(prompt_binding.context,prompt_binding.atoms,HISTORY_A) if prompt_binding else None);prompt_cut_causal,prompt_cut_programs=prompt_cut.compose_causal_component(matrix_effect.identity,channel=HISTORY_A)
        checks['prompt_relation_lesion_silences_open_self_report_not_causal_action']=(prompt_cut_state is None and len(prompt_cut_programs)==len(a_programs) and bytes(prompt_cut_causal)==a_causal)
        # Lesion only B's slow control history; action/background counts remain identical.
        history_cut=type(matrix).restore(copy.deepcopy(matrix.checkpoint()));history_context=history_cut._causal_dialogue_appraisal_context(matrix_programs,HISTORY_B);count_before=[]
        for pid in matrix_programs:
            credited=history_cut._causal_expression_credit_program(pid,history_context);local=history_cut.language_adult.credit.row(credited).contexts.get(history_context) if credited else None
            if local is None:continue
            count_before.append((pid,local.control_attempts,local.control_successes,local.background_attempts,local.background_successes));local.control_history_q16=0
        history_cut_state=bytes(history_cut.respond_context_affordance(prompt_binding.context,prompt_binding.atoms,HISTORY_B) or b'');history_cut_causal,history_cut_programs=history_cut.compose_causal_component(matrix_effect.identity,channel=HISTORY_B);count_after=[]
        for pid in matrix_programs:
            credited=history_cut._causal_expression_credit_program(pid,history_context);local=history_cut.language_adult.credit.row(credited).contexts.get(history_context) if credited else None
            if local is None:continue
            count_after.append((pid,local.control_attempts,local.control_successes,local.background_attempts,local.background_successes))
        checks['control_history_lesion_changes_report_and_action_without_changing_current_counts']=(count_before==count_after and len(history_cut_programs)<len(b_programs) and history_cut_state!=b_state and b'I have little control over what happens' in history_cut_state)
        # Lesion only slow body history: world/control remain, strain wording and truncation disappear.
        body_cut=type(matrix).restore(copy.deepcopy(matrix.checkpoint()));body_world=copy.deepcopy(body_cut.language_adult.world_causal_learning.checkpoint());body_cut.language_adult.slow_resource_history.lesion_history();body_state=bytes(body_cut.respond_context_affordance(prompt_binding.context,prompt_binding.atoms,HISTORY_C) or b'');body_causal,body_programs=body_cut.compose_causal_component(matrix_effect.identity,channel=HISTORY_C)
        checks['body_history_lesion_changes_strain_and_recovery_not_world_truth']=(body_cut.language_adult.world_causal_learning.checkpoint()==body_world and b'my body is settled' in body_state and b'my body is under strain' not in body_state and len(body_programs)==matrix_message_width>len(c_programs))

        d16_public=d16s_public=b''
        # Recompute one current causal closure through three independently lived
        # partner histories. World evidence and language stay fixed; partner-local
        # controllability and common body load jointly shape public depth/appraisal.
        def partner_profile(mark,partner):
            subject=type(loaded[mark].adult).restore(copy.deepcopy(loaded[mark].adult.checkpoint()))
            leaf=subject.language_adult.leaf(100,(0xA104,));surface,programs=subject.compose_causal_component(leaf.identity,channel=partner)
            state_programs=subject._longest_causal_program_component();context=subject._causal_dialogue_appraisal_context(state_programs,partner);credited=subject._causal_expression_credit_program(state_programs[0],context) if state_programs else 0
            felt=subject.language_adult.somatic_appraisal(credited or state_programs[0],context) if state_programs else None
            return subject,bytes(surface),felt
        partner_rows={mark:{partner:partner_profile(mark,partner) for partner in (HISTORY_A,HISTORY_B,HISTORY_C)} for mark in ('partner_history_matrix','partner_history_matrix_loaded','partner_history_matrix_recovered')}
        base_profiles=partner_rows['partner_history_matrix'];loaded_profiles=partner_rows['partner_history_matrix_loaded'];recovered_profiles=partner_rows['partner_history_matrix_recovered']
        a_loaded=loaded_profiles[HISTORY_A][1];b_loaded=loaded_profiles[HISTORY_B][1];c_loaded=loaded_profiles[HISTORY_C][1]
        checks['partner_history_and_body_state_jointly_change_visible_discourse']=(a_loaded.count(b'.')>=3 and 1<c_loaded.count(b'.')<b_loaded.count(b'.')<=a_loaded.count(b'.') and len(a_loaded)>len(b_loaded)>len(c_loaded) and all(b'my body is under strain' in row[1] for row in loaded_profiles.values()))
        checks['partner_local_controllability_is_graded_not_a_partner_label_opcode']=(loaded_profiles[HISTORY_A][2].controllability_q16>loaded_profiles[HISTORY_B][2].controllability_q16>loaded_profiles[HISTORY_C][2].controllability_q16 and b'I can influence what happens' in a_loaded and b'I have little control over what happens' in c_loaded)
        checks['same_world_language_and_correction_support_all_partner_perspectives']=all(row[0].language_adult.world_causal_learning.checkpoint()==loaded_profiles[HISTORY_A][0].language_adult.world_causal_learning.checkpoint() and row[0].language_adult.language.checkpoint()==loaded_profiles[HISTORY_A][0].language_adult.language.checkpoint() and b'heater warms the greenhouse' in row[1] and b'morning sunlight' not in row[1] for row in loaded_profiles.values())
        checks['recovery_restores_discourse_depth_without_erasing_partner_history']=all(recovered_profiles[p][1].count(b'.')==base_profiles[p][1].count(b'.')>=3 and recovered_profiles[p][2].controllability_q16==base_profiles[p][2].controllability_q16 for p in (HISTORY_A,HISTORY_B,HISTORY_C))
        partner_restart=type(loaded['partner_history_matrix_loaded'].adult).restore(copy.deepcopy(loaded['partner_history_matrix_loaded'].adult.checkpoint()));restart_leaf=partner_restart.language_adult.leaf(100,(0xA104,))
        checks['partner_conditioned_discourse_survives_exact_restart']=all(partner_restart.compose_causal_component(restart_leaf.identity,channel=p)[0]==loaded_profiles[p][1] for p in (HISTORY_C,HISTORY_A,HISTORY_B))

        # Same continuing Adult: indexed embedded raw-contact binding, rather than a
        # host parse, lets a public self-state/causal-affordance collision recruit a
        # learned productive inquiry.
        inquiry=loaded['endogenous_state_inquiry_loaded'].adult;inquiry_programs=inquiry._longest_causal_program_component();inquiry_leaf=inquiry.language_adult.leaf(100,(0xA104,));state_context=inquiry.language_adult._somatic_appraisal_language_context();inquiry_rows={}
        for partner in (HISTORY_A,HISTORY_B,HISTORY_C):
            context=inquiry._causal_dialogue_appraisal_context(inquiry_programs,partner);credited=inquiry._causal_expression_credit_program(inquiry_programs[0],context);felt=inquiry.language_adult.somatic_appraisal(credited or inquiry_programs[0],context);atoms=inquiry.language_adult.somatic_appraisal_atoms(felt);state=bytes(inquiry.language.realize(state_context,atoms) or b'');question=inquiry.endogenous_state_inquiry(partner);touches=inquiry.last_endogenous_inquiry_touches;surface,used=inquiry.compose_causal_component(inquiry_leaf.identity,channel=partner);inquiry_rows[partner]=(state,bytes(question or b''),tuple(used),felt,touches,bytes(surface))
        ia,ib,ic=inquiry_rows[HISTORY_A],inquiry_rows[HISTORY_B],inquiry_rows[HISTORY_C]
        checks['same_public_self_state_collision_recruits_productive_inquiry']=(ia[0]==ib[0]!=ic[0] and ia[3].controllability_q16>ib[3].controllability_q16>ic[3].controllability_q16 and len(ia[2])>len(ib[2])>len(ic[2]) and not ia[1] and bool(ib[1] and ic[1]) and ib[1]!=ic[1])
        checks['endogenous_inquiry_work_stays_local_to_lived_candidate_contexts']=(0<max(row[4] for row in inquiry_rows.values())<=len(inquiry_programs)<=16)
        inquiry_action=type(inquiry).restore(copy.deepcopy(inquiry.checkpoint()));action_question,action_receipt=inquiry_action.externalize_endogenous_inquiry(0xE4F0,HISTORY_B);blocked_question,blocked_receipt=inquiry_action.externalize_endogenous_inquiry(0xE4F1,HISTORY_B);pending_inquiry_cp=copy.deepcopy(inquiry_action.checkpoint());pending_inquiry_restart=type(inquiry).restore(copy.deepcopy(pending_inquiry_cp))
        checks['resident_selected_inquiry_is_checkpointed_public_action_not_host_turn_policy']=(action_question==ib[1] and action_receipt is not None and not blocked_question and blocked_receipt is None and pending_inquiry_restart.checkpoint()==pending_inquiry_cp and not pending_inquiry_restart.externalize_endogenous_inquiry(0xE4F2,HISTORY_B)[0])
        inquiry_outcomes={}
        for independent in (False,True):
            subject=type(inquiry).restore(copy.deepcopy(inquiry.checkpoint()));surface,receipt=subject.externalize_causal_component(inquiry_leaf.identity,0xE501+(1 if independent else 0),HISTORY_B);settled=subject.settle_causal_dialogue_return(receipt,0xE511+(1 if independent else 0),Q,0,independent) if receipt else False;after_question=bytes(subject.endogenous_state_inquiry(HISTORY_B) or b'');after_surface,after_programs=subject.compose_causal_component(inquiry_leaf.identity,channel=HISTORY_B);inquiry_outcomes[independent]=(settled,after_question,tuple(after_programs),bytes(after_surface))
        yoked_inquiry,causal_inquiry=inquiry_outcomes[False],inquiry_outcomes[True]
        inquiry_message_width=len(inquiry.causal_message_rows(inquiry_leaf.identity))
        checks['independent_return_resolves_inquiry_and_restores_causal_depth']=(yoked_inquiry[0] and bool(yoked_inquiry[1]) and len(yoked_inquiry[2])<=len(ib[2]) and causal_inquiry[0] and not causal_inquiry[1] and len(causal_inquiry[2])==inquiry_message_width>len(ib[2]))
        inquiry_recovered=loaded['endogenous_state_inquiry_recovered'].adult;recovered_inquiry_leaf=inquiry_recovered.language_adult.leaf(100,(0xA104,));recovered_inquiry_width=len(inquiry_recovered.causal_message_rows(recovered_inquiry_leaf.identity));checks['quiet_recovery_abolishes_inquiry_without_erasing_partner_history']=(inquiry_recovered.language_adult.slow_resource_history.pressure_q16()==0 and all(inquiry_recovered.endogenous_state_inquiry(partner) is None for partner in (HISTORY_A,HISTORY_B,HISTORY_C)) and all(len(inquiry_recovered.compose_causal_component(recovered_inquiry_leaf.identity,channel=partner)[1])==recovered_inquiry_width for partner in (HISTORY_A,HISTORY_B,HISTORY_C)))

        checks['peer_emergent_intermediate_relation_is_immediately_available_to_conversation']=(intermediate is not None and subset_reply==bytes(subset_expected) and subset_reply.count(b'.')==intermediate[0])
        checks['relation_gap_refuses_instead_of_inventing_missing_structure']=bool(gap) and not gap_reply
        evidence=target[1].ecology.pending[target[4]].evidence;by_source={}
        for row in evidence:
            if row.active:by_source.setdefault(int(row.source),set()).add(tuple(row.coalition))
        support_sources=tuple(sorted(source for source,coalitions in by_source.items() if len(coalitions)>=4))
        causal_cut=type(final.adult).restore(copy.deepcopy(final.adult.checkpoint()));causal_cut.language_adult.world_causal_learning.withdraw_source(support_sources[0] if support_sources else 0)
        lesioned_reply=resident_respond(causal_cut,effect_raw,0xF311)
        checks['causal_evidence_lesion_revises_reply_without_erasing_downstream_relation']=(bool(lesioned_reply) and lesioned_reply!=causal_expected and effect_raw in lesioned_reply and cause_raw not in lesioned_reply)
        relation_cut=type(final.adult).restore(copy.deepcopy(final.adult.checkpoint()));rbest=relation_cut.relation_basis.best_derived();rspaces=relation_cut.relation_basis.expand_spaces(rbest);rprimitive=next(row for row in relation_cut.relation_basis.relations.values() if row.direct_evidence and row.left_space==rspaces[0] and row.right_space==rspaces[1]);rprimitive_sources=tuple(rprimitive.direct_evidence)
        for evidence_source in rprimitive_sources:relation_cut.relation_basis.withdraw_evidence(evidence_source)
        checks['relation_evidence_lesion_silences_only_dependent_scrambled_reply']=bool(rprimitive_sources) and not resident_respond(relation_cut,full_raw,0xF312)
        conversation_restart=type(final.adult).restore(copy.deepcopy(final.adult.checkpoint()));conversation_source=Path(__file__).with_name('reference_mathematical_adult_workbench_v1.py').read_text();checks['resident_conversation_restarts_without_external_question_or_model_authority']=(resident_respond(conversation_restart,effect_raw,0xF313)==effect_reply and all(token not in conversation_source for token in ('GroundedCausalQuestion','causal_models=','answer_key')) and all(token not in json.dumps(conversation_restart.checkpoint(),sort_keys=True).lower() for token in ('question_state','conversation_buffer','transcript')))
        checks['causal_discourse_has_no_self_echo_motor_or_arbitrary_connective_fixture']=(all(token not in conversation_source for token in ('_CertifiedChainMotor','reafference(step,step[0])','THEREFORE=','HOWEVER=')) and 'def causal_chain_rows' in conversation_source)

        restored=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(final_cp));checks['final_checkpoint_restarts_exactly']=restored.checkpoint()==final_cp and restored.adult.checkpoint()==final_cp['adult']
        altered=list(curriculum.events);e=altered[0];altered[0]=LifeCurriculumEventV2(e.sequence,e.lane,e.source+1,e.payload);changed=type(curriculum)(tuple(altered));past_refused=False
        try:restored.run(changed)
        except ValueError:past_refused=True
        checks['changed_past_refuses_from_history_commitment_without_transcript']=past_refused and 'history_docs' not in final_cp and len(final_cp['history_root'])==64

        extension=(LifeCurriculumEventV2(1,'scene',7501,(100,NEW)),LifeCurriculumEventV2(1,'surface',7501,tuple(b'overnight')),LifeCurriculumEventV2(1,'scene',7502,(100,NEW)),LifeCurriculumEventV2(1,'surface',7502,tuple(b'overnight')))
        continued=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(final_cp));continued.run(curriculum.append(extension));post_life_surface=bytes(continued.adult.language_adult.leaf(_probe_context,_new_heldout).surface);current_subject=bytes(final.adult.language_adult.language.lexeme(_heldout[0]) or ())
        checks['same_final_adult_continues_learning_without_rebirth']=(refused(final.adult.language_adult,_new_heldout) and current_subject and post_life_surface.startswith(current_subject+b' ') and post_life_surface.endswith(b' overnight.'))

        naive_event_work=sum(curriculum.mark_cursor(mark) for mark in marks);checks['one_pass_factory_eliminates_prefix_retraining_work']=(naive_event_work>5*len(curriculum.events))
        checkpoint_bytes=len(json.dumps(final_cp,sort_keys=True,separators=(',',':')).encode());pre_tail_cp=loaded['causal_discourse_continuation'].checkpoint();pre_tail_bytes=len(json.dumps(pre_tail_cp,sort_keys=True,separators=(',',':')).encode());tail_start=curriculum.mark_cursor('causal_discourse_continuation');tail_document_bytes=len(json.dumps([event.document() for event in curriculum.events[tail_start:]],sort_keys=True,separators=(',',':')).encode());tail_checkpoint_growth=checkpoint_bytes-pre_tail_bytes;tail_events=max(1,len(curriculum.events)-tail_start);checkpoint_leaves=_checkpoint_leaf_count(final_cp);pre_tail_leaves=_checkpoint_leaf_count(pre_tail_cp)
        checkpoint_blob=json.dumps(final_cp,sort_keys=True).lower();checks['checkpoint_retains_active_tail_without_contact_transcript']=(tail_checkpoint_growth>0 and tail_document_bytes>0 and checkpoint_leaves>pre_tail_leaves and all(token not in checkpoint_blob for token in ('history_docs','transcript','conversation_buffer','answer_key')))
        curriculum_sources=(Path(__file__).with_name('reference_life_function_curriculum_v1.py'),)+tuple(Path(__file__).with_name(module+'.py') for module in CANONICAL_TAIL_MODULES_V2)
        source='\n'.join(path.read_text().lower() for path in curriculum_sources);checks['canonical_source_contains_no_fixture_private_teacher_or_answer_key']=all(token not in source for token in ('fixture','answer_key','private_teacher'))
        checks['obsolete_private_open_ended_conversation_fixture_is_deleted']=not Path(__file__).with_name('reference_open_ended_conversation_verify.py').exists()

    failed=[name for name,passed in checks.items() if not passed];elapsed=(time.perf_counter()-started)*1000
    language_phenotype_improved = all(checks[name] for name in (
        'world_earned_language_recomposes_never_taught_form',
        'independent_consequence_revises_visible_explanation_without_sentence_replay',
        'canonical_checkpoint_supports_continuing_relation_grounded_conversation',
        'public_causal_action_credits_exact_evidence_certified_expression_programs',
        'resident_message_boundary_visibly_composes_every_selected_certified_relation',
        'hierarchy_sensitive_plan_groups_selected_fanout_and_reduces_repetition',
        'motor_message_plan_preserves_certified_resident_action',
        'partner_adverse_uptake_reopening_is_in_one_life_factory',
        'same_words_with_different_causal_parents_refuse_false_coordination',
        'heldout_sibling_coordination_requires_two_distinct_common_cause_histories',
        'coordination_source_lesion_deoptimizes_wording_not_causal_truth',
        'branch_evidence_lesion_removes_only_its_claim_and_preserves_trunk',
        'sibling_evidence_lesion_removes_only_its_claim_and_preserves_peer',
        'developmental_history_changes_only_equal_depth_motor_order',
        'causal_structure_not_mapping_storage_order_owns_public_plan',
        'learned_control_history_buffers_later_load_against_yoked_current_contingency',
        'causal_evidence_lesion_revises_reply_without_erasing_downstream_relation',
        'partner_history_and_body_state_jointly_change_visible_discourse',
        'partner_local_controllability_is_graded_not_a_partner_label_opcode',
        'same_world_language_and_correction_support_all_partner_perspectives',
        'partner_conditioned_discourse_survives_exact_restart',
        'authenticated_partner_acceptance_focuses_later_certified_discourse',
        'heldout_structural_paraphrase_and_reversed_causality_control_share_one_plan',
        'partner_uptake_survives_exact_restart_without_transcript',
        'causal_reversal_originates_specific_repair_and_answer_changes_later_plan',
        'canonical_resource_recovery_reorders_somatic_then_source_causal_inquiry',
        'canonical_first_settlement_exposes_second_dispute_question',
        'canonical_testimony_conflict_recruits_inquiry_then_revises_source_relation_and_discourse',
        'same_public_self_state_collision_recruits_productive_inquiry',
        'resident_selected_inquiry_is_checkpointed_public_action_not_host_turn_policy',
        'independent_return_resolves_inquiry_and_restores_causal_depth',
        'quiet_recovery_abolishes_inquiry_without_erasing_partner_history',
        'cartesian_proposition_and_successor_fixture_is_absent_from_life',
        'current_world_contact_recruits_multi_relation_certified_composition',
        'public_causal_composition_is_productive_not_contact_replay',
        'developmental_history_changes_same_world_focus',
        'causal_evidence_withdrawal_changes_current_composition',
        'candidate_enumeration_permutation_cannot_choose_public_order',
        'partner_uptake_replans_instead_of_prefix_chopping',
        'quiet_without_new_evidence_cannot_grow_or_rewrite_discourse',
        'restart_preserves_causal_composition_without_stored_paragraph',
        'obsolete_fixture_authorities_are_deleted',
        'heldout_natural_causal_question_orients_same_certified_closure',
        'question_construction_requires_developmental_experience',
        'same_word_bag_malformed_question_does_not_route',
        'question_evidence_withdrawal_silences_question_not_declarative',
        'later_world_consequence_acquires_unseen_causal_topic_in_same_adult',
        'unseen_topic_requires_independent_consequence_and_live_sources',
        'unseen_topic_survives_checkpoint_without_question_or_answer',
        'unknown_contact_refuses_without_unowned_public_action_router',
        'lexical_revision_preserves_world_coordinates_while_revising_both_surfaces',
        'resident_action_consequences_recompose_existing_causal_discourse',
        'lexical_causal_revision_varies_with_source_and_counterhistory',
        'lexical_causal_revision_survives_restart_without_transcript',
        'same_final_adult_continues_learning_without_rebirth',
        'variable_arity_message_planning_is_in_one_life_factory',
        'cross_action_structural_continuity_is_in_one_life_factory',
    ))
    capability_frontier={
        'curriculum_events':len(curriculum.events),'curriculum_marks':len(marks),
        'operator_operations':0 if winner is None else int(winner.operations),
        'operator_clauses':0 if operator_surface is None else int(operator_surface.count(b'.')),
        'operator_bytes':0 if operator_surface is None else len(operator_surface),
        'relation_generation':0 if best<=0 else int(final_rel.relation_basis.generation(best)),
        'relation_spaces':0 if spaces is None else len(spaces),
        'self_organized_clauses':0 if organized is None else int(bytes(organized).count(b'.')),
        'world_causal_relations':len(target[1].current_resolutions()),
        'world_derived':world_visible.decode(errors='replace'),
        'conversation_causal_reply':effect_reply.decode(errors='replace'),
        'variable_arity_causal_reply':variable_arity['after'],
        'cross_action_continuation':cross_action['after'],
        'focal_causal_replies':{
            'root':cause_reply.decode(errors='replace'),'middle':mid_reply.decode(errors='replace'),
            'terminal':terminal_reply.decode(errors='replace'),
            'sibling':own_focus_replies[A_NEED].decode(errors='replace')},
        'natural_causal_question':natural_question,
        'causal_dialogue_depth':effect_reply.count(b'.'),
        'causal_expression_factors':len(expression_factors),
        'causal_public_bytes':len(effect_reply),
        'loaded_causal_dialogue_depth':loaded_reply.count(b'.'),
        'recovered_causal_dialogue_depth':recovered.count(b'.'),
        'partner_loaded_depths':{
            hex(p):loaded_profiles[p][1].count(b'.') for p in (HISTORY_A,HISTORY_B,HISTORY_C)},
        'causal_repair_question':common_repair_question.decode(errors='replace'),
        'open_field_question':open_question.decode(errors='replace'),
        'inquiry_allocation_loaded':allocation_loaded_question.decode(errors='replace'),
        'inquiry_allocation_recovered':allocation_recovered_question.decode(errors='replace'),
        'testimony_revision_question':testimony_question.decode(errors='replace'),
        'testimony_revision_second_question':second_question.decode(errors='replace'),
        'testimony_revision_relations':len(testimony_after_programs),
        'certified_causal_discourse_relations':len(certified_discourse['programs']),
        'certified_causal_discourse_bytes':len(certified_discourse['public']),
        'partner_replanned_causal_discourse_relations':len(certified_discourse['scarce_programs']),
        'endogenous_inquiry':action_question.decode(errors='replace'),
        'endogenous_inquiry_loaded_depths':{
            hex(p):len(inquiry_rows[p][2]) for p in (HISTORY_A,HISTORY_B,HISTORY_C)},
        'endogenous_inquiry_independent_depth':len(causal_inquiry[2]),
        'endogenous_inquiry_yoked_depth':len(yoked_inquiry[2]),
        'post_life':post_life_surface.decode(errors='replace')}
    capability_frontier['unseen_causal_topic']=topic_reply.decode(errors='replace')
    capability_frontier['lexically_revised_causal_topic']=integrated_lc_reply.decode(errors='replace')
    visible_language_gain={
        'variable_arity_before':variable_arity['before'],
        'variable_arity_after':variable_arity['after'],
        'cross_action_before':cross_action['before'],
        'cross_action_after':cross_action['after'],
        'resident_causal':effect_reply.decode(errors='replace'),
        'unseen_causal_topic':topic_reply.decode(errors='replace'),
        'lexically_revised_causal_topic':integrated_lc_reply.decode(errors='replace'),
        'natural_causal_question':natural_question,
        'focal_middle':mid_reply.decode(errors='replace'),
        'focal_terminal':terminal_reply.decode(errors='replace'),
        'focal_sibling':own_focus_replies[A_NEED].decode(errors='replace'),
        'exact_causal_action_programs':len(_planned_programs),
        'certified_causal_discourse':certified_discourse['public'].decode(errors='replace'),
        'partner_replanned_causal_discourse':certified_discourse['scarce'].decode(errors='replace'),
        'partner_a_loaded':a_loaded.decode(errors='replace'),
        'partner_b_loaded':b_loaded.decode(errors='replace'),
        'partner_c_loaded':c_loaded.decode(errors='replace'),
        'partner_common_ground_before':common_ground_before.decode(errors='replace'),
        'partner_common_ground_paraphrase':common_paraphrase.decode(errors='replace'),
        'partner_common_ground_repair_question':common_repair_question.decode(errors='replace'),
        'partner_common_ground_after':common_ground_after.decode(errors='replace'),
        'inquiry_allocation_loaded':allocation_loaded_question.decode(errors='replace'),
        'inquiry_allocation_recovered':allocation_recovered_question.decode(errors='replace'),
        'testimony_revision_before':testimony_before.decode(errors='replace'),
        'open_field_question':open_question.decode(errors='replace'),
        'testimony_revision_question':testimony_question.decode(errors='replace'),
        'testimony_revision_second_question':second_question.decode(errors='replace'),
        'testimony_revision_after':testimony_after.decode(errors='replace'),
        'endogenous_inquiry':action_question.decode(errors='replace'),
        'loaded':loaded_reply.decode(errors='replace'),
        'recovered':recovered.decode(errors='replace')}
    result={
        'schema':'cyber-lagoon.life-function-factory.v2',
        'contract':'FOUNDRY_LIFE_FUNCTION_FACTORY_'+('GREEN' if not failed else 'RED'),
        'pass':not failed,'reference_only':True,'one_birth':True,'one_training_chain':True,
        'events_lived_once':len(curriculum.events),'marks':marks,
        'species_root':species.root(),'curriculum_root':curriculum.root(),
        'source_semantics_root':source_semantics_root_v2(),
        'checkpoint_bytes':checkpoint_bytes,
        'checkpoint_bytes_per_event':round(checkpoint_bytes/len(curriculum.events),3),
        'tail_checkpoint_growth_bytes':tail_checkpoint_growth,
        'tail_document_bytes':tail_document_bytes,
        'checkpoint_leaves':checkpoint_leaves,'pre_tail_checkpoint_leaves':pre_tail_leaves,
        'checkpoint_bytes_per_leaf':round(checkpoint_bytes/checkpoint_leaves,3),
        'pre_tail_checkpoint_bytes_per_leaf':round(pre_tail_bytes/pre_tail_leaves,3),
        'event_work_reduction':round(naive_event_work/len(curriculum.events),3),
        'language_phenotype_improved':language_phenotype_improved,
        'capability_frontier':capability_frontier,'emergence_ledger':emergence,
        'visible_language_gain':visible_language_gain,
        'structural_probe':{
            'heldout':learned_heldout.decode(errors='replace'),
            'operator_clauses':capability_frontier['operator_clauses'],
            'self_organized':bytes(organized or ()).decode(errors='replace'),
            'causal_message':effect_reply.decode(errors='replace'),
            'post_life':post_life_surface.decode(errors='replace')},
        'checks':checks,'failed':failed,'elapsed_ms':round(elapsed,3),
        'remaining_red':['DIRECT_CURRICULUM_PARITY','GRAPH_PROMOTION',
                         'THREE_OR_MORE_SIMULTANEOUS_SOURCE_CAUSAL_CLAIMS',
                         'NOVEL_CONCEPT_RELATION_TOPOLOGY_TRANSFER',
                         'MULTIMODAL_COMMON_CAUSE_IN_CANONICAL_LIFE',
                         'NATURAL_DIALOGUE_SCHEDULER','BROAD_METACOGNITIVE_GENERALITY',
                         'SILENT_RELATION_SUPPORT_PATH_GROWTH']}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
