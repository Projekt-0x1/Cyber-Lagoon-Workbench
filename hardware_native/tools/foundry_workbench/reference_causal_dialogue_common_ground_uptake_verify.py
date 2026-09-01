#!/usr/bin/env python3
"""Partner-grounded discourse uptake on the one canonical Life-grown Adult."""
from __future__ import annotations
import copy,hashlib,json,sys,tempfile,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from life_function_factory_v1 import build_cache,load_mark
from reference_life_extension_endogenous_state_inquiry_v1 import (
    CAUSAL_INQUIRY_SOURCES,TESTIMONY_ALARM,TESTIMONY_MOTOR_SOURCE,
    TESTIMONY_PARTNER,TESTIMONY_WORLD_SOURCE,
)
from reference_life_extension_causal_depth_plus_v1 import A_ROOTS,A_STOMATA
from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2,
    canonical_life_function_curriculum_v2,canonical_species_program_v2,
)

Q=1<<16
PARTNER=0xEE01
OTHER_PARTNER=0xEE02
UNRELATED_PARTNER=0xFE02
MOTOR_SOURCE=0xEE11
WORLD_SOURCE=0xEE21
HEATER=0xA106
ALARM=TESTIMONY_ALARM
CONTINUATION_CONTACT=b"Please continue: What else happens because of that?"
HORIZON_CHANNEL=0xEE31


def _causal_horizon(runtime,root_identity,limit=5):
    """Traverse successive resident-local boundaries; retain no host discourse plan."""
    rows=[];surface,receipt=runtime.adult.externalize_causal_component(
        root_identity,HORIZON_CHANNEL,HORIZON_CHANNEL)
    for index in range(limit):
        if receipt is None:break
        coordinates=runtime.adult._causal_action_coordinates(receipt)
        row={'surface':bytes(surface),'programs':tuple(map(int,receipt.programs)),
             'causal_receipts':tuple(sorted(int(item[0]) for item in coordinates)),
             'coordinates':tuple(tuple(map(int,item[1:])) for item in coordinates),
             'settled':False}
        rows.append(row)
        if index+1>=limit:break
        row['settled']=bool(runtime.settle_contact_consequence(
            receipt.identity,HORIZON_CHANNEL,0,0,True))
        if not row['settled']:break
        runtime=ReferenceLifeFunctionRuntimeV2.restore(
            runtime.program,copy.deepcopy(runtime.checkpoint()))
        runtime.adult.observe_authenticated_causal_dialogue_contact(
            CONTINUATION_CONTACT,HORIZON_CHANNEL,HORIZON_CHANNEL)
        surface,identity=runtime.contact_utterance(
            CONTINUATION_CONTACT,HORIZON_CHANNEL,HORIZON_CHANNEL)
        receipt=runtime.adult.pending_causal_dialogue_actions.get(identity)
    return runtime,tuple(rows)


def main():
    started=time.perf_counter();checks={}
    with tempfile.TemporaryDirectory(prefix='foundry-common-ground-') as directory:
        manifest=build_cache(directory);adult=load_mark(directory,'relational_surplus_recovered').adult
        leaf=adult.language_adult.leaf(100,(0xA104,));world_before=copy.deepcopy(adult.language_adult.world_causal_learning.checkpoint());language_before=copy.deepcopy(adult.language.checkpoint())

        before,receipt=adult.externalize_causal_component(leaf.identity,MOTOR_SOURCE,PARTNER)
        checks['resident_public_action_precedes_partner_uptake']=bool(before and receipt and receipt.channel==PARTNER and receipt.source==MOTOR_SOURCE)
        checks['independent_world_return_closes_public_action_without_partner_semantics']=bool(receipt and adult.settle_causal_dialogue_return(receipt,WORLD_SOURCE,Q,0,True))

        rows=adult.causal_chain_rows(leaf.identity);first_row=rows[0];first_program=receipt.programs[0] if receipt else 0;first_factor=adult.language_adult.programs.factor(first_program);ack=adult._causal_self_contained_surface(first_row,first_factor);ack=bytes(ack or b'')
        orientation=adult.language_adult.world_causal_learning.grounding.orientation(first_factor);cause,effect=int(first_row[2]),int(first_row[3]);acted_children=((effect,cause) if orientation>0 else (cause,effect))
        checks['acceptance_surface_is_rematerialized_from_the_acted_certified_relation']=bool(ack and ack in before and adult.language_adult.world_causal_learning.language_relation_certificate(adult.language_adult,first_factor,*acted_children)[:1]==(1,))

        alternatives=tuple(factor for factor in adult._causal_self_contained_factors() if factor!=first_factor);paraphrase_factor=alternatives[0] if len(alternatives)==1 else 0;paraphrase=bytes(adult._causal_self_contained_surface(first_row,paraphrase_factor) or b'') if paraphrase_factor else b''
        paraphrase_orientation=adult.language_adult.world_causal_learning.grounding.orientation(paraphrase_factor) if paraphrase_factor else 0;paraphrase_children=((effect,cause) if paraphrase_orientation>0 else (cause,effect));pieces=adult.language.historical_span_pieces(paraphrase_factor) if paraphrase_factor else ()
        reversed_children=tuple(reversed(paraphrase_children));reversal=bytes(adult.language_adult._render_pieces(pieces,tuple(tuple(adult.language_adult._leaf_surface(child)) for child in reversed_children))) if pieces else b''
        developmental_source=Path(__file__).with_name('reference_life_extension_causal_discourse_forms_v2.py').read_bytes()
        checks['independently_learned_construction_recomposes_heldout_relation_paraphrase']=(bool(paraphrase and paraphrase!=ack and paraphrase not in before and paraphrase not in developmental_source) and adult.language_adult.world_causal_learning.language_relation_certificate(adult.language_adult,paraphrase_factor,*paraphrase_children)[:1]==(1,))

        own=adult.observe_authenticated_causal_dialogue_contact(ack,MOTOR_SOURCE)
        wrong=adult.observe_authenticated_causal_dialogue_contact(ack,UNRELATED_PARTNER)
        nonsense_surface=b'this contact has no earned causal relation';nonsense=adult.observe_authenticated_causal_dialogue_contact(nonsense_surface,PARTNER)
        checks['own_output_wrong_partner_and_ungrounded_contact_cannot_author_uptake']=(own==wrong==nonsense==0)

        one=adult.observe_authenticated_causal_dialogue_contact(ack,PARTNER);after_one,after_one_programs=adult.compose_causal_component(leaf.identity,channel=PARTNER)
        one_cp=copy.deepcopy(adult.checkpoint());one_restart=type(adult).restore(copy.deepcopy(one_cp))
        checks['one_acceptance_is_insufficient_and_checkpoint_exact']=(one==1 and ack in after_one and len(after_one_programs)==len(receipt.programs) and after_one.count(b'.')==before.count(b'.') and one_restart.checkpoint()==one_cp and one_restart.compose_causal_component(leaf.identity,channel=PARTNER)[0]==after_one)

        reversal_subject=type(adult).restore(copy.deepcopy(one_cp));reversal_first=reversal_subject.observe_authenticated_causal_dialogue_contact(reversal,PARTNER);reversal_second=reversal_subject.observe_authenticated_causal_dialogue_contact(reversal,PARTNER);reversal_surface=reversal_subject.compose_causal_component(leaf.identity,channel=PARTNER)[0]
        checks['same_words_and_construction_with_reversed_causality_cannot_fake_uptake']=(reversal and reversal_first==reversal_second==0 and ack in reversal_surface and reversal_subject.causal_dialogue_uptake_support(PARTNER,int(first_row[4]))==1)

        dispute_cp=copy.deepcopy(reversal_subject.checkpoint());repair_method=getattr(reversal_subject,'endogenous_causal_repair_inquiry',None);repair_question=(repair_method(PARTNER) if repair_method is not None else None);repair_question=bytes(repair_question or b'')
        question_surface,question_receipt=reversal_subject.externalize_endogenous_inquiry(MOTOR_SOURCE+2,PARTNER);question_surface=bytes(question_surface or b'')
        question_motor=bool(question_receipt and reversal_subject.settle_endogenous_inquiry_motor_return(question_receipt,PARTNER,True))
        try:question_bindings=reversal_subject.language.invert_surface(question_surface,max_candidates=8) if question_surface else ()
        except ValueError:question_bindings=()
        cause_leaf=reversal_subject.language_adult.current_leaf_for_historical(cause);effect_leaf=reversal_subject.language_adult.current_leaf_for_historical(effect);cause_signature=reversal_subject.language_adult.leaf_signature(cause);effect_signature=reversal_subject.language_adult.leaf_signature(effect);question_atoms=(tuple((*cause_signature[1],*effect_signature[1])) if cause_signature is not None and effect_signature is not None else ())
        inquiry_source=Path(__file__).with_name('reference_life_extension_endogenous_state_inquiry_v1.py').read_bytes()
        checks['causal_role_conflict_originates_heldout_specific_repair_question']=(bool(repair_question and question_surface==repair_question and question_receipt and question_motor and question_surface.endswith(b'?') and bytes(cause_leaf.surface) in question_surface and bytes(effect_leaf.surface) in question_surface and question_surface not in inquiry_source and any(tuple(binding.atoms)==question_atoms for binding in question_bindings)))
        checks['dispute_state_and_inquiry_work_are_bounded_to_relevant_relations']=(len(reversal_subject._causal_dialogue_dispute_evidence)==1 and reversal_subject.causal_dialogue_dispute_support(PARTNER,int(first_row[4]))==2 and 0<reversal_subject.last_causal_repair_inquiry_touches<=len(reversal_subject._context_affordance_sources))
        checks['pending_repair_question_is_checkpointed_without_transcript_and_inhibits_repoll']=(bool(question_receipt) and reversal_subject.externalize_endogenous_inquiry(MOTOR_SOURCE+3,PARTNER)==(b'',None) and question_surface.decode(errors='replace') not in json.dumps(reversal_subject.checkpoint(),sort_keys=True) and type(adult).restore(copy.deepcopy(reversal_subject.checkpoint())).externalize_endogenous_inquiry(MOTOR_SOURCE+3,PARTNER)==(b'',None))

        wrong_answer=type(adult).restore(copy.deepcopy(reversal_subject.checkpoint()));wrong_changed=wrong_answer.observe_authenticated_causal_dialogue_contact(reversal,PARTNER);wrong_closed=wrong_answer.settle_endogenous_inquiry_resolution(question_receipt,PARTNER);wrong_question,wrong_receipt=wrong_answer.externalize_endogenous_inquiry(MOTOR_SOURCE+4,PARTNER)
        checks['reasserted_reversal_cannot_close_or_duplicate_same_information_need']=(wrong_changed==0 and not wrong_closed and not wrong_question and wrong_receipt is None)

        repaired=type(adult).restore(copy.deepcopy(reversal_subject.checkpoint()));answer_changed=repaired.observe_authenticated_causal_dialogue_contact(paraphrase,PARTNER);answer_closed=repaired.settle_endogenous_inquiry_resolution(question_receipt,PARTNER);post_answer_question,post_answer_receipt=repaired.externalize_endogenous_inquiry(MOTOR_SOURCE+5,PARTNER);post_answer_surface,post_answer_programs=repaired.compose_causal_component(leaf.identity,channel=PARTNER)
        checks['structurally_valid_answer_resolves_dispute_and_changes_later_partner_plan']=(answer_changed==1 and answer_closed is True and post_answer_question==b'' and post_answer_receipt is None and ack not in post_answer_surface and len(post_answer_programs)==len(receipt.programs)-1)

        lesioned=type(adult).restore(copy.deepcopy(dispute_cp))
        for source in CAUSAL_INQUIRY_SOURCES:lesioned.language.withdraw_source(source)
        lesioned_question,lesioned_receipt=lesioned.externalize_endogenous_inquiry(MOTOR_SOURCE+6,PARTNER)
        for source in CAUSAL_INQUIRY_SOURCES:lesioned.language.restore_source(source)
        restored_question,restored_receipt=lesioned.externalize_endogenous_inquiry(MOTOR_SOURCE+7,PARTNER)
        checks['question_construction_lesion_silences_action_without_erasing_dispute_then_restores']=(lesioned_question==b'' and lesioned_receipt is None and bytes(restored_question)==question_surface and restored_receipt is not None)

        two=one_restart.observe_authenticated_causal_dialogue_contact(paraphrase,PARTNER);focused,focused_programs=one_restart.compose_causal_component(leaf.identity,channel=PARTNER)
        checks['cross_construction_partner_acceptance_focuses_next_turn']=(two==1 and focused and focused!=before and ack not in focused and paraphrase not in focused and len(focused_programs)==len(receipt.programs)-1)

        other=type(adult).restore(copy.deepcopy(adult.checkpoint()));other_before,other_receipt=other.externalize_causal_component(leaf.identity,MOTOR_SOURCE+1,OTHER_PARTNER)
        other_after_stage,other_after_programs=other.compose_causal_component(leaf.identity,channel=OTHER_PARTNER)
        checks['other_partner_retains_accepted_relation']=other_receipt is not None and ack in other_before
        checks['other_partner_retains_full_causal_plan']=other_receipt is not None and len(other_receipt.programs)==len(receipt.programs) and other_before.count(b'.')==before.count(b'.')
        checks['other_partner_plan_is_stable_after_staging']=other_receipt is not None and tuple(other_after_programs)==tuple(other_receipt.programs) and ack in other_after_stage

        language_after=one_restart.language.checkpoint();before_lexemes={json.dumps(row,sort_keys=True) for row in language_before['lexemes']};new_lexemes=tuple(row for row in language_after['lexemes'] if json.dumps(row,sort_keys=True) not in before_lexemes);before_without_lexemes={key:value for key,value in language_before.items() if key!='lexemes'};after_without_lexemes={key:value for key,value in language_after.items() if key!='lexemes'}
        checks['partner_acceptance_changes_neither_world_truth_nor_existing_language_competence']=(one_restart.language_adult.world_causal_learning.checkpoint()==world_before and before_without_lexemes==after_without_lexemes and before_lexemes.issubset({json.dumps(row,sort_keys=True) for row in language_after['lexemes']}) and len(new_lexemes)==1 and tuple(new_lexemes[0]['units'])==tuple(nonsense_surface) and new_lexemes[0]['sources']==[PARTNER])
        provisional=type(adult).restore(copy.deepcopy(one_restart.checkpoint()));second_nonsense=provisional.observe_authenticated_causal_dialogue_contact(nonsense_surface,OTHER_PARTNER,channel=PARTNER);second_continuation=provisional.last_causal_dialogue_contact_continuations;third_nonsense=provisional.observe_authenticated_causal_dialogue_contact(nonsense_surface,UNRELATED_PARTNER,channel=PARTNER);third_continuation=provisional.last_causal_dialogue_contact_continuations
        checks['one_source_novel_post_action_surface_is_provisional_not_semantic_authority']=(second_nonsense==third_nonsense==0 and not second_continuation and len(third_continuation)==1)
        focused_cp=copy.deepcopy(one_restart.checkpoint());focused_restart=type(adult).restore(copy.deepcopy(focused_cp));checkpoint_text=json.dumps(focused_cp,sort_keys=True);checks['focused_common_ground_survives_restart_without_transcript']=(focused_restart.checkpoint()==focused_cp and focused_restart.compose_causal_component(leaf.identity,channel=PARTNER)[0]==focused and ack.decode(errors='replace') not in checkpoint_text and paraphrase.decode(errors='replace') not in checkpoint_text)

        one_restart.language.withdraw_source(PARTNER);withdrawn,withdrawn_programs=one_restart.compose_causal_component(leaf.identity,channel=PARTNER)
        one_restart.language.restore_source(PARTNER);restored,restored_programs=one_restart.compose_causal_component(leaf.identity,channel=PARTNER)
        checks['source_withdrawal_reopens_and_restoration_refocuses_partner_common_ground']=(ack in withdrawn and len(withdrawn_programs)==len(receipt.programs) and ack not in restored and len(restored_programs)==len(receipt.programs)-1)
        checks['uptake_work_and_state_are_bounded']=(len(one_restart._causal_dialogue_uptake_evidence)==1 and one_restart.causal_dialogue_uptake_support(PARTNER,int(first_row[4]))==2)

        # The canonical one-Life chain now owns this continuation.  These mark
        # contrasts inspect the same Adult before claim, during unresolved
        # intervention, and after independently settled revision.
        curriculum=canonical_life_function_curriculum_v2();species=canonical_species_program_v2()
        old_world=load_mark(directory,'testimony_revision_old_world');old_adult=old_world.adult;revision_root=old_adult.language_adult.leaf(100,(0xA104,));alarm_leaf=old_adult.language_adult.leaf(100,(ALARM,))
        revision_before,revision_action=old_adult.externalize_causal_component(alarm_leaf.identity,MOTOR_SOURCE+20,PARTNER);old_alarm_rows=tuple(row for row in old_adult.language_adult.world_causal_learning.current_resolutions() if int(row[3])==alarm_leaf.identity);old_alarm_receipt=int(old_alarm_rows[0][4]) if len(old_alarm_rows)==1 else 0
        committed=load_mark(directory,'testimony_revision_committed');committed_actions=tuple(row for row in committed.adult.recent_causal_dialogue_actions.values() if int(row.channel)==PARTNER and any(b'the irrigation alarm sounds' in bytes(committed.adult.language_adult.public_surface(pid) or b'') for pid in row.programs))
        checks['one_life_publicly_commits_old_relation_before_new_testimony']=(bool(revision_before and revision_action and len(revision_action.programs)==1 and len(old_alarm_rows)==1 and int(old_alarm_rows[0][2])==old_adult.language_adult.leaf(100,(HEATER,)).identity and len(committed_actions)==1 and b'the irrigation alarm sounds' in revision_before and b'the heater warms the greenhouse' in revision_before))

        need=load_mark(directory,'testimony_revision_need');need_adult=need.adult;need_learner=need_adult.language_adult.world_causal_learning;partner_source=need_learner.testimony_source(PARTNER);need_alarm=need_adult.language_adult.leaf(100,(ALARM,));need_growth=need_adult.language_adult.leaf(100,(0xA10B,));testimony_rows=tuple(row for row in need_learner.current_testimony_resolutions() if int(row[1])==need_alarm.identity);growth_testimony_rows=tuple(row for row in need_learner.current_testimony_resolutions() if int(row[1])==need_growth.identity);adverse_claim=b'Because plant stomata close, the irrigation alarm sounds.'
        causal_candidates=need_adult._causal_repair_inquiry_candidates(PARTNER);ordered_candidates=tuple(sorted(causal_candidates,key=lambda row:(-int(row[1]),int(row[2]),int(row[3]))));candidate_bids={(int(row[2]),int(row[3])):(bytes(row[0]),int(row[1])) for row in ordered_candidates};winning_candidate=(ordered_candidates[0] if ordered_candidates else None);deferred_candidate=(ordered_candidates[1] if len(ordered_candidates)>1 else None)
        alarm_action=next((row for row in need_adult.recent_causal_dialogue_actions.values()
                           if any(int(coordinate[2])==int(need_alarm.identity) for coordinate in need_adult._causal_action_coordinates(row))),None)
        efficacy_reversal=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(need.checkpoint()))
        if alarm_action is not None:
            for offset in range(4):efficacy_reversal.adult.observe_causal_dialogue_background(alarm_action.identity,0xF900+offset,True)
        reversed_question=bytes(efficacy_reversal.adult.endogenous_causal_repair_inquiry(PARTNER) or b'')
        allocation=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(need.checkpoint()))
        for offset in range(6):
            sequence=0xF100+offset;digest=hashlib.sha256(str(sequence).encode()).hexdigest()
            allocation.adult.language_adult.settle_body_ingress('inquiry-allocation-probe',sequence,digest,Q//2)
        loaded_state_need=bytes(allocation.adult.endogenous_state_inquiry(PARTNER) or b'')
        loaded_causal_need=bytes(allocation.adult.endogenous_causal_repair_inquiry(PARTNER) or b'')
        allocation_checkpoint=copy.deepcopy(allocation.checkpoint());allocation_restart=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(allocation_checkpoint))
        repeated=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(allocation_checkpoint));support_before=sum(repeated.adult._causal_dialogue_dispute_evidence.values());bids_before={bytes(row[0]):int(row[1]) for row in repeated.adult._causal_repair_inquiry_candidates(PARTNER)};repeated.adult.observe_authenticated_causal_dialogue_contact(adverse_claim,PARTNER);repeated.adult.observe_authenticated_causal_dialogue_contact(adverse_claim,PARTNER);support_after=sum(repeated.adult._causal_dialogue_dispute_evidence.values());bids_after={bytes(row[0]):int(row[1]) for row in repeated.adult._causal_repair_inquiry_candidates(PARTNER)};repeated_priority,_=repeated.adult.externalize_endogenous_inquiry(MOTOR_SOURCE+26,PARTNER)
        state_lesion=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(allocation_checkpoint))
        for source in (0xDF30,0xDF31):state_lesion.adult.language.withdraw_source(source)
        lesioned_priority,_=state_lesion.adult.externalize_endogenous_inquiry(MOTOR_SOURCE+27,PARTNER)
        loaded_priority,loaded_priority_receipt=allocation.adult.externalize_endogenous_inquiry(MOTOR_SOURCE+24,PARTNER);loaded_priority=bytes(loaded_priority or b'')
        checks['same_adult_holds_distinct_somatic_and_source_causal_information_needs']=(bool(loaded_state_need and loaded_causal_need and loaded_state_need!=loaded_causal_need and allocation_restart.checkpoint()==allocation_checkpoint))
        checks['acute_resource_pressure_prioritizes_body_control_inquiry_without_erasing_causal_need']=(bool(loaded_priority_receipt) and loaded_priority==loaded_state_need and allocation.adult.endogenous_causal_repair_inquiry(PARTNER)==loaded_causal_need)
        checks['repeated_source_volume_cannot_outbid_resource_grounded_need']=(support_after>support_before and bytes(repeated_priority or b'')==loaded_state_need)
        checks['two_simultaneous_causal_disputes_compete_as_distinct_resident_questions']=(len(candidate_bids)==len(ordered_candidates)==2 and winning_candidate is not None and deferred_candidate is not None and int(winning_candidate[1])>int(deferred_candidate[1]) and bytes(winning_candidate[0])!=bytes(deferred_candidate[0]) and loaded_causal_need==bytes(winning_candidate[0]))
        checks['testimony_repetition_cannot_change_cost_or_control_bid']=(support_after>support_before and bids_after==bids_before)
        checks['learned_uncontrollability_reverses_question_order_without_new_claim_content']=(alarm_action is not None and deferred_candidate is not None and reversed_question==bytes(deferred_candidate[0]) and efficacy_reversal.adult._causal_dialogue_dispute_evidence==need_adult._causal_dialogue_dispute_evidence)
        checks['equal_actual_question_bids_produce_voluntary_silence']=(winning_candidate is not None and deferred_candidate is not None and need_adult._unique_inquiry_bid(((bytes(winning_candidate[0]),1,int(winning_candidate[2]),int(winning_candidate[3])),(bytes(deferred_candidate[0]),1,int(deferred_candidate[2]),int(deferred_candidate[3])))) is None)
        checks['focal_somatic_construction_lesion_exposes_deferred_causal_competitor']=(state_lesion.adult.endogenous_state_inquiry(PARTNER) is None and bytes(lesioned_priority or b'')==loaded_causal_need)
        loaded_motor=bool(loaded_priority_receipt and allocation.adult.settle_endogenous_inquiry_motor_return(loaded_priority_receipt,PARTNER,True));loaded_unrelated=bool(loaded_priority_receipt and allocation.adult.settle_endogenous_inquiry_resolution(loaded_priority_receipt,WORLD_SOURCE+24))
        for _ in range(64):allocation.adult.language_adult.internal_tick();allocation.adult.resident_silent_wave()
        loaded_closed=bool(loaded_priority_receipt and allocation.adult.settle_endogenous_inquiry_resolution(loaded_priority_receipt,WORLD_SOURCE+24))
        recovered_state_need=bytes(allocation.adult.endogenous_state_inquiry(PARTNER) or b'');recovered_causal_need=bytes(allocation.adult.endogenous_causal_repair_inquiry(PARTNER) or b'')
        recovered_priority,recovered_priority_receipt=allocation.adult.externalize_endogenous_inquiry(MOTOR_SOURCE+25,PARTNER);recovered_priority=bytes(recovered_priority or b'')
        checks['quiet_recovery_reverses_priority_to_deferred_causal_inquiry']=(loaded_motor and not loaded_unrelated and loaded_closed and not recovered_state_need and recovered_causal_need==loaded_causal_need and bool(recovered_priority_receipt) and recovered_priority==loaded_causal_need)
        allocation_text=json.dumps(allocation.checkpoint(),sort_keys=True)
        checks['cross_network_allocation_persists_need_not_question_surfaces']=(loaded_state_need.decode(errors='replace') not in allocation_text and loaded_causal_need.decode(errors='replace') not in allocation_text and allocation.adult.causal_dialogue_dispute_support(PARTNER,old_alarm_receipt)>0)
        question_probe=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(need.checkpoint()));revision_question,revision_question_receipt=question_probe.adult.externalize_endogenous_inquiry(TESTIMONY_MOTOR_SOURCE,TESTIMONY_PARTNER);revision_question=bytes(revision_question or b'')
        conflict=load_mark(directory,'testimony_revision_conflict');conflict_cp=copy.deepcopy(conflict.checkpoint());conflict_text=json.dumps(conflict_cp,sort_keys=True);lived_surfaces=b' '.join(bytes(event.payload) for event in curriculum.events if event.lane in {'surface','discourse_surface','utterance','authenticated_utterance'})
        checks['source_qualified_alternatives_remain_hypotheses_not_world_truth']=(len(testimony_rows)==len(growth_testimony_rows)==1 and int(testimony_rows[0][0])==need_adult.language_adult.leaf(100,(A_STOMATA,)).identity and int(growth_testimony_rows[0][0])==need_adult.language_adult.leaf(100,(A_ROOTS,)).identity and not any(int(row[3]) in (need_alarm.identity,need_growth.identity) for row in need_learner.current_resolutions()) and need_learner.testimony_accuracy.get(partner_source)==(1,0))
        checks['competing_cause_originates_heldout_specific_question_inside_full_discourse']=(bool(winning_candidate and revision_question_receipt) and revision_question==bytes(winning_candidate[0]) and (revision_question_receipt.obligation_candidates,revision_question_receipt.obligation_effect)==((int(winning_candidate[2]),),int(winning_candidate[3])) and revision_question not in adverse_claim and revision_question not in lived_surfaces and question_probe.adult.causal_dialogue_dispute_support(PARTNER,old_alarm_receipt)>0)
        checks['candidate_cause_and_pending_action_survive_without_transcript']=(revision_question.decode(errors='replace') not in conflict_text and adverse_claim.decode(errors='replace') not in conflict_text and ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(conflict_cp)).checkpoint()==conflict_cp)

        yoked=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(conflict_cp));pending=next(iter(yoked.adult.pending_endogenous_inquiry_actions.values()),None);yoked_closed=bool(pending and yoked.adult.settle_endogenous_inquiry_resolution(pending,WORLD_SOURCE+22));yoked_question,_=yoked.adult.externalize_endogenous_inquiry(MOTOR_SOURCE+22,PARTNER)
        checks['contact_without_discriminating_consequence_keeps_one_pending_question']=(not yoked_closed and not yoked_question and pending is not None)

        lesioned=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(conflict_cp));lesioned_learner=lesioned.adult.language_adult.world_causal_learning;lesioned_alarm=lesioned.adult.language_adult.leaf(100,(ALARM,));lesioned_learner.pending_testimony_guidance.clear()
        for key in tuple(lesioned_learner.testimony_claims):
            if int(key[0])==int(partner_source):lesioned_learner.testimony_claims.pop(key,None)
        lesioned.run(curriculum.prefix_at_mark('testimony_revision_settled'));lesion_rows=tuple(row for row in lesioned_learner.current_resolutions() if int(row[3])==lesioned_alarm.identity)

        second_conflict=load_mark(directory,'testimony_revision_second_conflict');second_pending=tuple(second_conflict.adult.pending_endogenous_inquiry_actions.values());second_probe=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(second_conflict.checkpoint()));second_probe.adult.pending_endogenous_inquiry_actions.clear();second_winner=second_probe.adult._unique_inquiry_bid(second_probe.adult._causal_inquiry_candidates(PARTNER));second_question=(bytes(second_winner[0]) if second_winner else b'')
        settled=load_mark(directory,'testimony_revision_settled');settled_adult=settled.adult;settled_learner=settled_adult.language_adult.world_causal_learning;settled_alarm=settled_adult.language_adult.leaf(100,(ALARM,));settled_growth=settled_adult.language_adult.leaf(100,(0xA10B,));settled_root=settled_adult.language_adult.leaf(100,(0xA104,));settled_rows=tuple(row for row in settled_learner.current_resolutions() if int(row[3])==settled_alarm.identity);settled_growth_rows=tuple(row for row in settled_learner.current_resolutions() if int(row[3])==settled_growth.identity);settled_question,_=settled_adult.externalize_endogenous_inquiry(MOTOR_SOURCE+23,PARTNER);revision_after,revision_after_programs=settled_adult.compose_causal_component(settled_root.identity,channel=PARTNER);revision_after_alarm,_=settled_adult.compose_causal_component(settled_alarm.identity,channel=PARTNER)
        revision_cp=copy.deepcopy(settled.checkpoint());revision_restart=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(revision_cp));restart_after=revision_restart.adult.compose_causal_component(settled_root.identity,channel=PARTNER)[0]
        pre_wrapper=ReferenceLifeFunctionRuntimeV2.restore(species,copy.deepcopy(revision_cp));_pre_wrapper_runtime,pre_wrapper_horizon=_causal_horizon(pre_wrapper,pre_wrapper.adult.language_adult.leaf(100,(0xA104,)).identity)
        horizon_subject=load_mark(directory,manifest['checkpoints'][-1]['mark']);horizon_adult=horizon_subject.adult;horizon_root=horizon_adult.language_adult.leaf(100,(0xA104,));horizon_learner=horizon_adult.language_adult.world_causal_learning
        horizon_alarm=horizon_adult.language_adult.leaf(100,(ALARM,));horizon_growth=horizon_adult.language_adult.leaf(100,(0xA10B,));horizon_alarm_rows=tuple(row for row in horizon_learner.current_resolutions() if int(row[3])==horizon_alarm.identity);horizon_growth_rows=tuple(row for row in horizon_learner.current_resolutions() if int(row[3])==horizon_growth.identity)
        horizon_runtime,horizon=_causal_horizon(horizon_subject,horizon_root.identity);horizon_surfaces=tuple(row['surface'] for row in horizon);horizon_checkpoint=json.dumps(horizon_runtime.checkpoint(),sort_keys=True)
        horizon_receipts={receipt for row in horizon for receipt in row['causal_receipts']};revised_receipts=({int(horizon_alarm_rows[0][4]),int(horizon_growth_rows[0][4])} if len(horizon_alarm_rows)==len(horizon_growth_rows)==1 else set())
        integrated_horizon_rows=tuple(row for row in horizon if b'the irrigation alarm sounds' in row['surface'] and b'plant growth slows' in row['surface'] and b'plant roots lose water' in row['surface'])
        checks['first_settlement_exposes_second_specific_question']=(deferred_candidate is not None and second_winner is not None and len(second_pending)==1 and second_pending[0].surface_digest==hashlib.sha256(second_question).hexdigest() and (second_pending[0].obligation_candidates,second_pending[0].obligation_effect)==(tuple(map(int,second_winner[2])),int(second_winner[3])) and int(deferred_candidate[2]) in second_pending[0].obligation_candidates and len(second_pending[0].obligation_candidates)==2)
        revised_cause=settled_adult.language_adult.leaf(100,(A_ROOTS,)).identity
        revision_contrasts={
            'two_world_relations_revised': len(settled_rows)==len(settled_growth_rows)==1 and int(settled_rows[0][2])==int(settled_growth_rows[0][2])==revised_cause,
            'mixed_testimony_history_remains_usable': settled_learner.testimony_accuracy.get(partner_source)==(2,1) and settled_learner.testimony_reliable(partner_source),
            'both_information_needs_are_closed': not settled_adult.pending_endogenous_inquiry_actions and not settled_adult._causal_dialogue_dispute_evidence and not settled_question,
            'later_horizon_composes_both_revisions': bool(len(horizon_surfaces)>=4 and len(integrated_horizon_rows)==1 and revised_receipts and revised_receipts<=set(integrated_horizon_rows[0]['causal_receipts']) and revised_receipts<=horizon_receipts),
            'revised_alarm_excludes_refuted_cause': b'plant roots lose water' in revision_after_alarm and b'plant stomata close' not in revision_after_alarm,
            'local_boundaries_advance_without_repeating': len(horizon_surfaces)>=4 and len(set(horizon_surfaces))==len(horizon_surfaces) and all(row['coordinates'] and row['programs'] for row in horizon),
            'continuation_requires_later_development': len(pre_wrapper_horizon)==1,
            'horizon_survives_restart_without_transcript': CONTINUATION_CONTACT.decode() not in horizon_checkpoint and all(surface.decode(errors='replace') not in horizon_checkpoint for surface in horizon_surfaces),
        }
        checks['independent_consequences_revise_two_relations_source_reliability_and_later_explanation_together']=all(revision_contrasts.values())
        checks['guidance_provenance_lesion_spares_world_learning_but_preserves_old_overtrust']=(len(lesion_rows)==1 and int(lesion_rows[0][2])==lesioned.adult.language_adult.leaf(100,(A_ROOTS,)).identity and lesioned_learner.testimony_accuracy.get(partner_source)==(1,0) and lesioned_learner.testimony_reliable(partner_source))
        checks['revised_multi_turn_explanation_survives_exact_restart']=(revision_restart.checkpoint()==revision_cp and restart_after==revision_after and revision_question.decode(errors='replace') not in json.dumps(revision_cp,sort_keys=True))

    failed=[name for name,passed in checks.items() if not passed]
    result={'schema':'cyber-lagoon.causal-dialogue-common-ground-uptake.v7','contract':'FOUNDRY_CAUSAL_DIALOGUE_COMMON_GROUND_UPTAKE_'+('GREEN' if not failed else 'RED'),'pass':not failed,'checks':checks,'failed':failed,'revision_contrasts':revision_contrasts,'reference_only':True,'runtime_llm':False,'language_phenotype_improved':not failed,'visible_language_gain':{'before':before.decode(errors='replace'),'repair_question':question_surface.decode(errors='replace'),'accepted_paraphrase':paraphrase.decode(errors='replace'),'after':focused.decode(errors='replace'),'testimony_revision_before':revision_before.decode(errors='replace'),'testimony_revision_question':revision_question.decode(errors='replace'),'second_revision_question':second_question.decode(errors='replace'),'efficacy_reversal_question':reversed_question.decode(errors='replace'),'testimony_revision_after':revision_after.decode(errors='replace'),'testimony_revision_horizon':[surface.decode(errors='replace') for surface in horizon_surfaces],'loaded_somatic_priority':loaded_priority.decode(errors='replace'),'recovered_causal_priority':recovered_priority.decode(errors='replace')},'before':before.decode(errors='replace'),'acted_relation':ack.decode(errors='replace'),'accepted_paraphrase':paraphrase.decode(errors='replace'),'reversed_misunderstanding':reversal.decode(errors='replace'),'repair_question':question_surface.decode(errors='replace'),'after':focused.decode(errors='replace'),'before_relations':len(receipt.programs),'after_relations':len(focused_programs),'persistent_uptake_rows':len(one_restart._causal_dialogue_uptake_evidence),'elapsed_ms':round((time.perf_counter()-started)*1000,3),'claim':'ONE_CONTINUING_ADULT_ALLOCATES_BODY_LOAD_AND_TWO_SIMULTANEOUS_CAUSAL_DISPUTES_THEN_ADVANCES_DISTINCT_LOCAL_BOUNDARIES_UNTIL_BOTH_REVISIONS_ARE_PUBLIC','remaining_red':['SPONTANEOUS_MULTI_BOUNDARY_NARRATION_WITHOUT_FOLLOWUP_CONTACT','THREE_OR_MORE_SIMULTANEOUS_SOURCE_CAUSAL_CLAIMS','LEXICALLY_NOVEL_PARAPHRASE_GENERALIZATION','MULTIMODAL_COMMON_CAUSE_IN_CANONICAL_LIFE','DIRECT_PARITY']}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1


if __name__=='__main__':raise SystemExit(main())
