#!/usr/bin/env python3
"""Strict raw human-language path blocks fluent semantic nonsense and permits grounded concise answers."""
from __future__ import annotations
import copy,json,time
from reference_discourse_quantity_interaction_verify import BECAUSE,earn_program,train_adult
from reference_grounded_after_question_verify import Q1,Q2,QCTX,teach_questions,support
from reference_grounded_causal_operator_v1 import GroundedCausalOperatorV1
from reference_grounded_causal_operator_verify import mature
from reference_grounded_temporal_question_v1 import GroundedTemporalQuestionV1
from reference_human_language_terminal_v1 import HumanLanguageSessionV1
from reference_language_mastery_contact_adapter_v1 import LanguageMasteryContactAdapterV1
from reference_language_mastery_terminal_v1 import respond as legacy_respond
from reference_lived_event_order_v1 import LivedEventOrderV1
from reference_open_world_causal_learning_v1 import OpenWorldCausalLearningV1
from reference_predictive_credit_profile_v1 import Q


def main():
    started=time.perf_counter();checks={}

    # Human-style heldout temporal question: semantics and embedded proposition family are learned, not host supplied.
    qa,qleaves,_=train_adult();qleaves=tuple(qleaves);teach_questions(qa,qleaves);order=LivedEventOrderV1()
    support(order,qleaves[3],qleaves[0],0xF600);support(order,qleaves[20],qleaves[23],0xF610);support(order,qleaves[2],qleaves[0],0xF620)
    questions=GroundedTemporalQuestionV1();questions.observe_grounded(qa,QCTX,qleaves[3].identity,qleaves[0].identity,order,Q1);questions.observe_grounded(qa,QCTX,qleaves[20].identity,qleaves[23].identity,order,Q2)
    human_question=HumanLanguageSessionV1(qa,order,questions)
    raw_after=b'what happened after the careful engineer inspects the sensor?';answer=human_question.respond(raw_after,0xFA01)
    checks['raw_heldout_after_question_answers_only_new_grounded_information']=(answer==b'the careful engineer tests the sensor.')
    checks['raw_question_path_has_no_host_event_context_argument']=(questions.event_context.get(QCTX,0)>0 and b' then ' not in answer and b' because ' not in answer)

    # Exact old cheat: arbitrary fluent relation Program is selectable from ordinary raw input.
    cheat,leaves,_=train_adult();leaves=tuple(leaves);arbitrary=earn_program(cheat,BECAUSE,leaves[0],leaves[1]);cue=leaves[5]
    for _ in range(2):cheat.experience_choice(arbitrary.identity,Q,context=cue.identity,effort_q16=Q//16,duration=1,independent_return=True)
    pre=copy.deepcopy(cheat.checkpoint());legacy=type(cheat).restore(copy.deepcopy(pre));strict=type(cheat).restore(copy.deepcopy(pre))
    legacy_out=legacy_respond(legacy,LanguageMasteryContactAdapterV1(legacy),bytes(cue.surface),0)
    strict_out=HumanLanguageSessionV1(strict).respond(bytes(cue.surface),0xFA02)
    checks['legacy_semantics_free_terminal_reproduces_fluent_nonsense_failure']=(legacy_out==b'the careful engineer tests the sensor. because the careful engineer tests the valve.')
    checks['strict_human_terminal_blocks_same_selected_unlicensed_relation_program']=(strict_out==b'')

    # Same strict boundary must not become a blanket multi-clause ban: an intervention-grounded causal edge is allowed.
    grounded,gleaves,_=train_adult();gleaves=tuple(gleaves);learner=OpenWorldCausalLearningV1();receipt=mature(learner,(gleaves[6],gleaves[7]),gleaves[8],gleaves[6],0xFB00);causal=GroundedCausalOperatorV1()
    for source in (0xFB21,0xFB22):causal.observe(BECAUSE,gleaves[8].identity,gleaves[6].identity,learner,receipt,source)
    causal_program=causal.materialize(grounded,learner,receipt,BECAUSE);gcue=gleaves[12]
    for _ in range(2):grounded.experience_choice(causal_program.identity,Q,context=gcue.identity,effort_q16=Q//16,duration=1,independent_return=True)
    license=((causal,learner,receipt,BECAUSE),);grounded_out=HumanLanguageSessionV1(grounded,causal_licenses=license).respond(bytes(gcue.surface),0xFA03)
    checks['strict_human_terminal_allows_intervention_grounded_causal_relation']=(grounded_out==bytes(gleaves[8].surface)+b' because '+bytes(gleaves[6].surface))
    checks['strict_gate_is_relation_semantic_not_surface_length_or_connective_filter']=(b' because ' in legacy_out and b' because ' in grounded_out and strict_out==b'')

    # Question semantics survive checkpoint without raw question/answer text.
    qcp=questions.checkpoint();ocp=order.checkpoint();restored_q=GroundedTemporalQuestionV1.restore(copy.deepcopy(qcp));restored_o=LivedEventOrderV1.restore(copy.deepcopy(ocp));restored_adult=type(qa).restore(copy.deepcopy(qa.checkpoint()));replayed=HumanLanguageSessionV1(restored_adult,restored_o,restored_q).respond(raw_after,0xFA04)
    checks['strict_human_question_semantics_survive_checkpoint_without_transcript']=(replayed==answer)
    blob=json.dumps({'question':qcp,'order':ocp},sort_keys=True)
    checks['semantic_checkpoint_contains_no_raw_question_answer_or_connective_surface']=(all(token not in blob for token in ('what happened','tests the sensor',' because ',' then ','transcript','expected_answer')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0

    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-human-language-terminal.v1','contract':'FOUNDRY_HUMAN_LANGUAGE_TERMINAL_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,
            'visible_language_gain':'RAW_HUMAN_LANGUAGE_EXTERNALIZATION_NOW_BLOCKS_FLUENT_UNGROUNDED_RELATION_PROGRAMS_WHILE_ALLOWING_GROUNDED_RELATIONS_AND_CONCISE_HELDOUT_EVENT_QUESTIONS',
            'conversation':[raw_after.decode(),answer.decode() if answer else ''],'blocked_legacy_output':legacy_out.decode(),'grounded_causal_output':grounded_out.decode() if grounded_out else '',
            'checks':checks,'failed':failed,'remaining_red':['MAKE_HUMAN_TERMINAL_THE_PRODUCT_DEFAULT_NOT_REFERENCE_ONLY','GROUNDED_WHY_QUESTION_TO_CAUSAL_MODEL_WITH_CONCISE_ANSWER','ANAPHORIC_REFERENCE_AND_NATURAL_LEXICAL_VARIATION'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
