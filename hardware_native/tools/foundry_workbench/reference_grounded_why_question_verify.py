#!/usr/bin/env python3
"""Human-style N+1: heldout why-question returns only an intervention-grounded cause proposition."""
from __future__ import annotations
import copy,json,time
from reference_discourse_quantity_interaction_verify import train_adult
from reference_grounded_after_question_verify import atoms
from reference_grounded_causal_operator_verify import mature
from reference_grounded_causal_question_v1 import GroundedCausalQuestionV1
from reference_human_language_terminal_v1 import HumanLanguageSessionV1
from reference_open_world_causal_learning_v1 import OpenWorldCausalLearningV1

QCTX=0xF901;Q1=0xF911;Q2=0xF912

def teach_questions(adult,leaves):
    a1=atoms(adult,leaves[2]);a2=atoms(adult,leaves[21])
    adult.observe_surface_construction(QCTX,a1,b'why is it that the careful engineer inspects the sensor?',0xF921)
    adult.observe_surface_construction(QCTX,a2,b'why is it that the quiet technician tests the valve?',0xF922)
    return a1,a2

def main():
    started=time.perf_counter();checks={};adult,leaves,_=train_adult();leaves=tuple(leaves);teach_questions(adult,leaves)
    learner=OpenWorldCausalLearningV1()
    r1=mature(learner,(leaves[0],leaves[1]),leaves[2],leaves[0],0xFA00)
    r2=mature(learner,(leaves[23],leaves[22]),leaves[21],leaves[23],0xFB00)
    held=mature(learner,(leaves[26],leaves[27]),leaves[24],leaves[26],0xFC00)
    question=GroundedCausalQuestionV1()
    question.observe_grounded(adult,QCTX,leaves[2].identity,leaves[0].identity,learner,r1,Q1)
    checks['one_grounded_why_example_remains_provisional']=(not question.supported(QCTX))
    question.observe_grounded(adult,QCTX,leaves[21].identity,leaves[23].identity,learner,r2,Q2)
    checks['two_independent_why_examples_ground_cause_retrieval_semantics']=question.supported(QCTX)

    raw=b'why is it that the quiet operator tests the sensor?';held_atoms=atoms(adult,leaves[24]);rows=adult.language.invert_surface(tuple(raw));valid=[r for r in rows if int(r.context)==QCTX and tuple(map(int,r.atoms))==held_atoms]
    checks['heldout_natural_why_surface_productively_reconstructs_without_question_training']=(len(valid)==1)
    session=HumanLanguageSessionV1(adult,causal_questions=question,causal_models=((learner,r1),(learner,r2),(learner,held)))
    answer=session.respond(raw,0xFD01)
    checks['heldout_why_question_returns_only_intervention_grounded_cause']=(answer==bytes(leaves[26].surface))
    checks['answer_does_not_replay_effect_or_insert_because_padding']=(bytes(leaves[24].surface) not in answer and b' because ' not in answer and b' then ' not in answer)

    # Same question grammar with no heldout causal model cannot infer a cause from language shape.
    no_model=HumanLanguageSessionV1(type(adult).restore(copy.deepcopy(adult.checkpoint())),causal_questions=GroundedCausalQuestionV1.restore(copy.deepcopy(question.checkpoint())),causal_models=((learner,r1),(learner,r2)))
    checks['same_question_without_heldout_causal_evidence_refuses']=(no_model.respond(raw,0xFD02)==b'')

    # Question semantics alone are insufficient before independent language quorum.
    weak=GroundedCausalQuestionV1();weak.observe_grounded(adult,QCTX,leaves[2].identity,leaves[0].identity,learner,r1,Q1)
    checks['one_question_example_plus_full_causal_model_still_refuses']=(HumanLanguageSessionV1(type(adult).restore(copy.deepcopy(adult.checkpoint())),causal_questions=weak,causal_models=((learner,held),)).respond(raw,0xFD03)==b'')

    # Two mature models that disagree about the same effect force refusal.
    rival_learner=OpenWorldCausalLearningV1();rival=mature(rival_learner,(leaves[26],leaves[27]),leaves[24],leaves[27],0xFE00)
    ambiguous=HumanLanguageSessionV1(type(adult).restore(copy.deepcopy(adult.checkpoint())),causal_questions=GroundedCausalQuestionV1.restore(copy.deepcopy(question.checkpoint())),causal_models=((learner,held),(rival_learner,rival)))
    checks['two_intervention_models_with_different_causes_for_same_effect_refuse']=(ambiguous.respond(raw,0xFD04)==b'')

    # Asking why about the cause itself cannot simply reverse the learned edge.
    cause_raw=b'why is it that the quiet operator inspects the sensor?';cause_answer=HumanLanguageSessionV1(type(adult).restore(copy.deepcopy(adult.checkpoint())),causal_questions=GroundedCausalQuestionV1.restore(copy.deepcopy(question.checkpoint())),causal_models=((learner,held),)).respond(cause_raw,0xFD05)
    checks['causal_direction_is_not_reversed_by_question_surface']=(cause_answer==b'')

    qcp=question.checkpoint();lcp=learner.checkpoint();restored=HumanLanguageSessionV1(type(adult).restore(copy.deepcopy(adult.checkpoint())),causal_questions=GroundedCausalQuestionV1.restore(copy.deepcopy(qcp)),causal_models=((OpenWorldCausalLearningV1.restore(copy.deepcopy(lcp)),held),))
    replay=restored.respond(raw,0xFD06)
    checks['causal_question_and_model_survive_checkpoint_without_transcript']=(replay==answer)
    blob=json.dumps({'question':qcp,'learner':lcp},sort_keys=True)
    checks['checkpoint_contains_no_raw_question_or_answer_surface']=(all(token not in blob for token in ('why is it','quiet operator','inspects the sensor','tests the sensor','because','transcript','expected_answer')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-grounded-why-question.v1','contract':'FOUNDRY_GROUNDED_WHY_QUESTION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,
            'conversation':[raw.decode(),answer.decode() if answer else ''],'visible_language_gain':'A_HELDOUT_NATURAL_WHY_QUESTION_NOW_RETURNS_ONLY_THE_UNIQUE_INTERVENTION_GROUNDED_CAUSE_PROPOSITION_WITHOUT_CONNECTIVE_PADDING_OR_ANSWER_TRAINING',
            'checks':checks,'failed':failed,'remaining_red':['ANAPHORIC_PRONOUN_REFERENCE_IN_EVENT_QUESTIONS','LEXICAL_PARAPHRASE_OF_GROUNDED_QUESTION_SEMANTICS','MAKE_STRICT_HUMAN_TERMINAL_PRODUCT_DEFAULT'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
