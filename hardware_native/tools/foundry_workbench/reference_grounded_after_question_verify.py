#!/usr/bin/env python3
"""Human-style N+1: grounded 'after' question returns only the lived successor event, not a glued two-clause replay."""
from __future__ import annotations
import copy,json,time
from reference_discourse_quantity_interaction_verify import CLAUSE,train_adult
from reference_grounded_temporal_question_v1 import GroundedTemporalQuestionV1
from reference_lived_event_order_v1 import LivedEventOrderV1

QCTX=0xF501
Q1=0xF511;Q2=0xF512

def atoms(adult,leaf):
    tid=adult._surface_leaf_family_index[int(leaf.identity)];lids=adult._surface_leaf_families[tid][int(leaf.identity)]
    return tuple(int(adult.language.historical_lexeme_binding(lid)[0]) for lid in lids)
def support(order,left,right,base):
    for source in (base+1,base+2):
        order.observe(source,1,int(left.identity));order.observe(source,2,int(right.identity))
def teach_questions(adult,leaves):
    a1=atoms(adult,leaves[3]);a2=atoms(adult,leaves[20])
    adult.observe_surface_construction(QCTX,a1,b'what happened after the careful engineer inspects the valve?',0xF521)
    adult.observe_surface_construction(QCTX,a2,b'what happened after the quiet technician tests the sensor?',0xF522)
    return a1,a2

def main():
    started=time.perf_counter();checks={};adult,leaves,_=train_adult();leaves=tuple(leaves);a1,a2=teach_questions(adult,leaves)
    held_anchor,held_answer=leaves[2],leaves[0]
    held_atoms=atoms(adult,held_anchor);held_raw=b'what happened after the careful engineer inspects the sensor?'
    rows=adult.language.invert_surface(tuple(held_raw));valid=[r for r in rows if int(r.context)==QCTX and tuple(map(int,r.atoms))==held_atoms]
    checks['heldout_full_question_is_productively_reconstructed_without_training_that_surface']=(len(valid)==1)

    order=LivedEventOrderV1();support(order,leaves[3],leaves[0],0xF600);support(order,leaves[20],leaves[23],0xF610);support(order,held_anchor,held_answer,0xF620)
    grounded=GroundedTemporalQuestionV1()
    checks['question_surface_structure_alone_has_no_after_semantics']=(grounded.answer_learned(adult,order,QCTX,held_atoms)==0)
    grounded.observe_grounded(adult,QCTX,leaves[3].identity,leaves[0].identity,order,Q1)
    checks['one_grounded_question_example_remains_provisional']=(not grounded.supported(QCTX))
    grounded.observe_grounded(adult,QCTX,leaves[20].identity,leaves[23].identity,order,Q2)
    checks['two_independent_question_examples_ground_successor_retrieval_semantics']=grounded.supported(QCTX)

    successor=grounded.answer(adult,order,valid[0].context,CLAUSE,valid[0].atoms);spoken=b'' if not successor else bytes(adult._leaf_surface(successor))
    checks['heldout_human_style_after_question_returns_only_successor_event']=(successor==held_answer.identity and spoken==bytes(held_answer.surface))
    checks['answer_does_not_replay_premise_or_insert_connective_padding']=(bytes(held_anchor.surface) not in spoken and b' then ' not in spoken and b' because ' not in spoken)

    # Matched syntax/question semantics with no heldout event-order evidence must refuse.
    no_order=LivedEventOrderV1();support(no_order,leaves[3],leaves[0],0xF700);support(no_order,leaves[20],leaves[23],0xF710)
    checks['same_question_without_heldout_lived_order_refuses']=(grounded.answer_learned(adult,no_order,QCTX,held_atoms)==0)

    reverse=LivedEventOrderV1();support(reverse,held_answer,held_anchor,0xF720)
    checks['same_question_when_world_order_is_reversed_refuses']=(grounded.answer_learned(adult,reverse,QCTX,held_atoms)==0)

    ambiguous=LivedEventOrderV1.restore(copy.deepcopy(order.checkpoint()));support(ambiguous,held_anchor,leaves[1],0xF730)
    checks['two_supported_successors_refuse_instead_of_guessing']=(ambiguous.successor(held_anchor.identity)==0 and grounded.answer_learned(adult,ambiguous,QCTX,held_atoms)==0)

    weakened=GroundedTemporalQuestionV1();weakened.observe_grounded(adult,QCTX,leaves[3].identity,leaves[0].identity,order,Q1)
    checks['question_semantics_requires_independent_language_examples']=(weakened.answer_learned(adult,order,QCTX,held_atoms)==0)

    checkpoint={'question':grounded.checkpoint(),'order':order.checkpoint()};blob=json.dumps(checkpoint,sort_keys=True)
    checks['checkpoint_contains_no_question_surface_answer_surface_or_transcript']=(all(token not in blob for token in ('what happened','careful engineer','tests the sensor','transcript','expected_answer')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-grounded-after-question.v1','contract':'FOUNDRY_GROUNDED_AFTER_QUESTION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,
            'conversation':[held_raw.decode(),spoken.decode() if spoken else ''],'visible_language_gain':'A_HELDOUT_HUMAN_STYLE_AFTER_QUESTION_NOW_RETURNS_ONLY_THE_UNIQUE_LIVED_SUCCESSOR_EVENT_FROM_GROUNDED_QUESTION_SEMANTICS_AND_WORLD_ORDER',
            'checks':checks,'failed':failed,'remaining_red':['GROUNDED_WHY_QUESTION_TO_CAUSAL_MODEL_WITH_CONCISE_ANSWER','ANAPHORIC_PRONOUN_REFERENCE_IN_EVENT_QUESTIONS','DIRECT_GROUNDED_QUESTION_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
