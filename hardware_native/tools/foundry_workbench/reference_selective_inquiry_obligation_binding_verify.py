#!/usr/bin/env python3
"""R2 hostile assay: a pending inquiry binds reply uptake to its causal obligation."""
from __future__ import annotations
import copy,hashlib,json,tempfile,time
from life_function_factory_v1 import build_cache,load_mark
from reference_mathematical_adult_workbench_v1 import EndogenousInquiryActionReceiptV1,_identity
from reference_partner_multisource_dispute_repair_verify import (
    EFFECT,MOTOR_SOURCE,PARTNER,Q,SOURCE_A,SOURCE_B,WORLD_SOURCE,relation_surfaces,
)


def clone(adult):return type(adult).restore(copy.deepcopy(adult.checkpoint()))


def plant_pending_obligation(adult,surface,candidate,effect):
    programs=adult._longest_causal_program_component();context=adult._causal_dialogue_appraisal_context(programs,PARTNER);born=int(adult.language_adult._advance());digest=hashlib.sha256(bytes(surface)).hexdigest()
    candidates=((int(candidate),) if int(candidate)>0 else ())
    identity=_identity('endogenous-inquiry-action-receipt-v1',(context,digest,MOTOR_SOURCE+9,PARTNER,born,candidates,int(effect)))
    receipt=EndogenousInquiryActionReceiptV1(context,digest,MOTOR_SOURCE+9,PARTNER,born,identity,candidates,int(effect));adult.pending_endogenous_inquiry_actions[identity]=receipt;adult.endogenous_inquiry_public_count+=1;return receipt


def main():
    started=time.perf_counter();checks={}
    with tempfile.TemporaryDirectory(prefix='foundry-selective-inquiry-') as directory:
        build_cache(directory);adult=load_mark(directory,'relational_surplus_recovered').adult;leaf=adult.language_adult.leaf(100,(EFFECT,))
        _surfaces,actions=adult.externalize_causal_groups(leaf.identity,MOTOR_SOURCE,PARTNER)
        for offset,action in enumerate(actions):assert adult.settle_causal_dialogue_return(action,WORLD_SOURCE+offset,Q,0,True)
        rows=adult.causal_focus_rows(leaf.identity);outgoing={int(row[2]) for row in rows};siblings=()
        for cause in sorted({int(row[2]) for row in rows}):
            terminal=tuple(row for row in rows if int(row[2])==cause and int(row[3]) not in outgoing)
            if len(terminal)>=2:siblings=terminal[:2];break
        programs=tuple(adult.causal_program_for_row(row,materialize=False) for row in siblings);pairs=tuple(relation_surfaces(adult,row,program) for row,program in zip(siblings,programs));accepted=tuple(pair[0] for pair in pairs);reversed_surfaces=tuple(pair[1] for pair in pairs)
        for surface in accepted:
            assert adult.observe_authenticated_causal_dialogue_contact(surface,SOURCE_A,channel=PARTNER)==1
            assert adult.observe_authenticated_causal_dialogue_contact(surface,SOURCE_A,channel=PARTNER)==1
        for surface in reversed_surfaces:
            assert adult.observe_authenticated_causal_dialogue_contact(surface,SOURCE_B,channel=PARTNER)==0
            assert adult.observe_authenticated_causal_dialogue_contact(surface,SOURCE_B,channel=PARTNER)==0
        candidates=tuple(adult._causal_repair_inquiry_candidates(PARTNER));checks['two_lived_reversible_disputes_produce_distinct_structural_questions']=(len(candidates)==2 and len({bytes(row[0]) for row in candidates})==2 and len({(int(row[2]),int(row[3])) for row in candidates})==2)
        checks['equal_lived_bids_normally_withhold_instead_of_host_tie_breaking']=(adult.externalize_endogenous_inquiry(MOTOR_SOURCE+8,PARTNER)==(b'',None))

        chosen=candidates[0];other=candidates[1];receipt=plant_pending_obligation(adult,*chosen[0:1],chosen[2],chosen[3]);assert adult.settle_endogenous_inquiry_motor_return(receipt,PARTNER,True);cp=copy.deepcopy(adult.checkpoint());restart=clone(adult);pending=next(iter(restart.pending_endogenous_inquiry_actions.values()),None)
        checks['checkpoint_preserves_only_opaque_winning_obligation_not_question_bytes']=(restart.checkpoint()==cp and pending is not None and (pending.obligation_candidates,pending.obligation_effect)==((int(chosen[2]),),int(chosen[3])) and bytes(chosen[0]).decode(errors='replace') not in json.dumps(cp,sort_keys=True))

        chosen_effect=int(chosen[3]);other_effect=int(other[3]);chosen_index=next(i for i,row in enumerate(siblings) if adult.language_adult.leaf_equivalent(int(row[3]),chosen_effect));other_index=next(i for i,row in enumerate(siblings) if adult.language_adult.leaf_equivalent(int(row[3]),other_effect))
        chosen_receipt=int(siblings[chosen_index][4]);other_receipt=int(siblings[other_index][4]);chosen_answer=accepted[chosen_index];other_answer=accepted[other_index]

        surface_only=clone(adult);surface_only.pending_endogenous_inquiry_actions.clear();plant_pending_obligation(surface_only,chosen[0],0,0);control_change=surface_only.observe_authenticated_causal_dialogue_contact(other_answer,SOURCE_B,channel=PARTNER)
        checks['obligation_ancestry_lesion_restores_surface_only_sibling_leak']=(control_change==1 and surface_only.causal_dialogue_dispute_support(PARTNER,other_receipt)==0 and surface_only.causal_dialogue_dispute_support(PARTNER,chosen_receipt)>0)

        wrong=clone(adult);wrong_before=copy.deepcopy(wrong._causal_dialogue_dispute_evidence);wrong_change=wrong.observe_authenticated_causal_dialogue_contact(other_answer,SOURCE_B,channel=PARTNER)
        checks['structurally_valid_sibling_reply_cannot_settle_different_pending_question']=(wrong_change==0 and wrong._causal_dialogue_dispute_evidence==wrong_before and wrong.causal_dialogue_dispute_support(PARTNER,other_receipt)>0 and wrong.causal_dialogue_dispute_support(PARTNER,chosen_receipt)>0)

        answered=clone(adult);answer_pending=next(iter(answered.pending_endogenous_inquiry_actions.values()));answer_change=answered.observe_authenticated_causal_dialogue_contact(chosen_answer,SOURCE_B,channel=PARTNER);chosen_support=answered.causal_dialogue_dispute_support(PARTNER,chosen_receipt);other_support=answered.causal_dialogue_dispute_support(PARTNER,other_receipt);answer_closed=answered.settle_endogenous_inquiry_resolution(answer_pending,WORLD_SOURCE+20);next_question,next_receipt=answered.externalize_endogenous_inquiry(MOTOR_SOURCE+10,PARTNER)
        checks['matching_reply_settles_only_originating_obligation_and_exposes_sibling']=(answer_change==1 and chosen_support==0 and other_support>0 and answer_closed and next_receipt is not None and bytes(next_question or b'')==bytes(other[0]))

    failed=[name for name,passed in checks.items() if not passed]
    result={'schema':'cyber-lagoon.selective-inquiry-obligation-binding.v1','contract':'FOUNDRY_SELECTIVE_INQUIRY_OBLIGATION_BINDING_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'runtime_llm':False,'mechanism_change':True,'checks':checks,'failed':failed,'elapsed_ms':round((time.perf_counter()-started)*1000,3),'visible_language_gain':{'before':'WHILE_ONE_CAUSAL_QUESTION_WAS_PENDING_A_STRUCTURALLY_VALID_REPLY_COULD_REVISE_A_DIFFERENT_LIVE_DISPUTE','after':'PENDING_INQUIRY_ANCESTRY_LIMITS_REPLY_UPTAKE_TO_ITS_CAUSAL_OBLIGATION;THE_UNANSWERED_SIBLING_SURVIVES_AS_THE_NEXT_QUESTION'},'remaining_red':['NATURAL_UNEQUAL_BID_SELECTIVE_REPLY_WITHOUT_INTERVENTION','THREE_OR_MORE_SIMULTANEOUS_SOURCE_CAUSAL_CLAIMS','RECURSIVE_EMBEDDED_QUD_BINDING','MULTIMODAL_COMMON_CAUSE_IN_CANONICAL_LIFE','DIRECT_PARITY','BROAD_HUMAN_DIALOGUE']}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1


if __name__=='__main__':raise SystemExit(main())
