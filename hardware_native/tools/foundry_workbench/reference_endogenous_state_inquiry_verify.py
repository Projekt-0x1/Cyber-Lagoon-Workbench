#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_life_function_curriculum_v1 import (
    LifeFunctionCurriculumV2,ReferenceLifeFunctionRuntimeV2,canonical_life_function_curriculum_v2,canonical_species_program_v2,
)
from reference_life_extension_history_matrix_v1 import A,B,C
from reference_life_extension_endogenous_state_inquiry_v1 import INQUIRY_CONTEXT

Q=1<<16

def at_mark(curriculum,species,mark):
    cursor=curriculum.mark_cursor(mark)
    return ReferenceLifeFunctionRuntimeV2(species).run(LifeFunctionCurriculumV2(curriculum.events[:cursor]))

def verify_adults(a,recovered):
    checks={}
    programs=a._longest_causal_program_component();leaf=a.language_adult.leaf(100,(0xA104,));state_context=a.language_adult._somatic_appraisal_language_context()
    rows={}
    for channel in (A,B,C):
        context=a.causal_dialogue_context(programs,channel);felt=a.language_adult.somatic_appraisal(programs[0],context);atoms=a.language_adult.somatic_appraisal_atoms(felt)
        state=bytes(a.language.realize(state_context,atoms) or b'');question=a.endogenous_state_inquiry(channel);touches=a.last_endogenous_inquiry_touches;surface,used=a.compose_causal_component(leaf.identity,channel=channel)
        rows[channel]=(state,question,tuple(used),felt,touches,bytes(surface))
    checks['same_public_self_state_can_hide_causally_distinct_resident_state']=(rows[A][0]==rows[B][0] and rows[A][3].controllability_q16>rows[B][3].controllability_q16 and len(rows[A][2])>len(rows[B][2]))
    checks['only_intermediate_collision_originates_learned_inquiry']=(rows[A][1] is None and bool(rows[B][1]) and rows[C][1] is None and len(rows[A][2])==len(programs)>len(rows[B][2])>len(rows[C][2])>0)
    checks['heldout_question_is_productive_construction_not_witness_replay']=(rows[B][1] is not None and a.language.template(INQUIRY_CONTEXT,3) is not None and rows[B][1] not in (
        b'Does my current state feel manageable: I can influence what happens, and my body is settled?',
        b'Does my current state feel aversive: I have little control over what happens, and my body is under strain?'))
    checks['inquiry_work_is_local_to_current_candidate_context_bindings']=max(rows[ch][4] for ch in (A,B,C))<=len(a.language_adult.credit.row(programs[0]).contexts)<=8
    before_cp=copy.deepcopy(a.checkpoint());replay=type(a).restore(copy.deepcopy(before_cp));checks['inquiry_is_rematerialized_and_checkpoint_exact']=(replay.endogenous_state_inquiry(B)==rows[B][1] and replay.checkpoint()==before_cp and a.checkpoint()==before_cp)
    outward=type(a).restore(copy.deepcopy(before_cp));question,question_receipt=outward.externalize_endogenous_inquiry(0xE3F0,B);second_question,second_receipt=outward.externalize_endogenous_inquiry(0xE3F1,B)
    pending_cp=copy.deepcopy(outward.checkpoint());pending_restart=type(a).restore(copy.deepcopy(pending_cp));restart_question,restart_receipt=pending_restart.externalize_endogenous_inquiry(0xE3F2,B)
    checks['resident_inquiry_is_public_action_with_turn_inhibition']=(question==rows[B][1] and question_receipt is not None and not second_question and second_receipt is None and not restart_question and restart_receipt is None)
    checks['pending_public_inquiry_restart_retains_digest_not_surface_or_transcript']=(pending_restart.checkpoint()==pending_cp and question.decode(errors='replace') not in json.dumps(pending_cp,sort_keys=True) and 'transcript' not in json.dumps(pending_cp,sort_keys=True).lower())
    before_motor=pending_restart.settle_endogenous_inquiry_resolution(question_receipt,0xE3F3)
    motor=pending_restart.settle_endogenous_inquiry_motor_return(question_receipt,B,True)
    unrelated=pending_restart.settle_endogenous_inquiry_resolution(question_receipt,0xE3F3)
    for _ in range(64):pending_restart.language_adult.internal_tick();pending_restart.resident_silent_wave()
    resolved=pending_restart.settle_endogenous_inquiry_resolution(question_receipt,0xE3F3);reopened,reopened_receipt=pending_restart.externalize_endogenous_inquiry(0xE3F4,B)
    checks['only_motor_realized_and_evidence_obsolete_state_inquiry_deactivates']=(not before_motor and motor and not unrelated and resolved and not reopened and reopened_receipt is None and pending_restart.endogenous_inquiry_public_count==1)
    cut=type(a).restore(copy.deepcopy(before_cp));before_depth=len(cut.compose_causal_component(leaf.identity,channel=B)[1]);cut.language.withdraw_source(0xDF40);cut.language.withdraw_source(0xDF41)
    checks['withdraw_question_relation_silences_inquiry_not_causal_policy']=(cut.endogenous_state_inquiry(B) is None and len(cut.compose_causal_component(leaf.identity,channel=B)[1])==before_depth)
    recovered_leaf=recovered.language_adult.leaf(100,(0xA104,))
    checks['quiet_recovery_abolishes_collision_without_erasing_partner_history']=(recovered.language_adult.slow_resource_history.pressure_q16()==0 and all(recovered.endogenous_state_inquiry(ch) is None for ch in (A,B,C)) and all(len(recovered.compose_causal_component(recovered_leaf.identity,channel=ch)[1])==len(programs) for ch in (A,B,C)))
    # Matched returned-contact arms. The outward action is only B's two-link prefix,
    # while context credit remains anchored to the full current causal situation.
    outcomes={}
    for independent in (False,True):
        subject=type(a).restore(copy.deepcopy(before_cp));world_before=copy.deepcopy(subject.language_adult.world_causal_learning.checkpoint());lang_before=copy.deepcopy(subject.language.checkpoint())
        surface,receipt=subject.externalize_causal_component(leaf.identity,0xE401+(1 if independent else 0),B)
        issued=len(receipt.programs) if receipt else 0;full_context=(receipt is not None and receipt.context==subject.causal_dialogue_context(programs,B))
        settled=subject.settle_causal_dialogue_return(receipt,0xE411+(1 if independent else 0),Q,0,independent) if receipt else False
        after_q=subject.endogenous_state_inquiry(B);after_depth=len(subject.compose_causal_component(leaf.identity,channel=B)[1]);after_control=subject.language_adult.somatic_appraisal(programs[0],subject.causal_dialogue_context(programs,B)).controllability_q16
        outcomes[independent]=(issued,full_context,settled,after_q,after_depth,after_control,world_before==subject.language_adult.world_causal_learning.checkpoint(),lang_before==subject.language.checkpoint())
    yoked=outcomes[False];causal=outcomes[True]
    checks['truncated_public_action_keeps_full_situation_credit_context']=(yoked[0]==causal[0]==len(rows[B][2])==2 and yoked[1] and causal[1])
    checks['yoked_return_cannot_resolve_information_need']=(yoked[2] and bool(yoked[3]) and yoked[4]<=len(rows[B][2]) and yoked[5]<rows[B][3].controllability_q16)
    checks['independent_return_updates_control_then_abolishes_inquiry_and_restores_depth']=(causal[2] and causal[3] is None and causal[4]==len(programs) and causal[5]>rows[B][3].controllability_q16)
    checks['returned_contact_changes_credit_not_world_truth_or_language_evidence']=(yoked[6] and yoked[7] and causal[6] and causal[7])
    adult_source=Path(__file__).with_name('reference_mathematical_adult_workbench_v1.py').read_text()
    checks['causal_owner_has_no_question_bytes_partner_branch_or_curiosity_scalar']=all(token not in adult_source for token in ('Does my current state feel','channel==0xDA02','curiosity_q16','partial_control'))
    metrics={'question':bytes(rows[B][1] or b''),'depths':{hex(k):len(rows[k][2]) for k in (A,B,C)},'controls_q16':{hex(k):rows[k][3].controllability_q16 for k in (A,B,C)},'max_context_touches':max(rows[ch][4] for ch in (A,B,C)),'active_contexts':len(a.language_adult.credit.row(programs[0]).contexts),'independent_after':{'depth':causal[4],'control_q16':causal[5]},'yoked_after':{'depth':yoked[4],'control_q16':yoked[5]}}
    return checks,metrics

def main():
    started=time.perf_counter();curriculum=canonical_life_function_curriculum_v2();species=canonical_species_program_v2()
    loaded=at_mark(curriculum,species,'endogenous_state_inquiry_loaded').adult
    recovered=at_mark(curriculum,species,'endogenous_state_inquiry_recovered').adult
    checks,metrics=verify_adults(loaded,recovered);failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.endogenous-state-inquiry.v1','pass':not failed,'checks':checks,'failed':failed,'reference_only':True,'runtime_llm':False,**{key:(value.decode(errors='replace') if key=='question' else value) for key,value in metrics.items()},'elapsed_ms':round((time.perf_counter()-started)*1000,3),'claim':'PUBLIC_SELF_DESCRIPTION_COLLISION_CAN_RECRUIT_PRODUCTIVE_INQUIRY_AND_ONLY_INDEPENDENT_RETURN_RESOLVES_IT','remaining_red':['DIRECT_PARITY','NATURAL_DIALOGUE_SCHEDULER','BROAD_METACOGNITIVE_GENERALITY']}
    print('FOUNDRY_ENDOGENOUS_STATE_INQUIRY '+('GREEN' if not failed else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
