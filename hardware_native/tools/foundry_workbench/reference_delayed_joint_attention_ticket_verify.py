#!/usr/bin/env python3
"""N+1: delayed unnamed shared-attention returns settle only the exact Adult action occurrence ticket."""
from __future__ import annotations
import copy,json,time
from reference_consequence_qualified_joint_attention_memory_v1 import ConsequenceQualifiedJointAttentionMemoryV1
from reference_consequence_qualified_joint_attention_memory_verify import prepare
from reference_language_mastery_terminal_v1 import emit_choice
from reference_nonvisible_unnamed_deictic_event_verify import THAT,CHANNEL,event_surface
from reference_predictive_credit_profile_v1 import Q

S1=0xE611;S2=0xE612

def stage_emit(memory,adult,o,g,tracker,left_pos,source):
    staged=memory.stage(adult,o,tracker,g,b'that',THAT,*left_pos,CHANNEL,source)
    if staged is None:raise RuntimeError('delayed_joint:stage')
    ticket,response,episode=staged;spoken=emit_choice(adult,response)
    if not spoken:raise RuntimeError('delayed_joint:emit')
    return ticket,response,episode,spoken

def main():
    started=time.perf_counter();checks={};adult,o,g,tracker,left,left_pos,event=prepare();memory=ConsequenceQualifiedJointAttentionMemoryV1()

    t1,response1,ep1,speech1=stage_emit(memory,adult,o,g,tracker,left_pos,S1)
    adult._clear_current_occurrence();bank1=copy.deepcopy(adult._pending_span_reply_actions.get(t1))
    t2,response2,ep2,speech2=stage_emit(memory,adult,o,g,tracker,left_pos,S2)
    adult._clear_current_occurrence();bank_before=copy.deepcopy(adult._pending_span_reply_actions)
    adult_cp=copy.deepcopy(adult.checkpoint());memory_cp=copy.deepcopy(memory.checkpoint())
    blob=json.dumps({'adult':adult_cp,'memory':memory_cp},sort_keys=True)

    checks['same_program_two_actions_have_distinct_adult_occurrence_tickets']=(response1==response2 and t1>0 and t2>t1 and bank1 is not None and set(bank_before)=={t1,t2})
    checks['checkpoint_preserves_same_exact_ticket_set_in_adult_and_memory']=(
        {row['ticket'] for row in adult_cp.get('pending_span_reply_actions',())}=={t1,t2}
        and {row['ticket'] for row in memory_cp.get('pending',())}=={t1,t2})

    resumed=type(adult).restore(copy.deepcopy(adult_cp));rm=ConsequenceQualifiedJointAttentionMemoryV1.restore(copy.deepcopy(memory_cp))
    no_ticket_refused=False
    try:resumed.experience_partner_choice(response1,Q,independent_return=True)
    except RuntimeError:no_ticket_refused=True
    wrong_ticket=max(t1,t2)+100
    before_wrong=(copy.deepcopy(resumed._pending_span_reply_actions),copy.deepcopy(rm.pending))
    wrong_result=rm.settle_partner_return(resumed,wrong_ticket,response1,Q,True)
    checks['detached_same_program_return_requires_occurrence_ticket']=(no_ticket_refused and not wrong_result)
    checks['wrong_ticket_cannot_consume_adult_or_memory_pending_state']=(before_wrong==(resumed._pending_span_reply_actions,rm.pending))

    second_ok=rm.settle_partner_return(resumed,t2,response2,Q,True)
    first_still=(t1 in resumed._pending_span_reply_actions and t1 in rm.pending and t2 not in resumed._pending_span_reply_actions and t2 not in rm.pending)
    pre_first=rm.resolve(resumed,o,b'that again',CHANNEL)
    first_ok=rm.settle_partner_return(resumed,t1,response1,Q,True)
    learned=rm.resolve(resumed,o,b'that again',CHANNEL);answer=b'' if learned is None else event_surface(resumed,learned.event)
    checks['out_of_order_second_ticket_settlement_preserves_first_pending_action']=(second_ok and first_still and pre_first is None)
    checks['later_first_ticket_settlement_combines_independent_sources_into_recall']=(first_ok and learned is not None and learned.entity==left and answer==b'the careful engineer tests the sensor.')

    # Exact checkpoint state contains opaque ticket/context/program/episode IDs only.
    checks['checkpoint_contains_no_public_surface_pointing_or_transcript']=(all(token not in blob for token in ('that again','tests the sensor','point_y2','point_x2','transcript','current_frame')))
    checks['both_public_actions_used_identical_normal_reafferent_surface']=(speech1==speech2==b'the careful engineer tests the sensor.')
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-delayed-joint-attention-ticket.v1','contract':'FOUNDRY_DELAYED_JOINT_ATTENTION_TICKET_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'TWO_IDENTICAL_PUBLIC_RESPONSES_NOW_RETAIN_DISTINCT_DELAYED_SHARED_ATTENTION_CREDIT_ACROSS_INTERRUPTION_AND_CHECKPOINT_AND_SETTLE_OUT_OF_ORDER_ONLY_BY_ADULT_ACTION_TICKET','tickets':[t1,t2],'conversation':['that again',answer.decode() if answer else ''],'checks':checks,'failed':failed,'remaining_red':['JOINT_ATTENTION_TICKET_EXPIRY_AND_MEMORY_PENDING_COHERENCE','DIRECT_ADULT_OWNERSHIP_OF_JOINT_ATTENTION_MEMORY','DIRECT_DELAYED_JOINT_MEMORY_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
