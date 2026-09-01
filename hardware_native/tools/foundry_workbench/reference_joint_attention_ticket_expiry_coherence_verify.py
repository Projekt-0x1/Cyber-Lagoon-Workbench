#!/usr/bin/env python3
"""Economic gate: joint-attention pending metadata cannot outlive the Adult action ticket that authorizes it."""
from __future__ import annotations
import copy,json,time
from reference_consequence_qualified_joint_attention_memory_v1 import ConsequenceQualifiedJointAttentionMemoryV1
from reference_consequence_qualified_joint_attention_memory_verify import prepare
from reference_language_mastery_terminal_v1 import emit_choice
from reference_language_mastery_adult_v1 import PARTNER_ACTION_TTL
from reference_nonvisible_unnamed_deictic_event_verify import THAT,CHANNEL
from reference_predictive_credit_profile_v1 import Q

SOURCE=0xE711

def main():
    started=time.perf_counter();checks={};adult,o,g,tracker,left,left_pos,event=prepare();memory=ConsequenceQualifiedJointAttentionMemoryV1()
    staged=memory.stage(adult,o,tracker,g,b'that',THAT,*left_pos,CHANNEL,SOURCE)
    if staged is None:raise RuntimeError('joint_expiry:stage')
    ticket,response,_episode=staged;spoken=emit_choice(adult,response);adult._clear_current_occurrence()
    checks['interrupted_joint_attention_action_is_owned_by_same_ticket_in_both_banks']=(ticket in adult._pending_span_reply_actions and ticket in memory.pending and bool(spoken))

    mid_adult=copy.deepcopy(adult.checkpoint());mid_memory=copy.deepcopy(memory.checkpoint())
    resumed=type(adult).restore(copy.deepcopy(mid_adult));rm=ConsequenceQualifiedJointAttentionMemoryV1.restore(copy.deepcopy(mid_memory))
    born=int(resumed._pending_span_reply_actions[ticket][3]);resumed._advance(max(PARTNER_ACTION_TTL+1,born+PARTNER_ACTION_TTL-resumed._tick+1));resumed._expire_partner_action_tickets()
    checks['adult_expiry_removes_action_occurrence_without_redirecting_it']=(ticket not in resumed._pending_span_reply_actions)
    before_evidence=copy.deepcopy(rm.evidence);settled=rm.settle_partner_return(resumed,ticket,response,Q,True)
    checks['expired_adult_ticket_cannot_settle_memory_and_orphan_metadata_is_discarded']=(not settled and ticket not in rm.pending and rm.evidence==before_evidence)

    # synchronize_pending is an explicit checkpoint hygiene boundary for any other
    # Adult-side expiry/eviction that happens before a return attempt.
    again=ConsequenceQualifiedJointAttentionMemoryV1.restore(copy.deepcopy(mid_memory));again_adult=type(adult).restore(copy.deepcopy(mid_adult));again_adult._advance(PARTNER_ACTION_TTL+2);again_adult._expire_partner_action_tickets();pruned=again.synchronize_pending(again_adult)
    checks['explicit_pending_synchronization_prunes_exactly_stale_memory_ticket']=(pruned==1 and not again.pending)
    cp=again.checkpoint();blob=json.dumps(cp,sort_keys=True)
    checks['future_checkpoint_cannot_retain_orphan_ticket_after_synchronization']=(cp.get('pending')==[] and str(ticket) not in json.dumps(cp.get('pending',()),sort_keys=True))
    checks['no_surface_transcript_or_expiry_redirect_state_is_persisted']=(all(token not in blob for token in ('that again','tests the sensor','transcript','redirect_target','current_frame')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-joint-attention-ticket-expiry-coherence.v1','contract':'FOUNDRY_JOINT_ATTENTION_TICKET_EXPIRY_COHERENCE_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'economic_refactor':True,'phenotype_preserved':True,'future_update_authority_preserved':True,'economic_gain':'MEMORY_PENDING_METADATA_CANNOT_OUTLIVE_OR_REDIRECT_AN_EXPIRED_ADULT_PARTNER_ACTION_TICKET','checks':checks,'failed':failed,'remaining_red':['DIRECT_ADULT_OWNERSHIP_OF_JOINT_ATTENTION_MEMORY','DIRECT_DELAYED_JOINT_MEMORY_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
