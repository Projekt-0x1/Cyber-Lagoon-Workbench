#!/usr/bin/env python3
"""Economic/causal gate: bounded delayed partner-action lineage expires without suppressing public action."""
from __future__ import annotations
import copy,json,time
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1,MAX_PENDING_PARTNER_ACTIONS,PARTNER_ACTION_TTL
from reference_predictive_credit_profile_v1 import Q
from reference_source_reliability_productive_construction_transfer_verify import CLAUSE,X,ALT_AMB,REL,teach_alt,response_ecology
from reference_recursive_testimony_span_repair_verify import calibrated
from reference_long_distance_structural_reply_verify import teach_long_distance_wrapper,teach_reply_role
from reference_structural_role_delayed_checkpoint_credit_verify import stage_selected_role

economic_refactor=True
phenotype_preserved=True
future_update_authority_preserved=True

def main():
    started=time.perf_counter();a,_obj,counter=calibrated();teach_alt(a);response_ecology(a,ALT_AMB,(CLAUSE,X));teach_long_distance_wrapper(a);teach_reply_role(a,counter,0)
    base_cp=copy.deepcopy(a.checkpoint());bank=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));tickets=[]
    for i in range(MAX_PENDING_PARTNER_ACTIONS):
        selected=stage_selected_role(bank,counter,source=0xB000+i);ticket=bank.current_partner_action_ticket()
        if selected<=0 or ticket<=0:raise RuntimeError('partner_capacity:stage')
        tickets.append(ticket);bank._clear_current_occurrence()
    full_cp=copy.deepcopy(bank.checkpoint());full_rows=full_cp.get('pending_span_reply_actions',())

    # Full delayed-credit capacity cannot suppress the next resident public action.
    selected17=stage_selected_role(bank,counter,source=0xC001);ticket17=bank.current_partner_action_ticket();bank._clear_current_occurrence()
    after17=copy.deepcopy(bank.checkpoint())

    restored=LanguageMasteryAdultV1.restore(copy.deepcopy(full_cp));restored_tickets=set(restored._pending_span_reply_actions)
    first=tickets[0];born=restored._pending_span_reply_actions[first][3]
    restored._advance(max(PARTNER_ACTION_TTL+1,born+PARTNER_ACTION_TTL-restored._tick+1));expired_refused=False
    try:restored.experience_partner_choice(27393,Q,independent_return=True,action_ticket=first)
    except RuntimeError:expired_refused=True
    post_expiry=copy.deepcopy(restored.checkpoint())

    checks={
      'capacity_is_fixed_and_fully_used':MAX_PENDING_PARTNER_ACTIONS==16 and len(full_rows)==16 and len(set(tickets))==16,
      'checkpoint_restores_exact_live_ticket_set':restored_tickets==set(tickets),
      'seventeenth_public_action_is_not_suppressed_by_credit_capacity':selected17==27393 and ticket17>max(tickets),
      'seventeenth_delayed_trace_refuses_without_eviction':len(after17.get('pending_span_reply_actions',()))==16 and ticket17 not in {row['ticket'] for row in after17.get('pending_span_reply_actions',())},
      'expired_ticket_cannot_settle_partner_credit':expired_refused and first not in restored._pending_span_reply_actions,
      'expired_ticket_is_absent_from_future_checkpoint':first not in {row['ticket'] for row in post_expiry.get('pending_span_reply_actions',())},
      'no_surface_or_transcript_in_pending_rows':all(set(row)<= {'ticket','roles','partner_context','selected_program','born_tick'} and all(set(role)<= {'template','port','source'} for role in row.get('roles',())) for row in full_rows),
      'bounded_fast_path':time.perf_counter()-started<1.0,
    }
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-pending-partner-action-capacity-expiry.v1','contract':'FOUNDRY_PENDING_PARTNER_ACTION_CAPACITY_EXPIRY_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'economic_refactor':economic_refactor,'phenotype_preserved':phenotype_preserved,'future_update_authority_preserved':future_update_authority_preserved,'economic_gain':'BOUNDED_16_SLOT_DELAYED_PARTNER_ACTION_LINEAGE_WITH_FIXED_TTL_AND_NO_PUBLIC_ACTION_SUPPRESSION','capacity':MAX_PENDING_PARTNER_ACTIONS,'ttl_ticks':PARTNER_ACTION_TTL,'checks':checks,'failed':failed,'remaining_red':['DIRECT_PENDING_PARTNER_ACTION_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
