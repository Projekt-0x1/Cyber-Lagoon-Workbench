#!/usr/bin/env python3
"""N+1 RED/GREEN: delayed structural-role consequence survives unrelated intervening dialogue safely."""
from __future__ import annotations
import copy,json,time
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
from reference_language_mastery_contact_adapter_v1 import CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1
from reference_predictive_credit_profile_v1 import Q
from reference_source_reliability_productive_construction_transfer_verify import CLAUSE,X,ALT_AMB,ALT_ANS,P_ACCEPT,P_VERIFY,REL,teach_alt,response_ecology,settle,emit_all
from reference_recursive_testimony_span_repair_verify import calibrated
from reference_long_distance_structural_reply_verify import DISTRACTOR,wrap,teach_long_distance_wrapper,teach_reply_role
from reference_structural_role_delayed_checkpoint_credit_verify import stage_selected_role,current_choice

UNRELATED=b'the careful engineer reviews the ledger.'

def main():
    started=time.perf_counter();a,_obj,counter=calibrated();teach_alt(a);response_ecology(a,ALT_AMB,(CLAUSE,X));teach_long_distance_wrapper(a);teach_reply_role(a,counter,0)
    heldout=wrap(ALT_ANS,DISTRACTOR);base_cp=copy.deepcopy(a.checkpoint())

    pending=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));selected=stage_selected_role(pending,counter)
    staged=tuple(getattr(pending,'_pending_span_reply_role',()));ticket=pending.current_partner_action_ticket()
    # Intervening unrelated authenticated language contact may change current occurrence,
    # but the first action moves into the bounded pending-action bank. A second structural
    # action may then occur before either consequence returns.
    m=LanguageMasteryContactAdapterV1(pending);m.contact(CONTACT_UTTERANCE,tuple(UNRELATED),0xD001,0xD002)
    first_banked=tuple(pending._pending_span_reply_actions.get(ticket,()))
    selected2=stage_selected_role(pending,counter,source=0x8E02);ticket2=pending.current_partner_action_ticket()
    pending._clear_current_occurrence()
    bank_before=copy.deepcopy(pending._pending_span_reply_actions);mid_cp=copy.deepcopy(pending.checkpoint())

    no_ticket=LanguageMasteryAdultV1.restore(copy.deepcopy(mid_cp));no_ticket_refused=False
    try:no_ticket.experience_partner_choice(selected,-Q,independent_return=True)
    except RuntimeError:no_ticket_refused=True
    wrong_ticket=LanguageMasteryAdultV1.restore(copy.deepcopy(mid_cp));wrong_ticket_refused=False
    try:wrong_ticket.experience_partner_choice(selected,-Q,independent_return=True,action_ticket=max(ticket,ticket2)+1)
    except RuntimeError:wrong_ticket_refused=True

    # Settle the second action first. The first must remain pending; then settle the first
    # adverse return and require the learned role to reopen under the same heldout wrapper.
    resumed=LanguageMasteryAdultV1.restore(copy.deepcopy(mid_cp));second_ok=first_ok=True
    try:resumed.experience_partner_choice(selected2,Q,independent_return=True,action_ticket=ticket2)
    except Exception:second_ok=False
    first_still_pending=ticket in resumed._pending_span_reply_actions and ticket2 not in resumed._pending_span_reply_actions
    try:resumed.experience_partner_choice(selected,-Q,independent_return=True,action_ticket=ticket)
    except Exception:first_ok=False
    rs,rc=current_choice(resumed,heldout,counter,0xD003) if second_ok and first_ok else (0,0)

    checks={
      'selected_structural_eligibility_exists_before_intervening_turn':len(staged)==3 and len(staged[0])>=1 and staged[-1]==selected and ticket>0,
      'unrelated_intervening_dialogue_moves_original_action_to_pending_bank':len(first_banked)==4 and first_banked[:3]==staged and first_banked[3]>0,
      'checkpoint_preserves_two_distinct_pending_action_tickets':set(bank_before)=={ticket,ticket2} and set(row['ticket'] for row in mid_cp.get('pending_span_reply_actions',()))=={ticket,ticket2},
      'detached_delayed_return_requires_action_ticket':no_ticket_refused,
      'wrong_action_ticket_cannot_steal_delayed_credit':wrong_ticket_refused,
      'out_of_order_ticket_settlement_preserves_other_pending_action':second_ok and first_still_pending,
      'later_independent_adverse_return_updates_original_role_after_interruption':first_ok and rs==0 and rc==0,
      'checkpoint_contains_no_intervening_transcript':UNRELATED.decode() not in json.dumps(mid_cp,sort_keys=True),
      'bounded_fast_path':time.perf_counter()-started<1.0,
    }
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-structural-role-multiturn-delayed-credit.v1','contract':'FOUNDRY_STRUCTURAL_ROLE_MULTITURN_DELAYED_CREDIT_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':not failed,'visible_language_gain':'MULTIPLE_DELAYED_STRUCTURAL_PARTNER_ACTIONS_SURVIVE_INTERVENING_DIALOGUE_AND_SETTLE_ONLY_BY_THEIR_OWN_TICKETS','checks':checks,'failed':failed,'remaining_red':['CROSS_PARTNER_STRUCTURAL_ROLE_CALIBRATION','PENDING_PARTNER_ACTION_EXPIRY_AND_CAPACITY_ECONOMICS','DIRECT_MULTITURN_DELAYED_ROLE_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
