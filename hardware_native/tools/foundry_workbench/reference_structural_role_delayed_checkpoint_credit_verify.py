#!/usr/bin/env python3
"""N+1: structural-role eligibility survives checkpoint until delayed consequence."""
from __future__ import annotations
import copy,json,time
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
from reference_language_mastery_contact_adapter_v1 import CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1
from reference_predictive_credit_profile_v1 import Q
from reference_source_reliability_productive_construction_transfer_verify import (
    CLAUSE,X,ALT_AMB,ALT_ANS,P_ACCEPT,P_VERIFY,REL,teach_alt,response_ecology,settle,emit_all,
)
from reference_recursive_testimony_span_repair_verify import calibrated
from reference_long_distance_structural_reply_verify import DISTRACTOR,wrap,teach_long_distance_wrapper,teach_reply_role

ROLE_SOURCE=0x8E01

def stage_selected_role(a,counter,source=ROLE_SOURCE):
    counter[0]+=8;base=counter[0];a._clear_current_occurrence();m=LanguageMasteryContactAdapterV1(a)
    gap=m.contact(CONTACT_UTTERANCE,tuple(ALT_AMB),base,REL);q=settle(a)
    if q!=P_VERIFY or not gap:raise RuntimeError('delayed_role:question')
    emit_all(a,q);scene=m.contact(CONTACT_UTTERANCE,tuple(wrap(ALT_ANS,b'delayed role note.')),source,REL);choice=a.choose()
    if not scene or choice!=P_ACCEPT:raise RuntimeError('delayed_role:choice')
    return choice

def current_choice(a,heldout,counter,source):
    counter[0]+=8;base=counter[0];a._clear_current_occurrence();m=LanguageMasteryContactAdapterV1(a)
    gap=m.contact(CONTACT_UTTERANCE,tuple(ALT_AMB),base,REL);q=settle(a)
    if q!=P_VERIFY or not gap:raise RuntimeError('delayed_role:probe_question')
    emit_all(a,q);scene=m.contact(CONTACT_UTTERANCE,tuple(heldout),source,REL)
    return int(scene),int(a.choose())

def main():
    started=time.perf_counter();a,_obj,counter=calibrated();teach_alt(a);response_ecology(a,ALT_AMB,(CLAUSE,X));teach_long_distance_wrapper(a);teach_reply_role(a,counter,0)
    heldout=wrap(ALT_ANS,DISTRACTOR);earned_cp=copy.deepcopy(a.checkpoint())

    # Stage a newly selected structural occurrence but do not deliver its consequence.
    pending=LanguageMasteryAdultV1.restore(copy.deepcopy(earned_cp));selected=stage_selected_role(pending,counter)
    pending_tuple=tuple(getattr(pending,'_pending_span_reply_role',()))
    mid_cp=copy.deepcopy(pending.checkpoint())
    mid_text=json.dumps(mid_cp,sort_keys=True)

    # Positive delayed return after restart should strengthen/retain the role; adverse
    # delayed return should cancel one of the two support witnesses and reopen ambiguity.
    positive=LanguageMasteryAdultV1.restore(copy.deepcopy(mid_cp));positive_ok=True
    try:positive.experience_partner_choice(selected,Q,independent_return=True)
    except Exception:positive_ok=False
    ps,pc=current_choice(positive,heldout,counter,0x9401) if positive_ok else (0,0)

    adverse=LanguageMasteryAdultV1.restore(copy.deepcopy(mid_cp));adverse_ok=True
    try:adverse.experience_partner_choice(selected,-Q,independent_return=True)
    except Exception:adverse_ok=False
    ads,adc=current_choice(adverse,heldout,counter,0x9402) if adverse_ok else (0,0)

    # Same checkpoint without any return must not itself change learned role support.
    no_return=LanguageMasteryAdultV1.restore(copy.deepcopy(mid_cp));ns,nc=current_choice(no_return,heldout,counter,0x9403)

    checks={
      'selected_structural_eligibility_exists_before_checkpoint':len(pending_tuple)==3 and len(pending_tuple[0])>=1 and pending_tuple[-1]==selected and pending.current_partner_action_ticket()>0,
      'checkpoint_contains_no_surface_or_transcript':b'delayed role note.' .decode() not in mid_text and heldout.decode() not in mid_text,
      'checkpoint_preserves_pending_structural_eligibility':bool(getattr(LanguageMasteryAdultV1.restore(copy.deepcopy(mid_cp)),'_pending_span_reply_role',())),
      'no_return_checkpoint_does_not_self_confirm_or_reopen':bool(ns) and nc==P_ACCEPT,
      'delayed_positive_independent_return_settles_after_restart':positive_ok and bool(ps) and pc==P_ACCEPT,
      'delayed_adverse_independent_return_reopens_after_restart':adverse_ok and ads==0 and adc==0,
      'bounded_fast_path':time.perf_counter()-started<1.0,
    }
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-structural-role-delayed-checkpoint-credit.v1','contract':'FOUNDRY_STRUCTURAL_ROLE_DELAYED_CHECKPOINT_CREDIT_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':not failed,'visible_language_gain':'STRUCTURAL_REPLY_ROLE_CREDIT_SURVIVES_RESTART_UNTIL_DELAYED_INDEPENDENT_CONSEQUENCE','checks':checks,'failed':failed,'remaining_red':['SOURCE_CONFLICT_ROLE_REFINEMENT','MULTI_TURN_DELAYED_ROLE_CREDIT','DIRECT_DELAYED_ROLE_CREDIT_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
