#!/usr/bin/env python3
"""N+1: structural role transfers across partner contexts while pragmatic consequence stays local."""
from __future__ import annotations
import copy,json,time
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
from reference_language_mastery_contact_adapter_v1 import CONTACT_UTTERANCE,LanguageMasteryContactAdapterV1
from reference_predictive_credit_profile_v1 import Q
from reference_source_reliability_productive_construction_transfer_verify import ALT_AMB,ALT_ANS,P_ACCEPT,P_VERIFY,REL,UNREL,settle,emit_all
from reference_recursive_testimony_span_repair_verify import calibrated
from reference_long_distance_structural_reply_verify import DISTRACTOR,wrap,teach_long_distance_wrapper,teach_reply_role

def ask(a,channel,payload,counter,source):
    counter[0]+=8;base=counter[0];a._clear_current_occurrence();m=LanguageMasteryContactAdapterV1(a)
    gap=m.contact(CONTACT_UTTERANCE,tuple(ALT_AMB),base,channel);q=settle(a)
    if q!=P_VERIFY or not gap:raise RuntimeError('cross_partner:question')
    emit_all(a,q);scene=m.contact(CONTACT_UTTERANCE,tuple(payload),source,channel);return int(scene),int(a.choose())

def port(a,heldout):
    spans=a.language.invert_span(tuple(heldout));return None if len(spans)!=1 else a.language.span_reply_port(spans[0].template_identity)

def main():
    started=time.perf_counter();a,_obj,counter=calibrated();teach_long_distance_wrapper(a);teach_reply_role(a,counter,0);heldout=wrap(ALT_ANS,DISTRACTOR);base_cp=copy.deepcopy(a.checkpoint())
    rel=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));rs,rc=ask(rel,REL,heldout,counter,0xA001)
    unrel=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));us,uc=ask(unrel,UNREL,heldout,counter,0xA002)

    # Revise only the UNREL partner-local policy. No structural-role evidence is changed:
    # these consequences apply to the already selected public program, not a new role example.
    revised=LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp));adverse=0
    for i in range(8):
        scene,choice=ask(revised,UNREL,heldout,counter,0xA100+i)
        if not scene or not choice:break
        revised.experience_partner_choice(choice,-Q);revised.experience_partner_background(choice,False);adverse+=1
    revised._clear_current_occurrence()
    revised_cp=copy.deepcopy(revised.checkpoint())
    u2s,u2c=ask(LanguageMasteryAdultV1.restore(copy.deepcopy(revised_cp)),UNREL,heldout,counter,0xA200)
    r2s,r2c=ask(LanguageMasteryAdultV1.restore(copy.deepcopy(revised_cp)),REL,heldout,counter,0xA201)

    checks={
      'same_structural_role_is_available_in_both_partner_contexts':port(LanguageMasteryAdultV1.restore(copy.deepcopy(base_cp)),heldout)==0 and bool(rs) and bool(us),
      'partner_context_changes_public_policy_without_rewriting_structure':rc==P_ACCEPT and uc==P_VERIFY,
      'adverse_history_revises_only_target_partner_context':adverse>=1 and bool(u2s) and u2c==0 and bool(r2s) and r2c==P_ACCEPT,
      'structural_role_survives_partner_local_policy_revision':port(LanguageMasteryAdultV1.restore(copy.deepcopy(revised_cp)),heldout)==0,
      'heldout_surface_not_persisted':heldout.decode() not in json.dumps(revised_cp,sort_keys=True),
      'bounded_fast_path':time.perf_counter()-started<1.0,
    }
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-structural-role-cross-partner-calibration.v1','contract':'FOUNDRY_STRUCTURAL_ROLE_CROSS_PARTNER_CALIBRATION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'STRUCTURAL_REPLY_KNOWLEDGE_TRANSFERS_ACROSS_PARTNERS_WHILE_PARTNER_LOCAL_CONSEQUENCE_CONTROLS_WHETHER_TO_ACCEPT_VERIFY_OR_REFUSE','checks':checks,'failed':failed,'choices':{'rel':rc,'unrel':uc,'revised_unrel':u2c,'preserved_rel':r2c},'remaining_red':['PENDING_PARTNER_ACTION_EXPIRY_AND_CAPACITY_ECONOMICS','DIRECT_CROSS_PARTNER_STRUCTURAL_ROLE_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
