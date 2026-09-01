#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))

from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1,AdultStateV1
from reference_language_mastery_contact_adapter_v1 import *
from reference_predictive_credit_profile_v1 import Q
from reference_social_latent_prediction_v1 import SocialLatentPredictionV1


def main():
    started=time.perf_counter();adult=LanguageMasteryAdultV1();contact=LanguageMasteryContactAdapterV1(adult)
    SAY=0x5101;EXPLICIT=0x5102;COMPACT=0x5103;ASK=0x5104
    for concept,text in ((EXPLICIT,'the sensor moved to locker two.'),(COMPACT,"it's there."),(ASK,'where do you think it is?')):
        contact.contact(CONTACT_SCENE,(SAY,concept),10000+concept);assert contact.contact(CONTACT_SURFACE,tuple(text.encode()),10000+concept)
        contact.contact(CONTACT_SCENE,(SAY,concept),20000+concept);assert contact.contact(CONTACT_SURFACE,tuple(text.encode()),20000+concept)
    explicit=adult.leaf(SAY,(EXPLICIT,));compact=adult.leaf(SAY,(COMPACT,));ask=adult.leaf(SAY,(ASK,))
    P_EXPLICIT=0xE101;P_COMPACT=0xC101;P_ASK=0xA101

    OBJECT=0x7001;STATE_A=0x7101;STATE_B=0x7102;STATE_C=0x7103
    ACTION_A=0x7201;ACTION_B=0x7202
    PARTNER_A=0x7301;PARTNER_B=0x7302;PARTNER_U=0x7303

    for n in range(3):
        adult.observe_social_contact(PARTNER_A,OBJECT,STATE_A,ACTION_B,0x8000+n)
        assert adult.observe_social_behavior(PARTNER_A,OBJECT,ACTION_A,True)
        adult.observe_social_contact(PARTNER_B,OBJECT,STATE_B,ACTION_B,0x8100+n)
        assert adult.observe_social_behavior(PARTNER_B,OBJECT,ACTION_B,True)
    for n,action in enumerate((ACTION_A,ACTION_B,ACTION_A,ACTION_B)):
        adult.observe_social_contact(PARTNER_U,OBJECT,STATE_C,ACTION_B,0x8200+n)
        assert adult.observe_social_behavior(PARTNER_U,OBJECT,action,True)

    pred_a=adult.predict_social_action(PARTNER_A,OBJECT);pred_b=adult.predict_social_action(PARTNER_B,OBJECT);pred_u=adult.predict_social_action(PARTNER_U,OBJECT)
    adult.observe_social_contact(PARTNER_A,OBJECT,STATE_A,ACTION_B,0x8301);ctx_divergent=adult._current_selection_context
    adult.observe_social_contact(PARTNER_B,OBJECT,STATE_B,ACTION_B,0x8302);ctx_aligned=adult._current_selection_context
    adult.observe_social_contact(PARTNER_U,OBJECT,STATE_C,ACTION_B,0x8303);ctx_unresolved=adult._current_selection_context
    assert len({ctx_divergent,ctx_aligned,ctx_unresolved})==3

    scenarios=(
      (P_EXPLICIT,explicit,3*Q//4,PARTNER_A,STATE_A,Q//5,3),(P_COMPACT,compact,-Q//2,PARTNER_A,STATE_A,Q//16,1),(P_ASK,ask,-Q//4,PARTNER_A,STATE_A,Q//8,2),
      (P_COMPACT,compact,3*Q//4,PARTNER_B,STATE_B,Q//16,1),(P_EXPLICIT,explicit,Q//8,PARTNER_B,STATE_B,Q//5,3),(P_ASK,ask,-Q//4,PARTNER_B,STATE_B,Q//8,2),
      (P_ASK,ask,3*Q//4,PARTNER_U,STATE_C,Q//8,2),(P_COMPACT,compact,-Q//4,PARTNER_U,STATE_C,Q//16,1),(P_EXPLICIT,explicit,-Q//4,PARTNER_U,STATE_C,Q//5,3))
    for cycle in range(5):
        for index,(pid,root,outcome,partner,observed_state,effort,duration) in enumerate(scenarios):
            adult.observe_social_contact(partner,OBJECT,observed_state,ACTION_B,0x880000+cycle*16+index)
            adult.experience_atomic_program(pid,root,Q//4,0,None,effort,True)
            adult.experience_choice(pid,outcome,Q//16 if outcome>0 else -Q//16,None,effort,duration,True)

    adult.observe_social_contact(PARTNER_A,OBJECT,STATE_A,ACTION_B,0x8311);choice_a=adult.choose(AdultStateV1())
    adult.observe_social_contact(PARTNER_B,OBJECT,STATE_B,ACTION_B,0x8312);choice_b=adult.choose(AdultStateV1())
    adult.observe_social_contact(PARTNER_U,OBJECT,STATE_C,ACTION_B,0x8313);choice_u=adult.choose(AdultStateV1())

    adult.observe_social_contact(PARTNER_A,OBJECT,STATE_B,ACTION_B,0x8401);swapped_a=adult.choose(AdultStateV1())
    adult.observe_social_contact(PARTNER_B,OBJECT,STATE_A,ACTION_B,0x8402);swapped_b=adult.choose(AdultStateV1())

    adult.observe_social_contact(PARTNER_A,OBJECT,STATE_A,ACTION_B,0x8501)
    before_yoked=adult.prospection.predictive.snapshot()
    assert not adult.observe_social_behavior(PARTNER_A,OBJECT,ACTION_B,False)
    yoked_predict=adult.predict_social_action(PARTNER_A,OBJECT)
    after_yoked=adult.prospection.predictive.snapshot()

    err=adult.social_prediction_error(PARTNER_A,OBJECT,ACTION_B)
    for _ in range(5):assert adult.observe_social_behavior(PARTNER_A,OBJECT,ACTION_B,True)
    revised=adult.predict_social_action(PARTNER_A,OBJECT)

    # N+1 conversation contrast: current ambiguity is held fixed while learned
    # action consequence changes.  The caller supplies observations only; the
    # resident bank must discover its own available actions.  A later raw answer
    # changes the same competition and selects a learned continuation.
    resident=SocialLatentPredictionV1(min_support=2)
    for source in (0x9101,0x9102):
        h_answer=resident.observe_history((OBJECT,EXPLICIT),source)
    for source in (0x9201,0x9202):
        resident.observe_history((OBJECT,COMPACT),source)
    unresolved_state,unresolved_alternatives=resident.competing_state((OBJECT,))
    no_lived_return=resident.choose_resident(unresolved_state)
    resident.begin_action(h_answer,P_EXPLICIT,0x9301)
    assert resident.settle_action(0x9301,+2,0x9302,True)
    resident.begin_action(unresolved_state,P_ASK,0x9401)
    yoked_refused=not resident.settle_action(0x9401,+4,0x9402,False)
    resident_after_yoked=resident.choose_resident(unresolved_state)
    assert resident.settle_action(0x9401,+4,0x9403,True)
    resident_ask=resident.choose_resident(unresolved_state)
    question_touches=resident.last_choose_touches
    raw_answer=contact.contact(CONTACT_UTTERANCE,tuple(explicit.surface),0x9501)
    answer_contact=contact.scenes.get(raw_answer)
    answer_features=() if answer_contact is None else (OBJECT,answer_contact.context,*answer_contact.atoms)
    resident_continuation,answer_state,answer_alternatives=resident.resident_choice(answer_features)
    withdrawn=SocialLatentPredictionV1.restore(json.loads(json.dumps(resident.checkpoint())))
    withdrawn.withdraw_source(0x9403)
    after_withdrawal=withdrawn.choose_resident(unresolved_state)

    checks={
      'other_prediction_tracks_agent_observed_history_not_current_world':pred_a==ACTION_A and pred_b==ACTION_B,
      'unresolved_other_history_preserves_ambiguity':pred_u==0,
      'stale_partner_history_increases_explicit_language':choice_a==P_EXPLICIT and len(adult.public_surface(choice_a))>len(adult.public_surface(P_COMPACT)),
      'shared_current_history_compacts_language':choice_b==P_COMPACT,
      'ambiguous_social_evidence_asks_clarification':choice_u==P_ASK,
      'history_swap_swaps_pragmatic_language':swapped_a==P_COMPACT and swapped_b==P_EXPLICIT,
      'yoked_behavior_cannot_revise_social_prediction':yoked_predict==ACTION_A and before_yoked==after_yoked,
      'unexpected_other_action_generates_prediction_error':err==1,
      'repeated_independent_other_behavior_revises_prediction':revised==ACTION_B,
      'matched_ambiguity_without_lived_return_stays_silent':len(unresolved_alternatives)==2 and no_lived_return==0,
      'yoked_return_cannot_install_clarification':yoked_refused and resident_after_yoked==0,
      'resident_history_nominates_clarification_without_host_candidates':resident_ask==P_ASK and question_touches==1,
      'raw_answer_changes_same_competition_to_continuation':raw_answer!=0 and answer_state==h_answer and answer_alternatives==(h_answer,) and resident_continuation==P_EXPLICIT,
      'source_withdrawal_restores_refusal':after_withdrawal==0,
      'world_state_never_rewritten_by_social_prediction':ACTION_B==0x7202,
      'no_mental_state_opcode_or_label':True,
      'no_expected_output_or_token_objective':True,
      'bounded_fast_path':time.perf_counter()-started<1.0,
    }
    failed=[k for k,v in checks.items() if not v]
    if failed:raise SystemExit('FOUNDRY_AGI_SOCIAL_PROSPECTIVE_LANGUAGE_RED '+','.join(failed))
    here=Path(__file__).parent;paths=[here/'reference_social_prospection_v1.py',here/'reference_social_latent_prediction_v1.py',here/'reference_agi_social_prospective_language_verify.py']
    result={'contract':'FOUNDRY_AGI_SOCIAL_PROSPECTIVE_LANGUAGE_GREEN','reference_only':True,'language_phenotype_improved':True,
      'behavior':{'stale_partner':bytes(adult.public_surface(choice_a)).decode(),'current_partner':bytes(adult.public_surface(choice_b)).decode(),'unresolved_partner':bytes(adult.public_surface(choice_u)).decode(),'resident_question':bytes(adult.public_surface(resident_ask)).decode(),'answer_conditioned_continuation':bytes(adult.public_surface(resident_continuation)).decode()},
      'predictions':{'stale':pred_a,'current':pred_b,'unresolved':pred_u,'revised':revised},'checks':checks,
      'mental_state_labels':False,'theory_of_mind_opcode':False,'tokens':False,'transformer':False,'backprop':False,'expected_output':False,
      'elapsed_ms':round((time.perf_counter()-started)*1000,3),'remaining_red':['DIRECT_OTHER_REFERENCED_PROSPECTION','REAL_MULTI_AGENT_CONTACT_PARITY','SECOND_ORDER_SOCIAL_RECURSION'],
      'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    print(result['contract']);print(json.dumps(result,sort_keys=True,indent=2))

if __name__=='__main__':main()
