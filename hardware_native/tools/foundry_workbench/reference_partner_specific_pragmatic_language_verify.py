#!/usr/bin/env python3
"""Partner-specific pragmatic consequence with generic social transfer intact."""
from __future__ import annotations

import copy
import json
import time

from reference_language_mastery_adult_v1 import AdultStateV1, LanguageMasteryAdultV1
from reference_language_mastery_contact_adapter_v1 import CONTACT_SCENE, CONTACT_SURFACE, LanguageMasteryContactAdapterV1
from reference_predictive_credit_profile_v1 import Q

SAY=0xD100;COMPACT=0xD101;EXPLICIT=0xD102;P_COMPACT=0xD201;P_EXPLICIT=0xD202
PARTNER_A=0xD301;PARTNER_B=0xD302;PARTNER_U=0xD303;OBJECT=0xD401;STATE=0xD402;ACTION=0xD403
SOURCE_A=0xDA01;SOURCE_B=0xDA02;SOURCE_U=0xDA03


def teach_surface(adult,contact,concept,text,source):
    raw=tuple(text.encode())
    for offset in (0,1):
        contact.contact(CONTACT_SCENE,(SAY,concept),source+offset)
        if not contact.contact(CONTACT_SURFACE,raw,source+offset):raise RuntimeError('partner_context:surface')
    return adult.leaf(SAY,(concept,))


def social_contact(adult,partner,source):
    # Event receipts vary, but the opaque active sensory-source population remains
    # stable until a separate continuity assay earns a new view into that population.
    sensory_source={PARTNER_A:SOURCE_A,PARTNER_B:SOURCE_B,PARTNER_U:SOURCE_U}.get(partner,source)
    adult.observe_social_contact(partner,OBJECT,STATE,ACTION,sensory_source)
    return int(adult._current_selection_context),int(adult._current_partner_context)


def prepare():
    adult=LanguageMasteryAdultV1();contact=LanguageMasteryContactAdapterV1(adult)
    compact=teach_surface(adult,contact,COMPACT,"it's there.",0xD500)
    explicit=teach_surface(adult,contact,EXPLICIT,"the sensor is in locker two.",0xD510)
    # Only A demonstrates the generic state/action regularity; B/U must inherit it.
    for n in range(3):
        social_contact(adult,PARTNER_A,0xD600+n)
        if not adult.observe_social_behavior(PARTNER_A,OBJECT,ACTION,True):raise RuntimeError('partner_context:behavior')
    for n in range(2):
        social_contact(adult,PARTNER_B,0xD680+n)
        if not adult.observe_social_behavior(PARTNER_B,OBJECT,ACTION,True):raise RuntimeError('partner_context:behavior')
    # Both public programs earn generic competence in the shared pragmatic relation.
    generic,_=social_contact(adult,PARTNER_B,0xD700)
    for _ in range(3):
        adult.experience_atomic_program(P_COMPACT,compact,Q//2,Q//16,None,Q//16,True)
        adult.experience_atomic_program(P_EXPLICIT,explicit,3*Q//4,Q//16,None,Q//16,True)
    return adult,compact,explicit,generic


def establish(adult):
    generic_a,partner_a=social_contact(adult,PARTNER_A,0xD710)
    for _ in range(2):
        adult.experience_partner_choice(P_COMPACT,Q,effort_q16=Q//16,duration=1)
        adult.experience_partner_background(P_COMPACT,False)
    generic_b,partner_b=social_contact(adult,PARTNER_B,0xD720)
    for _ in range(2):
        adult.experience_partner_choice(P_EXPLICIT,Q,effort_q16=Q//16,duration=1)
        adult.experience_partner_background(P_EXPLICIT,False)
    return generic_a,partner_a,generic_b,partner_b


def current_choice(adult,partner,source,state=AdultStateV1()):
    generic,local=social_contact(adult,partner,source)
    return adult.choose(state),generic,local


def main():
    started=time.perf_counter();checks={}
    adult,compact,explicit,generic=prepare()
    prediction_a=adult.predict_social_action(PARTNER_A,OBJECT)
    social_contact(adult,PARTNER_B,0xD701);prediction_b=adult.predict_social_action(PARTNER_B,OBJECT)
    social_contact(adult,PARTNER_U,0xD702);prediction_u=adult.predict_social_action(PARTNER_U,OBJECT)
    ga,pa,gb,pb=establish(adult)
    checkpoint=copy.deepcopy(adult.checkpoint())

    choice_a,test_ga,test_pa=current_choice(adult,PARTNER_A,0xD730)
    choice_b,test_gb,test_pb=current_choice(adult,PARTNER_B,0xD731)
    choice_u,test_gu,test_pu=current_choice(adult,PARTNER_U,0xD732)
    checks['generic_social_prediction_transfers_across_partners']=(prediction_a==prediction_b==prediction_u==ACTION)
    checks['same_present_generic_relation_is_held_fixed']=(ga==gb==test_ga==test_gb==test_gu==generic)
    checks['partner_local_coordinates_are_distinct_without_replacing_generic_relation']=(pa==test_pa and pb==test_pb and pa!=pb!=test_pu and pa!=test_pu)
    checks['adult_competition_selects_compact_for_partner_a']=(choice_a==P_COMPACT)
    checks['adult_competition_selects_explicit_for_partner_b']=(choice_b==P_EXPLICIT)
    checks['novel_partner_falls_back_to_generic_language']=(choice_u==P_EXPLICIT and not adult.partner_credit.candidates(test_pu))
    checks['source_swap_changes_visible_language']=(adult.public_surface(choice_a)==compact.surface and adult.public_surface(choice_b)==explicit.surface)

    # Development/timescale: one local success cannot overcommit.
    one,_,_,_=prepare();_,one_ctx=social_contact(one,PARTNER_A,0xD740)
    one.experience_partner_choice(P_COMPACT,Q)
    one_choice=one.choose();one_local=one.partner_credit.row(P_COMPACT).contexts[one_ctx]
    checks['one_shot_partner_history_does_not_override_generic']=(one_choice==P_EXPLICIT and not one_local.control_supported)

    # Controllability: same action outcome value, free/background outcomes destroy local causal bias.
    yoked,_,_,_=prepare();_,yoked_ctx=social_contact(yoked,PARTNER_A,0xD750)
    for _ in range(2):
        yoked.experience_partner_choice(P_COMPACT,Q);yoked.experience_partner_background(P_COMPACT,True)
    yrow=yoked.partner_credit.row(P_COMPACT);ylocal=yrow.contexts[yoked_ctx]
    checks['yoked_partner_precedent_cannot_override_generic']=(yoked.choose()==P_EXPLICIT and ylocal.controllability_q16==0 and yrow.outcome_mean_q16==Q)

    # Consequence authenticity: repeated non-independent return creates no local outcome/control support.
    forged,_,_,_=prepare();_,forged_ctx=social_contact(forged,PARTNER_A,0xD760)
    for _ in range(2):forged.experience_partner_choice(P_COMPACT,Q,independent_return=False)
    frow=forged.partner_credit.row(P_COMPACT);flocal=frow.contexts[forged_ctx]
    checks['non_independent_partner_return_cannot_override_generic']=(forged.choose()==P_EXPLICIT and flocal.outcome_samples==0 and not flocal.control_supported)

    # Recent history: pact violation extinguishes local bias; ordinary interaction reacquires it.
    revised=LanguageMasteryAdultV1.restore(copy.deepcopy(checkpoint));current_choice(revised,PARTNER_A,0xD770)
    before_violation=revised.choose()
    for _ in range(2):revised.experience_partner_choice(P_COMPACT,-Q)
    after_violation=revised.choose()
    for _ in range(2):revised.experience_partner_choice(P_COMPACT,Q)
    reacquired=revised.choose()
    checks['pact_violation_extinguishes_local_bias']=(before_violation==P_COMPACT and after_violation==P_EXPLICIT)
    checks['ordinary_partner_interaction_reacquires_precedent']=(reacquired==P_COMPACT)

    # Matched body/resource pressure must not masquerade as partner identity.
    press_a,_,_=current_choice(adult,PARTNER_A,0xD780,AdultStateV1(pressure_q16=Q//2))
    press_b,_,_=current_choice(adult,PARTNER_B,0xD781,AdultStateV1(pressure_q16=Q//2))
    checks['matched_resource_pressure_preserves_partner_factor']=(press_a==P_COMPACT and press_b==P_EXPLICIT)

    # Focal partner-history lesion restores generic language; generic prediction survives.
    lesioned=LanguageMasteryAdultV1.restore(copy.deepcopy(checkpoint));_,lesion_ctx=social_contact(lesioned,PARTNER_A,0xD790)
    for row in lesioned.partner_credit.rows.values():row.contexts.pop(lesion_ctx,None)
    lesioned.partner_credit.context_members.pop(lesion_ctx,None)
    lesion_choice=lesioned.choose();lesion_prediction=lesioned.predict_social_action(PARTNER_A,OBJECT)
    checks['focal_partner_history_lesion_restores_generic_not_silence']=(lesion_choice==P_EXPLICIT and lesion_prediction==ACTION)
    checks['focal_partner_history_lesion_preserves_generic_programs']=(lesioned.public_surface(P_COMPACT)==compact.surface and lesioned.public_surface(P_EXPLICIT)==explicit.surface)

    # Checkpoint retains learned divergence but not active partner occurrence.
    restored=LanguageMasteryAdultV1.restore(copy.deepcopy(checkpoint))
    checks['checkpoint_does_not_restore_active_partner_occurrence']=(restored._current_partner_context==0 and restored.choose()==0)
    restored_a,rest_ga,rest_pa=current_choice(restored,PARTNER_A,0xD7A0)
    restored_b,rest_gb,rest_pb=current_choice(restored,PARTNER_B,0xD7A1)
    checks['checkpoint_fresh_contact_restores_partner_divergence']=(restored_a==P_COMPACT and restored_b==P_EXPLICIT and rest_ga==rest_gb==generic and rest_pa!=rest_pb)

    # No current authenticated partner means no way to nominate local credit.
    atomic=LanguageMasteryAdultV1.restore(copy.deepcopy(checkpoint));atomic._clear_current_occurrence();before=atomic.partner_credit.snapshot();refused=False
    try:atomic.experience_partner_choice(P_COMPACT,Q)
    except RuntimeError:refused=True
    checks['absent_or_forged_partner_occurrence_refuses_atomically']=(refused and atomic.partner_credit.snapshot()==before)

    checks['generic_and_partner_factors_both_resident_in_competition']=(adult.last_select_touches>=2 and len(adult.credit.candidates(generic))>=2)
    checks['no_partner_specific_answer_table_or_partner_grammar']=(not hasattr(adult,'partner_answers') and not hasattr(adult,'partner_grammar'))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={
        'reference_only':True,
        'language_phenotype_improved':True,
        'visible_language_gain':'PARTNER_SPECIFIC_REFERENTIAL_EXTERNALIZATION_WITH_GENERIC_TRANSFER',
        'generic_prediction':{'a':prediction_a,'b':prediction_b,'novel':prediction_u},
        'generic_context':generic,'partner_contexts':{'a':pa,'b':pb,'novel':test_pu},
        'choices':{'a':choice_a,'b':choice_b,'novel':choice_u,'violation':after_violation,'reacquired':reacquired,'lesion':lesion_choice},
        'visible_language':{'a':bytes(compact.surface).decode(),'b':bytes(explicit.surface).decode(),'novel':bytes(explicit.surface).decode()},
        'checks':checks,
        'remaining_red':['DIRECT_AUTHENTICATED_PARTNER_PARITY','PARTNER_HISTORY_IN_LONG_FORM_DISCOURSE','COPIED_SOURCE_DEPENDENCE','LONGER_BODY_HISTORY_INTERACTION'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    if failed:
        print('FOUNDRY_PARTNER_SPECIFIC_PRAGMATIC_LANGUAGE_RED '+','.join(failed));print(json.dumps(result,sort_keys=True,indent=2));return 1
    print('FOUNDRY_PARTNER_SPECIFIC_PRAGMATIC_LANGUAGE_GREEN');print(json.dumps(result,sort_keys=True,indent=2));return 0

if __name__=='__main__':raise SystemExit(main())
