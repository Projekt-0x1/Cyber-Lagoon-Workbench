#!/usr/bin/env python3
"""Partner history biases resident discourse matter without replacing generic relevance."""
from __future__ import annotations

import copy
import json
import time

from reference_global_discourse_relevance_verify import fresh
from reference_language_mastery_adult_v1 import AdultStateV1, LanguageMasteryAdultV1
from reference_partner_specific_pragmatic_language_verify import (
    ACTION, OBJECT, PARTNER_A, PARTNER_B, PARTNER_U, STATE, social_contact,
)
from reference_predictive_credit_profile_v1 import Q


def prepare():
    adult,frontier,_factors,_ctx_a,_ctx_b=fresh()
    # One partner supplies the transferable state/action relation. B and the novel
    # partner must inherit it; neither is a separate topic or answer coordinate.
    for n in range(3):
        social_contact(adult,PARTNER_A,0xDB00+n)
        if not adult.observe_social_behavior(PARTNER_A,OBJECT,ACTION,True):
            raise RuntimeError('partner_discourse:social_prediction')
    for n in range(2):
        social_contact(adult,PARTNER_B,0xDB08+n)
        if not adult.observe_social_behavior(PARTNER_B,OBJECT,ACTION,True):
            raise RuntimeError('partner_discourse:social_prediction')
    generic,_=social_contact(adult,PARTNER_B,0xDB10)
    # All proposition leaves earn weak generic relevance in the shared relation.
    for leaf in frontier:
        for _ in range(2):
            adult.experience_discourse_candidate(leaf.identity,Q//4,context=generic)
            adult.experience_discourse_background(leaf.identity,False)
    return adult,frontier,generic


def establish_partner(adult,frontier,partner,parity,source,repetitions=2,
                      positive=Q,negative=-Q,effort=Q//16,duration=1):
    generic,local=social_contact(adult,partner,source)
    try:
        for i,leaf in enumerate(frontier):
            outcome=positive if i%2==parity else negative
            for _ in range(repetitions):
                adult.experience_partner_choice(
                    leaf.identity,outcome,effort_q16=effort,duration=duration)
                adult.experience_partner_background(leaf.identity,False)
    except RuntimeError:
        return generic,local,False
    return generic,local,True


def select(adult,frontier,partner,source,state=AdultStateV1()):
    generic,local=social_contact(adult,partner,source)
    root=adult.organize_relevant_frontier(frontier,state)
    selected=tuple(adult.last_discourse_selected)
    surface=() if root is None else tuple(root.surface)
    return root,selected,surface,generic,local


def remove_partner_context(adult,context):
    for row in adult.partner_credit.rows.values():row.contexts.pop(int(context),None)
    adult.partner_credit.context_members.pop(int(context),None)


def main():
    started=time.perf_counter();checks={}
    adult,frontier,generic=prepare()
    ga,pa,admitted_a=establish_partner(adult,frontier,PARTNER_A,0,0xDB20)
    gb,pb,admitted_b=establish_partner(adult,frontier,PARTNER_B,1,0xDB30)
    if not (admitted_a and admitted_b):
        print('FOUNDRY_PARTNER_SPECIFIC_DISCOURSE_SELECTION_RED '
              'partner_credit_rejects_proposition_leaf')
        return 1
    learned=copy.deepcopy(adult.checkpoint())

    root_a,selected_a,surface_a,test_ga,test_pa=select(adult,frontier,PARTNER_A,0xDB40)
    root_b,selected_b,surface_b,test_gb,test_pb=select(adult,frontier,PARTNER_B,0xDB41)
    root_u,selected_u,surface_u,test_gu,test_pu=select(adult,frontier,PARTNER_U,0xDB42)
    even=tuple(leaf.identity for i,leaf in enumerate(frontier) if i%2==0)
    odd=tuple(leaf.identity for i,leaf in enumerate(frontier) if i%2==1)
    all_ids=tuple(leaf.identity for leaf in frontier)
    checks['same_generic_relation_and_frontier_are_held_fixed']=(
        ga==gb==test_ga==test_gb==test_gu==generic and len(frontier)==16)
    checks['partner_coordinates_are_distinct_second_factors']=(
        pa==test_pa and pb==test_pb and pa!=pb and test_pu not in (pa,pb))
    checks['partner_a_selects_even_matter']=(selected_a==even)
    checks['partner_b_selects_odd_matter']=(selected_b==odd)
    checks['novel_partner_falls_back_to_generic_discourse']=(
        selected_u==all_ids and not adult.partner_credit.candidates(test_pu))
    checks['source_swap_changes_visible_long_form_language']=(
        root_a is not None and root_b is not None and root_u is not None
        and len(surface_a)>300 and len(surface_b)>300 and surface_a!=surface_b)
    checks['generic_social_prediction_transfers']=(
        adult.predict_social_action(PARTNER_A,OBJECT)
        ==adult.predict_social_action(PARTNER_B,OBJECT)
        ==adult.predict_social_action(PARTNER_U,OBJECT)==ACTION)

    # Development: one partner-local event does not pass the control-history gate.
    one,one_frontier,_=prepare();_,one_ctx=social_contact(one,PARTNER_A,0xDB50)
    one.experience_partner_choice(one_frontier[0].identity,Q)
    one.experience_partner_background(one_frontier[0].identity,False)
    _,one_selected,_,_,_=select(one,one_frontier,PARTNER_A,0xDB51)
    checks['one_shot_partner_matter_does_not_override_generic']=(
        one_selected==tuple(x.identity for x in one_frontier)
        and not one.partner_credit.row(one_frontier[0].identity).contexts[one_ctx].control_supported)

    # Controllability: equal partner return value, but only expression-contingent
    # return may rescue generically disfavored matter into current discourse.
    master,mfrontier,mgeneric=prepare();yoked,yfrontier,ygeneric=prepare()
    mp=mfrontier[1];yp=yfrontier[1]
    for _ in range(2):
        master.experience_discourse_candidate(mp.identity,-Q,context=mgeneric)
        yoked.experience_discourse_candidate(yp.identity,-Q,context=ygeneric)
    _,mctx=social_contact(master,PARTNER_A,0xDB60)
    _,yctx=social_contact(yoked,PARTNER_A,0xDB61)
    for _ in range(2):
        master.experience_partner_choice(mp.identity,Q)
        master.experience_partner_background(mp.identity,False)
        yoked.experience_partner_choice(yp.identity,Q)
        yoked.experience_partner_background(yp.identity,True)
    _,mselected,_,_,_=select(master,mfrontier,PARTNER_A,0xDB62)
    _,yselected,_,_,_=select(yoked,yfrontier,PARTNER_A,0xDB63)
    mlocal=master.partner_credit.row(mp.identity).contexts[mctx]
    ylocal=yoked.partner_credit.row(yp.identity).contexts[yctx]
    checks['matched_value_master_yoked_partner_history_diverges_by_contingency']=(
        master.partner_credit.row(mp.identity).outcome_mean_q16
        ==yoked.partner_credit.row(yp.identity).outcome_mean_q16==Q
        and mlocal.controllability_q16==Q and ylocal.controllability_q16==0
        and mp.identity in mselected and yp.identity not in yselected)

    forged,ffrontier,_=prepare();_,fctx=social_contact(forged,PARTNER_A,0xDB70);fp=ffrontier[1]
    for _ in range(2):
        forged.experience_partner_choice(fp.identity,Q,independent_return=False)
    _,fselected,_,_,_=select(forged,ffrontier,PARTNER_A,0xDB71)
    flocal=forged.partner_credit.row(fp.identity).contexts[fctx]
    checks['non_independent_partner_return_cannot_bias_discourse']=(
        fselected==tuple(x.identity for x in ffrontier)
        and flocal.outcome_samples==0 and not flocal.control_supported)

    # Even strong partner history cannot inject matter absent from the generic
    # current-situation incidence set.
    blocked,bfrontier,bgeneric=prepare();bp=bfrontier[0]
    social_contact(blocked,PARTNER_A,0xDB78)
    for _ in range(4):
        blocked.experience_partner_choice(bp.identity,Q)
        blocked.experience_partner_background(bp.identity,False)
    blocked.discourse_credit.row(bp.identity).contexts.pop(bgeneric,None)
    blocked.discourse_credit.context_members[bgeneric].discard(bp.identity)
    _,bselected,_,_,_=select(blocked,bfrontier,PARTNER_A,0xDB79)
    checks['partner_factor_cannot_inject_generically_absent_matter']=(
        bp.identity not in bselected and len(bselected)==15)

    # Recent partner consequence can extinguish and reacquire one item.
    revised=LanguageMasteryAdultV1.restore(copy.deepcopy(learned));target=frontier[0]
    select(revised,frontier,PARTNER_A,0xDB80)
    before=target.identity in revised.last_discourse_selected
    for _ in range(8):revised.experience_partner_choice(target.identity,-Q)
    select(revised,frontier,PARTNER_A,0xDB81);after=target.identity in revised.last_discourse_selected
    for _ in range(12):revised.experience_partner_choice(target.identity,Q)
    select(revised,frontier,PARTNER_A,0xDB82);reacquired=target.identity in revised.last_discourse_selected
    checks['recent_partner_consequence_extinguishes_then_reacquires_matter']=(
        before and not after and reacquired)

    # The partner factor remains inside current body/resource competition.
    _,press_a,_,_,_=select(adult,frontier,PARTNER_A,0xDB90,AdultStateV1(pressure_q16=Q//4))
    _,press_b,_,_,_=select(adult,frontier,PARTNER_B,0xDB91,AdultStateV1(pressure_q16=Q//4))
    checks['matched_pressure_preserves_partner_difference']=(press_a==even and press_b==odd)

    timed,timed_frontier,_=prepare();timed_target=timed_frontier[0]
    social_contact(timed,PARTNER_A,0xDB98)
    for _ in range(2):
        timed.experience_partner_choice(
            timed_target.identity,Q//16,effort_q16=Q//8,duration=2)
        timed.experience_partner_background(timed_target.identity,False)
    _,neutral_rows,_,_,_=select(timed,timed_frontier,PARTNER_A,0xDB99)
    _,urgent_rows,_,_,_=select(
        timed,timed_frontier,PARTNER_A,0xDB9A,AdultStateV1(urgency_q16=Q//4))
    checks['urgency_suppresses_marginal_partner_adaptation_not_all_discourse']=(
        timed_target.identity in neutral_rows and timed_target.identity not in urgent_rows
        and len(neutral_rows)==16 and len(urgent_rows)==15)

    # Focal partner lesion restores transferable generic relevance, not silence.
    lesioned=LanguageMasteryAdultV1.restore(copy.deepcopy(learned))
    _,lesion_ctx=social_contact(lesioned,PARTNER_A,0xDBA0);remove_partner_context(lesioned,lesion_ctx)
    lesion_root=lesioned.organize_relevant_frontier(frontier)
    lesion_selected=tuple(lesioned.last_discourse_selected)
    checks['focal_partner_lesion_restores_generic_discourse']=(
        lesion_root is not None and lesion_selected==all_ids
        and lesioned.predict_social_action(PARTNER_A,OBJECT)==ACTION)

    # Checkpoint carries learned factors but no active source; rapid source switching
    # must retrieve rather than blend the two histories.
    restored=LanguageMasteryAdultV1.restore(copy.deepcopy(learned))
    checks['checkpoint_drops_active_partner_occurrence']=(
        restored._current_partner_context==0 and not restored.select_discourse_frontier(frontier))
    switched=[]
    for n,partner in enumerate((PARTNER_A,PARTNER_B,PARTNER_A,PARTNER_B)):
        _,rows,_,_,_=select(restored,frontier,partner,0xDBB0+n);switched.append(rows)
    checks['checkpoint_random_partner_switching_retrieves_distinct_histories']=(
        switched==[even,odd,even,odd])

    restored._clear_current_occurrence();before_credit=restored.partner_credit.snapshot();refused=False
    try:restored.experience_partner_choice(frontier[0].identity,Q)
    except RuntimeError:refused=True
    checks['absent_partner_occurrence_refuses_atomically']=(
        refused and restored.partner_credit.snapshot()==before_credit)
    checks['generic_and_partner_factors_share_one_discourse_competition']=(
        adult.last_discourse_touches==16 and len(adult.discourse_credit.candidates(generic))==16)
    checks['no_partner_topic_answer_or_paragraph_map']=(
        not hasattr(adult,'partner_topics') and not hasattr(adult,'partner_answers')
        and not hasattr(adult,'paragraph_planner'))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0

    failed=[name for name,value in checks.items() if not value]
    result={
        'contract':'FOUNDRY_PARTNER_SPECIFIC_DISCOURSE_SELECTION_GREEN',
        'reference_only':True,'graph_flip':False,
        'language_phenotype_improved':True,
        'visible_language_gain':'PARTNER_HISTORY_SELECTS_DISTINCT_LONG_FORM_MATTER_WITH_GENERIC_TRANSFER',
        'generic_context':generic,'partner_contexts':{'a':pa,'b':pb,'novel':test_pu},
        'selected':{'a':len(selected_a),'b':len(selected_b),'novel':len(selected_u)},
        'visible_bytes':{'a':len(surface_a),'b':len(surface_b),'novel':len(surface_u)},
        'checks':checks,
        'remaining_red':['DIRECT_AUTHENTICATED_PARTNER_DISCOURSE_PARITY',
                         'CONTINUOUS_LIFE_LONG_DISCOURSE_INTERFERENCE',
                         'LONGER_BODY_HISTORY_INTERACTION','COPIED_SOURCE_DEPENDENCE'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    if failed:
        print('FOUNDRY_PARTNER_SPECIFIC_DISCOURSE_SELECTION_RED '+','.join(failed))
        print(json.dumps(result,sort_keys=True,indent=2));return 1
    print('FOUNDRY_PARTNER_SPECIFIC_DISCOURSE_SELECTION_GREEN')
    print(json.dumps(result,sort_keys=True,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
