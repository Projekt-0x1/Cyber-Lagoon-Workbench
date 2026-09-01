#!/usr/bin/env python3
"""Actual nonlinguistic world occurrence becomes the current long-discourse situation."""
from __future__ import annotations
import copy,inspect,json,time
from reference_global_discourse_relevance_verify import fresh
from reference_language_mastery_adult_v1 import AdultStateV1,LanguageMasteryAdultV1
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE,CONTACT_WITHDRAW_SOURCE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_slow_resource_history_v1 import LOAD_SAMPLE_CAP_Q16,SUSTAINED_MIN_CONTACTS,HISTORY_WINDOW_TICKS
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1

WORLD_A=(201,303,403);WORLD_B=(202,302,402)
SOURCE_A=0xA701;SOURCE_B=0xA702;BODY='world-discourse-body'


def digest(n):return format(int(n),'064x')[-64:]


def world(o,state,source,independent=True):
    o.contact(CONTACT_WORLD_STATE,tuple(state),int(source),True,bool(independent))
    return int(o.world_state_occurrence)


def activate_world(adult,o):
    return WorldDiscourseSituationBridgeV1.activate(adult,o)


def learn_relevance_from_current_world(adult,frontier,parity,outcome=Q,effort=Q//16,duration=1):
    selected=[]
    if not adult._current_selection_context:raise RuntimeError('world_discourse:no_current_world_situation')
    for i,leaf in enumerate(frontier):
        if i%2!=parity:continue
        selected.append(leaf.identity)
        for _ in range(2):
            adult.experience_discourse_candidate(
                leaf.identity,outcome,context=None,effort_q16=effort,duration=duration)
            adult.experience_discourse_background(leaf.identity,False)
    return tuple(selected)


def current_discourse(adult,frontier,state=AdultStateV1()):
    root=adult.organize_relevant_frontier(frontier,state)
    return root,tuple(getattr(adult,'last_discourse_selected',()))


def main():
    started=time.perf_counter();checks={}
    adult,frontier,factors,_ctx_a,_ctx_b=fresh()
    organism=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    programs_before=len(adult.programs.chunks)

    # Development is through actual world contacts. The fixture never names the
    # resulting discourse context in the relevance update.
    occ_a1=world(organism,WORLD_A,SOURCE_A,True);ctx_a=activate_world(adult,organism)
    expected_a=learn_relevance_from_current_world(adult,frontier,0)
    occ_b1=world(organism,WORLD_B,SOURCE_B,True);ctx_b=activate_world(adult,organism)
    expected_b=learn_relevance_from_current_world(adult,frontier,1)
    checks['different_lived_world_states_create_distinct_stable_situations']=(ctx_a>0 and ctx_b>0 and ctx_a!=ctx_b)

    # Held-out reentry has no language cue/topic id.
    occ_a2=world(organism,WORLD_A,SOURCE_A,True);re_ctx_a=activate_world(adult,organism)
    root_a,selected_a=current_discourse(adult,frontier)
    occ_b2=world(organism,WORLD_B,SOURCE_B,True);re_ctx_b=activate_world(adult,organism)
    root_b,selected_b=current_discourse(adult,frontier)
    checks['actual_world_a_selects_a_long_form_without_language_topic_cue']=(
        root_a is not None and re_ctx_a==ctx_a and selected_a==expected_a and len(selected_a)==8)
    checks['actual_world_b_selects_b_long_form_without_language_topic_cue']=(
        root_b is not None and re_ctx_b==ctx_b and selected_b==expected_b and len(selected_b)==8)
    checks['world_swap_changes_visible_multi_sentence_language']=(
        tuple(root_a.surface)!=tuple(root_b.surface) and len(root_a.surface)>300 and len(root_b.surface)>300)

    # Repeated same-world perception must mint a new occurrence but preserve the
    # stable situation coordinate and phenotype.
    occ_a3=world(organism,WORLD_A,SOURCE_A,True);repeat_ctx=activate_world(adult,organism)
    repeat_root,repeat_selected=current_discourse(adult,frontier)
    checks['new_world_occurrence_same_state_reuses_stable_situation']=(
        occ_a3 not in (occ_a1,occ_a2) and repeat_ctx==ctx_a
        and repeat_root is not None and repeat_selected==expected_a)

    # Yoked perception is still a present perception, but cannot by itself write
    # durable world revision or discourse relevance.
    yoked_org=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));yoked_adult,_,_,_,_=fresh()
    world(yoked_org,WORLD_A,SOURCE_A,False);yoked_ctx=activate_world(yoked_adult,yoked_org)
    yoked_root,yoked_selected=current_discourse(yoked_adult,frontier)
    checks['yoked_world_can_be_current_but_does_not_teach_discourse']=(
        yoked_ctx==ctx_a and yoked_org._world_revisions.row_count==0
        and yoked_root is None and not yoked_selected
        and not yoked_adult.discourse_credit.candidates(yoked_ctx))

    # No current world means no current discourse situation.
    empty_org=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));empty_adult=LanguageMasteryAdultV1.restore(copy.deepcopy(adult.checkpoint()))
    empty_ctx=activate_world(empty_adult,empty_org);empty_root,empty_selected=current_discourse(empty_adult,frontier)
    checks['no_current_world_refuses_discourse_situation']=(empty_ctx==0 and empty_root is None and not empty_selected)

    # Source withdrawal clears organism authority before bridge activation.
    withdrawn_org=ReferenceOrganismV2.restore(copy.deepcopy(organism.checkpoint()));withdrawn_adult=LanguageMasteryAdultV1.restore(copy.deepcopy(adult.checkpoint()))
    withdrawn_org.contact(CONTACT_WITHDRAW_SOURCE,(SOURCE_A,),0xA7F0,True,True)
    withdrawn_ctx=activate_world(withdrawn_adult,withdrawn_org)
    checks['withdrawing_active_world_source_blocks_bridge']=(
        withdrawn_org.world_state is None and withdrawn_org.world_state_occurrence==0
        and withdrawn_ctx==0 and not withdrawn_adult.select_discourse_frontier(frontier))

    # Checkpoint owns world only in the organism; Adult rematerializes situation from it.
    cp_org=ReferenceOrganismV2.restore(copy.deepcopy(organism.checkpoint()));cp_adult=LanguageMasteryAdultV1.restore(copy.deepcopy(adult.checkpoint()))
    cp_world=tuple(cp_org.world_state) if cp_org.world_state is not None else None
    cp_occ=int(cp_org.world_state_occurrence);cp_ctx=activate_world(cp_adult,cp_org);cp_root,cp_selected=current_discourse(cp_adult,frontier)
    checks['organism_checkpoint_world_reentry_rematerializes_discourse_situation']=(
        cp_world==WORLD_A and cp_occ==occ_a3 and cp_ctx==ctx_a
        and cp_root is not None and cp_selected==expected_a)
    adult_cp=cp_adult.checkpoint()
    checks['language_adult_checkpoint_does_not_duplicate_world_state']=(
        all(k not in adult_cp for k in ('world_state','world_source','world_state_occurrence','world_signature')))

    # Focal A relevance lesion: world remains current and relation operators remain,
    # while only A proposition selection disappears. B remains exact.
    lesioned=LanguageMasteryAdultV1.restore(copy.deepcopy(adult.checkpoint()))
    for row in lesioned.discourse_credit.rows.values():row.contexts.pop(ctx_a,None)
    lesioned.discourse_credit.context_members.pop(ctx_a,None)
    world(organism,WORLD_A,SOURCE_A,True);activate_world(lesioned,organism)
    cut_root,cut_selected=current_discourse(lesioned,frontier)
    world(organism,WORLD_B,SOURCE_B,True);activate_world(lesioned,organism)
    b_after_root,b_after_selected=current_discourse(lesioned,frontier)
    checks['focal_world_relevance_lesion_drops_a_not_b_or_relation_learning']=(
        cut_root is None and not cut_selected
        and b_after_root is not None and b_after_selected==expected_b
        and all(lesioned.organization_credit.candidates(leaf.identity) for leaf in frontier))

    # Slow resource history cannot invent relevance. A fresh Adult with current world
    # and sustained load still has no topic; learned A retains only learned A matter.
    resource_org=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));resource_adult=LanguageMasteryAdultV1.restore(copy.deepcopy(adult.checkpoint()))
    world(resource_org,WORLD_A,SOURCE_A,True);activate_world(resource_adult,resource_org)
    for seq in range(1,SUSTAINED_MIN_CONTACTS+1):
        resource_adult.settle_body_ingress(BODY,seq,digest(seq),LOAD_SAMPLE_CAP_Q16)
    loaded_selected=tuple(x.identity for x in resource_adult.select_discourse_frontier(frontier))
    checks['resource_history_cannot_author_world_topic_content']=(
        set(loaded_selected).issubset(set(expected_a)) and all(x in expected_a for x in loaded_selected))
    no_credit=LanguageMasteryAdultV1();activate_world(no_credit,resource_org)
    checks['resource_plus_world_without_relevance_history_stays_silent']=(
        not no_credit.select_discourse_frontier(frontier))

    bridge_source=inspect.getsource(WorldDiscourseSituationBridgeV1)
    checks['bridge_is_stateless_and_has_no_semantic_authority']=(
        not hasattr(WorldDiscourseSituationBridgeV1(),'__dict__') or not WorldDiscourseSituationBridgeV1().__dict__
        and all(token not in bridge_source for token in ('discourse_credit','partner_credit','expected','paragraph','topic_table')))
    checks['no_complete_paragraph_program_persisted_by_world_reentry']=(len(adult.programs.chunks)==programs_before)
    checks['bounded_reference_work']=time.perf_counter()-started<1.0

    failed=[k for k,v in checks.items() if not v]
    result={
        'contract':'FOUNDRY_LIVED_WORLD_DISCOURSE_SITUATION_GREEN',
        'reference_only':True,'graph_flip':False,
        'world_occurrences':{'a_first':occ_a1,'a_second':occ_a2,'a_third':occ_a3,'b_first':occ_b1,'b_second':occ_b2},
        'world_contexts':{'a':ctx_a,'b':ctx_b},
        'selected':{'a':len(selected_a),'b':len(selected_b),'resource_a':len(loaded_selected)},
        'checks':checks,'failed':failed,
        'remaining_red':['AUTONOMOUS_PROPOSITION_FRONTIER_RECRUITMENT_FROM_WORLD','DIRECT_WORLD_DISCOURSE_PARITY','UNBOUNDED_WORLD_TOPIC_FORMATION'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    if failed:
        print('FOUNDRY_LIVED_WORLD_DISCOURSE_SITUATION_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));return 1
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
