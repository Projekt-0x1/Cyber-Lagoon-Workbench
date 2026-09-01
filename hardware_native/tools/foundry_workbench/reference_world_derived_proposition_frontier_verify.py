#!/usr/bin/env python3
"""Current lived world recruits proposition frontier without a host candidate list."""
from __future__ import annotations
import copy,inspect,json,time
from reference_global_discourse_relevance_verify import fresh
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE,CONTACT_WITHDRAW_SOURCE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_slow_resource_history_v1 import LOAD_SAMPLE_CAP_Q16,SUSTAINED_MIN_CONTACTS
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1

# All proposition constituents are already grounded/productive in the Adult.  A/B
# worlds differ only in which two agents are physically present; both also contain
# one adjective, both learned verbs and both objects. Each therefore entails 8 of the
# first 16 proposition leaves. C entails all 16 but has no learned relevance history.
WORLD_A=(101,201,202,301,302,401,402)
WORLD_B=(101,203,204,301,302,401,402)
WORLD_C=(101,201,202,203,204,301,302,401,402)
WORLD_A_MINUS_VALVE=(101,201,202,301,302,401)
SOURCE_A=0xB701;SOURCE_B=0xB702;SOURCE_C=0xB703;BODY='world-frontier-body'


def digest(n):return format(int(n),'064x')[-64:]


def world(o,state,source):
    o.contact(CONTACT_WORLD_STATE,tuple(state),int(source),True,True)
    return int(o.world_state_occurrence)


def activate(adult,o):return WorldDiscourseSituationBridgeV1.activate_frontier(adult,o)


def train_current_world(adult,frontier,outcome=Q,effort=Q//16,duration=1):
    if not adult._current_selection_context:raise RuntimeError('world_frontier:no_situation')
    for leaf in frontier:
        for _ in range(2):
            adult.experience_discourse_candidate(
                leaf.identity,outcome,context=None,effort_q16=effort,duration=duration)
            adult.experience_discourse_background(leaf.identity,False)
    return tuple(leaf.identity for leaf in frontier)


def express(adult,frontier):
    root=adult.organize_relevant_frontier(frontier)
    return root,tuple(getattr(adult,'last_discourse_selected',()))


def main():
    started=time.perf_counter();checks={}
    adult,host_frontier,factors,_ca,_cb=fresh()
    organism=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    first16=tuple(x.identity for x in host_frontier)
    programs_before=len(adult.programs.chunks)

    # Development: both the discourse context and the candidate proposition frontier
    # come from actual world state. No host frontier is passed to the bridge.
    world(organism,WORLD_A,SOURCE_A);ctx_a,frontier_a=activate(adult,organism)
    expected_a=train_current_world(adult,frontier_a)
    world(organism,WORLD_B,SOURCE_B);ctx_b,frontier_b=activate(adult,organism)
    expected_b=train_current_world(adult,frontier_b)
    checks['world_a_mechanically_recruits_exact_eight_productive_propositions']=(
        len(frontier_a)==8 and set(expected_a).issubset(set(first16)))
    checks['world_b_mechanically_recruits_complementary_eight_productive_propositions']=(
        len(frontier_b)==8 and set(expected_b).issubset(set(first16))
        and set(expected_a).isdisjoint(set(expected_b))
        and set(expected_a)|set(expected_b)==set(first16))

    # Held-out reentry: only world contact is supplied. The same bridge derives both
    # situation and frontier, then resident relevance + local relation competition speak.
    world(organism,WORLD_A,SOURCE_A);re_ctx_a,re_frontier_a=activate(adult,organism)
    root_a,selected_a=express(adult,re_frontier_a)
    world(organism,WORLD_B,SOURCE_B);re_ctx_b,re_frontier_b=activate(adult,organism)
    root_b,selected_b=express(adult,re_frontier_b)
    checks['heldout_world_a_speaks_without_host_frontier_or_topic_id']=(
        re_ctx_a==ctx_a and tuple(x.identity for x in re_frontier_a)==expected_a
        and root_a is not None and selected_a==expected_a)
    checks['heldout_world_b_speaks_without_host_frontier_or_topic_id']=(
        re_ctx_b==ctx_b and tuple(x.identity for x in re_frontier_b)==expected_b
        and root_b is not None and selected_b==expected_b)
    checks['world_frontier_swap_changes_visible_long_form_language']=(
        tuple(root_a.surface)!=tuple(root_b.surface) and len(root_a.surface)>300 and len(root_b.surface)>300)

    # Entailment alone is not semantic importance: C mechanically recruits all 16,
    # but no C relevance history means silence.
    world(organism,WORLD_C,SOURCE_C);ctx_c,frontier_c=activate(adult,organism)
    root_c,selected_c=express(adult,frontier_c)
    checks['broader_world_recruits_sixteen_but_without_relevance_history_stays_silent']=(
        len(frontier_c)==16 and set(x.identity for x in frontier_c)==set(first16)
        and root_c is None and not selected_c and not adult.discourse_credit.candidates(ctx_c))

    # Remove one constituent: every valve proposition must disappear mechanically,
    # while sensor propositions remain. No discourse-credit intervention is involved.
    world(organism,WORLD_A_MINUS_VALVE,SOURCE_A);_,frontier_cut=activate(adult,organism)
    cut_ids=tuple(x.identity for x in frontier_cut)
    sensor_a=tuple(x.identity for x in frontier_a if b'sensor' in bytes(x.surface))
    checks['removing_world_atom_mechanically_removes_only_dependent_propositions']=(
        len(frontier_cut)==4 and cut_ids==sensor_a
        and all(b'valve' not in bytes(x.surface) for x in frontier_cut))

    # Focal relevance lesion leaves the mechanical world frontier exact but removes
    # selection. This dissociates current availability from learned importance.
    lesioned=LanguageMasteryAdultV1.restore(copy.deepcopy(adult.checkpoint()))
    for row in lesioned.discourse_credit.rows.values():row.contexts.pop(ctx_a,None)
    lesioned.discourse_credit.context_members.pop(ctx_a,None)
    world(organism,WORLD_A,SOURCE_A);_,lesion_frontier=activate(lesioned,organism)
    lesion_root,lesion_selected=express(lesioned,lesion_frontier)
    checks['relevance_lesion_changes_selection_not_world_frontier']=(
        tuple(x.identity for x in lesion_frontier)==expected_a
        and lesion_root is None and not lesion_selected)

    # Focal productive-leaf deletion changes the frontier while world state remains.
    productive_cut=LanguageMasteryAdultV1.restore(copy.deepcopy(adult.checkpoint()))
    target_identity=expected_a[0];removed=False
    for template_identity,family in productive_cut._surface_leaf_families.items():
        if target_identity in family:
            family.pop(target_identity,None);removed=True;break
    productive_cut._surface_leaf_family_index.pop(target_identity,None)
    if not removed:raise RuntimeError('world_frontier:productive_owner_missing')
    world(organism,WORLD_A,SOURCE_A);_,product_frontier=activate(productive_cut,organism)
    checks['productive_leaf_lesion_changes_frontier_not_world_state']=(
        organism.world_state==WORLD_A and len(product_frontier)==7
        and expected_a[0] not in tuple(x.identity for x in product_frontier))

    # Source withdrawal/no world empties both situation and frontier.
    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(organism.checkpoint()))
    withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(SOURCE_A,),0xB7F0,True,True)
    withdrawn_adult=LanguageMasteryAdultV1.restore(copy.deepcopy(adult.checkpoint()))
    withdrawn_ctx,withdrawn_frontier=activate(withdrawn_adult,withdrawn)
    checks['world_source_withdrawal_empties_frontier']=(
        withdrawn.world_state is None and withdrawn_ctx==0 and not withdrawn_frontier)

    # Checkpoint rematerializes both context and frontier; neither list is checkpointed.
    world(organism,WORLD_B,SOURCE_B);checkpoint_org=copy.deepcopy(organism.checkpoint())
    checkpoint_adult=copy.deepcopy(adult.checkpoint())
    restored_org=ReferenceOrganismV2.restore(checkpoint_org);restored_adult=LanguageMasteryAdultV1.restore(checkpoint_adult)
    restored_ctx,restored_frontier=activate(restored_adult,restored_org)
    restored_root,restored_selected=express(restored_adult,restored_frontier)
    checks['checkpoint_reentry_rematerializes_world_frontier']=(
        restored_ctx==ctx_b and tuple(x.identity for x in restored_frontier)==expected_b
        and restored_root is not None and restored_selected==expected_b)
    checkpoint_text=json.dumps(checkpoint_adult,sort_keys=True)
    checks['adult_checkpoint_has_no_topic_or_frontier_list']=(
        all(token not in checkpoint_text for token in ('topic_frontier','world_frontier','paragraph_plan')))

    # Slow resource history may suppress learned matter but cannot add anything that
    # the current world did not mechanically recruit.
    resource=LanguageMasteryAdultV1.restore(copy.deepcopy(adult.checkpoint()))
    world(organism,WORLD_A,SOURCE_A);_,resource_frontier=activate(resource,organism)
    for seq in range(1,SUSTAINED_MIN_CONTACTS+1):
        resource.settle_body_ingress(BODY,seq,digest(seq),LOAD_SAMPLE_CAP_Q16)
    resource_selected=tuple(x.identity for x in resource.select_discourse_frontier(resource_frontier))
    checks['resource_history_cannot_add_unrecruited_propositions']=(
        set(resource_selected).issubset(set(expected_a)))

    bridge_source=inspect.getsource(WorldDiscourseSituationBridgeV1.frontier)
    checks['frontier_bridge_has_no_relevance_or_answer_authority']=(
        all(token not in bridge_source for token in ('discourse_credit','expected','score','topic','paragraph','partner')))
    checks['no_complete_paragraph_program_persisted']=(len(adult.programs.chunks)==programs_before)
    checks['bounded_reference_work']=time.perf_counter()-started<1.0

    failed=[k for k,v in checks.items() if not v]
    result={
        'contract':'FOUNDRY_WORLD_DERIVED_PROPOSITION_FRONTIER_GREEN',
        'reference_only':True,'graph_flip':False,
        'contexts':{'a':ctx_a,'b':ctx_b,'c':ctx_c},
        'frontiers':{'a':len(frontier_a),'b':len(frontier_b),'c':len(frontier_c),'a_minus_valve':len(frontier_cut)},
        'selected':{'a':len(selected_a),'b':len(selected_b),'resource_a':len(resource_selected)},
        'checks':checks,'failed':failed,
        'remaining_red':['RAW_SENSORY_GROUNDING_OF_SHARED_CONCEPT_IDENTITIES','DIRECT_WORLD_FRONTIER_PARITY','OPEN_WORLD_ENTITY_DISCOVERY'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    if failed:
        print('FOUNDRY_WORLD_DERIVED_PROPOSITION_FRONTIER_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));return 1
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
