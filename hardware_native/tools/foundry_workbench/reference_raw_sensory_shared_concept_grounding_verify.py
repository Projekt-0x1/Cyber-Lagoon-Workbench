#!/usr/bin/env python3
"""Raw world features converge on independently learned language concepts through lived consequence."""
from __future__ import annotations
import copy,inspect,json,time
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1

RAW={101:9101,201:9201,202:9202,203:9203,204:9204,301:9301,302:9302,401:9401,402:9402}
SURFACE={101:b'careful',201:b'engineer',202:b'technician',203:b'operator',204:b'analyst',301:b'tests',302:b'inspects',401:b'sensor',402:b'valve'}
WORLD_A=tuple(RAW[x] for x in (101,201,202,301,302,401,402))
WORLD_B=tuple(RAW[x] for x in (101,203,204,301,302,401,402))
SOURCE=0xCA01


def world(o,raw,source=SOURCE):
    o.contact(CONTACT_WORLD_STATE,tuple(raw),int(source),True,True)
    return int(o.world_state_occurrence)


def pair(g,adult,o,raw,surface,source,effect=1,independent=True):
    world(o,(raw,),source)
    g.observe_world(o)
    feature=g.observe_language_surface(adult,surface)
    relation=g.settle_current_pair(source+1000,effect,independent)
    return feature,relation


def train_mapping(g,adult,o,concept,source):
    for n in range(2):
        feature,relation=pair(g,adult,o,RAW[concept],SURFACE[concept],source+n)
        if feature!=concept or relation<=0:raise RuntimeError('raw_grounding:training')


def train_world_relevance(adult,g,o,raw_world):
    world(o,raw_world,SOURCE)
    ctx,frontier=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
    for leaf in frontier:
        for _ in range(2):
            adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1)
            adult.experience_discourse_background(leaf.identity,False)
    return ctx,tuple(x.identity for x in frontier)


def express(adult,g,o,raw_world):
    world(o,raw_world,SOURCE)
    ctx,frontier=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
    root=adult.organize_relevant_frontier(frontier)
    return ctx,frontier,root,tuple(getattr(adult,'last_discourse_selected',()))


def main():
    started=time.perf_counter();checks={}
    adult,host_frontier,_factors,_ca,_cb=fresh()
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))

    # Modalities begin with different opaque identities.
    checks['raw_and_language_identities_are_not_pre_aligned']=all(RAW[c]!=c for c in RAW)

    one=CrossmodalConceptGroundingV1();pair(one,adult,o,RAW[101],SURFACE[101],0xCB10)
    checks['one_joint_episode_is_not_durable_concept_authority']=(one.resolve_raw_feature(adult,RAW[101])==0)

    yoked=CrossmodalConceptGroundingV1()
    for n in range(2):pair(yoked,adult,o,RAW[101],SURFACE[101],0xCB20+n,independent=False)
    checks['non_independent_joint_returns_cannot_ground_concept']=(yoked.resolve_raw_feature(adult,RAW[101])==0)

    g=CrossmodalConceptGroundingV1()
    for concept in RAW:train_mapping(g,adult,o,concept,0xCC00+concept*4)
    checks['repeated_independent_crossmodal_history_resolves_raw_to_language_concepts']=all(
        g.resolve_raw_feature(adult,RAW[c])==c for c in RAW)

    # Equal learned support for one raw feature and two concepts refuses.
    ambiguous=CrossmodalConceptGroundingV1()
    for n in range(2):pair(ambiguous,adult,o,RAW[101],SURFACE[101],0xCD10+n)
    for n in range(2):pair(ambiguous,adult,o,RAW[101],SURFACE[201],0xCD20+n)
    checks['equal_crossmodal_support_refuses_arbitrary_concept']=(ambiguous.resolve_raw_feature(adult,RAW[101])==0)

    # Reversal/reacquisition changes only bridge history.
    base_language=json.dumps(adult.language.checkpoint(),sort_keys=True,separators=(',',':'))
    raw_before=tuple(o.world_state) if o.world_state is not None else None
    before=g.resolve_raw_feature(adult,RAW[101])
    for n in range(3):pair(g,adult,o,RAW[101],SURFACE[101],0xCE10+n,effect=-1)
    after=g.resolve_raw_feature(adult,RAW[101])
    for n in range(3):pair(g,adult,o,RAW[101],SURFACE[101],0xCE20+n,effect=1)
    reacquired=g.resolve_raw_feature(adult,RAW[101])
    checks['recent_adverse_history_reverses_then_reacquires_bridge']=(before==101 and after==0 and reacquired==101)
    checks['bridge_revision_does_not_rewrite_language_ecology']=(json.dumps(adult.language.checkpoint(),sort_keys=True,separators=(',',':'))==base_language)

    # Checkpoint keeps learned bridges, never active modality occurrences.
    cp=copy.deepcopy(g.checkpoint());restored=CrossmodalConceptGroundingV1.restore(cp)
    checks['checkpoint_preserves_bridge_but_not_active_occurrences']=(
        restored.resolve_raw_feature(adult,RAW[101])==101
        and not restored._current_world and restored._current_language is None)

    # Focal raw-side lesion preserves words/concepts and unrelated bridges.
    lesioned=CrossmodalConceptGroundingV1.restore(cp);lesioned.lesion_raw_feature(RAW[101])
    checks['focal_bridge_lesion_preserves_modalities_and_unrelated_grounding']=(
        lesioned.resolve_raw_feature(adult,RAW[101])==0
        and lesioned.resolve_raw_feature(adult,RAW[201])==201
        and adult.language.lexeme(101)==tuple(SURFACE[101]))

    # Later unrelated bridges do not overwrite the early bridge.
    interference=CrossmodalConceptGroundingV1.restore(cp)
    for i,concept in enumerate((201,202,203,204,301,302,401,402)):
        # New raw views of the same independently learned concepts.
        raw=0xD000+i
        for n in range(2):pair(interference,adult,o,raw,SURFACE[concept],0xD100+i*4+n)
    checks['later_crossmodal_learning_does_not_overwrite_early_bridge']=(
        interference.resolve_raw_feature(adult,RAW[101])==101 and interference.last_touches<=2)

    # Strong phenotype: raw-feature worlds cannot recruit proposition leaves without
    # grounding. After grounding, A/B recruit complementary productive frontiers.
    no_ground_o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));world(no_ground_o,WORLD_A)
    _ctx0,frontier0=WorldDiscourseSituationBridgeV1.activate_frontier(adult,no_ground_o)
    checks['raw_world_without_crossmodal_grounding_recruits_no_language_propositions']=(not frontier0)

    discourse=type(adult).restore(copy.deepcopy(adult.checkpoint()))
    dg=CrossmodalConceptGroundingV1.restore(cp)
    do=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    ctx_a,expected_a=train_world_relevance(discourse,dg,do,WORLD_A)
    ctx_b,expected_b=train_world_relevance(discourse,dg,do,WORLD_B)
    ra,fa,root_a,selected_a=express(discourse,dg,do,WORLD_A)
    rb,fb,root_b,selected_b=express(discourse,dg,do,WORLD_B)
    first16=set(x.identity for x in host_frontier)
    checks['grounded_raw_world_a_recruits_and_speaks_eight_propositions']=(
        ra==ctx_a and len(fa)==8 and selected_a==expected_a and root_a is not None
        and set(expected_a).issubset(first16))
    checks['grounded_raw_world_b_recruits_complementary_eight_propositions']=(
        rb==ctx_b and len(fb)==8 and selected_b==expected_b and root_b is not None
        and set(expected_a).isdisjoint(set(expected_b)) and set(expected_a)|set(expected_b)==first16)
    checks['raw_world_swap_changes_visible_long_form_language']=(tuple(root_a.surface)!=tuple(root_b.surface))

    # Grounding checkpoint + Adult/organism checkpoint reproduce raw-world discourse.
    cp_d=copy.deepcopy(discourse.checkpoint());cp_o=copy.deepcopy(do.checkpoint());cp_g=copy.deepcopy(dg.checkpoint())
    replay_d=type(discourse).restore(cp_d);replay_o=ReferenceOrganismV2.restore(cp_o);replay_g=CrossmodalConceptGroundingV1.restore(cp_g)
    _rc,_rf,replay_root,replay_selected=express(replay_d,replay_g,replay_o,WORLD_B)
    checks['three_owner_checkpoint_replays_grounded_world_discourse']=(replay_root is not None and replay_selected==expected_b)

    source=inspect.getsource(CrossmodalConceptGroundingV1)
    checks['plastic_settlement_has_no_mapping_pair_arguments']=(
        list(inspect.signature(CrossmodalConceptGroundingV1.settle_current_pair).parameters)==['self','source','effect','independent'])
    checks['bridge_has_no_symbolic_mapping_table']=(
        all(token not in source for token in ('concept_map','feature_map','world_to_concept','expected_concept')))
    checks['bounded_reference_work']=time.perf_counter()-started<1.0

    failed=[k for k,v in checks.items() if not v]
    result={
        'contract':'FOUNDRY_RAW_SENSORY_SHARED_CONCEPT_GROUNDING_GREEN',
        'reference_only':True,'graph_flip':False,
        'resolved_examples':{str(RAW[101]):restored.resolve_raw_feature(adult,RAW[101]),str(RAW[401]):restored.resolve_raw_feature(adult,RAW[401])},
        'frontiers':{'a':len(fa),'b':len(fb)},'checks':checks,'failed':failed,
        'remaining_red':['RAW_PIXEL_AUDIO_FEATURE_EXTRACTION','OPEN_WORLD_PERCEPTUAL_CATEGORY_DISCOVERY','DIRECT_CROSSMODAL_GROUNDING_PARITY'],
        'elapsed_ms':round((time.perf_counter()-started)*1000,3),
    }
    if failed:
        print('FOUNDRY_RAW_SENSORY_SHARED_CONCEPT_GROUNDING_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));return 1
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
