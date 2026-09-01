#!/usr/bin/env python3
"""Unlabeled temporal adjacency over raw visual views supports grounded discourse."""
from __future__ import annotations
import copy,inspect,json,time
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_open_world_perceptual_category_grounding_verify import SURFACE,pair
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_raw_visual_elements_v1 import RawVisualElementsV1
from reference_raw_visual_signal_feature_extraction_verify import edge
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_unsupervised_perceptual_features_v1 import FEATURE_QUORUM,UnsupervisedPerceptualFeaturesV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1

A1=edge(1,1,64);A2=edge(1,-1,64);B1=edge(2,1,64);B2=edge(2,-1,64)
C1=edge(1,1,96);C2=edge(2,-1,96)
A=0x9101;B=0x9102;NOVEL=0x9201;OTHER=0x9202
FEATURE_SOURCE=0x9301;WORLD_SOURCE=0x9302
RAW_ADJ=0x9401;RAW_TEST=0x9402;RAW_INSPECT=0x9403;RAW_SENSOR=0x9404;RAW_VALVE=0x9405
DIRECT=((RAW_ADJ,101),(RAW_TEST,301),(RAW_INSPECT,302),(RAW_SENSOR,401),(RAW_VALVE,402))


def patches(image):return RawVisualElementsV1.pair_patches(image)


def train_level1():
    learner=UnsupervisedPerceptualFeaturesV1()
    for image in (A1,A2,B1,B2,C1,C2):
        row=patches(image)
        if len(row)!=1:raise RuntimeError('temporal_visual:primitive_shape')
        for _ in range(FEATURE_QUORUM):learner.observe_scene(row[0])
    return learner


def view_feature(level1,image):
    row=patches(image)
    if len(row)!=1:return 0
    return int(level1.feature(*row[0]))


def pair_sequence(adapter,level1,left,right,repeats=FEATURE_QUORUM):
    for _ in range(repeats):
        adapter.gap();adapter.observe_features((view_feature(level1,left),));adapter.observe_features((view_feature(level1,right),))
    adapter.gap()


def install(o,entity,features,source=FEATURE_SOURCE):
    row=tuple(map(int,features))
    if not row:raise RuntimeError('temporal_visual:no_entity_features')
    o.contact(CONTACT_ENTITY_FEATURES,(int(entity),len(row),*row),int(source),True,True)


def ground(g,adult,o,entity,concept,source):
    for n in range(2):pair(g,adult,o,entity,SURFACE[concept],source+n)


def train_direct(g,adult,o):
    for i,(raw,concept) in enumerate(DIRECT):
        for n in range(2):pair(g,adult,o,raw,SURFACE[concept],0x9500+i*4+n)


def world(o,atoms):o.contact(CONTACT_WORLD_STATE,tuple(map(int,atoms)),WORLD_SOURCE,True,True)


def main():
    started=time.perf_counter();checks={};level1=train_level1()
    fa1,fa2,fb1,fb2=(view_feature(level1,x) for x in (A1,A2,B1,B2))
    checks['view_specific_lower_features_are_distinct_before_temporal_learning']=(len({fa1,fa2,fb1,fb2})==4 and all((fa1,fa2,fb1,fb2)))

    one=TemporalVisualContinuityV1();one.observe_features((fa1,));one.observe_features((fa2,))
    checks['one_adjacent_view_pair_is_insufficient']=(one.relation(fa1,fa2)==0)

    coherent=TemporalVisualContinuityV1();pair_sequence(coherent,level1,A1,A2);pair_sequence(coherent,level1,B1,B2)
    ta=coherent.relation(fa1,fa2);tb=coherent.relation(fb1,fb2)
    checks['coherent_temporal_view_history_creates_distinct_invariant_features']=(ta>0 and tb>0 and ta!=tb)

    separated=TemporalVisualContinuityV1()
    for _ in range(FEATURE_QUORUM):
        separated.observe_features((fa1,));separated.gap();separated.observe_features((fa2,));separated.gap()
    checks['same_views_separated_by_gaps_do_not_create_temporal_relation']=(separated.relation(fa1,fa2)==0)

    shuffled=TemporalVisualContinuityV1()
    for _ in range(FEATURE_QUORUM):
        shuffled.gap();shuffled.observe_features((fa1,));shuffled.observe_features((fb2,))
        shuffled.gap();shuffled.observe_features((fb1,));shuffled.observe_features((fa2,))
    checks['same_frame_marginals_with_cross_object_adjacency_do_not_fabricate_a_relation']=(
        shuffled.relation(fa1,fa2)==0 and shuffled.relation(fa1,fb2)>0 and shuffled.relation(fb1,fa2)>0)

    checks['reversed_view_order_reactivates_same_temporal_relation']=(coherent.relation(fa2,fa1)==ta)
    fc1=view_feature(level1,C1)
    checks['replacing_one_view_changes_temporal_relation']=(fc1 not in (0,fa2) and coherent.relation(fa1,fc1)==0)

    lower_cut=UnsupervisedPerceptualFeaturesV1.restore(copy.deepcopy(level1.checkpoint()));lower_cut.lesion_pair(*patches(A1)[0])
    checks['lower_view_lesion_prevents_temporal_activation_preserves_other_view']=(
        view_feature(lower_cut,A1)==0 and view_feature(lower_cut,A2)==fa2
        and coherent.relation(view_feature(lower_cut,A1),fa2)==0)

    temporal_cut=TemporalVisualContinuityV1.restore(copy.deepcopy(coherent.checkpoint()));temporal_cut.lesion_relation(fa1,fa2)
    checks['temporal_relation_lesion_preserves_both_lower_views']=(
        temporal_cut.relation(fa1,fa2)==0 and view_feature(level1,A1)==fa1 and view_feature(level1,A2)==fa2)

    restored=TemporalVisualContinuityV1.restore(copy.deepcopy(coherent.checkpoint()))
    checks['checkpoint_preserves_temporal_relation_not_previous_frame']=(restored.relation(fa1,fa2)==ta and not restored.previous)
    pair_sequence(restored,level1,C1,C2)
    checks['later_unrelated_temporal_learning_does_not_overwrite_early_relation']=(restored.relation(fa1,fa2)==ta)
    checks['temporal_learning_has_zero_consequence_credit']=(
        coherent.learner.population.credit_events==0 and coherent.learner.population.revision_events==0
        and not coherent.learner.population.occurrences)

    # Visible downstream consequence: entity geometry is temporal-sequence-derived only.
    adult,host_frontier,_f,_ca,_cb=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1()
    install(o,A,(ta,));install(o,B,(tb,));install(o,NOVEL,(ta,));install(o,OTHER,(tb,))
    ground(g,adult,o,A,201,0x9600);ground(g,adult,o,B,203,0x9610)
    checks['novel_coherent_sequence_inherits_grounded_concept_without_language_pair']=(
        g.resolve_raw_feature(adult,NOVEL)==0 and g.resolve_world_atom(adult,o,NOVEL)==201)
    checks['other_temporal_sequence_remains_distinct']=(g.resolve_world_atom(adult,o,OTHER)==203)

    train_direct(g,adult,o)
    world(o,(RAW_ADJ,NOVEL,RAW_TEST,RAW_INSPECT,RAW_SENSOR,RAW_VALVE))
    ctx,frontier=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
    for leaf in frontier:
        for _ in range(2):
            adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1)
            adult.experience_discourse_background(leaf.identity,False)
    world(o,(RAW_ADJ,NOVEL,RAW_TEST,RAW_INSPECT,RAW_SENSOR,RAW_VALVE))
    re_ctx,re_frontier=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
    root=adult.organize_relevant_frontier(re_frontier);selected=tuple(adult.last_discourse_selected)
    expected=tuple(x.identity for x in host_frontier if b'careful' in bytes(x.surface) and b'engineer' in bytes(x.surface))
    checks['temporal_visual_invariance_supports_long_form_grounded_discourse']=(
        re_ctx==ctx and len(re_frontier)==4 and root is not None and set(selected)==set(expected))

    source=inspect.getsource(TemporalVisualContinuityV1)
    checks['temporal_adapter_has_no_object_category_language_or_reward_authority']=(
        all(token not in source for token in ('object','category','language','reward','concept','expected')))
    checks['adapter_checkpoint_contains_no_previous_frame']=('previous' not in json.dumps(coherent.checkpoint(),sort_keys=True))
    checks['bounded_reference_work']=time.perf_counter()-started<1.0

    failed=[k for k,v in checks.items() if not v]
    result={'contract':'FOUNDRY_TEMPORAL_VISUAL_CONTINUITY_INVARIANCE_GREEN','reference_only':True,'graph_flip':False,
            'view_features':[fa1,fa2,fb1,fb2],'temporal_features':{'a':ta,'b':tb},'novel_resolution':g.resolve_world_atom(adult,o,NOVEL),
            'frontier_count':len(re_frontier),'checks':checks,'failed':failed,
            'remaining_red':['CONTINUOUS_CAMERA_SENSOR_OWNERSHIP','OPTIC_FLOW_TRAJECTORY_FEATURES','RAW_AUDIO_WAVEFORM_ELEMENT_EXTRACTION','DIRECT_TEMPORAL_VISUAL_PARITY'],
            'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    if failed:
        print('FOUNDRY_TEMPORAL_VISUAL_CONTINUITY_INVARIANCE_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));return 1
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
