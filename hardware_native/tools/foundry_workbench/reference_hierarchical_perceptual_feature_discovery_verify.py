#!/usr/bin/env python3
"""Two-level unlabeled visual feature hierarchy reuses one generic exposure law recursively."""
from __future__ import annotations
import copy,inspect,json,time
from itertools import combinations
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_open_world_perceptual_category_grounding_verify import SURFACE,pair
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_raw_visual_signal_feature_extraction_verify import I3,I4,H4,edge,expose_image,raw_pairs
from reference_unsupervised_perceptual_features_v1 import FEATURE_QUORUM,UnsupervisedPerceptualFeaturesV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1

A=0x8101;B=0x8102;NOVEL=0x8201;AMBIG=0x8202
FEATURE_SOURCE=0x8301;WORLD_SOURCE=0x8302
RAW_ADJ=0x8401;RAW_TEST=0x8402;RAW_INSPECT=0x8403;RAW_SENSOR=0x8404;RAW_VALVE=0x8405
DIRECT=((RAW_ADJ,101),(RAW_TEST,301),(RAW_INSPECT,302),(RAW_SENSOR,401),(RAW_VALVE,402))


def compound(top,bottom):
    width=len(top[0]);separator=(tuple(128 for _ in range(width)),)*2
    return tuple(top)+separator+tuple(bottom)


def level1_features(level1,image):
    return tuple(sorted({feature for patch in raw_pairs(image)
                         for feature in (level1.feature(*patch),) if feature}))


def expose_level2(level1,level2,image,repeats=FEATURE_QUORUM):
    active=level1_features(level1,image)
    for _ in range(repeats):level2.observe_scene(active)
    return active


def level2_features(level1,level2,image):
    active=level1_features(level1,image);out=[]
    for pair_features in combinations(active,2):
        feature=level2.feature(*pair_features)
        if feature:out.append(int(feature))
    return tuple(sorted(set(out)))


def install(o,entity,features,source=FEATURE_SOURCE):
    row=tuple(map(int,features))
    if not row:raise RuntimeError('hierarchical_features:empty')
    o.contact(CONTACT_ENTITY_FEATURES,(int(entity),len(row),*row),int(source),True,True)
    return row


def ground(g,adult,o,entity,concept,source):
    for n in range(2):pair(g,adult,o,entity,SURFACE[concept],source+n)


def train_direct(g,adult,o):
    for i,(raw,concept) in enumerate(DIRECT):
        for n in range(2):pair(g,adult,o,raw,SURFACE[concept],0x8500+i*4+n)


def world(o,atoms):o.contact(CONTACT_WORLD_STATE,tuple(map(int,atoms)),WORLD_SOURCE,True,True)


def main():
    started=time.perf_counter();checks={}
    level1=UnsupervisedPerceptualFeaturesV1()
    for image in (I3,I4,H4):expose_image(level1,image)
    f3=level1.feature(*raw_pairs(I3)[0]);f4=level1.feature(*raw_pairs(I4)[0]);fh4=level1.feature(*raw_pairs(H4)[0])
    checks['lower_visual_features_exist_before_higher_learning']=(f3>0 and f4>0 and fh4>0 and len({f3,f4,fh4})==3)

    image_a=compound(I3,I4);active_a=level1_features(level1,image_a)
    checks['compound_raw_image_mechanically_activates_exact_two_lower_features']=(active_a==tuple(sorted((f3,f4))))

    one=UnsupervisedPerceptualFeaturesV1();one.observe_scene(active_a)
    checks['one_compound_image_is_insufficient_for_higher_feature']=(one.feature(*active_a)==0)

    level2=UnsupervisedPerceptualFeaturesV1();expose_level2(level1,level2,image_a)
    higher_a=level2.feature(*active_a)
    checks['repeated_same_image_lower_feature_cooccurrence_creates_higher_feature']=(higher_a>0 and level2.support(*active_a)>=FEATURE_QUORUM)

    # Same lower-feature marginals, never co-active in one image, cannot create the conjunction.
    shuffled=UnsupervisedPerceptualFeaturesV1()
    for _ in range(FEATURE_QUORUM):
        shuffled.observe_scene(level1_features(level1,I3));shuffled.observe_scene(level1_features(level1,I4))
    checks['separate_images_with_same_lower_marginals_do_not_fabricate_higher_feature']=(shuffled.feature(f3,f4)==0)

    image_b=compound(I3,H4);active_b=level1_features(level1,image_b)
    for _ in range(FEATURE_QUORUM):level2.observe_scene(active_b)
    higher_b=level2.feature(*active_b)
    checks['changing_one_lower_component_yields_distinct_higher_feature']=(
        active_b==tuple(sorted((f3,fh4))) and higher_b>0 and higher_b!=higher_a)

    # Translation of both constituent motifs preserves lower identities and therefore the higher identity.
    i3_shift=edge(2,1,64,boundary=2);i4_shift=edge(2,-1,64,boundary=2)
    translated=compound(i3_shift,i4_shift)
    checks['translated_motifs_reactivate_same_higher_feature']=(
        level1_features(level1,translated)==active_a and level2_features(level1,level2,translated)==(higher_a,))

    lower_cut=UnsupervisedPerceptualFeaturesV1.restore(copy.deepcopy(level1.checkpoint()));lower_cut.lesion_pair(*raw_pairs(I3)[0])
    checks['lower_feature_lesion_prevents_higher_activation_preserves_other_lower']=(
        lower_cut.feature(*raw_pairs(I3)[0])==0 and lower_cut.feature(*raw_pairs(I4)[0])==f4
        and not level2_features(lower_cut,level2,image_a))

    higher_cut=UnsupervisedPerceptualFeaturesV1.restore(copy.deepcopy(level2.checkpoint()));higher_cut.lesion_pair(f3,f4)
    checks['higher_feature_lesion_preserves_both_lower_features']=(
        higher_cut.feature(f3,f4)==0 and level1.feature(*raw_pairs(I3)[0])==f3 and level1.feature(*raw_pairs(I4)[0])==f4)

    cp1=copy.deepcopy(level1.checkpoint());cp2=copy.deepcopy(level2.checkpoint())
    r1=UnsupervisedPerceptualFeaturesV1.restore(cp1);r2=UnsupervisedPerceptualFeaturesV1.restore(cp2)
    checks['checkpoint_preserves_two_level_hierarchy_without_active_image_state']=(
        level2_features(r1,r2,image_a)==(higher_a,) and not r1.__dict__.get('current_image') and not r2.__dict__.get('current_image'))

    # Learn unrelated higher conjunction after checkpoint; early hierarchy remains exact.
    for _ in range(FEATURE_QUORUM):r2.observe_scene(active_b)
    checks['later_higher_feature_learning_does_not_overwrite_early_hierarchy']=(r2.feature(f3,f4)==higher_a)

    # Full stack uses second-order features only as entity geometry.
    adult,host_frontier,_f,_ca,_cb=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1()
    fa=install(o,A,level2_features(level1,level2,image_a));fb=install(o,B,level2_features(level1,level2,image_b))
    fn=install(o,NOVEL,level2_features(level1,level2,translated));install(o,AMBIG,tuple(sorted((higher_a,higher_b))))
    checks['organism_entity_geometry_is_composed_only_of_second_order_features']=(fa==(higher_a,) and fb==(higher_b,) and fn==(higher_a,))
    ground(g,adult,o,A,201,0x8600);ground(g,adult,o,B,203,0x8610)
    checks['novel_translated_hierarchical_visual_entity_inherits_grounded_concept']=(
        g.resolve_raw_feature(adult,NOVEL)==0 and g.resolve_world_atom(adult,o,NOVEL)==201)
    checks['mixed_higher_feature_entity_refuses_arbitrary_category']=(g.resolve_world_atom(adult,o,AMBIG)==0)

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
    checks['two_level_raw_visual_hierarchy_supports_long_form_discourse']=(
        re_ctx==ctx and len(re_frontier)==4 and root is not None and set(selected)==set(expected))

    checks['no_level_specific_feature_mechanism_was_added']=(
        type(level1) is type(level2) is UnsupervisedPerceptualFeaturesV1
        and list(inspect.signature(UnsupervisedPerceptualFeaturesV1.observe_scene).parameters)==['self','elements'])
    checks['both_hierarchy_layers_have_zero_consequence_credit']=(
        level1.population.credit_events==0 and level2.population.credit_events==0
        and level1.population.revision_events==0 and level2.population.revision_events==0)
    checks['bounded_reference_work']=time.perf_counter()-started<1.0

    failed=[k for k,v in checks.items() if not v]
    result={'contract':'FOUNDRY_HIERARCHICAL_PERCEPTUAL_FEATURE_DISCOVERY_GREEN','reference_only':True,'graph_flip':False,
            'lower_features':[f3,f4],'higher_feature':higher_a,'translated_higher':level2_features(level1,level2,translated),
            'novel_resolution':g.resolve_world_atom(adult,o,NOVEL),'frontier_count':len(re_frontier),'mechanism_change':False,
            'checks':checks,'failed':failed,
            'remaining_red':['ARBITRARY_DEPTH_PERCEPTUAL_HIERARCHY','TEMPORAL_MOTION_INVARIANCE','RAW_AUDIO_WAVEFORM_ELEMENT_EXTRACTION','CONTINUOUS_CAMERA_SENSOR_OWNERSHIP','DIRECT_HIERARCHICAL_PERCEPTION_PARITY'],
            'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    if failed:
        print('FOUNDRY_HIERARCHICAL_PERCEPTUAL_FEATURE_DISCOVERY_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));return 1
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
