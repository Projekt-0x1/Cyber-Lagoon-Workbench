#!/usr/bin/env python3
"""Unlabeled recurrent raw conjunctions become perceptual features used by grounded discourse."""
from __future__ import annotations
import copy,inspect,json,time
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES,CONTACT_WORLD_STATE
from reference_open_world_perceptual_category_grounding_verify import SURFACE,pair
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_unsupervised_perceptual_features_v1 import FEATURE_QUORUM,UnsupervisedPerceptualFeaturesV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1

P1=(11,12);P2=(21,22);P3=(31,32);P4=(41,42);G3=(51,52);G4=(61,62);F5=(71,72);H3=(81,82);H4=(91,92)
A=0xA101;B=0xA102;NOVEL=0xA201;WEAK=0xA202;AMBIG=0xA203
FEATURE_SOURCE=0xAA01;WORLD_SOURCE=0xAA02
RAW_ADJ=0xAB01;RAW_TEST=0xAB02;RAW_INSPECT=0xAB03;RAW_SENSOR=0xAB04;RAW_VALVE=0xAB05
DIRECT=((RAW_ADJ,101),(RAW_TEST,301),(RAW_INSPECT,302),(RAW_SENSOR,401),(RAW_VALVE,402))


def expose(learner,patches,repeats=FEATURE_QUORUM):
    for patch in patches:
        for _ in range(repeats):learner.observe_scene(patch)


def bundle(learner,patches):return learner.features_from_patches(patches)


def install(o,learner,entity,patches,source=FEATURE_SOURCE):
    features=bundle(learner,patches)
    if not features:raise RuntimeError('unsupervised_features:empty_bundle')
    o.contact(CONTACT_ENTITY_FEATURES,(int(entity),len(features),*features),int(source),True,True)
    return features


def ground(g,adult,o,entity,concept,source):
    for n in range(2):pair(g,adult,o,entity,SURFACE[concept],source+n)


def train_direct(g,adult,o):
    for i,(raw,concept) in enumerate(DIRECT):
        for n in range(2):pair(g,adult,o,raw,SURFACE[concept],0xAC00+i*4+n)


def world(o,atoms):o.contact(CONTACT_WORLD_STATE,tuple(map(int,atoms)),WORLD_SOURCE,True,True)


def main():
    started=time.perf_counter();checks={}

    # Statistical law in isolation.
    one=UnsupervisedPerceptualFeaturesV1();one.observe_scene(P1)
    checks['one_shot_conjunction_is_not_a_feature']=(one.feature(*P1)==0 and one.support(*P1)==1)

    learned=UnsupervisedPerceptualFeaturesV1();expose(learned,(P1,))
    f1=learned.feature(*P1)
    checks['repeated_unlabeled_conjunction_crosses_developmental_quorum']=(f1>0 and learned.support(*P1)>=FEATURE_QUORUM)
    checks['pair_feature_is_order_invariant']=(learned.feature(*reversed(P1))==f1)
    checks['changed_element_is_different_and_unlearned']=(learned.feature(P1[0],999)==0)

    shuffled=UnsupervisedPerceptualFeaturesV1();shuffled.observe_scene((1001,1002))
    for n in range(8):
        shuffled.observe_scene((1001,2000+n));shuffled.observe_scene((1002,3000+n))
    checks['equal_high_marginals_with_shuffled_pairing_do_not_fabricate_conjunction']=(
        shuffled.feature(1001,1002)==0 and shuffled.support(1001,1002)==1)
    checks['feature_learning_has_no_consequence_credit_or_retained_occurrences']=(
        learned.population.credit_events==0 and learned.population.revision_events==0 and not learned.population.occurrences)

    cp=copy.deepcopy(learned.checkpoint());restored=UnsupervisedPerceptualFeaturesV1.restore(cp)
    checks['checkpoint_preserves_discovered_feature']=(restored.feature(*P1)==f1)
    for n in range(16):restored.observe_scene((4000+n,5000+n))
    checks['unrelated_exposure_does_not_erase_old_feature_and_new_features_remain_learnable']=(restored.feature(*P1)==f1)
    expose(restored,((6001,6002),));checks['post_interference_new_conjunction_can_be_learned']=(restored.feature(6001,6002)>0)
    lesioned=UnsupervisedPerceptualFeaturesV1.restore(cp);lesioned.lesion_pair(*P1)
    checks['focal_support_lesion_removes_only_target_conjunction']=(lesioned.feature(*P1)==0)

    # Full stack: all entity feature IDs come from unlabeled conjunction statistics.
    learner=UnsupervisedPerceptualFeaturesV1()
    all_patches=(P1,P2,P3,P4,G3,G4,F5,H3,H4)
    expose(learner,all_patches)
    adult,host_frontier,_f,_ca,_cb=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1()
    fa=install(o,learner,A,(P1,P2,P3,P4));fb=install(o,learner,B,(P1,P2,G3,G4))
    fn=install(o,learner,NOVEL,(P1,P2,P3,F5));fw=install(o,learner,WEAK,(P1,P2,H3,H4));fm=install(o,learner,AMBIG,(P1,P2,P3,G3))
    checks['entity_feature_ids_are_outputs_of_unsupervised_learner']=(
        fa==bundle(learner,(P1,P2,P3,P4)) and fn==bundle(learner,(P1,P2,P3,F5))
        and len(fa)==len(fb)==len(fn)==len(fw)==len(fm)==4)

    ground(g,adult,o,A,201,0xAD10);ground(g,adult,o,B,203,0xAD20)
    checks['novel_entity_generalizes_from_discovered_feature_family']=(
        g.resolve_raw_feature(adult,NOVEL)==0 and g.resolve_world_atom(adult,o,NOVEL)==201)
    checks['weak_and_ambiguous_discovered_feature_bundles_refuse']=(
        g.resolve_world_atom(adult,o,WEAK)==0 and g.resolve_world_atom(adult,o,AMBIG)==0)

    # Feature lesion propagates causally to category transfer without rewriting language/grounding.
    lesion_learner=UnsupervisedPerceptualFeaturesV1.restore(copy.deepcopy(learner.checkpoint()));lesion_learner.lesion_pair(*P3)
    lesion_o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    install(lesion_o,lesion_learner,A,(P1,P2,P3,P4));install(lesion_o,lesion_learner,NOVEL,(P1,P2,P3,F5))
    checks['feature_discovery_lesion_changes_family_geometry_upstream_of_grounding']=(
        len(lesion_o._active_entity_features(NOVEL))==3 and len(lesion_o._active_entity_features(A))==3)

    # Strong phenotype: novel entity whose features were discovered without labels drives discourse.
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
    checks['unsupervised_features_support_novel_entity_long_form_discourse']=(
        re_ctx==ctx and len(re_frontier)==4 and root is not None and set(selected)==set(expected))

    feature_source=inspect.getsource(UnsupervisedPerceptualFeaturesV1.observe_scene)
    checks['feature_learning_api_contains_no_entity_category_language_or_reward']=(
        list(inspect.signature(UnsupervisedPerceptualFeaturesV1.observe_scene).parameters)==['self','elements']
        and all(token not in feature_source for token in ('entity','category','language','reward','effect','concept')))
    checks['bounded_reference_work']=time.perf_counter()-started<1.0

    failed=[k for k,v in checks.items() if not v]
    result={'contract':'FOUNDRY_UNSUPERVISED_PERCEPTUAL_FEATURE_DISCOVERY_GREEN','reference_only':True,'graph_flip':False,
            'feature_quorum':FEATURE_QUORUM,'feature_example':f1,'novel_resolution':g.resolve_world_atom(adult,o,NOVEL),
            'frontier_count':len(re_frontier),'checks':checks,'failed':failed,
            'remaining_red':['RAW_PIXEL_AUDIO_FEATURE_EXTRACTION','HIERARCHICAL_PERCEPTUAL_FEATURE_DISCOVERY','DIRECT_UNSUPERVISED_FEATURE_PARITY'],
            'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    if failed:
        print('FOUNDRY_UNSUPERVISED_PERCEPTUAL_FEATURE_DISCOVERY_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));return 1
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
