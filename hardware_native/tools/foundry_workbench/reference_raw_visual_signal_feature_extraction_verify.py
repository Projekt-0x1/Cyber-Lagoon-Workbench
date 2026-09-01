#!/usr/bin/env python3
"""Raw grayscale arrays generate local events that learn features and support grounded discourse."""
from __future__ import annotations
import copy,inspect,json,time
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_open_world_perceptual_category_grounding_verify import SURFACE,pair
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_raw_visual_elements_v1 import CONTRAST_FLOOR,RawVisualElementsV1
from reference_unsupervised_perceptual_features_v1 import FEATURE_QUORUM,UnsupervisedPerceptualFeaturesV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1

A=0x7101;B=0x7102;NOVEL=0x7201;WEAK=0x7202;AMBIG=0x7203
FEATURE_SOURCE=0x7301;WORLD_SOURCE=0x7302
RAW_ADJ=0x7401;RAW_TEST=0x7402;RAW_INSPECT=0x7403;RAW_SENSOR=0x7404;RAW_VALVE=0x7405
DIRECT=((RAW_ADJ,101),(RAW_TEST,301),(RAW_INSPECT,302),(RAW_SENSOR,401),(RAW_VALVE,402))


def edge(axis:int,polarity:int,contrast:int=128,offset:int=48,boundary:int=3,size:int=6):
    low=int(offset);high=low+int(contrast)
    if high>255:raise ValueError('raw_visual_verify:contrast')
    a,b=(low,high) if int(polarity)>0 else (high,low)
    if int(axis)==1:return tuple(tuple(a if x<boundary else b for x in range(size)) for _y in range(size))
    if int(axis)==2:return tuple(tuple(a if y<boundary else b for _x in range(size)) for y in range(size))
    raise ValueError('raw_visual_verify:axis')


def shift_brightness(image,delta):return tuple(tuple(int(v)+int(delta) for v in row) for row in image)

def invert(image):return tuple(tuple(255-int(v) for v in row) for row in image)

def flat_noise(size=6):return tuple(tuple(100+((x+y)&1)*(CONTRAST_FLOOR//2) for x in range(size)) for y in range(size))

# Nine distinct raw image regularities; no feature IDs are authored.
I1=edge(1,1,64);I2=edge(1,-1,64);I3=edge(2,1,64);I4=edge(2,-1,64)
G3=edge(1,1,96);G4=edge(2,1,96);F5=edge(1,-1,96);H3=edge(1,1,160);H4=edge(2,-1,160)


def raw_pairs(image):return RawVisualElementsV1.pair_patches(image)

def expose_image(learner,image,repeats=FEATURE_QUORUM):
    pairs=raw_pairs(image)
    for _ in range(repeats):
        for patch in pairs:learner.observe_scene(patch)
    return pairs

def image_features(learner,images):
    patches=[]
    for image in images:patches.extend(raw_pairs(image))
    return learner.features_from_patches(tuple(patches))

def install(o,learner,entity,images,source=FEATURE_SOURCE):
    feats=image_features(learner,images)
    if not feats:raise RuntimeError('raw_visual_verify:no_features')
    o.contact(CONTACT_ENTITY_FEATURES,(int(entity),len(feats),*feats),int(source),True,True)
    return feats

def ground(g,adult,o,entity,concept,source):
    for n in range(2):pair(g,adult,o,entity,SURFACE[concept],source+n)

def train_direct(g,adult,o):
    for i,(raw,concept) in enumerate(DIRECT):
        for n in range(2):pair(g,adult,o,raw,SURFACE[concept],0x7500+i*4+n)

def world(o,atoms):o.contact(CONTACT_WORLD_STATE,tuple(map(int,atoms)),WORLD_SOURCE,True,True)


def main():
    started=time.perf_counter();checks={}
    extractor=RawVisualElementsV1()
    vertical=edge(1,1,128);vertical_shift=edge(1,1,128,boundary=2)
    vertical_offset=shift_brightness(edge(1,1,128,offset=32),20)
    vertical_base=edge(1,1,128,offset=32);horizontal=edge(2,1,128);negative=invert(vertical)

    checks['uniform_raw_image_has_no_contrast_events']=(not extractor.event_windows(tuple(tuple(100 for _ in range(6)) for _ in range(6))))
    checks['global_brightness_offset_preserves_exact_event_vocabulary_and_pairs']=(
        extractor.event_vocabulary(vertical_base)==extractor.event_vocabulary(vertical_offset)
        and raw_pairs(vertical_base)==raw_pairs(vertical_offset))
    checks['spatial_translation_preserves_event_vocabulary_and_pair_feature_input']=(
        extractor.event_vocabulary(vertical)==extractor.event_vocabulary(vertical_shift)
        and raw_pairs(vertical)==raw_pairs(vertical_shift))
    checks['polarity_inversion_changes_primitive_event_pair']=(raw_pairs(vertical)!=raw_pairs(negative))
    checks['horizontal_and_vertical_boundaries_have_distinct_orientation_events']=(raw_pairs(vertical)!=raw_pairs(horizontal))
    checks['below_floor_pixel_noise_creates_no_events']=(not extractor.event_windows(flat_noise()))

    # Raw pixels -> fixed events -> unlabeled learned conjunction feature.
    one=UnsupervisedPerceptualFeaturesV1();vp=raw_pairs(vertical);assert len(vp)==1
    one.observe_scene(vp[0]);checks['one_raw_image_exposure_is_not_a_learned_feature']=(one.feature(*vp[0])==0)
    learned=UnsupervisedPerceptualFeaturesV1();expose_image(learned,vertical)
    vf=learned.feature(*vp[0])
    checks['repeated_raw_images_create_unlabeled_perceptual_feature']=(vf>0 and learned.support(*vp[0])>=FEATURE_QUORUM)
    checks['raw_visual_learning_has_zero_consequence_credit']=(
        learned.population.credit_events==0 and learned.population.revision_events==0 and not learned.population.occurrences)

    # Matched high/low pixel marginals arranged as horizontal instead of vertical do not teach target vertical conjunction.
    shuffled=UnsupervisedPerceptualFeaturesV1()
    for _ in range(FEATURE_QUORUM):
        for patch in raw_pairs(horizontal):shuffled.observe_scene(patch)
    checks['matched_pixel_marginals_different_structure_do_not_fabricate_target_feature']=(
        shuffled.feature(*vp[0])==0 and learned.feature(*vp[0])==vf)

    cp=copy.deepcopy(learned.checkpoint());restored=UnsupervisedPerceptualFeaturesV1.restore(cp)
    checks['checkpoint_preserves_raw_image_learned_feature']=(restored.feature(*vp[0])==vf)
    lesion=UnsupervisedPerceptualFeaturesV1.restore(cp);lesion.lesion_pair(*vp[0])
    checks['feature_lesion_removes_learned_feature_but_not_raw_transduction']=(
        lesion.feature(*vp[0])==0 and raw_pairs(vertical)==vp)

    # Full stack: entity feature bundles come only from repeated raw grayscale images.
    learner=UnsupervisedPerceptualFeaturesV1()
    for image in (I1,I2,I3,I4,G3,G4,F5,H3,H4):expose_image(learner,image)
    adult,host_frontier,_f,_ca,_cb=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1()
    fa=install(o,learner,A,(I1,I2,I3,I4));fb=install(o,learner,B,(I1,I2,G3,G4))
    fn=install(o,learner,NOVEL,(I1,I2,I3,F5));fw=install(o,learner,WEAK,(I1,I2,H3,H4));fm=install(o,learner,AMBIG,(I1,I2,I3,G3))
    checks['entity_features_are_derived_from_raw_pixels_not_fixture_ids']=(
        fa==image_features(learner,(I1,I2,I3,I4)) and fn==image_features(learner,(I1,I2,I3,F5))
        and len(fa)==len(fb)==len(fn)==len(fw)==len(fm)==4)

    ground(g,adult,o,A,201,0x7600);ground(g,adult,o,B,203,0x7610)
    checks['novel_raw_visual_entity_generalizes_without_language_pair']=(
        g.resolve_raw_feature(adult,NOVEL)==0 and g.resolve_world_atom(adult,o,NOVEL)==201)
    checks['weak_and_cross_category_raw_visual_entities_refuse']=(
        g.resolve_world_atom(adult,o,WEAK)==0 and g.resolve_world_atom(adult,o,AMBIG)==0)

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
    checks['raw_grayscale_development_supports_novel_entity_long_form_discourse']=(
        re_ctx==ctx and len(re_frontier)==4 and root is not None and set(selected)==set(expected))

    src=inspect.getsource(RawVisualElementsV1)
    checks['fixed_transducer_contains_no_semantic_or_category_authority']=(
        all(token not in src for token in ('entity','category','concept','language','reward','expected','engineer','sensor')))
    checks['fixed_transducer_has_no_checkpoint_or_learned_state']=(not hasattr(extractor,'checkpoint') and not extractor.__dict__)
    checks['bounded_reference_work']=time.perf_counter()-started<1.0

    failed=[k for k,v in checks.items() if not v]
    result={'contract':'FOUNDRY_RAW_VISUAL_SIGNAL_FEATURE_EXTRACTION_GREEN','reference_only':True,'graph_flip':False,
            'primitive_pair':vp[0],'learned_feature':vf,'novel_resolution':g.resolve_world_atom(adult,o,NOVEL),
            'frontier_count':len(re_frontier),'checks':checks,'failed':failed,
            'remaining_red':['RAW_AUDIO_WAVEFORM_ELEMENT_EXTRACTION','HIERARCHICAL_PERCEPTUAL_FEATURE_DISCOVERY','CONTINUOUS_CAMERA_SENSOR_OWNERSHIP','DIRECT_RAW_VISUAL_PARITY'],
            'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    if failed:
        print('FOUNDRY_RAW_VISUAL_SIGNAL_FEATURE_EXTRACTION_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));return 1
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
