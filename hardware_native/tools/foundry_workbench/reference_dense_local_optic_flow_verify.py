#!/usr/bin/env python3
"""Multiple local motion vectors survive when global centroid motion cancels."""
from __future__ import annotations
import copy,inspect,json,time
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_local_optic_flow_v1 import LocalOpticFlowV1
from reference_open_world_perceptual_category_grounding_verify import SURFACE,pair
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_visual_motion_trajectory_v1 import VisualMotionTrajectoryV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1
from reference_unsupervised_perceptual_features_v1 import FEATURE_QUORUM

SRC=0xC100;A=0xC201;B=0xC202;NOVEL=0xC301
FEATURE_SOURCE=0xC400;WORLD_SOURCE=0xC401
RAW_ADJ=0xC501;RAW_TEST=0xC502;RAW_INSPECT=0xC503;RAW_SENSOR=0xC504;RAW_VALVE=0xC505
DIRECT=((RAW_ADJ,101),(RAW_TEST,301),(RAW_INSPECT,302),(RAW_SENSOR,401),(RAW_VALVE,402))

def bar(left,right,offset=32,high=160,size=10):
    return tuple(tuple(high if left<=x<right else offset for x in range(size)) for _y in range(size))
def edge(boundary,offset=32,high=160,size=10):
    return tuple(tuple(offset if x<boundary else high for x in range(size)) for _y in range(size))
NARROW=bar(3,7);WIDE=bar(2,8);NARROW_SHIFT=bar(4,8);WIDE_SHIFT=bar(3,9)

def ingest(sensor,flow,seq,img,source=SRC):
    raw,contiguous=sensor.ingest(source,seq,img,VisualSensorIngressV1.frame_digest(img))
    return flow.observe_frame(raw,contiguous),contiguous

def episode(sensor,flow,start,a,b,source=SRC):
    ingest(sensor,flow,start,a,source);return ingest(sensor,flow,start+1,b,source)
def train(flow,frames=(NARROW,WIDE),source=SRC,start=1):
    sensor=VisualSensorIngressV1();seq=start;field=()
    for _ in range(FEATURE_QUORUM):
        field,_=episode(sensor,flow,seq,frames[0],frames[1],source);seq+=3
    return field,flow.flow_feature(field)
def install(o,entity,feature):o.contact(CONTACT_ENTITY_FEATURES,(entity,1,feature),FEATURE_SOURCE,True,True)
def ground(g,adult,o,entity,concept,source):
    for n in range(2):pair(g,adult,o,entity,SURFACE[concept],source+n)
def train_direct(g,adult,o):
    for i,(raw,concept) in enumerate(DIRECT):
        for n in range(2):pair(g,adult,o,raw,SURFACE[concept],0xC600+i*4+n)
def world(o,atoms):o.contact(CONTACT_WORLD_STATE,tuple(map(int,atoms)),WORLD_SOURCE,True,True)

def main():
    started=time.perf_counter();checks={}
    # Single local translation remains compatible with centroid motion.
    single=LocalOpticFlowV1();ss=VisualSensorIngressV1();field_single,_=episode(ss,single,1,edge(3),edge(4))
    checks['single_translating_edge_yields_one_local_vector']=(len(field_single)==1)

    # Symmetric expansion: global centroid stays fixed, local vectors survive.
    local=LocalOpticFlowV1();ls=VisualSensorIngressV1();expansion,_=episode(ls,local,1,NARROW,WIDE)
    centroid=VisualMotionTrajectoryV1();cs=VisualSensorIngressV1()
    raw,c=cs.ingest(SRC,1,NARROW,VisualSensorIngressV1.frame_digest(NARROW));centroid.observe_frame(raw,c)
    raw,c=cs.ingest(SRC,2,WIDE,VisualSensorIngressV1.frame_digest(WIDE));centroid_motion,_=centroid.observe_frame(raw,c)
    checks['symmetric_expansion_has_zero_global_centroid_motion_but_two_local_vectors']=(centroid_motion==0 and len(expansion)==2)

    contraction_flow=LocalOpticFlowV1();cts=VisualSensorIngressV1();contraction,_=episode(cts,contraction_flow,1,WIDE,NARROW)
    checks['expansion_and_contraction_fields_are_distinct']=(len(contraction)==2 and contraction!=expansion)

    static=LocalOpticFlowV1();sts=VisualSensorIngressV1();static_field,_=episode(sts,static,1,NARROW,NARROW)
    checks['static_multi_component_frame_emits_no_flow_vectors']=(not static_field)

    bright_n=bar(3,7,offset=64,high=192);bright_w=bar(2,8,offset=64,high=192)
    bright=LocalOpticFlowV1();bs=VisualSensorIngressV1();bright_field,_=episode(bs,bright,1,bright_n,bright_w)
    checks['brightness_offset_preserves_local_flow_field']=(bright_field==expansion)
    shifted=LocalOpticFlowV1();shs=VisualSensorIngressV1();shifted_field,_=episode(shs,shifted,1,NARROW_SHIFT,WIDE_SHIFT)
    checks['translating_whole_expansion_pattern_preserves_relative_flow_field']=(shifted_field==expansion)

    one=LocalOpticFlowV1();os=VisualSensorIngressV1();one_field,_=episode(os,one,1,NARROW,WIDE)
    checks['one_expansion_exposure_is_insufficient_for_global_flow_feature']=(one.flow_feature(one_field)==0)

    learned=LocalOpticFlowV1();expansion_field,expansion_feature=train(learned)
    checks['repeated_expansion_learns_global_flow_feature']=(expansion_field==expansion and expansion_feature>0)
    contract=LocalOpticFlowV1();contraction_field,contraction_feature=train(contract,(WIDE,NARROW),SRC+1)
    checks['repeated_contraction_learns_distinct_global_flow_feature']=(contraction_feature>0 and contraction_feature!=expansion_feature)

    gapped=LocalOpticFlowV1();gs=VisualSensorIngressV1();seq=1
    for _ in range(FEATURE_QUORUM):
        ingest(gs,gapped,seq,NARROW);field,_=ingest(gs,gapped,seq+2,WIDE);seq+=4
    checks['sensor_gaps_prevent_expansion_feature_learning']=(not field and gapped.flow_feature(expansion)==0)

    # Removing one boundary leaves only one local vector and cannot activate expansion feature.
    partial=LocalOpticFlowV1();ps=VisualSensorIngressV1();partial_field,_=episode(ps,partial,1,edge(3),edge(2))
    checks['single_component_motion_cannot_substitute_for_two_vector_expansion']=(len(partial_field)==1 and learned.flow_feature(partial_field)==0)

    restored=LocalOpticFlowV1.restore(copy.deepcopy(learned.checkpoint()))
    checks['checkpoint_preserves_learned_flow_not_active_components']=(restored.flow_feature(expansion)==expansion_feature and not restored.previous)
    _other_field,other_feature=train(restored,(WIDE,NARROW),SRC+2)
    checks['later_unrelated_flow_learning_does_not_overwrite_expansion']=(other_feature>0 and restored.flow_feature(expansion)==expansion_feature)
    checks['flow_learning_has_zero_consequence_credit']=(learned.learner.population.credit_events==0 and learned.learner.population.revision_events==0)

    # Downstream visible behavior uses only learned global-flow features.
    adult,host_frontier,_f,_ca,_cb=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1()
    install(o,A,expansion_feature);install(o,B,contraction_feature);install(o,NOVEL,expansion_feature)
    ground(g,adult,o,A,201,0xC700);ground(g,adult,o,B,203,0xC710)
    checks['novel_expansion_flow_entity_inherits_grounded_concept_without_language_pair']=(g.resolve_raw_feature(adult,NOVEL)==0 and g.resolve_world_atom(adult,o,NOVEL)==201)
    train_direct(g,adult,o);world(o,(RAW_ADJ,NOVEL,RAW_TEST,RAW_INSPECT,RAW_SENSOR,RAW_VALVE))
    ctx,frontier=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
    for leaf in frontier:
        for _ in range(2):adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1);adult.experience_discourse_background(leaf.identity,False)
    world(o,(RAW_ADJ,NOVEL,RAW_TEST,RAW_INSPECT,RAW_SENSOR,RAW_VALVE));re_ctx,re_frontier=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
    root=adult.organize_relevant_frontier(re_frontier);selected=tuple(adult.last_discourse_selected)
    expected=tuple(x.identity for x in host_frontier if b'careful' in bytes(x.surface) and b'engineer' in bytes(x.surface))
    checks['local_flow_grounded_entity_supports_long_form_discourse']=(re_ctx==ctx and len(re_frontier)==4 and root is not None and set(selected)==set(expected))

    src=inspect.getsource(LocalOpticFlowV1)
    checks['local_flow_extractor_has_no_semantic_flow_class_authority']=(all(token not in src for token in ('object','category','language','reward','expansion','contraction','heading','expected')))
    checks['bounded_reference_work']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'contract':'FOUNDRY_DENSE_LOCAL_OPTIC_FLOW_GREEN','reference_only':True,'graph_flip':False,'expansion_field':expansion,'contraction_field':contraction,'expansion_feature':expansion_feature,'contraction_feature':contraction_feature,'frontier_count':len(re_frontier),'checks':checks,'failed':failed,
            'remaining_red':['DENSE_PER_PIXEL_OPTIC_FLOW','ROTATION_HEADING_FEATURES','EYE_MOVEMENT_REAFFERENCE_COMPENSATION','PHYSICAL_CAMERA_DRIVER_OWNERSHIP','RAW_AUDIO_WAVEFORM_ELEMENT_EXTRACTION','DIRECT_DENSE_FLOW_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    if failed:
        print('FOUNDRY_DENSE_LOCAL_OPTIC_FLOW_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));return 1
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
