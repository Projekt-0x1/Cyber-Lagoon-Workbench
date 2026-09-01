#!/usr/bin/env python3
"""Owned grayscale streams yield direction/speed events and learned trajectory features."""
from __future__ import annotations
import copy,inspect,json,time
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_open_world_perceptual_category_grounding_verify import SURFACE,pair
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_raw_visual_signal_feature_extraction_verify import edge
from reference_unsupervised_perceptual_features_v1 import FEATURE_QUORUM
from reference_visual_motion_trajectory_v1 import VisualMotionTrajectoryV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1

SRC=0xB100; A=0xB201;B=0xB202;NOVEL=0xB301
FEATURE_SOURCE=0xB400;WORLD_SOURCE=0xB401
RAW_ADJ=0xB501;RAW_TEST=0xB502;RAW_INSPECT=0xB503;RAW_SENSOR=0xB504;RAW_VALVE=0xB505
DIRECT=((RAW_ADJ,101),(RAW_TEST,301),(RAW_INSPECT,302),(RAW_SENSOR,401),(RAW_VALVE,402))

def frame(x,offset=48):return edge(1,1,64,offset=offset,boundary=x,size=8)
A_FRAMES=(frame(1),frame(2),frame(4));A_SHIFT=(frame(2),frame(3),frame(5));B_FRAMES=(frame(6),frame(5),frame(3))

def ingest(sensor,motion,source,seq,img,digest=None):
    digest=VisualSensorIngressV1.frame_digest(img) if digest is None else digest
    raw,contiguous=sensor.ingest(source,seq,img,digest)
    return (*motion.observe_frame(raw,contiguous),contiguous)

def episode(sensor,motion,frames,start,source=SRC):
    out=[]
    for i,img in enumerate(frames):out.append(ingest(sensor,motion,source,start+i,img))
    return out

def train(sensor,motion,frames,start=1,source=SRC,repeats=FEATURE_QUORUM):
    seq=start
    first=second=0
    for _ in range(repeats):
        rows=episode(sensor,motion,frames,seq,source)
        first=rows[1][0];second=rows[2][0];seq+=4
    return first,second,motion.trajectory(first,second),seq

def install(o,entity,feature):o.contact(CONTACT_ENTITY_FEATURES,(entity,1,feature),FEATURE_SOURCE,True,True)
def ground(g,adult,o,entity,concept,source):
    for n in range(2):pair(g,adult,o,entity,SURFACE[concept],source+n)
def train_direct(g,adult,o):
    for i,(raw,concept) in enumerate(DIRECT):
        for n in range(2):pair(g,adult,o,raw,SURFACE[concept],0xB600+i*4+n)
def world(o,atoms):o.contact(CONTACT_WORLD_STATE,tuple(map(int,atoms)),WORLD_SOURCE,True,True)

def main():
    started=time.perf_counter();checks={}
    # Primitive motion geometry.
    sensor=VisualSensorIngressV1();motion=VisualMotionTrajectoryV1()
    r1=ingest(sensor,motion,SRC,1,frame(1));r2=ingest(sensor,motion,SRC,2,frame(1))
    checks['stationary_contiguous_frames_emit_no_motion']=(r1[0]==0 and r2[0]==0)

    s=VisualSensorIngressV1();m=VisualMotionTrajectoryV1();rows=episode(s,m,A_FRAMES,1)
    right1,right2=rows[1][0],rows[2][0]
    s2=VisualSensorIngressV1();m2=VisualMotionTrajectoryV1();left=episode(s2,m2,(frame(4),frame(2)),1)[1][0]
    checks['rightward_and_leftward_displacements_are_distinct']=(right1>0 and left>0 and right1!=left)
    checks['same_direction_different_displacement_magnitude_is_distinct']=(right1>0 and right2>0 and right1!=right2)

    bright=VisualSensorIngressV1();bm=VisualMotionTrajectoryV1();br=episode(bright,bm,tuple(frame(x,offset=80) for x in (1,2,4)),1)
    checks['brightness_offset_preserves_motion_event_sequence']=((br[1][0],br[2][0])==(right1,right2))
    trans=VisualSensorIngressV1();tm=VisualMotionTrajectoryV1();tr=episode(trans,tm,A_SHIFT,1)
    checks['translated_whole_trajectory_preserves_motion_event_sequence']=((tr[1][0],tr[2][0])==(right1,right2))

    one=VisualMotionTrajectoryV1();os=VisualSensorIngressV1();episode(os,one,A_FRAMES,1)
    checks['one_three_frame_trajectory_is_insufficient']=(one.trajectory(right1,right2)==0)

    coherent=VisualMotionTrajectoryV1();cs=VisualSensorIngressV1();ma1,ma2,ta,next_seq=train(cs,coherent,A_FRAMES)
    checks['repeated_coherent_displacement_sequence_creates_trajectory_feature']=(ma1==right1 and ma2==right2 and ta>0)

    shuffled=VisualMotionTrajectoryV1();ss=VisualSensorIngressV1();sm1,sm2,ts,_=train(ss,shuffled,(frame(1),frame(4),frame(2)))
    checks['identical_frame_set_different_order_yields_different_trajectory']=(
        (sm1,sm2)!=(ma1,ma2) and ts>0 and ts!=ta and shuffled.trajectory(ma1,ma2)==0)

    gapped=VisualMotionTrajectoryV1();gs=VisualSensorIngressV1();seq=1
    for _ in range(FEATURE_QUORUM):
        ingest(gs,gapped,SRC,seq,frame(1));ingest(gs,gapped,SRC,seq+2,frame(2));ingest(gs,gapped,SRC,seq+4,frame(4));seq+=6
    checks['same_frames_with_sensor_gaps_do_not_create_target_trajectory']=(gapped.trajectory(ma1,ma2)==0)

    switched=VisualMotionTrajectoryV1();sw=VisualSensorIngressV1()
    ingest(sw,switched,SRC,1,frame(1));ingest(sw,switched,SRC+1,1,frame(2));_mot,_traj,cont=ingest(sw,switched,SRC+1,2,frame(4))
    checks['source_switch_breaks_motion_history_before_trajectory_learning']=(not switched.trajectory(ma1,ma2) and cont)

    cut=VisualMotionTrajectoryV1.restore(copy.deepcopy(coherent.checkpoint()));cut.lesion_trajectory(ma1,ma2)
    checks['focal_trajectory_lesion_preserves_raw_motion_geometry']=(cut.trajectory(ma1,ma2)==0 and VisualMotionTrajectoryV1.active_centroid(frame(1))!=VisualMotionTrajectoryV1.active_centroid(frame(2)))
    restored=VisualMotionTrajectoryV1.restore(copy.deepcopy(coherent.checkpoint()))
    checks['checkpoint_preserves_trajectory_not_active_centroid_or_motion']=(restored.trajectory(ma1,ma2)==ta and restored.previous_centroid is None and restored.previous_motion==0)
    _b1,_b2,tb,_=train(VisualSensorIngressV1(),restored,B_FRAMES,1,SRC+2)
    checks['later_unrelated_trajectory_learning_does_not_overwrite_first']=(tb>0 and tb!=ta and restored.trajectory(ma1,ma2)==ta)
    checks['motion_and_trajectory_learning_have_zero_consequence_credit']=(coherent.learner.population.credit_events==0 and coherent.learner.population.revision_events==0)

    # Visible downstream behavior uses learned trajectory features only.
    adult,host_frontier,_f,_ca,_cb=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1()
    install(o,A,ta);install(o,B,tb);install(o,NOVEL,ta);ground(g,adult,o,A,201,0xB700);ground(g,adult,o,B,203,0xB710)
    checks['novel_trajectory_entity_inherits_grounded_concept_without_language_pair']=(g.resolve_raw_feature(adult,NOVEL)==0 and g.resolve_world_atom(adult,o,NOVEL)==201)
    train_direct(g,adult,o);world(o,(RAW_ADJ,NOVEL,RAW_TEST,RAW_INSPECT,RAW_SENSOR,RAW_VALVE))
    ctx,frontier=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
    for leaf in frontier:
        for _ in range(2):adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1);adult.experience_discourse_background(leaf.identity,False)
    world(o,(RAW_ADJ,NOVEL,RAW_TEST,RAW_INSPECT,RAW_SENSOR,RAW_VALVE));re_ctx,re_frontier=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
    root=adult.organize_relevant_frontier(re_frontier);selected=tuple(adult.last_discourse_selected)
    expected=tuple(x.identity for x in host_frontier if b'careful' in bytes(x.surface) and b'engineer' in bytes(x.surface))
    checks['trajectory_grounded_entity_supports_long_form_discourse']=(re_ctx==ctx and len(re_frontier)==4 and root is not None and set(selected)==set(expected))

    src=inspect.getsource(VisualMotionTrajectoryV1)
    checks['motion_transducer_contains_no_semantic_heading_or_object_authority']=(all(token not in src for token in ('object','category','language','reward','heading','expected','engineer')))
    checks['bounded_reference_work']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'contract':'FOUNDRY_OPTIC_FLOW_TRAJECTORY_FEATURES_GREEN','reference_only':True,'graph_flip':False,
            'motion_events':[ma1,ma2],'trajectory_feature':ta,'shuffled_trajectory':ts,'novel_resolution':g.resolve_world_atom(adult,o,NOVEL),'frontier_count':len(re_frontier),'checks':checks,'failed':failed,
            'remaining_red':['DENSE_OPTIC_FLOW_FIELDS','ROTATION_EXPANSION_HEADING_FEATURES','EYE_MOVEMENT_REAFFERENCE_COMPENSATION','PHYSICAL_CAMERA_DRIVER_OWNERSHIP','RAW_AUDIO_WAVEFORM_ELEMENT_EXTRACTION','DIRECT_OPTIC_FLOW_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    if failed:
        print('FOUNDRY_OPTIC_FLOW_TRAJECTORY_FEATURES_RED '+','.join(failed));print(json.dumps(result,indent=2,sort_keys=True));return 1
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
