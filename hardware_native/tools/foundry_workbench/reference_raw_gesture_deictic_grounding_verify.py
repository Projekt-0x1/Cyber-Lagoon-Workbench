#!/usr/bin/env python3
"""N+1: raw partner hand trajectories learn and resolve visual deictic reference without host target coordinates."""
from __future__ import annotations
import copy,inspect,json,time
from reference_continuous_visual_sensor_ownership_verify import A1,A2,train_level1
from reference_global_discourse_relevance_verify import fresh
from reference_multi_visual_object_file_tracker_v1 import MultiVisualObjectFileTrackerV1
from reference_multiple_visual_object_files_verify import canvas,train_temporal
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_raw_pointing_motion_v1 import RawPointingMotionV1
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_visual_deictic_distance_ecology_v1 import VisualDeicticDistanceEcologyV1
from reference_visual_motion_deictic_v1 import VisualMotionDeicticV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

SOURCE=0xDA01;M1=0xDA11;M2=0xDA12;RAW1=b'this';RAW2=b'that';S1=(0xDA21,0xDA22);S2=(0xDA31,0xDA32)
ORIGIN=(5,0)

def obs(t,o,s,temp,level1,seq,items,w=40):
    frame=canvas(items,h=8,w=w)
    return t.observe(o,s,temp,level1,SOURCE,seq,frame,VisualSensorIngressV1.frame_digest(frame))

def motion_to(y,x):
    oy,ox=ORIGIN;return ((oy,ox),((oy+y)//2,(ox+x)//2),(y,x))

def main():
    started=time.perf_counter();checks={};level1=train_level1();temporal=train_temporal(level1);adult,*_=fresh();eco=VisualDeicticDistanceEcologyV1()
    for source in S1:adult.observe_surface_item(M1,RAW1,source)
    for source in S2:adult.observe_surface_item(M2,RAW2,source)
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));sensor=VisualSensorIngressV1();tracker=MultiVisualObjectFileTrackerV1();temp=TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint()))
    train=obs(tracker,o,sensor,temp,level1,1,((A1,1,1),(A1,1,13)))
    near=(train[0][1],train[0][2]);far=(train[1][1],train[1][2]);near_motion=motion_to(*near);far_motion=motion_to(*far)
    for source in S1:VisualMotionDeicticV1.observe(adult,tracker,eco,RAW1,near_motion,source)
    for source in S2:VisualMotionDeicticV1.observe(adult,tracker,eco,RAW2,far_motion,source)
    checks['raw_hand_motion_alone_builds_two_distinct_deictic_profiles']=(eco.profile(adult,M1) is not None and eco.profile(adult,M2) is not None and eco.profile(adult,M1)<eco.profile(adult,M2))

    # Held-out positions/view change and new trajectories.
    test=obs(tracker,o,sensor,temp,level1,2,((A2,1,2),(A2,1,12)));n=(test[0][1],test[0][2]);f=(test[1][1],test[1][2])
    nref=VisualMotionDeicticV1.resolve(adult,tracker,eco,RAW1,motion_to(*n));fref=VisualMotionDeicticV1.resolve(adult,tracker,eco,RAW2,motion_to(*f))
    checks['heldout_dynamic_gesture_resolves_correct_visible_individual_without_target_coordinate']=(nref==test[0][0] and fref==test[1][0])
    checks['swapped_gesture_marker_combinations_refuse']=(VisualMotionDeicticV1.resolve(adult,tracker,eco,RAW2,motion_to(*n))==0 and VisualMotionDeicticV1.resolve(adult,tracker,eco,RAW1,motion_to(*f))==0)

    static=(ORIGIN,ORIGIN,ORIGIN);retract=(ORIGIN,(6,10),(6,5),(6,3))
    checks['static_hand_posture_does_not_substitute_for_dynamic_pointing']=(RawPointingMotionV1.target(tracker,static)==0 and VisualMotionDeicticV1.resolve(adult,tracker,eco,RAW1,static)==0)
    checks['nonmonotonic_retracting_motion_refuses']=(RawPointingMotionV1.target(tracker,retract)==0)

    # Two same-category objects exactly on one gesture ray create an angular tie.
    to=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));ts=VisualSensorIngressV1();tt=MultiVisualObjectFileTrackerV1();t_temp=TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint()))
    # centers y2=7; choose gesture origin on the same y2, x2=0.
    tie_rows=obs(tt,to,ts,t_temp,level1,1,((A1,1,1),(A1,1,13)));tie_motion=((7,0),(7,5),(7,15),(7,35))
    checks['two_collinear_targets_refuse_instead_of_nearest_or_id_tiebreak']=(RawPointingMotionV1.target(tt,tie_motion)==0)

    source=(inspect.getsource(RawPointingMotionV1)+inspect.getsource(VisualMotionDeicticV1)).lower();blob=json.dumps(eco.checkpoint(),sort_keys=True)
    checks['gesture_bridge_has_no_target_entity_coordinate_or_named_word_policy']=(all(token not in source for token in ('mira','nora','engineer','target_entity','expected','category')))
    checks['checkpoint_has_no_hand_trajectory_or_resolved_target']=(str(near_motion) not in blob and str(far_motion) not in blob and str(test[0][0]) not in blob and str(test[1][0]) not in blob)
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-raw-gesture-deictic-grounding.v1','contract':'FOUNDRY_RAW_GESTURE_DEICTIC_GROUNDING_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'RAW_DYNAMIC_HAND_TRAJECTORY_NOW_ESTABLISHES_THE_JOINTLY_ATTENDED_VISUAL_REFERENT_FOR_A_LEARNED_DEMONSTRATIVE_WITHOUT_HOST_POINT_OR_ENTITY','checks':checks,'failed':failed,'remaining_red':['LEARNED_GESTURE_STYLE_VARIATION','GAZE_PLUS_POINTING_MULTIMODAL_FUSION','DEICTIC_REFERENCE_TO_REMEMBERED_NONVISIBLE_OBJECT','DIRECT_RAW_GESTURE_DEICTIC_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
