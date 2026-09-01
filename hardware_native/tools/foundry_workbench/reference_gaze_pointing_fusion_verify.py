#!/usr/bin/env python3
"""N+1: raw gaze and pointing direction jointly disambiguate a visual individual that either cue alone misses."""
from __future__ import annotations
import copy,inspect,json,time
from reference_continuous_visual_sensor_ownership_verify import A1,train_level1
from reference_discovered_visual_individual_naming_verify import CLAUSE,MIRA
from reference_global_discourse_relevance_verify import fresh
from reference_multi_visual_object_file_tracker_v1 import MultiVisualObjectFileTrackerV1
from reference_multiple_visual_object_files_verify import canvas,train_temporal
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_raw_pointing_motion_v1 import RawPointingMotionV1
from reference_social_directional_cue_fusion_v1 import SocialDirectionalCueFusionV1
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

SOURCE=0xDB01

def motion(origin,endpoint):
    oy,ox=origin;ey,ex=endpoint;return ((oy,ox),((oy+ey)//2,(ox+ex)//2),(ey,ex))

def main():
    started=time.perf_counter();checks={};level1=train_level1();temporal=train_temporal(level1);adult,*_=fresh()
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));sensor=VisualSensorIngressV1();tracker=MultiVisualObjectFileTrackerV1();temp=TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint()))
    frame=canvas(((A1,1,1),(A1,1,8),(A1,1,15)),h=8,w=24)
    rows=tracker.observe(o,sensor,temp,level1,SOURCE,1,frame,VisualSensorIngressV1.frame_digest(frame))
    if len(rows)!=3:raise RuntimeError('fusion:three_objects')
    left,center,right=[row[0] for row in rows];positions=[(row[1],row[2]) for row in rows]
    checks['three_same_category_organism_individuals_are_simultaneously_visible']=(len({left,center,right})==3)

    origin=(77,21)  # far below center: target vectors ~= (-70,-14),(-70,0),(-70,+14)
    point_motion=motion(origin,(7,14))  # midway left/center
    gaze_motion=motion(origin,(7,28))   # midway center/right
    point_only=RawPointingMotionV1.target(tracker,point_motion);gaze_only=RawPointingMotionV1.target(tracker,gaze_motion)
    fused=SocialDirectionalCueFusionV1.resolve(tracker,point_motion,gaze_motion)
    checks['pointing_and_gaze_alone_prefer_different_flanking_objects']=(point_only==left and gaze_only==right)
    checks['equal_weight_multimodal_fusion_selects_shared_center_candidate']=(fused==center)

    # Expose the selected identity through ordinary productive language.
    adult.observe_surface_item(center,MIRA,0xDB11);adult.observe_surface_item(center,MIRA,0xDB12)
    # Two other subject exemplars establish productive subject diversity.
    adult.observe_surface_item(left,b'nora',0xDB21);adult.observe_surface_item(left,b'nora',0xDB22)
    adult.observe_surface_item(right,b'liam',0xDB31);adult.observe_surface_item(right,b'liam',0xDB32)
    for src in (0xDB41,0xDB42):adult.observe_surface_construction(CLAUSE,(101,left,301,401),b'the careful nora tests the sensor.',src)
    for src in (0xDB51,0xDB52):adult.observe_surface_construction(CLAUSE,(102,right,302,402),b'the quiet liam inspects the valve.',src)
    response=bytes(adult.leaf(CLAUSE,(101,fused,301,401)).surface) if fused else b''
    checks['fused_social_attention_target_drives_ordinary_productive_language']=(response==b'the careful mira tests the sensor.')

    # Strongly conflicting cues have no common candidate and refuse.
    direct_left=motion(origin,(7,0));direct_right=motion(origin,(7,42))
    conflict=SocialDirectionalCueFusionV1.resolve(tracker,direct_left,direct_right)
    checks['strongly_disagreeing_gaze_and_pointing_refuse_instead_of_priority_rule']=(conflict==0)

    # If either cue is static/invalid, fusion cannot substitute the other cue alone.
    static=(origin,origin,origin);one_invalid=SocialDirectionalCueFusionV1.resolve(tracker,point_motion,static)
    checks['one_invalid_directional_modality_cannot_be_silently_ignored']=(one_invalid==0)

    source=(inspect.getsource(SocialDirectionalCueFusionV1)+inspect.getsource(RawPointingMotionV1)).lower()
    checks['fusion_has_no_left_center_right_name_or_modality_priority_policy']=(all(token not in source for token in ('mira','nora','liam','left','center','right','priority','expected')))
    checks['fusion_is_transient_and_adds_no_checkpoint_state']=not hasattr(SocialDirectionalCueFusionV1(),'checkpoint')
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-gaze-pointing-fusion.v1','contract':'FOUNDRY_GAZE_POINTING_FUSION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'GAZE_AND_POINTING_NOW_FUSE_TO_SELECT_A_SHARED_VISUAL_INDIVIDUAL_THAT_EITHER_DIRECTIONAL_CUE_ALONE_SELECTS_INCORRECTLY','conversation':['joint-attention',response.decode() if response else ''],'checks':checks,'failed':failed,'remaining_red':['LEARNED_MODALITY_RELIABILITY_WEIGHTING','RAW_GAZE_SENSOR_OWNERSHIP','VERBAL_DEICTIC_PLUS_MULTIMODAL_FUSION','DIRECT_SOCIAL_CUE_FUSION_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
