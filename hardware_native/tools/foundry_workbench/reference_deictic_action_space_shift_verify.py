#!/usr/bin/env python3
"""N+1: demonstrative applicability shifts with current action extent while target position stays fixed."""
from __future__ import annotations
import copy,inspect,json,time
from reference_continuous_visual_sensor_ownership_verify import A1,train_level1
from reference_global_discourse_relevance_verify import fresh
from reference_multi_visual_object_file_tracker_v1 import MultiVisualObjectFileTrackerV1
from reference_multiple_visual_object_files_verify import canvas,train_temporal
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_visual_deictic_action_space_v1 import VisualDeicticActionSpaceV1,Q
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

SOURCE=0xD701;M1=0xD801;M2=0xD802;RAW1=b'this';RAW2=b'that'
S1=(0xD811,0xD812);S2=(0xD821,0xD822);SPEAKER=(5,0)

def main():
    started=time.perf_counter();checks={};level1=train_level1();temporal=train_temporal(level1);adult,*_=fresh()
    for s in S1:adult.observe_surface_item(M1,RAW1,s)
    for s in S2:adult.observe_surface_item(M2,RAW2,s)
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));sensor=VisualSensorIngressV1();tracker=MultiVisualObjectFileTrackerV1();temp=TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint()));eco=VisualDeicticActionSpaceV1()
    # Single target center x2=29,y2=7 -> speaker-relative distance 31.
    frame=canvas(((A1,1,12),),h=8,w=30);rows=tracker.observe(o,sensor,temp,level1,SOURCE,1,frame,VisualSensorIngressV1.frame_digest(frame));target=rows[0][0];point=(rows[0][1],rows[0][2]);distance=abs(point[0]-SPEAKER[0])+abs(point[1]-SPEAKER[1])
    # Development teaches only relative action-space profiles: M1 around 0.5x reach,
    # M2 around 2x reach, at the exact same target position.
    for source,extent in zip(S1,(60,64)):eco.observe(adult,tracker,RAW1,*point,*SPEAKER,extent,source)
    for source,extent in zip(S2,(15,16)):eco.observe(adult,tracker,RAW2,*point,*SPEAKER,extent,source)
    p1=eco.profile(adult,M1);p2=eco.profile(adult,M2)
    checks['two_opaque_markers_learn_distinct_target_to_action_extent_ratios']=(p1 is not None and p2 is not None and p1<p2)

    # Held-out body states, same raw frame/target/pointing. Extended action range makes
    # the M1 profile fit; contracted range makes M2 fit.
    extended=62;contracted=15
    m1_ext=eco.resolve(adult,tracker,RAW1,*point,*SPEAKER,extended)
    m2_ext=eco.resolve(adult,tracker,RAW2,*point,*SPEAKER,extended)
    m1_contract=eco.resolve(adult,tracker,RAW1,*point,*SPEAKER,contracted)
    m2_contract=eco.resolve(adult,tracker,RAW2,*point,*SPEAKER,contracted)
    checks['same_physical_target_changes_demonstrative_fit_when_action_extent_changes']=(m1_ext==target and m2_contract==target)
    checks['opposite_marker_refuses_in_each_action_space']=(m2_ext==0 and m1_contract==0)

    # Integer body extent may not realize an exact arithmetic tie. The nearest
    # realizable boundary must flip between adjacent body states with no object move.
    edge_a=eco.resolve(adult,tracker,RAW1,*point,*SPEAKER,25);edge_a_other=eco.resolve(adult,tracker,RAW2,*point,*SPEAKER,25)
    edge_b=eco.resolve(adult,tracker,RAW2,*point,*SPEAKER,24);edge_b_other=eco.resolve(adult,tracker,RAW1,*point,*SPEAKER,24)
    checks['adjacent_action_extents_flip_demonstrative_winner_at_learned_boundary']=(edge_a==target and edge_a_other==0 and edge_b==target and edge_b_other==0)

    cp=eco.checkpoint();restored=VisualDeicticActionSpaceV1.restore(copy.deepcopy(cp));source=inspect.getsource(VisualDeicticActionSpaceV1).lower();blob=json.dumps(cp,sort_keys=True)
    checks['checkpoint_stores_only_source_qualified_ratio_evidence_not_target_or_body_episode']=(restored.support==eco.support and str(target) not in blob and 'action_extent' not in blob)
    checks['action_space_ecology_contains_no_this_that_near_far_or_reachability_branch']=(all(token not in source for token in ("'this'","'that'",'near','far','reachable','engineer','expected')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-deictic-action-space-shift.v1','contract':'FOUNDRY_DEICTIC_ACTION_SPACE_SHIFT_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'THE_SAME_POINTED_OBJECT_NOW_SHIFTS_BETWEEN_TWO_LEARNED_DEMONSTRATIVES_WHEN_CURRENT_ACTION_EXTENT_CHANGES_WITHOUT_MOVING_THE_OBJECT','profiles_q16':{'a':p1,'b':p2},'checks':checks,'failed':failed,'remaining_red':['GESTURE_LEARNING_FROM_RAW_BODY_MOTION','DEICTIC_REFERENCE_TO_REMEMBERED_NONVISIBLE_OBJECT','MULTIFACTOR_SOCIAL_ATTENTION_DEMONSTRATIVE_SHIFT','DIRECT_ACTION_SPACE_DEICTIC_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
