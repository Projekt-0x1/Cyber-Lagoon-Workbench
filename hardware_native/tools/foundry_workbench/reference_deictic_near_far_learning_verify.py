#!/usr/bin/env python3
"""N+1: opaque demonstrative markers learn distinct spatial applicability from pointing episodes."""
from __future__ import annotations
import copy,inspect,json,time
from reference_continuous_visual_sensor_ownership_verify import A1,A2,train_level1
from reference_global_discourse_relevance_verify import fresh
from reference_multi_visual_object_file_tracker_v1 import MultiVisualObjectFileTrackerV1
from reference_multiple_visual_object_files_verify import canvas,train_temporal
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_visual_deictic_distance_ecology_v1 import VisualDeicticDistanceEcologyV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

SOURCE=0xD501;MARK_A=0xD601;MARK_B=0xD602
A_RAW=b'this';B_RAW=b'that';A_SOURCES=(0xD611,0xD612);B_SOURCES=(0xD621,0xD622)
SPEAKER=(5,0)

def obs(t,o,s,temp,level1,seq,items,w=40):
    frame=canvas(items,h=8,w=w)
    return t.observe(o,s,temp,level1,SOURCE,seq,frame,VisualSensorIngressV1.frame_digest(frame))

def main():
    started=time.perf_counter();checks={};level1=train_level1();temporal=train_temporal(level1);adult,*_=fresh()
    # Lexical forms are learned independently of their spatial applicability.
    for source in A_SOURCES:adult.observe_surface_item(MARK_A,A_RAW,source)
    for source in B_SOURCES:adult.observe_surface_item(MARK_B,B_RAW,source)
    checks['two_opaque_demonstrative_lexemes_are_learned_without_spatial_labels']=(adult.language.lexeme(MARK_A)==tuple(A_RAW) and adult.language.lexeme(MARK_B)==tuple(B_RAW))

    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));sensor=VisualSensorIngressV1();tracker=MultiVisualObjectFileTrackerV1();temp=TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint()));eco=VisualDeicticDistanceEcologyV1()
    # Development positions: left center x2=7, right center x2=31; both y2=7.
    r1=obs(tracker,o,sensor,temp,level1,1,((A1,1,1),(A1,1,13)))
    if len(r1)!=2:raise RuntimeError('deictic_distance:seed')
    left_pos=(r1[0][1],r1[0][2]);right_pos=(r1[1][1],r1[1][2])
    for source in A_SOURCES:eco.observe(adult,tracker,A_RAW,*left_pos,*SPEAKER,source)
    for source in B_SOURCES:eco.observe(adult,tracker,B_RAW,*right_pos,*SPEAKER,source)
    profile_a=eco.profile(adult,MARK_A);profile_b=eco.profile(adult,MARK_B)
    checks['independent_pointing_episodes_create_distinct_marker_distance_profiles']=(profile_a is not None and profile_b is not None and profile_a<profile_b)

    # Held-out positions after motion/view change: x2=9 and x2=29.
    r2=obs(tracker,o,sensor,temp,level1,2,((A2,1,2),(A2,1,12)))
    near_pos=(r2[0][1],r2[0][2]);far_pos=(r2[1][1],r2[1][2]);near_id,far_id=r2[0][0],r2[1][0]
    a_near=eco.resolve(adult,tracker,A_RAW,*near_pos,*SPEAKER)
    b_far=eco.resolve(adult,tracker,B_RAW,*far_pos,*SPEAKER)
    a_far=eco.resolve(adult,tracker,A_RAW,*far_pos,*SPEAKER)
    b_near=eco.resolve(adult,tracker,B_RAW,*near_pos,*SPEAKER)
    checks['heldout_distances_transfer_each_marker_to_its_learned_spatial_band']=(a_near==near_id and b_far==far_id)
    checks['swapped_marker_distance_combinations_refuse_without_hardcoded_this_that_branch']=(a_far==0 and b_near==0)

    # New single object exactly midway between profile centers -> equal marker error.
    mo=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));ms=VisualSensorIngressV1();mt=MultiVisualObjectFileTrackerV1();mtemp=TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint()))
    mid=obs(mt,mo,ms,mtemp,level1,1,((A1,1,7),),w=30)[0]
    mid_pos=(mid[1],mid[2]);mid_a=eco.resolve(adult,mt,A_RAW,*mid_pos,*SPEAKER);mid_b=eco.resolve(adult,mt,B_RAW,*mid_pos,*SPEAKER)
    checks['equidistant_profile_boundary_refuses_both_markers']=(mid_a==0 and mid_b==0)

    # One marker loses one of its two supporting sources: lexical authority and spatial
    # profile both cease to support its use; the other marker is unaffected.
    adult.language.withdraw_source(A_SOURCES[1]);withdrawn_a=eco.resolve(adult,tracker,A_RAW,*near_pos,*SPEAKER);still_b=eco.resolve(adult,tracker,B_RAW,*far_pos,*SPEAKER)
    checks['source_withdrawal_removes_one_profile_and_fail_closes_the_contrast_without_rewriting_other_profile']=(withdrawn_a==0 and still_b==0 and eco.profile(adult,MARK_A) is None and eco.profile(adult,MARK_B) is not None and adult.language.lexeme(MARK_B)==tuple(B_RAW))

    cp=eco.checkpoint();restored=VisualDeicticDistanceEcologyV1.restore(copy.deepcopy(cp));source=inspect.getsource(VisualDeicticDistanceEcologyV1).lower();blob=json.dumps(cp,sort_keys=True)
    checks['checkpoint_preserves_numeric_source_distance_evidence_not_words_or_targets']=(restored.support==eco.support and 'this' not in blob and 'that' not in blob and str(near_id) not in blob and str(far_id) not in blob)
    checks['distance_ecology_has_no_this_that_near_far_or_category_policy']=(all(token not in source for token in ("'this'","'that'",'near','far','category','concept','expected')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-deictic-near-far-learning.v1','contract':'FOUNDRY_DEICTIC_NEAR_FAR_LEARNING_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'TWO_OPAQUE_DEMONSTRATIVES_NOW_LEARN_DISTINCT_SPEAKER_RELATIVE_SPATIAL_APPLICABILITY_FROM_POINTING_EPISODES_AND_TRANSFER_TO_HELDOUT_POSITIONS','profiles':{'a':profile_a,'b':profile_b},'checks':checks,'failed':failed,'remaining_red':['GESTURE_LEARNING_FROM_RAW_BODY_MOTION','ACTION_SPACE_DEPENDENT_DEMONSTRATIVE_SHIFT','DEICTIC_REFERENCE_TO_REMEMBERED_NONVISIBLE_OBJECT','DIRECT_DEMONSTRATIVE_DISTANCE_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
