#!/usr/bin/env python3
"""N+1: learned deictic marker + pointing selects an organism-discovered individual."""
from __future__ import annotations
import copy,inspect,json,time
from reference_continuous_visual_sensor_ownership_verify import A1,A2,train_level1
from reference_discovered_visual_individual_naming_verify import CLAUSE,NORA,MIRA
from reference_global_discourse_relevance_verify import fresh
from reference_multi_visual_object_file_tracker_v1 import MultiVisualObjectFileTrackerV1
from reference_multiple_visual_object_files_verify import canvas,train_temporal
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_visual_deictic_reference_v1 import VisualDeicticReferenceV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

SOURCE=0xD101;THIS=0xD201

def obs(t,o,s,temp,level1,seq,frame):
    return t.observe(o,s,temp,level1,SOURCE,seq,frame,VisualSensorIngressV1.frame_digest(frame))

def main():
    started=time.perf_counter();checks={};level1=train_level1();temporal=train_temporal(level1);adult,*_=fresh()
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));sensor=VisualSensorIngressV1();tracker=MultiVisualObjectFileTrackerV1();temp=TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint()))
    first=obs(tracker,o,sensor,temp,level1,1,canvas(((A1,1,1),(A1,1,14))))
    second=obs(tracker,o,sensor,temp,level1,2,canvas(((A2,1,2),(A2,1,13))))
    if len(second)!=2:raise RuntimeError('deictic:visual_seed')
    left,right=second[0][0],second[1][0];left_pos=(second[0][1],second[0][2]);right_pos=(second[1][1],second[1][2])
    checks['two_same_category_organism_individuals_share_one_visual_scene']=(left!=right and all(row[4] for row in second))

    # Names only expose which individual the deictic selected downstream.
    adult.observe_surface_item(left,MIRA,0xD301);adult.observe_surface_item(left,MIRA,0xD302)
    adult.observe_surface_item(right,NORA,0xD311);adult.observe_surface_item(right,NORA,0xD312)
    for src in (0xD321,0xD322):adult.observe_surface_construction(CLAUSE,(101,left,301,401),b'the careful mira tests the sensor.',src)
    for src in (0xD331,0xD332):adult.observe_surface_construction(CLAUSE,(102,right,302,402),b'the quiet nora inspects the valve.',src)

    adult.observe_surface_item(THIS,b'this',0xD401);one=adult.language.lexeme(THIS)
    adult.observe_surface_item(THIS,b'this',0xD401);repeat=adult.language.lexeme(THIS)
    adult.observe_surface_item(THIS,b'this',0xD402);learned=adult.language.lexeme(THIS)
    checks['deictic_marker_requires_independent_source_support']=(one is None and repeat is None and learned==tuple(b'this'))

    left_ref=VisualDeicticReferenceV1.resolve(adult,tracker,b'this',THIS,*left_pos)
    right_ref=VisualDeicticReferenceV1.resolve(adult,tracker,b'this',THIS,*right_pos)
    left_response=bytes(adult.leaf(CLAUSE,(101,left_ref,301,401)).surface) if left_ref else b''
    right_response=bytes(adult.leaf(CLAUSE,(102,right_ref,302,402)).surface) if right_ref else b''
    checks['identical_this_surface_selects_left_or_right_individual_from_pointing_location']=(left_ref==left and right_ref==right and left_ref!=right_ref)
    checks['deictic_referents_drive_distinct_ordinary_productive_language']=(left_response==b'the careful mira tests the sensor.' and right_response==b'the quiet nora inspects the valve.')

    midpoint=((left_pos[0]+right_pos[0])//2,(left_pos[1]+right_pos[1])//2)
    tie=VisualDeicticReferenceV1.resolve(adult,tracker,b'this',THIS,*midpoint)
    unknown=VisualDeicticReferenceV1.resolve(adult,tracker,b'that',THIS,*left_pos)
    checks['equidistant_pointing_refuses_without_identity_tiebreak']=(tie==0)
    checks['unlearned_deictic_surface_refuses']=(unknown==0)

    third=obs(tracker,o,sensor,temp,level1,3,canvas(((A1,1,3),)))
    hidden_ref=VisualDeicticReferenceV1.resolve(adult,tracker,b'this',THIS,*right_pos)
    checks['dormant_or_occluded_individual_cannot_be_claimed_by_current_visual_deixis']=(len(third)==1 and hidden_ref!=right)

    adult.language.withdraw_source(0xD402);withdrawn=VisualDeicticReferenceV1.resolve(adult,tracker,b'this',THIS,*left_pos)
    checks['deictic_source_withdrawal_blocks_reference_without_erasing_entities']=(withdrawn==0 and left in o.entity_features and right in o.entity_features)

    source=inspect.getsource(VisualDeicticReferenceV1).lower();blob=json.dumps(adult.checkpoint(),sort_keys=True)
    checks['deictic_bridge_has_no_host_entity_argument_or_named_target_policy']=(all(token not in source for token in ('mira','nora','engineer','category','expected')))
    checks['checkpoint_has_no_pointing_location_or_deictic_referent_cache']=(all(token not in blob for token in ('point_y2','point_x2','deictic_referent','pointing_location')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-deictic-visual-individual-grounding.v1','contract':'FOUNDRY_DEICTIC_VISUAL_INDIVIDUAL_GROUNDING_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'IDENTICAL_THIS_NOW_REFERS_TO_DIFFERENT_ORGANISM_DISCOVERED_INDIVIDUALS_FROM_SHARED_POINTING_LOCATION_AND_DRIVES_DIFFERENT_PRODUCTIVE_LANGUAGE','conversation':[['this@left',left_response.decode()],['this@right',right_response.decode()]],'checks':checks,'failed':failed,'remaining_red':['DEICTIC_NEAR_FAR_CONTRAST_THIS_VS_THAT','GESTURE_LEARNING_FROM_RAW_BODY_MOTION','DEICTIC_REFERENCE_TO_REMEMBERED_NONVISIBLE_OBJECT','DIRECT_DEICTIC_GROUNDING_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
