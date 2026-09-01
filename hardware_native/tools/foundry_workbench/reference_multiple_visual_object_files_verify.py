#!/usr/bin/env python3
"""N+1: multiple raw visual components recruit and preserve distinct organism object files."""
from __future__ import annotations
import copy,inspect,json,time
from reference_continuous_visual_sensor_ownership_verify import (
    A1,A2,B1,B2,SRC_A,SRC_B,train_level1,vf,SURFACE,pair,
)
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_multi_visual_object_file_tracker_v1 import MultiVisualObjectFileTrackerV1
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES
from reference_population_v1 import PopulationSpecV1
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_unsupervised_perceptual_features_v1 import FEATURE_QUORUM
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

KNOWN_A=0xBC01;KNOWN_B=0xBC02;FEATURE_SOURCE=0xBC10;TRACK_SOURCE=0xBC20;MIXED_SOURCE=0xBC21

def canvas(items,h=8,w=22):
    out=[[0]*w for _ in range(h)]
    for image,y0,x0 in items:
        for y,row in enumerate(image):
            for x,value in enumerate(row):
                if out[y0+y][x0+x]!=0:raise RuntimeError('multi_object:overlap_fixture')
                out[y0+y][x0+x]=int(value)
    return tuple(tuple(row) for row in out)

def train_temporal(level1):
    t=TemporalVisualContinuityV1()
    for left,right in ((A1,A2),(B1,B2)):
        lf,rf=vf(level1,left),vf(level1,right)
        for _ in range(FEATURE_QUORUM):
            t.gap();t.observe_features((lf,));t.observe_features((rf,))
    return t

def observe(tracker,o,sensor,temporal,level1,source,seq,frame):
    return tracker.observe(o,sensor,temporal,level1,source,seq,frame,VisualSensorIngressV1.frame_digest(frame))

def install(o,entity,feature):o.contact(CONTACT_ENTITY_FEATURES,(entity,1,feature),FEATURE_SOURCE,True,True)
def ground(g,adult,o,entity,concept,source):
    for n in range(2):pair(g,adult,o,entity,SURFACE[concept],source+n)

def main():
    started=time.perf_counter();checks={};level1=train_level1();temporal=train_temporal(level1)
    ta=temporal.relation(vf(level1,A1),vf(level1,A2));tb=temporal.relation(vf(level1,B1),vf(level1,B2))
    if not ta or not tb or ta==tb:raise RuntimeError('multi_object:temporal')
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));sensor=VisualSensorIngressV1();tracker=MultiVisualObjectFileTrackerV1()

    # Two visually identical-category objects appear simultaneously at separated positions.
    frame1=canvas(((A1,1,1),(A1,1,14)))
    first=observe(tracker,o,sensor,temporal,level1,TRACK_SOURCE,1,frame1)
    checks['one_raw_frame_mints_two_distinct_same_category_entities']=(
        len(first)==2 and first[0][0]!=first[1][0] and all(not row[4] for row in first))
    left,right=first[0][0],first[1][0]
    before_motion=(tuple(o._active_entity_features(left)),tuple(o._active_entity_features(right)))

    # Both move inward one pixel and change view A1 -> A2. Nearest compatible
    # spatial assignment should preserve each identity without swapping.
    frame2=canvas(((A2,1,2),(A2,1,13)))
    second=observe(tracker,o,sensor,temporal,level1,TRACK_SOURCE,2,frame2)
    checks['two_same_category_objects_preserve_left_right_identity_through_motion_and_view_change']=(
        len(second)==2 and [row[0] for row in second]==[left,right]
        and all(row[4] and row[3]==ta for row in second)
        and o._active_entity_features(left)==o._active_entity_features(right)==(ta,))

    # Collision: crops become one connected component. Tracker must refuse this
    # frame rather than collapse/swap either identity. Sensor occurrence advances,
    # but active object map and organism entity features remain unchanged.
    active_before=copy.deepcopy(tracker.active);features_before={left:o._active_entity_features(left),right:o._active_entity_features(right)}
    collision=canvas(((A1,1,4),(A1,1,10)))  # adjacent filled rectangles merge
    collision_result=observe(tracker,o,sensor,temporal,level1,TRACK_SOURCE,3,collision)
    checks['merged_collision_blob_refuses_without_identity_swap_or_entity_mutation']=(
        collision_result==() and tracker.active==active_before
        and {left:o._active_entity_features(left),right:o._active_entity_features(right)}==features_before)

    # A sensor sequence gap breaks all current object files. Same raw two-object
    # category therefore mints two fresh identities, not stale reidentification.
    frame4=canvas(((A1,1,1),(A1,1,14)))
    fourth=observe(tracker,o,sensor,temporal,level1,TRACK_SOURCE,5,frame4)
    fresh={row[0] for row in fourth}
    checks['sequence_gap_breaks_both_files_and_mints_two_fresh_individuals']=(
        len(fourth)==2 and fresh.isdisjoint({left,right}) and all(not row[4] for row in fourth))

    # Independent clean stream with one A and one B tests simultaneous distinct
    # category identities and downstream category grounding.
    sensor2=VisualSensorIngressV1();tracker2=MultiVisualObjectFileTrackerV1();o2=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    mixed1=canvas(((A1,1,1),(B1,1,14)));mixed2=canvas(((A2,1,2),(B2,1,13)))
    m1=observe(tracker2,o2,sensor2,temporal,level1,MIXED_SOURCE,1,mixed1)
    m2=observe(tracker2,o2,sensor2,temporal,level1,MIXED_SOURCE,2,mixed2)
    checks['simultaneous_different_categories_preserve_two_identity_tracks']=(
        len(m1)==len(m2)==2 and [r[0] for r in m1]==[r[0] for r in m2]
        and {m2[0][3],m2[1][3]}=={ta,tb})

    from reference_global_discourse_relevance_verify import fresh as adult_fresh
    adult,*_=adult_fresh();g=CrossmodalConceptGroundingV1();install(o2,KNOWN_A,ta);install(o2,KNOWN_B,tb)
    ground(g,adult,o2,KNOWN_A,201,0xBD00);ground(g,adult,o2,KNOWN_B,203,0xBD10)
    mixed_ids=[row[0] for row in m2];resolved={g.resolve_world_atom(adult,o2,e) for e in mixed_ids}
    checks['two_organism_discovered_tracks_independently_inherit_grounded_concepts']=(resolved=={201,203})

    # Checkpoint keeps durable organism identities but not active multi-object files.
    ro=ReferenceOrganismV2.restore(copy.deepcopy(o2.checkpoint()));rt=MultiVisualObjectFileTrackerV1.restore(tracker2.checkpoint());rs=VisualSensorIngressV1.restore(copy.deepcopy(sensor2.checkpoint()))
    post=observe(rt,ro,rs,TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint())),level1,MIXED_SOURCE,3,mixed1)
    checks['restart_preserves_discovered_entities_but_not_active_multi_object_assignments']=(
        all(ro._active_entity_features(e) in ((ta,),(tb,)) for e in mixed_ids)
        and len(post)==2 and {r[0] for r in post}.isdisjoint(set(mixed_ids)))

    source=(inspect.getsource(MultiVisualObjectFileTrackerV1)+inspect.getsource(__import__('reference_raw_visual_object_candidates_v1').RawVisualObjectCandidatesV1)).lower()
    checks['multi_object_tracker_has_no_category_language_expected_or_reward_authority']=(all(token not in source for token in ('category','language','concept','expected','reward','engineer')))
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-multiple-visual-object-files.v1','contract':'FOUNDRY_MULTIPLE_VISUAL_OBJECT_FILES_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'ONE_RAW_FRAME_NOW_RECRUITS_MULTIPLE_DISTINCT_ORGANISM_ENTITY_FILES_THAT_PRESERVE_IDENTITY_THROUGH_MOTION_AND_REFUSE_COLLISION_SWAPS','entities':{'same_category':[left,right],'fresh_after_gap':sorted(fresh),'mixed':[r[0] for r in m2]},'checks':checks,'failed':failed,'remaining_red':['OCCLUSION_AND_REIDENTIFICATION','CROSSING_TRAJECTORY_IDENTITY_WITH_SAME_CATEGORY','DIRECT_MULTIPLE_OBJECT_FILE_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
