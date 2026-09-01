#!/usr/bin/env python3
"""N+1: same-category object identity follows motion prediction through a hidden crossing."""
from __future__ import annotations
import copy,json,time
from reference_continuous_visual_sensor_ownership_verify import A1,A2,train_level1,vf
from reference_multi_visual_object_file_tracker_v1 import MultiVisualObjectFileTrackerV1
from reference_multiple_visual_object_files_verify import train_temporal
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

SOURCE=0xBF01

def canvas(items,h=20,w=20):
    out=[[0]*w for _ in range(h)]
    for image,y0,x0 in items:
        for y,row in enumerate(image):
            for x,value in enumerate(row):
                if out[y0+y][x0+x]!=0:raise RuntimeError('crossing:overlap')
                out[y0+y][x0+x]=int(value)
    return tuple(tuple(row) for row in out)

def obs(t,o,s,temporal,level1,seq,frame):
    return t.observe(o,s,temporal,level1,SOURCE,seq,frame,VisualSensorIngressV1.frame_digest(frame))

def run_seed(level1,temporal):
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));s=VisualSensorIngressV1();t=MultiVisualObjectFileTrackerV1()
    r1=obs(t,o,s,temporal,level1,1,canvas(((A1,0,0),(A1,14,14))))
    r2=obs(t,o,s,temporal,level1,2,canvas(((A2,4,4),(A2,10,10))))
    if len(r1)!=len(r2)!=2:raise RuntimeError('crossing:seed')
    return o,s,t,r1,r2

def main():
    started=time.perf_counter();checks={};level1=train_level1();temporal=train_temporal(level1);ta=temporal.relation(vf(level1,A1),vf(level1,A2))
    o,s,t,r1,r2=run_seed(level1,temporal);northwest,southeast=r1[0][0],r1[1][0]
    checks['same_category_tracks_have_opposed_diagonal_velocities_before_hidden_crossing']=(
        [row[0] for row in r2]==[northwest,southeast]
        and t.active[northwest][2]>0 and t.active[northwest][3]>0
        and t.active[southeast][2]<0 and t.active[southeast][3]<0)

    # Collision/crossing itself is unobserved. One blank frame preserves one-step dormant tracks.
    blank=canvas(());obs(t,o,s,temporal,level1,3,blank)
    # Reappearance after trajectories have crossed: physical NW-origin object is now SE;
    # physical SE-origin object is now NW. Candidate ordering is NW then SE.
    crossed=canvas(((A1,2,2),(A1,12,12)));r4=obs(t,o,s,temporal,level1,4,crossed)
    checks['motion_prediction_preserves_identity_after_same_category_crossing']=(
        len(r4)==2 and r4[0][0]==southeast and r4[1][0]==northwest
        and all(row[4] and row[3]==ta for row in r4))

    # Destructive lesion: erase only velocity after frame2. Last-position matching
    # then assigns the opposite identities, proving the GREEN is not feature/category alone.
    lo,ls,lt,lr1,lr2=run_seed(level1,TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint())))
    for entity,row in tuple(lt.active.items()):
        y2,x2,_vy,_vx,feature,misses=row;lt.active[entity]=(y2,x2,0,0,feature,misses)
    obs(lt,lo,ls,TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint())),level1,3,blank)
    lesion_temporal=TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint()))
    lesion=obs(lt,lo,ls,lesion_temporal,level1,4,crossed)
    checks['velocity_lesion_reverses_crossing_identity_assignment']=(
        len(lesion)==2 and lesion[0][0]==lr1[0][0] and lesion[1][0]==lr1[1][0]
        and [row[0] for row in lesion]!=[row[0] for row in r4])

    # Exact geometric tie with same category remains fail-closed.
    ao=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));asensor=VisualSensorIngressV1();at=MultiVisualObjectFileTrackerV1();atemp=TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint()))
    a1=obs(at,ao,asensor,atemp,level1,1,canvas(((A1,0,0),(A1,14,14))))
    a2=obs(at,ao,asensor,atemp,level1,2,canvas(((A2,4,4),(A2,10,10))))
    # Force equal, opposed velocities to zero prediction around midpoint then hide.
    for entity,row in tuple(at.active.items()):
        y2,x2,_vy,_vx,feature,misses=row;at.active[entity]=(y2,x2,0,0,feature,misses)
    obs(at,ao,asensor,atemp,level1,3,blank);before=copy.deepcopy(at.active);features={e:ao._active_entity_features(e) for e in before}
    # One central candidate is equidistant from both last positions.
    ambiguous=obs(at,ao,asensor,atemp,level1,4,canvas(((A1,7,7),)))
    checks['same_category_geometric_tie_refuses_without_recency_or_id_tiebreak']=(
        ambiguous==() and at.active==before and {e:ao._active_entity_features(e) for e in before}==features)

    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-same-category-crossing-trajectory.v1','contract':'FOUNDRY_SAME_CATEGORY_CROSSING_TRAJECTORY_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'VISUALLY_IDENTICAL_ORGANISM_OBJECTS_NOW_KEEP_INDIVIDUAL_IDENTITY_THROUGH_A_HIDDEN_TRAJECTORY_CROSSING_USING_LEARNED_MOTION_NOT_LAST_POSITION','checks':checks,'failed':failed,'entities':{'northwest_origin':northwest,'southeast_origin':southeast},'remaining_red':['LONGER_OCCLUSION_FROM_LEARNED_MOTION_UNCERTAINTY','OBJECT_INTERACTION_AND_CAUSAL_ROLE_LEARNING','DIRECT_CROSSING_TRAJECTORY_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
