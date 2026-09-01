#!/usr/bin/env python3
"""N+1: bounded visual object files reidentify after one-frame occlusion from learned motion + feature continuity."""
from __future__ import annotations
import copy,json,time
from reference_continuous_visual_sensor_ownership_verify import A1,A2,B1,B2,train_level1,vf
from reference_multi_visual_object_file_tracker_v1 import MultiVisualObjectFileTrackerV1
from reference_multiple_visual_object_files_verify import train_temporal
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

SOURCE=0xBE01

def canvas(items,h=15,w=22):
    out=[[0]*w for _ in range(h)]
    for image,y0,x0 in items:
        for y,row in enumerate(image):
            for x,value in enumerate(row):
                if out[y0+y][x0+x]!=0:raise RuntimeError('occlusion:overlap_fixture')
                out[y0+y][x0+x]=int(value)
    return tuple(tuple(row) for row in out)

def obs(tracker,o,sensor,temporal,level1,seq,frame):
    return tracker.observe(o,sensor,temporal,level1,SOURCE,seq,frame,VisualSensorIngressV1.frame_digest(frame))

def main():
    started=time.perf_counter();checks={};level1=train_level1();temporal=train_temporal(level1)
    ta=temporal.relation(vf(level1,A1),vf(level1,A2));tb=temporal.relation(vf(level1,B1),vf(level1,B2))
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));sensor=VisualSensorIngressV1();tracker=MultiVisualObjectFileTrackerV1()

    # Two tracks acquire nonzero velocities on separate rows.
    f1=canvas(((A1,0,1),(B1,8,14)));r1=obs(tracker,o,sensor,temporal,level1,1,f1)
    f2=canvas(((A2,0,3),(B2,8,12)));r2=obs(tracker,o,sensor,temporal,level1,2,f2)
    if len(r1)!=2 or len(r2)!=2:raise RuntimeError('occlusion:seed')
    top,bottom=r1[0][0],r1[1][0]
    checks['two_tracks_acquire_distinct_motion_state_before_occlusion']=(
        [row[0] for row in r2]==[top,bottom] and tracker.active[top][3]>0 and tracker.active[bottom][3]<0)

    # Top disappears for one frame. Bottom continues. Top must remain dormant only.
    f3=canvas(((B1,8,10),));r3=obs(tracker,o,sensor,temporal,level1,3,f3)
    checks['one_missing_object_becomes_dormant_while_visible_peer_continues']=(
        len(r3)==1 and r3[0][0]==bottom and top in tracker.active and tracker.active[top][-1]==1
        and tracker.active[bottom][-1]==0)
    top_features_before=o._active_entity_features(top)

    # Reappearance exactly at top's 2-step predicted position; bottom continues on its row.
    f4=canvas(((A1,0,7),(B2,8,8)));r4=obs(tracker,o,sensor,temporal,level1,4,f4)
    checks['uniquely_predicted_reappearance_rebinds_original_identity_after_occlusion']=(
        len(r4)==2 and [row[0] for row in r4]==[top,bottom]
        and all(row[4] for row in r4) and r4[0][3]==ta and r4[1][3]==tb)
    checks['occlusion_did_not_mutate_hidden_entity_until_reappearance']=(top_features_before==(ta,) and o._active_entity_features(top)==(ta,))

    # Two-frame disappearance exceeds the one-frame budget. Reappearance must mint fresh identity.
    expired=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));es=VisualSensorIngressV1();et=MultiVisualObjectFileTrackerV1();temp2=TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint()))
    q1=obs(et,expired,es,temp2,level1,1,canvas(((A1,0,1),)));old=q1[0][0]
    q2=obs(et,expired,es,temp2,level1,2,canvas(((A2,0,3),)))
    obs(et,expired,es,temp2,level1,3,canvas(()));obs(et,expired,es,temp2,level1,4,canvas(()))
    q5=obs(et,expired,es,temp2,level1,5,canvas(((A1,0,9),)))
    checks['occlusion_beyond_budget_does_not_reidentify_stale_entity']=(len(q5)==1 and q5[0][0]!=old and not q5[0][4])

    # Ambiguity arm: two dormant same-category tracks are symmetrically placed around
    # one reappearing candidate. Equal predicted distance must refuse; no entity changes.
    amb_o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));amb_s=VisualSensorIngressV1();amb_t=MultiVisualObjectFileTrackerV1();amb_temp=TemporalVisualContinuityV1.restore(copy.deepcopy(temporal.checkpoint()))
    a1=obs(amb_t,amb_o,amb_s,amb_temp,level1,1,canvas(((A1,0,1),(A1,8,1))))
    a2=obs(amb_t,amb_o,amb_s,amb_temp,level1,2,canvas(((A2,0,3),(A2,8,3))))
    ids=[row[0] for row in a2];obs(amb_t,amb_o,amb_s,amb_temp,level1,3,canvas(()));active_before=copy.deepcopy(amb_t.active);feat_before={e:amb_o._active_entity_features(e) for e in ids}
    # Candidate centered vertically between both predicted tracks, same predicted x.
    ambiguous=obs(amb_t,amb_o,amb_s,amb_temp,level1,4,canvas(((A1,4,7),)))
    checks['equidistant_dormant_tracks_refuse_ambiguous_reidentification']=(
        ambiguous==() and amb_t.active==active_before and {e:amb_o._active_entity_features(e) for e in ids}==feat_before)

    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-visual-occlusion-reidentification.v1','contract':'FOUNDRY_VISUAL_OCCLUSION_REIDENTIFICATION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'ONE_FRAME_VISUAL_OCCLUSION_NOW_PRESERVES_ORGANISM_OBJECT_IDENTITY_WHEN_MOTION_AND_FEATURE_CONTINUITY_UNIQUELY_REIDENTIFY_IT','checks':checks,'failed':failed,'entities':{'top':top,'bottom':bottom},'remaining_red':['CROSSING_TRAJECTORY_IDENTITY_WITH_SAME_CATEGORY','LONGER_OCCLUSION_FROM_LEARNED_MOTION_UNCERTAINTY','DIRECT_VISUAL_OCCLUSION_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
