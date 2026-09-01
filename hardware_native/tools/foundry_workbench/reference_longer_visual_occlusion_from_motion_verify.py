#!/usr/bin/env python3
"""N+1: repeated stable motion earns a longer bounded visual occlusion horizon."""
from __future__ import annotations
import copy,json,time
from reference_continuous_visual_sensor_ownership_verify import A1,A2,train_level1,vf
from reference_multi_visual_object_file_tracker_v1 import MultiVisualObjectFileTrackerV1
from reference_multiple_visual_object_files_verify import train_temporal
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_temporal_visual_continuity_v1 import TemporalVisualContinuityV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1

SOURCE=0xC001

def canvas(image=None,x0=0,h=6,w=30):
    out=[[0]*w for _ in range(h)]
    if image is not None:
        for y,row in enumerate(image):
            for x,value in enumerate(row):out[y][x0+x]=int(value)
    return tuple(tuple(row) for row in out)

def obs(t,o,s,temp,level1,seq,frame):
    return t.observe(o,s,temp,level1,SOURCE,seq,frame,VisualSensorIngressV1.frame_digest(frame))

def seed_stable(level1,temp):
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));s=VisualSensorIngressV1();t=MultiVisualObjectFileTrackerV1()
    rows=[]
    for seq,(img,x) in enumerate(((A1,1),(A2,3),(A1,5),(A2,7)),start=1):rows.append(obs(t,o,s,temp,level1,seq,canvas(img,x)))
    return o,s,t,rows

def seed_erratic(level1,temp):
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));s=VisualSensorIngressV1();t=MultiVisualObjectFileTrackerV1()
    rows=[]
    # Velocity changes +2, +3, +1 pixels; feature continuity remains the same.
    for seq,(img,x) in enumerate(((A1,1),(A2,3),(A1,6),(A2,7)),start=1):rows.append(obs(t,o,s,temp,level1,seq,canvas(img,x)))
    return o,s,t,rows

def main():
    started=time.perf_counter();checks={};level1=train_level1();base_temp=train_temporal(level1);ta=base_temp.relation(vf(level1,A1),vf(level1,A2));blank=canvas()
    o,s,t,rows=seed_stable(level1,TemporalVisualContinuityV1.restore(copy.deepcopy(base_temp.checkpoint())));entity=rows[0][0][0]
    stability=t.motion_stability.get(entity,0)
    checks['repeated_constant_velocity_builds_motion_stability_without_new_entity_state']=(stability>=2 and len(t.active)==1)
    obs(t,o,s,base_temp,level1,5,blank);after_one=entity in t.active
    obs(t,o,s,base_temp,level1,6,blank);after_two=entity in t.active
    # Last observed x=7, velocity +2 px/frame, three intervals later => x=13.
    reappear=obs(t,o,s,base_temp,level1,7,canvas(A1,13))
    checks['stable_motion_survives_two_blank_frames_and_reidentifies_three_steps_ahead']=(after_one and after_two and len(reappear)==1 and reappear[0][0]==entity and reappear[0][4] and reappear[0][3]==ta)

    eo,es,et,erows=seed_erratic(level1,TemporalVisualContinuityV1.restore(copy.deepcopy(base_temp.checkpoint())));old=erows[0][0][0];err_stability=et.motion_stability.get(old,0)
    obs(et,eo,es,base_temp,level1,5,blank);obs(et,eo,es,base_temp,level1,6,blank)
    err_reappear=obs(et,eo,es,base_temp,level1,7,canvas(A1,10))
    checks['erratic_motion_does_not_earn_same_two_frame_occlusion_horizon']=(err_stability<2 and old not in {row[0] for row in err_reappear} and len(err_reappear)==1 and not err_reappear[0][4])

    # Even stable motion has a hard bounded ceiling: three blank frames exceed the
    # earned two-extra budget in this reference implementation.
    bo,bs,bt,brows=seed_stable(level1,TemporalVisualContinuityV1.restore(copy.deepcopy(base_temp.checkpoint())));bold=brows[0][0][0]
    obs(bt,bo,bs,base_temp,level1,5,blank);obs(bt,bo,bs,base_temp,level1,6,blank);obs(bt,bo,bs,base_temp,level1,7,blank)
    b8=obs(bt,bo,bs,base_temp,level1,8,canvas(A1,15))
    checks['stable_motion_horizon_remains_bounded_and_does_not_become_unlimited_tracking']=(bold not in {row[0] for row in b8} and len(b8)==1 and not b8[0][4])

    cp=t.checkpoint();restored=MultiVisualObjectFileTrackerV1.restore(cp)
    checks['motion_stability_and_active_prediction_are_transient_not_checkpointed']=(cp=={'schema':1} and not restored.active and not restored.motion_stability)
    checks['bounded_fast_path']=time.perf_counter()-started<1.0
    failed=[k for k,v in checks.items() if not v]
    result={'schema':'cyber-lagoon.reference-longer-visual-occlusion-from-motion.v1','contract':'FOUNDRY_LONGER_VISUAL_OCCLUSION_FROM_MOTION_'+('GREEN' if not failed else 'RED'),'pass':not failed,'reference_only':True,'language_phenotype_improved':not failed,'future_update_authority_preserved':True,'visible_language_gain':'REPEATED_STABLE_OBJECT_MOTION_NOW_EXTENDS_REIDENTIFICATION_ACROSS_TWO_MISSING_FRAMES_WHILE_ERRATIC_MOTION_EXPIRES','checks':checks,'failed':failed,'stability':{'stable':stability,'erratic':err_stability},'remaining_red':['LEARNED_OCCLUSION_DURATION_DISTRIBUTION','OBJECT_INTERACTION_AND_CAUSAL_ROLE_LEARNING','DIRECT_LONG_OCCLUSION_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True));return 0 if not failed else 1
if __name__=='__main__':raise SystemExit(main())
