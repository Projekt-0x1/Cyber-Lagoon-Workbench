#!/usr/bin/env python3
from __future__ import annotations
import json,time
from reference_heading_calibration_learned_lag_v1 import HeadingCalibrationLearnedLagV1
from reference_heading_center_of_flow_verify import radial
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_tilt_translation_disambiguator_v1 import TiltTranslationDisambiguatorV1
from reference_vestibular_sensor_ingress_v1 import VestibularSensorIngressV1
from reference_vestibular_translation_heading_v1 import VestibularTranslationHeadingV1
from reference_visual_heading_center_v1 import VisualHeadingCenterV1
SRC=0xEE10;OS=0xEE11

def organism():return ReferenceOrganismV2(PopulationSpecV1(8192,2,3,16,8))
def advance(o,n,seed):
 for i in range(n):o.contact(CONTACT_WORLD_STATE,(40000+seed*16+i,),SRC,True,True)
def visual(x):return VisualHeadingCenterV1().estimate(*radial(x,8))[0]
def vestibular(sample):
 s=VestibularSensorIngressV1();s.ingest(OS,1,sample,VestibularSensorIngressV1.sample_digest(sample));return VestibularTranslationHeadingV1.from_resolution(TiltTranslationDisambiguatorV1().resolve(s))
def event(owner,o,v,b,vd,bd,seed):
 # Both pathways start from the same conceptual event; only ready-time order differs.
 if vd<=bd:
  advance(o,vd,seed);owner.observe_visual(o,v);advance(o,bd-vd,seed+1);owner.observe_vestibular(o,b)
 else:
  advance(o,bd,seed);owner.observe_vestibular(o,b);advance(o,vd-bd,seed+1);owner.observe_visual(o,v)
 advance(o,1,seed+2)
def train(vd,bd):
 o=organism();h=HeadingCalibrationLearnedLagV1();vl,vc,vr=visual(4),visual(8),visual(12);al,ac,ar=vestibular((-500,1866)),vestibular((0,2000)),vestibular((500,1866))
 for i in range(4):event(h,o,vl,al,vd,bd,10+i*3)
 for i in range(2):event(h,o,vr,ar,vd,bd,40+i*3)
 return h,(vl,vc,vr),(al,ac,ar)
def main():
 t=time.perf_counter();c={};slow_visual,vis,vest=train(3,1);slow_vest,_,_=train(1,3);equal,_,_=train(2,2)
 c['visual_slower_than_vestibular_is_absorbed_as_negative_learned_lag']=slow_visual.learned_lag()==-2
 c['vestibular_slower_than_visual_is_absorbed_as_positive_learned_lag']=slow_vest.learned_lag()==2
 c['equal_effective_pathway_delays_learn_zero_lag']=equal.learned_lag()==0
 c['spatial_coordinate_calibration_is_invariant_across_latency_regimes']=(slow_visual.map_visual(vis[1])==slow_vest.map_visual(vis[1])==equal.map_visual(vis[1])==vest[1])
 # Unstable alternating effective latency must not invent a single conduction correction.
 o=organism();mixed=HeadingCalibrationLearnedLagV1();vl,vc,vr=vis;al,ac,ar=vest
 for i in range(3):event(mixed,o,vl,al,3,1,100+i*6);event(mixed,o,vr,ar,1,3,103+i*6)
 c['mixed_opposite_processing_delays_refuse_unique_timing_correction']=mixed.learned_lag() is None
 # Bounded new history reverses effective-latency compensation without a separate latency setter.
 o2=organism();rev=HeadingCalibrationLearnedLagV1()
 for i in range(6):event(rev,o2,vl if i%2==0 else vr,al if i%2==0 else ar,3,1,200+i*3)
 old=rev.learned_lag()
 for i in range(12):event(rev,o2,vl if i%2==0 else vr,al if i%2==0 else ar,1,3,300+i*3)
 c['later_opposite_latency_history_reverses_compensation_through_bounded_timing_state']=old==-2 and rev.learned_lag()==2
 c['no_separate_processing_latency_fields_are_added']=not any('latency' in k for k in slow_visual.__dict__)
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];r={'contract':'FOUNDRY_PREDICTABLE_PROCESSING_LATENCY_COMPENSATION_GREEN','visual_slow_lag':slow_visual.learned_lag(),'vestibular_slow_lag':slow_vest.learned_lag(),'equal_lag':equal.learned_lag(),'checks':c,'failed':fail,'remaining_red':['PATHWAY_READY_OCCURRENCE_OWNERSHIP','RAPIDLY_VARYING_INTERNAL_LATENCY','BIOLOGICAL_CONDUCTION_LATENCY_MEASUREMENT','PHYSICAL_HARDWARE_TIMESTAMP_PROVENANCE','DIRECT_PROCESSING_LATENCY_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_PREDICTABLE_PROCESSING_LATENCY_COMPENSATION_RED '+','.join(fail)) if fail else r['contract']);print(json.dumps(r,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
