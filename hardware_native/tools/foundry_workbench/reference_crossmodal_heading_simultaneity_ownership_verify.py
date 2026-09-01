#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_heading_calibration_synchrony_v1 import HeadingCalibrationSynchronyV1
from reference_heading_center_of_flow_verify import radial
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_tilt_translation_disambiguator_v1 import TiltTranslationDisambiguatorV1
from reference_vestibular_sensor_ingress_v1 import VestibularSensorIngressV1
from reference_vestibular_translation_heading_v1 import VestibularTranslationHeadingV1
from reference_visual_heading_center_v1 import VisualHeadingCenterV1
from reference_visual_vestibular_heading_fusion_v1 import VisualVestibularHeadingFusionV1
SRC=0xCD10;OS=0xCD11
def organism():return ReferenceOrganismV2(PopulationSpecV1(8192,2,3,16,8))
def advance(o,n):o.contact(CONTACT_WORLD_STATE,(1000+int(n),),SRC,True,True)
def visual(x):return VisualHeadingCenterV1().estimate(*radial(x,8))[0]
def vestibular(sample):
 s=VestibularSensorIngressV1();s.ingest(OS,1,sample,VestibularSensorIngressV1.sample_digest(sample));return VestibularTranslationHeadingV1.from_resolution(TiltTranslationDisambiguatorV1().resolve(s))
def refuses(fn):
 try:fn();return False
 except ValueError:return True
def sync_pair(owner,o,v,b,step):
 owner.observe_visual(o,v);paired=owner.observe_vestibular(o,b);advance(o,step);return paired
def main():
 t=time.perf_counter();c={};vl,vc,vr=visual(4),visual(8),visual(12);al,ac,ar=vestibular((-500,1866)),vestibular((0,2000)),vestibular((500,1866))
 o=organism();s=HeadingCalibrationSynchronyV1();c['same_tick_visual_then_vestibular_earns_one_pair']=s.observe_visual(o,vl) is False and s.observe_vestibular(o,al) is True and s.pair_count==1
 o2=organism();r=HeadingCalibrationSynchronyV1();c['same_tick_vestibular_then_visual_earns_same_pair']=r.observe_vestibular(o2,al) is False and r.observe_visual(o2,vl) is True and r.calibration.pairs==s.calibration.pairs
 async_o=organism();a=HeadingCalibrationSynchronyV1()
 a.observe_visual(async_o,vl);advance(async_o,1);a.observe_vestibular(async_o,al);advance(async_o,101)
 c['intervening_real_organism_contact_prevents_pairing']=a.pair_count==0 and a.relation() is None
 for i in range(2,8):a.observe_visual(async_o,vl if i%2 else vr);advance(async_o,i);a.observe_vestibular(async_o,al if i%2 else ar);advance(async_o,100+i)
 c['repeated_asynchronous_histories_cannot_acquire_calibration']=a.pair_count==0 and a.map_visual(vc) is None
 train_o=organism();train=HeadingCalibrationSynchronyV1()
 sync_pair(train,train_o,vl,al,1);sync_pair(train,train_o,vl,al,2);sync_pair(train,train_o,vr,ar,3);sync_pair(train,train_o,vr,ar,4)
 c['repeated_synchronous_two_anchor_history_acquires_heldout_mapping']=train.pair_count==4 and train.map_visual(vc)==ac
 dup_o=organism();dup=HeadingCalibrationSynchronyV1();dup.observe_visual(dup_o,vl)
 c['duplicate_same_modality_observation_in_one_tick_refuses']=refuses(lambda:dup.observe_visual(dup_o,vr))
 train.observe_visual(train_o,vc);cp=copy.deepcopy(train.checkpoint());restored=HeadingCalibrationSynchronyV1.restore(cp)
 c['checkpoint_preserves_learned_calibration_but_drops_unpaired_contact']=restored.map_visual(vc)==ac and restored._visual is None and restored._vestibular is None
 c['public_synchrony_api_has_no_pair_or_tick_argument']=(list(inspect.signature(HeadingCalibrationSynchronyV1.observe_visual).parameters)==['self','organism','value'] and list(inspect.signature(HeadingCalibrationSynchronyV1.observe_vestibular).parameters)==['self','organism','value'] and not hasattr(HeadingCalibrationSynchronyV1,'observe_pair'))
 f=VisualVestibularHeadingFusionV1()
 for x in (-1,1,-1,1):f.observe_visual(x);f.observe_vestibular(x)
 c['owned_synchronous_calibration_feeds_existing_fast_fuser']=f.fuse(restored.map_visual(vc),ac)==ac
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_CROSSMODAL_HEADING_SIMULTANEITY_OWNERSHIP_GREEN','synchronous_pairs':train.pair_count,'asynchronous_pairs':a.pair_count,'heldout_mapping':train.map_visual(vc),'checks':c,'failed':fail,'remaining_red':['LEARNED_WIDTH_MULTISENSORY_BINDING_WINDOW','PHYSICAL_CAMERA_IMU_CLOCK_SYNCHRONIZATION','DELAYED_CONDUCTION_COMPENSATION','NONLINEAR_THREE_DIMENSIONAL_HEADING_CALIBRATION','WORLD_CENTERED_NAVIGATION','DIRECT_MULTISENSORY_TIMING_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_CROSSMODAL_HEADING_SIMULTANEITY_OWNERSHIP_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
