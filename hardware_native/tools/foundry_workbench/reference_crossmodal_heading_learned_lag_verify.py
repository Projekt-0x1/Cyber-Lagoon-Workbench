#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_heading_calibration_learned_lag_v1 import HeadingCalibrationLearnedLagV1,LAG_HISTORY,LAG_QUORUM,MAX_LAG
from reference_heading_center_of_flow_verify import radial
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_tilt_translation_disambiguator_v1 import TiltTranslationDisambiguatorV1
from reference_vestibular_sensor_ingress_v1 import VestibularSensorIngressV1
from reference_vestibular_translation_heading_v1 import VestibularTranslationHeadingV1
from reference_visual_heading_center_v1 import VisualHeadingCenterV1
from reference_visual_vestibular_heading_fusion_v1 import VisualVestibularHeadingFusionV1
SRC=0xDA10;OS=0xDA11

def organism():return ReferenceOrganismV2(PopulationSpecV1(8192,2,3,16,8))
def advance(o,n,seed=0):
 for i in range(int(n)):o.contact(CONTACT_WORLD_STATE,(10000+seed*32+i,),SRC,True,True)
def visual(x):return VisualHeadingCenterV1().estimate(*radial(x,8))[0]
def vestibular(sample):
 s=VestibularSensorIngressV1();s.ingest(OS,1,sample,VestibularSensorIngressV1.sample_digest(sample));return VestibularTranslationHeadingV1.from_resolution(TiltTranslationDisambiguatorV1().resolve(s))
def lag_pair(owner,o,v,b,lag,seed):
 if lag>=0:
  owner.observe_visual(o,v);advance(o,lag,seed);accepted=owner.observe_vestibular(o,b)
 else:
  owner.observe_vestibular(o,b);advance(o,-lag,seed);accepted=owner.observe_visual(o,v)
 advance(o,1,seed+1000)
 return accepted

def main():
 t=time.perf_counter();c={};vl,vc,vr=visual(4),visual(8),visual(12);al,ac,ar=vestibular((-500,1866)),vestibular((0,2000)),vestibular((500,1866))
 one_o=organism();one=HeadingCalibrationLearnedLagV1();lag_pair(one,one_o,vl,al,2,1)
 c['one_lag_observation_is_insufficient']=one.learned_lag() is None and one.pair_count==0
 stable_o=organism();stable=HeadingCalibrationLearnedLagV1()
 for i in range(3):lag_pair(stable,stable_o,vl,al,2,10+i)
 c['three_repeated_same_lag_observations_earn_unique_offset']=stable.learned_lag()==2 and stable.pair_count==1
 c['learned_offset_alone_does_not_fake_coordinate_scale']=stable.map_visual(vc) is None
 train_o=organism();train=HeadingCalibrationLearnedLagV1()
 for i in range(4):lag_pair(train,train_o,vl,al,2,30+i)
 for i in range(2):lag_pair(train,train_o,vr,ar,2,40+i)
 c['stable_lag_plus_two_repeated_anchors_earns_affine_calibration']=train.learned_lag()==2 and train.map_visual(vc)==ac and train.pair_count==4
 before=train.pair_count;lag_pair(train,train_o,vc,ac,1,50)
 c['same_current_cues_at_wrong_lag_do_not_enter_calibration']=train.pair_count==before and train.learned_lag()==2
 far_o=organism();far=HeadingCalibrationLearnedLagV1();lag_pair(far,far_o,vl,al,MAX_LAG+1,60)
 c['out_of_candidate_horizon_lag_is_not_learned']=far.lag_history==[] and far.pair_count==0
 bimodal_o=organism();bimodal=HeadingCalibrationLearnedLagV1()
 for i in range(3):
  lag_pair(bimodal,bimodal_o,vl,al,1,70+i*2);lag_pair(bimodal,bimodal_o,vr,ar,-1,71+i*2)
 c['equally_supported_competing_lag_modes_remain_ambiguous']=bimodal.learned_lag() is None and bimodal.lag_history.count(1)==3 and bimodal.lag_history.count(-1)==3
 # Checkpoint keeps learned lag/calibration, but pending current cue is transient.
 train.observe_visual(train_o,vc);cp=copy.deepcopy(train.checkpoint());restored=HeadingCalibrationLearnedLagV1.restore(cp)
 c['checkpoint_preserves_lag_and_mapping_but_drops_pending_cue']=restored.learned_lag()==2 and restored.map_visual(vc)==ac and restored._visual is None and restored._vestibular is None
 # Bounded developmental reversal: new opposite-order history replaces old lag and then old coordinate pairs.
 rev_o=organism();rev=HeadingCalibrationLearnedLagV1()
 for i in range(4):lag_pair(rev,rev_o,vl,al,2,100+i)
 for i in range(4):lag_pair(rev,rev_o,vr,ar,2,110+i)
 old=rev.map_visual(vl)
 for i in range(16):
  if i%2==0:lag_pair(rev,rev_o,vl,ar,-2,200+i)
  else:lag_pair(rev,rev_o,vr,al,-2,200+i)
 c['bounded_history_reverses_learned_lag_and_coordinate_relation']=old==al and rev.learned_lag()==-2 and rev.map_visual(vl)==ar and rev.map_visual(vr)==al and len(rev.lag_history)==LAG_HISTORY
 # Fast fusion uses the held-out mapping only after the temporal law has earned it.
 f=VisualVestibularHeadingFusionV1()
 for x in (-1,1,-1,1):f.observe_visual(x);f.observe_vestibular(x)
 c['learned_temporal_offset_calibration_feeds_existing_fast_fuser']=f.fuse(restored.map_visual(vc),ac)==ac
 sigv=list(inspect.signature(HeadingCalibrationLearnedLagV1.observe_visual).parameters);sigb=list(inspect.signature(HeadingCalibrationLearnedLagV1.observe_vestibular).parameters)
 c['public_api_has_no_lag_tick_window_or_pair_argument']=(sigv==['self','organism','value'] and sigb==['self','organism','value'] and not hasattr(HeadingCalibrationLearnedLagV1,'observe_pair'))
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_CROSSMODAL_HEADING_LEARNED_LAG_GREEN','learned_lag':restored.learned_lag(),'reversed_lag':rev.learned_lag(),'lag_quorum':LAG_QUORUM,'lag_history_cap':LAG_HISTORY,'heldout_mapping':restored.map_visual(vc),'checks':c,'failed':fail,'remaining_red':['LEARNED_WIDTH_MULTISENSORY_BINDING_WINDOW','PHYSICAL_CAMERA_IMU_CLOCK_SYNCHRONIZATION','DELAYED_CONDUCTION_COMPENSATION','NONLINEAR_THREE_DIMENSIONAL_HEADING_CALIBRATION','WORLD_CENTERED_NAVIGATION','DIRECT_MULTISENSORY_TIMING_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_CROSSMODAL_HEADING_LEARNED_LAG_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
