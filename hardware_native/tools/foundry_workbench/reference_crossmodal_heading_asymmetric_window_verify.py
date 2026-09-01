#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_heading_calibration_asymmetric_window_v1 import HeadingCalibrationAsymmetricWindowV1,LAG_HISTORY
from reference_heading_center_of_flow_verify import radial
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_tilt_translation_disambiguator_v1 import TiltTranslationDisambiguatorV1
from reference_vestibular_sensor_ingress_v1 import VestibularSensorIngressV1
from reference_vestibular_translation_heading_v1 import VestibularTranslationHeadingV1
from reference_visual_heading_center_v1 import VisualHeadingCenterV1
from reference_visual_vestibular_heading_fusion_v1 import VisualVestibularHeadingFusionV1
SRC=0xDC10;OS=0xDC11

def organism():return ReferenceOrganismV2(PopulationSpecV1(8192,2,3,16,8))
def advance(o,n,seed=0):
 for i in range(int(n)):o.contact(CONTACT_WORLD_STATE,(30000+seed*32+i,),SRC,True,True)
def visual(x):return VisualHeadingCenterV1().estimate(*radial(x,8))[0]
def vestibular(sample):
 s=VestibularSensorIngressV1();s.ingest(OS,1,sample,VestibularSensorIngressV1.sample_digest(sample));return VestibularTranslationHeadingV1.from_resolution(TiltTranslationDisambiguatorV1().resolve(s))
def lag_pair(owner,o,v,b,lag,seed):
 if lag>=0:
  owner.observe_visual(o,v);advance(o,lag,seed);accepted=owner.observe_vestibular(o,b)
 else:
  owner.observe_vestibular(o,b);advance(o,-lag,seed);accepted=owner.observe_visual(o,v)
 advance(o,1,seed+1000);return accepted

def main():
 t=time.perf_counter();c={};vl,vc,vr=visual(4),visual(8),visual(12);al,ac,ar=vestibular((-500,1866)),vestibular((0,2000)),vestibular((500,1866))
 o=organism();a=HeadingCalibrationAsymmetricWindowV1()
 for i in range(4):lag_pair(a,o,vl,al,2,10+i)
 for i in range(2):lag_pair(a,o,vr,ar,2,20+i)
 c['center_history_starts_with_zero_width_on_both_sides']=a.learned_center()==2 and a.widths()==(0,0) and a.map_visual(vc)==ac
 for i in range(2):lag_pair(a,o,vc,ac,1,30+i)
 c['repeated_left_flank_widens_only_left_side']=a.widths()==(1,0)
 p=a.pair_count;left=lag_pair(a,o,vc,ac,1,40);right=lag_pair(a,o,vc,ac,3,41)
 c['symmetric_temporal_distances_can_bind_differently_from_lived_side_history']=left and not right and a.pair_count==p+1
 for i in range(2):lag_pair(a,o,vc,ac,3,50+i)
 c['later_right_flank_history_widens_right_without_erasing_left']=a.widths()==(1,1)
 for i in range(2):lag_pair(a,o,vc,ac,0,60+i)
 c['supported_intervening_left_bins_extend_left_width_to_two_only']=a.widths()==(2,1)
 # Missing +1 intervening support prevents direct jump from center 2 to lag 0.
 gap_o=organism();gap=HeadingCalibrationAsymmetricWindowV1()
 for i in range(4):lag_pair(gap,gap_o,vl,al,2,70+i)
 for i in range(2):lag_pair(gap,gap_o,vl,al,0,80+i)
 c['missing_intervening_lag_prevents_width_jump']=gap.widths()==(0,0)
 one_o=organism();one=HeadingCalibrationAsymmetricWindowV1()
 for i in range(4):lag_pair(one,one_o,vl,al,2,90+i)
 lag_pair(one,one_o,vl,al,1,95)
 c['single_side_outlier_is_insufficient_to_widen']=one.widths()==(0,0)
 # Temporal asymmetry still cannot invent spatial coordinate scale.
 ns_o=organism();ns=HeadingCalibrationAsymmetricWindowV1()
 for i in range(4):lag_pair(ns,ns_o,vl,al,2,100+i)
 for i in range(2):lag_pair(ns,ns_o,vl,al,1,110+i)
 c['asymmetric_timing_history_cannot_invent_coordinate_scale']=ns.widths()==(1,0) and ns.map_visual(vc) is None
 # Checkpoint preserves asymmetric widths/calibration, pending cue remains transient.
 a.observe_visual(o,vc);cp=copy.deepcopy(a.checkpoint());restored=HeadingCalibrationAsymmetricWindowV1.restore(cp)
 c['checkpoint_preserves_asymmetric_widths_and_mapping_but_drops_pending']=restored.widths()==(2,1) and restored.map_visual(vc)==ac and restored._visual is None and restored._vestibular is None
 # Replace bounded history with center/right-only evidence: left narrows independently, right survives.
 nr_o=organism();narrow=HeadingCalibrationAsymmetricWindowV1.restore(cp)
 for i in range(LAG_HISTORY):lag_pair(narrow,nr_o,vc,ac,3 if i%3==0 else 2,200+i)
 c['bounded_history_can_narrow_left_independently_while_right_survives']=narrow.learned_center()==2 and narrow.widths()==(0,1) and set(narrow.lag_history)=={2,3}
 f=VisualVestibularHeadingFusionV1()
 for x in (-1,1,-1,1):f.observe_visual(x);f.observe_vestibular(x)
 c['asymmetrically_windowed_mapping_feeds_existing_fast_fuser']=f.fuse(restored.map_visual(vc),ac)==ac
 sigv=list(inspect.signature(HeadingCalibrationAsymmetricWindowV1.observe_visual).parameters);sigb=list(inspect.signature(HeadingCalibrationAsymmetricWindowV1.observe_vestibular).parameters)
 c['public_api_has_no_side_width_kernel_lag_tick_or_pair_argument']=(sigv==['self','organism','value'] and sigb==['self','organism','value'] and not hasattr(HeadingCalibrationAsymmetricWindowV1,'observe_pair'))
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_CROSSMODAL_HEADING_ASYMMETRIC_WINDOW_GREEN','center':restored.learned_center(),'widths':restored.widths(),'narrowed_widths':narrow.widths(),'history_cap':LAG_HISTORY,'checks':c,'failed':fail,'remaining_red':['CONTINUOUS_MULTISENSORY_DELAY_DISTRIBUTION','PHYSICAL_CAMERA_IMU_CLOCK_SYNCHRONIZATION','DELAYED_CONDUCTION_COMPENSATION','NONLINEAR_THREE_DIMENSIONAL_HEADING_CALIBRATION','WORLD_CENTERED_NAVIGATION','DIRECT_MULTISENSORY_TIMING_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_CROSSMODAL_HEADING_ASYMMETRIC_WINDOW_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
