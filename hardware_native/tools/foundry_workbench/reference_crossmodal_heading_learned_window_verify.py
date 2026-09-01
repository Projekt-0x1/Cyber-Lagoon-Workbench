#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_heading_calibration_learned_window_v1 import HeadingCalibrationLearnedWindowV1,CENTER_QUORUM,FLANK_QUORUM,LAG_HISTORY
from reference_heading_center_of_flow_verify import radial
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_tilt_translation_disambiguator_v1 import TiltTranslationDisambiguatorV1
from reference_vestibular_sensor_ingress_v1 import VestibularSensorIngressV1
from reference_vestibular_translation_heading_v1 import VestibularTranslationHeadingV1
from reference_visual_heading_center_v1 import VisualHeadingCenterV1
from reference_visual_vestibular_heading_fusion_v1 import VisualVestibularHeadingFusionV1
SRC=0xDB10;OS=0xDB11

def organism():return ReferenceOrganismV2(PopulationSpecV1(8192,2,3,16,8))
def advance(o,n,seed=0):
 for i in range(int(n)):o.contact(CONTACT_WORLD_STATE,(20000+seed*32+i,),SRC,True,True)
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
 # Center-only development earns offset but no width.
 o=organism();w=HeadingCalibrationLearnedWindowV1()
 for i in range(4):lag_pair(w,o,vl,al,2,10+i)
 for i in range(2):lag_pair(w,o,vr,ar,2,20+i)
 c['repeated_center_lag_learns_center_with_zero_width']=w.learned_center()==2 and w.learned_width()==0 and w.map_visual(vc)==ac
 before=w.pair_count
 for i in range(2):lag_pair(w,o,vc,ac,1,30+i)
 c['one_sided_flank_history_does_not_widen']=w.learned_center()==2 and w.learned_width()==0 and w.pair_count==before
 for i in range(2):lag_pair(w,o,vc,ac,3,40+i)
 c['symmetric_repeated_flanks_widen_window_to_one']=w.learned_center()==2 and w.learned_width()==1
 # Once width=1, three neighboring lags are accepted but outsiders are not.
 p=w.pair_count;a1=lag_pair(w,o,vc,ac,1,50);a2=lag_pair(w,o,vc,ac,2,51);a3=lag_pair(w,o,vc,ac,3,52);a0=lag_pair(w,o,vc,ac,0,53);a4=lag_pair(w,o,vc,ac,4,54)
 c['learned_width_accepts_center_and_symmetric_neighbors_only']=(a1 and a2 and a3 and not a0 and not a4 and w.pair_count==p+3 and w.learned_width()==1)
 # A single symmetric outlier on each side is insufficient to broaden a fresh center.
 so=organism();single=HeadingCalibrationLearnedWindowV1()
 for i in range(4):lag_pair(single,so,vl,al,2,60+i)
 lag_pair(single,so,vl,al,1,65);lag_pair(single,so,vl,al,3,66)
 c['single_flank_outliers_do_not_widen']=single.learned_center()==2 and single.learned_width()==0
 # Width does not invent coordinate scale when only one spatial anchor was ever demonstrated.
 ns_o=organism();noscale=HeadingCalibrationLearnedWindowV1()
 for i in range(4):lag_pair(noscale,ns_o,vl,al,2,70+i)
 for i in range(2):lag_pair(noscale,ns_o,vl,al,1,80+i);lag_pair(noscale,ns_o,vl,al,3,90+i)
 c['learned_temporal_width_cannot_invent_coordinate_scale']=noscale.learned_width()==1 and noscale.map_visual(vc) is None
 # Competing equally supported centers remain unresolved.
 bm_o=organism();bimodal=HeadingCalibrationLearnedWindowV1()
 for i in range(3):lag_pair(bimodal,bm_o,vl,al,1,100+i);lag_pair(bimodal,bm_o,vr,ar,3,110+i)
 c['equally_supported_competing_centers_refuse_window_learning']=bimodal.learned_center() is None and bimodal.learned_width() is None
 # Checkpoint preserves width/calibration but drops pending current cue.
 w.observe_visual(o,vc);cp=copy.deepcopy(w.checkpoint());restored=HeadingCalibrationLearnedWindowV1.restore(cp)
 c['checkpoint_preserves_width_and_mapping_but_drops_pending_cue']=restored.learned_center()==2 and restored.learned_width()==1 and restored.map_visual(vc)==ac and restored._visual is None and restored._vestibular is None
 # Later stable center-only history narrows the window by bounded-history replacement.
 narrow_o=organism();narrow=HeadingCalibrationLearnedWindowV1.restore(cp)
 for i in range(LAG_HISTORY):lag_pair(narrow,narrow_o,vc,ac,2,200+i)
 c['bounded_center_only_history_narrows_window_back_to_zero']=narrow.learned_center()==2 and narrow.learned_width()==0 and set(narrow.lag_history)=={2}
 f=VisualVestibularHeadingFusionV1()
 for x in (-1,1,-1,1):f.observe_visual(x);f.observe_vestibular(x)
 c['temporally_windowed_calibration_feeds_existing_fast_fuser']=f.fuse(restored.map_visual(vc),ac)==ac
 sigv=list(inspect.signature(HeadingCalibrationLearnedWindowV1.observe_visual).parameters);sigb=list(inspect.signature(HeadingCalibrationLearnedWindowV1.observe_vestibular).parameters)
 c['public_api_has_no_center_width_lag_tick_tolerance_or_pair_argument']=(sigv==['self','organism','value'] and sigb==['self','organism','value'] and not hasattr(HeadingCalibrationLearnedWindowV1,'observe_pair'))
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_CROSSMODAL_HEADING_LEARNED_WINDOW_GREEN','center':restored.learned_center(),'width':restored.learned_width(),'narrowed_width':narrow.learned_width(),'center_quorum':CENTER_QUORUM,'flank_quorum':FLANK_QUORUM,'history_cap':LAG_HISTORY,'checks':c,'failed':fail,'remaining_red':['ASYMMETRIC_MULTISENSORY_TEMPORAL_KERNEL','PHYSICAL_CAMERA_IMU_CLOCK_SYNCHRONIZATION','DELAYED_CONDUCTION_COMPENSATION','NONLINEAR_THREE_DIMENSIONAL_HEADING_CALIBRATION','WORLD_CENTERED_NAVIGATION','DIRECT_MULTISENSORY_TIMING_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_CROSSMODAL_HEADING_LEARNED_WINDOW_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
