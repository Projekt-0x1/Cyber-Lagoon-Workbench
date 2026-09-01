#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_authenticated_sensor_timestamp_v1 import AuthenticatedSensorTimestampV1,SensorTimestampReceiptV1
from reference_multisensory_delay_distribution_v1 import HISTORY,MultisensoryDelayDistributionV1
from reference_sensor_clock_alignment_v1 import SensorClockAlignmentV1
CAM=0xEB10;IMU=0xEB11;PD='11'*32

def receipt(source,seq,timestamp):
 c=AuthenticatedSensorTimestampV1.commitment(source,seq,timestamp,PD)
 return SensorTimestampReceiptV1(source,seq,timestamp,PD,c)
def align(rows):
 a=SensorClockAlignmentV1();a.pairs=[tuple(map(int,r)) for r in rows];return a
def obs(d,a,delay,seq,cam_t):
 cr=receipt(CAM,seq,cam_t);ir=receipt(IMU,seq,a.map_camera(cam_t)+int(delay));return d.observe(cr,ir,a)
def main():
 t=time.perf_counter();c={};a=align(((1000,2500),(2000,4500),(3000,6500)))
 one=MultisensoryDelayDistributionV1();obs(one,a,100,1,4000);c['one_delay_sample_is_insufficient']=one.profile() is None
 stable=MultisensoryDelayDistributionV1()
 for i,d in enumerate((80,90,100,110,120),1):obs(stable,a,d,i,4000+i*100)
 c['stable_low_jitter_history_learns_narrow_empirical_envelope']=stable.profile()==(100,20,5)
 noisy=MultisensoryDelayDistributionV1()
 for i,d in enumerate((50,75,100,125,150),1):obs(noisy,a,d,i,5000+i*100)
 c['broader_jitter_history_learns_wider_envelope']=noisy.profile()==(100,50,5) and noisy.profile()[1]>stable.profile()[1]
 robust=MultisensoryDelayDistributionV1()
 for i,d in enumerate((80,90,100,110,120,10000),1):obs(robust,a,d,i,6000+i*100)
 c['single_extreme_outlier_does_not_dominate_center_or_width']=robust.profile()==(100,20,5)
 bimodal=MultisensoryDelayDistributionV1()
 for i,d in enumerate((0,0,0,1000,1000,1000),1):obs(bimodal,a,d,i,7000+i*100)
 c['equal_bimodal_delay_history_refuses_instead_of_choosing_one_mode']=bimodal.profile() is None
 inside_cam=8000;inside=(receipt(CAM,90,inside_cam),receipt(IMU,90,a.map_camera(inside_cam)+119));outside=(receipt(CAM,91,inside_cam+10),receipt(IMU,91,a.map_camera(inside_cam+10)+121))
 c['learned_envelope_accepts_inside_delay_and_refuses_nearby_outside']=stable.accepts(*inside,a) and not stable.accepts(*outside,a)
 cp=copy.deepcopy(stable.checkpoint());rest=MultisensoryDelayDistributionV1.restore(cp);c['checkpoint_preserves_empirical_distribution_without_sensor_payload']=rest.profile()==stable.profile() and 'payload' not in ''.join(cp.keys()).lower()
 forged=SensorTimestampReceiptV1(CAM,99,9000,PD,'0'*64)
 try:stable.accepts(forged,receipt(IMU,99,a.map_camera(9000)+100),a);forged_refused=False
 except ValueError:forged_refused=True
 c['forged_timestamp_receipt_refuses_inside_delay_estimation']=forged_refused
 # Same true delays under a different learned affine clock relation produce same delay profile.
 a2=align(((1000,3100),(2000,6100),(3000,9100)));other=MultisensoryDelayDistributionV1()
 for i,d in enumerate((80,90,100,110,120),1):obs(other,a2,d,i,10000+i*100)
 c['different_clock_offset_and_drift_preserve_aligned_delay_statistics']=other.profile()==stable.profile()
 replacement=MultisensoryDelayDistributionV1.restore(cp)
 for i in range(HISTORY):obs(replacement,a,300,100+i,12000+i*100)
 c['bounded_later_history_relocates_and_narrows_distribution']=replacement.profile()==(300,0,HISTORY) and set(replacement.history)=={300}
 sig=list(inspect.signature(MultisensoryDelayDistributionV1.observe).parameters)
 c['public_distribution_api_has_no_delay_mean_variance_tolerance_or_family_argument']=sig==['self','camera_receipt','imu_receipt','clock_alignment']
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_SUBTICK_MULTISENSORY_DELAY_DISTRIBUTION_GREEN','stable_profile':stable.profile(),'noisy_profile':noisy.profile(),'outlier_profile':robust.profile(),'replacement_profile':replacement.profile(),'checks':c,'failed':fail,'remaining_red':['SMOOTH_CONTINUOUS_DELAY_DENSITY','PHYSICAL_HARDWARE_TIMESTAMP_PROVENANCE','NONLINEAR_CLOCK_DRIFT','DELAYED_CONDUCTION_COMPENSATION','WORLD_CENTERED_NAVIGATION','DIRECT_SUBTICK_TIMING_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_SUBTICK_MULTISENSORY_DELAY_DISTRIBUTION_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
