#!/usr/bin/env python3
from __future__ import annotations
import copy,json,time
from reference_authenticated_sensor_timestamp_v1 import AuthenticatedSensorTimestampV1,SensorTimestampReceiptV1
from reference_multisensory_delay_distribution_v1 import MultisensoryDelayDistributionV1
from reference_sensor_clock_alignment_v1 import SensorClockAlignmentV1
PD='22'*32;CAM=0xEC10;IMU=0xEC11

def receipt(src,seq,t):
 c=AuthenticatedSensorTimestampV1.commitment(src,seq,t,PD)
 return SensorTimestampReceiptV1(src,seq,t,PD,c)
def add(a,x,y):a.pairs.append((int(x),int(y)));a.pairs=a.pairs[-a.history:]
def main():
 t=time.perf_counter();c={};old=((1000,2500),(2000,4500),(3000,6500));new=((4000,12100),(5000,15100),(6000,18100))
 local=SensorClockAlignmentV1(history=3)
 for row in old:add(local,*row)
 c['three_old_rows_learn_initial_affine_clock']=local.relation()==(1000,2500,4000,2000) and local.map_camera(1500)==3500
 add(local,*new[0]);c['mixed_old_new_local_window_refuses_after_first_regime_change']=local.relation() is None
 add(local,*new[1]);c['mixed_old_new_local_window_still_refuses_before_full_replacement']=local.relation() is None
 add(local,*new[2]);c['three_recent_new_rows_replace_old_regime_and_learn_new_clock']=local.relation()==(4000,12100,6000,2000) and local.map_camera(5500)==16600
 cp=copy.deepcopy(local.checkpoint());rest=SensorClockAlignmentV1.restore(cp);c['checkpoint_preserves_current_local_clock_regime']=rest.relation()==local.relation() and rest.map_camera(5500)==16600
 # Long history must refuse rather than fit through incompatible regimes.
 long=SensorClockAlignmentV1(history=8)
 for row in (*old,*new):add(long,*row)
 c['long_mixed_history_refuses_incompatible_clock_regimes']=long.relation() is None
 # Reversal after three new-old rows reacquires old law.
 for row in ((7000,14500),(8000,16500),(9000,18500)):add(local,*row)
 c['later_reversal_reacquires_old_clock_skew_locally']=local.relation()==(7000,14500,4000,2000) and local.map_camera(7500)==15500
 # Same physical 100us delay remains 100us after each valid map.
 d1=MultisensoryDelayDistributionV1();a1=SensorClockAlignmentV1(history=3)
 for row in old:add(a1,*row)
 d2=MultisensoryDelayDistributionV1();a2=SensorClockAlignmentV1(history=3)
 for row in new:add(a2,*row)
 for i,x in enumerate((1200,1400,1600,1800,2200),1):
  cr=receipt(CAM,i,x);ir=receipt(IMU,i,a1.map_camera(x)+100);d1.observe(cr,ir,a1)
 for i,x in enumerate((4200,4400,4600,4800,5200),10):
  cr=receipt(CAM,i,x);ir=receipt(IMU,i,a2.map_camera(x)+100);d2.observe(cr,ir,a2)
 c['same_physical_delay_is_invariant_across_clock_regime_change']=d1.profile()==d2.profile()==(100,0,5)
 forged=SensorTimestampReceiptV1(CAM,99,9999,PD,'0'*64)
 try:a2.observe_camera(type('O',(),{'tick_count':1})(),forged);ok=False
 except ValueError:ok=True
 c['forged_timestamp_receipt_still_refuses']=ok
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];r={'contract':'FOUNDRY_TIME_VARYING_SENSOR_CLOCK_DRIFT_GREEN','initial_relation':old,'new_relation':new,'current_relation':rest.relation(),'checks':c,'failed':fail,'remaining_red':['CONTINUOUS_NONLINEAR_INTRA_WINDOW_CLOCK_DRIFT','PHYSICAL_HARDWARE_TIMESTAMP_PROVENANCE','TEMPERATURE_COUPLED_OSCILLATOR_MODEL','DELAYED_CONDUCTION_COMPENSATION','SMOOTH_CONTINUOUS_DELAY_DENSITY','DIRECT_SENSOR_CLOCK_DRIFT_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_TIME_VARYING_SENSOR_CLOCK_DRIFT_RED '+','.join(fail)) if fail else r['contract']);print(json.dumps(r,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
