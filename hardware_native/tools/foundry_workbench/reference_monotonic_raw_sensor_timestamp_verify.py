#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_authenticated_sensor_timestamp_v1 import AuthenticatedSensorTimestampV1
from reference_monotonic_raw_sensor_timestamp_v1 import MonotonicRawSensorTimestampV1
from reference_sensor_clock_alignment_v1 import SensorClockAlignmentV1
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1
from reference_vestibular_sensor_ingress_v1 import VestibularSensorIngressV1
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
CAM=0xED10;IMU=0xED11;FRAME=((0,255),(0,255));SAMPLE=(3,4)
def visual(seq,frame=FRAME):
 s=VisualSensorIngressV1();s.ingest(CAM,seq,frame,VisualSensorIngressV1.frame_digest(frame));return s
def imu(seq,sample=SAMPLE):
 s=VestibularSensorIngressV1();s.ingest(IMU,seq,sample,VestibularSensorIngressV1.sample_digest(sample));return s
def refuses(fn):
 try:fn();return False
 except ValueError:return True
def main():
 started=time.perf_counter();c={};owner=MonotonicRawSensorTimestampV1();v1=visual(1);r1=owner.stamp(v1)
 c['public_stamp_api_has_no_timestamp_argument']=list(inspect.signature(MonotonicRawSensorTimestampV1.stamp).parameters)==['self','sensor']
 c['accepted_visual_payload_gets_positive_internal_monotonic_timestamp']=r1.timestamp_us>0 and r1.source==CAM and r1.sequence==1
 c['receipt_commitment_is_independently_verifiable']=r1.commitment==AuthenticatedSensorTimestampV1.commitment(r1.source,r1.sequence,r1.timestamp_us,r1.payload_digest)
 c['duplicate_stamping_same_source_sequence_refuses']=refuses(lambda:owner.stamp(v1))
 v2=visual(2);r2=owner.stamp(v2);c['new_sequence_gets_fresh_stamp_and_internal_clock_advances']=r2.sequence==2 and owner.last_timestamp_ns>0
 m1=imu(1);rm=owner.stamp(m1);c['different_sensor_source_can_be_stamped_by_same_owner']=rm.source==IMU and rm.sequence==1
 # Mutating an already accepted frame makes the recomputed payload digest differ from the ingress identity.
 broken=visual(3);broken.current_frame=((255,0),(255,0));c['payload_mutation_after_ingress_changes_timestamp_commitment_payload']=owner.stamp(broken).payload_digest!=VisualSensorIngressV1.frame_digest(FRAME)
 cp=copy.deepcopy(owner.checkpoint());rest=MonotonicRawSensorTimestampV1.restore(cp)
 c['checkpoint_preserves_chronology_without_payload']=rest.stamped==owner.stamped and rest.last_timestamp_ns==owner.last_timestamp_ns and 'payload' not in ''.join(cp.keys()).lower()
 c['restored_owner_cannot_restamp_prior_source_sequence']=refuses(lambda:rest.stamp(v2))
 # Downstream clock alignment can consume internally sampled receipts; use three fresh sequences on both sources.
 org=ReferenceOrganismV2(PopulationSpecV1(4096,2,3,8,8));align=SensorClockAlignmentV1();cam_owner=MonotonicRawSensorTimestampV1();imu_owner=MonotonicRawSensorTimestampV1()
 pairs=0
 for seq in range(10,13):
  cr=cam_owner.stamp(visual(seq));time.sleep(0.001);ir=imu_owner.stamp(imu(seq));pairs+=int(align.observe_camera(org,cr) or align.observe_imu(org,ir))
 c['internally_sampled_receipts_reach_clock_alignment_without_fixture_timestamp']=pairs==3 and len(align.pairs)==3
 c['os_monotonic_owner_does_not_claim_hardware_provider_fields']=all(x not in inspect.getsource(MonotonicRawSensorTimestampV1).lower() for x in ('ptp_clock','so_timestamping','hte_push','hardware_provider'))
 c['bounded_reference_work']=time.perf_counter()-started<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_MONOTONIC_RAW_SENSOR_TIMESTAMP_OWNERSHIP_GREEN','first_timestamp_us':r1.timestamp_us,'last_timestamp_ns':owner.last_timestamp_ns,'checks':c,'failed':fail,'remaining_red':['PHYSICAL_HARDWARE_TIMESTAMP_PROVENANCE','KERNEL_RECEIVE_TIMESTAMP_PROVENANCE','SENSOR_EXPOSURE_SAMPLE_TIME','PTP_HARDWARE_CLOCK_BINDING','SMOOTH_CONTINUOUS_DELAY_DENSITY','DIRECT_OS_TIMESTAMP_PARITY'],'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
 print(('FOUNDRY_MONOTONIC_RAW_SENSOR_TIMESTAMP_OWNERSHIP_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
