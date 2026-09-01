#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_heading_center_of_flow_verify import radial
from reference_heading_coordinate_calibration_v1 import HeadingCoordinateCalibrationV1
from reference_tilt_translation_disambiguator_v1 import TiltTranslationDisambiguatorV1
from reference_vestibular_sensor_ingress_v1 import VestibularSensorIngressV1
from reference_vestibular_translation_heading_v1 import VestibularTranslationHeadingV1
from reference_visual_heading_center_v1 import VisualHeadingCenterV1
from reference_visual_vestibular_heading_fusion_v1 import VisualVestibularHeadingFusionV1
OS=0xBC10
def vestibular(sample):
 s=VestibularSensorIngressV1();s.ingest(OS,1,sample,VestibularSensorIngressV1.sample_digest(sample));r=TiltTranslationDisambiguatorV1().resolve(s);return VestibularTranslationHeadingV1.from_resolution(r)
def visual(x):return VisualHeadingCenterV1().estimate(*radial(x,8))[0]
def main():
 t=time.perf_counter();c={};vl,vc,vr=visual(4),visual(8),visual(12);al=vestibular((-500,1866));ac=vestibular((0,2000));ar=vestibular((500,1866))
 c['actual_visual_and_derived_vestibular_anchors_are_distinct_units']=(vl,vc,vr)==(1024,2048,3072) and al<0 and ac==0 and ar>0
 one=HeadingCoordinateCalibrationV1();one.observe_pair(vl,al);c['one_pair_cannot_identify_scale']=one.map_visual(vc) is None
 anchor=HeadingCoordinateCalibrationV1();anchor.observe_pair(vl,al);anchor.observe_pair(vl,al);c['repeated_one_anchor_still_cannot_identify_scale']=anchor.map_visual(vc) is None
 learned=HeadingCoordinateCalibrationV1()
 for _ in range(2):learned.observe_pair(vl,al);learned.observe_pair(vr,ar)
 c['two_repeated_distinct_anchors_earn_affine_relation']=learned.relation() is not None
 c['heldout_visual_center_maps_to_independently_derived_vestibular_center']=learned.map_visual(vc)==ac==0
 conflicting=HeadingCoordinateCalibrationV1()
 for _ in range(2):conflicting.observe_pair(vl,al);conflicting.observe_pair(vr,ar)
 conflicting.observe_pair(vl,ar)
 c['contradictory_pairing_at_one_anchor_forces_refusal']=conflicting.relation() is None
 third=HeadingCoordinateCalibrationV1()
 for _ in range(2):third.observe_pair(vl,al);third.observe_pair(vc,ac);third.observe_pair(vr,ar)
 c['consistent_third_anchor_preserves_relation']=third.map_visual(vc)==ac
 badthird=HeadingCoordinateCalibrationV1()
 for _ in range(2):badthird.observe_pair(vl,al);badthird.observe_pair(vr,ar);badthird.observe_pair(vc,123)
 c['inconsistent_third_anchor_forces_refusal']=badthird.relation() is None
 remap=HeadingCoordinateCalibrationV1(window=8)
 for _ in range(2):remap.observe_pair(vl,al);remap.observe_pair(vr,ar)
 old_left=remap.map_visual(vl)
 for _ in range(4):remap.observe_pair(vl,ar);remap.observe_pair(vr,al)
 c['bounded_window_can_replace_old_relation_with_reversed_mapping']=old_left==al and remap.map_visual(vl)==ar and remap.map_visual(vr)==al
 restored=HeadingCoordinateCalibrationV1.restore(copy.deepcopy(learned.checkpoint()))
 c['checkpoint_preserves_heldout_mapping_without_fused_heading']=restored.map_visual(vc)==ac and 'fused' not in restored.checkpoint()
 fnoise=VisualVestibularHeadingFusionV1()
 for x in (100,500,200,400):fnoise.observe_visual(x)
 for x in (300,301,299,300):fnoise.observe_vestibular(x)
 c['fast_reliability_history_does_not_change_calibration_relation']=learned.map_visual(vc)==ac
 f=VisualVestibularHeadingFusionV1()
 for x in (-1,1,-1,1):f.observe_visual(x);f.observe_vestibular(x)
 mapped=learned.map_visual(vc);fused=f.fuse(mapped,ac)
 c['fast_fuser_now_consumes_learned_mapped_visual_and_derived_vestibular_heading']=mapped==ac and fused==ac
 sig=list(inspect.signature(HeadingCoordinateCalibrationV1.observe_pair).parameters)
 c['pairing_api_contains_no_gain_offset_weight_or_semantic_label']=sig==['self','visual_coordinate','vestibular_angle']
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_LEARNED_CROSSMODAL_HEADING_COORDINATE_CALIBRATION_GREEN','visual_anchors':[vl,vc,vr],'vestibular_anchors':[al,ac,ar],'relation':learned.relation(),'checks':c,'failed':fail,'remaining_red':['CROSS_MODAL_SIMULTANEITY_OWNERSHIP','NONLINEAR_THREE_DIMENSIONAL_HEADING_CALIBRATION','EXTERNAL_ACCURACY_SUPERVISION','PHYSICAL_CAMERA_IMU_CALIBRATION','WORLD_CENTERED_NAVIGATION','DIRECT_HEADING_CALIBRATION_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_LEARNED_CROSSMODAL_HEADING_COORDINATE_CALIBRATION_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
