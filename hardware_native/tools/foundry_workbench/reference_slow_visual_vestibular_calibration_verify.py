#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_open_world_perceptual_category_grounding_verify import SURFACE,pair
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_slow_visual_vestibular_calibration_v1 import CALIBRATION_QUORUM,SlowVisualVestibularCalibrationV1
from reference_visual_heading_center_v1 import VisualHeadingCenterV1
from reference_visual_vestibular_heading_fusion_v1 import VisualVestibularHeadingFusionV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1
A=0xB910;B=0xB911;NOVEL=0xB912;FS=0xB920;WS=0xB921
RA,RT,RI,RS,RV=(0xB930,0xB931,0xB932,0xB933,0xB934);DIRECT=((RA,101),(RT,301),(RI,302),(RS,401),(RV,402))
def train(cal,v=100,vest=120,n=8):
 for _ in range(n):cal.observe_pair(v,vest)
 return cal
def reliable_fuser():
 f=VisualVestibularHeadingFusionV1()
 for x in (1535,1537,1535,1537):f.observe_visual(x)
 for x in (1919,1921,1919,1921):f.observe_vestibular(x)
 return f
def install(o,e,f):o.contact(CONTACT_ENTITY_FEATURES,(e,1,f),FS,True,True)
def ground(g,a,o,e,c,s):
 for n in range(2):pair(g,a,o,e,SURFACE[c],s+n)
def main():
 t=time.perf_counter();c={}
 one=SlowVisualVestibularCalibrationV1();one.observe_pair(100,120)
 c['one_conflict_is_insufficient_for_slow_calibration']=(one.visual_offset,one.vestibular_offset)==(0,0)
 q=SlowVisualVestibularCalibrationV1()
 for _ in range(CALIBRATION_QUORUM):q.observe_pair(100,120)
 c['persistent_same_sign_conflict_crosses_slow_quorum']=(q.visual_offset,q.vestibular_offset)==(1,-2)
 c['vestibular_adaptation_is_qualitatively_twice_visual']=abs(q.vestibular_offset)==2*abs(q.visual_offset)
 # Fast reliability histories are deliberately opposite, but calibration sees neither.
 fast_visual=VisualVestibularHeadingFusionV1();fast_vest=VisualVestibularHeadingFusionV1()
 for x in (100,101,99,100):fast_visual.observe_visual(x)
 for x in (70,130,80,120):fast_visual.observe_vestibular(x)
 for x in (70,130,80,120):fast_vest.observe_visual(x)
 for x in (100,101,99,100):fast_vest.observe_vestibular(x)
 cv=train(SlowVisualVestibularCalibrationV1());cb=train(SlowVisualVestibularCalibrationV1())
 c['slow_calibration_is_independent_of_opposite_fast_reliability_histories']=(cv.visual_offset,cv.vestibular_offset)==(cb.visual_offset,cb.vestibular_offset)==(2,-4) and fast_visual.weights()!=fast_vest.weights()
 alt=SlowVisualVestibularCalibrationV1()
 for _ in range(6):alt.observe_pair(100,120);alt.observe_pair(120,100)
 c['alternating_conflict_signs_do_not_accumulate_calibration']=(alt.visual_offset,alt.vestibular_offset)==(0,0)
 rev=train(SlowVisualVestibularCalibrationV1())
 before=(rev.visual_offset,rev.vestibular_offset)
 for _ in range(8):rev.observe_pair(120,100)
 c['persistent_reversed_conflict_undoes_prior_calibration']=before==(2,-4) and (rev.visual_offset,rev.vestibular_offset)==(0,0)
 partial=SlowVisualVestibularCalibrationV1();partial.observe_pair(100,120);partial.observe_pair(100,120);restored=SlowVisualVestibularCalibrationV1.restore(copy.deepcopy(partial.checkpoint()));restored.observe_pair(100,120);updated=restored.observe_pair(100,120)
 c['checkpoint_preserves_partial_slow_conflict_history']=updated and (restored.visual_offset,restored.vestibular_offset)==(1,-2)
 lesions=train(SlowVisualVestibularCalibrationV1());vest=lesions.vestibular_offset;lesions.lesion_visual()
 c['focal_visual_calibration_lesion_preserves_vestibular_offset']=lesions.visual_offset==0 and lesions.vestibular_offset==vest
 slow=train(SlowVisualVestibularCalibrationV1());slow_state=copy.deepcopy(slow.checkpoint());fast=reliable_fuser();fast.reset_visual();fast.reset_vestibular()
 c['resetting_fast_reliability_history_does_not_change_slow_calibration']=slow.checkpoint()==slow_state
 # Fast output changes under the same current raw cues once slow offsets are applied.
 base_f=reliable_fuser();baseline=base_f.fuse(1536,1920);calibrated=base_f.fuse(slow.calibrate_visual(1536),slow.calibrate_vestibular(1920))
 c['slow_calibration_changes_existing_fast_fusion_without_changing_fast_history']=baseline==1728 and calibrated==1727 and len(base_f.visual)==4 and len(base_f.vestibular)==4
 # Public heading sector crosses at x=1728 after slow calibration.
 h=VisualHeadingCenterV1();baseline_token=h.token((baseline,2048,1),18,18);calibrated_token=h.token((calibrated,2048,1),18,18)
 c['slow_calibration_crosses_public_heading_sector_boundary']=baseline_token!=calibrated_token
 adult,host,*_=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1();install(o,A,calibrated_token);install(o,B,baseline_token);install(o,NOVEL,calibrated_token);ground(g,adult,o,A,201,0xBA00);ground(g,adult,o,B,203,0xBA10)
 c['novel_slow_calibrated_heading_inherits_grounded_concept']=g.resolve_world_atom(adult,o,NOVEL)==201
 for i,(raw,concept) in enumerate(DIRECT):
  for n in range(2):pair(g,adult,o,raw,SURFACE[concept],0xBB00+i*4+n)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);ctx,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
 for leaf in front:
  for _ in range(2):adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1);adult.experience_discourse_background(leaf.identity,False)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);_,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g);root=adult.organize_relevant_frontier(front)
 c['slow_calibrated_multisensory_heading_supports_long_form_discourse']=root is not None and len(front)==4
 sig=list(inspect.signature(SlowVisualVestibularCalibrationV1.observe_pair).parameters);c['calibration_api_has_no_fast_reliability_or_weight_argument']=sig==['self','visual','vestibular']
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];r={'contract':'FOUNDRY_SLOW_VISUAL_VESTIBULAR_CALIBRATION_GREEN','offsets':{'visual':slow.visual_offset,'vestibular':slow.vestibular_offset},'baseline_fused':baseline,'calibrated_fused':calibrated,'checks':c,'failed':fail,'remaining_red':['PHYSICAL_VESTIBULAR_TRANSDUCTION','LONG_HORIZON_MULTISENSORY_CALIBRATION','VESTIBULAR_VISUAL_REFERENCE_FRAME_TRANSFORM','SPIRAL_CONTINUUM_POOLING','DIRECT_SLOW_MULTISENSORY_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_SLOW_VISUAL_VESTIBULAR_CALIBRATION_RED '+','.join(fail)) if fail else r['contract']);print(json.dumps(r,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
