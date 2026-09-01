#!/usr/bin/env python3
from __future__ import annotations
import inspect,json,time
from reference_canal_sensor_ingress_v1 import CanalSensorIngressV1
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_open_world_perceptual_category_grounding_verify import SURFACE,pair
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_tilt_translation_disambiguator_v1 import TiltTranslationDisambiguatorV1
from reference_vestibular_sensor_ingress_v1 import VestibularSensorIngressV1
from reference_vestibular_translation_heading_v1 import VestibularTranslationHeadingV1
from reference_visual_vestibular_heading_fusion_v1 import VisualVestibularHeadingFusionV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1
CS=0xAB10;OS=0xAB11;A=0xAB20;NOVEL=0xAB21;FS=0xAB30;WS=0xAB31
RA,RT,RI,RS,RV=(0xAB40,0xAB41,0xAB42,0xAB43,0xAB44);DIRECT=((RA,101),(RT,301),(RI,302),(RS,401),(RV,402));GIF=(500,866)
def canal(v):
 s=CanalSensorIngressV1();s.ingest(CS,1,v,CanalSensorIngressV1.sample_digest(v));return s
def otolith(sample=GIF):
 s=VestibularSensorIngressV1();s.ingest(OS,1,sample,VestibularSensorIngressV1.sample_digest(sample));return s
def install(o,e,f):o.contact(CONTACT_ENTITY_FEATURES,(e,1,f),FS,True,True)
def ground(g,a,o,e,c,s):
 for n in range(2):pair(g,a,o,e,SURFACE[c],s+n)
def main():
 t=time.perf_counter();c={};h=VestibularTranslationHeadingV1()
 r34={'translation_feature':1,'translation_magnitude':5,'residual':(3,4)};r68={'translation_feature':2,'translation_magnitude':10,'residual':(6,8)}
 a34=h.from_resolution(r34);a68=h.from_resolution(r68)
 c['positive_scaling_preserves_heading_coordinate']=a34==a68
 c['lateral_sign_reversal_reverses_heading_sign']=h.from_resolution({'translation_feature':3,'translation_magnitude':5,'residual':(-3,4)})==-a34
 c['forward_sign_reversal_moves_heading_to_opposite_half_plane']=abs(h.from_resolution({'translation_feature':4,'translation_magnitude':5,'residual':(3,-4)}))>90*256
 c['zero_or_tilt_explained_resolution_has_no_heading']=h.from_resolution({'translation_feature':0,'translation_magnitude':0,'residual':(0,0)}) is None
 upright=TiltTranslationDisambiguatorV1();ur=upright.resolve(otolith());uh=h.from_resolution(ur)
 tilted=TiltTranslationDisambiguatorV1();tilted.observe_canal(canal(30));tr=tilted.resolve(otolith());th=h.from_resolution(tr)
 c['same_raw_otolith_different_canal_history_yields_heading_vs_none']=uh is not None and th is None and ur['residual']==(500,-134) and tr['residual']==(0,0)
 bad=TiltTranslationDisambiguatorV1();gs=CanalSensorIngressV1();gs.ingest(CS,1,10,CanalSensorIngressV1.sample_digest(10));bad.observe_canal(gs);gs.ingest(CS,3,20,CanalSensorIngressV1.sample_digest(20));bad.observe_canal(gs)
 c['invalidated_canal_history_propagates_to_heading_refusal']=h.from_resolution(bad.resolve(otolith())) is None
 cut=TiltTranslationDisambiguatorV1();cut.observe_canal(canal(30));cut.lesion_canal_history();c['focal_canal_history_lesion_removes_heading_without_raw_otolith_change']=h.from_resolution(cut.resolve(otolith())) is None
 c['public_heading_api_accepts_only_resolved_receipt']=list(inspect.signature(VestibularTranslationHeadingV1.from_resolution).parameters)==['resolution']
 # Fast fuser now consumes a derived vestibular coordinate. Visual angular alignment is intentionally supplied in the same unit for this bounded slice.
 f=VisualVestibularHeadingFusionV1();
 for x in (uh-2,uh+2,uh-2,uh+2):f.observe_visual(x);f.observe_vestibular(x)
 fused=f.fuse(uh,uh)
 c['fast_fuser_consumes_derived_vestibular_heading_coordinate']=fused==uh
 token=h.token(uh);adult,host,*_=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1();install(o,A,token);install(o,NOVEL,token);ground(g,adult,o,A,201,0xAC00)
 c['derived_vestibular_heading_token_grounds_novel_entity']=g.resolve_world_atom(adult,o,NOVEL)==201
 for i,(raw,concept) in enumerate(DIRECT):
  for n in range(2):pair(g,adult,o,raw,SURFACE[concept],0xAD00+i*4+n)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);_,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
 for leaf in front:
  for _ in range(2):adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1);adult.experience_discourse_background(leaf.identity,False)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);_,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g);root=adult.organize_relevant_frontier(front)
 c['derived_vestibular_heading_supports_long_form_discourse']=root is not None and len(front)==4
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_VESTIBULAR_HEADING_FROM_TRANSLATION_RESIDUAL_GREEN','upright_heading_q8':uh,'tilt_heading':th,'checks':c,'failed':fail,'remaining_red':['LEARNED_CROSS_MODAL_HEADING_COORDINATE_CALIBRATION','THREE_DIMENSIONAL_VESTIBULAR_HEADING','VELOCITY_STORAGE','PHYSICAL_IMU_DRIVER_OWNERSHIP','WORLD_CENTERED_NAVIGATION','DIRECT_VESTIBULAR_HEADING_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_VESTIBULAR_HEADING_FROM_TRANSLATION_RESIDUAL_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
