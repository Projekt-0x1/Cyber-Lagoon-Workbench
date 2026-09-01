#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_open_world_perceptual_category_grounding_verify import SURFACE,pair
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES,CONTACT_WORLD_STATE
from reference_otolith_gravito_inertial_v1 import OtolithGravitoInertialV1
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_vestibular_sensor_ingress_v1 import VestibularSensorIngressV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1
SRC=0xE810;OTHER=0xE811;A=0xE820;B=0xE821;NOVEL=0xE822;FS=0xE830;WS=0xE831
RA,RT,RI,RS,RV=(0xE840,0xE841,0xE842,0xE843,0xE844);DIRECT=((RA,101),(RT,301),(RI,302),(RS,401),(RV,402))
def refuse(fn):
 try:fn();return False
 except ValueError:return True
def install(o,e,f):o.contact(CONTACT_ENTITY_FEATURES,(e,1,f),FS,True,True)
def ground(g,a,o,e,c,s):
 for n in range(2):pair(g,a,o,e,SURFACE[c],s+n)
def main():
 t=time.perf_counter();c={};sensor=VestibularSensorIngressV1();trans=OtolithGravitoInertialV1()
 c['zero_gravito_inertial_vector_refuses']=trans.transduce((0,0)) is None
 d34,m34=trans.transduce((3,4));d68,m68=trans.transduce((6,8));dneg,mneg=trans.transduce((-3,4));dquad,_=trans.transduce((3,-4))
 c['scaled_raw_vector_preserves_direction_but_changes_magnitude']=(d34==d68 and m34==5 and m68==10)
 c['sign_and_quadrant_changes_are_distinct_direction_features']=(len({d34,dneg,dquad})==3)
 raw,cont=sensor.ingest(SRC,1,(3,4),VestibularSensorIngressV1.sample_digest((3,4)))
 c['first_authenticated_sample_is_current_but_not_contiguous']=(raw==(3,4) and not cont and sensor.current_sample==(3,4))
 before=(dict(sensor.last_sequence),sensor.active_source,sensor.active_sequence,sensor.current_sample)
 c['forged_digest_refuses_before_sensor_mutation']=refuse(lambda:sensor.ingest(SRC,2,(6,8),'forged')) and before==(dict(sensor.last_sequence),sensor.active_source,sensor.active_sequence,sensor.current_sample)
 raw2,cont2=sensor.ingest(SRC,2,(6,8),VestibularSensorIngressV1.sample_digest((6,8)))
 c['monotonic_same_source_sample_is_contiguous']=(raw2==(6,8) and cont2)
 c['duplicate_sequence_refuses']=refuse(lambda:sensor.ingest(SRC,2,(6,8),VestibularSensorIngressV1.sample_digest((6,8))))
 other,other_cont=sensor.ingest(OTHER,1,(-3,4),VestibularSensorIngressV1.sample_digest((-3,4)))
 c['source_switch_breaks_continuity_but_exposes_authenticated_sample']=(other==(-3,4) and not other_cont and sensor.current_sample==(-3,4))
 cp=copy.deepcopy(sensor.checkpoint());restored=VestibularSensorIngressV1.restore(cp)
 c['checkpoint_preserves_provenance_but_no_raw_current_sample']=(restored.last_sequence==sensor.last_sequence and restored.current_sample is None and restored.active_source==0)
 restored.withdraw_source(SRC);c['withdrawn_source_blocks_future_ingress_and_clears_if_active']=refuse(lambda:restored.ingest(SRC,3,(3,4),VestibularSensorIngressV1.sample_digest((3,4))))
 tilt_feature=trans.transduce((3,4));translation_feature=trans.transduce((3,4))
 c['identical_raw_tilt_and_translation_observer_labels_remain_indistinguishable']=tilt_feature==translation_feature
 src=inspect.getsource(OtolithGravitoInertialV1)
 c['otolith_transducer_has_no_semantic_heading_tilt_translation_authority']=(not hasattr(trans,'heading') and all(x not in src.lower() for x in ('heading','tilt','translation','category','reward','expected')))
 adult,host,*_=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1();install(o,A,d34);install(o,B,dneg);install(o,NOVEL,d34);ground(g,adult,o,A,201,0xE900);ground(g,adult,o,B,203,0xE910)
 c['novel_raw_gravito_inertial_direction_inherits_grounded_concept_without_heading_claim']=g.resolve_world_atom(adult,o,NOVEL)==201 and g.resolve_raw_feature(adult,NOVEL)==0
 for i,(raw_atom,concept) in enumerate(DIRECT):
  for n in range(2):pair(g,adult,o,raw_atom,SURFACE[concept],0xEA00+i*4+n)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);ctx,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
 for leaf in front:
  for _ in range(2):adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1);adult.experience_discourse_background(leaf.identity,False)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);_,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g);root=adult.organize_relevant_frontier(front)
 c['raw_gravito_inertial_feature_supports_generic_grounded_long_form_discourse']=root is not None and len(front)==4
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_RAW_VESTIBULAR_GRAVITO_INERTIAL_GREEN','direction_feature':d34,'magnitudes':[m34,m68],'checks':c,'failed':fail,'remaining_red':['TILT_TRANSLATION_DISAMBIGUATION','SEMICIRCULAR_CANAL_TRANSDUCTION','VESTIBULAR_HEADING_FROM_MULTISENSOR_CAUSES','PHYSICAL_IMU_DRIVER_OWNERSHIP','THREE_DIMENSIONAL_GRAVITY_ESTIMATION','DIRECT_VESTIBULAR_TRANSDUCTION_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_RAW_VESTIBULAR_GRAVITO_INERTIAL_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
