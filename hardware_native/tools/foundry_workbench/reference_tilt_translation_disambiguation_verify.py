#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_canal_sensor_ingress_v1 import CanalSensorIngressV1
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_open_world_perceptual_category_grounding_verify import SURFACE,pair
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES,CONTACT_WORLD_STATE
from reference_otolith_gravito_inertial_v1 import OtolithGravitoInertialV1
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_tilt_translation_disambiguator_v1 import TiltTranslationDisambiguatorV1
from reference_vestibular_sensor_ingress_v1 import VestibularSensorIngressV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1
CS=0xA810;OS=0xA811;A=0xA820;NOVEL=0xA821;FS=0xA830;WS=0xA831
RA,RT,RI,RS,RV=(0xA840,0xA841,0xA842,0xA843,0xA844);DIRECT=((RA,101),(RT,301),(RI,302),(RS,401),(RV,402));GIF=(500,866)
def canal(value,source=CS,seq=1):
 s=CanalSensorIngressV1();s.ingest(source,seq,value,CanalSensorIngressV1.sample_digest(value));return s
def otolith(sample=GIF,source=OS,seq=1):
 s=VestibularSensorIngressV1();s.ingest(source,seq,sample,VestibularSensorIngressV1.sample_digest(sample));return s
def install(o,e,f):o.contact(CONTACT_ENTITY_FEATURES,(e,1,f),FS,True,True)
def ground(g,a,o,e,c,s):
 for n in range(2):pair(g,a,o,e,SURFACE[c],s+n)
def main():
 t=time.perf_counter();c={};raw_feature=OtolithGravitoInertialV1.transduce(GIF)[0]
 upright=TiltTranslationDisambiguatorV1();u=otolith();ur=upright.resolve(u)
 c['upright_history_leaves_nonzero_translation_residual']=(ur is not None and ur['translation_feature']>0 and ur['residual']==(500,-134))
 tilted=TiltTranslationDisambiguatorV1();cs=canal(30);tilted.observe_canal(cs);to=otolith();tr=tilted.resolve(to)
 c['same_otolith_after_matching_canal_tilt_is_explained_as_gravity']=(tr is not None and tr['translation_feature']==0 and tr['translation_magnitude']==0 and tr['gravity']==GIF)
 opposite=TiltTranslationDisambiguatorV1();ocs=canal(-30);opposite.observe_canal(ocs);orr=opposite.resolve(otolith())
 c['opposite_canal_history_yields_different_nonzero_residual']=(orr is not None and orr['translation_feature']>0 and orr['translation_feature']!=ur['translation_feature'])
 reversal=TiltTranslationDisambiguatorV1();rs=CanalSensorIngressV1();rs.ingest(CS,1,30,CanalSensorIngressV1.sample_digest(30));reversal.observe_canal(rs);rs.ingest(CS,2,-30,CanalSensorIngressV1.sample_digest(-30));reversal.observe_canal(rs);rr=reversal.resolve(otolith())
 c['reversed_canal_history_back_to_upright_restores_original_translation_residual']=(rr is not None and rr['translation_feature']==ur['translation_feature'] and rr['residual']==ur['residual'])
 gap=TiltTranslationDisambiguatorV1();gs=CanalSensorIngressV1();gs.ingest(CS,1,10,CanalSensorIngressV1.sample_digest(10));gap.observe_canal(gs);gs.ingest(CS,3,20,CanalSensorIngressV1.sample_digest(20));gap.observe_canal(gs)
 c['canal_sequence_gap_invalidates_gravity_estimate_and_forces_refusal']=gap.resolve(otolith()) is None and not gap.orientation_valid
 switch=TiltTranslationDisambiguatorV1();ss=CanalSensorIngressV1();ss.ingest(CS,1,10,CanalSensorIngressV1.sample_digest(10));switch.observe_canal(ss);ss.ingest(CS+1,1,20,CanalSensorIngressV1.sample_digest(20));switch.observe_canal(ss)
 c['canal_source_switch_invalidates_gravity_estimate']=switch.resolve(otolith()) is None and not switch.orientation_valid
 cp=copy.deepcopy(tilted.checkpoint());restored=TiltTranslationDisambiguatorV1.restore(cp);cr=restored.resolve(otolith())
 c['checkpoint_preserves_integrated_orientation_without_raw_sensor_sample']=(restored.orientation_deg==30 and cr is not None and cr['translation_feature']==0 and 'raw' not in ''.join(cp.keys()).lower())
 cut=TiltTranslationDisambiguatorV1.restore(cp);cut.lesion_canal_history();c['focal_canal_history_lesion_refuses_without_changing_raw_otolith_truth']=(cut.resolve(otolith()) is None and OtolithGravitoInertialV1.transduce(GIF)[0]==raw_feature)
 withdrawn=otolith();withdrawn.withdraw_source(OS);c['otolith_source_withdrawal_prevents_resolution']=upright.resolve(withdrawn) is None
 c['raw_otolith_direction_is_identical_across_tilt_and_translation_histories']=(OtolithGravitoInertialV1.transduce(GIF)[0]==raw_feature and raw_feature==OtolithGravitoInertialV1.transduce(GIF)[0])
 c['resolver_api_has_no_tilt_translation_label_or_gravity_argument']=list(inspect.signature(TiltTranslationDisambiguatorV1.resolve).parameters)==['self','otolith_sensor']
 adult,host,*_=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1();install(o,A,ur['translation_feature']);install(o,NOVEL,ur['translation_feature']);ground(g,adult,o,A,201,0xA900)
 c['translation_residual_feature_can_ground_novel_entity_while_tilt_arm_has_no_feature']=(g.resolve_world_atom(adult,o,NOVEL)==201 and tr['translation_feature']==0)
 for i,(raw_atom,concept) in enumerate(DIRECT):
  for n in range(2):pair(g,adult,o,raw_atom,SURFACE[concept],0xAA00+i*4+n)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);ctx,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
 for leaf in front:
  for _ in range(2):adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1);adult.experience_discourse_background(leaf.identity,False)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);_,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g);root=adult.organize_relevant_frontier(front);c['disambiguated_translation_feature_supports_long_form_discourse']=root is not None and len(front)==4
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_TILT_TRANSLATION_DISAMBIGUATION_GREEN','raw_otolith_feature':raw_feature,'upright_residual':ur['residual'],'tilt_residual':tr['residual'],'orientation_deg':tilted.orientation_deg,'checks':c,'failed':fail,'remaining_red':['VESTIBULAR_HEADING_FROM_TRANSLATION_RESIDUAL','VELOCITY_STORAGE','THREE_DIMENSIONAL_GRAVITY_ESTIMATION','PHYSICAL_IMU_DRIVER_OWNERSHIP','LONG_HORIZON_GRAVITY_CALIBRATION','DIRECT_TILT_TRANSLATION_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_TILT_TRANSLATION_DISAMBIGUATION_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
