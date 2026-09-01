#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_heading_center_of_flow_verify import radial
from reference_open_world_perceptual_category_grounding_verify import SURFACE,pair
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_BODY_STATE,CONTACT_ENTITY_FEATURES,CONTACT_MOTOR_CONSEQUENCE,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_visual_heading_center_v1 import VisualHeadingCenterV1,SCALE
from reference_visual_pursuit_compensation_v1 import VisualPursuitCompensationV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1
SRC=0xF100;ACTION=0xF101;A=0xF201;B=0xF202;NOVEL=0xF301;PASSIVE=0xF302;FS=0xF400;WS=0xF401
RA,RT,RI,RS,RV=(0xF501,0xF502,0xF503,0xF504,0xF505);DIRECT=((RA,101),(RT,301),(RI,302),(RS,401),(RV,402))
def organism(pos=10):
 o=ReferenceOrganismV2(PopulationSpecV1(16384,2,3,16,8));o.contact(CONTACT_WORLD_STATE,(int(pos),),SRC,True,True);return o
def pursue(delta,independent=True,effect=1,action_id=ACTION):
 o=organism(10);a=o._issue_motor(int(action_id));o.contact(CONTACT_MOTOR_CONSEQUENCE,(a.ticket,int(effect),1,10+int(delta)),SRC,True,bool(independent));return o,a
def install(o,e,f):o.contact(CONTACT_ENTITY_FEATURES,(e,1,f),FS,True,True)
def ground(g,a,o,e,c,s):
 for n in range(2):pair(g,a,o,e,SURFACE[c],s+n)
def main():
 t=time.perf_counter();c={};h=VisualHeadingCenterV1();p=VisualPursuitCompensationV1(max_lag=8)
 retinal=h.estimate(*radial(6,8));true_center=h.estimate(*radial(8,8));assert retinal and true_center
 c['retinal_shift_is_real_before_compensation']=retinal[0]!=true_center[0]
 no_action=organism(10);c['identical_retinal_shift_without_motor_action_is_not_corrected']=p.correct(retinal,no_action)==retinal
 self1,a1=pursue(1);self2,a2=pursue(2)
 corrected1=p.correct(retinal,self1);corrected2=p.correct(retinal,self2)
 c['matching_self_generated_return_corrects_world_relative_heading']=corrected2[:2]==true_center[:2]
 c['pursuit_magnitude_scales_correction']=corrected1[0]-retinal[0]==SCALE and corrected2[0]-retinal[0]==2*SCALE
 yoked,ya=pursue(2,independent=False)
 c['yoked_non_independent_return_is_not_correction_authority']=p.correct(retinal,yoked)==retinal and not ya.independent_consequence
 zero,za=pursue(2,independent=True,effect=0)
 c['zero_effect_motor_return_is_not_correction_authority']=p.correct(retinal,zero)==retinal and za.effect==0
 mismatch,ma=pursue(2);mismatch.contact(CONTACT_BODY_STATE,(13,),SRC,True,True)
 c['later_mismatching_body_state_invalidates_compensation']=p.correct(retinal,mismatch)==retinal and ma.state_after==(12,)
 reverse,ra=pursue(-2);rev=p.correct(true_center,reverse)
 c['reversing_returned_actuator_delta_reverses_correction']=rev[0]==true_center[0]-2*SCALE
 same_id,_=pursue(-2,action_id=ACTION)
 c['action_id_does_not_determine_correction_sign']=p.resident_delta(self2)==2 and p.resident_delta(same_id)==-2
 cp=copy.deepcopy(self2.checkpoint());restored=ReferenceOrganismV2.restore(cp)
 c['immediate_checkpoint_replay_preserves_valid_causal_compensation']=p.correct(retinal,restored)==corrected2
 for n in range(10):restored.contact(CONTACT_WORLD_STATE,(100+n,),SRC,True,True)
 c['stale_motor_history_expires_without_rewriting_retinal_estimate']=p.correct(retinal,restored)==retinal
 cut=ReferenceOrganismV2.restore(cp);cut.motor_actions[-1].independent_consequence=False
 c['focal_motor_causal_evidence_lesion_removes_only_compensation']=p.correct(retinal,cut)==retinal and h.estimate(*radial(6,8))==retinal
 c['same_retinal_state_different_causal_histories_yield_different_world_estimates']=p.correct(retinal,self2)!=p.correct(retinal,no_action)
 true_token=h.token(true_center,18,18);self_token=h.token(corrected2,18,18);passive_token=h.token(retinal,18,18)
 c['self_generated_compensation_recovers_true_heading_token']=self_token==true_token and passive_token!=true_token
 adult,host,*_=fresh();wo=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1()
 install(wo,A,true_token);install(wo,B,passive_token);install(wo,NOVEL,self_token);install(wo,PASSIVE,passive_token)
 ground(g,adult,wo,A,201,0xF600);ground(g,adult,wo,B,203,0xF610)
 c['corrected_novel_inherits_true_heading_concept_while_passive_does_not']=g.resolve_world_atom(adult,wo,NOVEL)==201 and g.resolve_world_atom(adult,wo,PASSIVE)==203
 for i,(raw,concept) in enumerate(DIRECT):
  for n in range(2):pair(g,adult,wo,raw,SURFACE[concept],0xF700+i*4+n)
 wo.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);ctx,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,wo,g)
 for leaf in front:
  for _ in range(2):adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1);adult.experience_discourse_background(leaf.identity,False)
 wo.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);_,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,wo,g);root=adult.organize_relevant_frontier(front)
 c['reafference_compensated_heading_supports_long_form_discourse']=root is not None and len(front)==4
 src=inspect.getsource(VisualPursuitCompensationV1)
 c['public_correction_api_has_no_ticket_velocity_or_action_argument']=list(inspect.signature(VisualPursuitCompensationV1.correct).parameters)==['self','retinal_center','organism']
 c['bridge_has_no_semantic_pursuit_direction_table']=all(x not in src for x in ('eye_left','eye_right','heading_class','expected_heading','velocity_q16'))
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];r={'contract':'FOUNDRY_EYE_MOVEMENT_REAFFERENCE_HEADING_COMPENSATION_GREEN','retinal_center':retinal,'true_center':true_center,'corrected':corrected2,'resident_delta':p.resident_delta(self2),'checks':c,'failed':fail,'remaining_red':['VESTIBULAR_VISUAL_HEADING_FUSION','SPIRAL_CONTINUUM_POOLING','PHYSICAL_OCULOMOTOR_SENSOR_OWNERSHIP','DENSE_PER_PIXEL_OPTIC_FLOW','RAW_AUDIO_WAVEFORM_ELEMENT_EXTRACTION','DIRECT_REAFFERENCE_HEADING_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_EYE_MOVEMENT_REAFFERENCE_HEADING_COMPENSATION_RED '+','.join(fail)) if fail else r['contract']);print(json.dumps(r,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
