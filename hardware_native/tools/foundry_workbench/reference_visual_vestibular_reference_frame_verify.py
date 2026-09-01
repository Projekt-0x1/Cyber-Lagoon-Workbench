#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_open_world_perceptual_category_grounding_verify import SURFACE,pair
from reference_organism_v2 import (ReferenceOrganismV2,CONTACT_BODY_STATE,CONTACT_ENTITY_FEATURES,
    CONTACT_MOTOR_CONSEQUENCE,CONTACT_WITHDRAW_SOURCE,CONTACT_WORLD_STATE)
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_visual_heading_center_v1 import VisualHeadingCenterV1,SCALE
from reference_visual_vestibular_heading_fusion_v1 import VisualVestibularHeadingFusionV1
from reference_visual_vestibular_reference_frame_v1 import EYE_NEUTRAL,VisualVestibularReferenceFrameV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1
BODY=0xC810;WORLD=0xC811;A=0xC820;B=0xC821;NOVEL=0xC822;FS=0xC830;WS=0xC831
RA,RT,RI,RS,RV=(0xC840,0xC841,0xC842,0xC843,0xC844);DIRECT=((RA,101),(RT,301),(RI,302),(RS,401),(RV,402))
def organism(eye=None):
 o=ReferenceOrganismV2(PopulationSpecV1(16384,2,3,16,8))
 if eye is not None:o.contact(CONTACT_BODY_STATE,(EYE_NEUTRAL+int(eye),),BODY,True,True)
 return o
def stable_fuser():
 f=VisualVestibularHeadingFusionV1()
 for x in (1023,1025,1023,1025):f.observe_visual(x)
 for x in (2047,2049,2047,2049):f.observe_vestibular(x)
 return f
def install(o,e,f):o.contact(CONTACT_ENTITY_FEATURES,(e,1,f),FS,True,True)
def ground(g,a,o,e,c,s):
 for n in range(2):pair(g,a,o,e,SURFACE[c],s+n)
def main():
 t=time.perf_counter();c={};r=VisualVestibularReferenceFrameV1();ret4=(4*SCALE,8*SCALE,1);ret12=(12*SCALE,8*SCALE,1);head8=(8*SCALE,8*SCALE,1)
 plus=organism(4);minus=organism(-4);center=organism(0)
 c['opposite_retinal_eye_pairs_transform_to_same_head_heading']=(r.to_head(ret4,plus)==head8 and r.to_head(ret12,minus)==head8)
 c['same_retinal_heading_under_different_eye_positions_transforms_differently']=(r.to_head(ret4,plus)!=r.to_head(ret4,minus))
 c['centered_eye_leaves_retinal_heading_unchanged']=r.to_head(ret4,center)==ret4
 c['missing_current_body_occurrence_refuses']=r.to_head(ret4,ReferenceOrganismV2(PopulationSpecV1(4096,2,3,8,8))) is None
 withdrawn=organism(4);withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(BODY,),BODY,True,True)
 c['withdrawn_body_source_refuses_transform']=r.to_head(ret4,withdrawn) is None
 changed=organism(4);before=r.to_head(ret4,changed);changed.contact(CONTACT_BODY_STATE,(EYE_NEUTRAL+2,),BODY,True,True);after=r.to_head(ret4,changed)
 c['changing_current_body_state_changes_transform_not_retinal_input']=(before==head8 and after==(6*SCALE,8*SCALE,1) and ret4==(4*SCALE,8*SCALE,1))
 cp=copy.deepcopy(plus.checkpoint());restored=ReferenceOrganismV2.restore(cp)
 c['checkpoint_preserves_transform_via_body_state_without_transformed_cache']=(r.to_head(ret4,restored)==head8 and all('transform' not in str(k).lower() for k in cp.keys()))
 motor=ReferenceOrganismV2(PopulationSpecV1(8192,2,3,16,8));motor.contact(CONTACT_WORLD_STATE,(EYE_NEUTRAL,),WORLD,True,True);a=motor._issue_motor(0xC850);motor.contact(CONTACT_MOTOR_CONSEQUENCE,(a.ticket,1,1,EYE_NEUTRAL+4),WORLD,True,True)
 c['settled_motor_return_updates_same_current_body_transform_path']=(motor.body_state==(EYE_NEUTRAL+4,) and r.to_head(ret4,motor)==head8)
 f=stable_fuser();raw=f.fuse(ret4[0],head8[0]);transformed=f.fuse(r.to_head(ret4,plus)[0],head8[0])
 c['equal_reliability_fusion_becomes_congruent_only_after_reference_transform']=(raw==6*SCALE and transformed==8*SCALE)
 noisy=VisualVestibularHeadingFusionV1()
 for x in (500,1500,700,1300):noisy.observe_visual(x)
 for x in (2000,2050,2020,2030):noisy.observe_vestibular(x)
 c['fast_reliability_history_does_not_change_coordinate_transform']=r.to_head(ret4,plus)==head8
 src=inspect.getsource(VisualVestibularReferenceFrameV1)
 c['public_transform_has_no_eye_position_matrix_or_vestibular_argument']=(list(inspect.signature(VisualVestibularReferenceFrameV1.to_head).parameters)==['retinal_center','organism'] and all(x not in src for x in ('matrix','vestibular_heading','fast_weight','reliability')))
 h=VisualHeadingCenterV1();head_token=h.token(head8,18,18);raw_token=h.token((raw,8*SCALE,1),18,18)
 c['transformed_and_untransformed_fused_headings_have_distinct_grounding_tokens']=head_token!=raw_token
 adult,host,*_=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1();install(o,A,head_token);install(o,B,raw_token);install(o,NOVEL,head_token);ground(g,adult,o,A,201,0xC900);ground(g,adult,o,B,203,0xC910)
 c['novel_head_relative_entity_inherits_grounded_concept']=g.resolve_world_atom(adult,o,NOVEL)==201
 for i,(raw_atom,concept) in enumerate(DIRECT):
  for n in range(2):pair(g,adult,o,raw_atom,SURFACE[concept],0xCA00+i*4+n)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);ctx,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
 for leaf in front:
  for _ in range(2):adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1);adult.experience_discourse_background(leaf.identity,False)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);_,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g);root=adult.organize_relevant_frontier(front)
 c['head_relative_transformed_heading_supports_long_form_discourse']=root is not None and len(front)==4
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_VISUAL_VESTIBULAR_REFERENCE_FRAME_GREEN','retinal_x':ret4[0],'head_x':head8[0],'raw_fused':raw,'transformed_fused':transformed,'checks':c,'failed':fail,'remaining_red':['HEAD_ON_TRUNK_REFERENCE_TRANSFORM','PHYSICAL_EYE_POSITION_TRANSDUCTION','PHYSICAL_VESTIBULAR_TRANSDUCTION','LEARNED_NONLINEAR_REFERENCE_FRAME_TRANSFORM','SPIRAL_CONTINUUM_POOLING','DIRECT_REFERENCE_FRAME_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_VISUAL_VESTIBULAR_REFERENCE_FRAME_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
