#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_open_world_perceptual_category_grounding_verify import SURFACE,pair
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_BODY_STATE,CONTACT_ENTITY_FEATURES,CONTACT_WITHDRAW_SOURCE,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_visual_heading_center_v1 import VisualHeadingCenterV1,SCALE
from reference_visual_vestibular_body_frame_v1 import HEAD_NEUTRAL,VisualVestibularBodyFrameV1
from reference_visual_vestibular_reference_frame_v1 import EYE_NEUTRAL,VisualVestibularReferenceFrameV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1
BODY=0xD810;A=0xD820;B=0xD821;NOVEL=0xD822;FS=0xD830;WS=0xD831
RA,RT,RI,RS,RV=(0xD840,0xD841,0xD842,0xD843,0xD844);DIRECT=((RA,101),(RT,301),(RI,302),(RS,401),(RV,402))
def organism(eye,head):
 o=ReferenceOrganismV2(PopulationSpecV1(16384,2,3,16,8));o.contact(CONTACT_BODY_STATE,(EYE_NEUTRAL+eye,HEAD_NEUTRAL+head),BODY,True,True);return o
def install(o,e,f):o.contact(CONTACT_ENTITY_FEATURES,(e,1,f),FS,True,True)
def ground(g,a,o,e,c,s):
 for n in range(2):pair(g,a,o,e,SURFACE[c],s+n)
def main():
 t=time.perf_counter();c={};eye=VisualVestibularReferenceFrameV1();body=VisualVestibularBodyFrameV1();ret4=(4*SCALE,8*SCALE,1);ret12=(12*SCALE,8*SCALE,1);head8=(8*SCALE,8*SCALE,1);trunk11=(11*SCALE,8*SCALE,1)
 a=organism(4,3);b=organism(-4,3);neg=organism(4,-3);center=organism(4,0)
 c['two_distinct_retinal_eye_histories_compose_to_same_trunk_heading']=(body.visual_to_trunk(ret4,a)==trunk11 and body.visual_to_trunk(ret12,b)==trunk11)
 c['head_relative_vestibular_heading_uses_same_head_to_trunk_transform']=body.to_trunk(head8,a)==trunk11
 c['same_head_heading_under_different_head_positions_transforms_differently']=body.to_trunk(head8,a)!=body.to_trunk(head8,neg)
 c['centered_head_leaves_head_relative_heading_unchanged']=body.to_trunk(head8,center)==head8
 eye_cut=ReferenceOrganismV2(PopulationSpecV1(8192,2,3,16,8));eye_cut.contact(CONTACT_BODY_STATE,(999,HEAD_NEUTRAL+3),BODY,True,True)
 c['eye_coordinate_lesion_blocks_eye_to_head_but_preserves_head_to_trunk']=(eye.to_head(ret4,eye_cut) is None and body.to_trunk(head8,eye_cut)==trunk11)
 head_cut=ReferenceOrganismV2(PopulationSpecV1(8192,2,3,16,8));head_cut.contact(CONTACT_BODY_STATE,(EYE_NEUTRAL+4,999),BODY,True,True)
 c['head_coordinate_lesion_blocks_trunk_transform_but_preserves_eye_to_head']=(eye.to_head(ret4,head_cut)==head8 and body.to_trunk(head8,head_cut) is None)
 missing=ReferenceOrganismV2(PopulationSpecV1(4096,2,3,8,8));c['missing_body_occurrence_refuses_both_levels']=(eye.to_head(ret4,missing) is None and body.to_trunk(head8,missing) is None)
 withdrawn=organism(4,3);withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(BODY,),BODY,True,True);c['withdrawn_body_source_refuses_both_levels']=(eye.to_head(ret4,withdrawn) is None and body.to_trunk(head8,withdrawn) is None)
 cp=copy.deepcopy(a.checkpoint());restored=ReferenceOrganismV2.restore(cp);c['checkpoint_preserves_composed_transform_without_cached_heading']=(body.visual_to_trunk(ret4,restored)==trunk11 and all('trunk' not in str(k).lower() for k in cp.keys()))
 changed=organism(4,3);head_before=eye.to_head(ret4,changed);trunk_before=body.visual_to_trunk(ret4,changed);changed.contact(CONTACT_BODY_STATE,(EYE_NEUTRAL+4,HEAD_NEUTRAL-2),BODY,True,True);c['changing_only_head_coordinate_changes_trunk_not_eye_to_head']=(eye.to_head(ret4,changed)==head_before and body.visual_to_trunk(ret4,changed)!=trunk_before)
 src=inspect.getsource(VisualVestibularBodyFrameV1);c['trunk_transform_has_no_modality_weight_matrix_or_semantic_heading_table']=all(x not in src for x in ('weight','reliability','matrix','category','heading_class','reward'))
 h=VisualHeadingCenterV1();head_token=h.token(head8,18,18);trunk_token=h.token(trunk11,18,18);c['head_and_trunk_relative_tokens_are_distinct']=head_token!=trunk_token
 adult,host,*_=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1();install(o,A,trunk_token);install(o,B,head_token);install(o,NOVEL,trunk_token);ground(g,adult,o,A,201,0xD900);ground(g,adult,o,B,203,0xD910);c['novel_trunk_relative_entity_inherits_grounded_concept']=g.resolve_world_atom(adult,o,NOVEL)==201
 for i,(raw,concept) in enumerate(DIRECT):
  for n in range(2):pair(g,adult,o,raw,SURFACE[concept],0xDA00+i*4+n)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);ctx,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
 for leaf in front:
  for _ in range(2):adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1);adult.experience_discourse_background(leaf.identity,False)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);_,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g);root=adult.organize_relevant_frontier(front);c['trunk_relative_heading_supports_long_form_discourse']=root is not None and len(front)==4
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_HEAD_ON_TRUNK_REFERENCE_FRAME_GREEN','head_x':head8[0],'trunk_x':trunk11[0],'checks':c,'failed':fail,'remaining_red':['PHYSICAL_EYE_POSITION_TRANSDUCTION','PHYSICAL_NECK_PROPRIOCEPTION','PHYSICAL_VESTIBULAR_TRANSDUCTION','LEARNED_NONLINEAR_REFERENCE_FRAME_TRANSFORM','WORLD_CENTERED_NAVIGATION','DIRECT_BODY_FRAME_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_HEAD_ON_TRUNK_REFERENCE_FRAME_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
