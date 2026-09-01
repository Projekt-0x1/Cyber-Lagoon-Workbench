#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_open_world_perceptual_category_grounding_verify import SURFACE,pair
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES,CONTACT_MOTOR_CONSEQUENCE,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_visual_heading_center_v1 import VisualHeadingCenterV1
from reference_visual_pursuit_compensation_v1 import VisualPursuitCompensationV1
from reference_visual_vestibular_heading_fusion_v1 import VisualVestibularHeadingFusionV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1
SRC=0xA910;ACT=0xA911;A=0xA920;B=0xA921;NOVEL=0xA922;FS=0xA930;WS=0xA931
RA,RT,RI,RS,RV=(0xA940,0xA941,0xA942,0xA943,0xA944);DIRECT=((RA,101),(RT,301),(RI,302),(RS,401),(RV,402))
def fill(f,visual,vest):
 for x in visual:f.observe_visual(x)
 for x in vest:f.observe_vestibular(x)
 return f
def motor(delta,independent=True):
 o=ReferenceOrganismV2(PopulationSpecV1(8192,2,3,16,8));o.contact(CONTACT_WORLD_STATE,(10,),SRC,True,True);a=o._issue_motor(ACT);o.contact(CONTACT_MOTOR_CONSEQUENCE,(a.ticket,1,1,10+delta),SRC,True,independent);return o
def install(o,e,f):o.contact(CONTACT_ENTITY_FEATURES,(e,1,f),FS,True,True)
def ground(g,a,o,e,c,s):
 for n in range(2):pair(g,a,o,e,SURFACE[c],s+n)
def main():
 t=time.perf_counter();c={};low=[1000,1001,999,1000];high=[700,1300,800,1200]
 fv=fill(VisualVestibularHeadingFusionV1(),low,high);fb=fill(VisualVestibularHeadingFusionV1(),high,low);eq=fill(VisualVestibularHeadingFusionV1(),[900,1100,900,1100],[900,1100,900,1100])
 c['congruent_cues_reproduce_same_heading']=fv.fuse(1050,1050)==1050
 av=fv.fuse(1000,1200);ab=fb.fuse(1000,1200)
 c['low_noise_visual_history_pulls_conflict_toward_vision']=av is not None and abs(av-1000)<abs(av-1200)
 c['reversing_reliability_history_pulls_conflict_toward_vestibular']=ab is not None and abs(ab-1200)<abs(ab-1000)
 c['matched_reliability_compromises_without_arbitrary_winner']=eq.fuse(1000,1200)==1100
 same_pair_before=fill(VisualVestibularHeadingFusionV1(),low,low);before=same_pair_before.fuse(1000,1200)
 for x in high:same_pair_before.observe_visual(x)
 after=same_pair_before.fuse(1000,1200)
 c['increasing_visual_noise_alone_shifts_weight_away_from_vision']=after>before
 lesion=VisualVestibularHeadingFusionV1.restore(copy.deepcopy(fv.checkpoint()));vest_before=tuple(lesion.vestibular);lesion.reset_visual()
 c['visual_history_lesion_preserves_vestibular_history_and_refuses_until_relearned']=tuple(lesion.vestibular)==vest_before and lesion.fuse(1000,1200) is None
 restored=VisualVestibularHeadingFusionV1.restore(copy.deepcopy(fv.checkpoint()))
 c['checkpoint_preserves_modality_histories_not_fused_estimate']=restored.visual==fv.visual and restored.vestibular==fv.vestibular and 'fused' not in restored.checkpoint()
 bounded=fill(VisualVestibularHeadingFusionV1(window=6),high,low);old_visual_weight=bounded.weights()[0]
 for x in [1000,1001,999,1000,1001,999]:bounded.observe_visual(x)
 c['bounded_recent_window_replaces_old_noisy_visual_history']=(bounded.visual==[1000,1001,999,1000,1001,999] and bounded.weights()[0]>old_visual_weight)
 c['current_cue_swap_with_fixed_histories_changes_fusion']=fv.fuse(1000,1200)!=fv.fuse(1200,1000)
 insufficient=VisualVestibularHeadingFusionV1();insufficient.observe_visual(1000);fill(insufficient,[],low)
 c['insufficient_one_modality_history_cannot_silently_dominate']=insufficient.fuse(1000,1200) is None
 # Pursuit correction remains upstream of fusion.
 p=VisualPursuitCompensationV1();retinal=(1536,2048,1);self_o=motor(2,True);yoked_o=motor(2,False)
 corrected=p.correct(retinal,self_o);passive=p.correct(retinal,yoked_o)
 self_f=fill(VisualVestibularHeadingFusionV1(),[2047,2048,2049,2048],[1900,2200,1800,2300]);passive_f=fill(VisualVestibularHeadingFusionV1(),[1535,1536,1537,1536],[1900,2200,1800,2300])
 self_heading=self_f.fuse(corrected[0],2048);passive_heading=passive_f.fuse(passive[0],2048)
 c['pursuit_compensation_remains_upstream_and_yoked_history_cannot_borrow_it']=corrected[0]==2048 and passive[0]==1536 and self_heading!=passive_heading
 # Equal reliable conflict produces a genuinely fused midpoint heading token for public grounding.
 public=fill(VisualVestibularHeadingFusionV1(),[1279,1281,1279,1281],[2815,2817,2815,2817]);fused=public.fuse(1280,2816);h=VisualHeadingCenterV1();fused_token=h.token((fused,2048,1),18,18);left_token=h.token((1280,2048,1),18,18)
 c['equal_reliability_conflict_creates_midpoint_heading_coordinate']=fused==2048 and fused_token!=left_token
 adult,host,*_=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1();install(o,A,fused_token);install(o,B,left_token);install(o,NOVEL,fused_token);ground(g,adult,o,A,201,0xAA00);ground(g,adult,o,B,203,0xAA10)
 c['novel_fused_heading_entity_inherits_grounded_concept']=g.resolve_world_atom(adult,o,NOVEL)==201
 for i,(raw,concept) in enumerate(DIRECT):
  for n in range(2):pair(g,adult,o,raw,SURFACE[concept],0xAB00+i*4+n)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);ctx,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
 for leaf in front:
  for _ in range(2):adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1);adult.experience_discourse_background(leaf.identity,False)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);_,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g);root=adult.organize_relevant_frontier(front)
 c['fused_heading_supports_long_form_grounded_discourse']=root is not None and len(front)==4
 sig=list(inspect.signature(VisualVestibularHeadingFusionV1.fuse).parameters);c['public_fusion_call_has_no_weight_or_reliability_parameter']=sig==['self','current_visual','current_vestibular']
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];r={'contract':'FOUNDRY_VISUAL_VESTIBULAR_HEADING_FUSION_GREEN','visual_weighted':av,'vestibular_weighted':ab,'equal_fused':eq.fuse(1000,1200),'public_fused':fused,'checks':c,'failed':fail,'remaining_red':['SLOW_MULTISENSORY_CALIBRATION','PHYSICAL_VESTIBULAR_TRANSDUCTION','VESTIBULAR_VISUAL_REFERENCE_FRAME_TRANSFORM','SPIRAL_CONTINUUM_POOLING','DENSE_PER_PIXEL_OPTIC_FLOW','DIRECT_VISUAL_VESTIBULAR_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_VISUAL_VESTIBULAR_HEADING_FUSION_RED '+','.join(fail)) if fail else r['contract']);print(json.dumps(r,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
