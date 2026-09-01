#!/usr/bin/env python3
from __future__ import annotations
import copy,json,time
from reference_crossmodal_concept_grounding_v1 import CrossmodalConceptGroundingV1
from reference_global_discourse_relevance_verify import fresh
from reference_local_optic_flow_v1 import LocalOpticFlowV1
from reference_open_world_perceptual_category_grounding_verify import SURFACE,pair
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_ENTITY_FEATURES,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_unsupervised_perceptual_features_v1 import FEATURE_QUORUM
from reference_visual_sensor_ingress_v1 import VisualSensorIngressV1
from reference_world_discourse_situation_bridge_v1 import WorldDiscourseSituationBridgeV1
SRC=0xD100;A=0xD201;B=0xD202;NOVEL=0xD301;MIX=0xD302;FS=0xD400;WS=0xD401
RA,RT,RI,RS,RV=(0xD501,0xD502,0xD503,0xD504,0xD505);DIRECT=((RA,101),(RT,301),(RI,302),(RS,401),(RV,402))
def img(points,n=14,lo=20,hi=200):
 a=[[lo]*n for _ in range(n)]
 for x,y in points:
  for yy in (y,y+1):
   for xx in (x,x+1):a[yy][xx]=hi
 return tuple(tuple(r) for r in a)
RIGHT=([(2,4),(7,4)],[(3,4),(8,4)]);EXP=([(3,4),(7,4)],[(2,4),(8,4)])
CW=([(3,3),(7,7)],[(4,2),(6,8)]);CCW=([(3,3),(7,7)],[(2,4),(8,6)]);CW_SHIFT=([(5,4),(9,8)],[(6,3),(8,9)])
def field(pattern,source=SRC,lo=20,hi=200):
 f=LocalOpticFlowV1();s=VisualSensorIngressV1();out=()
 for q,p in enumerate(pattern,1):
  im=img(p,lo=lo,hi=hi);raw,c=s.ingest(source,q,im,VisualSensorIngressV1.frame_digest(im));out=f.observe_frame(raw,c)
 return out
def train(pattern,source):
 f=LocalOpticFlowV1();s=VisualSensorIngressV1();seq=1;out=()
 for _ in range(FEATURE_QUORUM):
  for p in pattern:
   im=img(p);raw,c=s.ingest(source,seq,im,VisualSensorIngressV1.frame_digest(im));out=f.observe_frame(raw,c);seq+=1
  seq+=1
 return f,out,f.flow_feature(out)
def install(o,e,features):o.contact(CONTACT_ENTITY_FEATURES,(e,len(features),*features),FS,True,True)
def ground(g,a,o,e,c,s):
 for n in range(2):pair(g,a,o,e,SURFACE[c],s+n)
def main():
 t=time.perf_counter();c={}
 fr,fe,fcw,fccw=(field(x) for x in (RIGHT,EXP,CW,CCW))
 c['matched_local_vector_count_distinguishes_global_organizations']=len(fr)==len(fe)==len(fcw)==len(fccw)==2 and len({fr,fe,fcw,fccw})==4
 c['translated_clockwise_pattern_is_position_invariant']=field(CW_SHIFT)==fcw
 c['brightness_offset_preserves_clockwise_pattern']=field(CW,lo=50,hi=230)==fcw
 one=LocalOpticFlowV1();s=VisualSensorIngressV1()
 for q,p in enumerate(CW,1):
  im=img(p);raw,co=s.ingest(SRC,q,im,VisualSensorIngressV1.frame_digest(im));ow=one.observe_frame(raw,co)
 c['one_clockwise_exposure_is_insufficient']=one.flow_feature(ow)==0
 lw,_,cw=train(CW,SRC);lc,_,ccw=train(CCW,SRC+1);le,_,exp=train(EXP,SRC+2);lr,_,right=train(RIGHT,SRC+3)
 c['repeated_global_fields_learn_distinct_pattern_features']=all(x>0 for x in (cw,ccw,exp,right)) and len({cw,ccw,exp,right})==4
 cut=LocalOpticFlowV1.restore(copy.deepcopy(lw.checkpoint()));cut.learner.lesion_pair(*fcw)
 c['focal_global_pattern_lesion_preserves_local_field']=cut.flow_feature(fcw)==0 and field(CW)==fcw
 restored=LocalOpticFlowV1.restore(copy.deepcopy(lw.checkpoint()))
 c['checkpoint_preserves_global_pattern_not_active_components']=restored.flow_feature(fcw)==cw and not restored.previous
 c['global_flow_learning_has_zero_consequence_credit']=lw.learner.population.credit_events==0 and lw.learner.population.revision_events==0
 adult,host,*_=fresh();o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));g=CrossmodalConceptGroundingV1()
 install(o,A,(cw,));install(o,B,(exp,));install(o,NOVEL,(cw,));install(o,MIX,(cw,exp));ground(g,adult,o,A,201,0xD600);ground(g,adult,o,B,203,0xD610)
 c['novel_clockwise_flow_entity_inherits_grounded_concept']=g.resolve_world_atom(adult,o,NOVEL)==201 and g.resolve_raw_feature(adult,NOVEL)==0
 c['mixed_global_flow_evidence_refuses_arbitrary_category']=g.resolve_world_atom(adult,o,MIX)==0
 for i,(raw,concept) in enumerate(DIRECT):
  for n in range(2):pair(g,adult,o,raw,SURFACE[concept],0xD700+i*4+n)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);ctx,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g)
 for leaf in front:
  for _ in range(2):adult.experience_discourse_candidate(leaf.identity,Q,context=None,effort_q16=Q//16,duration=1);adult.experience_discourse_background(leaf.identity,False)
 o.contact(CONTACT_WORLD_STATE,(RA,NOVEL,RT,RI,RS,RV),WS,True,True);_,front=WorldDiscourseSituationBridgeV1.activate_frontier(adult,o,g);root=adult.organize_relevant_frontier(front)
 c['clockwise_global_flow_supports_long_form_discourse']=root is not None and len(front)==4
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];r={'contract':'FOUNDRY_ROTATION_HEADING_FLOW_POOLING_GREEN','checks':c,'failed':fail,'features':{'translation':right,'expansion':exp,'clockwise':cw,'counterclockwise':ccw},'remaining_red':['HEADING_CENTER_OF_FLOW_ESTIMATION','SPIRAL_CONTINUUM_POOLING','EYE_MOVEMENT_REAFFERENCE_COMPENSATION','PHYSICAL_CAMERA_DRIVER_OWNERSHIP','RAW_AUDIO_WAVEFORM_ELEMENT_EXTRACTION','DIRECT_GLOBAL_FLOW_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_ROTATION_HEADING_FLOW_POOLING_RED '+','.join(fail)) if fail else r['contract']);print(json.dumps(r,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
