#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_heading_center_of_flow_verify import img,radial
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_recurrent_visual_heading_v1 import RecurrentVisualHeadingV1
from reference_visual_pathway_stage_occurrences_v1 import VisualPathwayStageOccurrencesV1
WORLD=0xF110

def organism():return ReferenceOrganismV2(PopulationSpecV1(8192,2,3,16,8))
def partial(cx,cy,which,expand=True,n=18):
 r0,r1=(3,4) if expand else (4,3)
 if which=='lr':p0=[(cx-r0,cy),(cx+r0,cy)];p1=[(cx-r1,cy),(cx+r1,cy)]
 else:p0=[(cx,cy-r0),(cx,cy+r0)];p1=[(cx,cy-r1),(cx,cy+r1)]
 return img(p0,n),img(p1,n)
def vector(stage,o,frames):
 c=stage.process_components(o,*frames);return stage.process_vectors(o,c)
def main():
 t=time.perf_counter();c={};o=organism();stage=VisualPathwayStageOccurrencesV1();rec=RecurrentVisualHeadingV1()
 # Clear full radial evidence resolves in one vector occurrence.
 full=vector(stage,o,radial(8,8));one=rec.observe_vector(o,full)
 c['clear_full_radial_evidence_resolves_in_one_vector_occurrence']=(one is not None and one.heading==(2048,2048,1) and one.vector_occurrences==1)
 # Degraded complementary evidence requires re-entry.
 lr=vector(stage,o,partial(8,8,'lr',True));first=rec.observe_vector(o,lr)
 tb=vector(stage,o,partial(8,8,'tb',True));second=rec.observe_vector(o,tb)
 c['two_row_partial_evidence_is_initially_unresolved']=first is None
 c['complementary_same_tick_reentry_resolves_same_heading']=(second is not None and second.heading==one.heading and second.vector_occurrences==2)
 c['clear_and_degraded_paths_share_final_heading_but_have_different_recurrence_depth']=(one.heading==second.heading and one.vector_occurrences==1 and second.vector_occurrences==2)
 # Duplicate geometry does not increase unique rows enough to resolve.
 dup=RecurrentVisualHeadingV1();ds=VisualPathwayStageOccurrencesV1();d1=vector(ds,o,partial(8,8,'lr',True));d2=vector(ds,o,partial(8,8,'lr',True));x1=dup.observe_vector(o,d1);x2=dup.observe_vector(o,d2)
 c['duplicate_partial_geometry_does_not_fake_recurrent_resolution']=(x1 is None and x2 is None and len(dup.rows)==2)
 # Contradictory signs remain unresolved rather than averaging.
 bad=RecurrentVisualHeadingV1();bs=VisualPathwayStageOccurrencesV1();b1=vector(bs,o,partial(8,8,'lr',True));b2=vector(bs,o,partial(8,8,'tb',False));q1=bad.observe_vector(o,b1);q2=bad.observe_vector(o,b2)
 c['contradictory_recurrent_evidence_refuses_heading']=(q1 is None and q2 is None)
 # Tick change clears unresolved evidence: top/bottom alone after world contact remains insufficient.
 gap=RecurrentVisualHeadingV1();gs=VisualPathwayStageOccurrencesV1();g1=vector(gs,o,partial(8,8,'lr',True));gap.observe_vector(o,g1);o.contact(CONTACT_WORLD_STATE,(777,),WORLD,True,True);g2=vector(gs,o,partial(8,8,'tb',True));gout=gap.observe_vector(o,g2)
 c['world_history_tick_change_prevents_partial_evidence_carryover']=(gout is None and len(gap.rows)==2 and gap.tick==o.tick_count)
 # A stale vector occurrence from the old tick refuses directly.
 c['stale_vector_occurrence_refuses_reentry']=False
 try:gap.observe_vector(o,g1)
 except ValueError:c['stale_vector_occurrence_refuses_reentry']=True
 # Checkpoint drops unresolved pool, keeps completed recurrence sequence.
 cp=copy.deepcopy(rec.checkpoint());rest=RecurrentVisualHeadingV1.restore(cp)
 c['checkpoint_preserves_completed_sequence_but_drops_unresolved_pool']=(rest.sequence==rec.sequence and rest.rows==[] and rest.sources==[] and rest.current is None)
 sig=list(inspect.signature(RecurrentVisualHeadingV1.observe_vector).parameters);src=inspect.getsource(RecurrentVisualHeadingV1)
 c['public_reentry_api_has_no_retry_depth_target_heading_ambiguity_or_latency_argument']=(sig==['self','o','vector_occurrence'] and all(x not in src for x in ('retry_count','target_heading','ambiguity_flag','sleep(','latency_ms')))
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_RECURRENT_VISUAL_STAGE_REENTRY_GREEN','clear_depth':None if one is None else one.vector_occurrences,'degraded_depth':None if second is None else second.vector_occurrences,'heading':None if second is None else second.heading,'checks':c,'failed':fail,'remaining_red':['BIOLOGICAL_RECURRENT_LOOP_TIMING','PREDICTIVE_ERROR_DRIVEN_REENTRY','RAPIDLY_VARYING_INTERNAL_LATENCY','VESTIBULAR_RECURRENT_STAGE_REENTRY','DIRECT_RECURRENT_VISUAL_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_RECURRENT_VISUAL_STAGE_REENTRY_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
