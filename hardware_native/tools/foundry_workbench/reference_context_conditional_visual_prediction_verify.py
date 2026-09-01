#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_context_prediction_driven_visual_reentry_v1 import ContextPredictionDrivenVisualReentryV1
from reference_contextual_visual_prediction_v1 import ContextConditionalVisualPredictionV1,HISTORY,QUORUM
from reference_heading_center_of_flow_verify import img,radial
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_visual_pathway_stage_occurrences_v1 import VisualPathwayStageOccurrencesV1
from reference_visual_vector_prediction_v1 import VisualVectorPredictionV1
SRC=0xF210;A=(10101,);B=(20202,);C=(30303,)
def organism():return ReferenceOrganismV2(PopulationSpecV1(8192,2,3,16,8))
def world(o,state):o.contact(CONTACT_WORLD_STATE,state,SRC,True,True)
def vector(o,frames):
 s=VisualPathwayStageOccurrencesV1();c=s.process_components(o,*frames);return s.process_vectors(o,c)
def train(p,o,state,frames,n=3):
 world(o,state)
 for _ in range(n):p.observe(o,vector(o,frames))
def partial(cx,cy):return img([(cx-3,cy),(cx+3,cy)]),img([(cx-4,cy),(cx+4,cy)])
def main():
 t=time.perf_counter();c={};o=organism();f8=radial(8,8);f10=radial(10,8);p=ContextConditionalVisualPredictionV1();train(p,o,A,f8,3);pa=p.prediction(o);train(p,o,B,f10,3);pb=p.prediction(o)
 c['two_world_contexts_learn_distinct_visual_expectations']=pa is not None and pb is not None and pa!=pb
 same=vector(o,f8);rb=p.residual(o,same);world(o,A);ra=p.residual(o,vector(o,f8))
 c['same_visual_field_is_expected_in_A_and_feature_mismatched_in_B']=(ra=={'missing':(),'unexpected':()} and rb is not None and bool(rb['missing']) and bool(rb['unexpected']))
 # Route changes solely via resident world-state switch.
 dr=ContextPredictionDrivenVisualReentryV1(copy.deepcopy(p));world(o,A);aout=dr.observe_vector(o,vector(o,f8),learn=False);fa=dr.feedforward_count;world(o,B);bout=dr.observe_vector(o,vector(o,f8),learn=False)
 c['same_visual_field_routes_feedforward_in_A_and_recurrently_in_B']=(aout is not None and aout.vector_occurrences==0 and fa==1 and bout is not None and bout.vector_occurrences==1 and dr.reentry_count==1)
 world(o,C);cpred=p.prediction(o);c['unseen_context_has_no_prediction_and_borrows_neither_A_nor_B']=cpred is None
 unseen=ContextPredictionDrivenVisualReentryV1(copy.deepcopy(p));cout=unseen.observe_vector(o,vector(o,f8),learn=False);c['unseen_context_uses_observed_recurrent_path_not_foreign_prediction']=(cout is not None and cout.vector_occurrences==1 and unseen.last_residual is None)
 world(o,A);pv=vector(o,partial(8,8));res=p.residual(o,pv);missing=set(pa)-set(VisualVectorPredictionV1.rows(pv));c['degraded_A_evidence_exposes_A_specific_missing_rows']=set(res['missing'])==missing and res['unexpected']==()
 # Tie within A should not damage B.
 tie=ContextConditionalVisualPredictionV1.restore(copy.deepcopy(p.checkpoint()));train(tie,o,A,f10,3);world(o,A);ta=tie.prediction(o);world(o,B);tb=tie.prediction(o)
 c['tied_A_history_refuses_A_without_damaging_B_prediction']=ta is None and tb==pb
 # Bounded later A remap while B stays stable.
 repl=ContextConditionalVisualPredictionV1.restore(copy.deepcopy(p.checkpoint()));train(repl,o,A,f10,HISTORY);world(o,A);actx=ContextConditionalVisualPredictionV1.context(o);newa=repl.prediction(o);world(o,B);newb=repl.prediction(o)
 c['bounded_later_A_history_remaps_A_without_altering_B']=newa==pb and newb==pb and len(repl.histories[actx])==HISTORY
 restored=ContextPredictionDrivenVisualReentryV1.restore(copy.deepcopy(dr.checkpoint()));world(o,A);c['checkpoint_preserves_contextual_prediction_history_but_drops_active_reentry']=restored.predictor.prediction(o)==pa and restored.recurrent.rows==[]
 sig=list(inspect.signature(ContextConditionalVisualPredictionV1.observe).parameters);src=inspect.getsource(ContextConditionalVisualPredictionV1)+inspect.getsource(ContextPredictionDrivenVisualReentryV1)
 c['public_api_has_no_context_id_expected_heading_or_route_selector']=(sig==['self','organism','vector_occurrence'] and all(x not in src for x in ('context_id','expected_heading','route_selector','corridor_position')))
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];r={'contract':'FOUNDRY_CONTEXT_CONDITIONAL_VISUAL_PREDICTION_GREEN','history_cap':HISTORY,'quorum':QUORUM,'checks':c,'failed':fail,'remaining_red':['LEARNED_RELATIONAL_CONTEXT_GENERALIZATION','PREDICTIVE_FEEDBACK_TO_EARLIER_STAGE','BIOLOGICAL_PREDICTION_ERROR_AMPLITUDE','RAPIDLY_VARYING_INTERNAL_LATENCY','DIRECT_CONTEXTUAL_VISUAL_PREDICTION_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_CONTEXT_CONDITIONAL_VISUAL_PREDICTION_RED '+','.join(fail)) if fail else r['contract']);print(json.dumps(r,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
