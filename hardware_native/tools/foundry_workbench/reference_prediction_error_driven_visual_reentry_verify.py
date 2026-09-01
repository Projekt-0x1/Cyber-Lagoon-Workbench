#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_heading_center_of_flow_verify import img,radial
from reference_organism_v2 import ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_prediction_driven_visual_reentry_v1 import PredictionDrivenVisualReentryV1
from reference_visual_pathway_stage_occurrences_v1 import VisualPathwayStageOccurrencesV1
from reference_visual_vector_prediction_v1 import VisualVectorPredictionV1,HISTORY,QUORUM

def organism():return ReferenceOrganismV2(PopulationSpecV1(8192,2,3,16,8))
def partial(cx,cy,which,expand=True,n=18):
 r0,r1=(3,4) if expand else (4,3)
 if which=='lr':p0=[(cx-r0,cy),(cx+r0,cy)];p1=[(cx-r1,cy),(cx+r1,cy)]
 else:p0=[(cx,cy-r0),(cx,cy+r0)];p1=[(cx,cy-r1),(cx,cy+r1)]
 return img(p0,n),img(p1,n)
def vector(stage,o,frames):
 c=stage.process_components(o,*frames);return stage.process_vectors(o,c)
def train_prediction(pred,o,frames,n=3):
 for _ in range(n):
  s=VisualPathwayStageOccurrencesV1();pred.observe(vector(s,o,frames))
def main():
 t=time.perf_counter();c={};o=organism();expected_frames=radial(8,8);shifted_frames=radial(10,8)
 # One exposure is insufficient; three repeated complete fields earn exact row-set expectation.
 p=VisualVectorPredictionV1();s=VisualPathwayStageOccurrencesV1();full=vector(s,o,expected_frames);p.observe(full)
 c['one_complete_field_exposure_is_insufficient_for_prediction']=p.prediction() is None
 train_prediction(p,o,expected_frames,2);expected=p.prediction();c['three_repeated_complete_fields_earn_exact_vector_row_prediction']=(expected is not None and expected==VisualVectorPredictionV1.rows(full))
 # Same clear field changes route with learned prediction: naive recurrence vs learned feedforward.
 naive=PredictionDrivenVisualReentryV1();nv=vector(VisualPathwayStageOccurrencesV1(),o,expected_frames);nr=naive.observe_vector(o,nv,learn=False)
 learned=PredictionDrivenVisualReentryV1(copy.deepcopy(p));lv=vector(VisualPathwayStageOccurrencesV1(),o,expected_frames);lr=learned.observe_vector(o,lv,learn=False)
 c['same_clear_field_route_changes_after_expectation_learning']=(nr is not None and nr.vector_occurrences==1 and naive.reentry_count==1 and lr is not None and lr.vector_occurrences==0 and learned.feedforward_count==1 and learned.last_residual=={'missing':(),'unexpected':()})
 # Degraded subset reveals exact missing feature rows and cannot resolve by prediction alone.
 degraded=PredictionDrivenVisualReentryV1(copy.deepcopy(p));ds=VisualPathwayStageOccurrencesV1();v1=vector(ds,o,partial(8,8,'lr',True));r1=degraded.observe_vector(o,v1,learn=False);res1=degraded.last_residual
 missing=set(expected)-set(VisualVectorPredictionV1.rows(v1))
 c['degraded_subset_exposes_feature_specific_missing_rows']=r1 is None and set(res1['missing'])==missing and res1['unexpected']==()
 c['predicted_missing_rows_are_not_hallucinated_into_recurrent_evidence']=set(degraded.recurrent.rows)==set(VisualVectorPredictionV1.rows(v1)) and not missing.issubset(set(degraded.recurrent.rows))
 v2=vector(ds,o,partial(8,8,'tb',True));r2=degraded.observe_vector(o,v2,learn=False)
 c['actual_complementary_evidence_resolves_after_prediction_error_reentry']=(r2 is not None and r2.heading==(2048,2048,1) and r2.vector_occurrences==2)
 # Contradictory actual evidence remains unresolved despite prediction.
 bad=PredictionDrivenVisualReentryV1(copy.deepcopy(p));bs=VisualPathwayStageOccurrencesV1();bad.observe_vector(o,vector(bs,o,partial(8,8,'lr',True)),learn=False);br=bad.observe_vector(o,vector(bs,o,partial(8,8,'tb',False)),learn=False)
 c['prediction_cannot_override_contradictory_actual_evidence']=br is None
 # Unexpected but complete alternate field has feature-specific residual and resolves from observed geometry.
 alt=PredictionDrivenVisualReentryV1(copy.deepcopy(p));av=vector(VisualPathwayStageOccurrencesV1(),o,shifted_frames);ar=alt.observe_vector(o,av,learn=False);ares=alt.last_residual
 c['unexpected_complete_field_has_feature_specific_residual_but_resolves_from_observation']=(ar is not None and ar.heading==(2560,2048,1) and ar.vector_occurrences==1 and bool(ares['missing']) and bool(ares['unexpected']))
 # Tie between two complete-field expectations refuses a unique prediction.
 tie=VisualVectorPredictionV1();train_prediction(tie,o,expected_frames,3);train_prediction(tie,o,shifted_frames,3)
 c['equally_supported_complete_field_histories_refuse_unique_prediction']=tie.prediction() is None
 # Bounded later history replaces expectation.
 repl=VisualVectorPredictionV1.restore(copy.deepcopy(p.checkpoint()));train_prediction(repl,o,shifted_frames,HISTORY)
 c['bounded_later_history_replaces_expected_field']=repl.prediction()==VisualVectorPredictionV1.rows(vector(VisualPathwayStageOccurrencesV1(),o,shifted_frames)) and len(repl.history)==HISTORY
 cp=copy.deepcopy(degraded.checkpoint());rest=PredictionDrivenVisualReentryV1.restore(cp)
 c['checkpoint_preserves_prediction_history_but_drops_active_recurrent_evidence']=(rest.predictor.prediction()==p.prediction() and rest.recurrent.rows==[] and rest.last_residual is None)
 sig=list(inspect.signature(PredictionDrivenVisualReentryV1.observe_vector).parameters);src=inspect.getsource(PredictionDrivenVisualReentryV1)+inspect.getsource(VisualVectorPredictionV1)
 c['public_api_has_no_expected_heading_scalar_surprise_retry_or_missing_row_injection']=(sig==['self','o','vector_occurrence','learn'] and all(x not in src for x in ('expected_heading','surprise_score','retry_count','inject_missing','target_heading')))
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_PREDICTION_ERROR_DRIVEN_VISUAL_REENTRY_GREEN','prediction_quorum':QUORUM,'history_cap':HISTORY,'expected_rows':0 if expected is None else len(expected),'degraded_missing_rows':len(missing),'checks':c,'failed':fail,'remaining_red':['LEARNED_CONTEXT_CONDITIONAL_VISUAL_PREDICTIONS','BIOLOGICAL_PREDICTION_ERROR_AMPLITUDE','PREDICTIVE_FEEDBACK_TO_EARLIER_STAGE','RAPIDLY_VARYING_INTERNAL_LATENCY','DIRECT_VISUAL_PREDICTION_ERROR_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_PREDICTION_ERROR_DRIVEN_VISUAL_REENTRY_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
