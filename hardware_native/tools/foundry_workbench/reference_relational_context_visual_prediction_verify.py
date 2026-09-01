#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_context_prediction_driven_visual_reentry_v1 import ContextPredictionDrivenVisualReentryV1
from reference_heading_center_of_flow_verify import radial
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_relational_context_visual_prediction_v1 import RelationalContextVisualPredictionV1,HISTORY,QUORUM
from reference_visual_pathway_stage_occurrences_v1 import VisualPathwayStageOccurrencesV1
from reference_visual_vector_prediction_v1 import VisualVectorPredictionV1
SRC=0xF310
A=(11,12,13,14);APRIME=(11,12,13,15);WEAK=(11,12,16,17)
B=(11,12,13,24);AMB=(11,12,13,99);D=(11,12,13,25)

def organism():return ReferenceOrganismV2(PopulationSpecV1(8192,2,3,16,8))
def world(o,state):o.contact(CONTACT_WORLD_STATE,state,SRC,True,True)
def vector(o,frames):
 s=VisualPathwayStageOccurrencesV1();c=s.process_components(o,*frames);return s.process_vectors(o,c)
def train(p,o,state,frames,n=3):
 world(o,state)
 for _ in range(n):p.observe(o,vector(o,frames))
def main():
 t=time.perf_counter();c={};o=organism();f8=radial(8,8);f10=radial(10,8);rows8=VisualVectorPredictionV1.rows(vector(o,f8));rows10=VisualVectorPredictionV1.rows(vector(o,f10))
 p=RelationalContextVisualPredictionV1();train(p,o,A,f8,3);world(o,APRIME)
 c['unseen_three_of_four_overlap_context_inherits_donor_prediction']=p.prediction(o)==rows8 and RelationalContextVisualPredictionV1.related(A,APRIME)
 world(o,WEAK);c['two_of_four_overlap_context_does_not_inherit']=p.prediction(o) is None and not RelationalContextVisualPredictionV1.related(A,WEAK)
 # Same visual field routes feedforward in a related context after transfer, despite never being trained there.
 route=ContextPredictionDrivenVisualReentryV1(copy.deepcopy(p));world(o,APRIME);out=route.observe_vector(o,vector(o,f8),learn=False)
 c['relationally_inherited_expectation_changes_route_in_never_trained_context']=(out is not None and out.vector_occurrences==0 and route.feedforward_count==1 and route.last_residual=={'missing':(),'unexpected':()})
 # Equal incompatible related donors must refuse.
 amb=RelationalContextVisualPredictionV1();train(amb,o,A,f8,3);train(amb,o,B,f10,3);world(o,AMB)
 c['equal_incompatible_related_donor_families_refuse']=amb.prediction(o) is None and RelationalContextVisualPredictionV1.related(A,AMB) and RelationalContextVisualPredictionV1.related(B,AMB)
 # Agreeing related donors strengthen one transferable expectation.
 agree=RelationalContextVisualPredictionV1();train(agree,o,A,f8,3);train(agree,o,D,f8,3);world(o,AMB)
 c['multiple_related_donors_agreeing_on_one_field_transfer']=agree.prediction(o)==rows8
 # Exact local experience overrides inherited donor after quorum.
 exact=RelationalContextVisualPredictionV1.restore(copy.deepcopy(p.checkpoint()));train(exact,o,APRIME,f10,3);world(o,APRIME)
 c['exact_experience_overrides_relational_donor_after_quorum']=exact.prediction(o)==rows10
 # Later remapping donor A must not rewrite the now-exact A' expectation.
 train(exact,o,A,f10,HISTORY);world(o,APRIME);aprime_after=exact.prediction(o);world(o,A);a_after=exact.prediction(o)
 c['remapping_donor_A_does_not_rewrite_exact_APRIME_memory']=aprime_after==rows10 and a_after==rows10
 # Exact ambiguity blocks donor substitution rather than falling back to a related context.
 block=RelationalContextVisualPredictionV1.restore(copy.deepcopy(p.checkpoint()));train(block,o,APRIME,f8,3);train(block,o,APRIME,f10,3);world(o,APRIME)
 c['ambiguous_exact_context_blocks_relational_donor_substitution']=block.prediction(o) is None
 cp=copy.deepcopy(agree.checkpoint());rest=RelationalContextVisualPredictionV1.restore(cp);world(o,AMB)
 c['checkpoint_preserves_relational_donor_generalization']=rest.prediction(o)==rows8
 sig=list(inspect.signature(RelationalContextVisualPredictionV1.observe).parameters);src=inspect.getsource(RelationalContextVisualPredictionV1)
 c['public_api_has_no_context_id_similarity_score_donor_or_expected_heading']=(sig==['self','o','v'] and all(x not in src for x in ('context_id','similarity_score','donor_id','expected_heading','euclidean')))
 c['opaque_state_ids_are_only_compared_by_set_overlap']=('a&b' in src and 'len(a&b)*4' in src and 'abs(' not in src and 'sqrt' not in src)
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_RELATIONAL_CONTEXT_VISUAL_PREDICTION_GREEN','history_cap':HISTORY,'quorum':QUORUM,'checks':c,'failed':fail,'remaining_red':['TRANSITION_STRUCTURE_CONTEXT_GENERALIZATION','ABSTRACT_RELATIONAL_GRAPH_INFERENCE','PREDICTIVE_FEEDBACK_TO_EARLIER_STAGE','BIOLOGICAL_HIPPOCAMPAL_PATTERN_COMPLETION','DIRECT_RELATIONAL_CONTEXT_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_RELATIONAL_CONTEXT_VISUAL_PREDICTION_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
