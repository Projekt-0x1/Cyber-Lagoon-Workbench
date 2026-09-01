#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_heading_center_of_flow_verify import radial
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE
from reference_population_v1 import PopulationSpecV1
from reference_rotation_heading_flow_pooling_verify import img,CW
from reference_visual_pathway_stage_occurrences_v1 import VisualPathwayStageOccurrencesV1,COMPONENT_STAGE,VECTOR_STAGE,HEADING_STAGE
WORLD=0xF010

def organism():return ReferenceOrganismV2(PopulationSpecV1(8192,2,3,16,8))
def refuses(fn):
 try:fn();return False
 except ValueError:return True
def main():
 t=time.perf_counter();c={};o=organism();p=VisualPathwayStageOccurrencesV1()
 # No structure: no stage occurrence.
 static=tuple(tuple(0 for _ in range(10)) for _ in range(10))
 c['structureless_input_stops_before_component_stage']=p.process_components(o,static,static) is None and p.sequences[COMPONENT_STAGE]==0
 # Non-radial clockwise motion reaches vectors but cannot produce heading.
 a,b=(img(x) for x in CW);comp=p.process_components(o,a,b);vec=p.process_vectors(o,comp);head=p.process_heading(o,vec)
 c['nonradial_motion_reaches_vector_stage_but_not_heading_stage']=(comp is not None and vec is not None and head is None and p.sequences[COMPONENT_STAGE]==1 and p.sequences[VECTOR_STAGE]==1 and p.sequences[HEADING_STAGE]==0)
 # Radial flow traverses all three stages.
 rp,rc=radial(8,8);comp2=p.process_components(o,rp,rc);vec2=p.process_vectors(o,comp2);head2=p.process_heading(o,vec2)
 c['radial_motion_traverses_component_vector_heading_stages']=(head2 is not None and head2.representation==(2048,2048,1) and head2.predecessor_sequence==vec2.sequence)
 c['stage_occurrences_preserve_causal_predecessor_sequence']=(vec2.predecessor_sequence==comp2.sequence and comp2.predecessor_sequence==0)
 # Stale predecessor refuses after a newer same-stage occurrence supersedes it.
 comp_old=comp2;comp_new=p.process_components(o,rp,rc)
 c['stale_component_predecessor_refuses_vector_stage']=refuses(lambda:p.process_vectors(o,comp_old)) and comp_new.sequence==comp_old.sequence+1
 vec_new=p.process_vectors(o,comp_new);other_comp=p.process_components(o,rp,rc)
 c['stale_vector_predecessor_refuses_heading_stage']=refuses(lambda:p.process_heading(o,vec_new))
 # Real organism history is reflected in stage occurrence tick, not inferred from runtime.
 before=o.tick_count;o.contact(CONTACT_WORLD_STATE,(999,),WORLD,True,True);fresh=p.process_components(o,rp,rc)
 c['stage_tick_reads_resident_history_after_real_contact']=fresh.organism_tick==before+1
 c['provenance_is_positive_but_no_latency_field_exists']=fresh.provenance_ns>0 and not hasattr(fresh,'latency')
 cp=copy.deepcopy(p.checkpoint());rest=VisualPathwayStageOccurrencesV1.restore(cp)
 c['checkpoint_preserves_stage_sequences_but_drops_current_occurrences']=rest.sequences==p.sequences and rest.current=={} and 'current' not in cp
 src=inspect.getsource(VisualPathwayStageOccurrencesV1)
 c['public_stage_api_has_no_mark_ready_delay_latency_or_stage_answer_setter']=all(x not in src for x in ('mark_ready','sleep(','latency_ms','expected_answer'))
 c['heading_stage_consumes_vector_rows_without_rereading_raw_frames']='estimate_rows' in src and 'process_heading' in src
 c['bounded_reference_work']=time.perf_counter()-t<1
 fail=[k for k,v in c.items() if not v];res={'contract':'FOUNDRY_VISUAL_PATHWAY_STAGE_OCCURRENCES_GREEN','stage_sequences':p.sequences,'checks':c,'failed':fail,'remaining_red':['RAPIDLY_VARYING_INTERNAL_LATENCY','BIOLOGICAL_RESPONSE_ONSET_MEASUREMENT','RECURRENT_STAGE_REENTRY_OWNERSHIP','VESTIBULAR_PATHWAY_STAGE_OCCURRENCES','DIRECT_PATHWAY_STAGE_PARITY'],'elapsed_ms':round((time.perf_counter()-t)*1000,3)}
 print(('FOUNDRY_VISUAL_PATHWAY_STAGE_OCCURRENCES_RED '+','.join(fail)) if fail else res['contract']);print(json.dumps(res,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=='__main__':raise SystemExit(main())
