#!/usr/bin/env python3
from __future__ import annotations
import copy,inspect,json,time
from reference_heading_calibration_synchrony_v1 import HeadingCalibrationSynchronyV1
from reference_heading_center_of_flow_verify import radial
from reference_organism_v2 import ReferenceOrganismV2,CONTACT_WORLD_STATE
from reference_pathway_ready_occurrence_v1 import PathwayReadyOccurrenceV1,VISUAL_PATH,VESTIBULAR_PATH
from reference_population_v1 import PopulationSpecV1
from reference_tilt_translation_disambiguator_v1 import TiltTranslationDisambiguatorV1
from reference_vestibular_sensor_ingress_v1 import VestibularSensorIngressV1
WORLD=0xEF10;OS=0xEF11

def organism(): return ReferenceOrganismV2(PopulationSpecV1(8192,2,3,16,8))
def vestib_resolution(sample=(0,2000)):
 s=VestibularSensorIngressV1();s.ingest(OS,1,sample,VestibularSensorIngressV1.sample_digest(sample));return TiltTranslationDisambiguatorV1().resolve(s)
def main():
 t=time.perf_counter();c={};o=organism();owner=PathwayReadyOccurrenceV1();prev,curr=radial(8,8)
 v=owner.process_visual_heading(o,prev,curr);c["valid_visual_transducer_completion_emits_ready_occurrence"]=(v is not None and v.pathway==VISUAL_PATH and v.sequence==1 and v.organism_tick==o.tick_count and v.representation==(2048,2048,1))
 br=owner.process_vestibular_heading(o,vestib_resolution());c["valid_vestibular_transducer_completion_emits_ready_occurrence"]=(br is not None and br.pathway==VESTIBULAR_PATH and br.sequence==1 and br.organism_tick==o.tick_count and isinstance(br.representation,int))
 c["ready_occurrences_have_positive_internal_provenance_without_latency_claim"]=(v.provenance_ns>0 and br.provenance_ns>=v.provenance_ns and not hasattr(v,"latency"))
 c["tilt_explained_no_heading_vestibular_input_emits_no_ready_occurrence"]=(owner.process_vestibular_heading(o,{"translation_feature":0,"translation_magnitude":0,"residual":(0,0)}) is None and owner.sequences[VESTIBULAR_PATH]==1)
 static=((0,0,0),(0,0,0),(0,0,0));c["visual_input_with_no_estimable_heading_emits_no_ready_occurrence"]=(owner.process_visual_heading(o,static,static) is None and owner.sequences[VISUAL_PATH]==1)
 before=o.tick_count;o.contact(CONTACT_WORLD_STATE,(12345,),WORLD,True,True);v2=owner.process_visual_heading(o,prev,curr);c["ready_tick_is_read_from_resident_organism_history"]=(v2.organism_tick==before+1 and v2.sequence==2)
 cp=copy.deepcopy(owner.checkpoint());rest=PathwayReadyOccurrenceV1.restore(cp);c["checkpoint_preserves_pathway_sequences_but_drops_current_ready_occurrence"]=(rest.sequences==owner.sequences and rest.current is None and "current" not in cp)
 src=inspect.getsource(PathwayReadyOccurrenceV1);c["public_api_has_no_mark_ready_tick_timestamp_latency_or_answer_setter"]=("mark_ready" not in src and "latency" not in src and list(inspect.signature(PathwayReadyOccurrenceV1.process_visual_heading).parameters)==["self","o","previous_frame","current_frame"])
 so=organism();ro=PathwayReadyOccurrenceV1();sync=HeadingCalibrationSynchronyV1();pv,cv=radial(4,8);vo=ro.process_visual_heading(so,pv,cv);bo=ro.process_vestibular_heading(so,vestib_resolution((-500,1866)));p1=sync.observe_visual(so,vo.representation[0]);p2=sync.observe_vestibular(so,bo.representation);c["same_tick_ready_occurrences_feed_existing_crossmodal_synchrony"]=(not p1 and p2 and sync.pair_count==1)
 ao=organism();ar=PathwayReadyOccurrenceV1();asy=HeadingCalibrationSynchronyV1();av=ar.process_visual_heading(ao,pv,cv);asy.observe_visual(ao,av.representation[0]);ao.contact(CONTACT_WORLD_STATE,(54321,),WORLD,True,True);ab=ar.process_vestibular_heading(ao,vestib_resolution((-500,1866)));paired=asy.observe_vestibular(ao,ab.representation);c["intervening_world_history_separates_ready_occurrences_and_prevents_pairing"]=(av.organism_tick!=ab.organism_tick and not paired and asy.pair_count==0)
 c["bounded_reference_work"]=time.perf_counter()-t<1
 fail=[k for k,vv in c.items() if not vv];r={"contract":"FOUNDRY_PATHWAY_READY_OCCURRENCE_OWNERSHIP_GREEN","checks":c,"failed":fail,"remaining_red":["RAPIDLY_VARYING_INTERNAL_LATENCY","BIOLOGICAL_RESPONSE_ONSET_MEASUREMENT","PATHWAY_STAGE_LATENCY_OWNERSHIP","PHYSICAL_HARDWARE_TIMESTAMP_PROVENANCE","DIRECT_PATHWAY_READY_PARITY"],"elapsed_ms":round((time.perf_counter()-t)*1000,3)}
 print(("FOUNDRY_PATHWAY_READY_OCCURRENCE_OWNERSHIP_RED "+",".join(fail)) if fail else r["contract"]);print(json.dumps(r,indent=2,sort_keys=True));return 1 if fail else 0
if __name__=="__main__": raise SystemExit(main())
