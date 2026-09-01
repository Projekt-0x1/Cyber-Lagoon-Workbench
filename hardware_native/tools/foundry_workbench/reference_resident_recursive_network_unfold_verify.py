#!/usr/bin/env python3
"""Hostile depth-three assay for recursive reuse of learned surface Networks."""
from __future__ import annotations

import hashlib,inspect,json,time
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parent))

from reference_resident_channel_sequence_grounding_v1 import admit_channel_sequence_boundary_v1
from reference_resident_parametric_span_network_v1 import (
    ParametricSpanNetworkRefuse,ResidentParametricSpanNetworkV1,_identity,
)
from reference_resident_parametric_span_network_verify import arrange,put
from reference_resident_recursive_network_unfold_v1 import (
    unfold_resident_recursive_network_v1,
)
def b(value): return tuple(value.encode("utf-8"))

ANN,EVE,CATS,DOGS=map(b,("Ann","Eve","Cats","Dogs"))
SEES,SEE,BOB,DAN=map(b,(" sees "," see ","Bob","Dan"))
CONNECT=b(" & ")


def refuses(fn,fragment=""):
    try: fn()
    except (ParametricSpanNetworkRefuse,TypeError,ValueError) as exc:
        return fragment in str(exc)
    return False


def feed(machine,boundary,source,*spans):
    for span in spans:
        for unit in span:
            put(boundary,machine,source,unit)


def acquire(machine,boundary,span,target,sources):
    for source in sources:
        for _ in range(2):
            for unit in (0,0,0,*span,target): put(boundary,machine,source,unit)
    for source in sources:
        for unit in (0,0,0,*span): put(boundary,machine,source,unit)
        phase=machine.close();assert phase.prediction_tickets
        put(boundary,machine,source,target)


def history_contains(machine,needle):
    groups={}
    for row in machine._inner.samples:
        groups.setdefault((row.source,row.channel),[]).append(row.unit)
    return any(any(tuple(units[i:i+len(needle)])==needle
                   for i in range(len(units)-len(needle)+1))
               for units in groups.values())


def main():
    started=time.perf_counter();checks={}
    boundary=admit_channel_sequence_boundary_v1()
    machine=ResidentParametricSpanNetworkV1(boundary)
    spans=(ANN,EVE,CATS,DOGS,SEES,SEE,BOB,DAN,CONNECT)
    for index,span in enumerate(spans):
        acquire(machine,boundary,span,239+index,(201+2*index,202+2*index))
    for source,left,middle,right in (
        (1001,ANN,SEES,BOB),(1001,ANN,SEES,DAN),(1002,EVE,SEES,DAN),
        (1101,CATS,SEE,BOB),(1101,CATS,SEE,DAN),(1102,DOGS,SEE,DAN)):
        arrange(boundary,machine,source,left,middle,right)
    expected=b("Eve sees Bob & Dogs see Bob & Eve sees Bob")
    checks["depth_three_complete_surface_absent_from_contact_history"]=(
        not history_contains(machine,expected))
    before_samples=len(machine._inner.samples)
    feed(machine,boundary,1301,EVE,BOB,CONNECT,DOGS,BOB,CONNECT,EVE,BOB)
    peak_samples=len(machine._inner.samples)
    current=tuple(row.unit for row in machine._inner.samples if row.source==1301)
    checks["current_contact_contains_slots_and_connectives_but_no_inflections"]=(
        SEES not in tuple(current[i:i+len(SEES)] for i in range(len(current)-len(SEES)+1))
        and SEE not in tuple(current[i:i+len(SEE)] for i in range(len(current)-len(SEE)+1)))
    frozen=machine.checkpoint();out=unfold_resident_recursive_network_v1(machine)
    checks["authored_fold_reuses_learned_networks_to_depth_three"]=(
        out.rank==3 and len(out.clause_roots)==3 and len(out.connective_recipe_roots)==2
        and len(out.closure_roots)==2 and out.units==expected)
    checks["depth_two_and_three_closures_are_distinct_ephemeral_roots"]=(
        out.closure_roots[0]!=out.closure_roots[1]
        and all(root not in machine.networks for root in out.closure_roots))
    sample_by_root={row.identity:row for row in machine._inner.samples}
    occurrence_by_root={row.identity:row for row in machine._inner.span_occurrences}
    witness_by_root={row.identity:row for row in machine._inner.prediction_witnesses}
    def valid_leaf(row):
        sample=sample_by_root.get(row.sample_root)
        occurrence=occurrence_by_root.get(row.span_occurrence_root)
        witness=witness_by_root.get(row.prediction_witness_root)
        return (sample is not None and occurrence is not None and witness is not None
            and (row.unit,row.raw_contact_root,row.contact_sequence,row.source,row.channel,
                 row.provenance)==(sample.unit,sample.contact_root,sample.contact_sequence,
                                   sample.source,sample.channel,sample.provenance)
            and row.raw_contact_root==_identity(b"variable-span-contact-v1",(
                machine.session,row.contact_sequence,row.source,row.channel,
                (row.unit,),row.provenance))
            and row.sample_root==_identity(b"variable-span-sample-v1",(
                row.raw_contact_root,row.contact_sequence))
            and occurrence.identity==_identity(b"variable-span-occurrence-v1",(
                occurrence.channel,occurrence.sample_roots))
            and sample.identity in occurrence.sample_roots
            and witness.identity==_identity(b"variable-span-prediction-witness-v1",(
                witness.ticket,witness.observed_sample,witness.difference))
            and witness.difference>0 and witness.channel==row.channel
            and witness.source in row.source_roots)
    rebuilt=[];current_root=out.clause_roots[0]
    for index,other in enumerate(out.clause_roots[1:]):
        current_root=_identity(b"resident-recursive-network-closure-v1",(
            current_root,out.connective_recipe_roots[index],other,index+2))
        rebuilt.append(current_root)
    checks["exact_per_byte_recursive_ancestry"]=(
        len(out.ancestry)==len(out.units)
        and tuple(row.offset for row in out.ancestry)==tuple(range(len(out.units)))
        and tuple(row.unit for row in out.ancestry)==out.units
        and tuple(rebuilt)==out.closure_roots
        and all(row.raw_contact_root and row.sample_root and row.span_occurrence_root
                and row.span_recipe_root and row.prediction_witness_root
                and row.leaf_rank==1 and row.closure_rank==3
                and row.closure_roots==out.closure_roots[(
                    row.clause_index if row.role==2 else max(0,row.clause_index-1)):]
                and (row.clause_root==out.clause_roots[row.clause_index]
                     if row.role!=2 else row.clause_root==0)
                and (bool(row.constructor_witness_roots)
                     if row.role!=2 else not row.constructor_witness_roots)
                and row.source_roots==out.source_roots and valid_leaf(row)
                for row in out.ancestry)
        and {row.role for row in out.ancestry}=={-1,0,1,2})
    checks["recursive_unfold_is_ephemeral_no_teach_and_exact_reexecution"]=(
        machine.checkpoint()==frozen
        and unfold_resident_recursive_network_v1(machine)==out)

    original_limit=machine.work_limit;machine.work_limit=1;low=machine.checkpoint()
    checks["low_resource_refuses_atomically"]=(
        refuses(lambda:unfold_resident_recursive_network_v1(machine),"resource")
        and machine.checkpoint()==low)
    machine.work_limit=original_limit
    constructor_root=out.ancestry[0].constructor_root
    constructor=machine.constructors.pop(constructor_root)
    lesion=machine.checkpoint()
    lesioned_constructors=tuple(machine.constructors.items())
    checks["one_required_clause_network_lesion_refuses_atomically"]=(
        refuses(lambda:unfold_resident_recursive_network_v1(machine))
        and machine.checkpoint()==lesion
        and tuple(machine.constructors.items())==lesioned_constructors)
    machine.constructors[constructor_root]=constructor

    connect_recipe=out.connective_recipe_roots[0]
    connect=next(row for row in machine._inner.recipes.values()
                 if row.identity==connect_recipe)
    withdrawal_source=connect.source_roots[0]
    withdrawal=boundary.seal_withdrawal(machine.session,machine.next_sequence,
                                        1999,7,withdrawal_source)
    machine.ingest_withdrawal(withdrawal)
    checks["connective_source_withdrawal_cascades_and_silences"]=(
        connect_recipe not in machine._inner.recipes
        and refuses(lambda:unfold_resident_recursive_network_v1(machine)))
    checks["bounded_work_state_and_runtime"]=(
        before_samples<peak_samples<=576
        and out.work_units<=original_limit and time.perf_counter()-started<60)
    checks["zero_argument_no_surface_or_tree_input_api"]=(
        tuple(inspect.signature(unfold_resident_recursive_network_v1).parameters)==("resident",))

    failed=sorted(name for name,value in checks.items() if not value)
    if failed: raise SystemExit("FOUNDRY_REFERENCE_RESIDENT_RECURSIVE_NETWORK_UNFOLD_RED "+",".join(failed))
    here=Path(__file__).parent;elapsed=time.perf_counter()-started
    paths=[here/"reference_resident_recursive_network_unfold_v1.py",
           here/"reference_resident_recursive_network_unfold_verify.py"]
    receipt={"contract":"FOUNDRY_REFERENCE_RESIDENT_RECURSIVE_NETWORK_UNFOLD_GREEN",
      "claim":"AUTHORED_GENERIC_RECURSIVE_FOLD_REUSES_CONTACT_LEARNED_SURFACE_NETWORKS_REFERENCE_ONLY",
      "reference_only":True,"adult_attached":False,"runtime_llm":False,
      "authored_starting_machinery":"GENERIC_SEQUENCE_FOLD",
      "learned_content":"BYTE_SPAN_RECIPES_AND_SLOT_CALL_SLOT_NETWORKS",
      "host_fixture_current_slot_and_connective_selection":True,
      "host_output_inflection_or_closure_selection":False,
      "complete_output_seen_in_training":False,"human_language_claim":False,
      "acquisition_source_partition":"UNIQUE_PAIR_PER_LEARNED_SPAN",
      "semantic_or_discourse_claim":False,"causal_claim":False,"graph_flip":False,
      "physical_direct_parity":"NOT_RUN/RED","production_ir":"ResidentRecipeIrProgram.vcurrent",
      "translation_status":"UNDEFINED","parity_status":"NOT_RUN/RED",
      "peak_resident_samples":peak_samples,"runtime_limit_seconds":60,
      "elapsed_ms":round(elapsed*1000,3),"checks":checks,
      "sha256":{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
      "remaining_red":["LEARNED_HIGHER_RANK_CONSTRUCTOR","GROUNDED_DISCOURSE",
        "MULTILINGUAL_RECURSION","CONSEQUENCE_CONDITIONED_PARTNER_STYLE",
        "SOURCE_REACQUISITION","PUBLIC_BODY_CHANNEL","PRODUCTION_RECIPE_IR_TRANSLATION",
        "DIRECT_PHYSICAL_PARITY"]}
    print("FOUNDRY_REFERENCE_RESIDENT_RECURSIVE_NETWORK_UNFOLD_GREEN")
    print(json.dumps(receipt,sort_keys=True,indent=2))


if __name__=="__main__":main()
