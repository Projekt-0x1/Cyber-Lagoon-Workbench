#!/usr/bin/env python3
"""Hostile controls for resident parametric three-span Networks."""
from __future__ import annotations

from dataclasses import replace
import hashlib, inspect, json, time
from pathlib import Path

from reference_resident_channel_sequence_grounding_v1 import (
    GroundingRefuse, admit_channel_sequence_boundary_v1,
)
from reference_resident_parametric_span_network_v1 import (
    OP_CALL, OP_SLOT, ParametricSpanNetworkRefuse,
    ResidentParametricSpanNetworkV1,
)
from reference_resident_variable_span_v1 import VariableSpanRefuse


A=(11,12,13,14); B=(21,22,23,24); M=(31,32,33,34)
X=(41,42,43,44); Y=(51,52,53,54)


def refuses(fn, fragment=""):
    try: fn()
    except (ParametricSpanNetworkRefuse,VariableSpanRefuse,GroundingRefuse) as exc:
        return fragment in str(exc)
    return False


def put(b,m,source,unit,channel=7):
    c=b.seal_sample(m.session,m.next_sequence,source,channel,(unit,),(source,))
    return m.ingest_sample(c)


def acquire(b,m,span,target):
    for source in (11,22):
        for _ in range(2):
            for unit in (0,0,0,*span,target): put(b,m,source,unit)
    for source in (11,22):
        for unit in (0,0,0,*span): put(b,m,source,unit)
        phase=m.close(); assert phase.prediction_tickets
        put(b,m,source,target)


def arrange(b,m,source,left,middle,right):
    put(b,m,source,7000+source)
    for span in (left,middle,right):
        for unit in span: put(b,m,source,unit)
    return m.close()


def has_recipe(m,units):
    for occurrence in m._inner.span_occurrences:
        if m._inner._units(occurrence) != units: continue
        if any((r.channel,r.length,r.span_hash)==(
                occurrence.channel,occurrence.length,occurrence.span_hash)
               for r in m._inner.recipes.values()): return True
    return False


def main(fixture_hook=None):
    started=time.perf_counter(); checks={}
    b=admit_channel_sequence_boundary_v1(); m=ResidentParametricSpanNetworkV1(b)
    for index,span in enumerate((A,B,M,X,Y)): acquire(b,m,span,90+index)
    checks["five_base_spans_acquired_in_one_resident"]=all(
        has_recipe(m,span) for span in (A,B,M,X,Y))

    r1=arrange(b,m,31,A,M,X)
    checks["one_corner_cannot_nominate"]=not m.constructors and not r1.rebinds
    r1copy=arrange(b,m,32,A,M,X)
    checks["repeated_exact_pair_is_not_binding_diversity"]=(
        not m.constructors and not r1copy.rebinds)
    r2=arrange(b,m,31,A,M,Y)
    checks["two_distinct_pairs_cannot_nominate"]=not m.constructors and not r2.rebinds
    r3=arrange(b,m,32,B,M,Y)
    checks["three_corners_cross_source_nominate_constructor"]=(
        len(m.constructors)==1 and not r3.rebinds)
    constructor=next(iter(m.constructors.values()))
    checks["constructor_is_numeric_slot_call_slot_only"]=(
        constructor.operations==((OP_SLOT,0),(OP_CALL,constructor.middle_recipe),(OP_SLOT,1))
        and len(constructor.left_bindings)==2 and len(constructor.right_bindings)==2
        and len(constructor.binding_pairs)==3 and len(constructor.binding_pair_sources)==3
        and len(constructor.binding_pair_hashes)==3)
    if fixture_hook is not None:
        return fixture_hook({"boundary":b,"machine":m,"put":put,
            "arrange":arrange,"constructor":constructor,"checks":checks})

    known=arrange(b,m,34,A,M,X)
    checks["known_pair_is_not_false_heldout_rebind"]=not known.rebinds
    constructor=next(iter(m.constructors.values()))
    held=arrange(b,m,35,B,M,X)
    checks["heldout_pair_rebinds_before_assimilation"]=(
        len(held.rebinds)==1 and held.rebinds[0].units==B+M+X)
    out=held.rebinds[0]
    checks["heldout_pair_absent_from_constructor_witnesses"]=(
        m._pair_hash(out.left_binding,out.right_binding)
        not in constructor.binding_pair_hashes)
    heldout_network=m.networks[out.network]
    checks["constructor_evidence_predates_heldout_batch"]=(
        out.constructor_evidence_root==constructor.evidence_root
        and all(m.networks[root].born_sequence < heldout_network.born_sequence
                for root in out.constructor_witness_roots))
    checks["exact_per_unit_network_constructor_ancestry"]=(
        len(out.ancestry)==12 and all(
            row.offset==i and row.unit==out.units[i]
            and row.raw_contact_root>0 and row.span_occurrence_root>0
            and row.span_recipe_root>0 and row.prediction_witness_root>0
            and row.network_root==out.network and row.constructor_root==out.constructor
            and row.constructor_evidence_root==out.constructor_evidence_root
            and out.constructor_evidence_root==constructor.evidence_root
            and row.constructor_witness_roots==constructor.witness_network_roots
            and out.constructor_witness_roots==constructor.witness_network_roots
            and row.network_position==i//4
            and row.slot==(1 if i<4 else 0 if i<8 else 2)
            for i,row in enumerate(out.ancestry)))
    swapped=arrange(b,m,36,X,M,B)
    checks["outer_slot_swap_with_unwitnessed_marginal_refuses"]=not swapped.rebinds
    novel=arrange(b,m,40,M,M,X)
    checks["unwitnessed_slot_marginal_refuses_rebind"]=not novel.rebinds

    blob=m.checkpoint(); replay=ResidentParametricSpanNetworkV1.restore(blob,b)
    checks["complete_checkpoint_replay"]=(
        replay.checkpoint()==blob and replay.constructors==m.constructors
        and replay.rebinds==m.rebinds and replay.trace==m.trace)
    corrupt=bytearray(blob);corrupt[-2]^=1
    checks["corrupt_checkpoint_refuses"]=refuses(
        lambda:ResidentParametricSpanNetworkV1.restore(bytes(corrupt),b),"checkpoint")
    checks["wrong_boundary_refuses"]=refuses(
        lambda:ResidentParametricSpanNetworkV1.restore(
            blob,admit_channel_sequence_boundary_v1()),"checkpoint_authentication")

    before=len(m.constructors)
    for target in (31,32,34,35,36):
        withdrawal=b.seal_withdrawal(m.session,m.next_sequence,99,7,target)
        m.ingest_withdrawal(withdrawal)
    checks["source_withdrawal_cascades_constructor_and_rebinds"]=(
        before==1 and not m.constructors and not m.rebinds)
    arrange(b,m,37,A,M,X)
    arrange(b,m,37,A,M,Y)
    arrange(b,m,38,B,M,Y)
    checks["replacement_support_restores_constructor"]=len(m.constructors)==1
    replacement=arrange(b,m,39,B,M,X)
    checks["replacement_support_allows_fresh_rebind"]=len(replacement.rebinds)==1
    remote=b.seal_withdrawal(m.session,m.next_sequence,99,7,999999)
    prior=tuple(m.constructors.values());m.ingest_withdrawal(remote)
    checks["remote_withdrawal_is_sham"]=tuple(m.constructors.values())==prior

    ab=admit_channel_sequence_boundary_v1(); atomic=ResidentParametricSpanNetworkV1(ab,77)
    sealed=ab.seal_sample(77,1,1,1,(5,)); before_blob=atomic.checkpoint()
    checks["tamper_refuses_atomically"]=(
        refuses(lambda:atomic.ingest_sample(replace(sealed,features=(6,))),"authentication")
        and atomic.checkpoint()==before_blob)
    checks["literal_and_chunk_bypass_refuse"]=(
        refuses(lambda:ab.seal_sample(77,1,1,1,b"x"),"literal")
        and refuses(lambda:ab.seal_sample(77,1,1,1,(1,2)),"sample_extent"))
    checks["zero_argument_close"]=(
        tuple(inspect.signature(ResidentParametricSpanNetworkV1.close).parameters)==("self",))
    forbidden={"compose","parse","template","children","context","expected",
               "winner","answer","prompt","emit","output","grammar","reward"}
    public={name.lower() for name in dir(ResidentParametricSpanNetworkV1)
            if not name.startswith("_")}
    checks["no_host_structure_or_semantic_api"]=not forbidden.intersection(public)
    elapsed=time.perf_counter()-started;checks["hard_runtime_bound"]=elapsed<60
    failed=sorted(k for k,v in checks.items() if not v)
    if failed: raise SystemExit("FOUNDRY_RESIDENT_PARAMETRIC_SPAN_NETWORK_RED "+",".join(failed))
    core=Path(__file__).with_name("reference_resident_parametric_span_network_v1.py")
    receipt={
      "contract":"FOUNDRY_RESIDENT_PARAMETRIC_SPAN_NETWORK_GREEN",
      "claim":"RESIDENT_LEARNED_ORDERED_MULTISPAN_RELATION_NOMINATION_AND_HELDOUT_STRUCTURAL_SLOT_REBINDING_FROM_AUTHENTICATED_SCALAR_CHRONOLOGY_REFERENCE_ONLY",
      "reference_only":True,"adult_attached":False,"runtime_llm":False,
      "host_numeric_fixture":True,"host_prompt":False,"human_language_claim":False,
      "semantic_or_grammar_claim":False,"surface_generation_claim":False,
      "graph_flip":False,"physical_direct_parity":"NOT_RUN/RED",
      "production_ir":"ResidentRecipeIrProgram.vcurrent","translation_status":"UNDEFINED",
      "constructor_payload":"NUMERIC_SLOT_CALL_SLOT_TOPOLOGY_AND_CAUSAL_ROOTS_ONLY",
      "rebind_output":"TRANSIENT_RECONSTRUCTION_OF_CURRENT_CONTACT/NOT_PUBLIC_ACTION",
      "elapsed_ms":round(elapsed*1000,3),"core_sha256":hashlib.sha256(core.read_bytes()).hexdigest(),
      "checks":checks,"remaining_red":["SEMANTIC_GROUNDING","GRAMMAR_AND_LANGUAGE",
        "SURFACE_GENERATION","CAUSAL_INTERVENTION_CREDIT","PHYSICAL_SOURCE_INDEPENDENCE",
        "EARNED_CONDENSATION_AND_WORK_REDUCTION","RECURSIVE_N_PLUS_TWO",
        "WORST_CASE_SPARSE_RESOURCE_ACCOUNTING","SAME_SOURCE_REACQUISITION",
        "PRODUCTION_RECIPE_IR_TRANSLATION","DIRECT_PHYSICAL_PARITY","CONTINUING_ADULT_LANGUAGE"]}
    print("FOUNDRY_RESIDENT_PARAMETRIC_SPAN_NETWORK_GREEN")
    print(json.dumps(receipt,sort_keys=True,indent=2))


if __name__=="__main__":main()
