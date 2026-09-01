#!/usr/bin/env python3
"""Hostile assay for held-out slot-call-slot Network unfolding."""
from __future__ import annotations

from dataclasses import replace
import hashlib, inspect, json, time
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parent))

from reference_resident_parametric_span_network_v1 import (
    ParametricSpanNetworkRefuse, ResidentParametricSpanNetworkV1, _identity,
)
from reference_resident_parametric_span_network_verify import (
    A,B,M,X, main as build_constructor,
)
from reference_resident_partial_network_unfold_v1 import (
    ResidentPartialNetworkTrajectoryV1, unfold_resident_partial_network_v1,
)

STARTED=time.perf_counter()


def refuses(fn,fragment=""):
    try: fn()
    except (ParametricSpanNetworkRefuse,TypeError,ValueError) as exc:
        return fragment in str(exc)
    return False


def assay(state):
    b=state["boundary"];m=state["machine"];put=state["put"]
    constructor=state["constructor"];checks=dict(state["checks"])

    heldout=B+M+X
    def histories(machine):
        groups={}
        for row in machine._inner.samples:
            groups.setdefault((row.source,row.channel),[]).append(row.unit)
        return tuple(tuple(values) for values in groups.values())
    checks["complete_heldout_trajectory_absent_from_contact_history"]=(
        all(not any(tuple(row[i:i+len(heldout)])==heldout
                    for i in range(len(row)-len(heldout)+1)) for row in histories(m)))

    for unit in B+X: put(b,m,41,unit)
    current=tuple(row.unit for row in m._inner.samples
                  if row.source==41 and row.channel==7)
    checks["current_frontier_contains_slots_but_not_call_member"]=(
        current[-len(B+X):]==B+X and M not in tuple(
            current[i:i+len(M)] for i in range(len(current)-len(M)+1)))
    before=m.checkpoint();state_before=(tuple(m.constructors.values()),tuple(m.rebinds),
        tuple(m.events),tuple(m.trace),tuple(m._inner.events),tuple(m._inner.trace))
    out=unfold_resident_partial_network_v1(m)
    checks["learned_network_unfolds_heldout_complete_bytes"]=(
        isinstance(out,ResidentPartialNetworkTrajectoryV1) and out.units==heldout
        and out.constructor_root==constructor.identity
        and (out.left_recipe_root,out.right_recipe_root)
            not in constructor.binding_pairs)
    checks["persistent_constructor_has_no_surface_payload"]=(
        not {"units","bytes","output","expected","semantic","prompt"}.intersection(
            constructor.__dataclass_fields__))
    checks["unfold_is_ephemeral_atomic_and_no_teach"]=(m.checkpoint()==before
        and state_before==(tuple(m.constructors.values()),tuple(m.rebinds),
            tuple(m.events),tuple(m.trace),tuple(m._inner.events),tuple(m._inner.trace)))
    def valid_leaf(row):
        samples={item.identity:item for item in m._inner.samples}
        occurrences={item.identity:item for item in m._inner.span_occurrences}
        witnesses={item.identity:item for item in m._inner.prediction_witnesses}
        sample=samples.get(row.sample_root);occurrence=occurrences.get(row.span_occurrence_root)
        witness=witnesses.get(row.prediction_witness_root)
        return (sample is not None and occurrence is not None and witness is not None
            and (row.unit,row.raw_contact_root,row.contact_sequence,row.source,
                 row.channel,row.provenance)==(sample.unit,sample.contact_root,
                 sample.contact_sequence,sample.source,sample.channel,sample.provenance)
            and row.raw_contact_root==_identity(b"variable-span-contact-v1",(
                m.session,row.contact_sequence,row.source,row.channel,(row.unit,),row.provenance))
            and row.sample_root==_identity(b"variable-span-sample-v1",(
                row.raw_contact_root,row.contact_sequence))
            and occurrence.identity==_identity(b"variable-span-occurrence-v1",(
                occurrence.channel,occurrence.sample_roots))
            and sample.identity in occurrence.sample_roots
            and witness.identity==_identity(b"variable-span-prediction-witness-v1",(
                witness.ticket,witness.observed_sample,witness.difference))
            and witness.difference>0 and witness.source not in m._inner.withdrawn_sources
            and witness.channel==row.channel and witness.source in row.source_roots
            and row.network_position in (0,1,2)
            and row.slot==(0 if row.network_position==0 else 1
                           if row.network_position==2 else -1))
    checks["complete_per_byte_network_ancestry"]=(len(out.ancestry)==len(out.units)
        and tuple(row.offset for row in out.ancestry)==tuple(range(len(out.units)))
        and tuple(row.unit for row in out.ancestry)==out.units
        and all(row.raw_contact_root and row.sample_root and row.span_occurrence_root
            and row.span_recipe_root and row.prediction_witness_root
            and row.constructor_root==out.constructor_root
            and row.constructor_evidence_root==out.constructor_evidence_root
            and row.constructor_witness_roots==constructor.witness_network_roots
            and row.source_roots==out.source_roots and valid_leaf(row)
            for row in out.ancestry))
    checks["complete_constituent_frontier"]=(
        {row.raw_contact_root for row in out.ancestry}.issubset(out.constituent_roots)
        and {row.sample_root for row in out.ancestry}.issubset(out.constituent_roots)
        and {row.span_occurrence_root for row in out.ancestry}.issubset(out.constituent_roots)
        and {row.span_recipe_root for row in out.ancestry}.issubset(out.constituent_roots)
        and {row.prediction_witness_root for row in out.ancestry}.issubset(out.constituent_roots)
        and set(constructor.witness_network_roots).issubset(out.constituent_roots))
    checks["bounded_resident_work"]=(0<out.work_units<=m.work_limit)
    old_limit=m.work_limit;m.work_limit=1;low=m.checkpoint()
    checks["low_budget_refuses_before_unfold_atomically"]=(
        refuses(lambda:unfold_resident_partial_network_v1(m),"resource")
        and m.checkpoint()==low)
    m.work_limit=old_limit
    replay=ResidentParametricSpanNetworkV1.restore(before,b)
    checks["same_complete_checkpoint_exact_replay"]=(
        unfold_resident_partial_network_v1(replay)==out)

    original=next(iter(m.constructors.values()))
    m.constructors[original.identity]=replace(original,evidence_root=original.evidence_root+1)
    checks["constructor_tamper_refuses_without_state_change"]=refuses(
        lambda:unfold_resident_partial_network_v1(m),"constructor")
    m.constructors[original.identity]=original
    removed_root=original.witness_network_roots[0]
    removed_network=m.networks.pop(removed_root)
    stale_before=m.checkpoint()
    checks["stale_constructor_witness_refuses_without_mutation"]=(
        refuses(lambda:unfold_resident_partial_network_v1(m),"constructor_witness")
        and m.checkpoint()==stale_before)
    m.networks[removed_root]=removed_network
    m.networks[removed_root]=replace(removed_network,identity=removed_network.identity+1)
    forged_before=m.checkpoint()
    checks["forged_live_network_identity_refuses_without_mutation"]=(
        refuses(lambda:unfold_resident_partial_network_v1(m),"constructor_witness")
        and m.checkpoint()==forged_before)
    m.networks[removed_root]=removed_network
    witness_root=out.ancestry[0].prediction_witness_root
    witness_index=next(i for i,row in enumerate(m._inner.prediction_witnesses)
                       if row.identity==witness_root)
    original_witness=m._inner.prediction_witnesses[witness_index]
    m._inner.prediction_witnesses[witness_index]=replace(
        original_witness,channel=original_witness.channel+1)
    forged_witness_before=m.checkpoint()
    checks["forged_live_prediction_witness_refuses_without_mutation"]=(
        refuses(lambda:unfold_resident_partial_network_v1(m),"prediction_witness")
        and m.checkpoint()==forged_witness_before)
    m._inner.prediction_witnesses[witness_index]=original_witness

    for unit in A+X: put(b,m,42,unit)
    checks["already_witnessed_pair_does_not_masquerade_as_heldout"]=(
        refuses(lambda:unfold_resident_partial_network_v1(m),"ambiguous"))

    middle_source=next(row.source for row in replay._inner.span_occurrences
        if row.identity==out.middle_occurrence_root)
    withdrawal=b.seal_withdrawal(replay.session,replay.next_sequence,991,7,middle_source)
    replay.ingest_withdrawal(withdrawal)
    checks["middle_source_withdrawal_cascades_and_silences"]=(
        replay.checkpoint()!=before
        and refuses(lambda:unfold_resident_partial_network_v1(replay)))
    checks["zero_argument_resident_selection"]=(tuple(inspect.signature(
        unfold_resident_partial_network_v1).parameters)==("resident",))

    elapsed=time.perf_counter()-STARTED;checks["hard_runtime_bound"]=elapsed<60
    failed=sorted(name for name,value in checks.items() if not value)
    if failed: raise SystemExit("FOUNDRY_REFERENCE_RESIDENT_PARTIAL_NETWORK_UNFOLD_RED "+",".join(failed))
    here=Path(__file__).parent;paths=[here/"reference_resident_partial_network_unfold_v1.py",
        here/"reference_resident_partial_network_unfold_verify.py",
        here/"reference_resident_parametric_span_network_verify.py"]
    receipt={"contract":"FOUNDRY_REFERENCE_RESIDENT_PARTIAL_NETWORK_UNFOLD_GREEN",
      "claim":"LEARNED_SLOT_CALL_SLOT_NETWORK_UNFOLDS_HELDOUT_OUTER_BYTES_REFERENCE_ONLY",
      "reference_only":True,"adult_attached":False,"runtime_llm":False,
      "host_numeric_fixture":True,"host_target_selection":False,
      "fixture_known_expected_bytes":True,"complete_output_seen_in_training":False,
      "human_language_claim":False,"semantic_claim":False,"causal_claim":False,
      "graph_flip":False,"physical_direct_parity":"NOT_RUN/RED",
      "production_ir":"ResidentRecipeIrProgram.vcurrent","translation_status":"UNDEFINED",
      "parity_status":"NOT_RUN/RED","runtime_limit_seconds":60,
      "elapsed_ms":round(elapsed*1000,3),"checks":checks,
      "sha256":{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
      "remaining_red":["NATURAL_LANGUAGE_ACQUISITION","SEMANTIC_WORLD_REFERENCE",
        "AGREEMENT_AND_RECURSION","PUBLIC_BODY_CHANNEL","PRODUCTION_RECIPE_IR_TRANSLATION",
        "DIRECT_PHYSICAL_PARITY"]}
    print("FOUNDRY_REFERENCE_RESIDENT_PARTIAL_NETWORK_UNFOLD_GREEN")
    print(json.dumps(receipt,sort_keys=True,indent=2));return receipt


if __name__=="__main__":build_constructor(fixture_hook=assay)
