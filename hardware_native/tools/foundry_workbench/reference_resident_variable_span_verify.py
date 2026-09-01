#!/usr/bin/env python3
"""Hostile reference controls for resident variable-span recruitment."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import time

from reference_resident_channel_sequence_grounding_v1 import (
    GroundingRefuse, admit_channel_sequence_boundary_v1,
)
from reference_resident_variable_span_v1 import (
    ResidentVariableSpanV1, VariableSpanRefuse,
)


def refuses(fn, fragment=""):
    try: fn()
    except (VariableSpanRefuse, GroundingRefuse) as exc: return fragment in str(exc)
    return False


def put(boundary, machine, source, unit, channel=7):
    contact = boundary.seal_sample(
        machine.session, machine.next_sequence, source, channel,
        (unit,), (source,))
    return machine.ingest_sample(contact)


def cue(boundary, machine, source):
    for unit in (0, 0, 0, 1, 2): put(boundary, machine, source, unit)
    return machine.tick()


def main():
    started = time.perf_counter(); checks = {}
    boundary = admit_channel_sequence_boundary_v1()
    machine = ResidentVariableSpanV1(boundary)

    # Raw undelimited scalar chronology; no packet/span/width is supplied.
    for source in (11, 22):
        for _ in range(6):
            for unit in (0, 0, 0, 1, 2, 9): put(boundary, machine, source, unit)
    checks["structural_recurrence_is_not_credit"] = not machine.recipes
    structural_lengths = {row.length for row in machine.span_occurrences
                          if row.source == 11}
    checks["resident_multiscale_lattice"] = set(range(2, 9)).issubset(structural_lengths)

    one_boundary = admit_channel_sequence_boundary_v1()
    one_source = ResidentVariableSpanV1(one_boundary, session=2)
    for _ in range(4):
        for unit in (0, 0, 0, 1, 2, 9): put(one_boundary, one_source, 71, unit)
    for unit in (0, 0, 0, 1, 2): put(one_boundary, one_source, 71, unit)
    checks["copied_one_source_cannot_nominate"] = (
        refuses(one_source.tick, "no_prediction") and not one_source.recipes)

    # Mid-ticket checkpoint and unrelated-channel ingress/neutral expiry.
    tickets = cue(boundary, machine, 33)
    mid = machine.checkpoint()
    replay = ResidentVariableSpanV1.restore(mid, boundary)
    for offset in range(14):
        contact = boundary.seal_sample(
            machine.session, machine.next_sequence, 33, 100 + offset,
            (700 + offset,), (33,))
        machine.ingest_sample(contact); replay.ingest_sample(contact)
    checks["unrelated_channels_reach_neutral_expiry"] = not machine.pending
    checks["deferred_contacts_drain_after_settlement"] = not machine.deferred_sample_roots
    checks["mid_ticket_replay_is_exact"] = machine.checkpoint() == replay.checkpoint()
    checks["deadline_expiry_is_neutral_not_invented_outcome"] = (
        tickets and machine.prediction_witnesses
        and all(row.difference == 0 and row.source == 0 and row.observed_sample == 0
                and row.settlement_trigger_sample > 0
                for row in machine.prediction_witnesses))

    # Context match must beat the resident marginal; one positive source cannot promote.
    cue(boundary, machine, 33); put(boundary, machine, 33, 0)
    checks["marginal_hit_is_negative_difference"] = (
        machine.prediction_witnesses[-1].difference == -1 and not machine.recipes)
    cue(boundary, machine, 11); put(boundary, machine, 11, 9)
    checks["one_positive_outcome_source_cannot_promote"] = not machine.recipes
    cue(boundary, machine, 22); put(boundary, machine, 22, 9)
    checks["two_positive_sources_plus_recurrence_margin_nominate"] = (
        bool(machine.recipes)
        and {row.length for row in machine.recipes.values()} == {2, 3, 4, 5}
        and all(row.retention_margin > 0 and row.prediction_gain > 0
                for row in machine.recipes.values()))
    checks["recipes_retain_no_units"] = all(
        not hasattr(row, "units") and not hasattr(row, "raw")
        and not hasattr(row, "payload") for row in machine.recipes.values())

    for unit in (0, 0, 0, 1, 2): put(boundary, machine, 44, unit)
    unfolded = machine.unfold()
    occurrence = machine._occurrence(unfolded.cue_occurrence)
    checks["unique_resident_span_unfolds"] = unfolded.units == (0, 0, 0, 1, 2)
    checks["exact_per_unit_ancestry"] = (
        len(unfolded.ancestry) == 5
        and all(row.offset == i and row.unit == unfolded.units[i]
                and row.raw_contact_root > 0
                and row.span_occurrence_root == occurrence.identity
                and row.span_recipe_root == unfolded.recipe
                and row.prediction_witness_root > 0
                for i, row in enumerate(unfolded.ancestry)))

    before_withdraw = len(machine.recipes)
    withdrawal = boundary.seal_withdrawal(
        machine.session, machine.next_sequence, 99, 7, 22)
    machine.ingest_withdrawal(withdrawal)
    checks["source_withdrawal_cascades"] = before_withdraw and not machine.recipes
    cue(boundary, machine, 44); put(boundary, machine, 44, 9)
    cue(boundary, machine, 55); put(boundary, machine, 55, 9)
    checks["replacement_cross_source_support_restores"] = bool(machine.recipes)
    remote = boundary.seal_withdrawal(
        machine.session, machine.next_sequence, 99, 7, 999999)
    prior = tuple(machine.recipes.values()); machine.ingest_withdrawal(remote)
    checks["remote_withdrawal_is_sham"] = tuple(machine.recipes.values()) == prior

    blob = machine.checkpoint()
    restored = ResidentVariableSpanV1.restore(blob, boundary)
    checks["complete_checkpoint_replay"] = (
        restored.checkpoint() == blob and restored.recipes == machine.recipes
        and restored.trace == machine.trace)
    corrupt = bytearray(blob); corrupt[-2] ^= 1
    checks["corrupt_checkpoint_refuses"] = refuses(
        lambda: ResidentVariableSpanV1.restore(bytes(corrupt), boundary), "checkpoint")
    checks["wrong_boundary_refuses"] = refuses(
        lambda: ResidentVariableSpanV1.restore(
            blob, admit_channel_sequence_boundary_v1()), "checkpoint_authentication")

    auth_boundary = admit_channel_sequence_boundary_v1()
    auth = ResidentVariableSpanV1(auth_boundary, session=77)
    sealed = auth_boundary.seal_sample(77, 1, 1, 1, (5,))
    before = auth.checkpoint()
    checks["tamper_refuses_atomically"] = (
        refuses(lambda: auth.ingest_sample(replace(sealed, features=(6,))), "authentication")
        and auth.checkpoint() == before)
    low_boundary = admit_channel_sequence_boundary_v1()
    low = ResidentVariableSpanV1(low_boundary, session=78, work_limit=1)
    put(low_boundary, low, 1, 5, channel=1)
    low_before = low.checkpoint()
    checks["resource_refusal_is_atomic"] = (
        refuses(lambda: put(low_boundary, low, 1, 6, channel=1), "resource")
        and low.checkpoint() == low_before)
    exact_boundary = admit_channel_sequence_boundary_v1()
    exact = ResidentVariableSpanV1(exact_boundary, session=79, work_limit=3)
    put(exact_boundary, exact, 1, 5, channel=1)
    put(exact_boundary, exact, 1, 6, channel=1)
    checks["exact_small_resource_bound_succeeds"] = len(exact.span_occurrences) == 1
    checks["literal_chunk_bypass_refuses"] = (
        refuses(lambda: auth_boundary.seal_sample(77, 1, 1, 1, b"x"), "literal")
        and refuses(lambda: auth_boundary.seal_sample(77, 1, 1, 1, (1, 2)), "sample_extent"))
    checks["zero_argument_resident_apis"] = (
        tuple(inspect.signature(ResidentVariableSpanV1.tick).parameters) == ("self",)
        and tuple(inspect.signature(ResidentVariableSpanV1.unfold).parameters) == ("self",))
    forbidden = {"segment", "chunk", "word", "prompt", "answer", "winner",
                 "expected", "reward", "truth", "motor", "emit", "output"}
    public = {name.lower() for name in dir(ResidentVariableSpanV1)
              if not name.startswith("_")}
    checks["no_host_semantic_or_span_selection_api"] = not forbidden.intersection(public)
    elapsed = time.perf_counter() - started
    checks["hard_runtime_bound"] = elapsed < 60
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SystemExit("FOUNDRY_RESIDENT_VARIABLE_SPAN_RED " + ",".join(failed))

    core = Path(__file__).with_name("reference_resident_variable_span_v1.py")
    receipt = {
        "contract": "FOUNDRY_RESIDENT_VARIABLE_SPAN_GREEN",
        "claim": "RESIDENT_BOUNDED_OVERLAPPING_SPAN_NOMINATION_AND_ASSOCIATIVE_NEXT_UNIT_ELIGIBILITY_FROM_AUTHENTICATED_SCALAR_CHRONOLOGY_REFERENCE_ONLY",
        "reference_only": True, "adult_attached": False, "runtime_llm": False,
        "host_prompt": False, "host_numeric_fixture": True,
        "human_language_claim": False, "tokenizer_claim": False,
        "graph_flip": False, "physical_direct_parity": "NOT_RUN/RED",
        "production_ir": "ResidentRecipeIrProgram.vcurrent",
        "translation_status": "UNDEFINED",
        "persistent_recipe_row_payload": "HASHES_LENGTHS_AND_CAUSAL_ROOTS_ONLY/RAW_HISTORY_RETAINED",
        "prediction_difference": "CONTEXT_HIT_MINUS_CHANNEL_MARGINAL_HIT/NOT_CAUSAL_INTERVENTION",
        "checkpoint_authority_custody": "SAME_LIVE_REFERENCE_BOUNDARY_ONLY",
        "dense_reference_scan": True,
        "core_sha256": hashlib.sha256(core.read_bytes()).hexdigest(),
        "elapsed_ms": round(elapsed * 1000, 3), "checks": checks,
        "remaining_red": ["PHYSICAL_SOURCE_INDEPENDENCE", "CROSS_CHANNEL_COMMON_CAUSE",
            "CAUSAL_ACTION_DIFFERENCE", "EARNED_CONDENSATION_AND_MEASURED_WORK_REDUCTION",
            "SAME_SOURCE_REACQUISITION", "WORST_CASE_INSTRUCTION_EXACT_RESOURCE_ACCOUNTING",
            "SPARSE_LARGE_SCALE_INCIDENCE",
            "PRODUCTION_RECIPE_IR_TRANSLATION", "DIRECT_PHYSICAL_PARITY",
            "LEARNED_MULTI_SPAN_COMPOSITION", "CONTINUING_ADULT_LANGUAGE"],
    }
    print("FOUNDRY_RESIDENT_VARIABLE_SPAN_GREEN")
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__": main()
