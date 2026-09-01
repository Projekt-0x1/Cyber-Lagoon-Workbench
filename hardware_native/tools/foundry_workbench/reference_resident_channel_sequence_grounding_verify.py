#!/usr/bin/env python3
"""Hostile controls for resident channel-sequence future grounding."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import time

from reference_resident_channel_sequence_grounding_v1 import (
    COACTIVITY_APERTURE,
    GroundingRefuse,
    ResidentChannelSequenceGroundingV1,
    admit_channel_sequence_boundary_v1,
)


A = (11, 12, 13)
B = (21, 22, 23)
C = (31, 32, 33)


def fresh(work_limit=4096, session=1):
    boundary = admit_channel_sequence_boundary_v1()
    return boundary, ResidentChannelSequenceGroundingV1(
        boundary, session=session, work_limit=work_limit)


def feed(boundary, machine, source, channel, values, provenance=()):
    rows = []
    for value in values:
        contact = boundary.seal_sample(
            machine.session, machine.next_sequence, source, channel,
            (value,), provenance)
        rows.append(machine.ingest_sample(contact))
    return tuple(rows)


def train_hypothesis(boundary, machine, cue=A, target=B,
                     cue_channel=1, target_channel=2, sources=(101, 102)):
    for source in sources:
        feed(boundary, machine, source, cue_channel, cue)
        feed(boundary, machine, source, target_channel, target)


def predict(boundary, machine, source, cue=A, target=B,
            cue_channel=1, target_channel=2):
    feed(boundary, machine, source, cue_channel, cue)
    tickets = machine.tick()
    feed(boundary, machine, source, target_channel, target)
    return tickets


def acquire(boundary, machine, cue=A, target=B,
            cue_channel=1, target_channel=2):
    train_hypothesis(boundary, machine, cue, target, cue_channel, target_channel)
    predict(boundary, machine, 201, cue, target, cue_channel, target_channel)
    predict(boundary, machine, 202, cue, target, cue_channel, target_channel)


def refuses(fn, fragment=""):
    try:
        fn()
    except GroundingRefuse as exc:
        return fragment in str(exc)
    return False


def structure(machine, result):
    return {
        "sequence_recipe_count": len(machine.sequence_recipes),
        "hypothesis_count": len(machine.hypotheses),
        "cross_recipe_count": len(machine.cross_recipes),
        "credits": sorted(row.credit for row in machine.cross_recipes.values()),
        "support": sorted(row.support for row in machine.cross_recipes.values()),
        "unfold_width": len(result.units),
        "ancestry_offsets": [row.offset for row in result.ancestry],
        "ancestry_complete": all(
            row.raw_contact_root > 0
            and row.sequence_recipe_root > 0
            and row.cross_channel_recipe_root > 0
            and row.prediction_witness_root > 0
            and row.recursive_occurrence_root > 0
            for row in result.ancestry),
    }


def main():
    started = time.perf_counter()
    checks = {}

    boundary, machine = fresh()
    train_hypothesis(boundary, machine)
    checks["coactivity_only_is_hypothesis_not_credit"] = (
        len(machine.hypotheses) >= 1
        and not machine.prediction_witnesses
        and not machine.cross_recipes
        and refuses(machine.unfold, "no_cross_channel_recipe"))
    first_tickets = predict(boundary, machine, 201)
    checks["one_future_source_is_insufficient"] = (
        first_tickets and not machine.pending
        and len(machine.prediction_witnesses) == len(first_tickets)
        and not machine.cross_recipes)
    second_tickets = predict(boundary, machine, 202)
    checks["independent_future_differences_recruit"] = (
        second_tickets and not machine.pending
        and len(machine.cross_recipes) == 1
        and next(iter(machine.cross_recipes.values())).support == 2
        and next(iter(machine.cross_recipes.values())).credit == 2)
    feed(boundary, machine, 301, 1, A)
    result = machine.unfold()
    checks["resident_zero_argument_unfold"] = result.units == B
    checks["per_unit_complete_ancestry"] = (
        len(result.ancestry) == len(B)
        and tuple(row.unit for row in result.ancestry) == B
        and all(row.offset == offset for offset, row in enumerate(result.ancestry))
        and structure(machine, result)["ancestry_complete"])
    cited_witness = next(
        row for row in machine.prediction_witnesses
        if row.identity == result.ancestry[0].prediction_witness_root)
    checks["ancestry_witness_matches_target_occurrence"] = (
        cited_witness.observed_occurrence == result.target_occurrence
        and all(row.prediction_witness_root == cited_witness.identity
                for row in result.ancestry))
    checks["recursive_occurrence_is_internal"] = (
        result.recursive_occurrence > 0
        and len(machine._recursive.occurrences) == 1)
    checks["compact_recipes_hold_hashes_not_samples"] = all(
        not hasattr(row, "units") and not hasattr(row, "samples")
        for row in (*machine.sequence_recipes.values(), *machine.cross_recipes.values()))

    no_teach_boundary, no_teach = fresh(session=2)
    for _ in range(3):
        feed(no_teach_boundary, no_teach, 401, 1, A)
        feed(no_teach_boundary, no_teach, 401, 2, B)
    checks["copied_source_is_not_independent_support"] = (
        not no_teach.sequence_recipes and not no_teach.hypotheses
        and not no_teach.cross_recipes)

    mismatch_boundary, mismatch = fresh(session=3)
    train_hypothesis(mismatch_boundary, mismatch)
    feed(mismatch_boundary, mismatch, 501, 1, A)
    mismatch.tick()
    feed(mismatch_boundary, mismatch, 501, 2, C)
    checks["mismatch_is_negative_difference"] = (
        not mismatch.pending and mismatch.prediction_witnesses
        and all(row.difference == -1 for row in mismatch.prediction_witnesses)
        and not mismatch.cross_recipes)

    pending_boundary, pending = fresh(session=4)
    train_hypothesis(pending_boundary, pending)
    feed(pending_boundary, pending, 601, 1, A)
    pending.tick()
    checks["pending_blocks_next_cognition"] = (
        refuses(pending.tick, "pending_prediction")
        and refuses(pending.unfold, "pending_prediction"))
    feed(pending_boundary, pending, 601, 2, B)
    checks["contact_owned_continuation_settles"] = not pending.pending

    channel_boundary, wrong_channel = fresh(session=17)
    train_hypothesis(channel_boundary, wrong_channel)
    feed(channel_boundary, wrong_channel, 602, 1, A)
    wrong_channel.tick()
    feed(channel_boundary, wrong_channel, 602, 3, B)
    checks["wrong_channel_cannot_settle_ticket"] = bool(wrong_channel.pending)
    feed(channel_boundary, wrong_channel, 602, 2, B)
    checks["exact_target_channel_settles_ticket"] = not wrong_channel.pending

    deadline_boundary, deadline = fresh(session=18)
    train_hypothesis(deadline_boundary, deadline)
    feed(deadline_boundary, deadline, 603, 1, A)
    deadline.tick()
    for offset in range(13):
        feed(deadline_boundary, deadline, 603, 8 + offset, (7000 + offset,))
    expired_before_late_target = not deadline.pending
    feed(deadline_boundary, deadline, 603, 2, B)
    checks["late_target_is_negative_not_credit"] = (
        expired_before_late_target and not deadline.pending
        and deadline.prediction_witnesses[-1].difference == -1
        and deadline.prediction_witnesses[-1].observed_occurrence == 0
        and not deadline.cross_recipes)

    ambiguous_boundary, ambiguous = fresh(session=5)
    for source in (701, 702):
        feed(ambiguous_boundary, ambiguous, source, 1, A)
        feed(ambiguous_boundary, ambiguous, source, 2, B)
        feed(ambiguous_boundary, ambiguous, source, 3, C)
    for source in (703, 704):
        feed(ambiguous_boundary, ambiguous, source, 1, A)
        tickets = ambiguous.tick()
        feed(ambiguous_boundary, ambiguous, source, 2, B)
        feed(ambiguous_boundary, ambiguous, source, 3, C)
        checks[f"all_alternatives_open_{source}"] = len(tickets) == 2
    feed(ambiguous_boundary, ambiguous, 705, 1, A)
    checks["equal_resident_alternatives_refuse"] = (
        len(ambiguous.cross_recipes) == 2
        and refuses(ambiguous.unfold, "ambiguous"))

    perm_boundary, permuted = fresh(session=6)
    cue_permuted = (901, 903, 907)
    target_permuted = (1009, 1013, 1019)
    acquire(perm_boundary, permuted, cue_permuted, target_permuted, 17, 29)
    feed(perm_boundary, permuted, 301, 17, cue_permuted)
    perm_result = permuted.unfold()
    checks["opaque_unit_and_channel_permutation_invariance"] = (
        structure(machine, result) == structure(permuted, perm_result))

    reverse_boundary, reverse = fresh(session=7)
    for source in (101, 102):
        feed(reverse_boundary, reverse, source, 2, B)
        feed(reverse_boundary, reverse, source, 1, A)
    feed(reverse_boundary, reverse, 801, 1, A)
    checks["chronology_reversal_diverges"] = refuses(
        reverse.tick, "no_prediction_alternative")

    sham_boundary, sham = fresh(session=8)
    for source in (101, 102):
        feed(sham_boundary, sham, source, 1, A)
        feed(sham_boundary, sham, source, 1, B)
    checks["same_channel_sham_has_no_cross_hypothesis"] = not sham.hypotheses

    distant_boundary, distant = fresh(session=9)
    for source in (101, 102):
        feed(distant_boundary, distant, source, 1, A)
        feed(distant_boundary, distant, source, 9,
             tuple(range(source, source + COACTIVITY_APERTURE + 1)))
        feed(distant_boundary, distant, source, 2, B)
    cue_ids = {row.identity for row in distant.sequence_recipes.values()
               if row.channel == 1}
    target_ids = {row.identity for row in distant.sequence_recipes.values()
                  if row.channel == 2}
    checks["outside_aperture_does_not_associate"] = not any(
        row.cue_recipe in cue_ids and row.target_recipe in target_ids
        for row in distant.hypotheses.values())

    checkpoint_boundary, checkpointed = fresh(session=10)
    train_hypothesis(checkpoint_boundary, checkpointed)
    feed(checkpoint_boundary, checkpointed, 901, 1, A)
    checkpointed.tick()
    mid = checkpointed.checkpoint()
    restored = ResidentChannelSequenceGroundingV1.restore(mid, checkpoint_boundary)
    for value in B:
        contact = checkpoint_boundary.seal_sample(
            checkpointed.session, checkpointed.next_sequence, 901, 2, (value,))
        checkpointed.ingest_sample(contact)
        restored.ingest_sample(contact)
    checks["complete_mid_prediction_checkpoint_replay"] = (
        checkpointed.checkpoint() == restored.checkpoint()
        and checkpointed.trace == restored.trace
        and checkpointed.prediction_witnesses == restored.prediction_witnesses)
    corrupt = bytearray(mid); corrupt[-2] ^= 1
    checks["corrupt_checkpoint_refuses"] = refuses(
        lambda: ResidentChannelSequenceGroundingV1.restore(
            bytes(corrupt), checkpoint_boundary), "checkpoint")
    wrong_boundary = admit_channel_sequence_boundary_v1()
    checks["wrong_authority_checkpoint_refuses"] = refuses(
        lambda: ResidentChannelSequenceGroundingV1.restore(mid, wrong_boundary),
        "checkpoint_authentication")
    forged_envelope = json.loads(mid)
    forged_envelope["body"]["events"][-1]["values"][0][0] += 1000
    forged_envelope["hmac"] = checkpoint_boundary._checkpoint_tag(
        forged_envelope["body"])
    forged = json.dumps(
        forged_envelope, sort_keys=True, separators=(",", ":")).encode()
    checks["valid_hmac_derived_event_mutation_refuses"] = refuses(
        lambda: ResidentChannelSequenceGroundingV1.restore(
            forged, checkpoint_boundary), "event_replay")

    atomic_boundary, atomic = fresh(work_limit=2, session=11)
    feed(atomic_boundary, atomic, 1001, 1, A[:2])
    before_resource = atomic.checkpoint()
    third = atomic_boundary.seal_sample(
        atomic.session, atomic.next_sequence, 1001, 1, (A[2],))
    checks["one_less_resource_refuses_atomically"] = (
        refuses(lambda: atomic.ingest_sample(third), "resource")
        and atomic.checkpoint() == before_resource)
    valid_boundary, valid = fresh(work_limit=4, session=12)
    feed(valid_boundary, valid, 1001, 1, A)
    checks["exact_resource_bound_succeeds"] = len(valid.sequence_occurrences) == 1

    auth_boundary, auth = fresh(session=13)
    original = auth_boundary.seal_sample(13, 1, 1101, 1, (A[0],))
    before_auth = auth.checkpoint()
    checks["tampered_contact_refuses_atomically"] = (
        refuses(lambda: auth.ingest_sample(
            replace(original, features=(999,))), "authentication")
        and auth.checkpoint() == before_auth)
    stale = auth_boundary.seal_sample(13, 2, 1101, 1, (A[0],))
    checks["stale_or_future_sequence_refuses_atomically"] = (
        refuses(lambda: auth.ingest_sample(stale), "session_sequence")
        and auth.checkpoint() == before_auth)
    foreign = auth_boundary.seal_sample(14, 1, 1101, 1, (A[0],))
    checks["wrong_session_refuses_atomically"] = (
        refuses(lambda: auth.ingest_sample(foreign), "session_sequence")
        and auth.checkpoint() == before_auth)
    checks["literal_and_chunk_bypass_refuse"] = (
        refuses(lambda: auth_boundary.seal_sample(13, 1, 1, 1, b"x"), "literal")
        and refuses(lambda: auth_boundary.seal_sample(13, 1, 1, 1, "x"), "literal")
        and refuses(lambda: auth_boundary.seal_sample(13, 1, 1, 1, A), "sample_extent"))
    oversized_provenance = auth_boundary.seal_sample(
        13, 1, 1101, 1, (A[0],), tuple(range(1, 14)))
    checks["provenance_bound_refuses_atomically"] = (
        refuses(lambda: auth.ingest_sample(oversized_provenance), "sample_shape")
        and auth.checkpoint() == before_auth)

    altered_boundary, altered = fresh(session=14)
    altered_target = (41, 43, 47)
    acquire(altered_boundary, altered, target=altered_target)
    feed(altered_boundary, altered, 301, 1, A)
    altered_result = altered.unfold()
    checks["altered_contact_diverges"] = (
        altered_result.units == altered_target
        and altered_result.units != result.units
        and altered_result.identity != result.identity)

    withdraw_boundary, withdraw = fresh(session=15)
    acquire(withdraw_boundary, withdraw)
    cross_before = len(withdraw.cross_recipes)
    removal = withdraw_boundary.seal_withdrawal(
        withdraw.session, withdraw.next_sequence, 1201, 7, 201)
    withdraw.ingest_withdrawal(removal)
    checks["source_withdrawal_cascades_credit"] = (
        cross_before == 1 and not withdraw.cross_recipes
        and all(201 not in row.source_roots for row in withdraw.prediction_witnesses))
    predict(withdraw_boundary, withdraw, 203)
    predict(withdraw_boundary, withdraw, 204)
    checks["ordinary_reacquisition_restores"] = len(withdraw.cross_recipes) == 1
    remote = withdraw_boundary.seal_withdrawal(
        withdraw.session, withdraw.next_sequence, 1202, 7, 999999)
    before_remote = tuple(withdraw.cross_recipes.values())
    withdraw.ingest_withdrawal(remote)
    checks["remote_withdrawal_is_sham"] = tuple(withdraw.cross_recipes.values()) == before_remote

    pending_withdraw_boundary, pending_withdraw = fresh(session=16)
    train_hypothesis(pending_withdraw_boundary, pending_withdraw)
    feed(pending_withdraw_boundary, pending_withdraw, 1301, 1, A)
    pending_withdraw.tick()
    withdrawal = pending_withdraw_boundary.seal_withdrawal(
        pending_withdraw.session, pending_withdraw.next_sequence, 1302, 1, 1301)
    pending_withdraw.ingest_withdrawal(withdrawal)
    checks["withdrawal_cancels_dependent_pending"] = not pending_withdraw.pending

    tick_signature = inspect.signature(ResidentChannelSequenceGroundingV1.tick)
    unfold_signature = inspect.signature(ResidentChannelSequenceGroundingV1.unfold)
    forbidden = {
        "prompt", "expected", "answer", "winner", "candidate", "semantic",
        "lexeme", "word", "sentence", "frame", "motor", "route", "consequence",
        "reward", "truth", "label", "domain", "emit", "output", "byte_action",
    }
    public_names = {name.lower() for name in dir(ResidentChannelSequenceGroundingV1)
                    if not name.startswith("_")}
    checks["zero_input_resident_apis"] = (
        tuple(tick_signature.parameters) == ("self",)
        and tuple(unfold_signature.parameters) == ("self",))
    checks["no_host_semantic_selection_api"] = not forbidden.intersection(public_names)

    elapsed = time.perf_counter() - started
    checks["hard_runtime_bound"] = elapsed < 60.0
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SystemExit("FOUNDRY_RESIDENT_CHANNEL_SEQUENCE_GROUNDING_RED " + ",".join(failed))

    core_path = Path(__file__).with_name(
        "reference_resident_channel_sequence_grounding_v1.py")
    receipt = {
        "contract": "FOUNDRY_RESIDENT_CHANNEL_SEQUENCE_GROUNDING_GREEN",
        "claim": "RESIDENT_CHANNEL_SEQUENCE_PREDICTIVE_ASSOCIATION_AND_COMPOSITOR_ANCESTRY_REFERENCE_ONLY",
        "reference_only": True,
        "adult_attached": False,
        "runtime_llm": False,
        "host_prompt": False,
        "contact_content_authority": "HOST_AUTHORED_NUMERIC_REFERENCE_FIXTURE",
        "human_language_claim": False,
        "graph_flip": False,
        "physical_direct_parity": "NOT_RUN/RED",
        "production_ir": "ResidentRecipeIrProgram.vcurrent",
        "translation_status": "UNDEFINED",
        "checkpoint_authority_custody": "SAME_LIVE_REFERENCE_BOUNDARY_ONLY",
        "resource_accounting_scope": "HEAVY_RECURRENCE_AND_PAIR_SCAN_PLUS_FIXED_STATE_CAPS/NOT_INSTRUCTION_EXACT",
        "observer_channel_interpretation": "ONE_CHANNEL_CALLED_SURFACE_ONLY_OUTSIDE_RUNTIME",
        "sequence_semantics": "RECURRENT_NUMERIC_SEQUENCE_NOT_WORD",
        "association_semantics": "FUTURE_PREDICTIVE_ELIGIBILITY_NOT_SHARED_CAUSE_TRUTH_OR_PHYSICAL_GROUNDING",
        "unfold_authority": "INTERNAL_STATE_OCCURRENCE_NOT_PUBLIC_BYTE_ACTION",
        "recursive_frontier_status": "BASE_OCCURRENCE_INGRESS_ONLY/NO_RECURSIVE_CONSTRUCTOR_CLAIM",
        "runtime_limit_seconds": 60,
        "elapsed_ms": round(elapsed * 1000, 3),
        "core_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
        "checks": checks,
        "remaining_red": [
            "PHYSICAL_CHANNEL_AND_COMMON_CAUSE_GROUNDING",
            "PHYSICAL_CAUSAL_DIFFERENCE_NOT_PROVEN",
            "AUTHENTICATED_PHYSICAL_SOURCE_INDEPENDENCE",
            "PUBLIC_MOTOR_REAFFERENCE",
            "PRODUCTION_RESIDENT_RECIPE_IR_TRANSLATION",
            "DIRECT_PHYSICAL_PARITY",
            "VARIABLE_WIDTH_AND_PACKETIZATION",
            "LARGE_SCALE_DEVELOPMENT",
            "CONTINUING_ADULT_LANGUAGE",
        ],
    }
    print("FOUNDRY_RESIDENT_CHANNEL_SEQUENCE_GROUNDING_GREEN")
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
