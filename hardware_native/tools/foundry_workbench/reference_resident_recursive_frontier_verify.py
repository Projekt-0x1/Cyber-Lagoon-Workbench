#!/usr/bin/env python3
"""Hostile assay for resident alternatives and learned constructor eligibility.

The recursive executor is authored starting machinery. Learned content here is
only generic-constructor eligibility after deterministic same-state yoked
reference outcomes, not physical causality, truth, language, or Adult capacity.
"""
from dataclasses import fields, replace
import hashlib
import json
import time

from reference_resident_recursive_frontier_v1 import (
    CONSTRUCTOR_LEFT, CONSTRUCTOR_RIGHT, FrontierRefuse,
    MAX_CHECKPOINT_BYTES, MAX_WORK, OccurrenceContactV1,
    ResidentRecursiveFrontierV1, admit_reference_boundary_v1,
)


def refuse(call, prefix, machine=None):
    before = machine.checkpoint() if machine is not None else None
    try:
        call()
    except (FrontierRefuse, ValueError, TypeError) as exc:
        if prefix and not str(exc).startswith(prefix):
            raise AssertionError((prefix, str(exc)))
        if machine is not None and machine.checkpoint() != before:
            raise AssertionError("non-atomic refusal")
        return True
    raise AssertionError("expected refusal")


def new_machine(session=1, work_limit=MAX_WORK):
    boundary = admit_reference_boundary_v1()
    return ResidentRecursiveFrontierV1(boundary, session, work_limit), boundary


def add_pair(machine, boundary, source, atoms, provenance_base):
    for offset, atom in enumerate(atoms):
        machine.ingest_occurrence(boundary.seal_occurrence(
            machine.session, machine.next_sequence, source, 7, (atom,),
            (provenance_base + offset,)))
    return machine.tick()


def yoked_outcomes(machine, tickets):
    """Score observable member order against the one resident chronology."""
    positions = {identity: index for index, identity in enumerate(machine.frontier)}
    effects = {ticket.ticket: int(tuple(sorted(
        ticket.candidate.members, key=positions.__getitem__))
        == ticket.candidate.members) for ticket in tickets}
    return {ticket.ticket: (effects[ticket.ticket], max(
        effects[other.ticket] for other in tickets
        if other.ticket != ticket.ticket)) for ticket in tickets}


def settle_yoked(machine, boundary, tickets, reverse_delivery=False):
    outcomes = yoked_outcomes(machine, tickets)
    ordered = tuple(reversed(tickets)) if reverse_delivery else tickets
    differences = {}
    for ticket in ordered:
        actual, baseline = outcomes[ticket.ticket]
        contact = boundary.seal_consequence(
            machine.session, machine.next_sequence, ticket, actual, baseline)
        differences[ticket.ticket] = machine.ingest_consequence(contact)
    return differences, outcomes


def settle_symmetric(machine, boundary, tickets):
    for ticket in tickets:
        machine.ingest_consequence(boundary.seal_consequence(
            machine.session, machine.next_sequence, ticket, 1, 0))


def train_unique(sources=(101, 102), shift=0, work_limit=MAX_WORK,
                 reverse_delivery=False):
    machine, boundary = new_machine(10 + shift, work_limit)
    observations = []
    for index, source in enumerate(sources):
        tickets = add_pair(machine, boundary, source + shift,
            (11 + shift + index * 2, 12 + shift + index * 2),
            10_000 + shift + index * 10)
        differences, outcomes = settle_yoked(
            machine, boundary, tickets, reverse_delivery)
        observations.append((tickets, differences, outcomes))
    return machine, boundary, observations


def add_held_out(machine, boundary, atoms, source=301,
                 provenance_base=30_000):
    for offset, atom in enumerate(atoms):
        machine.ingest_occurrence(boundary.seal_occurrence(
            machine.session, machine.next_sequence, source, 7, (atom,),
            (provenance_base + offset,)))


def structure(machine, output):
    recipe = next(iter(machine.recipes.values()))
    return (recipe.constructor, tuple(row.rank for row in output),
            tuple(len(row.source_roots) for row in output),
            tuple(len(row.provenance) for row in output),
            tuple(row.features[0] == recipe.identity for row in output))


def signed_variant(boundary, machine, candidate_ticket, **changes):
    altered = replace(candidate_ticket, **changes)
    return boundary.seal_consequence(
        machine.session, machine.next_sequence, altered, 1, 0)


def main():
    started = time.perf_counter(); checks = {}

    no_teach, _, _ = train_unique(sources=(101,))
    checks["one_source_cannot_promote_constructor"] = (
        not no_teach.recipes
        and refuse(no_teach.unfold, "frontier:no_constructor", no_teach))
    copied, _, _ = train_unique(sources=(111, 111))
    checks["copied_source_is_not_independent_support"] = not copied.recipes

    prediction, prediction_boundary = new_machine(20)
    pending = add_pair(prediction, prediction_boundary, 121, (31, 32), 20_000)
    checks["prediction_only_creates_no_recipe_or_credit"] = (
        len(pending) == 2 and not prediction.recipes and not prediction.witnesses
        and refuse(prediction.unfold, "frontier:pending_consequence", prediction))

    learned, learned_boundary, observations = train_unique()
    recipe = learned.recipes.get(CONSTRUCTOR_LEFT)
    all_tickets = [ticket for rows, _, _ in observations for ticket in rows]
    checks["resident_proposes_both_alternatives_without_host_winner"] = (
        all({ticket.candidate.constructor for ticket in rows}
            == {CONSTRUCTOR_LEFT, CONSTRUCTOR_RIGHT}
            for rows, _, _ in observations)
        and not any(hasattr(learned, name) for name in (
            "seal_occurrence", "seal_consequence", "seal_withdrawal")))
    checks["same_preaction_yoke_scores_observable_behavior"] = all(
        sorted(outcomes.values()) == [(0, 1), (1, 0)]
        and sorted(differences.values()) == [-1, 1]
        for _, differences, outcomes in observations)
    checks["yoked_eligibility_promotes_one_compact_constructor"] = (
        recipe is not None and recipe.support == 2 and recipe.credit == 2
        and CONSTRUCTOR_RIGHT not in learned.recipes and len(learned.recipes) == 1)
    checks["negative_eligibility_is_not_truth_or_recipe"] = (
        any(row.constructor == CONSTRUCTOR_RIGHT and row.difference < 0
            for row in learned.witnesses)
        and CONSTRUCTOR_RIGHT not in learned.recipes)

    before_unfold_count = len(learned.occurrences)
    add_held_out(learned, learned_boundary, (71, 72, 73, 74))
    learned_blob = learned.checkpoint(); recursive = learned.unfold()
    checks["authored_executor_reuses_learned_constructor_to_depth_three"] = (
        tuple(row.rank for row in recursive) == (1, 2, 3)
        and all(row.features[0] == recipe.identity for row in recursive)
        and len(learned.recipes) == 1)
    checks["recursive_occurrences_are_ephemeral_not_persistent_state"] = (
        len(learned.occurrences) == before_unfold_count + 4
        and all(row not in learned.occurrences for row in recursive))
    checks["recursive_provenance_covers_held_out_chronology"] = (
        tuple(len(row.provenance) for row in recursive) == (2, 3, 4)
        and set(recursive[-1].source_roots) == {301})

    replay_one = ResidentRecursiveFrontierV1.restore(learned_blob, learned_boundary)
    replay_two = ResidentRecursiveFrontierV1.restore(learned_blob, learned_boundary)
    output_one = replay_one.unfold(); output_two = replay_two.unfold()
    checks["event_sourced_checkpoint_recomputes_exact_state"] = (
        output_one == output_two
        and replay_one.recipes == replay_two.recipes == learned.recipes
        and replay_one.witnesses == replay_two.witnesses == learned.witnesses
        and replay_one.trace == replay_two.trace
        and replay_one.checkpoint() == replay_two.checkpoint())
    envelope = json.loads(learned_blob)
    checks["checkpoint_contains_events_not_derived_state"] = (
        set(envelope["body"]) == {
            "schema", "session", "incarnation", "work_limit", "events"}
        and len(learned_blob) <= MAX_CHECKPOINT_BYTES)
    corrupt = bytearray(learned_blob); corrupt[-2] ^= 1
    checks["corrupt_checkpoint_refuses"] = refuse(
        lambda: ResidentRecursiveFrontierV1.restore(
            bytes(corrupt), learned_boundary), "frontier:checkpoint")
    wrong_boundary = admit_reference_boundary_v1()
    checks["wrong_checkpoint_authority_refuses"] = refuse(
        lambda: ResidentRecursiveFrontierV1.restore(
            learned_blob, wrong_boundary), "frontier:checkpoint_authentication")
    envelope["body"]["events"][0]["values"][0][4][0] += 1
    forged = json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()
    checks["valid_shape_unauthenticated_history_mutation_refuses"] = refuse(
        lambda: ResidentRecursiveFrontierV1.restore(
            forged, learned_boundary), "frontier:checkpoint_authentication")

    permuted, permuted_boundary, _ = train_unique(shift=1_000)
    add_held_out(permuted, permuted_boundary, (9_071, 9_072, 9_073, 9_074),
                 source=1_301, provenance_base=90_000)
    permuted_output = permuted.unfold()
    checks["opaque_identity_permutation_preserves_constructor_law"] = (
        structure(learned, recursive) == structure(permuted, permuted_output)
        and recursive[-1].identity != permuted_output[-1].identity)

    swapped, swapped_boundary, _ = train_unique(
        shift=2_000, reverse_delivery=True)
    add_held_out(swapped, swapped_boundary, (2_071, 2_072, 2_073, 2_074),
                 source=2_301, provenance_base=50_000)
    checks["candidate_delivery_order_cannot_swap_learned_mapping"] = (
        structure(learned, recursive) == structure(swapped, swapped.unfold()))

    reversed_machine, reversed_boundary, _ = train_unique(shift=3_000)
    add_held_out(reversed_machine, reversed_boundary,
                 (3_074, 3_073, 3_072, 3_071), source=3_301,
                 provenance_base=60_000)
    reversed_output = reversed_machine.unfold()
    checks["chronology_reversal_changes_ephemeral_occurrence"] = (
        reversed_output[-1].identity != recursive[-1].identity
        and tuple(row.rank for row in reversed_output) == (1, 2, 3))

    ambiguous, ambiguous_boundary = new_machine(50)
    for index, source in enumerate((501, 502)):
        tickets = add_pair(ambiguous, ambiguous_boundary, source,
            (81 + index * 2, 82 + index * 2), 70_000 + index * 10)
        settle_symmetric(ambiguous, ambiguous_boundary, tickets)
    checks["equal_symmetric_eligibility_refuses_atomically"] = (
        set(ambiguous.recipes) == {CONSTRUCTOR_LEFT, CONSTRUCTOR_RIGHT}
        and refuse(ambiguous.unfold, "frontier:ambiguous", ambiguous))

    constrained, constrained_boundary = new_machine(60, work_limit=15)
    for atom in (1, 2):
        constrained.ingest_occurrence(constrained_boundary.seal_occurrence(
            constrained.session, constrained.next_sequence, 601, 7,
            (atom,), (atom,)))
    checks["one_less_resource_refuses_and_rolls_back"] = refuse(
        constrained.tick, "frontier:resource", constrained)

    mid, mid_boundary = new_machine(70)
    mid_tickets = add_pair(mid, mid_boundary, 701, (91, 92), 80_000)
    mid_blob = mid.checkpoint()
    mid_one = ResidentRecursiveFrontierV1.restore(mid_blob, mid_boundary)
    mid_two = ResidentRecursiveFrontierV1.restore(mid_blob, mid_boundary)
    restored_pending = (
        tuple(mid_one.pending) == tuple(ticket.ticket for ticket in mid_tickets)
        and tuple(mid_one.pending) == tuple(mid_two.pending))
    settle_yoked(mid_one, mid_boundary, tuple(mid_one.pending.values()))
    settle_yoked(mid_two, mid_boundary, tuple(mid_two.pending.values()))
    checks["pending_frontier_checkpoint_settlement_replay"] = (
        restored_pending and mid_one.checkpoint() == mid_two.checkpoint())

    wrong = ResidentRecursiveFrontierV1.restore(mid_blob, mid_boundary)
    first = next(iter(wrong.pending.values()))
    variants = (
        ("wrong_ticket", {"ticket": first.ticket + 10_000}, "frontier:ticket"),
        ("wrong_incarnation", {"incarnation": first.incarnation + 1},
         "frontier:consequence_binding"),
        ("wrong_deadline", {"deadline": first.deadline + 1},
         "frontier:consequence_binding"),
        ("wrong_channel", {"channel": first.channel + 1},
         "frontier:consequence_binding"),
        ("wrong_source", {"source": first.source + 1},
         "frontier:consequence_binding"))
    for name, changes, prefix in variants:
        contact = signed_variant(mid_boundary, wrong, first, **changes)
        checks[name + "_consequence_refuses_atomically"] = refuse(
            lambda contact=contact: wrong.ingest_consequence(contact), prefix, wrong)
    wrong_sequence = mid_boundary.seal_consequence(
        wrong.session, wrong.next_sequence + 1, first, 1, 0)
    checks["wrong_sequence_consequence_refuses_atomically"] = refuse(
        lambda: wrong.ingest_consequence(wrong_sequence),
        "frontier:session_sequence", wrong)
    alien = admit_reference_boundary_v1()
    unauthenticated = alien.seal_consequence(
        wrong.session, wrong.next_sequence, first, 1, 0)
    checks["wrong_consequence_authority_refuses_atomically"] = refuse(
        lambda: wrong.ingest_consequence(unauthenticated),
        "frontier:authentication", wrong)

    withdrawn = ResidentRecursiveFrontierV1.restore(learned_blob, learned_boundary)
    withdrawn.ingest_withdrawal(learned_boundary.seal_withdrawal(
        withdrawn.session, withdrawn.next_sequence, 900, 7, 101))
    checks["source_withdrawal_cascades_witnesses_and_recipe"] = (
        CONSTRUCTOR_LEFT not in withdrawn.recipes
        and all(101 not in row.source_roots for row in withdrawn.occurrences)
        and all(101 not in row.source_roots for row in withdrawn.witnesses))
    tickets = add_pair(withdrawn, learned_boundary, 103, (105, 106), 90_000)
    settle_yoked(withdrawn, learned_boundary, tickets)
    checks["new_independent_contact_reacquires_constructor"] = (
        withdrawn.recipes[CONSTRUCTOR_LEFT].support == 2)

    sham = ResidentRecursiveFrontierV1.restore(learned_blob, learned_boundary)
    sham.ingest_withdrawal(learned_boundary.seal_withdrawal(
        sham.session, sham.next_sequence, 900, 7, 999_999))
    checks["remote_source_withdrawal_is_matched_sham"] = (
        sham.recipes[CONSTRUCTOR_LEFT] == learned.recipes[CONSTRUCTOR_LEFT])
    pending_withdrawal = ResidentRecursiveFrontierV1.restore(mid_blob, mid_boundary)
    pending_withdrawal.ingest_withdrawal(mid_boundary.seal_withdrawal(
        pending_withdrawal.session, pending_withdrawal.next_sequence,
        900, 7, 701))
    checks["withdrawal_cascades_pending_candidates"] = (
        not pending_withdrawal.pending and not pending_withdrawal.occurrences)

    literal = OccurrenceContactV1(learned.session, learned.next_sequence,
                                  801, 7, ("surface",), (1,), 0)
    checks["literal_ingress_bypass_refuses_atomically"] = refuse(
        lambda: learned.ingest_occurrence(literal), "frontier:type", learned)
    checks["literal_boundary_bypass_refuses"] = refuse(
        lambda: learned_boundary.seal_occurrence(
            learned.session, learned.next_sequence, 801, 7, b"surface", (1,)),
        "frontier:type")
    forbidden = {"literal", "expected", "semantic", "motor", "route", "output"}
    candidate_fields = {row.name for row in fields(all_tickets[0].candidate)}
    checks["candidate_ir_has_no_surface_or_motor_authority_fields"] = (
        not forbidden.intersection(candidate_fields))
    checks["bounded_cpu_runtime"] = time.perf_counter() - started < 1.0

    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SystemExit("FOUNDRY_RESIDENT_RECURSIVE_FRONTIER_RED "
                         + ",".join(failed))
    receipt = {
        "contract": "FOUNDRY_RESIDENT_RECURSIVE_FRONTIER_GREEN",
        "claim": "RESIDENT_ALTERNATIVE_PROPOSAL_AND_YOKED_CONSTRUCTOR_ELIGIBILITY_REFERENCE_ONLY",
        "reference_only": True, "adult_attached": False, "runtime_llm": False,
        "host_tree_chunk_candidate_or_winner_input": False,
        "world_law_authorship": "AUTHORED_DETERMINISTIC_SAME_STATE_YOKED_REFERENCE_ENVIRONMENT",
        "credit_semantics": "REFERENCE_YOKED_ELIGIBILITY_DIFFERENCE/PHYSICAL_CAUSAL_DIFFERENCE_NOT_PROVEN/NOT_TRUTH",
        "recursive_executor_authorship": "AUTHORED_GENERIC_STARTING_MACHINERY",
        "learned_content": "COMPACT_CONSTRUCTOR_ELIGIBILITY_ONLY",
        "human_language_claim": False, "surface_bytes_emitted": False,
        "graph_flip": False, "physical_direct_parity": "NOT_RUN/RED",
        "production_ir": "ResidentRecipeIrProgram.vcurrent",
        "translation_status": "UNDEFINED",
        "constructor_count": len(learned.recipes),
        "constructor_support": recipe.support,
        "recursive_ranks": [row.rank for row in recursive],
        "checkpoint_bytes": len(learned_blob),
        "checkpoint_sha256": hashlib.sha256(learned_blob).hexdigest(),
        "remaining_red": [
            "RESIDENT_ACQUISITION_OF_SURFACE_AND_GROUNDING_NETWORKS",
            "EXTERNAL_PHYSICAL_CONSEQUENCE_AND_REAFFERENCE",
            "PHYSICAL_CAUSAL_DIFFERENCE_AND_COUNTERFACTUAL",
            "CONSEQUENCE_RECEIPT_TO_PER_MEMBER_ANCESTRY",
            "SOURCE_INDEPENDENCE_BEYOND_AUTHENTICATED_IDENTITY",
            "NATURAL_PENDING_DEADLINE_EXPIRY",
            "COMPLETE_VALIDATION_LINEAR_SCAN_AND_CHECKPOINT_WORK_ACCOUNTING",
            "COMPLETE_PACKETIZATION_AND_BOUNDARY_INVARIANCE",
            "EXTERNALIZED_WORLD_CHRONOLOGY_OBSERVER",
            "PERSISTENT_BOUNDARY_KEY_CUSTODY_ACROSS_PROCESS_RESTART",
            "PRODUCTION_RESIDENT_RECIPE_IR_TRANSLATION", "DIRECT_PHYSICAL_PARITY"],
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print("FOUNDRY_RESIDENT_RECURSIVE_FRONTIER_GREEN reference_only=true adult_attached=false graph_flip=false")
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
