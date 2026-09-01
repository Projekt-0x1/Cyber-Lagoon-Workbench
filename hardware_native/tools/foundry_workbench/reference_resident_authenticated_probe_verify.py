#!/usr/bin/env python3
"""Hostile CPU reference assay for resident-originated authenticated probes."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from types import MethodType

sys.path.insert(0, str(Path(__file__).parent))

import reference_resident_authenticated_probe_v1 as probe_module
from reference_resident_authenticated_probe_v1 import (
    ACTION_RECIPES, AuthenticatedProbeRefuse, ResidentAuthenticatedProbeV1,
    admit_body_probe_boundary_v1,
)
from reference_resident_channel_sequence_grounding_v1 import (
    GroundingRefuse, admit_channel_sequence_boundary_v1,
)
from reference_resident_composite_cue_prediction_v1 import CompositeCueRefuse
from reference_resident_composite_cue_prediction_verify import main as trained_fixture


CLAIM = (
    "RESIDENT_SURFACE_MEDIATED_GAP_TO_BOUNDED_OPAQUE_ACTION_SELECTION_"
    "AND_RETURN_CONDITIONED_POLICY_UPDATE_REFERENCE_ONLY"
)
RUN_STARTED = time.perf_counter()


def refuses(call, fragment=""):
    try:
        call()
    except Exception as exc:
        return isinstance(exc, (RuntimeError, TypeError, ValueError)) and fragment in str(exc)
    return False


def assay(ctx):
    boundary, trained = ctx["boundary"], ctx["machine"]
    world, consequence, distractor = ctx["world"], ctx["consequence"], ctx["distractor"]
    checks = dict(ctx["checks"])
    body = admit_body_probe_boundary_v1(boundary)
    routes = (
        (ACTION_RECIPES[0].identity, 7001, 31001, 9, distractor),
        (ACTION_RECIPES[1].identity, 7002, 31002, 9, consequence),
    )
    mapping = body.precommit(trained.session, 77, 11, 3, routes)
    swapped_seed = probe_module._unpack_inner(
        probe_module._pack_inner(trained), boundary)
    tie_seed = probe_module._unpack_inner(
        probe_module._pack_inner(trained), boundary)
    checks["mapping_order_is_not_policy"] = (
        body.precommit(trained.session, 77, 11, 3, reversed(routes)) == mapping)
    def put(machine, source, channel, unit):
        contact = boundary.seal_sample(
            machine.session, machine.next_sequence, source, channel, (unit,),
            (source, channel))
        return machine.ingest_sample(contact)

    def open_episode(machine, source):
        for unit in world:
            put(machine, source, 8, unit)
        result = machine.tick()
        assert result.command is not None
        return result.command

    def settle(machine, command):
        active_mapping = machine.mapping
        apply = machine.dispatch()
        returns = []
        evidence = None
        while machine.pending is not None:
            contact = body.seal_return(active_mapping, apply, machine.next_sequence)
            returns.append(contact)
            evidence = machine.ingest_return(contact)
        assert evidence is not None
        return apply, tuple(returns), evidence

    # A blank resident on the same authenticated boundary cannot invent a gap.
    idle_resident = ResidentAuthenticatedProbeV1(boundary, body, mapping)
    idle_before = idle_resident.checkpoint()
    idle_refused = refuses(idle_resident.tick, "quiescent")
    checks["no_resident_gap_no_command"] = (
        idle_refused and idle_resident.pending is None
        and idle_resident.checkpoint() == idle_before)
    resident = ResidentAuthenticatedProbeV1(
        boundary, body, mapping, initial_inner=trained)
    first = open_episode(resident, 32001)
    first_apply, first_returns, first_evidence = settle(resident, first)
    second = open_episode(resident, 32002)
    second_apply, second_returns, second_evidence = settle(resident, second)
    checks["mismatch_changes_next_structural_choice"] = (
        first.action_recipe == ACTION_RECIPES[0].identity
        and first_evidence.difference == -1
        and second.action_recipe == ACTION_RECIPES[1].identity
        and second_evidence.difference == 1)
    checks["body_mapping_is_private_from_resident_choice"] = (
        first.mapping_root == second.mapping_root == mapping.identity
        and mapping.route_root != mapping.identity)
    checks["exact_apply_and_return_authentication"] = (
        first_apply.command_hash == first.identity
        and second_apply.command_hash == second.identity
        and all(row.apply_receipt == first_apply.identity for row in first_returns)
        and all(row.apply_receipt == second_apply.identity for row in second_returns))

    witness = next(row for row in resident._inner.witnesses
                   if row.identity == second_evidence.structural_witness)
    scalar_samples = resident._inner._inner._inner.samples
    returned_contact_roots = tuple(next(
        row.contact_root for row in scalar_samples
        if row.contact_sequence == contact.occurrence.sequence)
        for contact in second_returns)
    checks["per_unit_structural_ancestry"] = (
        len(witness.ancestry) == len(consequence) == len(second_returns)
        and tuple(row.unit for row in witness.ancestry) == consequence
        and tuple(row.raw_contact_root for row in witness.ancestry)
        == returned_contact_roots
        and second_evidence.return_roots == tuple(row.identity for row in second_returns)
        and all(row.occurrence.source == 31002 and row.occurrence.channel == 9
                and row.occurrence.provenance[:2] == (77, 11)
                and row.occurrence.provenance[-1] == second_apply.identity
                for row in second_returns)
        and 31002 in second_evidence.source_roots)

    # Same complete state and same input replay identically; an altered source diverges.
    stable = resident.checkpoint()
    twin = ResidentAuthenticatedProbeV1.restore(stable, boundary, body)
    checks["complete_checkpoint_exact_replay"] = twin.checkpoint() == stable
    normal = open_episode(resident, 33001)
    def reversed_eligible(self, ticket):
        return tuple(reversed(ResidentAuthenticatedProbeV1._eligible(self, ticket)))
    twin._eligible = MethodType(reversed_eligible, twin)  # enumeration-order lesion
    permuted = open_episode(twin, 33001)
    checks["opaque_action_permutation_invariant"] = normal == permuted
    altered_twin = ResidentAuthenticatedProbeV1.restore(stable, boundary, body)
    altered = open_episode(altered_twin, 33002)
    checks["same_state_same_input_exact_altered_input_diverges"] = (
        normal == permuted and altered.identity != normal.identity)

    # A pending command accepts only its exact current authenticated application.
    third = normal
    evidence_before = tuple(resident.evidence)
    resource_before = resident.resource
    third_apply = resident.dispatch()
    before_wrong = resident.checkpoint()
    pending_twin = ResidentAuthenticatedProbeV1.restore(before_wrong, boundary, body)
    checks["pending_checkpoint_exact_replay"] = pending_twin.checkpoint() == before_wrong
    checks["host_cannot_supply_apply_receipt"] = not hasattr(
        resident, "ingest_apply") and not hasattr(body, "seal_accept") \
        and not hasattr(body, "_seal_accept") \
        and not hasattr(probe_module, "_BODY_DISPATCH")
    checks["yoked_stale_return_refuses_atomically"] = (
        refuses(lambda: resident.ingest_return(first_returns[0]), "return_authentication")
        and resident.checkpoint() == before_wrong)
    forged_device = replace(
        first_returns[0], apply_receipt=third_apply.identity,
        command_hash=third.identity, device=first_returns[0].device + 1)
    checks["wrong_device_return_refuses_atomically"] = (
        refuses(lambda: resident.ingest_return(forged_device), "return_authentication")
        and resident.checkpoint() == before_wrong)
    for _ in range(probe_module.PROBE_HORIZON):
        resident.tick()
    checks["withheld_return_neutral_expiry"] = (
        resident.pending is None and tuple(resident.evidence) == evidence_before
        and resident.resource == resource_before - third.resource_cost)

    # Returned units, not the opaque action identity, determine relative evidence.
    swapped_body = admit_body_probe_boundary_v1(boundary)
    swapped_routes = (
        (ACTION_RECIPES[0].identity, 7001, 35001, 9, consequence),
        (ACTION_RECIPES[1].identity, 7002, 35002, 9, distractor),
    )
    swapped_mapping = swapped_body.precommit(trained.session, 88, 12, 4, swapped_routes)
    swapped = ResidentAuthenticatedProbeV1(
        boundary, swapped_body, swapped_mapping, initial_inner=swapped_seed)
    swapped_first = open_episode(swapped, 35003)
    apply = swapped.dispatch()
    swapped_evidence = None
    while swapped.pending is not None:
        contact = swapped_body.seal_return(swapped_mapping, apply, swapped.next_sequence)
        swapped_evidence = swapped.ingest_return(contact)
    checks["mapping_swap_follows_authenticated_raw_return"] = (
        swapped_first.action_recipe == first.action_recipe
        and swapped_evidence is not None and swapped_evidence.difference == 1)

    # Settled evidence is withdrawn transitively and can be reacquired by new contact.
    withdrawal = boundary.seal_withdrawal(
        resident.session, resident.next_sequence, 39001, 79, 31002)
    resident.ingest_withdrawal(withdrawal)
    checks["source_withdrawal_cascades_evidence"] = (
        second_evidence.identity not in {row.identity for row in resident.evidence})
    renewed_routes = (
        (ACTION_RECIPES[0].identity, 7001, 31003, 9, consequence),
        (ACTION_RECIPES[1].identity, 7002, 31004, 9, distractor),
    )
    renewed_mapping = body.precommit(trained.session, 77, 12, 3, renewed_routes)
    resident.adopt_mapping(renewed_mapping)
    reacquire = open_episode(resident, 39002)
    apply, _returns, evidence = settle(resident, reacquire)
    checks["ordinary_contact_reacquires_return_evidence"] = (
        reacquire.action_recipe == ACTION_RECIPES[0].identity
        and apply.command_hash == reacquire.identity and evidence.difference == 1
        and all(row.mapping_root == renewed_mapping.identity
                for row in resident.evidence))

    # A structural tie in an independently constructed lesion resident refuses;
    # IDs never arbitrate it and the continuing runtime policy is untouched.
    tie_resident = ResidentAuthenticatedProbeV1(
        boundary, body, mapping, initial_inner=tie_seed)
    for unit in world:
        put(tie_resident, 39003, 8, unit)
    before_tie = tie_resident.checkpoint()
    def tied_eligible(self, ticket):
        return tuple((0, action) for _score, action in
                     ResidentAuthenticatedProbeV1._eligible(self, ticket))
    tie_resident._eligible = MethodType(tied_eligible, tie_resident)
    tie_resident._policy_uniqueness = MethodType(
        lambda self: (False, 1), tie_resident)
    tie_refused = refuses(tie_resident.tick, "action_ambiguous")
    tie_atomic = tie_resident.checkpoint() == before_tie
    checks["score_tie_refuses_atomically"] = tie_refused and tie_atomic

    checks["corrupt_checkpoint_refuses"] = refuses(
        lambda: ResidentAuthenticatedProbeV1.restore(stable[:-1] + b"0", boundary, body),
        "checkpoint")
    foreign_body = admit_body_probe_boundary_v1(boundary)
    checks["wrong_body_boundary_refuses_checkpoint"] = refuses(
        lambda: ResidentAuthenticatedProbeV1.restore(stable, boundary, foreign_body),
        "checkpoint_authentication")
    forged = json.loads(stable)
    forged["body"]["inner_layers"][0][0] = [99]
    forged_blob = json.dumps(forged, sort_keys=True, separators=(",", ":")).encode()
    body_before_forgery = body._snapshot_state()
    checks["authenticated_malformed_checkpoint_refuses_atomically"] = (
        refuses(lambda: ResidentAuthenticatedProbeV1.restore(
            forged_blob, boundary, body), "checkpoint_authentication")
        and body._snapshot_state() == body_before_forgery)
    checks["no_arbitrary_checkpoint_signing_oracle"] = (
        not hasattr(body, "_seal_checkpoint")
        and refuses(lambda: probe_module._state_decode([99], [0]),
                    "inner_snapshot_shape"))

    public_parameters = set()
    for name in ("__init__", "ingest_sample", "tick", "dispatch",
                 "ingest_return", "ingest_withdrawal", "adopt_mapping", "restore"):
        public_parameters.update(inspect.signature(
            getattr(ResidentAuthenticatedProbeV1, name)).parameters)
    forbidden = {"prompt", "expected", "answer", "effect", "outcome", "candidate",
                 "goal", "target", "surface", "scene", "bind", "emit"}
    checks["no_host_current_thought_or_literal_bypass_api"] = not (
        public_parameters & forbidden)
    checks["no_literal_surface_bytes_in_runtime_state"] = all(
        not isinstance(value, (str, bytes, bytearray, memoryview))
        for row in (first, second, first_evidence, second_evidence)
        for value in row.__dict__.values())
    checks["no_direct_world_consequence_recipe"] = ctx["checks"][
        "no_direct_w_c_hypothesis_or_recipe"]

    elapsed = time.perf_counter() - RUN_STARTED
    checks["bounded_cpu_reference_runtime"] = elapsed < 60.0
    if not all(checks.values()):
        raise AssertionError({key: value for key, value in checks.items() if not value})

    receipt = {
        "claim": CLAIM,
        "reference_only": True,
        "graph_flip": False,
        "adult_status": "RED",
        "language_status": "RED",
        "semantic_status": "RED",
        "physical_causal_status": "RED",
        "production_ir": "ResidentRecipeIrProgram.vcurrent",
        "translation_status": "UNDEFINED",
        "parity_status": "NOT_RUN",
        "physical_direct_parity": "NOT_RUN/RED",
        "runtime_llm": False,
        "host_current_thought": False,
        "executor": "CPython graph-neutral deterministic reference",
        "checks": checks,
        "check_count": len(checks),
        "mapping_root": mapping.identity,
        "private_route_root": mapping.route_root,
        "first_action": first.action_recipe,
        "second_action": second.action_recipe,
        "first_difference": first_evidence.difference,
        "second_difference": second_evidence.difference,
        "first_apply": first_apply.identity,
        "second_apply": second_apply.identity,
        "first_return_roots": first_evidence.return_roots,
        "second_return_roots": second_evidence.return_roots,
        "second_ancestry": [row.__dict__ for row in witness.ancestry],
        "candidate_set_root": second.candidate_set_root,
        "selection_state_root": second.selection_state_root,
        "resource_remaining": resident.resource,
        "resident_call_work_limit": resident.work_limit,
        "last_resident_call_work": resident.work,
        "transaction_state_node_bound": probe_module.MAX_STATE_NODES,
        "checkpoint_byte_bound": probe_module.MAX_CHECKPOINT_BYTES,
        "event_bound": probe_module.MAX_EVENTS,
        "evidence_bound": probe_module.MAX_EVIDENCE,
        "checkpoint_sha256": hashlib.sha256(stable).hexdigest(),
        "elapsed_seconds": round(elapsed, 6),
    }
    print("FOUNDRY_REFERENCE_RESIDENT_AUTHENTICATED_PROBE_GREEN")
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return receipt


def main():
    return trained_fixture(assay)


if __name__ == "__main__":
    main()
