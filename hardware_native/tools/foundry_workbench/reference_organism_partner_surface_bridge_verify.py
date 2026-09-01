#!/usr/bin/env python3
"""Hostile fast assay for the byte-blind organism selection bridge."""
from __future__ import annotations

import ast
import copy
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import time

from reference_multilingual_contextual_reference_verify import (
    HELDOUT_INDEX, PARTNER_ECOLOGIES, acquire_compact_ecologies, train,
    two_turn)
from reference_productive_register_discourse_verify import settle
from reference_organism_v2 import ActionV2
from reference_organism_partner_surface_bridge_v1 import (
    OrganismPartnerSurfaceBridgeRefuse,
    ReferenceOrganismSurfaceSelectionV1,
    capture_reference_organism_surface_selection_v1)


def refuses(call, fragment):
    try:
        call()
    except (OrganismPartnerSurfaceBridgeRefuse, ValueError) as exc:
        return fragment in str(exc)
    return False


def main():
    started = time.perf_counter()
    checks = {}
    organism, _atoms, _full, _unacquired = train()
    acquire_compact_ecologies(organism)
    authorities = []
    actions = []
    for index, (partner, _render, words) in enumerate(PARTNER_ECOLOGIES):
        heldout = tuple(words[position] for position in HELDOUT_INDEX)
        _first_scene, first, _second_scene, second = two_turn(
            organism, partner, heldout, 0x8100000 + index * 0x100000)
        assert isinstance(first, ActionV2) and isinstance(second, ActionV2)
        authorities.append(capture_reference_organism_surface_selection_v1(
            organism, second))
        actions.append(second)
        if index + 1 < len(PARTNER_ECOLOGIES):
            settle(organism, second, partner, 1)
    checks["four_language_ecologies_yield_numeric_authority"] = (
        len(authorities) == 5
        and all(isinstance(row, ReferenceOrganismSurfaceSelectionV1)
                for row in authorities))
    checks["bounded_structural_recipe_population"] = (
        len({row.action_recipe for row in authorities}) <= len(PARTNER_ECOLOGIES)
        and all(row.work_units < 14000 for row in authorities))

    action = actions[-1]
    before = capture_reference_organism_surface_selection_v1(organism, action)
    saved_surface = action.payload, action.planned_payload
    action.payload = (255, 0, 255, 0)
    action.planned_payload = (1, 2, 3, 4, 5)
    poisoned = capture_reference_organism_surface_selection_v1(organism, action)
    action.payload, action.planned_payload = saved_surface
    checks["stored_surface_poison_cannot_change_authority"] = poisoned == before

    saved_network = action.selection_network_identity
    action.selection_network_identity ^= 1
    checks["mutated_selection_network_refuses_before_recompute"] = refuses(
        lambda: capture_reference_organism_surface_selection_v1(organism, action),
        "occurrence_currentness")
    action.selection_network_identity = saved_network
    saved_members = action.selection_occurrences
    first_member = saved_members[0]
    action.selection_occurrences = (
        (*first_member[:3], first_member[3] ^ 1), *saved_members[1:])
    checks["mutated_member_root_refuses"] = refuses(
        lambda: capture_reference_organism_surface_selection_v1(organism, action),
        "")
    action.selection_occurrences = saved_members

    saved_source, saved_channel = action.source, action.channel
    action.source += 1
    checks["forged_partner_source_refuses"] = refuses(
        lambda: capture_reference_organism_surface_selection_v1(organism, action),
        "scene")
    action.source = saved_source
    action.channel += 1
    checks["forged_partner_channel_refuses"] = refuses(
        lambda: capture_reference_organism_surface_selection_v1(organism, action),
        "scene")
    action.channel = saved_channel

    duplicate = replace(action, ticket=action.ticket + 1000000)
    organism.actions.append(duplicate)
    organism._action_by_ticket[duplicate.ticket] = duplicate
    checks["host_cannot_choose_among_multiple_pending_actions"] = refuses(
        lambda: capture_reference_organism_surface_selection_v1(organism, action),
        "ticket")
    organism.actions.pop()
    organism._action_by_ticket.pop(duplicate.ticket)

    source_lesion = type(organism).restore(copy.deepcopy(organism.checkpoint()))
    source_action = source_lesion._action_by_ticket[action.ticket]
    source_lesion.withdrawn_sources.add(source_action.source)
    checks["withdrawn_current_source_refuses_before_export"] = refuses(
        lambda: capture_reference_organism_surface_selection_v1(
            source_lesion, source_action), "source")

    replay = copy.deepcopy(organism.checkpoint())
    restored = type(organism).restore(replay)
    restored_action = restored._action_by_ticket[action.ticket]
    checks["complete_checkpoint_replays_exact_authority"] = (
        capture_reference_organism_surface_selection_v1(restored, restored_action)
        == capture_reference_organism_surface_selection_v1(organism, action))

    core = Path(__file__).with_name(
        "reference_organism_partner_surface_bridge_v1.py")
    tree = ast.parse(core.read_text())
    accessed = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    forbidden_names = {"prompt", "expected", "answer", "router", "label",
                       "semantic", "transcript"}
    checks["bridge_never_reads_stored_surface_fields"] = (
        "payload" not in accessed and "planned_payload" not in accessed)
    checks["bridge_has_no_prompt_or_semantic_router"] = not forbidden_names.intersection(
        name.id.lower() for name in ast.walk(tree) if isinstance(name, ast.Name))
    checks["authority_schema_is_numeric_only"] = all(
        str(field.type) in ("int", "tuple[int, ...]", "tuple[tuple[int, int], ...]")
        for field in fields(ReferenceOrganismSurfaceSelectionV1))
    elapsed = (time.perf_counter() - started) * 1000
    checks["consumer_cpu_runtime_under_60_seconds"] = elapsed < 60000
    result = {
        "schema": "0x1.reference-organism-partner-surface-bridge.v1",
        "pass": all(checks.values()), "checks": checks,
        "reference_only": True, "adult_attached": False,
        "runtime_llm": False, "graph_flip": False,
        "bridge_reads_surface_units": False,
        "resident_surface_blind_commitment": False,
        "recipe_status": "AUTHORED_STRUCTURAL_QUOTIENT_IDENTIFIER",
        "voicebox_trajectory_integration": "NOT_YET_CONNECTED",
        "per_byte_action_ancestry": False,
        "production_ir": "ResidentRecipeIrProgram.vcurrent",
        "translation_status": "UNDEFINED",
        "physical_direct_parity": "NOT_RUN/RED",
        "human_language_mastery": False,
        "elapsed_ms": round(elapsed, 3),
        "core_sha256": hashlib.sha256(core.read_bytes()).hexdigest(),
    }
    print("FOUNDRY_ORGANISM_SURFACE_AUTHORITY "
          + ("GREEN" if result["pass"] else "RED")
          + " reference_only=1 byte_blind=1")
    print(json.dumps(result, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
