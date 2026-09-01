#!/usr/bin/env python3
"""Byte-blind authority over an organism-selected surface trajectory.

This adapter exposes only bounded numeric causal state.  It deliberately does
not inspect or materialize the action's already-realized surface; a separate
resident voicebox must do that work.
"""
from __future__ import annotations

from dataclasses import dataclass

from reference_organism_v2 import (ActionV2, PREF_BINDING, PREF_FORM,
    PREF_LEXEME, PREF_TEMPLATE, ReferenceOrganismV2, _digest)


MAX_MEMBERS = 32
MAX_ROOTS = 4096
MAX_POPULATION_OCCURRENCES = 16384


class OrganismPartnerSurfaceBridgeRefuse(ValueError):
    pass


@dataclass(frozen=True)
class ReferenceOrganismSurfaceSelectionV1:
    identity: int
    action_recipe: int
    ticket: int
    tick: int
    ingress_channel: int
    consequence_source: int
    selection_context: int
    scene_root: int
    population_root: int
    template_root: int
    closure_root: int
    selection_root: int
    member_roots: tuple[int, ...]
    member_shape: tuple[tuple[int, int], ...]
    relation_roots: tuple[int, ...]
    binding_root: int
    contributor_roots: tuple[int, ...]
    source_roots: tuple[int, ...]
    work_units: int


def _root(tag: str, values) -> int:
    return int(_digest(tag, values)[:15], 16) or 1


def capture_reference_organism_surface_selection_v1(
        organism: ReferenceOrganismV2,
        action: ActionV2) -> ReferenceOrganismSurfaceSelectionV1:
    """Authenticate current selection state and quotient it to a reusable Recipe."""
    if not isinstance(organism, ReferenceOrganismV2) or not isinstance(action, ActionV2):
        raise OrganismPartnerSurfaceBridgeRefuse("organism_surface_bridge:type")
    pending = tuple(row for row in organism.actions if not row.settled)
    current = organism._action_by_ticket.get(int(action.ticket))
    if (current is not action or pending != (action,)
            or action is not organism.actions[-1]
            or int(action.tick) != int(organism.tick_count)):
        raise OrganismPartnerSurfaceBridgeRefuse("organism_surface_bridge:ticket")
    members = tuple(tuple(int(value) for value in row)
                    for row in action.selection_occurrences)
    if (not 1 <= len(members) <= MAX_MEMBERS
            or any(len(row) != 4 or row[0] <= 0 or row[1] < 0
                   or row[2] <= 0 or row[3] <= 0 for row in members)
            or len({row[3] for row in members}) != len(members)):
        raise OrganismPartnerSurfaceBridgeRefuse("organism_surface_bridge:members")
    if (not int(action.selection_context) or not int(action.scene_identity)
            or not int(action.population_occurrence)
            or not int(action.template_identity)
            or not int(action.closure_identity)
            or not int(action.selection_network_identity)):
        raise OrganismPartnerSurfaceBridgeRefuse("organism_surface_bridge:roots")
    scene = organism._scene_by_id.get(int(action.scene_identity))
    closure = organism.utterances.lookup(int(action.closure_identity))
    if (scene is None or scene is not organism.current_scene or closure is None
            or int(action.source) != int(organism.partner_source)
            or int(action.channel) != int(organism.partner_channel)):
        raise OrganismPartnerSurfaceBridgeRefuse("organism_surface_bridge:scene")
    if len(organism.population.occurrences) > MAX_POPULATION_OCCURRENCES:
        raise OrganismPartnerSurfaceBridgeRefuse("organism_surface_bridge:resource")
    live = {int(row.identity): row for row in organism.population.occurrences}
    occurrence_roots = (int(action.population_occurrence),
                        *(row[3] for row in members),
                        *map(int, action.relation_occurrences))
    contributors = tuple(dict.fromkeys(map(int, action.contributors)))
    contributor_set = set(contributors)
    structural_roots = (int(action.scene_identity), int(action.population_occurrence),
        int(action.template_identity), int(action.closure_identity),
        int(action.selection_network_identity), *occurrence_roots)
    if (len(contributors) > MAX_ROOTS
            or any(root not in live or root not in contributor_set
                   for root in occurrence_roots)
            or any(root not in contributor_set for root in structural_roots)):
        raise OrganismPartnerSurfaceBridgeRefuse(
            "organism_surface_bridge:occurrence_currentness")
    configuration = tuple(row[:3] for row in members)
    expected_configuration = organism._selection_configuration(
        int(action.template_identity), action.lexical_identities,
        action.form_slots, int(action.span_identity), int(action.binding_identity))
    if configuration != expected_configuration:
        raise OrganismPartnerSurfaceBridgeRefuse(
            "organism_surface_bridge:configuration")
    active_atoms, used_context, binding_root, relation_roots = (
        organism._surface_view(scene))
    if (int(binding_root) != int(action.binding_identity)
            or tuple(map(int, relation_roots))
                != tuple(map(int, action.relation_occurrences))):
        raise OrganismPartnerSurfaceBridgeRefuse(
            "organism_surface_bridge:binding_currentness")
    surface_context, conditions = organism._surface_context(
        scene, used_context, active_atoms)
    if any(conditions) and int(surface_context) != int(used_context) and action.form_slots:
        candidates = organism.language.template_candidates(
            int(surface_context), len(active_atoms))
        if any(int(row.identity[:15], 16) == int(action.template_identity)
               for row in candidates):
            used_context = int(surface_context)
    for member_kind, slot, candidate, occurrence_root in members:
        qualifier = (used_context if member_kind == PREF_TEMPLATE else
            active_atoms[slot - 1] if member_kind in (PREF_LEXEME, PREF_FORM)
            and 0 < slot <= len(active_atoms) else 0)
        features = ((0x51EC7, member_kind, slot, candidate,
            int(action.selection_context), int(scene.population_occurrence),
            *map(int, relation_roots)) if member_kind == PREF_BINDING else
            (0x51EC7, member_kind, slot, candidate,
             int(action.selection_context), int(qualifier)))
        if live[occurrence_root].sites != organism.population.signature(features):
            raise OrganismPartnerSurfaceBridgeRefuse(
                "organism_surface_bridge:member_currentness")
    expected_network = organism._selection_network_identity(
        int(action.selection_context), int(action.population_occurrence),
        int(action.closure_identity), members)
    if int(action.selection_network_identity) != int(expected_network):
        raise OrganismPartnerSurfaceBridgeRefuse("organism_surface_bridge:network")
    member_shape = tuple((row[0], row[1]) for row in members)
    # Candidate lexical identities and occurrence identities are intentionally
    # absent: they remain exact ancestry, while this quotient can be reused by
    # a learned constructor binding across held-out lexical recombinations.
    recipe_values = (int(action.selection_context), int(action.template_identity),
        int(action.binding_identity), int(action.span_identity), member_shape,
        len(action.relation_occurrences), tuple(map(int, action.form_slots)))
    action_recipe = _root("organism-surface-action-recipe-v1", recipe_values)
    member_roots = tuple(row[3] for row in members)
    relation_roots = tuple(map(int, action.relation_occurrences))
    sources = tuple(sorted({int(action.source), int(action.body_source),
                            int(scene.source)} - {0}))
    if (not sources or any(source in organism.withdrawn_sources for source in sources)):
        raise OrganismPartnerSurfaceBridgeRefuse("organism_surface_bridge:source")
    work = (16 + len(organism.population.occurrences)
            + len(members) * 6 + len(relation_roots) * 2
            + len(contributors) + len(sources))
    if work > 16 + MAX_POPULATION_OCCURRENCES + MAX_MEMBERS * 6 + MAX_ROOTS * 3:
        raise OrganismPartnerSurfaceBridgeRefuse("organism_surface_bridge:resource")
    values = (action_recipe, int(action.ticket), int(action.tick),
        int(action.channel), int(action.source), int(action.selection_context),
        int(action.scene_identity), int(action.population_occurrence),
        int(action.template_identity), int(action.closure_identity),
        int(action.selection_network_identity), member_roots, member_shape,
        relation_roots, int(action.binding_identity), contributors, sources, work)
    return ReferenceOrganismSurfaceSelectionV1(
        _root("organism-surface-selection-authority-v1", values), *values)
