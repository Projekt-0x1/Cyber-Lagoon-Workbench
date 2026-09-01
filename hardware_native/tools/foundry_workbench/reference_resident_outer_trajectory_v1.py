#!/usr/bin/env python3
"""Transient outer-byte trajectory from a resident structural prediction.

The durable machine keeps learned Recipes, Occurrences, and relation-network
evidence.  This dead voice box accepts no target or surface argument: it can
only unfold the one current resident ticket into bytes already witnessed for
that learned target Recipe.  The returned trajectory is ephemeral and is not
fed back as contact or retained in the checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass

from reference_resident_composite_cue_prediction_v1 import (
    CompositeCueRefuse, NODE_SPAN, ResidentCompositeCuePredictionV1, _identity,
    _strict_numeric,
)
from reference_resident_variable_span_v1 import MAX_SPAN


MAX_OUTER_BYTES = 256
MAX_OUTER_WITNESSES = 256
MAX_OUTER_ROOTS = 1024


@dataclass(frozen=True)
class OuterByteAncestryV1:
    offset: int
    unit: int
    raw_contact_root: int
    sample_root: int
    contact_sequence: int
    source: int
    channel: int
    provenance: tuple[int, ...]
    target_occurrence_root: int
    target_recipe_root: int
    target_prediction_witness_root: int
    target_prediction_witness_source: int
    ticket: int
    ticket_envelope_root: int
    cue_node_root: int
    cue_evidence_revision: int
    relation_recipe_roots: tuple[int, ...]
    relation_witness_roots: tuple[int, ...]
    relation_source_roots: tuple[int, ...]
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class ResidentOuterTrajectoryV1:
    identity: int
    ticket: int
    ticket_envelope_root: int
    target_channel: int
    target_recipe_root: int
    target_occurrence_root: int
    units: tuple[int, ...]
    constituent_roots: tuple[int, ...]
    source_roots: tuple[int, ...]
    work_units: int
    ancestry: tuple[OuterByteAncestryV1, ...]


def unfold_resident_outer_trajectory_v1(
        resident: ResidentCompositeCuePredictionV1) -> ResidentOuterTrajectoryV1:
    """Unfold the unique resident-selected target; never choose current thought."""
    if not isinstance(resident, ResidentCompositeCuePredictionV1):
        raise CompositeCueRefuse("outer_trajectory:resident")
    if len(resident.pending) != 1:
        raise CompositeCueRefuse("outer_trajectory:pending")
    ticket = next(iter(resident.pending.values()))
    if (ticket.target_kind != NODE_SPAN
            or not resident._valid_ticket_envelope(ticket)
            or not (ticket.opened_sequence <= resident.next_sequence
                    <= ticket.deadline_sequence)):
        raise CompositeCueRefuse("outer_trajectory:ticket")
    base = resident._inner._inner
    work_units = (len(resident.pending) + len(resident.actual_nodes)
                  + len(base.recipes) + len(base.span_occurrences)
                  + len(base.samples) + len(base.prediction_witnesses))
    if work_units > resident.work_limit:
        raise CompositeCueRefuse("outer_trajectory:resource")
    cue_rows = [row for row in resident.actual_nodes
                if row.identity == ticket.cue_node]
    if (len(cue_rows) != 1
            or cue_rows[0].evidence_revision != ticket.cue_evidence_revision):
        raise CompositeCueRefuse("outer_trajectory:cue_currentness")

    recipe = resident._span_recipe(ticket.target_recipe)
    if (recipe is None or not (1 <= recipe.length <= MAX_OUTER_BYTES)
            or not (1 <= len(recipe.occurrence_witness_roots) <= MAX_OUTER_WITNESSES)
            or not (1 <= len(recipe.prediction_witness_roots) <= MAX_OUTER_WITNESSES)):
        raise CompositeCueRefuse("outer_trajectory:recipe")
    occurrences = [row for row in base.span_occurrences
                   if row.identity in recipe.occurrence_witness_roots]
    if len(occurrences) != len(recipe.occurrence_witness_roots):
        raise CompositeCueRefuse("outer_trajectory:witness_currentness")
    reserved_work_units = (work_units
        + len(occurrences) * recipe.length * (12 + MAX_SPAN * 4)
        + len(base.prediction_witnesses) * 6 + MAX_OUTER_ROOTS * 2)
    if reserved_work_units > resident.work_limit:
        raise CompositeCueRefuse("outer_trajectory:resource")

    sample_by_root = {row.identity: row for row in base.samples}
    if len(sample_by_root) != len(base.samples):
        raise CompositeCueRefuse("outer_trajectory:sample_identity")
    witnessed = []
    for occurrence in occurrences:
        try:
            samples = [sample_by_root[root] for root in occurrence.sample_roots]
        except KeyError as exc:
            raise CompositeCueRefuse("outer_trajectory:sample_currentness") from exc
        units = tuple(row.unit for row in samples)
        if (len(units) != recipe.length
                or any(not 0 <= unit <= 255 for unit in units)
                or any(row.source != occurrence.source
                       or row.channel != occurrence.channel
                       or row.source in base.withdrawn_sources
                       or row.contact_root != _identity(b"variable-span-contact-v1", (
                           resident.session, row.contact_sequence, row.source,
                           row.channel, (row.unit,), row.provenance))
                       or row.identity != _identity(b"variable-span-sample-v1", (
                           row.contact_root, row.contact_sequence))
                       for row in samples)
                or occurrence.identity != _identity(b"variable-span-occurrence-v1", (
                    occurrence.channel, tuple(row.identity for row in samples)))):
            raise CompositeCueRefuse("outer_trajectory:byte_extent")
        work_units += len(samples)
        witnessed.append((occurrence, samples, units))
    if not witnessed or len({row[2] for row in witnessed}) != 1:
        raise CompositeCueRefuse("outer_trajectory:witness_ambiguity")
    occurrence, samples, units = min(witnessed, key=lambda row: row[0].identity)
    if _identity(b"variable-span-content-v1", units) != recipe.span_hash:
        raise CompositeCueRefuse("outer_trajectory:content_root")

    source_roots = tuple(sorted({*ticket.source_roots, *recipe.source_roots,
                                 occurrence.source}))
    if (not source_roots
            or any(source in base.withdrawn_sources for source in source_roots)):
        raise CompositeCueRefuse("outer_trajectory:source_currentness")
    prediction_rows = [row for row in base.prediction_witnesses
                       if row.identity in recipe.prediction_witness_roots]
    if (len(prediction_rows) != len(recipe.prediction_witness_roots)
            or any(row.difference <= 0 or row.source in base.withdrawn_sources
                   or row.channel != recipe.channel
                   or row.identity != _identity(b"variable-span-prediction-witness-v1", (
                       row.ticket, row.observed_sample, row.difference))
                   for row in prediction_rows)):
        raise CompositeCueRefuse("outer_trajectory:prediction_witness_currentness")
    prediction = min(prediction_rows, key=lambda row: row.identity)
    prediction_root = prediction.identity
    constituent_roots = tuple(sorted({cue_rows[0].actual_root,
        *cue_rows[0].ancestry_roots, *ticket.relation_recipe_roots,
        *ticket.relation_witness_roots, recipe.identity, occurrence.identity,
        prediction_root, *(row.contact_root for row in samples),
        *(row.identity for row in samples)}))
    if (len(source_roots) > MAX_OUTER_ROOTS
            or len(constituent_roots) > MAX_OUTER_ROOTS):
        raise CompositeCueRefuse("outer_trajectory:root_bound")
    work_units += (len(source_roots) + len(constituent_roots)
                   + len(prediction_rows) + len(samples))
    if work_units > resident.work_limit:
        raise CompositeCueRefuse("outer_trajectory:resource")
    ancestry = tuple(OuterByteAncestryV1(
        offset, sample.unit, sample.contact_root, sample.identity,
        sample.contact_sequence, sample.source, sample.channel, sample.provenance,
        occurrence.identity, recipe.identity, prediction_root, prediction.source,
        ticket.ticket, ticket.envelope_root,
        ticket.cue_node, ticket.cue_evidence_revision,
        ticket.relation_recipe_roots, ticket.relation_witness_roots,
        ticket.relation_source_roots, source_roots)
        for offset, sample in enumerate(samples))
    identity = _identity(b"resident-outer-trajectory-v1", (
        ticket.envelope_root, recipe.identity, occurrence.identity, units,
        constituent_roots, source_roots, work_units,
        tuple(tuple(getattr(row, name) for name in row.__dataclass_fields__)
              for row in ancestry)))
    result = ResidentOuterTrajectoryV1(
        identity, ticket.ticket, ticket.envelope_root, ticket.target_channel,
        recipe.identity, occurrence.identity, units, constituent_roots,
        source_roots, work_units, ancestry)
    _strict_numeric(result, extent=MAX_OUTER_BYTES)
    return result
