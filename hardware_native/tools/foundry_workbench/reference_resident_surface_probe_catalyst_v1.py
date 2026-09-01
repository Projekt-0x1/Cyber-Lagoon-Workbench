#!/usr/bin/env python3
"""Learned outer spans can contextually bias an authenticated resident probe.

The mechanism is channel-neutral.  A compact span preceding the structural cue
is merely an outer-context nominee.  It earns a durable catalyst relation only
through later authenticated probe evidence from distinct outcome sources.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from types import MappingProxyType

import reference_resident_authenticated_probe_v1 as probe
import reference_resident_relation_ir_v1 as relation_ir
from reference_resident_authenticated_probe_v1 import (
    AuthenticatedProbeRefuse, BodyMappingContractV1, BodyProbeBoundaryV1,
    BodyReturnContactV1, ProbeCommandV1, ResidentAuthenticatedProbeV1,
)
from reference_resident_channel_sequence_grounding_v1 import (
    ReferenceChannelSequenceBoundaryV1,
)
from reference_resident_composite_cue_prediction_v1 import (
    EDGE_APERTURE, NODE_SPAN, ResidentCompositeCuePredictionV1,
)
from reference_resident_recursive_frontier_v1 import WithdrawalContactV1


SCHEMA_VERSION = 0x53504334
MAX_NOMINATIONS = 64
MAX_WITNESSES = 512
MAX_RECIPES = 128


@dataclass(frozen=True)
class OuterContextNominationV1:
    identity: int
    node: int
    span_recipe: int
    command: int
    command_fields: tuple
    mapping_root: int
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class SurfaceProbeCatalystWitnessV1:
    identity: int
    nomination: int
    span_recipe: int
    action_recipe: int
    mapping_root: int
    probe_evidence: int
    probe_evidence_fields: tuple
    outcome_source: int
    difference: int
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class SurfaceProbeCatalystRecipeV1:
    """No surface units and no action identity: witnesses resolve the relation."""

    identity: int
    span_recipe: int
    mapping_root: int
    credit: int
    witness_roots: tuple[int, ...]
    consumed_probe_evidence_roots: tuple[int, ...]
    source_roots: tuple[int, ...]


class ResidentSurfaceProbeCatalystV1(ResidentAuthenticatedProbeV1):
    def __init__(self, contact_boundary: ReferenceChannelSequenceBoundaryV1,
                 body_boundary: BodyProbeBoundaryV1,
                 mapping: BodyMappingContractV1, session: int = 1,
                 resource_limit: int = 64, work_limit: int = probe.MAX_WORK,
                 initial_inner: ResidentCompositeCuePredictionV1 | None = None):
        super().__init__(contact_boundary, body_boundary, mapping, session,
                         resource_limit, work_limit, initial_inner)
        self.__context_nominations: list[OuterContextNominationV1] = []
        self.__catalyst_witnesses: list[SurfaceProbeCatalystWitnessV1] = []
        self.__catalyst_recipes: dict[int, SurfaceProbeCatalystRecipeV1] = {}
        self.__pending_context: OuterContextNominationV1 | None = None
        self.__selecting_context: OuterContextNominationV1 | None = None

    @property
    def context_nominations(self): return tuple(self.__context_nominations)

    @property
    def catalyst_witnesses(self): return tuple(self.__catalyst_witnesses)

    @property
    def catalyst_recipes(self):
        return MappingProxyType(self.__catalyst_recipes.copy())

    def _snapshot(self):
        return (super()._snapshot(), copy.deepcopy(self.__context_nominations),
                copy.deepcopy(self.__catalyst_witnesses),
                copy.deepcopy(self.__catalyst_recipes),
                copy.deepcopy(self.__pending_context),
                copy.deepcopy(self.__selecting_context))

    def _rollback(self, state):
        (base, self.__context_nominations, self.__catalyst_witnesses,
         self.__catalyst_recipes, self.__pending_context,
         self.__selecting_context) = state
        super()._rollback(base)

    def _policy_uniqueness(self):
        # Current outer context is discovered only after the inner tick opens its
        # exact ticket, so retain the base atomic rollback path.
        return False, 1

    def _outer_context(self, ticket):
        cue = next((row for row in self._inner.actual_nodes
                    if row.identity == ticket.cue_node), None)
        relations = [self._inner.recipes.get(root)
                     for root in ticket.relation_recipe_roots]
        if cue is None or len(relations) != 2 or any(row is None for row in relations):
            return None
        structural = {cue.recipe, ticket.target_recipe,
                      ticket.prospective_middle_recipe}
        for row in relations:
            structural.update((row.cue_recipe, row.target_recipe))
        cue_raw = set(cue.raw_contact_roots)
        rows = [row for row in self._inner.actual_nodes
                if row.kind == NODE_SPAN and row.identity != cue.identity
                and row.source == cue.source
                and row.eligibility_epoch == cue.eligibility_epoch
                and 0 < cue.observed_sequence - row.observed_sequence <= EDGE_APERTURE
                and row.recipe not in structural
                and not cue_raw.intersection(row.raw_contact_roots)]
        if not rows:
            return None
        peak = max((row.observed_sequence, len(row.raw_contact_roots)) for row in rows)
        winners = [row for row in rows
                   if (row.observed_sequence, len(row.raw_contact_roots)) == peak]
        if len(winners) != 1:
            raise AuthenticatedProbeRefuse("surface_probe:context_ambiguous")
        row = winners[0]
        sources = tuple(sorted(set((row.source,) + row.source_roots)))
        values = (row.identity, row.recipe, self.mapping.identity, sources)
        return OuterContextNominationV1(
            probe._identity(b"surface-probe-context-nomination-v1", values),
            row.identity, row.recipe, 0, (), self.mapping.identity, sources)

    def _context_delta(self, span_recipe, action_recipe):
        value = 0
        for row in self.__catalyst_recipes.values():
            self._charge(13 + len(row.witness_roots) + len(row.source_roots))
            program = relation_ir.unfold_bound_relation_v1(
                row.span_recipe, row.mapping_root, row.credit,
                row.witness_roots, row.source_roots)
            roots = set(row.witness_roots)
            self._charge(len(self.__catalyst_witnesses))
            bindings = tuple(relation_ir.ResidentRelationBindingV1(
                witness.identity, witness.action_recipe, witness.source_roots)
                for witness in self.__catalyst_witnesses
                if witness.identity in roots)
            frame = relation_ir.ResidentRelationFrameV1(
                (span_recipe, self.mapping.identity, action_recipe), bindings)
            result = relation_ir.execute_relation_ir_v1(program, frame)
            self._charge(result.work)
            value += result.value
        return value

    def _eligible(self, ticket):
        scored = list(super()._eligible(ticket))
        context = self.__selecting_context
        if context is None:
            return tuple(scored)
        return tuple((score + self._context_delta(
            context.span_recipe, action.identity), action)
            for score, action in scored)

    def _select(self, ticket):
        context = self._outer_context(ticket)
        self.__selecting_context = context
        try:
            scored = self._eligible(ticket); self._charge(len(scored))
            if not scored:
                raise AuthenticatedProbeRefuse("probe:no_eligible_action")
            peak = max(row[0] for row in scored)
            winners = [row for score, row in scored if score == peak]
            if len(winners) != 1:
                raise AuthenticatedProbeRefuse("probe:action_ambiguous")
            action = winners[0]; mapping = dict(self.mapping.entries)
            candidates = tuple(sorted((row.identity, row.capability,
                row.structural_cost, row.resource_cost) for _score, row in scored))
            candidate_root = probe._identity(
                b"authenticated-probe-candidate-set-v1", candidates)
            score_rows = tuple(sorted((row.identity, score) for score, row in scored))
            catalyst_roots = tuple(sorted(row.identity for row in self.__catalyst_recipes.values()
                if context is not None and row.span_recipe == context.span_recipe
                and row.mapping_root == self.mapping.identity))
            state_values = (ticket.envelope_root, tuple(ticket.relation_recipe_roots),
                tuple(ticket.relation_witness_roots), candidates, score_rows,
                0 if context is None else context.identity, catalyst_roots,
                self.resource, tuple(row.identity for row in self.evidence))
            state_root = probe._identity(
                b"surface-probe-selection-state-v1", state_values)
            episode = probe._identity(b"authenticated-probe-episode-v1", [
                ticket.envelope_root, state_root, candidate_root, len(self.events)])
            values = (episode, ticket.ticket, ticket.envelope_root,
                ticket.session, ticket.incarnation, action.identity,
                action.capability, mapping[action.identity], self.mapping.identity,
                state_root, candidate_root, self.resource, action.resource_cost,
                self.next_sequence, ticket.deadline_sequence,
                self.resident_tick, self.resident_tick + probe.PROBE_HORIZON)
            command = ProbeCommandV1(
                probe._identity(b"authenticated-probe-command-v1", values), *values)
            if context is not None:
                nomination_values = (context.node, context.span_recipe,
                    command.identity, command.content_fields(),
                    self.mapping.identity, context.source_roots)
                context = OuterContextNominationV1(
                    probe._identity(b"surface-probe-context-nomination-v1",
                    nomination_values),
                    context.node, context.span_recipe, command.identity,
                    command.content_fields(), self.mapping.identity,
                    context.source_roots)
                if len(self.__context_nominations) >= MAX_NOMINATIONS:
                    raise AuthenticatedProbeRefuse("surface_probe:nomination_bound")
                self.__context_nominations.append(context)
            self.__pending_context = context
            return command
        finally:
            self.__selecting_context = None

    def _rebuild_catalysts(self):
        groups = {}
        for row in self.__catalyst_witnesses:
            self._charge(1)
            groups.setdefault((row.span_recipe, row.action_recipe,
                               row.mapping_root), []).append(row)
        recipes = {}
        for (span_recipe, _action, mapping_root), rows in groups.items():
            positive_sources = {row.outcome_source for row in rows
                                if row.difference > 0}
            credit = sum(row.difference for row in rows)
            if len(positive_sources) < 2 or credit <= 0:
                continue
            roots = tuple(sorted(row.identity for row in rows))
            sources = tuple(sorted({source for row in rows
                                    for source in row.source_roots}))
            context_rows = [row for row in self.__catalyst_witnesses
                            if row.span_recipe == span_recipe
                            and row.mapping_root == mapping_root]
            self._charge(len(self.__catalyst_witnesses))
            consumed = tuple(sorted({row.probe_evidence for row in context_rows}))
            values = (span_recipe, mapping_root, credit, roots, consumed, sources)
            identity = probe._identity(b"surface-probe-catalyst-recipe-v1", values)
            recipes[identity] = SurfaceProbeCatalystRecipeV1(
                identity, span_recipe, mapping_root, credit, roots, consumed, sources)
        if len(recipes) > MAX_RECIPES:
            raise AuthenticatedProbeRefuse("surface_probe:recipe_bound")
        consumed = {root for row in recipes.values()
                    for root in row.consumed_probe_evidence_roots}
        self.evidence = [row for row in self.evidence if row.identity not in consumed]
        self.__catalyst_recipes = recipes

    def ingest_return(self, contact: BodyReturnContactV1):
        before = self._snapshot()
        nomination = copy.deepcopy(self.__pending_context)
        try:
            evidence = super().ingest_return(contact)
            if evidence is None:
                return None
            self.__pending_context = None
            if nomination is None:
                return evidence
            if len(self.__catalyst_witnesses) >= MAX_WITNESSES:
                raise AuthenticatedProbeRefuse("surface_probe:witness_bound")
            sources = tuple(sorted(set(nomination.source_roots
                                       + evidence.source_roots
                                       + (contact.occurrence.source,))))
            evidence_fields = (evidence.episode, evidence.action_recipe,
                evidence.command_hash, evidence.mapping_root,
                evidence.apply_receipt, evidence.structural_ticket,
                evidence.structural_witness, evidence.difference,
                evidence.return_roots, evidence.source_roots)
            if evidence.identity != probe._identity(
                    b"authenticated-probe-evidence-v1", evidence_fields):
                raise AuthenticatedProbeRefuse("surface_probe:evidence_identity")
            values = (nomination.identity, nomination.span_recipe,
                evidence.action_recipe, evidence.mapping_root, evidence.identity,
                evidence_fields, contact.occurrence.source,
                evidence.difference, sources)
            witness = SurfaceProbeCatalystWitnessV1(
                probe._identity(b"surface-probe-catalyst-witness-v1", values), *values)
            self.__catalyst_witnesses.append(witness); self._rebuild_catalysts()
            self._bounded(); return evidence
        except Exception:
            self._rollback(before); raise

    def ingest_withdrawal(self, contact: WithdrawalContactV1):
        before = self._snapshot()
        try:
            target = contact.target_source
            super().ingest_withdrawal(contact)
            self.__context_nominations = [row for row in self.__context_nominations
                                          if target not in row.source_roots]
            self.__catalyst_witnesses = [row for row in self.__catalyst_witnesses
                                         if target not in row.source_roots
                                         and row.outcome_source != target]
            if (self.__pending_context is not None
                    and target in self.__pending_context.source_roots):
                self.__pending_context = None
            self._rebuild_catalysts(); self._bounded()
        except Exception:
            self._rollback(before); raise

    def adopt_mapping(self, mapping: BodyMappingContractV1):
        before = self._snapshot()
        try:
            identity = super().adopt_mapping(mapping)
            self.__context_nominations = [row for row in self.__context_nominations
                                          if row.mapping_root == identity]
            self.__catalyst_witnesses = [row for row in self.__catalyst_witnesses
                                         if row.mapping_root == identity]
            self.__pending_context = None; self._rebuild_catalysts(); self._bounded()
            return identity
        except Exception:
            self._rollback(before); raise

    def tick(self):
        had_context = self.__pending_context is not None
        result = super().tick()
        if had_context and self.pending is None and result.command is None:
            self.__pending_context = None
        return result

    def _validate_catalyst_state(self):
        nomination_roots = set()
        for row in self.__context_nominations:
            values = (row.node, row.span_recipe, row.command, row.command_fields,
                      row.mapping_root, row.source_roots)
            if (row.identity != probe._identity(
                    b"surface-probe-context-nomination-v1", values)
                    or row.command != probe._identity(
                        b"authenticated-probe-command-v1", row.command_fields)
                    or len(row.command_fields) != 17
                    or row.command_fields[8] != row.mapping_root
                    or (row.command_fields[5], row.command_fields[7])
                    not in self.mapping.entries
                    or row.mapping_root != self.mapping.identity
                    or row.identity in nomination_roots):
                raise AuthenticatedProbeRefuse("surface_probe:nomination_state")
            nomination_roots.add(row.identity)
        witness_by_root = {}
        for row in self.__catalyst_witnesses:
            values = (row.nomination, row.span_recipe, row.action_recipe,
                      row.mapping_root, row.probe_evidence,
                      row.probe_evidence_fields, row.outcome_source,
                      row.difference, row.source_roots)
            if (row.identity != probe._identity(
                    b"surface-probe-catalyst-witness-v1", values)
                    or row.probe_evidence != probe._identity(
                        b"authenticated-probe-evidence-v1",
                        row.probe_evidence_fields)
                    or row.nomination not in nomination_roots
                    or row.action_recipe not in {
                        action.identity for action in self._actions}
                    or row.mapping_root != self.mapping.identity
                    or row.difference == 0 or row.identity in witness_by_root):
                raise AuthenticatedProbeRefuse("surface_probe:witness_state")
            witness_by_root[row.identity] = row
        expected = {}
        groups = {}
        for row in self.__catalyst_witnesses:
            groups.setdefault((row.span_recipe, row.action_recipe,
                               row.mapping_root), []).append(row)
        for (span_recipe, _action, mapping_root), rows in groups.items():
            if (len({row.outcome_source for row in rows if row.difference > 0}) < 2
                    or sum(row.difference for row in rows) <= 0):
                continue
            roots = tuple(sorted(row.identity for row in rows))
            sources = tuple(sorted({source for row in rows
                                    for source in row.source_roots}))
            consumed = tuple(sorted({row.probe_evidence
                for row in self.__catalyst_witnesses
                if row.span_recipe == span_recipe
                and row.mapping_root == mapping_root}))
            credit = sum(row.difference for row in rows)
            values = (span_recipe, mapping_root, credit, roots, consumed, sources)
            identity = probe._identity(b"surface-probe-catalyst-recipe-v1", values)
            expected[identity] = SurfaceProbeCatalystRecipeV1(
                identity, span_recipe, mapping_root, credit,
                roots, consumed, sources)
        if self.__catalyst_recipes != expected:
            raise AuthenticatedProbeRefuse("surface_probe:recipe_state")
        if (self.__pending_context is not None
                and self.__pending_context.identity not in nomination_roots):
            raise AuthenticatedProbeRefuse("surface_probe:pending_context_state")
        if self.__selecting_context is not None:
            raise AuthenticatedProbeRefuse("surface_probe:selection_state")

    def _bounded(self):
        super()._bounded()
        if (len(self.__context_nominations) > MAX_NOMINATIONS
                or len(self.__catalyst_witnesses) > MAX_WITNESSES
                or len(self.__catalyst_recipes) > MAX_RECIPES):
            raise AuthenticatedProbeRefuse("surface_probe:state_bound")
        probe._numeric((tuple(self.__context_nominations),
            tuple(self.__catalyst_witnesses), tuple(self.__catalyst_recipes.values()),
            () if self.__pending_context is None else (self.__pending_context,)))
        self._validate_catalyst_state()

    def _checkpoint_body(self):
        view = object.__new__(ResidentAuthenticatedProbeV1)
        view.__dict__ = self.__dict__.copy()
        base = json.loads(self._body_boundary._seal_resident_checkpoint(view))
        return {"schema": SCHEMA_VERSION, "base": base,
                "context_nominations": [asdict(row) for row in self.__context_nominations],
                "catalyst_witnesses": [asdict(row) for row in self.__catalyst_witnesses],
                "catalyst_recipes": [asdict(row) for row in self.__catalyst_recipes.values()],
                "pending_context": None if self.__pending_context is None
                else asdict(self.__pending_context)}

    @classmethod
    def restore(cls, blob, contact_boundary, body_boundary):
        try:
            raw = bytes(blob); envelope = json.loads(raw); body = envelope["body"]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise AuthenticatedProbeRefuse("surface_probe:checkpoint") from exc
        if (len(raw) > probe.MAX_CHECKPOINT_BYTES or not isinstance(envelope, dict)
                or set(envelope) != {"version", "body", "hmac"}
                or envelope["version"] != probe.CHECKPOINT_VERSION
                or not body_boundary._valid_checkpoint(body, envelope["hmac"])
                or not isinstance(body, dict)
                or set(body) != {"schema", "base", "context_nominations",
                                 "catalyst_witnesses", "catalyst_recipes",
                                 "pending_context"}
                or body["schema"] != SCHEMA_VERSION):
            raise AuthenticatedProbeRefuse("surface_probe:checkpoint_authentication")
        for key, bound in (("context_nominations", MAX_NOMINATIONS),
                           ("catalyst_witnesses", MAX_WITNESSES),
                           ("catalyst_recipes", MAX_RECIPES)):
            if not isinstance(body[key], list) or len(body[key]) > bound:
                raise AuthenticatedProbeRefuse("surface_probe:checkpoint_bound")
        boundary_before = body_boundary._snapshot_state()
        try:
            base_blob = probe._canonical(body["base"])
            base = ResidentAuthenticatedProbeV1.restore(
                base_blob, contact_boundary, body_boundary)
            out = object.__new__(cls); out.__dict__ = base.__dict__.copy()
            out.__context_nominations = [probe._from_dict(
                OuterContextNominationV1, row) for row in body["context_nominations"]]
            out.__catalyst_witnesses = [probe._from_dict(
                SurfaceProbeCatalystWitnessV1, row) for row in body["catalyst_witnesses"]]
            recipes = [probe._from_dict(SurfaceProbeCatalystRecipeV1, row)
                       for row in body["catalyst_recipes"]]
            out.__catalyst_recipes = {row.identity: row for row in recipes}
            out.__pending_context = (None if body["pending_context"] is None else
                probe._from_dict(OuterContextNominationV1,
                                 body["pending_context"]))
            out.__selecting_context = None
            out._bounded()
            if out.checkpoint() != raw:
                raise AuthenticatedProbeRefuse("surface_probe:checkpoint_noncanonical")
            return out
        except Exception:
            body_boundary._restore_state(boundary_before)
            raise
