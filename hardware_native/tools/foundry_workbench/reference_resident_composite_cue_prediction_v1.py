#!/usr/bin/env python3
"""Two-edge structural prediction through an absent composite identity.

Only authenticated scalar contacts and zero-argument ticks enter this reference
machine.  Actual span and composite occurrences can nominate future edges, but
only later authenticated occurrences qualify them against a resident marginal.
The learned path is deliberately two-edge: span W -> actual composite L and
actual composite L -> span C.  A later W may traverse a prospective L identity
to open a C ticket, but that prospective identity is never admitted as evidence.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json

from reference_resident_channel_sequence_grounding_v1 import ReferenceChannelSequenceBoundaryV1
from reference_resident_recursive_frontier_v1 import OccurrenceContactV1, WithdrawalContactV1
from reference_resident_parametric_span_network_v1 import (
    ParametricSpanNetworkRefuse, ResidentParametricSpanNetworkV1,
)


SCHEMA_VERSION, CHECKPOINT_VERSION = 0x43435031, 1
NODE_SPAN, NODE_COMPOSITE = 1, 2
EDGE_APERTURE, PREDICTION_HORIZON = 20, 20
MAX_NODES = 1024
MAX_NOMINATIONS = 1024
MAX_HYPOTHESES = 256
MAX_PENDING = 64
MAX_WITNESSES = 1024
MAX_RECIPES = 128
MAX_EVENTS = MAX_TRACE = 4096
MAX_WORK = 32768
MAX_CHECKPOINT_BYTES = 2 << 20
EVENT_SAMPLE, EVENT_TICK, EVENT_WITHDRAWAL = 1, 2, 3


class CompositeCueRefuse(RuntimeError):
    pass


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _identity(domain: bytes, values) -> int:
    value = int.from_bytes(hashlib.sha256(
        domain + b"\0" + _canonical(values)).digest()[:8], "little") & ((1 << 63) - 1)
    return value or 1


def _strict_numeric(value, path="$", extent=MAX_EVENTS):
    if (isinstance(value, bool) or value is None
            or isinstance(value, (str, bytes, bytearray, memoryview))):
        raise CompositeCueRefuse(f"composite_cue:type:{path}")
    if isinstance(value, int):
        if not -(1 << 63) <= value < (1 << 63):
            raise CompositeCueRefuse(f"composite_cue:integer:{path}")
        return
    if hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            _strict_numeric(getattr(value, name), f"{path}.{name}", extent)
        return
    if isinstance(value, (tuple, list)):
        if len(value) > extent:
            raise CompositeCueRefuse(f"composite_cue:bound:{path}")
        for index, item in enumerate(value):
            _strict_numeric(item, f"{path}[{index}]", extent)
        return
    raise CompositeCueRefuse(f"composite_cue:type:{path}")


@dataclass(frozen=True)
class EventV1:
    kind: int
    values: tuple


@dataclass(frozen=True)
class ActualStructuralNodeV1:
    identity: int
    born_sequence: int
    observed_sequence: int
    source: int
    eligibility_epoch: int
    kind: int
    channel: int
    recipe: int
    actual_root: int
    raw_contact_roots: tuple[int, ...]
    ancestry_roots: tuple[int, ...]
    source_roots: tuple[int, ...]
    evidence_revision: int


@dataclass(frozen=True)
class EdgeNominationV1:
    identity: int
    cue_node: int
    target_node: int
    cue_kind: int
    cue_channel: int
    cue_recipe: int
    target_kind: int
    target_channel: int
    target_recipe: int
    source: int
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class EdgeHypothesisV1:
    identity: int
    cue_kind: int
    cue_channel: int
    cue_recipe: int
    target_kind: int
    target_channel: int
    target_recipe: int
    nomination_roots: tuple[int, ...]
    independent_sources: tuple[int, ...]
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class StructuralPredictionTicketV1:
    ticket: int
    envelope_root: int
    session: int
    incarnation: int
    opened_sequence: int
    deadline_sequence: int
    cue_node: int
    cue_evidence_revision: int
    target_kind: int
    target_channel: int
    target_recipe: int
    target_extent: int
    marginal_recipe: int
    hypothesis_roots: tuple[int, ...]
    nomination_roots: tuple[int, ...]
    relation_recipe_roots: tuple[int, ...]
    relation_witness_roots: tuple[int, ...]
    relation_source_roots: tuple[int, ...]
    prospective_middle_kind: int
    prospective_middle_recipe: int
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class SettlementLeafAncestryV1:
    offset: int
    unit: int
    raw_contact_root: int
    target_node: int
    target_span_occurrence: int
    target_span_recipe: int
    target_span_prediction_witness: int
    target_evidence_revision: int
    target_source_roots: tuple[int, ...]
    target_ancestry_roots: tuple[int, ...]
    ticket: int
    ticket_envelope_root: int
    cue_node: int
    hypothesis_root: int
    nomination_roots: tuple[int, ...]
    first_relation_recipe: int
    second_relation_recipe: int
    relation_witness_roots: tuple[int, ...]
    relation_source_roots: tuple[int, ...]
    prospective_middle_recipe: int


@dataclass(frozen=True)
class StructuralPredictionWitnessV1:
    identity: int
    ticket: int
    ticket_envelope_root: int
    source: int
    difference: int
    observed_node: int
    marginal_recipe: int
    hypothesis_roots: tuple[int, ...]
    relation_recipe_roots: tuple[int, ...]
    nomination_roots: tuple[int, ...]
    source_roots: tuple[int, ...]
    ancestry: tuple[SettlementLeafAncestryV1, ...]


@dataclass(frozen=True)
class StructuralEdgeRecipeV1:
    identity: int
    cue_kind: int
    cue_channel: int
    cue_recipe: int
    target_kind: int
    target_channel: int
    target_recipe: int
    marginal_recipe: int
    credit: int
    witness_roots: tuple[int, ...]
    nomination_roots: tuple[int, ...]
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class TickResultV1:
    identity: int
    inner_close_root: int
    prediction_tickets: tuple[int, ...]
    actual_composite_roots: tuple[int, ...]


class ResidentCompositeCuePredictionV1:
    def __init__(self, boundary: ReferenceChannelSequenceBoundaryV1,
                 session: int = 1, work_limit: int = MAX_WORK):
        if not isinstance(boundary, ReferenceChannelSequenceBoundaryV1):
            raise CompositeCueRefuse("composite_cue:boundary")
        if (isinstance(session, bool) or not isinstance(session, int) or session <= 0
                or isinstance(work_limit, bool) or not isinstance(work_limit, int)
                or not 1 <= work_limit <= MAX_WORK):
            raise CompositeCueRefuse("composite_cue:configuration")
        self._boundary = boundary
        self._inner = ResidentParametricSpanNetworkV1(boundary, session, work_limit)
        self.work, self.work_limit = 0, int(work_limit)
        self.next_ticket = 1
        self.events: list[EventV1] = []
        self.trace: list[tuple[int, ...]] = []
        self.actual_nodes: list[ActualStructuralNodeV1] = []
        self.nomination_eligible: set[int] = set()
        self.nominations: list[EdgeNominationV1] = []
        self.hypotheses: dict[int, EdgeHypothesisV1] = {}
        self.pending: dict[int, StructuralPredictionTicketV1] = {}
        self.witnesses: list[StructuralPredictionWitnessV1] = []
        self.recipes: dict[int, StructuralEdgeRecipeV1] = {}
        self.deferred_sample_roots: list[int] = []
        self.current_epoch = 1
        self.last_eligible_sequence = 0

    @property
    def session(self): return self._inner.session

    @property
    def next_sequence(self): return self._inner.next_sequence

    def _snapshot(self):
        state = {key: copy.deepcopy(value) for key, value in self.__dict__.items()
                 if key not in {"_boundary", "_inner"}}
        state["_inner_checkpoint"] = self._inner.checkpoint()
        return state

    def _rollback(self, state):
        boundary = self._boundary; blob = state.pop("_inner_checkpoint")
        self.__dict__.clear(); self._boundary = boundary; self.__dict__.update(state)
        self._inner = ResidentParametricSpanNetworkV1.restore(blob, boundary)

    def _charge(self, amount):
        self.work += int(amount)
        if self.work > self.work_limit: raise CompositeCueRefuse("composite_cue:resource")

    def _record(self, event):
        _strict_numeric(event)
        if len(self.events) >= MAX_EVENTS: raise CompositeCueRefuse("composite_cue:event_bound")
        self.events.append(event)

    def _append_trace(self, *values):
        if len(self.trace) >= MAX_TRACE: raise CompositeCueRefuse("composite_cue:trace_bound")
        self.trace.append(tuple(map(int, values)))

    @staticmethod
    def _contact_event(kind, contact):
        return EventV1(kind, (contact.signed_fields(), contact.auth_tag))

    def _span_recipe(self, identity):
        rows = [row for row in self._inner._inner.recipes.values()
                if row.identity == int(identity)]
        return rows[0] if len(rows) == 1 else None

    def _frontier(self, sample_identity):
        base = self._inner._inner
        recipes = {(row.channel, row.length, row.span_hash): row
                   for row in base.recipes.values()}
        rows = []
        for occurrence in base.span_occurrences:
            if not occurrence.sample_roots or occurrence.sample_roots[-1] != sample_identity:
                continue
            recipe = recipes.get((occurrence.channel, occurrence.length,
                                  occurrence.span_hash))
            if recipe is not None:
                rows.append((recipe, occurrence))
        if not rows: return None
        peak = max((row[0].length, row[0].prediction_gain,
                    row[0].retention_margin) for row in rows)
        winners = [row for row in rows if (row[0].length, row[0].prediction_gain,
                   row[0].retention_margin) == peak]
        if len(winners) != 1: raise CompositeCueRefuse("composite_cue:frontier_ambiguous")
        recipe, occurrence = winners[0]
        samples = [next(row for row in base.samples if row.identity == root)
                   for root in occurrence.sample_roots]
        prediction_root = min(recipe.prediction_witness_roots)
        source_roots = tuple(sorted(set(recipe.source_roots + (occurrence.source,))))
        evidence_revision = _identity(b"composite-cue-span-evidence-v1", [
            occurrence.identity, recipe.identity, recipe.support,
            recipe.retention_margin, recipe.prediction_gain,
            recipe.occurrence_witness_roots, recipe.prediction_witness_roots,
            source_roots])
        return ActualStructuralNodeV1(
            _identity(b"composite-cue-span-node-v1", [occurrence.identity, recipe.identity]),
            occurrence.born_sequence, max(row.contact_sequence for row in samples),
            occurrence.source, 0, NODE_SPAN, occurrence.channel,
            recipe.identity, occurrence.identity,
            tuple(row.contact_root for row in samples), (prediction_root,),
            source_roots, evidence_revision)

    def _rebuild_nominations(self):
        rows = [row for row in self.actual_nodes if row.identity in self.nomination_eligible]
        nominations = []
        for source, epoch in sorted({(row.source, row.eligibility_epoch) for row in rows}):
            ordered = sorted((row for row in rows
                              if row.source == source and row.eligibility_epoch == epoch),
                             key=lambda row: (row.observed_sequence, row.identity))
            for cue, target in zip(ordered, ordered[1:]):
                if (not 0 < target.observed_sequence - cue.observed_sequence <= EDGE_APERTURE
                        or (cue.kind, cue.channel, cue.recipe)
                        == (target.kind, target.channel, target.recipe)):
                    continue
                identity = _identity(b"composite-cue-edge-nomination-v1", [
                    cue.identity, target.identity])
                nominations.append(EdgeNominationV1(
                    identity, cue.identity, target.identity, cue.kind, cue.channel,
                    cue.recipe, target.kind, target.channel, target.recipe, source,
                    tuple(sorted(set((source,) + cue.source_roots
                                     + target.source_roots)))))
        if len(nominations) > MAX_NOMINATIONS:
            raise CompositeCueRefuse("composite_cue:nomination_bound")
        self.nominations = nominations; self._rebuild_hypotheses()

    def _rebuild_hypotheses(self):
        groups = {}
        for row in self.nominations:
            key = (row.cue_kind, row.cue_channel, row.cue_recipe,
                   row.target_kind, row.target_channel, row.target_recipe)
            groups.setdefault(key, []).append(row); self._charge(1)
        hypotheses = {}
        for key, rows in groups.items():
            independent_sources = tuple(sorted({row.source for row in rows}))
            if len(independent_sources) < 2: continue
            roots = tuple(sorted(row.identity for row in rows))
            source_roots = tuple(sorted({source for row in rows
                                         for source in row.source_roots}))
            identity = _identity(b"composite-cue-edge-hypothesis-evidence-v1", [
                key, roots, independent_sources, source_roots])
            hypotheses[identity] = EdgeHypothesisV1(
                identity, *key, roots, independent_sources, source_roots)
        if len(hypotheses) > MAX_HYPOTHESES:
            raise CompositeCueRefuse("composite_cue:hypothesis_bound")
        self.hypotheses = hypotheses; self._rebuild_recipes()

    def _rebuild_recipes(self):
        groups = {}
        for row in self.witnesses:
            # A chained W -> prospective-L -> C settlement tests the chain.  It
            # must never feed either constituent edge or turn prospective L into
            # self-teaching evidence.
            if row.relation_recipe_roots:
                continue
            for root in row.hypothesis_roots:
                hypothesis = self.hypotheses.get(root)
                if hypothesis is not None:
                    key = (hypothesis.cue_kind, hypothesis.cue_channel,
                           hypothesis.cue_recipe, hypothesis.target_kind,
                           hypothesis.target_channel, hypothesis.target_recipe)
                    groups.setdefault(key, []).append(row)
        recipes = {}
        for key, rows in groups.items():
            positives = {row.source for row in rows if row.difference > 0}
            credit = sum(row.difference for row in rows)
            marginals = {row.marginal_recipe for row in rows}
            if len(positives) < 2 or credit <= 0 or len(marginals) != 1: continue
            marginal = next(iter(marginals))
            witness_roots = tuple(sorted(row.identity for row in rows))
            nomination_roots = tuple(sorted({root for row in rows
                                             for root in row.nomination_roots}))
            source_roots = tuple(sorted({source for row in rows
                                         for source in row.source_roots}))
            identity = _identity(b"composite-cue-edge-recipe-evidence-v1", [
                key, marginal, credit, witness_roots, nomination_roots, source_roots])
            recipes[identity] = StructuralEdgeRecipeV1(
                identity, *key, marginal, credit, witness_roots,
                nomination_roots, source_roots)
        if len(recipes) > MAX_RECIPES: raise CompositeCueRefuse("composite_cue:recipe_bound")
        self.recipes = recipes

    def _observe_node(self, node, eligible=True):
        # Longer spans replace their own prefix nodes; actual composites replace all
        # component span nodes.  This prevents internal construction traffic from
        # becoming a shortcut edge.
        remove = set()
        node_raw = set(node.raw_contact_roots)
        for prior in self.actual_nodes:
            if (prior.source == node.source and prior.channel == node.channel
                    and set(prior.raw_contact_roots).issubset(node_raw)
                    and (prior.kind == NODE_SPAN or node.kind == NODE_COMPOSITE)):
                remove.add(prior.identity)
        self.actual_nodes = [row for row in self.actual_nodes if row.identity not in remove]
        self.nomination_eligible.difference_update(remove)
        if eligible:
            if (self.last_eligible_sequence
                    and node.observed_sequence - self.last_eligible_sequence > EDGE_APERTURE):
                self.current_epoch += 1
            self.last_eligible_sequence = node.observed_sequence
            node = replace(node, eligibility_epoch=self.current_epoch)
        self.actual_nodes.append(node)
        if eligible: self.nomination_eligible.add(node.identity)
        if len(self.actual_nodes) > MAX_NODES: raise CompositeCueRefuse("composite_cue:node_bound")
        self._rebuild_nominations()

    def _node_units(self, node):
        if node.kind != NODE_SPAN: return (), ()
        base = self._inner._inner
        occurrence = next(row for row in base.span_occurrences if row.identity == node.actual_root)
        samples = tuple(next(row for row in base.samples if row.identity == root)
                        for root in occurrence.sample_roots)
        return tuple(row.unit for row in samples), samples

    def _settle(self, node):
        settled = []
        for ticket in tuple(self.pending.values()):
            if (not self._valid_ticket_envelope(ticket)
                    or ticket.session != self.session or ticket.incarnation != 1
                    or node.born_sequence <= ticket.opened_sequence
                    or not ticket.opened_sequence < node.observed_sequence
                    <= ticket.deadline_sequence):
                continue
            if (ticket.target_kind, ticket.target_channel) != (node.kind, node.channel):
                continue
            if node.kind == NODE_SPAN:
                seen = sum(1 for row in self._inner._inner.samples
                           if row.channel == node.channel
                           and row.contact_sequence >= ticket.opened_sequence)
                if seen < ticket.target_extent:
                    continue
            difference = ((1 if node.recipe == ticket.target_recipe else 0)
                          - (1 if node.recipe == ticket.marginal_recipe else 0))
            units, samples = self._node_units(node)
            ancestry = ()
            if node.kind == NODE_SPAN:
                prediction_root = node.ancestry_roots[0]
                ancestry = tuple(SettlementLeafAncestryV1(
                    index, sample.unit, sample.contact_root, node.identity,
                    node.actual_root, node.recipe, prediction_root,
                    node.evidence_revision, node.source_roots,
                    node.ancestry_roots, ticket.ticket,
                    ticket.envelope_root, ticket.cue_node,
                    ticket.hypothesis_roots[0], ticket.nomination_roots,
                    ticket.relation_recipe_roots[0] if ticket.relation_recipe_roots else 0,
                    ticket.relation_recipe_roots[1] if len(ticket.relation_recipe_roots) > 1 else 0,
                    ticket.relation_witness_roots, ticket.relation_source_roots,
                    ticket.prospective_middle_recipe)
                    for index, sample in enumerate(samples))
            roots = tuple(sorted(set(ticket.source_roots + node.source_roots
                                     + (node.source,))))
            witness_body = (ticket.envelope_root, node.identity, node.evidence_revision,
                node.source, node.source_roots, node.ancestry_roots,
                difference, ticket.marginal_recipe,
                ticket.hypothesis_roots, ticket.nomination_roots,
                ticket.relation_recipe_roots, ticket.relation_witness_roots,
                ticket.relation_source_roots, roots,
                tuple(asdict(row) for row in ancestry))
            witness = StructuralPredictionWitnessV1(
                _identity(b"composite-cue-prediction-witness-body-v1", witness_body),
                ticket.ticket, ticket.envelope_root, node.source,
                difference, node.identity, ticket.marginal_recipe,
                ticket.hypothesis_roots,
                ticket.relation_recipe_roots, ticket.nomination_roots, roots, ancestry)
            self.witnesses.append(witness); settled.append(ticket.ticket); self._charge(1)
        for identity in settled: self.pending.pop(identity, None)
        if settled: self._rebuild_recipes()

    def _expire(self):
        expired = [identity for identity, row in self.pending.items()
                   if self.next_sequence > row.deadline_sequence]
        for identity in expired: self.pending.pop(identity, None)
        return bool(expired)

    def _invalidate_noncurrent_pending(self):
        hypotheses = set(self.hypotheses)
        nominations = {row.identity for row in self.nominations}
        recipes = set(self.recipes)
        self.pending = {identity: row for identity, row in self.pending.items()
            if set(row.hypothesis_roots).issubset(hypotheses)
            and set(row.nomination_roots).issubset(nominations)
            and set(row.relation_recipe_roots).issubset(recipes)
            and any(node.identity == row.cue_node
                    and node.evidence_revision == row.cue_evidence_revision
                    for node in self.actual_nodes)}

    def _drain_deferred(self):
        roots = tuple(self.deferred_sample_roots); self.deferred_sample_roots.clear()
        for root in roots:
            frontier = self._frontier(root)
            if frontier is not None:
                self._observe_node(frontier, False)

    def ingest_sample(self, contact: OccurrenceContactV1):
        before = self._snapshot()
        try:
            self.work = 0; sample = self._inner.ingest_sample(contact)
            frontier = self._frontier(sample.identity)
            if frontier is not None:
                had_pending = bool(self.pending)
                if had_pending: self._settle(frontier)
                if had_pending and self.pending:
                    self.deferred_sample_roots.append(sample.identity)
                elif had_pending:
                    self._drain_deferred(); self._observe_node(frontier, False)
                else:
                    self._observe_node(frontier, True)
            expired = self._expire()
            if expired and not self.pending: self._drain_deferred()
            self._append_trace(contact.sequence, EVENT_SAMPLE, sample.identity,
                               0 if frontier is None else frontier.identity)
            self._record(self._contact_event(EVENT_SAMPLE, contact))
            self._bounded(); self._check_checkpoint_bound(); return sample
        except Exception:
            self._rollback(before); raise

    @staticmethod
    def _key(row):
        return (row.cue_kind, row.cue_channel, row.cue_recipe,
                row.target_kind, row.target_channel, row.target_recipe)

    def _marginal(self, kind, channel, before_sequence, epoch):
        counts = {}
        for row in self.actual_nodes:
            if (row.kind == kind and row.channel == channel
                    and row.eligibility_epoch == epoch
                    and row.observed_sequence < before_sequence):
                counts[row.recipe] = counts.get(row.recipe, 0) + 1
        if not counts: return None
        peak = max(counts.values()); winners = [key for key, value in counts.items() if value == peak]
        return winners[0] if len(winners) == 1 else None

    @staticmethod
    def _valid_ticket_envelope(row):
        values = (row.ticket, row.session, row.incarnation, row.opened_sequence,
            row.deadline_sequence, row.cue_node, row.cue_evidence_revision,
            row.target_kind, row.target_channel, row.target_recipe,
            row.target_extent, row.marginal_recipe, row.hypothesis_roots,
            row.nomination_roots, row.relation_recipe_roots,
            row.relation_witness_roots, row.relation_source_roots,
            (row.prospective_middle_kind, row.prospective_middle_recipe),
            row.source_roots)
        return row.envelope_root == _identity(b"composite-cue-ticket-envelope-v1", values)

    def _open_ticket(self, cue, target_kind, target_channel, target_recipe,
                     hypotheses, relation_recipes=(), prospective=(0, 0)):
        marginal = (relation_recipes[-1].marginal_recipe if relation_recipes
                    else self._marginal(target_kind, target_channel,
                                        cue.observed_sequence, cue.eligibility_epoch))
        if marginal is None or marginal == target_recipe: return None
        target_extent = 1
        if target_kind == NODE_SPAN:
            target = self._span_recipe(target_recipe); baseline = self._span_recipe(marginal)
            if target is None or baseline is None: return None
            target_extent = max(target.length, baseline.length)
        hypothesis_roots = tuple(row.identity for row in hypotheses)
        nomination_roots = tuple(sorted({root for row in hypotheses
                                          for root in row.nomination_roots}))
        relation_recipe_roots = tuple(row.identity for row in relation_recipes)
        relation_witness_roots = tuple(sorted({root for row in relation_recipes
                                               for root in row.witness_roots}))
        relation_source_roots = tuple(sorted({source for row in relation_recipes
                                              for source in row.source_roots}))
        source_roots = tuple(sorted({*cue.source_roots, *(
            source for row in hypotheses for source in row.source_roots), *(
            source for row in relation_recipes for source in row.source_roots)}))
        envelope_values = (self.next_ticket, self.session, 1, self.next_sequence,
            self.next_sequence + PREDICTION_HORIZON, cue.identity,
            cue.evidence_revision, target_kind, target_channel, target_recipe,
            target_extent, marginal, hypothesis_roots, nomination_roots,
            relation_recipe_roots, relation_witness_roots,
            relation_source_roots, prospective, source_roots)
        envelope_root = _identity(b"composite-cue-ticket-envelope-v1", envelope_values)
        ticket = StructuralPredictionTicketV1(
            self.next_ticket, envelope_root, self.session, 1, self.next_sequence,
            self.next_sequence + PREDICTION_HORIZON, cue.identity,
            cue.evidence_revision,
            target_kind, target_channel, target_recipe, target_extent, marginal,
            hypothesis_roots, nomination_roots, relation_recipe_roots,
            relation_witness_roots, relation_source_roots,
            prospective[0], prospective[1], source_roots)
        self.next_ticket += 1; self.pending[ticket.ticket] = ticket; return ticket

    def _structural_close_ready(self):
        """Resident arbitration: a current three-span frontier closes first."""
        if not self.actual_nodes:
            return False
        newest = self.actual_nodes[-1]
        rows = [row for row in self.actual_nodes
                if row.kind == NODE_SPAN and row.source == newest.source
                and row.channel == newest.channel
                and row.identity in self.nomination_eligible
                and 0 <= newest.observed_sequence - row.observed_sequence <= EDGE_APERTURE]
        if len(rows) < 3:
            return False
        samples = [row for row in self._inner._inner.samples
                   if row.source == newest.source and row.channel == newest.channel]
        position = {row.identity: index for index, row in enumerate(samples)}
        occurrences = {row.identity: row for row in self._inner._inner.span_occurrences}
        ordered = sorted(rows, key=lambda row: (row.observed_sequence, row.identity))
        for left, middle, right in zip(ordered, ordered[1:], ordered[2:]):
            trio = [occurrences.get(row.actual_root) for row in (left, middle, right)]
            if any(row is None or not row.sample_roots for row in trio):
                continue
            if (position.get(trio[0].sample_roots[-1], -2) + 1
                    == position.get(trio[1].sample_roots[0], -1)
                    and position.get(trio[1].sample_roots[-1], -2) + 1
                    == position.get(trio[2].sample_roots[0], -1)):
                return True
        return False

    def _consume_processed_networks(self, inner_result):
        roots = set()
        for identity in inner_result.processed_networks:
            network = self._inner.networks.get(identity)
            if network is not None:
                roots.update(network.member_occurrence_roots)
        if not roots:
            return
        removed = {row.identity for row in self.actual_nodes
                   if row.kind == NODE_SPAN and row.actual_root in roots}
        self.actual_nodes = [row for row in self.actual_nodes if row.identity not in removed]
        self.nomination_eligible.difference_update(removed)
        self._rebuild_nominations()

    def tick(self) -> TickResultV1:
        before = self._snapshot()
        try:
            self.work = 0
            expired = self._expire()
            if expired and not self.pending:
                self._drain_deferred()
            # A pending composite target is settled only by an actual inner close.
            composite_pending = any(row.target_kind == NODE_COMPOSITE
                                    for row in self.pending.values())
            if self.pending and not composite_pending:
                raise CompositeCueRefuse("composite_cue:pending_prediction")
            if composite_pending:
                inner = self._inner.close(); self._consume_processed_networks(inner)
                composites = self._observe_inner_composites(inner, False)
                if self.pending: raise CompositeCueRefuse("composite_cue:composite_not_observed")
                result = TickResultV1(_identity(b"composite-cue-tick-v1", [inner.identity]),
                                      inner.identity, (), tuple(row.identity for row in composites))
            else:
                cue = self.actual_nodes[-1] if self.actual_nodes else None
                tickets = []
                close_ready = self._structural_close_ready()
                if cue is not None and not close_ready:
                    # Qualified two-edge traversal has priority.  The middle is
                    # prospective only and is never passed to _observe_node.
                    first = [row for row in self.recipes.values()
                             if (row.cue_kind, row.cue_channel, row.cue_recipe)
                             == (cue.kind, cue.channel, cue.recipe)
                             and row.target_kind == NODE_COMPOSITE]
                    chain_candidates = []
                    for left in first:
                        for right in self.recipes.values():
                            if ((right.cue_kind, right.cue_channel, right.cue_recipe)
                                    != (NODE_COMPOSITE, left.target_channel, left.target_recipe)):
                                continue
                            direct_key = (cue.kind, cue.channel, cue.recipe,
                                          right.target_kind, right.target_channel,
                                          right.target_recipe)
                            if any(self._key(row) == direct_key for row in (
                                    *self.nominations, *self.hypotheses.values(),
                                    *self.recipes.values())):
                                raise CompositeCueRefuse("composite_cue:direct_shortcut")
                            hypotheses = [row for row in self.hypotheses.values()
                                          if self._key(row) == self._key(right)]
                            if not hypotheses: continue
                            chain_candidates.append((left, right, tuple(hypotheses)))
                    unique = {(left.identity, right.identity): (left, right, hypotheses)
                              for left, right, hypotheses in chain_candidates}
                    if len(unique) > 1:
                        raise CompositeCueRefuse("composite_cue:chain_ambiguous")
                    for left, right, hypotheses in unique.values():
                        ticket = self._open_ticket(cue, right.target_kind,
                            right.target_channel, right.target_recipe, hypotheses,
                            (left, right), (NODE_COMPOSITE, left.target_recipe))
                        if ticket is not None: tickets.append(ticket)
                    if not tickets:
                        for hypothesis in self.hypotheses.values():
                            if (hypothesis.cue_kind, hypothesis.cue_channel,
                                hypothesis.cue_recipe) != (cue.kind, cue.channel, cue.recipe):
                                continue
                            ticket = self._open_ticket(cue, hypothesis.target_kind,
                                hypothesis.target_channel, hypothesis.target_recipe, (hypothesis,))
                            if ticket is not None: tickets.append(ticket)
                    if len(tickets) > 1:
                        raise CompositeCueRefuse("composite_cue:prediction_ambiguous")
                if tickets:
                    inner_root, composite_roots = 0, ()
                else:
                    inner = self._inner.close(); inner_root = inner.identity
                    self._consume_processed_networks(inner)
                    composites = self._observe_inner_composites(inner, True)
                    composite_roots = tuple(row.identity for row in composites)
                result = TickResultV1(_identity(b"composite-cue-tick-v1", [
                    inner_root, tuple(row.ticket for row in tickets), composite_roots]),
                    inner_root, tuple(row.ticket for row in tickets), composite_roots)
            self._append_trace(self.next_sequence, EVENT_TICK, result.identity,
                               *result.prediction_tickets)
            self._record(EventV1(EVENT_TICK, (result.identity, result.inner_close_root,
                result.prediction_tickets, result.actual_composite_roots)))
            self._bounded(); self._check_checkpoint_bound(); return result
        except Exception:
            self._rollback(before); raise

    def _observe_inner_composites(self, inner_result, eligible):
        rows = []
        for rebind in inner_result.rebinds:
            network = self._inner.networks.get(rebind.network)
            if network is None: raise CompositeCueRefuse("composite_cue:network")
            witnesses = [self._inner.networks.get(root)
                         for root in rebind.constructor_witness_roots]
            if (not rebind.constructor_evidence_root
                    or not rebind.constructor_witness_roots):
                raise CompositeCueRefuse("composite_cue:constructor_revision")
            if any(row.member_recipe_roots[1] != network.member_recipe_roots[1]
                   for row in witnesses):
                raise CompositeCueRefuse("composite_cue:constructor_structure")
            if any(row is None or row.born_sequence >= network.born_sequence
                   for row in witnesses):
                raise CompositeCueRefuse("composite_cue:constructor_chronology")
            if any(row.constructor_root != rebind.constructor
                   or row.constructor_evidence_root != rebind.constructor_evidence_root
                   or row.constructor_witness_roots != rebind.constructor_witness_roots
                   for row in rebind.ancestry):
                raise CompositeCueRefuse("composite_cue:constructor_ancestry")
            raw_roots = tuple(row.raw_contact_root for row in rebind.ancestry)
            ancestry = tuple(sorted({root for row in rebind.ancestry for root in (
                row.span_occurrence_root, row.span_recipe_root,
                row.prediction_witness_root, row.network_root, row.constructor_root)}))
            source_roots = tuple(sorted({network.source,
                                         *(row.source for row in witnesses)}))
            evidence_revision = _identity(b"composite-cue-composite-evidence-v1", [
                rebind.identity, network.identity, rebind.constructor,
                rebind.constructor_evidence_root, rebind.constructor_witness_roots,
                raw_roots, source_roots])
            node = ActualStructuralNodeV1(
                _identity(b"composite-cue-composite-node-v1", [rebind.identity]),
                network.born_sequence, self.next_sequence, network.source, 0,
                NODE_COMPOSITE,
                network.channel, rebind.constructor, rebind.identity,
                raw_roots, ancestry + (rebind.constructor_evidence_root,),
                source_roots, evidence_revision)
            self._settle(node); self._observe_node(node, eligible and not bool(self.pending)); rows.append(node)
        return tuple(rows)

    def ingest_withdrawal(self, contact: WithdrawalContactV1):
        before = self._snapshot()
        try:
            self.work = 0; self._inner.ingest_withdrawal(contact); target = contact.target_source
            active_span_occurrences = {row.identity for row in self._inner._inner.span_occurrences}
            active_span_recipes = {row.identity for row in self._inner._inner.recipes.values()}
            active_rebinds = {row.identity for row in self._inner.rebinds}
            self.actual_nodes = [row for row in self.actual_nodes
                if target not in row.source_roots and row.source != target
                and ((row.kind == NODE_SPAN and row.actual_root in active_span_occurrences
                      and row.recipe in active_span_recipes)
                     or (row.kind == NODE_COMPOSITE and row.actual_root in active_rebinds))]
            active = {row.identity for row in self.actual_nodes}
            self.nomination_eligible.intersection_update(active)
            self.pending = {key: row for key, row in self.pending.items()
                            if target not in row.source_roots}
            self.witnesses = [row for row in self.witnesses
                              if target not in row.source_roots and row.source != target]
            self._rebuild_nominations()
            self._invalidate_noncurrent_pending()
            self._append_trace(contact.sequence, EVENT_WITHDRAWAL, target)
            self._record(self._contact_event(EVENT_WITHDRAWAL, contact))
            self._bounded(); self._check_checkpoint_bound()
        except Exception:
            self._rollback(before); raise

    def _bounded(self):
        if (len(self.actual_nodes) > MAX_NODES or len(self.nominations) > MAX_NOMINATIONS
                or len(self.hypotheses) > MAX_HYPOTHESES or len(self.pending) > MAX_PENDING
                or len(self.witnesses) > MAX_WITNESSES or len(self.recipes) > MAX_RECIPES
                or len(self.deferred_sample_roots) > MAX_NODES
                or len(self.events) > MAX_EVENTS or len(self.trace) > MAX_TRACE
                or self.work > self.work_limit):
            raise CompositeCueRefuse("composite_cue:state_bound")
        for rows in (self.actual_nodes, tuple(self.nomination_eligible), self.nominations,
                     tuple(self.hypotheses.values()), tuple(self.pending.values()),
                     self.witnesses, tuple(self.recipes.values()),
                     tuple(self.deferred_sample_roots),
                     (self.current_epoch, self.last_eligible_sequence),
                     self.events, self.trace):
            _strict_numeric(tuple(rows))

    def _checkpoint_body(self):
        return {"schema": SCHEMA_VERSION, "session": self.session, "incarnation": 1,
                "work_limit": self.work_limit, "events": [asdict(row) for row in self.events]}

    def checkpoint(self):
        body = self._checkpoint_body()
        blob = _canonical({"version": CHECKPOINT_VERSION, "body": body,
                           "hmac": self._boundary._checkpoint_tag(body)})
        if len(blob) > MAX_CHECKPOINT_BYTES: raise CompositeCueRefuse("composite_cue:checkpoint_bound")
        return blob

    def _check_checkpoint_bound(self): self.checkpoint()

    @classmethod
    def restore(cls, blob, boundary):
        if not isinstance(boundary, ReferenceChannelSequenceBoundaryV1):
            raise CompositeCueRefuse("composite_cue:boundary")
        try:
            raw_blob = bytes(blob)
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise CompositeCueRefuse("composite_cue:checkpoint") from exc
        if len(raw_blob) > MAX_CHECKPOINT_BYTES:
            raise CompositeCueRefuse("composite_cue:checkpoint_bound")
        try:
            envelope = json.loads(raw_blob); body = envelope["body"]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise CompositeCueRefuse("composite_cue:checkpoint") from exc
        if (not isinstance(envelope, dict)
                or set(envelope) != {"version", "body", "hmac"}
                or envelope["version"] != CHECKPOINT_VERSION
                or not boundary._valid_checkpoint(body, envelope["hmac"])):
            raise CompositeCueRefuse("composite_cue:checkpoint_authentication")
        if (not isinstance(body, dict)
                or set(body) != {"schema", "session", "incarnation", "work_limit", "events"}
                or body["schema"] != SCHEMA_VERSION or body["incarnation"] != 1
                or any(isinstance(body[key], bool) or not isinstance(body[key], int)
                       for key in ("schema", "session", "incarnation", "work_limit"))
                or not isinstance(body["events"], list) or len(body["events"]) > MAX_EVENTS):
            raise CompositeCueRefuse("composite_cue:checkpoint_schema")
        out = cls(boundary, int(body["session"]), int(body["work_limit"]))
        for raw in body["events"]:
            event = EventV1(int(raw["kind"]), _tuplify(raw["values"])); before = len(out.events)
            if event.kind == EVENT_SAMPLE:
                fields, tag = event.values
                out.ingest_sample(OccurrenceContactV1(int(fields[0]), int(fields[1]),
                    int(fields[2]), int(fields[3]), tuple(map(int, fields[4])),
                    tuple(map(int, fields[5])), int(tag)))
            elif event.kind == EVENT_TICK:
                result = out.tick()
                if (result.identity, result.inner_close_root, result.prediction_tickets,
                    result.actual_composite_roots) != event.values:
                    raise CompositeCueRefuse("composite_cue:event_replay")
            elif event.kind == EVENT_WITHDRAWAL:
                fields, tag = event.values
                out.ingest_withdrawal(WithdrawalContactV1(
                    *(int(value) for value in fields), int(tag)))
            else: raise CompositeCueRefuse("composite_cue:event_kind")
            if len(out.events) != before + 1 or out.events[-1] != event:
                raise CompositeCueRefuse("composite_cue:event_replay")
        if out.checkpoint() != raw_blob: raise CompositeCueRefuse("composite_cue:checkpoint_noncanonical")
        return out


def _tuplify(value):
    return tuple(_tuplify(item) for item in value) if isinstance(value, list) else int(value)
