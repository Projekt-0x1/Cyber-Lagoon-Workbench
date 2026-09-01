#!/usr/bin/env python3
"""Resident channel-sequence acquisition and future-grounding reference kernel.

Every channel obeys the same fixed width-three recurrence law.  The boundary can
authenticate only extent-one numeric samples and source withdrawal.  It cannot
submit chunks, episodes, bindings, candidates, winners, consequences, scores, or
expected units.  Prediction tickets settle only when later authenticated channel
sequences arrive through the admitted reference boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import hmac
import json

from reference_resident_recursive_frontier_v1 import (
    CONTACT_OCCURRENCE, CONTACT_WITHDRAWAL, FrontierRefuse,
    OccurrenceContactV1, ResidentRecursiveFrontierV1, WithdrawalContactV1,
    admit_reference_boundary_v1,
)


SCHEMA_VERSION = CHECKPOINT_VERSION = 1
WIDTH = 3
COACTIVITY_APERTURE = 8
PREDICTION_HORIZON = 12
MAX_SAMPLES = 256
MAX_SEQUENCE_OCCURRENCES = 256
MAX_SEQUENCE_RECIPES = 128
MAX_HYPOTHESES = 512
MAX_PENDING = 64
MAX_PREDICTION_WITNESSES = 512
MAX_CROSS_RECIPES = 128
MAX_WITHDRAWN_SOURCES = 256
MAX_EVENTS = MAX_TRACE = 4096
MAX_WORK = 4096
MAX_CHECKPOINT_BYTES = 1 << 20

EVENT_SAMPLE = 1
EVENT_TICK = 2
EVENT_WITHDRAWAL = 3
EVENT_UNFOLD = 4

_CHANNEL_BOUNDARY_ADMISSION = object()


class GroundingRefuse(RuntimeError):
    pass


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _identity(domain: bytes, values) -> int:
    value = int.from_bytes(hashlib.sha256(
        domain + b"\0" + _canonical(values)).digest()[:8], "little") & ((1 << 63) - 1)
    return value or 1


def _strict_numeric(value, path="$", extent=MAX_EVENTS) -> None:
    if (isinstance(value, bool) or value is None
            or isinstance(value, (str, bytes, bytearray, memoryview))):
        raise GroundingRefuse(f"grounding:type:{path}")
    if isinstance(value, int):
        if not -(1 << 63) <= value < (1 << 63):
            raise GroundingRefuse(f"grounding:integer:{path}")
        return
    if hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            _strict_numeric(getattr(value, name), f"{path}.{name}", extent)
        return
    if isinstance(value, (tuple, list)):
        if len(value) > extent:
            raise GroundingRefuse(f"grounding:bound:{path}")
        for index, item in enumerate(value):
            _strict_numeric(item, f"{path}[{index}]", extent)
        return
    raise GroundingRefuse(f"grounding:type:{path}")


@dataclass(frozen=True)
class EventV1:
    kind: int
    values: tuple


@dataclass(frozen=True)
class SampleOccurrenceV1:
    identity: int
    contact_sequence: int
    source: int
    channel: int
    unit: int
    provenance: tuple[int, ...]
    contact_root: int


@dataclass(frozen=True)
class ChannelSequenceOccurrenceV1:
    identity: int
    born_sequence: int
    channel: int
    sample_roots: tuple[int, ...]
    source_roots: tuple[int, ...]
    sequence_hash: int


@dataclass(frozen=True)
class ChannelSequenceRecipeV1:
    identity: int
    channel: int
    sequence_hash: int
    length: int
    support: int
    witness_roots: tuple[int, ...]
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class CoactivityHypothesisV1:
    identity: int
    cue_recipe: int
    target_recipe: int
    cue_channel: int
    target_channel: int
    support: int
    witness_roots: tuple[int, ...]
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class PredictionTicketV1:
    ticket: int
    incarnation: int
    opened_sequence: int
    deadline_sequence: int
    cue_occurrence: int
    cue_recipe: int
    target_recipe: int
    target_channel: int
    hypothesis_root: int
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class PredictionWitnessV1:
    identity: int
    ticket: int
    source: int
    difference: int
    cue_recipe: int
    target_recipe: int
    target_channel: int
    observed_occurrence: int
    hypothesis_root: int
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class CrossChannelRecipeV1:
    identity: int
    cue_recipe: int
    target_recipe: int
    cue_channel: int
    target_channel: int
    support: int
    credit: int
    witness_roots: tuple[int, ...]
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class LeafAncestryV1:
    offset: int
    unit: int
    raw_contact_root: int
    sequence_recipe_root: int
    cross_channel_recipe_root: int
    prediction_witness_root: int
    recursive_occurrence_root: int


@dataclass(frozen=True)
class GroundedUnfoldV1:
    identity: int
    cue_occurrence: int
    target_occurrence: int
    cross_channel_recipe: int
    units: tuple[int, ...]
    ancestry: tuple[LeafAncestryV1, ...]
    recursive_occurrence: int


class ReferenceChannelSequenceBoundaryV1:
    """Narrow wrapper over the committed admitted boundary authority."""

    def __init__(self, admission=None):
        if admission is not _CHANNEL_BOUNDARY_ADMISSION:
            raise GroundingRefuse("grounding:boundary_admission")
        self.__authority = admit_reference_boundary_v1()

    def seal_sample(self, session: int, sequence: int, source: int, channel: int,
                    units, provenance=()) -> OccurrenceContactV1:
        if (isinstance(units, (str, bytes, bytearray, memoryview))
                or isinstance(provenance, (str, bytes, bytearray, memoryview))):
            raise GroundingRefuse("grounding:literal_boundary")
        if isinstance(units, bool):
            raise GroundingRefuse("grounding:sample_type")
        try:
            values = tuple(units)
            provenance = tuple(provenance)
        except TypeError as exc:
            raise GroundingRefuse("grounding:sample_extent") from exc
        if (len(values) != 1 or isinstance(values[0], bool)
                or not isinstance(values[0], int)
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in (session, sequence, source, channel, *provenance))):
            raise GroundingRefuse("grounding:sample_extent")
        return self.__authority.seal_occurrence(
            int(session), int(sequence), int(source), int(channel), values,
            provenance)

    def seal_withdrawal(self, session: int, sequence: int, source: int,
                        channel: int, target_source: int) -> WithdrawalContactV1:
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (session, sequence, source, channel, target_source)):
            raise GroundingRefuse("grounding:withdrawal_type")
        return self.__authority.seal_withdrawal(
            int(session), int(sequence), int(source), int(channel), int(target_source))

    def _valid_contact(self, kind: int, contact) -> bool:
        return self.__authority._valid_contact(kind, contact.signed_fields(), contact.auth_tag)

    def _checkpoint_tag(self, body) -> str:
        return self.__authority._checkpoint_tag(body)

    def _valid_checkpoint(self, body, tag) -> bool:
        return self.__authority._valid_checkpoint(body, tag)


def admit_channel_sequence_boundary_v1() -> ReferenceChannelSequenceBoundaryV1:
    return ReferenceChannelSequenceBoundaryV1(_CHANNEL_BOUNDARY_ADMISSION)


class ResidentChannelSequenceGroundingV1:
    def __init__(self, boundary: ReferenceChannelSequenceBoundaryV1,
                 session: int = 1, work_limit: int = MAX_WORK):
        if not isinstance(boundary, ReferenceChannelSequenceBoundaryV1):
            raise GroundingRefuse("grounding:boundary")
        if session <= 0 or not 1 <= work_limit <= MAX_WORK:
            raise GroundingRefuse("grounding:configuration")
        self._boundary = boundary
        self._recursive_authority = admit_reference_boundary_v1()
        self._recursive = ResidentRecursiveFrontierV1(
            self._recursive_authority, session + (1 << 20), work_limit=min(512, work_limit))
        self.session, self.incarnation = int(session), 1
        self.next_sequence = self.next_ticket = 1
        self.work = 0
        self.work_limit = int(work_limit)
        self.events: list[EventV1] = []
        self.samples: list[SampleOccurrenceV1] = []
        self.sequence_occurrences: list[ChannelSequenceOccurrenceV1] = []
        self.sequence_recipes: dict[tuple[int, int], ChannelSequenceRecipeV1] = {}
        self.hypotheses: dict[int, CoactivityHypothesisV1] = {}
        self.pending: dict[int, PredictionTicketV1] = {}
        self.prediction_witnesses: list[PredictionWitnessV1] = []
        self.cross_recipes: dict[int, CrossChannelRecipeV1] = {}
        self.withdrawn_sources: set[int] = set()
        self.trace: list[tuple[int, ...]] = []

    def _snapshot(self):
        state = {key: copy.deepcopy(value) for key, value in self.__dict__.items()
                 if key not in {"_boundary", "_recursive_authority", "_recursive"}}
        state["_recursive_checkpoint"] = self._recursive.checkpoint()
        return state

    def _rollback(self, state):
        boundary, authority = self._boundary, self._recursive_authority
        recursive_blob = state.pop("_recursive_checkpoint")
        self.__dict__.clear(); self._boundary = boundary; self._recursive_authority = authority
        self.__dict__.update(state)
        self._recursive = ResidentRecursiveFrontierV1.restore(recursive_blob, authority)

    def _charge(self, amount):
        self.work += int(amount)
        if self.work > self.work_limit:
            raise GroundingRefuse("grounding:resource")

    def _record(self, event):
        _strict_numeric(event)
        if len(self.events) >= MAX_EVENTS:
            raise GroundingRefuse("grounding:event_bound")
        self.events.append(event)

    def _append_trace(self, *values):
        if len(self.trace) >= MAX_TRACE:
            raise GroundingRefuse("grounding:trace_bound")
        self.trace.append(tuple(map(int, values)))

    def _verify(self, kind, contact):
        _strict_numeric(contact)
        if contact.session != self.session or contact.sequence != self.next_sequence:
            raise GroundingRefuse("grounding:session_sequence")
        if not self._boundary._valid_contact(kind, contact):
            raise GroundingRefuse("grounding:authentication")

    @staticmethod
    def _contact_event(kind, contact):
        return EventV1(kind, (contact.signed_fields(), contact.auth_tag))

    def ingest_sample(self, contact: OccurrenceContactV1) -> SampleOccurrenceV1:
        before = self._snapshot()
        try:
            self.work = 0; self._verify(CONTACT_OCCURRENCE, contact)
            if (contact.source <= 0 or contact.channel <= 0
                    or contact.source in self.withdrawn_sources
                    or len(contact.features) != 1 or isinstance(contact.features[0], bool)
                    or not isinstance(contact.features[0], int)
                    or len(contact.provenance) > WIDTH * 4
                    or any(root <= 0 for root in contact.provenance)):
                raise GroundingRefuse("grounding:sample_shape")
            if len(self.samples) >= MAX_SAMPLES:
                raise GroundingRefuse("grounding:sample_bound")
            contact_root = _identity(b"channel-sample-contact-v1", contact.signed_fields())
            sample = SampleOccurrenceV1(_identity(b"channel-sample-occurrence-v1", [
                contact_root, contact.sequence]), contact.sequence, contact.source,
                contact.channel, contact.features[0], contact.provenance, contact_root)
            self.samples.append(sample); self.next_sequence += 1
            sequence = self._form_new_sequence(sample.channel)
            if sequence is not None:
                self._settle_predictions(sequence)
            self._expire_predictions(sample)
            self._rebuild_derived()
            self._append_trace(contact.sequence, EVENT_SAMPLE, sample.identity,
                               0 if sequence is None else sequence.identity)
            self._record(self._contact_event(EVENT_SAMPLE, contact))
            self._bounded(); self._check_checkpoint_bound(); return sample
        except Exception:
            self._rollback(before); raise

    def _form_new_sequence(self, channel):
        rows = [row for row in self.samples if row.channel == int(channel)]
        if len(rows) < WIDTH:
            return None
        members = tuple(rows[-WIDTH:]); units = tuple(row.unit for row in members)
        if len({row.source for row in members}) != 1:
            return None
        sequence_hash = _identity(b"channel-sequence-content-v1", units)
        occurrence = ChannelSequenceOccurrenceV1(
            _identity(b"channel-sequence-occurrence-v1", [
                channel, tuple(row.identity for row in members)]),
            members[-1].contact_sequence, int(channel),
            tuple(row.identity for row in members),
            tuple(sorted(set(row.source for row in members))), sequence_hash)
        if len(self.sequence_occurrences) >= MAX_SEQUENCE_OCCURRENCES:
            raise GroundingRefuse("grounding:sequence_occurrence_bound")
        self.sequence_occurrences.append(occurrence); self._charge(WIDTH)
        return occurrence

    def _rebuild_derived(self):
        active = [row for row in self.sequence_occurrences
                  if not set(row.source_roots).intersection(self.withdrawn_sources)]
        grouped = {}
        for row in active:
            self._charge(1)
            grouped.setdefault((row.channel, row.sequence_hash), []).append(row)
        recipes = {}
        for key, rows in grouped.items():
            sources = tuple(sorted(set(source for row in rows for source in row.source_roots)))
            if len(sources) < 2:
                continue
            identity = _identity(b"channel-sequence-recipe-v1", [key[0], key[1], WIDTH])
            recipes[key] = ChannelSequenceRecipeV1(
                identity, key[0], key[1], WIDTH, len(sources),
                tuple(sorted(row.identity for row in rows)), sources)
        if len(recipes) > MAX_SEQUENCE_RECIPES:
            raise GroundingRefuse("grounding:sequence_recipe_bound")
        self.sequence_recipes = recipes

        groups = {}
        for left in active:
            if (left.channel, left.sequence_hash) not in recipes:
                continue
            for right in active:
                self._charge(1)
                if (right.channel == left.channel
                        or (right.channel, right.sequence_hash) not in recipes
                        or not set(left.source_roots).intersection(right.source_roots)
                        or not 0 < right.born_sequence - left.born_sequence <= COACTIVITY_APERTURE):
                    continue
                key = (recipes[(left.channel, left.sequence_hash)].identity,
                       recipes[(right.channel, right.sequence_hash)].identity,
                       left.channel, right.channel)
                groups.setdefault(key, []).append((left, right))
                self._charge(1)
        hypotheses = {}
        for key, pairs in groups.items():
            sources = tuple(sorted(set(source for pair in pairs
                                       for row in pair for source in row.source_roots)))
            if len(sources) < 2:
                continue
            roots = tuple(sorted(_identity(b"channel-coactivity-witness-v1", [
                left.identity, right.identity]) for left, right in pairs))
            identity = _identity(b"channel-coactivity-hypothesis-v1", [key, roots])
            hypotheses[identity] = CoactivityHypothesisV1(
                identity, key[0], key[1], key[2], key[3], len(sources), roots, sources)
        if len(hypotheses) > MAX_HYPOTHESES:
            raise GroundingRefuse("grounding:hypothesis_bound")
        self.hypotheses = hypotheses
        self._refresh_cross_recipes()

    def tick(self) -> tuple[PredictionTicketV1, ...]:
        before = self._snapshot()
        try:
            if self.pending:
                raise GroundingRefuse("grounding:pending_prediction")
            self.work = 0
            if not self.sequence_occurrences:
                raise GroundingRefuse("grounding:no_sequence_cue")
            cue = self.sequence_occurrences[-1]
            cue_recipe = self.sequence_recipes.get((cue.channel, cue.sequence_hash))
            if cue_recipe is None:
                raise GroundingRefuse("grounding:unsupported_sequence_cue")
            alternatives = [row for row in self.hypotheses.values()
                            if row.cue_recipe == cue_recipe.identity]
            if not alternatives:
                raise GroundingRefuse("grounding:no_prediction_alternative")
            if len(alternatives) > MAX_PENDING:
                raise GroundingRefuse("grounding:pending_bound")
            tickets = []
            for row in sorted(alternatives, key=lambda item: item.identity):
                ticket = PredictionTicketV1(
                    self.next_ticket, self.incarnation, self.next_sequence,
                    self.next_sequence + PREDICTION_HORIZON, cue.identity,
                    row.cue_recipe, row.target_recipe, row.target_channel,
                    row.identity, tuple(sorted(set(cue.source_roots + row.source_roots))))
                self.next_ticket += 1; self.pending[ticket.ticket] = ticket
                tickets.append(ticket); self._charge(4)
            self._append_trace(self.next_sequence, EVENT_TICK,
                               *(row.ticket for row in tickets))
            event = EventV1(EVENT_TICK, (tuple(row.ticket for row in tickets),))
            self._record(event); self._bounded(); self._check_checkpoint_bound()
            return tuple(tickets)
        except Exception:
            self._rollback(before); raise

    def _settle_predictions(self, observed):
        settled = []
        for ticket in tuple(self.pending.values()):
            if observed.channel != ticket.target_channel:
                continue
            difference = (1 if self.next_sequence <= ticket.deadline_sequence
                          and observed.sequence_hash
                          == self._recipe(ticket.target_recipe).sequence_hash else -1)
            if difference > 0:
                roots = tuple(sorted(set(ticket.source_roots + observed.source_roots)))
                witness = PredictionWitnessV1(
                    _identity(b"channel-prediction-witness-v1", [
                        ticket.ticket, observed.identity, difference]), ticket.ticket,
                    observed.source_roots[-1], difference, ticket.cue_recipe,
                    ticket.target_recipe, ticket.target_channel, observed.identity,
                    ticket.hypothesis_root, roots)
                if len(self.prediction_witnesses) >= MAX_PREDICTION_WITNESSES:
                    raise GroundingRefuse("grounding:prediction_witness_bound")
                self.prediction_witnesses.append(witness)
            else:
                roots = tuple(sorted(set(ticket.source_roots + observed.source_roots)))
                witness = PredictionWitnessV1(
                    _identity(b"channel-prediction-witness-v1", [
                        ticket.ticket, observed.identity, -1]), ticket.ticket,
                    observed.source_roots[-1], -1, ticket.cue_recipe,
                    ticket.target_recipe, ticket.target_channel, observed.identity,
                    ticket.hypothesis_root, roots)
                if len(self.prediction_witnesses) >= MAX_PREDICTION_WITNESSES:
                    raise GroundingRefuse("grounding:prediction_witness_bound")
                self.prediction_witnesses.append(witness)
            settled.append(ticket.ticket); self._charge(2)
        for ticket in settled:
            self.pending.pop(ticket, None)

    def _expire_predictions(self, observed_sample):
        expired = []
        for ticket in tuple(self.pending.values()):
            if self.next_sequence <= ticket.deadline_sequence:
                continue
            roots = tuple(sorted(set(ticket.source_roots + (observed_sample.source,))))
            witness = PredictionWitnessV1(
                _identity(b"channel-prediction-timeout-v1", [
                    ticket.ticket, observed_sample.identity, -1]), ticket.ticket,
                observed_sample.source, -1, ticket.cue_recipe, ticket.target_recipe,
                ticket.target_channel, 0, ticket.hypothesis_root, roots)
            if len(self.prediction_witnesses) >= MAX_PREDICTION_WITNESSES:
                raise GroundingRefuse("grounding:prediction_witness_bound")
            self.prediction_witnesses.append(witness)
            expired.append(ticket.ticket); self._charge(2)
        for ticket in expired:
            self.pending.pop(ticket, None)

    def _recipe(self, identity):
        rows = [row for row in self.sequence_recipes.values() if row.identity == int(identity)]
        if len(rows) != 1:
            raise GroundingRefuse("grounding:sequence_recipe")
        return rows[0]

    def _refresh_cross_recipes(self):
        groups = {}
        for row in self.prediction_witnesses:
            if set(row.source_roots).intersection(self.withdrawn_sources):
                continue
            try:
                cue = self._recipe(row.cue_recipe); target = self._recipe(row.target_recipe)
            except GroundingRefuse:
                continue
            key = (row.cue_recipe, row.target_recipe, cue.channel, target.channel)
            groups.setdefault(key, []).append(row)
        recipes = {}
        for key, rows in groups.items():
            sources = {row.source for row in rows if row.difference > 0}
            credit = sum(row.difference for row in rows)
            if len(sources) < 2 or credit <= 0:
                continue
            identity = _identity(b"cross-channel-recipe-v1", key)
            recipes[identity] = CrossChannelRecipeV1(
                identity, key[0], key[1], key[2], key[3], len(sources), credit,
                tuple(sorted(row.identity for row in rows if row.difference > 0)),
                tuple(sorted(set(source for row in rows for source in row.source_roots))))
        if len(recipes) > MAX_CROSS_RECIPES:
            raise GroundingRefuse("grounding:cross_recipe_bound")
        self.cross_recipes = recipes

    def ingest_withdrawal(self, contact: WithdrawalContactV1) -> None:
        before = self._snapshot()
        try:
            self.work = 0; self._verify(CONTACT_WITHDRAWAL, contact)
            if contact.source <= 0 or contact.channel <= 0 or contact.target_source <= 0:
                raise GroundingRefuse("grounding:withdrawal_shape")
            target = contact.target_source; self.next_sequence += 1
            if (target not in self.withdrawn_sources
                    and len(self.withdrawn_sources) >= MAX_WITHDRAWN_SOURCES):
                raise GroundingRefuse("grounding:withdrawn_source_bound")
            self.withdrawn_sources.add(target)
            removed_samples = {row.identity for row in self.samples if row.source == target}
            self.samples = [row for row in self.samples if row.identity not in removed_samples]
            removed_sequences = {row.identity for row in self.sequence_occurrences
                                 if target in row.source_roots}
            self.sequence_occurrences = [row for row in self.sequence_occurrences
                                         if row.identity not in removed_sequences]
            self.pending = {key: row for key, row in self.pending.items()
                            if target not in row.source_roots}
            self.prediction_witnesses = [row for row in self.prediction_witnesses
                                         if target not in row.source_roots]
            self._rebuild_derived()
            recursive_withdrawal = self._recursive_authority.seal_withdrawal(
                self._recursive.session, self._recursive.next_sequence,
                contact.source, contact.channel, target)
            self._recursive.ingest_withdrawal(recursive_withdrawal)
            self._append_trace(contact.sequence, EVENT_WITHDRAWAL, target,
                               len(removed_samples), len(removed_sequences))
            self._record(self._contact_event(EVENT_WITHDRAWAL, contact))
            self._bounded(); self._check_checkpoint_bound()
        except Exception:
            self._rollback(before); raise

    def unfold(self) -> GroundedUnfoldV1:
        before = self._snapshot()
        try:
            if self.pending:
                raise GroundingRefuse("grounding:pending_prediction")
            self.work = 0
            if not self.sequence_occurrences:
                raise GroundingRefuse("grounding:no_sequence_cue")
            cue = self.sequence_occurrences[-1]
            cue_recipe = self.sequence_recipes.get((cue.channel, cue.sequence_hash))
            candidates = ([] if cue_recipe is None else [row for row in self.cross_recipes.values()
                          if row.cue_recipe == cue_recipe.identity])
            if not candidates:
                raise GroundingRefuse("grounding:no_cross_channel_recipe")
            peak = max(row.credit for row in candidates)
            winners = [row for row in candidates if row.credit == peak]
            if len(winners) != 1:
                raise GroundingRefuse("grounding:ambiguous")
            cross = winners[0]
            witness_rows = [row for row in self.prediction_witnesses
                            if row.identity in cross.witness_roots and row.difference > 0]
            selected_witness = min(witness_rows, key=lambda row: row.identity)
            occurrences = [self._sequence_occurrence(row.observed_occurrence)
                           for row in witness_rows]
            unit_rows = {self._sequence_units(row) for row in occurrences}
            if len(unit_rows) != 1:
                raise GroundingRefuse("grounding:target_witness_ambiguity")
            units = next(iter(unit_rows))
            target = self._sequence_occurrence(selected_witness.observed_occurrence)
            recursive_contact = self._recursive_authority.seal_occurrence(
                self._recursive.session, self._recursive.next_sequence,
                target.source_roots[-1], target.channel, (cross.identity,),
                tuple(sorted(set((*target.sample_roots, cross.identity))))[:32])
            recursive_occurrence = self._recursive.ingest_occurrence(recursive_contact)
            prediction_root = selected_witness.identity
            samples = [self._sample(identity) for identity in target.sample_roots]
            ancestry = tuple(LeafAncestryV1(
                offset, sample.unit, sample.contact_root, cross.target_recipe,
                cross.identity, prediction_root, recursive_occurrence.identity)
                for offset, sample in enumerate(samples))
            result = GroundedUnfoldV1(_identity(b"grounded-unfold-v1", [
                cue.identity, target.identity, cross.identity, recursive_occurrence.identity]),
                cue.identity, target.identity, cross.identity, units, ancestry,
                recursive_occurrence.identity)
            self._charge(WIDTH + len(witness_rows))
            self._append_trace(self.next_sequence, EVENT_UNFOLD, result.identity,
                               result.recursive_occurrence)
            self._record(EventV1(EVENT_UNFOLD, (result.identity,
                result.cross_channel_recipe, result.recursive_occurrence)))
            self._bounded(); self._check_checkpoint_bound(); return result
        except Exception:
            self._rollback(before); raise

    def _sample(self, identity):
        row = next((row for row in self.samples if row.identity == int(identity)), None)
        if row is None: raise GroundingRefuse("grounding:sample")
        return row

    def _sequence_occurrence(self, identity):
        row = next((row for row in self.sequence_occurrences
                    if row.identity == int(identity)), None)
        if row is None: raise GroundingRefuse("grounding:sequence_occurrence")
        return row

    def _sequence_units(self, occurrence):
        return tuple(self._sample(identity).unit for identity in occurrence.sample_roots)

    def _bounded(self):
        if (len(self.samples) > MAX_SAMPLES
                or len(self.sequence_occurrences) > MAX_SEQUENCE_OCCURRENCES
                or len(self.sequence_recipes) > MAX_SEQUENCE_RECIPES
                or len(self.hypotheses) > MAX_HYPOTHESES or len(self.pending) > MAX_PENDING
                or len(self.prediction_witnesses) > MAX_PREDICTION_WITNESSES
                or len(self.cross_recipes) > MAX_CROSS_RECIPES
                or len(self.withdrawn_sources) > MAX_WITHDRAWN_SOURCES
                or len(self.events) > MAX_EVENTS or len(self.trace) > MAX_TRACE
                or self.work > self.work_limit):
            raise GroundingRefuse("grounding:state_bound")
        for rows in (self.samples, self.sequence_occurrences,
                     tuple(self.sequence_recipes.values()), tuple(self.hypotheses.values()),
                     tuple(self.pending.values()), self.prediction_witnesses,
                     tuple(self.cross_recipes.values()), self.events, self.trace):
            _strict_numeric(tuple(rows))

    def _checkpoint_body(self):
        return {"schema": SCHEMA_VERSION, "session": self.session,
                "incarnation": self.incarnation, "work_limit": self.work_limit,
                "events": [asdict(row) for row in self.events]}

    def checkpoint(self) -> bytes:
        body = self._checkpoint_body()
        blob = _canonical({"version": CHECKPOINT_VERSION, "body": body,
                           "hmac": self._boundary._checkpoint_tag(body)})
        if len(blob) > MAX_CHECKPOINT_BYTES:
            raise GroundingRefuse("grounding:checkpoint_bound")
        return blob

    def _check_checkpoint_bound(self):
        self.checkpoint()

    @classmethod
    def restore(cls, blob: bytes, boundary: ReferenceChannelSequenceBoundaryV1):
        if len(bytes(blob)) > MAX_CHECKPOINT_BYTES:
            raise GroundingRefuse("grounding:checkpoint_bound")
        try:
            envelope = json.loads(bytes(blob)); body = envelope["body"]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise GroundingRefuse("grounding:checkpoint") from exc
        if (not isinstance(envelope, dict) or set(envelope) != {"version", "body", "hmac"}
                or envelope["version"] != CHECKPOINT_VERSION
                or not boundary._valid_checkpoint(body, envelope["hmac"])):
            raise GroundingRefuse("grounding:checkpoint_authentication")
        if (not isinstance(body, dict)
                or set(body) != {"schema", "session", "incarnation", "work_limit", "events"}
                or body["schema"] != SCHEMA_VERSION or body["incarnation"] != 1
                or len(body["events"]) > MAX_EVENTS):
            raise GroundingRefuse("grounding:checkpoint_schema")
        out = cls(boundary, int(body["session"]), int(body["work_limit"]))
        for raw in body["events"]:
            if not isinstance(raw, dict) or set(raw) != {"kind", "values"}:
                raise GroundingRefuse("grounding:event_shape")
            event = EventV1(int(raw["kind"]), _tuplify(raw["values"])); before = len(out.events)
            if event.kind == EVENT_SAMPLE:
                fields, tag = event.values
                out.ingest_sample(OccurrenceContactV1(int(fields[0]), int(fields[1]),
                    int(fields[2]), int(fields[3]), tuple(map(int, fields[4])),
                    tuple(map(int, fields[5])), int(tag)))
            elif event.kind == EVENT_TICK: out.tick()
            elif event.kind == EVENT_WITHDRAWAL:
                fields, tag = event.values
                out.ingest_withdrawal(WithdrawalContactV1(
                    *(int(value) for value in fields), int(tag)))
            elif event.kind == EVENT_UNFOLD: out.unfold()
            else: raise GroundingRefuse("grounding:event_kind")
            if len(out.events) != before + 1 or out.events[-1] != event:
                raise GroundingRefuse("grounding:event_replay")
        if out.checkpoint() != bytes(blob):
            raise GroundingRefuse("grounding:checkpoint_noncanonical")
        return out


def _tuplify(value):
    return tuple(_tuplify(item) for item in value) if isinstance(value, list) else int(value)
