#!/usr/bin/env python3
"""Resident variable-span recruitment from authenticated scalar chronology.

The boundary admits one opaque integer at a time.  It has no span, boundary,
word, candidate, winner, or outcome API.  The resident enumerates every bounded
suffix span, compares its online next-unit prediction with the channel marginal,
and keeps a compact Recipe only after cross-source recurrence, positive storage
recurrence margin, and later authenticated predictive advantage. Recipes contain hashes and
witness roots, never the units they transiently unfold.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from reference_resident_channel_sequence_grounding_v1 import (
    ReferenceChannelSequenceBoundaryV1,
)
from reference_resident_recursive_frontier_v1 import (
    CONTACT_OCCURRENCE,
    CONTACT_WITHDRAWAL,
    OccurrenceContactV1,
    WithdrawalContactV1,
)


SCHEMA_VERSION = CHECKPOINT_VERSION = 1
MIN_SPAN, MAX_SPAN = 2, 8
PREDICTION_HORIZON = 12
RECIPE_HEADER_CELLS = 4
MAX_SAMPLES = 576
MAX_DEFERRED_SAMPLES = 64
MAX_SPAN_OCCURRENCES = 4096
MAX_PENDING = 64
MAX_PREDICTION_WITNESSES = 2048
MAX_RECIPES = 256
MAX_WITHDRAWN_SOURCES = 256
MAX_EVENTS = MAX_TRACE = 4096
MAX_WORK = 32768
MAX_CHECKPOINT_BYTES = 2 << 20

EVENT_SAMPLE = 1
EVENT_TICK = 2
EVENT_WITHDRAWAL = 3
EVENT_UNFOLD = 4


class VariableSpanRefuse(RuntimeError):
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
        raise VariableSpanRefuse(f"span:type:{path}")
    if isinstance(value, int):
        if not -(1 << 63) <= value < (1 << 63):
            raise VariableSpanRefuse(f"span:integer:{path}")
        return
    if hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            _strict_numeric(getattr(value, name), f"{path}.{name}", extent)
        return
    if isinstance(value, (tuple, list)):
        if len(value) > extent:
            raise VariableSpanRefuse(f"span:bound:{path}")
        for index, item in enumerate(value):
            _strict_numeric(item, f"{path}[{index}]", extent)
        return
    raise VariableSpanRefuse(f"span:type:{path}")


@dataclass(frozen=True)
class EventV1:
    kind: int
    values: tuple


@dataclass(frozen=True)
class ScalarOccurrenceV1:
    identity: int
    contact_sequence: int
    source: int
    channel: int
    unit: int
    provenance: tuple[int, ...]
    contact_root: int


@dataclass(frozen=True)
class SpanOccurrenceV1:
    identity: int
    born_sequence: int
    source: int
    channel: int
    length: int
    span_hash: int
    sample_roots: tuple[int, ...]


@dataclass(frozen=True)
class SpanPredictionTicketV1:
    ticket: int
    incarnation: int
    opened_sequence: int
    deadline_sequence: int
    channel: int
    context_length: int
    context_span_hash: int
    predicted_unit_hash: int
    marginal_unit_hash: int
    cue_occurrence: int
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class SpanPredictionWitnessV1:
    identity: int
    ticket: int
    source: int
    difference: int
    observed_sample: int
    settlement_trigger_sample: int
    channel: int
    context_length: int
    context_span_hash: int
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class VariableSpanRecipeV1:
    identity: int
    channel: int
    length: int
    span_hash: int
    support: int
    retention_margin: int
    prediction_gain: int
    occurrence_witness_roots: tuple[int, ...]
    prediction_witness_roots: tuple[int, ...]
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class SpanLeafAncestryV1:
    offset: int
    unit: int
    raw_contact_root: int
    span_occurrence_root: int
    span_recipe_root: int
    prediction_witness_root: int


@dataclass(frozen=True)
class VariableSpanUnfoldV1:
    identity: int
    cue_occurrence: int
    recipe: int
    units: tuple[int, ...]
    ancestry: tuple[SpanLeafAncestryV1, ...]


class ResidentVariableSpanV1:
    """Bounded resident span census, prediction, promotion, and unfold."""

    def __init__(self, boundary: ReferenceChannelSequenceBoundaryV1,
                 session: int = 1, work_limit: int = MAX_WORK):
        if not isinstance(boundary, ReferenceChannelSequenceBoundaryV1):
            raise VariableSpanRefuse("span:boundary")
        if (isinstance(session, bool) or not isinstance(session, int) or session <= 0
                or isinstance(work_limit, bool) or not isinstance(work_limit, int)
                or not 1 <= work_limit <= MAX_WORK):
            raise VariableSpanRefuse("span:configuration")
        self._boundary = boundary
        self.session, self.incarnation = int(session), 1
        self.next_sequence = self.next_ticket = 1
        self.work, self.work_limit = 0, int(work_limit)
        self.events: list[EventV1] = []
        self.samples: list[ScalarOccurrenceV1] = []
        self.deferred_sample_roots: list[int] = []
        self.span_occurrences: list[SpanOccurrenceV1] = []
        self.pending: dict[int, SpanPredictionTicketV1] = {}
        self.prediction_witnesses: list[SpanPredictionWitnessV1] = []
        self.recipes: dict[int, VariableSpanRecipeV1] = {}
        self.withdrawn_sources: set[int] = set()
        self.trace: list[tuple[int, ...]] = []

    def _snapshot(self):
        return {key: (value.copy() if isinstance(value, (list, dict, set))
                      else value)
                for key, value in self.__dict__.items() if key != "_boundary"}

    def _rollback(self, state):
        boundary = self._boundary
        self.__dict__.clear(); self._boundary = boundary; self.__dict__.update(state)

    def _charge(self, amount):
        self.work += int(amount)
        if self.work > self.work_limit:
            raise VariableSpanRefuse("span:resource")

    def _record(self, event):
        _strict_numeric(event)
        if len(self.events) >= MAX_EVENTS:
            raise VariableSpanRefuse("span:event_bound")
        self.events.append(event)

    def _append_trace(self, *values):
        if len(self.trace) >= MAX_TRACE:
            raise VariableSpanRefuse("span:trace_bound")
        self.trace.append(tuple(map(int, values)))

    def _verify(self, kind, contact):
        _strict_numeric(contact)
        if contact.session != self.session or contact.sequence != self.next_sequence:
            raise VariableSpanRefuse("span:session_sequence")
        if not self._boundary._valid_contact(kind, contact):
            raise VariableSpanRefuse("span:authentication")

    @staticmethod
    def _contact_event(kind, contact):
        return EventV1(kind, (contact.signed_fields(), contact.auth_tag))

    def _sample(self, identity):
        row = next((row for row in self.samples if row.identity == int(identity)), None)
        if row is None:
            raise VariableSpanRefuse("span:sample")
        return row

    def _occurrence(self, identity):
        row = next((row for row in self.span_occurrences
                    if row.identity == int(identity)), None)
        if row is None:
            raise VariableSpanRefuse("span:occurrence")
        return row

    def _units(self, occurrence):
        return tuple(self._sample(root).unit for root in occurrence.sample_roots)

    @staticmethod
    def _unit_hash(unit):
        return _identity(b"variable-span-unit-v1", [int(unit)])

    def _matching_groups(self):
        groups = {}
        for row in self.span_occurrences:
            if row.source in self.withdrawn_sources:
                continue
            self._charge(1)
            groups.setdefault((row.channel, row.length, row.span_hash), []).append(row)
        for rows in groups.values():
            units = {self._units(row) for row in rows}
            if len(units) != 1:
                raise VariableSpanRefuse("span:hash_collision")
        return groups

    @staticmethod
    def _unique_peak(counts):
        if not counts:
            return None
        peak = max(counts.values())
        winners = [key for key, value in counts.items() if value == peak]
        return winners[0] if len(winners) == 1 else None

    def _next_sample(self, occurrence):
        return next((row for row in self.samples
                     if row.channel == occurrence.channel
                     and row.source == occurrence.source
                     and row.contact_sequence > occurrence.born_sequence), None)

    def _settle(self, observed):
        settled = []
        observed_hash = self._unit_hash(observed.unit)
        for ticket in tuple(self.pending.values()):
            if ticket.channel != observed.channel:
                continue
            in_time = observed.contact_sequence <= ticket.deadline_sequence
            difference = ((1 if in_time and observed_hash == ticket.predicted_unit_hash else 0)
                          - (1 if in_time and observed_hash == ticket.marginal_unit_hash else 0))
            roots = tuple(sorted(set(ticket.source_roots + (observed.source,))))
            witness = SpanPredictionWitnessV1(
                _identity(b"variable-span-prediction-witness-v1", [
                    ticket.ticket, observed.identity, difference]),
                ticket.ticket, observed.source, difference, observed.identity,
                observed.identity,
                ticket.channel, ticket.context_length, ticket.context_span_hash, roots)
            if len(self.prediction_witnesses) >= MAX_PREDICTION_WITNESSES:
                raise VariableSpanRefuse("span:prediction_witness_bound")
            self.prediction_witnesses.append(witness)
            settled.append(ticket.ticket); self._charge(1)
        for ticket in settled:
            self.pending.pop(ticket, None)

    def _expire(self, observed):
        expired = []
        for ticket in tuple(self.pending.values()):
            if observed.contact_sequence <= ticket.deadline_sequence:
                continue
            witness = SpanPredictionWitnessV1(
                _identity(b"variable-span-prediction-timeout-v1", [
                    ticket.ticket, observed.identity]),
                ticket.ticket, 0, 0, 0, observed.identity, ticket.channel,
                ticket.context_length, ticket.context_span_hash,
                ticket.source_roots)
            if len(self.prediction_witnesses) >= MAX_PREDICTION_WITNESSES:
                raise VariableSpanRefuse("span:prediction_witness_bound")
            self.prediction_witnesses.append(witness)
            expired.append(ticket.ticket); self._charge(1)
        for ticket in expired:
            self.pending.pop(ticket, None)

    def _form_suffixes(self, sample):
        rows = [row for row in self.samples if row.channel == sample.channel
                and row.contact_sequence <= sample.contact_sequence]
        made = []
        for length in range(MIN_SPAN, min(MAX_SPAN, len(rows)) + 1):
            members = tuple(rows[-length:])
            self._charge(length)
            if len({row.source for row in members}) != 1:
                continue
            units = tuple(row.unit for row in members)
            span_hash = _identity(b"variable-span-content-v1", units)
            occurrence = SpanOccurrenceV1(
                _identity(b"variable-span-occurrence-v1", [
                    sample.channel, tuple(row.identity for row in members)]),
                sample.contact_sequence, sample.source, sample.channel, length,
                span_hash, tuple(row.identity for row in members))
            if len(self.span_occurrences) >= MAX_SPAN_OCCURRENCES:
                raise VariableSpanRefuse("span:occurrence_bound")
            self.span_occurrences.append(occurrence); made.append(occurrence)
        return tuple(made)

    def _drain_deferred(self, extra=()):
        roots = tuple(self.deferred_sample_roots) + tuple(extra)
        self.deferred_sample_roots.clear()
        made = []
        for root in roots:
            made.extend(self._form_suffixes(self._sample(root)))
        self._rebuild_recipes()
        return tuple(made)

    def _rebuild_recipes(self):
        groups = self._matching_groups()
        recipes = {}
        for key, rows in groups.items():
            sources = tuple(sorted({row.source for row in rows}))
            if len(sources) < 2:
                continue
            witnesses = [row for row in self.prediction_witnesses
                         if (row.channel, row.context_length, row.context_span_hash) == key
                         and not set(row.source_roots).intersection(self.withdrawn_sources)]
            positive_sources = tuple(sorted({row.source for row in witnesses
                                             if row.difference > 0}))
            prediction_gain = sum(row.difference for row in witnesses)
            if len(positive_sources) < 2 or prediction_gain <= 0:
                continue
            occurrence_roots = tuple(min(row.identity for row in rows if row.source == source)
                                     for source in sources)
            prediction_roots = tuple(min(row.identity for row in witnesses
                                         if row.source == source and row.difference > 0)
                                     for source in positive_sources)
            all_sources = tuple(sorted(set(sources + positive_sources)))
            compact_cells = (RECIPE_HEADER_CELLS + len(rows) + len(occurrence_roots)
                             + len(prediction_roots) + len(all_sources))
            retention_margin = len(rows) * key[1] - compact_cells
            if retention_margin <= 0:
                continue
            identity = _identity(b"variable-span-recipe-v1", key)
            recipes[identity] = VariableSpanRecipeV1(
                identity, key[0], key[1], key[2], len(rows), retention_margin,
                prediction_gain, occurrence_roots, prediction_roots, all_sources)
        if len(recipes) > MAX_RECIPES:
            raise VariableSpanRefuse("span:recipe_bound")
        self.recipes = recipes

    def ingest_sample(self, contact: OccurrenceContactV1) -> ScalarOccurrenceV1:
        before = self._snapshot()
        try:
            self.work = 0; self._verify(CONTACT_OCCURRENCE, contact)
            if (contact.source <= 0 or contact.channel <= 0
                    or contact.source in self.withdrawn_sources
                    or len(contact.features) != 1
                    or isinstance(contact.features[0], bool)
                    or not isinstance(contact.features[0], int)
                    or len(contact.provenance) > MAX_SPAN * 4
                    or any(root <= 0 for root in contact.provenance)):
                raise VariableSpanRefuse("span:sample_shape")
            if len(self.samples) >= MAX_SAMPLES:
                raise VariableSpanRefuse("span:sample_bound")
            root = _identity(b"variable-span-contact-v1", contact.signed_fields())
            sample = ScalarOccurrenceV1(
                _identity(b"variable-span-sample-v1", [root, contact.sequence]),
                contact.sequence, contact.source, contact.channel,
                contact.features[0], contact.provenance, root)
            self._settle(sample)
            self.samples.append(sample); self.next_sequence += 1
            self._expire(sample)
            if self.pending:
                if len(self.deferred_sample_roots) >= MAX_DEFERRED_SAMPLES:
                    raise VariableSpanRefuse("span:deferred_bound")
                self.deferred_sample_roots.append(sample.identity); made = ()
            else:
                made = self._drain_deferred((sample.identity,))
            self._append_trace(contact.sequence, EVENT_SAMPLE, sample.identity,
                               *(row.identity for row in made))
            self._record(self._contact_event(EVENT_SAMPLE, contact))
            self._bounded(); self._check_checkpoint_bound(); return sample
        except Exception:
            self._rollback(before); raise

    def tick(self) -> tuple[SpanPredictionTicketV1, ...]:
        before = self._snapshot()
        try:
            if self.pending:
                raise VariableSpanRefuse("span:pending_prediction")
            self.work = 0
            if not self.samples:
                raise VariableSpanRefuse("span:no_cue")
            newest = self.samples[-1]
            groups = self._matching_groups()
            baselines = {}
            for row in self.samples[:-1]:
                if row.channel == newest.channel:
                    unit_hash = self._unit_hash(row.unit)
                    baselines[unit_hash] = baselines.get(unit_hash, 0) + 1
                    self._charge(1)
            marginal = self._unique_peak(baselines)
            if marginal is None:
                raise VariableSpanRefuse("span:ambiguous_marginal")
            cues = [row for row in self.span_occurrences
                    if row.born_sequence == newest.contact_sequence]
            tickets = []
            for cue in sorted(cues, key=lambda row: (row.length, row.span_hash)):
                rows = groups.get((cue.channel, cue.length, cue.span_hash), ())
                sources = tuple(sorted({row.source for row in rows}))
                if len(sources) < 2:
                    continue
                counts = {}
                for row in rows:
                    following = self._next_sample(row)
                    if following is None:
                        continue
                    unit_hash = self._unit_hash(following.unit)
                    counts[unit_hash] = counts.get(unit_hash, 0) + 1
                    self._charge(1)
                predicted = self._unique_peak(counts)
                if predicted is None or predicted == marginal:
                    continue
                if len(self.pending) + len(tickets) >= MAX_PENDING:
                    raise VariableSpanRefuse("span:pending_bound")
                ticket = SpanPredictionTicketV1(
                    self.next_ticket, self.incarnation, self.next_sequence,
                    self.next_sequence + PREDICTION_HORIZON, cue.channel, cue.length,
                    cue.span_hash, predicted, marginal, cue.identity, sources)
                self.next_ticket += 1; tickets.append(ticket); self._charge(2)
            if not tickets:
                raise VariableSpanRefuse("span:no_prediction")
            for ticket in tickets:
                self.pending[ticket.ticket] = ticket
            self._append_trace(self.next_sequence, EVENT_TICK,
                               *(row.ticket for row in tickets))
            self._record(EventV1(EVENT_TICK, (tuple(row.ticket for row in tickets),)))
            self._bounded(); self._check_checkpoint_bound(); return tuple(tickets)
        except Exception:
            self._rollback(before); raise

    def ingest_withdrawal(self, contact: WithdrawalContactV1) -> None:
        before = self._snapshot()
        try:
            self.work = 0; self._verify(CONTACT_WITHDRAWAL, contact)
            if contact.source <= 0 or contact.channel <= 0 or contact.target_source <= 0:
                raise VariableSpanRefuse("span:withdrawal_shape")
            target = contact.target_source; self.next_sequence += 1
            if (target not in self.withdrawn_sources
                    and len(self.withdrawn_sources) >= MAX_WITHDRAWN_SOURCES):
                raise VariableSpanRefuse("span:withdrawn_source_bound")
            self.withdrawn_sources.add(target)
            sample_roots = {row.identity for row in self.samples if row.source == target}
            self.samples = [row for row in self.samples if row.identity not in sample_roots]
            self.deferred_sample_roots = [root for root in self.deferred_sample_roots
                                          if root not in sample_roots]
            occurrence_roots = {row.identity for row in self.span_occurrences
                                if row.source == target
                                or set(row.sample_roots).intersection(sample_roots)}
            self.span_occurrences = [row for row in self.span_occurrences
                                     if row.identity not in occurrence_roots]
            self.pending = {key: row for key, row in self.pending.items()
                            if target not in row.source_roots}
            self.prediction_witnesses = [row for row in self.prediction_witnesses
                                         if target not in row.source_roots
                                         and row.source != target]
            self.recipes = {key: row for key, row in self.recipes.items()
                            if target not in row.source_roots}
            if not self.pending:
                self._drain_deferred()
            self._append_trace(contact.sequence, EVENT_WITHDRAWAL, target,
                               len(sample_roots), len(occurrence_roots))
            self._record(self._contact_event(EVENT_WITHDRAWAL, contact))
            self._bounded(); self._check_checkpoint_bound()
        except Exception:
            self._rollback(before); raise

    def unfold(self) -> VariableSpanUnfoldV1:
        before = self._snapshot()
        try:
            if self.pending:
                raise VariableSpanRefuse("span:pending_prediction")
            self.work = 0
            if not self.samples:
                raise VariableSpanRefuse("span:no_cue")
            newest = self.samples[-1]
            occurrences = {row.identity: row for row in self.span_occurrences
                           if row.born_sequence == newest.contact_sequence}
            candidates = [(recipe, occurrence) for recipe in self.recipes.values()
                          for occurrence in occurrences.values()
                          if (recipe.channel, recipe.length, recipe.span_hash)
                          == (occurrence.channel, occurrence.length, occurrence.span_hash)]
            if not candidates:
                raise VariableSpanRefuse("span:no_recipe")
            peak = max((row[0].prediction_gain, row[0].retention_margin)
                       for row in candidates)
            winners = [row for row in candidates
                       if (row[0].prediction_gain, row[0].retention_margin) == peak]
            if len(winners) != 1:
                raise VariableSpanRefuse("span:ambiguous")
            recipe, occurrence = winners[0]
            units = self._units(occurrence)
            if (_identity(b"variable-span-content-v1", units) != recipe.span_hash
                    or len(units) != recipe.length):
                raise VariableSpanRefuse("span:witness_mismatch")
            for root in recipe.occurrence_witness_roots:
                witness = self._occurrence(root)
                if self._units(witness) != units:
                    raise VariableSpanRefuse("span:witness_ambiguity")
            prediction_root = min(recipe.prediction_witness_roots)
            samples = [self._sample(root) for root in occurrence.sample_roots]
            ancestry = tuple(SpanLeafAncestryV1(
                offset, sample.unit, sample.contact_root, occurrence.identity,
                recipe.identity, prediction_root)
                for offset, sample in enumerate(samples))
            result = VariableSpanUnfoldV1(
                _identity(b"variable-span-unfold-v1", [
                    occurrence.identity, recipe.identity, prediction_root]),
                occurrence.identity, recipe.identity, units, ancestry)
            self._charge(len(units) + len(recipe.occurrence_witness_roots))
            self._append_trace(self.next_sequence, EVENT_UNFOLD, result.identity,
                               result.recipe)
            self._record(EventV1(EVENT_UNFOLD, (result.identity, result.recipe)))
            self._bounded(); self._check_checkpoint_bound(); return result
        except Exception:
            self._rollback(before); raise

    def _bounded(self):
        if (len(self.samples) > MAX_SAMPLES
                or len(self.deferred_sample_roots) > MAX_DEFERRED_SAMPLES
                or len(self.span_occurrences) > MAX_SPAN_OCCURRENCES
                or len(self.pending) > MAX_PENDING
                or len(self.prediction_witnesses) > MAX_PREDICTION_WITNESSES
                or len(self.recipes) > MAX_RECIPES
                or len(self.withdrawn_sources) > MAX_WITHDRAWN_SOURCES
                or len(self.events) > MAX_EVENTS or len(self.trace) > MAX_TRACE
                or self.work > self.work_limit):
            raise VariableSpanRefuse("span:state_bound")
        for rows in (self.samples, tuple(self.deferred_sample_roots),
                     self.span_occurrences, tuple(self.pending.values()),
                     self.prediction_witnesses, tuple(self.recipes.values()),
                     self.events, self.trace):
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
            raise VariableSpanRefuse("span:checkpoint_bound")
        return blob

    def _check_checkpoint_bound(self):
        self.checkpoint()

    @classmethod
    def restore(cls, blob: bytes, boundary: ReferenceChannelSequenceBoundaryV1):
        if not isinstance(boundary, ReferenceChannelSequenceBoundaryV1):
            raise VariableSpanRefuse("span:boundary")
        if isinstance(blob, str):
            raise VariableSpanRefuse("span:checkpoint_bound")
        try:
            raw_blob = bytes(blob)
            if len(raw_blob) > MAX_CHECKPOINT_BYTES:
                raise VariableSpanRefuse("span:checkpoint_bound")
            envelope = json.loads(raw_blob); body = envelope["body"]
        except VariableSpanRefuse:
            raise
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise VariableSpanRefuse("span:checkpoint") from exc
        if (not isinstance(envelope, dict)
                or set(envelope) != {"version", "body", "hmac"}
                or envelope["version"] != CHECKPOINT_VERSION
                or not boundary._valid_checkpoint(body, envelope["hmac"])):
            raise VariableSpanRefuse("span:checkpoint_authentication")
        if (not isinstance(body, dict)
                or set(body) != {"schema", "session", "incarnation", "work_limit", "events"}
                or body["schema"] != SCHEMA_VERSION or body["incarnation"] != 1
                or any(isinstance(body[key], bool) or not isinstance(body[key], int)
                       for key in ("schema", "session", "incarnation", "work_limit"))
                or not isinstance(body["events"], list)
                or len(body["events"]) > MAX_EVENTS):
            raise VariableSpanRefuse("span:checkpoint_schema")
        out = cls(boundary, body["session"], body["work_limit"])
        for raw in body["events"]:
            if not isinstance(raw, dict) or set(raw) != {"kind", "values"}:
                raise VariableSpanRefuse("span:event_shape")
            event = EventV1(int(raw["kind"]), _tuplify(raw["values"])); before = len(out.events)
            if event.kind == EVENT_SAMPLE:
                fields, tag = event.values
                out.ingest_sample(OccurrenceContactV1(
                    int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]),
                    tuple(map(int, fields[4])), tuple(map(int, fields[5])), int(tag)))
            elif event.kind == EVENT_TICK:
                tickets = out.tick()
                if tuple(row.ticket for row in tickets) != event.values[0]:
                    raise VariableSpanRefuse("span:event_replay")
            elif event.kind == EVENT_WITHDRAWAL:
                fields, tag = event.values
                out.ingest_withdrawal(WithdrawalContactV1(
                    *(int(value) for value in fields), int(tag)))
            elif event.kind == EVENT_UNFOLD:
                result = out.unfold()
                if (result.identity, result.recipe) != event.values:
                    raise VariableSpanRefuse("span:event_replay")
            else:
                raise VariableSpanRefuse("span:event_kind")
            if len(out.events) != before + 1 or out.events[-1] != event:
                raise VariableSpanRefuse("span:event_replay")
        if out.checkpoint() != raw_blob:
            raise VariableSpanRefuse("span:checkpoint_noncanonical")
        return out


def _tuplify(value):
    return tuple(_tuplify(item) for item in value) if isinstance(value, list) else int(value)
