#!/usr/bin/env python3
"""Resident three-span Networks and structural SLOT/CALL/SLOT rebinding.

This graph-neutral reference layer owns a variable-span resident and accepts only
its authenticated extent-one contacts.  It derives non-overlapping Networks from
chronology, never from host-provided spans, parses, children, contexts, or
templates.  A stable middle Recipe with varied outer bindings can recruit one
numeric structural constructor.  A previously unseen outer pair is rebound only
against the constructor pool that existed before the current Network is learned.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from reference_resident_channel_sequence_grounding_v1 import (
    ReferenceChannelSequenceBoundaryV1,
)
from reference_resident_recursive_frontier_v1 import (
    OccurrenceContactV1,
    WithdrawalContactV1,
)
from reference_resident_variable_span_v1 import (
    ResidentVariableSpanV1,
    VariableSpanRefuse,
)


SCHEMA_VERSION, CHECKPOINT_VERSION = 0x53504E31, 1
NETWORK_ARITY = 3
OP_SLOT, OP_CALL = 1, 2
MAX_NETWORKS = 1024
MAX_CONSTRUCTORS = 128
MAX_PROCESSED = 1024
MAX_REBINDS = 256
MAX_EVENTS = MAX_TRACE = 4096
MAX_WORK = 32768
MAX_CHECKPOINT_BYTES = 2 << 20

EVENT_SAMPLE = 1
EVENT_CLOSE = 2
EVENT_WITHDRAWAL = 3


class ParametricSpanNetworkRefuse(RuntimeError):
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
        raise ParametricSpanNetworkRefuse(f"span_network:type:{path}")
    if isinstance(value, int):
        if not -(1 << 63) <= value < (1 << 63):
            raise ParametricSpanNetworkRefuse(f"span_network:integer:{path}")
        return
    if hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            _strict_numeric(getattr(value, name), f"{path}.{name}", extent)
        return
    if isinstance(value, (tuple, list)):
        if len(value) > extent:
            raise ParametricSpanNetworkRefuse(f"span_network:bound:{path}")
        for index, item in enumerate(value):
            _strict_numeric(item, f"{path}[{index}]", extent)
        return
    raise ParametricSpanNetworkRefuse(f"span_network:type:{path}")


@dataclass(frozen=True)
class EventV1:
    kind: int
    values: tuple


@dataclass(frozen=True)
class SpanNetworkV1:
    identity: int
    born_sequence: int
    source: int
    channel: int
    member_occurrence_roots: tuple[int, ...]
    member_recipe_roots: tuple[int, ...]
    covered_units: int
    evidence: int


@dataclass(frozen=True)
class ParametricSpanConstructorV1:
    identity: int
    evidence_root: int
    arity: int
    operations: tuple[tuple[int, int], ...]
    middle_recipe: int
    support: int
    left_bindings: tuple[int, ...]
    right_bindings: tuple[int, ...]
    binding_pairs: tuple[tuple[int, int], ...]
    binding_pair_sources: tuple[tuple[int, int, tuple[int, ...]], ...]
    binding_pair_hashes: tuple[int, ...]
    witness_network_roots: tuple[int, ...]
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class ParametricSpanLeafAncestryV1:
    offset: int
    unit: int
    raw_contact_root: int
    span_occurrence_root: int
    span_recipe_root: int
    prediction_witness_root: int
    network_root: int
    constructor_root: int
    constructor_evidence_root: int
    constructor_witness_roots: tuple[int, ...]
    network_position: int
    slot: int


@dataclass(frozen=True)
class ParametricSpanRebindV1:
    identity: int
    network: int
    constructor: int
    constructor_evidence_root: int
    constructor_witness_roots: tuple[int, ...]
    left_binding: int
    right_binding: int
    units: tuple[int, ...]
    ancestry: tuple[ParametricSpanLeafAncestryV1, ...]


@dataclass(frozen=True)
class CloseResultV1:
    identity: int
    prediction_tickets: tuple[int, ...]
    processed_networks: tuple[int, ...]
    rebinds: tuple[ParametricSpanRebindV1, ...]


class ResidentParametricSpanNetworkV1:
    """One continuing scalar resident with a fixed structural close law."""

    def __init__(self, boundary: ReferenceChannelSequenceBoundaryV1,
                 session: int = 1, work_limit: int = MAX_WORK):
        if not isinstance(boundary, ReferenceChannelSequenceBoundaryV1):
            raise ParametricSpanNetworkRefuse("span_network:boundary")
        if (isinstance(session, bool) or not isinstance(session, int) or session <= 0
                or isinstance(work_limit, bool) or not isinstance(work_limit, int)
                or not 1 <= work_limit <= MAX_WORK):
            raise ParametricSpanNetworkRefuse("span_network:configuration")
        self._boundary = boundary
        self._inner = ResidentVariableSpanV1(boundary, session, work_limit)
        self.work, self.work_limit = 0, int(work_limit)
        self.events: list[EventV1] = []
        self.trace: list[tuple[int, ...]] = []
        self.networks: dict[int, SpanNetworkV1] = {}
        self.processed_networks: set[int] = set()
        self.constructors: dict[int, ParametricSpanConstructorV1] = {}
        self.rebinds: list[ParametricSpanRebindV1] = []
        self._checkpoint_event_bytes = 0

    @property
    def session(self):
        return self._inner.session

    @property
    def next_sequence(self):
        return self._inner.next_sequence

    def _snapshot(self):
        state = {key: (value.copy() if isinstance(value, (list, dict, set))
                       else value)
                 for key, value in self.__dict__.items()
                 if key not in {"_boundary", "_inner"}}
        state["_inner_state"] = self._inner._snapshot()
        return state

    def _rollback(self, state):
        boundary, inner = self._boundary, self._inner
        inner_state = state.pop("_inner_state")
        self.__dict__.clear(); self._boundary = boundary
        self.__dict__.update(state)
        self._inner = inner; self._inner._rollback(inner_state)

    def _charge(self, amount):
        self.work += int(amount)
        if self.work > self.work_limit:
            raise ParametricSpanNetworkRefuse("span_network:resource")

    def _record(self, event):
        _strict_numeric(event)
        if len(self.events) >= MAX_EVENTS:
            raise ParametricSpanNetworkRefuse("span_network:event_bound")
        encoded = len(_canonical(asdict(event)))
        if self._projected_checkpoint_size(encoded) > MAX_CHECKPOINT_BYTES:
            raise ParametricSpanNetworkRefuse("span_network:checkpoint_bound")
        self.events.append(event)
        self._checkpoint_event_bytes += encoded

    def _projected_checkpoint_size(self, added_event_bytes=0):
        empty = {"schema": SCHEMA_VERSION, "session": self.session,
                 "incarnation": 1, "work_limit": self.work_limit, "events": []}
        base = len(_canonical({"version": CHECKPOINT_VERSION, "body": empty,
                               "hmac": "0" * 64}))
        count = len(self.events) + int(bool(added_event_bytes))
        return (base + self._checkpoint_event_bytes + int(added_event_bytes)
                + max(0, count - 1))

    def _append_trace(self, *values):
        if len(self.trace) >= MAX_TRACE:
            raise ParametricSpanNetworkRefuse("span_network:trace_bound")
        self.trace.append(tuple(map(int, values)))

    @staticmethod
    def _contact_event(kind, contact):
        return EventV1(kind, (contact.signed_fields(), contact.auth_tag))

    def _active_rows(self):
        recipes = {(row.channel, row.length, row.span_hash): row
                   for row in self._inner.recipes.values()}
        samples = {row.identity: row for row in self._inner.samples}
        active = []
        for occurrence in self._inner.span_occurrences:
            recipe = recipes.get((occurrence.channel, occurrence.length,
                                  occurrence.span_hash))
            if recipe is None or any(root not in samples for root in occurrence.sample_roots):
                continue
            members = tuple(samples[root] for root in occurrence.sample_roots)
            if (not members or len({row.source for row in members}) != 1
                    or len({row.channel for row in members}) != 1):
                raise ParametricSpanNetworkRefuse("span_network:span_witness")
            units = tuple(row.unit for row in members)
            if (_identity(b"variable-span-content-v1", units) != recipe.span_hash
                    or len(units) != recipe.length):
                raise ParametricSpanNetworkRefuse("span_network:span_hash")
            active.append((occurrence, recipe, members))
            self._charge(len(members))
        return active

    def _derive_networks(self):
        active = self._active_rows()
        by_end = {}
        for occurrence, recipe, members in active:
            key_end = (occurrence.source, occurrence.channel, members[-1].identity)
            by_end.setdefault(key_end, []).append((occurrence, recipe, members))
        previous = {}
        channel_rows = {}
        for sample in self._inner.samples:
            channel_rows.setdefault((sample.source, sample.channel), []).append(sample)
        for key, rows in channel_rows.items():
            for index in range(1, len(rows)):
                previous[(key[0], key[1], rows[index].identity)] = rows[index - 1].identity

        candidates = {}
        for right, right_recipe, right_members in active:
            prior_right = previous.get((right.source, right.channel,
                                        right_members[0].identity))
            for middle, middle_recipe, middle_members in by_end.get(
                    (right.source, right.channel, prior_right), ()):
                prior_middle = previous.get((middle.source, middle.channel,
                                             middle_members[0].identity))
                for left, left_recipe, left_members in by_end.get(
                        (middle.source, middle.channel, prior_middle), ()):
                    roots = (left.identity, middle.identity, right.identity)
                    if len(set(roots)) != NETWORK_ARITY:
                        continue
                    recipe_roots = (left_recipe.identity, middle_recipe.identity,
                                    right_recipe.identity)
                    covered = left.length + middle.length + right.length
                    evidence = sum(row.prediction_gain + row.retention_margin
                                   for row in (left_recipe, middle_recipe, right_recipe))
                    network = SpanNetworkV1(
                        _identity(b"resident-parametric-span-network-v1", roots),
                        right.born_sequence, right.source, right.channel,
                        roots, recipe_roots, covered, evidence)
                    endpoint = (right.source, right.channel, right_members[-1].identity)
                    candidates.setdefault(endpoint, []).append(network)
                    self._charge(1)

        networks = {}
        for rows in candidates.values():
            peak = max((row.covered_units, row.evidence) for row in rows)
            winners = [row for row in rows
                       if (row.covered_units, row.evidence) == peak]
            unique = {row.identity: row for row in winners}
            if len(unique) != 1:
                raise ParametricSpanNetworkRefuse("span_network:network_ambiguous")
            winner = next(iter(unique.values()))
            networks[winner.identity] = winner
        if len(networks) > MAX_NETWORKS:
            raise ParametricSpanNetworkRefuse("span_network:network_bound")
        return networks

    @staticmethod
    def _pair_hash(left, right):
        return _identity(b"resident-parametric-span-binding-pair-v1", [left, right])

    @staticmethod
    def _three_corner_support(pairs, left, right):
        if (left, right) in pairs:
            return False
        return any((left, other_right) in pairs
                   and (other_left, right) in pairs
                   and (other_left, other_right) in pairs
                   for other_left, other_right in pairs
                   if other_left != left and other_right != right)

    @staticmethod
    def _cross_source_three_corner(pair_sources, left, right):
        if (left, right) in pair_sources:
            return False
        return any(len(set(pair_sources[(left, other_right)])
                       | set(pair_sources[(other_left, right)])
                       | set(pair_sources[(other_left, other_right)])) >= 2
                   for other_left, other_right in pair_sources
                   if other_left != left and other_right != right
                   and (left, other_right) in pair_sources
                   and (other_left, right) in pair_sources
                   and (other_left, other_right) in pair_sources)

    def _refresh_constructors(self):
        groups = {}
        for identity in self.processed_networks:
            row = self.networks.get(identity)
            if row is not None:
                groups.setdefault(row.member_recipe_roots[1], []).append(row)
                self._charge(1)
        constructors = {}
        for middle, rows in groups.items():
            sources = tuple(sorted({row.source for row in rows}))
            lefts = tuple(sorted({row.member_recipe_roots[0] for row in rows}))
            rights = tuple(sorted({row.member_recipe_roots[2] for row in rows}))
            pairs = {(row.member_recipe_roots[0], row.member_recipe_roots[2])
                     for row in rows}
            pair_sources = {}
            for row in rows:
                pair = (row.member_recipe_roots[0], row.member_recipe_roots[2])
                pair_sources.setdefault(pair, set()).add(row.source)
            if len(sources) < 2 or len(lefts) < 2 or len(rights) < 2 or len(pairs) < 3:
                continue
            if not any(self._cross_source_three_corner(pair_sources, left, right)
                       for left in lefts for right in rights):
                continue
            operations = ((OP_SLOT, 0), (OP_CALL, middle), (OP_SLOT, 1))
            identity = _identity(b"resident-parametric-span-constructor-v1",
                                 [NETWORK_ARITY, operations])
            pair_hashes = tuple(sorted(self._pair_hash(*pair) for pair in pairs))
            witness_roots = tuple(sorted(row.identity for row in rows))
            evidence_root = _identity(b"resident-parametric-span-constructor-evidence-v1",
                                      [pair_hashes, sorted((left, right, sorted(values))
                                       for (left, right), values in pair_sources.items()),
                                       witness_roots, sources])
            constructor = ParametricSpanConstructorV1(
                identity, evidence_root, NETWORK_ARITY, operations, middle, len(rows), lefts,
                rights, tuple(sorted(pairs)),
                tuple(sorted((left, right, tuple(sorted(values)))
                             for (left, right), values in pair_sources.items())),
                pair_hashes, witness_roots, sources)
            for row in rows:
                rebuilt = (row.member_recipe_roots[0], middle,
                           row.member_recipe_roots[2])
                if rebuilt != row.member_recipe_roots:
                    raise ParametricSpanNetworkRefuse("span_network:shadow")
            constructors[identity] = constructor
        if len(constructors) > MAX_CONSTRUCTORS:
            raise ParametricSpanNetworkRefuse("span_network:constructor_bound")
        self.constructors = constructors

    def _span_recipe(self, identity):
        rows = [row for row in self._inner.recipes.values()
                if row.identity == int(identity)]
        if len(rows) != 1:
            raise ParametricSpanNetworkRefuse("span_network:recipe")
        return rows[0]

    def _span_occurrence(self, identity):
        rows = [row for row in self._inner.span_occurrences
                if row.identity == int(identity)]
        if len(rows) != 1:
            raise ParametricSpanNetworkRefuse("span_network:occurrence")
        return rows[0]

    def _sample(self, identity):
        rows = [row for row in self._inner.samples if row.identity == int(identity)]
        if len(rows) != 1:
            raise ParametricSpanNetworkRefuse("span_network:sample")
        return rows[0]

    def _rebind(self, network, constructor):
        left, middle, right = network.member_recipe_roots
        pairs = set(constructor.binding_pairs)
        pair_sources = {(row[0], row[1]): row[2]
                        for row in constructor.binding_pair_sources}
        if (middle != constructor.middle_recipe
                or left not in constructor.left_bindings
                or right not in constructor.right_bindings
                or not self._cross_source_three_corner(pair_sources, left, right)):
            raise ParametricSpanNetworkRefuse("span_network:not_heldout")
        rebuilt = (left, constructor.middle_recipe, right)
        if rebuilt != network.member_recipe_roots:
            raise ParametricSpanNetworkRefuse("span_network:rebind_shadow")
        units, ancestry = [], []
        for position, (occurrence_root, recipe_root) in enumerate(zip(
                network.member_occurrence_roots, network.member_recipe_roots)):
            occurrence = self._span_occurrence(occurrence_root)
            recipe = self._span_recipe(recipe_root)
            samples = tuple(self._sample(root) for root in occurrence.sample_roots)
            member_units = tuple(row.unit for row in samples)
            if (_identity(b"variable-span-content-v1", member_units) != recipe.span_hash
                    or len(member_units) != recipe.length
                    or not recipe.prediction_witness_roots):
                raise ParametricSpanNetworkRefuse("span_network:unfold_witness")
            prediction_root = min(recipe.prediction_witness_roots)
            slot = (1 if position == 0 else 2 if position == 2 else 0)
            for sample in samples:
                ancestry.append(ParametricSpanLeafAncestryV1(
                    len(units), sample.unit, sample.contact_root, occurrence.identity,
                    recipe.identity, prediction_root, network.identity,
                    constructor.identity, constructor.evidence_root,
                    constructor.witness_network_roots, position, slot))
                units.append(sample.unit)
        identity = _identity(b"resident-parametric-span-rebind-v1", [
            network.identity, constructor.identity, constructor.evidence_root,
            left, right])
        return ParametricSpanRebindV1(identity, network.identity,
                                      constructor.identity, constructor.evidence_root,
                                      constructor.witness_network_roots,
                                      left, right,
                                      tuple(units), tuple(ancestry))

    def ingest_sample(self, contact: OccurrenceContactV1):
        event = self._contact_event(EVENT_SAMPLE, contact)
        self._bounded(); _strict_numeric(event)
        if len(self.events) >= MAX_EVENTS or len(self.trace) >= MAX_TRACE:
            raise ParametricSpanNetworkRefuse("span_network:state_bound")
        encoded = len(_canonical(asdict(event)))
        if self._projected_checkpoint_size(encoded) > MAX_CHECKPOINT_BYTES:
            raise ParametricSpanNetworkRefuse("span_network:checkpoint_bound")
        # Inner ingress is transactional.  Everything after it is a prebounded
        # append, so a second full inner checkpoint adds no rollback authority.
        self.work = 0
        try:
            result = self._inner.ingest_sample(contact)
        except (VariableSpanRefuse, ParametricSpanNetworkRefuse):
            raise
        except Exception as exc:
            raise ParametricSpanNetworkRefuse("span_network:sample") from exc
        self.events.append(event)
        self._checkpoint_event_bytes += encoded
        self.trace.append((contact.sequence, EVENT_SAMPLE, result.identity))
        return result

    def close(self) -> CloseResultV1:
        before = self._snapshot()
        try:
            if self._inner.pending:
                raise ParametricSpanNetworkRefuse("span_network:pending_prediction")
            self.work = 0
            self.networks = self._derive_networks()
            self.processed_networks.intersection_update(self.networks)
            self._refresh_constructors()
            frozen_constructors = tuple(self.constructors.values())
            unprocessed = sorted((row for key, row in self.networks.items()
                                  if key not in self.processed_networks),
                                 key=lambda row: (row.born_sequence, row.identity))
            processed, rebound = [], []
            for network in unprocessed:
                matches = [row for row in frozen_constructors
                           if row.middle_recipe == network.member_recipe_roots[1]
                           and network.member_recipe_roots[0] in row.left_bindings
                           and network.member_recipe_roots[2] in row.right_bindings
                           and self._three_corner_support(
                               set(row.binding_pairs),
                               network.member_recipe_roots[0],
                               network.member_recipe_roots[2])
                           and self._cross_source_three_corner(
                               {(item[0], item[1]): item[2]
                                for item in row.binding_pair_sources},
                               network.member_recipe_roots[0],
                               network.member_recipe_roots[2])]
                if len(matches) > 1:
                    raise ParametricSpanNetworkRefuse("span_network:constructor_ambiguous")
                if matches:
                    if len(self.rebinds) + len(rebound) >= MAX_REBINDS:
                        raise ParametricSpanNetworkRefuse("span_network:rebind_bound")
                    result = self._rebind(network, matches[0])
                    rebound.append(result)
                self.processed_networks.add(network.identity)
                processed.append(network.identity)
            self._refresh_constructors()
            tickets = ()
            if not processed:
                try:
                    tickets = tuple(row.ticket for row in self._inner.tick())
                except VariableSpanRefuse as exc:
                    raise ParametricSpanNetworkRefuse("span_network:quiescent") from exc
            self.rebinds.extend(rebound)
            identity = _identity(b"resident-parametric-span-close-v1", [
                tickets, tuple(processed), tuple(row.identity for row in rebound)])
            result = CloseResultV1(identity, tickets, tuple(processed), tuple(rebound))
            self._append_trace(self.next_sequence, EVENT_CLOSE, result.identity,
                               *(row.identity for row in rebound))
            self._record(EventV1(EVENT_CLOSE, (result.identity, tickets,
                tuple(processed), tuple(row.identity for row in rebound))))
            self._bounded(); self._check_checkpoint_bound(); return result
        except Exception:
            self._rollback(before); raise

    def ingest_withdrawal(self, contact: WithdrawalContactV1) -> None:
        before = self._snapshot()
        try:
            self.work = 0
            self._inner.ingest_withdrawal(contact)
            self.networks = self._derive_networks()
            self.processed_networks.intersection_update(self.networks)
            active_networks = set(self.networks)
            self._refresh_constructors()
            self.rebinds = [row for row in self.rebinds
                            if row.network in active_networks
                            and set(row.constructor_witness_roots).issubset(active_networks)]
            self._append_trace(contact.sequence, EVENT_WITHDRAWAL,
                               contact.target_source)
            self._record(self._contact_event(EVENT_WITHDRAWAL, contact))
            self._bounded(); self._check_checkpoint_bound()
        except Exception:
            self._rollback(before); raise

    def _bounded(self):
        if (len(self.networks) > MAX_NETWORKS
                or len(self.processed_networks) > MAX_PROCESSED
                or len(self.constructors) > MAX_CONSTRUCTORS
                or len(self.rebinds) > MAX_REBINDS
                or len(self.events) > MAX_EVENTS or len(self.trace) > MAX_TRACE
                or self.work > self.work_limit):
            raise ParametricSpanNetworkRefuse("span_network:state_bound")
        for rows in (tuple(self.networks.values()), tuple(self.processed_networks),
                     tuple(self.constructors.values()), self.rebinds,
                     self.events, self.trace):
            _strict_numeric(tuple(rows))

    def _checkpoint_body(self):
        return {"schema": SCHEMA_VERSION, "session": self.session,
                "incarnation": 1, "work_limit": self.work_limit,
                "events": [asdict(row) for row in self.events]}

    def _checkpoint_blob(self, body):
        return _canonical({"version": CHECKPOINT_VERSION, "body": body,
                           "hmac": self._boundary._checkpoint_tag(body)})

    def checkpoint(self) -> bytes:
        body = self._checkpoint_body()
        blob = self._checkpoint_blob(body)
        if (len(blob) != self._projected_checkpoint_size()
                or len(blob) > MAX_CHECKPOINT_BYTES):
            raise ParametricSpanNetworkRefuse("span_network:checkpoint_bound")
        return blob

    def _check_checkpoint_bound(self):
        self.checkpoint()

    @classmethod
    def restore(cls, blob: bytes, boundary: ReferenceChannelSequenceBoundaryV1):
        if not isinstance(boundary, ReferenceChannelSequenceBoundaryV1):
            raise ParametricSpanNetworkRefuse("span_network:boundary")
        if isinstance(blob, str):
            raise ParametricSpanNetworkRefuse("span_network:checkpoint")
        try:
            raw_blob = bytes(blob)
            if len(raw_blob) > MAX_CHECKPOINT_BYTES:
                raise ParametricSpanNetworkRefuse("span_network:checkpoint_bound")
            envelope = json.loads(raw_blob); body = envelope["body"]
        except ParametricSpanNetworkRefuse:
            raise
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise ParametricSpanNetworkRefuse("span_network:checkpoint") from exc
        if (not isinstance(envelope, dict)
                or set(envelope) != {"version", "body", "hmac"}
                or envelope["version"] != CHECKPOINT_VERSION
                or not boundary._valid_checkpoint(body, envelope["hmac"])):
            raise ParametricSpanNetworkRefuse("span_network:checkpoint_authentication")
        if (not isinstance(body, dict)
                or set(body) != {"schema", "session", "incarnation", "work_limit", "events"}
                or body["schema"] != SCHEMA_VERSION or body["incarnation"] != 1
                or any(isinstance(body[key], bool) or not isinstance(body[key], int)
                       for key in ("schema", "session", "incarnation", "work_limit"))
                or not isinstance(body["events"], list)
                or len(body["events"]) > MAX_EVENTS):
            raise ParametricSpanNetworkRefuse("span_network:checkpoint_schema")
        out = cls(boundary, body["session"], body["work_limit"])
        for raw in body["events"]:
            if not isinstance(raw, dict) or set(raw) != {"kind", "values"}:
                raise ParametricSpanNetworkRefuse("span_network:event_shape")
            event = EventV1(int(raw["kind"]), _tuplify(raw["values"])); before = len(out.events)
            if event.kind == EVENT_SAMPLE:
                fields, tag = event.values
                out.ingest_sample(OccurrenceContactV1(
                    int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3]),
                    tuple(map(int, fields[4])), tuple(map(int, fields[5])), int(tag)))
            elif event.kind == EVENT_CLOSE:
                result = out.close()
                expected = (result.identity, result.prediction_tickets,
                            result.processed_networks,
                            tuple(row.identity for row in result.rebinds))
                if expected != event.values:
                    raise ParametricSpanNetworkRefuse("span_network:event_replay")
            elif event.kind == EVENT_WITHDRAWAL:
                fields, tag = event.values
                out.ingest_withdrawal(WithdrawalContactV1(
                    *(int(value) for value in fields), int(tag)))
            else:
                raise ParametricSpanNetworkRefuse("span_network:event_kind")
            if len(out.events) != before + 1 or out.events[-1] != event:
                raise ParametricSpanNetworkRefuse("span_network:event_replay")
        if out.checkpoint() != raw_blob:
            raise ParametricSpanNetworkRefuse("span_network:checkpoint_noncanonical")
        return out


def _tuplify(value):
    return tuple(_tuplify(item) for item in value) if isinstance(value, list) else int(value)
