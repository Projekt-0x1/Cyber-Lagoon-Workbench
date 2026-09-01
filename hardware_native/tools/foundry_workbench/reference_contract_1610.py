#!/usr/bin/env python3
"""Graph-neutral, CPU-only reference semantics for the logical #1610 law.

This module is intentionally not connected to a Direct Adult.  A small, fixed
interpreter executes numeric Recipe IR over a bounded, offline-authored starting
state.  Surface units can enter only through authenticated contact records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import hashlib
import hmac
import json
import secrets
from typing import Iterable


SCHEMA_VERSION = 1
CHECKPOINT_VERSION = 2
PRODUCTION_IR = "ResidentRecipeIrProgram.vcurrent"
TRANSLATION_STATUS = "UNDEFINED"
PARITY_STATUS = "NOT_RUN/RED"

MAX_RECIPES = 16
MAX_PROGRAM = 32
MAX_CONTACT_PAYLOAD = 256
MAX_RELATIONS = 512
MAX_QUERIES = 16
MAX_CURRENT_OCCURRENCES = 64
MAX_PHYSICAL_TERMINAL_CLAIMS = 64
MAX_TRACE = 4096
MAX_PUBLIC_TRAJECTORY_EXTENT = 512
MAX_ACTIVE_COMPOSITIONAL_DEPTH = 3
MAX_CAUSAL_ANCESTRY_DEPTH = 8
MAX_ROUTE_TRANSPORT_WIDTH = 128
MAX_FROZEN_ACTION_CLOSURE = 128
MAX_CONDENSATION_WITNESS_WIDTH = 128
MAX_WORK_PER_TICK = 4096
MAX_WITHDRAWN_SOURCES = 64
MAX_CHECKPOINT_BYTES = 1 << 20
MAX_CONTINUATIONS = 64

OP_ARBITRATE = 1
OP_FRONTIER = 2
OP_COMPOSE = 3
OP_PUBLISH = 4
OP_SETTLE = 5
OP_CONDENSE = 6

CONTACT_LEXEME = 1
CONTACT_FRAME = 2
CONTACT_QUERY = 3
CONTACT_CONSEQUENCE = 4
CONTACT_STIMULUS = 5

_POOL_ADMISSION = object()
_POOL_RECEIPT_KEY = b"0x1-foundry-reference-pool-admission-v1"


class Refuse(RuntimeError):
    pass


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(domain: str, value) -> str:
    return hashlib.sha256(domain.encode() + b"\0" + _canonical(value)).hexdigest()


def _strict_numeric(value, path: str = "$") -> None:
    """Reject every dynamic string/byte and enforce bounded numeric containers."""
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raise Refuse(f"ir:literal:{path}")
    if isinstance(value, bool) or value is None:
        raise Refuse(f"ir:type:{path}")
    if isinstance(value, int):
        if not -(1 << 63) <= value < (1 << 63):
            raise Refuse(f"ir:integer:{path}")
        return
    if isinstance(value, (tuple, list)):
        if len(value) > MAX_TRACE:
            raise Refuse(f"ir:container_bound:{path}")
        for index, item in enumerate(value):
            _strict_numeric(item, f"{path}[{index}]")
        return
    if isinstance(value, set):
        if len(value) > MAX_TRACE:
            raise Refuse(f"ir:container_bound:{path}")
        for index, item in enumerate(sorted(value)):
            _strict_numeric(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        if len(value) > MAX_TRACE:
            raise Refuse(f"ir:container_bound:{path}")
        for key, item in value.items():
            _strict_numeric(key, f"{path}.key")
            _strict_numeric(item, f"{path}[{key}]")
        return
    if hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            _strict_numeric(getattr(value, name), f"{path}.{name}")
        return
    raise Refuse(f"ir:type:{path}")


@dataclass(frozen=True)
class InstructionV1:
    opcode: int
    a: int = 0
    b: int = 0
    c: int = 0


@dataclass(frozen=True)
class RecipeIrV1:
    schema_version: int
    identity: int
    program: tuple[InstructionV1, ...]
    work_quote: int

    def validate(self) -> None:
        _strict_numeric(self)
        if self.schema_version != SCHEMA_VERSION:
            raise Refuse("ir:version")
        if not 0 < len(self.program) <= MAX_PROGRAM:
            raise Refuse("ir:program_bound")
        if not 0 < self.work_quote <= MAX_WORK_PER_TICK:
            raise Refuse("ir:work_quote")
        allowed = {OP_ARBITRATE, OP_FRONTIER, OP_COMPOSE, OP_PUBLISH,
                   OP_SETTLE, OP_CONDENSE}
        if any(i.opcode not in allowed for i in self.program):
            raise Refuse("ir:opcode")
        if any((i.a, i.b, i.c) != (0, 0, 0) for i in self.program):
            raise Refuse("ir:semantic_operand")


class VerifiedRecipePool:
    """Content-addressed pool: callers cannot supply or assert a pass receipt."""
    def __init__(self, recipes: Iterable[RecipeIrV1], *, _admission=None):
        if _admission is not _POOL_ADMISSION:
            raise Refuse("pool:admission")
        rows = list(recipes)
        if not 0 < len(rows) <= MAX_RECIPES:
            raise Refuse("pool:bound")
        self._rows: dict[int, RecipeIrV1] = {}
        for recipe in rows:
            recipe.validate()
            if recipe.identity in self._rows:
                raise Refuse("pool:duplicate")
            self._rows[recipe.identity] = copy.deepcopy(recipe)
        material = [asdict(self._rows[k]) for k in sorted(self._rows)]
        self.root = digest("foundry-verified-recipe-pool-v1", material)
        self.admission_receipt = hmac.new(
            _POOL_RECEIPT_KEY, self.root.encode(), hashlib.sha256).hexdigest()

    def program(self, identity: int) -> tuple[InstructionV1, ...]:
        if identity not in self._rows:
            raise Refuse("pool:unknown_recipe")
        recipe = self._rows[identity]
        recipe.validate()
        return recipe.program

    def recompute(self) -> str:
        material = [asdict(self._rows[k]) for k in sorted(self._rows)]
        return digest("foundry-verified-recipe-pool-v1", material)

    def admitted(self) -> bool:
        expected = hmac.new(
            _POOL_RECEIPT_KEY, self.recompute().encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.admission_receipt, expected)


@dataclass
class RelationV1:
    identity: int
    kind: int
    values: tuple[int, ...]
    source: int
    provenance: tuple[int, ...]
    active: int = 1


@dataclass
class QueryV1:
    identity: int
    contact_identity: int
    language: int
    frame: int
    root: tuple
    source: int
    active: int = 1
    # Legacy observer name only. contact_identity==0 means this is an already-
    # resident endogenous work front, not a prompted input. parent_identity is
    # causal ancestry between current work fronts, not a semantic "thought" id.
    parent_identity: int = 0


@dataclass
class OccurrenceV1:
    identity: int
    recipe_identity: int
    query_identity: int
    constituent_root: int
    frontier: tuple[int, ...]
    tick: int
    current: int = 1


@dataclass
class ByteAncestryV1:
    offset: int
    unit: int
    constituent_root: int
    relation_roots: tuple[int, ...]
    occurrence_identity: int = 0
    ancestry_depth: int = 1


@dataclass
class ActionV1:
    ticket: int
    incarnation: int
    born_tick: int
    deadline: int
    channel: int
    source: int
    occurrence_identity: int
    payload: tuple[int, ...]
    constituent_root: int
    frontier: tuple[int, ...]
    ancestry: tuple[ByteAncestryV1, ...]
    settled: int = 0
    consequence_identity: int = 0
    active: int = 1


@dataclass
class ContactV1:
    ticket: int
    incarnation: int
    deadline: int
    source: int
    channel: int
    kind: int
    payload: tuple[int, ...]
    authenticated: int
    independent: int
    session_epoch: int = 0
    ingress_sequence: int = 0
    auth_tag: int = 0

    def signed_fields(self):
        return (self.ticket, self.incarnation, self.deadline, self.source,
                self.channel, self.kind, self.payload, self.authenticated,
                self.independent, self.session_epoch, self.ingress_sequence)

    def validate(self, tick: int, incarnation: int, session_epoch: int,
                 ingress_sequence: int, key: bytes) -> None:
        _strict_numeric(self.signed_fields())
        if self.incarnation != incarnation or self.deadline < tick:
            raise Refuse("contact:stale_incarnation_deadline")
        if self.authenticated != 1 or self.independent not in {0, 1}:
            raise Refuse("contact:authentication")
        if self.session_epoch != session_epoch or self.ingress_sequence != ingress_sequence:
            raise Refuse("contact:session_sequence")
        expected = int.from_bytes(hmac.new(
            key, _canonical(self.signed_fields()), hashlib.sha256).digest()[:8], "little") & ((1 << 63) - 1)
        if self.auth_tag != expected:
            raise Refuse("contact:receipt")
        if not 0 < self.channel < (1 << 16) or not 0 < self.source < (1 << 31):
            raise Refuse("contact:route")
        if len(self.payload) > MAX_CONTACT_PAYLOAD:
            raise Refuse("contact:payload_bound")


@dataclass
class ReferenceStateV1:
    schema_version: int = SCHEMA_VERSION
    incarnation: int = 1
    tick: int = 0
    work: int = 0
    work_limit: int = MAX_WORK_PER_TICK
    relations: list[RelationV1] = field(default_factory=list)
    queries: list[QueryV1] = field(default_factory=list)
    occurrences: list[OccurrenceV1] = field(default_factory=list)
    actions: list[ActionV1] = field(default_factory=list)
    trace: list[tuple[int, ...]] = field(default_factory=list)
    withdrawn_sources: set[int] = field(default_factory=set)
    compact_roots: dict[int, tuple[int, ...]] = field(default_factory=dict)
    compact_lesions: set[int] = field(default_factory=set)
    use_counts: dict[int, int] = field(default_factory=dict)
    credit: dict[int, int] = field(default_factory=dict)
    contact_receipts: list[ContactV1] = field(default_factory=list)
    # Dormant resident continuation graph between already-present work fronts.
    # `contact_identity == 0` marks an endogenous work front. Runtime settlement
    # may activate a successor but may not create a new work front or surface.
    continuations: dict[int, int] = field(default_factory=dict)
    session_epoch: int = 1
    next_ingress_sequence: int = 1
    next_identity: int = 1


def authored_starting_state() -> ReferenceStateV1:
    """Authored machinery only: no facts, surfaces, queries, or mature wiring."""
    return ReferenceStateV1()


def authored_recipe_pool() -> VerifiedRecipePool:
    # Six reusable laws, not one Recipe per word, construction, or test case.
    return VerifiedRecipePool((
        RecipeIrV1(1, 1, (InstructionV1(OP_ARBITRATE),), 8),
        RecipeIrV1(1, 2, (InstructionV1(OP_FRONTIER), InstructionV1(OP_COMPOSE)), 256),
        RecipeIrV1(1, 3, (InstructionV1(OP_PUBLISH),), 64),
        RecipeIrV1(1, 4, (InstructionV1(OP_SETTLE),), 64),
        RecipeIrV1(1, 5, (InstructionV1(OP_CONDENSE),), 64),
        RecipeIrV1(1, 6, (InstructionV1(OP_FRONTIER), InstructionV1(OP_COMPOSE),
                           InstructionV1(OP_PUBLISH), InstructionV1(OP_CONDENSE)), 512),
    ), _admission=_POOL_ADMISSION)


class ReferenceMachineV1:
    def __init__(self, state: ReferenceStateV1, pool: VerifiedRecipePool):
        if (state.schema_version != SCHEMA_VERSION or pool.recompute() != pool.root
                or not pool.admitted()):
            raise Refuse("machine:frozen_input")
        self.state = copy.deepcopy(state)
        self.pool = pool
        # Rebuildable execution-only incidence index. It is never serialized and
        # carries no semantic/evidentiary authority; it exists solely so active
        # work scales with touched resident morphology rather than total dormant
        # relation count.
        self._relation_index: dict[tuple[int, int, tuple[int, ...]], list[int]] = {}
        self._rebuild_relation_index()
        self.last_lookup_touches = 0
        self._contact_key = hmac.new(
            _POOL_RECEIPT_KEY,
            f"{pool.admission_receipt}:{state.incarnation}:{state.session_epoch}".encode(),
            hashlib.sha256).digest()
        self._bounded()

    def seal_contact(self, contact: ContactV1) -> ContactV1:
        """Body-boundary receipt creation; callers cannot assert a valid tag."""
        sealed = copy.deepcopy(contact)
        sealed.session_epoch = self.state.session_epoch
        sealed.ingress_sequence = self.state.next_ingress_sequence
        sealed.auth_tag = int.from_bytes(hmac.new(
            self._contact_key, _canonical(sealed.signed_fields()),
            hashlib.sha256).digest()[:8], "little") & ((1 << 63) - 1)
        return sealed

    def _index_relation(self, index: int, relation: RelationV1) -> None:
        # Prefixes up to four fields cover the current compact ABI and are
        # intentionally derived. A future production backend may use BVH/RT,
        # tensor-friendly sparse tables, or another lowering without changing
        # causal semantics.
        limit = min(4, len(relation.values))
        for width in range(limit + 1):
            key = (relation.kind, width, tuple(relation.values[:width]))
            self._relation_index.setdefault(key, []).append(index)

    def _rebuild_relation_index(self) -> None:
        self._relation_index.clear()
        for index, relation in enumerate(self.state.relations):
            self._index_relation(index, relation)

    def _bounded(self) -> None:
        s = self.state
        _strict_numeric(s)
        if len(s.relations) > MAX_RELATIONS or len(s.queries) > MAX_QUERIES:
            raise Refuse("state:relation_query_bound")
        if len(s.occurrences) > MAX_CURRENT_OCCURRENCES:
            raise Refuse("state:occurrence_bound")
        if len(s.actions) > MAX_PHYSICAL_TERMINAL_CLAIMS or len(s.trace) > MAX_TRACE:
            raise Refuse("state:action_trace_bound")
        if len(s.withdrawn_sources) > MAX_WITHDRAWN_SOURCES:
            raise Refuse("state:withdrawal_bound")
        if len(s.contact_receipts) > MAX_TRACE:
            raise Refuse("state:contact_receipt_bound")
        sequences = [c.ingress_sequence for c in s.contact_receipts]
        if sequences != list(range(1, s.next_ingress_sequence)):
            raise Refuse("state:contact_sequence")
        for receipt in s.contact_receipts:
            receipt.validate(0, s.incarnation, s.session_epoch,
                             receipt.ingress_sequence, self._contact_key)
        if any(len(r.values) > MAX_CONTACT_PAYLOAD or len(r.provenance) > MAX_ROUTE_TRANSPORT_WIDTH
               for r in s.relations):
            raise Refuse("state:relation_payload_provenance_bound")
        if any(len(a.payload) > MAX_PUBLIC_TRAJECTORY_EXTENT or len(a.frontier) > MAX_FROZEN_ACTION_CLOSURE
               or len(a.ancestry) != len(a.payload) for a in s.actions):
            raise Refuse("state:action_ancestry_bound")
        occurrence_ids = {o.identity for o in s.occurrences}
        if any(a.occurrence_identity not in occurrence_ids
               or any(x.occurrence_identity != a.occurrence_identity for x in a.ancestry)
               for a in s.actions):
            raise Refuse("state:occurrence_ancestry")
        if any(x.ancestry_depth < 1 or x.ancestry_depth > MAX_CAUSAL_ANCESTRY_DEPTH
               for a in s.actions for x in a.ancestry):
            raise Refuse("state:causal_ancestry_depth")
        if any(len(v) > MAX_CONDENSATION_WITNESS_WIDTH for v in s.compact_roots.values()):
            raise Refuse("state:compact_provenance_bound")
        if any(len(x) > MAX_RELATIONS for x in (s.compact_roots, s.compact_lesions,
                                                s.use_counts, s.credit)):
            raise Refuse("state:learned_mutation_bound")
        if s.continuations:
            raise Refuse("state:host_continuation")
        work_ids = {q.identity for q in s.queries}
        if any(parent not in work_ids or child not in work_ids
               for parent, child in s.continuations.items()):
            raise Refuse("state:continuation_identity")
        if any(parent == child for parent, child in s.continuations.items()):
            raise Refuse("state:continuation_self_loop")
        relation_ids = [r.identity for r in s.relations]
        query_ids = [q.identity for q in s.queries]
        occurrence_ids_list = [o.identity for o in s.occurrences]
        action_ids = [a.ticket for a in s.actions]
        consequence_ids = [a.consequence_identity for a in s.actions
                           if a.consequence_identity]
        identities = relation_ids + query_ids + occurrence_ids_list + action_ids + consequence_ids
        if any(identity <= 0 for identity in identities) or len(identities) != len(set(identities)):
            raise Refuse("state:identity")
        if s.next_identity <= max(identities, default=0):
            raise Refuse("state:next_identity")
        if any(r.kind not in {CONTACT_LEXEME, CONTACT_FRAME, CONTACT_QUERY, CONTACT_STIMULUS}
               or r.active not in {0, 1} or r.source <= 0
               or any(p <= 0 for p in r.provenance) for r in s.relations):
            raise Refuse("state:relation_shape")
        relation_by_id = {r.identity: r for r in s.relations}
        receipt_roots = {(c.ticket, c.ingress_sequence, c.session_epoch)
                         for c in s.contact_receipts}
        if any(tuple(r.provenance) not in receipt_roots for r in s.relations):
            raise Refuse("state:contact_receipt_lineage")
        query_by_id = {q.identity: q for q in s.queries}
        for query in s.queries:
            if query.active not in {0, 1} or query.source <= 0:
                raise Refuse("state:query_shape")
            if query.contact_identity:
                contact = relation_by_id.get(query.contact_identity)
                if contact is None or contact.kind != CONTACT_QUERY:
                    raise Refuse("state:query_contact")
            self._validate_query_node(query.root, 1)
        for occurrence in s.occurrences:
            if (occurrence.current not in {0, 1} or occurrence.query_identity not in query_by_id
                    or occurrence.constituent_root <= 0):
                raise Refuse("state:occurrence_shape")
            self.pool.program(occurrence.recipe_identity)
            if any(root not in relation_by_id for root in occurrence.frontier):
                raise Refuse("state:occurrence_frontier")
        occurrence_by_id = {o.identity: o for o in s.occurrences}
        for action in s.actions:
            if (action.settled not in {0, 1} or action.active not in {0, 1} or action.source <= 0
                    or action.channel <= 0 or action.deadline < action.born_tick
                    or action.occurrence_identity not in occurrence_by_id):
                raise Refuse("state:action_shape")
            if bool(action.consequence_identity) != bool(action.settled):
                raise Refuse("state:action_consequence")
            if tuple(item.offset for item in action.ancestry) != tuple(range(len(action.payload))):
                raise Refuse("state:ancestry_offset")
            if any(item.unit != action.payload[item.offset]
                   or item.constituent_root <= 0
                   or any(root not in relation_by_id for root in item.relation_roots)
                   for item in action.ancestry):
                raise Refuse("state:ancestry_lineage")
            if any(root not in relation_by_id for root in action.frontier):
                raise Refuse("state:action_frontier")
        if any(root <= 0 or any(dep not in relation_by_id for dep in deps)
               for root, deps in s.compact_roots.items()):
            raise Refuse("state:compact_lineage")
        if any(root not in relation_by_id for root in s.credit):
            raise Refuse("state:credit_lineage")
        seen: set[int] = set()
        for start in s.continuations:
            cursor = start
            chain: set[int] = set()
            while cursor in s.continuations:
                if cursor in chain:
                    raise Refuse("state:continuation_cycle")
                chain.add(cursor)
                cursor = s.continuations[cursor]
            seen.update(chain)
        if not 0 <= s.work <= s.work_limit <= MAX_WORK_PER_TICK:
            raise Refuse("state:resource_bound")
        if s.session_epoch <= 0 or s.next_ingress_sequence <= 0:
            raise Refuse("state:ingress_identity")

    @classmethod
    def _validate_query_node(cls, node, depth: int) -> None:
        if not isinstance(node, tuple) or len(node) != 3:
            raise Refuse("state:query_tree")
        concept, number, children = node
        if not isinstance(concept, int) or not isinstance(number, int):
            raise Refuse("state:query_tree")
        if not isinstance(children, tuple) or len(children) > 4:
            raise Refuse("state:query_tree")
        if depth > MAX_ACTIVE_COMPOSITIONAL_DEPTH and children:
            raise Refuse("state:query_depth")
        for child in children:
            cls._validate_query_node(child, depth + 1)

    def _new_id(self) -> int:
        value = self.state.next_identity
        self.state.next_identity += 1
        return value

    def contact(self, contact: ContactV1) -> int:
        before = copy.deepcopy(self.state)
        try:
            contact.validate(self.state.tick, self.state.incarnation,
                             self.state.session_epoch,
                             self.state.next_ingress_sequence,
                             self._contact_key)
            self.state.contact_receipts.append(copy.deepcopy(contact))
            self.state.next_ingress_sequence += 1
            if contact.kind == CONTACT_CONSEQUENCE:
                self._settle(contact)
                self._bounded()
                return contact.ticket
            rid = self._new_id()
            relation = RelationV1(
                rid, contact.kind, contact.payload, contact.source,
                (contact.ticket, contact.ingress_sequence, contact.session_epoch), 1)
            self.state.relations.append(relation)
            self._index_relation(len(self.state.relations) - 1, relation)
            if contact.kind == CONTACT_QUERY:
                self.state.queries.append(self._parse_query(rid, contact))
            elif contact.kind not in {CONTACT_LEXEME, CONTACT_FRAME, CONTACT_STIMULUS}:
                raise Refuse("contact:kind")
            self.state.trace.append((self.state.tick, 10, rid, contact.ticket))
            self._bounded()
            return rid
        except Exception:
            self.state = before
            self._rebuild_relation_index()
            raise

    def _parse_query(self, rid: int, contact: ContactV1) -> QueryV1:
        p = contact.payload
        if len(p) < 5:
            raise Refuse("query:shape")
        cue, language, used, depth = p[:4]
        if used != len(p) - 4 or not 1 <= depth <= MAX_ACTIVE_COMPOSITIONAL_DEPTH:
            raise Refuse("query:bound")
        root, end = self._parse_node(p, 4, 1)
        if end != len(p):
            raise Refuse("query:trailing")
        # Learned dormant candidate. A later opaque stimulus, not this contact,
        # makes it eligible for resident arbitration.
        return QueryV1(self._new_id(), rid, language, cue, root, contact.source, 1)

    def _parse_node(self, values: tuple[int, ...], pos: int, depth: int):
        if pos + 3 > len(values):
            raise Refuse("query:node")
        concept, number, children = values[pos:pos + 3]
        if children < 0 or children > 4:
            raise Refuse("query:arity")
        if depth > MAX_ACTIVE_COMPOSITIONAL_DEPTH and children:
            raise Refuse("query:depth")
        out = []
        pos += 3
        for _ in range(children):
            child, pos = self._parse_node(values, pos, depth + 1)
            out.append(child)
        return (concept, number, tuple(out)), pos

    def withdraw_source(self, source: int) -> None:
        before = copy.deepcopy(self.state)
        try:
            self.state.withdrawn_sources.add(source)
            changed = True
            withdrawn: set[int] = set()
            while changed:
                changed = False
                for relation in self.state.relations:
                    if not relation.active:
                        continue
                    if relation.source == source or any(p in withdrawn for p in relation.provenance):
                        relation.active = 0
                        withdrawn.add(relation.identity)
                        changed = True
            for root, dependencies in list(self.state.compact_roots.items()):
                if any(dep in withdrawn for dep in dependencies):
                    del self.state.compact_roots[root]
            for query in self.state.queries:
                if query.source == source or query.contact_identity in withdrawn:
                    query.active = 0
            inactive_queries = {q.identity for q in self.state.queries if not q.active}
            for occurrence in self.state.occurrences:
                if occurrence.query_identity in inactive_queries or any(
                        root in withdrawn for root in occurrence.frontier):
                    occurrence.current = 0
            inactive_occurrences = {o.identity for o in self.state.occurrences if not o.current}
            for action in self.state.actions:
                if (action.source == source or action.occurrence_identity in inactive_occurrences
                        or any(root in withdrawn for root in action.frontier)):
                    action.active = 0
            self.state.continuations = {
                parent: child for parent, child in self.state.continuations.items()
                if parent not in inactive_queries and child not in inactive_queries}
            for root in withdrawn:
                self.state.use_counts.pop(root, None)
                self.state.credit.pop(root, None)
            self.state.trace.append((self.state.tick, 11, source, len(withdrawn)))
            self._bounded()
        except Exception:
            self.state = before
            self._rebuild_relation_index()
            raise

    def tick(self) -> ActionV1 | None:
        before = copy.deepcopy(self.state)
        try:
            if any(action.active and not action.settled for action in self.state.actions):
                raise Refuse("tick:pending_consequence")
            self.state.tick += 1
            self.state.work = 0
            stimuli = self._active(CONTACT_STIMULUS)
            if not stimuli:
                self.state.trace.append((self.state.tick, OP_ARBITRATE, 0, 0))
                return None
            stimulus = min(stimuli, key=lambda r: r.identity)
            if len(stimulus.values) != 1:
                raise Refuse("stimulus:shape")
            cue = stimulus.values[0]
            candidates = [q for q in self.state.queries if q.active and q.frame == cue
                          and q.source not in self.state.withdrawn_sources]
            if not candidates:
                raise Refuse("arbitrate:no_candidate")
            scored = []
            support_touches = 0
            for q in candidates:
                score, touches = self._candidate_credit(q)
                support_touches += touches
                scored.append((score, q))
            peak = max(score for score, _ in scored)
            winners = [q for score, q in scored if score == peak]
            if len(winners) != 1:
                raise Refuse("arbitrate:ambiguous")
            self.state.work += len(candidates) + support_touches
            if self.state.work > self.state.work_limit:
                raise Refuse("tick:resource")
            action = self._execute_query(winners[0], stimulus)
            stimulus.active = 0
            self._bounded()
            return action
        except Exception:
            self.state = before
            raise

    def _active(self, kind: int) -> list[RelationV1]:
        return [r for r in self.state.relations if r.active and r.kind == kind
                and r.source not in self.state.withdrawn_sources]

    def _candidate_credit(self, query: QueryV1) -> tuple[int, int]:
        """Credit follows the resident relations a candidate would recruit."""
        roots = {query.contact_identity}
        touches = 0
        frame = self._select_unique(CONTACT_FRAME, query.language, 1)
        roots.add(frame.identity)
        touches += self.last_lookup_touches

        def visit(node) -> None:
            nonlocal touches
            concept, number, children = node
            lexeme = self._select_unique(CONTACT_LEXEME, query.language,
                                          concept, number)
            roots.add(lexeme.identity)
            touches += self.last_lookup_touches
            for child in children:
                visit(child)

        visit(query.root)
        return sum(self.state.credit.get(root, 0) for root in roots), touches

    def _execute_query(self, query: QueryV1, stimulus: RelationV1) -> ActionV1:
        for instruction in self.pool.program(6):
            self.state.work += {OP_FRONTIER: 8, OP_COMPOSE: 32,
                               OP_PUBLISH: 16, OP_CONDENSE: 8}[instruction.opcode]
            if self.state.work > self.state.work_limit:
                raise Refuse("tick:resource")
            self.state.trace.append((self.state.tick, instruction.opcode, query.identity,
                                     self.state.work))
        frame = self._select_unique(CONTACT_FRAME, query.language, 1)
        payload, ancestry, frontier, root = self._compose(query.root, query.language,
                                                          1, frame, 1)
        frontier.update((query.contact_identity, stimulus.identity))
        if not payload or len(payload) > MAX_PUBLIC_TRAJECTORY_EXTENT:
            raise Refuse("compose:payload_bound")
        occurrence_id = self._new_id()
        occurrence = OccurrenceV1(occurrence_id, 6, query.identity, root,
                                  tuple(sorted(frontier)), self.state.tick)
        self.state.occurrences.append(occurrence)
        ancestry = tuple(ByteAncestryV1(item.offset, item.unit,
                         item.constituent_root, item.relation_roots, occurrence_id, item.ancestry_depth)
                         for item in ancestry)
        ticket = self._new_id()
        action = ActionV1(ticket, self.state.incarnation, self.state.tick,
                          self.state.tick + 4, 2, stimulus.source, occurrence_id,
                          tuple(payload), root, tuple(sorted(frontier)), ancestry)
        self.state.actions.append(action)
        self.state.use_counts[root] = self.state.use_counts.get(root, 0) + 1
        if self.state.use_counts[root] >= 2 and root not in self.state.compact_lesions:
            self.state.compact_roots[root] = tuple(sorted(frontier))
        return action

    def _select_unique(self, kind: int, *prefix: int) -> RelationV1:
        key = (kind, len(prefix), tuple(prefix))
        indices = self._relation_index.get(key, ())
        rows = [self.state.relations[i] for i in indices
                if self.state.relations[i].active
                and self.state.relations[i].source not in self.state.withdrawn_sources]
        self.last_lookup_touches = len(indices)
        if len(rows) != 1:
            raise Refuse("compose:missing_or_ambiguous")
        return rows[0]

    def _compose(self, node: tuple, language: int, frame_id: int,
                 frame: RelationV1, depth: int):
        concept, number, children = node
        if depth > MAX_ACTIVE_COMPOSITIONAL_DEPTH and children:
            raise Refuse("compose:depth")
        lexeme = self._select_unique(CONTACT_LEXEME, language, concept, number)
        count = lexeme.values[3]
        units = tuple(lexeme.values[4:])
        if count != len(units) or any(not 0 <= u <= 255 for u in units):
            raise Refuse("compose:lexeme")
        required_arity = frame.values[2]
        order_count = frame.values[3]
        order = tuple(frame.values[4:])
        if (depth == 1 or children) and len(children) != required_arity:
            raise Refuse("compose:omitted_constituent")
        if order_count != len(order) or set(order) != set(range(required_arity + 1)):
            raise Refuse("compose:frame_order")
        child_parts = [self._compose(child, language, frame_id, frame, depth + 1)
                       for child in children]
        root = int(digest("constituent-root-v1", [concept, number,
                   [part[3] for part in child_parts]])[:15], 16)
        frontier = {lexeme.identity, frame.identity}
        own = (list(units),
               [ByteAncestryV1(i, unit, root, (lexeme.identity, frame.identity), 0, 1)
                for i, unit in enumerate(units)],
               {lexeme.identity, frame.identity}, root)
        parts = [own] + child_parts
        if not children:
            return own
        payload: list[int] = []
        ancestry: list[ByteAncestryV1] = []
        separator = self._select_unique(CONTACT_LEXEME, language, 0, 0)
        for part_index, slot in enumerate(order):
            child_payload, child_ancestry, child_frontier, child_root = parts[slot]
            if part_index:
                base = len(payload)
                payload.extend(separator.values[4:])
                ancestry.extend(ByteAncestryV1(base + i, unit, root,
                                (separator.identity, frame.identity), 0, 1)
                                for i, unit in enumerate(separator.values[4:]))
            base = len(payload)
            payload.extend(child_payload)
            ancestry.extend(ByteAncestryV1(base + item.offset, item.unit,
                            item.constituent_root,
                            tuple(sorted(set(item.relation_roots + (frame.identity,)))),
                            0, item.ancestry_depth + 1)
                            for item in child_ancestry)
            frontier.update(child_frontier)
        frontier.add(separator.identity)
        return payload, ancestry, frontier, root

    def _settle(self, contact: ContactV1) -> None:
        action = next((a for a in self.state.actions if a.ticket == contact.ticket), None)
        if action is None or action.settled or not action.active:
            raise Refuse("consequence:ticket")
        if (contact.incarnation != action.incarnation or contact.deadline != action.deadline
                or contact.channel != action.channel or contact.source != action.source):
            raise Refuse("consequence:binding")
        if len(contact.payload) != 2:
            raise Refuse("consequence:shape")
        effect, counterfactual = contact.payload
        action.settled = 1
        action.consequence_identity = self._new_id()
        occurrence = next(o for o in self.state.occurrences
                          if o.identity == action.occurrence_identity)
        occurrence.current = 0
        # Independence is necessary, but causal difference from the zero-effect
        # counterfactual is separately required for positive credit.
        causal_difference = contact.independent and effect != counterfactual
        if causal_difference:
            for relation in action.frontier:
                delta = 1 if effect > counterfactual else -1
                self.state.credit[relation] = self.state.credit.get(relation, 0) + delta
        self.state.trace.append((self.state.tick, OP_SETTLE, action.ticket,
                                 action.consequence_identity, effect,
                                 int(bool(causal_difference))))

    def lesion_compact(self, root: int) -> None:
        before = copy.deepcopy(self.state)
        self.state.compact_lesions.add(root)
        self.state.compact_roots.pop(root, None)
        try:
            self._bounded()
        except Exception:
            self.state = before
            raise

    def deoptimize(self, root: int) -> None:
        self.state.compact_roots.pop(root, None)

    def checkpoint(self) -> bytes:
        body = self._checkpoint_body()
        envelope = {"version": CHECKPOINT_VERSION, "pool_root": self.pool.root,
                    "body": body,
                    "checksum": digest("foundry-checkpoint-v1", body)}
        blob = _canonical(envelope)
        if len(blob) > MAX_CHECKPOINT_BYTES:
            raise Refuse("checkpoint:bound")
        return blob

    def _checkpoint_body(self):
        def convert(value):
            if hasattr(value, "__dataclass_fields__"):
                return {name: convert(getattr(value, name)) for name in value.__dataclass_fields__}
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, set):
                return sorted(value)
            if isinstance(value, (tuple, list)):
                return [convert(v) for v in value]
            return value
        return convert(self.state)

    @classmethod
    def restore(cls, blob: bytes, pool: VerifiedRecipePool):
        if not isinstance(blob, bytes) or len(blob) > MAX_CHECKPOINT_BYTES:
            raise Refuse("checkpoint:bound")
        try:
            envelope = json.loads(blob)
        except Exception as exc:
            raise Refuse("checkpoint:decode") from exc
        if not isinstance(envelope, dict) or set(envelope) != {
                "version", "pool_root", "body", "checksum"}:
            raise Refuse("checkpoint:envelope")
        if envelope.get("version") != CHECKPOINT_VERSION:
            raise Refuse("checkpoint:version")
        if envelope.get("pool_root") != pool.root or pool.recompute() != pool.root:
            raise Refuse("checkpoint:pool")
        body = envelope.get("body")
        expected_body = set(ReferenceStateV1.__dataclass_fields__)
        if not isinstance(body, dict) or set(body) != expected_body:
            raise Refuse("checkpoint:body_fields")
        if envelope.get("checksum") != digest("foundry-checkpoint-v1", body):
            raise Refuse("checkpoint:corrupt")
        try:
            state = ReferenceStateV1(
                schema_version=body["schema_version"], incarnation=body["incarnation"],
                tick=body["tick"], work=body["work"], work_limit=body["work_limit"],
                relations=[RelationV1(r["identity"], r["kind"], tuple(r["values"]),
                           r["source"], tuple(r["provenance"]), r["active"])
                           for r in body["relations"]],
                queries=[QueryV1(q["identity"], q["contact_identity"], q["language"],
                         q["frame"], cls._tuple_tree(q["root"]), q["source"], q["active"],
                         q.get("parent_identity", 0))
                         for q in body["queries"]],
                occurrences=[OccurrenceV1(o["identity"], o["recipe_identity"],
                             o["query_identity"], o["constituent_root"],
                             tuple(o["frontier"]), o["tick"], o["current"])
                             for o in body["occurrences"]],
                actions=[ActionV1(a["ticket"], a["incarnation"], a["born_tick"],
                         a["deadline"], a["channel"], a["source"],
                         a["occurrence_identity"], tuple(a["payload"]),
                         a["constituent_root"], tuple(a["frontier"]),
                         tuple(ByteAncestryV1(x["offset"], x["unit"],
                               x["constituent_root"], tuple(x["relation_roots"]),
                               x["occurrence_identity"], x["ancestry_depth"])
                               for x in a["ancestry"]), a["settled"],
                         a["consequence_identity"], a.get("active", 1))
                         for a in body["actions"]],
                trace=[tuple(x) for x in body["trace"]],
                withdrawn_sources=set(body["withdrawn_sources"]),
                compact_roots={int(k): tuple(v) for k, v in body["compact_roots"].items()},
                compact_lesions=set(body["compact_lesions"]),
                use_counts={int(k): v for k, v in body["use_counts"].items()},
                credit={int(k): v for k, v in body["credit"].items()},
                contact_receipts=[ContactV1(
                    c["ticket"], c["incarnation"], c["deadline"], c["source"],
                    c["channel"], c["kind"], tuple(c["payload"]),
                    c["authenticated"], c["independent"], c["session_epoch"],
                    c["ingress_sequence"], c["auth_tag"])
                    for c in body["contact_receipts"]],
                continuations={int(k): int(v) for k, v in body.get("continuations", {}).items()},
                session_epoch=body["session_epoch"],
                next_ingress_sequence=body["next_ingress_sequence"],
                next_identity=body["next_identity"])
        except Exception as exc:
            raise Refuse("checkpoint:shape") from exc
        machine = cls(state, pool)
        machine._bounded()
        return machine

    @staticmethod
    def _tuple_tree(value):
        if isinstance(value, list):
            return tuple(ReferenceMachineV1._tuple_tree(v) for v in value)
        return value

    def state_hash(self) -> str:
        return digest("foundry-reference-state-v1", self._checkpoint_body())


# Experimental replacement kernel.  It deliberately shares the strict contact,
# consequence and ancestry types above while making no language/world distinction.
OPAQUE_EXPERIENCE = 20
OPAQUE_CURRENT = 21
OPAQUE_WITHDRAW = 22
OPAQUE_CHECKPOINT_VERSION = 1
MAX_OPAQUE_EXPERIENCES = 128
MAX_OPAQUE_UNITS = 64
MAX_OPAQUE_SITES = 64
_OPAQUE_RESTORE_ADMISSION = object()


@dataclass
class OpaqueExperienceV2:
    identity: int
    source: int
    coalition: int
    port_a: int
    port_b: int
    sites: tuple[int, ...]
    prior: tuple[int, ...]
    later: tuple[int, ...]
    occurrence_identity: int
    active: int = 1


@dataclass
class OpaqueCurrentV2:
    identity: int
    source: int
    channel: int
    coalition: int
    port_a: int
    port_b: int
    sites: tuple[int, ...]
    present: tuple[int, ...]


@dataclass
class OpaqueActionV2:
    ticket: int
    incarnation: int
    born_tick: int
    deadline: int
    channel: int
    source: int
    occurrence_identity: int
    recipe_identity: int
    payload: tuple[int, ...]
    contributors: tuple[int, ...]
    ancestry: tuple[ByteAncestryV1, ...]
    provisional: int = 1
    settled: int = 0
    consequence_identity: int = 0
    active: int = 1


@dataclass
class OpaqueRelationStateV2:
    incarnation: int = 1
    tick: int = 0
    next_identity: int = 1
    next_ingress_sequence: int = 0
    session_epoch: int = 1
    work: int = 0
    work_limit: int = MAX_WORK_PER_TICK
    experiences: list[OpaqueExperienceV2] = field(default_factory=list)
    current: OpaqueCurrentV2 | None = None
    actions: list[OpaqueActionV2] = field(default_factory=list)
    credit: dict[int, int] = field(default_factory=dict)
    credit_lineage: dict[int, tuple[int, ...]] = field(default_factory=dict)
    withdrawn_sources: set[int] = field(default_factory=set)
    receipts: list[ContactV1] = field(default_factory=list)
    trace: list[tuple[int, ...]] = field(default_factory=list)


class OpaqueContactAuthorityV2:
    """External membrane capability; its signing material is not resident state."""

    def __init__(self, key: bytes | None = None):
        self.__key = bytes(key or secrets.token_bytes(32))
        if len(self.__key) != 32:
            raise Refuse("opaque:authority")

    def seal(self, raw: ContactV1, session_epoch: int, ingress_sequence: int):
        sealed = copy.deepcopy(raw)
        sealed.authenticated = 1
        sealed.session_epoch = session_epoch
        sealed.ingress_sequence = ingress_sequence
        sealed.auth_tag = int.from_bytes(hmac.new(
            self.__key, _canonical(sealed.signed_fields()), hashlib.sha256
        ).digest()[:8], "little") & ((1 << 63) - 1)
        return sealed

    def verify(self, contact: ContactV1, tick: int, incarnation: int,
               session_epoch: int, ingress_sequence: int):
        contact.validate(tick, incarnation, session_epoch, ingress_sequence, self.__key)


class _OpaqueCheckpointAuthorityV2:
    """Distinct capability retained by checkpoint custody, never contact ingress."""

    def __init__(self):
        self.__key = secrets.token_bytes(32)

    def tag(self, body) -> int:
        return int.from_bytes(hmac.new(
            self.__key, b"opaque-checkpoint-v2\0" + _canonical(body), hashlib.sha256
        ).digest()[:8], "little") & ((1 << 63) - 1)


def authored_opaque_authorities_v2():
    """Workbench custody split: ingress receives only the first capability."""
    return OpaqueContactAuthorityV2(), _OpaqueCheckpointAuthorityV2()


class OpaqueRelationMachineV2:
    """One bounded consequence-revised relation law over opaque numeric ports."""

    def __init__(self, state: OpaqueRelationStateV2 | None = None,
                 authority: OpaqueContactAuthorityV2 | None = None,
                 checkpoint_authority: _OpaqueCheckpointAuthorityV2 | None = None,
                 _admission=None, work_limit=MAX_WORK_PER_TICK):
        if authority is None or checkpoint_authority is None:
            raise Refuse("opaque:authority")
        if state is not None and _admission is not _OPAQUE_RESTORE_ADMISSION:
            raise Refuse("opaque:state_install")
        self._authority = authority
        self.__checkpoint_authority = checkpoint_authority
        self._state = copy.deepcopy(state or OpaqueRelationStateV2(work_limit=work_limit))
        self._index: dict[tuple[int, int, int], list[int]] = {}
        self.last_touches = 0
        self._rebuild_index()
        self._bounded()
        self._validate_receipt_history()

    @property
    def state(self):
        """Observer snapshot; mutation never reaches resident state."""
        return copy.deepcopy(self._state)

    def ingress_coordinates(self):
        return (self._state.incarnation, self._state.tick,
                self._state.session_epoch, self._state.next_ingress_sequence)

    def _new_id(self):
        value = self._state.next_identity
        self._state.next_identity += 1
        return value

    def _charge(self, extent):
        self._state.work += int(extent)
        if self._state.work > self._state.work_limit:
            raise Refuse("opaque:work")

    def _append_trace(self, row):
        if len(self._state.trace) == MAX_TRACE:
            self._state.trace.pop(0)
        self._state.trace.append(tuple(row))

    def _retire_settled_action(self):
        if len(self._state.actions) < MAX_FROZEN_ACTION_CLOSURE:
            return
        for index, action in enumerate(self._state.actions):
            if action.settled or not action.active:
                self._state.actions.pop(index)
                return
        raise Refuse("opaque:action_bound")

    def _rebuild_index(self):
        self._index.clear()
        for index, row in enumerate(self._state.experiences):
            self._index.setdefault(
                (row.coalition, row.port_a, row.port_b), []).append(index)

    def _bounded(self):
        s = self._state
        for collection in (s.experiences, s.actions, s.receipts, s.trace,
                           s.credit, s.credit_lineage, s.withdrawn_sources):
            _strict_numeric(collection)
        if (len(s.experiences) > MAX_OPAQUE_EXPERIENCES
                or len(s.actions) > MAX_FROZEN_ACTION_CLOSURE
                or len(s.credit) > MAX_OPAQUE_EXPERIENCES
                or len(s.credit_lineage) > MAX_OPAQUE_EXPERIENCES
                or len(s.withdrawn_sources) > MAX_WITHDRAWN_SOURCES
                or len(s.receipts) > MAX_RELATIONS or len(s.trace) > MAX_TRACE):
            raise Refuse("opaque:state_bound")
        if (s.incarnation <= 0 or s.session_epoch <= 0 or s.tick < 0
                or s.next_ingress_sequence < 0 or not 0 < s.work_limit <= MAX_WORK_PER_TICK
                or not 0 <= s.work <= s.work_limit):
            raise Refuse("opaque:state_scalar")
        if s.current is not None:
            self._validate_parts(s.current.sites, s.current.present)
            if s.current.source in s.withdrawn_sources:
                raise Refuse("opaque:current_source")
        for row in s.experiences:
            self._validate_parts(row.sites, row.prior, row.later)
        rows = {row.identity: row for row in s.experiences}
        for action in s.actions:
            if (not action.recipe_identity or not action.contributors
                    or action.provisional not in {0, 1}
                    or len(action.payload) != len(action.ancestry)
                    or len(set(action.contributors)) != len(action.contributors)
                    or any(root not in rows for root in action.contributors)):
                raise Refuse("opaque:action_link")
            for offset, ancestry in enumerate(action.ancestry):
                if (ancestry.offset != offset or ancestry.unit != action.payload[offset]
                        or ancestry.constituent_root != action.recipe_identity
                        or ancestry.relation_roots != action.contributors
                        or ancestry.occurrence_identity != action.occurrence_identity):
                    raise Refuse("opaque:ancestry")
            if bool(action.settled) != bool(action.consequence_identity):
                raise Refuse("opaque:settlement")
        if (any(type(k) is not int or k <= 0 or type(v) is not int
                or not -MAX_FROZEN_ACTION_CLOSURE <= v <= MAX_FROZEN_ACTION_CLOSURE
                for k,v in s.credit.items())
                or set(s.credit_lineage) != set(s.credit)
                or any(not roots or len(set(roots)) != len(roots)
                       or any(root not in rows for root in roots)
                       for roots in s.credit_lineage.values())
                or any(type(x) is not int or not 0 < x < (1 << 31)
                       for x in s.withdrawn_sources)):
            raise Refuse("opaque:credit_source")
        identities = [x.identity for x in s.experiences]
        identities += [x.occurrence_identity for x in s.experiences]
        if s.current is not None:
            identities.append(s.current.identity)
        identities += [a.ticket for a in s.actions] + [a.occurrence_identity for a in s.actions]
        identities += [a.consequence_identity for a in s.actions if a.consequence_identity]
        if (any(x <= 0 for x in identities) or len(identities) != len(set(identities))
                or s.next_identity <= max(identities, default=0)):
            raise Refuse("opaque:identity")

    @staticmethod
    def _validate_parts(sites, *sequences):
        if not sites or len(sites) > MAX_OPAQUE_SITES or len(set(sites)) != len(sites):
            raise Refuse("opaque:sites")
        if any(type(x) is not int or x <= 0 for x in sites):
            raise Refuse("opaque:sites")
        for values in sequences:
            if not values or len(values) > MAX_OPAQUE_UNITS:
                raise Refuse("opaque:units")
            if any(type(x) is not int or not 0 <= x <= 255 for x in values):
                raise Refuse("opaque:units")

    def _validate_receipt_history(self):
        if len(self._state.receipts) > MAX_RELATIONS:
            raise Refuse("opaque:receipt_bound")
        first = self._state.next_ingress_sequence - len(self._state.receipts)
        if first < 0:
            raise Refuse("opaque:receipt_sequence")
        for offset, receipt in enumerate(self._state.receipts):
            self._authority.verify(receipt, min(self._state.tick, receipt.deadline),
                                   self._state.incarnation, self._state.session_epoch,
                                   first + offset)

    @staticmethod
    def _parts(payload, experience):
        p = tuple(payload)
        if len(p) < 7:
            raise Refuse("opaque:shape")
        coalition, port_a, port_b, site_count = p[:4]
        pos = 4
        sites = p[pos:pos + site_count]
        pos += site_count
        if pos >= len(p):
            raise Refuse("opaque:shape")
        base_count = p[pos]
        pos += 1
        base = p[pos:pos + base_count]
        pos += base_count
        later = ()
        if experience:
            if pos >= len(p):
                raise Refuse("opaque:shape")
            later_count = p[pos]
            pos += 1
            later = p[pos:pos + later_count]
            pos += later_count
        if pos != len(p) or any(x <= 0 for x in (coalition, port_a, port_b)):
            raise Refuse("opaque:shape")
        OpaqueRelationMachineV2._validate_parts(sites, base, *(later,) if experience else ())
        return coalition, port_a, port_b, tuple(sites), tuple(base), tuple(later)

    def contact(self, contact: ContactV1):
        if type(contact) is not ContactV1:
            raise Refuse("opaque:contact_type")
        receipt = copy.deepcopy(contact)
        self._authority.verify(receipt, self._state.tick, self._state.incarnation,
                               self._state.session_epoch,
                               self._state.next_ingress_sequence)
        if receipt.source in self._state.withdrawn_sources:
            raise Refuse("opaque:source_withdrawn")
        if any(a.active and not a.settled for a in self._state.actions):
            if receipt.kind not in {CONTACT_CONSEQUENCE, OPAQUE_WITHDRAW}:
                raise Refuse("opaque:pending_consequence")
        if receipt.kind == CONTACT_CONSEQUENCE:
            result = self._settle(receipt)
        elif receipt.kind == OPAQUE_EXPERIENCE:
            result = self._experience(receipt)
        elif receipt.kind == OPAQUE_CURRENT:
            result = self._current(receipt)
        elif receipt.kind == OPAQUE_WITHDRAW:
            result = self._withdraw(receipt)
        else:
            raise Refuse("opaque:kind")
        if len(self._state.receipts) == MAX_RELATIONS:
            self._state.receipts.pop(0)
        self._state.receipts.append(receipt)
        self._state.next_ingress_sequence += 1
        return result

    def _experience(self, contact):
        if len(self._state.experiences) >= MAX_OPAQUE_EXPERIENCES:
            raise Refuse("opaque:experience_bound")
        coalition, port_a, port_b, sites, prior, later = self._parts(contact.payload, True)
        identity, occurrence = self._new_id(), self._new_id()
        row = OpaqueExperienceV2(identity, contact.source, coalition, port_a,
                                 port_b, sites, prior, later, occurrence)
        self._state.experiences.append(row)
        self._index.setdefault((coalition, port_a, port_b), []).append(
            len(self._state.experiences) - 1)
        self._append_trace((self._state.tick, OPAQUE_EXPERIENCE, identity, occurrence))
        return identity

    def _current(self, contact):
        if self._state.current is not None:
            raise Refuse("opaque:current_pending")
        coalition, port_a, port_b, sites, present, _ = self._parts(contact.payload, False)
        identity = self._new_id()
        self._state.current = OpaqueCurrentV2(identity, contact.source, contact.channel,
                                             coalition, port_a, port_b, sites, present)
        return identity

    @staticmethod
    def _edit(prior, later):
        prefix = 0
        while prefix < min(len(prior), len(later)) and prior[prefix] == later[prefix]:
            prefix += 1
        return len(prior) - prefix, later[prefix:]

    def tick(self):
        before = (self._state.tick, self._state.work, self._state.next_identity,
                  copy.deepcopy(self._state.current), list(self._state.actions),
                  self.last_touches)
        try:
            self._state.tick += 1
            self._state.work = 0
            if any(a.active and not a.settled for a in self._state.actions):
                raise Refuse("opaque:pending_consequence")
            current = self._state.current
            if current is None:
                return None
            indices = self._index.get(
                (current.coalition, current.port_a, current.port_b), ())
            self.last_touches = len(indices)
            groups = {}
            for index in indices:
                row = self._state.experiences[index]
                self._charge(1 + len(row.sites) + len(row.prior) + len(row.later))
                if not row.active or row.source in self._state.withdrawn_sources:
                    continue
                groups.setdefault(self._edit(row.prior, row.later), []).append(row)
            candidates = []
            for edit, rows in groups.items():
                sources = {r.source for r in rows}
                support = set(rows[0].sites)
                for row in rows[1:]:
                    self._charge(len(support) + len(row.sites))
                    support.intersection_update(row.sites)
                if len(rows) < 2 or len(sources) < 2 or not support.issubset(current.sites):
                    continue
                drop, append = edit
                if not support or drop > len(current.present):
                    continue
                output = current.present[:len(current.present) - drop] + append
                self._charge(len(current.present) + len(append) + len(output))
                contributors = tuple(sorted(r.identity for r in rows))
                recipe = int(digest("opaque-relation-recipe-v2", [
                    current.coalition, current.port_a, current.port_b,
                    sorted(support), drop, list(append)])[:15], 16) or 1
                candidates.append((self._state.credit.get(recipe, 0), output,
                                   recipe, contributors))
            if not candidates:
                raise Refuse("opaque:no_candidate")
            peak = max(x[0] for x in candidates)
            if peak < 0:
                raise Refuse("opaque:no_candidate")
            winners = [x for x in candidates if x[0] == peak]
            if len(winners) != 1:
                raise Refuse("opaque:ambiguous")
            _, output, recipe, contributors = winners[0]
            occurrence = self._new_id()
            action = self._action(current, occurrence, recipe, output, contributors,
                                  int(peak <= 0))
            self._state.current = None
            return copy.deepcopy(action)
        except Exception:
            (self._state.tick, self._state.work, self._state.next_identity,
             self._state.current, self._state.actions, self.last_touches) = before
            raise

    def _action(self, contact, occurrence, recipe, payload, contributors, provisional):
        self._retire_settled_action()
        ticket = self._new_id()
        ancestry = tuple(ByteAncestryV1(i, unit, recipe, contributors,
                                        occurrence, 1) for i, unit in enumerate(payload))
        action = OpaqueActionV2(ticket, self._state.incarnation, self._state.tick,
                               self._state.tick + 4, contact.channel, contact.source,
                               occurrence, recipe, tuple(payload),
                               tuple(contributors), ancestry, provisional)
        self._state.actions.append(action)
        return action

    def _settle(self, contact):
        action = next((a for a in self._state.actions if a.ticket == contact.ticket), None)
        if action is None or action.settled or not action.active:
            raise Refuse("opaque:consequence_ticket")
        if (contact.incarnation, contact.deadline, contact.channel, contact.source) != (
                action.incarnation, action.deadline, action.channel, action.source):
            raise Refuse("opaque:consequence_binding")
        if len(contact.payload) != 2:
            raise Refuse("opaque:consequence_shape")
        effect, counterfactual = contact.payload
        difference = contact.independent and effect != counterfactual
        if (difference and action.recipe_identity not in self._state.credit
                and len(self._state.credit) >= MAX_OPAQUE_EXPERIENCES):
            raise Refuse("opaque:credit_bound")
        action.settled = 1
        action.consequence_identity = self._new_id()
        if difference:
            revised = (self._state.credit.get(action.recipe_identity, 0)
                       + (1 if effect > counterfactual else -1))
            self._state.credit[action.recipe_identity] = max(
                -MAX_FROZEN_ACTION_CLOSURE,
                min(MAX_FROZEN_ACTION_CLOSURE, revised))
            prior = self._state.credit_lineage.get(action.recipe_identity, ())
            self._state.credit_lineage[action.recipe_identity] = tuple(sorted(
                set(prior).union(action.contributors)))
        self._append_trace((self._state.tick, OP_SETTLE, action.ticket,
                            action.consequence_identity, int(bool(difference))))
        return action.consequence_identity

    def _withdraw(self, contact):
        if len(contact.payload) != 1:
            raise Refuse("opaque:withdraw_shape")
        source = contact.payload[0]
        if type(source) is not int or not 0 < source < (1 << 31):
            raise Refuse("opaque:withdraw_source")
        if (source not in self._state.withdrawn_sources
                and len(self._state.withdrawn_sources) >= MAX_WITHDRAWN_SOURCES):
            raise Refuse("opaque:withdraw_bound")
        self._state.withdrawn_sources.add(source)
        for row in self._state.experiences:
            if row.source == source:
                row.active = 0
        if self._state.current is not None and self._state.current.source == source:
            self._state.current = None
        for action in self._state.actions:
            if action.source == source or any(
                    next((r.source for r in self._state.experiences
                          if r.identity == contributor), 0) == source
                    for contributor in action.contributors):
                action.active = 0
        tainted = [recipe for recipe, roots in self._state.credit_lineage.items()
                   if any(rowsource == source for rowsource in (
                       next((r.source for r in self._state.experiences
                             if r.identity == root), 0) for root in roots))]
        for recipe in tainted:
            self._state.credit.pop(recipe, None)
            self._state.credit_lineage.pop(recipe, None)
        self._append_trace((self._state.tick, OPAQUE_WITHDRAW, source))
        return source

    def checkpoint(self):
        body = self._checkpoint_body()
        signed = {"version": OPAQUE_CHECKPOINT_VERSION, "body": body}
        envelope = {**signed, "auth_tag": self.__checkpoint_authority.tag(signed)}
        blob = _canonical(envelope)
        if len(blob) > MAX_CHECKPOINT_BYTES:
            raise Refuse("opaque:checkpoint_bound")
        return blob

    @classmethod
    def restore(cls, blob, authority: OpaqueContactAuthorityV2,
                checkpoint_authority: _OpaqueCheckpointAuthorityV2):
        if len(blob) > MAX_CHECKPOINT_BYTES:
            raise Refuse("opaque:checkpoint_bound")
        try:
            envelope = json.loads(blob)
        except Exception as exc:
            raise Refuse("opaque:checkpoint_decode") from exc
        if (not isinstance(envelope, dict) or set(envelope) != {"version", "body", "auth_tag"}
                or envelope["version"] != OPAQUE_CHECKPOINT_VERSION
                or envelope["auth_tag"] != checkpoint_authority.tag(
                    {"version": envelope["version"], "body": envelope["body"]})):
            raise Refuse("opaque:checkpoint_corrupt")
        b = envelope["body"]
        try:
            if set(b) != set(OpaqueRelationStateV2.__dataclass_fields__):
                raise ValueError("state fields")
            for x in b["experiences"]:
                if set(x) != set(OpaqueExperienceV2.__dataclass_fields__):
                    raise ValueError("experience fields")
            for x in b["actions"]:
                if set(x) != set(OpaqueActionV2.__dataclass_fields__):
                    raise ValueError("action fields")
                if any(set(a) != set(ByteAncestryV1.__dataclass_fields__)
                       for a in x["ancestry"]):
                    raise ValueError("ancestry fields")
            for x in b["receipts"]:
                if set(x) != set(ContactV1.__dataclass_fields__):
                    raise ValueError("receipt fields")
            current = b["current"]
            if current is not None and set(current) != set(OpaqueCurrentV2.__dataclass_fields__):
                raise ValueError("current fields")
            state = OpaqueRelationStateV2(
                incarnation=b["incarnation"], tick=b["tick"],
                next_identity=b["next_identity"],
                next_ingress_sequence=b["next_ingress_sequence"],
                session_epoch=b["session_epoch"], work=b["work"],
                work_limit=b["work_limit"],
                experiences=[OpaqueExperienceV2(
                    x["identity"], x["source"], x["coalition"], x["port_a"],
                    x["port_b"], tuple(x["sites"]), tuple(x["prior"]),
                    tuple(x["later"]), x["occurrence_identity"], x["active"])
                    for x in b["experiences"]],
                current=None if current is None else OpaqueCurrentV2(
                    current["identity"], current["source"], current["channel"],
                    current["coalition"], current["port_a"], current["port_b"],
                    tuple(current["sites"]), tuple(current["present"])),
                actions=[OpaqueActionV2(
                    x["ticket"], x["incarnation"], x["born_tick"],
                    x["deadline"], x["channel"], x["source"],
                    x["occurrence_identity"], x["recipe_identity"],
                    tuple(x["payload"]), tuple(x["contributors"]),
                    tuple(ByteAncestryV1(
                        a["offset"], a["unit"], a["constituent_root"],
                        tuple(a["relation_roots"]), a["occurrence_identity"],
                        a["ancestry_depth"]) for a in x["ancestry"]),
                    x["provisional"], x["settled"],
                    x["consequence_identity"], x["active"])
                    for x in b["actions"]],
                credit={int(k): int(v) for k, v in b["credit"].items()},
                credit_lineage={int(k): tuple(v)
                                for k,v in b["credit_lineage"].items()},
                withdrawn_sources=set(b["withdrawn_sources"]),
                receipts=[ContactV1(
                    x["ticket"], x["incarnation"], x["deadline"], x["source"],
                    x["channel"], x["kind"], tuple(x["payload"]),
                    x["authenticated"], x["independent"], x["session_epoch"],
                    x["ingress_sequence"], x["auth_tag"])
                    for x in b["receipts"]],
                trace=[tuple(x) for x in b["trace"]])
        except Exception as exc:
            raise Refuse("opaque:checkpoint_shape") from exc
        return cls(state, authority, checkpoint_authority, _OPAQUE_RESTORE_ADMISSION)

    def _checkpoint_body(self):
        body = asdict(self._state)
        body["withdrawn_sources"] = sorted(self._state.withdrawn_sources)
        return body

    def state_hash(self):
        return digest("opaque-relation-state-v2", self._checkpoint_body())
