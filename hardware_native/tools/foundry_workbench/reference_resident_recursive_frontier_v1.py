#!/usr/bin/env python3
"""Event-sourced graph-neutral resident recursive-frontier reference kernel.

An admitted boundary authority authenticates flat numeric contacts but cannot
submit candidate state, a tree, winner, or Recipe. Checkpoints contain only the
authenticated event journal; restore recomputes resident state through public
operations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import hmac
import json
import secrets

SCHEMA_VERSION = CHECKPOINT_VERSION = 1
MAX_PAYLOAD, MAX_PROVENANCE = 64, 32
MAX_OCCURRENCES, MAX_PENDING, MAX_RECIPES, MAX_WITNESSES = 64, 16, 8, 64
MAX_EVENTS = MAX_TRACE = 4096
MAX_WORK, MAX_DEPTH, MAX_CHECKPOINT_BYTES, DEADLINE_TICKS = 512, 3, 1 << 20, 4
CONTACT_OCCURRENCE, CONTACT_CONSEQUENCE, CONTACT_WITHDRAWAL = 1, 2, 3
EVENT_TICK, EVENT_UNFOLD = 4, 5
CONSTRUCTOR_LEFT, CONSTRUCTOR_RIGHT = 1, 2
_BOUNDARY_ADMISSION = object()


class FrontierRefuse(RuntimeError):
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
        raise FrontierRefuse(f"frontier:type:{path}")
    if isinstance(value, int):
        if not -(1 << 63) <= value < (1 << 63):
            raise FrontierRefuse(f"frontier:integer:{path}")
        return
    if hasattr(value, "__dataclass_fields__"):
        for name in value.__dataclass_fields__:
            _strict_numeric(getattr(value, name), f"{path}.{name}", extent)
        return
    if isinstance(value, (tuple, list)):
        if len(value) > extent:
            raise FrontierRefuse(f"frontier:bound:{path}")
        for index, item in enumerate(value):
            _strict_numeric(item, f"{path}[{index}]", extent)
        return
    raise FrontierRefuse(f"frontier:type:{path}")


@dataclass(frozen=True)
class OccurrenceContactV1:
    session: int
    sequence: int
    source: int
    channel: int
    features: tuple[int, ...]
    provenance: tuple[int, ...]
    auth_tag: int

    def signed_fields(self):
        return (self.session, self.sequence, self.source, self.channel,
                self.features, self.provenance)


@dataclass(frozen=True)
class ConsequenceContactV1:
    session: int
    sequence: int
    ticket: int
    incarnation: int
    deadline: int
    channel: int
    source: int
    actual: int
    baseline: int
    auth_tag: int

    def signed_fields(self):
        return (self.session, self.sequence, self.ticket, self.incarnation,
                self.deadline, self.channel, self.source, self.actual, self.baseline)


@dataclass(frozen=True)
class WithdrawalContactV1:
    session: int
    sequence: int
    source: int
    channel: int
    target_source: int
    auth_tag: int

    def signed_fields(self):
        return (self.session, self.sequence, self.source, self.channel,
                self.target_source)


@dataclass(frozen=True)
class EventV1:
    kind: int
    values: tuple


@dataclass(frozen=True)
class OccurrenceV1:
    identity: int
    tick: int
    rank: int
    channel: int
    source_roots: tuple[int, ...]
    provenance: tuple[int, ...]
    features: tuple[int, ...]


@dataclass(frozen=True)
class CandidateV1:
    identity: int
    incarnation: int
    born_tick: int
    deadline: int
    rank: int
    constructor: int
    members: tuple[int, ...]
    member_ranks: tuple[int, ...]
    witness_roots: tuple[int, ...]
    source_roots: tuple[int, ...]
    provenance_root: int
    status: int


@dataclass(frozen=True)
class CandidateTicketV1:
    ticket: int
    incarnation: int
    deadline: int
    channel: int
    source: int
    candidate: CandidateV1


@dataclass(frozen=True)
class ConstructorWitnessV1:
    identity: int
    constructor: int
    source: int
    candidate: int
    consequence_sequence: int
    difference: int
    provenance_root: int
    source_roots: tuple[int, ...]


@dataclass(frozen=True)
class CompactConstructorV1:
    identity: int
    constructor: int
    support: int
    credit: int
    witness_roots: tuple[int, ...]


class ReferenceBoundaryAuthorityV1:
    """Per-instance authority admitted only by ``admit_reference_boundary_v1``."""

    def __init__(self, admission=None):
        if admission is not _BOUNDARY_ADMISSION:
            raise FrontierRefuse("frontier:boundary_admission")
        self.__contact_key = secrets.token_bytes(32)
        self.__checkpoint_key = secrets.token_bytes(32)
        self.__admitted = True

    def _is_admitted(self):
        return self.__admitted

    def _contact_tag(self, kind: int, fields) -> int:
        return int.from_bytes(hmac.new(
            self.__contact_key, _canonical((int(kind), fields)), hashlib.sha256,
        ).digest()[:8], "little") & ((1 << 63) - 1)

    def _valid_contact(self, kind: int, fields, tag: int) -> bool:
        expected = self._contact_tag(kind, fields)
        return hmac.compare_digest(int(tag).to_bytes(8, "little"),
                                   expected.to_bytes(8, "little"))

    def _checkpoint_tag(self, body) -> str:
        return hmac.new(self.__checkpoint_key, _canonical(body), hashlib.sha256).hexdigest()

    def _valid_checkpoint(self, body, tag) -> bool:
        return isinstance(tag, str) and hmac.compare_digest(tag, self._checkpoint_tag(body))

    def seal_occurrence(self, session: int, sequence: int, source: int,
                        channel: int, features, provenance=()) -> OccurrenceContactV1:
        _strict_numeric((session, sequence, source, channel, features, provenance))
        features = tuple(map(int, features)); provenance = tuple(map(int, provenance))
        fields = (int(session), int(sequence), int(source), int(channel),
                  features, provenance)
        return OccurrenceContactV1(*fields, self._contact_tag(CONTACT_OCCURRENCE, fields))

    def seal_consequence(self, session: int, sequence: int,
                         ticket: CandidateTicketV1, actual: int,
                         baseline: int) -> ConsequenceContactV1:
        if not isinstance(ticket, CandidateTicketV1):
            raise FrontierRefuse("frontier:boundary_ticket")
        _strict_numeric((session, sequence, ticket, actual, baseline))
        fields = (int(session), int(sequence), ticket.ticket, ticket.incarnation,
                  ticket.deadline, ticket.channel, ticket.source,
                  int(actual), int(baseline))
        return ConsequenceContactV1(
            *fields, self._contact_tag(CONTACT_CONSEQUENCE, fields))

    def seal_withdrawal(self, session: int, sequence: int, source: int,
                        channel: int, target_source: int) -> WithdrawalContactV1:
        _strict_numeric((session, sequence, source, channel, target_source))
        fields = (int(session), int(sequence), int(source), int(channel),
                  int(target_source))
        return WithdrawalContactV1(
            *fields, self._contact_tag(CONTACT_WITHDRAWAL, fields))


def admit_reference_boundary_v1() -> ReferenceBoundaryAuthorityV1:
    return ReferenceBoundaryAuthorityV1(_BOUNDARY_ADMISSION)


class ResidentRecursiveFrontierV1:
    """Bounded machine whose only durable authority is its numeric event log."""

    def __init__(self, authority: ReferenceBoundaryAuthorityV1,
                 session: int = 1, work_limit: int = MAX_WORK):
        if (not isinstance(authority, ReferenceBoundaryAuthorityV1)
                or not authority._is_admitted()):
            raise FrontierRefuse("frontier:boundary_authority")
        if session <= 0 or not 1 <= work_limit <= MAX_WORK:
            raise FrontierRefuse("frontier:configuration")
        self._authority = authority
        self.session, self.incarnation = int(session), 1
        self.next_sequence = self.next_identity = 1
        self.tick_count = self.work = 0
        self.work_limit = int(work_limit)
        self.events: list[EventV1] = []
        self.occurrences: list[OccurrenceV1] = []
        self.frontier: list[int] = []
        self.pending: dict[int, CandidateTicketV1] = {}
        self.witnesses: list[ConstructorWitnessV1] = []
        self.recipes: dict[int, CompactConstructorV1] = {}
        self.withdrawn_sources: set[int] = set()
        self.trace: list[tuple[int, ...]] = []

    def _new_id(self):
        value = self.next_identity; self.next_identity += 1; return value

    def _append_trace(self, *values):
        if len(self.trace) >= MAX_TRACE:
            raise FrontierRefuse("frontier:trace_bound")
        self.trace.append(tuple(map(int, values)))

    def _charge(self, amount):
        self.work += int(amount)
        if self.work > self.work_limit:
            raise FrontierRefuse("frontier:resource")

    def _verify(self, kind, contact):
        _strict_numeric(contact); fields = contact.signed_fields()
        if contact.session != self.session or contact.sequence != self.next_sequence:
            raise FrontierRefuse("frontier:session_sequence")
        if not self._authority._valid_contact(kind, fields, contact.auth_tag):
            raise FrontierRefuse("frontier:authentication")

    @staticmethod
    def _contact_event(kind, contact):
        return EventV1(kind, (contact.signed_fields(), contact.auth_tag))

    def _record(self, event):
        _strict_numeric(event)
        if len(self.events) >= MAX_EVENTS:
            raise FrontierRefuse("frontier:event_bound")
        self.events.append(event)

    def _snapshot(self):
        return {key: copy.deepcopy(value) for key, value in self.__dict__.items()
                if key != "_authority"}

    def _rollback(self, state):
        authority = self._authority; self.__dict__.clear()
        self.__dict__["_authority"] = authority; self.__dict__.update(state)

    def ingest_occurrence(self, contact: OccurrenceContactV1) -> OccurrenceV1:
        before = self._snapshot()
        try:
            if self.pending:
                raise FrontierRefuse("frontier:pending_consequence")
            self._verify(CONTACT_OCCURRENCE, contact)
            if (contact.source <= 0 or contact.channel <= 0
                    or contact.source in self.withdrawn_sources
                    or not 0 < len(contact.features) <= MAX_PAYLOAD
                    or len(contact.provenance) > MAX_PROVENANCE
                    or any(root <= 0 for root in contact.provenance)):
                raise FrontierRefuse("frontier:occurrence_shape")
            if len(self.occurrences) >= MAX_OCCURRENCES:
                raise FrontierRefuse("frontier:occurrence_bound")
            identity = _identity(b"resident-flat-occurrence-v1", contact.signed_fields())
            if any(row.identity == identity for row in self.occurrences):
                raise FrontierRefuse("frontier:occurrence_duplicate")
            occurrence = OccurrenceV1(identity, self.tick_count, 0, contact.channel,
                                      (contact.source,), contact.provenance, contact.features)
            self.occurrences.append(occurrence); self.frontier.append(identity)
            self.next_sequence += 1
            self._append_trace(self.tick_count, CONTACT_OCCURRENCE, identity, contact.source)
            self._record(self._contact_event(CONTACT_OCCURRENCE, contact))
            self._bounded(); self._check_checkpoint_bound(); return occurrence
        except Exception:
            self._rollback(before); raise

    def tick(self) -> tuple[CandidateTicketV1, ...]:
        if self.pending:
            raise FrontierRefuse("frontier:pending_consequence")
        before = self.checkpoint()
        try:
            self.tick_count += 1; self.work = 0
            live = [self._occurrence(identity) for identity in self.frontier]
            if len(live) < 2:
                tickets = (); self._append_trace(self.tick_count, 0, len(live), 0)
            else:
                left, right = live[-2:]
                tickets = tuple(self._propose(kind, left, right)
                                for kind in (CONSTRUCTOR_LEFT, CONSTRUCTOR_RIGHT))
                self._charge(len(tickets) * 8)
                self._append_trace(self.tick_count, EVENT_TICK,
                                   tickets[0].ticket, tickets[1].ticket)
            self._record(EventV1(EVENT_TICK, (tuple(row.ticket for row in tickets),)))
            self._bounded(); self._check_checkpoint_bound(); return tickets
        except Exception:
            restored = type(self).restore(before, self._authority)
            self.__dict__.update(restored.__dict__); raise

    def _propose(self, constructor, left, right):
        members = ((left, right) if constructor == CONSTRUCTOR_LEFT else (right, left))
        sources = tuple(sorted(set(members[0].source_roots + members[1].source_roots)))
        roots = tuple(row.identity for row in members)
        provenance = tuple(sorted(set(members[0].provenance + members[1].provenance)))
        provenance_root = _identity(b"resident-candidate-provenance-v1", provenance or roots)
        candidate = CandidateV1(
            _identity(b"resident-recursive-candidate-v1", [self.incarnation,
                self.tick_count, constructor, roots, tuple(row.rank for row in members),
                sources, provenance_root]), self.incarnation, self.tick_count,
            self.tick_count + DEADLINE_TICKS, 1, constructor, roots,
            tuple(row.rank for row in members), roots, sources, provenance_root, 1)
        ticket = CandidateTicketV1(self._new_id(), self.incarnation,
            candidate.deadline, members[-1].channel, members[-1].source_roots[-1], candidate)
        if len(self.pending) >= MAX_PENDING:
            raise FrontierRefuse("frontier:pending_bound")
        self.pending[ticket.ticket] = ticket; return ticket

    def ingest_consequence(self, contact: ConsequenceContactV1) -> int:
        before = self._snapshot()
        try:
            self._verify(CONTACT_CONSEQUENCE, contact)
            pending = self.pending.get(contact.ticket)
            if pending is None:
                raise FrontierRefuse("frontier:ticket")
            if (contact.incarnation != pending.incarnation
                    or contact.deadline != pending.deadline
                    or contact.channel != pending.channel or contact.source != pending.source
                    or self.tick_count > pending.deadline):
                raise FrontierRefuse("frontier:consequence_binding")
            difference = max(-127, min(127, contact.actual - contact.baseline))
            if difference and len(self.witnesses) >= MAX_WITNESSES:
                raise FrontierRefuse("frontier:witness_bound")
            self.pending.pop(contact.ticket); self.next_sequence += 1
            if difference:
                witness = ConstructorWitnessV1(_identity(
                    b"resident-constructor-witness-v1",
                    [pending.candidate.identity, contact.sequence, difference]),
                    pending.candidate.constructor, contact.source, pending.candidate.identity,
                    contact.sequence, difference, pending.candidate.provenance_root,
                    pending.candidate.source_roots)
                self.witnesses.append(witness); self._refresh_recipe(witness.constructor)
            self._append_trace(self.tick_count, CONTACT_CONSEQUENCE,
                               contact.ticket, difference)
            self._record(self._contact_event(CONTACT_CONSEQUENCE, contact))
            self._bounded(); self._check_checkpoint_bound(); return difference
        except Exception:
            self._rollback(before); raise

    def _refresh_recipe(self, constructor):
        rows = [row for row in self.witnesses if row.constructor == constructor
                and not set(row.source_roots).intersection(self.withdrawn_sources)]
        sources = {row.source for row in rows if row.difference > 0}
        credit = sum(row.difference for row in rows)
        if len(sources) >= 2 and credit > 0:
            self.recipes[constructor] = CompactConstructorV1(
                _identity(b"resident-compact-constructor-v1", [constructor]),
                constructor, len(sources), credit,
                tuple(sorted(row.identity for row in rows if row.difference > 0)))
        else:
            self.recipes.pop(constructor, None)

    def ingest_withdrawal(self, contact: WithdrawalContactV1) -> None:
        before = self._snapshot()
        try:
            self._verify(CONTACT_WITHDRAWAL, contact)
            if contact.source <= 0 or contact.channel <= 0 or contact.target_source <= 0:
                raise FrontierRefuse("frontier:withdrawal_shape")
            target = contact.target_source; self.next_sequence += 1
            self.withdrawn_sources.add(target)
            cancelled = [ticket for ticket, row in self.pending.items()
                         if target in row.candidate.source_roots]
            self.pending = {ticket: row for ticket, row in self.pending.items()
                            if ticket not in cancelled}
            removed = {row.identity for row in self.occurrences
                       if target in row.source_roots}
            self.occurrences = [row for row in self.occurrences if row.identity not in removed]
            self.frontier = [identity for identity in self.frontier if identity not in removed]
            self.witnesses = [row for row in self.witnesses if target not in row.source_roots]
            for constructor in (CONSTRUCTOR_LEFT, CONSTRUCTOR_RIGHT):
                self._refresh_recipe(constructor)
            self._append_trace(self.tick_count, CONTACT_WITHDRAWAL, target,
                               len(removed), len(cancelled))
            self._record(self._contact_event(CONTACT_WITHDRAWAL, contact))
            self._bounded(); self._check_checkpoint_bound()
        except Exception:
            self._rollback(before); raise

    def unfold(self) -> tuple[OccurrenceV1, ...]:
        if self.pending:
            raise FrontierRefuse("frontier:pending_consequence")
        before = self.checkpoint()
        try:
            candidates = [row for row in self.recipes.values() if row.credit > 0]
            if not candidates:
                raise FrontierRefuse("frontier:no_constructor")
            peak = max(row.credit for row in candidates)
            winners = [row for row in candidates if row.credit == peak]
            if len(winners) != 1:
                raise FrontierRefuse("frontier:ambiguous")
            bases = [self._occurrence(identity) for identity in self.frontier[-4:]]
            if len(bases) < 2:
                raise FrontierRefuse("frontier:insufficient_occurrences")
            recipe, current, outputs = winners[0], bases[0], []; self.work = 0
            for other in bases[1:]:
                members = ((current, other) if recipe.constructor == CONSTRUCTOR_LEFT
                           else (other, current))
                rank = max(row.rank for row in members) + 1
                if rank > MAX_DEPTH: break
                sources = tuple(sorted(set(members[0].source_roots + members[1].source_roots)))
                provenance = tuple(sorted(set(members[0].provenance + members[1].provenance)))
                if len(sources) > MAX_WITNESSES or len(provenance) > MAX_PROVENANCE:
                    raise FrontierRefuse("frontier:recursive_provenance_bound")
                identity = _identity(b"resident-ephemeral-recursive-occurrence-v1",
                    [recipe.identity, rank, tuple(row.identity for row in members),
                     sources, provenance])
                current = OccurrenceV1(identity, self.tick_count, rank, members[-1].channel,
                    sources, provenance, (recipe.identity,) + tuple(row.identity for row in members))
                outputs.append(current); self._charge(8 + len(sources) + len(provenance))
                self._append_trace(self.tick_count, EVENT_UNFOLD,
                                   recipe.identity, identity, rank)
            self._record(EventV1(EVENT_UNFOLD, (tuple(row.identity for row in outputs),
                                                tuple(row.rank for row in outputs))))
            self._bounded(); self._check_checkpoint_bound(); return tuple(outputs)
        except Exception:
            restored = type(self).restore(before, self._authority)
            self.__dict__.update(restored.__dict__); raise

    def _occurrence(self, identity):
        row = next((row for row in self.occurrences if row.identity == int(identity)), None)
        if row is None: raise FrontierRefuse("frontier:occurrence")
        return row

    def _bounded(self):
        if (len(self.events) > MAX_EVENTS or len(self.occurrences) > MAX_OCCURRENCES
                or len(self.frontier) > MAX_OCCURRENCES or len(self.pending) > MAX_PENDING
                or len(self.recipes) > MAX_RECIPES or len(self.witnesses) > MAX_WITNESSES
                or len(self.withdrawn_sources) > MAX_WITNESSES
                or len(self.trace) > MAX_TRACE or self.work > self.work_limit):
            raise FrontierRefuse("frontier:state_bound")
        for rows in (self.events, self.occurrences, tuple(self.pending.values()),
                     self.witnesses, tuple(self.recipes.values()), self.trace):
            _strict_numeric(tuple(rows))
        if any(not 0 < len(row.features) <= MAX_PAYLOAD
               or len(row.provenance) > MAX_PROVENANCE for row in self.occurrences):
            raise FrontierRefuse("frontier:occurrence_payload_bound")
        occurrence_ids = {row.identity for row in self.occurrences}
        if any(identity not in occurrence_ids for identity in self.frontier):
            raise FrontierRefuse("frontier:frontier_occurrence")
        if any(target in row.candidate.source_roots for target in self.withdrawn_sources
               for row in self.pending.values()):
            raise FrontierRefuse("frontier:withdrawal_pending")

    def _checkpoint_body(self):
        return {"schema": SCHEMA_VERSION, "session": self.session,
                "incarnation": self.incarnation, "work_limit": self.work_limit,
                "events": [asdict(row) for row in self.events]}

    def _check_checkpoint_bound(self):
        if len(self.checkpoint()) > MAX_CHECKPOINT_BYTES:
            raise FrontierRefuse("frontier:checkpoint_bound")

    def checkpoint(self) -> bytes:
        body = self._checkpoint_body()
        blob = _canonical({"version": CHECKPOINT_VERSION, "body": body,
                           "hmac": self._authority._checkpoint_tag(body)})
        if len(blob) > MAX_CHECKPOINT_BYTES:
            raise FrontierRefuse("frontier:checkpoint_bound")
        return blob

    @classmethod
    def restore(cls, blob: bytes, authority: ReferenceBoundaryAuthorityV1):
        if len(bytes(blob)) > MAX_CHECKPOINT_BYTES:
            raise FrontierRefuse("frontier:checkpoint_bound")
        try:
            envelope = json.loads(bytes(blob)); body = envelope["body"]
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise FrontierRefuse("frontier:checkpoint") from exc
        if (not isinstance(envelope, dict) or set(envelope) != {"version", "body", "hmac"}
                or envelope["version"] != CHECKPOINT_VERSION
                or not authority._valid_checkpoint(body, envelope["hmac"])):
            raise FrontierRefuse("frontier:checkpoint_authentication")
        if (not isinstance(body, dict)
                or set(body) != {"schema", "session", "incarnation", "work_limit", "events"}
                or body["schema"] != SCHEMA_VERSION or body["incarnation"] != 1
                or len(body["events"]) > MAX_EVENTS):
            raise FrontierRefuse("frontier:checkpoint_schema")
        out = cls(authority, int(body["session"]), int(body["work_limit"]))
        for raw in body["events"]:
            if not isinstance(raw, dict) or set(raw) != {"kind", "values"}:
                raise FrontierRefuse("frontier:event_shape")
            event = EventV1(int(raw["kind"]), _tuplify(raw["values"])); before = len(out.events)
            if event.kind == CONTACT_OCCURRENCE:
                fields, tag = event.values
                out.ingest_occurrence(OccurrenceContactV1(int(fields[0]), int(fields[1]),
                    int(fields[2]), int(fields[3]), tuple(map(int, fields[4])),
                    tuple(map(int, fields[5])), int(tag)))
            elif event.kind == CONTACT_CONSEQUENCE:
                fields, tag = event.values
                out.ingest_consequence(ConsequenceContactV1(
                    *(int(value) for value in fields), int(tag)))
            elif event.kind == CONTACT_WITHDRAWAL:
                fields, tag = event.values
                out.ingest_withdrawal(WithdrawalContactV1(
                    *(int(value) for value in fields), int(tag)))
            elif event.kind == EVENT_TICK: out.tick()
            elif event.kind == EVENT_UNFOLD: out.unfold()
            else: raise FrontierRefuse("frontier:event_kind")
            if len(out.events) != before + 1 or out.events[-1] != event:
                raise FrontierRefuse("frontier:event_replay")
        if out.checkpoint() != bytes(blob):
            raise FrontierRefuse("frontier:checkpoint_noncanonical")
        return out


def _tuplify(value):
    return tuple(_tuplify(item) for item in value) if isinstance(value, list) else int(value)
