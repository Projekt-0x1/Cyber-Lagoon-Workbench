#!/usr/bin/env python3
"""Fast graph-neutral assay for learned testimonial influence.

This is an observer-side reference experiment, not an Adult implementation.
Numeric source contact enters through a sealed route capability; callers never
pass a source identity into the ecology.  Runtime values are bounded integers and
compact Recipe cells.  There is no trusted flag, semantic label, prompt, language
model, motor identity, or output-byte path.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import hmac
import json
import time


MAX_CELLS = 32
MAX_PENDING = 1
MAX_ASSERTIONS = 8
MAX_CONTEXTS = 32
MAX_CREDIT_ROWS = 64
MAX_WITHDRAWN = 32
MAX_ASSERTION_RECEIPTS = 64
CHECKPOINT_VERSION = 2


@dataclass
class RecipeCellV1:
    source: int
    source_incarnation: int
    context: int
    support: int = 0
    counter: int = 0
    revision: int = 0
    active: int = 1


@dataclass(frozen=True)
class AssertionOccurrenceV1:
    identity: int
    source: int
    source_incarnation: int
    context: int
    candidate: int
    provenance_class: int


@dataclass(frozen=True)
class AssertionEnvelopeV1:
    identity: int
    ticket: int
    incarnation: int
    deadline: int
    source: int
    source_incarnation: int
    physical_port: int
    codec_identity: int
    context: int
    candidate: int
    payload: tuple[int, ...]
    session_epoch: int
    ingress_sequence: int
    provenance_class: int
    auth_tag: int

    def signed_fields(self) -> tuple[int | tuple[int, ...], ...]:
        return (
            self.ticket, self.incarnation, self.deadline, self.source,
            self.source_incarnation, self.physical_port, self.codec_identity,
            self.context, self.candidate, self.payload, self.session_epoch,
            self.ingress_sequence, self.provenance_class,
        )


class SourceRouteV1:
    """Opaque body-route capability; it cannot choose or expose source identity."""

    def __init__(self, authority: "AssertionIngressAuthorityV1", physical_port: int):
        self.__authority = authority
        self.__physical_port = physical_port

    def emit(
        self, ticket: int, incarnation: int, deadline: int, context: int,
        candidate: int, payload: tuple[int, ...], session_epoch: int,
        ingress_sequence: int,
    ) -> AssertionEnvelopeV1:
        return self.__authority._seal(
            self.__physical_port, ticket, incarnation, deadline, context,
            candidate, payload, session_epoch, ingress_sequence,
        )


class AssertionIngressAuthorityV1:
    """Reference route custody; signing material is not resident trust state."""

    def __init__(self, routes: dict[int, tuple[int, int, int]], key: bytes):
        if not routes or len(routes) > 8 or len(key) != 32:
            raise ValueError("assertion_authority_bound")
        self.__routes = dict(routes)
        self.__lineage = {
            (port, source, source_incarnation, codec)
            for port, (source, source_incarnation, codec) in routes.items()
        }
        self.__key = bytes(key)

    def route(self, physical_port: int) -> SourceRouteV1:
        if physical_port not in self.__routes:
            raise ValueError("unknown_source_route")
        return SourceRouteV1(self, physical_port)

    def reincarnate(self, physical_port: int) -> SourceRouteV1:
        source, incarnation, codec = self.__routes[physical_port]
        current = (source, incarnation + 1, codec)
        self.__routes[physical_port] = current
        self.__lineage.add((physical_port, *current))
        return SourceRouteV1(self, physical_port)

    @staticmethod
    def _canonical(fields: tuple[int | tuple[int, ...], ...]) -> bytes:
        return json.dumps(fields, separators=(",", ":")).encode()

    def _seal(
        self, physical_port: int, ticket: int, incarnation: int, deadline: int,
        context: int, candidate: int, payload: tuple[int, ...], session_epoch: int,
        ingress_sequence: int,
    ) -> AssertionEnvelopeV1:
        source, source_incarnation, codec = self.__routes[physical_port]
        if (
            not payload or len(payload) > 16
            or any(type(value) is not int for value in payload)
        ):
            raise ValueError("assertion_payload_bound")
        fields = (
            ticket, incarnation, deadline, source, source_incarnation,
            physical_port, codec, context, candidate, payload, session_epoch,
            ingress_sequence, 7,
        )
        tag = int.from_bytes(
            hmac.new(self.__key, self._canonical(fields), hashlib.sha256).digest()[:8],
            "little",
        )
        identity = int.from_bytes(
            hashlib.sha256(self._canonical(fields) + tag.to_bytes(8, "little")).digest()[:8],
            "little",
        ) or 1
        return AssertionEnvelopeV1(identity, *fields, tag)

    def verify(
        self, envelope: AssertionEnvelopeV1, incarnation: int, tick: int,
        session_epoch: int, ingress_sequence: int,
    ) -> None:
        if type(envelope) is not AssertionEnvelopeV1:
            raise TypeError("assertion_envelope_type")
        if (
            envelope.incarnation != incarnation or envelope.deadline < tick
            or envelope.session_epoch != session_epoch
            or envelope.ingress_sequence != ingress_sequence
            or envelope.provenance_class != 7
            or (envelope.physical_port, envelope.source, envelope.source_incarnation,
                envelope.codec_identity) not in self.__lineage
        ):
            raise ValueError("assertion_envelope_coordinates")
        fields = envelope.signed_fields()
        expected_tag = int.from_bytes(
            hmac.new(self.__key, self._canonical(fields), hashlib.sha256).digest()[:8],
            "little",
        )
        expected_identity = int.from_bytes(
            hashlib.sha256(
                self._canonical(fields) + expected_tag.to_bytes(8, "little")
            ).digest()[:8],
            "little",
        ) or 1
        if envelope.auth_tag != expected_tag or envelope.identity != expected_identity:
            raise ValueError("assertion_envelope_authentication")


@dataclass
class PendingV1:
    ticket: int
    deadline: int
    context: int
    selected: int
    counterfactual: int
    assertions: tuple[tuple[int, int, int, int], ...]
    contributions: tuple[tuple[int, int, int], ...]
    base_scores: tuple[tuple[int, int], ...]
    body_source: int
    consequence_channel: int
    challenge: int


@dataclass(frozen=True)
class ConsequenceV1:
    ticket: int
    incarnation: int
    deadline: int
    source: int
    channel: int
    challenge: int
    returned_value: int
    tag: str


@dataclass(frozen=True)
class ContextContactV1:
    identity: int
    context: int
    occurrences: tuple[int, ...]
    source: int
    tag: str


class SealedToyBodyV1:
    """Precommitted body/world mapping; settlement never accepts an effect flag."""

    def __init__(
        self, mapping: dict[int, int], context_occurrences: dict[int, tuple[int, ...]],
        key: bytes,
    ):
        self._mapping = dict(mapping)
        self._context_occurrences = dict(context_occurrences)
        self._key = bytes(key)
        self.incarnation = 1
        self.source = 9001
        self.channel = 17

    def consequence(
        self, ticket: int, deadline: int, challenge: int, context: int
    ) -> ConsequenceV1:
        returned_value = self._mapping[context]
        payload = (
            f"{ticket}:{self.incarnation}:{deadline}:{self.source}:"
            f"{self.channel}:{challenge}:{returned_value}"
        ).encode()
        tag = hmac.new(self._key, payload, hashlib.sha256).hexdigest()
        return ConsequenceV1(
            ticket, self.incarnation, deadline, self.source, self.channel,
            challenge, returned_value, tag
        )

    def context_contact(self, context: int) -> ContextContactV1:
        occurrences = self._context_occurrences[context]
        payload = json.dumps((context, occurrences, self.source), separators=(",", ":")).encode()
        tag = hmac.new(self._key, payload, hashlib.sha256).hexdigest()
        identity = int.from_bytes(hashlib.sha256(payload + tag.encode()).digest()[:8], "little") or 1
        return ContextContactV1(identity, context, occurrences, self.source, tag)


class RelationalTrustNetworkV1:
    def __init__(self, body_key: bytes, assertion_authority: AssertionIngressAuthorityV1):
        self.cells: dict[tuple[int, int, int], RecipeCellV1] = {}
        self.context_links: set[tuple[int, int]] = set()
        self.context_occurrences: dict[int, tuple[int, ...]] = {}
        self.withdrawn: set[tuple[int, int]] = set()
        self.pending: dict[int, PendingV1] = {}
        self.belief_support: dict[int, int] = {}
        self.credit: dict[int, int] = {}
        self.assertions: dict[int, AssertionOccurrenceV1] = {}
        self.assertion_receipts: dict[int, AssertionEnvelopeV1] = {}
        self.tick = 0
        self.next_ticket = 1
        self.session_epoch = 1
        self.next_ingress_sequence = 1
        self.body_incarnation = 1
        self._body_key = bytes(body_key)
        self._assertion_authority = assertion_authority

    def contact_context(self, contact: ContextContactV1) -> None:
        if type(contact) is not ContextContactV1:
            raise TypeError("context_contact_type")
        payload = json.dumps(
            (contact.context, contact.occurrences, contact.source), separators=(",", ":")
        ).encode()
        expected_tag = hmac.new(self._body_key, payload, hashlib.sha256).hexdigest()
        expected_identity = int.from_bytes(
            hashlib.sha256(payload + expected_tag.encode()).digest()[:8], "little"
        ) or 1
        if (
            contact.source != 9001
            or not hmac.compare_digest(contact.tag, expected_tag)
            or contact.identity != expected_identity
        ):
            raise ValueError("context_contact_authentication")
        context = contact.context
        occurrences = contact.occurrences
        if not occurrences or len(occurrences) > 8 or any(type(x) is not int for x in occurrences):
            raise ValueError("context_occurrence_bound")
        if context not in self.context_occurrences and len(self.context_occurrences) >= MAX_CONTEXTS:
            raise ValueError("context_bound")
        current = tuple(sorted(set(occurrences)))
        self.context_occurrences[context] = current
        self.context_links.clear()
        rows = sorted(self.context_occurrences.items())
        for index, (left, left_occurrences) in enumerate(rows):
            for right, right_occurrences in rows[index + 1:]:
                if set(left_occurrences).intersection(right_occurrences):
                    self.context_links.add((left, right))

    def capture_assertion(self, envelope: AssertionEnvelopeV1) -> AssertionOccurrenceV1:
        if len(self.assertion_receipts) >= MAX_ASSERTION_RECEIPTS:
            raise ValueError("assertion_receipt_bound")
        self._assertion_authority.verify(
            envelope, self.body_incarnation, self.tick, self.session_epoch,
            self.next_ingress_sequence,
        )
        if envelope.identity in self.assertion_receipts:
            raise ValueError("assertion_replay")
        if (envelope.source, envelope.source_incarnation) in self.withdrawn:
            raise ValueError("assertion_source_withdrawn")
        occurrence = AssertionOccurrenceV1(
            envelope.identity, envelope.source, envelope.source_incarnation,
            envelope.context, envelope.candidate, envelope.provenance_class,
        )
        self.assertion_receipts[envelope.identity] = envelope
        self.assertions[occurrence.identity] = occurrence
        self.next_ingress_sequence += 1
        return occurrence

    def _cell(self, source: int, source_incarnation: int, context: int) -> RecipeCellV1:
        key = (source, source_incarnation, context)
        row = self.cells.get(key)
        if row is None:
            if len(self.cells) >= MAX_CELLS:
                raise ValueError("cell_bound")
            row = RecipeCellV1(source, source_incarnation, context)
            self.cells[key] = row
        return row

    def calibration(self, source: int, source_incarnation: int, context: int) -> int:
        if (source, source_incarnation) in self.withdrawn:
            return 0
        exact = self.cells.get((source, source_incarnation, context))
        score = 0 if exact is None or not exact.active else 4 * (exact.support - exact.counter)
        for a, b in self.context_links:
            other = b if a == context else a if b == context else None
            row = None if other is None else self.cells.get((source, source_incarnation, other))
            if row is not None and row.active:
                score += row.support - row.counter
        return score

    def deliberate(
        self,
        context: int,
        alternatives: tuple[int, ...],
        base_support: tuple[int, ...],
        assertions: tuple[AssertionOccurrenceV1, ...],
    ) -> tuple[int | None, int, int]:
        if self.pending:
            raise ValueError("pending_consequence")
        if len(assertions) > MAX_ASSERTIONS:
            raise ValueError("assertion_bound")
        if not 1 < len(alternatives) <= 8 or len(alternatives) != len(base_support):
            raise ValueError("alternative_bound")
        if any(type(x) is not int for x in alternatives + base_support):
            raise TypeError("numeric_ir")

        scores = dict(zip(alternatives, base_support))
        counterfactual = min(((-v, k) for k, v in scores.items()))[1]
        # Repetition by one source is one causal voice, not independent voting.
        unique: dict[tuple[int, int], AssertionOccurrenceV1] = {}
        for assertion in assertions:
            if self.assertions.get(assertion.identity) != assertion:
                raise ValueError("assertion_not_admitted")
            if assertion.context != context or assertion.candidate not in scores:
                continue
            key = (assertion.source, assertion.source_incarnation)
            if key in self.withdrawn:
                continue
            unique.setdefault(key, assertion)
        contributions = []
        for key, assertion in unique.items():
            learned = self.calibration(assertion.source, assertion.source_incarnation, context)
            influence = max(0, learned)
            if influence:
                scores[assertion.candidate] += influence
                self.belief_support[assertion.identity] = influence
                contributions.append((assertion.identity, assertion.candidate, influence))

        ranked = sorted((value, candidate) for candidate, value in scores.items())
        if len(ranked) > 1 and ranked[-1][0] == ranked[-2][0]:
            if contributions:
                return None, counterfactual, 0
            # Neutral exploration is resident chronology, never source influence.
            selected = alternatives[self.next_ticket % len(alternatives)]
        else:
            selected = ranked[-1][1]
        ticket = self.next_ticket
        self.next_ticket += 1
        deadline = self.tick + 2
        challenge = ((ticket * 0x9E3779B1) ^ context ^ selected) & 0x7FFFFFFF
        self.pending[ticket] = PendingV1(
            ticket,
            deadline,
            context,
            selected,
            counterfactual,
            tuple((a.identity, a.source, a.source_incarnation, a.candidate) for a in unique.values()),
            tuple(contributions),
            tuple(sorted(zip(alternatives, base_support))),
            9001,
            17,
            challenge,
        )
        return selected, counterfactual, ticket

    def settle(self, consequence: ConsequenceV1) -> None:
        pending = self.pending.get(consequence.ticket)
        if pending is None:
            raise ValueError("unknown_ticket")
        if consequence.incarnation != self.body_incarnation:
            raise ValueError("wrong_incarnation")
        if consequence.deadline != pending.deadline or self.tick > pending.deadline:
            raise ValueError("wrong_deadline")
        if (
            consequence.source != pending.body_source
            or consequence.channel != pending.consequence_channel
            or consequence.challenge != pending.challenge
        ):
            raise ValueError("wrong_consequence_route")
        payload = (
            f"{consequence.ticket}:{consequence.incarnation}:{consequence.deadline}:"
            f"{consequence.source}:{consequence.channel}:{consequence.challenge}:"
            f"{consequence.returned_value}"
        ).encode()
        expected = hmac.new(self._body_key, payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, consequence.tag):
            raise ValueError("bad_authentication")
        if consequence.returned_value not in dict(pending.base_scores):
            raise ValueError("bad_returned_value")

        result = 1 if pending.selected == consequence.returned_value else -1

        # Raw returned state evaluates the asserted candidate without a reward bit.
        observable = list(pending.assertions)
        missing = {
            (source, incarnation, pending.context)
            for _, source, incarnation, _ in observable
            if (source, incarnation, pending.context) not in self.cells
        }
        if len(self.cells) + len(missing) > MAX_CELLS:
            raise ValueError("cell_bound")
        all_contributions = list(pending.contributions)
        new_credit_rows = sum(
            identity not in self.credit for identity, _, _ in all_contributions
        )
        if len(self.credit) + new_credit_rows > MAX_CREDIT_ROWS:
            raise ValueError("credit_bound")
        rows = [self._cell(source, incarnation, pending.context) for _, source, incarnation, _ in observable]
        for row in rows:
            asserted_candidate = next(a[3] for a in observable if a[1] == row.source and a[2] == row.source_incarnation)
            if asserted_candidate == consequence.returned_value:
                row.support += 1
            else:
                row.counter += 1
            row.revision += 1
        base = dict(pending.base_scores)
        for identity, _, _ in all_contributions:
            scores = dict(base)
            for other_identity, candidate, influence in all_contributions:
                if other_identity != identity:
                    scores[candidate] += influence
            ranked = sorted((value, candidate) for candidate, value in scores.items())
            without = None if ranked[-1][0] == ranked[-2][0] else ranked[-1][1]
            if without != pending.selected:
                self.credit[identity] = self.credit.get(identity, 0) + result
        for identity, _, _, _ in pending.assertions:
            self.belief_support.pop(identity, None)
        del self.pending[consequence.ticket]
        self.tick += 1

    def withdraw(self, source: int, incarnation: int) -> None:
        key = (source, incarnation)
        if key not in self.withdrawn and len(self.withdrawn) >= MAX_WITHDRAWN:
            raise ValueError("withdrawn_bound")
        self.withdrawn.add(key)
        for row in self.cells.values():
            if row.source == source and row.source_incarnation == incarnation:
                row.active = 0
        for ticket, pending in list(self.pending.items()):
            if any((source_id, source_incarnation) == key for _, source_id, source_incarnation, _ in pending.assertions):
                for identity, _, _, _ in pending.assertions:
                    self.belief_support.pop(identity, None)
                del self.pending[ticket]

    def zero_input_continue(self) -> int:
        self.tick += 1
        expired = 0
        for ticket, pending in list(self.pending.items()):
            if self.tick > pending.deadline:
                for identity, _, _, _ in pending.assertions:
                    self.belief_support.pop(identity, None)
                del self.pending[ticket]
                expired += 1
        return expired

    def checkpoint(self) -> bytes:
        state = {
            "version": CHECKPOINT_VERSION,
            "cells": [asdict(self.cells[k]) for k in sorted(self.cells)],
            "links": sorted(self.context_links),
            "context_occurrences": sorted(self.context_occurrences.items()),
            "withdrawn": sorted(self.withdrawn),
            "pending": [asdict(self.pending[k]) for k in sorted(self.pending)],
            "belief": sorted(self.belief_support.items()),
            "credit": sorted(self.credit.items()),
            "assertions": [asdict(self.assertions[k]) for k in sorted(self.assertions)],
            "assertion_receipts": [
                asdict(self.assertion_receipts[k]) for k in sorted(self.assertion_receipts)
            ],
            "tick": self.tick,
            "next_ticket": self.next_ticket,
            "session_epoch": self.session_epoch,
            "next_ingress_sequence": self.next_ingress_sequence,
            "body_incarnation": self.body_incarnation,
        }
        payload = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
        digest = hmac.new(self._body_key, payload, hashlib.sha256).hexdigest().encode()
        return digest + b"\n" + payload

    @classmethod
    def restore(
        cls, blob: bytes, body_key: bytes,
        assertion_authority: AssertionIngressAuthorityV1,
    ) -> "RelationalTrustNetworkV1":
        digest, payload = blob.split(b"\n", 1)
        expected = hmac.new(body_key, payload, hashlib.sha256).hexdigest().encode()
        if not hmac.compare_digest(digest, expected):
            raise ValueError("checkpoint_authentication")
        state = json.loads(payload)
        if state.get("version") != CHECKPOINT_VERSION:
            raise ValueError("checkpoint_version")
        out = cls(body_key, assertion_authority)
        out.cells = {
            (r["source"], r["source_incarnation"], r["context"]): RecipeCellV1(**r)
            for r in state["cells"]
        }
        if len(out.cells) > MAX_CELLS:
            raise ValueError("checkpoint_cell_bound")
        out.context_links = {tuple(x) for x in state["links"]}
        out.context_occurrences = {int(k): tuple(v) for k, v in state["context_occurrences"]}
        out.withdrawn = {tuple(x) for x in state["withdrawn"]}
        out.pending = {
            r["ticket"]: PendingV1(**{
                **r,
                "assertions": tuple(map(tuple, r["assertions"])),
                "contributions": tuple(map(tuple, r["contributions"])),
                "base_scores": tuple(map(tuple, r["base_scores"])),
            })
            for r in state["pending"]
        }
        if len(out.pending) > MAX_PENDING:
            raise ValueError("checkpoint_pending_bound")
        out.belief_support = dict(state["belief"])
        out.credit = dict(state["credit"])
        out.assertions = {
            r["identity"]: AssertionOccurrenceV1(**r) for r in state["assertions"]
        }
        out.assertion_receipts = {}
        for raw in state["assertion_receipts"]:
            envelope = AssertionEnvelopeV1(**{**raw, "payload": tuple(raw["payload"])})
            assertion_authority.verify(
                envelope, state["body_incarnation"], 0, state["session_epoch"],
                envelope.ingress_sequence,
            )
            out.assertion_receipts[envelope.identity] = envelope
        if (
            len(out.assertions) != len(out.assertion_receipts)
            or len(out.assertions) > MAX_ASSERTION_RECEIPTS
            or any(
                out.assertions.get(identity) != AssertionOccurrenceV1(
                    envelope.identity, envelope.source, envelope.source_incarnation,
                    envelope.context, envelope.candidate, envelope.provenance_class,
                )
                for identity, envelope in out.assertion_receipts.items()
            )
        ):
            raise ValueError("checkpoint_assertion_lineage")
        out.tick = state["tick"]
        out.next_ticket = state["next_ticket"]
        out.session_epoch = state["session_epoch"]
        out.next_ingress_sequence = state["next_ingress_sequence"]
        if out.next_ingress_sequence != len(out.assertion_receipts) + 1:
            raise ValueError("checkpoint_ingress_sequence")
        out.body_incarnation = state["body_incarnation"]
        return out


def admit(
    net: RelationalTrustNetworkV1, route: SourceRouteV1, ticket: int,
    context: int, candidate: int, payload: tuple[int, ...] | None = None,
) -> AssertionOccurrenceV1:
    envelope = route.emit(
        ticket, net.body_incarnation, net.tick + 8, context, candidate,
        payload or (context, candidate, ticket & 0xFFFF), net.session_epoch,
        net.next_ingress_sequence,
    )
    return net.capture_assertion(envelope)


def transact(net: RelationalTrustNetworkV1, body: SealedToyBodyV1, item: AssertionOccurrenceV1) -> int:
    selected, _, ticket = net.deliberate(item.context, (41, 73), (0, 0), (item,))
    if selected is None:
        raise AssertionError("unexpected ambiguity")
    pending = net.pending[ticket]
    net.settle(body.consequence(
        ticket, pending.deadline, pending.challenge, item.context
    ))
    return selected


def main() -> None:
    started = time.perf_counter()
    key = b"reference-only-body-key-v1"
    authority = AssertionIngressAuthorityV1(
        {1: (11, 1, 1011), 2: (22, 1, 1022), 3: (33, 1, 1033), 4: (44, 1, 1044)},
        hashlib.sha256(b"reference-assertion-route-custody-v1").digest(),
    )
    route11, route22 = authority.route(1), authority.route(2)
    route33, route44 = authority.route(3), authority.route(4)
    body = SealedToyBodyV1(
        {101: 41, 102: 41, 909: 73},
        {101: (2, 3), 102: (2, 5), 909: (8, 9)}, key,
    )
    net = RelationalTrustNetworkV1(key, authority)
    net.contact_context(body.context_contact(101))
    net.contact_context(body.context_contact(102))
    net.contact_context(body.context_contact(909))

    # Lived accuracy condenses in-place into one source/context Recipe cell.
    for i in range(3):
        transact(net, body, admit(net, route11, 1000 + i, 101, 41))
    exact_before = net.calibration(11, 1, 101)
    near_before = net.calibration(11, 1, 102)
    remote_before = net.calibration(11, 1, 909)

    # Testimony immediately creates belief support and changes the tied choice.
    live = admit(net, route11, 2000, 101, 41)
    selected, counterfactual, ticket = net.deliberate(101, (41, 73), (0, 1), (live,))
    immediate_belief = net.belief_support[live.identity] > 0
    pending = net.pending[ticket]
    net.settle(body.consequence(
        ticket, pending.deadline, pending.challenge, 101
    ))
    earned_credit = net.credit[live.identity] > 0 and selected != counterfactual

    # One source repeated cannot outvote an equally calibrated conflicting source.
    for i in range(4):
        transact(net, body, admit(net, route22, 3000 + i, 101, 41))
    conflict_net = RelationalTrustNetworkV1(key, authority)
    conflict_net.cells = {k: RecipeCellV1(**asdict(v)) for k, v in net.cells.items()}
    repeated = tuple(admit(conflict_net, route11, 4000 + i, 101, 41) for i in range(5))
    conflict = repeated + (admit(conflict_net, route22, 5000, 101, 73),)
    conflict_selected, _, _ = conflict_net.deliberate(101, (41, 73), (0, 0), conflict)

    # Betrayal-like failure revises only local evidence; no hardcoded lie state.
    betray = admit(net, route11, 6000, 101, 73)
    transact(net, body, betray)
    exact_after_failure = net.calibration(11, 1, 101)
    remote_after_failure = net.calibration(11, 1, 909)
    for i in range(3):
        transact(net, body, admit(net, route11, 6100 + i, 101, 41))
    exact_after_recovery = net.calibration(11, 1, 101)

    checkpoint = net.checkpoint()
    replay = RelationalTrustNetworkV1.restore(checkpoint, key, authority)
    corrupt_refused = False
    try:
        damaged = checkpoint[:-1] + bytes([checkpoint[-1] ^ 1])
        RelationalTrustNetworkV1.restore(damaged, key, authority)
    except ValueError:
        corrupt_refused = True

    refusal_net = RelationalTrustNetworkV1(key, authority)
    transact(refusal_net, body, admit(refusal_net, route33, 6999, 101, 41))
    probe = admit(refusal_net, route33, 7000, 101, 41)
    probe_selected, _, probe_ticket = refusal_net.deliberate(101, (41, 73), (0, 0), (probe,))
    probe_pending = refusal_net.pending[probe_ticket]
    valid_probe = body.consequence(
        probe_ticket, probe_pending.deadline, probe_pending.challenge, 101
    )
    before_refusals = refusal_net.checkpoint()
    refused = []
    for invalid in (
        ConsequenceV1(valid_probe.ticket, 2, valid_probe.deadline, valid_probe.source, valid_probe.channel, valid_probe.challenge, valid_probe.returned_value, valid_probe.tag),
        ConsequenceV1(valid_probe.ticket, 1, valid_probe.deadline + 1, valid_probe.source, valid_probe.channel, valid_probe.challenge, valid_probe.returned_value, valid_probe.tag),
        ConsequenceV1(valid_probe.ticket, 1, valid_probe.deadline, valid_probe.source + 1, valid_probe.channel, valid_probe.challenge, valid_probe.returned_value, valid_probe.tag),
        ConsequenceV1(valid_probe.ticket, 1, valid_probe.deadline, valid_probe.source, valid_probe.channel + 1, valid_probe.challenge, valid_probe.returned_value, valid_probe.tag),
        ConsequenceV1(valid_probe.ticket, 1, valid_probe.deadline, valid_probe.source, valid_probe.channel, valid_probe.challenge + 1, valid_probe.returned_value, valid_probe.tag),
        ConsequenceV1(valid_probe.ticket, 1, valid_probe.deadline, valid_probe.source, valid_probe.channel, valid_probe.challenge, valid_probe.returned_value, "0" * 64),
    ):
        try:
            refusal_net.settle(invalid)
            refused.append(False)
        except ValueError:
            refused.append(refusal_net.checkpoint() == before_refusals)
    prediction_only_has_belief = probe.identity in refusal_net.belief_support
    prediction_only_has_no_credit = probe.identity not in refusal_net.credit
    expiry_count = sum(refusal_net.zero_input_continue() for _ in range(3))
    expiry_clears_belief = (
        expiry_count == 1
        and probe.identity not in refusal_net.belief_support
        and probe_ticket not in refusal_net.pending
    )

    unknown_net = RelationalTrustNetworkV1(key, authority)
    unknown = admit(unknown_net, route44, 7100, 101, 41)
    unknown_selected, _, unknown_ticket = unknown_net.deliberate(
        101, (41, 73), (0, 1), (unknown,)
    )
    unknown_has_zero_influence = (
        unknown_selected == 73 and unknown.identity not in unknown_net.belief_support
    )
    unknown_net.zero_input_continue()
    unknown_net.zero_input_continue()
    unknown_net.zero_input_continue()

    withdrawal_net = RelationalTrustNetworkV1.restore(checkpoint, key, authority)
    pending_withdrawal = admit(withdrawal_net, route11, 7200, 101, 41)
    _, _, withdrawal_ticket = withdrawal_net.deliberate(
        101, (41, 73), (0, 1), (pending_withdrawal,)
    )
    withdrawal_net.withdraw(11, 1)
    pending_withdrawal_cascades = (
        withdrawal_ticket not in withdrawal_net.pending
        and pending_withdrawal.identity not in withdrawal_net.belief_support
        and withdrawal_net.calibration(11, 1, 101) == 0
        and withdrawal_net.calibration(11, 2, 101) == 0
    )

    context_revision_net = RelationalTrustNetworkV1.restore(checkpoint, key, authority)
    revision_body = SealedToyBodyV1(
        {101: 41, 102: 41, 909: 73},
        {101: (2, 3), 102: (6, 7), 909: (8, 9)}, key,
    )
    context_revision_net.contact_context(revision_body.context_contact(102))
    context_revision_removes_transfer = (
        context_revision_net.calibration(11, 1, 102) == 0
    )

    ingress_net = RelationalTrustNetworkV1(key, authority)
    envelope = route11.emit(
        8000, ingress_net.body_incarnation, ingress_net.tick + 8, 101, 41,
        (4, 5, 6), ingress_net.session_epoch, ingress_net.next_ingress_sequence,
    )
    before_ingress = ingress_net.checkpoint()
    ingress_refusals = []
    for forged in (
        replace(envelope, source=22),
        replace(envelope, source_incarnation=2),
        replace(envelope, physical_port=2),
        replace(envelope, codec_identity=1022),
        replace(envelope, context=102),
        replace(envelope, candidate=73),
        replace(envelope, payload=(4, 5, 7)),
        replace(envelope, session_epoch=2),
        replace(envelope, ingress_sequence=2),
        replace(envelope, incarnation=2),
        replace(envelope, deadline=-1),
        replace(envelope, auth_tag=envelope.auth_tag ^ 1),
    ):
        try:
            ingress_net.capture_assertion(forged)
            ingress_refusals.append(False)
        except (TypeError, ValueError):
            ingress_refusals.append(ingress_net.checkpoint() == before_ingress)
    admitted = ingress_net.capture_assertion(envelope)
    after_admission = ingress_net.checkpoint()
    try:
        ingress_net.capture_assertion(envelope)
        replay_refused = False
    except ValueError:
        replay_refused = ingress_net.checkpoint() == after_admission
    try:
        ingress_net.deliberate(
            101, (41, 73), (0, 0),
            (AssertionOccurrenceV1(999999, 11, 1, 101, 41, 7),),
        )
        direct_occurrence_refused = False
    except ValueError:
        direct_occurrence_refused = ingress_net.checkpoint() == after_admission

    routes_net = RelationalTrustNetworkV1(key, authority)
    shared_payload = (77, 88, 99)
    left_route_occurrence = admit(routes_net, route11, 8100, 101, 41, shared_payload)
    right_route_occurrence = admit(routes_net, route22, 8101, 101, 41, shared_payload)
    distinct_route_identity = (
        left_route_occurrence.source != right_route_occurrence.source
        and left_route_occurrence.identity != right_route_occurrence.identity
    )
    ingress_checkpoint = routes_net.checkpoint()
    ingress_replay = RelationalTrustNetworkV1.restore(ingress_checkpoint, key, authority)
    ingress_checkpoint_replay = ingress_replay.checkpoint() == ingress_checkpoint

    reincarnated_route = authority.reincarnate(1)
    reincarnation_net = RelationalTrustNetworkV1(key, authority)
    reincarnated = admit(reincarnation_net, reincarnated_route, 8200, 101, 41)
    clean_reincarnation = (
        reincarnated.source == 11 and reincarnated.source_incarnation == 2
        and reincarnation_net.calibration(11, 2, 101) == 0
    )

    context_auth_net = RelationalTrustNetworkV1(key, authority)
    context_probe = body.context_contact(101)
    before_context = context_auth_net.checkpoint()
    try:
        context_auth_net.contact_context(replace(context_probe, occurrences=(2, 4)))
        context_tamper_refused = False
    except ValueError:
        context_tamper_refused = context_auth_net.checkpoint() == before_context

    hidden_left = RelationalTrustNetworkV1(key, authority)
    hidden_right = RelationalTrustNetworkV1(key, authority)
    hidden_world_left = SealedToyBodyV1({101: 41}, {101: (2, 3)}, key)
    hidden_world_right = SealedToyBodyV1({101: 73}, {101: (2, 3)}, key)
    hidden_left.contact_context(hidden_world_left.context_contact(101))
    hidden_right.contact_context(hidden_world_right.context_contact(101))
    hidden_assertion_left = admit(hidden_left, route44, 8300, 101, 41, (12, 13))
    hidden_assertion_right = admit(hidden_right, route44, 8300, 101, 41, (12, 13))
    hidden_left_choice, _, _ = hidden_left.deliberate(
        101, (41, 73), (0, 0), (hidden_assertion_left,)
    )
    hidden_right_choice, _, _ = hidden_right.deliberate(
        101, (41, 73), (0, 0), (hidden_assertion_right,)
    )
    altered_hidden_world_same_preaction = hidden_left_choice == hidden_right_choice

    net.withdraw(11, 1)
    withdrawn_influence = net.calibration(11, 1, 101)

    controls = {
        "no_trusted_flag_or_global_score": not hasattr(RecipeCellV1, "trusted"),
        "unknown_source_has_zero_learned_influence": unknown_has_zero_influence,
        "testimony_creates_immediate_belief": immediate_belief,
        "testimony_can_earn_causal_credit": earned_credit,
        "learned_context_relation_bounds_transfer": 0 < near_before < exact_before,
        "unrelated_context_does_not_inherit": remote_before == 0,
        "betrayal_revises_local_authority": exact_after_failure < exact_before + 4,
        "betrayal_does_not_globally_blacklist": remote_after_failure == remote_before,
        "experience_allows_recovery": exact_after_recovery > exact_after_failure,
        "repeated_source_is_not_multiple_votes": conflict_selected is None,
        "withdrawal_cascades_live_influence": withdrawn_influence == 0,
        "pending_withdrawal_and_new_incarnation_are_clean": pending_withdrawal_cascades,
        "context_contact_revision_removes_transfer": context_revision_removes_transfer,
        "checkpoint_exact_replay": replay.checkpoint() == checkpoint,
        "checkpoint_corruption_refused": corrupt_refused,
        "wrong_incarnation_deadline_auth_refuse_atomically": all(refused),
        "compact_recipe_cells": len(net.cells) <= 3,
        "no_deception_label_installed": not hasattr(RecipeCellV1, "lie"),
        "prediction_only_belief_without_settled_credit": (
            prediction_only_has_belief and prediction_only_has_no_credit
        ),
        "zero_input_expiry_clears_unsettled_belief": expiry_clears_belief,
        "assertion_route_tamper_refuses_atomically": all(ingress_refusals),
        "assertion_replay_refuses_atomically": replay_refused,
        "direct_occurrence_forgery_refused": direct_occurrence_refused,
        "source_identity_derived_from_route": (
            admitted.source == 11 and admitted.source_incarnation == 1
        ),
        "identical_payload_distinct_routes_remain_distinct": distinct_route_identity,
        "assertion_ingress_checkpoint_replay": ingress_checkpoint_replay,
        "new_source_incarnation_starts_uncalibrated": clean_reincarnation,
        "context_contact_tamper_refused": context_tamper_refused,
        "body_returns_raw_value_not_reward": (
            "result" not in ConsequenceV1.__dataclass_fields__
        ),
        "hidden_world_mapping_cannot_change_preaction": altered_hidden_world_same_preaction,
    }
    result = {
        "schema": "0x1.reference-relational-trust-network.v1",
        "contract_status": "CONTRACT" if all(controls.values()) else "RED",
        "reference_only": True,
        "adult_attached": False,
        "runtime_llm": False,
        "physical_direct_parity": "NOT_RUN/RED",
        "production_ir": "ResidentRecipeIrProgram.vcurrent",
        "translation_status": "UNDEFINED",
        "assertion_ingress": "SYNTHETIC_ROUTE_AUTHENTICATED/PHYSICAL_DIRECT_RED",
        "assertion_binding": "OPAQUE_CANDIDATE_BINDING_ASSUMED/RED",
        "toy_world_mapping": "AUTHORED_OBSERVER_FIXTURE/NOT_RESIDENT_INPUT",
        "context_contact": "SIGNED_FIXTURE/CHRONOLOGY_PARITY_RED",
        "contract_scope": "ROUTE_AUTHENTICATED_TESTIMONIAL_UPDATE",
        "graph_flip": False,
        "human_language_claim": False,
        "controls": controls,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    status = result["contract_status"]
    print(f"FOUNDRY_RELATIONAL_TRUST_NETWORK {status} reference_only=1 adult=0 graph_flip=0")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if status == "CONTRACT" else 1)


if __name__ == "__main__":
    main()
