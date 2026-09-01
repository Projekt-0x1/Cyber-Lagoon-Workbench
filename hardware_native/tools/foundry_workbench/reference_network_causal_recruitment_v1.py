#!/usr/bin/env python3
"""Reference hypothesis for causal credit on recruited distributed Networks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from reference_population_v1 import PopulationBankV1, PopulationOccurrenceV1, PopulationSpecV1


MAX_NETWORKS = 128
MAX_MEMBERS = 16
MAX_ACTIVE = 128
MAX_REVISIONS = 512
MAX_HORIZON = 64
DEFAULT_CAPACITY = 32_768


class Refuse(ValueError):
    pass


def _identity(tag, value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(tag.encode() + b"\0" + body).digest()[:8], "big") or 1


@dataclass(frozen=True)
class NetworkRecruitmentRevisionV1:
    identity: int
    member_signatures: tuple[tuple[int, ...], ...]
    couplings: tuple[tuple[int, int], ...]
    quorum: int


@dataclass(frozen=True)
class ActiveNetworkV1:
    identity: int
    recruitment_revision: int
    generation: int
    member_occurrences: tuple[int, ...]
    live_members: tuple[int, ...]
    sites: tuple[int, ...]
    born_tick: int


@dataclass
class PendingNetworkActionV1:
    ticket: int
    active_network: int
    recruitment_revision: int
    source: int
    consequence_channel: int
    incarnation: int
    challenge: int
    opened_tick: int
    deadline: int
    settled: int = 0


@dataclass(frozen=True)
class NetworkRevisionEventV1:
    identity: int
    recruitment_revision: int
    active_network: int
    source: int
    direction: int


class NetworkCausalRecruitmentV1:
    """Persistent recruitment revisions forming transient active Networks."""

    def __init__(self, resident_capacity=DEFAULT_CAPACITY):
        self.population = PopulationBankV1(PopulationSpecV1(site_count=resident_capacity))
        self.tick = 0
        self.next_generation = 1
        self.recruitment_revisions: dict[int, NetworkRecruitmentRevisionV1] = {}
        self.active_networks: dict[int, ActiveNetworkV1] = {}
        self.pending: dict[int, PendingNetworkActionV1] = {}
        self.revisions: list[NetworkRevisionEventV1] = []
        self.withdrawn_sources: set[int] = set()
        self.last_touched_members = 0

    def learn_recruitment_revision(self, members, couplings, quorum=None):
        members = tuple(members)
        if not 2 <= len(members) <= MAX_MEMBERS:
            raise Refuse("network member bound")
        if any(not isinstance(row, PopulationOccurrenceV1)
               or row not in self.population.occurrences for row in members):
            raise Refuse("network actual member occurrence")
        signatures = tuple(tuple(row.sites) for row in members)
        if len(set(signatures)) != len(signatures):
            raise Refuse("network duplicate member")
        edges = tuple(sorted({tuple(sorted(map(int, edge))) for edge in couplings}))
        if any(len(edge) != 2 or edge[0] == edge[1]
               or not 0 <= edge[0] < len(members) or not 0 <= edge[1] < len(members)
               for edge in edges):
            raise Refuse("network coupling")
        if not self._connected(range(len(members)), edges):
            raise Refuse("network disconnected preparation")
        quorum = int(quorum if quorum is not None else (2 * len(members) + 2) // 3)
        if not 2 <= quorum <= len(members):
            raise Refuse("network quorum")
        identity = _identity("agi-resident-network-v1", [signatures, edges, quorum])
        revision = NetworkRecruitmentRevisionV1(identity, signatures, edges, quorum)
        prior = self.recruitment_revisions.get(identity)
        if prior is not None and prior != revision:
            raise Refuse("network identity collision")
        if prior is None and len(self.recruitment_revisions) >= MAX_NETWORKS:
            raise Refuse("network bound")
        self.recruitment_revisions[identity] = revision
        return revision

    @staticmethod
    def _connected(live_members, couplings):
        live = set(live_members)
        if not live:
            return False
        seen = {min(live)}
        changed = True
        while changed:
            changed = False
            for left, right in couplings:
                if left not in live or right not in live:
                    continue
                if (left in seen) != (right in seen):
                    seen.update((left, right))
                    changed = True
        return seen == live

    def activate(self, revision, members, unavailable_sites=()):
        if (not isinstance(revision, NetworkRecruitmentRevisionV1)
                or self.recruitment_revisions.get(revision.identity) != revision):
            raise Refuse("network recruitment revision")
        members = tuple(members)
        if (len(members) != len(revision.member_signatures)
                or any(not isinstance(row, PopulationOccurrenceV1)
                       or row not in self.population.occurrences
                       for row in members)
                or tuple(row.sites for row in members) != revision.member_signatures):
            raise Refuse("network current member occurrences")
        unavailable = set(map(int, unavailable_sites))
        live = []
        sites = set()
        for index, signature in enumerate(revision.member_signatures):
            present = set(signature) - unavailable
            if len(present) * 2 >= len(signature):
                live.append(index)
                sites.update(present)
        self.last_touched_members = len(revision.member_signatures)
        if len(live) < revision.quorum or not self._connected(live, revision.couplings):
            return None
        generation = self.next_generation
        self.next_generation += 1
        identity = _identity("agi-network-occurrence-v1", [
            revision.identity, generation, [row.identity for row in members],
            live, sorted(sites), self.tick])
        occurrence = ActiveNetworkV1(
            identity, revision.identity, generation,
            tuple(row.identity for row in members), tuple(live),
            tuple(sorted(sites)), self.tick)
        if len(self.active_networks) >= MAX_ACTIVE:
            raise Refuse("network occurrence bound")
        self.active_networks[identity] = occurrence
        self.tick += 1
        return occurrence

    def participate(self, occurrence, source, consequence_channel, horizon=16):
        source = int(source)
        consequence_channel = int(consequence_channel)
        if (not isinstance(occurrence, ActiveNetworkV1)
                or self.active_networks.get(occurrence.identity) != occurrence):
            raise Refuse("network actual occurrence")
        if source <= 0 or consequence_channel <= 0 or source in self.withdrawn_sources:
            raise Refuse("network source")
        if not 1 <= int(horizon) <= MAX_HORIZON:
            raise Refuse("network horizon")
        if any(not row.settled and row.active_network == occurrence.identity
               for row in self.pending.values()):
            raise Refuse("network occurrence replay")
        if len(self.pending) >= MAX_ACTIVE:
            raise Refuse("network pending bound")
        ticket = _identity("agi-network-action-v1", [
            occurrence.identity, occurrence.generation, source,
            consequence_channel, self.tick])
        challenge = _identity("agi-network-consequence-challenge-v2", [
            ticket, occurrence.identity, occurrence.generation, source,
            consequence_channel, self.tick + int(horizon)])
        self.pending[ticket] = PendingNetworkActionV1(
            ticket, occurrence.identity, occurrence.recruitment_revision,
            source, consequence_channel, occurrence.generation, challenge,
            self.tick, self.tick + int(horizon))
        self.tick += 1
        return ticket

    def consequence_envelope(self, ticket):
        row = self.pending.get(int(ticket))
        if row is None or row.settled:
            raise Refuse("network pending action")
        return (row.ticket, row.incarnation, row.deadline, row.source,
                row.consequence_channel, row.challenge)

    def settle(self, envelope, occurrence, source, consequence_channel,
               actual_effect, matched_effect):
        if not isinstance(envelope, tuple) or len(envelope) != 6:
            raise Refuse("network consequence envelope")
        ticket, incarnation, deadline, expected_source, expected_channel, challenge = envelope
        row = self.pending.get(int(ticket))
        before = len(self.revisions)
        if row is None or row.settled:
            raise Refuse("network pending action")
        if (not isinstance(occurrence, ActiveNetworkV1)
                or occurrence.identity != row.active_network
                or occurrence.recruitment_revision != row.recruitment_revision):
            raise Refuse("network consequence binding")
        if (envelope != self.consequence_envelope(ticket)
                or incarnation != occurrence.generation or deadline != row.deadline
                or expected_source != row.source or expected_channel != row.consequence_channel
                or challenge != row.challenge or type(actual_effect) is not int
                or type(matched_effect) is not int or int(source) != row.source
                or int(consequence_channel) != row.consequence_channel
                or row.source in self.withdrawn_sources or self.tick > row.deadline):
            raise Refuse("network consequence route")
        difference = actual_effect - matched_effect
        if difference != 0:
            if len(self.revisions) >= MAX_REVISIONS:
                raise Refuse("network revision bound")
            direction = 1 if difference > 0 else -1
            identity = _identity("agi-network-revision-v1", [
                row.recruitment_revision, row.active_network, row.source, direction])
            self.revisions.append(NetworkRevisionEventV1(
                identity, row.recruitment_revision, row.active_network,
                row.source, direction))
        row.settled = 1
        self.tick += 1
        return len(self.revisions) - before

    def withdraw_source(self, source):
        source = int(source)
        if source <= 0:
            raise Refuse("network source")
        self.withdrawn_sources.add(source)

    def score(self, revision):
        return sum(row.direction for row in self.revisions
                   if row.recruitment_revision == revision.identity
                   and row.source not in self.withdrawn_sources)

    def select(self, candidates):
        rows = [(self.score(self.recruitment_revisions[network.recruitment_revision]),
                 network.identity)
                for network in candidates
                if self.active_networks.get(network.identity) == network]
        if not rows:
            return 0
        peak = max(score for score, _ in rows)
        winners = [identity for score, identity in rows if score == peak]
        return winners[0] if peak > 0 and len(winners) == 1 else 0

    def checkpoint(self):
        body = {
            "schema": 2,
            "population": self.population.checkpoint(),
            "tick": self.tick,
            "next_generation": self.next_generation,
            "recruitment_revisions": [asdict(row) for _, row in sorted(self.recruitment_revisions.items())],
            "active_networks": [asdict(row) for _, row in sorted(self.active_networks.items())],
            "pending": [asdict(row) for _, row in sorted(self.pending.items())],
            "revisions": [asdict(row) for row in self.revisions],
            "withdrawn_sources": sorted(self.withdrawn_sources),
            "last_touched_members": self.last_touched_members,
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":"))

    @classmethod
    def restore(cls, blob):
        body = json.loads(blob)
        expected = {"schema", "population", "tick", "next_generation", "recruitment_revisions",
                    "active_networks", "pending", "revisions", "withdrawn_sources",
                    "last_touched_members"}
        if set(body) != expected or body["schema"] != 2:
            raise Refuse("network checkpoint schema")
        out = cls(body["population"]["spec"]["site_count"])
        out.population = PopulationBankV1.restore(body["population"])
        out.tick = int(body["tick"])
        out.next_generation = int(body["next_generation"])
        for packed in body["recruitment_revisions"]:
            packed["member_signatures"] = tuple(tuple(row) for row in packed["member_signatures"])
            packed["couplings"] = tuple(tuple(row) for row in packed["couplings"])
            row = NetworkRecruitmentRevisionV1(**packed)
            out.recruitment_revisions[row.identity] = row
        for packed in body["active_networks"]:
            packed["member_occurrences"] = tuple(packed["member_occurrences"])
            packed["live_members"] = tuple(packed["live_members"])
            packed["sites"] = tuple(packed["sites"])
            row = ActiveNetworkV1(**packed)
            out.active_networks[row.identity] = row
        out.pending = {int(row["ticket"]): PendingNetworkActionV1(**row)
                       for row in body["pending"]}
        if len(out.pending) != len(body["pending"]):
            raise Refuse("network checkpoint duplicate pending")
        out.revisions = [NetworkRevisionEventV1(**row) for row in body["revisions"]]
        out.withdrawn_sources = set(map(int, body["withdrawn_sources"]))
        out.last_touched_members = int(body["last_touched_members"])
        if (len(out.recruitment_revisions) != len(body["recruitment_revisions"])
                or len(out.active_networks) != len(body["active_networks"])
                or len(out.recruitment_revisions) > MAX_NETWORKS
                or len(out.active_networks) > MAX_ACTIVE
                or len(out.pending) > MAX_ACTIVE or len(out.revisions) > MAX_REVISIONS):
            raise Refuse("network checkpoint bound")
        for revision in out.recruitment_revisions.values():
            rebuilt = NetworkRecruitmentRevisionV1(
                _identity("agi-resident-network-v1", [revision.member_signatures,
                          revision.couplings, revision.quorum]),
                revision.member_signatures, revision.couplings, revision.quorum)
            if revision != rebuilt:
                raise Refuse("network checkpoint identity")
        for occurrence in out.active_networks.values():
            revision = out.recruitment_revisions.get(occurrence.recruitment_revision)
            members = [next((row for row in out.population.occurrences
                             if row.identity == identity), None)
                       for identity in occurrence.member_occurrences]
            union = set().union(*map(set, revision.member_signatures)) if revision else set()
            derived_live = tuple(index for index, signature in enumerate(revision.member_signatures)
                                 if len(set(signature) & set(occurrence.sites)) * 2 >= len(signature)) if revision else ()
            expected_identity = _identity("agi-network-occurrence-v1", [
                occurrence.recruitment_revision, occurrence.generation,
                occurrence.member_occurrences, occurrence.live_members,
                occurrence.sites, occurrence.born_tick])
            if (revision is None or any(member is None for member in members)
                    or tuple(member.sites for member in members) != revision.member_signatures
                    or occurrence.identity != expected_identity
                    or occurrence.generation <= 0 or occurrence.born_tick < 0
                    or occurrence.live_members != derived_live
                    or len(derived_live) < revision.quorum
                    or not out._connected(derived_live, revision.couplings)
                    or not set(occurrence.sites) <= union):
                raise Refuse("network checkpoint occurrence")
        if out.active_networks and out.next_generation <= max(
                row.generation for row in out.active_networks.values()):
            raise Refuse("network checkpoint generation")
        for pending in out.pending.values():
            occurrence = out.active_networks.get(pending.active_network)
            expected_ticket = _identity("agi-network-action-v1", [
                pending.active_network, pending.incarnation, pending.source,
                pending.consequence_channel, pending.opened_tick])
            expected_challenge = _identity("agi-network-consequence-challenge-v2", [
                pending.ticket, pending.active_network, pending.incarnation,
                pending.source, pending.consequence_channel, pending.deadline])
            if (occurrence is None or pending.ticket != expected_ticket
                    or pending.recruitment_revision != occurrence.recruitment_revision
                    or pending.incarnation != occurrence.generation
                    or pending.challenge != expected_challenge
                    or pending.source <= 0 or pending.consequence_channel <= 0
                    or pending.opened_tick < occurrence.born_tick
                    or pending.deadline < pending.opened_tick
                    or pending.settled not in (0, 1)):
                raise Refuse("network checkpoint pending")
        for revision in out.revisions:
            pending = next((row for row in out.pending.values()
                            if row.active_network == revision.active_network), None)
            if (pending is None or not pending.settled
                    or pending.recruitment_revision != revision.recruitment_revision
                    or pending.source != revision.source
                    or revision.direction not in (-1, 1)
                    or revision.identity != _identity("agi-network-revision-v1", [
                        revision.recruitment_revision, revision.active_network,
                        revision.source, revision.direction])):
                raise Refuse("network checkpoint revision")
        return out

    def state_hash(self):
        return hashlib.sha256(self.checkpoint().encode()).hexdigest()
