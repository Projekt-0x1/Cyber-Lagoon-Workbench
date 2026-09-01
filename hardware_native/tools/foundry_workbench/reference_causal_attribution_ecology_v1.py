#!/usr/bin/env python3
"""Sparse matched-intervention reference for resident causal attribution.

Participation opens a temporary eligibility receipt.  Only independent lived
consequences from two complete, agreeing source blocks can settle credit and
change later selection.  This is a bounded reference hypothesis, not Direct.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import itertools
import json

from reference_population_v1 import PopulationBankV1, PopulationOccurrenceV1, PopulationSpecV1


MAX_PARTICIPANTS = 2
MAX_PENDING = 64
MAX_EVIDENCE = 256
MAX_NOMINATIONS = 256
MAX_HORIZON = 256
MAX_EFFECT = 1_000_000
DEFAULT_RESIDENT_CAPACITY = 32_768


class Refuse(ValueError):
    pass


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise Refuse(name)
    return value


def _identity(parts):
    raw = json.dumps(parts, separators=(",", ":"), sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(b"causal-attribution-v1\0" + raw).digest()[:8], "big") or 1


def _coalitions(participants):
    return tuple(
        tuple(group)
        for width in range(len(participants) + 1)
        for group in itertools.combinations(participants, width)
    )


@dataclass
class EvidenceV1:
    occurrence: int
    source: int
    coalition: tuple[int, ...]
    effect: int
    tick: int
    active: int = 1


@dataclass(frozen=True)
class AttributionV1:
    receipt: int
    participant_credit: tuple[tuple[int, int, int], ...]
    tags: tuple[str, ...]
    coalition_effects: tuple[tuple[tuple[int, ...], int], ...]
    evidence_occurrences: tuple[int, ...]
    touched_work: int


@dataclass(frozen=True)
class ConditionalAttributionV1:
    receipt: int
    established: int
    newcomer: int
    prior_effect: int
    baseline: int
    established_effect: int
    joint_effect: int
    residual: int
    evidence_occurrences: tuple[int, ...]
    touched_work: int


@dataclass
class InterventionNominationV1:
    identity: int
    receipt: int
    occurrence: int
    coalition: tuple[int, ...]
    opened_tick: int
    deadline: int
    settled: int = 0


@dataclass
class PendingV1:
    receipt: int
    occurrence: int
    participants: tuple[int, ...]
    opened_tick: int
    deadline: int
    evidence: list[EvidenceV1]
    result: AttributionV1 | None = None


@dataclass(frozen=True)
class SelectionV1:
    candidate: int
    participants: tuple[int, ...]
    credit_numerator: int
    credit_denominator: int
    alternatives: int


class CausalAttributionEcologyV1:
    """Two-participant exact coalition assay with sparse resident state."""

    def __init__(self, resident_capacity=DEFAULT_RESIDENT_CAPACITY):
        self.resident_capacity = _positive(resident_capacity, "resident capacity")
        self.population = PopulationBankV1(PopulationSpecV1(site_count=self.resident_capacity))
        self.tick = 0
        self.pending: dict[int, PendingV1] = {}
        self.used_occurrences: set[int] = set()
        self.credit: dict[int, Fraction] = {}
        self.conditionals: dict[int, ConditionalAttributionV1] = {}
        self.nominations: dict[int, InterventionNominationV1] = {}
        self.revision_events = 0
        self.last_touched_work = 0

    def _prune_expired(self):
        expired = [receipt for receipt, row in self.pending.items()
                   if row.result is None and receipt not in self.conditionals
                   and self.tick > row.deadline]
        expired_nominations = [identity for identity, row in self.nominations.items()
                               if (not row.settled and self.tick > row.deadline)
                               or row.receipt in expired]
        for identity in expired_nominations:
            del self.nominations[identity]
        for receipt in expired:
            row=self.pending.pop(receipt)
            retired={int(row.occurrence),*(int(item.occurrence) for item in row.evidence)}
            self.used_occurrences.difference_update(retired)
            self.population.occurrences=[item for item in self.population.occurrences
                                         if int(item.identity) not in retired]

    def participate(self, participants, horizon=32):
        participants = tuple(sorted(set(participants)))
        if (len(participants) != MAX_PARTICIPANTS
                or any(type(value) is not int or not 0 < value < self.resident_capacity
                       for value in participants)):
            raise Refuse("exact bounded participants")
        if type(horizon) is not int or not 1 <= horizon <= MAX_HORIZON:
            raise Refuse("pending bound")
        self._prune_expired()
        if len(self.pending) >= MAX_PENDING:
            raise Refuse("pending bound")
        occurrence = self.population.recruit(participants)
        receipt = _identity([occurrence.identity, participants, self.tick])
        if receipt in self.pending:
            raise Refuse("receipt collision")
        self.used_occurrences.add(occurrence.identity)
        self.pending[receipt] = PendingV1(
            receipt, occurrence.identity, participants, self.tick, self.tick + horizon, [])
        self.tick += 1
        return receipt

    def advance(self, steps=1):
        if type(steps) is not int or steps < 0:
            raise Refuse("advance")
        self.tick += steps

    def _open_nomination(self,row,coalition,horizon):
        if any(not item.settled and item.receipt == row.receipt
               for item in self.nominations.values()):
            raise Refuse("pending nomination")
        if len(self.nominations) >= MAX_NOMINATIONS:
            raise Refuse("nomination bound")
        coalition=tuple(sorted(set(map(int,coalition))))
        if not set(coalition)<=set(row.participants):raise Refuse("nomination coalition")
        occurrence=self.population.recruit(coalition)
        identity=_identity(["resident-intervention",row.receipt,occurrence.identity,coalition,self.tick])
        if identity in self.nominations:raise Refuse("nomination collision")
        nomination=InterventionNominationV1(identity,row.receipt,occurrence.identity,coalition,self.tick,self.tick+horizon)
        self.nominations[identity]=nomination;self.tick+=1;return nomination,occurrence

    def nominate_intervention_at(self,receipt,coalition,horizon=32):
        """Open one exact resident intervention chosen by a higher causal strategy."""
        self._prune_expired();row=self.pending.get(int(receipt))
        if row is None or row.result is not None or row.receipt in self.conditionals:raise Refuse("unresolved pending receipt")
        if type(horizon) is not int or not 1<=horizon<=MAX_HORIZON:raise Refuse("nomination bound")
        return self._open_nomination(row,coalition,horizon)

    def nominate_intervention(self, receipt, horizon=32, preferred_coalition=None):
        self._prune_expired()
        row = self.pending.get(int(receipt))
        if row is None or row.result is not None or row.receipt in self.conditionals:
            raise Refuse("unresolved pending receipt")
        if type(horizon) is not int or not 1 <= horizon <= MAX_HORIZON:
            raise Refuse("nomination bound")
        coalitions = _coalitions(row.participants)
        counts = {coalition: 0 for coalition in coalitions}
        for evidence in row.evidence:
            if evidence.active:counts[evidence.coalition] += 1
        coalition=min(coalitions,key=lambda value:(counts[value],coalitions.index(value)))
        return self._open_nomination(row,coalition,horizon)

    def consequence(self, receipt, occurrence, source, coalition, effect, independent=True):
        return self._record_consequence(
            receipt, occurrence, source, coalition, effect, independent, 0)

    def _record_consequence(self, receipt, occurrence, source, coalition, effect,
                            independent, nomination_identity):
        row = self.pending.get(int(receipt))
        source = _positive(source, "source")
        if row is None:
            raise Refuse("pending receipt")
        coalition = tuple(sorted(set(coalition)))
        if any(type(value) is not int for value in coalition) or not set(coalition) <= set(row.participants):
            raise Refuse("coalition participation")
        if type(effect) is not int or abs(effect) > MAX_EFFECT:
            raise Refuse("bounded effect")
        if independent is not True:
            raise Refuse("independent consequence")
        if (not isinstance(occurrence, PopulationOccurrenceV1)
                or occurrence not in self.population.occurrences
                or occurrence.sites != self.population.signature(coalition)):
            raise Refuse("actual occurrence participation")
        bound_nomination = next((item for item in self.nominations.values()
                                 if not item.settled
                                 and item.occurrence == occurrence.identity), None)
        if (bound_nomination is not None
                and bound_nomination.identity != nomination_identity):
            raise Refuse("nominated occurrence requires settlement")
        if self.tick > row.deadline:
            raise Refuse("late consequence")
        if occurrence.identity in self.used_occurrences:
            raise Refuse("occurrence replay")
        if row.result is not None or sum(len(item.evidence) for item in self.pending.values()) >= MAX_EVIDENCE:
            raise Refuse("settled or evidence bound")
        if any(item.active and item.source == source and item.coalition == coalition for item in row.evidence):
            raise Refuse("duplicate source coalition")
        self.used_occurrences.add(occurrence.identity)
        row.evidence.append(EvidenceV1(occurrence.identity, source, coalition, effect, self.tick))
        self.tick += 1

    def settle_intervention(self, nomination, occurrence, source, effect,
                            independent=True):
        identity = nomination.identity if isinstance(
            nomination, InterventionNominationV1) else int(nomination)
        row = self.nominations.get(identity)
        if row is None or row.settled:
            raise Refuse("pending nomination")
        if self.tick > row.deadline:
            raise Refuse("late nomination consequence")
        if (not isinstance(occurrence, PopulationOccurrenceV1)
                or occurrence.identity != row.occurrence):
            raise Refuse("nominated occurrence binding")
        self._record_consequence(row.receipt, occurrence, source, row.coalition,
                                 effect, independent, row.identity)
        row.settled = 1

    def _matched_effects(self, row):
        required = set(_coalitions(row.participants))
        by_source = {}
        touched = 0
        for evidence in row.evidence:
            if not evidence.active:
                continue
            touched += 1
            by_source.setdefault(evidence.source, {})[evidence.coalition] = evidence.effect
        complete = [values for values in by_source.values() if set(values) == required]
        if len(complete) < 2:
            return None, touched
        effects = {}
        for coalition in required:
            values = {source_block[coalition] for source_block in complete}
            if len(values) != 1:
                return None, touched
            effects[coalition] = values.pop()
        return effects, touched

    def _matched_conditioned_effects(self, row, established):
        joint = tuple(row.participants)
        required = {(), (established,), joint}
        by_source = {}
        touched = 0
        for evidence in row.evidence:
            if not evidence.active:
                continue
            touched += 1
            by_source.setdefault(evidence.source, {})[evidence.coalition] = evidence.effect
        complete = [values for values in by_source.values() if required <= set(values)]
        if len(complete) < 2:
            return None, touched
        effects = {}
        for coalition in required:
            values = {source_block[coalition] for source_block in complete}
            if len(values) != 1:
                return None, touched
            effects[coalition] = values.pop()
        return effects, touched

    def _learned_singleton_effect(self, participant):
        values = []
        for row in self.pending.values():
            if row.result is None or participant not in row.participants:
                continue
            effects = dict(row.result.coalition_effects)
            values.append(effects[(participant,)] - effects[()])
        return values[0] if values and len(set(values)) == 1 and values[0] > 0 else None

    @staticmethod
    def _conditioned_attribution(row, established, prior_effect, effects, touched):
        newcomer = next(value for value in row.participants if value != established)
        baseline = effects[()]
        established_effect = effects[(established,)] - baseline
        joint_effect = effects[row.participants] - baseline
        evidence_occurrences = tuple(sorted(
            evidence.occurrence for evidence in row.evidence if evidence.active))
        return ConditionalAttributionV1(
            row.receipt, established, newcomer, prior_effect, baseline,
            established_effect, joint_effect, joint_effect - established_effect,
            evidence_occurrences, touched)

    @staticmethod
    def _attribution(row, effects, touched):
        a, b = row.participants
        baseline = effects[()]
        da = effects[(a,)] - baseline
        db = effects[(b,)] - baseline
        interaction = effects[(a, b)] - effects[(a,)] - effects[(b,)] + baseline
        credit_a = Fraction(da, 1) + Fraction(interaction, 2)
        credit_b = Fraction(db, 1) + Fraction(interaction, 2)
        tags = set()
        if credit_a > 0 or credit_b > 0:
            tags.add("support")
        if credit_a < 0 or credit_b < 0:
            tags.add("counter")
        if interaction > 0:
            tags.add("synergy")
        if interaction < 0 and da and db and (da > 0) == (db > 0):
            tags.add("redundancy")
        if (credit_a > 0 > credit_b) or (credit_b > 0 > credit_a):
            tags.add("opposition")
        evidence_occurrences = tuple(sorted(
            evidence.occurrence for evidence in row.evidence if evidence.active))
        return AttributionV1(
            row.receipt,
            ((a, credit_a.numerator, credit_a.denominator),
             (b, credit_b.numerator, credit_b.denominator)),
            tuple(sorted(tags)),
            tuple((coalition, effects[coalition]) for coalition in _coalitions(row.participants)),
            evidence_occurrences,
            touched,
        )

    def resolve(self, receipt):
        row = self.pending.get(int(receipt))
        if row is None:
            raise Refuse("pending receipt")
        if row.receipt in self.conditionals:
            raise Refuse("conditioned attribution already resolved")
        if row.result is not None:
            return row.result
        effects, touched = self._matched_effects(row)
        if effects is None:
            return None
        self.last_touched_work = touched
        result = self._attribution(row, effects, touched)
        row.result = result
        for participant, numerator, denominator in result.participant_credit:
            self.credit[participant] = self.credit.get(participant, Fraction()) + Fraction(numerator, denominator)
            if not self.credit[participant]:
                self.credit.pop(participant)
        coalition_effects = dict(result.coalition_effects)
        a, b = row.participants
        interaction = (coalition_effects[(a, b)] - coalition_effects[(a,)]
                       - coalition_effects[(b,)] + coalition_effects[()])
        self.revision_events += (sum(numerator != 0 for _, numerator, _ in result.participant_credit)
                                 + int(interaction != 0))
        return result

    def resolve_conditioned(self, receipt, established):
        row = self.pending.get(int(receipt))
        established = _positive(established, "established participant")
        if row is None:
            raise Refuse("pending receipt")
        if established not in row.participants:
            raise Refuse("established participation")
        if row.result is not None:
            raise Refuse("full attribution already resolved")
        if row.receipt in self.conditionals:
            return self.conditionals[row.receipt]
        prior_effect = self._learned_singleton_effect(established)
        if prior_effect is None:
            return None
        effects, touched = self._matched_conditioned_effects(row, established)
        if effects is None:
            return None
        current_effect = effects[(established,)] - effects[()]
        if current_effect != prior_effect:
            return None
        result = self._conditioned_attribution(
            row, established, prior_effect, effects, touched)
        self.conditionals[row.receipt] = result
        self.last_touched_work = touched
        self.revision_events += int(result.residual != 0)
        return result

    def withdraw_source(self, source):
        source = _positive(source, "source")
        affected = []
        for row in self.pending.values():
            changed = False
            for evidence in row.evidence:
                if evidence.source == source and evidence.active:
                    evidence.active = 0
                    changed = True
            if changed and row.result is not None:
                affected.append(row)
            if changed:
                self.conditionals.pop(row.receipt, None)
        for row in affected:
            for participant, numerator, denominator in row.result.participant_credit:
                self.credit[participant] = self.credit.get(participant, Fraction()) - Fraction(numerator, denominator)
                if not self.credit[participant]:
                    self.credit.pop(participant)
            row.result = None

    def select(self, candidates, unavailable=()):
        unavailable = set(unavailable)
        scored = []
        for candidate, participants in candidates:
            candidate = _positive(candidate, "candidate")
            participants = tuple(sorted(set(participants)))
            if not participants or any(value in unavailable for value in participants):
                continue
            present = set(participants)
            score = Fraction()
            for row in self.pending.values():
                if row.result is None:
                    continue
                effects = dict(row.result.coalition_effects)
                a, b = row.participants
                baseline = effects[()]
                if a in present:
                    score += effects[(a,)] - baseline
                if b in present:
                    score += effects[(b,)] - baseline
                if a in present and b in present:
                    score += effects[(a, b)] - effects[(a,)] - effects[(b,)] + baseline
            for result in self.conditionals.values():
                if {result.established, result.newcomer} <= present:
                    score += result.residual
            scored.append((score, candidate, participants))
        if not scored:
            return None
        peak = max(score for score, _, _ in scored)
        winners = [item for item in scored if item[0] == peak]
        if peak <= 0 or len(winners) != 1:
            return None
        score, candidate, participants = winners[0]
        return SelectionV1(candidate, participants, score.numerator, score.denominator, len(scored))

    def quantity(self):
        evidence = sum(len(row.evidence) for row in self.pending.values())
        materialized = set(self.credit)
        for row in self.pending.values():
            materialized.update(row.participants)
        return {
            "resident_capacity": self.resident_capacity,
            "materialized_participants": len(materialized),
            "pending_receipts": len(self.pending),
            "intervention_nominations": len(self.nominations),
            "evidence_rows": evidence,
            "last_resolve_touched": self.last_touched_work,
            "revision_events": self.revision_events,
            "checkpoint_bytes": len(self.checkpoint().encode()),
        }

    def checkpoint(self):
        body = {
            "schema": 3,
            "resident_capacity": self.resident_capacity,
            "population": self.population.checkpoint(),
            "tick": self.tick,
            "used_occurrences": sorted(self.used_occurrences),
            "revision_events": self.revision_events,
            "last_touched_work": self.last_touched_work,
            "credit": [[key, value.numerator, value.denominator] for key, value in sorted(self.credit.items())],
            "conditionals": [asdict(value) for _, value in sorted(self.conditionals.items())],
            "nominations": [asdict(value) for _, value in sorted(self.nominations.items())],
            "pending": [],
        }
        for row in sorted(self.pending.values(), key=lambda item: item.receipt):
            packed = {key: value for key, value in asdict(row).items() if key != "result"}
            packed["result"] = None if row.result is None else asdict(row.result)
            body["pending"].append(packed)
        return json.dumps(body, separators=(",", ":"), sort_keys=True)

    @classmethod
    def restore(cls, blob):
        body = json.loads(blob)
        expected = {"schema", "resident_capacity", "population", "tick", "used_occurrences", "revision_events",
                    "last_touched_work", "credit", "conditionals", "nominations", "pending"}
        if set(body) != expected or body["schema"] != 3:
            raise Refuse("checkpoint schema")
        out = cls(body["resident_capacity"])
        out.population = PopulationBankV1.restore(body["population"])
        if out.population.spec.site_count != out.resident_capacity:
            raise Refuse("checkpoint population")
        out.tick = int(body["tick"])
        used_occurrences = tuple(map(int, body["used_occurrences"]))
        if len(used_occurrences) != len(set(used_occurrences)) or any(value <= 0 for value in used_occurrences):
            raise Refuse("checkpoint occurrence identities")
        out.used_occurrences = set(used_occurrences)
        out.revision_events = int(body["revision_events"])
        out.last_touched_work = int(body["last_touched_work"])
        if out.tick < 0 or out.revision_events < 0 or not 0 <= out.last_touched_work <= MAX_EVIDENCE:
            raise Refuse("checkpoint counters")
        if len(body["credit"]) != len({int(row[0]) for row in body["credit"]}):
            raise Refuse("checkpoint credit identities")
        out.credit = {int(key): Fraction(int(numerator), int(denominator))
                      for key, numerator, denominator in body["credit"]}
        conditional_fields = set(ConditionalAttributionV1.__dataclass_fields__)
        for packed in body["conditionals"]:
            if set(packed) != conditional_fields:
                raise Refuse("checkpoint conditional fields")
            packed["evidence_occurrences"] = tuple(packed["evidence_occurrences"])
            result = ConditionalAttributionV1(**packed)
            if result.receipt in out.conditionals:
                raise Refuse("checkpoint duplicate conditional")
            out.conditionals[result.receipt] = result
        nomination_fields = set(InterventionNominationV1.__dataclass_fields__)
        for packed in body["nominations"]:
            if set(packed) != nomination_fields:
                raise Refuse("checkpoint nomination fields")
            packed["coalition"] = tuple(packed["coalition"])
            nomination = InterventionNominationV1(**packed)
            if nomination.identity in out.nominations:
                raise Refuse("checkpoint duplicate nomination")
            out.nominations[nomination.identity] = nomination
        pending_fields = {"receipt", "occurrence", "participants", "opened_tick", "deadline", "evidence", "result"}
        evidence_fields = set(EvidenceV1.__dataclass_fields__)
        result_fields = set(AttributionV1.__dataclass_fields__)
        for packed in body["pending"]:
            if set(packed) != pending_fields:
                raise Refuse("checkpoint pending fields")
            evidence = []
            for item in packed["evidence"]:
                if set(item) != evidence_fields:
                    raise Refuse("checkpoint evidence fields")
                item["coalition"] = tuple(item["coalition"])
                evidence.append(EvidenceV1(**item))
            result = packed["result"]
            if result is not None:
                if set(result) != result_fields:
                    raise Refuse("checkpoint result fields")
                result["participant_credit"] = tuple(tuple(item) for item in result["participant_credit"])
                result["tags"] = tuple(result["tags"])
                result["coalition_effects"] = tuple(
                    (tuple(coalition), effect) for coalition, effect in result["coalition_effects"])
                result["evidence_occurrences"] = tuple(result["evidence_occurrences"])
                result = AttributionV1(**result)
            row = PendingV1(int(packed["receipt"]), int(packed["occurrence"]),
                            tuple(packed["participants"]), int(packed["opened_tick"]),
                            int(packed["deadline"]), evidence, result)
            if row.receipt in out.pending:
                raise Refuse("checkpoint duplicate receipt")
            out.pending[row.receipt] = row
        if len(out.pending) > MAX_PENDING or sum(len(row.evidence) for row in out.pending.values()) > MAX_EVIDENCE:
            raise Refuse("checkpoint bound")
        seen = set()
        rebuilt = {}
        for row in out.pending.values():
            opening = next((item for item in out.population.occurrences
                            if item.identity == row.occurrence), None)
            if (len(row.participants) != MAX_PARTICIPANTS
                    or row.participants != tuple(sorted(set(row.participants)))
                    or any(not 0 < value < out.resident_capacity for value in row.participants)
                    or opening is None or opening.sites != out.population.signature(row.participants)
                    or row.opened_tick < 0 or row.deadline < row.opened_tick
                    or row.receipt != _identity([row.occurrence, row.participants, row.opened_tick])):
                raise Refuse("checkpoint participation")
            seen.add(row.occurrence)
            source_coalitions = set()
            for evidence in row.evidence:
                occurrence = next((item for item in out.population.occurrences
                                   if item.identity == evidence.occurrence), None)
                if (occurrence is None or occurrence.sites != out.population.signature(evidence.coalition)
                        or evidence.tick < row.opened_tick or evidence.tick > row.deadline
                        or evidence.occurrence in seen or evidence.active not in (0, 1)
                        or evidence.source <= 0
                        or not set(evidence.coalition) <= set(row.participants)
                        or abs(evidence.effect) > MAX_EFFECT
                        or (evidence.source, evidence.coalition) in source_coalitions):
                    raise Refuse("checkpoint consequence")
                seen.add(evidence.occurrence)
                source_coalitions.add((evidence.source, evidence.coalition))
            if row.result is not None:
                effects, touched = out._matched_effects(row)
                if effects is None or row.result != out._attribution(row, effects, touched):
                    raise Refuse("checkpoint attribution")
                for participant, numerator, denominator in row.result.participant_credit:
                    rebuilt[participant] = rebuilt.get(participant, Fraction()) + Fraction(numerator, denominator)
                    if not rebuilt[participant]:
                        rebuilt.pop(participant)
        for receipt, result in out.conditionals.items():
            row = out.pending.get(receipt)
            if row is None or row.result is not None:
                raise Refuse("checkpoint conditional receipt")
            prior = out._learned_singleton_effect(result.established)
            effects, touched = out._matched_conditioned_effects(row, result.established)
            if (prior is None or effects is None
                    or result != out._conditioned_attribution(
                        row, result.established, prior, effects, touched)):
                raise Refuse("checkpoint conditional attribution")
        if len(out.nominations) > MAX_NOMINATIONS:
            raise Refuse("checkpoint nomination bound")
        for nomination in out.nominations.values():
            row = out.pending.get(nomination.receipt)
            occurrence = next((item for item in out.population.occurrences
                               if item.identity == nomination.occurrence), None)
            expected_identity = _identity([
                "resident-intervention", nomination.receipt,
                nomination.occurrence, nomination.coalition,
                nomination.opened_tick])
            evidence_match = row is not None and any(
                evidence.occurrence == nomination.occurrence
                and evidence.coalition == nomination.coalition
                for evidence in row.evidence)
            if (row is None or occurrence is None
                    or occurrence.sites != out.population.signature(nomination.coalition)
                    or not set(nomination.coalition) <= set(row.participants)
                    or nomination.identity != expected_identity
                    or nomination.opened_tick < row.opened_tick
                    or nomination.deadline < nomination.opened_tick
                    or nomination.settled not in (0, 1)
                    or bool(nomination.settled) != evidence_match):
                raise Refuse("checkpoint nomination")
        if seen != out.used_occurrences or rebuilt != out.credit:
            raise Refuse("checkpoint causal invariant")
        return out

    def state_hash(self):
        return hashlib.sha256(self.checkpoint().encode()).hexdigest()
