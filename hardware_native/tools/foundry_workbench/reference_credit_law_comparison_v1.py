#!/usr/bin/env python3
"""Head-to-head assay for scalar versus matched resident credit laws."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time

from reference_causal_attribution_ecology_v1 import CausalAttributionEcologyV1, DEFAULT_RESIDENT_CAPACITY
from reference_population_v1 import PopulationBankV1, PopulationSpecV1


PARTICIPANTS = (11, 29)
COALITIONS = ((), (11,), (29,), (11, 29))
CANDIDATES = ((701, (11,)), (702, (29,)), (703, (11, 29)))


@dataclass(frozen=True)
class ArmResultV1:
    selected: int
    focal_lesion_selected: int
    revisions: int
    touched_work: int
    checkpoint_bytes: int
    elapsed_us: int


@dataclass(frozen=True)
class ComparisonV1:
    name: str
    causal_optimum: int
    focal_lesion_optimum: int
    scalar: ArmResultV1
    matched: ArmResultV1

    def receipt(self):
        return asdict(self)


def _records(effects, order=COALITIONS):
    return tuple((source, coalition, effects[coalition])
                 for source in (501, 502) for coalition in order)


def _unique_positive(scored):
    peak = max((score for score, _ in scored), default=0)
    winners = [candidate for score, candidate in scored if score == peak]
    return winners[0] if peak > 0 and len(winners) == 1 else 0


def causal_optimum(effects, unavailable=()):
    unavailable = set(unavailable)
    baseline = effects[()]
    scored = []
    for candidate, coalition in CANDIDATES:
        if unavailable & set(coalition):
            continue
        scored.append((effects[coalition] - baseline, candidate))
    return _unique_positive(scored)


class ScalarArmV1:
    """The current local signed update, with no matched causal contrast."""

    def __init__(self, resident_capacity):
        self.population = PopulationBankV1(PopulationSpecV1(
            site_count=resident_capacity, eligibility_horizon=64))
        if set(self.population.signature((11,))) & set(self.population.signature((29,))):
            raise ValueError("participant signature collision")

    def _score(self, features):
        sites = self.population.signature(features)
        edges = (site * self.population.spec.fanout + lane
                 for site in sites for lane in range(self.population.spec.fanout))
        return sum(self.population.edge_weight[edge] - 1 for edge in edges)

    def select(self, unavailable=()):
        unavailable = set(unavailable)
        return _unique_positive([
            (self._score(features), candidate)
            for candidate, features in CANDIDATES
            if not unavailable & set(features)
        ])

    def run(self, records):
        started = time.perf_counter_ns()
        occurrences = [self.population.recruit(coalition) for _, coalition, _ in records]
        first_revision_row = 0
        for index in reversed(range(len(records))):
            _source, _coalition, effect = records[index]
            before = self.population.revision_events
            self.population.settle(occurrences[index], effect, True)
            if not first_revision_row and self.population.revision_events > before:
                first_revision_row = len(records) - index
        elapsed = (time.perf_counter_ns() - started) // 1000
        checkpoint = json.dumps(self.population.checkpoint(), separators=(",", ":"), sort_keys=True)
        touched = (self.population.materialized_site_count()
                   + self.population.touched_incidence_count())
        return ArmResultV1(
            self.select(), self.select((11,)), self.population.revision_events,
            touched, len(checkpoint.encode()), elapsed,
        ), first_revision_row


class MatchedArmV1:
    def __init__(self, resident_capacity):
        self.ecology = CausalAttributionEcologyV1(resident_capacity)

    def run(self, records):
        started = time.perf_counter_ns()
        receipt = self.ecology.participate(PARTICIPANTS, 64)
        occurrences = [self.ecology.population.recruit(coalition) for _, coalition, _ in records]
        for index in reversed(range(len(records))):
            source, coalition, effect = records[index]
            self.ecology.consequence(receipt, occurrences[index], source, coalition, effect, True)
        before = self.ecology.revision_events
        result = self.ecology.resolve(receipt)
        first_revision_row = len(records) if result is not None and self.ecology.revision_events > before else 0
        elapsed = (time.perf_counter_ns() - started) // 1000
        selected = self.ecology.select(CANDIDATES)
        focal = self.ecology.select(CANDIDATES, unavailable=(11,))
        quantity = self.ecology.quantity()
        population_touched = (self.ecology.population.materialized_site_count()
                              + self.ecology.population.touched_incidence_count())
        return ArmResultV1(
            0 if selected is None else selected.candidate,
            0 if focal is None else focal.candidate,
            self.ecology.revision_events,
            population_touched + quantity["evidence_rows"],
            quantity["checkpoint_bytes"], elapsed,
        ), first_revision_row


def compare(name, effects, resident_capacity=DEFAULT_RESIDENT_CAPACITY, order=COALITIONS):
    if set(effects) != set(COALITIONS):
        raise ValueError("complete intervention square required")
    records = _records(effects, order)
    scalar, scalar_first_revision = ScalarArmV1(resident_capacity).run(records)
    matched, matched_first_revision = MatchedArmV1(resident_capacity).run(records)
    return ComparisonV1(
        name, causal_optimum(effects), causal_optimum(effects, unavailable=(11,)),
        scalar, matched,
    ), {
        "evidence_rows": len(records),
        "scalar_first_revision_row": scalar_first_revision,
        "matched_first_revision_row": matched_first_revision,
    }
