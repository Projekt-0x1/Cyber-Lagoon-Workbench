#!/usr/bin/env python3
"""Deterministic anti-stranding metaplasticity over the reference population.

OBJECTIVE: let fixed-capacity resident matter learn novel consequences by revising
low-commitment eligible incidence before cannibalizing consequence-established
incidence, while repeated contradictory consequences can reopen established matter.
DONE: the paired verifier proves reserve-first acquisition, established-route
retention, reversal-driven reopening, fixed capacity, sparse touched work, and
checkpoint/exact-replay identity.
NON-GOALS: backprop, IDBD emulation, random reinitialization, semantic routing,
production Direct parity, or a claim that this reference mechanism is canonical.
CONSTRAINTS: deterministic state+ordered-input execution, consequence-only credit,
no capacity growth, no host-selected winning edge, and sparse persistent state.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

from reference_population_v1 import PopulationBankV1, PopulationOccurrenceV1

Q16 = 1 << 16


@dataclass
class EdgePlasticityStateV1:
    commitment_q16: int = 0
    conflict_q16: int = 0
    last_direction: int = 0
    revisions: int = 0


class AntiStrandingMetaplasticityV1:
    """Sparse utilization-dependent writeability for eligible population edges.

    The ecology never receives a task label, desired edge, or expected output.
    It sees only one actual retained occurrence plus an independent signed
    consequence. Candidate incidence is exactly the occurrence's currently
    eligible edges. Low-commitment incidence is more writable; repeated opposing
    consequence accumulates local conflict until established incidence reopens.
    """

    SCHEMA = 1
    BASE_WRITEABILITY_Q16 = Q16
    COMMITMENT_GAIN_Q16 = Q16 // 2
    COMMITMENT_EROSION_Q16 = Q16 // 2
    COMMITMENT_PROTECTION_DIVISOR = 2
    CONFLICT_GAIN_Q16 = Q16 // 2
    CONFLICT_DECAY_Q16 = Q16 // 4
    MAX_COMMITMENT_Q16 = 8 * Q16
    MAX_CONFLICT_Q16 = 8 * Q16

    def __init__(self, revision_budget: int = 2):
        revision_budget = int(revision_budget)
        if not 1 <= revision_budget <= 16:
            raise ValueError("anti_stranding:revision_budget")
        self.revision_budget = revision_budget
        self.edge_state: dict[int, EdgePlasticityStateV1] = {}
        self.settlements = 0
        self.revisions = 0
        self.last_candidate_touches = 0
        self.last_revised_edges: tuple[int, ...] = ()

    def _state(self, edge: int) -> EdgePlasticityStateV1:
        edge = int(edge)
        state = self.edge_state.get(edge)
        if state is None:
            state = EdgePlasticityStateV1()
            self.edge_state[edge] = state
        return state

    @classmethod
    def _writeability_q16(cls, state: EdgePlasticityStateV1, direction: int) -> int:
        protection = state.commitment_q16 // cls.COMMITMENT_PROTECTION_DIVISOR
        opposition = state.last_direction != 0 and state.last_direction != direction
        conflict = state.conflict_q16 if opposition else 0
        return cls.BASE_WRITEABILITY_Q16 + conflict - protection

    def settle(self, bank: PopulationBankV1, occurrence: PopulationOccurrenceV1,
               effect: int, independent: bool = True):
        if occurrence not in bank.occurrences:
            raise ValueError("anti_stranding:occurrence")
        if not independent or int(effect) == 0:
            self.last_candidate_touches = 0
            self.last_revised_edges = ()
            return {"credit": 0, "revisions": 0, "candidates": 0,
                    "revised_edges": ()}

        direction = 1 if int(effect) > 0 else -1
        candidates = []
        for edge in occurrence.edges:
            edge = int(edge)
            if edge < 0 or edge >= bank.allocated_edge_count:
                raise ValueError("anti_stranding:edge")
            if not bank.edge_eligibility[edge]:
                continue
            state = self._state(edge)
            opposition = state.last_direction != 0 and state.last_direction != direction
            if opposition:
                state.conflict_q16 = min(
                    self.MAX_CONFLICT_Q16,
                    state.conflict_q16 + self.CONFLICT_GAIN_Q16,
                )
            elif state.conflict_q16:
                state.conflict_q16 = max(0, state.conflict_q16 - self.CONFLICT_DECAY_Q16)
            candidates.append((self._writeability_q16(state, direction), edge))

        self.last_candidate_touches = len(candidates)
        # High writeability wins. Edge identity is only a deterministic tie-breaker;
        # it carries no semantic or desired-answer authority.
        candidates.sort(key=lambda row: (-row[0], row[1]))
        chosen = tuple(edge for score, edge in candidates[:self.revision_budget] if score > 0)

        revisions = 0
        for edge in chosen:
            state = self._state(edge)
            prior = int(bank.edge_weight[edge])
            new = max(-127, min(127, prior + direction))
            if new == prior:
                continue
            bank.edge_weight[edge] = new
            revisions += 1
            state.revisions += 1
            if state.last_direction in (0, direction):
                state.commitment_q16 = min(
                    self.MAX_COMMITMENT_Q16,
                    state.commitment_q16 + self.COMMITMENT_GAIN_Q16,
                )
            else:
                state.commitment_q16 = max(
                    0, state.commitment_q16 - self.COMMITMENT_EROSION_Q16,
                )
                if state.commitment_q16 == 0:
                    state.last_direction = direction
                    state.conflict_q16 = 0
            if state.last_direction == 0:
                state.last_direction = direction

        credit = sum(1 for site in occurrence.sites if bank.eligibility[site])
        bank.credit_events += credit
        bank.revision_events += revisions
        self.settlements += 1
        self.revisions += revisions
        self.last_revised_edges = chosen
        return {"credit": credit, "revisions": revisions,
                "candidates": len(candidates), "revised_edges": chosen}

    def checkpoint(self):
        return {
            "schema": self.SCHEMA,
            "revision_budget": self.revision_budget,
            "settlements": self.settlements,
            "revisions": self.revisions,
            "edge_state": [
                [edge, state.commitment_q16, state.conflict_q16,
                 state.last_direction, state.revisions]
                for edge, state in sorted(self.edge_state.items())
            ],
        }

    @classmethod
    def restore(cls, data):
        if int(data.get("schema", 0)) != cls.SCHEMA:
            raise ValueError("anti_stranding:checkpoint_schema")
        out = cls(int(data["revision_budget"]))
        out.settlements = int(data["settlements"])
        out.revisions = int(data["revisions"])
        for raw in data.get("edge_state", ()):
            if len(raw) != 5:
                raise ValueError("anti_stranding:checkpoint_state")
            edge, commitment, conflict, direction, revisions = map(int, raw)
            if edge < 0 or commitment < 0 or conflict < 0 or direction not in (-1, 0, 1):
                raise ValueError("anti_stranding:checkpoint_state")
            out.edge_state[edge] = EdgePlasticityStateV1(
                commitment, conflict, direction, revisions)
        return out

    def digest(self):
        h = hashlib.sha256(b"anti-stranding-metaplasticity-v1\0")
        for edge, state in sorted(self.edge_state.items()):
            for value in (edge, state.commitment_q16, state.conflict_q16,
                          state.last_direction, state.revisions):
                h.update(int(value).to_bytes(8, "little", signed=True))
        for value in (self.revision_budget, self.settlements, self.revisions):
            h.update(int(value).to_bytes(8, "little", signed=False))
        return h.hexdigest()
