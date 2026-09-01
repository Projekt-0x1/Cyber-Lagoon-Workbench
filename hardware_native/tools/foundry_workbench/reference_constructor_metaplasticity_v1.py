#!/usr/bin/env python3
"""Content-free consequence-backed construction-horizon metaplasticity.

The constructor does not know language, mathematics, tasks, answers, or operator
families. It owns only how deep already-supported dependency structure may be
unfolded. A bounded horizon increase requires an independently returned useful
consequence from a child whose exact structural ancestry is carried in a
constructor-issued receipt. Rehearsal, a yoked return, or a forged receipt cannot
revise the constructor.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

Q = 1 << 16
BASE_DEPENDENCY_DEPTH = 2
MAX_DEPENDENCY_DEPTH = 8
PLASTICITY_STEP_Q16 = Q // 8
MAX_META_REVISIONS = MAX_DEPENDENCY_DEPTH - BASE_DEPENDENCY_DEPTH
MAX_CONSUMED_VIABILITY = 256


def _digest(tag, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=list).encode()
    return hashlib.sha256(tag.encode() + b"\\0" + raw).hexdigest()


def _identity(tag, value):
    return int(_digest(tag, value)[:16], 16) or 1


@dataclass(frozen=True)
class ConstructedChildReceiptV1:
    constructor_identity: int
    constructor_revision: int
    structure_identity: int
    dependency_depth: int
    ancestry_digest: str
    ancestry_roots: tuple[int, ...]
    identity: int


class GenericConstructorMetaplasticityV1:
    """One generic constructor state shared by heterogeneous construction paths."""

    def __init__(self, base_depth=BASE_DEPENDENCY_DEPTH,
                 max_depth=MAX_DEPENDENCY_DEPTH):
        base_depth = int(base_depth)
        max_depth = int(max_depth)
        if not 1 <= base_depth <= max_depth <= MAX_DEPENDENCY_DEPTH:
            raise ValueError("constructor_meta:depth")
        self.base_depth = base_depth
        self.max_depth_cap = max_depth
        self.revision = 0
        self.plasticity_q16 = 0
        self.max_dependency_depth = base_depth
        self.consumed_viability_evidence: set[int] = set()
        self.exact_constructor_evidence_lineage: list[int] = []
        self.identity = _identity(
            "generic-constructor-metaplasticity-v1",
            (self.base_depth, self.max_depth_cap, PLASTICITY_STEP_Q16),
        )

    def admits(self, dependency_depth):
        depth = int(dependency_depth)
        return 0 <= depth <= int(self.max_dependency_depth)

    @staticmethod
    def ancestry_digest(structure_identity, dependency_depth, ancestry):
        return _digest(
            "constructor-child-ancestry-v1",
            (int(structure_identity), int(dependency_depth), tuple(map(int, ancestry))),
        )

    def _receipt_identity(self, revision, structure_identity, dependency_depth,
                          ancestry_digest, ancestry_roots):
        return _identity(
            "constructor-child-receipt-v1",
            (
                self.identity,
                int(revision),
                int(structure_identity),
                int(dependency_depth),
                str(ancestry_digest),
                tuple(map(int, ancestry_roots)),
            ),
        )

    def receipt(self, structure_identity, dependency_depth, ancestry,
                ancestry_roots=()):
        """Mint structural ancestry for a child that this horizon can actually build."""
        structure_identity = int(structure_identity)
        dependency_depth = int(dependency_depth)
        roots = tuple(sorted(set(int(x) for x in ancestry_roots if int(x) > 0)))
        ancestry = tuple(map(int, ancestry))
        if structure_identity <= 0 or not self.admits(dependency_depth):
            return None
        if dependency_depth > 0 and not ancestry:
            return None
        digest = self.ancestry_digest(structure_identity, dependency_depth, ancestry)
        rid = self._receipt_identity(
            self.revision, structure_identity, dependency_depth, digest, roots)
        return ConstructedChildReceiptV1(
            self.identity, self.revision, structure_identity, dependency_depth,
            digest, roots, rid)

    def _valid_receipt(self, receipt):
        if not isinstance(receipt, ConstructedChildReceiptV1):
            return False
        if int(receipt.constructor_identity) != int(self.identity):
            return False
        if not 0 <= int(receipt.constructor_revision) <= int(self.revision):
            return False
        if not 0 <= int(receipt.dependency_depth) <= self.max_depth_cap:
            return False
        expected = self._receipt_identity(
            receipt.constructor_revision,
            receipt.structure_identity,
            receipt.dependency_depth,
            receipt.ancestry_digest,
            receipt.ancestry_roots,
        )
        return int(receipt.identity) == int(expected)

    def settle_child(self, receipt, outcome_q16, consequence_source,
                     independent=True, controllable=True, endogenous=False):
        """Consume one downstream viability witness and, at most, one bounded step."""
        source = int(consequence_source)
        outcome = int(outcome_q16)
        if (not self._valid_receipt(receipt) or outcome <= 0 or source <= 0
                or not bool(independent) or not bool(controllable)
                or bool(endogenous)):
            return False
        if source in set(map(int, receipt.ancestry_roots)):
            return False
        # A successful construction can expand structural reach only when it
        # actually exercised the constructor's current frontier. Success far
        # below that frontier says nothing about whether one more dependency
        # level is viable, and a stale pre-revision receipt cannot be replayed
        # later to ratchet depth again.
        if int(receipt.dependency_depth) != int(self.max_dependency_depth):
            return False
        evidence = _identity(
            "constructor-downstream-viability-v1",
            (int(receipt.identity), source),
        )
        if evidence in self.consumed_viability_evidence:
            return False
        if len(self.consumed_viability_evidence) >= MAX_CONSUMED_VIABILITY:
            return False
        self.consumed_viability_evidence.add(evidence)
        self.exact_constructor_evidence_lineage.append(evidence)
        if (self.revision >= MAX_META_REVISIONS
                or self.max_dependency_depth >= self.max_depth_cap):
            return False
        self.revision += 1
        self.plasticity_q16 = min(Q, self.plasticity_q16 + PLASTICITY_STEP_Q16)
        self.max_dependency_depth = min(
            self.max_depth_cap, self.base_depth + self.revision)
        return True

    def lesion_meta_state(self):
        """Experimental focal lesion: remove susceptibility gain, not evidence lineage."""
        self.revision = 0
        self.plasticity_q16 = 0
        self.max_dependency_depth = self.base_depth

    def checkpoint(self):
        return {
            "schema": 1,
            "base_depth": self.base_depth,
            "max_depth_cap": self.max_depth_cap,
            "revision": self.revision,
            "plasticity_q16": self.plasticity_q16,
            "max_dependency_depth": self.max_dependency_depth,
            "consumed_viability_evidence": sorted(self.consumed_viability_evidence),
            "exact_constructor_evidence_lineage": list(self.exact_constructor_evidence_lineage),
        }

    @classmethod
    def restore(cls, data):
        if int(data.get("schema", 0)) != 1:
            raise ValueError("constructor_meta:checkpoint_schema")
        out = cls(int(data["base_depth"]), int(data["max_depth_cap"]))
        out.revision = int(data["revision"])
        out.plasticity_q16 = int(data["plasticity_q16"])
        out.max_dependency_depth = int(data["max_dependency_depth"])
        consumed = tuple(map(int, data.get("consumed_viability_evidence", ())))
        lineage = tuple(map(int, data.get("exact_constructor_evidence_lineage", ())))
        if (not 0 <= out.revision <= MAX_META_REVISIONS
                or out.max_dependency_depth != min(
                    out.max_depth_cap, out.base_depth + out.revision)
                or out.plasticity_q16 != min(Q, out.revision * PLASTICITY_STEP_Q16)
                or len(consumed) > MAX_CONSUMED_VIABILITY
                or len(set(consumed)) != len(consumed)
                or tuple(sorted(consumed)) != tuple(sorted(lineage))
                or len(lineage) != len(consumed)):
            raise ValueError("constructor_meta:checkpoint_state")
        out.consumed_viability_evidence = set(consumed)
        out.exact_constructor_evidence_lineage = list(lineage)
        return out

    def digest(self):
        return _digest("constructor-metaplasticity-checkpoint-v1", self.checkpoint())
