#!/usr/bin/env python3
"""Tiny exact Workbench interpreter for Direct white-box relation equations.

This is intentionally not a second cognitive architecture. It mirrors the four
bounded exact reductions in direct_mathematical_relation_algebra.cuh so mixed
composition can be falsified in milliseconds instead of recompiling Direct.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

Q = 1 << 16
AFFINE = 0
POLYNOMIAL = 1
SCHUR = 2
BISIMULATION = 3


def _identity(tag: str, payload) -> int:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    value = int.from_bytes(hashlib.sha256(tag.encode() + b"\0" + raw).digest()[:8], "big")
    return value or 1


def qadd(left: int, right: int) -> int | None:
    value = int(left) + int(right)
    return value if -(1 << 31) <= value < (1 << 31) else None


def qmul(left: int, right: int) -> int | None:
    product = int(left) * int(right)
    if product % Q:
        return None
    value = product // Q
    return value if -(1 << 31) <= value < (1 << 31) else None


def qdiv(numerator: int, denominator: int) -> int | None:
    if not denominator:
        return None
    scaled = int(numerator) * Q
    if scaled % int(denominator):
        return None
    value = scaled // int(denominator)
    return value if -(1 << 31) <= value < (1 << 31) else None


@dataclass(frozen=True)
class ExactRelationSourceV1:
    kind: int
    coefficients_q16: tuple[int, int, int, int] = (0, 0, 0, 0)
    successors: tuple[int, int, int, int] = (0, 0, 0, 0)
    outputs_q16: tuple[int, int, int, int] = (0, 0, 0, 0)
    state_count: int = 0
    source_identity: int = 0

    def sealed(self):
        payload = (
            int(self.kind), tuple(map(int, self.coefficients_q16)),
            tuple(map(int, self.successors)), tuple(map(int, self.outputs_q16)),
            int(self.state_count),
        )
        return ExactRelationSourceV1(
            int(self.kind), tuple(map(int, self.coefficients_q16)),
            tuple(map(int, self.successors)), tuple(map(int, self.outputs_q16)),
            int(self.state_count), _identity("exact-relation-source-v1", payload),
        )


@dataclass(frozen=True)
class ExactRelationWitnessV1:
    source: ExactRelationSourceV1
    result_q16: tuple[int, ...]
    result_u32: tuple[int, ...]
    class_by_state: tuple[int, ...]
    eliminated_count: int
    witness_identity: int


def _witness(source, result_q16=(), result_u32=(), classes=(), eliminated=0):
    payload = (
        int(source.source_identity), tuple(map(int, result_q16)),
        tuple(map(int, result_u32)), tuple(map(int, classes)), int(eliminated),
    )
    return ExactRelationWitnessV1(
        source, tuple(map(int, result_q16)), tuple(map(int, result_u32)),
        tuple(map(int, classes)), int(eliminated),
        _identity("exact-relation-witness-v1", payload),
    )


def reduce_exact(source: ExactRelationSourceV1):
    if source.source_identity <= 0 or source.sealed().source_identity != source.source_identity:
        return None
    kind = int(source.kind)
    if kind not in (AFFINE, POLYNOMIAL, SCHUR, BISIMULATION):
        return None
    c = tuple(map(int, source.coefficients_q16))
    if kind == AFFINE:
        linear = qmul(c[2], c[0])
        scaled_offset = qmul(c[2], c[1])
        offset = None if scaled_offset is None else qadd(scaled_offset, c[3])
        if linear is None or offset is None:
            return None
        return _witness(source, (linear, offset), eliminated=1)
    if kind == POLYNOMIAL:
        a2, ab, b2 = qmul(c[0], c[0]), qmul(c[0], c[1]), qmul(c[1], c[1])
        if None in (a2, ab, b2):
            return None
        quadratic = qmul(c[2], a2)
        half_linear = qmul(c[2], ab)
        constant = qmul(c[2], b2)
        linear = None if half_linear is None else qadd(half_linear, half_linear)
        constant = None if constant is None else qadd(constant, c[3])
        if None in (quadratic, linear, constant):
            return None
        return _witness(source, (quadratic, linear, constant), eliminated=1)
    if kind == SCHUR:
        denominator = Q - c[3]
        product = qmul(c[1], c[2])
        quotient = None if product is None else qdiv(product, denominator)
        result = None if quotient is None else qadd(c[0], quotient)
        if result is None:
            return None
        return _witness(source, (result,), eliminated=1)

    n = int(source.state_count)
    if not 1 <= n <= 4 or any(not 0 <= int(s) < n for s in source.successors[:n]):
        return None
    classes = []
    for state in range(n):
        assigned = next((classes[p] for p in range(state)
                         if source.outputs_q16[p] == source.outputs_q16[state]), None)
        classes.append(state if assigned is None else assigned)
    for _ in range(n):
        next_classes = []
        next_count = 0
        for state in range(n):
            assigned = None
            for prior in range(state):
                if (source.outputs_q16[prior] == source.outputs_q16[state]
                        and classes[source.successors[prior]] == classes[source.successors[state]]):
                    assigned = next_classes[prior]
                    break
            if assigned is None:
                assigned = next_count
                next_count += 1
            next_classes.append(assigned)
        if next_classes == classes:
            break
        classes = next_classes
    class_count = max(classes) + 1
    result_u32 = [0] * class_count
    result_q16 = [0] * class_count
    for state in range(n):
        cls = classes[state]
        result_u32[cls] = classes[source.successors[state]]
        result_q16[cls] = int(source.outputs_q16[state])
    return _witness(source, result_q16, result_u32, classes, n - class_count)


def algebraic_source(kind, a, b, c, d):
    return ExactRelationSourceV1(int(kind), (int(a), int(b), int(c), int(d))).sealed()


def bisimulation_source(successors, outputs):
    successors = tuple(map(int, successors)); outputs = tuple(map(int, outputs))
    if len(successors) != len(outputs) or not 1 <= len(successors) <= 4:
        raise ValueError("exact-relation:bisimulation-shape")
    pad_s = successors + (0,) * (4 - len(successors))
    pad_o = outputs + (0,) * (4 - len(outputs))
    return ExactRelationSourceV1(BISIMULATION, successors=pad_s,
                                 outputs_q16=pad_o, state_count=len(successors)).sealed()


def compose_affine(left: tuple[int, int], right: tuple[int, int]):
    """Return right(left(x)) without materializing an unretained certificate."""
    linear = qmul(right[0], left[0])
    scaled_offset = qmul(right[0], left[1])
    offset = None if scaled_offset is None else qadd(scaled_offset, right[1])
    if linear is None or offset is None:
        return None
    return linear, offset
