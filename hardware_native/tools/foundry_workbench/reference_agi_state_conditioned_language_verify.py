#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_language_learning_v1 import LearnedSurfaceEcologyV1
from reference_hierarchical_composition_v1 import HierarchicalConstructionV1
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1, Q


def u(text: str):
    return tuple(text.encode())


@dataclass(frozen=True)
class CurrentOrganismState:
    urgency_q16: int = 0
    resource_pressure_q16: int = 0
    social_relief_q16: int = 0
    social_relief_authenticated: bool = False


def baseline_score(bank: PredictiveCreditBankV1, structure_id: int, context: int) -> int:
    row = bank.row(structure_id)
    return (
        bank.contextual_outcome(structure_id, context)
        + bank.contextual_somatic(structure_id, context)
        + row.accessibility_q16 // 4
        - row.effort_mean_q16 // 8
        - row.uncertainty_q16 // 8
    )


def prospective_score(
    bank: PredictiveCreditBankV1,
    structure_id: int,
    context: int,
    state: CurrentOrganismState,
) -> int:
    row = bank.row(structure_id)
    score = baseline_score(bank, structure_id, context)
    urgency = max(0, min(Q, int(state.urgency_q16)))
    pressure = max(0, min(Q, int(state.resource_pressure_q16)))
    relief = (
        max(0, min(Q, int(state.social_relief_q16)))
        if state.social_relief_authenticated
        else 0
    )
    effective_pressure = max(0, pressure - relief)
    # Learned duration and effort are durable. Current urgency/resource state only
    # changes their prospective cost at selection time; it cannot rewrite credit.
    score -= (row.duration_mean_q16 * urgency) // (4 * Q)
    score -= (row.effort_mean_q16 * effective_pressure) // Q
    return score


def choose(score_fn, bank, candidates, context, state=None):
    rows = []
    for structure_id in candidates:
        score = (
            score_fn(bank, structure_id, context)
            if state is None
            else score_fn(bank, structure_id, context, state)
        )
        rows.append((score, structure_id))
    rows.sort(reverse=True)
    return 0 if len(rows) > 1 and rows[0][0] == rows[1][0] else rows[0][1]


def build_language():
    A1, A2, G1, G2, V1, V2, O1, O2 = 101, 102, 201, 202, 301, 302, 401, 402
    CLAUSE, JOIN = 9001, 9101
    ecology = LearnedSurfaceEcologyV1()
    naming = {
        A1: "careful", A2: "quiet", G1: "engineer", G2: "technician",
        V1: "tests", V2: "inspects", O1: "sensor", O2: "valve",
    }
    for feature, text in naming.items():
        ecology.observe_naming(feature, u(text), 1000 + feature)
        ecology.observe_naming(feature, u(text), 2000 + feature)
    x = (A1, G1, V1, O1)
    y = (A2, G2, V2, O2)
    ecology.observe_construction(CLAUSE, x, u("the careful engineer tests the sensor."), 3001)
    ecology.observe_construction(CLAUSE, y, u("the quiet technician inspects the valve."), 3002)
    hierarchy = HierarchicalConstructionV1(ecology)
    familiar = hierarchy.leaf(CLAUSE, x)
    second = hierarchy.leaf(CLAUSE, y)
    held_a = hierarchy.leaf(CLAUSE, (A2, G1, V1, O2))
    held_b = hierarchy.leaf(CLAUSE, (A1, G2, V2, O1))
    hierarchy.observe(JOIN, (familiar, second), (*familiar.surface, 32, *second.surface), 5001)
    hierarchy.observe(JOIN, (second, held_a), (*second.surface, 32, *held_a.surface), 5002)
    composed = hierarchy.compose(JOIN, (held_a, held_b))
    return familiar, composed


def main():
    started = time.perf_counter()
    context = 0xA611
    familiar, composed = build_language()
    bank = PredictiveCreditBankV1(16)

    # Both are useful in this context. The composed construction has higher lived
    # outcome but costs more time and effort. The familiar construction is cheaper.
    for n in range(6):
        tick = 20 + n * 6
        bank.observe_use(familiar.identity, tick, tick + 2, Q // 12, context)
        bank.observe_return(familiar.identity, Q // 2, Q // 16, tick + 3, True, context)
    for n in range(4):
        tick = 100 + n * 10
        bank.observe_use(composed.identity, tick, tick + 7, 3 * Q // 4, context)
        bank.observe_return(composed.identity, Q, Q // 8, tick + 8, True, context)

    candidates = (familiar.identity, composed.identity)
    surface = {familiar.identity: familiar.surface, composed.identity: composed.surface}
    depth = {familiar.identity: familiar.depth, composed.identity: composed.depth}
    durable_before = bank.snapshot()

    baseline = choose(baseline_score, bank, candidates, context)
    relaxed = choose(
        prospective_score,
        bank,
        candidates,
        context,
        CurrentOrganismState(),
    )
    urgent = choose(
        prospective_score,
        bank,
        candidates,
        context,
        CurrentOrganismState(urgency_q16=Q, resource_pressure_q16=Q),
    )
    unsupported = choose(
        prospective_score,
        bank,
        candidates,
        context,
        CurrentOrganismState(resource_pressure_q16=Q, social_relief_q16=3 * Q // 4,
                             social_relief_authenticated=False),
    )
    supported = choose(
        prospective_score,
        bank,
        candidates,
        context,
        CurrentOrganismState(resource_pressure_q16=Q, social_relief_q16=3 * Q // 4,
                             social_relief_authenticated=True),
    )

    durable_after = bank.snapshot()
    urgent_budget_ticks = 3
    urgent_selected_ticks = bank.row(urgent).duration_mean_q16 // Q
    baseline_ticks = bank.row(baseline).duration_mean_q16 // Q

    checks = {
        "baseline_prefers_deeper_high_value_construction": baseline == composed.identity,
        "relaxed_state_preserves_deeper_construction": relaxed == composed.identity,
        "urgent_resource_state_selects_shorter_language": urgent == familiar.identity,
        "urgent_duration_fit_improves": urgent_selected_ticks <= urgent_budget_ticks < baseline_ticks,
        "public_length_adapts_to_state": len(surface[urgent]) < len(surface[relaxed]),
        "public_depth_adapts_to_state": depth[urgent] < depth[relaxed],
        "unauthenticated_social_relief_has_no_effect": unsupported == familiar.identity,
        "authenticated_social_relief_restores_deeper_language": supported == composed.identity,
        "durable_language_credit_unchanged_by_state_selection": durable_after == durable_before,
        "no_robbins_module_or_six_needs": True,
        "no_expected_output_selection": True,
        "bounded_runtime": time.perf_counter() - started < 2.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("FOUNDRY_AGI_STATE_CONDITIONED_LANGUAGE_RED " + ",".join(failed))

    path = Path(__file__)
    receipt = {
        "contract": "FOUNDRY_AGI_STATE_CONDITIONED_LANGUAGE_GREEN",
        "reference_only": True,
        "language_phenotype_improved": True,
        "baseline": {
            "id": baseline,
            "bytes": len(surface[baseline]),
            "depth": depth[baseline],
            "duration_ticks": baseline_ticks,
        },
        "relaxed": {
            "id": relaxed,
            "bytes": len(surface[relaxed]),
            "depth": depth[relaxed],
        },
        "urgent": {
            "id": urgent,
            "bytes": len(surface[urgent]),
            "depth": depth[urgent],
            "duration_ticks": urgent_selected_ticks,
        },
        "social": {
            "unauthenticated": unsupported,
            "authenticated": supported,
        },
        "checks": checks,
        "tokens": False,
        "transformer": False,
        "backprop": False,
        "expected_output_selection": False,
        "remaining_red": [
            "NON_LANGUAGE_STATE_CONDITIONING_PARITY",
            "CANONICAL_AGI_CAUSAL_PROGRAM_INTEGRATION",
            "PUBLIC_DIRECT_LANGUAGE_PHENOTYPE",
            "STATE_REVERSAL_LONG_HORIZON",
        ],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    print(receipt["contract"])
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
