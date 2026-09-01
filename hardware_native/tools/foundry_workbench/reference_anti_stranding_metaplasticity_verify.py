#!/usr/bin/env python3
"""End-to-end hostile assay for Sutton-style anti-stranding plasticity."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from reference_anti_stranding_metaplasticity_v1 import AntiStrandingMetaplasticityV1, Q16
from reference_population_v1 import PopulationBankV1, PopulationSpecV1


def edge_set(bank: PopulationBankV1, feature: int) -> set[int]:
    occurrence = bank.activate((feature,), retain=False)
    return set(map(int, occurrence.edges))


def find_partial_overlap_pair(bank: PopulationBankV1):
    rows = [(feature, edge_set(bank, feature)) for feature in range(1, 257)]
    for left, left_edges in rows:
        for right, right_edges in rows:
            if right <= left:
                continue
            overlap = left_edges & right_edges
            left_only = left_edges - right_edges
            right_only = right_edges - left_edges
            if len(overlap) >= 2 and len(left_only) >= 2 and len(right_only) >= 2:
                return left, right, overlap, left_only, right_only
    raise AssertionError("no deterministic partial-overlap feature pair")


def train(bank, ecology, feature, effect, rounds):
    revised = []
    for _ in range(rounds):
        occurrence = bank.recruit((feature,))
        receipt = ecology.settle(bank, occurrence, effect, True)
        revised.extend(receipt["revised_edges"])
    return tuple(revised)


def main():
    started = time.perf_counter()
    checks = {}
    spec = PopulationSpecV1(site_count=64, fanout=2, sites_per_feature=2,
                            territory_count=8, eligibility_horizon=8)
    bank = PopulationBankV1(spec)
    ecology = AntiStrandingMetaplasticityV1(revision_budget=2)
    capacity_before = (bank.spec.site_count, bank.allocated_edge_count)

    feature_a, feature_b, overlap, a_only, b_only = find_partial_overlap_pair(bank)
    checks["fixture_has_real_partial_incidence_overlap"] = (
        len(overlap) >= 2 and len(a_only) >= 2 and len(b_only) >= 2)

    # Establish A until every one of its eligible edges has consequence-earned
    # commitment. This is the protected Sutton-like backbone.
    a_edges = edge_set(bank, feature_a)
    train(bank, ecology, feature_a, +1, 4 * len(a_edges))
    checks["established_a_has_distributed_commitment"] = all(
        ecology.edge_state[edge].commitment_q16 >= Q16 for edge in a_edges)
    a_weights_before_b = {edge: int(bank.edge_weight[edge]) for edge in a_edges}
    a_commit_before_b = {edge: ecology.edge_state[edge].commitment_q16 for edge in a_edges}

    # No independent consequence means no update authority.
    no_credit_occurrence = bank.recruit((feature_b,))
    no_credit_bank = bank.digest(); no_credit_meta = ecology.digest()
    no_credit = ecology.settle(bank, no_credit_occurrence, +1, False)
    checks["nonindependent_contact_cannot_teach"] = (
        no_credit["revisions"] == 0 and bank.digest() == no_credit_bank
        and ecology.digest() == no_credit_meta)

    checkpoint_bank = copy.deepcopy(bank.checkpoint())
    checkpoint_meta = copy.deepcopy(ecology.checkpoint())

    # One novel B consequence must spend its tiny revision budget on B-only
    # fringe incidence, not the high-commitment A/B overlap.
    b_occurrence = bank.recruit((feature_b,))
    first_b = ecology.settle(bank, b_occurrence, +1, True)
    first_b_edges = set(first_b["revised_edges"])
    checks["novel_b_prefers_uncommitted_fringe"] = (
        len(first_b_edges) == ecology.revision_budget
        and first_b_edges <= b_only and not (first_b_edges & overlap))
    checks["established_overlap_survives_first_novel_update"] = all(
        int(bank.edge_weight[edge]) == a_weights_before_b[edge] for edge in overlap)
    checks["established_commitment_not_erased_by_novel_b"] = all(
        ecology.edge_state[edge].commitment_q16 == a_commit_before_b[edge]
        for edge in overlap)

    # Continue B learning long enough to establish its own distinct incidence.
    b_revised = first_b["revised_edges"] + train(bank, ecology, feature_b, +1, 6)
    checks["novel_b_acquires_distinct_resident_structure"] = (
        len(set(b_revised) & b_only) >= 2
        and all(ecology.edge_state[edge].commitment_q16 > 0
                for edge in set(b_revised) & b_only))

    # A naive homogeneous settlement on the exact same pre-B state writes every
    # eligible B edge, including established overlap. This positive control makes
    # the anti-stranding result non-vacuous.
    naive = PopulationBankV1.restore(copy.deepcopy(checkpoint_bank))
    naive_b = naive.recruit((feature_b,))
    naive_before = {edge: int(naive.edge_weight[edge]) for edge in overlap}
    naive.settle(naive_b, +1, True)
    checks["homogeneous_positive_control_cannibalizes_overlap"] = any(
        int(naive.edge_weight[edge]) != naive_before[edge] for edge in overlap)

    # Repeated contradictory consequence must not freeze A forever. Conflict is
    # local resident evidence; it eventually reopens previously protected matter.
    pre_reverse = {edge: int(bank.edge_weight[edge]) for edge in a_edges}
    reversed_edges = []
    for _ in range(12):
        reversed_edges.extend(train(bank, ecology, feature_a, -1, 1))
    checks["contradiction_reopens_established_matter"] = any(
        edge in a_edges and int(bank.edge_weight[edge]) < pre_reverse[edge]
        for edge in set(reversed_edges))
    checks["reversal_is_local_not_global_reset"] = any(
        ecology.edge_state[edge].commitment_q16 > 0 for edge in a_edges)

    # Exact replay from the pre-B checkpoint must reproduce allocation choice and
    # resident state bit-for-bit under the same ordered consequences.
    replay_bank = PopulationBankV1.restore(copy.deepcopy(checkpoint_bank))
    replay_meta = AntiStrandingMetaplasticityV1.restore(copy.deepcopy(checkpoint_meta))
    replay_first = replay_meta.settle(replay_bank, replay_bank.recruit((feature_b,)), +1, True)
    replay_tail = train(replay_bank, replay_meta, feature_b, +1, 6)
    for _ in range(12):
        train(replay_bank, replay_meta, feature_a, -1, 1)
    checks["same_state_same_inputs_same_structural_search"] = (
        tuple(replay_first["revised_edges"]) == tuple(first_b["revised_edges"])
        and replay_bank.digest() == bank.digest()
        and replay_meta.digest() == ecology.digest())

    capacity_after = (bank.spec.site_count, bank.allocated_edge_count)
    checks["fixed_capacity_no_neurogenesis_escape"] = capacity_after == capacity_before
    checks["persistent_metaplastic_state_is_sparse"] = (
        len(ecology.edge_state) < bank.allocated_edge_count)
    checks["touched_work_bounded_by_current_occurrence"] = (
        ecology.last_candidate_touches <= max(len(a_edges), len(edge_set(bank, feature_b))))
    checks["bounded_runtime"] = time.perf_counter() - started < 30.0

    failed = sorted(name for name, value in checks.items() if not value)
    if failed:
        raise SystemExit("FOUNDRY_REFERENCE_ANTI_STRANDING_METAPLASTICITY_RED " + ",".join(failed))

    here = Path(__file__).parent
    paths = [here / "reference_anti_stranding_metaplasticity_v1.py",
             here / "reference_anti_stranding_metaplasticity_verify.py"]
    receipt = {
        "contract": "FOUNDRY_REFERENCE_ANTI_STRANDING_METAPLASTICITY_GREEN",
        "claim": "DETERMINISTIC_RESERVE_FIRST_CONSEQUENCE_PLASTICITY_WITH_REVERSAL_REOPENING_REFERENCE_ONLY",
        "reference_only": True,
        "adult_attached": False,
        "production_direct_parity": "NOT_RUN/RED",
        "runtime_llm": False,
        "randomness": False,
        "gradient_descent": False,
        "host_selected_revision_edge": False,
        "semantic_ids": False,
        "expected_output": False,
        "capacity_before": capacity_before,
        "capacity_after": capacity_after,
        "feature_a": feature_a,
        "feature_b": feature_b,
        "a_edges": len(a_edges),
        "overlap_edges": len(overlap),
        "b_only_edges": len(b_only),
        "revision_budget": ecology.revision_budget,
        "persistent_metaplastic_edges": len(ecology.edge_state),
        "allocated_edges": bank.allocated_edge_count,
        "checks": checks,
        "remaining_red": [
            "DIRECT_PHYSICAL_PARITY",
            "RESIDENT_RECIPE_CONSTRUCTOR_INTEGRATION",
            "LONG_HORIZON_INTERFERENCE_QUANTITY_SWEEP",
            "LANGUAGE_DEVELOPMENT_TRANSFER",
        ],
        "sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in paths},
    }
    print("FOUNDRY_REFERENCE_ANTI_STRANDING_METAPLASTICITY_GREEN")
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
