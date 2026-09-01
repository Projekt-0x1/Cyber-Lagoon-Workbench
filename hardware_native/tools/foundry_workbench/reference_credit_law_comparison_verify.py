#!/usr/bin/env python3
from __future__ import annotations

import json
import time

from reference_credit_law_comparison_v1 import compare


CASES = {
    "additive_control": {(): 0, (11,): 4, (29,): 1, (11, 29): 5},
    "contingency_degraded": {(): 3, (11,): 3, (29,): 3, (11, 29): 3},
    "pure_synergy": {(): 0, (11,): 0, (29,): 0, (11, 29): 6},
    "redundant_causes": {(): 0, (11,): 4, (29,): 4, (11, 29): 4},
    "negative_patterning": {(): 0, (11,): 4, (29,): 4, (11, 29): 0},
    "opposed_causes": {(): 0, (11,): 5, (29,): -3, (11, 29): 2},
}


def main():
    started = time.perf_counter()
    comparisons = {}
    acquisition = {}
    for name, effects in CASES.items():
        result, cost = compare(name, effects, resident_capacity=32_768)
        comparisons[name] = result
        acquisition[name] = cost

    checks = {
        "scalar_sufficient_additive_control": (
            comparisons["additive_control"].scalar.selected
            == comparisons["additive_control"].causal_optimum
            == comparisons["additive_control"].matched.selected),
        "matched_rejects_noncontingent_reward": (
            comparisons["contingency_degraded"].causal_optimum == 0
            and comparisons["contingency_degraded"].matched.selected == 0),
        "scalar_false_credit_under_contingency_degradation": (
            comparisons["contingency_degraded"].scalar.selected != 0),
        "matched_preserves_synergy_after_focal_lesion": (
            comparisons["pure_synergy"].matched.selected
            == comparisons["pure_synergy"].causal_optimum == 703
            and comparisons["pure_synergy"].matched.focal_lesion_selected
            == comparisons["pure_synergy"].focal_lesion_optimum == 0),
        "scalar_spreads_synergy_to_survivor": (
            comparisons["pure_synergy"].scalar.focal_lesion_selected != 0),
        "matched_redundancy_refuses_arbitrary_winner": (
            comparisons["redundant_causes"].matched.selected
            == comparisons["redundant_causes"].causal_optimum == 0),
        "scalar_double_counts_redundant_coalition": (
            comparisons["redundant_causes"].scalar.selected != 0),
        "matched_preserves_negative_patterning": (
            comparisons["negative_patterning"].matched.selected
            == comparisons["negative_patterning"].causal_optimum == 0
            and comparisons["negative_patterning"].matched.focal_lesion_selected
            == comparisons["negative_patterning"].focal_lesion_optimum == 702),
        "scalar_loses_negative_patterning": (
            comparisons["negative_patterning"].scalar.selected != 0),
        "matched_opposition_selects_only_supportive_cause": (
            comparisons["opposed_causes"].matched.selected
            == comparisons["opposed_causes"].causal_optimum == 701),
        "scalar_loses_opposed_signs": (
            comparisons["opposed_causes"].scalar.selected
            != comparisons["opposed_causes"].causal_optimum),
        "matched_pays_for_causal_evidence": all(
            cost["matched_first_revision_row"] in (0, cost["evidence_rows"])
            and 0 <= cost["scalar_first_revision_row"] < cost["evidence_rows"]
            for cost in acquisition.values()),
        "bounded_sparse_work": all(
            result.matched.touched_work < 256 and result.scalar.touched_work < 256
            for result in comparisons.values()),
    }

    scalar_correct = sum(result.scalar.selected == result.causal_optimum
                         and result.scalar.focal_lesion_selected == result.focal_lesion_optimum
                         for result in comparisons.values())
    matched_correct = sum(result.matched.selected == result.causal_optimum
                          and result.matched.focal_lesion_selected == result.focal_lesion_optimum
                          for result in comparisons.values())
    checks["matched_behavioral_advantage"] = matched_correct > scalar_correct
    elapsed_ms = (time.perf_counter() - started) * 1000
    checks["bounded_runtime"] = elapsed_ms < 5000

    receipt = {
        "schema": "agi.reference-credit-law-comparison.v1",
        "pass": all(checks.values()),
        "checks": checks,
        "behavior": {
            "cases": len(comparisons),
            "lesion_conditions_per_case": 2,
            "scalar_correct": scalar_correct,
            "matched_correct": matched_correct,
            "scalar_readout": "ASSAY_ONLY_EDGE_WEIGHT_DELTA_NOT_ORGANISM_SELECTION",
        },
        "acquisition": acquisition,
        "comparisons": {name: result.receipt() for name, result in comparisons.items()},
        "elapsed_ms": round(elapsed_ms, 3),
        "reference_only": True,
        "adult_attached": False,
        "runtime_llm": False,
        "graph_flip": False,
        "physical_direct_parity": "NOT_RUN/RED",
        "claim": "CAUSAL_CREDIT_ADOPTION_BENCHMARK_REFERENCE_ONLY",
        "known_red": {
            "sequential_blocking": "SEPARATE_CONDITIONED_REFERENCE_ASSAY_GREEN/DIRECT_RED",
            "adult_authenticated_independent_sources": "NOT_ATTACHED/RED",
            "resident_originated_intervention_square": "SEPARATE_REFERENCE_ASSAY_GREEN/DIRECT_RED",
        },
    }
    print("FOUNDRY_CREDIT_LAW_COMPARISON " + ("GREEN" if receipt["pass"] else "RED"))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if receipt["pass"] else 1)


if __name__ == "__main__":
    main()
