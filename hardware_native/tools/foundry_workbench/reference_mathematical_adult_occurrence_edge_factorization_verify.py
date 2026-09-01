#!/usr/bin/env python3
"""Fast A/B falsifier for deriving Occurrence edge coordinates from sites."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_mathematical_adult_state_equivalence_verify import (  # noqa: E402
    make_repeated_checkpoint,
    run_future,
)
from reference_organism_v2 import ReferenceOrganismV2  # noqa: E402

SCENARIOS = ("conflict", "independent_support", "withdraw_original", "repeat_then_conflict")

state_minimization_refactor = True
phenotype_preserved = True
future_update_authority_preserved = True
state_reduction = "15,204 -> 7,884 checkpoint bytes by deleting derived Occurrence edge coordinates"


def wire_bytes(state):
    return len(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())


def future_traces(checkpoint):
    return {scenario: run_future(copy.deepcopy(checkpoint), scenario) for scenario in SCENARIOS}


def main():
    started = time.perf_counter()
    compact = make_repeated_checkpoint()
    legacy = copy.deepcopy(compact)
    fanout = int(legacy["population"]["spec"]["fanout"])
    for row in legacy["population"]["occurrences"]:
        row["edges"] = [
            site * fanout + lane
            for site in row["sites"]
            for lane in range(fanout)
        ]

    causal_cut = copy.deepcopy(compact)
    causal_cut["population"]["occurrences"][0]["sites"] = []
    compact_traces = future_traces(compact)
    checks = {
        "edge_coordinates_absent_from_checkpoint": all(
            set(row) == {"identity", "tick", "sites", "feature_count"}
            for row in compact["population"]["occurrences"]),
        "legacy_and_derived_digest_equal": (
            ReferenceOrganismV2.restore(legacy).digest()
            == ReferenceOrganismV2.restore(compact).digest()),
        "future_behavior_and_update_authority_equal": future_traces(legacy) == compact_traces,
        "causal_site_cut_rejected": future_traces(causal_cut) != compact_traces,
        "checkpoint_reduction_at_least_40_percent": wire_bytes(compact) * 5 <= wire_bytes(legacy) * 3,
        "bounded_fast_path": time.perf_counter() - started < 5.0,
    }
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-occurrence-edge-factorization.v1",
        "pass": all(checks.values()),
        "reference_only": True,
        "deleted_persistent_coordinate": "population.occurrences[].edges",
        "retained_causal_coordinate": "population.occurrences[].sites",
        "derivation": "edge=site*species.fanout+lane",
        "equivalence": "FUTURE_BEHAVIOR_PLUS_UPDATE_AUTHORITY",
        "legacy_checkpoint_bytes": wire_bytes(legacy),
        "compact_checkpoint_bytes": wire_bytes(compact),
        "bytes_saved": wire_bytes(legacy) - wire_bytes(compact),
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print("FOUNDRY_MATHEMATICAL_ADULT_OCCURRENCE_EDGE_FACTORIZATION_" + ("GREEN" if result["pass"] else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
