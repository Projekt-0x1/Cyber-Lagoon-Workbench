#!/usr/bin/env python3
"""Fast A/B falsifier for deleting observer-only checkpoint state from AGI.

`last_retrieval` is useful live instrumentation for retrieval assays, but no later
organism transition reads its previous value.  This verifier compares the legacy
serialized diagnostic with the compact checkpoint and requires future behaviour
and future update authority to remain identical.  A causal-state cut is included
as a negative control so output-only equivalence cannot make the battery GREEN.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_mathematical_adult_state_equivalence_verify import (  # noqa: E402
    make_repeated_checkpoint, run_future,
)
from reference_organism_v2 import ReferenceOrganismV2  # noqa: E402


SCENARIOS = (
    "conflict", "independent_support", "withdraw_original",
    "repeat_then_conflict",
)
EMPTY_RETRIEVAL = {"status": 0, "winner": 0, "score": 0, "alternatives": 0}
LEGACY_OBSERVER_VALUE = {"status": 2, "winner": 771, "score": 19, "alternatives": 3}


def wire_bytes(state):
    return len(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())


def future_traces(checkpoint):
    return {scenario: run_future(copy.deepcopy(checkpoint), scenario) for scenario in SCENARIOS}


def main():
    started = time.perf_counter()
    compact = make_repeated_checkpoint()
    legacy = copy.deepcopy(compact)
    legacy["last_retrieval"] = dict(LEGACY_OBSERVER_VALUE)

    restore_started = time.perf_counter_ns()
    restored_legacy = ReferenceOrganismV2.restore(copy.deepcopy(legacy))
    legacy_restore_us = (time.perf_counter_ns() - restore_started) / 1000.0
    restore_started = time.perf_counter_ns()
    restored_compact = ReferenceOrganismV2.restore(copy.deepcopy(compact))
    compact_restore_us = (time.perf_counter_ns() - restore_started) / 1000.0

    future_started = time.perf_counter_ns()
    legacy_traces = future_traces(legacy)
    compact_traces = future_traces(compact)
    future_trace_ms = (time.perf_counter_ns() - future_started) / 1_000_000.0

    # Negative control: source assertions are causal evidence/update authority.
    # The same future battery must reject deleting them.
    causal_cut = copy.deepcopy(compact)
    causal_cut["source_assertions"] = []
    causal_cut_traces = future_traces(causal_cut)

    legacy_bytes = wire_bytes(legacy)
    compact_bytes = wire_bytes(compact)
    differing_negative_scenarios = tuple(
        scenario for scenario in SCENARIOS
        if compact_traces[scenario] != causal_cut_traces[scenario]
    )
    checks = {
        "current_checkpoint_excludes_observer_diagnostic": "last_retrieval" not in compact,
        "legacy_checkpoint_contains_nondefault_observer_diagnostic": legacy["last_retrieval"] != EMPTY_RETRIEVAL,
        "legacy_observer_value_is_ignored_on_restore": restored_legacy.last_retrieval == EMPTY_RETRIEVAL,
        "compact_restore_uses_empty_observer_diagnostic": restored_compact.last_retrieval == EMPTY_RETRIEVAL,
        "immediate_causal_checkpoint_is_identical": restored_legacy.checkpoint() == restored_compact.checkpoint(),
        "immediate_causal_digest_is_identical": restored_legacy.digest() == restored_compact.digest(),
        "future_behavior_and_update_authority_are_identical": legacy_traces == compact_traces,
        "causal_negative_control_is_rejected": bool(differing_negative_scenarios),
        "persistent_representation_is_smaller": compact_bytes < legacy_bytes,
        "bounded_seconds_lane": time.perf_counter() - started < 5.0,
    }
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-observational-state-deletion.v1",
        "pass": all(checks.values()),
        "reference_only": True,
        "state_role": "OBSERVATIONAL",
        "deleted_persistent_field": "last_retrieval",
        "retained_live_diagnostic": True,
        "equivalence": "FUTURE_BEHAVIOR_PLUS_UPDATE_AUTHORITY",
        "scenarios": list(SCENARIOS),
        "negative_control_differing_scenarios": list(differing_negative_scenarios),
        "legacy_checkpoint_bytes": legacy_bytes,
        "compact_checkpoint_bytes": compact_bytes,
        "bytes_saved": legacy_bytes - compact_bytes,
        "compression_ratio": compact_bytes / legacy_bytes,
        "work_shape": {
            "logical_fields_deleted": 1,
            "added_transition_touches": 0,
            "added_materializations": 0,
            "restore_reset_work": "O(1)",
        },
        "timing": {
            "legacy_restore_us": round(legacy_restore_us, 3),
            "compact_restore_us": round(compact_restore_us, 3),
            "future_trace_battery_ms": round(future_trace_ms, 3),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "checks": checks,
    }
    print("FOUNDRY_MATHEMATICAL_ADULT_OBSERVATIONAL_STATE_DELETION_" + ("GREEN" if result["pass"] else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
