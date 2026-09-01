#!/usr/bin/env python3
"""Fast A/B verifier for conditionally factoring pending language-action state.

For an ordinary pending action, `planned_payload == payload`; serializing both is a
duplicate coordinate choice, not an additional future distinction.  Faulted output
is the negative control: when actual bytes diverge from the plan, both values remain
causal because repair and credit eligibility depend on the distinction.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_organism_repair_verify import CTX, P1, partner, scene, train  # noqa: E402
from reference_organism_v2 import CONTACT_CONSEQUENCE, ReferenceOrganismV2  # noqa: E402
from reference_population_v1 import PopulationSpecV1  # noqa: E402

# Pending-action factorization is independent of dormant population quantity.
# The factory's population-delta checkpoint assay retains the 32,768-site control.
SPEC = PopulationSpecV1(1024, 2, 4, 42, 8)
HELDOUT = (102, 201, 301, 402)


def wire_bytes(state):
    return len(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())


def build_pending(source_scene):
    o = ReferenceOrganismV2(SPEC)
    train(o)
    partner(o, P1)
    scene(o, CTX, HELDOUT, source_scene)
    action = o.tick()
    if action is None:
        raise RuntimeError("pending-action-factorization:no_action")
    return o, action


def main():
    started = time.perf_counter()
    ordinary, action = build_pending(45001)
    compact = ordinary.checkpoint()
    if len(compact["actions"]) != 1:
        raise RuntimeError("pending-action-factorization:action_count")
    compact_row = compact["actions"][0]

    # Reconstruct the legacy duplicate representation for the A/B donor.
    legacy = copy.deepcopy(compact)
    legacy["actions"][0]["planned_payload"] = tuple(action.planned_payload)
    legacy_bytes = wire_bytes(legacy)
    compact_bytes = wire_bytes(compact)

    legacy_restore_started = time.perf_counter_ns()
    donor = ReferenceOrganismV2.restore(copy.deepcopy(legacy))
    legacy_restore_us = (time.perf_counter_ns() - legacy_restore_started) / 1000.0
    compact_restore_started = time.perf_counter_ns()
    challenger = ReferenceOrganismV2.restore(copy.deepcopy(compact))
    compact_restore_us = (time.perf_counter_ns() - compact_restore_started) / 1000.0

    before_equal = donor.digest() == challenger.digest()
    donor_return = donor.contact(CONTACT_CONSEQUENCE, (action.ticket, 1), P1, True, True)
    challenger_return = challenger.contact(CONTACT_CONSEQUENCE, (action.ticket, 1), P1, True, True)
    after_equal = donor_return == challenger_return and donor.digest() == challenger.digest()

    # Negative control: a real output fault makes the plan/actual distinction causal.
    faulted = ReferenceOrganismV2(SPEC)
    train(faulted); partner(faulted, P1); scene(faulted, CTX, HELDOUT, 45002)
    faulted.inject_output_fault(0, ord("X"))
    bad = faulted.tick()
    fault_checkpoint = faulted.checkpoint()
    fault_row = fault_checkpoint["actions"][0]
    fault_replay = ReferenceOrganismV2.restore(copy.deepcopy(fault_checkpoint))
    fault_replay.contact(CONTACT_CONSEQUENCE, (bad.ticket, -1), P1, True, True)
    repair = fault_replay.tick()

    checks = {
        "ordinary_action_plan_equals_actual": tuple(action.planned_payload) == tuple(action.payload),
        "ordinary_checkpoint_derives_planned_payload": "planned_payload" not in compact_row,
        "legacy_duplicate_is_larger": legacy_bytes > compact_bytes,
        "legacy_and_compact_restore_same_causal_state": before_equal,
        "settlement_and_future_state_are_identical": after_equal,
        "fault_plan_differs_from_actual": tuple(bad.planned_payload) != tuple(bad.payload),
        "fault_checkpoint_retains_irreducible_plan": "planned_payload" in fault_row,
        "fault_checkpoint_plan_is_exact": tuple(fault_row["planned_payload"]) == tuple(bad.planned_payload),
        "fault_replay_repairs_from_retained_plan": repair is not None and tuple(repair.payload) == tuple(bad.planned_payload),
        "bounded_seconds_lane": time.perf_counter() - started < 5.0,
    }
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-pending-action-factorization.v1",
        "pass": all(checks.values()),
        "reference_only": True,
        "state_role": "PENDING_WITH_CONDITIONAL_DERIVATION",
        "factored_field": "actions[].planned_payload",
        "derivation_guard": "planned_payload == payload",
        "negative_control": "faulted payload != planned_payload keeps both",
        "legacy_checkpoint_bytes": legacy_bytes,
        "compact_checkpoint_bytes": compact_bytes,
        "bytes_saved_for_one_pending_action": legacy_bytes - compact_bytes,
        "compression_ratio": compact_bytes / legacy_bytes,
        "work_shape": {
            "added_transition_touches": 0,
            "added_runtime_materializations": 0,
            "checkpoint_compare_per_pending_action": 1,
            "restore_derivation": "existing a.get(planned_payload, payload)",
        },
        "timing": {
            "legacy_restore_us": round(legacy_restore_us, 3),
            "compact_restore_us": round(compact_restore_us, 3),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "checks": checks,
    }
    print("FOUNDRY_MATHEMATICAL_ADULT_PENDING_ACTION_FACTORIZATION_" + ("GREEN" if result["pass"] else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
