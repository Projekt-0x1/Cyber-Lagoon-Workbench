#!/usr/bin/env python3
"""Fast falsifier for mathematical-Adult state minimization.

This does not claim a new Adult implementation.  It defines when two persistent
representations are allowed to be treated as the same individual state: future
public behaviour *and* future causal/update authority must remain identical
under the declared intervention battery.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

A, B, C = 4101, 4102, 4103
X, Y = 101, 202
WORLD = 9001
# Population quantity is a nuisance variable for this state-equivalence battery.
# A separate factory assay retains the 32,768-site checkpoint scale control.
SPEC = PopulationSpecV1(1024, 2, 4, 42, 8)


def setup(o):
    o.contact(CONTACT_WORLD_STATE, (11,), WORLD, True, True)
    o.contact(CONTACT_BODY_TARGET, (99,), 8001, True, True)
    o.contact(CONTACT_AFFORDANCES, (X, Y), 8002, True, True)


def claim(o, action, source):
    return o.contact(CONTACT_SOURCE_ASSERTION, (action,), source, True, True)


def settle(o, action, effect=1):
    return o.contact(CONTACT_MOTOR_CONSEQUENCE,
                     (action.ticket, effect, 1, 99), WORLD, True, True)


def public_action(action):
    if isinstance(action, MotorActionV2):
        return ("motor", int(action.action_id))
    if isinstance(action, ActionV2):
        return ("language", bytes(action.payload))
    return None


def causal_action(action):
    if isinstance(action, MotorActionV2):
        return (
            "motor", int(action.action_id),
            tuple(map(int, action.source_assertion_ids)),
            int(action.source_counterfactual_action),
            int(action.prospective_recipe),
        )
    if isinstance(action, ActionV2):
        return (
            "language", bytes(action.payload), int(action.template_identity),
            tuple(map(int, action.lexical_identities)),
            tuple(map(int, action.contributors)),
        )
    return None


def state_projection(checkpoint):
    """Candidate compression projection: source repetition count is observer-only.

    This projection is deliberately tiny.  The assay does not assume other fields
    are redundant; it asks whether deleting this one candidate distinction survives
    future interventions.
    """
    out = copy.deepcopy(checkpoint)
    for row in out.get("source_assertions", ()):
        row.pop("repetitions", None)
    return out


def run_future(checkpoint, scenario):
    o = ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    settlement = None
    if scenario == "conflict":
        claim(o, Y, B)
        action = o.tick()
    elif scenario == "independent_support":
        claim(o, Y, B)
        claim(o, X, C)
        action = o.tick()
        if isinstance(action, MotorActionV2):
            settlement = settle(o, action, 1)
    elif scenario == "withdraw_original":
        o.contact(CONTACT_WITHDRAW_SOURCE, (A,), 9901, True, True)
        claim(o, Y, B)
        action = o.tick()
    elif scenario == "repeat_then_conflict":
        claim(o, X, A)
        claim(o, Y, B)
        action = o.tick()
    else:
        raise ValueError(scenario)
    return {
        "public": public_action(action),
        "causal": causal_action(action),
        "information_need": tuple(o.information_need),
        "source_updates": None if settlement is None else int(settlement.get("source_updates", 0)),
        "projection": state_projection(o.checkpoint()),
    }


def make_repeated_checkpoint():
    o = ReferenceOrganismV2(SPEC)
    setup(o)
    identity = claim(o, X, A)
    for _ in range(19):
        assert claim(o, X, A) == identity
    return o.checkpoint()


def make_calibrating_source_state(source):
    o = ReferenceOrganismV2(SPEC)
    o.contact(CONTACT_WORLD_STATE, (21,), WORLD, True, True)
    o.contact(CONTACT_BODY_TARGET, (501,), 8001, True, True)
    o.contact(CONTACT_AFFORDANCES, (X, Y), 8002, True, True)
    baseline = o._exploration_candidate()
    chosen = Y if baseline == X else X
    claim(o, chosen, source)
    return o, chosen


def main():
    started = time.perf_counter()
    checks = {}

    donor = make_repeated_checkpoint()
    compact = copy.deepcopy(donor)
    assert compact["source_assertions"][0]["repetitions"] == 20
    compact["source_assertions"][0]["repetitions"] = 1

    checks["candidate_representation_actually_differs"] = donor != compact
    checks["candidate_projection_merges_only_repetition_count"] = (
        state_projection(donor) == state_projection(compact)
    )

    scenarios = (
        "conflict", "independent_support", "withdraw_original",
        "repeat_then_conflict",
    )
    futures_equal = True
    for scenario in scenarios:
        left = run_future(donor, scenario)
        right = run_future(compact, scenario)
        if left != right:
            futures_equal = False
            break
    checks["deleted_repetition_count_is_intervention_equivalent"] = futures_equal

    # Negative control: two histories can yield the same outward motor action while
    # still being different Adult states because consequence credit belongs to a
    # different source lineage.
    source_a, chosen_a = make_calibrating_source_state(A)
    source_c, chosen_c = make_calibrating_source_state(C)
    action_a = source_a.tick(); action_c = source_c.tick()
    same_public = (
        chosen_a == chosen_c
        and public_action(action_a) == public_action(action_c) == ("motor", chosen_a)
    )
    settle_a = source_a.contact(
        CONTACT_MOTOR_CONSEQUENCE, (action_a.ticket, 1, 1, 501), WORLD, True, True)
    settle_c = source_c.contact(
        CONTACT_MOTOR_CONSEQUENCE, (action_c.ticket, 1, 1, 501), WORLD, True, True)
    checks["same_public_action_does_not_imply_same_adult_state"] = same_public
    checks["source_lineage_changes_future_update_authority"] = (
        causal_action(action_a) != causal_action(action_c)
        and int(settle_a.get("source_updates", 0)) == 1
        and int(settle_c.get("source_updates", 0)) == 1
        and source_a.source_calibrations != source_c.source_calibrations
    )

    # The operator-level interpretation is stateful: the same current input on two
    # causally distinct states is permitted to produce the same public action while
    # their lawful next-state updates differ.
    checks["stateful_operator_not_stateless_input_output_function"] = (
        same_public and source_a.digest() != source_c.digest()
    )
    checks["bounded_fast_path"] = time.perf_counter() - started < 5.0

    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise SystemExit("FOUNDRY_MATHEMATICAL_ADULT_STATE_EQUIVALENCE_RED " + ",".join(failed))

    raw_bytes = len(json.dumps(donor, sort_keys=True, separators=(",", ":")))
    compact_bytes = len(json.dumps(compact, sort_keys=True, separators=(",", ":")))
    result = {
        "contract": "FOUNDRY_MATHEMATICAL_ADULT_STATE_EQUIVALENCE_GREEN",
        "reference_only": True,
        "whole_adult_semantics": "STATEFUL_OPERATOR",
        "equivalence": "FUTURE_BEHAVIOR_PLUS_UPDATE_AUTHORITY",
        "candidate_deleted_field": "source_assertion.repetitions",
        "intervention_scenarios": list(scenarios),
        "raw_checkpoint_bytes": raw_bytes,
        "candidate_checkpoint_bytes": compact_bytes,
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print(result["contract"])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
