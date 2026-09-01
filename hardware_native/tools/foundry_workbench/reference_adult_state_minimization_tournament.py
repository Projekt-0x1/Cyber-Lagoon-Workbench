#!/usr/bin/env python3
"""Fast practical A/B tournament for compact Mathematical Adult state.

This is deliberately representation-first.  It starts from a real
ReferenceOrganismV2 checkpoint, applies one candidate persistent-state deletion at
a time, and accepts a deletion only when all declared future interventions preserve
both public behaviour and causal/update authority.

The goal is not to prove that the current candidate compact state is optimal.  The
goal is to make state minimization a cheap executable search instead of an
architecture argument.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_mathematical_adult_state_equivalence_verify import (  # noqa: E402
    A, B, C, X, Y, WORLD, SPEC,
    claim, settle, setup, public_action, causal_action,
)
from reference_organism_v2 import (  # noqa: E402
    CONTACT_WITHDRAW_SOURCE, MotorActionV2, ReferenceOrganismV2,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    apply: object


def _wire_bytes(state):
    return len(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())


def _delete_top(name):
    def apply(state):
        out = copy.deepcopy(state)
        out.pop(name, None)
        return out
    return apply


def _source_repetitions_to_one(state):
    out = copy.deepcopy(state)
    for row in out.get("source_assertions", ()):
        row["repetitions"] = 1
    return out


def _drop_source_repetitions(state):
    out = copy.deepcopy(state)
    for row in out.get("source_assertions", ()):
        row.pop("repetitions", None)
    return out


def _canonicalize_empty_pending_repair(state):
    out = copy.deepcopy(state)
    if out.get("pending_repair") is None:
        out.pop("pending_repair", None)
    return out


def _canonicalize_no_output_fault(state):
    out = copy.deepcopy(state)
    if int(out.get("output_fault_offset", -1)) == -1 and int(out.get("output_fault_value", 0)) == 0:
        out.pop("output_fault_offset", None)
        out.pop("output_fault_value", None)
    return out


def _canonicalize_empty_action_commitments(state):
    out = copy.deepcopy(state)
    if not out.get("action_commitments"):
        out.pop("action_commitments", None)
    return out


def _restore(state):
    try:
        return ReferenceOrganismV2.restore(copy.deepcopy(state))
    except Exception as exc:  # fail closed: non-restorable is not equivalent
        return exc


def _future(checkpoint, scenario):
    restored = _restore(checkpoint)
    if isinstance(restored, Exception):
        return {"restore_error": type(restored).__name__, "message": str(restored)}
    o = restored
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
    elif scenario == "quiet_tick":
        action = o.tick()
    else:
        raise ValueError(scenario)
    return {
        "public": public_action(action),
        "causal": causal_action(action),
        "information_need": tuple(o.information_need),
        "source_updates": None if settlement is None else int(settlement.get("source_updates", 0)),
        "source_calibrations": tuple(
            (int(r.source), int(r.context), int(r.support), int(r.counter), int(r.revision), bool(r.active))
            for r in o.source_calibrations
        ),
        "withdrawn_sources": tuple(sorted(map(int, o.withdrawn_sources))),
        "selection_revision_rows": int(o._selection_revisions.row_count),
        "somatic_revision_rows": int(o._somatic_revisions.row_count),
        "world_revision_rows": int(o._world_revisions.row_count),
    }


def _equivalent(left, right, scenarios):
    traces = []
    for scenario in scenarios:
        a = _future(left, scenario)
        b = _future(right, scenario)
        same = a == b
        traces.append({"scenario": scenario, "equal": same})
        if not same:
            return False, traces
    return True, traces


def _build_donor():
    o = ReferenceOrganismV2(SPEC)
    setup(o)
    identity = claim(o, X, A)
    # Deliberately create repetition so the tournament has one known removable
    # representation distinction and a negative control for careless deletion.
    for _ in range(19):
        assert claim(o, X, A) == identity
    return o.checkpoint()


def main():
    started = time.perf_counter()
    donor = _build_donor()
    scenarios = (
        "quiet_tick", "conflict", "independent_support",
        "withdraw_original", "repeat_then_conflict",
    )

    candidates = (
        Candidate("source_repetitions_to_one", _source_repetitions_to_one),
        Candidate("source_repetitions_implicit_one", _drop_source_repetitions),
        Candidate("drop_last_retrieval", _delete_top("last_retrieval")),
        Candidate("drop_exploration_trials", _delete_top("exploration_trials")),
        Candidate("drop_information_need", _delete_top("information_need")),
        Candidate("drop_shared_episode_relations", _delete_top("shared_episode_relations")),
        Candidate("canonicalize_empty_pending_repair", _canonicalize_empty_pending_repair),
        Candidate("canonicalize_no_output_fault", _canonicalize_no_output_fault),
        Candidate("canonicalize_empty_action_commitments", _canonicalize_empty_action_commitments),
    )

    rows = []
    accepted_state = copy.deepcopy(donor)
    for candidate in candidates:
        before = _wire_bytes(accepted_state)
        proposed = candidate.apply(accepted_state)
        after = _wire_bytes(proposed)
        differs = proposed != accepted_state
        if not differs:
            rows.append({
                "candidate": candidate.name,
                "status": "NOOP",
                "bytes_before": before,
                "bytes_after": after,
                "bytes_saved": 0,
                "traces": [],
            })
            continue
        same, traces = _equivalent(accepted_state, proposed, scenarios)
        status = "ACCEPT" if same and after <= before else "REJECT"
        rows.append({
            "candidate": candidate.name,
            "status": status,
            "bytes_before": before,
            "bytes_after": after,
            "bytes_saved": before - after,
            "traces": traces,
        })
        if status == "ACCEPT":
            accepted_state = proposed

    donor_bytes = _wire_bytes(donor)
    compact_bytes = _wire_bytes(accepted_state)
    final_equal, final_traces = _equivalent(donor, accepted_state, scenarios)
    accepted = [r["candidate"] for r in rows if r["status"] == "ACCEPT"]
    rejected = [r["candidate"] for r in rows if r["status"] == "REJECT"]
    checks = {
        "final_compact_state_intervention_equivalent": final_equal,
        "no_candidate_can_win_by_growing_state": all(
            r["bytes_after"] <= r["bytes_before"] for r in rows if r["status"] == "ACCEPT"
        ),
        "known_repetition_representation_is_challenged": any(
            r["candidate"].startswith("source_repetitions") and r["status"] == "ACCEPT"
            for r in rows
        ),
        "tournament_contains_negative_candidates": bool(rejected),
        "bounded_fast_path": time.perf_counter() - started < 5.0,
    }
    result = {
        "schema": "0x1.reference-adult-state-minimization-tournament.v1",
        "pass": all(checks.values()),
        "reference_only": True,
        "purpose": "REPRESENTATION_PORT_A_B_NOT_CAPABILITY_PROMOTION",
        "equivalence": "PUBLIC_BEHAVIOUR_PLUS_CAUSAL_UPDATE_AUTHORITY",
        "scenarios": list(scenarios),
        "donor_checkpoint_bytes": donor_bytes,
        "compact_checkpoint_bytes": compact_bytes,
        "bytes_saved": donor_bytes - compact_bytes,
        "compression_ratio": compact_bytes / donor_bytes,
        "accepted": accepted,
        "rejected": rejected,
        "rows": rows,
        "final_traces": final_traces,
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print("FOUNDRY_ADULT_STATE_MINIMIZATION_" + ("GREEN" if result["pass"] else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
