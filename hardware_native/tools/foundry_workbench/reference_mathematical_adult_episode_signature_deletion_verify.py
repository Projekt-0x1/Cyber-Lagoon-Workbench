#!/usr/bin/env python3
"""Fast A/B verifier for deleting derived episodic population signatures.

Episode context/atoms/source/tick are lived evidence. The sparse population signature
used to nominate retrieval is an exact deterministic function of context + atoms and
the fixed PopulationBank topology. Persisting both stores one evidence distinction
twice. This verifier removes only the signature and keeps the episode itself causal.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_organism_v2 import ActionV2, CONTACT_CONSEQUENCE, ReferenceOrganismV2  # noqa: E402
from reference_organism_v2_verify import partner, scene, surface, train, u  # noqa: E402
from reference_population_v1 import PopulationSpecV1  # noqa: E402

# This assay challenges derived episodic state, not population capacity.
# The factory's population-delta checkpoint assay retains the 32,768-site control.
SPEC = PopulationSpecV1(1024, 2, 4, 42, 8)
PARTIAL = (101, 201, 301, 0)
TARGET = (101, 201, 301, 401)
ALTERNATE = (101, 201, 301, 402)


def wire_bytes(state):
    return len(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())


def build():
    o = ReferenceOrganismV2(SPEC)
    _name, context, _mapping = train(o)
    return o, context


def add_legacy_signatures(o, checkpoint):
    out = copy.deepcopy(checkpoint)
    for row in out["episodes"]:
        row["signature"] = list(o._scene_signature(int(row["context"]), tuple(row["atoms"])))
    return out


def unique_future(checkpoint, context):
    o = ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    partner(o, 9100)
    scene(o, context, PARTIAL, 9100)
    action = o.tick()
    retrieval = dict(o.last_retrieval)
    completed = 0 if o.current_scene is None else int(o.current_scene.completed_from_episode)
    settlement = None
    if isinstance(action, ActionV2):
        settlement = o.contact(CONTACT_CONSEQUENCE, (action.ticket, 1), 9100, True, True)
    return {
        "public": None if action is None else bytes(action.payload),
        "completed_from_episode": completed,
        "retrieval_status": int(retrieval.get("status", 0)),
        "retrieval_winner": int(retrieval.get("winner", 0)),
        "settlement_credit": None if settlement is None else int(settlement.get("credit", 0)),
        "settlement_revisions": None if settlement is None else int(settlement.get("revisions", 0)),
        "post_digest": o.digest(),
    }


def ambiguous_future(checkpoint, context):
    o = ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    scene(o, context, ALTERNATE, 9200)
    surface(o, "the careful engineer tests the valve.", 3003)
    scene(o, context, PARTIAL, 9300)
    action = o.tick()
    return {
        "public": None if action is None else bytes(action.payload),
        "retrieval_status": int(o.last_retrieval.get("status", 0)),
        "alternatives": int(o.last_retrieval.get("alternatives", 0)),
        "post_digest": o.digest(),
    }


def main():
    started = time.perf_counter()
    o, context = build()
    compact = o.checkpoint()
    legacy = add_legacy_signatures(o, compact)
    episode_count = len(compact["episodes"])
    signature_ints = sum(len(row["signature"]) for row in legacy["episodes"])

    restore_started = time.perf_counter_ns()
    legacy_restored = ReferenceOrganismV2.restore(copy.deepcopy(legacy))
    legacy_restore_us = (time.perf_counter_ns() - restore_started) / 1000.0
    restore_started = time.perf_counter_ns()
    compact_restored = ReferenceOrganismV2.restore(copy.deepcopy(compact))
    compact_restore_us = (time.perf_counter_ns() - restore_started) / 1000.0

    # Derived legacy bytes must have no semantic authority. Even a forged legacy
    # signature is ignored and canonically rebuilt from the retained evidence.
    forged_legacy = copy.deepcopy(legacy)
    forged_legacy["episodes"][0]["signature"] = [1, 2, 3]
    forged = ReferenceOrganismV2.restore(forged_legacy)

    unique_legacy = unique_future(legacy, context)
    unique_compact = unique_future(compact, context)
    ambiguous_legacy = ambiguous_future(legacy, context)
    ambiguous_compact = ambiguous_future(compact, context)

    # Negative control: delete the lived target episode itself. The partial cue must
    # lose the future that depended on that evidence.
    evidence_cut = copy.deepcopy(compact)
    evidence_cut["episodes"] = [row for row in evidence_cut["episodes"] if tuple(row["atoms"]) != TARGET]
    cut_future = unique_future(evidence_cut, context)

    legacy_bytes = wire_bytes(legacy)
    compact_bytes = wire_bytes(compact)
    checks = {
        "compact_checkpoint_deletes_all_episode_signatures": all("signature" not in row for row in compact["episodes"]),
        "legacy_representation_contains_signatures": all("signature" in row for row in legacy["episodes"]),
        "restored_causal_state_is_identical": legacy_restored.digest() == compact_restored.digest(),
        "forged_legacy_signature_has_no_authority": forged.digest() == compact_restored.digest(),
        "unique_retrieval_future_is_identical": unique_legacy == unique_compact,
        "unique_retrieval_still_completes": unique_compact["public"] == b"the careful engineer tests the sensor." and unique_compact["completed_from_episode"] > 0,
        "ambiguity_future_is_identical": ambiguous_legacy == ambiguous_compact,
        "ambiguity_remains_silent": ambiguous_compact["public"] is None and ambiguous_compact["retrieval_status"] == 2 and ambiguous_compact["alternatives"] >= 2,
        "episode_evidence_negative_control_is_rejected": cut_future != unique_compact,
        "persistent_representation_is_smaller": compact_bytes < legacy_bytes,
        "bounded_seconds_lane": time.perf_counter() - started < 5.0,
    }
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-episode-signature-deletion.v1",
        "pass": all(checks.values()),
        "reference_only": True,
        "state_role": "EVIDENCE_WITH_DERIVED_RETRIEVAL_COORDINATE",
        "deleted_persistent_field": "episodes[].signature",
        "retained_evidence": ["identity", "scene_identity", "context", "atoms", "source", "tick"],
        "derivation": "population.signature((context,*atoms))",
        "episode_count": episode_count,
        "derived_signature_ints_removed": signature_ints,
        "legacy_checkpoint_bytes": legacy_bytes,
        "compact_checkpoint_bytes": compact_bytes,
        "bytes_saved": legacy_bytes - compact_bytes,
        "bytes_saved_per_episode": round((legacy_bytes - compact_bytes) / max(1, episode_count), 3),
        "compression_ratio": compact_bytes / legacy_bytes,
        "work_shape": {
            "interactive_transition_delta": 0,
            "restore_signature_rebuilds": episode_count,
            "persistent_coordinate_ints_deleted": signature_ints,
        },
        "timing": {
            "legacy_restore_us": round(legacy_restore_us, 3),
            "compact_restore_us": round(compact_restore_us, 3),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "checks": checks,
    }
    print("FOUNDRY_MATHEMATICAL_ADULT_EPISODE_SIGNATURE_DELETION_" + ("GREEN" if result["pass"] else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
