#!/usr/bin/env python3
"""Hostile proof for an offline-authored, numeric starting causal state."""
from __future__ import annotations
import copy
import hashlib
import json

from interchange import run_scenario
from reference_contract_1610 import (
    CONTACT_FRAME,
    CONTACT_LEXEME,
    CONTACT_QUERY,
    ReferenceMachineV1,
    ReferenceStateV1,
    RelationV1,
    Refuse,
    _canonical,
    authored_recipe_pool,
    digest,
)


def learned_state(offset: int = 0) -> ReferenceStateV1:
    rows = []
    identity = 1 + offset
    for concept, number, surface in (
        (0, 0, b" "),
        (10, 2, b"cats"),
        (11, 1, b"dog"),
        (20, 1, b"sees"),
    ):
        rows.append(RelationV1(
            identity, CONTACT_LEXEME,
            (1, concept, number, len(surface), *surface),
            101, (1000 + identity,), 1))
        identity += 1
    rows.append(RelationV1(
        identity, CONTACT_FRAME, (1, 1, 2, 3, 1, 0, 2),
        101, (1000 + identity,), 1))
    return ReferenceStateV1(relations=rows, next_identity=identity + 1)


def query_steps(right_concept: int = 11, right_number: int = 1):
    tree = [20, 1, 2, 10, 2, 0, right_concept, right_number, 0]
    return [
        [1, 9000, 1, 100, 900, 1, CONTACT_QUERY, 1, 1,
         1, 1, len(tree), 3, *tree],
        [2],
    ]


def reseal(document: dict) -> bytes:
    document["checksum"] = digest("foundry-checkpoint-v1", document["body"])
    return _canonical(document)


def refuses(blob: bytes) -> bool:
    try:
        ReferenceMachineV1.restore(blob, authored_recipe_pool())
    except Refuse:
        return True
    return False


def main() -> None:
    pool = authored_recipe_pool()
    machine = ReferenceMachineV1(learned_state(), pool)
    checkpoint = machine.checkpoint()
    result = run_scenario({"schema_version": 1, "steps": query_steps()}, checkpoint)
    action = result["events"][-1]["result"]

    replay = run_scenario({"schema_version": 1, "steps": query_steps()}, checkpoint)
    altered = run_scenario({"schema_version": 1, "steps": query_steps(10, 2)}, checkpoint)
    permuted_checkpoint = ReferenceMachineV1(learned_state(100), pool).checkpoint()
    permuted = run_scenario(
        {"schema_version": 1, "steps": query_steps()}, permuted_checkpoint)

    raw = json.loads(checkpoint)
    unknown = copy.deepcopy(raw)
    unknown["body"]["unknown"] = 1
    duplicate = copy.deepcopy(raw)
    duplicate["body"]["relations"].append(
        copy.deepcopy(duplicate["body"]["relations"][0]))
    literal = copy.deepcopy(raw)
    literal["body"]["relations"][0]["values"][0] = "surface"
    stale_next = copy.deepcopy(raw)
    stale_next["body"]["next_identity"] = 1

    checks = {
        "offline_state_to_surface": bytes(action["payload"]) == b"cats sees dog",
        "adapter_did_not_supply_surface": all(len(step) == 1 or step[0] != 1
                                              or step[6] == CONTACT_QUERY
                                              for step in query_steps()),
        "starting_checkpoint_bound": (
            result["starting_checkpoint_supplied"]
            and result["starting_checkpoint_sha256"] == hashlib.sha256(checkpoint).hexdigest()),
        "exact_replay": result == replay,
        "altered_input_divergence": (
            altered["state_hash"] != result["state_hash"]
            and bytes(altered["events"][-1]["result"]["payload"]) == b"cats sees cats"),
        "opaque_state_identity_permutation": (
            bytes(permuted["events"][-1]["result"]["payload"]) == bytes(action["payload"])
            and permuted["events"][-1]["result"]["frontier"] != action["frontier"]),
        "unknown_state_field_refused": refuses(reseal(unknown)),
        "duplicate_identity_refused": refuses(reseal(duplicate)),
        "literal_state_value_refused": refuses(reseal(literal)),
        "stale_next_identity_refused": refuses(reseal(stale_next)),
        "per_byte_ancestry": len(action["ancestry"]) == len(action["payload"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("FOUNDRY_STARTING_STATE_RED " + ",".join(failed))
    print(
        "FOUNDRY_STARTING_STATE_GREEN offline_authored=1 numeric_state=1 "
        "adapter_surface_bytes=0 exact_replay=1 altered_input_divergence=1 "
        "opaque_identity_permutation=1 tamper_refusal=1 per_byte_ancestry=1"
    )


if __name__ == "__main__":
    main()
