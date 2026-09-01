#!/usr/bin/env python3
"""Hostile controls for the offline starting-state freezer."""
from __future__ import annotations
import copy
import hashlib

from reference_contract_1610 import (
    Refuse, ReferenceMachineV1, authored_recipe_pool, authored_starting_state,
)
from starting_state_freezer import freeze_document


def refused(document) -> bool:
    try:
        freeze_document(document)
    except Refuse:
        return True
    return False


def main() -> None:
    pool = authored_recipe_pool()
    machine = ReferenceMachineV1(authored_starting_state(), pool)
    body = machine._checkpoint_body()
    document = {"schema_version": 1, "state": body}
    checkpoint, manifest = freeze_document(document)

    literal = copy.deepcopy(document)
    literal["state"]["trace"].append(["prompt"])
    stale = copy.deepcopy(document)
    stale["state"]["tick"] = 1
    active_work = copy.deepcopy(document)
    active_work["state"]["queries"].append({
        "identity": 1, "contact_identity": 0, "language": 1, "frame": 1,
        "root": [1, 1, []], "source": 1, "active": 1,
        "parent_identity": 0,
    })
    active_work["state"]["next_identity"] = 2

    checks = {
        "canonical_roundtrip": checkpoint == machine.checkpoint(),
        "manifest_binds_checkpoint": (
            manifest["checkpoint_sha256"] == hashlib.sha256(checkpoint).hexdigest()
            and manifest["state_hash"] == machine.state_hash()),
        "manifest_has_no_state_or_surface": (
            "state" not in manifest and "payload" not in manifest
            and "surface" not in manifest),
        "recipe_declaration_refused": refused({**document, "recipe": []}),
        "expected_output_refused": refused({**document, "expected_output": []}),
        "literal_value_refused": refused(literal),
        "stale_identity_refused": refused(stale),
        "host_authored_active_work_refused": refused(active_work),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("FOUNDRY_STARTING_STATE_FREEZER_RED " + ",".join(failed))
    print(
        "FOUNDRY_STARTING_STATE_FREEZER_GREEN authored_vehicle=1 canonical=1 "
        "pool_bound=1 manifest_surface=0 recipe_declaration=0 expected_output=0 "
        "literal_value=0 host_current_thought=0 tamper_refusal=1"
    )


if __name__ == "__main__":
    main()
