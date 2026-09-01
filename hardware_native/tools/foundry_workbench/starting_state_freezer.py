#!/usr/bin/env python3
"""Offline freezer for a complete typed numeric reference-state body.

This authoring tool has no runtime cognition authority. It validates and freezes
state before the separate interchange process receives authenticated input.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from reference_contract_1610 import (
    CHECKPOINT_VERSION,
    Refuse,
    ReferenceMachineV1,
    ReferenceStateV1,
    _canonical,
    authored_recipe_pool,
    authored_starting_state,
    digest,
)


SCHEMA_VERSION = 1
ALLOWED_FIELDS = {"schema_version", "state"}


def freeze_document(document: dict) -> tuple[bytes, dict]:
    if not isinstance(document, dict) or set(document) != ALLOWED_FIELDS:
        raise Refuse("starting_state:document_fields")
    if document["schema_version"] != SCHEMA_VERSION:
        raise Refuse("starting_state:version")
    body = document["state"]
    if not isinstance(body, dict) or set(body) != set(ReferenceStateV1.__dataclass_fields__):
        raise Refuse("starting_state:body_fields")
    pool = authored_recipe_pool()
    # This boundary freezes the authored vehicle, never a host-authored current
    # or future thought. Live relations, work, trace, actions, credit and
    # continuations can arise only through the resident runtime and its complete
    # checkpoint path.
    authored_body = ReferenceMachineV1(authored_starting_state(), pool)._checkpoint_body()
    if body != authored_body:
        raise Refuse("starting_state:resident_state_required")
    envelope = {
        "version": CHECKPOINT_VERSION,
        "pool_root": pool.root,
        "body": body,
        "checksum": digest("foundry-checkpoint-v1", body),
    }
    checkpoint = _canonical(envelope)
    machine = ReferenceMachineV1.restore(checkpoint, pool)
    # Re-serialization proves the accepted body has one canonical representation.
    canonical = machine.checkpoint()
    if canonical != checkpoint:
        raise Refuse("starting_state:noncanonical")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "reference_only": True,
        "adult_attached": False,
        "graph_flip": False,
        "experimental_ir": "ReferenceRecipeIrV1",
        "production_ir": "ResidentRecipeIrProgram.vcurrent",
        "translation_status": "UNDEFINED",
        "physical_direct_parity": "NOT_RUN/RED",
        "pool_root": pool.root,
        "state_hash": machine.state_hash(),
        "checkpoint_sha256": hashlib.sha256(canonical).hexdigest(),
        "relation_count": len(machine.state.relations),
        "query_count": len(machine.state.queries),
        "occurrence_count": len(machine.state.occurrences),
        "action_count": len(machine.state.actions),
    }
    return canonical, manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a complete numeric ReferenceStateV1 body")
    parser.add_argument("state_document")
    parser.add_argument("--checkpoint-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    args = parser.parse_args()
    document = json.loads(Path(args.state_document).read_text())
    checkpoint, manifest = freeze_document(document)
    Path(args.checkpoint_out).write_bytes(checkpoint)
    Path(args.manifest_out).write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    print(
        "FOUNDRY_STARTING_STATE_FROZEN "
        f"state_hash={manifest['state_hash']} "
        f"checkpoint_sha256={manifest['checkpoint_sha256']} "
        f"relations={manifest['relation_count']} actions={manifest['action_count']} "
        "reference_only=true adult_attached=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
