#!/usr/bin/env python3
"""Strict numeric scenario transport for the graph-neutral reference machine.

The interchange transports authenticated contacts, ticks, and declared causal
interventions. It cannot declare Recipes verified, enqueue goals, provide
expected output, or carry a prompt/semantic summary.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from reference_contract_1610 import (
    ContactV1,
    Refuse,
    ReferenceMachineV1,
    authored_recipe_pool,
    authored_starting_state,
)


SCHEMA_VERSION = 1
MAX_STEPS = 4096
MAX_STEP_WIDTH = 512

STEP_CONTACT = 1
STEP_TICK = 2
STEP_WITHDRAW_SOURCE = 3
STEP_CHECKPOINT_RESTORE = 4
STEP_LESION_COMPACT = 5
STEP_DEOPTIMIZE = 6
STEP_SET_WORK_LIMIT = 7

ALLOWED_DOCUMENT_FIELDS = {"schema_version", "steps"}
FORBIDDEN_DOCUMENT_FIELDS = {
    "verified_recipes", "recipe", "program", "goal", "expected",
    "expected_output", "answer", "prompt", "semantic", "domain", "motor",
    "route", "bind", "emit", "channel", "surface_bytes", "assertions",
}


def _numeric(value, path="$", *, depth=0) -> None:
    if depth > 8:
        raise Refuse(f"interchange:depth:{path}")
    if isinstance(value, bool) or not isinstance(value, (int, list, tuple)):
        raise Refuse(f"interchange:numeric_only:{path}")
    if isinstance(value, int):
        if not -(1 << 63) <= value < (1 << 63):
            raise Refuse(f"interchange:integer:{path}")
        return
    if len(value) > MAX_STEP_WIDTH:
        raise Refuse(f"interchange:width:{path}")
    for index, item in enumerate(value):
        _numeric(item, f"{path}[{index}]", depth=depth + 1)


def validate_document(document: dict) -> None:
    if not isinstance(document, dict):
        raise Refuse("interchange:document")
    fields = set(document)
    if fields & FORBIDDEN_DOCUMENT_FIELDS:
        raise Refuse("interchange:forbidden_field")
    if fields != ALLOWED_DOCUMENT_FIELDS:
        raise Refuse("interchange:fields")
    if document["schema_version"] != SCHEMA_VERSION:
        raise Refuse("interchange:version")
    steps = document["steps"]
    if not isinstance(steps, list) or len(steps) > MAX_STEPS:
        raise Refuse("interchange:step_bound")
    for index, step in enumerate(steps):
        _numeric(step, f"$.steps[{index}]")
        if not step:
            raise Refuse("interchange:empty_step")


def _action_record(action):
    if action is None:
        return None
    return {
        "ticket": action.ticket,
        "incarnation": action.incarnation,
        "born_tick": action.born_tick,
        "deadline": action.deadline,
        "channel": action.channel,
        "source": action.source,
        "occurrence_identity": action.occurrence_identity,
        "payload": list(action.payload),
        "constituent_root": action.constituent_root,
        "frontier": list(action.frontier),
        "ancestry": [asdict(item) for item in action.ancestry],
    }


def run_scenario(document: dict, checkpoint: bytes | None = None) -> dict:
    validate_document(document)
    pool = authored_recipe_pool()
    starting_checkpoint_sha256 = (hashlib.sha256(checkpoint).hexdigest()
                                  if checkpoint is not None else None)
    machine = (ReferenceMachineV1.restore(checkpoint, pool) if checkpoint is not None
               else ReferenceMachineV1(authored_starting_state(), pool))
    events = []
    for index, step in enumerate(document["steps"]):
        before = machine.state_hash()
        opcode = step[0]
        if opcode == STEP_CONTACT:
            if len(step) < 10:
                raise Refuse("interchange:contact_shape")
            ticket, incarnation, deadline, source, channel, kind, authenticated, independent = step[1:9]
            payload = tuple(step[9:])
            # Membrane seals session/ingress/auth. Scenario bytes cannot assert a
            # valid receipt tag — that would be host authority laundering.
            raw = ContactV1(
                ticket, incarnation, deadline, source, channel, kind, payload,
                authenticated, independent)
            result = machine.contact(machine.seal_contact(raw))
        elif opcode == STEP_TICK:
            if len(step) != 1:
                raise Refuse("interchange:tick_shape")
            result = _action_record(machine.tick())
        elif opcode == STEP_WITHDRAW_SOURCE:
            if len(step) != 2:
                raise Refuse("interchange:withdraw_shape")
            machine.withdraw_source(step[1]); result = None
        elif opcode == STEP_CHECKPOINT_RESTORE:
            if len(step) != 1:
                raise Refuse("interchange:checkpoint_shape")
            machine = ReferenceMachineV1.restore(machine.checkpoint(), pool); result = None
        elif opcode == STEP_LESION_COMPACT:
            if len(step) != 2:
                raise Refuse("interchange:lesion_shape")
            machine.lesion_compact(step[1]); result = None
        elif opcode == STEP_DEOPTIMIZE:
            if len(step) != 2:
                raise Refuse("interchange:deoptimize_shape")
            machine.deoptimize(step[1]); result = None
        elif opcode == STEP_SET_WORK_LIMIT:
            if len(step) != 2 or not 0 <= step[1] <= 4096:
                raise Refuse("interchange:work_limit")
            machine.state.work_limit = step[1]; machine._bounded(); result = None
        else:
            raise Refuse("interchange:opcode")
        events.append({"index": index, "opcode": opcode, "before": before,
                       "after": machine.state_hash(), "result": result})
    return {
        "schema_version": SCHEMA_VERSION,
        "reference_only": True,
        "adult_attached": False,
        "graph_flip": False,
        "physical_direct_parity": "NOT_RUN/RED",
        "starting_checkpoint_supplied": checkpoint is not None,
        "starting_checkpoint_sha256": starting_checkpoint_sha256,
        "events": events,
        "state_hash": machine.state_hash(),
        "checkpoint_sha256": hashlib.sha256(machine.checkpoint()).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a strict numeric Foundry scenario")
    parser.add_argument("scenario")
    parser.add_argument("--checkpoint")
    parser.add_argument("--out")
    args = parser.parse_args()
    document = json.loads(Path(args.scenario).read_text())
    checkpoint = Path(args.checkpoint).read_bytes() if args.checkpoint else None
    result = run_scenario(document, checkpoint)
    text = json.dumps(result, sort_keys=True, separators=(",", ":"))
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
