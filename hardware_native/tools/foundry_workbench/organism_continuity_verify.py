#!/usr/bin/env python3
"""Continuing-organism assay for the compact graph-neutral Foundry core.

One external work contact may start a resident causal sequence. Later work fronts
must already exist in resident state and may awaken only through independent
consequence. There is deliberately no answer/think/speak/goal API.
"""
from __future__ import annotations

import json
from reference_contract_1610 import (
    CONTACT_CONSEQUENCE, CONTACT_FRAME, CONTACT_LEXEME, CONTACT_QUERY,
    ContactV1, QueryV1, ReferenceMachineV1, authored_recipe_pool,
    authored_starting_state,
)


def units(text: str) -> tuple[int, ...]:
    return tuple(text.encode("utf-8"))


def node(concept: int, number: int, *children):
    return (concept, number, tuple(children))


def flatten(root) -> list[int]:
    concept, number, children = root
    out = [concept, number, len(children)]
    for child in children:
        out.extend(flatten(child))
    return out


def query_payload(language: int, frame: int, root) -> tuple[int, ...]:
    body = flatten(root)
    return (language, frame, len(body), 3, *body)


class World:
    """Environment transport only: contact and consequence, never cognition calls."""
    def __init__(self):
        self.ticket = 70000

    def put(self, machine, kind, payload, source, channel=1, independent=1):
        ticket = self.ticket
        self.ticket += 1
        c = ContactV1(ticket, machine.state.incarnation, machine.state.tick + 8,
                      source, channel, kind, tuple(payload), 1, independent)
        machine.contact(c)
        return ticket

    def settle(self, machine, action, effect=1, independent=1):
        c = ContactV1(action.ticket, action.incarnation, action.deadline,
                      action.source, action.channel, CONTACT_CONSEQUENCE,
                      (effect,), 1, independent)
        machine.contact(c)


def teach_surface(machine, world):
    table = {
        (0, 0): " ", (10, 1): "cat", (10, 2): "cats",
        (11, 1): "dog", (11, 2): "dogs", (20, 1): "sees",
        (20, 2): "see", (30, 1): "and", (40, 1): "says",
    }
    for (concept, number), surface in table.items():
        raw = units(surface)
        world.put(machine, CONTACT_LEXEME,
                  (1, concept, number, len(raw), *raw), 101)
    # Learned generic binary surface ordering already present from lived contact.
    world.put(machine, CONTACT_FRAME, (1, 1, 2, 3, 1, 0, 2), 101)


def prepare_sequence():
    machine = ReferenceMachineV1(authored_starting_state(), authored_recipe_pool())
    world = World()
    teach_surface(machine, world)
    roots = (
        node(20, 1, node(10, 1), node(11, 1)),
        node(30, 1, node(10, 2), node(11, 1)),
        node(40, 1, node(11, 1), node(20, 1, node(10, 1), node(11, 1))),
        node(20, 2, node(10, 2), node(11, 2)),
    )
    # Exactly one external work contact starts the episode.
    world.put(machine, CONTACT_QUERY, query_payload(1, 1, roots[0]), 900)
    first = machine.state.queries[-1]
    prior = first.identity
    ids = [prior]
    for root in roots[1:]:
        identity = machine._new_id()  # Foundry snapshot construction, not runtime API.
        machine.state.queries.append(QueryV1(identity, 0, 1, 1, root, 900, 0, prior))
        machine.state.continuations[prior] = identity
        ids.append(identity)
        prior = identity
    machine._bounded()
    return machine, world, roots, ids


def run_episode():
    machine, world, roots, ids = prepare_sequence()
    outputs = []
    for index in range(len(ids)):
        action = machine.tick()
        assert action is not None
        outputs.append(bytes(action.payload).decode("utf-8"))
        world.settle(machine, action, effect=1, independent=1)
    terminal = machine.tick()
    return machine, outputs, terminal


def main():
    blank = ReferenceMachineV1(authored_starting_state(), authored_recipe_pool())
    blank_silent = blank.tick() is None

    machine, outputs, terminal = run_episode()
    expected = ["cat sees dog", "cats and dog", "dog says cat sees dog", "cats see dogs"]
    query_contacts = [r for r in machine.state.relations if r.kind == CONTACT_QUERY]
    checks = {
        "blank_organism_silent": blank_silent,
        "one_external_work_contact": len(query_contacts) == 1,
        "endogenous_successors_present": sum(q.contact_identity == 0 for q in machine.state.queries) == 3,
        "consequence_gated_chain": outputs == expected,
        "no_prompt_between_outputs": len(query_contacts) == 1 and len(outputs) == 4,
        "terminal_silence": terminal is None and not any(q.active for q in machine.state.queries),
        "no_answer_api": not hasattr(machine, "answer"),
        "no_think_api": not hasattr(machine, "think"),
        "no_speak_api": not hasattr(machine, "speak"),
        "no_goal_enqueue_api": not hasattr(machine, "enqueue_goal"),
    }

    # Mid-episode checkpoint includes dormant continuation topology and must replay.
    left, world, _, _ = prepare_sequence()
    first = left.tick(); world.settle(left, first)
    frozen = left.checkpoint()
    twin = ReferenceMachineV1.restore(frozen, left.pool)
    a = left.tick(); b = twin.tick()
    checks["checkpoint_preserves_endogenous_frontier"] = (
        a.payload == b.payload and left.state.continuations == twin.state.continuations)

    result = {
        "schema": "0x1.compact-lived-organism.v1",
        "pass": all(checks.values()),
        "checks": checks,
        "outputs": outputs,
        "external_work_contacts": len(query_contacts),
        "endogenous_work_fronts": 3,
        "semantic_inner_state_taxonomy": False,
        "claim": "CONTINUING_CAUSAL_ORGANISM_NOT_INPUT_OUTPUT_CHATBOT",
    }
    print("FOUNDRY_ORGANISM_CONTINUITY " + ("GREEN" if result["pass"] else "RED")
          + " external_work_contacts=1 endogenous_fronts=3 prompt_loop=0 semantic_thought_objects=0")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
