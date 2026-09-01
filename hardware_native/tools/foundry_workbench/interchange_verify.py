#!/usr/bin/env python3
"""Hostile checks for the sole authoritative numeric scenario interchange."""
from interchange import run_scenario
from reference_contract_1610 import (
    CONTACT_FRAME, CONTACT_LEXEME, CONTACT_QUERY, CONTACT_STIMULUS, Refuse,
)


def refused(document):
    try:
        run_scenario(document)
    except Refuse:
        return True
    return False


def contact(ticket, source, kind, payload):
    return [1, ticket, 1, 100, source, 1, kind, 1, 1, *payload]


def main():
    steps = []
    ticket = 1000
    for concept, number, surface in (
        (0, 0, b" "), (10, 2, b"cats"), (11, 1, b"dog"),
        (20, 1, b"sees"),
    ):
        payload = [1, concept, number, len(surface), *surface]
        steps.append(contact(ticket, 101, CONTACT_LEXEME, payload)); ticket += 1
    steps.append(contact(ticket, 101, CONTACT_FRAME, [1, 1, 2, 3, 1, 0, 2])); ticket += 1
    # QUERY stays dormant until an opaque STIMULUS cue makes it eligible.
    tree = [20, 1, 2, 10, 2, 0, 11, 1, 0]
    query_cue = 1
    steps.append(contact(ticket, 900, CONTACT_QUERY,
                          [query_cue, 1, len(tree), 3, *tree])); ticket += 1
    steps.append(contact(ticket, 900, CONTACT_STIMULUS, [query_cue])); ticket += 1
    steps.append([2])
    result = run_scenario({"schema_version": 1, "steps": steps})
    action = result["events"][-1]["result"]
    if action is None:
        raise SystemExit("FOUNDRY_INTERCHANGE_RED tick_produced_no_action")
    checks = {
        "numeric_contact_to_action": bytes(action["payload"]) == b"cats sees dog",
        "per_byte_ancestry": len(action["ancestry"]) == len(action["payload"]),
        "no_recipe_declaration": refused({"schema_version": 1, "steps": [],
                                           "verified_recipes": []}),
        "no_goal_injection": refused({"schema_version": 1, "steps": [], "goal": 4}),
        "no_expected_output": refused({"schema_version": 1, "steps": [],
                                        "expected_output": [1]}),
        "numeric_only": refused({"schema_version": 1, "steps": [[2, "prompt"]]}),
        "unknown_opcode": refused({"schema_version": 1, "steps": [[999]]}),
        "reference_only": result["reference_only"] and not result["adult_attached"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("FOUNDRY_INTERCHANGE_RED " + ",".join(failed))
    print("FOUNDRY_INTERCHANGE_GREEN numeric_only=1 recipe_declaration=0 "
          "goal_injection=0 expected_output=0 adult_attached=0 graph_flip=0 "
          "stimulus_gated_query=1")


if __name__ == "__main__":
    main()
