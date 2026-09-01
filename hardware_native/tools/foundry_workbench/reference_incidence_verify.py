#!/usr/bin/env python3
"""White-box touched-work guard for compact reference relation lookup."""
from __future__ import annotations
import json
from reference_contract_1610 import (
    CONTACT_LEXEME, ReferenceMachineV1, RelationV1, Refuse,
    authored_recipe_pool, authored_starting_state,
)


def main():
    state = authored_starting_state()
    # Dense dormant vocabulary-like relation population. Only one exact prefix
    # should be touched by the execution index.
    for i in range(450):
        state.relations.append(RelationV1(i + 1, CONTACT_LEXEME,
            (1, 10000 + i, 1, 1, 65 + (i % 26)), 100 + (i % 7), (900000 + i,)))
    state.next_identity = 451
    target = state.relations[317]
    machine = ReferenceMachineV1(state, authored_recipe_pool())
    picked = machine._select_unique(CONTACT_LEXEME, 1, target.values[1], 1)
    touches_one = machine.last_lookup_touches

    # Exact ambiguity remains fail-closed even through the derived index.
    duplicate = RelationV1(9999, CONTACT_LEXEME, target.values, 777, (999999,))
    machine.state.relations.append(duplicate)
    machine.state.next_identity = 10000
    machine._index_relation(len(machine.state.relations) - 1, duplicate)
    try:
        machine._select_unique(CONTACT_LEXEME, 1, target.values[1], 1)
    except Refuse as exc:
        ambiguous = str(exc).startswith("compose:missing_or_ambiguous")
    else:
        ambiguous = False
    touches_two = machine.last_lookup_touches

    checks = {
        "exact_lookup": picked.identity == target.identity,
        "dormant_population_not_scanned": touches_one == 1,
        "ambiguity_refuses": ambiguous,
        "ambiguity_touches_exact_posting": touches_two == 2,
        "index_not_checkpoint_state": "_relation_index" not in machine._checkpoint_body(),
    }
    result = {
        "schema": "0x1.reference-incidence-quantity.v1",
        "pass": all(checks.values()),
        "dormant_relations": 450,
        "lookup_touches": touches_one,
        "ambiguous_lookup_touches": touches_two,
        "checks": checks,
        "claim": "DERIVED_INDEX_TOUCHED_WORK_NOT_SEMANTIC_AUTHORITY",
    }
    print("REFERENCE_INCIDENCE " + ("GREEN" if result["pass"] else "RED")
          + f" dormant_relations=450 exact_touches={touches_one} ambiguous_touches={touches_two}")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
