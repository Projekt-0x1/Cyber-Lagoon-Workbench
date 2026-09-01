#!/usr/bin/env python3
"""Fast causal contrast for retiring settled language-action Occurrences."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_organism_repair_verify import CTX, P1, partner, scene, train  # noqa: E402
from reference_organism_v2 import ActionV2, CONTACT_CONSEQUENCE, ReferenceOrganismV2  # noqa: E402
from reference_population_v1 import PopulationSpecV1  # noqa: E402

state_minimization_refactor = True
semantics_free_refactor = True
phenotype_preserved = True
future_update_authority_preserved = True
state_reduction = "settled action-private Occurrences no longer grow lifetime dialogue state"

SPEC = PopulationSpecV1(1024, 2, 4, 42, 8)
FIRST = (102, 201, 301, 402)
HELDOUT = (101, 202, 302, 401)


def wire_bytes(state):
    return len(json.dumps(state, sort_keys=True, separators=(",", ":")).encode())


def build_pending(source_scene, fault=False):
    organism = ReferenceOrganismV2(SPEC)
    train(organism)
    partner(organism, P1)
    scene(organism, CTX, FIRST, source_scene)
    if fault:
        organism.inject_output_fault(0, ord("X"))
    action = organism.tick()
    if not isinstance(action, ActionV2):
        raise RuntimeError("settled-language-occurrence-retirement:no_action")
    return organism, action


def private_ids(action):
    return {
        int(action.population_occurrence),
        *(int(row[3]) for row in action.selection_occurrences),
    }


def population_ids(checkpoint):
    return {int(row["identity"]) for row in checkpoint["population"]["occurrences"]}


def without_occurrences(checkpoint, identities):
    out = copy.deepcopy(checkpoint)
    removed = set(map(int, identities))
    out["population"]["occurrences"] = [
        row for row in out["population"]["occurrences"]
        if int(row["identity"]) not in removed
    ]
    return out


def with_legacy_occurrences(compact, pending, identities):
    out = copy.deepcopy(compact)
    wanted = set(map(int, identities))
    rows = [copy.deepcopy(row) for row in pending["population"]["occurrences"]
            if int(row["identity"]) in wanted]
    out["population"]["occurrences"].extend(rows)
    out["population"]["occurrences"].sort(key=lambda row: int(row["identity"]))
    return out


def future(checkpoint, legacy_ids):
    organism = ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    scene(organism, CTX, HELDOUT, 46001)
    action = organism.tick()
    if not isinstance(action, ActionV2):
        raise RuntimeError("settled-language-occurrence-retirement:no_future_action")
    learned = organism.contact(
        CONTACT_CONSEQUENCE, (action.ticket, 1), P1, True, True)
    projected = without_occurrences(organism.checkpoint(), legacy_ids)
    return {
        "surface": bytes(action.payload),
        "credit": int(learned.get("credit", 0)),
        "revisions": int(learned.get("revisions", 0)),
        "selection_credit": int(learned.get("selection_credit", 0)),
        "selection_network_updates": int(learned.get("selection_network_updates", 0)),
        "projected_checkpoint": projected,
    }


def main():
    started = time.perf_counter()
    organism, action = build_pending(45001)
    pending = organism.checkpoint()
    retired = private_ids(action)
    pending_ids = population_ids(pending)
    learned = organism.contact(
        CONTACT_CONSEQUENCE, (action.ticket, 1), P1, True, True)
    compact = organism.checkpoint()
    compact_ids = population_ids(compact)
    legacy = with_legacy_occurrences(compact, pending, retired)

    donor_future = future(legacy, retired)
    compact_future = future(compact, retired)

    # RED control: the same rows are irreducible while the return is pending.
    premature = without_occurrences(pending, (action.population_occurrence,))
    premature_refused = False
    try:
        cut = ReferenceOrganismV2.restore(premature)
        cut.contact(CONTACT_CONSEQUENCE, (action.ticket, 1), P1, True, True)
    except ValueError as exc:
        premature_refused = str(exc) == "organism:consequence_occurrence"

    # A punished output still needs its action metadata for endogenous repair, but
    # never needs the already-settled population computations themselves.
    faulted, bad = build_pending(45002, fault=True)
    fault_private = private_ids(bad)
    faulted.contact(CONTACT_CONSEQUENCE, (bad.ticket, -1), P1, True, True)
    repair = faulted.tick()
    repair_ok = isinstance(repair, ActionV2) and repair.repair and repair.payload == bad.planned_payload
    if repair_ok:
        faulted.contact(CONTACT_CONSEQUENCE, (repair.ticket, 1), P1, True, True)

    legacy_bytes = wire_bytes(legacy)
    compact_bytes = wire_bytes(compact)
    checks = {
        "pending_private_occurrences_exist": retired <= pending_ids,
        "settled_private_occurrences_retired": retired.isdisjoint(compact_ids),
        "settlement_still_earns_credit": int(learned.get("credit", 0)) > 0,
        "legacy_rows_increase_checkpoint": legacy_bytes > compact_bytes,
        "future_language_and_update_authority_equal": donor_future == compact_future,
        "premature_retirement_refused": premature_refused,
        "fault_repair_survives_retirement": repair_ok,
        "fault_and_repair_private_occurrences_retired": (
            fault_private | (set() if not repair_ok else private_ids(repair))
        ).isdisjoint({row.identity for row in faulted.population.occurrences}),
        "bounded_seconds_lane": time.perf_counter() - started < 5.0,
    }
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-settled-language-occurrence-retirement.v1",
        "pass": all(checks.values()),
        "reference_only": True,
        "state_role": "PENDING_UNTIL_RETURN",
        "retired_coordinates": [
            "ActionV2.population_occurrence",
            "ActionV2.selection_occurrences[].occurrence",
        ],
        "retired_rows_per_turn": len(retired),
        "legacy_checkpoint_bytes": legacy_bytes,
        "compact_checkpoint_bytes": compact_bytes,
        "bytes_saved_for_one_settled_turn": legacy_bytes - compact_bytes,
        "future_surface": donor_future["surface"].decode(errors="replace"),
        "negative_control": "deleting the action Occurrence before return rejects settlement",
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print("FOUNDRY_MATHEMATICAL_ADULT_SETTLED_LANGUAGE_OCCURRENCE_RETIREMENT_" +
          ("GREEN" if result["pass"] else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
