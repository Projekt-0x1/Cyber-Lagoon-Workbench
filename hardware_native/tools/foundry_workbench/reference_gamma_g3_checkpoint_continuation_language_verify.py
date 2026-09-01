#!/usr/bin/env python3
"""G3 O1 N+1: checkpointed same Adult uses old structure with new lived language."""
from __future__ import annotations

import copy
import json
import time

from reference_contextual_invariant_consolidation_v1 import ContextualInvariantBankV1
from reference_gamma_g3_contextual_invariant_language_verify import (
    A1, G1, O2, V2, C, JOIN, SEP, ContextualLanguageLifeV1, prepare, contact, refuses,
)
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1

NEW_ATOMS = (A1, G1, V2, O2)
NEW_SURFACE = "the careful engineer inspects the valve."
NEW_SOURCE = 9901


def restore_life(adult_checkpoint, invariant_checkpoint):
    life = ContextualLanguageLifeV1()
    life.adult = LanguageMasteryAdultV1.restore(copy.deepcopy(adult_checkpoint))
    life.invariants = ContextualInvariantBankV1.restore(copy.deepcopy(invariant_checkpoint))
    # All relation episodes used before the checkpoint have already been
    # materialized into ordinary learned span state. Pending lists are transient
    # construction work, not future-causal authority once promotion completed.
    life.pending = []
    life.promoted = set()
    return life


def earn_precheckpoint_relation(life, leaves):
    life.relation_episode(leaves[0], leaves[1], 8801, 9701)
    life.relation_episode(leaves[2], leaves[3], 8802, 9702)
    if not life.invariants.reusable(JOIN):
        raise RuntimeError("checkpoint_continuation:precheckpoint_invariant_missing")


def main():
    started = time.perf_counter()

    continuing = ContextualLanguageLifeV1()
    old_leaves = prepare(continuing)
    earn_precheckpoint_relation(continuing, old_leaves)

    # A matched control has all pre-checkpoint lexical/clause competence but no
    # lived relation episodes. It receives the same post-checkpoint clause.
    no_relation = ContextualLanguageLifeV1()
    prepare(no_relation)

    adult_checkpoint = continuing.adult.checkpoint()
    invariant_checkpoint = continuing.invariants.checkpoint()
    restored = restore_life(adult_checkpoint, invariant_checkpoint)

    # Checkpoint must not carry an active public-contact occurrence.
    checks = {
        "restored_current_occurrence_is_cleared":
            int(restored.adult._current_selection_context) == 0,
        "restored_invariant_checkpoint_exact":
            restored.invariants.checkpoint() == invariant_checkpoint,
        "restored_adult_checkpoint_stable":
            restored.adult.checkpoint() == adult_checkpoint,
    }

    # Both organisms now receive the same genuinely new lived clause after the
    # checkpoint boundary. It was never demonstrated before checkpoint.
    contact(restored.adult, (C, *NEW_ATOMS), NEW_SURFACE, NEW_SOURCE)
    contact(no_relation.adult, (C, *NEW_ATOMS), NEW_SURFACE, NEW_SOURCE)
    new_restored = restored.adult.leaf(C, NEW_ATOMS)
    new_control = no_relation.adult.leaf(C, NEW_ATOMS)
    checks["same_new_postcheckpoint_clause"] = (
        bytes(new_restored.surface) == bytes(new_control.surface) == NEW_SURFACE.encode()
    )

    rleaves = [restored.adult.leaf(C, atoms) for atoms, _ in (
        ((A1, 201, 301, 401), ""),
        ((102, 202, 302, 402), ""),
        ((A1, 202, 302, 401), ""),
        ((102, G1, 301, O2), ""),
    )]

    first = restored.adult.compose(JOIN, rleaves[3], rleaves[0])
    second = restored.adult.compose(JOIN, first, rleaves[2])
    longer = restored.adult.compose(JOIN, second, new_restored)
    visible = bytes(longer.surface).decode()

    checks.update({
        "continuing_adult_reuses_precheckpoint_relation_with_new_clause":
            longer.depth == 3 and bytes(longer.surface).count(SEP) == 3,
        "no_relation_control_refuses_same_new_clause":
            refuses(no_relation.adult, no_relation.adult.leaf(C, NEW_ATOMS), new_control),
        "new_clause_not_in_checkpoint":
            NEW_SURFACE.encode() not in json.dumps(adult_checkpoint, sort_keys=True).encode(),
        "visible_discourse_is_longer_than_precheckpoint_n_plus_1":
            bytes(longer.surface).count(SEP) == 3,
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    })

    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "contract": "FOUNDRY_GAMMA_G3_CHECKPOINT_CONTINUATION_LANGUAGE_GREEN",
        "pass": not failed,
        "reference_only": True,
        "graph_flip": False,
        "visible_language_gain": "POSTCHECKPOINT_NEW_CLAUSE_EXTENDS_PRIOR_RECURSIVE_DISCOURSE",
        "direct_phase_boundary": "OWNED_EXACT_HISTORY_ASSIMILATION_PLUS_ACTUAL_MOTOR_FINALIZATION",
        "direct_launch_geometry": "ONE_WARP_THREAD0_GUARDED",
        "postcheckpoint_clause": NEW_SURFACE,
        "continued_surface": visible,
        "continued_depth": longer.depth,
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "remaining_red": [
            "DIRECT_LANGUAGE_NOMINATION_PLANNING",
            "DIRECT_CONTEXTUAL_INVARIANT_LOWERING",
            "PHYSICAL_CHECKPOINT_CONTINUATION_ASSAY",
            "GRAPH_PROMOTION",
        ],
    }
    print(result["contract"] if not failed else
          "FOUNDRY_GAMMA_G3_CHECKPOINT_CONTINUATION_LANGUAGE_RED " + ",".join(failed))
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
