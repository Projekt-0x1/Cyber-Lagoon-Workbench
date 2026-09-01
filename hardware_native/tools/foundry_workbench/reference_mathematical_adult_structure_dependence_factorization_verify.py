#!/usr/bin/env python3
"""Structure-dependence falsifier for productive-leaf factorization.

Two distinct learned recursive program trees must preserve distinct causal structure even
when they rematerialize the same linear public bytes. A focal internal-factor lesion must
break only the dependent tree; a shared historical terminal lesion must break both.
Reference-only: this adds no grammar mechanism and makes no Direct capability claim.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_hierarchical_composition_v1 import HierarchicalRefuse  # noqa: E402
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1  # noqa: E402
from reference_predictive_credit_profile_v1 import Q  # noqa: E402

CLAUSE, JOIN, CTX = 9001, 9101, 0xA11


def earn(adult, children, witness):
    program = None
    for _ in range(3):
        program = adult.experience_program(
            tuple(int(x) for x in children), witness,
            3 * Q // 4, Q // 8, CTX, Q // 3, True)
    if program is None:
        raise RuntimeError("structure-dependence:program_not_earned")
    return program


def build():
    adult = LanguageMasteryAdultV1()
    names = {
        101: "careful", 102: "quiet",
        201: "engineer", 202: "technician",
        301: "tests", 302: "inspects",
        401: "sensor", 402: "valve",
    }
    for feature, text in names.items():
        adult.observe_surface_item(feature, text.encode(), 1000 + feature)
        adult.observe_surface_item(feature, text.encode(), 2000 + feature)
    adult.observe_surface_construction(
        CLAUSE, (101, 201, 301, 401),
        b"the careful engineer tests the sensor.", 3001)
    adult.observe_surface_construction(
        CLAUSE, (102, 202, 302, 402),
        b"the quiet technician inspects the valve.", 3002)

    atoms = (
        (101, 201, 301, 401),
        (102, 201, 301, 402),
        (101, 202, 302, 401),
        (102, 202, 302, 402),
    )
    leaves = tuple(adult.leaf(CLAUSE, row) for row in atoms)
    adult.observe_join(JOIN, leaves[0], leaves[1], 5001)
    adult.observe_join(JOIN, leaves[2], leaves[3], 5002)

    p01 = earn(adult, (leaves[0].identity, leaves[1].identity),
               adult.compose(JOIN, leaves[0], leaves[1]))
    p23 = earn(adult, (leaves[2].identity, leaves[3].identity),
               adult.compose(JOIN, leaves[2], leaves[3]))
    balanced = earn(adult, (p01.identity, p23.identity),
                    adult.compose(JOIN, p01, p23))

    p123 = earn(adult, (leaves[1].identity, p23.identity),
                adult.compose(JOIN, leaves[1], p23))
    right_deep = earn(adult, (leaves[0].identity, p123.identity),
                      adult.compose(JOIN, leaves[0], p123))
    return adult, leaves, p01, p23, p123, balanced, right_deep


def refuses_surface(adult, program_identity):
    try:
        adult.public_surface(int(program_identity))
        return False
    except RuntimeError:
        return True


def main():
    started = time.perf_counter()
    adult, leaves, p01, p23, p123, balanced, right_deep = build()
    balanced_surface = adult.public_surface(balanced.identity)
    right_surface = adult.public_surface(right_deep.identity)
    checkpoint = copy.deepcopy(adult.checkpoint())
    program_surface_state = checkpoint["program_surfaces"]

    restored = LanguageMasteryAdultV1.restore(copy.deepcopy(checkpoint))
    replay_balanced = restored.public_surface(balanced.identity)
    replay_right = restored.public_surface(right_deep.identity)

    # Structural lesions are cold counterfactual forks of durable state. The Adult's
    # rematerialized leaf/public holds are transient recall/output state and therefore
    # must not mask removal of a persistent program factor after checkpoint restore.
    internal_lesion = LanguageMasteryAdultV1.restore(copy.deepcopy(checkpoint))
    internal_factor = internal_lesion.programs.factor(p123.identity)
    if internal_factor is None:
        raise RuntimeError("structure-dependence:missing_internal_factor")
    del internal_lesion.programs.factors[p123.identity]
    right_internal_refused = refuses_surface(internal_lesion, right_deep.identity)
    balanced_after_internal_lesion = internal_lesion.public_surface(balanced.identity)
    internal_lesion.programs.factors[p123.identity] = int(internal_factor)
    right_after_restore = internal_lesion.public_surface(right_deep.identity)

    # Shared terminal lesion: both structures ultimately require leaf 1's historical
    # quiet lexeme. Use a separate cold fork so no prior output tape survives.
    terminal_lesion = LanguageMasteryAdultV1.restore(copy.deepcopy(checkpoint))
    quiet_units = terminal_lesion.language.historical_lexeme_units(
        terminal_lesion.language.lexeme_identity(102, tuple(b"quiet")))
    if quiet_units != tuple(b"quiet"):
        raise RuntimeError("structure-dependence:quiet_witness")
    quiet_identity = terminal_lesion.language.lexeme_identity(102, quiet_units)
    quiet_key = next(
        key for key in terminal_lesion.language._lexeme_sources
        if terminal_lesion.language.lexeme_identity(key[0], key[1]) == quiet_identity)
    quiet_sources = terminal_lesion.language._lexeme_sources.pop(quiet_key)
    terminal_lesion.language._rebuild_indices()
    shared_terminal_lesion_breaks_balanced = refuses_surface(terminal_lesion, balanced.identity)
    shared_terminal_lesion_breaks_right = refuses_surface(terminal_lesion, right_deep.identity)
    terminal_lesion.language._lexeme_sources[quiet_key] = quiet_sources
    terminal_lesion.language._rebuild_indices()
    recovered_balanced = terminal_lesion.public_surface(balanced.identity)
    recovered_right = terminal_lesion.public_surface(right_deep.identity)

    checks = {
        "same_linear_surface_distinct_structural_programs": (
            balanced.identity != right_deep.identity
            and balanced_surface == right_surface
            and balanced.depth == 2 and right_deep.depth == 3),
        "productive_leaves_remain_factored_not_raw": (
            not program_surface_state["raw_leaf_surfaces"]
            and len(program_surface_state["leaf_families"]) == 1
            and len(program_surface_state["leaf_families"][0]["leaves"]) == 4),
        "checkpoint_preserves_both_structures_exactly": (
            replay_balanced == balanced_surface and replay_right == right_surface
            and restored.program_depth(balanced.identity) == 2
            and restored.program_depth(right_deep.identity) == 3),
        "right_internal_factor_lesion_is_branch_local": (
            right_internal_refused and balanced_after_internal_lesion == balanced_surface),
        "right_internal_factor_restore_recovers": right_after_restore == right_surface,
        "shared_terminal_lesion_breaks_both_structures": (
            shared_terminal_lesion_breaks_balanced and shared_terminal_lesion_breaks_right),
        "shared_terminal_restore_recovers_both": (
            recovered_balanced == balanced_surface and recovered_right == right_surface),
        "decision_width_remains_one": (
            adult.current_width(balanced.identity) == 1
            and adult.current_width(right_deep.identity) == 1),
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    }
    failed = [key for key, value in checks.items() if not value]
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-structure-dependence-factorization.v1",
        "pass": not failed,
        "reference_only": True,
        "mechanism_change": False,
        "falsifier_class": "STRUCTURE_DEPENDENCE_SAME_LINEAR_SURFACE_DISTINCT_CAUSAL_TREE",
        "balanced_program": int(balanced.identity),
        "right_deep_program": int(right_deep.identity),
        "balanced_depth": int(balanced.depth),
        "right_deep_depth": int(right_deep.depth),
        "public_bytes": len(balanced_surface),
        "raw_productive_leaf_bytes_persisted": sum(
            len(row["surface"]) for row in program_surface_state["raw_leaf_surfaces"]),
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print("FOUNDRY_MATHEMATICAL_ADULT_STRUCTURE_DEPENDENCE_FACTORIZATION_" +
          ("GREEN" if not failed else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
