#!/usr/bin/env python3
"""Destructive Sapolsky audit over the factored productive-language substrate.

This adds no organism mechanism. It holds the learned productive surface/program
checkpoint fixed while matched counterfactual Adults vary lived consequence,
recent history, current resource state, controllability history, and recovery
trajectory. The point is to make the research pass causally destructive: selection
must vary for the grounded causes while exact earned language factors do not mutate.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_language_mastery_adult_v1 import AdultStateV1, LanguageMasteryAdultV1  # noqa: E402
from reference_mathematical_adult_operator_factorization_verify import build  # noqa: E402
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1, Q  # noqa: E402

# Documentation/test-only maintenance lane: this verifier changes no Adult law.
semantics_free_maintenance = True
phenotype_preserved = True
future_update_authority_preserved = True

CTX = 0x5A01
SHORT = 0x5A02


def surface_state(adult: LanguageMasteryAdultV1) -> str:
    return json.dumps(adult.program_surface_checkpoint(), sort_keys=True, separators=(",", ":"))


def choose(adult: LanguageMasteryAdultV1, state: AdultStateV1 = AdultStateV1()) -> int:
    return int(adult._probe_choice(CTX, state))


def history_bank(short_id: int, deep_id: int) -> PredictiveCreditBankV1:
    """Same value evidence for both control-history orderings below."""
    bank = PredictiveCreditBankV1(32)
    for n in range(4):
        tick = 10 + n * 6
        bank.observe_use(short_id, tick, tick + 2, Q // 12, CTX)
        bank.observe_return(short_id, Q // 2, Q // 16, tick + 3, True, CTX)
        bank.observe_control(short_id, True, True)
    for n in range(2):
        tick = 80 + n * 10
        bank.observe_use(deep_id, tick, tick + 7, Q // 2, CTX)
        bank.observe_return(deep_id, Q, Q // 8, tick + 8, True, CTX)
    return bank


def adult_with_bank(base_checkpoint, bank: PredictiveCreditBankV1) -> LanguageMasteryAdultV1:
    adult = LanguageMasteryAdultV1.restore(copy.deepcopy(base_checkpoint))
    adult.credit = bank
    adult._select_epoch += 1
    return adult


def main() -> int:
    started = time.perf_counter()
    seed, leaves, _deep, top, _programs, _roots = build(True)
    seed.experience_atomic_program(SHORT, leaves[0], Q // 2, Q // 16, CTX, Q // 16, True)

    base_checkpoint = copy.deepcopy(seed.checkpoint())
    base_surface_state = surface_state(seed)
    short_surface = seed.public_surface(SHORT)
    deep_surface = seed.public_surface(top.identity)
    program_surface = seed.program_surface_checkpoint()

    # Consequence/history twins: same learned language state and same present context,
    # but different lived outcome histories. The internal resident competition must
    # choose differently without rewriting the productive surface factors.
    favorable = LanguageMasteryAdultV1.restore(copy.deepcopy(base_checkpoint))
    for _ in range(7):
        favorable.experience_choice(SHORT, -Q // 2, -Q // 8, CTX, Q // 16, 1, True)
    for _ in range(5):
        favorable.experience_choice(top.identity, 3 * Q // 4, Q // 8, CTX, Q // 2, 10, True)
    favorable_choice = choose(favorable)

    adverse = LanguageMasteryAdultV1.restore(copy.deepcopy(base_checkpoint))
    for _ in range(5):
        adverse.experience_choice(SHORT, 3 * Q // 4, Q // 8, CTX, Q // 16, 1, True)
    for _ in range(5):
        adverse.experience_choice(top.identity, -Q // 2, -Q // 8, CTX, Q // 2, 10, True)
    adverse_choice = choose(adverse)

    # Current body/resource state is a different causal level from learned value.
    # Pressure can suppress the expensive deep trajectory; authenticated relief can
    # recover it without changing learned structural evidence.
    pressured_choice = choose(favorable, AdultStateV1(urgency_q16=Q, pressure_q16=Q))
    forged_relief_choice = choose(
        favorable,
        AdultStateV1(pressure_q16=Q, relief_q16=3 * Q // 4, relief_authenticated=False),
    )
    recovered_state_choice = choose(
        favorable,
        AdultStateV1(pressure_q16=Q, relief_q16=3 * Q // 4, relief_authenticated=True),
    )

    # Recent counterevidence must be able to reverse an earlier developmental/lived
    # preference. This is deliberately ordinary experience, not a host "switch".
    recent = LanguageMasteryAdultV1.restore(copy.deepcopy(favorable.checkpoint()))
    for _ in range(12):
        recent.experience_choice(SHORT, 3 * Q // 4, Q // 8, CTX, Q // 16, 1, True)
        recent.experience_choice(top.identity, -3 * Q // 4, -Q // 8, CTX, Q // 2, 10, True)
    recent_choice = choose(recent)

    # Matched-current-contingency control-history twins. They finish with exactly the
    # same A/O incidence and current DeltaP=0, but only one lived trajectory first
    # acquired repeated control. Order/history therefore remains causally visible.
    prior_control = history_bank(SHORT, top.identity)
    for _ in range(2):
        prior_control.observe_control(top.identity, True, True)
    for _ in range(2):
        prior_control.observe_control(top.identity, False, True)

    interleaved_yoked = history_bank(SHORT, top.identity)
    for public_action in (False, True, False, True):
        interleaved_yoked.observe_control(top.identity, public_action, True)

    prior_row = prior_control.row(top.identity)
    yoked_row = interleaved_yoked.row(top.identity)
    prior_cells = (
        prior_row.control_attempts,
        prior_row.control_successes,
        prior_row.background_attempts,
        prior_row.background_successes,
    )
    yoked_cells = (
        yoked_row.control_attempts,
        yoked_row.control_successes,
        yoked_row.background_attempts,
        yoked_row.background_successes,
    )
    prior_adult = adult_with_bank(base_checkpoint, prior_control)
    yoked_adult = adult_with_bank(base_checkpoint, interleaved_yoked)
    prior_choice = choose(prior_adult)
    yoked_choice = choose(yoked_adult)

    # Destructive history lesion, then timescale extinction/recovery.
    lesioned_bank = PredictiveCreditBankV1.restore(prior_control.checkpoint())
    lesioned_bank.row(top.identity).control_history_q16 = 0
    lesioned_adult = adult_with_bank(base_checkpoint, lesioned_bank)
    lesioned_choice = choose(lesioned_adult)

    extinguished_bank = PredictiveCreditBankV1.restore(prior_control.checkpoint())
    for _ in range(3):
        extinguished_bank.observe_control(top.identity, False, True)
    extinguished_adult = adult_with_bank(base_checkpoint, extinguished_bank)
    extinguished_choice = choose(extinguished_adult)
    for _ in range(5):
        extinguished_bank.observe_control(top.identity, False, False)
    extinguished_bank.observe_control(top.identity, True, True)
    reacquired_adult = adult_with_bank(base_checkpoint, extinguished_bank)
    reacquired_choice = choose(reacquired_adult)

    branches = (
        favorable,
        adverse,
        recent,
        prior_adult,
        yoked_adult,
        lesioned_adult,
        extinguished_adult,
        reacquired_adult,
    )
    exact_surface_state_invariant = all(surface_state(adult) == base_surface_state for adult in branches)
    exact_public_surfaces_invariant = all(
        adult.public_surface(SHORT) == short_surface
        and adult.public_surface(top.identity) == deep_surface
        for adult in branches
    )

    checks = {
        "productive_surface_is_factored_not_raw": (
            not program_surface["raw_leaf_surfaces"]
            and len(program_surface["leaf_families"]) == 1
        ),
        "same_current_context_different_consequence_history_changes_winner": (
            favorable_choice == top.identity and adverse_choice == SHORT
        ),
        "current_resource_state_changes_transition_without_relearning": (
            pressured_choice == SHORT
            and forged_relief_choice != top.identity
            and recovered_state_choice == top.identity
        ),
        "recent_counterevidence_reverses_prior_preference": recent_choice == SHORT,
        "matched_current_contingency_cells_hide_different_developmental_history": (
            prior_cells == yoked_cells == (2, 2, 2, 2)
            and prior_row.controllability_q16 == yoked_row.controllability_q16 == 0
            and prior_row.control_history_q16 > yoked_row.control_history_q16 == 0
        ),
        "controllability_history_changes_internal_competition": (
            prior_choice == top.identity and yoked_choice == SHORT
        ),
        "focal_control_history_lesion_destroys_only_history_advantage": lesioned_choice == SHORT,
        "timescale_disconfirmation_extinguishes_and_lived_control_recovers": (
            extinguished_choice == SHORT and reacquired_choice == top.identity
        ),
        "cross_level_variation_never_rewrites_exact_productive_surface_state": exact_surface_state_invariant,
        "cross_level_variation_never_rewrites_exact_public_surfaces": exact_public_surfaces_invariant,
        "resident_competition_not_host_expected_answer": (
            set(prior_control.candidates(CTX)) == {SHORT, top.identity}
            and set(interleaved_yoked.candidates(CTX)) == {SHORT, top.identity}
        ),
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema": "cyber-lagoon.reference-productive-leaf-sapolsky-destructive-audit.v1",
        "pass": not failed,
        "reference_only": True,
        "mechanism_change": False,
        "semantics_free_maintenance": semantics_free_maintenance,
        "phenotype_preserved": phenotype_preserved,
        "future_update_authority_preserved": future_update_authority_preserved,
        "economic_gain": {
            "sapolsky_axes_with_executable_counterfactuals": 6,
            "supporting_context_only_axes_allowed": 0,
        },
        "audit_axes": [
            "prior_developmental_history",
            "recent_history",
            "current_body_resource_state",
            "controllability",
            "consequence",
            "timescale_recovery",
        ],
        "same_current_context": CTX,
        "short_program": SHORT,
        "deep_program": int(top.identity),
        "short_bytes": len(short_surface),
        "deep_bytes": len(deep_surface),
        "matched_control_cells": prior_cells,
        "current_controllability_q16": int(prior_row.controllability_q16),
        "prior_control_history_q16": int(prior_row.control_history_q16),
        "interleaved_control_history_q16": int(yoked_row.control_history_q16),
        "checks": checks,
        "remaining_red": [
            "PERSISTENT_BODY_RESOURCE_HISTORY_BEYOND_CURRENT_ADULT_STATE",
            "SOCIAL_SOURCE_HISTORY_IN_THIS_FACTORIZATION_ASSAY",
            "DIRECT_CUDA_PARITY",
        ],
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print("FOUNDRY_PRODUCTIVE_LEAF_SAPOLSKY_DESTRUCTIVE_AUDIT_" + ("GREEN" if not failed else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
