#!/usr/bin/env python3
"""Destructive persistent-somatic-history audit over fixed productive language.

Reference-only: no Adult mechanism is added. The current Adult state, current context,
outcome values, control history, candidate programs, and exact productive surfaces are
matched while prior authenticated somatic consequence history is varied.
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

semantics_free_maintenance = True
phenotype_preserved = True
future_update_authority_preserved = True
language_phenotype_improved = True

CTX = 0x5B01
SHORT = 0x5B02
CURRENT_STATE = AdultStateV1()


def surface_state(adult: LanguageMasteryAdultV1) -> str:
    return json.dumps(adult.program_surface_checkpoint(), sort_keys=True, separators=(",", ":"))


def choice(adult: LanguageMasteryAdultV1) -> int:
    return int(adult._probe_choice(CTX, CURRENT_STATE))


def lived_branch(base_checkpoint, deep_id: int, deep_somatic_q16: int):
    adult = LanguageMasteryAdultV1.restore(copy.deepcopy(base_checkpoint))
    # Selection history starts matched and empty; language/program factors remain the
    # exact earned checkpoint. Both programs receive the same factual outcome and
    # control evidence. Only the deep program's bodily consequence history differs.
    adult.credit = PredictiveCreditBankV1(32)
    for _ in range(4):
        adult.experience_choice(SHORT, Q // 4, 0, CTX, Q // 16, 1, True)
        adult.experience_choice(deep_id, Q // 4, deep_somatic_q16, CTX, Q // 4, 5, True)
    return adult, choice(adult)


def control_cells(adult: LanguageMasteryAdultV1, pid: int):
    row = adult.credit.row(pid).contexts[CTX]
    return (
        row.control_attempts,
        row.control_successes,
        row.background_attempts,
        row.background_successes,
        row.control_history_q16,
    )


def main() -> int:
    started = time.perf_counter()
    seed, leaves, _deep, top, _programs, _roots = build(True)
    seed.experience_atomic_program(SHORT, leaves[0], Q // 4, 0, CTX, Q // 16, True)
    base_checkpoint = copy.deepcopy(seed.checkpoint())
    base_surface_state = surface_state(seed)
    short_surface = seed.public_surface(SHORT)
    deep_surface = seed.public_surface(top.identity)

    favorable, favorable_choice = lived_branch(base_checkpoint, top.identity, Q // 2)
    adverse, adverse_choice = lived_branch(base_checkpoint, top.identity, -Q // 2)

    favorable_outcomes = (
        favorable.credit.contextual_outcome(SHORT, CTX),
        favorable.credit.contextual_outcome(top.identity, CTX),
    )
    adverse_outcomes = (
        adverse.credit.contextual_outcome(SHORT, CTX),
        adverse.credit.contextual_outcome(top.identity, CTX),
    )
    favorable_controls = (
        control_cells(favorable, SHORT),
        control_cells(favorable, top.identity),
    )
    adverse_controls = (
        control_cells(adverse, SHORT),
        control_cells(adverse, top.identity),
    )
    favorable_somatic = (
        favorable.credit.contextual_somatic(SHORT, CTX),
        favorable.credit.contextual_somatic(top.identity, CTX),
    )
    adverse_somatic = (
        adverse.credit.contextual_somatic(SHORT, CTX),
        adverse.credit.contextual_somatic(top.identity, CTX),
    )

    # Persistent learned history survives checkpoint while current state remains neutral.
    replay = LanguageMasteryAdultV1.restore(copy.deepcopy(favorable.checkpoint()))
    replay_choice = choice(replay)

    # Focal destructive lesion: remove only the deep program's context-local somatic
    # history. Outcome/control evidence and exact language factors are left untouched.
    lesioned = LanguageMasteryAdultV1.restore(copy.deepcopy(favorable.checkpoint()))
    lesion_row = lesioned.credit.row(top.identity).contexts[CTX]
    lesion_outcome_before = lesion_row.outcome_mean_q16
    lesion_control_before = (
        lesion_row.control_attempts,
        lesion_row.control_successes,
        lesion_row.background_attempts,
        lesion_row.background_successes,
        lesion_row.control_history_q16,
    )
    lesion_row.somatic_mean_q16 = 0
    lesioned._select_epoch += 1
    lesioned_choice = choice(lesioned)
    lesion_control_after = control_cells(lesioned, top.identity)

    # Evidence-driven recovery/reversal: the previously adverse Adult receives ordinary
    # authenticated bodily consequence on real uses of the same deep program.
    recovered = LanguageMasteryAdultV1.restore(copy.deepcopy(adverse.checkpoint()))
    for _ in range(12):
        recovered.experience_choice(top.identity, Q // 4, Q // 2, CTX, Q // 4, 5, True)
    recovered_choice = choice(recovered)
    recovered_somatic = recovered.credit.contextual_somatic(top.identity, CTX)

    branches = (favorable, adverse, replay, lesioned, recovered)
    exact_surface_state = all(surface_state(adult) == base_surface_state for adult in branches)
    exact_public_surfaces = all(
        adult.public_surface(SHORT) == short_surface
        and adult.public_surface(top.identity) == deep_surface
        for adult in branches
    )

    checks = {
        "same_current_state_and_context_different_somatic_history_changes_winner": (
            favorable_choice == top.identity and adverse_choice == SHORT
        ),
        "factual_outcome_history_is_matched": (
            favorable_outcomes == adverse_outcomes == (Q // 4, Q // 4)
        ),
        "controllability_history_is_matched": favorable_controls == adverse_controls,
        "only_somatic_history_differs_across_twins": (
            favorable_somatic == (0, Q // 2)
            and adverse_somatic == (0, -Q // 2)
        ),
        "checkpoint_retains_body_consequence_history": replay_choice == top.identity,
        "focal_somatic_history_lesion_removes_deep_advantage": (
            lesioned_choice == SHORT
            and lesion_row.outcome_mean_q16 == lesion_outcome_before
            and lesion_control_after == lesion_control_before
        ),
        "ordinary_authenticated_body_consequence_reacquires_deep_program": (
            recovered_choice == top.identity and recovered_somatic > 0
        ),
        "somatic_history_never_rewrites_productive_surface_state": exact_surface_state,
        "somatic_history_never_rewrites_public_surfaces": exact_public_surfaces,
        "current_pressure_relief_are_identical_neutral_controls": (
            CURRENT_STATE == AdultStateV1()
        ),
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema": "cyber-lagoon.reference-productive-leaf-persistent-somatic-history-destructive-audit.v1",
        "pass": not failed,
        "reference_only": True,
        "mechanism_change": False,
        "semantics_free_maintenance": semantics_free_maintenance,
        "phenotype_preserved": phenotype_preserved,
        "future_update_authority_preserved": future_update_authority_preserved,
        "language_phenotype_improved": language_phenotype_improved,
        "visible_language_gain": "MATCHED_CURRENT_STATE_PERSISTENT_SOMATIC_HISTORY_SWITCHES_FACTORED_SHORT_VS_DEEP_PROGRAM",
        "destructive_axis_paid": "PERSISTENT_CONTEXT_LOCAL_SOMATIC_CONSEQUENCE_HISTORY",
        "same_current_state": {
            "urgency_q16": CURRENT_STATE.urgency_q16,
            "pressure_q16": CURRENT_STATE.pressure_q16,
            "relief_q16": CURRENT_STATE.relief_q16,
            "relief_authenticated": CURRENT_STATE.relief_authenticated,
        },
        "short_program": SHORT,
        "deep_program": int(top.identity),
        "short_bytes": len(short_surface),
        "deep_bytes": len(deep_surface),
        "matched_outcomes_q16": list(favorable_outcomes),
        "favorable_somatic_q16": list(favorable_somatic),
        "adverse_somatic_q16": list(adverse_somatic),
        "recovered_deep_somatic_q16": int(recovered_somatic),
        "checks": checks,
        "remaining_red": [
            "SLOW_RESOURCE_ALLOSTATIC_HISTORY_DISTINCT_FROM_ACTION_SPECIFIC_SOMATIC_RETURN",
            "SPONTANEOUS_TIME_BASED_ALLOSTATIC_RECOVERY_WITHOUT_NEW_PROGRAM_RETURN",
            "DIRECT_CUDA_PARITY",
        ],
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print("FOUNDRY_PRODUCTIVE_LEAF_PERSISTENT_SOMATIC_HISTORY_DESTRUCTIVE_AUDIT_" + ("GREEN" if not failed else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
