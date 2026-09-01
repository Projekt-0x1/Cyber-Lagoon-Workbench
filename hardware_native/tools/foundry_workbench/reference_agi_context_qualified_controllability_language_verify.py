#!/usr/bin/env python3
"""RED/contrast for context-qualified control in Adult-owned competition."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_language_mastery_adult_v1 import AdultStateV1, LanguageMasteryAdultV1  # noqa: E402
from reference_mathematical_adult_operator_factorization_verify import build  # noqa: E402
from reference_predictive_credit_profile_v1 import Q  # noqa: E402

CTX_A = 0xC0A1
CTX_B = 0xC0B1
CTX_FORGED = 0xC0F1
FAMILIAR = 0xC001


def choose(adult: LanguageMasteryAdultV1, context: int) -> int:
    """Use the resident Adult tournament, never a host-selected maximum."""
    return int(adult._probe_choice(context, AdultStateV1()))


def observe_local_control(adult, program, context, public_action, independent_return):
    """Call the challenger API; fall back only to expose predecessor behavior."""
    try:
        adult.credit.observe_control(
            program, public_action, independent_return, context=context)
        return True
    except TypeError:
        adult.credit.observe_control(program, public_action, independent_return)
        return False


def main() -> int:
    started = time.perf_counter()
    adult, leaves, _retired, deep, _programs, _roots = build(True)

    # The same resident candidates participate in both opaque contexts. Familiar
    # language has modest consequence and repeated control in both. The deeper
    # held-out composition has greater value, but earns action control only in A.
    for context in (CTX_A, CTX_B):
        for _ in range(2):
            adult.experience_atomic_program(
                FAMILIAR, leaves[0], Q // 2, Q // 16, context, Q // 16, True)
    for _ in range(2):
        adult.experience_choice(
            deep.identity, Q, Q // 8, CTX_A, Q // 3, deep.depth + 1, True)

    # B supplies one no-action/non-outcome opportunity and one matched free
    # outcome. It changes controllability, not value or candidate membership.
    adult.experience_choice(
        deep.identity, Q, Q // 8, CTX_B, Q // 3, deep.depth + 1, False)
    contextual_api = observe_local_control(adult, deep.identity, CTX_B, False, True)

    initial_a = choose(adult, CTX_A)
    initial_b = choose(adult, CTX_B)
    familiar_surface = adult.public_surface(FAMILIAR)
    deep_surface = adult.public_surface(deep.identity)
    local = adult.credit.row(deep.identity).contexts.get(CTX_B)
    local_state_present = bool(
        local is not None
        and hasattr(local, "control_history_q16")
        and hasattr(local, "controllability_q16")
    )

    checks = {
        "same_resident_candidates_compete_in_a_and_b": (
            set(adult.credit.candidates(CTX_A)) == {FAMILIAR, deep.identity}
            and set(adult.credit.candidates(CTX_B)) == {FAMILIAR, deep.identity}
        ),
        "heldout_language_is_deeper_and_visible": (
            deep_surface is not None
            and familiar_surface is not None
            and deep.depth > 1
            and len(deep_surface) > len(familiar_surface)
        ),
        "contextual_control_state_exists": contextual_api and local_state_present,
        "adult_keeps_deep_language_in_controlled_a": initial_a == deep.identity,
        "adult_refuses_to_borrow_a_control_in_b": initial_b == FAMILIAR,
    }

    if contextual_api and local_state_present:
        # Ordinary B-local action consequences must recover the deeper winner.
        for _ in range(2):
            adult.experience_choice(
                deep.identity, Q, Q // 8, CTX_B, Q // 3, deep.depth + 1, True)
        recovered_b = choose(adult, CTX_B)
        preserved_a = choose(adult, CTX_A)

        checkpoint = copy.deepcopy(adult.checkpoint())
        restored = LanguageMasteryAdultV1.restore(checkpoint)
        restored_a = choose(restored, CTX_A)
        restored_b = choose(restored, CTX_B)

        # A focal Workbench lesion removes B-local causal support only. Global
        # developmental history and A-local support must remain physically present.
        lesioned = LanguageMasteryAdultV1.restore(checkpoint)
        lesion = lesioned.credit.row(deep.identity).contexts[CTX_B]
        lesion.control_attempts = 0
        lesion.control_successes = 0
        lesion.background_attempts = 0
        lesion.background_successes = 0
        lesion.control_history_q16 = 0
        lesioned._select_epoch += 1
        lesioned_b = choose(lesioned, CTX_B)
        lesioned_a = choose(lesioned, CTX_A)

        forged_before = lesioned.credit.snapshot()
        forged_refused = False
        try:
            lesioned.credit.observe_control(
                deep.identity, True, True, context=CTX_FORGED)
        except ValueError:
            forged_refused = True
        forged_atomic = forged_before == lesioned.credit.snapshot()

        for _ in range(2):
            lesioned.experience_choice(
                deep.identity, Q, Q // 8, CTX_B, Q // 3, deep.depth + 1, True)
        lesion_recovered_b = choose(lesioned, CTX_B)

        checks.update({
            "ordinary_b_consequence_recovers_deeper_language": recovered_b == deep.identity,
            "b_recovery_does_not_rewrite_a": preserved_a == deep.identity,
            "checkpoint_retains_contextual_control": (
                restored_a == deep.identity and restored_b == deep.identity),
            "focal_b_lesion_preserves_a_and_global_history": (
                lesioned_b == FAMILIAR
                and lesioned_a == deep.identity
                and lesioned.credit.row(deep.identity).control_history_q16 > 0
            ),
            "nonparticipating_context_control_refusal_is_atomic": (
                forged_refused and forged_atomic),
            "ordinary_control_recovers_after_focal_lesion": (
                lesion_recovered_b == deep.identity),
        })

    checks["bounded_runtime"] = time.perf_counter() - started < 1.0
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print("FOUNDRY_AGI_CONTEXT_QUALIFIED_CONTROLLABILITY_LANGUAGE_RED " + ",".join(failed))
        print(json.dumps({
            "initial_a": initial_a,
            "initial_b": initial_b,
            "familiar": FAMILIAR,
            "deep": deep.identity,
            "global_control_history_q16": adult.credit.row(deep.identity).control_history_q16,
            "checks": checks,
        }, sort_keys=True, indent=2))
        return 1

    print("FOUNDRY_AGI_CONTEXT_QUALIFIED_CONTROLLABILITY_LANGUAGE_GREEN")
    print(json.dumps({
        "reference_only": True,
        "adult_owned_competition": True,
        "contexts_are_opaque_lived_identities": True,
        "initial": {"a": initial_a, "b": initial_b},
        "recovered_b": recovered_b,
        "lesioned": {"a": lesioned_a, "b": lesioned_b},
        "lesion_recovered_b": lesion_recovered_b,
        "language": {
            "familiar_bytes": len(familiar_surface),
            "deep_bytes": len(deep_surface),
            "deep_depth": deep.depth,
        },
        "checks": checks,
        "remaining_red": [
            "CURRENT_BODY_RESOURCE_STATE_INDEPENDENT_VARIATION",
            "SOCIAL_SOURCE_CONTEXT_INDEPENDENT_VARIATION",
            "DIRECT_AUTHENTICATED_PARITY",
        ],
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
