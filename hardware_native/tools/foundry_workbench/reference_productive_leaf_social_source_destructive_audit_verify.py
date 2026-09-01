#!/usr/bin/env python3
"""Destructive social/source-history audit over fixed productive-language factors.

Reference-only. This adds no Adult mechanism. The exact productive surface/program
checkpoint and current partner cues are held fixed while lived partner-return history
or active source authority changes resident social action competition.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_language_mastery_adult_v1 import LanguageMasteryAdultV1  # noqa: E402
from reference_mathematical_adult_operator_factorization_verify import build  # noqa: E402
from reference_predictive_credit_profile_v1 import Q  # noqa: E402

semantics_free_maintenance = True
phenotype_preserved = True
future_update_authority_preserved = True
language_phenotype_improved = True

CTX = 0x5A01
SHORT = 0x5A02
PARTNER_FEATURES = (11, 12, 13)
CURRENT_PARTNER_CUES = (11, 12)
SOURCE_PARTNER_FEATURES = (31, 32, 33)
SOURCE_CURRENT_CUES = (31, 32)


def surface_state(adult: LanguageMasteryAdultV1) -> str:
    return json.dumps(adult.program_surface_checkpoint(), sort_keys=True, separators=(",", ":"))


def make_partner_history(adult: LanguageMasteryAdultV1, features, source_base: int) -> int:
    hypotheses = [adult.observe_social_history(features, source_base + i) for i in range(3)]
    if len(set(hypotheses)) != 1 or hypotheses[0] == 0:
        raise RuntimeError("social_source_audit:partner_hypothesis")
    return int(hypotheses[0])


def social_history_branch(base_checkpoint, deep_id: int, short_return: int, deep_return: int):
    adult = LanguageMasteryAdultV1.restore(copy.deepcopy(base_checkpoint))
    hypothesis = make_partner_history(adult, PARTNER_FEATURES, 1001)
    for n in range(3):
        adult.settle_social_action(hypothesis, SHORT, short_return, 3000 + n, True)
        adult.settle_social_action(hypothesis, deep_id, deep_return, 3100 + n, True)
    choice, inferred = adult.choose_social(CURRENT_PARTNER_CUES, (SHORT, deep_id), 0)
    return adult, hypothesis, int(choice), int(inferred)


def active_return_sources(social_checkpoint) -> set[int]:
    return {
        int(source["identity"])
        for row in social_checkpoint.get("action_returns", ())
        for source in row.get("sources", ())
    }


def main() -> int:
    started = time.perf_counter()
    seed, leaves, _deep, top, _programs, _roots = build(True)
    seed.experience_atomic_program(SHORT, leaves[0], Q // 2, Q // 16, CTX, Q // 16, True)
    base_checkpoint = copy.deepcopy(seed.checkpoint())
    base_surface_state = surface_state(seed)
    short_surface = seed.public_surface(SHORT)
    deep_surface = seed.public_surface(top.identity)

    # Same current partner cues and same inferred partner identity. Only lived social
    # action-return history differs, so the resident social action competition must flip.
    compact_history, compact_h, compact_choice, compact_inferred = social_history_branch(
        base_checkpoint, top.identity, +3, +1
    )
    explicit_history, explicit_h, explicit_choice, explicit_inferred = social_history_branch(
        base_checkpoint, top.identity, -1, +3
    )

    # A non-independent/yoked return is not authenticated social consequence and cannot
    # revise the winner even when its magnitude is extreme.
    yoked = LanguageMasteryAdultV1.restore(copy.deepcopy(compact_history.checkpoint()))
    yoked_before = yoked.social.expected_return(compact_h, top.identity)
    yoked_accepted = yoked.settle_social_action(compact_h, top.identity, +100, 9991, False)
    yoked_after = yoked.social.expected_return(compact_h, top.identity)
    yoked_choice, yoked_inferred = yoked.choose_social(
        CURRENT_PARTNER_CUES, (SHORT, top.identity), 0
    )

    # Checkpoint is causal history, not a transient conversational cache.
    explicit_replay = LanguageMasteryAdultV1.restore(copy.deepcopy(explicit_history.checkpoint()))
    replay_choice, replay_inferred = explicit_replay.choose_social(
        CURRENT_PARTNER_CUES, (SHORT, top.identity), 0
    )

    # Same current source-partner cues and same stored return history. Two active sources
    # disagree. Source 5001 dominates the pre-withdrawal average toward SHORT; withdrawing
    # only its authority exposes source 5002 and flips the resident winner to the deep program.
    source_history = LanguageMasteryAdultV1.restore(copy.deepcopy(base_checkpoint))
    source_h = make_partner_history(source_history, SOURCE_PARTNER_FEATURES, 4001)
    for _ in range(2):
        source_history.settle_social_action(source_h, SHORT, +5, 5001, True)
        source_history.settle_social_action(source_h, top.identity, -2, 5001, True)
    source_history.settle_social_action(source_h, SHORT, -2, 5002, True)
    source_history.settle_social_action(source_h, top.identity, +3, 5002, True)
    source_before_choice, source_before_inferred = source_history.choose_social(
        SOURCE_CURRENT_CUES, (SHORT, top.identity), 0
    )
    source_before_scores = (
        source_history.social.expected_return(source_h, SHORT),
        source_history.social.expected_return(source_h, top.identity),
    )
    source_history.social.withdraw_source(5001)
    source_after_choice, source_after_inferred = source_history.choose_social(
        SOURCE_CURRENT_CUES, (SHORT, top.identity), 0
    )
    source_after_scores = (
        source_history.social.expected_return(source_h, SHORT),
        source_history.social.expected_return(source_h, top.identity),
    )
    source_checkpoint = copy.deepcopy(source_history.social.checkpoint())
    source_replay = LanguageMasteryAdultV1.restore(copy.deepcopy(source_history.checkpoint()))
    source_replay_choice, source_replay_inferred = source_replay.choose_social(
        SOURCE_CURRENT_CUES, (SHORT, top.identity), 0
    )

    branches = (
        compact_history,
        explicit_history,
        yoked,
        explicit_replay,
        source_history,
        source_replay,
    )
    exact_surface_state = all(surface_state(adult) == base_surface_state for adult in branches)
    exact_public_surfaces = all(
        adult.public_surface(SHORT) == short_surface
        and adult.public_surface(top.identity) == deep_surface
        for adult in branches
    )

    checks = {
        "same_current_partner_cues_different_social_history_changes_winner": (
            compact_h == explicit_h == compact_inferred == explicit_inferred
            and compact_choice == SHORT
            and explicit_choice == top.identity
        ),
        "social_choice_is_resident_competition_over_same_candidates": (
            compact_history.social.last_choose_touches == 2
            and explicit_history.social.last_choose_touches == 2
        ),
        "yoked_social_return_cannot_revise_competition": (
            not yoked_accepted
            and yoked_before == yoked_after
            and yoked_inferred == compact_h
            and yoked_choice == SHORT
        ),
        "checkpoint_preserves_social_history_choice": (
            replay_inferred == explicit_h and replay_choice == top.identity
        ),
        "source_authority_withdrawal_changes_winner_without_new_return": (
            source_before_inferred == source_after_inferred == source_h
            and source_before_choice == SHORT
            and source_after_choice == top.identity
            and source_before_scores[0] > source_before_scores[1]
            and source_after_scores[0] < source_after_scores[1]
        ),
        "withdrawal_preserves_source_history_but_changes_active_authority": (
            active_return_sources(source_checkpoint) == {5001, 5002}
            and set(map(int, source_checkpoint.get("withdrawn", ()))) == {5001}
        ),
        "checkpoint_preserves_source_withdrawal_effect": (
            source_replay_inferred == source_h and source_replay_choice == top.identity
        ),
        "social_source_variation_never_rewrites_productive_surface_state": exact_surface_state,
        "social_source_variation_never_rewrites_public_surfaces": exact_public_surfaces,
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema": "cyber-lagoon.reference-productive-leaf-social-source-destructive-audit.v1",
        "pass": not failed,
        "reference_only": True,
        "mechanism_change": False,
        "semantics_free_maintenance": semantics_free_maintenance,
        "phenotype_preserved": phenotype_preserved,
        "future_update_authority_preserved": future_update_authority_preserved,
        "language_phenotype_improved": language_phenotype_improved,
        "visible_language_gain": "SAME_CURRENT_PARTNER_CUES_HISTORY_SWITCHES_FACTORED_SHORT_VS_DEEP_PROGRAM",
        "additional_destructive_axes_paid": [
            "social_partner_return_history",
            "authenticated_social_consequence",
            "source_authority_withdrawal",
        ],
        "same_current_partner_cues": list(CURRENT_PARTNER_CUES),
        "social_hypothesis": compact_h,
        "short_program": SHORT,
        "deep_program": int(top.identity),
        "short_bytes": len(short_surface),
        "deep_bytes": len(deep_surface),
        "source_scores_before_withdrawal": list(source_before_scores),
        "source_scores_after_withdrawal": list(source_after_scores),
        "checks": checks,
        "remaining_red": [
            "PERSISTENT_BODY_RESOURCE_HISTORY_BEYOND_CURRENT_ADULT_STATE",
            "LEARNED_SOURCE_RELIABILITY_BEYOND_EXPLICIT_WITHDRAWAL",
            "DIRECT_CUDA_PARITY",
        ],
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print("FOUNDRY_PRODUCTIVE_LEAF_SOCIAL_SOURCE_DESTRUCTIVE_AUDIT_" + ("GREEN" if not failed else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
