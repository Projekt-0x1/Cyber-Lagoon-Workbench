#!/usr/bin/env python3
"""Fast falsifier for duplicate cached support in social latent prediction.

Social hypothesis support is exactly |feature_sources - withdrawn_sources|. Persisting
that scalar separately makes every source withdrawal rescan all hypotheses just to
refresh a value already derivable at the point of inference.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_social_latent_prediction_v1 import SocialLatentPredictionV1  # noqa: E402

HYPOTHESES = 4096
TARGET_FEATURES = (11, 12, 13)
TARGET_SOURCES = (1001, 1002)
ACTION_GOOD = 7001
ACTION_BAD = 7002


class CountingFeatureSets(dict):
    def __init__(self, source):
        super().__init__(source)
        self.item_touches = 0

    def items(self):
        for row in super().items():
            self.item_touches += 1
            yield row


def build():
    social = SocialLatentPredictionV1(min_support=2)
    social.observe_history(TARGET_FEATURES, TARGET_SOURCES[0])
    social.observe_history(TARGET_FEATURES, TARGET_SOURCES[1])
    target = social.infer(TARGET_FEATURES)
    if not target:
        raise RuntimeError("social-support-factorization:no_target")
    social.observe_action_return(target, ACTION_GOOD, 10, 2001, True)
    social.observe_action_return(target, ACTION_GOOD, 6, 2002, True)
    social.observe_action_return(target, ACTION_BAD, -4, 2003, True)
    for i in range(HYPOTHESES - 1):
        features = (100_000 + i * 3, 100_001 + i * 3, 100_002 + i * 3)
        social.observe_history(features, 300_000 + i * 2)
        social.observe_history(features, 300_001 + i * 2)
    return social, target


def projection(social):
    return {
        "hypotheses": tuple((h.identity, h.features, h.support) for h in social.hypotheses()),
        "target_infer": social.infer(TARGET_FEATURES),
        "target_choice": social.choose(social.infer(TARGET_FEATURES), (ACTION_GOOD, ACTION_BAD)),
        "target_good_return": social.expected_return(social.infer(TARGET_FEATURES), ACTION_GOOD),
        "target_bad_return": social.expected_return(social.infer(TARGET_FEATURES), ACTION_BAD),
    }


def main():
    started = time.perf_counter()
    social, target = build()
    before = projection(social)
    live_cached_support_rows = len(getattr(social, "_support", {}))

    counted = CountingFeatureSets(social._feature_sets)
    social._feature_sets = counted
    social.withdraw_source(TARGET_SOURCES[0])
    withdrawal_touches = counted.item_touches
    one_withdrawn = projection(social)
    social.withdraw_source(TARGET_SOURCES[1])
    two_withdrawn = projection(social)

    # Negative control: the source-qualified feature evidence itself is causal. If it
    # is removed rather than merely deriving support from it, the target disappears.
    control, _ = build()
    control._feature_sets.pop(TARGET_FEATURES, None)
    if hasattr(control, "_support"):
        control._support.pop(TARGET_FEATURES, None)
    evidence_cut = projection(control)

    checks = {
        "target_exists_before_withdrawal": before["target_infer"] == target,
        "choice_uses_return_evidence": before["target_choice"] == ACTION_GOOD,
        "one_source_withdrawal_drops_below_support": one_withdrawn["target_infer"] == 0,
        "second_withdrawal_remains_absent": two_withdrawn["target_infer"] == 0,
        "source_evidence_negative_control_differs": evidence_cut != before,
        "derived_support_cache_deleted": not hasattr(social, "_support") and live_cached_support_rows == 0,
        "withdrawal_has_no_global_hypothesis_rescan": withdrawal_touches == 0,
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    }
    failed = [key for key, value in checks.items() if not value]
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-social-support-factorization.v1",
        "pass": not failed,
        "reference_only": True,
        "representation": "FEATURE_SOURCE_SETS_PLUS_WITHDRAWN_SET_DERIVE_SUPPORT_ON_QUERY",
        "hypotheses": HYPOTHESES,
        "legacy_duplicate_support_rows": HYPOTHESES,
        "live_cached_support_rows": live_cached_support_rows,
        "withdrawal_feature_set_touches": withdrawal_touches,
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print("FOUNDRY_MATHEMATICAL_ADULT_SOCIAL_SUPPORT_FACTORIZATION_" + ("GREEN" if not failed else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
