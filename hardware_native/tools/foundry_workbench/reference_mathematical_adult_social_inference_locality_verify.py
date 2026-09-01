#!/usr/bin/env python3
"""Fast falsifier for global social-latent inference scans.

Observed features should nominate only hypotheses that share learned evidence. Features
that occur in no learned social hypothesis are irrelevant, while cues supporting a
real competing hypothesis remain causal ambiguity.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_social_latent_prediction_v1 import SocialLatentPredictionV1  # noqa: E402

HYPOTHESES = 4096
A = (11, 12, 13)
B = (21, 22, 23)
NOISE = tuple(range(9001, 9033))


def main():
    started = time.perf_counter()
    social = SocialLatentPredictionV1(min_support=2)
    for source in (1001, 1002):
        aid = social.observe_history(A, source)
    for source in (2001, 2002):
        bid = social.observe_history(B, source)
    for i in range(HYPOTHESES - 2):
        features = (100_000 + i * 3, 100_001 + i * 3, 100_002 + i * 3)
        social.observe_history(features, 300_000 + i * 2)
        social.observe_history(features, 300_001 + i * 2)

    noisy = social.infer((11, 12, *NOISE))
    noisy_touches = int(social.last_touches)
    clean = social.infer((11, 12))
    clean_touches = int(social.last_touches)
    competing = social.infer((11, 21))
    competing_touches = int(social.last_touches)
    absent = social.infer(NOISE)
    absent_touches = int(social.last_touches)

    checks = {
        "strong_target_survives_irrelevant_cues": noisy == aid,
        "clean_target_still_resolves": clean == aid,
        "real_competing_cue_remains_ambiguous": competing == 0,
        "unlearned_cues_remain_unresolved": absent == 0,
        "noisy_inference_touches_only_nominated_hypotheses": noisy_touches <= 2,
        "clean_inference_touches_only_target": clean_touches <= 1,
        "competing_inference_touches_only_real_competitors": competing_touches <= 2,
        "unlearned_cues_touch_no_hypotheses": absent_touches == 0,
        "supported_hypothesis_scale_is_large": len(social.hypotheses()) == HYPOTHESES,
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    }
    failed = [key for key, value in checks.items() if not value]
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-social-inference-locality.v1",
        "pass": not failed,
        "reference_only": True,
        "hypotheses": HYPOTHESES,
        "representation": "FEATURE_INDEX_NOMINATES_SOCIAL_HYPOTHESES",
        "touches": {
            "noisy_target": noisy_touches,
            "clean_target": clean_touches,
            "competing": competing_touches,
            "unlearned": absent_touches,
        },
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print("FOUNDRY_MATHEMATICAL_ADULT_SOCIAL_INFERENCE_LOCALITY_" + ("GREEN" if not failed else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
