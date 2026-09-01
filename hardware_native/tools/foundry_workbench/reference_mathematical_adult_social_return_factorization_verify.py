#!/usr/bin/env python3
"""Fast falsifier for lifetime social-return event logs.

The social predictor reads action-return history only through a source-filtered
arithmetic mean. Event order is therefore not causal, but repetition count is: a
source that produced 4,096 lived returns must outweigh a source that produced one.
The sufficient state is source -> (sum, count), not one row per event and not naive
source deduplication.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_social_latent_prediction_v1 import SocialLatentPredictionV1  # noqa: E402

FEATURES = (11, 12, 13)
ACTION_A = 7001
ACTION_B = 7002
RETURN_SOURCE_A = 8001
RETURN_SOURCE_B = 8002
RETURN_SOURCE_C = 8003
REPEATS = 4096


class CountingList(list):
    def __init__(self, values):
        super().__init__(values); self.touches = 0
    def __iter__(self):
        for value in super().__iter__():
            self.touches += 1
            yield value


class CountingMap(dict):
    def __init__(self, values):
        super().__init__(values); self.touches = 0
    def items(self):
        for item in super().items():
            self.touches += 1
            yield item
    def values(self):
        for value in super().values():
            self.touches += 1
            yield value


def instrument_row(social, key):
    row = social._action_returns[key]
    if isinstance(row, dict):
        counted = CountingMap(row)
    else:
        counted = CountingList(row)
    social._action_returns[key] = counted
    return counted


def main():
    started = time.perf_counter()
    social = SocialLatentPredictionV1(min_support=2)
    h1 = social.observe_history(FEATURES, 1001)
    h2 = social.observe_history(FEATURES, 1002)
    if h1 != h2:
        raise RuntimeError("social-return-factorization:hypothesis")

    for _ in range(REPEATS):
        social.observe_action_return(h1, ACTION_A, 4, RETURN_SOURCE_A, True)
    social.observe_action_return(h1, ACTION_A, -10, RETURN_SOURCE_B, True)
    for _ in range(4):
        social.observe_action_return(h1, ACTION_B, 1, RETURN_SOURCE_C, True)

    key_a = (h1, ACTION_A)
    key_b = (h1, ACTION_B)
    row_a = instrument_row(social, key_a)
    before_a = social.expected_return(h1, ACTION_A)
    return_touches = int(row_a.touches)
    before_b = social.expected_return(h1, ACTION_B)
    before_choice = social.choose(h1, (ACTION_A, ACTION_B))

    expected_a = (REPEATS * 4 - 10) / (REPEATS + 1)
    naive_source_dedup_mean = (4 - 10) / 2

    # Event order is not read by the social choice law. A reversed chronology with
    # identical source-qualified multiplicities must yield the same current mean.
    reordered = SocialLatentPredictionV1(min_support=2)
    rh = reordered.observe_history(FEATURES, 1001);reordered.observe_history(FEATURES, 1002)
    reordered.observe_action_return(rh, ACTION_A, -10, RETURN_SOURCE_B, True)
    for _ in range(REPEATS):reordered.observe_action_return(rh, ACTION_A, 4, RETURN_SOURCE_A, True)
    for _ in range(4):reordered.observe_action_return(rh, ACTION_B, 1, RETURN_SOURCE_C, True)
    reordered_before_a = reordered.expected_return(rh, ACTION_A)
    reordered_before_choice = reordered.choose(rh, (ACTION_A, ACTION_B))

    social.withdraw_source(RETURN_SOURCE_A)
    after_withdraw_a = social.expected_return(h1, ACTION_A)
    after_withdraw_b = social.expected_return(h1, ACTION_B)
    after_withdraw_choice = social.choose(h1, (ACTION_A, ACTION_B))
    reordered.withdraw_source(RETURN_SOURCE_A)
    reordered_after_a = reordered.expected_return(rh, ACTION_A)
    reordered_after_choice = reordered.choose(rh, (ACTION_A, ACTION_B))

    stored_a = social._action_returns[key_a]
    stored_b = social._action_returns[key_b]
    aggregate_shape = isinstance(stored_a, dict) and isinstance(stored_b, dict)
    source_rows_a = len(stored_a)
    source_rows_b = len(stored_b)

    checks = {
        "event_weighted_mean_is_exact": abs(before_a - expected_a) < 1e-12,
        "repetition_count_is_causal_not_naively_deduplicable": abs(before_a - naive_source_dedup_mean) > 1.0,
        "prewithdrawal_choice_uses_repeated_lived_success": before_choice == ACTION_A and before_a > before_b,
        "event_order_is_future_irrelevant": reordered_before_a == before_a and reordered_before_choice == before_choice,
        "source_withdrawal_exposes_remaining_counterevidence": after_withdraw_a == -10 and after_withdraw_b == 1,
        "source_withdrawal_flips_action_choice": after_withdraw_choice == ACTION_B,
        "withdrawal_future_is_order_invariant": reordered_after_a == after_withdraw_a and reordered_after_choice == after_withdraw_choice,
        "source_qualified_aggregate_representation": aggregate_shape,
        "same_source_repetition_collapses_to_one_source_row": source_rows_a == 2 and source_rows_b == 1,
        "expected_return_touches_sources_not_events": return_touches <= 2,
        "event_log_would_be_large": REPEATS + 1 > 4000,
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    }
    failed = [key for key, value in checks.items() if not value]
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-social-return-factorization.v1",
        "pass": not failed,
        "reference_only": True,
        "representation": "ACTION_HYPOTHESIS_TO_SOURCE_SUM_COUNT",
        "economic_gain": True,
        "state_reduction": {"action_a_event_rows_before": REPEATS + 1, "action_a_source_rows_after": source_rows_a},
        "work_reduction": {"expected_return_event_touches_before": REPEATS + 1, "expected_return_source_touches_after": return_touches},
        "repeated_returns": REPEATS,
        "stored_source_rows_action_a": source_rows_a,
        "stored_source_rows_action_b": source_rows_b,
        "expected_return_touches": return_touches,
        "before": {"action_a": before_a, "action_b": before_b, "choice": before_choice},
        "after_withdrawal": {"action_a": after_withdraw_a, "action_b": after_withdraw_b, "choice": after_withdraw_choice},
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print("FOUNDRY_MATHEMATICAL_ADULT_SOCIAL_RETURN_FACTORIZATION_" + ("GREEN" if not failed else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
