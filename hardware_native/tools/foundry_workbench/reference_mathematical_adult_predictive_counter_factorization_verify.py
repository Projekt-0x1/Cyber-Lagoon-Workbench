#!/usr/bin/env python3
"""Fast falsifier for duplicate predictive-credit use counters.

A Profile has one observe-use chronology. `exposures`, `duration_samples`, and
`effort_samples` therefore advance together; only exposures has separate future
retention authority. Context exposure count is observer-only. The candidate keeps
one causal count and derives both online mean denominators from it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1, Q, ema  # noqa: E402

state_minimization_refactor = True
phenotype_preserved = True
future_update_authority_preserved = True

SID = 7101
USES = 257
RETURNS = 193


def donor_means():
    duration_mean = 0
    effort_mean = 0
    duration_error = 0
    effort_error = 0
    duration_samples = 0
    effort_samples = 0
    for n in range(USES):
        duration = (2 + n % 11) * Q
        effort = ((n % 7) + 1) * Q // 16
        old_duration = duration_mean
        duration_samples += 1
        duration_mean = ema(duration_mean, duration, duration_samples)
        duration_error = ema(duration_error, abs(duration - old_duration), duration_samples)
        old_effort = effort_mean
        effort_samples += 1
        effort_mean = ema(effort_mean, effort, effort_samples)
        effort_error = ema(effort_error, abs(effort - old_effort), effort_samples)
    return duration_mean, effort_mean, duration_error, effort_error


def donor_return_means():
    outcome_mean = 0
    somatic_mean = 0
    prediction_error = 0
    for n in range(RETURNS):
        outcome = ((n % 9) - 4) * Q // 8
        somatic = ((n % 5) - 2) * Q // 16
        old_outcome = outcome_mean
        count = n + 1
        outcome_mean = ema(outcome_mean, outcome, count)
        prediction_error = ema(prediction_error, abs(outcome - old_outcome), count)
        somatic_mean = ema(somatic_mean, somatic, count)
    return outcome_mean, somatic_mean, prediction_error


def populate(bank):
    tick = 10
    for n in range(USES):
        duration = 2 + n % 11
        effort = ((n % 7) + 1) * Q // 16
        bank.observe_use(SID, tick, tick + duration, effort, context=100 + n % 5)
        tick += duration + 2
    for n in range(RETURNS):
        outcome = ((n % 9) - 4) * Q // 8
        somatic = ((n % 5) - 2) * Q // 16
        bank.observe_return(SID, outcome, somatic, tick + n, True, context=200 + n % 3)
    return bank.row(SID)


def eviction_victim(cut_exposures=False):
    bank = PredictiveCreditBankV1(2)
    for n in range(5):bank.observe_use(8101, 10 + n * 3, 11 + n * 3, Q // 16)
    bank.observe_use(8102, 100, 101, Q // 16)
    if cut_exposures:bank.row(8101).exposures = 0
    bank.row(8103)
    return bank.last_evicted


def return_eviction_victim(cut_outcome_samples=False):
    bank = PredictiveCreditBankV1(2)
    bank.observe_use(8201, 10, 11, Q // 16);bank.observe_use(8202, 20, 21, Q // 16)
    for n in range(5):bank.observe_return(8201, Q // 4, Q // 16, 30 + n, True)
    bank.observe_return(8202, Q // 4, Q // 16, 40, True)
    if cut_outcome_samples:bank.row(8201).outcome_samples = 0
    bank.row(8203)
    return bank.last_evicted


def main():
    started = time.perf_counter()
    bank = PredictiveCreditBankV1(32)
    row = populate(bank)
    donor_duration, donor_effort, donor_duration_error, donor_effort_error = donor_means()
    donor_outcome, donor_somatic, donor_prediction_error = donor_return_means()
    contexts = tuple(row.contexts.values())
    snapshot = bank.snapshot()

    intact_victim = eviction_victim(False)
    cut_victim = eviction_victim(True)
    intact_return_victim = return_eviction_victim(False)
    cut_return_victim = return_eviction_victim(True)

    checks = {
        "one_causal_use_count_retained": row.exposures == USES,
        "duplicate_duration_counter_deleted": not hasattr(row, "duration_samples"),
        "duplicate_effort_counter_deleted": not hasattr(row, "effort_samples"),
        "context_observer_exposure_deleted": all(not hasattr(context, "exposures") for context in contexts),
        "one_causal_return_count_retained": row.outcome_samples == RETURNS,
        "duplicate_profile_somatic_counter_deleted": not hasattr(row, "somatic_samples"),
        "duplicate_context_somatic_counter_deleted": all(not hasattr(context, "somatic_samples") for context in contexts),
        "duration_mean_exactly_preserved": row.duration_mean_q16 == donor_duration,
        "effort_mean_exactly_preserved": row.effort_mean_q16 == donor_effort,
        "duration_error_exactly_preserved": row.duration_abs_error_q16 == donor_duration_error,
        "effort_error_exactly_preserved": row.effort_abs_error_q16 == donor_effort_error,
        "outcome_mean_exactly_preserved": row.outcome_mean_q16 == donor_outcome,
        "somatic_mean_exactly_preserved": row.somatic_mean_q16 == donor_somatic,
        "prediction_error_exactly_preserved": row.prediction_error_q16 == donor_prediction_error,
        "snapshot_remains_available": bool(snapshot),
        "exposures_negative_control_is_causal": intact_victim == 8102 and cut_victim == 8101,
        "outcome_samples_negative_control_is_causal": intact_return_victim == 8202 and cut_return_victim == 8201,
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    }
    failed = [key for key, value in checks.items() if not value]
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-predictive-counter-factorization.v1",
        "pass": not failed,
        "reference_only": True,
        "representation": "ONE_CAUSAL_USE_COUNT_PLUS_ONE_CAUSAL_RETURN_COUNT",
        "economic_gain": True,
        "state_reduction": {
            "profile_use_counters_before": 3,
            "profile_use_counters_after": 1,
            "profile_return_counters_before": 2,
            "profile_return_counters_after": 1,
            "context_observer_use_counters_before": 5,
            "context_observer_use_counters_after": 0,
            "context_return_counters_before_per_active_context": 2,
            "context_return_counters_after_per_active_context": 1,
        },
        "uses": USES,
        "returns": RETURNS,
        "contexts": len(contexts),
        "retained_causal_counters": ["exposures", "outcome_samples"],
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print("FOUNDRY_MATHEMATICAL_ADULT_PREDICTIVE_COUNTER_FACTORIZATION_" + ("GREEN" if not failed else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
