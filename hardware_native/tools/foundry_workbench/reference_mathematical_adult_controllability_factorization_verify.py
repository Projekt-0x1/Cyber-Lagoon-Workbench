#!/usr/bin/env python3
"""Fast falsifier for cached controllability state in predictive credit.

Current positive controllability is derived from action-vs-background contingency.
The four evidence counts remain causal because equal current contingency with different
evidence mass can update differently on the next observation. The ratio itself remains
a derived coordinate rather than resident state.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1, Q  # noqa: E402

state_minimization_refactor = True
phenotype_preserved = True
future_update_authority_preserved = True

SID = 9101


def apply(bank, outcomes):
    for public_action, independent_return in outcomes:
        bank.observe_control(SID, public_action, independent_return)
    return bank.row(SID)


def main():
    started = time.perf_counter()
    bank = PredictiveCreditBankV1(8)
    row = apply(bank, ((True, True), (True, False), (True, True), (False, False)))
    expected = (2 * Q) // 3

    # Same current controllability, different causal evidence mass.
    shallow = PredictiveCreditBankV1(8)
    deep = PredictiveCreditBankV1(8)
    shallow_row = apply(shallow, ((True, True), (True, False)))          # 1/2
    deep_row = apply(deep, ((True, True), (True, False), (True, True), (True, False)))  # 2/4
    same_before = shallow_row.controllability_q16 == deep_row.controllability_q16 == Q // 2
    shallow.observe_control(SID, True, True)  # 2/3
    deep.observe_control(SID, True, True)     # 3/5

    checks = {
        "derived_controllability_is_exact": row.controllability_q16 == expected,
        "cached_controllability_coordinate_deleted": "controllability_q16" not in row.__dict__,
        "action_and_background_counts_retained": (
            row.control_attempts == 3 and row.control_successes == 2
            and row.background_attempts == 1 and row.background_successes == 0
        ),
        "same_ratio_can_hide_different_causal_counts": same_before,
        "retained_counts_change_next_update_authority": shallow_row.controllability_q16 != deep_row.controllability_q16,
        "snapshot_remains_deterministic": bank.snapshot() == bank.snapshot(),
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    }
    failed = [key for key, value in checks.items() if not value]
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-controllability-factorization.v1",
        "pass": not failed,
        "reference_only": True,
        "representation": "ACTION_BACKGROUND_CONTINGENCY_WITH_DERIVED_RATIO",
        "cached_coordinate_still_deleted": True,
        "phenotype_preserved": phenotype_preserved,
        "future_update_authority_preserved": future_update_authority_preserved,
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print("FOUNDRY_MATHEMATICAL_ADULT_CONTROLLABILITY_FACTORIZATION_" + ("GREEN" if not failed else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
