#!/usr/bin/env python3
"""Fast causal test for retiring chunk quorum state after program admission.

Pre-quorum observation count is future-causal: at 2/3, one more lived use admits the
chunk. Once the chunk exists, that scalar has no remaining transition read and must
not survive as lifelong Adult state.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_causal_program_chunk_v1 import CausalChunkBankV1, CHUNK_QUORUM, Q  # noqa: E402

MEMBERS = (11, 22, 33, 44)
ALT = (55, 66)


def observe(bank, n, members=MEMBERS, outcome=Q // 2):
    start = 10 + n * 7
    return bank.observe(members, start, start + 4, Q // 4, outcome, Q // 16, successor=77)


def main():
    started = time.perf_counter()

    pre = CausalChunkBankV1()
    for n in range(CHUNK_QUORUM - 1):
        assert observe(pre, n) is None
    pre_count = pre.counts.get(MEMBERS)

    intact = copy.deepcopy(pre)
    cut = copy.deepcopy(pre)
    cut.counts.pop(MEMBERS, None)
    intact_chunk = observe(intact, 10)
    cut_chunk = observe(cut, 10)

    earned = CausalChunkBankV1()
    chunk = None
    for n in range(CHUNK_QUORUM):
        chunk = observe(earned, n)
    if chunk is None:
        raise RuntimeError("chunk-quorum-state:admission")
    count_after_admission = earned.counts.get(MEMBERS)
    predictive_after_admission = copy.deepcopy(earned.predictive.snapshot())

    for n in range(CHUNK_QUORUM, CHUNK_QUORUM + 512):
        assert observe(earned, n) is not None
    count_after_reuse = earned.counts.get(MEMBERS)

    # The admitted program still has ordinary future plasticity and can compose into
    # a higher program; deleting only its spent quorum scalar must not freeze it.
    before_duration = earned.predictive.row(chunk.identity).duration_mean_q16
    before_value = earned.predictive.row(chunk.identity).outcome_mean_q16
    earned.devalue(chunk.identity, -2 * Q)
    devalued = earned.predictive.row(chunk.identity).outcome_mean_q16 < before_value
    observe(earned, 1000, outcome=Q)
    recalibrated = earned.predictive.row(chunk.identity).duration_mean_q16 >= before_duration
    alt = None
    for n in range(CHUNK_QUORUM):
        alt = observe(earned, 1100 + n, ALT)
    higher = None
    for n in range(CHUNK_QUORUM):
        higher = observe(earned, 1200 + n, (chunk.identity, alt.identity))

    checks = {
        "pre_quorum_count_retained": pre_count == CHUNK_QUORUM - 1,
        "pre_quorum_count_is_causal": intact_chunk is not None and cut_chunk is None,
        "earned_chunk_exists": chunk.identity in earned.chunks,
        "spent_quorum_state_deleted_on_admission": count_after_admission is None,
        "repeated_earned_use_does_not_recreate_quorum_state": count_after_reuse is None,
        "predictive_state_exists_independently_of_quorum_count": bool(predictive_after_admission),
        "earned_program_remains_devaluable": devalued,
        "earned_program_remains_revisable": recalibrated,
        "earned_program_still_composes": higher is not None and higher.depth >= 2,
        "higher_program_also_retires_spent_quorum": (chunk.identity, alt.identity) not in earned.counts,
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    }
    failed = [key for key, value in checks.items() if not value]
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-chunk-quorum-state-deletion.v1",
        "pass": not failed,
        "reference_only": True,
        "state_boundary": "PRE_QUORUM_COUNT_CAUSAL_POST_QUORUM_COUNT_SPENT",
        "quorum": CHUNK_QUORUM,
        "reuse_observations": 512,
        "count_before_admission": pre_count,
        "count_after_admission": count_after_admission,
        "count_after_reuse": count_after_reuse,
        "pending_count_rows_after_higher_admission": len(earned.counts),
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }
    print("FOUNDRY_MATHEMATICAL_ADULT_CHUNK_QUORUM_STATE_DELETION_" + ("GREEN" if not failed else "RED"))
    print(json.dumps(result, indent=2, sort_keys=True, default=list))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
