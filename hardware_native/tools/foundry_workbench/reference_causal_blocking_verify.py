#!/usr/bin/env python3
"""Sequential blocking/unblocking controls for the matched credit ecology."""
from __future__ import annotations

import json
import time

from reference_causal_attribution_ecology_v1 import (
    CausalAttributionEcologyV1, DEFAULT_RESIDENT_CAPACITY, Refuse)


A, N, B = 11, 17, 29
SOURCES = (501, 502)


def refuse(call, prefix):
    try:
        call()
    except Refuse as exc:
        return str(exc).startswith(prefix)
    return False


def add_rows(ecology, receipt, effects, sources=SOURCES, reverse=False):
    rows = [(source, coalition, effect)
            for source in sources for coalition, effect in effects.items()]
    if reverse:
        rows.reverse()
    for source, coalition, effect in rows:
        occurrence = ecology.population.recruit(coalition)
        ecology.consequence(receipt, occurrence, source, coalition, effect, True)


def establish(ecology, effect=10, neutral=N, reverse=False):
    receipt = ecology.participate((A, neutral), 128)
    add_rows(ecology, receipt, {
        (): 0, (A,): effect, (neutral,): 0, tuple(sorted((A, neutral))): effect,
    }, reverse=reverse)
    return ecology.resolve(receipt)


def extend(ecology, joint_effect, *, established_effect=10,
           sources=SOURCES, reverse=False):
    receipt = ecology.participate((A, B), 128)
    add_rows(ecology, receipt, {
        (): 0, (A,): established_effect, (A, B): joint_effect,
    }, sources=sources, reverse=reverse)
    return receipt


def candidate(ecology, rows):
    selected = ecology.select(rows)
    return 0 if selected is None else selected.candidate


def trained_extension(joint_effect, reverse=False):
    ecology = CausalAttributionEcologyV1()
    prior = establish(ecology, reverse=reverse)
    receipt = extend(ecology, joint_effect, reverse=reverse)
    return ecology, prior, receipt, ecology.resolve_conditioned(receipt, A)


def main():
    started = time.perf_counter()
    checks = {}

    blocked, prior, blocked_receipt, blocked_result = trained_extension(10)
    checks["prior_causal_sufficiency_learned"] = (
        prior is not None and dict((p, n / d) for p, n, d in prior.participant_credit)[A] == 10)
    checks["blocking_assigns_zero_new_residual"] = (
        blocked_result is not None and blocked_result.prior_effect == 10
        and blocked_result.established_effect == 10
        and blocked_result.joint_effect == 10 and blocked_result.residual == 0)
    checks["blocked_newcomer_not_singleton_credit"] = (
        B not in blocked.credit
        and candidate(blocked, ((701, (A,)), (702, (B,)))) == 701)
    checks["blocked_compound_preserves_unresolved_tie"] = (
        candidate(blocked, ((701, (A,)), (703, (A, B)))) == 0)

    unblocked, _, unblocked_receipt, unblocked_result = trained_extension(14)
    checks["unblocking_is_conditional_residual"] = (
        unblocked_result is not None and unblocked_result.residual == 4
        and candidate(unblocked, ((701, (A,)), (703, (A, B)))) == 703
        and candidate(unblocked, ((701, (A,)), (702, (B,)))) == 701)

    downshifted, _, _, downshifted_result = trained_extension(6)
    checks["downshift_is_negative_conditional_not_global_countercredit"] = (
        downshifted_result is not None and downshifted_result.residual == -4
        and B not in downshifted.credit
        and candidate(downshifted, ((701, (A,)), (703, (A, B)))) == 701)

    naive = CausalAttributionEcologyV1()
    naive_receipt = extend(naive, 10)
    before = naive.state_hash()
    checks["no_prior_no_blocking_inference"] = (
        naive.resolve_conditioned(naive_receipt, A) is None
        and naive.state_hash() == before)

    drift = CausalAttributionEcologyV1()
    establish(drift)
    drift_receipt = extend(drift, 10, established_effect=9)
    before = drift.state_hash()
    checks["changed_established_effect_refuses_transfer"] = (
        drift.resolve_conditioned(drift_receipt, A) is None
        and drift.state_hash() == before)

    unstable = CausalAttributionEcologyV1()
    establish(unstable, 10, N)
    establish(unstable, 8, 19)
    unstable_receipt = extend(unstable, 10)
    checks["conflicting_prior_sufficiency_refuses"] = (
        unstable.resolve_conditioned(unstable_receipt, A) is None)

    one_source = CausalAttributionEcologyV1()
    establish(one_source)
    one_receipt = extend(one_source, 14, sources=(501,))
    before = one_source.state_hash()
    checks["two_source_partial_match_required"] = (
        one_source.resolve_conditioned(one_receipt, A) is None
        and one_source.state_hash() == before)

    reversed_ecology, _, _, reversed_result = trained_extension(14, True)
    checks["evidence_order_invariant"] = (
        reversed_result is not None
        and reversed_result.residual == unblocked_result.residual
        and candidate(reversed_ecology, ((701, (A,)), (703, (A, B)))) == 703)

    replay = CausalAttributionEcologyV1.restore(unblocked.checkpoint())
    checks["conditioned_checkpoint_exact"] = (
        replay.checkpoint() == unblocked.checkpoint()
        and replay.conditionals[unblocked_receipt] == unblocked_result
        and candidate(replay, ((701, (A,)), (703, (A, B)))) == 703)

    before = unblocked.state_hash()
    checks["conditioned_then_full_double_revision_refuses"] = (
        refuse(lambda: unblocked.resolve(unblocked_receipt),
               "conditioned attribution already resolved")
        and unblocked.state_hash() == before)

    withdrawn, _, _, withdrawn_result = trained_extension(14)
    before_choice = candidate(withdrawn, ((701, (A,)), (703, (A, B))))
    withdrawn.withdraw_source(501)
    checks["source_withdrawal_removes_conditional_revision"] = (
        withdrawn_result is not None and before_choice == 703
        and not withdrawn.conditionals
        and candidate(withdrawn, ((701, (A,)), (703, (A, B)))) == 0)

    quantity = replay.quantity()
    checks["bounded_sparse_history"] = (
        quantity["resident_capacity"] == DEFAULT_RESIDENT_CAPACITY
        and quantity["materialized_participants"] == 3
        and quantity["evidence_rows"] == 14
        and replay.conditionals[unblocked_receipt].touched_work == 6)

    body = json.loads(replay.checkpoint())
    body["conditionals"][0]["residual"] += 1
    checks["checkpoint_conditional_invariant_refusal"] = refuse(
        lambda: CausalAttributionEcologyV1.restore(json.dumps(body)),
        "checkpoint conditional attribution")

    elapsed_ms = (time.perf_counter() - started) * 1000
    checks["bounded_runtime"] = elapsed_ms < 5000
    receipt = {
        "schema": "agi.reference-causal-blocking.v1",
        "pass": all(checks.values()),
        "checks": checks,
        "behavior": {
            "prior_effect": 10,
            "blocked_residual": blocked_result.residual,
            "unblocked_residual": unblocked_result.residual,
            "downshifted_residual": downshifted_result.residual,
        },
        "quantity": quantity,
        "elapsed_ms": round(elapsed_ms, 3),
        "reference_only": True,
        "adult_attached": False,
        "runtime_llm": False,
        "graph_flip": False,
        "physical_direct_parity": "NOT_RUN/RED",
        "claim": "LEARNED_CAUSAL_SUFFICIENCY_BLOCKING_REFERENCE_PROPERTY_ONLY",
    }
    print("FOUNDRY_CAUSAL_BLOCKING " + ("GREEN" if receipt["pass"] else "RED"))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if receipt["pass"] else 1)


if __name__ == "__main__":
    main()
