#!/usr/bin/env python3
"""Controls for resident nomination of a matched causal intervention square."""
from __future__ import annotations

import json
import time

from reference_causal_attribution_ecology_v1 import (
    CausalAttributionEcologyV1, DEFAULT_RESIDENT_CAPACITY, Refuse)


PARTICIPANTS = (11, 29)
COALITIONS = ((), (11,), (29,), (11, 29))
EFFECTS = {(): 0, (11,): 2, (29,): 1, (11, 29): 3}


def refuse(call, prefix):
    try:
        call()
    except Refuse as exc:
        return str(exc).startswith(prefix)
    return False


def occurrence_by_identity(ecology, identity):
    return next(row for row in ecology.population.occurrences
                if row.identity == identity)


def main():
    started = time.perf_counter()
    checks = {}

    ecology = CausalAttributionEcologyV1()
    receipt = ecology.participate(PARTICIPANTS, 128)
    sequence = []
    no_credit_before_resolution = True
    bypass_refused = True
    for index in range(8):
        evidence_before_nomination = len(ecology.pending[receipt].evidence)
        nomination, occurrence = ecology.nominate_intervention(receipt, 16)
        sequence.append(nomination.coalition)
        no_credit_before_resolution &= (
            not ecology.credit
            and len(ecology.pending[receipt].evidence) == evidence_before_nomination)
        before = ecology.state_hash()
        bypass_refused &= (
            refuse(lambda: ecology.consequence(
                receipt, occurrence, 501 if index < 4 else 502,
                nomination.coalition, EFFECTS[nomination.coalition]),
                "nominated occurrence requires settlement")
            and ecology.state_hash() == before)
        ecology.settle_intervention(
            nomination, occurrence, 501 if index < 4 else 502,
            EFFECTS[nomination.coalition], True)
        no_credit_before_resolution &= not ecology.credit
    checks["resident_balances_missing_coalitions"] = tuple(sequence) == COALITIONS * 2
    checks["nomination_is_not_evidence_or_credit"] = no_credit_before_resolution
    checks["nominated_occurrence_cannot_bypass_binding"] = bypass_refused
    result = ecology.resolve(receipt)
    selected = ecology.select(((701, (11,)), (702, (29,))))
    checks["settled_square_earns_future_revision"] = (
        result is not None and selected is not None and selected.candidate == 701)
    checks["all_evidence_was_nominated_actual_participation"] = (
        set(result.evidence_occurrences)
        == {row.occurrence for row in ecology.nominations.values()}
        and all(row.settled for row in ecology.nominations.values()))

    pending = CausalAttributionEcologyV1()
    pending_receipt = pending.participate(PARTICIPANTS, 32)
    pending_nomination, pending_occurrence = pending.nominate_intervention(pending_receipt)
    before = pending.state_hash()
    checks["one_live_nomination_per_receipt"] = (
        refuse(lambda: pending.nominate_intervention(pending_receipt), "pending nomination")
        and pending.state_hash() == before)
    foreign = CausalAttributionEcologyV1().population.recruit(())
    checks["foreign_occurrence_refuses_atomically"] = (
        refuse(lambda: pending.settle_intervention(
            pending_nomination, foreign, 501, 0), "nominated occurrence binding")
        and pending.state_hash() == before)
    checks["prediction_cannot_settle_nomination"] = (
        refuse(lambda: pending.settle_intervention(
            pending_nomination, pending_occurrence, 501, 0, False),
            "independent consequence")
        and pending.state_hash() == before)

    midpoint = CausalAttributionEcologyV1.restore(pending.checkpoint())
    restored_nomination = midpoint.nominations[pending_nomination.identity]
    restored_occurrence = occurrence_by_identity(midpoint, pending_occurrence.identity)
    midpoint.settle_intervention(restored_nomination, restored_occurrence, 501, 0)
    pending.settle_intervention(pending_nomination, pending_occurrence, 501, 0)
    checks["pending_nomination_checkpoint_exact"] = midpoint.checkpoint() == pending.checkpoint()

    late = CausalAttributionEcologyV1()
    late_receipt = late.participate(PARTICIPANTS, 32)
    late_nomination, late_occurrence = late.nominate_intervention(late_receipt, 1)
    late.advance(2)
    before = late.state_hash()
    checks["late_world_return_refuses"] = (
        refuse(lambda: late.settle_intervention(
            late_nomination, late_occurrence, 501, 0), "late nomination consequence")
        and late.state_hash() == before)
    replacement, _ = late.nominate_intervention(late_receipt, 4)
    checks["expired_nomination_reclaimed"] = (
        replacement.identity != late_nomination.identity
        and late_nomination.identity not in late.nominations)

    repeated = CausalAttributionEcologyV1()
    repeated_receipt = repeated.participate(PARTICIPANTS, 64)
    for _ in range(4):
        nomination, occurrence = repeated.nominate_intervention(repeated_receipt)
        repeated.settle_intervention(
            nomination, occurrence, 501, EFFECTS[nomination.coalition])
    duplicate, duplicate_occurrence = repeated.nominate_intervention(repeated_receipt)
    before = repeated.state_hash()
    checks["same_source_repetition_not_independent_block"] = (
        refuse(lambda: repeated.settle_intervention(
            duplicate, duplicate_occurrence, 501, EFFECTS[duplicate.coalition]),
            "duplicate source coalition")
        and repeated.state_hash() == before)

    withdrawn = CausalAttributionEcologyV1.restore(ecology.checkpoint())
    withdrawn.withdraw_source(501)
    checks["source_withdrawal_removes_resident_revision"] = (
        withdrawn.resolve(receipt) is None
        and withdrawn.select(((701, (11,)), (702, (29,)))) is None)

    body = json.loads(pending.checkpoint())
    body["nominations"][0]["coalition"] = [29]
    checks["checkpoint_nomination_invariant_refusal"] = refuse(
        lambda: CausalAttributionEcologyV1.restore(json.dumps(body)),
        "checkpoint nomination")

    quantity = ecology.quantity()
    checks["bounded_sparse_intervention_work"] = (
        quantity["resident_capacity"] == DEFAULT_RESIDENT_CAPACITY
        and quantity["materialized_participants"] == 2
        and quantity["intervention_nominations"] == 8
        and quantity["evidence_rows"] == 8
        and quantity["last_resolve_touched"] == 8)

    elapsed_ms = (time.perf_counter() - started) * 1000
    checks["bounded_runtime"] = elapsed_ms < 5000
    report = {
        "schema": "agi.reference-resident-intervention.v1",
        "pass": all(checks.values()),
        "checks": checks,
        "nomination_sequence": [list(row) for row in sequence],
        "quantity": quantity,
        "elapsed_ms": round(elapsed_ms, 3),
        "reference_only": True,
        "adult_attached": False,
        "runtime_llm": False,
        "graph_flip": False,
        "physical_direct_parity": "NOT_RUN/RED",
        "claim": "RESIDENT_NOMINATED_MATCHED_INTERVENTIONS_REFERENCE_PROPERTY_ONLY",
    }
    print("FOUNDRY_RESIDENT_INTERVENTION " + ("GREEN" if report["pass"] else "RED"))
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
