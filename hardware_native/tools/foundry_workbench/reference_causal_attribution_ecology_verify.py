#!/usr/bin/env python3
from __future__ import annotations

import json
import time

from reference_causal_attribution_ecology_v1 import (
    CausalAttributionEcologyV1, DEFAULT_RESIDENT_CAPACITY, MAX_PENDING, Refuse)


COALITIONS = ((), (11,), (29,), (11, 29))


def refuse(call, prefix):
    try:
        call()
    except Refuse as exc:
        return str(exc).startswith(prefix)
    return False


def observed(effects, *, capacity=DEFAULT_RESIDENT_CAPACITY, sources=(501, 502), missing=None, conflict=None):
    ecology = CausalAttributionEcologyV1(capacity)
    receipt = ecology.participate((11, 29), 32)
    for source_index, source in enumerate(sources):
        for coalition in COALITIONS:
            if missing == (source_index, coalition):
                continue
            effect = effects[coalition]
            if conflict == (source_index, coalition):
                effect += 1
            occurrence = ecology.population.recruit(coalition)
            ecology.consequence(receipt, occurrence, source, coalition, effect, True)
    return ecology, receipt


def normalized_credit(result):
    return tuple((numerator, denominator) for _, numerator, denominator in result.participant_credit)


def main():
    started = time.perf_counter()
    checks = {}

    support, support_receipt = observed({(): 0, (11,): 4, (29,): 1, (11, 29): 5})
    before_resolution = support.select(((701, (11,)), (702, (29,))))
    support_result = support.resolve(support_receipt)
    selected = support.select(((701, (11,)), (702, (29,))))
    checks["participation_is_not_credit"] = before_resolution is None
    checks["supportive_credit_changes_future_selection"] = (
        support_result is not None and support_result.tags == ("support",)
        and selected is not None and selected.candidate == 701)
    focal = support.select(((701, (11,)), (702, (29,))), unavailable=(11,))
    remote = support.select(((701, (11,)), (702, (29,))), unavailable=(999_001,))
    checks["focal_vs_remote_participant_lesion"] = (
        focal is not None and focal.candidate == 702 and remote == selected)

    counter, counter_receipt = observed({(): 0, (11,): -4, (29,): -1, (11, 29): -5})
    counter_result = counter.resolve(counter_receipt)
    checks["counter_attribution"] = counter_result is not None and counter_result.tags == ("counter",)

    synergy, synergy_receipt = observed({(): 0, (11,): 0, (29,): 0, (11, 29): 6})
    synergy_result = synergy.resolve(synergy_receipt)
    synergy_choice = synergy.select(((801, (11,)), (802, (29,)), (803, (11, 29))))
    checks["synergy_requires_coalition"] = (
        synergy_result is not None and "synergy" in synergy_result.tags
        and synergy_choice is not None and synergy_choice.candidate == 803)
    checks["synergy_focal_lesion_abolishes_selection"] = (
        synergy.select(((801, (11,)), (802, (29,)), (803, (11, 29))),
                       unavailable=(11,)) is None)

    redundant, redundant_receipt = observed({(): 0, (11,): 4, (29,): 4, (11, 29): 4})
    redundant_result = redundant.resolve(redundant_receipt)
    checks["redundancy_not_double_credited"] = (
        redundant_result is not None and "redundancy" in redundant_result.tags
        and sum(numerator / denominator for _, numerator, denominator
                in redundant_result.participant_credit) == 4
        and redundant.select(((811, (11,)), (812, (29,)), (813, (11, 29)))) is None)

    opposed, opposed_receipt = observed({(): 0, (11,): 5, (29,): -3, (11, 29): 2})
    opposed_result = opposed.resolve(opposed_receipt)
    checks["opposed_contributions_preserved"] = (
        opposed_result is not None and set(opposed_result.tags) == {"counter", "opposition", "support"})

    noncontingent, noncontingent_receipt = observed(
        {(): 3, (11,): 3, (29,): 3, (11, 29): 3})
    noncontingent_result = noncontingent.resolve(noncontingent_receipt)
    checks["background_outcome_rate_prevents_false_credit"] = (
        noncontingent_result is not None and not noncontingent_result.tags
        and all(numerator == 0 for _, numerator, _ in noncontingent_result.participant_credit)
        and noncontingent.select(((701, (11,)), (702, (29,)))) is None)

    interleaved = CausalAttributionEcologyV1()
    first_receipt = interleaved.participate((11, 29), 64)
    second_receipt = interleaved.participate((41, 43), 64)
    first_effects = {(): 0, (11,): 4, (29,): 1, (11, 29): 5}
    second_effects = {(): 0, (41,): -4, (43,): -1, (41, 43): -5}
    for source in (501, 502):
        for first_coalition, second_coalition in zip(COALITIONS, ((), (41,), (43,), (41, 43))):
            second_occurrence = interleaved.population.recruit(second_coalition)
            interleaved.consequence(second_receipt, second_occurrence, source,
                                    second_coalition, second_effects[second_coalition])
            first_occurrence = interleaved.population.recruit(first_coalition)
            interleaved.consequence(first_receipt, first_occurrence, source,
                                    first_coalition, first_effects[first_coalition])
    second_result = interleaved.resolve(second_receipt)
    first_result = interleaved.resolve(first_receipt)
    checks["intervening_occurrences_preserve_pending_identity"] = (
        second_result is not None and second_result.tags == ("counter",)
        and first_result is not None and first_result.tags == ("support",)
        and not set(second_result.evidence_occurrences) & set(first_result.evidence_occurrences))

    one_source, one_source_receipt = observed(
        {(): 0, (11,): 4, (29,): 1, (11, 29): 5}, sources=(501,))
    atomic = one_source.state_hash()
    checks["two_source_minimum_atomic_refusal"] = (
        one_source.resolve(one_source_receipt) is None and one_source.state_hash() == atomic)

    disjoint = CausalAttributionEcologyV1()
    disjoint_receipt = disjoint.participate((11, 29), 32)
    for index, coalition in enumerate(COALITIONS):
        for source in (700 + index * 2, 701 + index * 2):
            occurrence = disjoint.population.recruit(coalition)
            disjoint.consequence(disjoint_receipt, occurrence, source, coalition,
                                 {(): 0, (11,): 4, (29,): 1, (11, 29): 5}[coalition])
    disjoint_before = disjoint.state_hash()
    checks["same_sources_span_matched_square"] = (
        disjoint.resolve(disjoint_receipt) is None and disjoint.state_hash() == disjoint_before)

    omitted, omitted_receipt = observed(
        {(): 0, (11,): 4, (29,): 1, (11, 29): 5}, missing=(1, (11, 29)))
    checks["focal_participant_omission_unresolved"] = omitted.resolve(omitted_receipt) is None

    conflict, conflict_receipt = observed(
        {(): 0, (11,): 4, (29,): 1, (11, 29): 5}, conflict=(1, (11,)))
    conflict_before = conflict.state_hash()
    checks["conflicting_matched_set_atomic_refusal"] = (
        conflict.resolve(conflict_receipt) is None and conflict.state_hash() == conflict_before)

    withdrawn, withdrawn_receipt = observed({(): 0, (11,): 4, (29,): 1, (11, 29): 5})
    withdrawn.resolve(withdrawn_receipt)
    withdrawn.withdraw_source(501)
    checks["source_withdrawal_removes_revision"] = (
        withdrawn.select(((701, (11,)), (702, (29,)))) is None
        and withdrawn.resolve(withdrawn_receipt) is None)
    sham_before = support.state_hash()
    support.withdraw_source(999_001)
    checks["remote_withdrawal_sham"] = support.state_hash() == sham_before

    late = CausalAttributionEcologyV1()
    late_receipt = late.participate((11, 29), 1)
    late.advance(2)
    late_before = late.state_hash()
    late_occurrence = late.population.recruit(())
    checks["late_consequence_refusal"] = (
        refuse(lambda: late.consequence(late_receipt, late_occurrence, 501, (), 0), "late consequence")
        and len(late.pending[late_receipt].evidence) == 0)

    non_independent = CausalAttributionEcologyV1()
    ni_receipt = non_independent.participate((11, 29), 8)
    ni_before = non_independent.state_hash()
    ni_occurrence = non_independent.population.recruit(())
    checks["non_independent_consequence_refusal"] = (
        refuse(lambda: non_independent.consequence(ni_receipt, ni_occurrence, 501, (), 0, False),
               "independent consequence") and len(non_independent.pending[ni_receipt].evidence) == 0)

    replay = CausalAttributionEcologyV1()
    replay_receipt = replay.participate((11, 29), 8)
    replay_occurrence = replay.population.recruit(())
    replay.consequence(replay_receipt, replay_occurrence, 501, (), 0)
    replay_before = replay.state_hash()
    checks["occurrence_replay_refusal"] = (
        refuse(lambda: replay.consequence(replay_receipt, replay_occurrence, 502, (), 1),
               "occurrence replay") and replay.state_hash() == replay_before)

    bound = CausalAttributionEcologyV1()
    bound_receipt = bound.participate((11, 29), 8)
    foreign = CausalAttributionEcologyV1()
    foreign_occurrence = foreign.population.recruit(())
    bound_before = bound.state_hash()
    checks["foreign_occurrence_refusal"] = (
        refuse(lambda: bound.consequence(bound_receipt, foreign_occurrence, 501, (), 0),
               "actual occurrence participation") and bound.state_hash() == bound_before)
    mismatched_occurrence = bound.population.recruit((11,))
    mismatch_before = bound.state_hash()
    checks["occurrence_coalition_mismatch_refusal"] = (
        refuse(lambda: bound.consequence(bound_receipt, mismatched_occurrence, 501, (), 0),
               "actual occurrence participation") and bound.state_hash() == mismatch_before)

    expiring = CausalAttributionEcologyV1()
    for index in range(MAX_PENDING + 8):
        expiring.participate((100 + index * 2, 101 + index * 2), 1)
    checks["expired_pending_reclaimed"] = (
        len(expiring.pending) < MAX_PENDING and expiring.quantity()["evidence_rows"] == 0)

    partial = CausalAttributionEcologyV1()
    partial_receipt = partial.participate((11, 29), 8)
    partial_nomination, partial_occurrence = partial.nominate_intervention_at(
        partial_receipt, (11,), 8)
    partial.settle_intervention(partial_nomination, partial_occurrence, 601, 1)
    retired_occurrences = {
        partial.pending[partial_receipt].occurrence, partial_occurrence.identity}
    partial.advance(9)
    partial.participate((41, 43), 8)
    partial_checkpoint = partial.checkpoint()
    checks["expired_partial_closure_reclaims_provenance_and_restores"] = (
        partial_receipt not in partial.pending
        and partial_nomination.identity not in partial.nominations
        and not retired_occurrences & partial.used_occurrences
        and not retired_occurrences & {
            occurrence.identity for occurrence in partial.population.occurrences}
        and CausalAttributionEcologyV1.restore(partial_checkpoint).checkpoint()
        == partial_checkpoint)

    restored = CausalAttributionEcologyV1.restore(synergy.checkpoint())
    checks["exact_checkpoint_replay"] = (
        restored.checkpoint() == synergy.checkpoint()
        and restored.select(((801, (11,)), (802, (29,)), (803, (11, 29)))) == synergy_choice)

    permuted = CausalAttributionEcologyV1(DEFAULT_RESIDENT_CAPACITY)
    permuted_receipt = permuted.participate((103, 407), 32)
    remapped = {(): 0, (103,): 0, (407,): 0, (103, 407): 6}
    for source in (1501, 2502):
        for coalition in ((), (103,), (407,), (103, 407)):
            occurrence = permuted.population.recruit(coalition)
            permuted.consequence(permuted_receipt, occurrence, source, coalition, remapped[coalition])
    permuted_result = permuted.resolve(permuted_receipt)
    checks["opaque_id_permutation"] = (
        permuted_result is not None and permuted_result.tags == synergy_result.tags
        and normalized_credit(permuted_result) == normalized_credit(synergy_result))

    huge, huge_receipt = observed(
        {(): 0, (11,): 4, (29,): 1, (11, 29): 5}, capacity=DEFAULT_RESIDENT_CAPACITY)
    huge_result = huge.resolve(huge_receipt)
    quantity = huge.quantity()
    checks["bounded_touched_work"] = (
        huge_result is not None and huge_result.touched_work == 8
        and quantity["resident_capacity"] == DEFAULT_RESIDENT_CAPACITY
        and quantity["materialized_participants"] == 2
        and quantity["evidence_rows"] == 8)

    try:
        body = json.loads(synergy.checkpoint())
        body["expected_output"] = [1]
        CausalAttributionEcologyV1.restore(json.dumps(body))
        checks["literal_bypass_refusal"] = False
    except Exception:
        checks["literal_bypass_refusal"] = True
    try:
        body = json.loads(synergy.checkpoint())
        body["used_occurrences"] = body["used_occurrences"][1:]
        CausalAttributionEcologyV1.restore(json.dumps(body))
        checks["checkpoint_causal_invariant_refusal"] = False
    except Exception:
        checks["checkpoint_causal_invariant_refusal"] = True

    elapsed_ms = (time.perf_counter() - started) * 1000
    checks["bounded_runtime"] = elapsed_ms < 5000
    receipt = {
        "schema": "0x1.reference-causal-attribution-ecology.v1",
        "pass": all(checks.values()),
        "checks": checks,
        "quantity": quantity,
        "elapsed_ms": round(elapsed_ms, 3),
        "reference_only": True,
        "adult_attached": False,
        "runtime_llm": False,
        "graph_flip": False,
        "physical_direct_parity": "NOT_RUN/RED",
        "claim": "MATCHED_CAUSAL_ATTRIBUTION_ECOLOGY_REFERENCE_PROPERTY_ONLY",
    }
    print("FOUNDRY_CAUSAL_ATTRIBUTION_ECOLOGY " + ("GREEN" if receipt["pass"] else "RED"))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    raise SystemExit(0 if receipt["pass"] else 1)


if __name__ == "__main__":
    main()
