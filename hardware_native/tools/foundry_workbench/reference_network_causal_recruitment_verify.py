#!/usr/bin/env python3
"""Adult-math-aligned Network closure, lesion and causal-credit controls."""
from __future__ import annotations

import json
import time

from reference_network_causal_recruitment_v1 import (
    DEFAULT_CAPACITY, NetworkCausalRecruitmentV1, Refuse)


A_FEATURES = (101, 102, 103, 104)
B_FEATURES = (104, 201, 202, 203)
CHAIN = ((0, 1), (1, 2), (2, 3))


def refuse(call, prefix):
    try:
        call()
    except Refuse as exc:
        return str(exc).startswith(prefix)
    return False


def members(ecology, features):
    return tuple(ecology.population.recruit((feature,)) for feature in features)


def build():
    ecology = NetworkCausalRecruitmentV1()
    revision_a = ecology.learn_recruitment_revision(members(ecology, A_FEATURES), CHAIN, 3)
    revision_b = ecology.learn_recruitment_revision(members(ecology, B_FEATURES), CHAIN, 3)
    return ecology, revision_a, revision_b


def activate(ecology, revision, features, unavailable=()):
    return ecology.activate(revision, members(ecology, features), unavailable)


def live(ecology, revision, features, source, actual_effect, matched_effect=0):
    network = activate(ecology, revision, features)
    channel = source + 7000
    ticket = ecology.participate(network, source, channel)
    envelope = ecology.consequence_envelope(ticket)
    return network, ecology.settle(
        envelope, network, source, channel, actual_effect, matched_effect)


def flat_fragment_score(revision, member_credit, unavailable=()):
    unavailable = set(unavailable)
    return sum(member_credit.get(signature, 0) for signature in revision.member_signatures
               if len(set(signature) - unavailable) * 2 >= len(signature))


def main():
    started = time.perf_counter()
    checks = {}
    ecology, revision_a, revision_b = build()

    active_a1, revised = live(ecology, revision_a, A_FEATURES, 501, 1)
    checks["actual_network_closure_earns_revision"] = revised == 1 and ecology.score(revision_a) == 1
    checks["overlap_does_not_leak_revision"] = ecology.score(revision_b) == 0
    active_a2, _ = live(ecology, revision_a, A_FEATURES, 502, 1)
    active_b1, _ = live(ecology, revision_b, B_FEATURES, 601, 1)
    checks["network_competition_uses_distinct_causal_history"] = (
        ecology.select((active_a2, active_b1)) == active_a2.identity)
    checks["fresh_occurrences_form_each_active_network"] = (
        active_a1.member_occurrences != active_a2.member_occurrences
        and active_a1.identity != active_a2.identity)

    leaf_sites = revision_a.member_signatures[0]
    leaf_lesioned = activate(ecology, revision_a, A_FEATURES, leaf_sites)
    bridge_sites = revision_a.member_signatures[1]
    bridge_lesioned = activate(ecology, revision_a, A_FEATURES, bridge_sites)
    b_after_bridge = activate(ecology, revision_b, B_FEATURES, bridge_sites)
    checks["redundant_leaf_lesion_preserves_active_network"] = (
        leaf_lesioned is not None and leaf_lesioned.live_members == (1, 2, 3))
    checks["bridge_lesion_dissolves_only_disconnected_network"] = (
        bridge_lesioned is None and b_after_bridge is not None
        and ecology.select((b_after_bridge,)) == b_after_bridge.identity)

    member_credit = {signature: 2 for signature in revision_a.member_signatures}
    for signature in revision_b.member_signatures:
        member_credit[signature] = member_credit.get(signature, 0) + 1
    flat_a = flat_fragment_score(revision_a, member_credit, bridge_sites)
    flat_b = flat_fragment_score(revision_b, member_credit, bridge_sites)
    checks["flat_member_credit_false_positive_after_bridge_lesion"] = flat_a > flat_b

    prediction = activate(ecology, revision_b, B_FEATURES)
    ticket = ecology.participate(prediction, 602, 7602)
    envelope = ecology.consequence_envelope(ticket)
    checks["prediction_not_revision"] = (
        ecology.settle(envelope, prediction, 602, 7602, 1, 1) == 0
        and ecology.score(revision_b) == 1)
    wrong = activate(ecology, revision_a, A_FEATURES)
    wrong_ticket = ecology.participate(wrong, 503, 7503)
    wrong_envelope = ecology.consequence_envelope(wrong_ticket)
    before = ecology.state_hash()
    checks["wrong_active_network_refuses_atomically"] = (
        refuse(lambda: ecology.settle(
            wrong_envelope, active_b1, 503, 7503, 1, 0),
               "network consequence binding")
        and ecology.state_hash() == before)
    forged = list(wrong_envelope);forged[-1] ^= 1
    before = ecology.state_hash()
    checks["wrong_challenge_refuses_atomically"] = (
        refuse(lambda: ecology.settle(
            tuple(forged), wrong, 503, 7503, 1, 0),
               "network consequence route")
        and ecology.state_hash() == before)
    checks["wrong_channel_refuses_atomically"] = (
        refuse(lambda: ecology.settle(
            wrong_envelope, wrong, 503, 7504, 1, 0),
               "network consequence route")
        and ecology.state_hash() == before)
    wrong_incarnation = list(wrong_envelope);wrong_incarnation[1] += 1
    checks["wrong_incarnation_refuses_atomically"] = (
        refuse(lambda: ecology.settle(
            tuple(wrong_incarnation), wrong, 503, 7503, 1, 0),
               "network consequence route")
        and ecology.state_hash() == before)
    stale = NetworkCausalRecruitmentV1.restore(ecology.checkpoint())
    stale_wrong = stale.active_networks[wrong.identity]
    stale.tick = wrong_envelope[2] + 1
    stale_before = stale.state_hash()
    checks["stale_consequence_refuses_atomically"] = (
        refuse(lambda: stale.settle(
            wrong_envelope, stale_wrong, 503, 7503, 1, 0),
               "network consequence route")
        and stale.state_hash() == stale_before)

    replay = NetworkCausalRecruitmentV1.restore(ecology.checkpoint())
    checks["complete_checkpoint_replay"] = (
        replay.checkpoint() == ecology.checkpoint()
        and replay.select((replay.active_networks[active_a2.identity],
                           replay.active_networks[active_b1.identity])) == active_a2.identity)
    replay.withdraw_source(501)
    checks["source_withdrawal_is_recruitment_revision_local"] = (
        replay.score(replay.recruitment_revisions[revision_a.identity]) == 1
        and replay.score(replay.recruitment_revisions[revision_b.identity]) == 1
        and replay.select((replay.active_networks[active_a2.identity],
                           replay.active_networks[active_b1.identity])) == 0)

    body = json.loads(ecology.checkpoint())
    body["revisions"][0]["recruitment_revision"] = revision_b.identity
    checks["checkpoint_causal_binding_refusal"] = refuse(
        lambda: NetworkCausalRecruitmentV1.restore(json.dumps(body)),
        "network checkpoint revision")
    body = json.loads(ecology.checkpoint())
    body["active_networks"][0]["sites"] = [999]
    checks["checkpoint_network_shape_refusal"] = refuse(
        lambda: NetworkCausalRecruitmentV1.restore(json.dumps(body)),
        "network checkpoint occurrence")
    body = json.loads(ecology.checkpoint())
    pending = next(row for row in body["pending"] if row["ticket"] == wrong_ticket)
    pending["challenge"] ^= 1
    checks["checkpoint_consequence_envelope_refusal"] = refuse(
        lambda: NetworkCausalRecruitmentV1.restore(json.dumps(body)),
        "network checkpoint pending")
    revision_count = len(ecology.recruitment_revisions)
    checks["disconnected_recruitment_revision_refusal"] = (
        refuse(lambda: ecology.learn_recruitment_revision(
            members(ecology, (301, 302, 303)), ((0, 1),), 2),
            "network disconnected preparation")
        and len(ecology.recruitment_revisions) == revision_count)

    checks["active_network_is_not_persistent_membership_object"] = (
        all(network.member_occurrences for network in ecology.active_networks.values())
        and all(not hasattr(revision, "current")
                for revision in ecology.recruitment_revisions.values()))
    checks["sparse_touched_work"] = (
        ecology.population.spec.site_count == DEFAULT_CAPACITY
        and ecology.last_touched_members == 4
        and len(ecology.recruitment_revisions) == 2
        and len(ecology.revisions) == 3)
    checks["no_language_or_answer_runtime"] = (
        not hasattr(ecology, "answer") and not hasattr(ecology, "expected_output"))

    elapsed_ms = (time.perf_counter() - started) * 1000
    checks["bounded_runtime"] = elapsed_ms < 5000
    report = {
        "schema": "agi.reference-network-causal-recruitment.v3",
        "pass": all(checks.values()),
        "checks": checks,
        "behavior": {
            "revision_a_credit": 2,
            "revision_b_credit": 1,
            "leaf_lesion_a_active": leaf_lesioned is not None,
            "bridge_lesion_a_active": bridge_lesioned is not None,
            "bridge_lesion_b_active": b_after_bridge is not None,
            "flat_fragment_a_score": flat_a,
            "flat_fragment_b_score": flat_b,
        },
        "quantity": {
            "resident_sites": ecology.population.spec.site_count,
            "resident_edges": ecology.population.allocated_edge_count,
            "recruitment_revisions": len(ecology.recruitment_revisions),
            "population_occurrences": len(ecology.population.occurrences),
            "active_networks": len(ecology.active_networks),
            "revision_events": len(ecology.revisions),
            "last_touched_members": ecology.last_touched_members,
            "checkpoint_bytes": len(ecology.checkpoint().encode()),
        },
        "elapsed_ms": round(elapsed_ms, 3),
        "reference_only": True,
        "adult_attached": False,
        "runtime_llm": False,
        "graph_flip": False,
        "physical_direct_parity": "NOT_RUN/RED",
        "claim": "ACTIVE_NETWORK_CLOSURE_MATCHED_CAUSAL_DIFFERENCE_REFERENCE_PROPERTY_ONLY",
    }
    print("FOUNDRY_NETWORK_CAUSAL_RECRUITMENT " + ("GREEN" if report["pass"] else "RED"))
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
