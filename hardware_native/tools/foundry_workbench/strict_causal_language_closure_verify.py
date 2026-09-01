#!/usr/bin/env python3
"""Falsify a missing consequence -> plan -> public-byte causal closure.

This is a graph-neutral reference experiment.  All organism-facing state is the
strict numeric state of ``ReferenceMachineV1``. The host authors this fixture's
numeric contact contents, delivers sealed contacts, and compares receipts. No
expected surface form is supplied.
"""
import copy
import hashlib
import json
import time

from reference_contract_1610 import (
    CONTACT_CONSEQUENCE,
    CONTACT_FRAME,
    CONTACT_LEXEME,
    CONTACT_QUERY,
    CONTACT_STIMULUS,
    ContactV1,
    ReferenceMachineV1,
    Refuse,
    authored_recipe_pool,
    authored_starting_state,
)


class ContactRoute:
    def __init__(self, ticket=20_000):
        self.ticket = ticket

    def put(self, machine, kind, payload, source, channel=1, independent=1):
        contact = ContactV1(
            self.ticket, machine.state.incarnation, machine.state.tick + 8,
            source, channel, kind, tuple(payload), 1, independent,
        )
        self.ticket += 1
        return machine.contact(machine.seal_contact(contact))


def leaf(category, number=1):
    return (category, number, ())


def clause(head, children=(10, 11)):
    return (head, 1, tuple(leaf(category) for category in children))


def tree_depth(node):
    return 1 + max((tree_depth(child) for child in node[2]), default=0)


def flatten(node):
    category, number, children = node
    out = [category, number, len(children)]
    for child in children:
        out.extend(flatten(child))
    return tuple(out)


def add_lexeme(route, machine, category, units, source, number=1):
    route.put(machine, CONTACT_LEXEME,
              (1, category, number, len(units), *units), source)


def add_query(route, machine, cue, head, source, children=(10, 11)):
    body = flatten(clause(head, children))
    route.put(machine, CONTACT_QUERY,
              (cue, 1, len(body), 1, *body), source)


def add_tree_query(route, machine, cue, root, source):
    body = flatten(root)
    route.put(machine, CONTACT_QUERY,
              (cue, 1, len(body), tree_depth(root), *body), source)


def ask(route, machine, cue, source=900):
    route.put(machine, CONTACT_STIMULUS, (cue,), source)
    return machine.tick()


def settle(route, machine, action, effect, counter):
    contact = ContactV1(
        action.ticket, action.incarnation, action.deadline, action.source,
        action.channel, CONTACT_CONSEQUENCE, (effect, counter), 1, 1,
    )
    return machine.contact(machine.seal_contact(contact))


def refuse(call, prefix):
    try:
        call()
    except Refuse as exc:
        assert str(exc).startswith(prefix), (prefix, str(exc))
        return True
    raise AssertionError("expected refusal")


def trained_preference(preferred, permutation=0):
    """Return equal final candidates with only causal difference selecting a root."""
    machine = ReferenceMachineV1(authored_starting_state(), authored_recipe_pool())
    route = ContactRoute(20_000 + permutation * 10_000)
    cue = 777 + permutation * 1_000
    categories = (10 + permutation * 70, 11 + permutation * 70)
    heads = (20 + permutation * 70, 21 + permutation * 70)
    sources = (301 + permutation * 100, 302 + permutation * 100)
    add_lexeme(route, machine, 0, (32,), 101 + permutation * 100, number=0)
    add_lexeme(route, machine, categories[0], (61, 62), 101 + permutation * 100)
    add_lexeme(route, machine, categories[1], (63, 64), 101 + permutation * 100)
    add_lexeme(route, machine, heads[0], (71, 72, 73), 101 + permutation * 100)
    add_lexeme(route, machine, heads[1], (81, 82, 83, 84), 101 + permutation * 100)
    route.put(machine, CONTACT_FRAME, (1, 1, 2, 3, 1, 0, 2),
              102 + permutation * 100)
    add_query(route, machine, cue, heads[0], sources[0], categories)
    training_action = ask(route, machine, cue)
    settle(route, machine, training_action, int(preferred == 0), int(preferred == 1))
    add_query(route, machine, cue, heads[1], sources[1], categories)
    return machine, route, cue, heads, sources


def recursive_candidate_state(permutation=0):
    """Authored numeric cognition fixture; no surface plan or expected bytes."""
    shift = permutation * 1_000
    machine = ReferenceMachineV1(authored_starting_state(), authored_recipe_pool())
    route = ContactRoute(70_000 + shift)
    cue = 900 + shift
    categories = tuple(value + shift for value in (10, 11, 12, 13, 20, 22))
    surfaces = ((51, 52), (53, 54, 55), (56, 57), (58, 59, 60),
                (71, 72), (76, 77))
    source = 700 + shift
    add_lexeme(route, machine, 0, (32,), source, number=0)
    for category, units in zip(categories, surfaces):
        add_lexeme(route, machine, category, units, source)
    route.put(machine, CONTACT_FRAME, (1, 1, 2, 3, 1, 0, 2), source + 1)
    a, b, c, d, p, r = categories
    training = (p, 1, (leaf(a), leaf(c)))
    preferred = (r, 1, ((p, 1, (leaf(a), leaf(b))), leaf(c)))
    alternative = (r, 1, ((p, 1, (leaf(a), leaf(b))), leaf(d)))
    training_source = source + 2
    preferred_source, alternative_source = source + 3, source + 4
    training_cue = cue - 1
    add_tree_query(route, machine, training_cue, training, training_source)
    training_action = ask(route, machine, training_cue)
    settle(route, machine, training_action, 1, 0)
    add_tree_query(route, machine, cue, preferred, preferred_source)
    add_tree_query(route, machine, cue, alternative, alternative_source)
    return machine, route, cue, preferred_source, alternative_source


def action_receipt(action):
    return {
        "payload_hash": hashlib.sha256(bytes(action.payload)).hexdigest(),
        "plan_occurrence": action.occurrence_identity,
        "constituent_root": action.constituent_root,
        "frontier": action.frontier,
        "ancestry_occurrences": tuple(x.occurrence_identity for x in action.ancestry),
        "ancestry_roots": tuple(x.constituent_root for x in action.ancestry),
    }


def surface_contact_topology(machine):
    """Mask consequence value; every authored/surface contact must still match."""
    return tuple(
        (contact.kind,
         () if contact.kind == CONTACT_CONSEQUENCE else contact.payload,
         contact.source, contact.channel, contact.ticket)
        for contact in machine.state.contact_receipts
    )


def main():
    started = time.perf_counter()
    checks = {}

    left, left_route, cue, _, left_sources = trained_preference(0)
    right, right_route, right_cue, _, _ = trained_preference(1)
    left_action = ask(left_route, left, cue)
    right_action = ask(right_route, right, right_cue)
    left_receipt = action_receipt(left_action)
    right_receipt = action_receipt(right_action)
    checks["opposite_authenticated_history_diverges"] = (
        left_receipt["payload_hash"] != right_receipt["payload_hash"]
        and left_action.constituent_root != right_action.constituent_root
    )
    checks["only_consequence_value_differs"] = (
        surface_contact_topology(left) == surface_contact_topology(right)
    )
    checks["plan_in_every_byte_ancestry"] = all(
        item.occurrence_identity == action.occurrence_identity
        for action in (left_action, right_action) for item in action.ancestry
    )
    checks["complete_per_byte_ancestry"] = all(
        len(action.ancestry) == len(action.payload)
        and tuple(item.offset for item in action.ancestry) == tuple(range(len(action.payload)))
        and tuple(item.unit for item in action.ancestry) == action.payload
        for action in (left_action, right_action)
    )

    tied = ReferenceMachineV1(authored_starting_state(), authored_recipe_pool())
    tied_route = ContactRoute(50_000)
    add_lexeme(tied_route, tied, 0, (32,), 101, number=0)
    add_lexeme(tied_route, tied, 10, (61, 62), 101)
    add_lexeme(tied_route, tied, 11, (63, 64), 101)
    add_lexeme(tied_route, tied, 20, (71, 72, 73), 101)
    add_lexeme(tied_route, tied, 21, (81, 82, 83, 84), 101)
    tied_route.put(tied, CONTACT_FRAME, (1, 1, 2, 3, 1, 0, 2), 102)
    add_query(tied_route, tied, 777, 20, 301)
    add_query(tied_route, tied, 777, 21, 302)
    tied_route.put(tied, CONTACT_STIMULUS, (777,), 900)
    before = tied.state_hash()
    checks["equal_credit_refuses_atomically"] = (
        refuse(tied.tick, "arbitrate:ambiguous") and tied.state_hash() == before
    )

    replay, replay_route, replay_cue, _, _ = trained_preference(0)
    frozen = replay.checkpoint()
    one = ReferenceMachineV1.restore(frozen, replay.pool)
    two = ReferenceMachineV1.restore(frozen, replay.pool)
    route_one = copy.deepcopy(replay_route)
    route_two = copy.deepcopy(replay_route)
    one_action = ask(route_one, one, replay_cue)
    two_action = ask(route_two, two, replay_cue)
    checks["complete_checkpoint_replay"] = (
        one_action == two_action and one.state_hash() == two.state_hash()
    )

    opaque, opaque_route, opaque_cue, _, _ = trained_preference(0, permutation=1)
    opaque_action = ask(opaque_route, opaque, opaque_cue)
    checks["opaque_id_permutation_invariant"] = (
        opaque_action.payload == one_action.payload
        and opaque_action.constituent_root != one_action.constituent_root
    )

    withdrawn, withdrawal_route, withdrawal_cue, _, withdrawal_sources = trained_preference(0)
    withdrawn.withdraw_source(withdrawal_sources[0])
    fallback = ask(withdrawal_route, withdrawn, withdrawal_cue)
    checks["focal_withdrawal_changes_plan"] = (
        hashlib.sha256(bytes(fallback.payload)).hexdigest()
        != left_receipt["payload_hash"]
    )
    remote, remote_route, remote_cue, _, _ = trained_preference(0)
    remote.withdraw_source(999_001)
    sham = ask(remote_route, remote, remote_cue)
    checks["remote_withdrawal_sham"] = sham.payload == one_action.payload

    recursive, recursive_route, recursive_cue, preferred_source, _ = recursive_candidate_state()
    recursive_blob = recursive.checkpoint()
    recursive_snapshot = ReferenceMachineV1.restore(
        recursive_blob, recursive.pool).state
    active_queries = [query for query in recursive.state.queries
                      if query.active and query.frame == recursive_cue]
    checks["bounded_authored_candidate_snapshot"] = (
        len(active_queries) == 2
        and len(recursive.state.relations) < 64
        and len(recursive_blob) < 64 * 1024
        and recursive.state.work <= recursive.state.work_limit
    )
    candidate_support = {
        query.identity: recursive._candidate_credit(query)[0]
        for query in active_queries
    }
    recursive_action = ask(recursive_route, recursive, recursive_cue)
    occurrence = next(row for row in recursive.state.occurrences
                      if row.identity == recursive_action.occurrence_identity)
    selected_query = next(row for row in recursive.state.queries
                          if row.identity == occurrence.query_identity)
    other_queries = [row for row in recursive.state.queries
                     if row.active and row.frame == recursive_cue
                     and row.identity != selected_query.identity]
    checks["resident_frontier_credit_selects_novel_recursive_candidate"] = (
        selected_query.source == preferred_source
        and recursive.state.credit.get(selected_query.contact_identity, 0) == 0
        and all(recursive.state.credit.get(row.contact_identity, 0) == 0
                for row in other_queries)
        and all(candidate_support[selected_query.identity]
                > candidate_support[row.identity]
                for row in other_queries)
    )
    checks["recursive_surface_not_historical_replay"] = all(
        recursive_action.payload != action.payload
        for action in recursive_snapshot.actions
    )
    checks["recursive_constituent_per_byte_ancestry"] = (
        len(recursive_action.ancestry) == len(recursive_action.payload)
        and max(item.ancestry_depth for item in recursive_action.ancestry) >= 3
        and len({item.constituent_root for item in recursive_action.ancestry}) >= 5
        and all(item.occurrence_identity == recursive_action.occurrence_identity
                for item in recursive_action.ancestry)
    )
    positive_credit_roots = {
        root for root, value in recursive.state.credit.items() if value > 0
    }
    checks["recursive_credit_overlaps_emitted_constituent_ancestry"] = (
        bool(positive_credit_roots.intersection(
            root for item in recursive_action.ancestry
            for root in item.relation_roots))
        and all(item.relation_roots for item in recursive_action.ancestry)
    )
    historical_queries = [query for query in recursive_snapshot.queries
                          if query.active and query.frame != recursive_cue]
    historical_occurrences = [occurrence for occurrence in recursive_snapshot.occurrences
                              if historical_queries
                              and occurrence.query_identity
                              == historical_queries[0].identity]
    historical_actions = [action for action in recursive_snapshot.actions
                          if historical_occurrences
                          and action.occurrence_identity
                          == historical_occurrences[0].identity]
    checks["recursive_credit_episode_remains_authenticated"] = (
        len(historical_queries) == 1
        and len(historical_occurrences) == 1
        and len(historical_actions) == 1
        and historical_queries[0].source not in recursive.state.withdrawn_sources
        and historical_actions[0].settled
        and bool(historical_actions[0].active)
    )
    query_payloads = [row.values for row in recursive.state.relations
                      if row.kind == CONTACT_QUERY]
    checks["recursive_surface_not_supplied_by_candidate_query"] = all(
        not any(tuple(values[index:index + len(recursive_action.payload)])
                == recursive_action.payload
                for index in range(max(0, len(values) - len(recursive_action.payload) + 1)))
        for values in query_payloads
    )
    checks["recursive_work_is_bounded"] = (
        recursive.state.work <= recursive.state.work_limit
        and recursive.last_lookup_touches <= len(recursive.state.relations)
    )

    replay_one = ReferenceMachineV1.restore(recursive_blob, recursive.pool)
    replay_two = ReferenceMachineV1.restore(recursive_blob, recursive.pool)
    recursive_route_one = copy.deepcopy(recursive_route)
    recursive_route_two = copy.deepcopy(recursive_route)
    replay_action_one = ask(recursive_route_one, replay_one, recursive_cue)
    replay_action_two = ask(recursive_route_two, replay_two, recursive_cue)
    checks["recursive_complete_checkpoint_replay"] = (
        replay_action_one == replay_action_two
        and replay_one.state_hash() == replay_two.state_hash()
    )

    permuted, permuted_route, permuted_cue, _, _ = recursive_candidate_state(1)
    permuted_action = ask(permuted_route, permuted, permuted_cue)
    checks["recursive_opaque_identity_permutation"] = (
        permuted_action.payload == recursive_action.payload
        and permuted_action.constituent_root != recursive_action.constituent_root
        and tuple(item.constituent_root for item in permuted_action.ancestry)
        != tuple(item.constituent_root for item in recursive_action.ancestry)
    )

    withdrawn_recursive = ReferenceMachineV1.restore(recursive_blob, recursive.pool)
    withdrawn_route = copy.deepcopy(recursive_route)
    withdrawn_recursive.withdraw_source(preferred_source)
    fallback_recursive = ask(withdrawn_route, withdrawn_recursive, recursive_cue)
    checks["recursive_source_withdrawal_changes_candidate"] = (
        fallback_recursive.payload != recursive_action.payload
    )
    sham_recursive = ReferenceMachineV1.restore(recursive_blob, recursive.pool)
    sham_route = copy.deepcopy(recursive_route)
    sham_recursive.withdraw_source(999_002)
    checks["recursive_remote_withdrawal_sham"] = (
        ask(sham_route, sham_recursive, recursive_cue).payload == recursive_action.payload
    )

    omitted, omitted_route, omitted_cue, omitted_source, _ = recursive_candidate_state(2)
    root = next(row.root for row in omitted.state.queries
                if row.source == omitted_source)
    omitted_root = (root[0], root[1], (root[2][0],))
    add_tree_query(omitted_route, omitted, omitted_cue + 1, omitted_root,
                   omitted_source + 20)
    omitted_route.put(omitted, CONTACT_STIMULUS, (omitted_cue + 1,),
                      omitted_source + 21)
    before_omission = omitted.state_hash()
    checks["recursive_omitted_constituent_refuses_atomically"] = (
        refuse(omitted.tick, "compose:omitted_constituent")
        and omitted.state_hash() == before_omission
    )

    checks["bounded_runtime"] = time.perf_counter() - started < 1.0
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SystemExit("FOUNDRY_STRICT_CAUSAL_LANGUAGE_CLOSURE_RED " + ",".join(failed))
    receipt = {
        "contract": "FOUNDRY_STRICT_CAUSAL_LANGUAGE_CLOSURE_GREEN",
        "claim": "CONSEQUENCE_TO_RECURSIVE_PLAN_TO_BYTES_REFERENCE_PROPERTY_ONLY",
        "reference_only": True,
        "adult_attached": False,
        "runtime_llm": False,
        "host_prompt": False,
        "human_language_claim": False,
        "candidate_state_authorship": "AUTHORED_NUMERIC_FIXTURE/NOT_RESIDENT_COGNITION",
        "contact_content_authority": "HOST_AUTHORED_REFERENCE_FIXTURE",
        "adapter_attached": False,
        "publication_owner": "REFERENCE_EXECUTOR_RECIPE_6",
        "graph_flip": False,
        "physical_direct_parity": "NOT_RUN/RED",
        "expected_surface_fields": False,
        "recursive_public_payload": list(recursive_action.payload),
        "recursive_public_payload_sha256": hashlib.sha256(
            bytes(recursive_action.payload)).hexdigest(),
        "recursive_snapshot_bytes": len(recursive_blob),
        "recursive_snapshot_counts": {
            "relations": len(recursive_snapshot.relations),
            "unresolved_candidates": len(active_queries),
            "authenticated_historical_candidates": sum(
                query.active and query.frame != recursive_cue
                for query in recursive_snapshot.queries),
            "contact_receipts": len(recursive_snapshot.contact_receipts),
            "settled_historical_actions": sum(
                action.settled for action in recursive_snapshot.actions),
            "work_limit": recursive.state.work_limit,
        },
        "recursive_candidate_support": sorted(
            candidate_support.values(), reverse=True),
        "recursive_candidate_direct_credit": sorted(
            recursive.state.credit.get(query.contact_identity, 0)
            for query in active_queries),
        "recipe_pool_root": recursive.pool.root,
        "remaining_red": [
            "RESIDENT_ACQUISITION_OF_CANDIDATE_STATE",
            "HOST_AUTHORED_FRAME_AND_LEXEME_CONTACTS",
            "CONSEQUENCE_RECEIPT_ID_IN_PER_BYTE_ANCESTRY",
            "CUMULATIVE_LOOKUP_AND_COMPOSE_WORK_ACCOUNTING",
            "PHYSICAL_MOTOR_ADAPTER_AND_REAFFERENCE",
            "PRODUCTION_RESIDENT_RECIPE_IR_TRANSLATION",
            "DIRECT_PHYSICAL_PARITY",
        ],
        "recursive_max_ancestry_depth": max(
            item.ancestry_depth for item in recursive_action.ancestry),
        "checks": checks,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print("FOUNDRY_STRICT_CAUSAL_LANGUAGE_CLOSURE_GREEN")
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
