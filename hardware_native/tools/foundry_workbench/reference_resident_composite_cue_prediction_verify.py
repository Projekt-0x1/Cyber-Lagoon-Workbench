#!/usr/bin/env python3
"""Hostile reference assay for two-edge composite-cue prediction."""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

from reference_resident_channel_sequence_grounding_v1 import (
    GroundingRefuse, admit_channel_sequence_boundary_v1,
)
from reference_resident_composite_cue_prediction_v1 import (
    CompositeCueRefuse, NODE_COMPOSITE, NODE_SPAN,
    ResidentCompositeCuePredictionV1,
)


def refuses(call, fragment=""):
    try:
        call()
    except Exception as exc:
        return isinstance(exc, (CompositeCueRefuse, GroundingRefuse,
                                TypeError, ValueError)) and fragment in str(exc)
    return False


def main(fixture_hook=None):
    started = time.perf_counter(); checks = {}
    boundary = admit_channel_sequence_boundary_v1()
    machine = ResidentCompositeCuePredictionV1(boundary)

    def put(source, channel, unit):
        contact = boundary.seal_sample(machine.session, machine.next_sequence,
                                       source, channel, (unit,), (source, channel))
        return machine.ingest_sample(contact)

    def acquire(span, marker, channel, sources):
        # Two recurrence occurrences, then two authenticated future outcomes.
        repeats = ((2, 2) if len(span) == 3 else (1, 1))
        for source, count in zip(sources, repeats):
            for _ in range(count):
                for unit in (*span, marker): put(source, channel, unit)
        for source in sources:
            for unit in span: put(source, channel, unit)
            phase = machine.tick()
            assert phase.inner_close_root and machine._inner._inner.pending
            put(source, channel, marker)
        rows = [row for row in machine._inner._inner.recipes.values()
                if row.channel == channel and row.length == len(span)]
        matching = []
        for recipe in rows:
            occurrences = [row for row in machine._inner._inner.span_occurrences
                           if row.channel == channel and row.span_hash == recipe.span_hash]
            if any(machine._inner._inner._units(row) == tuple(span) for row in occurrences):
                matching.append(recipe)
        assert len(matching) == 1
        return matching[0].identity

    # A single marginal value is established per channel.  Every learned opaque
    # span uses its own source pair, preventing acquisition traffic from earning
    # cross-span support.
    for channel, source, count in ((7, 9001, 7), (8, 9002, 5), (9, 9003, 5)):
        for _ in range(count): put(source, channel, 0)

    outer = ((101, 102, 103), (111, 112, 113),
             (121, 122, 123), (131, 132, 133))
    middle_l = (141, 142, 143)
    middle_q = (151, 152, 153)
    world = (161, 162, 163, 164)
    consequence = (171, 172, 173, 174)
    distractor = (181, 182, 183, 184)
    learned = {}
    specifications = [*( (span, 201 + i, 7) for i, span in enumerate(outer) ),
                      (middle_l, 210, 7), (middle_q, 211, 7),
                      (world, 220, 8), (consequence, 230, 9),
                      (distractor, 231, 9)]
    for index, (span, marker, channel) in enumerate(specifications):
        learned[(channel, span)] = acquire(span, marker, channel,
                                           (10000 + 2 * index, 10001 + 2 * index))
    checks["multiple_channel_general_spans_acquired"] = len(learned) == 9
    active_base = {row.identity for row in machine._inner._inner.recipes.values()}
    assert set(learned.values()).issubset(active_base), (learned, active_base)

    A, B, X, Y = outer
    training_pairs = ((A, B), (A, X), (B, X))
    # Each tuple is one resident expansion batch.  The final pair has all three
    # required corners in the frozen pre-batch constructor and therefore rebinds;
    # any earlier pair merely expands a slot marginal and must not rebind.
    expansion_batches = (((B, B),), ((A, Y), (B, Y)),
                         ((Y, B), (Y, X)), ((Y, Y),),
                         ((X, B), (X, X)), ((X, Y),),
                         ((A, A), (B, A)), ((Y, A),))

    def arrangement(source, middle, pair):
        for span in (pair[0], middle, pair[1]):
            for unit in span: put(source, 7, unit)
        derived = machine._inner._derive_networks()
        assert any(row.source == source for row in derived.values()), (
            source, pair, middle, tuple(machine._inner._inner.recipes.values()))
        return machine.tick()

    # Recruit L and a separate real composite Q used as the W->L marginal.
    source_cursor = 20000
    for middle in (middle_l, middle_q):
        for pair in training_pairs:
            phase = arrangement(source_cursor, middle, pair); source_cursor += 1
            assert not phase.actual_composite_roots
    assert len(machine._inner.constructors) >= 2
    constructor_l = next(row.identity for row in machine._inner.constructors.values()
                         if row.middle_recipe == learned[(7, middle_l)])
    constructor_q = next(row.identity for row in machine._inner.constructors.values()
                         if row.middle_recipe == learned[(7, middle_q)])
    checks["two_actual_composite_classes_recruited"] = constructor_l != constructor_q

    l_pair_index = 0; q_pair_index = 0; auxiliary_source = 24000
    def actual_composite(source, middle, constructor, which, expand=True):
        nonlocal l_pair_index, q_pair_index, auxiliary_source
        index = l_pair_index if which == "l" else q_pair_index
        batch = expansion_batches[index]
        if expand:
            for pair in batch[:-1]:
                phase = arrangement(auxiliary_source, middle, pair); auxiliary_source += 1
                assert not phase.actual_composite_roots
        phase = arrangement(source, middle, batch[-1])
        if which == "l": l_pair_index += 1
        else: q_pair_index += 1
        assert len(phase.actual_composite_roots) == 1, (phase, batch[-1])
        node = machine.actual_nodes[-1]
        assert node.kind == NODE_COMPOSITE and node.recipe == constructor
        return node

    def feed_span(source, channel, span):
        for unit in span: put(source, channel, unit)
        node = machine.actual_nodes[-1]
        assert node.kind == NODE_SPAN and node.channel == channel
        return node

    def washout(source, first_target):
        before = (machine.recipes.copy(), tuple(machine.actual_nodes))
        first_sequence = machine.next_sequence
        for offset in range(21):
            contact = boundary.seal_withdrawal(
                machine.session, machine.next_sequence, source, 79,
                first_target + offset)
            machine.ingest_withdrawal(contact)
        assert machine.next_sequence == first_sequence + 21
        assert (machine.recipes, tuple(machine.actual_nodes)) == before

    l_to_c_sources = []
    # Epoch one: nominate actual L -> C twice.  W is absent.
    for source in (21001, 21002):
        actual_composite(source, middle_l, constructor_l, "l")
        feed_span(source, 9, consequence)
        l_to_c_sources.append(source)
    lc_hypotheses = [row for row in machine.hypotheses.values()
                     if row.cue_kind == NODE_COMPOSITE and row.cue_recipe == constructor_l
                     and row.target_kind == NODE_SPAN
                     and row.target_recipe == learned[(9, consequence)]]
    assert len(lc_hypotheses) == 1

    # Make a resident non-C marginal on channel 9 from C's already learned
    # three-unit suffix; no host supplies a target or baseline identity.
    for source in (21101, 21102, 21103, 21104): feed_span(source, 9, distractor)

    # Qualify L -> C from two new future-outcome sources.
    for source in (21201, 21202):
        actual_composite(source, middle_l, constructor_l, "l")
        cue = machine.actual_nodes[-1]
        eligible_lc = [row for row in machine.hypotheses.values()
                       if (row.cue_kind, row.cue_channel, row.cue_recipe)
                       == (cue.kind, cue.channel, cue.recipe)]
        marginal_lc = machine._marginal(
            NODE_SPAN, 9, cue.observed_sequence, cue.eligibility_epoch)
        assert (eligible_lc and marginal_lc is not None
                and marginal_lc != learned[(9, consequence)]
                and machine._span_recipe(marginal_lc) is not None), (
            cue, eligible_lc, marginal_lc,
            [(row.recipe, row.born_sequence) for row in machine.actual_nodes
             if row.kind == NODE_SPAN and row.channel == 9])
        phase = machine.tick(); assert phase.prediction_tickets
        feed_span(source, 9, consequence)
    lc_recipes = [row for row in machine.recipes.values()
                  if row.cue_kind == NODE_COMPOSITE and row.cue_recipe == constructor_l
                  and row.target_kind == NODE_SPAN
                  and row.target_recipe == learned[(9, consequence)]]
    assert len(lc_recipes) == 1
    checks["l_to_c_requires_two_future_sources"] = lc_recipes[0].credit > 0

    # Washout contains no W or C structural occurrence.
    washout(22000, 800000)

    # Epoch two: nominate W -> actual L with C absent.
    for source in (22101, 22102):
        feed_span(source, 8, world)
        actual_composite(source, middle_l, constructor_l, "l")
    wl_hypotheses = [row for row in machine.hypotheses.values()
                     if row.cue_kind == NODE_SPAN and row.cue_recipe == learned[(8, world)]
                     and row.target_kind == NODE_COMPOSITE
                     and row.target_recipe == constructor_l]
    assert len(wl_hypotheses) == 1

    # Four actual Q composites establish a real, resident composite marginal.
    for source in (22201, 22202, 22203, 22204):
        actual_composite(source, middle_q, constructor_q, "q")

    # Qualify W -> L twice.  Pending composite tickets settle only when the
    # inner close creates an actual L rebind.
    for source in (22301, 22302):
        batch = expansion_batches[l_pair_index]
        for pair in batch[:-1]:
            phase = arrangement(auxiliary_source, middle_l, pair)
            auxiliary_source += 1
            assert not phase.actual_composite_roots
        feed_span(source, 8, world)
        phase = machine.tick(); assert phase.prediction_tickets
        actual_composite(source, middle_l, constructor_l, "l", expand=False)
    wl_recipes = [row for row in machine.recipes.values()
                  if row.cue_kind == NODE_SPAN and row.cue_recipe == learned[(8, world)]
                  and row.target_kind == NODE_COMPOSITE
                  and row.target_recipe == constructor_l]
    assert len(wl_recipes) == 1
    checks["w_to_l_requires_two_actual_future_sources"] = wl_recipes[0].credit > 0

    direct_key = (NODE_SPAN, 8, learned[(8, world)],
                  NODE_SPAN, 9, learned[(9, consequence)])
    def key(row):
        return (row.cue_kind, row.cue_channel, row.cue_recipe,
                row.target_kind, row.target_channel, row.target_recipe)
    checks["no_direct_w_c_nomination"] = not any(
        (row.cue_kind, row.cue_channel, row.cue_recipe,
         row.target_kind, row.target_channel, row.target_recipe) == direct_key
        for row in machine.nominations)
    checks["no_direct_w_c_hypothesis_or_recipe"] = (
        not any(key(row) == direct_key for row in machine.hypotheses.values())
        and not any(key(row) == direct_key for row in machine.recipes.values()))

    # A second disjoint epoch makes the final W-only traversal prospective;
    # neither actual L nor C is present in this washout.
    washout(23000, 900000)

    if fixture_hook is not None:
        return fixture_hook({
            "boundary": boundary, "machine": machine, "put": put,
            "feed_span": feed_span, "washout": washout,
            "world": world, "consequence": consequence,
            "distractor": distractor, "constructor_l": constructor_l,
            "learned": learned, "checks": checks,
        })

    edge_witnesses_before = len([row for row in machine.witnesses
                                 if not row.relation_recipe_roots])
    learning_revision_before = (tuple(machine.nominations),
                                tuple(machine.hypotheses.values()),
                                tuple(machine.recipes.values()))
    prospective_before = len(machine.actual_nodes)
    feed_span(23101, 8, world)
    chained = machine.tick()
    assert chained.prediction_tickets
    ticket = next(row for row in machine.pending.values()
                  if row.prospective_middle_recipe == constructor_l)
    checks["world_alone_opens_two_edge_c_ticket"] = (
        ticket.target_recipe == learned[(9, consequence)]
        and len(ticket.relation_recipe_roots) == 2
        and ticket.prospective_middle_kind == NODE_COMPOSITE)
    checks["prospective_l_not_actual_or_self_teaching"] = len(machine.actual_nodes) == prospective_before + 1
    feed_span(23101, 9, consequence)
    chain_witness = machine.witnesses[-1]
    settled_node = next(row for row in machine.actual_nodes
                        if row.identity == chain_witness.observed_node)
    checks["exact_chain_settlement_and_ancestry"] = (
        chain_witness.difference > 0 and len(chain_witness.ancestry) == len(consequence)
        and all(row.ticket == ticket.ticket
                and row.prospective_middle_recipe == constructor_l
                and row.ticket_envelope_root == ticket.envelope_root
                and row.nomination_roots == ticket.nomination_roots
                and row.first_relation_recipe and row.second_relation_recipe
                and row.relation_witness_roots == ticket.relation_witness_roots
                and row.relation_source_roots == ticket.relation_source_roots
                and row.target_evidence_revision == settled_node.evidence_revision
                and row.target_source_roots == settled_node.source_roots
                and row.target_ancestry_roots == settled_node.ancestry_roots
                for row in chain_witness.ancestry))
    checks["chain_does_not_credit_constituent_edges"] = (
        len([row for row in machine.witnesses if not row.relation_recipe_roots])
        == edge_witnesses_before)
    checks["pending_consequence_never_becomes_teaching_evidence"] = (
        learning_revision_before == (tuple(machine.nominations),
                                     tuple(machine.hypotheses.values()),
                                     tuple(machine.recipes.values())))
    checks["ticket_session_incarnation_and_deadline_authenticate"] = (
        ticket.session == machine.session and ticket.incarnation == 1
        and ticket.opened_sequence < settled_node.observed_sequence
        <= ticket.deadline_sequence
        and ticket.opened_sequence < settled_node.born_sequence
        and bool(ticket.relation_witness_roots))
    checks["ticket_envelope_detects_incarnation_tamper"] = (
        machine._valid_ticket_envelope(ticket)
        and not machine._valid_ticket_envelope(replace(ticket, incarnation=2)))

    # A matched resident marginal is negative causal-difference evidence, not
    # positive credit, and prospective chain settlement never trains an edge.
    feed_span(23102, 8, world)
    wrong_phase = machine.tick(); assert wrong_phase.prediction_tickets
    wrong_ticket = next(row for row in machine.pending.values()
                        if row.ticket in wrong_phase.prediction_tickets)
    direct_before_wrong = len([row for row in machine.witnesses
                               if not row.relation_recipe_roots])
    feed_span(23102, 9, distractor)
    wrong_witness = machine.witnesses[-1]
    checks["wrong_consequence_is_negative_not_credit"] = (
        wrong_witness.ticket == wrong_ticket.ticket
        and wrong_witness.difference == -1
        and bool(wrong_witness.relation_recipe_roots)
        and len([row for row in machine.witnesses
                 if not row.relation_recipe_roots]) == direct_before_wrong)

    # A target arriving beyond the frozen deadline cannot settle the old ticket.
    feed_span(23103, 8, world)
    stale_phase = machine.tick(); assert stale_phase.prediction_tickets
    stale_ticket = stale_phase.prediction_tickets[0]
    stale_witnesses = {row.ticket for row in machine.witnesses}
    washout(23200, 910000)
    machine.tick()
    checks["stale_ticket_expires_without_witness_or_credit"] = (
        stale_ticket not in machine.pending
        and stale_ticket not in {row.ticket for row in machine.witnesses}
        and stale_ticket not in stale_witnesses)

    # The stale-expiry tick opens a fresh chain ticket from the same current W.
    # Withdrawing one frozen underlying edge-evidence source must cancel that
    # ticket and cascade the exact relation revision before any consequence.
    dependency_ticket = next(iter(machine.pending.values()))
    dependency_source = next(source for source in settled_node.source_roots
                             if source != settled_node.source)
    dependency_relation_roots = {row.identity for row in machine.recipes.values()
                                 if dependency_source in row.source_roots}
    assert dependency_relation_roots and dependency_source in chain_witness.source_roots
    dependency_withdrawal = boundary.seal_withdrawal(
        machine.session, machine.next_sequence, 98901, 79, dependency_source)
    machine.ingest_withdrawal(dependency_withdrawal)
    checks["mid_ticket_underlying_source_withdrawal_cancels"] = (
        dependency_ticket.ticket not in machine.pending
        and dependency_relation_roots.isdisjoint(machine.recipes)
        and chain_witness.identity not in {row.identity for row in machine.witnesses}
        and all(dependency_source not in row.source_roots
                for row in machine.recipes.values()))
    checks["settled_target_transitive_withdrawal_cascades_credit"] = (
        all(dependency_source not in row.source_roots
            for row in machine.witnesses))

    blob = machine.checkpoint()
    wrong_boundary = admit_channel_sequence_boundary_v1()
    checks["wrong_boundary_checkpoint_refuses"] = refuses(
        lambda: ResidentCompositeCuePredictionV1.restore(blob, wrong_boundary))
    replay = ResidentCompositeCuePredictionV1.restore(blob, boundary)
    checks["event_only_checkpoint_exact"] = (
        replay.checkpoint() == blob and replay.recipes == machine.recipes
        and replay.witnesses == machine.witnesses)
    before_altered = replay.checkpoint()
    altered = boundary.seal_withdrawal(replay.session, replay.next_sequence,
                                       99100, 79, 99999991)
    replay.ingest_withdrawal(altered)
    checks["altered_authenticated_input_diverges"] = replay.checkpoint() != before_altered
    corrupt = bytearray(blob); corrupt[-2] ^= 1
    checks["corrupt_checkpoint_refuses"] = refuses(
        lambda: ResidentCompositeCuePredictionV1.restore(bytes(corrupt), boundary))
    checks["literal_and_extent_bypass_refuse"] = (
        refuses(lambda: boundary.seal_sample(
            1, 1, 1, 1, b"x", (1, 1)), "literal")
        and refuses(lambda: boundary.seal_sample(
            1, 1, 1, 1, (1, 2), (1, 1)), "extent"))
    public = {name.lower() for name in dir(ResidentCompositeCuePredictionV1)
              if not name.startswith("_")}
    forbidden = {"bind", "scene", "target", "outcome", "candidate", "template",
                 "children", "expected", "surface", "emit", "prompt"}
    checks["no_host_role_or_surface_api"] = not forbidden.intersection(public)

    withdrawal_source = l_to_c_sources[0]
    withdrawal = boundary.seal_withdrawal(replay.session, replay.next_sequence,
                                          99001, 7, withdrawal_source)
    replay.ingest_withdrawal(withdrawal)
    checks["source_withdrawal_cascades"] = all(
        withdrawal_source not in row.source_roots for row in replay.recipes.values())

    elapsed = time.perf_counter() - started
    checks["hard_runtime_bound"] = elapsed < 60.0
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise SystemExit("FOUNDRY_RESIDENT_COMPOSITE_CUE_PREDICTION_RED " + ",".join(failed))

    core = Path(__file__).with_name("reference_resident_composite_cue_prediction_v1.py")
    receipt = {
        "contract": "FOUNDRY_RESIDENT_COMPOSITE_CUE_PREDICTION_GREEN",
        "claim": "RESIDENT_DISJOINT_TWO_EDGE_STRUCTURAL_FUTURE_PREDICTION_REFERENCE_ONLY",
        "reference_only": True, "adult_attached": False, "runtime_llm": False,
        "host_numeric_fixture": True, "human_language_claim": False,
        "semantic_claim": False, "causal_claim": False, "surface_generation": False,
        "coactivity": "HYPOTHESIS_ONLY",
        "prediction_difference": "CANDIDATE_HIT_MINUS_RESIDENT_KIND_CHANNEL_MARGINAL_HIT",
        "prospective_middle": "TICKET_ONLY/NEVER_ACTUAL_OR_TEACHING_EVIDENCE",
        "direct_w_c_support": 0, "graph_flip": False,
        "physical_direct_parity": "NOT_RUN/RED",
        "production_ir": "ResidentRecipeIrProgram.vcurrent",
        "translation_status": "UNDEFINED",
        "parity_status": "NOT_RUN/RED",
        "executor": "CPU_ONLY_STABLE_PYTHON_REFERENCE",
        "capability_status": "NON_CAPABILITY_REFERENCE_ASSAY",
        "runtime_limit_seconds": 60, "elapsed_ms": round(elapsed * 1000, 3),
        "core_sha256": hashlib.sha256(core.read_bytes()).hexdigest(),
        "checks": checks,
        "remaining_red": ["PHYSICAL_COMMON_CAUSE", "CAUSAL_INTERVENTION",
                          "SEMANTIC_WORLD_REFERENCE", "PUBLIC_ACTION",
                          "PRODUCTION_RECIPE_IR_TRANSLATION", "DIRECT_PHYSICAL_PARITY"],
    }
    print("FOUNDRY_RESIDENT_COMPOSITE_CUE_PREDICTION_GREEN")
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__": main()
