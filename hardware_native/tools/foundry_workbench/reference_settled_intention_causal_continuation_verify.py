#!/usr/bin/env python3
"""Held-out RED: one settled request sustains a causal explanation through silence."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import time

from life_function_factory_v1 import build_cache, load_mark
from reference_claude_body_causal_uptake_verify import _focus_and_paraphrase
from reference_life_extension_causal_depth_plus_v1 import CONVERGENT_CAUSAL_SOURCES
from reference_language_mastery_claude_gateway_v1 import body_source_identity
from reference_slow_resource_history_v1 import (
    LOAD_SAMPLE_CAP_Q16, SUSTAINED_MIN_CONTACTS,
)


REQUEST = b"Please continue: What else happens because of that?"
WRAPPER_SOURCES = (0xFA41, 0xFA42)


def _pending(runtime, surface, channel):
    digest = hashlib.sha256(surface).hexdigest()
    rows = tuple(row for row in runtime.adult.pending_causal_dialogue_actions.values()
                 if int(row.channel) == int(channel)
                 and row.surface_digest == digest)
    return rows[0] if len(rows) == 1 else None


def _request(runtime, channel, settle=True):
    _effect, question, *_ = _focus_and_paraphrase(runtime)
    first, first_action = runtime.contact_utterance(question, channel, channel)
    if not first_action or not runtime.settle_contact_consequence(
            first_action, channel, 0, 0, True):
        return b"", 0, None
    runtime.adult.observe_authenticated_causal_dialogue_contact(
        REQUEST, channel, channel)
    surface, action = runtime.contact_utterance(REQUEST, channel, channel)
    receipt = runtime.adult.pending_causal_dialogue_actions.get(action)
    if settle and action:
        runtime.settle_contact_consequence(action, channel, 0, 0, True)
    return surface, action, receipt


def _quiet_chain(runtime, channel, limit=8, lesion_lineage=False):
    rows = []
    for _ in range(limit):
        if lesion_lineage:
            runtime.adult.recent_causal_dialogue_actions.clear()
        surface = runtime.quiet_public_opportunity(channel, channel)
        if not surface:
            break
        receipt = _pending(runtime, surface, channel)
        coordinate = (() if receipt is None else
                      runtime.adult._causal_action_leading_coordinate(receipt))
        rows.append((bytes(surface), receipt, coordinate))
        if receipt is None or not runtime.settle_contact_consequence(
                receipt.identity, channel, 0, 0, True):
            break
    return tuple(rows), runtime.quiet_public_opportunity(channel, channel)


def _arm(base, channel, *, settle=True, withdraw=False, interrupt=False,
         lesion_lineage=False, permute_lineage=False,
         withdraw_convergent=False):
    runtime = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
    requested, action, request_receipt = _request(runtime, channel, settle)
    if withdraw:
        for source in WRAPPER_SOURCES:
            runtime.adult.language.withdraw_source(source)
    if withdraw_convergent:
        for source in CONVERGENT_CAUSAL_SOURCES:
            runtime.adult.language_adult.world_causal_learning.withdraw_source(source)
    if interrupt:
        other = body_source_identity("settled-intention-interrupt")
        runtime.contact_utterance(b"an unrelated physical interruption", other, other)
    if permute_lineage:
        runtime.adult.recent_causal_dialogue_actions = dict(reversed(tuple(
            runtime.adult.recent_causal_dialogue_actions.items())))
    rows, final_silence = _quiet_chain(runtime, channel,
                                      lesion_lineage=lesion_lineage)
    return {
        "requested": requested,
        "action": action,
        "request_receipt": request_receipt,
        "rows": rows,
        "final_silence": final_silence,
        "runtime": runtime,
    }


def _continuity_probe(base, channel):
    correct = _arm(base, channel)
    explicit = _arm(base, channel, lesion_lineage=True)
    permuted = _arm(base, channel, permute_lineage=True)
    edge_cut = _arm(base, channel, withdraw_convergent=True)
    coordinates = tuple(row[2] for row in correct["rows"])
    explicit_coordinates = tuple(row[2] for row in explicit["rows"])
    reduced_causes = tuple(bytes(correct["runtime"].adult.language_adult.
                          _leaf_surface(int(row[2][1])))
                           for row in correct["rows"])
    continuation_factors = set(correct["runtime"].adult.
                               _causal_continuation_factors())
    used_continuations = {int(factor) for _surface, receipt, _coordinate
                          in correct["rows"] for factor in receipt.factors
                          if int(factor) in continuation_factors}
    def first_clause(surface):
        boundaries = tuple(index for index, value in enumerate(surface)
                           if value in (ord('.'), ord('?'), ord('!')))
        return surface if not boundaries else surface[:boundaries[0] + 1]
    def diamond_witness(arm):
        if len(arm["rows"]) < 2 or arm["rows"][0][1] is None:return False
        first_surface, first_receipt, first_coordinate = arm["rows"][0]
        second_coordinate = arm["rows"][1][2]
        acted = arm["runtime"].adult._causal_action_coordinates(first_receipt)
        sibling_effects = tuple(int(row[2]) for row in acted
                                if int(row[1]) == int(first_coordinate[1])
                                and int(row[2]) != int(first_coordinate[2]))
        resolutions = arm["runtime"].adult.language_adult.world_causal_learning.current_resolutions()
        for sibling in sibling_effects:
            sibling_surface = bytes(arm["runtime"].adult.language_adult._leaf_surface(sibling))
            leading_surface = bytes(arm["runtime"].adult.language_adult._leaf_surface(int(first_coordinate[2])))
            converges = any(
                arm["runtime"].adult.language_adult.leaf_equivalent(int(row[2]), sibling)
                and arm["runtime"].adult.language_adult.leaf_equivalent(
                    int(row[3]), int(second_coordinate[2])) for row in resolutions)
            if (converges and first_surface.find(sibling_surface)
                    > first_surface.find(leading_surface) >= 0
                    and int(second_coordinate[1]) == int(first_coordinate[2])):
                return True
        return False
    checks = {
        "prior_settled_effect_structurally_reduces_each_next_action":
            bool(len(reduced_causes) == len(correct["rows"])
                 and all(cause not in first_clause(surface)
                         for cause, (surface, _receipt, _row)
                         in zip(reduced_causes, correct["rows"]))),
        "diamond_topology_beats_last_mentioned_surface_branch":
            diamond_witness(correct),
        "convergent_source_withdrawal_removes_only_diamond_not_original_chain":
            bool(not diamond_witness(edge_cut)
                 and [row[0] for row in edge_cut["rows"]]
                 == [row[0] for row in correct["rows"]]),
        "learned_recent_factor_competition_varies_continuation_realization":
            len(used_continuations) >= 2,
        "continuity_reduces_bytes_without_losing_certified_coverage":
            bool(explicit_coordinates == coordinates
                 and sum(len(row[0]) for row in correct["rows"]) * 4
                 <= sum(len(row[0]) for row in explicit["rows"]) * 3),
        "lineage_lesion_deoptimizes_to_explicit_causes":
            bool(explicit["rows"] and all(
                bytes(explicit["runtime"].adult.language_adult._leaf_surface(
                    int(coordinate[1]))) in surface
                for surface, _receipt, coordinate in explicit["rows"])),
        "recent_action_storage_permutation_cannot_choose_wording":
            [row[0] for row in permuted["rows"]]
            == [row[0] for row in correct["rows"]],
    }
    return {"checks": checks, "correct": correct, "explicit": explicit,
            "permuted": permuted, "edge_cut": edge_cut,
            "coordinates": coordinates}


def verify_loaded(base):
    """Probe the shared one-Life Adult without birth or developmental replay."""
    probe = _continuity_probe(
        base, body_source_identity("settled-intention-shared-factory"))
    failed = sorted(name for name, passed in probe["checks"].items() if not passed)
    return {
        "pass": not failed, "checks": probe["checks"], "failed": failed,
        "before": [row[0].decode(errors="replace")
                   for row in probe["explicit"]["rows"]],
        "after": [row[0].decode(errors="replace")
                  for row in probe["correct"]["rows"]],
        "before_bytes": sum(len(row[0]) for row in probe["explicit"]["rows"]),
        "after_bytes": sum(len(row[0]) for row in probe["correct"]["rows"]),
        "certified_coordinates": [list(row) for row in probe["coordinates"]],
    }


def main():
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="foundry-settled-intention-") as directory:
        manifest = build_cache(directory)
        base = load_mark(directory, manifest["checkpoints"][-1]["mark"])
        channel = body_source_identity("settled-intention-main")
        continuity = _continuity_probe(base, channel)
        correct = continuity["correct"]
        explicit = continuity["explicit"]
        coordinates = continuity["coordinates"]
        chained = all(coordinates[index - 1][2] == coordinates[index][1]
                      for index in range(1, len(coordinates)))

        no_request = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
        _effect, question, *_ = _focus_and_paraphrase(no_request)
        _surface, action = no_request.contact_utterance(question, channel, channel)
        no_request.settle_contact_consequence(action, channel, 0, 0, True)
        no_request_quiet = no_request.quiet_public_opportunity(channel, channel)

        unsettled = _arm(base, channel, settle=False)
        wrong = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
        _wrong_surface, wrong_action, _wrong_receipt = _request(
            wrong, channel, settle=False)
        wrong_return = wrong.settle_contact_consequence(
            wrong_action + 1, channel, 0, 0, True) if wrong_action else False
        wrong_quiet = wrong.quiet_public_opportunity(channel, channel)
        failed = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
        _failed_surface, _failed_action, failed_receipt = _request(
            failed, channel, settle=False)
        failed_return = (failed.adult.settle_causal_dialogue_return(
            failed_receipt, channel, -(1 << 16), 0, True, False)
                         if failed_receipt is not None else False)
        failed_quiet = failed.quiet_public_opportunity(channel, channel)
        lesioned = _arm(base, channel, withdraw=True)
        interrupted = _arm(base, channel, interrupt=True)

        pressured = type(base).restore(base.program, copy.deepcopy(base.checkpoint()))
        _request(pressured, channel, True)
        for sequence in range(1, SUSTAINED_MIN_CONTACTS + 1):
            pressured.adult.language_adult.settle_body_ingress(
                "settled-intention-pressure", sequence, format(sequence, "064x"),
                LOAD_SAMPLE_CAP_Q16)
        pressure_rows, _pressure_end = _quiet_chain(pressured, channel)
        pressure_causal = tuple(row for row in pressure_rows if row[1] is not None)
        baseline_width = tuple(len(row[1].programs) for row in correct["rows"])
        pressure_width = tuple(len(row[1].programs) for row in pressure_causal)

        checkpoint_subject = type(base).restore(
            base.program, copy.deepcopy(base.checkpoint()))
        _request(checkpoint_subject, channel, True)
        checkpoint = checkpoint_subject.checkpoint()
        restored = type(base).restore(base.program, copy.deepcopy(checkpoint))
        restored_rows, restored_silence = _quiet_chain(restored, channel)

        before = load_mark(directory, "relational_surplus_recovered")
        pre_development = _arm(before, channel)

        checks = {
            **continuity["checks"],
            "one_request_precedes_multi_action_quiet_continuation":
                bool(correct["requested"] and len(correct["rows"]) >= 2),
            "quiet_actions_advance_successive_certified_relations":
                bool(coordinates and all(len(row) == 3 and min(row) > 0
                                         for row in coordinates) and chained),
            "exhausted_frontier_selects_silence":
                not correct["final_silence"],
            "ordinary_answer_without_continuation_request_stays_silent":
                not no_request_quiet,
            "unsettled_and_wrong_ticket_returns_cannot_advance":
                bool(not unsettled["rows"] and not wrong_return and not wrong_quiet),
            "failed_exact_action_cannot_advance_continuation":
                bool(failed_return and not failed_quiet),
            "learned_wrapper_source_withdrawal_abolishes_continuation":
                not lesioned["rows"],
            "unrelated_other_session_interruption_preserves_commitment":
                len(interrupted["rows"]) >= 2,
            "resource_pressure_narrows_but_preserves_grounded_progression":
                bool(len(pressure_causal) >= 2 and max(pressure_width) < max(baseline_width)),
            "checkpoint_preserves_commitment_not_future_surface":
                bool(len(restored_rows) >= 2 and not restored_silence
                     and REQUEST.decode() not in json.dumps(checkpoint, sort_keys=True)
                     and [row[0] for row in restored_rows]
                     == [row[0] for row in correct["rows"]]),
            "pre_development_same_contact_cannot_create_commitment":
                not pre_development["rows"],
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        result = {
            "schema": "cyber-lagoon.settled-intention-causal-continuation.v1",
            "contract": "FOUNDRY_SETTLED_INTENTION_CAUSAL_CONTINUATION_" +
                        ("GREEN" if not failed else "RED"),
            "pass": not failed,
            "reference_only": True,
            "runtime_llm": False,
            "language_phenotype_improved": not failed,
            "visible_language_gain": (
                "LEARNED_STRUCTURAL_CONTINUATIONS_REPLACE_REPEATED_CAUSES_"
                "ACROSS_SILENT_MESSAGE_BOUNDARIES_WITHOUT_LOSING_CERTIFICATES"),
            "public_actions_after_last_contact": len(correct["rows"]),
            "public_bytes_after_last_contact": [len(row[0]) for row in correct["rows"]],
            "program_width_after_last_contact": list(baseline_width),
            "program_width_under_pressure": list(pressure_width),
            "certified_leading_coordinates": [list(row) for row in coordinates],
            "visible_surfaces": [row[0].decode(errors="replace")
                                 for row in correct["rows"]],
            "before_explicit_surfaces": [row[0].decode(errors="replace")
                                         for row in explicit["rows"]],
            "checks": checks,
            "failed": failed,
            "remaining_red": ["STRUCTURAL_WRAPPER_FAMILY",
                              "DIRECT_PARITY", "BROAD_HUMAN_DIALOGUE"],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        print(result["contract"])
        print(json.dumps(result, indent=2, sort_keys=True))
        raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
