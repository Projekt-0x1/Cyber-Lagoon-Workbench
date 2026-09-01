#!/usr/bin/env python3
"""Ambiguous lexical contact must recruit resident clarification, not vanish."""
from __future__ import annotations

import copy
import json
import time

from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2,
    canonical_life_function_curriculum_v2,
    canonical_species_program_v2,
)


RAW = b"morning sunlight warms the greenhouse aka glimmer heats the glasshouse"
ANCHOR = b"morning sunlight warms the greenhouse"
KNOWN = 0xA103
OTHER = 0xBEEF


def fork(runtime):
    return type(runtime).restore(runtime.program, copy.deepcopy(runtime.checkpoint()))


def develop_to_mark(mark):
    runtime = ReferenceLifeFunctionRuntimeV2(canonical_species_program_v2())
    for event in canonical_life_function_curriculum_v2().events:
        runtime.apply(event)
        if event.lane == "checkpoint_mark" and event.payload == (mark,):
            return runtime
    raise KeyError(mark)


def main():
    started = time.perf_counter()
    base = develop_to_mark("lexical_carrier_ready")

    unique = fork(base)
    unique_surface, unique_action = unique.contact_utterance(
        RAW, 0xFA01, 0xFA01)

    ambiguous = fork(base)
    language = ambiguous.adult.language_adult.language
    for source in (0xFA10, 0xFA11):
        language.observe_naming(OTHER, ANCHOR, source)
    before = language.checkpoint()
    ambiguous_surface, ambiguous_action = ambiguous.contact_utterance(
        RAW, 0xFA01, 0xFA01)
    after = language.checkpoint()
    alternatives = tuple(
        (feature, bytes(row[1]).decode(), tuple(row[2]))
        for feature in (KNOWN, OTHER)
        for row in language.lexeme_observations(feature)
        if bytes(row[1]).startswith(b"glimmer"))
    clarification_settled = ambiguous.settle_contact_consequence(
        ambiguous_action, 0xFD10, 1, 0, True)
    revised_surface = bytes(language.lexeme(KNOWN) or ())
    effect = ambiguous.adult.language_adult.leaf(100, (0xA105,))
    heldout_question = (
        b"why is it the case that " + bytes(effect.surface).lower() + b"?")
    heldout_reply, heldout_action = ambiguous.contact_utterance(
        heldout_question, 0xFA02, 0xFA02)
    heldout_receipt = ambiguous.adult.pending_causal_dialogue_actions.get(
        heldout_action)
    heldout_certificate = (() if heldout_receipt is None else
                           ambiguous.adult._causal_action_coordinates(
                               heldout_receipt))

    yoked = fork(base)
    yoked_language = yoked.adult.language_adult.language
    for source in (0xFA10, 0xFA11, 0xFA12):
        yoked_language.observe_naming(OTHER, ANCHOR, source)
    yoked_surface, yoked_action = yoked.contact_utterance(
        RAW, 0xFA01, 0xFA01)

    checks = {
        "unique_history_still_externalizes_one_resident_hypothesis":
            bool(unique_surface) and unique_action > 0,
        "ambiguous_history_retains_two_future_causal_alternatives":
            len(alternatives) == 2,
        "ambiguity_originates_earned_clarification_action":
            bool(ambiguous_surface) and ambiguous_action > 0,
        "ambiguity_changes_resident_state_without_guessing":
            before != after and len(
                alternatives) == 2,
        "equal_prior_support_preserves_uncertainty_and_silence":
            not yoked_surface and yoked_action == 0 and not
            yoked.adult.pending_endogenous_inquiry_actions,
        "clarification_consequence_revises_the_selected_relation":
            clarification_settled and revised_surface.startswith(b"glimmer"),
        "heldout_composition_has_resident_causal_certificate":
            bool(heldout_reply) and bool(heldout_certificate) and
            revised_surface in heldout_reply,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    print(json.dumps({
        "schema": "cyber-lagoon.resident-lexical-ambiguity-clarification.v1",
        "status": "GREEN" if not failed else "RED",
        "checks": checks,
        "failed_checks": failed,
        "unique_surface": unique_surface.decode(),
        "ambiguous_surface": ambiguous_surface.decode(),
        "equal_support_surface": yoked_surface.decode(),
        "revised_surface": revised_surface.decode(),
        "heldout_question": heldout_question.decode(),
        "heldout_reply": heldout_reply.decode(),
        "heldout_relation_certificate": heldout_certificate,
        "retained_alternatives": alternatives,
        "language_state_changed": before != after,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "visible_language_gain":
            "AMBIGUITY_RECRUITS_RESIDENT_CLARIFICATION_THEN_HELDOUT_COMPOSITION",
    }, indent=2))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
