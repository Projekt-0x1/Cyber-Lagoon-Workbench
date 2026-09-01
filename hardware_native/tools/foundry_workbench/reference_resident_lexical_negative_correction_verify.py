#!/usr/bin/env python3
"""A rejected resident lexical proposal must visibly expose the next referent."""
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
OTHER = 0xA105
CONTACT = 0xFA01


def fork(runtime):
    return type(runtime).restore(runtime.program, copy.deepcopy(runtime.checkpoint()))


def develop_to_mark(mark):
    runtime = ReferenceLifeFunctionRuntimeV2(canonical_species_program_v2())
    for event in canonical_life_function_curriculum_v2().events:
        runtime.apply(event)
        if event.lane == "checkpoint_mark" and event.payload == (mark,):
            return runtime
    raise KeyError(mark)


def prepare():
    runtime = develop_to_mark("lexical_carrier_ready")
    language = runtime.adult.language_adult.language
    for source in (0xFA10, 0xFA11):
        language.observe_naming(OTHER, ANCHOR, source)
    return runtime


def receipt(runtime, action):
    return runtime.adult.pending_endogenous_inquiry_actions.get(action)


def proposed_feature(runtime, action):
    row = receipt(runtime, action)
    return 0 if row is None else int(row.obligation_effect)


def main():
    started = time.perf_counter()
    runtime = prepare()
    incumbent_other = bytes(runtime.adult.language_adult.language.lexeme(OTHER) or ())

    first_surface, first_action = runtime.contact_utterance(RAW, CONTACT, CONTACT)
    first_feature = proposed_feature(runtime, first_action)
    first_rejected = runtime.settle_contact_consequence(
        first_action, 0xFD10, -1, 0, True)
    corrected_checkpoint = copy.deepcopy(runtime.checkpoint())

    second_surface, second_action = runtime.contact_utterance(RAW, CONTACT, CONTACT)
    second_feature = proposed_feature(runtime, second_action)
    second_accepted = runtime.settle_contact_consequence(
        second_action, 0xFD11, 1, 0, True)
    revised_other = bytes(runtime.adult.language_adult.language.lexeme(OTHER) or ())
    effect = runtime.adult.language_adult.leaf(100, (OTHER,))
    heldout_question = b"why is it the case that " + bytes(effect.surface).lower() + b"?"
    heldout_reply, heldout_action = runtime.contact_utterance(
        heldout_question, 0xFA02, 0xFA02)
    heldout_receipt = runtime.adult.pending_causal_dialogue_actions.get(heldout_action)
    heldout_certificate = (() if heldout_receipt is None else
                           runtime.adult._causal_action_coordinates(heldout_receipt))

    exhausted = ReferenceLifeFunctionRuntimeV2.restore(
        canonical_species_program_v2(), copy.deepcopy(corrected_checkpoint))
    exhausted_surface, exhausted_action = exhausted.contact_utterance(
        RAW, CONTACT, CONTACT)
    exhausted_rejected = exhausted.settle_contact_consequence(
        exhausted_action, 0xFD12, -1, 0, True)
    silence_surface, silence_action = exhausted.contact_utterance(
        RAW, CONTACT, CONTACT)

    uncorrected = prepare()
    uncorrected_surface, uncorrected_action = uncorrected.contact_utterance(
        RAW, CONTACT, CONTACT)
    uncorrected_receipt = receipt(uncorrected, uncorrected_action)
    motor = uncorrected.adult.settle_endogenous_inquiry_motor_return(
        uncorrected_receipt, CONTACT, True)
    closed = uncorrected.adult.settle_endogenous_inquiry_resolution(
        uncorrected_receipt, 0xFD13)
    repeated_surface, repeated_action = uncorrected.contact_utterance(
        b"an unrelated contact", CONTACT, CONTACT)

    restored = ReferenceLifeFunctionRuntimeV2.restore(
        canonical_species_program_v2(), copy.deepcopy(corrected_checkpoint))
    restored_surface, restored_action = restored.contact_utterance(
        RAW, CONTACT, CONTACT)

    checks = {
        "first_question_identifies_supported_candidate":
            bool(first_surface) and first_action > 0 and first_feature == KNOWN,
        "exact_adverse_return_is_bound_to_first_candidate": first_rejected,
        "same_contact_exposes_distinct_remaining_candidate":
            bool(second_surface) and second_action > 0 and
            second_surface != first_surface and second_feature == OTHER and
            incumbent_other in second_surface,
        "favorable_return_revises_only_second_candidate":
            second_accepted and revised_other.startswith(b"glimmer"),
        "corrected_binding_enters_heldout_certified_composition":
            bool(heldout_reply) and revised_other in heldout_reply and
            bool(heldout_certificate),
        "all_rejected_candidates_produce_honest_silence":
            exhausted_rejected and not silence_surface and silence_action == 0,
        "generic_resolution_cannot_bypass_answer_conditioned_lexical_settlement":
            motor and not closed and not repeated_surface and repeated_action == 0
            and uncorrected_receipt.identity in
                uncorrected.adult.pending_endogenous_inquiry_actions,
        "checkpoint_preserves_correction_not_contact":
            restored_action > 0 and restored_surface == second_surface and
            proposed_feature(restored, restored_action) == OTHER,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    print(json.dumps({
        "schema": "cyber-lagoon.resident-lexical-negative-correction.v1",
        "status": "GREEN" if not failed else "RED",
        "checks": checks,
        "failed_checks": failed,
        "first_surface": first_surface.decode(),
        "second_surface": second_surface.decode(),
        "uncorrected_repeat_surface": repeated_surface.decode(),
        "restored_surface": restored_surface.decode(),
        "exhausted_surface": silence_surface.decode(),
        "revised_other": revised_other.decode(),
        "heldout_question": heldout_question.decode(),
        "heldout_reply": heldout_reply.decode(),
        "heldout_relation_certificate": heldout_certificate,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "visible_language_gain":
            "ADVERSE_PROPOSAL_EXPOSES_DISTINCT_REFERENT_THEN_CERTIFIED_COMPOSITION",
    }, indent=2))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
