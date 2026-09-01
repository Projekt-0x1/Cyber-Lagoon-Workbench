#!/usr/bin/env python3
"""Natural contact keeps exact resident action authority through consequence."""
from __future__ import annotations

import copy
import hashlib
import tempfile
import time

from life_function_factory_v1 import build_cache, load_mark
from reference_predictive_credit_profile_v1 import Q


CHANNEL = 0xFC01


def _focus(runtime):
    candidates = []
    for row in runtime.adult.language_adult.world_causal_learning.current_resolutions():
        effect = int(row[3])
        surface = bytes(runtime.adult.language_adult._leaf_surface(effect) or b"")
        closure = runtime.adult.causal_focus_rows(effect)
        if surface and closure:
            candidates.append((len(closure), effect, surface))
    if not candidates:
        return 0, b""
    _depth, effect, proposition = max(candidates)
    return effect, b"why is it the case that " + proposition.rstrip(b".?").lower() + b"?"


def _contact(runtime, question, source):
    action = runtime.contact_utterance(question, source, CHANNEL)
    if not isinstance(action, tuple) or len(action) != 2:
        return bytes(action or b""), 0
    return bytes(action[0] or b""), int(action[1] or 0)


def _apply_history(runtime, question, outcome, independent, count=3, offset=0):
    public = []
    for index in range(count):
        surface, action = _contact(runtime, question, 0xFC10 + offset + index)
        public.append(surface)
        if action <= 0:
            return public, False
        if not runtime.settle_contact_consequence(
                action, 0xFC20 + offset + index, outcome, outcome, independent):
            return public, False
        if not runtime.observe_contact_background(
                action, 0xFC30 + offset + index, False):
            return public, False
    return public, True


def _history(runtime, question, outcome, independent, count=3, offset=0):
    public, settled = _apply_history(
        runtime, question, outcome, independent, count, offset)
    if not settled:
        return public, 0, False
    surface, action = _contact(runtime, question, 0xFC40)
    public.append(surface)
    return public, action, bool(surface and action)


def _receipt(runtime, action):
    return runtime.adult.pending_causal_dialogue_actions.get(int(action))


def _coordinates(runtime, action):
    row = _receipt(runtime, action)
    return (() if row is None else
            tuple(runtime.adult._causal_action_coordinates(row)))


def verify(runtime):
    effect, question = _focus(runtime)
    if not question:
        return {"heldout_causal_focus_exists": False}, {}

    def fork():
        return type(runtime).restore(runtime.program, copy.deepcopy(runtime.checkpoint()))

    exposed = fork()
    exposed_surface, exposed_action = _contact(exposed, question, 0xFC50)
    exposed_receipt = exposed.adult.pending_causal_dialogue_actions.get(exposed_action)

    delayed = fork()
    delayed_surface, delayed_action = _contact(delayed, question, 0xFC51)
    delayed_restart = type(delayed).restore(
        delayed.program, copy.deepcopy(delayed.checkpoint()))
    wrong_ticket_refused = bool(delayed_action) and not delayed_restart.settle_contact_consequence(
        delayed_action + 1, 0xFC61, Q, Q, True)
    delayed_settled = bool(delayed_action) and delayed_restart.settle_contact_consequence(
        delayed_action, 0xFC61, Q, Q, True)
    replay_refused = bool(delayed_action) and not delayed_restart.settle_contact_consequence(
        delayed_action, 0xFC62, Q, Q, True)

    favorable = fork()
    adverse = fork()
    yoked = fork()
    one_adverse = fork()
    yoked_adverse = fork()
    favorable_public, favorable_action, favorable_ok = _history(
        favorable, question, Q, True)
    adverse_public, adverse_action, adverse_ok = _history(
        adverse, question, -Q, True)
    yoked_public, yoked_action, yoked_ok = _history(
        yoked, question, Q, False)
    one_adverse_public, one_adverse_action, one_adverse_ok = _history(
        one_adverse, question, -Q, True, 1)
    yoked_adverse_public, yoked_adverse_action, yoked_adverse_ok = _history(
        yoked_adverse, question, -Q, False)

    adverse_receipt = _receipt(adverse, adverse_action)
    favorable_receipt = _receipt(favorable, favorable_action)
    one_adverse_receipt = _receipt(one_adverse, one_adverse_action)
    yoked_adverse_receipt = _receipt(yoked_adverse, yoked_adverse_action)
    adverse_checkpoint = copy.deepcopy(adverse.checkpoint())
    restored_adverse = type(adverse).restore(adverse.program, adverse_checkpoint)
    restored_surface, restored_programs = restored_adverse.adult.compose_causal_component(
        effect, channel=CHANNEL)

    withdrawn = type(adverse).restore(
        adverse.program, copy.deepcopy(adverse_checkpoint))
    adverse_factor = (0 if adverse_receipt is None or not adverse_receipt.factors
                      else int(adverse_receipt.factors[0]))
    withdrawn_sources = set()
    for (context, arity, pieces), sources in withdrawn.adult.language._span_sources.items():
        if int(withdrawn.adult.language.span_factor_identity(
                context, arity, pieces)) == adverse_factor:
            withdrawn_sources.update(map(int, sources))
    grounding = withdrawn.adult.language_adult.world_causal_learning.grounding
    withdrawn_sources.update(map(int, grounding.rows.get(adverse_factor, ())))
    for source in withdrawn_sources:
        withdrawn.adult.language.withdraw_source(source)
        grounding.withdraw_source(source)
    withdrawn_surface, withdrawn_programs = withdrawn.adult.compose_causal_component(
        effect, channel=CHANNEL)

    reversal = fork()
    reversal_adverse_public, reversal_adverse_ok = _apply_history(
        reversal, question, -Q, True, 4, 0x80)
    reversal_favorable_public, reversal_favorable_ok = _apply_history(
        reversal, question, Q, True, 3, 0x90)
    reversal_surface, reversal_action = _contact(reversal, question, 0xFCE0)
    reversal_receipt = _receipt(reversal, reversal_action)

    def state(subject, action):
        receipt = subject.adult.pending_causal_dialogue_actions.get(int(action))
        if receipt is None or not receipt.programs:
            return None, 0
        _rows, _programs, _context, felt = subject.adult._causal_discourse_frontier(
            effect, CHANNEL)
        return felt, len(receipt.programs)

    favorable_felt, favorable_programs = state(favorable, favorable_action)
    adverse_felt, adverse_programs = state(adverse, adverse_action)
    yoked_felt, yoked_programs = state(yoked, yoked_action)

    checks = {
        "heldout_causal_focus_exists": bool(effect and question),
        "natural_contact_exposes_exact_resident_action": bool(
            exposed_surface and exposed_action and exposed_receipt is not None
            and exposed_receipt.surface_digest == hashlib.sha256(exposed_surface).hexdigest()
            and exposed_receipt.source == 0xFC50
            and exposed_receipt.channel == CHANNEL),
        "pending_action_survives_checkpoint_for_delayed_return": bool(
            delayed_surface and delayed_settled),
        "wrong_or_already_settled_ticket_refuses": bool(
            wrong_ticket_refused and replay_refused),
        "all_matched_history_arms_remain_public": bool(
            favorable_ok and adverse_ok and yoked_ok
            and one_adverse_ok and yoked_adverse_ok
            and all(favorable_public) and all(adverse_public) and all(yoked_public)
            and all(one_adverse_public) and all(yoked_adverse_public)),
        "independent_action_background_contrast_earns_control": bool(
            favorable_felt is not None and adverse_felt is not None
            and favorable_felt.controllability_q16 >= Q // 2
            and adverse_felt.controllability_q16 >= Q // 2),
        "favorable_yoked_return_cannot_mint_control": bool(
            yoked_felt is not None and yoked_felt.controllability_q16 < Q // 2),
        "favorable_and_adverse_consequences_remain_dissociated": bool(
            favorable_felt is not None and adverse_felt is not None
            and favorable_felt.valence_q16 > 0 > adverse_felt.valence_q16
            and favorable_public[-1] != adverse_public[-1]
            and _coordinates(favorable, favorable_action) ==
                _coordinates(adverse, adverse_action)
            and favorable_receipt is not None and adverse_receipt is not None
            and favorable_receipt.factors != adverse_receipt.factors),
        "one_adverse_return_does_not_create_valence_opcode": bool(
            one_adverse_receipt is not None and exposed_receipt is not None
            and one_adverse_receipt.factors[:1] == exposed_receipt.factors[:1]),
        "yoked_adverse_return_cannot_select_reformulation": bool(
            yoked_adverse_receipt is not None and exposed_receipt is not None
            and yoked_adverse_receipt.factors[:1] == exposed_receipt.factors[:1]),
        "checkpoint_restores_changed_formulation_without_transcript": bool(
            adverse_receipt is not None
            and restored_surface == adverse_public[-1]
            and tuple(restored_programs) == tuple(adverse_receipt.programs)
            and question.decode() not in str(adverse_checkpoint)
            and adverse_public[-1].decode() not in str(adverse_checkpoint)),
        "withdrawn_alternate_deoptimizes_to_remaining_certificate": bool(
            adverse_receipt is not None and withdrawn_sources and withdrawn_surface
            and withdrawn_surface != adverse_public[-1]
            and len(withdrawn_programs) == len(adverse_receipt.programs)),
        "later_reversal_can_restore_incumbent_without_retry_state": bool(
            reversal_adverse_ok and reversal_favorable_ok and reversal_action > 0
            and reversal_receipt is not None and favorable_receipt is not None
            and reversal_receipt.factors[:1] == favorable_receipt.factors[:1]
            and reversal_surface != adverse_public[-1]
            and set(_coordinates(favorable, favorable_action)).issubset(
                set(_coordinates(reversal, reversal_action)))),
        "same_question_different_lived_control_changes_composition": bool(
            favorable_programs == adverse_programs == 1
            and yoked_programs > favorable_programs
            and len(yoked_public[-1]) > len(favorable_public[-1])
            and len(yoked_public[-1]) > len(adverse_public[-1])),
    }
    phenotype = {
        "heldout_focus_identity": hex(effect),
        "same_final_question": question.decode(errors="replace"),
        "initial_public_bytes": len(exposed_surface),
        "favorable": favorable_public[-1].decode(errors="replace") if favorable_public else "",
        "adverse": adverse_public[-1].decode(errors="replace") if adverse_public else "",
        "yoked": yoked_public[-1].decode(errors="replace") if yoked_public else "",
        "program_counts": {
            "favorable": favorable_programs,
            "adverse": adverse_programs,
            "yoked": yoked_programs,
        },
        "one_adverse": one_adverse_public[-1].decode(errors="replace"),
        "yoked_adverse": yoked_adverse_public[-1].decode(errors="replace"),
        "restored_adverse": bytes(restored_surface).decode(errors="replace"),
        "withdrawn_alternate": bytes(withdrawn_surface).decode(errors="replace"),
        "reversal": bytes(reversal_surface).decode(errors="replace"),
        "factors": {
            "favorable": [] if favorable_receipt is None else list(favorable_receipt.factors),
            "adverse": [] if adverse_receipt is None else list(adverse_receipt.factors),
        },
        "controllability_q16": {
            "favorable": 0 if favorable_felt is None else favorable_felt.controllability_q16,
            "adverse": 0 if adverse_felt is None else adverse_felt.controllability_q16,
            "yoked": 0 if yoked_felt is None else yoked_felt.controllability_q16,
        },
    }
    return checks, phenotype


def main():
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="foundry-natural-dialogue-consequence-") as directory:
        manifest = build_cache(directory)
        runtime = load_mark(directory, manifest["checkpoints"][-1]["mark"])
        checks, phenotype = verify(runtime)
    failed = sorted(name for name, passed in checks.items() if not passed)
    print("FOUNDRY_NATURAL_DIALOGUE_ACTION_CONSEQUENCE_" + ("GREEN" if not failed else "RED"))
    print({"checks": checks, "phenotype": phenotype,
           "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
           "failed": failed})
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
