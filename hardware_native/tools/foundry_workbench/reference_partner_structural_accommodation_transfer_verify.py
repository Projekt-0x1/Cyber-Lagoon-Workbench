#!/usr/bin/env python3
"""Partner-certified form feedback transfers to held-out causal composition."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import time

from life_function_factory_v1 import build_cache, load_mark
from reference_predictive_credit_profile_v1 import Q


CHANNEL = 0xFEA1
OTHER_CHANNEL = 0xFEA2
MOTOR = 0xFEB0


def _clone(runtime):
    return type(runtime).restore(runtime.program, copy.deepcopy(runtime.checkpoint()))


def _relations(runtime):
    adult = runtime.adult
    factors = adult._causal_self_contained_factors()
    rows = []
    for resolution in adult.language_adult.world_causal_learning.current_resolutions():
        effect = int(resolution[3])
        closure = adult.causal_focus_rows(effect)
        if len(closure) != 1:
            continue
        surfaces = tuple(bytes(adult._causal_self_contained_surface(
            closure[0], factor) or b"") for factor in factors)
        proposition = bytes(adult.language_adult._leaf_surface(effect) or b"")
        if proposition and len(surfaces) >= 2 and all(surfaces) \
                and len(set(surfaces)) == len(surfaces):
            question = b"why is it the case that " + \
                proposition.rstrip(b".?").lower() + b"?"
            rows.append((effect, closure[0], question, surfaces))
    return tuple(rows[:2]), tuple(factors)


def _contact(runtime, question, channel=CHANNEL):
    surface, action = runtime.contact_utterance(question, channel, channel)
    receipt = runtime.adult.pending_causal_dialogue_actions.get(int(action))
    return bytes(surface), receipt


def _history(runtime, train, partner_surface, count=2, *,
             partner_channel=CHANNEL, pre_action=False):
    public = []
    changed = []
    if not pre_action:
        surface, receipt = _contact(runtime, train[2])
        public.append(surface)
        if receipt is None or not runtime.adult.settle_causal_dialogue_return(
                receipt, CHANNEL + 0x10, Q, 0, True, True):
            return tuple(public), tuple(changed), False
    for index in range(count):
        changed.append(runtime.adult.observe_authenticated_causal_dialogue_contact(
            partner_surface, CHANNEL + 0x20 + index, partner_channel))
    return tuple(public), tuple(changed), True


def _settled_final(runtime, question):
    restored = type(runtime).restore(runtime.program, copy.deepcopy(runtime.checkpoint()))
    surface, receipt = _contact(restored, question)
    return restored, surface, receipt


def _factor_and_coordinates(runtime, receipt):
    return ((0, ()) if receipt is None or not receipt.factors else
            (int(receipt.factors[0]),
             tuple(runtime.adult._causal_action_coordinates(receipt))))


def verify(runtime):
    relations, factors = _relations(runtime)
    if len(relations) != 2 or len(factors) < 2:
        return {"two_heldout_single_relation_closures_exist": False}, {}
    train, heldout = relations
    cause_factor, alternate_factor = factors[:2]
    same_surface, alternate_surface = train[3][:2]

    baseline = _clone(runtime)
    baseline_surface, baseline_receipt = _contact(baseline, heldout[2])
    baseline_factor, baseline_coordinates = _factor_and_coordinates(
        baseline, baseline_receipt)

    alternative = _clone(runtime)
    alt_public, alt_changed, alt_ok = _history(
        alternative, train, alternate_surface)
    learned_checkpoint = copy.deepcopy(alternative.checkpoint())
    alternative, learned_surface, learned_receipt = _settled_final(
        alternative, heldout[2])
    learned_factor, learned_coordinates = _factor_and_coordinates(
        alternative, learned_receipt)

    same = _clone(runtime)
    same_public, same_changed, same_ok = _history(same, train, same_surface)
    same, same_final, same_receipt = _settled_final(same, heldout[2])
    same_factor, _ = _factor_and_coordinates(same, same_receipt)

    one = _clone(runtime)
    one_public, one_changed, one_ok = _history(
        one, train, alternate_surface, 1)
    one, one_final, one_receipt = _settled_final(one, heldout[2])
    one_factor, _ = _factor_and_coordinates(one, one_receipt)

    yoked = _clone(runtime)
    _yp, yoked_changed, yoked_ok = _history(
        yoked, train, alternate_surface, pre_action=True)
    yoked, yoked_final, yoked_receipt = _settled_final(yoked, heldout[2])
    yoked_factor, _ = _factor_and_coordinates(yoked, yoked_receipt)

    wrong = _clone(runtime)
    wrong_public, wrong_changed, wrong_ok = _history(
        wrong, train, alternate_surface, partner_channel=OTHER_CHANNEL)
    wrong, wrong_final, wrong_receipt = _settled_final(wrong, heldout[2])
    wrong_factor, _ = _factor_and_coordinates(wrong, wrong_receipt)

    restored = type(alternative).restore(
        alternative.program, copy.deepcopy(learned_checkpoint))
    restored_surface, restored_receipt = _contact(restored, heldout[2])
    restored_factor, _ = _factor_and_coordinates(restored, restored_receipt)

    withdrawn = type(alternative).restore(
        alternative.program, copy.deepcopy(learned_checkpoint))
    for source in (CHANNEL + 0x20, CHANNEL + 0x21):
        withdrawn.adult.language.withdraw_source(source)
    withdrawn_surface, withdrawn_receipt = _contact(withdrawn, heldout[2])
    withdrawn_factor, _ = _factor_and_coordinates(withdrawn, withdrawn_receipt)

    world_before = runtime.adult.language_adult.world_causal_learning.checkpoint()
    world_after = alternative.adult.language_adult.world_causal_learning.checkpoint()
    checks = {
        "two_heldout_single_relation_closures_exist": True,
        "training_forms_are_independently_certified_and_distinct": bool(
            same_surface and alternate_surface
            and same_surface != alternate_surface),
        "all_action_bound_partner_histories_are_public_and_assimilated": bool(
            alt_ok and same_ok and one_ok and wrong_ok
            and all(alt_public) and all(same_public) and all(one_public)
            and alt_changed == same_changed == (1, 1)),
        "repeated_alternative_form_transfers_to_heldout_relation": bool(
            learned_receipt is not None and learned_factor == alternate_factor
            and learned_surface == heldout[3][1]
            and learned_surface != baseline_surface),
        "transfer_changes_form_not_resident_causal_coordinates": bool(
            baseline_coordinates and learned_coordinates
            and learned_coordinates == baseline_coordinates),
        "same_form_support_retains_incumbent": bool(
            same_receipt is not None and same_factor == cause_factor
            and same_final == baseline_surface),
        "one_partner_occurrence_cannot_become_form_opcode": bool(
            one_receipt is not None and one_factor == cause_factor
            and one_final == baseline_surface),
        "pre_action_yoked_contact_cannot_train_formulation": bool(
            yoked_ok and yoked_changed == (0, 0)
            and yoked_receipt is not None and yoked_factor == cause_factor
            and yoked_final == baseline_surface),
        "wrong_discourse_channel_cannot_train_focal_partner": bool(
            wrong_receipt is not None and wrong_factor == cause_factor
            and wrong_final == baseline_surface),
        "checkpoint_restores_preference_not_contact_or_surface": bool(
            restored_receipt is not None and restored_factor == alternate_factor
            and restored_surface == learned_surface
            and train[2].decode() not in str(learned_checkpoint)
            and alternate_surface.decode() not in str(learned_checkpoint)),
        "withdrawing_feedback_sources_deoptimizes_to_incumbent": bool(
            withdrawn_receipt is not None and withdrawn_factor == cause_factor
            and withdrawn_surface == baseline_surface),
        "partner_form_feedback_does_not_rewrite_world_truth":
            world_after == world_before,
    }
    phenotype = {
        "training_question": train[2].decode(),
        "partner_alternative": alternate_surface.decode(),
        "heldout_question": heldout[2].decode(),
        "before": baseline_surface.decode(),
        "after": learned_surface.decode(),
        "same_form_control": same_final.decode(),
        "one_occurrence_control": one_final.decode(),
        "yoked_control": yoked_final.decode(),
        "wrong_channel_control": wrong_final.decode(),
        "withdrawn": withdrawn_surface.decode(),
        "factors": {
            "incumbent": cause_factor,
            "alternate": alternate_factor,
            "before": baseline_factor,
            "after": learned_factor,
        },
        "causal_coordinates_sha256": hashlib.sha256(
            repr(learned_coordinates).encode()).hexdigest(),
    }
    return checks, phenotype


def main():
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(
            prefix="foundry-partner-structural-accommodation-") as directory:
        manifest = build_cache(directory)
        runtime = load_mark(directory, manifest["checkpoints"][-1]["mark"])
        checks, phenotype = verify(runtime)
    failed = tuple(name for name, passed in checks.items() if not passed)
    print("FOUNDRY_PARTNER_STRUCTURAL_ACCOMMODATION_" +
          ("GREEN" if not failed else "RED"))
    print(json.dumps({
        "schema": "cyber-lagoon.partner-structural-accommodation.v1",
        "status": "GREEN" if not failed else "RED",
        "checks": checks,
        "failed": failed,
        "phenotype": phenotype,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
        "runtime_llm": False,
        "visible_language_gain":
            "PARTNER_CERTIFIED_FORM_FEEDBACK_TRANSFERS_TO_HELDOUT_CAUSAL_CONTENT",
    }, indent=2))
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
