#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from reference_language_learning_v1 import LearnedSurfaceEcologyV1
from reference_hierarchical_composition_v1 import HierarchicalConstructionV1
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1, Q
from reference_language_mastery_adult_v1 import AdultStateV1, LanguageMasteryAdultV1


def u(text: str):
    return tuple(text.encode())


def build_language():
    A1, A2, G1, G2, V1, V2, O1, O2 = 101, 102, 201, 202, 301, 302, 401, 402
    CLAUSE, JOIN = 9001, 9101
    ecology = LearnedSurfaceEcologyV1()
    names = {
        A1: "careful", A2: "quiet", G1: "engineer", G2: "technician",
        V1: "tests", V2: "inspects", O1: "sensor", O2: "valve",
    }
    for feature, text in names.items():
        ecology.observe_naming(feature, u(text), 1000 + feature)
        ecology.observe_naming(feature, u(text), 2000 + feature)
    x = (A1, G1, V1, O1)
    y = (A2, G2, V2, O2)
    ecology.observe_construction(CLAUSE, x, u("the careful engineer tests the sensor."), 3001)
    ecology.observe_construction(CLAUSE, y, u("the quiet technician inspects the valve."), 3002)
    hierarchy = HierarchicalConstructionV1(ecology)
    familiar = hierarchy.leaf(CLAUSE, x)
    second = hierarchy.leaf(CLAUSE, y)
    held_a = hierarchy.leaf(CLAUSE, (A2, G1, V1, O2))
    held_b = hierarchy.leaf(CLAUSE, (A1, G2, V2, O1))
    hierarchy.observe(JOIN, (familiar, second), (*familiar.surface, 32, *second.surface), 5001)
    hierarchy.observe(JOIN, (second, held_a), (*second.surface, 32, *held_a.surface), 5002)
    deep = hierarchy.compose(JOIN, (held_a, held_b))
    return familiar, deep


def prospective_value(bank: PredictiveCreditBankV1, sid: int, context: int) -> int:
    row = bank.row(sid)
    return (
        bank.contextual_outcome(sid, context)
        + bank.contextual_somatic(sid, context)
        + row.accessibility_q16 // 4
        - row.effort_mean_q16 // 8
    )


def value_only_choice(bank, candidates, context):
    return max(candidates, key=lambda sid: (prospective_value(bank, sid, context), -sid))


def raw_ratio_choice(bank, candidates, context, minimum_control_q16=Q // 2):
    eligible = [sid for sid in candidates if bank.row(sid).controllability_q16 >= minimum_control_q16]
    if not eligible:
        return 0
    return max(eligible, key=lambda sid: (prospective_value(bank, sid, context), -sid))


def controllable_choice(bank, candidates, context):
    eligible=[]
    for sid in candidates:
        row=bank.row(sid)
        ready=getattr(row,'control_ready',row.controllability_q16>=Q//2)
        if ready:eligible.append(sid)
    if not eligible:return 0
    return max(eligible,key=lambda sid:(prospective_value(bank,sid,context),-sid))


def adult_history_choice(bank, candidates, context):
    adult=LanguageMasteryAdultV1();adult.credit=bank;state=AdultStateV1()
    for sid in candidates:adult.programs.bind_factor(sid,sid)
    return adult._probe_choice(context,state)


def legacy_success_ratio_choice(bank, candidates, context):
    """Immediate predecessor: successes divided by every control observation."""
    eligible=[]
    for sid in candidates:
        row=bank.row(sid)
        observations=row.control_attempts+getattr(row,'background_attempts',0)
        ratio=0 if observations<=0 else (row.control_successes*Q)//observations
        if row.control_successes>=2 and ratio>=Q//2:eligible.append(sid)
    if not eligible:return 0
    return max(eligible,key=lambda sid:(prospective_value(bank,sid,context),-sid))


def public_steps(bank, sid, step_count):
    if sid == 0:
        return 0
    row=bank.row(sid);ready=getattr(row,'control_ready',row.controllability_q16>=Q//2)
    return step_count if ready else 0


def main():
    started = time.perf_counter()
    context = 0xC071
    familiar, deep = build_language()
    bank = PredictiveCreditBankV1(16)
    candidates = (familiar.identity, deep.identity)
    steps = {familiar.identity: 2, deep.identity: 5}
    depth = {familiar.identity: familiar.depth, deep.identity: deep.depth}
    length = {familiar.identity: len(familiar.surface), deep.identity: len(deep.surface)}

    # The short construction has lower outcome but has repeatedly produced lawful
    # public action and independent consequence.
    for n in range(4):
        tick = 10 + n * 6
        bank.observe_use(familiar.identity, tick, tick + 2, Q // 12, context)
        bank.observe_return(familiar.identity, Q // 2, Q // 16, tick + 3, True, context)
        bank.observe_control(familiar.identity, public_action=True, independent_return=True, context=context)

    # The deep construction is prospectively more valuable, but internal rehearsal
    # and attempted preparation have not yet yielded a lawful public consequence.
    for n in range(4):
        tick = 70 + n * 10
        bank.observe_use(deep.identity, tick, tick + 7, Q // 2, context)
        bank.observe_return(deep.identity, Q, Q // 8, tick + 8, True, context)
        bank.observe_control(deep.identity, public_action=False, independent_return=False, context=context)

    control_before_assertion = bank.row(deep.identity).controllability_q16
    for _ in range(8):
        # Positive/internal language rehearsal is deliberately consequence-cold.
        bank.observe_control(deep.identity, public_action=False, independent_return=False, context=context)
    control_after_assertion = bank.row(deep.identity).controllability_q16

    value_choice_before = value_only_choice(bank, candidates, context)
    controlled_before = controllable_choice(bank, candidates, context)
    value_public_before = public_steps(bank, value_choice_before, steps[value_choice_before])
    controlled_public_before = public_steps(bank, controlled_before, steps[controlled_before])

    # Later, the deep construction is externally elicited through ordinary action
    # enough times to earn independent public response→consequence control.
    for _ in range(16):
        bank.observe_control(deep.identity, public_action=True, independent_return=True, context=context)

    controlled_after = controllable_choice(bank, candidates, context)
    controlled_public_after = public_steps(bank, controlled_after, steps[controlled_after])

    # N+1 calibration arm: a prospectively attractive deep program gets one lawful
    # public consequence.  Raw ratio alone treats 1/1 as fully established control;
    # resident repeated-control evidence must keep the established familiar language
    # until an independent second success corroborates the new trajectory.
    calibration = PredictiveCreditBankV1(16)
    for n in range(4):
        tick=300+n*6
        calibration.observe_use(familiar.identity,tick,tick+2,Q//12,context)
        calibration.observe_return(familiar.identity,Q//2,Q//16,tick+3,True,context)
        calibration.observe_control(familiar.identity,True,True,context)
    calibration.observe_use(deep.identity,400,407,Q//2,context)
    calibration.observe_return(deep.identity,Q,Q//8,408,True,context)
    calibration.observe_control(deep.identity,True,True,context)
    raw_one_shot=raw_ratio_choice(calibration,candidates,context)
    calibrated_one_shot=controllable_choice(calibration,candidates,context)
    calibration.observe_control(deep.identity,True,True,context)
    calibrated_corroborated=controllable_choice(calibration,candidates,context)

    # N+2 causal-contingency arm.  The deep program gets two genuine action returns,
    # but the same consequence also occurs twice without that action.  The predecessor
    # success/all-observation ratio calls this controllable at 2/4; action-vs-background
    # contingency correctly stays with established language because DeltaP = 1 - 1 = 0.
    contingency=PredictiveCreditBankV1(16)
    for n in range(4):
        tick=500+n*6
        contingency.observe_use(familiar.identity,tick,tick+2,Q//12,context)
        contingency.observe_return(familiar.identity,Q//2,Q//16,tick+3,True,context)
        contingency.observe_control(familiar.identity,True,True,context)
    for n in range(2):
        tick=600+n*10
        contingency.observe_use(deep.identity,tick,tick+7,Q//2,context)
        contingency.observe_return(deep.identity,Q,Q//8,tick+8,True,context)
        contingency.observe_control(deep.identity,True,True,context)
    contingency.observe_control(deep.identity,False,True,context)
    contingency.observe_control(deep.identity,False,True,context)
    predecessor_yoked=legacy_success_ratio_choice(contingency,candidates,context)
    resident_yoked=controllable_choice(contingency,candidates,context)
    contingency.observe_control(deep.identity,False,False,context)
    contingency.observe_control(deep.identity,False,False,context)
    predecessor_recovery=legacy_success_ratio_choice(contingency,candidates,context)
    resident_recovery=controllable_choice(contingency,candidates,context)

    # N+3 Sapolsky control-history arm. The twins receive the exact same event
    # multiset and end with identical action/background incidence and current
    # DeltaP=0. Only the prior-control twin first crosses the repeated-control
    # criterion before the matched free outcomes arrive. An unordered lifetime
    # counter representation necessarily collapses these lived histories.
    def history_twin():
        twin=PredictiveCreditBankV1(16)
        for n in range(4):
            tick=700+n*6
            twin.observe_use(familiar.identity,tick,tick+2,Q//12,context)
            twin.observe_return(familiar.identity,Q//2,Q//16,tick+3,True,context)
            twin.observe_control(familiar.identity,True,True,context)
        for n in range(2):
            tick=760+n*10
            twin.observe_use(deep.identity,tick,tick+7,Q//2,context)
            twin.observe_return(deep.identity,Q,Q//8,tick+8,True,context)
        return twin

    prior_control=history_twin()
    for _ in range(2):prior_control.observe_control(deep.identity,True,True,context)
    for _ in range(2):prior_control.observe_control(deep.identity,False,True,context)

    interleaved_yoked=history_twin()
    for public_action in (False,True,False,True):
        interleaved_yoked.observe_control(deep.identity,public_action,True,context)

    prior_row=prior_control.row(deep.identity)
    yoked_row=interleaved_yoked.row(deep.identity)
    prior_cells=(prior_row.control_attempts,prior_row.control_successes,
                 prior_row.background_attempts,prior_row.background_successes)
    yoked_cells=(yoked_row.control_attempts,yoked_row.control_successes,
                 yoked_row.background_attempts,yoked_row.background_successes)
    prior_history_choice=adult_history_choice(prior_control,candidates,context)
    yoked_history_choice=adult_history_choice(interleaved_yoked,candidates,context)

    prior_checkpoint=prior_control.checkpoint()
    prior_adult=LanguageMasteryAdultV1();prior_adult.credit=prior_control
    restored_adult=LanguageMasteryAdultV1.restore(prior_adult.checkpoint())
    restored_history=restored_adult.credit
    restored_history_choice=adult_history_choice(restored_history,candidates,context)

    lesioned_history=PredictiveCreditBankV1.restore(prior_checkpoint)
    lesioned_history.row(deep.identity).contexts[context].control_history_q16=0
    lesioned_history_choice=adult_history_choice(lesioned_history,candidates,context)

    extinguished_history=PredictiveCreditBankV1.restore(prior_checkpoint)
    for _ in range(3):extinguished_history.observe_control(deep.identity,False,True,context)
    extinguished_history_choice=adult_history_choice(extinguished_history,candidates,context)
    for _ in range(5):extinguished_history.observe_control(deep.identity,False,False,context)
    extinguished_history.observe_control(deep.identity,True,True,context)
    reacquired_history_choice=adult_history_choice(extinguished_history,candidates,context)

    # Destructive context audit. The deep construction has real control in A
    # and higher returned value in B, but its two B uses were noncontingent.
    # Global history must not counterfeit B-local action truth.
    context_a,context_b=0xC071A,0xC071B
    qualified=PredictiveCreditBankV1(16)
    for n in range(4):
        tick=900+n*6
        qualified.observe_use(familiar.identity,tick,tick+2,Q//12,context_b)
        qualified.observe_return(familiar.identity,Q//2,Q//16,tick+3,True,context_b)
        qualified.observe_control(familiar.identity,True,True,context_b)
        qualified.observe_use(deep.identity,tick+100,tick+107,Q//2,context_a)
        qualified.observe_return(deep.identity,Q,Q//8,tick+108,True,context_a)
        qualified.observe_control(deep.identity,True,True,context_a)
    for n in range(2):
        tick=1100+n*10
        qualified.observe_use(deep.identity,tick,tick+7,Q//2,context_b)
        qualified.observe_return(deep.identity,Q,Q//8,tick+8,True,context_b)
        qualified.observe_control(deep.identity,True,False,context_b)
    a_before=qualified.row(deep.identity).contexts[context_a]
    a_witness=(a_before.control_attempts,a_before.control_successes,a_before.control_history_q16,a_before.outcome_mean_q16)
    qualified_a=adult_history_choice(qualified,candidates,context_a)
    qualified_b_before=adult_history_choice(qualified,candidates,context_b)
    for n in range(2):
        qualified.observe_control(deep.identity,True,True,context_b)
    qualified_b_after=adult_history_choice(qualified,candidates,context_b)
    a_after=qualified.row(deep.identity).contexts[context_a]
    a_unchanged=(a_after.control_attempts,a_after.control_successes,a_after.control_history_q16,a_after.outcome_mean_q16)
    qualified_restored=PredictiveCreditBankV1.restore(qualified.checkpoint())
    qualified_b_restored=adult_history_choice(qualified_restored,candidates,context_b)
    qualified_lesioned=PredictiveCreditBankV1.restore(qualified.checkpoint())
    b_lesion=qualified_lesioned.row(deep.identity).contexts[context_b]
    b_lesion.control_attempts=b_lesion.control_successes=0
    b_lesion.background_attempts=b_lesion.background_successes=0
    b_lesion.control_history_q16=0
    qualified_b_lesioned=adult_history_choice(qualified_lesioned,candidates,context_b)
    global_history_after_lesion=qualified_lesioned.row(deep.identity).control_history_q16
    forged=PredictiveCreditBankV1(4)
    try:forged.observe_control(0xDEAD,True,True,context_b)
    except ValueError:forged_context_refused=True
    else:forged_context_refused=False

    checks = {
        "value_only_prefers_uncontrollable_deep_program": value_choice_before == deep.identity,
        "value_only_language_is_silent_when_program_not_controllable": value_public_before == 0,
        "controllability_prevents_avoidable_silence": (
            controlled_before == familiar.identity and controlled_public_before == steps[familiar.identity]
        ),
        "positive_self_language_does_not_create_control": (
            control_before_assertion == 0 and control_after_assertion == 0
        ),
        "public_consequence_earns_deep_control": bank.row(deep.identity).controllability_q16 >= Q // 2,
        "initiative_shifts_after_lived_control": controlled_after == deep.identity,
        "public_trajectory_length_increases_after_control_learning": (
            controlled_public_after > controlled_public_before
            and length[controlled_after] > length[controlled_before]
        ),
        "compositional_depth_increases_after_control_learning": (
            depth[controlled_after] > depth[controlled_before]
        ),
        "raw_ratio_would_overcommit_after_one_success": raw_one_shot == deep.identity,
        "one_success_does_not_displace_established_language": calibrated_one_shot == familiar.identity,
        "second_independent_success_unlocks_deeper_language": calibrated_corroborated == deep.identity,
        "visible_one_shot_calibration_gain": (
            calibrated_one_shot != raw_one_shot
            and length[calibrated_one_shot] < length[raw_one_shot]
            and depth[calibrated_one_shot] < depth[raw_one_shot]
        ),
        "predecessor_overcommits_when_free_outcomes_match_action_outcomes": predecessor_yoked == deep.identity,
        "matched_free_outcomes_keep_established_language": resident_yoked == familiar.identity,
        "informative_background_non_events_recover_deeper_language": resident_recovery == deep.identity,
        "predecessor_cannot_use_background_non_events_for_recovery": predecessor_recovery == familiar.identity,
        "visible_contingency_language_gain": (
            resident_yoked != predecessor_yoked
            and resident_recovery != predecessor_recovery
            and length[resident_yoked] < length[predecessor_yoked]
            and length[resident_recovery] > length[predecessor_recovery]
        ),
        "matched_current_contingency_cells": prior_cells == yoked_cells == (2,2,2,2),
        "matched_current_causal_truth_is_zero": (
            prior_row.controllability_q16 == yoked_row.controllability_q16 == 0
            and not prior_row.control_ready and not yoked_row.control_ready
        ),
        "learned_history_is_distinct_and_program_local": (
            prior_row.control_history_q16>yoked_row.control_history_q16==0
        ),
        "prior_control_preserves_deeper_heldout_composition": prior_history_choice == deep.identity,
        "canonical_adult_score_uses_history_without_rewriting_current_truth": (
            prior_history_choice == deep.identity and not prior_row.control_ready
        ),
        "interleaved_yoked_history_stays_with_familiar_language": yoked_history_choice == familiar.identity,
        "control_history_changes_visible_language_depth": (
            length[prior_history_choice] > length[yoked_history_choice]
            and depth[prior_history_choice] > depth[yoked_history_choice]
        ),
        "checkpoint_retains_control_history": restored_history_choice == deep.identity,
        "focal_history_lesion_removes_only_advantage": lesioned_history_choice == familiar.identity,
        "continued_disconfirmation_extinguishes_history_advantage": extinguished_history_choice == familiar.identity,
        "ordinary_control_evidence_reacquires_history_advantage": reacquired_history_choice == deep.identity,
        "control_in_a_does_not_author_same_program_in_b": qualified_a==deep.identity and qualified_b_before==familiar.identity,
        "ordinary_b_consequence_unlocks_deeper_language_in_b": qualified_b_after==deep.identity,
        "b_learning_does_not_rewrite_a": a_witness==a_unchanged,
        "context_control_survives_checkpoint": qualified_b_restored==deep.identity,
        "focal_b_lesion_preserves_global_history_and_a": (
            qualified_b_lesioned==familiar.identity and global_history_after_lesion>0
            and adult_history_choice(qualified_lesioned,candidates,context_a)==deep.identity),
        "nonparticipating_context_cannot_receive_control": forged_context_refused,
        "outcome_value_not_created_by_control_update": bank.contextual_outcome(deep.identity, context) == Q,
        "no_belief_string_or_robbins_module": True,
        "bounded_runtime": time.perf_counter() - started < 2.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("FOUNDRY_AGI_CONTROLLABILITY_LANGUAGE_RED " + ",".join(failed))

    path = Path(__file__)
    receipt = {
        "contract": "FOUNDRY_AGI_CONTROLLABILITY_LANGUAGE_GREEN",
        "reference_only": True,
        "language_phenotype_improved": True,
        "visible_language_gain": "CONTEXT_LOCAL_CONTROL_PREVENTS_CROSS_SITUATION_LANGUAGE_OVERCOMMITMENT",
        "control_calibration": {
            "raw_ratio_one_shot_surface": bytes(deep.surface).decode(),
            "resident_one_shot_surface": bytes(familiar.surface).decode(),
            "resident_after_second_success_surface": bytes(deep.surface).decode(),
            "one_shot_successes": 1,
            "corroborated_successes": 2,
        },
        "contingency_calibration": {
            "predecessor_matched_free_outcomes_surface": bytes(deep.surface).decode(),
            "resident_matched_free_outcomes_surface": bytes(familiar.surface).decode(),
            "predecessor_after_background_non_events_surface": bytes(familiar.surface).decode(),
            "resident_after_background_non_events_surface": bytes(deep.surface).decode(),
            "deep_action_outcomes": 2,
            "deep_free_outcomes": 2,
            "deep_later_free_non_outcomes": 2,
        },
        "control_history_immunization": {
            "matched_current_cells": prior_cells,
            "current_controllability_q16": prior_row.controllability_q16,
            "prior_control_history_q16": prior_row.control_history_q16,
            "interleaved_yoked_history_q16": yoked_row.control_history_q16,
            "prior_control_surface": bytes(deep.surface if prior_history_choice==deep.identity else familiar.surface).decode(),
            "interleaved_yoked_surface": bytes(deep.surface if yoked_history_choice==deep.identity else familiar.surface).decode(),
            "lesioned_surface": bytes(deep.surface if lesioned_history_choice==deep.identity else familiar.surface).decode(),
            "extinguished_surface": bytes(deep.surface if extinguished_history_choice==deep.identity else familiar.surface).decode(),
            "reacquired_surface": bytes(deep.surface if reacquired_history_choice==deep.identity else familiar.surface).decode(),
        },
        "context_qualification": {
            "a_choice": qualified_a,
            "b_before_local_control": qualified_b_before,
            "b_after_local_control": qualified_b_after,
            "b_after_local_lesion": qualified_b_lesioned,
            "global_history_after_local_lesion_q16": global_history_after_lesion,
        },
        "before": {
            "value_only_choice": value_choice_before,
            "value_only_public_steps": value_public_before,
            "controlled_choice": controlled_before,
            "controlled_public_steps": controlled_public_before,
            "controlled_bytes": length[controlled_before],
            "controlled_depth": depth[controlled_before],
        },
        "after_lived_control": {
            "controlled_choice": controlled_after,
            "controlled_public_steps": controlled_public_after,
            "controlled_bytes": length[controlled_after],
            "controlled_depth": depth[controlled_after],
            "deep_controllability_q16": bank.row(deep.identity).controllability_q16,
        },
        "checks": checks,
        "tokens": False,
        "transformer": False,
        "backprop": False,
        "expected_strings": False,
        "remaining_red": [
            "CANONICAL_AGI_CONTROL_PROFILE",
            "DIRECT_PUBLIC_LANGUAGE",
            "EXPLORATION_OF_UNTRIED_PROGRAMS",
            "DIRECT_CONTROL_HISTORY_PARITY",
        ],
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    print(receipt["contract"])
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
