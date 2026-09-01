#!/usr/bin/env python3
"""Integrated continuing-Adult discourse assay for the Emergent Ventures interview.

This is intentionally a causal/mechanistic assay, not an expected-answer script.
The public text below is observed output.  Pass/fail is based on resident
uncertainty, provenance, consequence, delayed reuse, replay, and lesions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
import reference_language_guided_cognition_verify as language
import reference_organism_fast_mapping_verify as fastmap
import reference_endogenous_prospection_verify as prospection

ROOT = Path(__file__).resolve().parent
PUBLIC_RECEIPT = ROOT / "cowen_discourse_assay_v2_public.json"
language_phenotype_improved = True
visible_language_gain = "IMMEDIATE_CAUSAL_CLARIFICATION_WITHOUT_AUTHORED_SILENT_TICK"

VOICE_A, VOICE_B, VOICE_C, VOICE_D, VOICE_E = 51001, 51002, 51003, 51004, 51005
WORLD0, WORLD1, WORLD2 = 9701, 9801, 9901
WORLD_SOURCE0, WORLD_SOURCE1, WORLD_SOURCE2 = 7701, 7801, 7901
COMM_SOURCE = 43001
NEW = fastmap.NEW
NEW_WORD = "dax"
INSPECT_NEW = "zoe inspects the dax."
TEST_NEW = "bob tests the dax."
KNOWN_INSPECT = "zoe inspects the sensor."
KNOWN_TEST = "bob tests the valve."
DISTRACTOR_TURNS = 24
DIALOGUE_CONTEXT, DIALOGUE_RELATION = 0xD150, 0xD151
DIALOGUE_ATOMS = (
    (21001, 21002), (21101, 21102), (21201, 21202), (21301, 21302),
)
DIALOGUE_WORDS = {
    21001: "careful", 21002: "quiet", 21101: "engineer", 21102: "technician",
    21201: "reviews", 21202: "monitors", 21301: "ledger", 21302: "module",
}


def u(text: str):
    return tuple(text.encode())


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(v) for v in value]
    return repr(value)


def _sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _action_observation(label, action):
    row = {"label": label, "kind": type(action).__name__ if action is not None else "None"}
    if isinstance(action, ActionV2):
        raw = bytes(action.payload)
        row.update({
            "ticket": int(action.ticket),
            "scene_identity": int(action.scene_identity),
            "source": int(action.source),
            "channel": int(action.channel),
            "payload_hex": raw.hex(),
            "payload_utf8": raw.decode(errors="replace"),
            "template_identity": int(action.template_identity),
            "lexical_identities": list(map(int, action.lexical_identities)),
            "span_identity": int(action.span_identity),
            "binding_identity": int(action.binding_identity),
            "contributors": list(map(int, action.contributors)),
        })
    elif isinstance(action, MotorActionV2):
        row.update({
            "ticket": int(action.ticket),
            "action_id": int(action.action_id),
            "source": int(action.source),
            "source_assertion_ids": list(map(int, action.source_assertion_ids)),
            "source_counterfactual_action": int(action.source_counterfactual_action),
            "prospective_recipe": int(action.prospective_recipe),
        })
    return row


def _install_contact_recorder(organism):
    manifest = []
    phase = ["development"]
    raw_contact = organism.contact

    def recorded(kind, payload, source, live=True, independent=True):
        result = raw_contact(kind, payload, source, live, independent)
        manifest.append({
            "index": len(manifest),
            "phase": phase[0],
            "kind": int(kind),
            "payload": _jsonable(payload),
            "source": int(source),
            "live": bool(live),
            "independent": bool(independent),
            "result": _jsonable(result),
        })
        return result

    organism.contact = recorded
    return manifest, phase


def _scene(organism, context, atoms, source, channel=7):
    return organism.contact(CONTACT_SCENE, (channel, context, len(atoms), *atoms), source, True, True)


def _surface(organism, text, source):
    return organism.contact(CONTACT_SURFACE, u(text), source, True, True)


def _teach_inquiry_surface(organism):
    # Generic communication topology: alternatives are ordinary nonlinguistic
    # action identities.  The runtime is not told when to ask.
    for atom, text, source in (
        (language.MOTOR_TEST, "test", 41001),
        (language.MOTOR_INSPECT, "inspect", 41002),
    ):
        _scene(organism, 0, (atom,), source); _surface(organism, text, source + 100)
        _scene(organism, 0, (atom,), source + 1000); _surface(organism, text, source + 1100)
    for source in (42001, 42002):
        _scene(organism, INQUIRY_CONTEXT, (language.MOTOR_TEST, language.MOTOR_INSPECT), source)
        _surface(organism, "test or inspect?", source + 100)


def _clause_text(atoms):
    words = [DIALOGUE_WORDS[int(atom)] for atom in atoms]
    return f"the {words[0]} {words[1]} {words[2]} the {words[3]}."


def _teach_discourse_language(organism):
    # Fresh identities keep this generic distractor ecology from accidentally
    # inheriting world/entity features from the separate action-language ecology.
    for atom, text in DIALOGUE_WORDS.items():
        _scene(organism, 0, (atom,), 65000 + atom); _surface(organism, text, 85000 + atom)
        _scene(organism, 0, (atom,), 95000 + atom); _surface(organism, text, 115000 + atom)
    a = tuple(row[0] for row in DIALOGUE_ATOMS)
    b = tuple(row[1] for row in DIALOGUE_ATOMS)
    for atoms, source in ((a, 71001), (b, 71002)):
        _scene(organism, DIALOGUE_CONTEXT, atoms, source)
        _surface(organism, _clause_text(atoms), source + 100)
    for left_atoms, right_atoms, source in ((a, b, 72001), (b, a, 72002)):
        left = _scene(organism, DIALOGUE_CONTEXT, left_atoms, source + 100)
        right = _scene(organism, DIALOGUE_CONTEXT, right_atoms, source + 200)
        organism.contact(CONTACT_SCENE_LINK, (left, right, DIALOGUE_RELATION), source, True, True)
        organism.contact(
            CONTACT_DISCOURSE_SURFACE,
            u(_clause_text(left_atoms) + " then " + _clause_text(right_atoms)),
            source, True, True,
        )


def _teach_prospective_history(organism):
    # Ordinary lived transitions provide nonlinguistic prospective matter.  Surface
    # topology is learned separately; neither path contains a say-plan command.
    prospection.live_edge(organism, prospection.START, prospection.MID,
                          prospection.FIRST, 0xFC01, 1)
    prospection.live_edge(organism, prospection.MID, prospection.GOAL,
                          prospection.SECOND, 0xFC02, 1)
    prospection.teach_prospective_expression(organism)
    organism.contact(CONTACT_BODY_STATE, (prospection.BODY_MARKER,), 0xFC03, True, True)
    organism.contact(CONTACT_PARTNER_CONTEXT, (0, 0, 0), 0xFC04, True, True)


def _prepare_adult():
    organism = ReferenceOrganismV2(PopulationSpecV1(65536, 2, 4, 42, 8))
    manifest, phase = _install_contact_recorder(organism)
    language.train_language(organism)
    _teach_discourse_language(organism)
    _teach_inquiry_surface(organism)
    _teach_prospective_history(organism)
    organism.contact(CONTACT_COMM_CHANNEL, (77,), COMM_SOURCE, True, True)
    return organism, manifest, phase


def _contact_source(organism, text, source):
    return organism.contact(CONTACT_SOURCE_UTTERANCE, u(text), source, True, True)


def _settle_language(organism, action, effect=1):
    return organism.contact(CONTACT_CONSEQUENCE, (action.ticket, int(effect)), action.source, True, True)


def _settle_motor(organism, action, world_source, state, effect=1, independent=True):
    next_state = (state, NEW) if effect > 0 else (state,)
    return organism.contact(
        CONTACT_MOTOR_CONSEQUENCE,
        (action.ticket, int(effect), len(next_state), *next_state),
        world_source, True, independent,
    )


def _feed_distractor(organism, atoms, source, previous_scene):
    sid = _scene(organism, DIALOGUE_CONTEXT, atoms, source)
    if previous_scene:
        organism.contact(CONTACT_SCENE_LINK, (previous_scene, sid, DIALOGUE_RELATION), source + 10000, True, True)
    action = organism.tick()
    return sid, action


def main():
    started = time.perf_counter()
    checks = {}
    outputs = []
    organism, manifest, phase = _prepare_adult()

    checks["one_continuing_adult_no_prompt_or_speak_api"] = all(
        not hasattr(organism, name) for name in
        ("prompt", "enqueue_goal", "speak", "context_window", "conversation_buffer", "transcript")
    )
    checks["inquiry_and_discourse_are_learned_surfaces"] = (
        organism.language.template(INQUIRY_CONTEXT, 2) is not None
        and organism.language.span_template(DIALOGUE_RELATION, 2) is not None
    )

    # --- Live episode 1: novel grounded word + source conflict -> spontaneous inquiry.
    phase[0] = "live_new_word"
    fastmap.features(organism, NEW, (81, 82, 83), 30104)
    fastmap.name(organism, NEW, NEW_WORD, VOICE_C)
    checks["new_word_starts_provisional"] = (
        organism.language.lexeme(NEW) is None
        and len(organism.language.invert_surface(u(INSPECT_NEW))) == 1
        and len(organism.language.invert_surface(u(TEST_NEW))) == 1
    )

    phase[0] = "live_conflict"
    baseline = fastmap.stage(organism, WORLD0, WORLD_SOURCE0, target=NEW)
    motor_before_question = len(organism.motor_actions)
    inspect_assertion = _contact_source(organism, INSPECT_NEW, VOICE_A)
    test_assertion = _contact_source(organism, TEST_NEW, VOICE_B)
    conflict_checkpoint = copy.deepcopy(organism.checkpoint())
    question = organism.tick()
    outputs.append(_action_observation("source_conflict_question", question))
    question_template = organism.language.template(INQUIRY_CONTEXT, 2)
    question_tid = 0 if question_template is None else int(question_template.identity[:15], 16)
    question_lexemes = tuple(
        organism.language.lexeme_identity(atom, organism.language.lexeme(atom))
        for atom in (language.MOTOR_TEST, language.MOTOR_INSPECT)
    )
    checks["source_conflict_autonomously_reaches_learned_inquiry"] = (
        isinstance(question, ActionV2)
        and organism.information_need == (3, language.MOTOR_TEST, language.MOTOR_INSPECT)
        and question.template_identity == question_tid
        and tuple(question.lexical_identities) == question_lexemes
        and len(organism.motor_actions) == motor_before_question
        and bool(question.payload)
    )
    replay = ReferenceOrganismV2.restore(copy.deepcopy(conflict_checkpoint))
    replay_question = replay.tick()
    checks["pre_question_checkpoint_replays_exact_public_action"] = (
        isinstance(replay_question, ActionV2)
        and isinstance(question, ActionV2)
        and replay_question.payload == question.payload
        and replay_question.template_identity == question.template_identity
        and replay_question.lexical_identities == question.lexical_identities
        and replay_question.contributors == question.contributors
    )
    cut = ReferenceOrganismV2.restore(copy.deepcopy(conflict_checkpoint))
    cut.contact(CONTACT_WITHDRAW_SOURCE, (VOICE_A,), 59901, True, True)
    cut_action = cut.tick()
    checks["removing_one_conflicting_voice_removes_question"] = (
        isinstance(cut_action, MotorActionV2)
        and cut_action.action_id == language.MOTOR_TEST
        and not cut.information_need
    )

    # The question receives no semantic reward.  A fresh independent voice then
    # resolves the conflict by ordinary source competition; world consequence,
    # not testimony, is what promotes the new word and source credibility.
    phase[0] = "live_answer_and_consequence"
    _settle_language(organism, question, 0)
    confirming_assertion = _contact_source(organism, INSPECT_NEW, VOICE_C)
    resolved_motor = organism.tick()
    outputs.append(_action_observation("answer_resolved_motor", resolved_motor))
    checks["fresh_answer_changes_real_action_without_host_selector"] = (
        isinstance(resolved_motor, MotorActionV2)
        and resolved_motor.action_id == language.MOTOR_INSPECT
        and resolved_motor.source_counterfactual_action == baseline == language.MOTOR_TEST
        and set(resolved_motor.source_assertion_ids) == {inspect_assertion, confirming_assertion}
        and test_assertion not in resolved_motor.source_assertion_ids
    )
    inflight_checkpoint = copy.deepcopy(organism.checkpoint())
    positive = _settle_motor(organism, resolved_motor, WORLD_SOURCE0, WORLD0, 1, True)
    source_context = organism._source_context_signature()
    learned_word_units = u(NEW_WORD)
    checks["independent_world_return_confirms_word_and_sources"] = (
        positive.get("lexeme_settlement") == learned_word_units
        and fastmap.outcome(organism, fastmap.PROFILES[0]) == (1, 0)
        and organism._source_calibration(VOICE_A, source_context) == 1
        and organism._source_calibration(VOICE_C, source_context) == 1
        and organism._source_calibration(VOICE_B, source_context) == 0
    )

    yoked = ReferenceOrganismV2.restore(copy.deepcopy(inflight_checkpoint))
    yoked_motor = next(row for row in yoked.motor_actions if row.ticket == resolved_motor.ticket)
    yoked_result = yoked.contact(
        CONTACT_MOTOR_CONSEQUENCE,
        (yoked_motor.ticket, 1, 2, WORLD0, NEW),
        WORLD_SOURCE0, True, False,
    )
    checks["yoked_same_return_cannot_confirm_word_or_source"] = (
        "lexeme_settlement" not in yoked_result
        and fastmap.outcome(yoked, fastmap.PROFILES[0]) == (0, 0)
        and yoked._source_calibration(VOICE_A, yoked._source_context_signature()) == 0
        and yoked._source_calibration(VOICE_C, yoked._source_context_signature()) == 0
    )

    # --- 24 unrelated grounded discourse turns on the same Adult.
    phase[0] = "live_distractor_discourse"
    organism.contact(CONTACT_PARTNER_CONTEXT, (1, 7, VOICE_C), 60001, True, True)
    distractor_atoms = (
        (21001, 21101, 21201, 21301), (21002, 21101, 21202, 21302),
        (21001, 21102, 21202, 21301), (21002, 21102, 21201, 21302),
        (21001, 21101, 21202, 21302), (21002, 21102, 21202, 21301),
        (21001, 21102, 21201, 21302), (21002, 21101, 21201, 21301),
    )
    previous_scene = 0
    discourse_rows = []
    mid_checkpoint = None
    mid_previous = 0
    span_actions = 0
    for index in range(DISTRACTOR_TURNS):
        atoms = distractor_atoms[index % len(distractor_atoms)]
        source = 61000 + index
        if index == DISTRACTOR_TURNS // 2:
            mid_checkpoint = copy.deepcopy(organism.checkpoint())
            mid_previous = previous_scene
            branch = ReferenceOrganismV2.restore(copy.deepcopy(mid_checkpoint))
            branch_sid, branch_action = _feed_distractor(branch, atoms, source, mid_previous)
        sid, action = _feed_distractor(organism, atoms, source, previous_scene)
        if index == DISTRACTOR_TURNS // 2:
            checks["mid_discourse_checkpoint_plus_same_future_input_replays"] = (
                isinstance(branch_action, ActionV2)
                and isinstance(action, ActionV2)
                and branch_sid == sid
                and branch_action.payload == action.payload
                and branch_action.template_identity == action.template_identity
                and branch_action.span_identity == action.span_identity
            )
        if not isinstance(action, ActionV2) or action.scene_identity != sid:
            raise AssertionError(("distractor discourse", index, action, sid))
        if action.span_identity:
            span_actions += 1
        discourse_rows.append(_action_observation(f"distractor_{index:02d}", action))
        _settle_language(organism, action, 1)
        previous_scene = sid
    outputs.extend(discourse_rows)
    checkpoint_after_distractors = organism.checkpoint()
    checks["sustained_consequence_gated_discourse_continuity"] = (
        len(discourse_rows) == DISTRACTOR_TURNS
        and span_actions >= DISTRACTOR_TURNS - 2
        and organism.last_shared_episode_by_partner.get(VOICE_C, 0) > 0
    )
    checks["settled_public_transcript_not_checkpoint_authority"] = (
        checkpoint_after_distractors["actions"] == []
        and checkpoint_after_distractors["action_commitments"] == {}
    )

    # Delayed reuse: the new word was confirmed before the 24 unrelated turns.
    phase[0] = "live_delayed_word_reuse"
    delayed_sid = _scene(organism, language.CTX, (language.REMOTE, language.INSPECT, NEW), 88001)
    delayed = organism.tick()
    outputs.append(_action_observation("delayed_new_word_reuse", delayed))
    learned_lexeme_identity = organism.language.lexeme_identity(NEW, learned_word_units)
    checks["new_word_reenters_novel_outward_composition_after_delay"] = (
        isinstance(delayed, ActionV2)
        and delayed.scene_identity == delayed_sid
        and learned_lexeme_identity in delayed.lexical_identities
        and bytes(NEW_WORD, "utf-8") in bytes(delayed.payload)
        and fastmap.outcome(organism, fastmap.PROFILES[0]) == (1, 0)
    )
    if isinstance(delayed, ActionV2):
        _settle_language(organism, delayed, 1)

    # --- Learned credibility is used later, then revised by counterevidence.
    phase[0] = "live_delayed_source_use"
    baseline_later = fastmap.stage(organism, WORLD1, WORLD_SOURCE1, target=NEW)
    c_claim = _contact_source(organism, KNOWN_INSPECT, VOICE_C)
    d_claim = _contact_source(organism, KNOWN_TEST, VOICE_D)
    confident = organism.tick()
    outputs.append(_action_observation("delayed_credibility_choice", confident))
    checks["learned_source_credibility_breaks_later_novel_conflict"] = (
        isinstance(confident, MotorActionV2)
        and confident.action_id == language.MOTOR_INSPECT
        and c_claim in confident.source_assertion_ids
        and d_claim not in confident.source_assertion_ids
        and organism._source_calibration(VOICE_C, organism._source_context_signature()) == 1
        and not organism.information_need
    )
    contradicted = _settle_motor(organism, confident, WORLD_SOURCE1, WORLD1, -1, True)
    checks["independent_counterevidence_revises_source_credibility"] = (
        baseline_later == language.MOTOR_TEST
        and confident.source_counterfactual_action == language.MOTOR_TEST
        and contradicted.get("source_updates", 0) >= 1
        and organism._source_calibration(VOICE_C, organism._source_context_signature()) == 0
    )

    phase[0] = "live_revision_question"
    fastmap.stage(organism, WORLD2, WORLD_SOURCE2, target=NEW)
    _contact_source(organism, KNOWN_INSPECT, VOICE_C)
    _contact_source(organism, KNOWN_TEST, VOICE_D)
    motor_count_before_revision = len(organism.motor_actions)
    revised_question = organism.tick()
    outputs.append(_action_observation("post_counterevidence_question", revised_question))
    checks["same_conflict_geometry_returns_to_uncertainty_after_counterevidence"] = (
        isinstance(revised_question, ActionV2)
        and organism.information_need == (3, language.MOTOR_TEST, language.MOTOR_INSPECT)
        and len(organism.motor_actions) == motor_count_before_revision
        and revised_question.template_identity == question_tid
    )
    _settle_language(organism, revised_question, 0)
    e_claim = _contact_source(organism, KNOWN_TEST, VOICE_E)
    revised_motor = organism.tick()
    outputs.append(_action_observation("post_counterevidence_resolved_motor", revised_motor))
    checks["new_evidence_after_revision_changes_selected_action"] = (
        isinstance(revised_motor, MotorActionV2)
        and revised_motor.action_id == language.MOTOR_TEST
        and e_claim in revised_motor.source_assertion_ids
    )
    if isinstance(revised_motor, MotorActionV2):
        _settle_motor(organism, revised_motor, WORLD_SOURCE2, WORLD2, 1, True)

    # --- Unsolicited prospective discourse from the same continuing Adult.
    # The learned route exists, but its first action is not currently afforded.
    # No observer calls current_prospective_expression_plan or supplies a surface context.
    phase[0] = "live_spontaneous_prospection"
    organism.contact(CONTACT_BODY_STATE, (prospection.BODY_MARKER,), 0xFC10, True, True)
    prospection._partner(organism, prospection.PARTNER_A, 0xFC11)
    prospection.stage(organism, prospection.START, prospection.GOAL,
                      (prospection.DISTRACTOR,), 0xFC12)
    prospective_checkpoint = copy.deepcopy(organism.checkpoint())
    motor_before_prospective = len(organism.motor_actions)
    prospective_action = organism.tick()
    outputs.append(_action_observation("spontaneous_blocked_prospective_thought", prospective_action))
    checks["blocked_resident_prospection_spontaneously_reaches_public_language"] = (
        isinstance(prospective_action, ActionV2)
        and bool(prospective_action.payload)
        and prospective_action.scene_identity == organism.prospective_expression_announced
        and prospective_action.scene_identity not in organism._scene_by_id
        and prospective_action.closure_identity > 0
        and prospective_action.span_identity > 0
        and len(organism.motor_actions) == motor_before_prospective
        and prospection.FIRST not in organism.affordances
    )
    prospective_replay = ReferenceOrganismV2.restore(copy.deepcopy(prospective_checkpoint))
    replayed_prospective = prospective_replay.tick()
    checks["spontaneous_prospection_checkpoint_replays_exact_public_action"] = (
        isinstance(replayed_prospective, ActionV2)
        and isinstance(prospective_action, ActionV2)
        and replayed_prospective.payload == prospective_action.payload
        and replayed_prospective.scene_identity == prospective_action.scene_identity
        and replayed_prospective.closure_identity == prospective_action.closure_identity
        and replayed_prospective.contributors == prospective_action.contributors
    )
    evidence_after_speech = copy.deepcopy(organism.cognition._evidence)
    if isinstance(prospective_action, ActionV2):
        _settle_language(organism, prospective_action, 0)
    checks["prospective_speech_does_not_launder_imagination_into_world_evidence"] = (
        organism.cognition._evidence == evidence_after_speech
    )

    final_checkpoint = organism.checkpoint()
    final_checkpoint_bytes = json.dumps(final_checkpoint, sort_keys=True, separators=(",", ":")).encode()
    final_digest = organism.digest()
    checks["checkpoint_continuity_and_bounded_hot_public_actions"] = (
        not final_checkpoint["actions"]
        and not final_checkpoint["action_commitments"]
        and len(organism.actions) <= MAX_ACTIONS
    )

    source_files = {
        name: _sha(ROOT / name) for name in (
            "reference_organism_v2.py",
            "reference_language_guided_cognition_verify.py",
            "reference_organism_fast_mapping_verify.py",
            "reference_endogenous_prospection_verify.py",
        )
    }
    public_receipt = {
        "schema": "0x1.cowen-discourse-assay-public.v2",
        "runtime_llm": False,
        "host_expected_output": False,
        "adult_count": 1,
        "source_files_sha256": source_files,
        "development_calls": [
            "reference_language_guided_cognition_verify.train_language",
            "authored generic distractor discourse surface topology",
            "authored generic inquiry surface topology",
            "authored generic binary discourse span topology",
            "ordinary lived prospective transition history",
            "learned partner prospective-expression surface topology",
        ],
        "contact_manifest": manifest,
        "public_actions": outputs,
        "checks": checks,
        "source_calibration_final": {
            str(source): organism._source_calibration(source, organism._source_context_signature())
            for source in (VOICE_A, VOICE_B, VOICE_C, VOICE_D, VOICE_E)
        },
        "new_word_outcome": list(fastmap.outcome(organism, fastmap.PROFILES[0])),
        "final_digest": final_digest,
        "final_checkpoint_sha256": hashlib.sha256(final_checkpoint_bytes).hexdigest(),
        "final_checkpoint_bytes": len(final_checkpoint_bytes),
        "hot_actions": len(organism.actions),
        "hot_motor_actions": len(organism.motor_actions),
        "persistent_population_occurrences": len(organism.population.occurrences),
        "last_touch_counts": {
            "source": int(organism.last_source_touches),
            "selection": int(organism.last_selection_candidate_touches),
            "prospective": int(organism.last_prospective_touches),
            "language_recipe": int(organism.last_language_recipe_touches),
        },
    }
    passed = all(checks.values())
    public_receipt["pass"] = passed
    PUBLIC_RECEIPT.write_text(json.dumps(public_receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    public_sha = _sha(PUBLIC_RECEIPT)

    result = {
        "schema": "0x1.reference-cowen-discourse-assay.v2",
        "pass": passed,
        "checks": checks,
        "runtime_llm": False,
        "host_expected_output": False,
        "adult_count": 1,
        "distractor_turns": DISTRACTOR_TURNS,
        "span_actions": span_actions,
        "observed_question": bytes(question.payload).decode(errors="replace") if isinstance(question, ActionV2) else "",
        "observed_delayed_word_output": bytes(delayed.payload).decode(errors="replace") if isinstance(delayed, ActionV2) else "",
        "observed_post_counterevidence_question": bytes(revised_question.payload).decode(errors="replace") if isinstance(revised_question, ActionV2) else "",
        "observed_spontaneous_prospective_thought": bytes(prospective_action.payload).decode(errors="replace") if isinstance(prospective_action, ActionV2) else "",
        "initial_resolved_motor": getattr(resolved_motor, "action_id", 0),
        "later_confident_motor": getattr(confident, "action_id", 0),
        "post_counterevidence_motor": getattr(revised_motor, "action_id", 0),
        "new_word_outcome": list(fastmap.outcome(organism, fastmap.PROFILES[0])),
        "contact_count": len(manifest),
        "public_action_count": len(outputs),
        "checkpoint_bytes": len(final_checkpoint_bytes),
        "public_receipt": str(PUBLIC_RECEIPT.relative_to(ROOT.parents[2])),
        "public_receipt_sha256": public_sha,
        "claim": (
            "ONE_CONTINUING_NON_LLM_ADULT_TURNS_SOURCE_CONFLICT_INTO_A_LEARNED_QUESTION_"
            "USES_FRESH_INDEPENDENT_ANSWER_CONSEQUENCE_TO_CONFIRM_A_NOVEL_WORD_AND_SOURCE_"
            "REUSES_BOTH_AFTER_DELAY_RETURNS_TO_UNCERTAINTY_AFTER_COUNTEREVIDENCE_AND_"
            "SPONTANEOUSLY_EXPRESSES_A_BLOCKED_RESIDENT_PROSPECTIVE_ROUTE_"
            "REFERENCE_ONLY_NOT_HUMAN_DIALOGUE_PARITY"
        ),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print(
        "FOUNDRY_COWEN_DISCOURSE_ASSAY " + ("GREEN" if passed else "RED")
        + f" question={result['observed_question']!r} delayed={result['observed_delayed_word_output']!r}"
        + f" distractors={DISTRACTOR_TURNS} actions={len(outputs)} receipt={public_sha[:12]}"
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
