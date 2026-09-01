#!/usr/bin/env python3
"""Raw-language source competition without testimony-to-truth laundering."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import reference_language_guided_cognition_verify as language
from reference_organism_v2 import (
    CONTACT_AFFORDANCES,
    CONTACT_BODY_TARGET,
    CONTACT_MOTOR_CONSEQUENCE,
    CONTACT_SOURCE_UTTERANCE,
    CONTACT_WITHDRAW_SOURCE,
    CONTACT_WORLD_STATE,
    MotorActionV2,
    ReferenceOrganismV2,
)
from reference_population_v1 import PopulationSpecV1


A, B, C, D, E = 8501, 8502, 8503, 8504, 8505
TEST_PREDICTION = "bob tests the target."
GERMAN_TEST_PREDICTION = "bob testet ziel."
MIXED_TEST_PREDICTION = "bob testet target."
MANDARIN_TEST_PREDICTION = "鲍勃测试目标。"
RUSSIAN_TEST_PREDICTION = "Боб тестирует цель."
CHIN_TEST_PREDICTION = "bob nih target kha a chek."


def stage(organism, state, source, target=(language.GOAL,), actions=None):
    actions = actions or (language.MOTOR_TEST, language.MOTOR_INSPECT)
    organism.contact(CONTACT_WORLD_STATE, tuple(state), source, True, True)
    organism.contact(CONTACT_BODY_TARGET, tuple(target), source + 10000, True, True)
    organism.contact(CONTACT_AFFORDANCES, tuple(actions), source + 20000, True, True)


def say(organism, text, source):
    return organism.contact(CONTACT_SOURCE_UTTERANCE, language.u(text), source, True, True)


def settle(organism, action, source, next_state, independent=True):
    payload = (action.ticket, 1, len(next_state), *next_state)
    return organism.contact(CONTACT_MOTOR_CONSEQUENCE, payload, source, True, independent)


def prepared(spec):
    organism = ReferenceOrganismV2(spec)
    language.train_language(organism)
    return organism


def prepared_multilingual(spec):
    """Add four learned surface ecologies to the same atoms and Recipes."""
    organism = prepared(spec)
    profiles = (
        (100000,
         ((language.REMOTE, "zoe"), (language.BOB, "bob"),
          (language.INSPECT, "prueft"), (language.TEST, "testet"),
          (language.GOAL, "ziel"), (language.SENSOR, "sensor"),
          (language.VALVE, "ventil")),
         (((language.BOB, language.TEST, language.VALVE), "bob testet ventil."),
          ((language.REMOTE, language.INSPECT, language.SENSOR), "zoe prueft sensor.")),
         ((language.INSPECT, "prueft", language.MOTOR_INSPECT),
          (language.TEST, "testet", language.MOTOR_TEST))),
        (200000,
         ((language.REMOTE, "佐伊"), (language.BOB, "鲍勃"),
          (language.INSPECT, "检查"), (language.TEST, "测试"),
          (language.GOAL, "目标"), (language.SENSOR, "传感器"),
          (language.VALVE, "阀门")),
         (((language.BOB, language.TEST, language.VALVE), "鲍勃测试阀门。"),
          ((language.REMOTE, language.INSPECT, language.SENSOR), "佐伊检查传感器。")),
         ((language.INSPECT, "检查", language.MOTOR_INSPECT),
          (language.TEST, "测试", language.MOTOR_TEST))),
        (300000,
         ((language.REMOTE, "Зои"), (language.BOB, "Боб"),
          (language.INSPECT, "проверяет"), (language.TEST, "тестирует"),
          (language.GOAL, "цель"), (language.SENSOR, "датчик"),
          (language.VALVE, "клапан")),
         (((language.BOB, language.TEST, language.VALVE), "Боб тестирует клапан."),
          ((language.REMOTE, language.INSPECT, language.SENSOR), "Зои проверяет датчик.")),
         ((language.INSPECT, "проверяет", language.MOTOR_INSPECT),
          (language.TEST, "тестирует", language.MOTOR_TEST))),
        (400000,
         ((language.REMOTE, "zoe"), (language.BOB, "bob"),
          (language.INSPECT, "zoh"), (language.TEST, "chek"),
          (language.GOAL, "target"), (language.SENSOR, "sensor"),
          (language.VALVE, "valve")),
         (((language.BOB, language.VALVE, language.TEST), "bob nih valve kha a chek."),
          ((language.REMOTE, language.SENSOR, language.INSPECT), "zoe nih sensor kha a zoh.")),
         ((language.INSPECT, "zoh", language.MOTOR_INSPECT),
          (language.TEST, "chek", language.MOTOR_TEST))),
    )
    for base, names, clauses, actions in profiles:
        for entity, text in names:
            language.name(organism, entity, text, base + entity)
            language.name(organism, entity, text, base + 10000 + entity)
        for offset, (atoms, text) in enumerate(clauses):
            language.clause(organism, atoms, text, base + 20000 + 2 * offset)
            language.clause(organism, atoms, text, base + 20001 + 2 * offset)
        for offset, (atom, text, motor) in enumerate(actions):
            lexeme = organism.language.lexeme_identity(atom, language.u(text))
            if not organism._ground_language_action_recruitment(
                lexeme, motor, base + 30000 + offset, 1, True
            ):
                raise AssertionError("multilingual action grounding")
    return organism


def main():
    started = time.perf_counter()
    checks = {}
    spec = PopulationSpecV1(32768, 2, 4, 42, 8)

    # Repetition by one source is one causal voice. An unresolved conflict
    # becomes resident information need rather than a host-selected winner.
    repeated = prepared(spec)
    stage(repeated, (11,), 9501)
    repeated_id = say(repeated, language.PREDICTION, A)
    for _ in range(19):
        assert say(repeated, language.PREDICTION, A) == repeated_id
    say(repeated, TEST_PREDICTION, B)
    conflict_checkpoint = copy.deepcopy(repeated.checkpoint())
    conflict_action = repeated.tick()
    checks["repetition_is_one_voice_and_conflict_seeks_information"] = (
        conflict_action is None
        and repeated.information_need == (3, language.MOTOR_TEST, language.MOTOR_INSPECT)
        and next(row for row in repeated.source_assertions if row.identity == repeated_id).repetitions == 20
        and not repeated.cognition._evidence
    )
    conflict_replay = ReferenceOrganismV2.restore(copy.deepcopy(conflict_checkpoint))
    checks["conflict_checkpoint_replays_uncertainty"] = (
        conflict_replay.tick() is None
        and conflict_replay.information_need == repeated.information_need
    )
    say(repeated, language.PREDICTION, C)
    nonindependent_action = repeated.tick()
    nonindependent_result = settle(repeated, nonindependent_action, 9501, (language.STATE, language.GOAL), False)
    checks["nonindependent_return_cannot_calibrate_sources"] = (
        isinstance(nonindependent_action, MotorActionV2)
        and nonindependent_action.action_id == language.MOTOR_INSPECT
        and len(nonindependent_action.source_assertion_ids) == 2
        and nonindependent_result["source_updates"] == 0
        and not repeated.source_calibrations
    )

    # A real consequence calibrates only the sources that caused the action.
    # Later contradiction erases their advantage without deleting history.
    organism = prepared(spec)
    stage(organism, (101,), 9601)
    first_assertion = say(organism, language.PREDICTION, A)
    first = organism.tick()
    first_result = settle(organism, first, 9601, (language.STATE, language.GOAL), True)
    context = organism._source_context_signature()
    checks["actual_return_calibrates_participating_language_source"] = (
        first.source_assertion_ids == (first_assertion,)
        and first_result["source_updates"] == 1
        and organism._source_calibration(A, context) == 1
        and not organism.cognition.edges()
    )
    stage(organism, (102,), 9602)
    say(organism, TEST_PREDICTION, B)
    before_choice = copy.deepcopy(organism.checkpoint())
    replay = ReferenceOrganismV2.restore(copy.deepcopy(before_choice))
    preferred = organism.tick()
    replayed = replay.tick()
    checks["calibrated_source_wins_new_state_and_checkpoint"] = (
        isinstance(preferred, MotorActionV2)
        and preferred.action_id == language.MOTOR_INSPECT
        and preferred.source_assertion_ids == replayed.source_assertion_ids
        and preferred.action_id == replayed.action_id
    )
    contradicted = settle(organism, preferred, 9602, (language.STATE, language.WRONG), True)
    # Establish an ordinary physical exploration history for the same state but
    # an unrelated body target. This makes the later witness-selected test
    # action differ from the organism's current exploration counterfactual.
    stage(organism, (103,), 96025, target=(424242,), actions=(language.MOTOR_TEST,))
    unrelated = organism.tick()
    settle(organism, unrelated, 96025, (104,), True)
    stage(organism, (103,), 9603)
    after_contradiction = organism.tick()
    checks["contradiction_reopens_equal_witness_uncertainty"] = (
        contradicted["source_updates"] == 1
        and organism._source_calibration(A, context) == 0
        and after_contradiction is None
        and organism.information_need == (3, language.MOTOR_TEST, language.MOTOR_INSPECT)
        and any(row.source == A and row.active for row in organism.source_assertions)
    )
    say(organism, TEST_PREDICTION, C)
    two_witness_checkpoint = copy.deepcopy(organism.checkpoint())
    two_witness = organism.tick()
    checks["two_independent_language_sources_nominate_one_action"] = (
        isinstance(two_witness, MotorActionV2)
        and two_witness.action_id == language.MOTOR_TEST
        and two_witness.source_counterfactual_action == language.MOTOR_INSPECT
        and len(two_witness.source_assertion_ids) == 2
    )
    withdrawn = ReferenceOrganismV2.restore(copy.deepcopy(two_witness_checkpoint))
    withdrawn.contact(CONTACT_WITHDRAW_SOURCE, (C,), 9650, True, True)
    checks["withdrawing_second_witness_restores_uncertainty"] = (
        withdrawn.tick() is None
        and withdrawn.information_need == (3, language.MOTOR_TEST, language.MOTOR_INSPECT)
    )
    two_witness_result = settle(organism, two_witness, 9603, (language.STATE, language.GOAL), True)
    checks["one_return_calibrates_each_actual_participating_source"] = (
        two_witness_result["source_updates"] == 2
        and organism._source_calibration(B, context) == 1
        and organism._source_calibration(C, context) == 1
    )

    # Learned surface ecologies are an outer interface to the same source and
    # motor competition. The mixed utterance is a held-out recombination of a
    # German action lexeme and English target lexeme, not a routed translation.
    multilingual = prepared_multilingual(spec)
    english_binding = multilingual.language.invert_surface(language.u(language.PREDICTION))
    german_binding = multilingual.language.invert_surface(language.u(GERMAN_TEST_PREDICTION))
    mixed_binding = multilingual.language.invert_surface(language.u(MIXED_TEST_PREDICTION))
    checks["english_german_and_mixed_surfaces_share_nonlinguistic_atoms"] = (
        len(english_binding) == len(german_binding) == len(mixed_binding) == 1
        and english_binding[0].atoms == (language.REMOTE, language.INSPECT, language.GOAL)
        and german_binding[0].atoms == mixed_binding[0].atoms
        == (language.BOB, language.TEST, language.GOAL)
        and mixed_binding[0].lexical_identities[1] == german_binding[0].lexical_identities[1]
        and mixed_binding[0].lexical_identities[2] == english_binding[0].lexical_identities[2]
    )
    stage(multilingual, (201,), 9900, target=(424242,), actions=(language.MOTOR_TEST,))
    unrelated = multilingual.tick()
    settle(multilingual, unrelated, 9900, (202,), True)
    stage(multilingual, (201,), 9901)
    english_assertion = say(multilingual, language.PREDICTION, A)
    german_assertion = say(multilingual, GERMAN_TEST_PREDICTION, B)
    cross_language_conflict = multilingual.tick()
    checks["cross_language_conflict_recruits_information_need"] = (
        cross_language_conflict is None
        and multilingual.information_need == (3, language.MOTOR_TEST, language.MOTOR_INSPECT)
        and not multilingual.cognition.edges()
    )
    mixed_assertion = say(multilingual, MIXED_TEST_PREDICTION, C)
    multilingual_checkpoint = copy.deepcopy(multilingual.checkpoint())
    cross_language_choice = multilingual.tick()
    checks["learned_codeswitch_corroborates_without_language_router"] = (
        isinstance(cross_language_choice, MotorActionV2)
        and cross_language_choice.action_id == language.MOTOR_TEST
        and cross_language_choice.source_counterfactual_action == language.MOTOR_INSPECT
        and set(cross_language_choice.source_assertion_ids) == {german_assertion, mixed_assertion}
        and english_assertion not in cross_language_choice.source_assertion_ids
        and all(not hasattr(multilingual, name) for name in ("language_id", "language_router", "translate"))
    )
    multilingual_withdrawn = ReferenceOrganismV2.restore(copy.deepcopy(multilingual_checkpoint))
    multilingual_withdrawn.contact(CONTACT_WITHDRAW_SOURCE, (C,), 9950, True, True)
    checks["codeswitch_source_withdrawal_restores_cross_language_uncertainty"] = (
        multilingual_withdrawn.tick() is None
        and multilingual_withdrawn.information_need
        == (3, language.MOTOR_TEST, language.MOTOR_INSPECT)
    )
    multilingual_result = settle(
        multilingual, cross_language_choice, 9901, (language.STATE, language.GOAL), True
    )
    multilingual_context = multilingual._source_context_signature()
    checks["counterfactual_return_calibrates_only_multilingual_participants"] = (
        multilingual_result["source_updates"] == 2
        and multilingual._source_calibration(A, multilingual_context) == 0
        and multilingual._source_calibration(B, multilingual_context) == 1
        and multilingual._source_calibration(C, multilingual_context) == 1
    )
    multilingual.contact(CONTACT_WITHDRAW_SOURCE, (C,), 9951, True, True)
    stage(multilingual, (203,), 99020, target=(424243,), actions=(language.MOTOR_TEST,))
    unrelated_transfer = multilingual.tick()
    settle(multilingual, unrelated_transfer, 99020, (204,), True)
    stage(multilingual, (203,), 9902)
    multilingual_before_transfer = copy.deepcopy(multilingual.checkpoint())
    multilingual_before_digest = multilingual.digest()
    transferred = multilingual.tick()
    checks["source_calibration_transfers_across_language_and_changed_state"] = (
        isinstance(transferred, MotorActionV2)
        and transferred.action_id == language.MOTOR_TEST
        and transferred.source_assertion_ids == (german_assertion,)
        and transferred.source_counterfactual_action == language.MOTOR_INSPECT
    )
    multilingual_replay = ReferenceOrganismV2.restore(multilingual_before_transfer)
    multilingual_replay_digest = multilingual_replay.digest()
    replayed_transfer = multilingual_replay.tick()
    checks["multilingual_checkpoint_replays_same_source_competition"] = (
        multilingual_replay_digest == multilingual_before_digest
        and isinstance(replayed_transfer, MotorActionV2)
        and replayed_transfer.action_id == transferred.action_id
        and replayed_transfer.source_assertion_ids == transferred.source_assertion_ids
    )

    # Five learned languages enter one source ecology. Hakha Chin contributes a
    # genuinely different S-O-V construction; no host step normalizes it into
    # the S-V-O order used by the other four surface histories.
    five_language = prepared_multilingual(spec)
    five_surfaces = (
        (language.PREDICTION, (language.REMOTE, language.INSPECT, language.GOAL)),
        (GERMAN_TEST_PREDICTION, (language.BOB, language.TEST, language.GOAL)),
        (MANDARIN_TEST_PREDICTION, (language.BOB, language.TEST, language.GOAL)),
        (RUSSIAN_TEST_PREDICTION, (language.BOB, language.TEST, language.GOAL)),
        (CHIN_TEST_PREDICTION, (language.BOB, language.GOAL, language.TEST)),
    )
    five_bindings = tuple(
        five_language.language.invert_surface(language.u(surface))
        for surface, _ in five_surfaces
    )
    checks["five_languages_recombine_heldout_surface_relations"] = all(
        len(bindings) == 1 and bindings[0].atoms == expected
        for bindings, (_, expected) in zip(five_bindings, five_surfaces)
    )
    checks["chin_sov_is_learned_order_not_host_normalization"] = (
        five_bindings[-1][0].atoms == (language.BOB, language.GOAL, language.TEST)
        and five_bindings[-1][0].atoms != five_bindings[1][0].atoms
        and set(five_bindings[-1][0].atoms) == set(five_bindings[1][0].atoms)
        and five_bindings[-1][0].template_identity != five_bindings[1][0].template_identity
        and len(five_language.language.template_candidates(language.CTX, 3)) == 4
    )
    stage(five_language, (301,), 10900, target=(424244,), actions=(language.MOTOR_TEST,))
    five_unrelated = five_language.tick()
    settle(five_language, five_unrelated, 10900, (302,), True)
    stage(five_language, (301,), 10901)
    five_assertions = (
        say(five_language, language.PREDICTION, A),
        say(five_language, GERMAN_TEST_PREDICTION, B),
        say(five_language, MANDARIN_TEST_PREDICTION, C),
        say(five_language, RUSSIAN_TEST_PREDICTION, D),
        say(five_language, CHIN_TEST_PREDICTION, E),
    )
    five_before_choice = copy.deepcopy(five_language.checkpoint())
    five_choice = five_language.tick()
    five_source_touches = five_language.last_source_touches
    checks["five_language_sources_compete_in_one_nonlinguistic_network"] = (
        isinstance(five_choice, MotorActionV2)
        and five_choice.action_id == language.MOTOR_TEST
        and five_choice.source_counterfactual_action == language.MOTOR_INSPECT
        and set(five_choice.source_assertion_ids) == set(five_assertions[1:])
        and five_assertions[0] not in five_choice.source_assertion_ids
        and five_source_touches == 5
        and not five_language.cognition.edges()
    )
    five_replay = ReferenceOrganismV2.restore(copy.deepcopy(five_before_choice))
    replayed_five_choice = five_replay.tick()
    checks["five_language_checkpoint_replays_competition"] = (
        isinstance(replayed_five_choice, MotorActionV2)
        and replayed_five_choice.action_id == five_choice.action_id
        and replayed_five_choice.source_assertion_ids == five_choice.source_assertion_ids
    )
    five_withdrawn = ReferenceOrganismV2.restore(copy.deepcopy(five_before_choice))
    for source in (C, D, E):
        five_withdrawn.contact(CONTACT_WITHDRAW_SOURCE, (source,), 11000 + source, True, True)
    checks["withdrawing_three_languages_restores_cross_language_tie"] = (
        five_withdrawn.tick() is None
        and five_withdrawn.information_need
        == (3, language.MOTOR_TEST, language.MOTOR_INSPECT)
    )
    five_result = settle(
        five_language, five_choice, 10901, (language.STATE, language.GOAL), True
    )
    five_context = five_language._source_context_signature()
    checks["five_language_return_calibrates_only_causal_sources"] = (
        five_result["source_updates"] == 4
        and five_language._source_calibration(A, five_context) == 0
        and all(five_language._source_calibration(source, five_context) == 1
                for source in (B, C, D, E))
    )

    # Prefix-sharing decoys exercise the derived inverse index without changing
    # the current utterance or retaining a tokenized sentence representation.
    quantity = prepared_multilingual(spec)
    quantity_probe = quantity.language.invert_surface(language.u(language.PREDICTION))
    quantity_baseline_touches = quantity_probe[0].candidate_touches
    for i in range(128):
        subject, verb, target = 600000 + 3 * i, 600001 + 3 * i, 600002 + 3 * i
        subject_text = f"zoex{i:03d}"
        verb_text = f"inspectsx{i:03d}"
        target_text = f"targetx{i:03d}"
        source_base = 6000000 + 10000 * i
        for offset, (atom, text) in enumerate((
            (subject, subject_text), (verb, verb_text), (target, target_text)
        )):
            language.name(quantity, atom, text, source_base + 2 * offset)
            language.name(quantity, atom, text, source_base + 2 * offset + 1)
        decoy_surface = f"{subject_text} {verb_text} {target_text}.q{i:03d}"
        language.clause(quantity, (subject, verb, target), decoy_surface, source_base + 2000)
        language.clause(quantity, (subject, verb, target), decoy_surface, source_base + 2001)
    quantity_scaled = quantity.language.invert_surface(language.u(language.PREDICTION))
    quantity_scaled_touches = quantity_scaled[0].candidate_touches
    checks["inverse_lookup_is_sparse_under_prefix_sharing_quantity"] = (
        len(quantity_scaled) == 1
        and quantity_scaled[0].atoms == quantity_probe[0].atoms
        and quantity_scaled[0].template_identity == quantity_probe[0].template_identity
        and quantity_scaled_touches <= quantity_baseline_touches + 2
        and len(quantity.language.template_candidates(language.CTX, 3)) == 132
    )
    quantity_checkpoint = quantity.checkpoint()
    checks["inverse_index_is_rebuildable_not_checkpoint_authority"] = (
        "_inverse_surface_trie" not in json.dumps(quantity_checkpoint, sort_keys=True)
        and ReferenceOrganismV2.restore(copy.deepcopy(quantity_checkpoint)).digest()
        == quantity.digest()
    )

    # Authoritative lived transition planning precedes even numerous raw
    # testimony Recipes. The utterances remain causal proposals, not truth.
    direct = prepared(spec)
    for source in (9701, 9702):
        stage(direct, (701,), source, actions=(language.MOTOR_TEST,))
        action = direct.tick()
        settle(direct, action, source, (language.GOAL,), True)
    stage(direct, (701,), 9703)
    for source in range(9800, 9820):
        say(direct, language.PREDICTION, source)
    authoritative = direct.tick()
    checks["lived_world_relation_precedes_many_language_shadows"] = (
        isinstance(authoritative, MotorActionV2)
        and authoritative.action_id == language.MOTOR_TEST
        and not authoritative.source_assertion_ids
    )

    # Context incidence must avoid a whole source-history scan.
    scaled = prepared(spec)
    for i in range(128):
        stage(scaled, (10000 + i,), 11000 + i, (language.GOAL, 20000 + i))
        say(scaled, language.PREDICTION, 12000 + i)
    stage(scaled, (20000,), 13000)
    say(scaled, language.PREDICTION, 14001)
    say(scaled, TEST_PREDICTION, 14002)
    scaled._source_nomination(False)
    checks["shadow_competition_is_sparse_to_current_context"] = (
        scaled.last_source_touches == 2
        and len(scaled.source_assertions) == 130
    )
    restored = ReferenceOrganismV2.restore(copy.deepcopy(organism.checkpoint()))
    checks["complete_checkpoint_preserves_source_ecology"] = restored.digest() == organism.digest()
    checks["no_global_trust_or_language_truth_object"] = (
        not hasattr(organism, "trust")
        and not hasattr(organism, "belief")
        and not hasattr(organism, "language_truth")
    )

    result = {
        "schema": "agi.reference-language-two-witness.v3",
        "pass": all(checks.values()),
        "checks": checks,
        "reference_only": True,
        "adult_attached": False,
        "runtime_llm": False,
        "graph_flip": False,
        "physical_direct_parity": "NOT_RUN/RED",
        "claim": "MULTILINGUAL_RAW_LANGUAGE_SOURCE_COMPETITION_AND_REVISION_REFERENCE_ONLY_NOT_HUMAN_SOCIAL_COGNITION",
        "resource": {
            "resident_sites": spec.site_count,
            "source_rows_scaled": len(scaled.source_assertions),
            "current_source_touches": scaled.last_source_touches,
            "checkpoint_bytes": len(json.dumps(organism.checkpoint(), sort_keys=True, separators=(",", ":"))),
            "repeated_utterances_one_source": 20,
            "misleading_sources_against_direct_world": 20,
            "languages": 5,
            "mixed_surface_ecologies": 1,
            "five_language_source_touches": five_source_touches,
            "five_language_binding_touches": sum(
                bindings[0].candidate_touches for bindings in five_bindings
            ),
            "shared_construction_recipes": len(
                five_language.language.template_candidates(language.CTX, 3)
            ),
            "quantity_decoy_lexemes": 384,
            "quantity_decoy_constructions": 128,
            "quantity_baseline_touches": quantity_baseline_touches,
            "quantity_scaled_touches": quantity_scaled_touches,
            "quantity_checkpoint_bytes": len(
                json.dumps(quantity_checkpoint, sort_keys=True, separators=(",", ":"))
            ),
            "multilingual_checkpoint_bytes": len(
                json.dumps(multilingual.checkpoint(), sort_keys=True, separators=(",", ":"))
            ),
            "five_language_checkpoint_bytes": len(
                json.dumps(five_language.checkpoint(), sort_keys=True, separators=(",", ":"))
            ),
        },
        "remaining_red": [
            "OPEN_ENDED_MULTILINGUAL_LEXICON_MORPHOLOGY_AND_ACQUISITION",
            "OPEN_ENDED_SOCIAL_HISTORY",
            "PHYSICAL_SOURCE_AUTHENTICATION",
            "DIRECT_TRANSLATION",
        ],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print(
        "FOUNDRY_LANGUAGE_TWO_WITNESS "
        + ("GREEN" if result["pass"] else "RED")
        + " repetition_vote=0 conflict_inquiry=1 calibrated_revision=1 languages=5 sov=1 codeswitch=1 direct_world_precedence=1"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
