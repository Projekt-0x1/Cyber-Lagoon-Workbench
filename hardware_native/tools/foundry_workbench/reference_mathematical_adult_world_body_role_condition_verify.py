#!/usr/bin/env python3
"""Destructive audit for lawful world/body seeding of Adult structural role conditions."""
from __future__ import annotations

import copy
import inspect
import json
import time
from pathlib import Path

from reference_language_mastery_adult_v1 import AdultStateV1, LanguageMasteryAdultV1
from reference_mathematical_adult_structural_dependency_migration_verify import (
    ROLE_BASE,
    atom_for_leaf,
    build_program,
    canonical_surface,
    train,
)
from reference_organism_v2 import (
    CONTACT_BODY_STATE,
    CONTACT_BODY_TARGET,
    CONTACT_ENTITY_FEATURES,
    CONTACT_WITHDRAW_SOURCE,
    CONTACT_WORLD_STATE,
    ReferenceOrganismV2,
)
from reference_population_v1 import PopulationSpecV1
from reference_predictive_credit_profile_v1 import Q
from reference_program_role_embodied_condition_bridge_v1 import (
    ProgramRoleEmbodiedConditionBridgeV1,
)

language_phenotype_improved = True
future_update_authority_preserved = True

WORLD_SOURCE = 0xD901
FEATURE_BASE = 0xDA00
FORM_SOURCE_A = 0xDB01
FORM_SOURCE_B = 0xDB02
BODY_SOURCE = 0xDC01
BODY_STATE = (831, 832, 833)


def organism():
    return ReferenceOrganismV2(PopulationSpecV1(32768, 2, 4, 42, 8))


def install_shared_feature(owner, feature, base, same_source=False):
    identities = (0x700000 + int(base) * 4, 0x700001 + int(base) * 4)
    sources = (
        0x710000 + int(base) * 4,
        0x710000 + int(base) * 4 if same_source else 0x710001 + int(base) * 4,
    )
    for entity, source in zip(identities, sources):
        owner.contact(
            CONTACT_ENTITY_FEATURES,
            (int(entity), 1, int(feature)),
            int(source),
            True,
            True,
        )
    return identities, sources


def set_world(owner, members, source=WORLD_SOURCE):
    return owner.contact(
        CONTACT_WORLD_STATE,
        tuple(map(int, members)),
        int(source),
        True,
        True,
    )


def form_units(prefix: bytes, atom: int) -> tuple[int, ...]:
    return tuple(prefix + ("%04x" % int(atom)).encode())


def teach_form(adult, atom, condition, prefix):
    units = form_units(prefix, atom)
    for source in (FORM_SOURCE_A + int(atom), FORM_SOURCE_B + int(atom)):
        adult.language.observe_form(int(atom), (int(condition),), units, int(source))
    if adult.language.form(int(atom), (int(condition),), require_conditioned=True) != units:
        raise RuntimeError("world_role:form")
    return units


def mapping_for(bridge, adult, owner, program):
    return bridge.conditions(adult, owner, int(program), ROLE_BASE)


def main():
    started = time.perf_counter()
    bridge = ProgramRoleEmbodiedConditionBridgeV1
    state = train()
    adult = state["adult"]
    held = state["groups"][2]
    leaves_h = state["leafsets"][2]
    program_h = int(state["programs"][2])
    view_h = state["views"][2]
    held_controller = int(leaves_h[0].identity)
    held_distractor = int(leaves_h[6].identity)
    held_target = int(leaves_h[7].identity)
    controller_atom = atom_for_leaf(view_h, held_controller)
    distractor_atom = atom_for_leaf(view_h, held_distractor)
    target_atom = atom_for_leaf(view_h, held_target)

    # Developmental feature history is organism-owned and independently sourced.
    world_owner = organism()
    controller_entities, controller_sources = install_shared_feature(
        world_owner, controller_atom, FEATURE_BASE + 1
    )
    distractor_entities, distractor_sources = install_shared_feature(
        world_owner, distractor_atom, FEATURE_BASE + 2
    )

    world_a = (*controller_entities, distractor_entities[0])
    set_world(world_owner, world_a)
    map_a_pre = mapping_for(bridge, adult, world_owner, program_h)
    if held_controller not in map_a_pre or held_distractor not in map_a_pre:
        raise RuntimeError("world_role:world_a_conditions")
    controller_many = int(map_a_pre[held_controller][0])
    distractor_one = int(map_a_pre[held_distractor][0])
    if controller_many == distractor_one:
        raise RuntimeError("world_role:numerosity_collision")

    # World B gives the same two opaque morphologies to the opposite feature populations.
    world_b = (controller_entities[0], *distractor_entities)
    set_world(world_owner, world_b)
    map_b_pre = mapping_for(bridge, adult, world_owner, program_h)
    controller_one = int(map_b_pre[held_controller][0])
    distractor_many = int(map_b_pre[held_distractor][0])
    if not (controller_one == distractor_one and distractor_many == controller_many):
        raise RuntimeError("world_role:world_swap_conditions")

    # Conditioned lexical forms are independently learned language evidence.  The bridge
    # learns none of them; it only supplies whichever current opaque morphology exists.
    world_forms = {}
    for atom in (controller_atom, distractor_atom, target_atom):
        world_forms[(atom, controller_one)] = teach_form(adult, atom, controller_one, b"one-")
        world_forms[(atom, controller_many)] = teach_form(adult, atom, controller_many, b"many-")

    adult_checkpoint = copy.deepcopy(adult.checkpoint())
    adult_checkpoint_text = json.dumps(adult_checkpoint, sort_keys=True, separators=(",", ":"))
    language_checkpoint_before = json.dumps(
        adult.language.checkpoint(), sort_keys=True, separators=(",", ":")
    )

    set_world(world_owner, world_a)
    world_a_checkpoint = copy.deepcopy(world_owner.checkpoint())
    map_a = mapping_for(bridge, adult, world_owner, program_h)
    output_a = bridge.realize(adult, world_owner, program_h, ROLE_BASE)
    expected_a = canonical_surface(
        adult,
        view_h,
        {
            held_controller: world_forms[(controller_atom, controller_many)],
            held_distractor: world_forms[(distractor_atom, distractor_one)],
            held_target: world_forms[(target_atom, controller_many)],
        },
    )

    set_world(world_owner, world_b)
    map_b = mapping_for(bridge, adult, world_owner, program_h)
    output_b = bridge.realize(adult, world_owner, program_h, ROLE_BASE)
    expected_b = canonical_surface(
        adult,
        view_h,
        {
            held_controller: world_forms[(controller_atom, controller_one)],
            held_distractor: world_forms[(distractor_atom, distractor_many)],
            held_target: world_forms[(target_atom, controller_one)],
        },
    )

    # Same current world + same terminal concepts, but changed local structural ancestry.
    wrong_adult = LanguageMasteryAdultV1.restore(copy.deepcopy(adult_checkpoint))
    wrong_concepts = list(held)
    wrong_concepts[0], wrong_concepts[2] = wrong_concepts[2], wrong_concepts[0]
    _wrong_leaves, wrong_program = build_program(wrong_adult, tuple(wrong_concepts), 2)
    set_world(world_owner, world_a)
    wrong_output = bridge.realize(wrong_adult, world_owner, wrong_program, ROLE_BASE)

    # Destructive feature-source lesion at fixed current world tuple.
    feature_cut = ReferenceOrganismV2.restore(copy.deepcopy(world_a_checkpoint))
    feature_cut.contact(
        CONTACT_WITHDRAW_SOURCE,
        (int(controller_sources[1]),),
        0xDD01,
        True,
        True,
    )
    cut_map = mapping_for(bridge, adult, feature_cut, program_h)
    cut_output = bridge.realize(adult, feature_cut, program_h, ROLE_BASE)
    fixed_world_after_feature_cut = tuple(feature_cut.world_state or ())

    # New independent feature evidence restores the current condition without touching Adult.
    feature_reacquired = ReferenceOrganismV2.restore(copy.deepcopy(feature_cut.checkpoint()))
    feature_reacquired.contact(
        CONTACT_ENTITY_FEATURES,
        (int(controller_entities[1]), 1, int(controller_atom)),
        0xDD02,
        True,
        True,
    )
    reacquired_map = mapping_for(bridge, adult, feature_reacquired, program_h)
    reacquired_output = bridge.realize(adult, feature_reacquired, program_h, ROLE_BASE)

    # Same count, but no independent developmental source geometry: must not fabricate numerosity.
    fake = organism()
    fake_entities, _fake_sources = install_shared_feature(
        fake, controller_atom, FEATURE_BASE + 3, same_source=True
    )
    set_world(fake, fake_entities, 0xD903)
    fake_map = mapping_for(bridge, adult, fake, program_h)

    # Current world source withdrawal abolishes world projection without touching feature history.
    world_cut = ReferenceOrganismV2.restore(copy.deepcopy(world_a_checkpoint))
    world_cut.contact(CONTACT_WITHDRAW_SOURCE, (WORLD_SOURCE,), 0xDD03, True, True)
    world_cut_map = mapping_for(bridge, adult, world_cut, program_h)
    world_cut_output = bridge.realize(adult, world_cut, program_h, ROLE_BASE)

    # Body-only arm.  No world feature geometry exists in this organism.
    body_owner = organism()
    body_owner.contact(
        CONTACT_BODY_TARGET, (controller_atom,), BODY_SOURCE, True, True
    )
    body_owner.contact(CONTACT_BODY_STATE, BODY_STATE, BODY_SOURCE, True, True)
    body_map_pre = mapping_for(bridge, adult, body_owner, program_h)
    if held_controller not in body_map_pre:
        raise RuntimeError("world_role:body_condition")
    body_condition = int(body_map_pre[held_controller][0])
    body_forms = {}
    for atom in (controller_atom, distractor_atom, target_atom):
        body_forms[atom] = teach_form(adult, atom, body_condition, b"body-")
    body_language_checkpoint = json.dumps(
        adult.language.checkpoint(), sort_keys=True, separators=(",", ":")
    )
    body_adult_checkpoint = copy.deepcopy(adult.checkpoint())
    body_owner_checkpoint = copy.deepcopy(body_owner.checkpoint())
    body_map = mapping_for(bridge, adult, body_owner, program_h)
    body_output = bridge.realize(adult, body_owner, program_h, ROLE_BASE)
    body_expected = canonical_surface(
        adult,
        view_h,
        {
            held_controller: body_forms[controller_atom],
            held_target: body_forms[target_atom],
        },
    )

    body_cut = ReferenceOrganismV2.restore(copy.deepcopy(body_owner_checkpoint))
    body_cut.contact(CONTACT_WITHDRAW_SOURCE, (BODY_SOURCE,), 0xDD04, True, True)
    body_cut_map = mapping_for(bridge, adult, body_cut, program_h)
    body_cut_output = bridge.realize(adult, body_cut, program_h, ROLE_BASE)

    body_distractor = ReferenceOrganismV2.restore(copy.deepcopy(body_owner_checkpoint))
    body_distractor.contact(
        CONTACT_BODY_TARGET, (distractor_atom,), BODY_SOURCE, True, True
    )
    distractor_body_map = mapping_for(bridge, adult, body_distractor, program_h)
    distractor_body_output = bridge.realize(
        adult, body_distractor, program_h, ROLE_BASE
    )

    # Projection is current computation.  Adult pressure/control state is not an input.
    conditions_before_pressure = mapping_for(bridge, adult, feature_cut, program_h)
    adult.choose(AdultStateV1(urgency_q16=Q, pressure_q16=Q))
    conditions_after_pressure = mapping_for(bridge, adult, feature_cut, program_h)

    target_many = bytes(world_forms[(target_atom, controller_many)])
    target_one = bytes(world_forms[(target_atom, controller_one)])
    target_body = bytes(body_forms[target_atom])
    distractor_body_mark = bytes(body_forms[distractor_atom])
    bridge_source = inspect.getsource(ProgramRoleEmbodiedConditionBridgeV1)

    checks = {
        "world_a_controller_condition_beats_nearer_distractor": (
            map_a.get(held_controller) == (controller_many,)
            and map_a.get(held_distractor) == (distractor_one,)
            and bytes(output_a or ()) == expected_a
            and target_many in bytes(output_a or ())
        ),
        "world_swap_changes_remote_target_with_adult_state_fixed": (
            map_b.get(held_controller) == (controller_one,)
            and map_b.get(held_distractor) == (distractor_many,)
            and bytes(output_b or ()) == expected_b
            and target_one in bytes(output_b or ())
            and target_many not in bytes(output_b or ())
        ),
        "same_world_same_terminals_changed_ancestry_loses_remote_condition": (
            wrong_output is not None and target_many not in bytes(wrong_output)
        ),
        "feature_source_lesion_at_fixed_world_tuple_is_causal": (
            fixed_world_after_feature_cut == tuple(sorted(world_a))
            and held_controller not in cut_map
            and cut_map.get(held_distractor) == (distractor_one,)
            and cut_output is not None
            and target_many not in bytes(cut_output)
        ),
        "independent_feature_reacquisition_restores_without_language_relearning": (
            reacquired_map.get(held_controller) == (controller_many,)
            and bytes(reacquired_output or ()) == expected_a
            and json.dumps(adult.language.checkpoint(), sort_keys=True, separators=(",", ":"))
            == body_language_checkpoint
        ),
        "one_source_two_entity_fake_cannot_create_numerosity": held_controller not in fake_map,
        "world_source_withdrawal_empties_world_projection": (
            held_controller not in world_cut_map
            and held_distractor not in world_cut_map
            and world_cut_output is not None
            and target_many not in bytes(world_cut_output)
        ),
        "authenticated_body_condition_routes_through_structural_controller": (
            body_map.get(held_controller) == (body_condition,)
            and bytes(body_output or ()) == body_expected
            and target_body in bytes(body_output or ())
        ),
        "body_source_withdrawal_removes_remote_body_form": (
            held_controller not in body_cut_map
            and body_cut_output is not None
            and target_body not in bytes(body_cut_output)
        ),
        "nearer_body_target_does_not_become_structural_controller": (
            held_controller not in distractor_body_map
            and distractor_body_map.get(held_distractor) == (body_condition,)
            and distractor_body_output is not None
            and distractor_body_mark in bytes(distractor_body_output)
            and target_body not in bytes(distractor_body_output)
        ),
        "pressure_and_action_selection_cannot_fabricate_projection": (
            conditions_before_pressure == conditions_after_pressure == cut_map
        ),
        "bridge_projection_does_not_mutate_adult_checkpoint": (
            json.dumps(body_adult_checkpoint, sort_keys=True, separators=(",", ":"))
            == json.dumps(adult.checkpoint(), sort_keys=True, separators=(",", ":"))
        ),
        "organism_remains_world_body_state_owner": (
            "world_state" in world_a_checkpoint
            and "body_state" in body_owner_checkpoint
            and all(
                token not in adult_checkpoint_text
                for token in ("world_state", "body_state", "world_source", "body_state_source")
            )
        ),
        "bridge_is_stateless_and_has_no_semantic_role_authority": (
            not vars(ProgramRoleEmbodiedConditionBridgeV1())
            and not hasattr(ProgramRoleEmbodiedConditionBridgeV1, "checkpoint")
            and all(
                token not in bridge_source.lower()
                for token in (
                    "controller",
                    "target",
                    "subject",
                    "agreement",
                    "plural",
                    "singular",
                    "expected",
                    "score",
                    "credit",
                )
            )
        ),
        "bridge_writes_no_language_evidence": (
            language_checkpoint_before
            != body_language_checkpoint
            and json.dumps(adult.language.checkpoint(), sort_keys=True, separators=(",", ":"))
            == body_language_checkpoint
        ),
        "productive_terminals_remain_factored_not_raw": adult.program_surface_checkpoint()[
            "raw_leaf_surfaces"
        ]
        == [],
        "structural_role_view_stays_fixed_across_world_body_changes": (
            adult.program_role_view(program_h, ROLE_BASE) == view_h
        ),
        "both_fast_unions_run_world_body_role_condition_audit": (
            "adult-world-body-role:reference_mathematical_adult_world_body_role_condition_verify.py"
            in (Path(__file__).parent / "run_language_mastery_fast.sh").read_text()
            and "reference_mathematical_adult_world_body_role_condition_verify.py"
            in (Path(__file__).parent / "run_language_mastery_factory_fast.sh").read_text()
        ),
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-world-body-role-condition.v1",
        "pass": not failed,
        "reference_only": True,
        "language_phenotype_improved": language_phenotype_improved,
        "visible_language_gain": "LIVE_WORLD_AND_BODY_STATE_SEED_REMOTE_MATHEMATICAL_ADULT_STRUCTURAL_FORM_WITHOUT_HOST_ROLE_FLAG",
        "future_update_authority_preserved": future_update_authority_preserved,
        "conditions": {
            "world_one": controller_one,
            "world_many": controller_many,
            "body": body_condition,
        },
        "checks": checks,
        "failed": failed,
        "remaining_red": [
            "DIRECT_WORLD_BODY_STRUCTURAL_DEPENDENCY_PARITY",
            "MULTISOURCE_CONSENSUS_AND_DECEPTION",
            "CONTINUOUS_LIFE_STRUCTURAL_WORLD_INTERFERENCE",
        ],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print(
        "FOUNDRY_MATHEMATICAL_ADULT_WORLD_BODY_ROLE_CONDITION_"
        + ("GREEN" if not failed else "RED")
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
