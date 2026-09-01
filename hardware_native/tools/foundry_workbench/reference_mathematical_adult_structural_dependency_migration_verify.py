#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from reference_hierarchical_composition_v1 import HierarchicalRefuse
from reference_language_mastery_adult_v1 import AdultStateV1, LanguageMasteryAdultV1
from reference_predictive_credit_profile_v1 import Q

language_phenotype_improved = True
future_update_authority_preserved = True

WORD = 0xA701
ROLE_BASE = 0xA702
COND = 0xA703
OTHER = 0xA704
PAIR = (0xB100, 0xB101, 0xB102, 0xB103)
HIGH = (0xB200, 0xB201, 0xB202, 0xB203, 0xB204, 0xB205)
ROOT = (0xB300, 0xB301, 0xB302)
ROLE_SOURCES = (0xC001, 0xC002)
REACQUIRE_SOURCE = 0xC003
FORM_SOURCES = (0xC101, 0xC102)
TRAIN_CTX = 0xD001


def token(concept: int) -> bytes:
    return ("w%04x" % int(concept)).encode()


def marked(prefix: bytes, concept: int) -> bytes:
    return prefix + ("%04x" % int(concept)).encode()


def earn(adult, operator, left_obj, left_id, right_obj, right_id):
    witness = adult.compose(operator, left_obj, right_obj)
    chunk = None
    for _ in range(3):
        chunk = adult.experience_program(
            (left_id, right_id), witness, Q // 2, 0, TRAIN_CTX, Q // 32, True
        )
    if chunk is None:
        raise RuntimeError("migration:chunk")
    return witness, int(chunk.identity)


def teach_words(adult, groups):
    for concepts in groups:
        for concept in concepts:
            raw = tuple(token(concept))
            for source in (0xE001, 0xE002):
                adult.observe_surface_item(concept, raw, source + concept)
            for source in (0xE101, 0xE102):
                assert adult.observe_surface_construction(WORD, (concept,), raw, source + concept)


def leaves(adult, concepts):
    return tuple(adult.leaf(WORD, (concept,)) for concept in concepts)


def learn_operator(adult, operator, left, right):
    for source in (0xF001 + operator, 0xF101 + operator):
        assert adult.observe_join(operator, left, right, source, b" ")


def prepare_operators(adult, seed_leaves):
    for i, operator in enumerate(PAIR):
        learn_operator(adult, operator, seed_leaves[2 * i], seed_leaves[2 * i + 1])
    pairs = [adult.compose(PAIR[i], seed_leaves[2 * i], seed_leaves[2 * i + 1]) for i in range(4)]
    for operator, left, right in (
        (HIGH[0], pairs[0], pairs[1]),
        (HIGH[1], pairs[2], pairs[3]),
        (HIGH[2], pairs[0], pairs[2]),
        (HIGH[3], pairs[1], pairs[3]),
        (HIGH[4], pairs[3], pairs[2]),
        (HIGH[5], pairs[1], pairs[0]),
    ):
        learn_operator(adult, operator, left, right)
    mids = (
        adult.compose(HIGH[0], pairs[0], pairs[1]),
        adult.compose(HIGH[1], pairs[2], pairs[3]),
        adult.compose(HIGH[2], pairs[0], pairs[2]),
        adult.compose(HIGH[3], pairs[1], pairs[3]),
        adult.compose(HIGH[4], pairs[3], pairs[2]),
        adult.compose(HIGH[5], pairs[1], pairs[0]),
    )
    for operator, left, right in (
        (ROOT[0], mids[0], mids[1]),
        (ROOT[1], mids[2], mids[3]),
        (ROOT[2], mids[4], mids[5]),
    ):
        learn_operator(adult, operator, left, right)


def build_program(adult, concepts, variant):
    terminal = leaves(adult, concepts)
    pair_witnesses = []
    pair_programs = []
    for i, operator in enumerate(PAIR):
        witness, program = earn(
            adult,
            operator,
            terminal[2 * i],
            terminal[2 * i].identity,
            terminal[2 * i + 1],
            terminal[2 * i + 1].identity,
        )
        pair_witnesses.append(witness)
        pair_programs.append(program)
    if variant == 0:
        specs = ((HIGH[0], 0, 1), (HIGH[1], 2, 3))
        root_operator = ROOT[0]
    elif variant == 1:
        specs = ((HIGH[2], 0, 2), (HIGH[3], 1, 3))
        root_operator = ROOT[1]
    else:
        specs = ((HIGH[4], 3, 2), (HIGH[5], 1, 0))
        root_operator = ROOT[2]
    mid_witnesses = []
    mid_programs = []
    for operator, left_i, right_i in specs:
        witness, program = earn(
            adult,
            operator,
            pair_witnesses[left_i],
            pair_programs[left_i],
            pair_witnesses[right_i],
            pair_programs[right_i],
        )
        mid_witnesses.append(witness)
        mid_programs.append(program)
    _root_witness, root_program = earn(
        adult,
        root_operator,
        mid_witnesses[0],
        mid_programs[0],
        mid_witnesses[1],
        mid_programs[1],
    )
    return terminal, root_program


def canonical_surface(adult, view, overrides=None):
    overrides = overrides or {}
    parts = []
    for leaf, atom in reversed(tuple(zip(view["leaves"], view["atoms"]))):
        units = overrides.get(int(leaf))
        if units is None:
            units = adult.language.lexeme(int(atom))
        if units is None:
            raise RuntimeError("migration:surface")
        parts.append(bytes(units))
    return b" ".join(parts)


def traversal_leaves(adult, program_identity):
    out = []

    def walk(identity):
        chunk = adult.programs.chunks.get(int(identity))
        if chunk is None:
            out.append(int(identity))
            return
        for member in chunk.members:
            walk(member)

    walk(program_identity)
    return tuple(out)


def role_for_leaf(view, leaf_identity):
    return dict(zip(view["leaves"], view["roles"]))[int(leaf_identity)]


def atom_for_leaf(view, leaf_identity):
    return dict(zip(view["leaves"], view["atoms"]))[int(leaf_identity)]


def train():
    adult = LanguageMasteryAdultV1()
    donor_a = tuple(range(0x1101, 0x1109))
    donor_b = tuple(range(0x1201, 0x1209))
    held = tuple(range(0x1301, 0x1309))
    teach_words(adult, (donor_a, donor_b, held))
    prepare_operators(adult, leaves(adult, donor_a))
    leaves_a, program_a = build_program(adult, donor_a, 0)
    leaves_b, program_b = build_program(adult, donor_b, 1)
    leaves_h, program_h = build_program(adult, held, 2)
    view_a = adult.program_role_view(program_a, ROLE_BASE)
    view_b = adult.program_role_view(program_b, ROLE_BASE)
    view_h = adult.program_role_view(program_h, ROLE_BASE)
    if not (
        view_a["context"] == view_b["context"] == view_h["context"]
        and len({view_a["topology"], view_b["topology"], view_h["topology"]}) == 3
    ):
        raise RuntimeError("migration:role_view")

    for program, view, source in (
        (program_a, view_a, ROLE_SOURCES[0]),
        (program_b, view_b, ROLE_SOURCES[1]),
    ):
        assert adult.observe_program_role_construction(
            program, tuple(canonical_surface(adult, view)), source, ROLE_BASE
        )

    controller_role = role_for_leaf(view_a, leaves_a[0].identity)
    distractor_role = role_for_leaf(view_a, leaves_a[6].identity)
    target_role = role_for_leaf(view_a, leaves_a[7].identity)
    assert role_for_leaf(view_b, leaves_b[0].identity) == controller_role
    assert role_for_leaf(view_b, leaves_b[6].identity) == distractor_role
    assert role_for_leaf(view_b, leaves_b[7].identity) == target_role

    for terminal, program, view, source in (
        (leaves_a, program_a, view_a, ROLE_SOURCES[0]),
        (leaves_b, program_b, view_b, ROLE_SOURCES[1]),
    ):
        controller = terminal[0].identity
        target = terminal[7].identity
        changed = canonical_surface(
            adult,
            view,
            {
                controller: marked(b"c", atom_for_leaf(view, controller)),
                target: marked(b"t", atom_for_leaf(view, target)),
            },
        )
        assert adult.observe_program_role_conditioned_contact(
            program, {controller: (COND,)}, tuple(changed), source, ROLE_BASE
        )

    held_controller = leaves_h[0].identity
    held_distractor = leaves_h[6].identity
    held_target = leaves_h[7].identity
    for leaf, condition, prefix in (
        (held_controller, COND, b"c"),
        (held_target, COND, b"t"),
        (held_distractor, OTHER, b"d"),
    ):
        atom = atom_for_leaf(view_h, leaf)
        units = marked(prefix, atom)
        for source in FORM_SOURCES:
            adult.language.observe_form(atom, (condition,), tuple(units), source + atom)

    return {
        "adult": adult,
        "groups": (donor_a, donor_b, held),
        "leafsets": (leaves_a, leaves_b, leaves_h),
        "programs": (program_a, program_b, program_h),
        "views": (view_a, view_b, view_h),
        "roles": (controller_role, distractor_role, target_role),
    }


def main():
    started = time.perf_counter()
    state = train()
    adult = state["adult"]
    donor_a, _donor_b, held = state["groups"]
    _leaves_a, _leaves_b, leaves_h = state["leafsets"]
    _program_a, _program_b, program_h = state["programs"]
    view_a, view_b, view_h = state["views"]
    controller_role, distractor_role, target_role = state["roles"]
    held_controller = leaves_h[0].identity
    held_distractor = leaves_h[6].identity
    held_target = leaves_h[7].identity

    checkpoint = copy.deepcopy(adult.checkpoint())
    restored = LanguageMasteryAdultV1.restore(copy.deepcopy(checkpoint))
    restored_view = restored.program_role_view(program_h, ROLE_BASE)
    output = restored.realize_program_role_conditioned(
        program_h, {held_controller: (COND,), held_distractor: (OTHER,)}, ROLE_BASE
    )
    expected = canonical_surface(
        restored,
        restored_view,
        {
            held_controller: marked(b"c", atom_for_leaf(restored_view, held_controller)),
            held_distractor: marked(b"d", atom_for_leaf(restored_view, held_distractor)),
            held_target: marked(b"t", atom_for_leaf(restored_view, held_target)),
        },
    )

    wrong_concepts = list(held)
    wrong_concepts[0], wrong_concepts[2] = wrong_concepts[2], wrong_concepts[0]
    wrong_leaves, wrong_program = build_program(restored, tuple(wrong_concepts), 2)
    wrong_view = restored.program_role_view(wrong_program, ROLE_BASE)
    held_atom = atom_for_leaf(restored_view, held_controller)
    distractor_atom = atom_for_leaf(restored_view, held_distractor)
    wrong_controller = next(
        leaf for leaf, atom in zip(wrong_view["leaves"], wrong_view["atoms"]) if atom == held_atom
    )
    wrong_distractor = next(
        leaf for leaf, atom in zip(wrong_view["leaves"], wrong_view["atoms"]) if atom == distractor_atom
    )
    wrong_output = restored.realize_program_role_conditioned(
        wrong_program, {wrong_controller: (COND,), wrong_distractor: (OTHER,)}, ROLE_BASE
    )

    lesion = LanguageMasteryAdultV1.restore(copy.deepcopy(checkpoint))
    lesion_view = lesion.program_role_view(program_h, ROLE_BASE)
    slots = {role: index for index, role in enumerate(lesion_view["roles"])}
    dep_key = (lesion_view["context"], slots[controller_role], slots[target_role])
    lesion.language._dependency_sources.pop(dep_key, None)
    lesion.language._rebuild_indices()
    lesion_output = lesion.realize_program_role_conditioned(
        program_h, {held_controller: (COND,), held_distractor: (OTHER,)}, ROLE_BASE
    )
    lesion.choose(AdultStateV1(pressure_q16=Q))
    pressure_output = lesion.realize_program_role_conditioned(
        program_h, {held_controller: (COND,), held_distractor: (OTHER,)}, ROLE_BASE
    )

    withdrawn = LanguageMasteryAdultV1.restore(copy.deepcopy(checkpoint))
    withdrawn.language.withdraw_source(ROLE_SOURCES[0])
    withdrawn_output = withdrawn.realize_program_role_conditioned(
        program_h, {held_controller: (COND,), held_distractor: (OTHER,)}, ROLE_BASE
    )
    reacquire_leaves, reacquire_program = build_program(withdrawn, donor_a, 0)
    reacquire_view = withdrawn.program_role_view(reacquire_program, ROLE_BASE)
    assert withdrawn.observe_program_role_construction(
        reacquire_program,
        tuple(canonical_surface(withdrawn, reacquire_view)),
        REACQUIRE_SOURCE,
        ROLE_BASE,
    )
    reacquire_controller = reacquire_leaves[0].identity
    reacquire_target = reacquire_leaves[7].identity
    changed = canonical_surface(
        withdrawn,
        reacquire_view,
        {
            reacquire_controller: marked(
                b"c", atom_for_leaf(reacquire_view, reacquire_controller)
            ),
            reacquire_target: marked(b"t", atom_for_leaf(reacquire_view, reacquire_target)),
        },
    )
    assert withdrawn.observe_program_role_conditioned_contact(
        reacquire_program,
        {reacquire_controller: (COND,)},
        tuple(changed),
        REACQUIRE_SOURCE,
        ROLE_BASE,
    )
    reacquired_output = withdrawn.realize_program_role_conditioned(
        program_h, {held_controller: (COND,), held_distractor: (OTHER,)}, ROLE_BASE
    )

    seeds = tuple(
        COND if role == controller_role else OTHER if role == distractor_role else 0
        for role in restored_view["roles"]
    )
    before = restored.language.complete_dependencies(restored_view["context"], seeds)
    touches_before = restored.language.last_lookup_touches
    for index in range(512):
        restored.language.observe_dependency(0x50000 + index, 0, 1, 0x60000 + index)
        restored.language.observe_dependency(0x50000 + index, 0, 1, 0x70000 + index)
    after = restored.language.complete_dependencies(restored_view["context"], seeds)
    touches_after = restored.language.last_lookup_touches

    duplicate_roles_refused = False
    try:
        dup_leaves = leaves(restored, held)
        w0, p0 = earn(restored, PAIR[0], dup_leaves[0], dup_leaves[0].identity, dup_leaves[1], dup_leaves[1].identity)
        w1, p1 = earn(restored, PAIR[0], dup_leaves[2], dup_leaves[2].identity, dup_leaves[3], dup_leaves[3].identity)
        w2, p2 = earn(restored, PAIR[2], dup_leaves[4], dup_leaves[4].identity, dup_leaves[5], dup_leaves[5].identity)
        w3, p3 = earn(restored, PAIR[3], dup_leaves[6], dup_leaves[6].identity, dup_leaves[7], dup_leaves[7].identity)
        m0, pm0 = earn(restored, HIGH[0], w0, p0, w1, p1)
        m1, pm1 = earn(restored, HIGH[1], w2, p2, w3, p3)
        _rw, rp = earn(restored, ROOT[0], m0, pm0, m1, pm1)
        restored.program_role_view(rp, ROLE_BASE)
    except HierarchicalRefuse as exc:
        duplicate_roles_refused = str(exc) == "adult:program_role_ambiguous"

    training_topologies = {
        topology
        for rows in adult.language._role_template_topologies.values()
        for _source, topology in rows
    }
    traversal = traversal_leaves(adult, program_h)
    controller_pos = traversal.index(held_controller)
    distractor_pos = traversal.index(held_distractor)
    target_pos = traversal.index(held_target)
    target_mark = marked(b"t", atom_for_leaf(restored_view, held_target))
    distractor_mark = marked(b"d", atom_for_leaf(restored_view, held_distractor))

    checks = {
        "heldout_remote_form_follows_structural_role_dependency": bytes(output or ()) == expected
        and target_mark in bytes(output or ()),
        "nearer_conflicting_distractor_does_not_control_target": abs(distractor_pos - target_pos)
        < abs(controller_pos - target_pos)
        and distractor_mark in bytes(output or ())
        and bytes(output or ()) == expected,
        "same_terminals_changed_local_ancestry_refuses_remote_target": wrong_output is not None
        and target_mark not in bytes(wrong_output),
        "role_support_uses_topology_diversity_not_heldout_tree": adult.language.role_context_supported(
            view_h["context"], len(view_h["atoms"])
        )
        and view_a["topology"] != view_b["topology"]
        and view_h["topology"] not in training_topologies,
        "focal_dependency_lesion_blocks_remote_form": lesion_output is not None
        and target_mark not in bytes(lesion_output),
        "pressure_cannot_fabricate_lesioned_structure": pressure_output == lesion_output,
        "source_withdrawal_drops_and_reacquisition_restores": withdrawn_output is None
        and reacquired_output is not None
        and bytes(reacquired_output) == expected,
        "checkpoint_rematerializes_roles_without_role_state": restored_view == view_h
        and all(
            key not in json.dumps(checkpoint)
            for key in ("program_role_view", "program_role_topology", "program_role_context")
        ),
        "productive_terminals_remain_factored_not_raw": adult.program_surface_checkpoint()[
            "raw_leaf_surfaces"
        ]
        == [],
        "local_dependency_lookup_survives_512_decoys": before == after
        and touches_before == touches_after == 1,
        "repeated_indistinguishable_local_roles_refuse": duplicate_roles_refused,
        "no_second_role_or_dependency_bank_on_adult": not any(
            hasattr(adult, name)
            for name in (
                "program_role_bank",
                "structural_dependency_bank",
                "syntax_tree",
                "grammar_rules",
                "controller_role",
            )
        ),
        "both_fast_unions_run_structural_migration_audit": (
            "adult-structural-migration:reference_mathematical_adult_structural_dependency_migration_verify.py"
            in (Path(__file__).parent / "run_language_mastery_fast.sh").read_text()
            and "reference_mathematical_adult_structural_dependency_migration_verify.py"
            in (Path(__file__).parent / "run_language_mastery_factory_fast.sh").read_text()
        ),
        "bounded_fast_path": time.perf_counter() - started < 1.0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    result = {
        "schema": "cyber-lagoon.reference-mathematical-adult-structural-dependency-migration.v1",
        "pass": not failed,
        "reference_only": True,
        "language_phenotype_improved": language_phenotype_improved,
        "visible_language_gain": "MATHEMATICAL_ADULT_REMOTE_FORM_FOLLOWS_CAUSAL_PROGRAM_ROLE_NOT_LINEAR_NEIGHBOR",
        "future_update_authority_preserved": future_update_authority_preserved,
        "role_context": restored_view["context"],
        "heldout_topology": restored_view["topology"],
        "positions": {
            "controller": controller_pos,
            "distractor": distractor_pos,
            "target": target_pos,
        },
        "checks": checks,
        "failed": failed,
        "remaining_red": [
            "MATHEMATICAL_ADULT_WORLD_BODY_ROLE_CONDITION_SEEDING",
            "MULTISOURCE_CONSENSUS_AND_DECEPTION",
            "DIRECT_STRUCTURAL_DEPENDENCY_PARITY",
        ],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
    }
    print(
        "FOUNDRY_MATHEMATICAL_ADULT_STRUCTURAL_DEPENDENCY_MIGRATION_"
        + ("GREEN" if not failed else "RED")
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()
