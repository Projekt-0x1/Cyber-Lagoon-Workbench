#!/usr/bin/env python3
"""Real-language falsifier for compact persistent Recipe / ephemeral Network state.

Only committed payload bytes reach the learner.  Package split metadata is used
observer-side to choose train/held-out ranges.  The higher language state may
persist compact Recipe mathematics, but current port bindings and Network state
must disappear after the current computation.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from language_development_v1 import LanguageDevelopmentV1
from reference_chunk_relation_induction_v1 import ChunkRelationInductionV1, ChunkRelationRefuse
from reference_organism_v2 import (
    CONTACT_AFFORDANCES, CONTACT_BODY_STATE, CONTACT_BODY_TARGET, CONTACT_CHANNEL_SAMPLE,
    CONTACT_EPISODE_BOUNDARY, CONTACT_MOTOR_CONSEQUENCE, CONTACT_WITHDRAW_SOURCE,
    CONTACT_WORLD_STATE,
    MotorActionV2, ReferenceOrganismV2,
)
from reference_population_v1 import (
    PopulationBankV1, PopulationRecruitmentEcologyV1, PopulationSpecV1,
    ResidentEventRecruitmentV1,
)

HARDWARE = Path(__file__).resolve().parents[2]
ROOT = HARDWARE / "data/organism_packages"
TIERS = (
    "language-residue-fastgate-v1",
    "language-residue-capability-v1",
    "language-residue-soak-v1",
)


def load(package):
    root = ROOT / package
    manifest = json.loads((root / "package.json").read_text())
    rows = [json.loads(line) for line in (root / "contacts.ndjson").read_text().splitlines() if line.strip()]
    payloads = [bytes.fromhex(row["payload_hex"]) for row in rows]
    return manifest, payloads


def extent(ranges):
    out = []
    for lo, hi in ranges:
        out.extend(range(int(lo), int(hi) + 1))
    return tuple(out)


def run(package):
    manifest, payloads = load(package)
    train_ids = extent(manifest["curriculum"]["training_ranges"])
    hlo, hhi = map(int, manifest["curriculum"]["heldout_range"])
    held_ids = tuple(range(hlo, hhi + 1))
    training = [payloads[i - 1] for i in train_ids]
    heldout = [payloads[i - 1] for i in held_ids]

    surface = LanguageDevelopmentV1()
    t0 = time.perf_counter()
    surface.consolidate_raw_stream(
        training, 7001, max_chunk=12, min_count=4, min_contexts=2, max_units=2048,
    )
    surface_ms = (time.perf_counter() - t0) * 1000

    ecology = ChunkRelationInductionV1(surface)
    for raw in training:
        ecology.ingest(raw)
    t0 = time.perf_counter()
    ecology.consolidate()
    recipe_ms = (time.perf_counter() - t0) * 1000

    before = [ecology.score(raw) for raw in heldout]
    packed_before = ecology.packed_state()
    state_text = json.dumps(ecology.persistent_state(), sort_keys=True, separators=(",", ":"))

    realized = 0
    attempted = 0
    for raw in heldout:
        occurrence = ecology.unfold(raw)
        if occurrence is None:
            continue
        try:
            bindings = ecology.current_bindings(raw, occurrence)
        except ChunkRelationRefuse:
            continue
        attempted += 1
        if ecology.realize(occurrence.recipe_identity, bindings) == raw:
            realized += 1

    released = ecology.compact_training_buffer()
    after = [ecology.score(raw) for raw in heldout]
    packed_after = ecology.packed_state()
    restored = ChunkRelationInductionV1.restore_packed(surface, packed_after)
    probe_ids = tuple(sorted({0, len(heldout) // 2, len(heldout) - 1}))
    restore_behavior = all(restored.score(heldout[i]) == after[i] for i in probe_ids)
    damaged = bytearray(packed_after); damaged[len(damaged) // 2] ^= 1
    try:
        ChunkRelationInductionV1.restore_packed(surface, bytes(damaged))
    except ChunkRelationRefuse:
        tamper_refused = True
    else:
        tamper_refused = False

    quantity = ecology.quantity()
    return {
        "package": package,
        "training_contacts": len(training),
        "heldout_contacts": len(heldout),
        "training_bytes": sum(map(len, training)),
        "recipes": len(ecology.recipes),
        "surface_units": len(surface.surface_units),
        "heldout_recruited": sum(row.recipe_identity != 0 for row in after),
        "heldout_recruit_fraction": sum(row.recipe_identity != 0 for row in after) / len(after),
        "heldout_binding_attempts": attempted,
        "heldout_exact_realizations": realized,
        "surface_ms": round(surface_ms, 3),
        "recipe_ms": round(recipe_ms, 3),
        "mean_candidate_touches": round(sum(row.candidate_touches for row in after) / len(after), 3),
        "max_candidate_touches": max(row.candidate_touches for row in after),
        "persistent_bytes_packed": quantity["persistent_bytes_packed"],
        "persistent_vs_raw_ratio": quantity["persistent_bytes_packed"] / sum(map(len, training)),
        "released_training_bytes": released,
        "hot_training_bytes_after": quantity["hot_training_bytes"],
        "post_compaction_behavior_exact": before == after,
        "post_compaction_state_exact": packed_before == packed_after,
        "packed_restore_behavior": restore_behavior,
        "packed_tamper_refused": tamper_refused,
        "transient_fields_persisted": any(token in state_text for token in (
            "anchor_occurrences", "port_intervals", "candidate_touches", "lattice_occurrences",
        )),
        "package_metadata_persisted": any(token in state_text for token in (
            "surface_id", "origin", "heldout", "partition_id", "payload_hex",
        )),
        "construction_root_nonzero": ecology.construction_root != bytes(32),
    }


def _packets(data: bytes, mode: int):
    sizes = (257,) if mode == 0 else (113, 509, 37, 1021, 211, 73)
    out = []
    cursor = index = 0
    while cursor < len(data):
        extent = sizes[index % len(sizes)]
        out.append(data[cursor:cursor + extent])
        cursor += extent
        index += 1
    return out


def _alice_build(data: bytes, mode: int):
    packets = _packets(data, mode)
    surface = LanguageDevelopmentV1()
    t0 = time.perf_counter()
    surface.consolidate_raw_stream(
        packets, 7001, max_chunk=12, min_count=4, min_contexts=2, max_units=2048,
    )
    surface_ms = (time.perf_counter() - t0) * 1000
    ecology = ChunkRelationInductionV1(surface)
    for packet in packets:
        ecology.ingest(packet)
    t0 = time.perf_counter()
    ecology.consolidate()
    recipe_ms = (time.perf_counter() - t0) * 1000
    surface_state = json.dumps(
        surface.raw_ecology_checkpoint(), sort_keys=True, separators=(",", ":"),
    )
    return surface, ecology, surface_state, surface_ms, recipe_ms


def _windows(data: bytes, width: int):
    return [data[i:i + width] for i in range(0, max(1, len(data) - width + 1), width)]


def _matched_shams(surface, ecology, count=3):
    """Preserve Recipe count, anchor marginals, supports, and guard matter."""
    rows = sorted(ecology.recipes.values(), key=lambda row: row.identity)
    shams = []
    for attempt in range(1, 4097):
        anchors_by_row = [list(row.anchors) for row in rows]
        rng = random.Random(0xC0A5E + attempt)
        for width in sorted({len(row.anchors) for row in rows}):
            indices = [index for index, row in enumerate(rows) if len(row.anchors) == width]
            for position in range(1, width):
                column = [rows[index].anchors[position] for index in indices]
                rng.shuffle(column)
                for index, value in zip(indices, column):
                    anchors_by_row[index][position] = value
        changed = sum(tuple(anchors) != row.anchors
                      for row, anchors in zip(rows, anchors_by_row))
        if changed < len(rows) // 2:
            continue
        sham = ChunkRelationInductionV1(surface)
        sham.contact_count = ecology.contact_count
        sham.contact_bytes = ecology.contact_bytes
        sham.construction_root = ecology.construction_root
        recipes = {}
        for row, anchors in zip(rows, anchors_by_row):
            anchors = tuple(anchors)
            identity = sham._identity(b"chunk-recipe-v3", [
                list(map(int, anchors)),
                [[guard.min_bytes, guard.max_bytes, guard.may_be_empty]
                 for guard in row.port_guards],
                [[guard.left_port, guard.right_port, guard.relation]
                 for guard in row.pair_guards],
            ])
            recipes[identity] = replace(row, identity=identity, anchors=anchors)
        if len(recipes) != len(rows):
            continue
        sham.recipes = recipes
        sham._build_active_surface()
        sham._rebuild_index()
        shams.append((sham, changed))
        if len(shams) == count:
            return shams
    raise RuntimeError("insufficient equal-matter Recipe permutations")


def _relation_specificity(surface, ecology, heldout):
    sham_rows = _matched_shams(surface, ecology)
    shams = [row[0] for row in sham_rows]
    sham_packed_sizes = [len(candidate.packed_state()) for candidate in shams]
    learned_packed_size = len(ecology.packed_state())
    rows = []
    for width in (16, 24, 32, 96):
        probes = _windows(heldout, width)

        def metrics(candidate):
            if width != 96:
                scores = [candidate.score(probe) for probe in probes]
                return (
                    sum(score.recipe_identity != 0 for score in scores) / len(scores),
                    sum(score.anchor_count >= 3 for score in scores) / len(scores),
                    sum(score.structure_score for score in scores) / len(scores),
                )
            recruited = higher_closures = structure = 0.0
            for probe in probes:
                occurrences = candidate.unfold_all(probe)
                recruited += bool(occurrences)
                higher_closures += sum(
                    candidate.recipes[row.recipe_identity].anchor_count >= 3
                    for row in occurrences
                )
                values = []
                for occurrence in occurrences:
                    recipe = candidate.recipes[occurrence.recipe_identity]
                    information = sum(
                        1 / candidate._anchor_support(uid) for uid in recipe.anchors
                    )
                    values.append(
                        recipe.anchor_count / max(1, occurrence.lattice_occurrences)
                        * math.log2(1 + recipe.support) * (1 + information)
                    )
                structure += max(values, default=0.0)
            return (
                recruited / len(probes),
                higher_closures / len(probes),
                structure / len(probes),
            )

        learned, learned_higher, learned_structure = metrics(ecology)
        sham_metrics = [metrics(candidate) for candidate in shams]
        sham = [row[0] for row in sham_metrics]
        sham_higher = [row[1] for row in sham_metrics]
        sham_structure = [row[2] for row in sham_metrics]
        rows.append({
            "aperture_bytes": width,
            "probe_count": len(probes),
            "learned_recruit_fraction": learned,
            "sham_min_recruit_fraction": min(sham),
            "sham_mean_recruit_fraction": sum(sham) / len(sham),
            "sham_max_recruit_fraction": max(sham),
            "learned_mean_higher_order_closures": learned_higher,
            "sham_min_mean_higher_order_closures": min(sham_higher),
            "sham_mean_mean_higher_order_closures": sum(sham_higher) / len(sham_higher),
            "sham_max_mean_higher_order_closures": max(sham_higher),
            "learned_mean_structure_score": learned_structure,
            "sham_min_mean_structure_score": min(sham_structure),
            "sham_mean_mean_structure_score": sum(sham_structure) / len(sham_structure),
            "sham_max_mean_structure_score": max(sham_structure),
            "matched_sham_count": len(shams),
            "sham_min_changed_recipes": min(row[1] for row in sham_rows),
            "recipe_count": len(ecology.recipes),
            "sham_packed_size_equal": all(
                size == learned_packed_size for size in sham_packed_sizes
            ),
        })
    return rows


def _run_alternate_slice(training, heldout, observer_source, observer_split):
    surface, ecology, _, surface_ms, recipe_ms = _alice_build(training, 0)
    return {
        "training_bytes": len(training),
        "heldout_bytes": len(heldout),
        "training_sha256": hashlib.sha256(training).hexdigest(),
        "heldout_sha256": hashlib.sha256(heldout).hexdigest(),
        "observer_source": observer_source,
        "observer_split": observer_split,
        "recipes": len(ecology.recipes),
        "persistent_bytes_packed": len(ecology.packed_state()),
        "surface_checkpoint_bytes": len(json.dumps(
            surface.raw_ecology_checkpoint(), sort_keys=True, separators=(",", ":"),
        )),
        "relation_specificity": _relation_specificity(surface, ecology, heldout),
        "surface_ms": round(surface_ms, 3),
        "recipe_ms": round(recipe_ms, 3),
    }


def _productive_core_falsifier(training, heldout_windows, ecology, minimum_cases=3):
    """Mix independently lived child Networks inside one learned parent Recipe.

    This is deliberately observer-side falsification, not a text decoder.  The
    candidate population is only current held-out Recipe occurrences.  A donor
    internal binding is admissible only when that byte span independently unfolds
    a learned child Recipe Network.  No tokenization, text scoring, expected output,
    or training/held-out target surface participates in candidate selection.
    """
    grouped = {}
    for window_index, raw_window in enumerate(heldout_windows):
        occurrence = ecology.unfold(raw_window)
        if occurrence is None:
            continue
        try:
            bindings = ecology.current_core_bindings(raw_window, occurrence)
        except ChunkRelationRefuse:
            continue
        grouped.setdefault(occurrence.recipe_identity, []).append(
            (window_index, raw_window, occurrence, bindings)
        )

    heldout_exact = set(heldout_windows)
    cases = []
    generated_seen = set()
    for recipe_identity, rows in sorted(grouped.items()):
        case = None
        for base_pos, (base_index, _base_raw, _base_occurrence, base_bindings) in enumerate(rows):
            for donor_index, _donor_raw, _donor_occurrence, donor_bindings in rows[base_pos + 1:]:
                for port in range(1, min(len(base_bindings), len(donor_bindings)) - 1):
                    donor_binding = donor_bindings[port]
                    if not donor_binding or donor_binding == base_bindings[port]:
                        continue
                    child = ecology.unfold(donor_binding)
                    if child is None:
                        continue
                    mixed = list(base_bindings)
                    mixed[port] = donor_binding
                    try:
                        generated = ecology.realize(recipe_identity, mixed)
                    except ChunkRelationRefuse:
                        continue
                    if (not generated or generated in generated_seen
                            or generated in training or generated in heldout_exact):
                        continue
                    reentered = ecology.unfold_all(generated)
                    if not any(row.recipe_identity == recipe_identity for row in reentered):
                        continue
                    generated_seen.add(generated)
                    case = {
                        "parent_recipe_identity": recipe_identity,
                        "child_recipe_identity": child.recipe_identity,
                        "internal_port": port,
                        "base_heldout_window": base_index,
                        "donor_heldout_window": donor_index,
                        "generated_bytes": len(generated),
                        "generated_hex": generated.hex(),
                        "generated_text": generated.decode("utf-8", errors="replace"),
                        "absent_from_training": generated not in training,
                        "not_exact_heldout_replay": generated not in heldout_exact,
                        "reentered_same_parent_recipe": True,
                    }
                    break
                if case is not None:
                    break
            if case is not None:
                break
        if case is not None:
            cases.append(case)
        if len(cases) >= minimum_cases:
            break

    return {
        "minimum_cases": minimum_cases,
        "cases": cases,
        "case_count": len(cases),
        "distinct_parent_recipes": len({row["parent_recipe_identity"] for row in cases}),
        "all_absent_from_training": bool(cases) and all(row["absent_from_training"] for row in cases),
        "all_not_exact_heldout_replay": bool(cases) and all(row["not_exact_heldout_replay"] for row in cases),
        "all_reentered_same_parent_recipe": bool(cases) and all(
            row["reentered_same_parent_recipe"] for row in cases
        ),
        "no_text_scorer_tokenizer_or_expected_output": True,
    }


def _shared_cognition_child_selection_falsifier(training, heldout_windows, ecology):
    """Let independent consequence, not surface quality, choose a child family.

    Parent/port/current sibling morphology forms one opaque current context cue.
    Candidate children are represented only by their learned Recipe morphology.
    Generic PopulationRecruitmentEcologyV1 then earns or suppresses a relation
    through an actual joint Network Occurrence plus independent consequence.
    The language layer only consumes the uniquely nominated child morphology.
    """
    grouped = {}
    for window_index, raw_window in enumerate(heldout_windows):
        occurrence = ecology.unfold(raw_window)
        if occurrence is None:
            continue
        try:
            bindings = ecology.current_core_bindings(raw_window, occurrence)
        except ChunkRelationRefuse:
            continue
        grouped.setdefault(occurrence.recipe_identity, []).append(
            (window_index, raw_window, occurrence, bindings)
        )

    heldout_exact = set(heldout_windows)
    for parent_recipe, rows in sorted(grouped.items()):
        for base_index, _base_raw, _base_occurrence, base_bindings in rows:
            sibling_children = []
            for span in base_bindings[1:-1]:
                child = ecology.unfold(span) if span else None
                sibling_children.append(0 if child is None else child.recipe_identity)
            for port in range(1, len(base_bindings) - 1):
                base_span = base_bindings[port]
                base_child = ecology.unfold(base_span) if base_span else None
                if base_child is None:
                    continue
                base_boundary = ecology.boundary_network_morphology_features(base_span)

                def boundary_sets(features):
                    left_count = int(features[0])
                    left = set(map(int, features[1:1 + left_count]))
                    cursor = 1 + left_count
                    right_count = int(features[cursor])
                    right = set(map(int, features[cursor + 1:cursor + 1 + right_count]))
                    return left, right

                base_left, base_right = boundary_sets(base_boundary)
                if not base_left or not base_right:
                    continue
                candidates = {}
                for donor_index, _donor_raw, _donor_occurrence, donor_bindings in rows:
                    if donor_index == base_index or port >= len(donor_bindings) - 1:
                        continue
                    donor_span = donor_bindings[port]
                    if not donor_span or donor_span == base_span:
                        continue
                    donor_child = ecology.unfold(donor_span)
                    if donor_child is None:
                        continue
                    try:
                        donor_geometry = ecology.recursive_network_morphology_features(
                            donor_span, donor_child, depth=2,
                        )
                    except ChunkRelationRefuse:
                        continue
                    donor_boundary = ecology.boundary_network_morphology_features(donor_span)
                    donor_left, donor_right = boundary_sets(donor_boundary)
                    boundary_compatible = bool(
                        base_left & donor_left and base_right & donor_right
                    )
                    mixed = list(base_bindings)
                    mixed[port] = donor_span
                    try:
                        generated = ecology.realize(parent_recipe, mixed)
                    except ChunkRelationRefuse:
                        continue
                    if (not generated or generated in training or generated in heldout_exact
                            or not any(row.recipe_identity == parent_recipe
                                       for row in ecology.unfold_all(generated))):
                        continue
                    candidates.setdefault(donor_geometry, []).append((
                        donor_index, donor_span, generated, donor_child.recipe_identity,
                        boundary_compatible, donor_boundary,
                    ))
                # Current boundary geometry is eligibility. Consequence may resolve
                # competition only among exact compatible current candidates; it may
                # never make an incompatible child speakable.
                positive_geometry = next((
                    geometry for geometry in sorted(candidates)
                    if sum(1 for row in candidates[geometry] if row[4]) == 1
                ), None)
                negative_geometry = next((
                    geometry for geometry in sorted(candidates)
                    if geometry != positive_geometry
                    and candidates[geometry]
                    and all(not row[4] for row in candidates[geometry])
                ), None)
                if positive_geometry is None or negative_geometry is None:
                    continue

                # Add actual non-language world/body Occurrences to the cue. Two
                # different current shared-cognition states must be able to reverse
                # the winner while the same language candidates remain available.
                shared = ReferenceOrganismV2(PopulationSpecV1(4096, 2, 4, 8, 16))
                language_context = (
                    0x4C435458,
                    parent_recipe & 0xFFFFFFFF,
                    (parent_recipe >> 32) & 0xFFFFFFFF,
                    port,
                    *tuple(value & 0xFFFFFFFF for value in sibling_children),
                    *tuple((value >> 32) & 0xFFFFFFFF for value in sibling_children),
                )

                def current_shared_context(world_value, body_value, source):
                    shared.contact(CONTACT_WORLD_STATE, (world_value,), source, True, True)
                    body_occurrence = shared.contact(
                        CONTACT_BODY_STATE, (body_value,), source + 1, True, True,
                    )
                    world_row = next(
                        row for row in shared.population.occurrences
                        if row.identity == shared.world_state_occurrence
                    )
                    body_row = next(
                        row for row in shared.population.occurrences
                        if row.identity == body_occurrence
                    )
                    world_morphology = PopulationRecruitmentEcologyV1.morphology_identity(
                        world_row.sites
                    )
                    body_morphology = PopulationRecruitmentEcologyV1.morphology_identity(
                        body_row.sites
                    )
                    return (
                        *language_context,
                        world_morphology & 0xFFFFFFFF,
                        (world_morphology >> 32) & 0xFFFFFFFF,
                        body_morphology & 0xFFFFFFFF,
                        (body_morphology >> 32) & 0xFFFFFFFF,
                    )

                context_a = current_shared_context(0xA501, 0xB501, 0xD501)
                context_b = current_shared_context(0xA502, 0xB502, 0xD502)
                bank = PopulationBankV1(PopulationSpecV1(4096, 2, 2, 8, 16))
                relations = PopulationRecruitmentEcologyV1()
                geometry_morphologies = {}
                relation_ids = {}
                for context, geometry, effect, source in (
                    (context_a, positive_geometry, 1, 0xCA551),
                    (context_a, negative_geometry, -1, 0xCA552),
                    (context_b, positive_geometry, -1, 0xCA553),
                    (context_b, negative_geometry, 1, 0xCA554),
                ):
                    cue = bank.recruit(context)
                    child_features = (0x43484C44, *geometry)
                    child = bank.recruit(child_features)
                    geometry_morphologies[geometry] = relations.morphology_identity(child.sites)
                    network = bank.recruit(relations.network_occurrence_features((cue, child)))
                    relation_ids[(context, geometry)] = relations.record_qualified_network(
                        bank, network, (cue, child), source, effect, True,
                    )

                fresh_a = bank.recruit(context_a)
                fresh_b = bank.recruit(context_b)
                nominated_a = relations.unfold_candidates(fresh_a)
                nominated_b = relations.unfold_candidates(fresh_b)
                if (nominated_a != (geometry_morphologies[positive_geometry],)
                        or nominated_b != (geometry_morphologies[negative_geometry],)):
                    continue

                # A yoked/non-independent return cannot create the same authority.
                yoked_bank = PopulationBankV1(PopulationSpecV1(4096, 2, 2, 8, 16))
                yoked_relations = PopulationRecruitmentEcologyV1()
                yoked_cue = yoked_bank.recruit(context_a)
                yoked_child = yoked_bank.recruit((0x43484C44, *positive_geometry))
                yoked_network = yoked_bank.recruit(
                    yoked_relations.network_occurrence_features((yoked_cue, yoked_child))
                )
                yoked_relation = yoked_relations.record_qualified_network(
                    yoked_bank, yoked_network, (yoked_cue, yoked_child),
                    0xCA551, 1, False,
                )
                yoked_fresh = yoked_bank.recruit(context_a)
                yoked_nomination = yoked_relations.unfold_candidates(yoked_fresh)

                (donor_index, _donor_span, generated, positive_family,
                 positive_boundary_compatible, positive_boundary) = next(
                    row for row in candidates[positive_geometry] if row[4]
                )
                (negative_index, _negative_span, negative_counterfactual, negative_family,
                 negative_boundary_compatible, negative_boundary) = candidates[negative_geometry][0]
                # Context B earns the incompatible morphology but eligibility remains
                # false, so the outward path must refuse rather than emit its bytes.
                reverse_boundary_refused = (
                    nominated_b == (geometry_morphologies[negative_geometry],)
                    and not negative_boundary_compatible
                )
                relations.withdraw_source(0xCA551)
                withdrawal_blocks = not relations.unfold_candidates(fresh_a)
                return {
                    "found": True,
                    "parent_recipe_identity": parent_recipe,
                    "internal_port": port,
                    "base_heldout_window": base_index,
                    "selected_donor_window": donor_index,
                    "positive_child_recipe": positive_family,
                    "negative_child_recipe": negative_family,
                    "positive_recursive_morphology": list(positive_geometry),
                    "negative_recursive_morphology": list(negative_geometry),
                    "positive_relation": relation_ids[(context_a, positive_geometry)],
                    "negative_relation": relation_ids[(context_a, negative_geometry)],
                    "reverse_positive_relation": relation_ids[(context_b, negative_geometry)],
                    "world_somatic_context_reverses_selection": True,
                    "unique_consequence_selected": True,
                    "yoked_relation": yoked_relation,
                    "yoked_nomination_count": len(yoked_nomination),
                    "withdrawal_blocks": withdrawal_blocks,
                    "generated_bytes": len(generated),
                    "generated_hex": generated.hex(),
                    "generated_text": generated.decode("utf-8", errors="replace"),
                    "positive_boundary_compatible": positive_boundary_compatible,
                    "positive_boundary_morphology": list(positive_boundary),
                    "reverse_selected_donor_window": negative_index,
                    "reverse_boundary_compatible": negative_boundary_compatible,
                    "reverse_boundary_morphology": list(negative_boundary),
                    "reverse_boundary_refused": reverse_boundary_refused,
                    "reverse_candidate_bytes_not_emitted": len(negative_counterfactual),
                    "absent_from_training": generated not in training,
                    "not_exact_heldout_replay": generated not in heldout_exact,
                    "reentered_same_parent_recipe": any(
                        row.recipe_identity == parent_recipe for row in ecology.unfold_all(generated)
                    ),
                    "surface_quality_used_for_selection": False,
                }
    return {"found": False}


def _multi_recipe_outward_closure_falsifier(training, heldout_windows, ecology):
    """Transfer earned authority from lived coactive Recipe closures to novel output.

    A closure is the exact set of learned Recipes simultaneously valid on one
    current surface.  Prior lived held-out cores supply closure occurrences; no
    output bytes enter persistent evidence.  A later novel rebinding may be emitted
    only when its closure was consequence-qualified and its current port boundary
    leaves exactly one eligible concrete binding.
    """
    heldout_exact = set(heldout_windows)
    rows = []
    lived = {}
    for window_index, raw_window in enumerate(heldout_windows):
        occurrence = ecology.unfold(raw_window)
        if occurrence is None:
            continue
        try:
            bindings = ecology.current_core_bindings(raw_window, occurrence)
            core = ecology.realize(occurrence.recipe_identity, bindings)
        except ChunkRelationRefuse:
            continue
        closure = ecology.current_network_closure_features(core)
        if closure:
            lived.setdefault(closure, []).append((window_index, core))
        rows.append((window_index, raw_window, occurrence, bindings))

    def boundary_sets(features):
        left_count = int(features[0]); left = set(map(int, features[1:1 + left_count]))
        cursor = 1 + left_count; right_count = int(features[cursor])
        right = set(map(int, features[cursor + 1:cursor + 1 + right_count]))
        return left, right

    grouped = {}
    for base_index, _base_raw, base_occurrence, base_bindings in rows:
        for port in range(1, len(base_bindings) - 1):
            base_span = base_bindings[port]
            if not base_span:
                continue
            base_left, base_right = boundary_sets(
                ecology.boundary_network_morphology_features(base_span)
            )
            if not base_left or not base_right:
                continue
            for donor_index, _donor_raw, donor_occurrence, donor_bindings in rows:
                if (donor_index == base_index
                        or donor_occurrence.recipe_identity != base_occurrence.recipe_identity
                        or port >= len(donor_bindings) - 1):
                    continue
                donor_span = donor_bindings[port]
                if not donor_span or donor_span == base_span:
                    continue
                donor_left, donor_right = boundary_sets(
                    ecology.boundary_network_morphology_features(donor_span)
                )
                if not (base_left & donor_left and base_right & donor_right):
                    continue
                mixed = list(base_bindings); mixed[port] = donor_span
                try:
                    generated = ecology.realize(base_occurrence.recipe_identity, mixed)
                except ChunkRelationRefuse:
                    continue
                if not generated or generated in training or generated in heldout_exact:
                    continue
                closure = ecology.current_network_closure_features(generated)
                lived_rows = lived.get(closure, ())
                if not closure or len(lived_rows) < 2:
                    continue
                key = (base_occurrence.recipe_identity, port, closure)
                grouped.setdefault(key, []).append((
                    base_index, donor_index, generated, lived_rows,
                ))

    # Exact current closure + port must resolve one concrete binding.  If several
    # byte bindings remain inside the same qualified closure, outward action refuses.
    candidates = [
        (key, values[0]) for key, values in grouped.items() if len(values) == 1
    ]
    candidates.sort(key=lambda row: (
        -int(row[0][2][1]), -len(row[1][3]), row[1][0], row[1][1], row[0][0], row[0][1],
    ))
    if len(candidates) < 2:
        return {"found": False, "qualified_unique_candidates": len(candidates)}

    positive_key, positive = candidates[0]
    negative_row = next((row for row in candidates[1:] if row[0][2] != positive_key[2]), None)
    if negative_row is None:
        return {"found": False, "qualified_unique_candidates": len(candidates)}
    negative_key, negative = negative_row
    positive_closure = positive_key[2]; negative_closure = negative_key[2]

    # Two independently lived examples of the same closure establish that the
    # persistent relation targets reusable closure morphology rather than one text.
    bank = PopulationBankV1(PopulationSpecV1(4096, 2, 2, 8, 16))
    relations = PopulationRecruitmentEcologyV1()
    context_features = (0x4F555443, positive_key[0] & 0xFFFFFFFF, positive_key[1])
    positive_sources = (0xE101, 0xE102)
    positive_relation = 0
    for source, _lived_row in zip(positive_sources, positive[3][:2]):
        cue = bank.recruit(context_features)
        closure_occurrence = bank.recruit((0x434C4F53, *positive_closure))
        network = bank.recruit(relations.network_occurrence_features((cue, closure_occurrence)))
        positive_relation = relations.record_qualified_network(
            bank, network, (cue, closure_occurrence), source, 1, True,
        )
    negative_cue = bank.recruit(context_features)
    negative_occurrence = bank.recruit((0x434C4F53, *negative_closure))
    negative_network = bank.recruit(
        relations.network_occurrence_features((negative_cue, negative_occurrence))
    )
    negative_relation = relations.record_qualified_network(
        bank, negative_network, (negative_cue, negative_occurrence), 0xE103, -1, True,
    )
    fresh_cue = bank.recruit(context_features)
    nominated = relations.unfold_candidates(fresh_cue)
    expected_morphology = relations.morphology_identity(
        bank.signature((0x434C4F53, *positive_closure))
    )

    yoked_bank = PopulationBankV1(PopulationSpecV1(4096, 2, 2, 8, 16))
    yoked_relations = PopulationRecruitmentEcologyV1()
    yc = yoked_bank.recruit(context_features)
    yo = yoked_bank.recruit((0x434C4F53, *positive_closure))
    yn = yoked_bank.recruit(yoked_relations.network_occurrence_features((yc, yo)))
    yoked_relation = yoked_relations.record_qualified_network(
        yoked_bank, yn, (yc, yo), 0xE101, 1, False,
    )

    ambiguous_refused = any(len(values) > 1 for values in grouped.values())
    generated = positive[2]
    return {
        "found": True,
        "coactive_recipes": int(positive_closure[1]),
        "lived_closure_support": len(positive[3]),
        "base_heldout_window": positive[0],
        "donor_heldout_window": positive[1],
        "parent_recipe_identity": positive_key[0],
        "internal_port": positive_key[1],
        "closure_features": list(positive_closure),
        "positive_relation": positive_relation,
        "negative_relation": negative_relation,
        "unique_closure_nominated": nominated == (expected_morphology,),
        "ambiguous_same_closure_refuses": ambiguous_refused,
        "yoked_relation": yoked_relation,
        "generated_bytes": len(generated),
        "generated_hex": generated.hex(),
        "generated_text": generated.decode("utf-8", errors="replace"),
        "absent_from_training": generated not in training,
        "not_exact_heldout_replay": generated not in heldout_exact,
        "closure_reentered": ecology.current_network_closure_features(generated) == positive_closure,
        "no_surface_quality_oracle": True,
    }


def run_alice():
    raw = (HARDWARE / "data/alice.txt").read_bytes()
    training = raw[:131072]
    heldout = raw[131072:]
    surface, ecology, surface_state, surface_ms, recipe_ms = _alice_build(training, 0)
    _, packetized, packetized_surface, packet_surface_ms, packet_recipe_ms = _alice_build(training, 1)
    packetization_invariant = (
        surface_state == packetized_surface
        and ecology.packed_state() == packetized.packed_state()
    )

    windows = _windows(heldout, 96)
    relation_specificity = _relation_specificity(surface, ecology, heldout)
    recruited = exact = attempts = touches = 0
    inconsistent_occurrence_refused = False
    outer_current_raw = None
    outer_current_occurrence = None
    coverage = []
    for raw_window in windows:
        score = ecology.score(raw_window)
        recruited += score.recipe_identity != 0
        touches += score.candidate_touches
        coverage.append(score.multi_byte_coverage)
        occurrence = ecology.unfold(raw_window)
        if occurrence is None:
            continue
        if outer_current_occurrence is None:
            outer_current_raw = raw_window
            outer_current_occurrence = occurrence
        if not inconsistent_occurrence_refused:
            try:
                ecology.current_bindings(raw_window,replace(
                    occurrence,identity=occurrence.identity+1))
            except ChunkRelationRefuse:
                inconsistent_occurrence_refused=True
        try:
            bindings = ecology.current_bindings(raw_window, occurrence)
            realized = ecology.realize(occurrence.recipe_identity, bindings)
        except ChunkRelationRefuse:
            continue
        attempts += 1
        exact += realized == raw_window

    productivity = _productive_core_falsifier(training, windows, ecology)
    shared_cognition_selection = _shared_cognition_child_selection_falsifier(
        training, windows, ecology,
    )
    multi_recipe_closure = _multi_recipe_outward_closure_falsifier(
        training, windows, ecology,
    )

    # A current language Network joins the generic causal event ecology by its
    # opaque persistent Recipe morphology.  No raw bytes or current bindings are
    # retained.  A changed later contact must unfold that Recipe again to cue it.
    outer_recipe_identity = 0 if outer_current_occurrence is None else outer_current_occurrence.recipe_identity
    outer_lived_windows = []
    for raw_window in windows:
        candidate = ecology.unfold(raw_window)
        if (candidate is not None and candidate.recipe_identity == outer_recipe_identity
                and raw_window not in (row[0] for row in outer_lived_windows)):
            outer_lived_windows.append((raw_window, candidate))
            if len(outer_lived_windows) == 3:
                break
    outer_current_features = ()
    outer_lived_features = []
    outer_tamper_refused = False
    if outer_current_occurrence is not None:
        outer_current_features = ecology.network_morphology_features(
            outer_current_raw, outer_current_occurrence,
        )
        try:
            ecology.network_morphology_features(
                outer_current_raw,
                replace(outer_current_occurrence, identity=outer_current_occurrence.identity + 1),
            )
        except ChunkRelationRefuse:
            outer_tamper_refused = True
    for raw_window, occurrence in outer_lived_windows:
        outer_lived_features.append(ecology.network_morphology_features(
            raw_window, occurrence,
        ))

    outer_source = 0xA11CE
    other_features = (0xB0, 0xB1, 0xB2)
    outer_goal = (0xD00D,)
    outer_affordances = (0xA701, 0xA702)

    def lived_event(organism, episode, features, source, affordances, independent):
        organism.contact(CONTACT_CHANNEL_SAMPLE,
                         (episode, 1, len(features), *features),
                         source, True, True)
        organism.contact(CONTACT_CHANNEL_SAMPLE,
                         (episode, 2, len(other_features), *other_features),
                         source, True, True)
        ticket = organism.contact(
            CONTACT_EPISODE_BOUNDARY, (episode,), source, True, True,
        )
        organism.contact(CONTACT_BODY_TARGET, outer_goal, source, True, True)
        organism.contact(CONTACT_AFFORDANCES, affordances, source, True, True)
        action = organism.tick()
        relations_before = len(organism.recruitment.relations)
        learned = organism.contact(
            CONTACT_MOTOR_CONSEQUENCE,
            (action.ticket, 1, len(outer_goal), *outer_goal),
            source, True, independent,
        )
        return ticket, action, relations_before, learned

    def cue_action(organism, episode, features, source, affordances):
        cue_identity = organism.contact(
            CONTACT_CHANNEL_SAMPLE,
            (episode, 1, len(features), *features), source, True, True,
        )
        cue = next(row for row in organism.population.occurrences
                   if row.identity == cue_identity)
        candidates = organism.recruitment.unfold_candidates(cue)
        organism.contact(CONTACT_BODY_TARGET, outer_goal, source, True, True)
        organism.contact(CONTACT_AFFORDANCES, affordances, source, True, True)
        return organism.tick(), candidates

    outer_organism = ReferenceOrganismV2(PopulationSpecV1(4096, 2, 4, 8, 16))
    outer_ticket, outer_action, outer_relations_before, outer_learned = lived_event(
        outer_organism, 1, outer_lived_features[0], outer_source,
        outer_affordances, True,
    )
    outer_relation = int(outer_learned.get("event_relation", 0))
    second_ticket, second_action, second_relations_before, second_learned = lived_event(
        outer_organism, 2, outer_lived_features[1], outer_source + 1,
        (outer_action.action_id,), True,
    )
    second_relation = int(second_learned.get("event_relation", 0))
    outer_edges = outer_organism.cognition.edges()
    outer_checkpoint = outer_organism.checkpoint()
    outer_probe_action, outer_candidates = cue_action(
        outer_organism, 3, outer_lived_features[2], outer_source + 2,
        tuple(reversed(outer_affordances)),
    )

    restored_outer = ReferenceOrganismV2.restore(
        json.loads(json.dumps(outer_checkpoint)),
    )
    restored_probe_action, restored_candidates = cue_action(
        restored_outer, 3, outer_lived_features[2], outer_source + 2,
        tuple(reversed(outer_affordances)),
    )

    yoked_organism = ReferenceOrganismV2(PopulationSpecV1(4096, 2, 4, 8, 16))
    yoked_ticket, yoked_action, yoked_relations_before, yoked_learned = lived_event(
        yoked_organism, 1, outer_lived_features[0], outer_source,
        outer_affordances, True,
    )
    yoked_second_ticket, yoked_second_action, _, yoked_second_learned = lived_event(
        yoked_organism, 2, outer_lived_features[1], outer_source + 1,
        (yoked_action.action_id,), False,
    )
    yoked_relation = int(yoked_second_learned.get("event_relation", 0))
    yoked_probe_action, yoked_candidates = cue_action(
        yoked_organism, 3, outer_lived_features[2], outer_source + 2,
        tuple(reversed(outer_affordances)),
    )
    outer_checkpoint_text = json.dumps(
        outer_checkpoint["recruitment"], sort_keys=True, separators=(",", ":"),
    )
    withdrawn_outer = ReferenceOrganismV2.restore(
        json.loads(json.dumps(outer_checkpoint)),
    )
    withdrawn_outer.contact(
        CONTACT_WITHDRAW_SOURCE, (outer_source + 1,), outer_source + 3, True, True,
    )
    withdrawn_probe_action, withdrawn_candidates = cue_action(
        withdrawn_outer, 3, outer_lived_features[2], outer_source + 2,
        tuple(reversed(outer_affordances)),
    )
    outer_relation_has_no_surface_payload = (
        heldout[:96].hex() not in outer_checkpoint_text.lower()
        and "payload_hex" not in outer_checkpoint_text
        and "alice" not in outer_checkpoint_text.lower()
    )

    # Two independently induced raw-language Recipes acquire opposed physical
    # action/outcome histories.  Every developmental encounter exposes both
    # affordances; only the resident tick selects the actual action.
    windows_by_recipe = {}
    for raw_window in windows:
        candidate = ecology.unfold(raw_window)
        if candidate is not None:
            windows_by_recipe.setdefault(candidate.recipe_identity, []).append(
                (raw_window, candidate)
            )
    competitive_recipe_ids = tuple(
        recipe_identity for recipe_identity, rows in sorted(
            windows_by_recipe.items(), key=lambda row: (-len(row[1]), row[0])
        ) if len(rows) >= 5
    )[:2]
    competitive_rows = tuple(
        tuple(windows_by_recipe[recipe_identity][:5])
        for recipe_identity in competitive_recipe_ids
    )
    competitive_features = tuple(tuple(
        ecology.network_morphology_features(raw_window, occurrence)
        for raw_window, occurrence in rows
    ) for rows in competitive_rows)
    competitive_other = ((0xC001, 0xC002, 0xC003), (0xD001, 0xD002, 0xD003))
    competitive_failure = (0xBAD,)
    competitive_targets = outer_affordances
    competitive_schedule = (0, 0, 0, 0, 1, 1, 1, 1)

    def train_competition(yoked_index=None):
        organism = ReferenceOrganismV2(PopulationSpecV1(4096, 2, 4, 8, 16))
        records = []
        per_recipe_index = [0, 0]
        for index, recipe_index in enumerate(competitive_schedule):
            row_index = per_recipe_index[recipe_index]
            per_recipe_index[recipe_index] += 1
            features = competitive_features[recipe_index][row_index]
            other = competitive_other[recipe_index]
            source = 0xC1000 + index
            episode = 100 + index
            organism.contact(
                CONTACT_CHANNEL_SAMPLE,
                (episode, 1, len(features), *features), source, True, True,
            )
            organism.contact(
                CONTACT_CHANNEL_SAMPLE,
                (episode, 2, len(other), *other), source, True, True,
            )
            ticket = organism.contact(
                CONTACT_EPISODE_BOUNDARY, (episode,), source, True, True,
            )
            organism.contact(CONTACT_BODY_TARGET, outer_goal, source, True, True)
            organism.contact(
                CONTACT_AFFORDANCES, outer_affordances, source, True, True,
            )
            action = organism.tick()
            beneficial = action.action_id == competitive_targets[recipe_index]
            next_state = outer_goal if beneficial else competitive_failure
            independent = index != yoked_index
            organism.contact(
                CONTACT_MOTOR_CONSEQUENCE,
                (action.ticket, 1 if beneficial else -1,
                 len(next_state), *next_state),
                source, True, independent,
            )
            records.append((recipe_index, source, ticket, action, beneficial,
                            independent))
        return organism, tuple(records)

    def probe_competition(checkpoint, order, withdrawn_sources=()):
        organism = ReferenceOrganismV2.restore(json.loads(json.dumps(checkpoint)))
        for source in withdrawn_sources:
            organism.contact(
                CONTACT_WITHDRAW_SOURCE, (source,), 0xC3000 + source, True, True,
            )
        actions = []
        recalls = []
        directives = []
        for probe_offset, recipe_index in enumerate(order):
            features = competitive_features[recipe_index][4]
            source = 0xC2000 + probe_offset
            cue_identity = organism.contact(
                CONTACT_CHANNEL_SAMPLE,
                (200 + probe_offset, 1, len(features), *features),
                source, True, True,
            )
            cue = next(row for row in organism.population.occurrences
                       if row.identity == cue_identity)
            directives.append(organism.recruitment.unfold_candidates(cue))
            recalls.append(organism.recruitment.unfold_candidates(
                cue, causal_recall=True,
            ))
            organism.contact(CONTACT_BODY_TARGET, outer_goal, source, True, True)
            organism.contact(
                CONTACT_AFFORDANCES, tuple(reversed(outer_affordances)),
                source, True, True,
            )
            action = organism.tick()
            actions.append((recipe_index, action))
            beneficial = action.action_id == competitive_targets[recipe_index]
            next_state = outer_goal if beneficial else competitive_failure
            organism.contact(
                CONTACT_MOTOR_CONSEQUENCE,
                (action.ticket, 1 if beneficial else -1,
                 len(next_state), *next_state),
                source, True, True,
            )
        return organism, tuple(actions), tuple(recalls), tuple(directives)

    competitive_organism, competitive_records = train_competition()
    competitive_checkpoint = competitive_organism.checkpoint()
    competitive_edges = competitive_organism.cognition.edges()
    competitive_forward = probe_competition(competitive_checkpoint, (0, 1))
    competitive_reverse = probe_competition(competitive_checkpoint, (1, 0))

    def probe_joint_competition(checkpoint, order, context_index=None,
                                checkpoint_ambiguity=False,
                                withdrawn_sources=()):
        organism = ReferenceOrganismV2.restore(json.loads(json.dumps(checkpoint)))
        for withdrawn in withdrawn_sources:
            organism.contact(
                CONTACT_WITHDRAW_SOURCE, (withdrawn,), 0xC3A00 + withdrawn,
                True, True,
            )
        source=0xC4000;episode=400
        organism.contact(CONTACT_BODY_TARGET,outer_goal,source,True,True)
        organism.contact(
            CONTACT_AFFORDANCES,tuple(reversed(outer_affordances)),source,True,True,
        )
        for recipe_index in order:
            features=competitive_features[recipe_index][4]
            organism.contact(
                CONTACT_CHANNEL_SAMPLE,
                (episode,1,len(features),*features),source,True,True,
            )
        unresolved_state=organism.world_state
        unresolved_need=organism.information_need
        unresolved_action=organism.tick()
        if checkpoint_ambiguity:
            organism=ReferenceOrganismV2.restore(json.loads(json.dumps(
                organism.checkpoint()
            )))
        if context_index is not None:
            context=(competitive_other[context_index]
                     if context_index in (0,1) else (0xE001,0xE002,0xE003))
            organism.contact(
                CONTACT_CHANNEL_SAMPLE,
                (episode,2,len(context),*context),source,True,True,
            )
        resolved_state=organism.world_state
        resolved_need=organism.information_need
        resolved_action=organism.tick()
        return {
            'unresolved_state':unresolved_state,
            'unresolved_need':unresolved_need,
            'unresolved_action':unresolved_action,
            'resolved_state':resolved_state,
            'resolved_need':resolved_need,
            'resolved_action':resolved_action,
            'checkpoint_bytes':len(json.dumps(
                organism.checkpoint(),sort_keys=True,separators=(',',':'),
            )),
        }

    competitive_joint_forward=probe_joint_competition(
        competitive_checkpoint,(0,1),
    )
    competitive_joint_reverse=probe_joint_competition(
        competitive_checkpoint,(1,0),
    )
    competitive_contextual=tuple(probe_joint_competition(
        competitive_checkpoint,(0,1),context_index,
    ) for context_index in (0,1))
    competitive_mismatched=probe_joint_competition(
        competitive_checkpoint,(0,1),2,
    )
    competitive_checkpointed_ambiguity=probe_joint_competition(
        competitive_checkpoint,(0,1),0,True,
    )
    positive_sources = tuple(tuple(
        source for kind, source, _ticket, _action, beneficial, independent
        in competitive_records
        if kind == recipe_index and beneficial and independent
    ) for recipe_index in (0, 1))
    competitive_withdrawn = tuple(
        probe_competition(
            competitive_checkpoint, (recipe_index,),
            positive_sources[recipe_index][:(2 if recipe_index==0 else 1)],
        ) for recipe_index in (0, 1)
    )
    competitive_evidence_biased=probe_joint_competition(
        competitive_checkpoint,(0,1),withdrawn_sources=(
            competitive_records[0][1],
        ),
    )
    competitive_yoked_organism, competitive_yoked_records = train_competition(
        yoked_index=len(competitive_schedule) - 1,
    )
    competitive_yoked_checkpoint = competitive_yoked_organism.checkpoint()
    competitive_yoked = probe_competition(competitive_yoked_checkpoint, (1,))
    competitive_relation_rows = tuple(
        competitive_organism.recruitment.relations.get(
            competitive_records[0 if recipe_index == 0 else 4][3].event_relation
        ) for recipe_index in (0, 1)
    )
    competitive_states = tuple(next(
        action.state_before for kind, _source, _ticket, action, _beneficial, _independent
        in competitive_records if kind == recipe_index
    ) for recipe_index in (0, 1))
    competitive_positive = tuple(
        competitive_organism.cognition.transition(
            competitive_states[recipe_index], competitive_targets[recipe_index], 2,
        ) for recipe_index in (0, 1)
    )
    competitive_negative = tuple(
        competitive_organism.cognition.transition(
            competitive_states[recipe_index], competitive_targets[1 - recipe_index], 1,
        ) for recipe_index in (0, 1)
    )
    yoked_state = next(
        action.state_before for kind, _source, _ticket, action, _beneficial, _independent
        in competitive_yoked_records if kind == 1
    )
    yoked_positive_one_source = competitive_yoked_organism.cognition.transition(
        yoked_state, competitive_targets[1], 1,
    )
    yoked_positive_authoritative = competitive_yoked_organism.cognition.transition(
        yoked_state, competitive_targets[1], 2,
    )
    competitive_checkpoint_text = json.dumps(
        competitive_checkpoint["recruitment"], sort_keys=True, separators=(",", ":"),
    )

    # Quantity control: all relations are earned through the actual event/action/
    # independent-reafference path.  Only the transient posting index is an
    # engineering lowering; canonical relation rows remain the checkpoint truth.
    scale_bank=PopulationBankV1(PopulationSpecV1(65536,2,4,16,16))
    scale_ecology=PopulationRecruitmentEcologyV1()
    scale_events=ResidentEventRecruitmentV1(scale_ecology,max_lag=8)
    scale_relations=[];scale_sources=[];scale_left=[]
    scale_started=time.perf_counter()
    for index in range(512):
        source=0xD1000+index;episode=1000+index
        left=(0x71000000+index*8+1,0x71000000+index*8+2)
        right=(0x71000000+index*8+5,0x71000000+index*8+6)
        scale_events.contact(scale_bank,episode,1,source,left)
        scale_events.contact(scale_bank,episode,2,source,right)
        closure=scale_events.close(scale_bank,episode,source)
        scale_events.issue_action(scale_bank,closure.ticket,0xA701)
        relation,_consequence=scale_events.reafference(
            scale_bank,closure.ticket,1,source,True,
        )
        scale_relations.append(relation);scale_sources.append(source);scale_left.append(left)
    scale_training_ms=(time.perf_counter()-scale_started)*1000
    scale_target_index=257;scale_source=0xD9000;scale_episode=9000
    scale_cue=scale_events.contact(
        scale_bank,scale_episode,1,scale_source,scale_left[scale_target_index],
    )
    scale_probe_started=time.perf_counter()
    scale_rows=scale_ecology.unfold_candidate_rows(scale_cue,causal_recall=True)
    scale_probe_us=(time.perf_counter()-scale_probe_started)*1_000_000
    scale_target_relation=scale_relations[scale_target_index]
    scale_checkpoint=scale_ecology.checkpoint()
    scale_checkpoint_text=json.dumps(scale_checkpoint,sort_keys=True,separators=(',',':'))
    scale_restored=PopulationRecruitmentEcologyV1.restore(json.loads(scale_checkpoint_text))
    scale_restored_rows=scale_restored.unfold_candidate_rows(scale_cue,causal_recall=True)
    scale_lesioned=PopulationRecruitmentEcologyV1.restore(json.loads(scale_checkpoint_text))
    scale_cue_morphology=scale_ecology.morphology_identity(scale_cue.sites)
    scale_lesioned.lesion_morphology(scale_cue_morphology)
    scale_lesioned_rows=scale_lesioned.unfold_candidate_rows(scale_cue,causal_recall=True)
    scale_withdrawn=PopulationRecruitmentEcologyV1.restore(json.loads(scale_checkpoint_text))
    scale_withdrawn.withdraw_source(scale_sources[scale_target_index])
    scale_withdrawn_rows=scale_withdrawn.unfold_candidate_rows(scale_cue,causal_recall=True)
    scale_index_tampered=PopulationRecruitmentEcologyV1.restore(json.loads(scale_checkpoint_text))
    scale_index_tampered._relations_by_morphology.setdefault(
        scale_cue_morphology,set(),
    ).add((1<<63)-1)
    scale_tampered_rows=scale_index_tampered.unfold_candidate_rows(
        scale_cue,causal_recall=True,
    )
    scale_index_cleared=PopulationRecruitmentEcologyV1.restore(json.loads(scale_checkpoint_text))
    scale_index_cleared._relations_by_morphology.clear()
    scale_cleared_rows=scale_index_cleared.unfold_candidate_rows(
        scale_cue,causal_recall=True,
    )
    scale_rebuilt=PopulationRecruitmentEcologyV1.restore(json.loads(scale_checkpoint_text))
    scale_rebuilt_rows=scale_rebuilt.unfold_candidate_rows(scale_cue,causal_recall=True)

    untrained=ChunkRelationInductionV1(surface)
    untrained_recruited=sum(untrained.score(raw_window).recipe_identity!=0
                            for raw_window in windows)
    reversed_recruited=sum(ecology.score(raw_window[::-1]).recipe_identity!=0
                           for raw_window in windows)

    packed = ecology.packed_state()
    released = ecology.compact_training_buffer()
    restored = ChunkRelationInductionV1.restore_packed(surface, packed)
    restore_probe = all(
        restored.score(windows[index]) == ecology.score(windows[index])
        for index in (0, len(windows) // 2, len(windows) - 1)
    )
    restore_networks = all(
        restored.unfold_all(windows[index]) == ecology.unfold_all(windows[index])
        for index in (0, len(windows) // 2, len(windows) - 1)
    )

    blocks = [training[i:i + 4096] for i in range(0, len(training), 4096)]
    scrambled = b"".join(reversed(blocks))
    scrambled_ecology = ChunkRelationInductionV1(surface)
    for packet in _packets(scrambled, 1):
        scrambled_ecology.ingest(packet)
    t0 = time.perf_counter()
    scrambled_ecology.consolidate()
    scramble_ms = (time.perf_counter() - t0) * 1000

    def recipe_rows(candidate):
        """Compare learned Recipe content without chronology telemetry."""
        return tuple(
            (
                row.identity,
                row.anchors,
                row.support,
                tuple((guard.min_bytes, guard.max_bytes, guard.may_be_empty)
                      for guard in row.port_guards),
                tuple((guard.left_port, guard.right_port, guard.relation)
                      for guard in row.pair_guards),
                row.min_bytes,
                row.max_bytes,
            )
            for row in sorted(candidate.recipes.values(), key=lambda value: value.identity)
        )

    return {
        "training_bytes": len(training),
        "heldout_bytes": len(heldout),
        "observer_source": "alice.txt",
        "observer_split": "TRAIN_[0,131072)/HELDOUT_[131072,EOF)",
        "heldout_windows": len(windows),
        "transport_packets_a": len(_packets(training, 0)),
        "transport_packets_b": len(_packets(training, 1)),
        "packetization_invariant": packetization_invariant,
        "recipes": len(ecology.recipes),
        "construction_windows": ecology.contact_count,
        "pair_candidates": ecology.pair_candidates_examined,
        "persistent_bytes_packed": len(packed),
        "surface_checkpoint_bytes": len(json.dumps(
            surface.raw_ecology_checkpoint(),sort_keys=True,separators=(",",":"))),
        "heldout_recruited": recruited,
        "heldout_recruit_fraction": recruited / len(windows),
        "heldout_binding_attempts": attempts,
        "heldout_exact_realizations": exact,
        "productive_core_generation": productivity,
        "shared_cognition_child_selection": shared_cognition_selection,
        "multi_recipe_outward_closure": multi_recipe_closure,
        "untrained_recruited":untrained_recruited,
        "reversed_recruited":reversed_recruited,
        "reversed_recruit_fraction":reversed_recruited/len(windows),
        "inconsistent_occurrence_refused":inconsistent_occurrence_refused,
        "outer_recipe_identity":outer_recipe_identity,
        "outer_recipe_was_current_winner":(
            outer_current_occurrence is not None
            and outer_recipe_identity != min(ecology.recipes, default=0)
        ),
        "outer_changed_raw_reinstated_same_recipe":(
            len(outer_lived_windows) == 3
            and len({raw_window for raw_window, _ in outer_lived_windows}) == 3
            and all(features == outer_current_features
                    for features in outer_lived_features)
        ),
        "outer_transient_tamper_refused":outer_tamper_refused,
        "outer_event_relation":outer_relation,
        "outer_second_event_relation":second_relation,
        "outer_event_candidates":len(outer_candidates),
        "outer_action_id":outer_action.action_id,
        "outer_action_was_resident_selected":(
            isinstance(outer_action, MotorActionV2)
            and outer_action.action_id in outer_affordances
            and outer_action.event_ticket == outer_ticket
            and outer_action.event_relation == outer_relation
            and outer_relations_before == 0
        ),
        "outer_second_episode_repeats_actual_action":(
            isinstance(second_action, MotorActionV2)
            and second_action.action_id == outer_action.action_id
            and second_action.event_ticket == second_ticket
            and second_action.event_relation == second_relation
            and second_relation == outer_relation
            and second_relations_before == 1
        ),
        "outer_authoritative_transition_support":(
            outer_edges[0].support if len(outer_edges) == 1 else 0
        ),
        "outer_learned_action_transfers":(
            isinstance(outer_probe_action, MotorActionV2)
            and outer_probe_action.action_id == outer_action.action_id
            and outer_probe_action.action_id in outer_affordances
            and outer_probe_action.event_ticket == 0
        ),
        "outer_independent_transition_evidence":len(outer_organism.cognition._evidence),
        "outer_yoked_transition_evidence":len(yoked_organism.cognition._evidence),
        "outer_yoked_relation":yoked_relation,
        "outer_yoked_counterfactual_action":(
            yoked_probe_action.action_id if isinstance(yoked_probe_action, MotorActionV2) else 0
        ),
        "outer_yoked_cannot_confirm_action":(
            isinstance(yoked_action, MotorActionV2)
            and isinstance(yoked_second_action, MotorActionV2)
            and isinstance(yoked_probe_action, MotorActionV2)
            and yoked_action.action_id == outer_action.action_id
            and yoked_second_action.action_id == outer_action.action_id
            and yoked_action.event_ticket == yoked_ticket
            and yoked_second_action.event_ticket == yoked_second_ticket
            and yoked_relations_before == 0
            and not yoked_organism.cognition.edges()
            and yoked_probe_action.action_id != outer_action.action_id
            and len(yoked_candidates) == 1
        ),
        "outer_checkpoint_replays_candidates":(
            restored_candidates == outer_candidates
            and isinstance(restored_probe_action, MotorActionV2)
            and restored_probe_action.action_id == outer_probe_action.action_id
        ),
        "outer_event_checkpoint_bytes":len(outer_checkpoint_text),
        "outer_organism_checkpoint_bytes":len(json.dumps(
            outer_checkpoint, sort_keys=True, separators=(",", ":"),
        )),
        "outer_resident_sites":outer_organism.population.spec.site_count,
        "outer_population_occurrences":len(outer_organism.population.occurrences),
        "outer_affordance_candidates":len(outer_affordances),
        "outer_event_withdrawal_blocks":(
            not withdrawn_outer.cognition.edges()
            and len(withdrawn_candidates) == 1
            and isinstance(withdrawn_probe_action, MotorActionV2)
            and withdrawn_probe_action.action_id == yoked_probe_action.action_id
            and withdrawn_probe_action.action_id != outer_action.action_id
        ),
        "outer_relation_has_no_surface_payload":outer_relation_has_no_surface_payload,
        "competitive_recipe_ids":competitive_recipe_ids,
        "competitive_raw_windows_distinct":(
            len(competitive_recipe_ids) == 2
            and competitive_recipe_ids[0] != competitive_recipe_ids[1]
            and all(len(rows) == len({raw_window for raw_window, _ in rows}) == 5
                    for rows in competitive_rows)
            and all(len(set(features)) == 1 for features in competitive_features)
            and competitive_features[0][0] != competitive_features[1][0]
        ),
        "competitive_training_actions":tuple(
            action.action_id for _kind, _source, _ticket, action, _beneficial, _independent
            in competitive_records
        ),
        "competitive_training_outcomes":tuple(
            1 if beneficial else -1
            for _kind, _source, _ticket, _action, beneficial, _independent
            in competitive_records
        ),
        "competitive_all_actions_resident_bound":all(
            isinstance(action, MotorActionV2)
            and action.action_id in outer_affordances
            and action.event_ticket == ticket
            and action.event_relation in action.state_before
            for _kind, _source, ticket, action, _beneficial, _independent
            in competitive_records
        ),
        "competitive_relation_ids":tuple(
            row.identity if row is not None else 0 for row in competitive_relation_rows
        ),
        "competitive_relation_signed_credit":tuple(
            row.credit if row is not None else 0 for row in competitive_relation_rows
        ),
        "competitive_relation_source_support":tuple(
            len(row.source_evidence) if row is not None else 0
            for row in competitive_relation_rows
        ),
        "competitive_positive_transition_support":tuple(
            row.support if row is not None else 0 for row in competitive_positive
        ),
        "competitive_negative_transition_support":tuple(
            row.support if row is not None else 0 for row in competitive_negative
        ),
        "competitive_forward_probe_actions":tuple(
            (recipe_index, action.action_id)
            for recipe_index, action in competitive_forward[1]
        ),
        "competitive_reverse_probe_actions":tuple(
            (recipe_index, action.action_id)
            for recipe_index, action in competitive_reverse[1]
        ),
        "competitive_forward_causal_candidates":tuple(
            len(rows) for rows in competitive_forward[2]
        ),
        "competitive_forward_directive_candidates":tuple(
            len(rows) for rows in competitive_forward[3]
        ),
        "competitive_withdrawn_probe_actions":tuple(
            row[1][0][1].action_id for row in competitive_withdrawn
        ),
        "competitive_withdrawn_causal_candidates":tuple(
            len(row[2][0]) for row in competitive_withdrawn
        ),
        "competitive_yoked_probe_action":competitive_yoked[1][0][1].action_id,
        "competitive_yoked_causal_candidates":len(competitive_yoked[2][0]),
        "competitive_yoked_positive_support":(
            yoked_positive_one_source.support
            if yoked_positive_one_source is not None else 0
        ),
        "competitive_yoked_authoritative_positive":(
            yoked_positive_authoritative is not None
        ),
        "competitive_joint_forward_alternatives":tuple(
            competitive_joint_forward['unresolved_need'][1:]
        ),
        "competitive_joint_reverse_alternatives":tuple(
            competitive_joint_reverse['unresolved_need'][1:]
        ),
        "competitive_joint_actions":(
            competitive_joint_forward['unresolved_action'].action_id
            if isinstance(competitive_joint_forward['unresolved_action'],MotorActionV2) else 0,
            competitive_joint_reverse['unresolved_action'].action_id
            if isinstance(competitive_joint_reverse['unresolved_action'],MotorActionV2) else 0,
        ),
        "competitive_contextual_relations":tuple(
            row['resolved_state'][-1] if row['resolved_state'] else 0
            for row in competitive_contextual
        ),
        "competitive_contextual_actions":tuple(
            row['resolved_action'].action_id
            if isinstance(row['resolved_action'],MotorActionV2) else 0
            for row in competitive_contextual
        ),
        "competitive_mismatched_unresolved":(
            competitive_mismatched['resolved_state'] is None
            and tuple(competitive_mismatched['resolved_need'][1:])
            == tuple(competitive_joint_forward['unresolved_need'][1:])
            and competitive_mismatched['resolved_action'] is None
        ),
        "competitive_ambiguity_checkpoint_replays":(
            competitive_checkpointed_ambiguity['resolved_state']
            == competitive_contextual[0]['resolved_state']
            and isinstance(competitive_checkpointed_ambiguity['resolved_action'],MotorActionV2)
            and isinstance(competitive_contextual[0]['resolved_action'],MotorActionV2)
            and competitive_checkpointed_ambiguity['resolved_action'].action_id
            == competitive_contextual[0]['resolved_action'].action_id
        ),
        "competitive_evidence_biased_relation":(
            competitive_evidence_biased['unresolved_state'][-1]
            if competitive_evidence_biased['unresolved_state'] else 0
        ),
        "competitive_evidence_biased_action":(
            competitive_evidence_biased['unresolved_action'].action_id
            if isinstance(competitive_evidence_biased['unresolved_action'],MotorActionV2)
            else 0
        ),
        "competitive_ambiguity_checkpoint_bytes":(
            competitive_checkpointed_ambiguity['checkpoint_bytes']
        ),
        "competitive_ambiguity_checkpoint_overhead":(
            competitive_checkpointed_ambiguity['checkpoint_bytes']
            - len(json.dumps(competitive_checkpoint,sort_keys=True,separators=(',',':')))
        ),
        "competitive_checkpoint_bytes":len(json.dumps(
            competitive_checkpoint, sort_keys=True, separators=(",", ":"),
        )),
        "competitive_population_occurrences":len(
            competitive_organism.population.occurrences
        ),
        "competitive_relation_checkpoint_bytes":len(competitive_checkpoint_text),
        "competitive_relation_has_no_surface_payload":all(
            raw_window.hex() not in competitive_checkpoint_text.lower()
            for rows in competitive_rows for raw_window, _ in rows
        ),
        "scale_relations":len(scale_ecology.relations),
        "scale_all_relations_independently_earned":(
            len(scale_relations)==len(set(scale_relations))==512
            and all(scale_relations)
            and all(scale_ecology.relations[identity].source_evidence==(source,)
                    for identity,source in zip(scale_relations,scale_sources))
        ),
        "scale_target_relation":scale_target_relation,
        "scale_probe_relation_ids":tuple(row[1] for row in scale_rows),
        "scale_probe_relation_touches":scale_ecology.last_touches,
        "scale_restore_relation_ids":tuple(row[1] for row in scale_restored_rows),
        "scale_restore_relation_touches":scale_restored.last_touches,
        "scale_lesion_excludes_target":all(
            row[1]!=scale_target_relation for row in scale_lesioned_rows
        ),
        "scale_withdrawal_excludes_target":all(
            row[1]!=scale_target_relation for row in scale_withdrawn_rows
        ),
        "scale_fake_posting_has_no_authority":(
            scale_tampered_rows==scale_restored_rows
        ),
        "scale_cleared_index_blocks_until_restore":(
            not scale_cleared_rows and scale_rebuilt_rows==scale_restored_rows
        ),
        "scale_posting_entries":sum(
            len(rows) for rows in scale_ecology._relations_by_morphology.values()
        ),
        "scale_index_not_checkpointed":(
            "relations_by_morphology" not in scale_checkpoint_text
            and "posting" not in scale_checkpoint_text
        ),
        "scale_relation_checkpoint_bytes":len(scale_checkpoint_text),
        "scale_population_occurrences":len(scale_bank.occurrences),
        "scale_training_ms":round(scale_training_ms,3),
        "scale_probe_us":round(scale_probe_us,3),
        "mean_candidate_touches": touches / len(windows),
        "mean_surface_coverage": sum(coverage) / len(coverage),
        "relation_specificity": relation_specificity,
        "construction_root_changes_with_chronology": (
            scrambled_ecology.construction_root != ecology.construction_root
        ),
        "block_reorder_changes_recipe_content": (
            recipe_rows(scrambled_ecology) != recipe_rows(ecology)
        ),
        "released_training_bytes": released,
        "restore_reconstructs_unfolding": restore_probe,
        "restore_reconstructs_coactive_networks": restore_networks,
        "surface_ms": round(surface_ms, 3),
        "recipe_ms": round(recipe_ms, 3),
        "packet_surface_ms": round(packet_surface_ms, 3),
        "packet_recipe_ms": round(packet_recipe_ms, 3),
        "scramble_recipe_ms": round(scramble_ms, 3),
    }


def _short_apertures_discriminate(row):
    short = [probe for probe in row["relation_specificity"]
             if probe["aperture_bytes"] in (16, 24, 32)]
    return len(short) == 3 and all(
        probe["matched_sham_count"] == 3
        and probe["sham_min_changed_recipes"] >= probe["recipe_count"] // 2
        and probe["sham_packed_size_equal"]
        and probe["learned_recruit_fraction"] > probe["sham_max_recruit_fraction"]
        for probe in short
    )


def _wide_aperture_discriminates(row):
    probes = [probe for probe in row["relation_specificity"]
              if probe["aperture_bytes"] == 96]
    return len(probes) == 1 and (
        probes[0]["matched_sham_count"] == 3
        and probes[0]["sham_min_changed_recipes"] >= probes[0]["recipe_count"] // 2
        and probes[0]["sham_packed_size_equal"]
        and probes[0]["learned_mean_higher_order_closures"]
        > probes[0]["sham_max_mean_higher_order_closures"]
    )


def _run_tiers():
    return [run(package) for package in TIERS]


def _run_shifted_alice():
    raw = (HARDWARE / "data/alice.txt").read_bytes()
    return _run_alternate_slice(
        raw[16384:147456], raw[:16384] + raw[147456:],
        "alice.txt", "TRAIN_[16384,147456)/HELDOUT_[0,16384)+[147456,EOF)",
    )


def _run_alternate_book():
    raw = (HARDWARE / "data/tinyshakespeare.txt").read_bytes()
    return _run_alternate_slice(
        raw[:131072], raw[131072:174314],
        "tinyshakespeare.txt", "TRAIN_[0,131072)/HELDOUT_[131072,174314)",
    )


def main():
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=4) as executor:
        tier_work = executor.submit(_run_tiers)
        alice_work = executor.submit(run_alice)
        shifted_work = executor.submit(_run_shifted_alice)
        alternate_work = executor.submit(_run_alternate_book)
        tiers = tier_work.result()
        alice = alice_work.result()
        shifted_alice = shifted_work.result()
        alternate_book = alternate_work.result()
    fast, capability, soak = tiers
    checks = {
        # Synthetic same-generator packages are economics/regression controls only;
        # whole-record heldout recruitment is deliberately not a capability law.
        "synthetic_exact_when_recruited": (
            fast["heldout_binding_attempts"] > 0
            and all(row["heldout_exact_realizations"] == row["heldout_binding_attempts"] for row in tiers)
        ),
        "construction_buffer_released": all(
            row["released_training_bytes"] == row["training_bytes"] and row["hot_training_bytes_after"] == 0
            for row in tiers
        ),
        "compaction_preserves_behavior": all(row["post_compaction_behavior_exact"] for row in tiers),
        "compaction_preserves_persistent_state": all(row["post_compaction_state_exact"] for row in tiers),
        "packed_restore_reconstructs_unfolding": all(row["packed_restore_behavior"] for row in tiers),
        "packed_tamper_refuses": all(row["packed_tamper_refused"] for row in tiers),
        "transient_network_not_persistent": all(not row["transient_fields_persisted"] for row in tiers),
        "package_metadata_not_persistent": all(not row["package_metadata_persisted"] for row in tiers),
        "construction_witness_root_retained": all(row["construction_root_nonzero"] for row in tiers),
        "soak_recipe_state_smaller_than_raw_experience": soak["persistent_bytes_packed"] < soak["training_bytes"],
        "current_work_sublinear_to_recipe_population": all(
            row["max_candidate_touches"] < max(256, row["recipes"] * 8) for row in tiers
        ),
        "capability_inner_loop_fast": capability["surface_ms"] + capability["recipe_ms"] < 1500,
        "soak_constructor_bounded": soak["surface_ms"] + soak["recipe_ms"] < 12000,
        "alice_packetization_invariant": alice["packetization_invariant"],
        "alice_nontoy_heldout_recruitment": alice["heldout_recruit_fraction"] >= 0.75,
        "alice_short_apertures_beat_equal_matter_shams": _short_apertures_discriminate(alice),
        "shifted_split_short_apertures_beat_equal_matter_shams": (
            _short_apertures_discriminate(shifted_alice)
        ),
        "alternate_book_short_apertures_beat_equal_matter_shams": (
            _short_apertures_discriminate(alternate_book)
        ),
        "alice_wide_aperture_beats_equal_matter_shams": _wide_aperture_discriminates(alice),
        "shifted_split_wide_aperture_beats_equal_matter_shams": (
            _wide_aperture_discriminates(shifted_alice)
        ),
        "alternate_book_wide_aperture_beats_equal_matter_shams": (
            _wide_aperture_discriminates(alternate_book)
        ),
        "alternate_slices_keep_compact_recipe_state": (
            shifted_alice["persistent_bytes_packed"] < 4096
            and alternate_book["persistent_bytes_packed"] < 4096
        ),
        "alice_exact_ephemeral_unfolding": (
            alice["heldout_binding_attempts"] > 0
            and alice["heldout_exact_realizations"] == alice["heldout_binding_attempts"]
        ),
        "alice_productive_core_rebinding": (
            alice["productive_core_generation"]["case_count"]
            >= alice["productive_core_generation"]["minimum_cases"]
            and alice["productive_core_generation"]["distinct_parent_recipes"]
            >= alice["productive_core_generation"]["minimum_cases"]
            and alice["productive_core_generation"]["all_absent_from_training"]
            and alice["productive_core_generation"]["all_not_exact_heldout_replay"]
            and alice["productive_core_generation"]["all_reentered_same_parent_recipe"]
            and alice["productive_core_generation"]["no_text_scorer_tokenizer_or_expected_output"]
        ),
        "alice_shared_cognition_selects_child_after_independent_consequence": (
            alice["shared_cognition_child_selection"].get("found", False)
            and alice["shared_cognition_child_selection"].get("unique_consequence_selected", False)
            and alice["shared_cognition_child_selection"].get(
                "world_somatic_context_reverses_selection", False
            )
            and alice["shared_cognition_child_selection"].get("yoked_relation", 1) == 0
            and alice["shared_cognition_child_selection"].get("yoked_nomination_count", 1) == 0
            and alice["shared_cognition_child_selection"].get("withdrawal_blocks", False)
            and alice["shared_cognition_child_selection"].get("absent_from_training", False)
            and alice["shared_cognition_child_selection"].get("not_exact_heldout_replay", False)
            and alice["shared_cognition_child_selection"].get("reentered_same_parent_recipe", False)
            and alice["shared_cognition_child_selection"].get("positive_boundary_compatible", False)
            and not alice["shared_cognition_child_selection"].get("reverse_boundary_compatible", True)
            and alice["shared_cognition_child_selection"].get("reverse_boundary_refused", False)
            and not alice["shared_cognition_child_selection"].get("surface_quality_used_for_selection", True)
        ),
        "alice_multi_recipe_outward_closure_transfers_consequence": (
            alice["multi_recipe_outward_closure"].get("found", False)
            and alice["multi_recipe_outward_closure"].get("coactive_recipes", 0) >= 2
            and alice["multi_recipe_outward_closure"].get("lived_closure_support", 0) >= 2
            and alice["multi_recipe_outward_closure"].get("positive_relation", 0) != 0
            and alice["multi_recipe_outward_closure"].get("unique_closure_nominated", False)
            and alice["multi_recipe_outward_closure"].get("ambiguous_same_closure_refuses", False)
            and alice["multi_recipe_outward_closure"].get("yoked_relation", 1) == 0
            and alice["multi_recipe_outward_closure"].get("absent_from_training", False)
            and alice["multi_recipe_outward_closure"].get("not_exact_heldout_replay", False)
            and alice["multi_recipe_outward_closure"].get("closure_reentered", False)
            and alice["multi_recipe_outward_closure"].get("no_surface_quality_oracle", False)
        ),
        "alice_no_training_cannot_recruit":alice["untrained_recruited"]==0,
        "alice_order_perturbation_reduces_recruitment":(
            alice["reversed_recruit_fraction"]
            < alice["heldout_recruit_fraction"]*.5),
        "alice_inconsistent_transient_occurrence_refused":alice["inconsistent_occurrence_refused"],
        "alice_current_recipe_network_enters_causal_event_ecology":(
            alice["outer_recipe_identity"] != 0
            and alice["outer_recipe_was_current_winner"]
            and alice["outer_event_relation"] != 0
            and alice["outer_event_candidates"] == 1
        ),
        "alice_changed_raw_reinstates_recipe_morphology":(
            alice["outer_changed_raw_reinstated_same_recipe"]
        ),
        "alice_transient_recipe_network_tamper_refused":(
            alice["outer_transient_tamper_refused"]
        ),
        "alice_yoked_return_cannot_create_outer_relation":(
            alice["outer_yoked_relation"] == 0
            and alice["outer_yoked_transition_evidence"] == 1
            and alice["outer_yoked_cannot_confirm_action"]
        ),
        "alice_two_sources_transfer_resident_event_action":(
            alice["outer_action_was_resident_selected"]
            and alice["outer_second_episode_repeats_actual_action"]
            and alice["outer_authoritative_transition_support"] == 2
            and alice["outer_learned_action_transfers"]
            and alice["outer_independent_transition_evidence"] == 1
        ),
        "alice_outer_checkpoint_replays_candidates":(
            alice["outer_checkpoint_replays_candidates"]
        ),
        "alice_outer_relation_withdrawal_blocks":alice["outer_event_withdrawal_blocks"],
        "alice_outer_relation_stores_no_surface_payload":alice["outer_relation_has_no_surface_payload"],
        "alice_two_recipe_populations_are_raw_and_distinct":(
            alice["competitive_raw_windows_distinct"]
            and len(alice["competitive_recipe_ids"]) == 2
            and all(alice["competitive_relation_ids"])
            and len(set(alice["competitive_relation_ids"])) == 2
        ),
        "alice_opposed_action_histories_are_resident_and_causal":(
            alice["competitive_all_actions_resident_bound"]
            and alice["competitive_training_outcomes"]
            == (1, -1, 1, 1, -1, 1, -1, 1)
            and alice["competitive_positive_transition_support"] == (3, 2)
            and alice["competitive_negative_transition_support"] == (1, 2)
            and alice["competitive_relation_signed_credit"] == (2, 0)
            and alice["competitive_relation_source_support"] == (4, 4)
        ),
        "alice_changed_recipe_cues_select_opposed_actions_order_invariantly":(
            alice["competitive_forward_probe_actions"]
            == ((0, 0xA701), (1, 0xA702))
            and alice["competitive_reverse_probe_actions"]
            == ((1, 0xA702), (0, 0xA701))
            and alice["competitive_forward_causal_candidates"] == (1, 1)
        ),
        "alice_negative_relation_is_recall_not_directive":(
            alice["competitive_forward_directive_candidates"] == (1, 0)
            and alice["competitive_forward_causal_candidates"] == (1, 1)
        ),
        "alice_load_bearing_source_withdrawal_restores_counterfactual":(
            alice["competitive_withdrawn_probe_actions"] == (0xA702, 0xA701)
            and alice["competitive_withdrawn_causal_candidates"] == (1, 1)
        ),
        "alice_yoked_return_cannot_confirm_competing_action":(
            alice["competitive_yoked_probe_action"] == 0xA701
            and alice["competitive_yoked_causal_candidates"] == 1
            and alice["competitive_yoked_positive_support"] == 1
            and not alice["competitive_yoked_authoritative_positive"]
        ),
        "alice_same_episode_relation_competition_is_order_invariant":(
            alice["competitive_joint_forward_alternatives"]
            == alice["competitive_joint_reverse_alternatives"]
            == tuple(sorted(alice["competitive_relation_ids"]))
            and alice["competitive_joint_actions"] == (0,0)
        ),
        "alice_nonlinguistic_context_resolves_opposed_relations":(
            alice["competitive_contextual_relations"]
            == alice["competitive_relation_ids"]
            and alice["competitive_contextual_actions"] == (0xA701,0xA702)
            and alice["competitive_mismatched_unresolved"]
        ),
        "alice_ambiguity_survives_checkpoint_then_resolves":(
            alice["competitive_ambiguity_checkpoint_replays"]
        ),
        "alice_active_source_support_is_load_bearing_in_competition":(
            alice["competitive_evidence_biased_relation"]
            == alice["competitive_relation_ids"][1]
            and alice["competitive_evidence_biased_action"] == 0xA702
        ),
        "alice_quantity_relations_are_actual_and_consequence_earned":(
            alice["scale_relations"] == 512
            and alice["scale_all_relations_independently_earned"]
        ),
        "alice_quantity_recall_touches_only_posted_candidates":(
            alice["scale_probe_relation_ids"] == (alice["scale_target_relation"],)
            and alice["scale_probe_relation_touches"] <= 4
            and alice["scale_probe_relation_touches"] * 100
            < alice["scale_relations"]
        ),
        "alice_quantity_index_rebuilds_exactly_from_checkpoint":(
            alice["scale_restore_relation_ids"]
            == alice["scale_probe_relation_ids"]
            and alice["scale_restore_relation_touches"]
            == alice["scale_probe_relation_touches"]
            and alice["scale_cleared_index_blocks_until_restore"]
        ),
        "alice_quantity_index_is_non_authoritative_lowering":(
            alice["scale_index_not_checkpointed"]
            and alice["scale_fake_posting_has_no_authority"]
            and alice["scale_posting_entries"] == 2*alice["scale_relations"]
        ),
        "alice_quantity_lesion_and_withdrawal_remain_load_bearing":(
            alice["scale_lesion_excludes_target"]
            and alice["scale_withdrawal_excludes_target"]
        ),
        "alice_competing_relation_state_is_compact_and_surface_free":(
            alice["competitive_relation_has_no_surface_payload"]
            and alice["competitive_relation_checkpoint_bytes"] < 2048
            and alice["competitive_ambiguity_checkpoint_overhead"] < 8192
        ),
        "alice_sparse_recipe_alignment_work": alice["mean_candidate_touches"] < 32,
        "alice_compact_recipe_state": alice["persistent_bytes_packed"] < 4096,
        "alice_training_buffer_released": alice["released_training_bytes"] == alice["training_bytes"],
        "alice_restore_reconstructs_unfolding": alice["restore_reconstructs_unfolding"],
        "alice_restore_reconstructs_coactive_networks": (
            alice["restore_reconstructs_coactive_networks"]
        ),
        "alice_block_reorder_changes_recipe_content": (
            alice["construction_root_changes_with_chronology"]
            and alice["block_reorder_changes_recipe_content"]
        ),
    }
    elapsed_ms = (time.perf_counter() - started) * 1000
    checks["named_runner_under_60_seconds"] = elapsed_ms < 60_000
    report = {
        "schema": "agi.ephemeral-language-recipe-reference.v8",
        "pass": all(checks.values()),
        "checks": checks,
        "tiers": tiers,
        "alice": alice,
        "shifted_alice": shifted_alice,
        "alternate_book": alternate_book,
        "elapsed_ms": round(elapsed_ms, 3),
        "persistent_law": "compact recipe/recruitment mathematics",
        "transient_law": "current bound occurrences + ephemeral network computation",
        "runtime_llm": False,
        "host_tokenizer": False,
        "host_semantic_features": False,
        "outer_bridge_authority":"CURRENT_RECIPE_NETWORK_PLUS_SIGNED_CAUSAL_RECALL_PLUS_TWO_SOURCE_RESIDENT_ACTION_SELECTION/REFERENCE_ONLY",
        "physical_direct_parity": "NOT_RUN/RED",
        "roundtrip_status":"INPUT_BOUND_BINDING_RECONSTRUCTION/NOT_GENERATION",
        "packed_size_scope":"HIGHER_RECIPE_STATE_ONLY/LOWER_SURFACE_REPORTED_SEPARATELY",
        "candidate_touch_scope":"RECIPE_ALIGNMENTS_AFTER_BITSET_NOMINATION/EXCLUDES_LATTICE_AND_BYTE_SCANS",
        "relation_control_status":"THREE_POSITION_MARGINAL_PRESERVING_DETERMINISTIC_COUNTERFACTUALS/NOT_STATISTICAL_GENERALIZATION",
        "aperture_selection_status":"AUTHORED_DURING_DEVELOPMENT/NOT_BLIND",
        "claim": "REFERENCE_REAL_BYTE_CONTEXTUAL_RELATION_COMPETITION_PLUS_SPARSE_512_RELATION_CAUSAL_QUANTITY/NOT_COMPREHENSION_GENERATION_OR_LANGUAGE_MASTERY",
        "remaining_red":["CANONICAL_BODY_ROUTE_AND_SOURCE_PROVENANCE",
                         "CANONICAL_DIRECT_RESIDENT_OUTER_BINDING_ACTION_AND_CONSEQUENCE",
                         "AUTHENTICATED_PHYSICAL_CONTACT_AND_CAUSAL_CREDIT",
                         "SHARED_OBJECT_OR_CAUSE_IDENTIFICATION",
                         "WIDE_APERTURE_PRODUCTIVE_TRANSFER_BEYOND_AUTHORED_ENGLISH_SPLITS",
                         "MULTILINGUAL_AND_CROSS_LANGUAGE_TRANSFER",
                         "BLIND_CORPUS_AND_WINDOW_SELECTION",
                         "TOTAL_PERSISTENT_STATE_BUDGET","DIRECT_TRANSLATION"],
    }
    print(
        "FOUNDRY_EPHEMERAL_LANGUAGE_RECIPE " + ("CONTRACT" if report["pass"] else "RED")
        + f" alice_train_bytes={alice['training_bytes']} alice_heldout_bytes={alice['heldout_bytes']}"
        + f" alice_recipes={alice['recipes']} alice_recipe_bytes={alice['persistent_bytes_packed']}"
        + f" alice_recruit={alice['heldout_recruit_fraction']:.4f}"
        + f" packetization_invariant={int(alice['packetization_invariant'])}"
        + " tokenizer=0 direct_parity=RED"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
