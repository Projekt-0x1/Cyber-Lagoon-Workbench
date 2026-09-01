#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from reference_organism_v2 import CONTACT_CONSEQUENCE, CONTACT_SCENE, ActionV2, ReferenceOrganismV2
from reference_population_v1 import PopulationSpecV1
from reference_raw_surface_recipe_critic_v1 import RawSurfaceRecipeCriticV1
import reference_ephemeral_language_recipe_verify as alice_recipe
import reference_organism_register_verify as register


ALICE_PATH = Path(__file__).resolve().parents[2] / "data" / "alice.txt"
HELDOUT_SETS = (
    ((113, 213, 313, 413), ("senior", "auditor", "checks", "record")),
    ((114, 214, 314, 414), ("careful", "planner", "revises", "procedure")),
    ((115, 215, 315, 415), ("curious", "scientist", "reviews", "result")),
    ((116, 216, 316, 416), ("patient", "operator", "monitors", "system")),
    ((117, 217, 317, 417), ("precise", "analyst", "tests", "sensor")),
    ((118, 218, 318, 418), ("quiet", "engineer", "inspects", "valve")),
    ((119, 219, 319, 419), ("rapid", "inspector", "examines", "device")),
    ((120, 220, 320, 420), ("methodical", "researcher", "verifies", "report")),
)
HELDOUT_ATOMS, HELDOUT_WORDS = HELDOUT_SETS[0]
FORMAL = b"the senior auditor checks the record."
TERSE = b"senior auditor: checks record."


def add_heldout_words(o, atoms=HELDOUT_ATOMS, words=HELDOUT_WORDS, source_base=610000):
    for index, (atom, word) in enumerate(zip(atoms, words)):
        for witness in range(2):
            source = int(source_base) + index * 10 + witness
            o.contact(CONTACT_SCENE, (7, 100, 1, atom), source, True, True)
            register.surf(o, word, source + 1000)


def prepare_ambiguous_scene():
    o = register.train(ReferenceOrganismV2(PopulationSpecV1(32768, 2, 4, 42, 8)))
    add_heldout_words(o)
    register.partner(o, register.P_NEW)
    scene_id = o.contact(
        CONTACT_SCENE, (7, register.CTX, len(HELDOUT_ATOMS), *HELDOUT_ATOMS),
        700000, True, True,
    )
    return o, scene_id


def build_alice_ecology():
    raw = ALICE_PATH.read_bytes()
    surface, ecology, _state, surface_ms, recipe_ms = alice_recipe._alice_build(
        raw[:131072], 0,
    )
    ecology.compact_training_buffer()
    return surface, ecology, surface_ms, recipe_ms


def main():
    started = time.perf_counter()
    checks = {}

    organism, scene_id = prepare_ambiguous_scene()
    # The resident surface ecology alone has two equally supported realizations.
    checks["resident_surface_competition_is_ambiguous"] = organism.tick() is None
    before_digest = organism.digest()
    before_templates = tuple(
        row.identity for row in organism.language.template_candidates(register.CTX, 4)
    )
    before_lexemes = tuple(
        tuple((support, bytes(units), tuple(sources))
              for support, units, sources in organism._lexeme_rows(atom))
        for atom in HELDOUT_ATOMS
    )

    _surface, alice, surface_ms, recipe_ms = build_alice_ecology()
    critic = RawSurfaceRecipeCriticV1(alice)
    scene = organism._scene_by_id[scene_id]
    candidates = critic.current_candidates(organism, scene)
    candidate_bytes = tuple(row.surface for row in candidates)
    checks["complete_current_candidate_set_has_two_surfaces"] = (
        len(candidates) == 2 and set(candidate_bytes) == {FORMAL, TERSE}
    )
    checks["developmental_support_exactly_tied"] = (
        len({row.developmental_support for row in candidates}) == 1
    )

    selected = critic.select(organism, scene)
    checks["alice_raw_recipes_select_existing_formal_surface"] = (
        selected is not None
        and selected.candidate.surface == FORMAL
        and selected.candidate.surface in candidate_bytes
        and selected.higher_closures >= 1
        and selected.rank_mass > 0
        and selected.alternatives == 2
    )
    formal_occurrences = alice.unfold_all(FORMAL)
    terse_occurrences = alice.unfold_all(TERSE)
    checks["raw_structure_discriminator_is_measured"] = (
        len(formal_occurrences) > len(terse_occurrences)
        and len(formal_occurrences) >= 1
        and len(terse_occurrences) == 0
    )

    # Selection is a read-only structural pressure. It does not create an Action,
    # learn a lexical item/template, or revise the resident selection Network.
    checks["critic_does_not_mutate_resident"] = (
        organism.digest() == before_digest
        and not organism.actions
        and tuple(row.identity for row in organism.language.template_candidates(register.CTX, 4)) == before_templates
        and tuple(
            tuple((support, bytes(units), tuple(sources))
                  for support, units, sources in organism._lexeme_rows(atom))
            for atom in HELDOUT_ATOMS
        ) == before_lexemes
        and not organism.selection_configuration_revisions
    )

    # The critic lowers only opaque resident candidate identities. The organism
    # recomputes its live tie and validates scene/template/lexeme/cardinality before
    # the ordinary ActionV2 path may admit the proposal.
    proposal_checkpoint = copy.deepcopy(organism.checkpoint())
    proposal = critic.propose(organism, scene)
    action = organism.tick(proposal)
    checks["validated_raw_proposal_enters_normal_action_path"] = (
        proposal is not None
        and not hasattr(proposal, "surface")
        and isinstance(action, ActionV2)
        and action.payload == tuple(FORMAL)
        and action.scene_identity == scene_id
        and action.template_identity == proposal.template_identity
        and tuple(action.lexical_identities) == tuple(proposal.lexical_identities)
        and len(action.selection_occurrences) == 5
    )
    learned = (
        organism.contact(
            CONTACT_CONSEQUENCE, (action.ticket, 1), register.P_NEW, True, True,
        ) if isinstance(action, ActionV2) else {}
    )
    checks["only_independent_return_makes_proposed_surface_durable"] = (
        learned.get("selection_credit", 0) > 0
        and learned.get("selection_network_updates", 0) >= 1
        and bool(organism.selection_configuration_revisions)
    )

    stale = ReferenceOrganismV2.restore(copy.deepcopy(proposal_checkpoint))
    stale_proposal = type(proposal)(
        proposal.scene_identity + 1, proposal.template_identity,
        proposal.lexical_identities, proposal.alternatives,
    )
    checks["stale_scene_proposal_refuses"] = (
        stale.tick(stale_proposal) is None and not stale.actions
        and not stale.selection_configuration_revisions
    )
    forged = ReferenceOrganismV2.restore(copy.deepcopy(proposal_checkpoint))
    forged_proposal = type(proposal)(
        proposal.scene_identity, proposal.template_identity ^ 1,
        proposal.lexical_identities, proposal.alternatives,
    )
    checks["unknown_candidate_identity_refuses"] = (
        forged.tick(forged_proposal) is None and not forged.actions
        and not forged.selection_configuration_revisions
    )

    # Higher raw-language Recipes, not the candidate byte literals or a hidden
    # preference, are the new information. Removing them restores the original tie.
    clean_probe = ReferenceOrganismV2.restore(copy.deepcopy(proposal_checkpoint))
    clean_scene = clean_probe._scene_by_id[scene_id]
    lesioned = copy.deepcopy(alice)
    lesioned.recipes.clear()
    lesioned._rebuild_index()
    lesion_critic = RawSurfaceRecipeCriticV1(lesioned)
    checks["raw_recipe_lesion_restores_ambiguity"] = (
        lesion_critic.select(clean_probe, clean_scene) is None
        and set(row.surface for row in lesion_critic.current_candidates(clean_probe, clean_scene))
        == {FORMAL, TERSE}
        and clean_probe.tick() is None
    )

    # Packed raw ecology is sufficient to reconstruct the same structural decision;
    # no Alice source bytes or candidate output text are retained in that packed state.
    packed = alice.packed_state()
    restored_alice = type(alice).restore_packed(_surface, packed)
    restored_pick = RawSurfaceRecipeCriticV1(restored_alice).select(clean_probe, clean_scene)
    packed_lower = packed.lower()
    checks["packed_raw_recipe_state_replays_selection"] = (
        restored_pick is not None and restored_pick.candidate.surface == FORMAL
    )
    checks["packed_state_has_no_candidate_or_corpus_payload"] = (
        FORMAL.lower() not in packed_lower
        and TERSE.lower() not in packed_lower
        and b"alice" not in packed_lower
        and len(packed) < 4096
    )
    sweep_rows = []
    sweep_green = True
    for index, (atoms, words) in enumerate(HELDOUT_SETS):
        probe = register.train(ReferenceOrganismV2(PopulationSpecV1(32768, 2, 4, 42, 8)))
        add_heldout_words(probe, atoms, words, 620000 + index * 1000)
        register.partner(probe, register.P_NEW)
        sid = probe.contact(
            CONTACT_SCENE, (7, register.CTX, len(atoms), *atoms),
            720000 + index, True, True,
        )
        resident_refuses = probe.tick() is None
        pscene = probe._scene_by_id[sid]
        pcandidates = critic.current_candidates(probe, pscene)
        pick = critic.select(probe, pscene)
        formal = f"the {words[0]} {words[1]} {words[2]} the {words[3]}.".encode()
        terse = f"{words[0]} {words[1]}: {words[2]} {words[3]}.".encode()
        formal_n = len(alice.unfold_all(formal))
        terse_n = len(alice.unfold_all(terse))
        row_green = (
            resident_refuses
            and len(pcandidates) == 2
            and set(x.surface for x in pcandidates) == {formal, terse}
            and len({x.developmental_support for x in pcandidates}) == 1
            and pick is not None
            and pick.candidate.surface == formal
            and formal_n > terse_n
            and terse_n == 0
        )
        sweep_green = sweep_green and row_green
        sweep_rows.append({
            "formal": formal.decode(), "terse": terse.decode(),
            "formal_higher_closures": formal_n,
            "terse_higher_closures": terse_n,
            "selected": "" if pick is None else pick.candidate.surface.decode(),
            "pass": row_green,
        })
    checks["eight_heldout_lexical_sets_share_raw_surface_preference"] = sweep_green
    formal_discourse = (
        b"the senior auditor checks the record. Then "
        b"the careful planner revises the procedure."
    )
    terse_discourse = (
        b"senior auditor: checks record. Then "
        b"careful planner: revises procedure."
    )
    formal_discourse_occurrences = alice.unfold_all(formal_discourse)
    terse_discourse_occurrences = alice.unfold_all(terse_discourse)
    formal_discourse_mass = sum(
        alice.recipes[row.recipe_identity].anchor_count
        * alice.recipes[row.recipe_identity].support
        for row in formal_discourse_occurrences
    )
    terse_discourse_mass = sum(
        alice.recipes[row.recipe_identity].anchor_count
        * alice.recipes[row.recipe_identity].support
        for row in terse_discourse_occurrences
    )
    checks["two_sentence_surface_preserves_raw_recipe_advantage"] = (
        len(formal_discourse_occurrences) >= 2
        and len(formal_discourse_occurrences) > len(terse_discourse_occurrences)
        and formal_discourse_mass > terse_discourse_mass
    )
    checks["bounded_current_work"] = (
        critic.last_surface_candidates == 2 and critic.last_raw_touches < 64
    )
    checks["no_runtime_llm_or_surface_oracle"] = all(
        not hasattr(critic, name)
        for name in ("prompt", "expected", "answer", "llm", "tokenizer", "grammar")
    )

    elapsed = (time.perf_counter() - started) * 1000
    report = {
        "schema": "0x1.reference-raw-corpus-outward-selection.v1",
        "pass": all(checks.values()),
        "checks": checks,
        "selected": "" if selected is None else selected.candidate.surface.decode(),
        "candidate_surfaces": [row.surface.decode() for row in candidates],
        "heldout_sweep": sweep_rows,
        "two_sentence_formal_higher_closures": len(formal_discourse_occurrences),
        "two_sentence_terse_higher_closures": len(terse_discourse_occurrences),
        "two_sentence_formal_rank_mass": formal_discourse_mass,
        "two_sentence_terse_rank_mass": terse_discourse_mass,
        "formal_higher_closures": len(formal_occurrences),
        "terse_higher_closures": len(terse_occurrences),
        "raw_recipe_bytes": len(packed),
        "raw_candidate_touches": critic.last_raw_touches,
        "alice_surface_ms": round(surface_ms, 3),
        "alice_recipe_ms": round(recipe_ms, 3),
        "elapsed_ms": round(elapsed, 3),
        "runtime_llm": False,
        "graph_flip": False,
        "claim": "REAL_RAW_CORPUS_RECIPE_PRESSURE_SELECTS_ONE_EXISTING_HELDOUT_RESIDENT_SURFACE_REFERENCE_ONLY_NOT_LLM_GENERATION",
    }
    print(
        "FOUNDRY_RAW_CORPUS_OUTWARD_SELECTION "
        + ("GREEN" if report["pass"] else "RED")
        + f" candidates={len(candidates)} formal_closures={len(formal_occurrences)} "
          f"terse_closures={len(terse_occurrences)} raw_bytes={len(packed)}"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
