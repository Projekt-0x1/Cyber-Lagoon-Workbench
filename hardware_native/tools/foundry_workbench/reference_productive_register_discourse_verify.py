#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from reference_organism_v2 import (ActionV2, CONTACT_CONSEQUENCE,
    CONTACT_ENTITY_FEATURES, CONTACT_PARTNER_CONTEXT, CONTACT_SCENE,
    CONTACT_SCENE_LINK, CONTACT_SURFACE, CONTACT_WITHDRAW_SOURCE, PREF_BINDING,
    ReferenceOrganismV2)
from reference_population_v1 import PopulationSpecV1
from reference_packed_selection_revision_v1 import PackedSelectionRevisionV1


ATOM_CONTEXT = 0x4100
PAIR_RELATION = 0x4101
CLAUSE_RELATION = 0x4102
COORDINATION_RELATION = 0x4103
FORMAL = 0x5101
TERSE = 0x5102
GERMAN = 0x5103
MANDARIN = 0x5104
RUSSIAN = 0x5105
UNKNOWN = 0x51FF
FORMAL_CONTACTS = (0x6101, 0x6102)
TERSE_CONTACTS = (0x6201, 0x6202)
GERMAN_CONTACTS = (0x6301, 0x6302)
MANDARIN_CONTACTS = (0x6401, 0x6402)
RUSSIAN_CONTACTS = (0x6501, 0x6502)

ENGLISH_A = ("careful", "engineer", "checks", "sensor",
             "quiet", "pilot", "inspects", "motor")
ENGLISH_B = ("patient", "technician", "monitors", "process",
             "senior", "auditor", "reviews", "report")
GERMAN_A = ("sorgfältige", "Ingenieur", "prüft", "Sensor",
            "ruhige", "Pilot", "inspiziert", "Motor")
GERMAN_B = ("geduldige", "Techniker", "überwacht", "Prozess",
            "erfahrene", "Prüfer", "untersucht", "Bericht")
MANDARIN_A = ("细心的", "工程师", "检查", "传感器",
              "安静的", "飞行员", "检视", "电机")
MANDARIN_B = ("耐心的", "技术员", "监控", "流程",
              "资深的", "审计员", "审查", "报告")
RUSSIAN_A = ("внимательный", "инженер", "проверяет", "датчик",
             "спокойный", "пилот", "осматривает", "двигатель")
RUSSIAN_B = ("терпеливый", "техник", "контролирует", "процесс",
             "опытный", "аудитор", "проверяет", "отчёт")
HELDOUT_INDEX = (0, 9, 2, 11, 12, 5, 14, 7)


def units(text):
    return tuple(text.encode())


def formal(words):
    return (f"the {words[0]} {words[1]} {words[2]} the {words[3]}, and "
            f"the {words[4]} {words[5]} {words[6]} the {words[7]}.")


def terse(words):
    return (f"{words[0]} {words[1]} {words[2]} {words[3]}; "
            f"{words[4]} {words[5]} {words[6]} {words[7]}.")


def german(words):
    return (f"der {words[0]} {words[1]} {words[2]} den {words[3]}, und "
            f"der {words[4]} {words[5]} {words[6]} den {words[7]}.")


def mandarin(words):
    return (f"{words[0]}{words[1]}{words[2]}{words[3]}，"
            f"{words[4]}{words[5]}{words[6]}{words[7]}。")


def russian(words):
    return (f"{words[0]} {words[1]} {words[2]} {words[3]}, а "
            f"{words[4]} {words[5]} {words[6]} {words[7]}.")


ECOLOGIES = (
    (*FORMAL_CONTACTS, formal, ENGLISH_A + ENGLISH_B,
     (FORMAL_CONTACTS[0],TERSE_CONTACTS[0])),
    (*TERSE_CONTACTS, terse, ENGLISH_A + ENGLISH_B, ()),
    (*GERMAN_CONTACTS, german, GERMAN_A + GERMAN_B, GERMAN_CONTACTS),
    (*MANDARIN_CONTACTS, mandarin, MANDARIN_A + MANDARIN_B,
     MANDARIN_CONTACTS),
    (*RUSSIAN_CONTACTS, russian, RUSSIAN_A + RUSSIAN_B,
     RUSSIAN_CONTACTS),
)
PARTNER_ECOLOGIES = (
    (FORMAL, formal, ENGLISH_A + ENGLISH_B),
    (TERSE, terse, ENGLISH_A + ENGLISH_B),
    (GERMAN, german, GERMAN_A + GERMAN_B),
    (MANDARIN, mandarin, MANDARIN_A + MANDARIN_B),
    (RUSSIAN, russian, RUSSIAN_A + RUSSIAN_B),
)
PARTNERS=tuple(row[0] for row in PARTNER_ECOLOGIES)
LEXICAL_ECOLOGIES = (
    ENGLISH_A + ENGLISH_B,
    GERMAN_A + GERMAN_B,
    MANDARIN_A + MANDARIN_B,
    RUSSIAN_A + RUSSIAN_B,
)


def contact_entity(organism, identity, feature, source):
    organism.contact(CONTACT_ENTITY_FEATURES,
                     (identity, 4, feature, feature + 0x100,
                      feature + 0x200, identity + 0x10000),
                     source, True, True)


def contact_name(organism, identity, text, source):
    organism.contact(CONTACT_SCENE, (7, ATOM_CONTEXT, 1, identity),
                     source, True, True)
    organism.contact(CONTACT_SURFACE, units(text), source, True, True)


def link(organism, left, right, relation, source):
    before = organism.next_scene
    organism.contact(CONTACT_SCENE_LINK, (left, right, relation),
                     source, True, True)
    assert organism.next_scene == before + 1
    return before


def relation_tree(organism, atoms, source, relations=(PAIR_RELATION,
                  CLAUSE_RELATION, COORDINATION_RELATION)):
    pair_relation,clause_relation,coordination_relation=relations
    leaves = [organism.contact(CONTACT_SCENE,
        (7, ATOM_CONTEXT, 1, atom), source + i, True, True)
        for i, atom in enumerate(atoms)]
    pairs = [link(organism, leaves[i], leaves[i + 1], pair_relation,
                  source + 16 + i) for i in range(0, 8, 2)]
    clauses = [link(organism, pairs[i], pairs[i + 1], clause_relation,
                    source + 32 + i) for i in range(0, 4, 2)]
    return link(organism, clauses[0], clauses[1], coordination_relation,
                source + 48)


def set_partner(organism, partner, source):
    organism.contact(CONTACT_PARTNER_CONTEXT, (1, 7, partner),
                     source, True, True)


def clear_partner(organism,source):
    organism.contact(CONTACT_PARTNER_CONTEXT,(0,0,0),source,True,True)


def demonstrate(organism, atoms, render, words, source):
    high=relation_tree(organism,atoms,source+0x1000)
    assert organism.current_scene.identity==high
    organism.contact(CONTACT_SURFACE,units(render(words)),source,True,True)


def earn_partner(organism,partner,target_render,target_words,example,source):
    clear_partner(organism,source-1)
    atoms=tuple(1000+i for i in range(example*8,(example+1)*8))
    words=target_words[example*8:(example+1)*8]
    # One temporary, fully ordinary contact advantage nominates a real trial.
    # After the returned consequence is settled, all alternatives are restored
    # to equal developmental support; only resident Network credit can transfer.
    for atom,word in zip(atoms,words):contact_name(organism,atom,word,source)
    demonstrate(organism,atoms,target_render,words,source+1)
    set_partner(organism,partner,source+2)
    high=relation_tree(organism,atoms,source+0x2000)
    action=organism.tick()
    assert isinstance(action,ActionV2) and action.scene_identity==high
    assert bytes(action.payload).decode()==target_render(words),(
        partner,example,bytes(action.payload).decode(),target_render(words),
        [(row.support,row.sources,row.identity) for row in
         organism.language.template_candidates(
             organism._surface_view(organism._scene_by_id[high])[1],8)])
    learned=settle(organism,action,partner,1)
    assert learned.get('selection_credit',0)>0
    clear_partner(organism,source+3)

    for ecology, alternative_words in enumerate(LEXICAL_ECOLOGIES):
        if alternative_words == target_words:
            continue
        for atom,word in zip(atoms,
                alternative_words[example*8:(example+1)*8]):
            contact_name(organism,atom,word,source+0x100+ecology)
    for _partner,render,all_words in PARTNER_ECOLOGIES:
        if render is target_render:continue
        demonstrate(organism,atoms,render,
                    all_words[example*8:(example+1)*8],
                    source+0x200+_partner)
    return action


def train():
    organism = ReferenceOrganismV2(PopulationSpecV1(131072, 2, 4, 42, 8))
    training_atoms = tuple(1000 + i for i in range(16))
    heldout_atoms = tuple(2000 + i for i in range(16))
    for i, (training_atom, heldout_atom) in enumerate(
            zip(training_atoms, heldout_atoms)):
        contact_entity(organism, training_atom, 0x7000 + i, 0x80000 + i)
        contact_entity(organism, heldout_atom, 0x7000 + i, 0x81000 + i)
    training_surfaces = set()
    for primary, secondary, render, words, name_sources in ECOLOGIES:
        for atom, word in zip(training_atoms, words):
            for source in name_sources:
                contact_name(organism, atom, word, source)
        for example, atom_slice in enumerate((training_atoms[:8],
                                               training_atoms[8:])):
            high = relation_tree(organism, atom_slice,
                                 0x90000 + primary * 64 + example * 128)
            assert organism.current_scene.identity == high
            text = render(words[example * 8:(example + 1) * 8])
            training_surfaces.add(units(text))
            organism.contact(CONTACT_SURFACE, units(text),
                             primary if example == 0 else secondary,
                             True, True)
    unacquired=copy.deepcopy(organism.checkpoint())
    for index,(partner,render,words) in enumerate(PARTNER_ECOLOGIES):
        for example in range(2):
            earn_partner(organism,partner,render,words,example,
                         0xA00000+index*0x10000+example*0x4000)
    return (organism,tuple(heldout_atoms[i] for i in HELDOUT_INDEX),
            training_surfaces,unacquired)


def express(organism, partner, atoms, source):
    set_partner(organism, partner, source)
    high = relation_tree(organism, atoms, source + 64)
    scene = organism._scene_by_id[high]
    action = organism.tick()
    return scene, action


def settle(organism, action, partner, effect):
    return organism.contact(CONTACT_CONSEQUENCE, (action.ticket, effect),
                            partner, True, True)


def main():
    started = time.perf_counter()
    checks = {}
    organism, atoms, training_surfaces, unacquired_checkpoint = train()
    checkpoint = copy.deepcopy(organism.checkpoint())
    english_words = tuple((ENGLISH_A + ENGLISH_B)[i] for i in HELDOUT_INDEX)
    german_words = tuple((GERMAN_A + GERMAN_B)[i] for i in HELDOUT_INDEX)
    mandarin_words = tuple((MANDARIN_A + MANDARIN_B)[i]
                           for i in HELDOUT_INDEX)
    russian_words = tuple((RUSSIAN_A + RUSSIAN_B)[i]
                          for i in HELDOUT_INDEX)
    expected = {FORMAL: formal(english_words), TERSE: terse(english_words),
                GERMAN: german(german_words),
                MANDARIN: mandarin(mandarin_words),
                RUSSIAN: russian(russian_words)}
    outputs = {}
    relation_counts = {}

    unacquired=ReferenceOrganismV2.restore(copy.deepcopy(unacquired_checkpoint))
    _scene,unacquired_action=express(unacquired,FORMAL,atoms,0x9F000)
    checks['partner_identity_without_lived_credit_cannot_route_surface']=(
        unacquired_action is None)
    contact_sources={source for primary,secondary,_render,_words,_names in ECOLOGIES
                     for source in (primary,secondary)}
    checks['partner_ids_are_disjoint_from_training_sources']=(
        not contact_sources.intersection(PARTNERS))

    # One continuing organism changes partner context and produces five unseen
    # surfaces from the same depth-three current relation computation.
    for turn, partner in enumerate(PARTNERS):
        scene, action = express(organism, partner, atoms, 0xA0000 + turn * 256)
        outputs[partner] = "" if action is None else bytes(action.payload).decode()
        relation_counts[partner] = len(scene.relation_occurrences)
        checks[f"partner_{partner}_heldout_surface"] = (
            isinstance(action, ActionV2) and outputs[partner] == expected[partner]
            and action.payload not in training_surfaces)
        checks[f"partner_{partner}_recursive_network_load_bearing"] = (
            isinstance(action, ActionV2) and scene.binding_identity > 0
            and action.binding_identity == scene.binding_identity
            and len(scene.relation_occurrences) == 7
            and tuple(action.relation_occurrences) == scene.relation_occurrences
            and all(identity in action.contributors
                    for identity in scene.relation_occurrences)
            and any(row[0] == PREF_BINDING
                    for row in action.selection_occurrences))
        if isinstance(action, ActionV2):
            if turn==0:
                forged=ReferenceOrganismV2.restore(
                    copy.deepcopy(organism.checkpoint()))
                before=forged.digest()
                try:
                    forged.contact(CONTACT_CONSEQUENCE,(action.ticket,1),
                                   partner+0x1000,True,True)
                except ValueError as exc:
                    checks['wrong_consequence_source_refuses_atomically']=(
                        str(exc)=='organism:consequence_source'
                        and forged.digest()==before)
                else:checks['wrong_consequence_source_refuses_atomically']=False
            learned = settle(organism, action, partner, 1)
            checks[f"partner_{partner}_consequence_credit"] = (
                learned.get("selection_credit", 0) > 0)
        else:
            checks[f"partner_{partner}_consequence_credit"] = False

    checks["one_continuing_organism_five_surface_ecologies"] = (
        len(set(outputs.values())) == 5 and len(organism.actions) == 15)
    checks["complete_heldout_outputs_absent_from_training"] = all(
        units(text) not in training_surfaces for text in expected.values())

    unknown = ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    _scene, unknown_action = express(unknown, UNKNOWN, atoms, 0xB0000)
    checks["unfamiliar_partner_refuses_surface_tie"] = unknown_action is None

    creditless=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    creditless._selection_revisions=PackedSelectionRevisionV1()
    creditless._selection_construction_index.clear()
    _scene,creditless_action=express(creditless,FORMAL,atoms,0xB1000)
    checks['learned_network_credit_is_required_for_heldout_choice']=(
        creditless_action is None)

    flat=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    set_partner(flat,FORMAL,0xB2000)
    flat.contact(CONTACT_SCENE,(7,COORDINATION_RELATION,8,*atoms),
                 0xB2001,True,True)
    checks['ordered_atoms_without_recursive_binding_refuse']=(flat.tick() is None)

    regrouped=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    set_partner(regrouped,FORMAL,0xB3000)
    relation_tree(regrouped,atoms,0xB3100,
                  (PAIR_RELATION+0x100,CLAUSE_RELATION,
                   COORDINATION_RELATION))
    checks['altered_relation_grouping_refuses']=(regrouped.tick() is None)

    binding_lesion=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    set_partner(binding_lesion,FORMAL,0xB4000)
    high=relation_tree(binding_lesion,atoms,0xB4100)
    binding_lesion._scene_by_id[high].binding_identity=0
    binding_lesion._scene_by_id[high].relation_occurrences=()
    checks['recursive_binding_lesion_refuses']=(binding_lesion.tick() is None)

    closure_lesion=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    set_partner(closure_lesion,FORMAL,0xB5000)
    high=relation_tree(closure_lesion,atoms,0xB5100)
    retained_binding=closure_lesion._scene_by_id[high].binding_identity
    closure_lesion._scene_by_id[high].relation_occurrences=()
    checks['closure_only_lesion_with_binding_retained_refuses']=(
        retained_binding > 0 and closure_lesion.tick() is None)

    binding_substitution=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    set_partner(binding_substitution,FORMAL,0xB6000)
    high=relation_tree(binding_substitution,atoms,0xB6100)
    retained_closure=binding_substitution._scene_by_id[high].relation_occurrences
    binding_substitution._scene_by_id[high].binding_identity += 0x100000
    checks['binding_only_substitution_with_closure_retained_refuses']=(
        len(retained_closure) == 7 and binding_substitution.tick() is None)

    replay = ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    replay_outputs = []
    for turn, partner in enumerate(PARTNERS):
        _scene, action = express(replay, partner, atoms, 0xA0000 + turn * 256)
        replay_outputs.append("" if action is None else bytes(action.payload).decode())
        if isinstance(action, ActionV2):
            settle(replay, action, partner, 1)
    checks["checkpoint_replays_identical_ordered_dialogue"] = (
        replay_outputs == [outputs[p] for p in PARTNERS]
        and replay.digest() == organism.digest())

    negative = ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    _scene, first = express(negative, FORMAL, atoms, 0xC0000)
    if isinstance(first, ActionV2):
        settle(negative, first, FORMAL, -1)
    _scene, revised = express(negative, FORMAL, atoms, 0xC1000)
    checks["negative_consequence_revises_only_partner_choice"] = (
        isinstance(first, ActionV2) and
        (revised is None or revised.payload != first.payload))
    if isinstance(revised, ActionV2):
        settle(negative, revised, FORMAL, 1)
    _scene, german_after = express(negative, GERMAN, atoms, 0xC2000)
    checks["other_partner_language_survives_negative_return"] = (
        isinstance(german_after, ActionV2)
        and bytes(german_after.payload).decode() == expected[GERMAN])

    # Remove one developmental witness from the German ecology. The organism
    # becomes silent, then ordinary naming and construction contacts under new
    # independent sources reconstruct enough support for the held-out use.
    reacquired = ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    german_name_sources=set()
    for atom,word in zip(tuple(1000+i for i in range(16)),
                         GERMAN_A+GERMAN_B):
        for _support,candidate,sources in reacquired.language.lexeme_candidates(atom):
            if candidate==units(word):german_name_sources.update(sources)
    for offset,source in enumerate(sorted(german_name_sources)):
        reacquired.contact(CONTACT_WITHDRAW_SOURCE,(source,),
                           0xD0000+offset,True,True)
    _scene, withdrawn_action = express(reacquired, GERMAN, atoms,
                                       0xD1000)
    checks["source_withdrawal_removes_language_ecology"] = (
        withdrawn_action is None)
    clear_partner(reacquired,0xD7000)
    training_atoms = tuple(1000 + i for i in range(16))
    reacquire_support=max(row[0] for atom in training_atoms
        for row in reacquired.language.lexeme_candidates(atom))
    new_sources=tuple(0xD8000+i for i in range(reacquire_support))
    for atom, word in zip(training_atoms, GERMAN_A + GERMAN_B):
        for source in new_sources:contact_name(reacquired,atom,word,source)
    demonstrate(reacquired,training_atoms[:8],german,GERMAN_A,0xD9000)
    demonstrate(reacquired,training_atoms[8:],german,GERMAN_B,0xD9001)
    _scene, reacquired_action = express(reacquired, GERMAN, atoms,
                                        0xD3000)
    checks["ordinary_contact_reacquires_withdrawn_language_ecology"] = (
        isinstance(reacquired_action, ActionV2)
        and bytes(reacquired_action.payload).decode() == expected[GERMAN])

    checks["no_language_or_style_router"] = all(not hasattr(organism, name)
        for name in ("language_id", "language_router", "translate", "style",
                     "register", "persona", "prompt", "context_window"))
    checks["runtime_llm_absent"] = "llm" not in vars(organism)
    checks["exclusive_consequence_origin_authority_not_claimed"] = True
    elapsed = (time.perf_counter() - started) * 1000
    checks["consumer_cpu_runtime_under_60_seconds"] = elapsed < 60000
    result = {
        "schema": "0x1.reference-productive-register-discourse.v2",
        "pass": all(checks.values()),
        "checks": checks,
        "outputs": {"formal_english": outputs.get(FORMAL, ""),
                    "terse_english": outputs.get(TERSE, ""),
                    "german": outputs.get(GERMAN, ""),
                    "mandarin": outputs.get(MANDARIN, ""),
                    "russian": outputs.get(RUSSIAN, "")},
        "relation_occurrences_per_depth3_output": relation_counts,
        "training_complete_outputs": len(training_surfaces),
        "heldout_complete_outputs": 5,
        "language_count": 4,
        "surface_ecology_count": 5,
        "continuing_organisms": 1,
        "training_partner_source_alias": False,
        "candidate_selection_basis": "LIVED_RUN_MEMBER_EVIDENCE",
        "reference_credit_store_tamperproof": False,
        "exclusive_consequence_origin_authenticated": False,
        "host_authored_relation_contact": True,
        "fixture_returned_scalar_consequence": True,
        "physical_returned_consequence": False,
        "per_byte_ancestry": False,
        "continuous_open_dialogue": False,
        "runtime_llm": False,
        "adult_attached": False,
        "reference_only": True,
        "graph_flip": False,
        "physical_direct_parity": "NOT_RUN/RED",
        "human_language_mastery": False,
        "claim": "CONTACT_AND_RETURN_ACQUIRED_FOUR_LANGUAGE_PARTNER_CONDITIONED_DEPTH3_COMPOSITION_REFERENCE_ONLY",
        "remaining_red": ["PHYSICAL_RETURNED_CONSEQUENCE",
                          "CREDIT_ORIGIN_AUTHENTICATION",
                          "PER_BYTE_OUTPUT_ANCESTRY",
                          "OPEN_MULTILINGUAL_DIALOGUE",
                          "CHIN_ACQUISITION",
                          "PRODUCTION_RECIPE_IR_TRANSLATION",
                          "DIRECT_PHYSICAL_PARITY",
                          "HUMAN_LANGUAGE_MASTERY"],
        "elapsed_ms": round(elapsed, 3),
    }
    print("FOUNDRY_PRODUCTIVE_REGISTER_DISCOURSE " +
          ("GREEN" if result["pass"] else "RED") +
          " continuing=1 languages=4 depth3=1 heldout=5")
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
