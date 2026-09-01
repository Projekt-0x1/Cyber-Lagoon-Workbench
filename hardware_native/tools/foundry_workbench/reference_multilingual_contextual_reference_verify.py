#!/usr/bin/env python3
"""Two-turn multilingual reference reuse on one continuing reference organism."""
from __future__ import annotations

import copy
import json
import time

from reference_organism_v2 import (ActionV2, CONTACT_ENTITY_FEATURES,
    CONTACT_CONSEQUENCE, CONTACT_SCENE, CONTACT_SURFACE, COND_REINSTATED,
    CONTACT_WITHDRAW_SOURCE, PREF_BINDING, PREF_FORM, ReferenceOrganismV2)
from reference_productive_register_discourse_verify import (
    ATOM_CONTEXT, ECOLOGIES, FORMAL, GERMAN, MANDARIN, PARTNER_ECOLOGIES,
    PARTNERS, RUSSIAN, TERSE, HELDOUT_INDEX, clear_partner, contact_name, express,
    relation_tree, set_partner, settle, train, units)


PRONOUNS = {
    FORMAL: "it",
    TERSE: "it",
    GERMAN: "ihn",
    MANDARIN: "它",
    RUSSIAN: "его",
}


def compact(partner, words):
    if partner == FORMAL:
        return (f"the {words[0]} {words[1]} {words[2]} the {words[3]}, and "
                f"the {words[4]} {words[5]} {words[6]} it.")
    if partner == TERSE:
        return (f"{words[0]} {words[1]} {words[2]} {words[3]}; "
                f"{words[4]} {words[5]} {words[6]} it.")
    if partner == GERMAN:
        return (f"der {words[0]} {words[1]} {words[2]} den {words[3]}, und "
                f"der {words[4]} {words[5]} {words[6]} ihn.")
    if partner == MANDARIN:
        return (f"{words[0]}{words[1]}{words[2]}{words[3]}，"
                f"{words[4]}{words[5]}{words[6]}它。")
    if partner == RUSSIAN:
        return (f"{words[0]} {words[1]} {words[2]} {words[3]}, а "
                f"{words[4]} {words[5]} {words[6]} его.")
    raise ValueError("contextual-reference:partner")


def add_entity(organism, identity, source):
    root=identity*16
    features=(root+1,root+2,root+3,root+4)
    organism.contact(CONTACT_ENTITY_FEATURES,
        (identity,len(features),*features),source,True,True)


def name_all_ecologies(organism, atoms, word_offset, source):
    lexical=[]
    for _partner,_render,words in PARTNER_ECOLOGIES:
        group=tuple(words[word_offset:word_offset+len(atoms)])
        if group not in lexical:lexical.append(group)
    for language,words in enumerate(lexical):
        for slot,(atom,word) in enumerate(zip(atoms,words)):
            contact_name(organism,atom,word,source+language*0x100+slot)
            contact_name(organism,atom,word,source+language*0x100+0x40+slot)


def fresh_atoms(organism, base, count, source):
    atoms=tuple(base+i for i in range(count))
    for i,atom in enumerate(atoms):add_entity(organism,atom,source+i)
    return atoms


def name_ecology(organism,atoms,words,source):
    for slot,(atom,word) in enumerate(zip(atoms,words)):
        contact_name(organism,atom,word,source+slot)
        contact_name(organism,atom,word,source+0x40+slot)


def demonstrate_compact(organism,partner,words,referent,base,source):
    atoms=(*fresh_atoms(organism,base,7,source),referent)
    for slot,(atom,word) in enumerate(zip(atoms[:7],words[:7])):
        contact_name(organism,atom,word,source+0x1000+slot)
        contact_name(organism,atom,word,source+0x1100+slot)
    set_partner(organism,partner,source+0x2000)
    relation_tree(organism,atoms,source+0x3000)
    context,conditions=organism._surface_context(organism.current_scene)
    assert context != ATOM_CONTEXT and conditions[-1]==(COND_REINSTATED,)
    organism.contact(CONTACT_SURFACE,units(compact(partner,words)),
                     source+0x4000,True,True)
    clear_partner(organism,source+0x5000)


def teach_compact_form(organism,partner,referent,source):
    set_partner(organism,partner,source)
    for witness in range(2):
        organism.contact(CONTACT_SCENE,(7,ATOM_CONTEXT,1,referent),
                         source+1+witness,True,True)
        organism.contact(CONTACT_SURFACE,units(PRONOUNS[partner]),
                         source+0x10+witness,True,True)
    clear_partner(organism,source+0x20)


def acquire_compact_ecologies(organism):
    # Two independent referents/examples give the ecology a reusable form law.
    referents=(1007,1015)
    for index,(partner,_render,_words) in enumerate(PARTNER_ECOLOGIES):
        for example,referent in enumerate(referents):
            teach_compact_form(organism,partner,referent,
                0xF00000+index*0x1000+example*0x100)
            if partner not in (FORMAL,TERSE):
                teach_compact_form(organism,partner,referent,
                    0xF80000+index*0x1000+example*0x100)
    for index,(partner,_render,words) in enumerate(PARTNER_ECOLOGIES):
        for example,referent in enumerate(referents):
            demonstrate_compact(organism,partner,
                words[example*8:(example+1)*8],referent,
                0x30000+index*0x1000+example*0x100,
                0x1000000+index*0x20000+example*0x8000)
    # A temporary ordinary-contact advantage nominates each first live trial.
    # Equalization removes it immediately; subsequent use must come from credit.
    for index,(partner,_render,words) in enumerate(PARTNER_ECOLOGIES):
        teach_compact_form(organism,partner,1007,
            0x1F00000+index*0x1000)
        demonstrate_compact(organism,partner,words[:8],1007,
            0x50000+index*0x1000,0x2000000+index*0x20000)
        atoms=(*fresh_atoms(organism,0x60000+index*0x1000,7,
                           0x2100000+index*0x20000),1007)
        name_all_ecologies(organism,atoms[:7],0,
                           0x2200000+index*0x20000)
        for slot,(atom,word) in enumerate(zip(atoms[:7],words[:7])):
            contact_name(organism,atom,word,
                         0x2280000+index*0x20000+slot)
        set_partner(organism,partner,0x2300000+index*0x20000)
        high=relation_tree(organism,atoms,0x2400000+index*0x20000)
        action=organism.tick()
        assert isinstance(action,ActionV2),partner
        assert bytes(action.payload).decode()==compact(partner,words[:8]),(
            partner,bytes(action.payload).decode(),compact(partner,words[:8]))
        assert action.scene_identity==high and action.form_slots
        learned=settle(organism,action,partner,1)
        assert learned.get("selection_credit",0)>0
        clear_partner(organism,0x2500000+index*0x20000)
        for other in (FORMAL,GERMAN,MANDARIN,RUSSIAN):
            if PRONOUNS[other]==PRONOUNS[partner]:continue
            teach_compact_form(organism,other,1007,
                0x1F80000+index*0x10000+other)
        for language,(_other_partner,_other_render,other_words) in enumerate(
                PARTNER_ECOLOGIES):
            if other_words==words:continue
            for slot,(atom,word) in enumerate(zip(atoms[:7],other_words[:7])):
                contact_name(organism,atom,word,
                    0x2580000+index*0x100000+language*0x1000+slot)
        for other,(other_partner,_other_render,other_words) in enumerate(
                PARTNER_ECOLOGIES):
            if other_partner==partner:continue
            demonstrate_compact(organism,other_partner,other_words[:8],1007,
                0x70000+index*0x10000+other*0x1000,
                0x2600000+index*0x100000+other*0x10000)
    return {units(compact(partner,words[example*8:(example+1)*8]))
            for partner,_render,words in PARTNER_ECOLOGIES
            for example in range(2)}


def two_turn(organism,partner,words,base):
    first_atoms=fresh_atoms(organism,base,8,base+0x10000)
    name_ecology(organism,first_atoms,words[:8],base+0x20000)
    assert not organism._shared_reinstated(partner,first_atoms[-1])
    first_scene,first=express(organism,partner,first_atoms,base+0x30000)
    if not isinstance(first,ActionV2):return first_scene,first,None,None
    settle(organism,first,partner,1)
    became_shared=organism._shared_reinstated(partner,first_atoms[-1])
    second_prefix=fresh_atoms(organism,base+0x40000,7,base+0x50000)
    name_ecology(organism,second_prefix,words[:7],base+0x60000)
    second_atoms=(*second_prefix,first_atoms[-1])
    second_scene,second=express(organism,partner,second_atoms,base+0x70000)
    return first_scene,first,second_scene,second if became_shared else None


def compact_selected(action):
    return (isinstance(action,ActionV2)
            and any(row[0]==PREF_FORM for row in action.selection_occurrences))


def consequence_control(checkpoint,effect,base):
    organism=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    partner,_render,words=PARTNER_ECOLOGIES[0]
    first_atoms=fresh_atoms(organism,base,8,base+0x10000)
    name_ecology(organism,first_atoms,words[:8],base+0x20000)
    _scene,first=express(organism,partner,first_atoms,base+0x30000)
    if not isinstance(first,ActionV2):return False,False,None
    settle(organism,first,partner,effect)
    shared=organism._shared_reinstated(partner,first_atoms[-1])
    prefix=fresh_atoms(organism,base+0x40000,7,base+0x50000)
    name_ecology(organism,prefix,words[:7],base+0x60000)
    _scene,second=express(organism,partner,(*prefix,first_atoms[-1]),
                          base+0x70000)
    return True,shared,second


def main():
    started=time.perf_counter();checks={}
    organism,_atoms,training_surfaces,_unacquired=train()
    compact_training_surfaces=acquire_compact_ecologies(organism)
    checkpoint=copy.deepcopy(organism.checkpoint())
    outputs={}
    for index,(partner,_render,words) in enumerate(PARTNER_ECOLOGIES):
        heldout=tuple(words[i] for i in HELDOUT_INDEX)
        first_scene,first,second_scene,second=two_turn(
            organism,partner,heldout,0x4000000+index*0x100000)
        outputs[partner]="" if second is None else bytes(second.payload).decode()
        expected=compact(partner,heldout)
        checks[f"partner_{partner}_positive_first_turn_enables_compaction"]=(
            isinstance(first,ActionV2) and isinstance(second,ActionV2)
            and outputs[partner]==expected)
        checks[f"partner_{partner}_recursive_compact_network"]=(
            isinstance(second,ActionV2) and len(second.relation_occurrences)==7
            and any(row[0]==PREF_BINDING for row in second.selection_occurrences)
            and any(row[0]==PREF_FORM for row in second.selection_occurrences)
            and all(row in second.contributors
                    for row in second.relation_occurrences))
        if isinstance(second,ActionV2):settle(organism,second,partner,1)
    withheld=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    atoms=fresh_atoms(withheld,0x6000000,8,0x6010000)
    formal_words=PARTNER_ECOLOGIES[0][2]
    name_ecology(withheld,atoms,formal_words[:8],0x6020000)
    _scene,first=express(withheld,FORMAL,atoms,0x6030000)
    prefix=fresh_atoms(withheld,0x6040000,7,0x6050000)
    name_all_ecologies(withheld,prefix,0,0x6060000)
    _scene,second=express(withheld,FORMAL,(*prefix,atoms[-1]),0x6070000)
    checks["withheld_first_return_cannot_compact"]=(
        isinstance(first,ActionV2) and second is None)
    for label,effect,base in (("zero",0,0x6100000),
                              ("negative",-1,0x6200000)):
        issued,shared,second=consequence_control(checkpoint,effect,base)
        checks[f"{label}_first_return_cannot_create_common_ground"]=(
            issued and not shared and not compact_selected(second))

    authenticated=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    words=PARTNER_ECOLOGIES[0][2]
    atoms=fresh_atoms(authenticated,0x6300000,8,0x6310000)
    name_ecology(authenticated,atoms,words[:8],0x6320000)
    _scene,action=express(authenticated,FORMAL,atoms,0x6330000)
    before=authenticated.digest()
    try:
        authenticated.contact(CONTACT_CONSEQUENCE,(action.ticket,1),
                              FORMAL+0x100,True,True)
    except ValueError as exc:
        checks["wrong_first_return_source_refuses_atomically"]=(
            str(exc)=="organism:consequence_source"
            and authenticated.digest()==before)
    else:checks["wrong_first_return_source_refuses_atomically"]=False

    wrong_partner=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    atoms=fresh_atoms(wrong_partner,0x7000000,8,0x7010000)
    formal_words=PARTNER_ECOLOGIES[0][2]
    german_words=PARTNER_ECOLOGIES[2][2]
    name_ecology(wrong_partner,atoms,formal_words[:8],0x7020000)
    _scene,first=express(wrong_partner,FORMAL,atoms,0x7030000)
    if isinstance(first,ActionV2):settle(wrong_partner,first,FORMAL,1)
    contact_name(wrong_partner,atoms[-1],german_words[7],0x7031000)
    contact_name(wrong_partner,atoms[-1],german_words[7],0x7031001)
    prefix=fresh_atoms(wrong_partner,0x7040000,7,0x7050000)
    name_ecology(wrong_partner,prefix,
                 german_words[:7],0x7060000)
    _scene,borrowed=express(wrong_partner,GERMAN,(*prefix,atoms[-1]),0x7070000)
    checks["wrong_partner_cannot_borrow_shared_reference"]=(
        isinstance(first,ActionV2) and not compact_selected(borrowed))

    withdrawal=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    words=PARTNER_ECOLOGIES[0][2]
    atoms=fresh_atoms(withdrawal,0x7100000,8,0x7110000)
    name_ecology(withdrawal,atoms,words[:8],0x7120000)
    _scene,first=express(withdrawal,FORMAL,atoms,0x7130000)
    if isinstance(first,ActionV2):settle(withdrawal,first,FORMAL,1)
    relation=next((row for row in reversed(withdrawal.shared_episode_relations)
                   if row.partner==FORMAL and row.source_roots),None)
    all_roots_refuse=True
    if relation is not None:
        shared_checkpoint=copy.deepcopy(withdrawal.checkpoint())
        episode=withdrawal._episode_by_id[relation.episode_identity]
        for offset,root in enumerate(relation.source_roots):
            probe=ReferenceOrganismV2.restore(copy.deepcopy(shared_checkpoint))
            probe.contact(CONTACT_WITHDRAW_SOURCE,(root,),
                          0x7138000+offset,True,True)
            set_partner(probe,FORMAL,0x7139000+offset)
            probe.world_state=tuple(episode.atoms)
            probe.world_source=0x713A000+offset
            probe.world_state_occurrence=0x713B000+offset
            all_roots_refuse=(all_roots_refuse
                and not probe._shared_reinstated(FORMAL,atoms[-1])
                and probe._resident_world_scene() is None)
    if relation is not None:
        withdrawal.contact(CONTACT_WITHDRAW_SOURCE,(relation.source_roots[0],),
                           0x7140000,True,True)
    removed=not withdrawal._shared_reinstated(FORMAL,atoms[-1])
    _scene,reacquired=express(withdrawal,FORMAL,atoms,0x7150000)
    if isinstance(reacquired,ActionV2):settle(withdrawal,reacquired,FORMAL,1)
    checks["shared_source_withdrawal_cascades_and_fresh_return_reacquires"]=(
        isinstance(first,ActionV2) and relation is not None and all_roots_refuse
        and removed
        and isinstance(reacquired,ActionV2)
        and withdrawal._shared_reinstated(FORMAL,atoms[-1]))
    replay=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    replay_rows=[]
    for index,(partner,_render,words) in enumerate(PARTNER_ECOLOGIES):
        heldout=tuple(words[i] for i in HELDOUT_INDEX)
        _a,_b,_c,second=two_turn(replay,partner,heldout,
                                 0x4000000+index*0x100000)
        replay_rows.append("" if second is None else bytes(second.payload).decode())
        if isinstance(second,ActionV2):settle(replay,second,partner,1)
    checks["checkpoint_replays_identical_two_turn_dialogue"]=(
        replay_rows==[outputs[p] for p in PARTNERS]
        and replay.digest()==organism.digest())
    checks["bounded_conditioned_candidate_work"]=(
        organism.last_selection_candidate_touches<=4096)
    generalized=organism.language.condition_form_candidates((COND_REINSTATED,))
    checks["compact_forms_require_multiple_entity_witnesses"]=(
        len({row[3] for row in generalized})==4
        and all(row[5]>=2 and row[6]>=2 for row in generalized))
    checks["all_compact_outputs_absent_from_training"] = all(
        units(text) not in training_surfaces|compact_training_surfaces
        for text in outputs.values())
    checks["no_prompt_router_or_transcript_api"] = all(not hasattr(organism,name)
        for name in ("prompt","answer","translate","pronoun","language_router",
                     "context_window","transcript","enqueue_goal"))
    elapsed=(time.perf_counter()-started)*1000
    checks["consumer_cpu_runtime_under_60_seconds"]=elapsed<60000
    result={
        "schema":"0x1.reference-multilingual-contextual-reference.v1",
        "pass":all(checks.values()),"checks":checks,
        "outputs":{str(k):v for k,v in outputs.items()},
        "languages":4,"surface_ecologies":5,"turns_per_partner":2,
        "runtime_llm":False,"adult_attached":False,"reference_only":True,
        "host_authored_relation_contact":True,
        "host_fixture_partner_selection":True,
        "fixture_returned_scalar_consequence":True,
        "exclusive_consequence_origin_authenticated":False,
        "prior_mention_condition_law":"AUTHORED_REFERENCE_FIXTURE",
        "contextual_reference_condition_acquired":False,
        "learned_conditioned_surface_selection":True,
        "per_byte_ancestry":False,"physical_consequence":False,
        "production_ir_translation":"UNDEFINED",
        "direct_parity":"NOT_RUN/RED","human_language_mastery":False,
        "claim":"FOUR_LANGUAGE_LEARNED_SURFACE_UNDER_AUTHORED_PRIOR_MENTION_CONDITION_REFERENCE_ONLY",
        "remaining_red":["RESIDENT_ACQUIRED_REFERENCE_CONDITION",
                         "AUTHENTICATED_SURFACE_TRAJECTORY_INTEGRATION",
                         "OPEN_MULTILINGUAL_DIALOGUE","QUESTION_ANSWER_BINDING",
                         "PRODUCTION_RECIPE_IR_TRANSLATION",
                         "DIRECT_PHYSICAL_PARITY","HUMAN_LANGUAGE_MASTERY"],
        "elapsed_ms":round(elapsed,3)}
    print("FOUNDRY_MULTILINGUAL_CONTEXTUAL_REFERENCE "+
          ("GREEN" if result["pass"] else "RED")+
          " languages=4 ecologies=5 turns=2")
    print(json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False))
    raise SystemExit(0 if result["pass"] else 1)


if __name__=="__main__":main()
