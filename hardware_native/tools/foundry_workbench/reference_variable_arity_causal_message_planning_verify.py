#!/usr/bin/env python3
"""Held-out exact-arity message grouping in the one continuing Life."""
from __future__ import annotations
import copy,json,tempfile,time
from pathlib import Path
from life_function_factory_v1 import build_cache,load_mark
from reference_life_function_curriculum_v1 import canonical_life_function_curriculum_v2
from reference_life_extension_causal_depth_plus_v1 import A_HUMIDITY,A_ROOTS

TARGET_CAUSE=0xA104
GROUP_SOURCES=(0xFE01,0xFE02)
HUMIDITY_CAUSAL_SOURCES=(0xF910,0xF911,0xF912)

def _fresh(runtime):
    return type(runtime.adult).restore(copy.deepcopy(runtime.adult.checkpoint()))

def _surface(adult,atom):
    leaf=adult.language_adult.leaf(100,(int(atom),))
    return bytes(adult.compose_causal_component(leaf.identity)[0]),leaf

def _direct_effects(adult,leaf):
    return tuple(int(row[3]) for row in adult.causal_message_rows(leaf.identity)
                 if int(row[2])==int(leaf.identity))

def verify_loaded(loaded,curriculum):
    before=_fresh(loaded['causal_siblings_8'])
    after=_fresh(loaded['lexical_causal_integration'])
    before_surface,before_leaf=_surface(before,TARGET_CAUSE)
    after_surface,after_leaf=_surface(after,TARGET_CAUSE)
    effects=_direct_effects(after,after_leaf)
    exact=after.language_adult.common_cause_span_expression(*effects)
    group_surface=bytes(exact[1]) if exact else b''
    effect_surfaces=tuple(bytes(after.language_adult._leaf_surface(effect)) for effect in effects)

    withdrawn=_fresh(loaded['lexical_causal_integration'])
    for source in GROUP_SOURCES:withdrawn.language.withdraw_source(source)
    withdrawn_surface,_=_surface(withdrawn,TARGET_CAUSE)

    member_cut=_fresh(loaded['lexical_causal_integration'])
    for source in HUMIDITY_CAUSAL_SOURCES:
        member_cut.language_adult.world_causal_learning.withdraw_source(source)
    member_cut_surface,_=_surface(member_cut,TARGET_CAUSE)

    permuted=_fresh(loaded['lexical_causal_integration'])
    learner=permuted.language_adult.world_causal_learning
    learner.bindings=dict(reversed(tuple(learner.bindings.items())))
    learner.ecology.pending=dict(reversed(tuple(learner.ecology.pending.items())))
    permuted_surface,_=_surface(permuted,TARGET_CAUSE)

    roots=after.language_adult.leaf(100,(A_ROOTS,))
    unlearned_effects=(*effects,int(roots.identity))
    unlearned=after.language_adult.common_cause_span_expression(*unlearned_effects)
    checkpoint=loaded['lexical_causal_integration'].checkpoint()
    blob=json.dumps(checkpoint,sort_keys=True)
    lived_surfaces=tuple(bytes(event.payload) for event in curriculum.events
                         if event.lane in {'surface','discourse_surface','utterance','authenticated_utterance'})
    transport_relations=checkpoint['transport']['relations']
    implementation=(Path(__file__).with_name('reference_language_mastery_adult_v1.py').read_text()
                    +Path(__file__).with_name('reference_language_mastery_contact_adapter_v1.py').read_text())

    checks={
        'before_reproduces_recursive_binary_fixture':(
            before_surface.count(b' and ')==2 and b', and ' not in before_surface),
        'heldout_three_member_closure_uses_one_learned_exact_arity_factor':(
            len(effects)==3 and bool(exact) and all(after_surface.count(surface)==1 for surface in effect_surfaces)
            and tuple(after_surface.find(surface) for surface in effect_surfaces)==tuple(sorted(after_surface.find(surface) for surface in effect_surfaces))),
        'visible_prose_replaces_binary_chain_with_learned_grouping':(
            after_surface!=before_surface and group_surface.count(b', ')==2
            and group_surface.count(b', and ')==1 and after_surface.count(group_surface)==1
            and group_surface not in before_surface),
        'target_sentence_is_neither_lived_contact_nor_checkpoint_state':(
            after_surface not in lived_surfaces and after_surface.decode() not in blob),
        'construction_source_withdrawal_deoptimizes_without_erasing_causal_truth':(
            withdrawn_surface==before_surface and all(surface in withdrawn_surface for surface in effect_surfaces)),
        'one_causal_member_lesion_removes_only_that_claim':(
            bytes(after.language_adult._leaf_surface(after.language_adult.leaf(100,(A_HUMIDITY,)).identity)) not in member_cut_surface
            and effect_surfaces[0] in member_cut_surface and effect_surfaces[1] in member_cut_surface),
        'unlearned_arity_cannot_borrow_three_port_construction':(
            len(unlearned_effects)==4 and not unlearned),
        'relation_storage_permutation_cannot_choose_group_order':(
            permuted_surface==after_surface),
        'checkpoint_restores_variable_arity_occurrence_without_binary_fields':(
            any(len(row.get('scenes',()))==3 for row in transport_relations)
            and all('left' not in row and 'right' not in row for row in transport_relations)
            and _surface(_fresh(loaded['lexical_causal_integration']),TARGET_CAUSE)[0]==after_surface),
        'punctuation_is_learned_contact_not_host_grouping_rule':(
            "b', and '" not in implementation and 'GROUP_SOURCES' not in implementation),
    }
    failed=[name for name,passed in checks.items() if not passed]
    return {
        'contract':'FOUNDRY_VARIABLE_ARITY_CAUSAL_MESSAGE_PLANNING_'+('GREEN' if not failed else 'RED'),
        'pass':not failed,'reference_only':True,'before':before_surface.decode(errors='replace'),
        'after':after_surface.decode(errors='replace'),'exact_arity':len(effects),
        'checks':checks,'failed':failed,
        'remaining_red':['OPEN_ARITY_ACQUISITION','MULTILINGUAL_COORDINATION',
                         'DIRECT_PARITY','BROAD_HUMAN_DISCOURSE']}

def main():
    started=time.perf_counter();curriculum=canonical_life_function_curriculum_v2()
    with tempfile.TemporaryDirectory(prefix='foundry-variable-arity-') as directory:
        build_cache(directory);marks=tuple(event.payload[0] for event in curriculum.events
                                          if event.lane=='checkpoint_mark')
        loaded={mark:load_mark(directory,mark) for mark in marks}
        result=verify_loaded(loaded,curriculum)
    result['elapsed_ms']=round((time.perf_counter()-started)*1000,3)
    print(result['contract']);print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if result['pass'] else 1

if __name__=='__main__':raise SystemExit(main())
