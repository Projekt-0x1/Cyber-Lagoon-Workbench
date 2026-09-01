#!/usr/bin/env python3
"""Replace Cartesian clause surplus with certified current causal discourse."""
from __future__ import annotations

import copy
import json
import time

from reference_life_function_curriculum_v1 import (
    ReferenceLifeFunctionRuntimeV2,
    canonical_life_function_curriculum_v2,
    canonical_species_program_v2,
)

language_phenotype_improved=True
visible_language_gain='CERTIFIED_CURRENT_CAUSAL_DISCOURSE_REPLACES_CARTESIAN_CLAUSE_INVENTORY'
RECOVERED='relational_surplus_recovered'  # Historical checkpoint name only.
PRE_DEVELOPMENT='causal_depth_3'
Q=1<<16


def _clone(runtime):
    return type(runtime).restore(runtime.program,copy.deepcopy(runtime.checkpoint()))


def _focus(runtime):
    identity=int(runtime.contact.current_scene)
    scene=runtime.contact.scenes.get(identity)
    return None if scene is None else (int(scene.context),tuple(map(int,scene.atoms)))


def _project(runtime,focus,channel=0,externalize=False):
    adult=runtime.adult;context,atoms=focus
    try:leaf=adult.language_adult.leaf(context,atoms)
    except RuntimeError:return {'surface':b'','programs':(),'factors':(),'coordinates':(),'receipt':None}
    surface,programs,factors=adult.compose_causal_component(
        leaf.identity,with_expression_factors=True,channel=channel)
    receipt=None;coordinates=()
    if externalize and surface:
        public,receipt=adult.externalize_causal_component(leaf.identity,0xFA80,channel)
        if public!=surface:return {'surface':b'','programs':(),'factors':(),'coordinates':(),'receipt':None}
        coordinates=(() if receipt is None else adult._causal_action_coordinates(receipt))
    return {'surface':bytes(surface),'programs':tuple(map(int,programs)),
            'factors':tuple(map(int,factors)),'coordinates':tuple(coordinates),
            'receipt':receipt,'leaf':int(leaf.identity)}


def _withdraw_first_relation(runtime,projection):
    coordinates=tuple(projection['coordinates'])
    if not coordinates:return ()
    causal_receipt=int(coordinates[0][0]);pending=runtime.adult.language_adult.world_causal_learning.ecology.pending.get(causal_receipt)
    sources=tuple(sorted({int(row.source) for row in (() if pending is None else pending.evidence)}))
    for source in sources:runtime.adult.language_adult.world_causal_learning.withdraw_source(source)
    return sources


def _partner_uptake(runtime,focus,projection,channel=0xFA90):
    receipt=projection['receipt'];adult=runtime.adult
    if receipt is None or not receipt.programs or not receipt.factors:return b'',(),False
    rows=adult.causal_message_rows(projection['leaf'])
    acceptance=(adult._causal_self_contained_surface(rows[0],receipt.factors[0]) if rows else None)
    settled=bool(acceptance and adult.settle_causal_dialogue_return(receipt,0xFA80,Q,0,True))
    changed=0
    if settled:
        changed+=int(adult.observe_authenticated_causal_dialogue_contact(acceptance,channel,channel))
        changed+=int(adult.observe_authenticated_causal_dialogue_contact(acceptance,channel,channel))
    revised=_project(runtime,focus,channel)
    return revised['surface'],revised['programs'],bool(settled and changed)


def verify_loaded(loaded,curriculum,final_adult,current_baseline):
    del final_adult,current_baseline
    if RECOVERED not in loaded or PRE_DEVELOPMENT not in loaded:
        return {'pass':False,'checks':{'certified_causal_discourse_marks_are_mandatory':False},
                'failed':['certified_causal_discourse_marks_are_mandatory'],'public':b'','scarce':b''}
    developed=_clone(loaded[RECOVERED]);focus=_focus(developed)
    if focus is None:
        return {'pass':False,'checks':{'current_world_contact_owns_discourse_focus':False},
                'failed':['current_world_contact_owns_discourse_focus'],'public':b'','scarce':b''}
    public=_project(developed,focus,externalize=True)
    lived=tuple(bytes(event.payload) for event in curriculum.events
                if event.lane in {'surface','discourse_surface','utterance','authenticated_utterance'})

    pre=_clone(loaded[PRE_DEVELOPMENT]);pre_public=_project(pre,focus)
    permuted=_clone(loaded[RECOVERED]);learner=permuted.adult.language_adult.world_causal_learning
    learner.bindings=dict(reversed(tuple(learner.bindings.items())))
    permuted_public=_project(permuted,focus)
    withdrawn=_clone(loaded[RECOVERED]);withdrawn_public=_project(withdrawn,focus,externalize=True)
    withdrawn_sources=_withdraw_first_relation(withdrawn,withdrawn_public)
    after_withdrawal=_project(withdrawn,focus)
    social=_clone(loaded[RECOVERED]);social_public=_project(
        social,focus,channel=0xFA90,externalize=True)
    after_social,after_social_programs,social_changed=_partner_uptake(
        social,focus,social_public)
    quiet=_clone(loaded[RECOVERED]);quiet_before=_project(quiet,focus)
    quiet.adult.language_adult.internal_tick();quiet_after=_project(quiet,focus)
    restarted=_clone(loaded[RECOVERED]);restart_public=_project(restarted,focus)
    checkpoint_text=json.dumps(loaded[RECOVERED].checkpoint(),sort_keys=True)

    legacy_features={111,112,113,211,212,311,312,411,412}
    checks={
      'certified_causal_discourse_marks_are_mandatory':True,
      'cartesian_proposition_and_successor_fixture_is_absent_from_life':(
          all(not (event.lane=='scene' and legacy_features.intersection(map(int,event.payload[1:])))
              for event in curriculum.events)
          and all(not (event.lane=='relation_basis_edge' and 908000<=int(event.source)<909000)
                  for event in curriculum.events)),
      'current_world_contact_recruits_multi_relation_certified_composition':(
          len(public['programs'])>=3 and len(public['coordinates'])==len(public['programs'])
          and public['surface'] and all(min(map(int,row))>0 for row in public['coordinates'])),
      'public_causal_composition_is_productive_not_contact_replay':(
          all(public['surface'] not in contact for contact in lived)),
      'developmental_history_changes_same_world_focus':(
          len(pre_public['programs'])<len(public['programs']) and pre_public['surface']!=public['surface']),
      'causal_evidence_withdrawal_changes_current_composition':(
          bool(withdrawn_sources) and after_withdrawal['surface']!=public['surface']
          and len(after_withdrawal['programs'])<len(public['programs'])),
      'candidate_enumeration_permutation_cannot_choose_public_order':(
          permuted_public['surface']==public['surface']
          and permuted_public['programs']==public['programs']),
      'partner_uptake_replans_instead_of_prefix_chopping':(
          social_changed and after_social and after_social!=public['surface']
          and len(after_social_programs)<len(public['programs'])
          and not public['surface'].startswith(after_social)),
      'quiet_without_new_evidence_cannot_grow_or_rewrite_discourse':(
          quiet_after['surface']==quiet_before['surface']
          and quiet_after['programs']==quiet_before['programs']),
      'restart_preserves_causal_composition_without_stored_paragraph':(
          restart_public['surface']==public['surface']
          and restart_public['programs']==public['programs']
          and public['surface'].decode(errors='replace') not in checkpoint_text),
    }
    failed=[name for name,passed in checks.items() if not passed]
    return {'pass':not failed,'checks':checks,'failed':failed,
            'public':public['surface'],'scarce':after_social,
            'scarce_programs':after_social_programs,
            'programs':public['programs'],'coordinates':public['coordinates'],
            'retired_fixture':{'propositions':24,'authored_successor_edges':23,
                               'fixed_quiet_ticks':128,'byte_threshold':512}}


def main():
    started=time.perf_counter();curriculum=canonical_life_function_curriculum_v2()
    runtime=ReferenceLifeFunctionRuntimeV2(canonical_species_program_v2());loaded={}
    for event in curriculum.events:
        runtime.apply(event)
        if event.lane=='checkpoint_mark' and event.payload[0] in {RECOVERED,PRE_DEVELOPMENT}:
            loaded[event.payload[0]]=runtime.fork_for_probe()
    result=verify_loaded(loaded,curriculum,runtime.adult,b'')
    result.update({'contract':'FOUNDRY_CERTIFIED_CAUSAL_DISCOURSE_COMPOSITION_'+
                   ('GREEN' if result['pass'] else 'RED'),
                   'language_phenotype_improved':result['pass'],
                   'future_update_authority_preserved':True,
                   'visible_language_gain':visible_language_gain,
                   'events_lived_once':len(curriculum.events),
                   'visible':result['public'].decode(errors='replace'),
                   'elapsed_ms':round((time.perf_counter()-started)*1000,3)})
    printable=dict(result);printable['public']=result['public'].decode(errors='replace')
    printable['scarce']=result['scarce'].decode(errors='replace')
    print(result['contract']);print('visible_language_gain='+visible_language_gain)
    print(json.dumps(printable,indent=2,sort_keys=True));return 0 if result['pass'] else 1


if __name__=='__main__':raise SystemExit(main())
