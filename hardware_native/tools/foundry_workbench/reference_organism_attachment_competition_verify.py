#!/usr/bin/env python3
"""Competing resident relation ancestries require causal selection history."""
from __future__ import annotations

import copy
import json
import time

from reference_organism_v2 import *
from reference_organism_structural_dependency_verify import LEFT,WRONG_LEFT,relation_scene,u
from reference_population_v1 import PopulationSpecV1

PARTNER=8842
A1=(101,102,103,104);A2=(201,202,203,204)
B1=(501,502,503,504);B2=(601,602,603,604)
HELD=(301,302,303,304);HELD2=(401,402,403,404)
ALL=(*A1,*A2,*B1,*B2,*HELD,*HELD2)
WORDS={atom:f'w{atom}' for atom in ALL}
A_ACTION_SOURCES=(21001,22001)


def surface(atoms):return ' '.join(WORDS[atom] for atom in atoms)


def name(o,atom):
    for source in (8101,8102):
        o.contact(CONTACT_SCENE,(7,1,1,atom),source+atom,True,True)
        o.contact(CONTACT_SURFACE,u(WORDS[atom]),source,True,True)


def demonstrate(o,atoms,relation,source):
    relation_scene(o,atoms,source,relation)
    o.contact(CONTACT_SURFACE,u(surface(atoms)),source,True,True)


def settle(o,action,effect=1,independent=True):
    return o.contact(CONTACT_CONSEQUENCE,(action.ticket,effect),action.source,True,independent)


def build(reward_b=False,independent_a=True):
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    for atom in ALL:name(o,atom)
    for rank,atoms in enumerate((A1,A2)):
        demonstrate(o,atoms,LEFT,11001+rank*100)
    for rank,atoms in enumerate((B1,B2)):
        demonstrate(o,atoms,WRONG_LEFT,13001+rank*100)
    o.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),14001,True,True)
    for atoms,source in zip((A1,A2),A_ACTION_SOURCES):
        relation_scene(o,atoms,source,LEFT);action=o.tick();assert action is not None
        settle(o,action,1,independent_a)
    if reward_b:
        for rank,atoms in enumerate((B1,B2)):
            relation_scene(o,atoms,23001+rank*100,WRONG_LEFT);action=o.tick();assert action is not None
            settle(o,action)
    return o


def alternatives(o,atoms,source,rewarded_first=False):
    order=(LEFT,WRONG_LEFT) if rewarded_first else (WRONG_LEFT,LEFT)
    scenes=[relation_scene(o,atoms,source+rank*100,relation) for rank,relation in enumerate(order)]
    return scenes,o.tick()


def main():
    started=time.perf_counter();checks={};o=build();checkpoint=copy.deepcopy(o.checkpoint())
    scenes,action=alternatives(o,HELD,30001)
    checks['earned_attachment_beats_contact_order']=(
        isinstance(action,ActionV2) and action.binding_identity==scenes[1].binding_identity
        and action.scene_identity==scenes[1].identity and action.payload==u(surface(HELD)))
    learned={} if not isinstance(action,ActionV2) else settle(o,action)
    checks['winning_relation_occurrences_participate_before_credit']=(
        isinstance(action,ActionV2) and len(action.relation_occurrences)==3
        and all(oid in action.contributors for oid in action.relation_occurrences)
        and any(row[0]==PREF_BINDING for row in action.selection_occurrences)
        and learned.get('credit',0)>0 and learned.get('selection_network_updates',0)==1)

    equal=build(reward_b=True);_equal_scenes,equal_action=alternatives(equal,HELD,31001)
    checks['equal_causal_histories_preserve_attachment_ambiguity']=equal_action is None
    yoked=build(independent_a=False);_yoked_scenes,yoked_action=alternatives(yoked,HELD,32001)
    checks['yoked_history_cannot_choose_attachment']=yoked_action is None

    revised=build(reward_b=True)
    relation_scene(revised,HELD,33001,LEFT);punished=revised.tick();assert punished is not None
    settle(revised,punished,-1)
    reversed_scenes,reversed_action=alternatives(revised,HELD2,33101,rewarded_first=True)
    checks['independent_counterevidence_reverses_attachment_preference']=(
        isinstance(reversed_action,ActionV2)
        and reversed_action.binding_identity==reversed_scenes[1].binding_identity)

    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(PARTNER,),34000,True,True)
    withdrawn.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER+1),34001,True,True)
    _withdrawn_scenes,withdrawn_action=alternatives(withdrawn,HELD,34001)
    checks['credit_source_withdrawal_reopens_attachment']=withdrawn_action is None
    restored=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));restored_digest=restored.digest()
    _restored_scenes,restored_action=alternatives(restored,HELD,35001)
    checks['checkpoint_rebuilds_attachment_competition']=(
        restored_digest==ReferenceOrganismV2.restore(copy.deepcopy(checkpoint)).digest()
        and isinstance(restored_action,ActionV2))

    baseline=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    relation_scene(baseline,HELD,60001,WRONG_LEFT);relation_scene(baseline,HELD,60101,LEFT)
    baseline_action=baseline.tick();baseline_touches=baseline.last_pending_lookup_touches
    quantity=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    quantity_scenes=[relation_scene(quantity,HELD,60001,WRONG_LEFT),
                     relation_scene(quantity,HELD,60101,LEFT)]
    for index in range(512):
        atoms=tuple(100000+index*4+slot for slot in range(4))
        source=40000+index*20;relation_scene(quantity,atoms,source,LEFT)
        quantity.contact(CONTACT_WITHDRAW_SOURCE,(source,),70000+index,True,True)
    quantity_action=quantity.tick()
    touches=quantity.last_pending_lookup_touches
    checks['attachment_nomination_survives_512_decoy_networks']=(
        isinstance(baseline_action,ActionV2) and isinstance(quantity_action,ActionV2)
        and touches<=baseline_touches and touches<=16
        and touches/quantity.population.spec.site_count<0.001)

    result={'schema':'agi.reference-organism-attachment-competition.v1','pass':all(checks.values()),
        'checks':checks,'metrics':{'decoy_networks':512,'pending_touches_before_after':[baseline_touches,touches],
        'selection_revision_bytes':o._selection_revisions.persistent_bytes},'runtime_llm':False,
        'expected_output_ingress':False,'graph_flip':False,'human_language_mastery':False,
        'direct_parity':False,'claim':'CAUSAL_RESIDENT_ATTACHMENT_COMPETITION_REFERENCE_ONLY',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_ATTACHMENT_COMPETITION '+('GREEN' if result['pass'] else 'RED')+
          f" checks={sum(checks.values())}/{len(checks)} decoys=512 touches={touches}")
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)


if __name__=='__main__':main()
