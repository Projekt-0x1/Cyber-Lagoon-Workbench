#!/usr/bin/env python3
"""Recruit one structural hypothesis from raw pair episodes, never authored links."""
from __future__ import annotations

import copy
import json
import time

import reference_organism_v2 as organism
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

META=7711;LEFT_CONTEXT=7712;RIGHT_CONTEXT=7713;PARTNER=7714
D1=(101,102,103,104);D2=(201,202,203,204);HELD=(301,302,303,304);HELD2=(401,402,403,404)
ALL=(*D1,*D2,*HELD,*HELD2);WORDS={atom:f'w{atom}' for atom in ALL}


def u(text):return tuple(text.encode())
def surface(atoms):return ' '.join(WORDS[atom] for atom in atoms)


def name(o,atom):
    for source in (8001,8002):
        o.contact(CONTACT_SCENE,(7,1,1,atom),source+atom,True,True)
        o.contact(CONTACT_SURFACE,u(WORDS[atom]),source,True,True)


def pair_evidence(o,atoms,base,left_context=LEFT_CONTEXT,right_context=RIGHT_CONTEXT):
    sources=[]
    for rank,(pair,context) in enumerate(((atoms[:2],left_context),(atoms[2:],right_context))):
        pair_sources=[]
        for witness in range(2):
            source=base+rank*100+witness;o.contact(CONTACT_SCENE,(7,context,2,*pair),source,True,True)
            o.contact(CONTACT_SURFACE,u(surface(pair)),source,True,True);pair_sources.append(source)
        sources.append(tuple(pair_sources))
    return tuple(sources)


def demonstrate(o,atoms,source):
    o.contact(CONTACT_SCENE,(7,META,4,*atoms),source,True,True)
    o.contact(CONTACT_SURFACE,u(surface(atoms)),source,True,True)


def build():
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8))
    for atom in ALL:name(o,atom)
    for rank,atoms in enumerate((D1,D2)):
        pair_evidence(o,atoms,10001+rank*1000);demonstrate(o,atoms,12001+rank)
    return o


def stage(o,atoms,base):
    sources=pair_evidence(o,atoms,base)
    o.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),base+300,True,True)
    raw=o.contact(CONTACT_SCENE,(7,META,4,*atoms),base+301,True,True)
    return sources,raw,o.tick()


def configuration(action):return tuple(row[:3] for row in action.selection_occurrences)


def main():
    started=time.perf_counter();checks={};o=build();checkpoint=copy.deepcopy(o.checkpoint())
    checks['training_contains_no_authored_scene_links']=not o.scene_links
    held=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));_sources,raw,action=stage(held,HELD,20001)
    checks['raw_pair_history_recruits_heldout_relation_hypothesis']=(
        isinstance(action,ActionV2) and action.binding_identity>0
        and len(action.relation_occurrences)>=6 and action.payload==u(surface(HELD))
        and raw!=action.scene_identity)
    learned={} if not isinstance(action,ActionV2) else held.contact(
        CONTACT_CONSEQUENCE,(action.ticket,1),action.source,True,True)
    checks['raw_pair_and_hypothesis_occurrences_participate_before_credit']=(
        isinstance(action,ActionV2) and all(oid in action.contributors for oid in action.relation_occurrences)
        and any(row[0]==PREF_BINDING for row in action.selection_occurrences)
        and learned.get('credit',0)>0 and learned.get('selection_network_updates',0)==1)

    missing=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    missing.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),21001,True,True)
    missing.contact(CONTACT_SCENE,(7,META,4,*HELD2),21002,True,True)
    checks['missing_pair_history_cannot_manufacture_grouping']=missing.tick() is None
    ambiguous=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    pair_evidence(ambiguous,HELD,22001)
    pair_evidence(ambiguous,(HELD[0],HELD[2],HELD[1],HELD[3]),23001)
    ambiguous.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),24001,True,True)
    ambiguous.contact(CONTACT_SCENE,(7,META,4,*HELD),24002,True,True)
    checks['two_supported_pair_covers_preserve_grouping_ambiguity']=ambiguous.tick() is None

    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));sources=pair_evidence(withdrawn,HELD,25001)
    withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(sources[0][0],),26001,True,True)
    withdrawn.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),26002,True,True)
    withdrawn.contact(CONTACT_SCENE,(7,META,4,*HELD),26003,True,True)
    checks['pair_source_withdrawal_prevents_hypothesis']=withdrawn.tick() is None
    lesion=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));pair_evidence(lesion,HELD,27001)
    if hasattr(lesion,'_pair_episode_index'):lesion._pair_episode_index.clear()
    lesion.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),28001,True,True)
    lesion.contact(CONTACT_SCENE,(7,META,4,*HELD),28002,True,True)
    checks['pair_index_lesion_prevents_hypothesis']=lesion.tick() is None

    yoked=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));_ys,_yr,yoked_action=stage(yoked,HELD,29001)
    yoked_learned={} if not isinstance(yoked_action,ActionV2) else yoked.contact(
        CONTACT_CONSEQUENCE,(yoked_action.ticket,1),yoked_action.source,True,False)
    checks['yoked_consequence_cannot_credit_hypothesis']=(
        isinstance(yoked_action,ActionV2) and not yoked_learned.get('selection_network_updates',0)
        and yoked._selection_configuration_evidence(yoked_action.selection_context,configuration(yoked_action))==(0,0))

    restored=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));restored_digest=restored.digest()
    _rs,_rr,restored_action=stage(restored,HELD,30001)
    checks['checkpoint_rebuilds_pair_index_without_unfolded_hypothesis']=(
        restored_digest==ReferenceOrganismV2.restore(copy.deepcopy(checkpoint)).digest()
        and isinstance(restored_action,ActionV2) and 'pair_episode_index' not in json.dumps(checkpoint))

    baseline=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));_bs,_br,baseline_action=stage(baseline,HELD,31001)
    baseline_touches=getattr(baseline,'last_relation_hypothesis_touches',0)
    quantity=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint))
    for index in range(512):
        atoms=(100000+index*2,100001+index*2)
        source=40000+index*2
        quantity.contact(CONTACT_SCENE,(7,50000+index,2,*atoms),source,True,True)
        quantity.contact(CONTACT_WITHDRAW_SOURCE,(source,),70000+index,True,True)
    _qs,_qr,quantity_action=stage(quantity,HELD,60001)
    touches=getattr(quantity,'last_relation_hypothesis_touches',0)
    checks['pair_nomination_survives_512_unrelated_ecologies']=(
        isinstance(baseline_action,ActionV2) and isinstance(quantity_action,ActionV2)
        and 0<baseline_touches==touches<=8)

    result={'schema':'agi.reference-organism-relation-hypothesis.v1','pass':all(checks.values()),
        'checks':checks,'metrics':{'decoy_pair_ecologies':512,
        'hypothesis_touches_before_after':[baseline_touches,touches]},'runtime_llm':False,
        'expected_output_ingress':False,'authored_scene_links':False,'graph_flip':False,
        'human_language_mastery':False,'direct_parity':False,
        'claim':'RAW_PAIR_EPISODE_RELATION_HYPOTHESIS_REFERENCE_ONLY',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_RELATION_HYPOTHESIS '+('GREEN' if result['pass'] else 'RED')+
          f" checks={sum(checks.values())}/{len(checks)} decoys=512 touches={touches}")
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)


if __name__=='__main__':main()
