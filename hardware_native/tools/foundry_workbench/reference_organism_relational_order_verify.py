#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CTX=9601;NAME=9600;REL=9701;OTHER_REL=9702;PARTNER=9801;LINK_SOURCE=9901
PAIRS=((101,102),(201,202),(301,302));WORDS={101:'ka',102:'lu',201:'mi',202:'no',301:'pa',302:'ru'}
def u(text):return tuple(text.encode())
def surface(o,text,source):return o.contact(CONTACT_SURFACE,u(text),source,True,True)
def partner(o):return o.contact(CONTACT_PARTNER_CONTEXT,(1,7,PARTNER),PARTNER,True,True)
def settle(o,action,effect=1,independent=True):return o.contact(CONTACT_CONSEQUENCE,(action.ticket,effect),PARTNER,True,independent)
def configuration(action):return tuple(row[:3] for row in action.selection_occurrences)

def teach(o):
    for rank,(atom,word) in enumerate(WORDS.items()):
        for repeat in range(2):
            source=10000+rank*10+repeat;o.contact(CONTACT_SCENE,(7,NAME,1,atom),source,True,True);surface(o,word,source)

def event(o,pair,source,atoms=None,relation=REL,demonstration=False,duplicate_path=False):
    left,right=pair
    left_scene=o.contact(CONTACT_SCENE,(7,NAME,1,left),source,True,True);surface(o,WORDS[left],source)
    right_scene=o.contact(CONTACT_SCENE,(7,NAME,1,right),source+1,True,True);surface(o,WORDS[right],source+1)
    ordered=pair if atoms is None else tuple(atoms)
    compound=o.contact(CONTACT_SCENE,(7,CTX,2,*ordered),source+2,True,True)
    o.contact(CONTACT_SCENE_LINK,(left_scene,compound,relation),LINK_SOURCE,True,True)
    o.contact(CONTACT_SCENE_LINK,(compound,right_scene,relation),LINK_SOURCE,True,True)
    if duplicate_path:
        o.contact(CONTACT_SCENE_LINK,(left_scene,compound,relation),LINK_SOURCE+1,True,True)
        o.contact(CONTACT_SCENE_LINK,(compound,right_scene,relation),LINK_SOURCE+1,True,True)
    if demonstration:surface(o,WORDS[left]+' > '+WORDS[right],source+2)
    return compound

def trained():
    o=ReferenceOrganismV2(PopulationSpecV1(32768,2,4,42,8));teach(o)
    event(o,PAIRS[0],20000,demonstration=True);event(o,PAIRS[1],20100,demonstration=True);partner(o)
    actions=[]
    for rank,pair in enumerate(PAIRS[:2]):
        event(o,pair,30000+rank*100);action=o.tick();assert action is not None;settle(o,action);actions.append(action)
    return o,actions

def path_order(surface,left,right):
    raw=bytes(surface);return raw.find(bytes(u(WORDS[left])))<raw.find(bytes(u(WORDS[right])))

def main():
    started=time.perf_counter();checks={};o,trained_actions=trained();checkpoint=o.checkpoint();pair=PAIRS[2]
    checks['settled_sentence_actions_not_checkpointed']=not checkpoint['actions']
    checks['no_semantic_role_runtime']=all(not hasattr(o,name) for name in ('agent_role','patient_role','subject','object'))
    normal=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));event(normal,pair,40000);normal_action=normal.tick()
    shuffled=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));event(shuffled,pair,41000,atoms=tuple(reversed(pair)));shuffled_action=shuffled.tick()
    checks['adapter_permutation_preserves_relation_binding']=normal_action is not None and shuffled_action is not None and normal_action.payload==shuffled_action.payload and path_order(normal_action.payload,*pair)
    checks['heldout_surface_not_training_replay']=normal_action is not None and all(pair[0] not in candidate for candidate in PAIRS[:2]) and normal_action.payload not in {u(WORDS[a]+' > '+WORDS[b]) for a,b in PAIRS[:2]}
    checks['binding_is_actual_selection_network_member']=shuffled_action is not None and any(row[0]==PREF_BINDING for row in shuffled_action.selection_occurrences) and len(shuffled_action.selection_occurrences)==4 and all(row[3] in shuffled_action.contributors for row in shuffled_action.selection_occurrences)
    checks['link_contact_occurrences_participate']=shuffled_action is not None and len(shuffled_action.relation_occurrences)==2 and all(oid in shuffled_action.contributors for oid in shuffled_action.relation_occurrences)
    checks['heldout_configuration_starts_without_credit']=shuffled_action is not None and shuffled._selection_configuration_evidence(shuffled_action.selection_context,configuration(shuffled_action))==(0,0)

    credited=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));event(credited,pair,42000,atoms=tuple(reversed(pair)));ca=credited.tick();learned=settle(credited,ca,1,True)
    checks['independent_consequence_credits_complete_binding_network']=learned.get('selection_network_updates')==1 and credited._selection_configuration_evidence(ca.selection_context,configuration(ca))[0]==1
    yoked=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));event(yoked,pair,43000,atoms=tuple(reversed(pair)));ya=yoked.tick();settle(yoked,ya,1,False)
    checks['yoked_consequence_cannot_credit_binding']=yoked._selection_configuration_evidence(ya.selection_context,configuration(ya))==(0,0)

    missing=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));left,right=pair
    for rank,atom in enumerate(pair):missing.contact(CONTACT_SCENE,(7,NAME,1,atom),44000+rank,True,True);surface(missing,WORDS[atom],44000+rank)
    missing.contact(CONTACT_SCENE,(7,CTX,2,*reversed(pair)),44002,True,True);checks['missing_relation_path_refuses']=missing.tick() is None
    ambiguous=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));event(ambiguous,pair,45000,atoms=tuple(reversed(pair)),duplicate_path=True);checks['ambiguous_relation_path_refuses']=ambiguous.tick() is None
    permuted=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));event(permuted,pair,46000,atoms=tuple(reversed(pair)),relation=OTHER_REL);checks['unearned_relation_root_refuses']=permuted.tick() is None
    withdrawn=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));event(withdrawn,pair,47000,atoms=tuple(reversed(pair)));withdrawn.contact(CONTACT_WITHDRAW_SOURCE,(LINK_SOURCE,),47010,True,True);checks['binding_source_withdrawal_refuses']=withdrawn.tick() is None
    restored=ReferenceOrganismV2.restore(copy.deepcopy(checkpoint));event(restored,pair,48000,atoms=tuple(reversed(pair)));ra=restored.tick()
    checks['checkpoint_preserves_relation_occurrences']=ra is not None and path_order(ra.payload,*pair) and all(link.population_occurrence for link in restored.scene_links)
    checks['sparse_relation_nomination']=restored.last_relation_binding_touches<=4 and restored.last_selection_candidate_touches==1 and restored.last_relation_binding_touches/restored.population.spec.site_count<0.001

    encoded=lambda value:len(json.dumps(value,sort_keys=True,separators=(',',':')).encode())
    result={'schema':'agi.reference-organism-relational-order.v1','pass':all(checks.values()),'checks':checks,'trained_bindings':2,'heldout_bindings':1,'relation_touches':restored.last_relation_binding_touches,'candidate_configurations':restored.last_selection_candidate_touches,'selection_occurrences':0 if ra is None else len(ra.selection_occurrences),'checkpoint_bytes':encoded(checkpoint),'population_checkpoint_bytes':encoded(checkpoint['population']),'language_recipe_bytes':encoded(checkpoint['language']),'relation_receipt_bytes':encoded(checkpoint['scene_links']),'packed_revision_bytes':o._selection_revisions.persistent_bytes,'resident_sites':restored.population.spec.site_count,'runtime_llm':False,'expected_output_path':False,'persistent_settled_action_network':bool(checkpoint['actions']),'claim':'EPHEMERAL_RELATION_PATH_BOUND_LANGUAGE_NETWORK_REFERENCE_ONLY_NOT_HUMAN_SYNTAX_OR_DIRECT_PARITY','elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_RELATIONAL_ORDER '+('GREEN' if result['pass'] else 'RED')+' adapter_invariant=1 binding_network=1 ephemeral_sentence=1 sparse=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)

if __name__=='__main__':main()
