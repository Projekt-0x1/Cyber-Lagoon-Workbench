#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CTX=8801;EN,EN2,DE,DE2,NEUTRAL=1101,1102,2101,2102,9999;SEE=77001
ALICE,BOB,INSPECT,SENSOR,TEST,VALVE=201,202,303,403,302,402
FA=(11,12,13,14)

def u(s):return tuple(s.encode())
def partner(o,p):o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),70000+p,True,True)
def feat(o,e,f,src):o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)
def name(o,e,text,src):
    o.contact(CONTACT_SCENE,(7,0,1,e),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)
def clause(o,atoms,text,src):
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)
def see(o,entity,src,independent=True):
    return o.contact(CONTACT_WORLD_STATE,(entity,),src,True,independent)
def speak(o,atoms,src,p):
    partner(o,p);o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);return o.tick()

def train(o):
    feat(o,ALICE,FA,8001);feat(o,BOB,(21,22,23,24),8002)
    feat(o,INSPECT,(35,36),8015);feat(o,SENSOR,(45,46),8016);feat(o,TEST,(33,34),8013);feat(o,VALVE,(43,44),8014)
    for src,alice,inspect,sensor in ((EN,'alice','inspects','sensor'),(EN2,'alice','inspects','sensor'),
                                     (DE,'Alice','prueft','Sensor'),(DE2,'Alice','prueft','Sensor')):
        name(o,ALICE,alice,src);name(o,INSPECT,inspect,src);name(o,SENSOR,sensor,src)
    for src,bob,test,valve in ((EN,'bob','tests','valve'),(EN2,'bob','tests','valve')):
        name(o,BOB,bob,src);name(o,TEST,test,src);name(o,VALVE,valve,src)
    clause(o,(ALICE,INSPECT,SENSOR),'alice inspects sensor.',EN)
    clause(o,(ALICE,INSPECT,SENSOR),'alice inspects sensor.',EN2)
    clause(o,(ALICE,INSPECT,SENSOR),'Alice prueft Sensor.',DE)
    clause(o,(ALICE,INSPECT,SENSOR),'Alice prueft Sensor.',DE2)
    clause(o,(BOB,TEST,VALVE),'bob tests valve.',EN)
    clause(o,(BOB,TEST,VALVE),'bob tests valve.',EN2)
    assert o.language.template(CTX,3) is not None

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,2,4,42,8)
    o=ReferenceOrganismV2(spec);train(o)
    checks['no_language_action_before_world']=not o.actions
    before_lang=len(o.selection_configuration_revisions)
    see(o,ALICE,SEE,True)
    checks['one_world_network_not_two_language_models']=o._world_revisions.row_count==1 and ALICE in o._world_marked_entities() and len(o.selection_configuration_revisions)==before_lang
    cp=copy.deepcopy(o.checkpoint())
    en_o=ReferenceOrganismV2.restore(copy.deepcopy(cp));en=speak(en_o,(ALICE,INSPECT,SENSOR),42001,EN);ew=en_o._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    de_o=ReferenceOrganismV2.restore(copy.deepcopy(cp));de=speak(de_o,(ALICE,INSPECT,SENSOR),42002,DE);dw=de_o._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['english_partner_emits_english']=isinstance(en,ActionV2) and en.payload==u('alice inspects sensor.')
    checks['german_partner_emits_german']=isinstance(de,ActionV2) and de.payload==u('Alice prueft Sensor.')
    checks['both_surfaces_recruit_same_world_entities']=bool(ew) and {row[0] for row in ew}=={row[0] for row in dw}=={ALICE} and all(row[3] in en.contributors for row in ew) and all(row[3] in de.contributors for row in dw)
    checks['world_is_not_a_language_member']=not any(k==PREF_VIEW for k,_,_,_ in en.selection_occurrences) and not any(k==PREF_VIEW for k,_,_,_ in de.selection_occurrences)
    neu=ReferenceOrganismV2.restore(copy.deepcopy(cp));neutral=speak(neu,(ALICE,INSPECT,SENSOR),43001,NEUTRAL)
    checks['neutral_partner_refuses_surface_tie']=neutral is None
    bob_o=ReferenceOrganismV2.restore(copy.deepcopy(cp));bob=speak(bob_o,(BOB,TEST,VALVE),44001,EN)
    checks['unmarked_referent_still_speaks']=isinstance(bob,ActionV2) and bob.payload==u('bob tests valve.') and not bob_o._world_state_occurrences(CTX,(BOB,TEST,VALVE))
    yoked=ReferenceOrganismV2(spec);train(yoked);see(yoked,ALICE,SEE,False)
    checks['yoked_see_cannot_write_world']=yoked._world_revisions.row_count==0
    cut=ReferenceOrganismV2.restore(copy.deepcopy(cp));cut.contact(CONTACT_WITHDRAW_SOURCE,(SEE,),88002,True,True)
    checks['world_withdrawal_drops_both_surfaces']=isinstance(speak(cut,(ALICE,INSPECT,SENSOR),45001,EN),ActionV2) and not cut._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['no_translate_opcode']=not hasattr(o,'translate') and not hasattr(o,'language_id') and not hasattr(o,'english') and not hasattr(o,'german')
    checks['no_stored_media']=all(k not in cp for k in ('image','audio','png','wav'))
    result={'schema':'0x1.reference-organism-world-multilingual.v1','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,'english':bytes(en.payload).decode() if isinstance(en,ActionV2) else '','german':bytes(de.payload).decode() if isinstance(de,ActionV2) else '','claim':'MULTILINGUAL_SURFACES_CONVERGE_ON_ONE_SHARED_WORLD_NETWORK_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_WORLD_MULTILINGUAL '+('GREEN' if result['pass'] else 'RED')+' shared_world=1 surfaces=2 translate=0 media=0 language_outer=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
