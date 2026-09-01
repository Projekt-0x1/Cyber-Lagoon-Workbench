#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CTX=8801;P=9901;GOAL=7;ACT=101;BODY=88001
ALICE,BOB,INSPECT,SENSOR,TEST,VALVE=201,202,303,403,302,402
FA=(11,12,13,14)

def u(s):return tuple(s.encode())
def partner(o,p=P):o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),70000+p,True,True)
def feat(o,e,f,src):o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)
def name(o,e,text,src):
    o.contact(CONTACT_SCENE,(7,0,1,e),src,True,True);o.contact(CONTACT_SURFACE,u(text),src+1000,True,True)
def clause(o,atoms,text,src):
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);o.contact(CONTACT_SURFACE,u(text),src+2000,True,True)

def train(o):
    feat(o,ALICE,FA,8001);feat(o,BOB,(21,22,23,24),8002);feat(o,INSPECT,(35,36),8015);feat(o,SENSOR,(45,46),8016);feat(o,TEST,(33,34),8013);feat(o,VALVE,(43,44),8014)
    for e,s in ((ALICE,'alice'),(BOB,'bob'),(INSPECT,'inspects'),(SENSOR,'sensor'),(TEST,'tests'),(VALVE,'valve')):
        name(o,e,s,10000+e);name(o,e,s,11000+e)
    clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30003);clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30004)
    clause(o,(ALICE,TEST,VALVE),'alice tests the valve.',30007);clause(o,(ALICE,TEST,VALVE),'alice tests the valve.',30008)
    clause(o,(BOB,TEST,VALVE),'bob tests the valve.',30005);clause(o,(BOB,TEST,VALVE),'bob tests the valve.',30006)
    assert o.language.template(CTX,3) is not None

def body(o,entity,src,independent=True,effect=-1):
    o.contact(CONTACT_WORLD_STATE,(entity,),src,True,True)
    o.contact(CONTACT_BODY_STATE,(entity,),src,True,True)
    o.contact(CONTACT_BODY_TARGET,(GOAL,),src+1,True,True)
    o.contact(CONTACT_AFFORDANCES,(ACT,),src+2,True,True)
    motor=o.tick()
    if not isinstance(motor,MotorActionV2):raise AssertionError(('motor',motor))
    nxt=tuple(sorted({int(entity),GOAL}))
    learned=o.contact(CONTACT_MOTOR_CONSEQUENCE,(motor.ticket,effect,len(nxt),*nxt),src,True,independent)
    return motor,learned

def speak(o,atoms,src,p=P):
    partner(o,p);o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);return o.tick()

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,2,4,42,8)
    o=ReferenceOrganismV2(spec);train(o)
    checks['no_language_action_before_body']=not o.actions
    before_lang=len(o.selection_configuration_revisions)
    motor,learned=body(o,ALICE,BODY,True,-1)
    checks['body_return_writes_sibling_soma']=learned.get('somatic_updates',0)>0 and ALICE in o._somatic_marked_entities()
    checks['body_return_does_not_credit_language_network']=len(o.selection_configuration_revisions)==before_lang and not o.actions
    cp=copy.deepcopy(o.checkpoint())
    en_o=ReferenceOrganismV2.restore(copy.deepcopy(cp));en=speak(en_o,(ALICE,INSPECT,SENSOR),42001);es=en_o._somatic_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['later_language_recruits_body_marker']=isinstance(en,ActionV2) and en.payload==u('alice inspects the sensor.') and bool(es) and all(row[3] in en.contributors for row in es)
    other_o=ReferenceOrganismV2.restore(copy.deepcopy(cp));other=speak(other_o,(ALICE,TEST,VALVE),43001);os=other_o._somatic_state_occurrences(CTX,(ALICE,TEST,VALVE))
    checks['other_construction_recruits_same_referent']=isinstance(other,ActionV2) and bool(os) and all(row[3] in other.contributors for row in os)
    bob_o=ReferenceOrganismV2.restore(copy.deepcopy(cp));bob=speak(bob_o,(BOB,TEST,VALVE),44001)
    checks['unmarked_referent_still_speaks']=isinstance(bob,ActionV2) and bob.payload==u('bob tests the valve.') and not bob_o._somatic_state_occurrences(CTX,(BOB,TEST,VALVE))
    yoked=ReferenceOrganismV2(spec);train(yoked);body(yoked,ALICE,BODY,False,-1)
    checks['yoked_body_cannot_write_soma']=yoked._somatic_revisions.row_count==0 and isinstance(speak(yoked,(ALICE,INSPECT,SENSOR),45001),ActionV2)
    cut=ReferenceOrganismV2.restore(copy.deepcopy(cp));cut.contact(CONTACT_WITHDRAW_SOURCE,(BODY,),88002,True,True);ca=speak(cut,(ALICE,INSPECT,SENSOR),46001)
    checks['body_source_withdrawal_drops_language_recruitment']=isinstance(ca,ActionV2) and not cut._somatic_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['language_is_outer_not_the_model']=not hasattr(o,'translate') and not hasattr(o,'imagine') and not hasattr(o,'valence')
    checks['no_stored_media']=all(k not in cp for k in ('image','audio','png','wav'))
    result={'schema':'0x1.reference-organism-somatic-body-language.v1','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,'claim':'NONLINGUISTIC_BODY_RETURN_BIASES_LATER_LANGUAGE_SURFACES_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_SOMATIC_BODY_LANGUAGE '+('GREEN' if result['pass'] else 'RED')+' body_to_language=1 surfaces=2 media=0 language_outer=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
