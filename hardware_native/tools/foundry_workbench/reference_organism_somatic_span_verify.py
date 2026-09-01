#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1
from reference_organism_discourse_verify import train,scene,surface,partner,settle,u,CTX,REL_NEXT

SENSOR,VALVE=401,402;BODY=88001;START=7;ACT=101
FS=(45,46,47,48);FV=(75,76,77,78)

def feat(o,e,f,src):o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)

def teach_span(o):
    left_a=scene(o,(101,201,301,401),70001);right_a=scene(o,(102,202,302,402),70002)
    o.contact(CONTACT_SCENE_LINK,(left_a,right_a,REL_NEXT),71001,True,True)
    o.contact(CONTACT_DISCOURSE_SURFACE,u('the careful engineer tests the sensor. then the quiet technician inspects the valve.'),71001)
    left_b=scene(o,(102,202,302,402),70003);right_b=scene(o,(101,201,301,401),70004)
    o.contact(CONTACT_SCENE_LINK,(left_b,right_b,REL_NEXT),71002,True,True)
    o.contact(CONTACT_DISCOURSE_SURFACE,u('the quiet technician inspects the valve. then the careful engineer tests the sensor.'),71002)
    assert o.language.span_template(REL_NEXT,2) is not None

def mark_sensor(o,src=BODY,independent=True,effect=-1):
    o.contact(CONTACT_WORLD_STATE,(SENSOR,),src,True,True)
    o.contact(CONTACT_BODY_STATE,(SENSOR,),src,True,True)
    o.contact(CONTACT_BODY_TARGET,(START,),src+1,True,True)
    o.contact(CONTACT_AFFORDANCES,(ACT,),src+2,True,True)
    motor=o.tick()
    if not isinstance(motor,MotorActionV2):raise AssertionError(('motor',motor))
    nxt=tuple(sorted({SENSOR,START}))
    learned=o.contact(CONTACT_MOTOR_CONSEQUENCE,(motor.ticket,effect,len(nxt),*nxt),src,True,independent)
    o.contact(CONTACT_WORLD_STATE,(START,),src+3,True,True)
    o.contact(CONTACT_BODY_TARGET,(START,),src+4,True,True)
    return motor,learned

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,2,4,42,8);P=9901
    o=ReferenceOrganismV2(spec);train(o);partner(o,P);teach_span(o)
    feat(o,SENSOR,FS,8001);feat(o,VALVE,FV,8002)
    _motor,learned=mark_sensor(o)
    checks['body_return_marks_sensor_not_valve']=learned.get('somatic_updates',0)>0 and SENSOR in o._somatic_marked_entities() and VALVE not in o._somatic_marked_entities()
    left=scene(o,(101,201,301,401),73001);first=o.tick();fs=o._somatic_state_occurrences(CTX,(101,201,301,401))
    checks['first_clause_recruits_sensor_soma']=isinstance(first,ActionV2) and {row[0] for row in fs}=={SENSOR} and all(row[3] in first.contributors for row in fs) and all(oid in first.somatic_occurrences for oid in (row[3] for row in fs))
    settle(o,first,P,1)
    right=scene(o,(102,202,302,402),73002);o.contact(CONTACT_SCENE_LINK,(left,right,REL_NEXT),73003,True,True)
    cont=o.tick();cs=o._somatic_state_occurrences(CTX,(101,201,301,401))
    checks['span_suffix_recruits_prior_soma']=isinstance(cont,ActionV2) and cont.payload==u(' then the quiet technician inspects the valve.') and any(k==PREF_SPAN for k,_,_,_ in cont.selection_occurrences) and {row[0] for row in cs}=={SENSOR} and all(row[3] in cont.contributors for row in cs) and all(oid in cont.somatic_occurrences for oid in (row[3] for row in cs))
    checks['current_unmarked_valve_stays_off']=VALVE not in {row[0] for row in cs}
    fix=ReferenceOrganismV2(spec);train(fix);partner(fix,P);teach_span(fix);feat(fix,SENSOR,FS,8001);feat(fix,VALVE,FV,8002);mark_sensor(fix)
    fl=scene(fix,(101,201,301,401),74001);ff=fix.tick();settle(fix,ff,P,1)
    fr=scene(fix,(102,202,302,402),74002);fix.contact(CONTACT_SCENE_LINK,(fl,fr,REL_NEXT),74003,True,True)
    fix.inject_output_fault(0,ord('X'));fb=fix.tick();prior=fix._somatic_state_occurrences(CTX,(101,201,301,401))
    settle(fix,fb,P,-1);frep=fix.tick()
    checks['span_repair_keeps_prior_soma']=isinstance(frep,ActionV2) and frep.repair and frep.span_identity and bool(prior) and all(row[3] in frep.contributors for row in prior) and all(row[3] in frep.somatic_occurrences for row in prior) and VALVE not in {row[0] for row in prior}
    moved=ReferenceOrganismV2(spec);train(moved);partner(moved,P);teach_span(moved);feat(moved,SENSOR,FS,8001);feat(moved,VALVE,FV,8002);mark_sensor(moved)
    ml=scene(moved,(101,201,301,401),75001);mf=moved.tick();settle(moved,mf,P,1)
    mr=scene(moved,(102,202,302,402),75002);moved.contact(CONTACT_SCENE_LINK,(ml,mr,REL_NEXT),75003,True,True)
    moved.inject_output_fault(0,ord('X'));mb=moved.tick();mprior=moved._somatic_state_occurrences(CTX,(101,201,301,401))
    settle(moved,mb,P,-1);partner(moved,9902);mrep=moved.tick()
    checks['span_repair_keeps_prior_soma_after_partner_switch']=isinstance(mrep,ActionV2) and mrep.repair and mrep.source==P and bool(mprior) and all(row[3] in mrep.contributors for row in mprior) and all(row[3] in mrep.somatic_occurrences for row in mprior)
    settle(o,cont,P,-1)
    right2=scene(o,(102,202,302,402),73004);o.contact(CONTACT_SCENE_LINK,(left,right2,REL_NEXT),73005,True,True)
    leaf=o.tick();ls=o._somatic_state_occurrences(CTX,(102,202,302,402))
    checks['leaf_fallback_drops_prior_soma']=isinstance(leaf,ActionV2) and leaf.payload==u('the quiet technician inspects the valve.') and not any(k==PREF_SPAN for k,_,_,_ in leaf.selection_occurrences) and not ls and not leaf.somatic_occurrences
    yoked=ReferenceOrganismV2(spec);train(yoked);partner(yoked,P);teach_span(yoked);feat(yoked,SENSOR,FS,8001);mark_sensor(yoked,independent=False)
    checks['yoked_body_cannot_write_soma']=yoked._somatic_revisions.row_count==0
    cut=ReferenceOrganismV2(spec);train(cut);partner(cut,P);teach_span(cut);feat(cut,SENSOR,FS,8001);feat(cut,VALVE,FV,8002);mark_sensor(cut)
    cl=scene(cut,(101,201,301,401),83001);ca=cut.tick();settle(cut,ca,P,1)
    cut.contact(CONTACT_WITHDRAW_SOURCE,(BODY,),88002,True,True)
    cut.contact(CONTACT_WORLD_STATE,(START,),88003,True,True);cut.contact(CONTACT_BODY_TARGET,(START,),88004,True,True)
    cr=scene(cut,(102,202,302,402),83002);cut.contact(CONTACT_SCENE_LINK,(cl,cr,REL_NEXT),83003,True,True);cc=cut.tick()
    checks['body_withdrawal_keeps_span_drops_soma']=isinstance(cc,ActionV2) and b'then' in bytes(cc.payload) and not cut._somatic_state_occurrences(CTX,(101,201,301,401)) and not cc.somatic_occurrences
    checks['language_is_outer']=not hasattr(o,'feel') and not hasattr(o,'then') and PREF_SOMA not in (PREF_TEMPLATE,PREF_LEXEME,PREF_SPAN)
    result={'schema':'0x1.reference-organism-somatic-span.v1','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,'span':bytes(cont.payload).decode() if isinstance(cont,ActionV2) else '','claim':'SPAN_CLOSURE_RECRUITS_PRIOR_SOMA_LEAF_DOES_NOT_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_SOMATIC_SPAN '+('GREEN' if result['pass'] else 'RED')+' span_keeps_prior=1 leaf_drops=1 body=1 language_outer=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
