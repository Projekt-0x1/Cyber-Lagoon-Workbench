#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CTX=8801;START=7;GOAL=9;ACT=101;BODY=88001
ALICE,BOB,INSPECT,SENSOR,TEST,VALVE=201,202,303,403,302,402
FA=(11,12,13,14)

def u(s):return tuple(s.encode())
def feat(o,e,f,src):o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)
def name(o,e,text,src):
    o.contact(CONTACT_SCENE,(7,0,1,e),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)
def clause(o,atoms,text,src):
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)
def stage(o,src=60001):
    o.contact(CONTACT_WORLD_STATE,(START,),src,True,True)
    o.contact(CONTACT_BODY_TARGET,(GOAL,),src+1,True,True)
    o.contact(CONTACT_AFFORDANCES,(INSPECT,TEST),src+2,True,True)

def train(o):
    feat(o,ALICE,FA,8001);feat(o,BOB,(21,22,23,24),8002)
    feat(o,INSPECT,(35,36),8015);feat(o,SENSOR,(45,46),8016);feat(o,TEST,(33,34),8013);feat(o,VALVE,(43,44),8014)
    for e,s in ((ALICE,'alice'),(BOB,'bob'),(INSPECT,'inspects'),(SENSOR,'sensor'),(TEST,'tests'),(VALVE,'valve')):
        name(o,e,s,10000+e);name(o,e,s,11000+e)
    clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30003);clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30004)
    clause(o,(BOB,TEST,VALVE),'bob tests the valve.',30005);clause(o,(BOB,TEST,VALVE),'bob tests the valve.',30006)
    assert o.language.template(CTX,3) is not None
    for atom,src in ((INSPECT,91001),(TEST,91002)):
        units=o.language.lexeme(atom)
        if units is None or not o._ground_language_action_recruitment(o.language.lexeme_identity(atom,units),atom,src,1,True):
            raise AssertionError('ground')

def mark(o,entity,src=BODY,independent=True,effect=-1):
    o.contact(CONTACT_WORLD_STATE,(entity,),src,True,True)
    o.contact(CONTACT_BODY_STATE,(entity,),src,True,True)
    o.contact(CONTACT_BODY_TARGET,(START,),src+1,True,True)
    o.contact(CONTACT_AFFORDANCES,(ACT,),src+2,True,True)
    motor=o.tick()
    if not isinstance(motor,MotorActionV2):raise AssertionError(('motor',motor))
    nxt=tuple(sorted({int(entity),START}))
    learned=o.contact(CONTACT_MOTOR_CONSEQUENCE,(motor.ticket,effect,len(nxt),*nxt),src,True,independent)
    stage(o,src+10)
    return motor,learned

def command(o,text,src=81001):
    return o.contact(CONTACT_SOURCE_UTTERANCE,u(text),src,True,True)

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,2,4,42,8)
    o=ReferenceOrganismV2(spec);train(o)
    _m,learned=mark(o,ALICE)
    checks['body_return_marks_alice']=learned.get('somatic_updates',0)>0 and ALICE in o._somatic_marked_entities()
    before_lang=len(o.selection_configuration_revisions)
    aid=command(o,'alice inspects the sensor.');motor=o.tick()
    soma=() if not isinstance(motor,MotorActionV2) else getattr(motor,'somatic_occurrences',())
    recruited=o._somatic_state_occurrences(0,(ALICE,INSPECT,SENSOR))
    checks['command_binds_unique_affordance']=isinstance(motor,MotorActionV2) and motor.action_id==INSPECT and aid in motor.source_assertion_ids
    checks['command_does_not_credit_language']=len(o.selection_configuration_revisions)==before_lang and not o.actions
    checks['command_recruits_body_marker']=bool(recruited) and {row[0] for row in recruited}=={ALICE} and set(soma)=={row[3] for row in recruited}
    checks['command_still_recruits_seen_world']=bool(getattr(motor,'world_occurrences',())) and ALICE in o._world_marked_entities()
    numeric=ReferenceOrganismV2(spec);train(numeric);mark(numeric,ALICE)
    nid=numeric.contact(CONTACT_SOURCE_ASSERTION,(INSPECT,),82001,True,True);nm=numeric.tick()
    checks['bare_assertion_is_not_a_command']=isinstance(nm,MotorActionV2) and nm.action_id==INSPECT and nid in nm.source_assertion_ids and not getattr(nm,'somatic_occurrences',())
    bob=ReferenceOrganismV2(spec);train(bob);mark(bob,ALICE);bid=command(bob,'bob tests the valve.',83001);bm=bob.tick()
    checks['unmarked_command_referents_still_act']=isinstance(bm,MotorActionV2) and bm.action_id==TEST and bid in bm.source_assertion_ids and not getattr(bm,'somatic_occurrences',())
    yoked=ReferenceOrganismV2(spec);train(yoked);mark(yoked,ALICE,independent=False)
    checks['yoked_body_cannot_write_soma']=yoked._somatic_revisions.row_count==0
    yid=command(yoked,'alice inspects the sensor.',84001);ym=yoked.tick()
    checks['yoked_body_cannot_join_command']=isinstance(ym,MotorActionV2) and ym.action_id==INSPECT and not getattr(ym,'somatic_occurrences',())
    cut=ReferenceOrganismV2(spec);train(cut);mark(cut,ALICE)
    cut.contact(CONTACT_WITHDRAW_SOURCE,(BODY,),88002,True,True);stage(cut,61001)
    cid=command(cut,'alice inspects the sensor.',85001);cm=cut.tick()
    checks['body_withdrawal_keeps_motor_drops_soma']=isinstance(cm,MotorActionV2) and cm.action_id==INSPECT and not getattr(cm,'somatic_occurrences',())
    held=ReferenceOrganismV2(spec);train(held);mark(held,ALICE);DEAD=77001
    src,occ=held.body_state_source,held.body_state_occurrence
    held.contact(CONTACT_WITHDRAW_SOURCE,(DEAD,),88010,True,True)
    held.contact(CONTACT_BODY_STATE,(BOB,),DEAD,True,True)
    checks['withdrawn_body_cannot_replace_live_body']=held.body_state_source==src and held.body_state_occurrence==occ
    checks['language_is_outer']=not hasattr(o,'feel') and not hasattr(o,'command') and PREF_SOMA not in (PREF_TEMPLATE,PREF_LEXEME)
    result={'schema':'0x1.reference-organism-somatic-command.v1','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,'claim':'COMMAND_RECRUITS_BODY_MARKER_WITHOUT_FEEL_OPCODE_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_SOMATIC_COMMAND '+('GREEN' if result['pass'] else 'RED')+' command_to_motor=1 soma_join=1 language_outer=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
