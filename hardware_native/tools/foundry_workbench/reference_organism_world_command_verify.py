#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CTX=8801;P=9901;SEE=77001;START=7;GOAL=9
ALICE,BOB,INSPECT,SENSOR,TEST,VALVE=201,202,303,403,302,402
FA=(11,12,13,14)

def u(s):return tuple(s.encode())
def feat(o,e,f,src):o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)
def name(o,e,text,src):
    o.contact(CONTACT_SCENE,(7,0,1,e),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)
def clause(o,atoms,text,src):
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)
def see(o,entity,src,independent=True):
    return o.contact(CONTACT_WORLD_STATE,(entity,),src,True,independent)
def stage(o,src=60001):
    o.contact(CONTACT_WORLD_STATE,(START,),src,True,True)
    o.contact(CONTACT_BODY_TARGET,(GOAL,),src+1,True,True)
    o.contact(CONTACT_AFFORDANCES,(INSPECT,TEST),src+2,True,True)
def command(o,text,src,independent=True):
    return o.contact(CONTACT_SOURCE_UTTERANCE,u(text),src,True,independent)

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

def hear_then_act(o,text,src=81001):
    aid=command(o,text,src,True);motor=o.tick();return aid,motor

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,2,4,42,8)
    o=ReferenceOrganismV2(spec);train(o);see(o,ALICE,SEE,True);stage(o)
    before_world=o._world_revisions.row_count;before_lang=len(o.selection_configuration_revisions)
    aid,motor=hear_then_act(o,'alice inspects the sensor.')
    world=() if not isinstance(motor,MotorActionV2) else getattr(motor,'world_occurrences',())
    recruited=o._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['command_binds_unique_affordance']=isinstance(motor,MotorActionV2) and motor.action_id==INSPECT and aid in motor.source_assertion_ids
    checks['command_does_not_write_world_or_language']=o._world_revisions.row_count==before_world and len(o.selection_configuration_revisions)==before_lang and not o.actions
    checks['command_recruits_seen_world']=bool(recruited) and set(world)=={row[3] for row in recruited} and ALICE in o._world_marked_entities()
    checks['unseen_sensor_stays_off']=all(row[0]==ALICE for row in recruited)
    numeric=ReferenceOrganismV2(spec);train(numeric);see(numeric,ALICE,SEE,True);stage(numeric)
    nid=numeric.contact(CONTACT_SOURCE_ASSERTION,(INSPECT,),82001,True,True);nm=numeric.tick()
    checks['bare_assertion_is_not_a_command']=isinstance(nm,MotorActionV2) and nm.action_id==INSPECT and nid in nm.source_assertion_ids and not getattr(nm,'world_occurrences',())
    bob=ReferenceOrganismV2(spec);train(bob);see(bob,ALICE,SEE,True);stage(bob)
    _bid,bm=hear_then_act(bob,'bob tests the valve.',83001)
    checks['unseen_command_referents_still_act']=isinstance(bm,MotorActionV2) and bm.action_id==TEST and not getattr(bm,'world_occurrences',())
    yoked=ReferenceOrganismV2(spec);train(yoked);see(yoked,ALICE,SEE,False);stage(yoked)
    _yid,ym=hear_then_act(yoked,'alice inspects the sensor.',84001)
    checks['yoked_see_cannot_join_command']=isinstance(ym,MotorActionV2) and ym.action_id==INSPECT and not getattr(ym,'world_occurrences',())
    cut=ReferenceOrganismV2(spec);train(cut);see(cut,ALICE,SEE,True);stage(cut);cut.contact(CONTACT_WITHDRAW_SOURCE,(SEE,),88002,True,True);stage(cut,61001)
    _cid,cm=hear_then_act(cut,'alice inspects the sensor.',85001)
    checks['see_withdrawal_keeps_motor_drops_world']=isinstance(cm,MotorActionV2) and cm.action_id==INSPECT and not getattr(cm,'world_occurrences',())
    aim=ReferenceOrganismV2(spec);train(aim);see(aim,ALICE,SEE,True);stage(aim);DEAD=77001
    goal,aff=aim.body_target,set(aim.affordances)
    aim.contact(CONTACT_WITHDRAW_SOURCE,(DEAD,),88022,True,True)
    aim.contact(CONTACT_BODY_TARGET,(ALICE,),DEAD,True,True)
    aim.contact(CONTACT_AFFORDANCES,(TEST,),DEAD,True,True)
    checks['withdrawn_source_cannot_retarget']=aim.body_target==goal and aim.affordances==aff
    mute=ReferenceOrganismV2(spec);train(mute);see(mute,ALICE,SEE,True);stage(mute);TEACH=81001
    before=len(mute.source_assertions)
    mute.contact(CONTACT_WITHDRAW_SOURCE,(TEACH,),88021,True,True)
    try:
        command(mute,'alice inspects the sensor.',TEACH);heard=True
    except ValueError:
        heard=False
    checks['withdrawn_teacher_cannot_command']=not heard and len(mute.source_assertions)==before
    checks['language_is_outer']=not hasattr(o,'command') and not hasattr(o,'imagine') and not hasattr(o,'translate')
    result={'schema':'0x1.reference-organism-world-command.v1','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,'claim':'COMMAND_BINDS_SHARED_WORLD_TO_ACTION_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_WORLD_COMMAND '+('GREEN' if result['pass'] else 'RED')+' command_to_motor=1 world_join=1 language_outer=1')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
