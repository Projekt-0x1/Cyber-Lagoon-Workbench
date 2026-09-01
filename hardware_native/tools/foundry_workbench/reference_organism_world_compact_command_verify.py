#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

CTX=8801;P=9901;SEE=77001;START=7;GOAL=9
ALICE,BOB,INSPECT,SENSOR,TEST,VALVE=201,202,303,403,302,402
FA=(11,12,13,14);FS=(45,46,47,48)

def u(s):return tuple(s.encode())
def partner(o,p=P):o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),70000+p,True,True)
def feat(o,e,f,src):o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)
def name(o,e,text,src):
    o.contact(CONTACT_SCENE,(7,0,1,e),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)
def clause(o,atoms,text,src):
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);o.contact(CONTACT_SURFACE,u(text),src,True,True)
def see(o,entity,src,independent=True):
    return o.contact(CONTACT_WORLD_STATE,(entity,),src,True,independent)
def speak(o,atoms,src,p=P):
    partner(o,p);o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True);return o.tick()
def stage(o,src=60001):
    o.contact(CONTACT_WORLD_STATE,(START,),src,True,True)
    o.contact(CONTACT_BODY_TARGET,(GOAL,),src+1,True,True)
    o.contact(CONTACT_AFFORDANCES,(INSPECT,TEST),src+2,True,True)

def train(o):
    feat(o,ALICE,FA,8001);feat(o,BOB,(21,22,23,24),8002)
    feat(o,INSPECT,(35,36),8015);feat(o,SENSOR,FS,8016);feat(o,TEST,(33,34),8013);feat(o,VALVE,(43,44),8014)
    for e,s in ((ALICE,'alice'),(BOB,'bob'),(INSPECT,'inspects'),(SENSOR,'sensor'),(TEST,'tests'),(VALVE,'valve')):
        name(o,e,s,10000+e);name(o,e,s,11000+e)
    clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30003);clause(o,(ALICE,INSPECT,SENSOR),'alice inspects the sensor.',30004)
    clause(o,(BOB,TEST,SENSOR),'bob tests the sensor.',30007);clause(o,(BOB,TEST,SENSOR),'bob tests the sensor.',30008)
    clause(o,(BOB,TEST,VALVE),'bob tests the valve.',30005);clause(o,(BOB,TEST,VALVE),'bob tests the valve.',30006)
    assert o.language.template(CTX,3) is not None
    for atom,src in ((INSPECT,91001),(TEST,91002)):
        units=o.language.lexeme(atom)
        if units is None or not o._ground_language_action_recruitment(o.language.lexeme_identity(atom,units),atom,src,1,True):
            raise AssertionError('ground')

def establish(o,p,atoms,src):
    a=speak(o,atoms,src,p)
    if not isinstance(a,ActionV2):raise AssertionError(('establish',a))
    o.contact(CONTACT_CONSEQUENCE,(a.ticket,1),p,True,True);return a

def teach_it(o,p,feature,src):
    partner(o,p)
    for i in range(2):
        o.contact(CONTACT_SCENE,(7,0,1,feature),src+i,True,True);o.contact(CONTACT_SURFACE,u('it'),src+100+i,True,True)
    assert o.language.form(feature,(COND_REINSTATED,),require_conditioned=True)==u('it')

def teach_compact(o,p,atoms,text,src):
    partner(o,p)
    o.contact(CONTACT_SCENE,(7,CTX,len(atoms),*atoms),src,True,True)
    o.contact(CONTACT_SURFACE,u(text),src,True,True)

def compact_command(o,text,src):
    try:return o.contact(CONTACT_SOURCE_UTTERANCE,u(text),src,True,True)
    except ValueError:return None

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,2,4,42,8)
    o=ReferenceOrganismV2(spec);train(o)
    establish(o,P,(BOB,TEST,SENSOR),41001)
    teach_it(o,P,SENSOR,41101)
    teach_compact(o,P,(ALICE,INSPECT,SENSOR),'alice inspects it.',43101)
    teach_compact(o,P,(ALICE,INSPECT,SENSOR),'alice inspects it.',43102)
    checks['talking_is_not_seeing']=o._world_revisions.row_count==0 and SENSOR not in o._world_marked_entities()
    see(o,SENSOR,SEE,True);stage(o)
    before_world=o._world_revisions.row_count
    aid=compact_command(o,'alice inspects it.',P);motor=o.tick() if aid else None
    recruited=o._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    world=() if not isinstance(motor,MotorActionV2) else getattr(motor,'world_occurrences',())
    checks['compact_command_binds_affordance']=isinstance(motor,MotorActionV2) and motor.action_id==INSPECT and aid in motor.source_assertion_ids
    checks['compact_command_does_not_write_world']=o._world_revisions.row_count==before_world and all(isinstance(a,ActionV2) and a.payload!=u('alice inspects it.') for a in o.actions)
    checks['compact_command_recruits_seen_it']=bool(recruited) and {row[0] for row in recruited}=={SENSOR} and set(world)=={row[3] for row in recruited}
    stranger=ReferenceOrganismV2(spec);train(stranger);establish(stranger,P,(BOB,TEST,SENSOR),42001);teach_it(stranger,P,SENSOR,42101)
    teach_compact(stranger,P,(ALICE,INSPECT,SENSOR),'alice inspects it.',42201);teach_compact(stranger,P,(ALICE,INSPECT,SENSOR),'alice inspects it.',42202)
    see(stranger,SENSOR,SEE,True);stage(stranger)
    checks['stranger_cannot_compact_command']=compact_command(stranger,'alice inspects it.',85001) is None
    yoked=ReferenceOrganismV2(spec);train(yoked);establish(yoked,P,(BOB,TEST,SENSOR),43001);teach_it(yoked,P,SENSOR,43101)
    teach_compact(yoked,P,(ALICE,INSPECT,SENSOR),'alice inspects it.',43201);teach_compact(yoked,P,(ALICE,INSPECT,SENSOR),'alice inspects it.',43202)
    see(yoked,SENSOR,SEE,False);stage(yoked)
    yid=compact_command(yoked,'alice inspects it.',P);ym=yoked.tick() if yid else None
    checks['yoked_see_cannot_join_compact_command']=isinstance(ym,MotorActionV2) and ym.action_id==INSPECT and not getattr(ym,'world_occurrences',())
    cut=ReferenceOrganismV2(spec);train(cut);establish(cut,P,(BOB,TEST,SENSOR),44001);teach_it(cut,P,SENSOR,44101)
    teach_compact(cut,P,(ALICE,INSPECT,SENSOR),'alice inspects it.',44201);teach_compact(cut,P,(ALICE,INSPECT,SENSOR),'alice inspects it.',44202)
    see(cut,SENSOR,SEE,True);stage(cut)
    cut.contact(CONTACT_WITHDRAW_SOURCE,(SEE,),88002,True,True);stage(cut,61001)
    cid=compact_command(cut,'alice inspects it.',P);cm=cut.tick() if cid else None
    checks['see_withdrawal_keeps_motor_drops_world']=isinstance(cm,MotorActionV2) and cm.action_id==INSPECT and not getattr(cm,'world_occurrences',())
    explicit=ReferenceOrganismV2(spec);train(explicit);see(explicit,SENSOR,SEE,True);stage(explicit)
    eid=compact_command(explicit,'alice inspects the sensor.',81001);em=explicit.tick() if eid else None
    ew=explicit._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['explicit_command_still_binds']=isinstance(em,MotorActionV2) and em.action_id==INSPECT and eid in em.source_assertion_ids and {row[0] for row in ew}=={SENSOR}
    both=ReferenceOrganismV2(spec);train(both);P_DE=85001
    establish(both,P,(BOB,TEST,SENSOR),45001);teach_it(both,P,SENSOR,45101)
    establish(both,P_DE,(BOB,TEST,SENSOR),45011)
    partner(both,P_DE)
    for src in (P_DE,P_DE+1):
        both.contact(CONTACT_SCENE,(7,0,1,SENSOR),src,True,True);both.contact(CONTACT_SURFACE,u('es'),src,True,True)
    teach_compact(both,P,(ALICE,INSPECT,SENSOR),'alice inspects it.',45201)
    teach_compact(both,P,(ALICE,INSPECT,SENSOR),'alice inspects it.',45202)
    see(both,SENSOR,SEE,True);stage(both)
    bid=compact_command(both,'alice inspects it.',P);bm=both.tick() if bid else None
    bw=both._world_state_occurrences(CTX,(ALICE,INSPECT,SENSOR))
    checks['compact_it_survives_other_form']=isinstance(bm,MotorActionV2) and bm.action_id==INSPECT and bid in bm.source_assertion_ids and {row[0] for row in bw}=={SENSOR} and set(getattr(bm,'world_occurrences',()))=={row[3] for row in bw}
    checks['no_pronoun_opcode']=not hasattr(o,'pronoun') and not hasattr(o,'command') and not hasattr(o,'imagine')
    result={'schema':'0x1.reference-organism-world-compact-command.v1','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,'claim':'COMPACT_COMMAND_KEEPS_SEEN_IT_WITHOUT_PRONOUN_OPCODE_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_WORLD_COMPACT_COMMAND '+('GREEN' if result['pass'] else 'RED')+' compact_command=1 talking_not_seeing=1 pronoun=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
