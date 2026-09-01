#!/usr/bin/env python3
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

BODY=88001;START=7;ACT=101
FL=(11,12,13,14);FR=(71,72,73,74)

def u(s):return tuple(s.encode())
def world(o,state,source):o.contact(CONTACT_WORLD_STATE,tuple(state),source,True,True)
def target(o,state):o.contact(CONTACT_BODY_TARGET,tuple(state),70000,True,True)
def afford(o,*actions):o.contact(CONTACT_AFFORDANCES,tuple(actions),71000,True,True)
def settle_motor(o,a,effect,next_state,source):o.contact(CONTACT_MOTOR_CONSEQUENCE,(a.ticket,effect,len(next_state),*next_state),source,True,True)
def scene(o,context,atoms,source,channel=7):o.contact(CONTACT_SCENE,(channel,context,len(atoms),*atoms),source,True,True)
def surface(o,text,source):o.contact(CONTACT_SURFACE,u(text),source,True,True)
def feat(o,e,f,src):o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)

def experience(o,start,action,next_state,effect,source):
    world(o,start,source);target(o,next_state);afford(o,action);a=o.tick();assert isinstance(a,MotorActionV2) and a.action_id==action;settle_motor(o,a,effect,next_state,source)

def equal_graph(o):
    S0=(10,);S1=(20,);ALT=(25,);S2=(30,40);A1,A2,A3=101,102,103
    for src in (1000,1001):experience(o,S0,A1,S1,1,src)
    for src in (1100,1101):experience(o,S1,A2,S2,1,src)
    for src in (1200,1201):experience(o,S0,A3,ALT,1,src)
    for src in (1300,1301):experience(o,ALT,A2,S2,1,src)
    return S0,S2,A1,A2,A3

def teach_inquiry(o,A1,A3):
    NAME=100;A4,A5=104,105
    for feature,text in ((A1,'left'),(A3,'right'),(A4,'up'),(A5,'down')):
        scene(o,NAME,(feature,),50000+feature);surface(o,text,10000+feature)
        scene(o,NAME,(feature,),60000+feature);surface(o,text,20000+feature)
    scene(o,INQUIRY_CONTEXT,(A1,A3),8100);surface(o,'left or right?',30001)
    scene(o,INQUIRY_CONTEXT,(A4,A5),8101);surface(o,'up or down?',30002)

def mark(o,entity,src=BODY,independent=True,effect=-1):
    o.contact(CONTACT_WORLD_STATE,(entity,),src,True,True)
    o.contact(CONTACT_BODY_STATE,(entity,),src,True,True)
    o.contact(CONTACT_BODY_TARGET,(START,),src+1,True,True)
    o.contact(CONTACT_AFFORDANCES,(ACT,),src+2,True,True)
    motor=o.tick()
    if not isinstance(motor,MotorActionV2):raise AssertionError(('motor',motor))
    nxt=tuple(sorted({int(entity),START}))
    return o.contact(CONTACT_MOTOR_CONSEQUENCE,(motor.ticket,effect,len(nxt),*nxt),src,True,independent)

def ask(o,S0,A1,A3,src=9000):
    o.contact(CONTACT_COMM_CHANNEL,(77,),72000,True,True);world(o,S0,src);target(o,(40,));afford(o,A1,A3);return o.tick()

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,2,4,42,8)
    o=ReferenceOrganismV2(spec);S0,S2,A1,A2,A3=equal_graph(o);teach_inquiry(o,A1,A3)
    feat(o,A1,FL,8001);feat(o,A3,FR,8002)
    learned=mark(o,A1)
    checks['body_return_marks_left']=learned.get('somatic_updates',0)>0 and A1 in o._somatic_marked_entities() and A3 not in o._somatic_marked_entities()
    checks['planning_ints_are_not_soma_referents']=10 not in o._somatic_marked_entities() and 20 not in o._somatic_marked_entities()
    q=ask(o,S0,A1,A3)
    qs=o._somatic_state_occurrences(INQUIRY_CONTEXT,(A1,A3))
    checks['question_is_outer_language']=isinstance(q,ActionV2) and q.payload==u('left or right?') and q.channel==77
    checks['question_carries_live_body']=isinstance(q,ActionV2) and q.body_occurrence==o.body_state_occurrence and q.body_occurrence in q.contributors and q.body_signature!=0 and q.body_source==BODY
    checks['question_recruits_marked_alternative']=bool(qs) and {row[0] for row in qs}=={A1} and all(row[3] in q.contributors for row in qs) and all(row[3] in q.somatic_occurrences for row in qs)
    checks['unmarked_alternative_stays_off']=A3 not in {row[0] for row in qs}
    rows=tuple(o._somatic_revisions.iter_revisions());learned=o.contact(CONTACT_CONSEQUENCE,(q.ticket,-1),9000,True,True)
    checks['settling_question_does_not_write_soma']=learned.get('somatic_updates',0)==0 and tuple(o._somatic_revisions.iter_revisions())==rows and A3 not in o._somatic_marked_entities()
    yoked=ReferenceOrganismV2(spec);equal_graph(yoked);teach_inquiry(yoked,A1,A3);feat(yoked,A1,FL,8001);feat(yoked,A3,FR,8002);mark(yoked,A1,independent=False)
    checks['yoked_body_cannot_write_soma']=yoked._somatic_revisions.row_count==0
    cut=ReferenceOrganismV2(spec);S0c,_,A1c,_,A3c=equal_graph(cut);teach_inquiry(cut,A1c,A3c);feat(cut,A1c,FL,8001);feat(cut,A3c,FR,8002);mark(cut,A1c)
    cut.contact(CONTACT_WITHDRAW_SOURCE,(BODY,),88002,True,True);cq=ask(cut,S0c,A1c,A3c)
    checks['body_withdrawal_keeps_question_drops_soma']=isinstance(cq,ActionV2) and cq.payload==u('left or right?') and not cut._somatic_state_occurrences(INQUIRY_CONTEXT,(A1c,A3c)) and not cq.somatic_occurrences and cq.body_source==0 and not cq.body_occurrence
    faulted=ReferenceOrganismV2(spec);S0f,_,A1f,_,A3f=equal_graph(faulted);teach_inquiry(faulted,A1f,A3f);feat(faulted,A1f,FL,8001);feat(faulted,A3f,FR,8002);mark(faulted,A1f)
    faulted.contact(CONTACT_COMM_CHANNEL,(77,),72000,True,True);world(faulted,S0f,9000);target(faulted,(40,));afford(faulted,A1f,A3f)
    faulted.inject_output_fault(0,ord('X'));bad=faulted.tick()
    if isinstance(bad,ActionV2):faulted.contact(CONTACT_CONSEQUENCE,(bad.ticket,-1),9000,True,True)
    rq=faulted.tick() if isinstance(bad,ActionV2) else None
    rqs=faulted._somatic_state_occurrences(INQUIRY_CONTEXT,(A1f,A3f))
    checks['question_repair_keeps_marked_soma']=isinstance(rq,ActionV2) and rq.repair and rq.payload==u('left or right?') and {row[0] for row in rqs}=={A1f} and all(row[3] in rq.contributors for row in rqs) and all(row[3] in rq.somatic_occurrences for row in rqs) and rq.body_occurrence==faulted.body_state_occurrence and rq.body_source==BODY
    checks['no_qa_opcode']=not hasattr(o,'ask') and not hasattr(o,'feel') and not hasattr(o,'imagine')
    result={'schema':'0x1.reference-organism-somatic-inquiry.v1','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,'question':bytes(q.payload).decode() if isinstance(q,ActionV2) else '','claim':'INQUIRY_SURFACE_RECRUITS_BODY_MARKED_ALTERNATIVE_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_SOMATIC_INQUIRY '+('GREEN' if result['pass'] else 'RED')+' question_recruits_soma=1 unmarked_off=1 qa_opcode=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
