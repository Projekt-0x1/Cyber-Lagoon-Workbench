#!/usr/bin/env python3
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

SEE=77001;EN,EN2,DE,DE2=1101,1102,2101,2102
FL=(11,12,13,14);FR=(71,72,73,74)

def u(s):return tuple(s.encode())
def world(o,state,source):o.contact(CONTACT_WORLD_STATE,tuple(state),source,True,True)
def target(o,state):o.contact(CONTACT_BODY_TARGET,tuple(state),70000,True,True)
def afford(o,*actions):o.contact(CONTACT_AFFORDANCES,tuple(actions),71000,True,True)
def settle_motor(o,a,effect,next_state,source):o.contact(CONTACT_MOTOR_CONSEQUENCE,(a.ticket,effect,len(next_state),*next_state),source,True,True)
def scene(o,context,atoms,source,channel=7):o.contact(CONTACT_SCENE,(channel,context,len(atoms),*atoms),source,True,True)
def surface(o,text,source):o.contact(CONTACT_SURFACE,u(text),source,True,True)
def feat(o,e,f,src):o.contact(CONTACT_ENTITY_FEATURES,(e,len(f),*f),src,True,True)
def see(o,entity,src,independent=True):return o.contact(CONTACT_WORLD_STATE,(entity,),src,True,independent)

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

def ask(o,S0,A1,A3,src=9000):
    o.contact(CONTACT_COMM_CHANNEL,(77,),72000,True,True);world(o,S0,src);target(o,(40,));afford(o,A1,A3);return o.tick()

def partner(o,p):o.contact(CONTACT_PARTNER_CONTEXT,(1,7,p),70000+p,True,True)

def teach_named_inquiry(o,A1,A3,left,right,question,s0,s1):
    NAME=100
    for feature,text in ((A1,left),(A3,right)):
        scene(o,NAME,(feature,),s0);surface(o,text,s0)
        scene(o,NAME,(feature,),s1);surface(o,text,s1)
    scene(o,INQUIRY_CONTEXT,(A1,A3),s0);surface(o,question,s0)
    scene(o,INQUIRY_CONTEXT,(A1,A3),s1);surface(o,question,s1)

def ask_with_partner(o,S0,A1,A3,p,src=9000):
    partner(o,p);return ask(o,S0,A1,A3,src)

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,2,4,42,8)
    o=ReferenceOrganismV2(spec);S0,S2,A1,A2,A3=equal_graph(o);teach_inquiry(o,A1,A3)
    feat(o,A1,FL,8001);feat(o,A3,FR,8002)
    before=o._world_revisions.row_count
    see(o,A1,SEE,True)
    checks['seeing_an_alternative_writes_world']=A1 in o._world_marked_entities() and A3 not in o._world_marked_entities()
    checks['planning_ints_are_not_world_referents']=10 not in o._world_marked_entities() and 20 not in o._world_marked_entities()
    q=ask(o,S0,A1,A3)
    qw=o._world_state_occurrences(INQUIRY_CONTEXT,(A1,A3))
    checks['question_is_outer_language']=isinstance(q,ActionV2) and q.payload==u('left or right?') and q.channel==77
    checks['question_recruits_seen_alternative']=bool(qw) and {row[0] for row in qw}=={A1} and all(row[3] in q.contributors for row in qw)
    checks['unseen_alternative_stays_off_world']=A3 not in {row[0] for row in qw}
    checks['question_does_not_mint_world_for_unseen']=o._world_revisions.row_count==before+1
    yoked=ReferenceOrganismV2(spec);equal_graph(yoked);teach_inquiry(yoked,A1,A3);feat(yoked,A1,FL,8001);feat(yoked,A3,FR,8002);see(yoked,A1,SEE,False)
    checks['yoked_see_cannot_write_world']=yoked._world_revisions.row_count==0
    cut=ReferenceOrganismV2(spec);S0c,_,A1c,_,A3c=equal_graph(cut);teach_inquiry(cut,A1c,A3c);feat(cut,A1c,FL,8001);feat(cut,A3c,FR,8002);see(cut,A1c,SEE,True)
    cut.contact(CONTACT_WITHDRAW_SOURCE,(SEE,),88002,True,True);cq=ask(cut,S0c,A1c,A3c)
    checks['world_withdrawal_keeps_question_drops_world']=isinstance(cq,ActionV2) and cq.payload==u('left or right?') and not cut._world_state_occurrences(INQUIRY_CONTEXT,(A1c,A3c))
    dead=ReferenceOrganismV2(spec);S0d,_,A1d,_,A3d=equal_graph(dead);teach_inquiry(dead,A1d,A3d)
    dead.contact(CONTACT_COMM_CHANNEL,(77,),72000,True,True);world(dead,S0d,SEE);target(dead,(40,));afford(dead,A1d,A3d)
    dead.contact(CONTACT_WITHDRAW_SOURCE,(SEE,),88011,True,True);dq=dead.tick()
    checks['withdrawn_world_source_does_not_author_question']=dq is None
    ch=ReferenceOrganismV2(spec);equal_graph(ch);teach_inquiry(ch,A1,A3);DEAD=72000
    ch.contact(CONTACT_WITHDRAW_SOURCE,(DEAD,),88024,True,True)
    ch.contact(CONTACT_COMM_CHANNEL,(77,),DEAD,True,True)
    checks['withdrawn_source_cannot_open_channel']=ch.communication_channel==0
    faulted=ReferenceOrganismV2(spec);S0f,_,A1f,_,A3f=equal_graph(faulted);teach_inquiry(faulted,A1f,A3f);feat(faulted,A1f,FL,8001);feat(faulted,A3f,FR,8002);see(faulted,A1f,SEE,True)
    faulted.contact(CONTACT_COMM_CHANNEL,(77,),72000,True,True);world(faulted,S0f,9000);target(faulted,(40,));afford(faulted,A1f,A3f)
    faulted.inject_output_fault(0,ord('X'));bad=faulted.tick()
    checks['question_output_is_faultable']=isinstance(bad,ActionV2) and bad.payload!=u('left or right?') and tuple(bad.planned_payload)==u('left or right?')
    if isinstance(bad,ActionV2):faulted.contact(CONTACT_CONSEQUENCE,(bad.ticket,-1),9000,True,True)
    rq=faulted.tick() if isinstance(bad,ActionV2) else None
    rqw=faulted._world_state_occurrences(INQUIRY_CONTEXT,(A1f,A3f))
    checks['question_repair_keeps_seen_world']=isinstance(rq,ActionV2) and rq.repair and rq.payload==u('left or right?') and {row[0] for row in rqw}=={A1f} and all(row[3] in rq.contributors for row in rqw)
    both=ReferenceOrganismV2(spec);S0m,_,A1m,_,A3m=equal_graph(both);feat(both,A1m,FL,8001);feat(both,A3m,FR,8002)
    teach_named_inquiry(both,A1m,A3m,'left','right','left or right?',EN,EN2)
    teach_named_inquiry(both,A1m,A3m,'links','rechts','links oder rechts?',DE,DE2)
    see(both,A1m,SEE,True);buried=copy.deepcopy(both.checkpoint())
    en_o=ReferenceOrganismV2.restore(copy.deepcopy(buried));en=ask_with_partner(en_o,S0m,A1m,A3m,EN);ew=en_o._world_state_occurrences(INQUIRY_CONTEXT,(A1m,A3m))
    de_o=ReferenceOrganismV2.restore(copy.deepcopy(buried));de=ask_with_partner(de_o,S0m,A1m,A3m,DE);dw=de_o._world_state_occurrences(INQUIRY_CONTEXT,(A1m,A3m))
    checks['english_partner_asks_english']=isinstance(en,ActionV2) and en.payload==u('left or right?')
    checks['german_partner_asks_german']=isinstance(de,ActionV2) and de.payload==u('links oder rechts?')
    checks['both_questions_recruit_same_seen_world']=isinstance(en,ActionV2) and isinstance(de,ActionV2) and {row[0] for row in ew}=={row[0] for row in dw}=={A1m} and all(row[3] in en.contributors for row in ew) and all(row[3] in de.contributors for row in dw)
    checks['no_qa_opcode']=not hasattr(o,'ask') and not hasattr(o,'imagine') and not hasattr(o,'prompt')
    result={'schema':'0x1.reference-organism-world-inquiry.v1','pass':all(checks.values()),'checks':checks,'runtime_llm':False,'graph_flip':False,'question':bytes(q.payload).decode() if isinstance(q,ActionV2) else '','claim':'INQUIRY_SURFACE_RECRUITS_SEEN_WORLD_ALTERNATIVE_REFERENCE_ONLY','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_WORLD_INQUIRY '+('GREEN' if result['pass'] else 'RED')+' question_recruits_world=1 unseen_off=1 qa_opcode=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
