#!/usr/bin/env python3
"""Information-need -> learned question whole-organism assay."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

def u(s):return tuple(s.encode())
def world(o,state,source):o.contact(CONTACT_WORLD_STATE,tuple(state),source,True,True)
def target(o,state):o.contact(CONTACT_BODY_TARGET,tuple(state),70000,True,True)
def afford(o,*actions):o.contact(CONTACT_AFFORDANCES,tuple(actions),71000,True,True)
def settle_motor(o,a,effect,next_state,source):o.contact(CONTACT_MOTOR_CONSEQUENCE,(a.ticket,effect,len(next_state),*next_state),source,True,True)
def scene(o,context,atoms,source,channel=7):o.contact(CONTACT_SCENE,(channel,context,len(atoms),*atoms),source,True,True)
def surface(o,text,source):o.contact(CONTACT_SURFACE,u(text),source,True,True)

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

def main():
    t=time.perf_counter();checks={};spec=PopulationSpecV1(65536,fanout=2,sites_per_feature=4,eligibility_horizon=8)
    o=ReferenceOrganismV2(spec);S0,S2,A1,A2,A3=equal_graph(o);teach_inquiry(o,A1,A3)
    checks['question_surface_learned']=o.language.template(INQUIRY_CONTEXT,2) is not None
    o.contact(CONTACT_COMM_CHANNEL,(77,),72000,True,True);world(o,S0,9000);target(o,(40,));afford(o,A1,A3)
    before_motor=len(o.motor_actions);q=o.tick()
    checks['ambiguity_asks_without_host_goal']=isinstance(q,ActionV2) and q.payload==u('left or right?') and q.channel==77 and q.source==9000 and len(o.motor_actions)==before_motor
    checks['numeric_information_need']=o.information_need[0]==2 and set(o.information_need[1:])=={A1,A3}
    checks['question_traced_to_population_and_template']=q.population_occurrence>0 and q.template_identity>0 and len(q.contributors)>=3
    # Pending question blocks repeated cognition until a partner consequence returns.
    checks['pending_question_blocks_repeat']=o.tick() is None and len(o.actions)==1
    o.contact(CONTACT_CONSEQUENCE,(q.ticket,0),9000,True,True);checks['settled_question_not_positive_credit']=q.settled and q.effect==0
    checks['no_immediate_repeat_after_settlement']=o.tick() is None and len(o.actions)==1 and o.information_need_asked
    # New world evidence resets the information-need episode; reaching the goal ends the need.
    world(o,S2,9000);checks['new_world_evidence_clears_need']=not o.information_need and o.tick() is None

    # Without learned inquiry surface, the same cognition remains uncertain but silent.
    silent=ReferenceOrganismV2(spec);S0b,S2b,A1b,A2b,A3b=equal_graph(silent);silent.contact(CONTACT_COMM_CHANNEL,(77,),72000,True,True);world(silent,S0b,9100);target(silent,(40,));afford(silent,A1b,A3b);none=silent.tick()
    checks['unlearned_question_stays_silent']=none is None and silent.information_need and not silent.actions

    net=ReferenceOrganismV2(spec);S0n,S2n,A1n,A2n,A3n=equal_graph(net);teach_inquiry(net,A1n,A3n)
    net.contact(CONTACT_COMM_CHANNEL,(77,),72000,True,True);world(net,S0n,9000);target(net,(40,));afford(net,A1n,A3n)
    qn=net.tick()
    checks['inquiry_joins_selection_network']=isinstance(qn,ActionV2) and any(k==PREF_TEMPLATE for k,_,_,_ in qn.selection_occurrences) and len(qn.lexical_identities)==2
    learned=net.contact(CONTACT_CONSEQUENCE,(qn.ticket,-1),9000,True,True) if qn else {}
    checks['independent_negative_credits_inquiry_network']=learned.get('selection_network_updates',0)>=1
    world(net,S2n,9000);world(net,S0n,9000);target(net,(40,));afford(net,A1n,A3n)
    checks['negative_inquiry_network_stays_silent']=net.tick() is None and bool(net.information_need)
    yoked=ReferenceOrganismV2(spec);S0y,S2y,A1y,A2y,A3y=equal_graph(yoked);teach_inquiry(yoked,A1y,A3y)
    yoked.contact(CONTACT_COMM_CHANNEL,(77,),72000,True,True);world(yoked,S0y,9000);target(yoked,(40,));afford(yoked,A1y,A3y)
    yq=yoked.tick();yoked.contact(CONTACT_CONSEQUENCE,(yq.ticket,1),9000,True,False)
    world(yoked,S2y,9000);world(yoked,S0y,9000);target(yoked,(40,));afford(yoked,A1y,A3y)
    checks['yoked_return_cannot_punish_inquiry_network']=isinstance(yoked.tick(),ActionV2)

    cp=o.checkpoint();a=ReferenceOrganismV2.restore(cp);b=ReferenceOrganismV2.restore(cp);checks['checkpoint_replay']=a.digest()==b.digest() and a.information_need==b.information_need
    result={'schema':'0x1.reference-organism-inquiry.v1','pass':all(checks.values()),'checks':checks,'question_bytes':len(q.payload),'information_need':list(o.information_need),'claim':'LEARNED_INFORMATION_SEEKING_EXPRESSION_NOT_PROMPTED_QA_OR_HUMAN_DIALOGUE','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_INQUIRY '+('GREEN' if result['pass'] else 'RED')+' endogenous_need=1 learned_question=1 host_goal=0 prompt=0 llm=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
