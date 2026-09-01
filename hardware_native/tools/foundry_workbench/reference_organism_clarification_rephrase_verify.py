#!/usr/bin/env python3
"""Visible discourse ratchet: failed learned clarification earns a clearer rephrase."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

def u(s): return tuple(s.encode())
def world(o,state,source=9000): o.contact(CONTACT_WORLD_STATE,tuple(state),source,True,True)
def target(o,state): o.contact(CONTACT_BODY_TARGET,tuple(state),70000,True,True)
def afford(o,*actions): o.contact(CONTACT_AFFORDANCES,tuple(actions),71000,True,True)
def scene(o,context,atoms,source,channel=7): o.contact(CONTACT_SCENE,(channel,context,len(atoms),*atoms),source,True,True)
def surface(o,text,source): o.contact(CONTACT_SURFACE,u(text),source,True,True)
def settle_motor(o,a,effect,next_state,source): o.contact(CONTACT_MOTOR_CONSEQUENCE,(a.ticket,effect,len(next_state),*next_state),source,True,True)

def experience(o,start,action,next_state,effect,source):
    world(o,start,source); target(o,next_state); afford(o,action)
    a=o.tick(); assert isinstance(a,MotorActionV2) and a.action_id==action
    settle_motor(o,a,effect,next_state,source)

def equal_graph(o):
    S0=(10,); S1=(20,); ALT=(25,); S2=(30,40); A1,A2,A3=101,102,103
    for src in (1000,1001): experience(o,S0,A1,S1,1,src)
    for src in (1100,1101): experience(o,S1,A2,S2,1,src)
    for src in (1200,1201): experience(o,S0,A3,ALT,1,src)
    for src in (1300,1301): experience(o,ALT,A2,S2,1,src)
    return S0,S2,A1,A3

def teach(o,A1,A3):
    NAME=100
    for feature,text in ((A1,'left'),(A3,'right')):
        for src in (50000+feature,60000+feature):
            scene(o,NAME,(feature,),src); surface(o,text,src)
    # Terse wording has stronger developmental support and therefore wins first.
    for src in (8100,8101,8102):
        scene(o,INQUIRY_CONTEXT,(A1,A3),src); surface(o,'left or right?',src)
    # Clearer wording is learned but initially lower-support.
    for src in (8200,8201):
        scene(o,INQUIRY_CONTEXT,(A1,A3),src); surface(o,'do you mean left or right?',src)

def stage_ambiguity(o,S0,A1,A3,channel=77,source=9000):
    o.contact(CONTACT_COMM_CHANNEL,(channel,),72000+channel,True,True)
    world(o,S0,source); target(o,(40,)); afford(o,A1,A3)
    return o.tick()

def refuses_consequence(o,ticket,effect,source):
    try:o.contact(CONTACT_CONSEQUENCE,(ticket,effect),source,True,True)
    except ValueError:return True
    return False

def main():
    started=time.perf_counter(); checks={}
    o=ReferenceOrganismV2(PopulationSpecV1(65536,fanout=2,sites_per_feature=4,eligibility_horizon=8))
    S0,S2,A1,A3=equal_graph(o); teach(o,A1,A3)
    first=stage_ambiguity(o,S0,A1,A3)
    checks['first_question_is_terse']=isinstance(first,ActionV2) and bytes(first.payload)==b'left or right?'
    learned=o.contact(CONTACT_CONSEQUENCE,(first.ticket,-1),9000,True,True) if isinstance(first,ActionV2) else {}
    checks['failed_question_gets_independent_selection_feedback']=learned.get('selection_network_updates',0)>=1
    # Start a genuinely later ambiguity episode; do not prompt a second answer synchronously.
    world(o,S2,9000); world(o,S0,9000); target(o,(40,)); afford(o,A1,A3)
    second=o.tick()
    checks['later_ambiguity_rephrases_more_explicitly']=(
        isinstance(second,ActionV2) and bytes(second.payload)==b'do you mean left or right?')

    # A separate continuing Adult must retain two independently returnable
    # public commitments.  Admission never assigns queue/turn authority.
    concurrent=ReferenceOrganismV2(PopulationSpecV1(65536,fanout=2,sites_per_feature=4,eligibility_horizon=8))
    C0,C2,C1,C3=equal_graph(concurrent);teach(concurrent,C1,C3)
    ca=stage_ambiguity(concurrent,C0,C1,C3,77,9000)
    silent=concurrent.tick()
    world(concurrent,C2,9001)
    cb=stage_ambiguity(concurrent,C0,C1,C3,78,9001)
    concurrent_checkpoint=concurrent.checkpoint()
    resumed=ReferenceOrganismV2.restore(concurrent_checkpoint)
    unsettled=lambda adult:{int(a.ticket) for a in adult.actions if not a.settled}
    two_tickets=({ca.ticket,cb.ticket} if isinstance(ca,ActionV2) and isinstance(cb,ActionV2) else set())
    before_wrong=unsettled(resumed)
    wrong_source=(refuses_consequence(resumed,cb.ticket,-1,9000)
                  if isinstance(cb,ActionV2) else False)
    wrong_ticket=refuses_consequence(resumed,resumed.next_ticket+99,-1,9001)
    after_wrong=unsettled(resumed)
    second_return=(resumed.contact(CONTACT_CONSEQUENCE,(cb.ticket,-1),9001,True,True)
                   if isinstance(cb,ActionV2) else {})
    after_second=unsettled(resumed)
    first_return=(resumed.contact(CONTACT_CONSEQUENCE,(ca.ticket,-1),9000,True,True)
                  if isinstance(ca,ActionV2) else {})
    after_first=unsettled(resumed)
    checkpoint_text=json.dumps(concurrent_checkpoint,sort_keys=True).lower()
    checks['later_episode_can_act_while_prior_public_consequence_is_pending']=(
        isinstance(ca,ActionV2) and isinstance(cb,ActionV2)
        and bytes(ca.payload)==bytes(cb.payload)==b'left or right?'
        and ca.channel==77 and cb.channel==78 and silent is None)
    checks['two_exact_commitments_survive_checkpoint_without_turn_authority']=(
        unsettled(ReferenceOrganismV2.restore(concurrent_checkpoint))==two_tickets
        and not any(name in checkpoint_text for name in ('queue_position','turn_number','semantic_priority')))
    checks['wrong_source_and_wrong_ticket_cannot_settle_commitment']=(
        wrong_source and wrong_ticket and before_wrong==after_wrong==two_tickets)
    checks['reverse_consequence_order_settles_only_exact_ticket']=(
        bool(second_return) and after_second=={ca.ticket}
        and bool(first_return) and not after_first)
    checks['same_adult_not_host_selected']=not hasattr(o,'ask') and not hasattr(o,'prompt')
    checks['visible_discussion_improvement']=(checks['first_question_is_terse']
        and checks['later_ambiguity_rephrases_more_explicitly']
        and checks['later_episode_can_act_while_prior_public_consequence_is_pending'])
    result={'contract':'FOUNDRY_ORGANISM_CLARIFICATION_REPHRASE_GREEN','pass':all(checks.values()),
            'checks':checks,
            'conversation':{'before':bytes(first.payload).decode() if isinstance(first,ActionV2) else '',
                            'after':bytes(second.payload).decode() if isinstance(second,ActionV2) else '',
                            'concurrent':[bytes(a.payload).decode() for a in (ca,cb) if isinstance(a,ActionV2)]},
            'runtime_llm':False,'host_answer_selector':False,
            'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    if not result['pass']:
        print('FOUNDRY_ORGANISM_CLARIFICATION_REPHRASE_RED '+','.join(k for k,v in checks.items() if not v))
    else: print(result['contract'])
    print(json.dumps(result,indent=2,sort_keys=True)); raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__': main()
