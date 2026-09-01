#!/usr/bin/env python3
"""Whole-organism motor learning / planning checks."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

def world(o,state,source):return o.contact(CONTACT_WORLD_STATE,tuple(state),source,True,True)
def target(o,state,source=70000):return o.contact(CONTACT_BODY_TARGET,tuple(state),source,True,True)
def afford(o,*actions):return o.contact(CONTACT_AFFORDANCES,tuple(actions),71000,True,True)
def settle_motor(o,a,effect,next_state,source):return o.contact(CONTACT_MOTOR_CONSEQUENCE,(a.ticket,effect,len(next_state),*next_state),source,True,True)

def experience(o,start,action,next_state,effect,source):
    world(o,start,source);target(o,next_state);afford(o,action);a=o.tick()
    if a is None or not isinstance(a,MotorActionV2) or a.action_id!=action:raise AssertionError(('experience action',action,a))
    settle_motor(o,a,effect,next_state,source);return a

def train_graph(o,equal=False):
    S0=(10,);S1=(20,);ALT=(25,);S2=(30,40);A1,A2,A3=101,102,103
    e12=1 if equal else 2
    for src in (1000,1001):experience(o,S0,A1,S1,1,src)
    for src in (1100,1101):experience(o,S1,A2,S2,e12,src)
    for src in (1200,1201):experience(o,S0,A3,ALT,1,src)
    for src in (1300,1301):experience(o,ALT,A2,S2,1,src)
    return S0,S1,ALT,S2,A1,A2,A3

def main():
    t=time.perf_counter();spec=PopulationSpecV1(65536,fanout=2,sites_per_feature=4,eligibility_horizon=8);checks={}
    o=ReferenceOrganismV2(spec);S0,S1,ALT,S2,A1,A2,A3=train_graph(o,False)
    checks['transition_learning_from_owned_actions']=len(o.cognition.edges())==4 and len(o.motor_actions)==8
    # Learned plan now controls behavior without a host action selector.
    world(o,S0,9000);target(o,(40,));afford(o,A1,A3);first=o.tick();checks['planned_first_action']=isinstance(first,MotorActionV2) and first.action_id==A1
    settle_motor(o,first,1,S1,9000);afford(o,A2);second=o.tick();checks['planned_second_action']=isinstance(second,MotorActionV2) and second.action_id==A2
    settle_motor(o,second,2,S2,9000);checks['goal_completion_stops_action']=o.tick() is None and o.cognition.satisfies(o.world_state,o.body_target)
    checks['counterfactual_planning_did_not_mutate_world']=first.state_before==S0 and second.state_before==S1

    # Fresh organism explores without a model using resident physical trajectory cost; no arbitrary external salt exists.
    x1=ReferenceOrganismV2(spec);world(x1,S0,8000);target(x1,S2);afford(x1,A3,A1);xa=x1.tick()
    x2=ReferenceOrganismV2(spec);world(x2,S0,8000);target(x2,S2);afford(x2,A1,A3);xb=x2.tick()
    checks['endogenous_exploration']=isinstance(xa,MotorActionV2) and xa.action_id in (A1,A3) and xa.action_id==xb.action_id

    # Equal learned routes create an information need and no action.
    tie=ReferenceOrganismV2(spec);train_graph(tie,True);world(tie,S0,9100);target(tie,(40,));afford(tie,A1,A3);before=len(tie.motor_actions);none=tie.tick()
    checks['equal_plan_information_need']=none is None and len(tie.motor_actions)==before and tie.information_need and tie.information_need[0]==2 and set(tie.information_need[1:])=={A1,A3}

    # Withdrawing one support source invalidates the high-value route and replans through the remaining path.
    w=ReferenceOrganismV2(spec);train_graph(w,False);w.contact(CONTACT_WITHDRAW_SOURCE,(1101,),99999,True,True);world(w,S0,9200);target(w,(40,));afford(w,A1,A3);wa=w.tick()
    checks['source_withdrawal_changes_plan']=isinstance(wa,MotorActionV2) and wa.action_id==A3

    # Checkpoint preserves the learned graph, population, world state and next planned action.
    base=ReferenceOrganismV2(spec);train_graph(base,False);world(base,S0,9300);target(base,(40,));afford(base,A1,A3);cp=base.checkpoint();left=ReferenceOrganismV2.restore(cp);right=ReferenceOrganismV2.restore(cp);la=left.tick();ra=right.tick()
    checks['checkpoint_replay']=la==ra and left.digest()==right.digest()
    q=o.population.quantity_vector(None,alternatives=0,horizon=2,trajectory=0);checks['planning_uses_population_state']=q['R']==65536 and q['F']>0 and q['O']>=len(o.motor_actions)
    result={'schema':'agi.reference-organism-cognition.v1','pass':all(checks.values()),'checks':checks,'learned_edges':len(o.cognition.edges()),'motor_actions':len(o.motor_actions),'quantity':q,'claim':'CONTINUING_NONLINGUISTIC_PLANNING_LOOP_NOT_HUMAN_REASONING_OR_PHYSICAL_ADULT','elapsed_ms':round((time.perf_counter()-t)*1000,3)}
    print('FOUNDRY_REFERENCE_ORGANISM_COGNITION '+('GREEN' if result['pass'] else 'RED')+' owned_action_learning=1 planning=1 exploration=1 information_need=1 llm=0 host_action_selector=0')
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if result['pass'] else 1)
if __name__=='__main__':main()
