#!/usr/bin/env python3
"""Whole-organism receipt for action-control / goal / procedure / valence separation."""
from __future__ import annotations
import copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_organism_v2 import *
from reference_population_v1 import PopulationSpecV1

WORLD=4101;TARGET_SRC=4102;AFF_SRC=4103;A=41;B=42
S0=(101,);S1=(201,);WRONG=(301,)

def organism():return ReferenceOrganismV2(PopulationSpecV1(32768,fanout=2,sites_per_feature=4,eligibility_horizon=8))
def prepare(o,state=S0,target=S1):
    o.contact(CONTACT_WORLD_STATE,state,WORLD,True,True);o.contact(CONTACT_BODY_TARGET,target,TARGET_SRC,True,True);o.contact(CONTACT_AFFORDANCES,(A,B),AFF_SRC,True,True)
def train_expected(o,state=S0,nxt=S1):
    o.cognition.observe(state,A,nxt,0,4201,True);o.cognition.observe(state,A,nxt,0,4202,True)
def settle(o,action,nxt,effect,independent=True):
    motor=o._issue_motor(action)
    if not isinstance(motor,MotorActionV2):raise AssertionError(('outcome-category:motor',motor))
    o.contact(CONTACT_MOTOR_CONSEQUENCE,(motor.ticket,int(effect),len(nxt),*nxt),WORLD,True,bool(independent));return motor,o.last_action_outcome_decomposition()

def main():
    started=time.perf_counter();checks={}

    # Controlled transition can be aversive.
    controlled=organism();train_expected(controlled);prepare(controlled)
    before=controlled.recursive_metacontrol.outcome_count
    _m,r=settle(controlled,A,S1,-1,True)
    checks['negative_valence_can_still_be_control_match']=bool(r.control_evaluable and r.control_match and r.valence==-1)
    checks['negative_valence_control_match_increases_self_competence']=controlled.recursive_metacontrol.outcome_count==before+1 and controlled.self_reliability_q16(A)>CTXQ//2
    checks['goal_attainment_is_separate_from_valence']=bool(r.goal_evaluable and r.goal_attained and r.valence==-1)

    # Pleasant consequence can accompany the wrong transition.
    wrong=organism();train_expected(wrong);prepare(wrong)
    _m,w=settle(wrong,A,WRONG,1,True)
    checks['positive_valence_can_still_be_control_mismatch']=bool(w.control_evaluable and not w.control_match and w.valence==1)
    checks['positive_valence_does_not_create_false_self_competence']=wrong.self_reliability_q16(A)<CTXQ//2
    checks['positive_valence_does_not_imply_goal_attainment']=bool(w.goal_evaluable and not w.goal_attained and w.valence==1)

    # No pre-outcome transition model means unknown self competence, regardless of reward.
    novel=organism();prepare(novel)
    before_novel=novel.recursive_metacontrol.outcome_count
    _m,n=settle(novel,A,S1,1,True)
    checks['novel_positive_outcome_without_control_witness_is_uncalibrated']=not n.control_evaluable and novel.recursive_metacontrol.outcome_count==before_novel

    # Non-independent return carries valence but cannot calibrate action control.
    yoked=organism();train_expected(yoked);prepare(yoked)
    before_yoked=yoked.recursive_metacontrol.outcome_count
    _m,y=settle(yoked,A,S1,1,False)
    checks['nonindependent_match_cannot_calibrate_self_control']=y.control_match and not y.independent and yoked.recursive_metacontrol.outcome_count==before_yoked

    # Procedure execution is sequence/provenance, not reward sign.
    p=organism();prepare(p)
    pid=p.compose_cultural_instruction((A,B),5101)
    if not pid:raise AssertionError('outcome-category:program')
    p._culture_active_program=(int(pid),0);p._culture_reset_execution_evidence()
    _m,p1=settle(p,A,(211,),-1,True)
    cursor_after_first=tuple(p._culture_active_program)
    # Refresh state/affordance for second instructed action; same consequence source remains WORLD.
    p.contact(CONTACT_WORLD_STATE,(211,),WORLD,True,True);p.contact(CONTACT_BODY_TARGET,(212,),TARGET_SRC,True,True);p.contact(CONTACT_AFFORDANCES,(A,B),AFF_SRC,True,True)
    _m,p2=settle(p,B,(212,),-1,True)
    program=next((row for row in p.recursive_self_culture._programs if int(row.identity)==int(pid)),None)
    confirmations=() if program is None else tuple(program.confirmations)
    checks['aversive_first_step_still_advances_procedure_cursor']=cursor_after_first==(int(pid),1) and p1.valence==-1
    checks['aversive_sequence_can_complete_execution']=not p._culture_active_program and bool(confirmations) and p2.valence==-1
    checks['procedure_execution_confirmation_is_not_goal_or_valence_claim']=bool(confirmations) and p2.goal_attained and p2.valence==-1

    # Decomposition is transient; learned relations survive checkpoint, transient receipt does not.
    cp=controlled.checkpoint();restored=ReferenceOrganismV2.restore(copy.deepcopy(cp))
    checks['outcome_decomposition_is_not_shadow_transcript']=restored.last_action_outcome_decomposition() is None
    checks['control_competence_survives_checkpoint']=restored.recursive_metacontrol.checkpoint()==controlled.recursive_metacontrol.checkpoint()

    failed=sorted(k for k,v in checks.items() if not v)
    result={'schema':'cyber-lagoon.action-outcome-category-separation.v1','pass':not failed,'checks':checks,'failed':failed,
        'controlled':r.__dict__,'wrong':w.__dict__,'novel':n.__dict__,'claim':'ACTION_CONTROL_GOAL_PROCEDURE_EXECUTION_AND_OUTCOME_VALENCE_ARE_DISTINCT_LIVED_RELATIONS',
        'elapsed_ms':round((time.perf_counter()-started)*1000,3)}
    print('FOUNDRY_ACTION_OUTCOME_CATEGORY_SEPARATION '+('GREEN' if not failed else 'RED'))
    print(json.dumps(result,indent=2,sort_keys=True));raise SystemExit(0 if not failed else 1)
if __name__=='__main__':main()
