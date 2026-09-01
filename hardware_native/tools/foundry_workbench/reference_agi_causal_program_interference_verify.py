#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_language_learning_v1 import LearnedSurfaceEcologyV1
from reference_hierarchical_composition_v1 import HierarchicalConstructionV1
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q

def u(s):return tuple(s.encode())
def build():
 A1,A2,G1,G2,V1,V2,O1,O2=101,102,201,202,301,302,401,402;CLAUSE=9001;JOIN=9101
 e=LearnedSurfaceEcologyV1();m={A1:'careful',A2:'quiet',G1:'engineer',G2:'technician',V1:'tests',V2:'inspects',O1:'sensor',O2:'valve'}
 for f,t in m.items():e.observe_naming(f,u(t),1000+f);e.observe_naming(f,u(t),2000+f)
 x=(A1,G1,V1,O1);y=(A2,G2,V2,O2);e.observe_construction(CLAUSE,x,u('the careful engineer tests the sensor.'),3001);e.observe_construction(CLAUSE,y,u('the quiet technician inspects the valve.'),3002)
 h=HierarchicalConstructionV1(e);short=h.leaf(CLAUSE,x);s=h.leaf(CLAUSE,y);a=h.leaf(CLAUSE,(A2,G1,V1,O2));b=h.leaf(CLAUSE,(A1,G2,V2,O1));h.observe(JOIN,(short,s),(*short.surface,32,*s.surface),5001);h.observe(JOIN,(s,a),(*s.surface,32,*a.surface),5002);deep=h.compose(JOIN,(a,b));return short,deep

def value(bank,sid,ctx):
 r=bank.row(sid);return bank.contextual_outcome(sid,ctx)+bank.contextual_somatic(sid,ctx)+r.accessibility_q16//4-r.effort_mean_q16//8

def choose(bank,cands,ctx):
 ok=[s for s in cands if bank.row(s).controllability_q16>=Q//2]
 return max(ok,key=lambda s:(value(bank,s,ctx),-s)) if ok else 0

def main():
 st=time.perf_counter();CTX=0x1F31;short,deep=build();bank=PredictiveCreditBankV1(128)
 for n in range(6):
  t=10+n*5;bank.observe_use(short.identity,t,t+2,Q//12,CTX);bank.observe_return(short.identity,Q//2,Q//16,t+3,True,CTX);bank.observe_control(short.identity,True,True)
 for n in range(4):
  t=70+n*9;bank.observe_use(deep.identity,t,t+7,Q//2,CTX);bank.observe_return(deep.identity,Q,Q//8,t+8,True,CTX);bank.observe_control(deep.identity,True,True)
 before=bank.snapshot();choice_before=choose(bank,(short.identity,deep.identity),CTX);deep_before=(bank.row(deep.identity).duration_mean_q16,bank.row(deep.identity).outcome_mean_q16,bank.row(deep.identity).controllability_q16)
 # 112 unrelated structures nearly fill the fixed profile table. None share target identity/context.
 for i in range(112):
  sid=0x100000+i;t=200+i*3;ctx=0x8000+(i%17);bank.observe_use(sid,t,t+(i%5)+1,(i%7+1)*Q//32,ctx);bank.observe_return(sid,((i%3)-1)*Q//4,((i%5)-2)*Q//16,t+8,True,ctx);bank.observe_control(sid,True,(i%4)!=0);bank.observe_successor(sid,0x200000+(i%19),i%4)
 choice_after=choose(bank,(short.identity,deep.identity),CTX);deep_after=(bank.row(deep.identity).duration_mean_q16,bank.row(deep.identity).outcome_mean_q16,bank.row(deep.identity).controllability_q16)
 # Add one legitimate new consequence for deep after distractors to prove plasticity remains.
 old_out=bank.contextual_outcome(deep.identity,CTX);bank.observe_return(deep.identity,-Q//2,-Q//8,999,True,CTX);new_out=bank.contextual_outcome(deep.identity,CTX)
 checks={
  'fixed_capacity_not_increased':bank.capacity==128,
  'distractors_fit_without_capacity_refusal':bank.capacity_refusals==0 and len(bank.rows)==114,
  'old_language_choice_survives_interference':choice_before==deep.identity and choice_after==deep.identity,
  'old_duration_value_control_survive_exactly':deep_after==deep_before,
  'heldout_language_still_longer':len(deep.surface)>len(short.surface),
  'plasticity_remains_after_stability':new_out<old_out,
  'distractor_contexts_do_not_globally_rewrite_target':bank.row(deep.identity).last_context==CTX,
  'bounded_runtime':time.perf_counter()-st<2,
 }
 failed=[k for k,v in checks.items() if not v]
 if failed:raise SystemExit('FOUNDRY_AGI_CAUSAL_PROGRAM_INTERFERENCE_RED '+','.join(failed))
 p=Path(__file__);r={'contract':'FOUNDRY_AGI_CAUSAL_PROGRAM_INTERFERENCE_GREEN','reference_only':True,'language_phenotype_preserved_under_interference':True,'capacity':bank.capacity,'rows':len(bank.rows),'distractors':112,'target_before':{'choice':choice_before,'bytes':len(deep.surface),'depth':deep.depth},'target_after':{'choice':choice_after,'bytes':len(deep.surface),'depth':deep.depth},'checks':checks,'remaining_red':['CAPACITY_EVICTION_POLICY','THOUSANDS_OF_PROGRAMS_GPU','CANONICAL_AGI_MIGRATION','RAW_DIRECT_LANGUAGE'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 print(r['contract']);print(json.dumps(r,sort_keys=True,indent=2))
if __name__=='__main__':main()
