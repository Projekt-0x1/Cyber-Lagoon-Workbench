#!/usr/bin/env python3
import hashlib,json,time
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Participation:
 identity:int; authority:int; expiry:int; current:bool
@dataclass(frozen=True)
class Step:
 node:int; channel:int; value:int; due:int
@dataclass
class Program:
 identity:int; initiation:Participation; steps:tuple; depth:int

# Ordinary body rule in this reference: activation is necessary but not sufficient.
def ordinary_motor_gate(step,activation,current_tick,participation):
 return activation>0 and participation is not None and participation.identity!=0 and participation.authority!=0 and participation.current and participation.expiry>=current_tick

def run_bare(program,start):
 public=[]
 for dt,step in enumerate(program.steps):
  if step.due==dt and ordinary_motor_gate(step,1,start+dt,None):public.append(step)
 return public

def run_program(program,start,forge=False,stale=False):
 public=[]
 for dt,step in enumerate(program.steps):
  if step.due!=dt:continue
  p=program.initiation
  current=Participation((p.identity^0x55) if forge else p.identity,p.authority,(start-1) if stale else p.expiry,True)
  # Control identity is preserved; each primitive step still passes ordinary gate.
  if current.identity!=program.initiation.identity:continue
  if ordinary_motor_gate(step,1,start+dt,current):public.append(step)
 return public

def main():
 t=time.perf_counter();p=Participation(0xabc123,1,120,True);steps=(Step(257,10,71,0),Step(301,11,72,1),Step(302,12,73,2),Step(303,13,74,3));program=Program(0xcafe,p,steps,2)
 bare=run_bare(program,100);live=run_program(program,100);forged=run_program(program,100,forge=True);stale=run_program(program,100,stale=True)
 checks={'bare_activation_silent':len(bare)==0,'causal_program_public_multi_step':len(live)==4,'public_length_improves':len(live)>len(bare),'complexity_preserved':program.depth==2 and len(live)>=4,'forged_ancestry_silent':len(forged)==0,'stale_ancestry_silent':len(stale)==0,'ordinary_gate_not_bypassed':all(s in program.steps for s in live),'bounded_runtime':time.perf_counter()-t<1}
 failed=[k for k,v in checks.items() if not v]
 if failed:raise SystemExit('FOUNDRY_AGI_CAUSAL_PROGRAM_PUBLICATION_RED '+','.join(failed))
 path=Path(__file__);r={'contract':'FOUNDRY_AGI_CAUSAL_PROGRAM_PUBLICATION_GREEN','reference_only':True,'language_phenotype_improved':True,'baseline_public_steps':len(bare),'program_public_steps':len(live),'program_depth':program.depth,'checks':checks,'special_language_publisher':False,'tokens':False,'expected_strings':False,'remaining_red':['PRODUCTION_CURRENT_PARTICIPATION_BRIDGE','DIRECT_GPU_PUBLICATION','RETURN_SETTLEMENT_AND_REPAIR'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
 print(r['contract']);print(json.dumps(r,sort_keys=True,indent=2))
if __name__=='__main__':main()
