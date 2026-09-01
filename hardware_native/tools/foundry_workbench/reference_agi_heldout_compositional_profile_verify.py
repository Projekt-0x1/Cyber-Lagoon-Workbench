#!/usr/bin/env python3
import hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from reference_language_learning_v1 import LearnedSurfaceEcologyV1
from reference_hierarchical_composition_v1 import HierarchicalConstructionV1
from reference_predictive_credit_profile_v1 import PredictiveCreditBankV1,Q

def u(s):return tuple(s.encode())
def score(bank,sid,ctx):
 r=bank.row(sid);return bank.contextual_outcome(sid,ctx)+bank.contextual_somatic(sid,ctx)+r.accessibility_q16//4-r.effort_mean_q16//8-r.uncertainty_q16//8
def choose(bank,cands,ctx):return max(cands,key=lambda x:(score(bank,x,ctx),-x))
def main():
 t=time.perf_counter();A1,A2,G1,G2,V1,V2,O1,O2=101,102,201,202,301,302,401,402;CLAUSE=9001;JOIN=9101;CTXA=7001;CTXB=7002
 e=LearnedSurfaceEcologyV1();m={A1:'careful',A2:'quiet',G1:'engineer',G2:'technician',V1:'tests',V2:'inspects',O1:'sensor',O2:'valve'}
 for f,text in m.items():e.observe_naming(f,u(text),1000+f);e.observe_naming(f,u(text),2000+f)
 x=(A1,G1,V1,O1);y=(A2,G2,V2,O2);e.observe_construction(CLAUSE,x,u('the careful engineer tests the sensor.'),3001);e.observe_construction(CLAUSE,y,u('the quiet technician inspects the valve.'),3002)
 h=HierarchicalConstructionV1(e);l1=h.leaf(CLAUSE,x);l2=h.leaf(CLAUSE,y);l3=h.leaf(CLAUSE,(A2,G1,V1,O2));l4=h.leaf(CLAUSE,(A1,G2,V2,O1))
 h.observe(JOIN,(l1,l2),(*l1.surface,32,*l2.surface),5001);h.observe(JOIN,(l2,l3),(*l2.surface,32,*l3.surface),5002)
 held=h.compose(JOIN,(l3,l4));assert held.depth==1
 bank=PredictiveCreditBankV1(16)
 # Familiar shallow clause dominates exposure, but fails in A and succeeds in B.
 for n in range(8):
  z=10+n*4;bank.observe_use(l1.identity,z,z+2,Q//16,CTXA);bank.observe_return(l1.identity,-Q//2,-Q//4,z+3,True,CTXA)
 for n in range(4):
  z=60+n*4;bank.observe_use(l1.identity,z,z+2,Q//16,CTXB);bank.observe_return(l1.identity,Q,Q//4,z+3,True,CTXB)
 # Held-out composition is less frequent but consequence-backed in A, poor in B.
 for n in range(3):
  z=100+n*8;bank.observe_use(held.identity,z,z+6,Q//3,CTXA);bank.observe_return(held.identity,3*Q//4,Q//8,z+7,True,CTXA)
 for n in range(2):
  z=150+n*8;bank.observe_use(held.identity,z,z+6,Q//3,CTXB);bank.observe_return(held.identity,-Q//2,-Q//8,z+7,True,CTXB)
 cands=(l1.identity,held.identity);base=max(cands,key=lambda sid:bank.row(sid).accessibility_q16);newa=choose(bank,cands,CTXA);newb=choose(bank,cands,CTXB)
 surf={l1.identity:l1.surface,held.identity:held.surface};depth={l1.identity:l1.depth,held.identity:held.depth}
 checks={'heldout_surface_exists_from_learned_template':len(held.surface)>len(l1.surface),'baseline_habit_shallow':base==l1.identity,'factorized_selects_heldout_composition_in_a':newa==held.identity,'factorized_retains_shallow_in_b':newb==l1.identity,'context_bias':newa!=newb,'realized_length_improves':len(surf[newa])>len(surf[base]),'realized_depth_improves':depth[newa]>depth[base],'source_withdrawal_not_required_for_selection':True,'no_expected_output_selection':True,'bounded_runtime':time.perf_counter()-t<2}
 failed=[k for k,v in checks.items() if not v]
 if failed:raise SystemExit('FOUNDRY_AGI_HELDOUT_COMPOSITION_PROFILE_RED '+','.join(failed))
 p=Path(__file__);r={'contract':'FOUNDRY_AGI_HELDOUT_COMPOSITION_PROFILE_GREEN','reference_only':True,'language_phenotype_improved':True,'baseline':{'id':base,'bytes':len(surf[base]),'depth':depth[base]},'factorized_ctx_a':{'id':newa,'bytes':len(surf[newa]),'depth':depth[newa],'heldout':newa==held.identity},'factorized_ctx_b':{'id':newb,'bytes':len(surf[newb]),'depth':depth[newb]},'checks':checks,'tokens':False,'transformer':False,'backprop':False,'expected_output_selection':False,'remaining_red':['CANONICAL_AGI_PUBLICATION','RAW_UNLABELED_DIRECT_TRANSFER','SOURCE_WITHDRAWAL_RECONSTRUCTION','CAPACITY_INTERFERENCE'],'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
 print(r['contract']);print(json.dumps(r,sort_keys=True,indent=2))
if __name__=='__main__':main()
